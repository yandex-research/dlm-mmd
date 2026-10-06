#!/usr/bin/env python
"""Post-train an ELF generator with MMD or iterative refinement distillation (IRD)."""

import argparse
import copy
import logging
import os
import random
import time
from itertools import islice
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from transformers import AutoTokenizer

from configs.config import (
    apply_config_overrides, config_to_dict, load_config_from_yaml, parse_seeds, validate_config,
)
from generation import run_generation
from modules.model import build_model
from train_step import ird_train_step, mmd_train_step
from utils.checkpoint_utils import load_checkpoint, load_pretrained_elf_weights, save_checkpoint
from utils.data_utils import get_dataloader, get_pad_token_id, load_dataset, prepare_batch
from utils.encoder_utils import get_encoder, resolve_latent_stats
from utils.logging_utils import _process_index, log_for_0
from utils.train_utils import (
    TrainState, attach_lr_scheduler, create_learning_rate_fn, ema_update, get_optimizer,
    prefetch_to_device,
)
from utils.wandb_utils import finish_wandb, init_wandb, log_wandb


def _init_distributed(force_cpu: bool) -> torch.device:
    """Initialize torch.distributed if launched via torchrun and return this rank's device."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    use_cuda = torch.cuda.is_available() and not force_cpu
    device = torch.device(f"cuda:{local_rank}") if use_cuda else torch.device("cpu")
    if use_cuda:
        torch.cuda.set_device(device)
    if "WORLD_SIZE" in os.environ and not dist.is_initialized():
        dist.init_process_group("nccl" if use_cuda else "gloo")
    return device


def configure_batch_sizes(config, world: int) -> int:
    """Split global_batch_size over ranks and add decoder rows; return the decoder row count.

    Sets `config.distill_batch_size` (distillation samples per device) and
    `config.batch_size` (rows loaded per device). TinyGSM MMD loads one
    reference per prompt and generates mmd_batch_size responses for it.
    """
    if config.global_batch_size % world:
        raise ValueError("global_batch_size must be divisible by the number of devices")
    distill_rows = config.global_batch_size // world
    references = distill_rows
    if config.objective == "mmd":
        if distill_rows % config.mmd_batch_size:
            raise ValueError(f"per-device batch {distill_rows} must be divisible by mmd_batch_size")
        if config.task == "tinygsm":
            references = distill_rows // config.mmd_batch_size
    decoder_rows = round(distill_rows * config.decoder_prob / (1 - config.decoder_prob))
    if decoder_rows < 1:
        raise ValueError("decoder_prob rounds to zero decoder rows; increase it or the batch size")
    config.distill_batch_size = distill_rows
    config.batch_size = references + decoder_rows
    return decoder_rows


def _average_metrics(window, device, world):
    """Mean of each metric over the logging window and over ranks (one sync per log)."""
    names = list(window[0])
    values = torch.stack([
        torch.stack([torch.as_tensor(m[name], dtype=torch.float32, device=device) for m in window]).mean()
        for name in names
    ])
    if world > 1:
        dist.all_reduce(values)
        values /= world
    return dict(zip(names, values.tolist()))


def run_training(config, *, force_cpu: bool = False):
    validate_config(config)
    train_step = mmd_train_step if config.objective == "mmd" else ird_train_step
    device = _init_distributed(force_cpu)
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1

    train_seed = parse_seeds(config.seed)[0]
    torch.manual_seed(train_seed)
    np.random.seed((train_seed + rank) % 2 ** 32)
    random.seed(train_seed + rank)
    # TF32 for fp32 matmuls on Ampere/Hopper; generation switches it off while sampling.
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    decoder_rows = configure_batch_sizes(config, world)

    output_dir = Path(config.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "config.yml").open("w") as f:
            yaml.safe_dump(config_to_dict(config), f, sort_keys=False)
    init_wandb(config)
    log_for_0(
        f"{config.objective.upper()} on {config.task} | {world} device(s) | "
        f"per device: {config.distill_batch_size} distillation + {decoder_rows} decoder rows | lr={config.lr:.3g}"
    )

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name or config.encoder_model_name)
    pad_token_id = get_pad_token_id(tokenizer, config.pad_token)
    train_dataset, eval_dataset = load_dataset(config, tokenizer=tokenizer)
    encoder_config, encoder = get_encoder(config.encoder_model_name, torch.float32, config.encoder_family)
    encoder = encoder.to(device)
    resolve_latent_stats(config, device)

    # The frozen model extracts MMD features or produces IRD targets; the
    # generator starts as a trainable copy of it. Both come from teacher_checkpoint.
    frozen_model = build_model(config, encoder_config.d_model, len(tokenizer)).to(device)
    load_pretrained_elf_weights(config.teacher_checkpoint, frozen_model)
    frozen_model.eval().requires_grad_(False)
    if config.objective == "mmd" and not 0 <= config.feature_layer < frozen_model.depth:
        raise ValueError(f"feature_layer must be a block index in [0, {frozen_model.depth - 1}]")
    generator = copy.deepcopy(frozen_model).train().requires_grad_(True)

    optimizer = get_optimizer(generator, config, lr=config.lr)
    # step_offset=1: the first update already uses lr / warmup_steps.
    lr_fn = create_learning_rate_fn(
        config.max_iters, config.warmup_steps, config.lr,
        schedule=config.lr_schedule, min_lr=config.min_lr, step_offset=1,
    )
    state = TrainState(
        model=generator, optimizer=optimizer, lr_scheduler=attach_lr_scheduler(optimizer, lr_fn),
        ema_params1=TrainState.init_ema(generator),
    )
    # Keep initialization identical across ranks, then make runtime stochastic ops rank-specific.
    torch.manual_seed(train_seed + rank)
    if config.resume:
        state, _ = load_checkpoint(config.resume, state)

    # torch.compile before DDP so only the inner module is compiled and
    # checkpoint I/O (which uses unwrap_model -> _orig_mod) still works.
    if config.use_compile:
        log_for_0("Compiling the generator and frozen model (first steps will be slower)...")
        state = state.replace(model=torch.compile(state.model))
        if config.objective == "mmd":
            # MMD only calls forward_features on the frozen model.
            frozen_model.forward_features = torch.compile(frozen_model.forward_features)
        else:
            frozen_model = torch.compile(frozen_model)
    if world > 1:
        state = state.replace(model=DDP(
            state.model, device_ids=[device.index] if device.type == "cuda" else None,
            broadcast_buffers=False, find_unused_parameters=False, gradient_as_bucket_view=True,
        ))

    loader = get_dataloader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, drop_last=True,
        max_seq_length=config.max_length, max_input_seq_length=config.max_input_length,
        pad_token_id=pad_token_id, distributed=True,
        encoder_mask_2d=config.encoder_family == "gpt2",
    )
    # A dedicated generator keeps DataLoader worker seeds out of the model RNG state.
    loader.generator = torch.Generator().manual_seed(train_seed)
    batches_per_epoch = len(loader)

    state.optimizer.zero_grad(set_to_none=True)
    last_saved_step = None
    window, window_start = [], time.time()
    description = f"{config.objective.upper()} training"
    log_for_0(f"Training from step {state.step} to {config.max_iters}")
    progress = tqdm(total=config.max_iters, initial=state.step, desc=description,
                    unit="step", dynamic_ncols=True, disable=rank != 0)
    while state.step < config.max_iters:
        state.epoch, offset = divmod(state.step, batches_per_epoch)
        loader.sampler.set_epoch(state.epoch)
        # On resume, skip the batches this epoch already used before they are loaded.
        loader.batch_sampler.sampler = islice(loader.sampler, offset * config.batch_size, None)
        for batch in prefetch_to_device(iter(loader), size=4):
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in prepare_batch(batch).items()}
            state.step += 1
            state.model.train()
            loss, metrics = train_step(state, encoder, batch, config, frozen_model)
            loss.backward()
            metrics["grad_norm"] = torch.nn.utils.clip_grad_norm_(state.model.parameters(), 1.0)
            state.optimizer.step()
            ema_update(state.ema_params1, state.model, config.ema_decay1)
            state.optimizer.zero_grad(set_to_none=True)
            state.lr_scheduler.step()
            progress.update(1)
            window.append(metrics)

            if config.log_freq and state.step % config.log_freq == 0:
                values = _average_metrics(window, device, world)
                elapsed = time.time() - window_start
                values["lr"] = state.optimizer.param_groups[0]["lr"]
                log_for_0(f"step={state.step} " + " ".join(f"{k}={v:.5g}" for k, v in values.items()))
                perf = {
                    "perf/steps_per_sec": len(window) / elapsed,
                    "perf/samples_per_sec": len(window) * config.global_batch_size / elapsed,
                }
                if device.type == "cuda":
                    perf["perf/max_gpu_memory_gb"] = torch.cuda.max_memory_allocated(device) / 2 ** 30
                progress.set_postfix({k: f"{v:.4g}" for k, v in values.items() if k.endswith("loss") or k == "lr"},
                                     refresh=False)
                log_wandb({**{f"train/{k}": v for k, v in values.items()}, "train/epoch": state.epoch, **perf},
                          state.step)
                window, window_start = [], time.time()

            if config.save_freq and state.step % config.save_freq == 0:
                progress.set_description(f"{config.objective.upper()} saving")
                save_checkpoint(state, config.output_dir, state.step, config.hf_repo_id)
                last_saved_step = state.step
                progress.set_description(description)
                window_start = time.time()

            late = config.late_eval_freq and state.step > config.late_eval_start
            eval_freq = config.late_eval_freq if late else config.eval_freq
            if eval_freq and state.step % eval_freq == 0:
                progress.set_description(f"{config.objective.upper()} evaluating")
                log_for_0(f"Starting evaluation at step {state.step}")
                run_generation(
                    state=state, encoder=encoder, eval_dataset=eval_dataset, tokenizer=tokenizer,
                    config=config, local_batch_size=config.distill_batch_size,
                )
                progress.set_description(description)
                window_start = time.time()

            if state.step >= config.max_iters:
                break
    progress.close()

    if last_saved_step != state.step:
        save_checkpoint(state, config.output_dir, state.step, config.hf_repo_id)
    finish_wandb()
    log_for_0(f"Training finished at step {state.step}. Outputs: {output_dir}")
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument("--config_override", action="append", default=[],
                        help="Override a config value (field_name=value). Repeatable.")
    parser.add_argument("--use_cpu", action="store_true", help="Force CPU even when CUDA is available.")
    args = parser.parse_args()
    config = apply_config_overrides(load_config_from_yaml(args.config), args.config_override)

    # Rank 0 also appends timestamped messages to <output_dir>/training.log.
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    file_handler = None
    if _process_index() == 0:
        os.makedirs(config.output_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(config.output_dir, "training.log"), encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
        logging.getLogger().addHandler(file_handler)
    try:
        with logging_redirect_tqdm():
            run_training(config, force_cpu=args.use_cpu)
    except Exception:
        if file_handler is not None:
            logging.getLogger().exception("Training failed")
        raise
    finally:
        if file_handler is not None:
            logging.getLogger().removeHandler(file_handler)
            file_handler.close()


if __name__ == "__main__":
    main()
