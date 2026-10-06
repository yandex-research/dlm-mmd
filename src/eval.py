#!/usr/bin/env python
"""Evaluation script for trained ELF models: load a checkpoint and generate text samples."""

import argparse
import logging
import os
import sys
from types import SimpleNamespace

import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from utils.encoder_utils import get_encoder, resolve_latent_stats
from modules.model import build_model
from utils.logging_utils import log_for_0
from utils.checkpoint_utils import load_pretrained_elf_weights
from utils.data_utils import load_jsonl_dataset, load_dataset_split, get_pad_token_id
from utils.wandb_utils import finish_wandb, init_wandb
from generation import run_generation
from configs.config import load_config_from_yaml, apply_config_overrides, parse_seeds, validate_config

logging.basicConfig(
    format="%(levelname)s - %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO, force=True,
)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained ELF model by generating text samples")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration YAML file")
    parser.add_argument("--config_override", action="append", default=[],
                        help="Override config values (field_name=value). Repeatable.")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Checkpoint file or directory (e.g. outputs/tinygsm_gpt2-mmd/checkpoint_7000), "
                             "or a HF repo id with an optional sub-path. Defaults to checkpoint_path in the config.")
    parser.add_argument("--use_cpu", action="store_true",
                        help="Force CPU even when CUDA is available.")
    return parser.parse_args()


def _init_distributed():
    if "WORLD_SIZE" in os.environ and not dist.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        # Eval only gathers CPU objects (text strings), so gloo is sufficient.
        dist.init_process_group(backend="gloo")


def main():
    args = parse_args()
    _init_distributed()

    device = torch.device("cpu") if args.use_cpu or not torch.cuda.is_available() else torch.device("cuda")

    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False

    log_for_0("Loading configuration...")
    config = apply_config_overrides(load_config_from_yaml(args.config), args.config_override)
    checkpoint_path = args.checkpoint_path or config.checkpoint_path
    if not checkpoint_path:
        raise ValueError("Set checkpoint_path in the config or pass --checkpoint_path for evaluation")
    validate_config(config, training=False)

    world = dist.get_world_size() if dist.is_initialized() else 1
    local_batch_size = config.global_batch_size // world
    if local_batch_size < 1:
        raise ValueError("global_batch_size must be at least the number of ranks")

    log_for_0(f"Config loaded from {args.config}")
    log_for_0(f"Model: {config.model} | encoder: {config.encoder_model_name} | max length: {config.max_length}")
    log_for_0(f"Samples: {config.num_samples} per setting, {len(config.sampling_configs)} sampling config(s), "
              f"seeds {parse_seeds(config.seed)}")
    log_for_0(f"Global batch size: {config.global_batch_size} | BF16: {config.use_bf16 and device.type == 'cuda'} | "
              f"torch.compile: {config.use_compile}")

    log_for_0("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name or config.encoder_model_name)
    pad_token_id = get_pad_token_id(tokenizer, config.pad_token)
    log_for_0(f"Using {'EOS' if config.pad_token == 'eos' else 'PAD'} token for padding: {pad_token_id}")

    eval_dataset = None
    if config.eval_data_path is not None:
        log_for_0("Loading dataset for conditional generation...")
        if config.eval_data_path.endswith(".jsonl"):
            eval_dataset = load_jsonl_dataset(config.eval_data_path, tokenizer)
        else:
            eval_dataset = load_dataset_split(config.eval_data_path, split=config.eval_data_split)
        log_for_0(f"Eval dataset size: {len(eval_dataset)}")

    init_wandb(config, job_type="eval")

    log_for_0(f"Loading Encoder: {config.encoder_model_name}...")
    encoder_config, encoder = get_encoder(config.encoder_model_name, torch.float32, config.encoder_family)
    encoder = encoder.to(device)
    resolve_latent_stats(config, device)

    log_for_0(f"Creating {config.model} model...")
    model = build_model(config, encoder_config.d_model, len(tokenizer)).to(device)
    log_for_0(f"Loading checkpoint from: {checkpoint_path}")
    metadata = load_pretrained_elf_weights(checkpoint_path, model)
    model.eval()
    state = SimpleNamespace(model=model, ema_params1=None, **metadata)

    run_generation(
        state=state, encoder=encoder, eval_dataset=eval_dataset,
        tokenizer=tokenizer, config=config, local_batch_size=local_batch_size,
    )
    finish_wandb()
    log_for_0("\nEvaluation complete!")


if __name__ == "__main__":
    main()
