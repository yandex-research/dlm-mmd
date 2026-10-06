import copy
import itertools
import json
import os
import random
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from tqdm import tqdm

from configs.config import Config, SamplingConfig, parse_seeds
from utils.logging_utils import log_for_0
from utils.checkpoint_utils import upload_output_dir_to_hf
from utils.train_utils import unwrap_model
from utils.data_utils import get_dataloader, get_pad_token_id
from utils.encoder_utils import encode_text
from utils.sampling_utils import get_sampling_steps
from utils.generation_utils import (
    mask_after_eos, shift_left,
    _generate_samples_single_batch, _dlm_decode_batch,
    _build_run_name,
)
from utils.wandb_utils import log_wandb, log_wandb_samples

NUM_LOGGED_SAMPLES = 16  # Samples per sampling setting shown in W&B tables.


def _rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def _world() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def _build_eval_model(state, use_compile: bool = False) -> nn.Module:
    """Return an eval-mode model copy loaded with EMA params (if available)."""
    model = unwrap_model(state.model)
    eval_model = copy.deepcopy(model)
    if state.ema_params1:
        eval_model.load_state_dict(state.ema_params1)
    eval_model.eval()
    if use_compile:
        log_for_0("Compiling eval model with torch.compile (first batch will be slower)...")
        eval_model = torch.compile(eval_model)
    return eval_model


def _sampling_steps(sampling_config, num_sampling_steps, config, device, dtype):
    """Time grid for ODE/SDE sampling; iterative refinement has none."""
    if sampling_config.sampling_method == "iterative_refinement":
        return None
    return get_sampling_steps(
        n_steps=num_sampling_steps, time_schedule=sampling_config.time_schedule,
        P_mean=config.denoiser_p_mean, P_std=config.denoiser_p_std,
        device=device, dtype=dtype, shift=sampling_config.time_shift,
    )


def _write_jsonl(path, records, mode="w"):
    with open(path, mode, encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ============================================
# Generation Helper
# ============================================
def run_generation(
    state,
    encoder: nn.Module,
    eval_dataset,
    tokenizer,
    config,
    local_batch_size: int,
):
    """Sample every sampling setting for every seed, then summarize across seeds.

    Sampling runs without TF32; the caller's precision and NumPy/Python RNG state are restored.
    """
    seeds = parse_seeds(config.seed)
    model = _build_eval_model(state, use_compile=config.use_compile)
    device = next(model.parameters()).device
    cuda_devices = [device.index] if device.type == "cuda" else []
    numpy_state, python_state = np.random.get_state(), random.getstate()
    matmul_precision = torch.get_float32_matmul_precision()
    cudnn_allow_tf32 = torch.backends.cudnn.allow_tf32
    results_by_run = {}

    try:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cudnn.allow_tf32 = False
        with torch.random.fork_rng(devices=cuda_devices):
            for seed in seeds:
                # Per-rank offset so ranks generate different samples when sharding.
                rank_seed = seed + _rank() * 1_000_003
                generator = torch.Generator().manual_seed(rank_seed)
                torch.manual_seed(rank_seed)
                np.random.seed(rank_seed % 2 ** 32)
                random.seed(rank_seed)

                seed_config = copy.copy(config)
                seed_config.seed = seed
                if len(seeds) > 1:
                    seed_config.output_dir = os.path.join(config.output_dir, f"seed_{seed}")
                log_for_0(f"\nSampling seed: {seed}")
                for sc_idx, sc in enumerate(config.sampling_configs):
                    if len(config.sampling_configs) > 1:
                        log_for_0(f"\n--- Sampling config {sc_idx + 1}/{len(config.sampling_configs)} ---")
                    common_kwargs = dict(
                        model=model,
                        state=state,
                        tokenizer=tokenizer,
                        generator=generator,
                        config=seed_config,
                        sampling_config=sc,
                        batch_size=local_batch_size,
                        num_samples=config.num_samples,
                    )
                    if eval_dataset is None:
                        results = test_generation_uncond(**common_kwargs)
                    else:
                        results = test_generation_cond(
                            **common_kwargs, encoder=encoder, dataset=eval_dataset,
                        )
                    for name, (metrics, samples) in results.items():
                        results_by_run.setdefault(name, []).append((seed, metrics, samples))
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)
        torch.set_float32_matmul_precision(matmul_precision)
        torch.backends.cudnn.allow_tf32 = cudnn_allow_tf32

    if _rank() != 0:
        return {}
    summaries = {name: _summarize_seeds(name, runs, state, config) for name, runs in results_by_run.items()}
    for name, runs in results_by_run.items():
        _, _, samples = runs[0]
        if samples:
            log_wandb_samples(f"samples/{name}", list(samples[0]), [list(s.values()) for s in samples], state.step)
    upload_output_dir_to_hf(config.output_dir, config.hf_repo_id, reason="generation")
    return summaries


def _summarize_seeds(name, runs, state, config):
    """Write mean and population std (ddof=0) of each metric across seeds, and log them to W&B."""
    summary = {"epoch": int(state.epoch), "step": int(state.step), "seeds": [seed for seed, _, _ in runs]}
    wandb_metrics = {}
    for key in sorted({key for _, metrics, _ in runs for key in metrics}):
        values = [metrics[key] for _, metrics, _ in runs if metrics.get(key) is not None]
        summary[f"{key}_n_seeds"] = len(values)
        summary[f"{key}_mean"] = float(np.mean(values)) if values else None
        summary[f"{key}_std"] = float(np.std(values)) if values else None
        if values:
            log_for_0(f"{name}: {key} = {summary[f'{key}_mean']:.4f} "
                      f"± {summary[f'{key}_std']:.4f} ({len(values)} seeds)")
            wandb_metrics[f"eval/{name}/{key}"] = summary[f"{key}_mean"]
            wandb_metrics[f"eval/{name}/{key}_std"] = summary[f"{key}_std"]
    run_dir = os.path.join(config.output_dir, name)
    os.makedirs(run_dir, exist_ok=True)
    _write_jsonl(os.path.join(run_dir, "metrics_summary.jsonl"), [summary], mode="a")
    log_wandb(wandb_metrics, state.step)
    return summary


# ============================================
# Unconditional generation
# ============================================
def test_generation_uncond(
    model: nn.Module,
    state,
    tokenizer,
    generator: torch.Generator,
    config: Config,
    sampling_config: SamplingConfig,
    num_samples: int = 64,
    batch_size: int = 64,
):
    """Generate OWT samples and score them with GPT-2 Large perplexity and unigram entropy.

    Returns {run_name: (metrics, first samples)}; only rank 0 returns results.
    """
    sampling_method = sampling_config.sampling_method
    log_for_0(f"Config: {sampling_config}")

    log_for_0("\n" + "=" * 70)
    log_for_0("              UNCONDITIONAL GENERATION EXAMPLES")
    log_for_0("=" * 70)

    device = next(model.parameters()).device
    d_model = unwrap_model(model).text_encoder_dim
    log_for_0(f"Per-device batch size: {batch_size}")

    pad_token_id = get_pad_token_id(tokenizer, config.pad_token)
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 1

    cfg_list = [1]
    steps_list = sampling_config.num_sampling_steps
    self_cond_cfg_scales_list = sampling_config.self_cond_cfg_scales
    results = {}
    ppl_metrics = None
    if config.online_eval and _rank() == 0:
        from utils.metrics_utils import Metrics as PPLMetrics
        ppl_metrics = PPLMetrics(
            gen_ppl_eval_model_name_or_path=config.eval_ppl_model,
            eval_ppl_batch_size=config.eval_ppl_batch_size,
            eval_context_size=config.eval_ppl_max_length,
        )

    world = _world()
    rank = _rank()
    param_dtype = next(model.parameters()).dtype

    for num_sampling_steps, cfg_scale, self_cond_cfg_scale in itertools.product(
        steps_list, cfg_list, self_cond_cfg_scales_list
    ):
        log_for_0(f"\n--- Method: {sampling_method}, Steps: {num_sampling_steps}, "
                  f"CFG Scale: {cfg_scale}, SC-CFG: {self_cond_cfg_scale} ---")

        # Shard work across ranks: each rank generates ceil(num_samples/world);
        # the extras are trimmed after the gather on rank 0.
        local_num_samples = (num_samples + world - 1) // world
        local_generated = []
        generation_time = 0.0
        decode_time = 0.0
        num_batches = (local_num_samples + batch_size - 1) // batch_size
        local_processed = 0

        for batch_idx in tqdm(range(num_batches), desc="Generating samples", disable=(rank != 0)):
            if local_processed >= local_num_samples:
                break
            current_batch = min(batch_size, local_num_samples - local_processed)
            t_steps = _sampling_steps(sampling_config, num_sampling_steps, config, device, param_dtype)
            if device.type == "cuda":
                z = torch.randn(
                    (current_batch, config.max_length, d_model),
                    dtype=param_dtype, device=device,
                ) * config.denoiser_noise_scale
            else:
                z = (torch.randn((current_batch, config.max_length, d_model),
                                 generator=generator, dtype=param_dtype)
                     * config.denoiser_noise_scale).to(device)

            gen_start = time.time()
            latent = _generate_samples_single_batch(
                model=model, generator=generator, z=z,
                num_steps=num_sampling_steps, t_steps=t_steps,
                cond_seq=None, cond_seq_mask=None,
                config=config, sampling_config=sampling_config,
                cfg_scale=cfg_scale, self_cond_cfg_scale=self_cond_cfg_scale,
            )
            generation_time += time.time() - gen_start

            dec_start = time.time()
            predicted_ids = _dlm_decode_batch(
                z=latent, model=model,
                config=config, self_cond_cfg_scale=self_cond_cfg_scale,
            )
            decode_time += time.time() - dec_start

            # GPT-2 OWT sequences can start with BOS, which shares the EOS token.
            if config.encoder_family != "gpt2":
                predicted_ids = mask_after_eos(predicted_ids, eos_token_id=eos_token_id, pad_token_id=pad_token_id)

            for i in range(predicted_ids.shape[0]):
                if local_processed >= local_num_samples:
                    break
                text = tokenizer.decode(predicted_ids[i].detach().cpu().numpy(), skip_special_tokens=True)
                local_generated.append(text)
                local_processed += 1

        # Gather shards to rank 0, then assemble final ID-tagged list.
        if world > 1:
            gathered = [None] * world
            dist.all_gather_object(gathered, local_generated)
            local_generated = [text for shard in gathered for text in shard]
        log_for_0(f"Generation: {generation_time:.2f}s ({num_sampling_steps} steps) | Decode: {decode_time:.2f}s")
        log_for_0("-" * 70)
        if rank != 0:
            continue

        records = [{"id": i, "generated": text} for i, text in enumerate(local_generated[:num_samples])]
        name = _build_run_name(sampling_config, num_sampling_steps, cfg_scale, self_cond_cfg_scale, suffix="uncond")
        run_dir = os.path.join(config.output_dir, name)
        os.makedirs(run_dir, exist_ok=True)
        out_path = os.path.join(run_dir, f"all_generated_{int(state.epoch)}_{int(state.step)}.jsonl")
        _write_jsonl(out_path, records)
        log_for_0(f"Saved {len(records)} generated texts to {out_path}")

        metrics = {}
        if ppl_metrics is not None:
            log_for_0("\n" + "=" * 70)
            log_for_0("              PPL EVALUATION")
            log_for_0("=" * 70)
            ppl_metrics.reset()
            nonempty_samples = [r["generated"] for r in records if r["generated"].strip()]
            skipped = len(records) - len(nonempty_samples)
            if skipped > 0:
                log_for_0(f"PPL eval: skipped {skipped} empty samples")
            if not nonempty_samples:
                log_for_0("PPL eval: all samples empty; skipping perplexity computation")
            else:
                ppl_results = ppl_metrics.record_generative_perplexity(
                    text_samples=nonempty_samples, max_length=config.eval_ppl_max_length,
                )
                metrics = {"ppl": ppl_results["ppl"], "mean_entropy": ppl_results["mean_entropy"]}
                log_for_0(f"Perplexity: {metrics['ppl']:.4f}")
                log_for_0(f"Mean Entropy: {metrics['mean_entropy']:.4f}")
                _write_jsonl(os.path.join(run_dir, "metrics.jsonl"), [
                    {"epoch": int(state.epoch), "step": int(state.step), "seed": config.seed, **metrics},
                ], mode="a")
            log_for_0("=" * 70 + "\n")
        results[name] = (metrics, records[:NUM_LOGGED_SAMPLES])

    log_for_0("=" * 70 + "\n")
    return results


# ============================================
# Conditional generation
# ============================================
def test_generation_cond(
    model: nn.Module,
    state,
    encoder: nn.Module,
    tokenizer,
    generator: torch.Generator,
    config: Config,
    sampling_config: SamplingConfig,
    dataset,
    num_samples: int = 64,
    batch_size: int = 64,
):
    """Generate TinyGSM programs for the eval prompts and score answer accuracy.

    Returns {run_name: (metrics, first samples)}; only rank 0 returns results.
    """
    sampling_method = sampling_config.sampling_method
    log_for_0(f"Config: {sampling_config}")

    log_for_0("\n" + "=" * 70)
    log_for_0("              CONDITIONAL GENERATION EXAMPLES")
    log_for_0("=" * 70)

    device = next(model.parameters()).device
    param_dtype = next(model.parameters()).dtype
    d_model = unwrap_model(model).text_encoder_dim

    pad_token_id = get_pad_token_id(tokenizer, config.pad_token)
    eos_token_id = tokenizer.eos_token_id

    rank, world = _rank(), _world()
    sample_indices = list(range(min(num_samples, len(dataset))))[rank::world]
    dataloader = get_dataloader(
        torch.utils.data.Subset(dataset, sample_indices), batch_size=batch_size,
        shuffle=False, num_workers=0, drop_last=False,
        max_seq_length=config.max_length, pad_token_id=pad_token_id,
        max_input_seq_length=config.max_input_length, distributed=False,
        encoder_mask_2d=config.encoder_family == "gpt2",
    )

    results = {}
    cfg_list = sampling_config.cfgs
    steps_list = sampling_config.num_sampling_steps
    self_cond_cfg_scales_list = sampling_config.self_cond_cfg_scales

    for num_sampling_steps, cfg_scale, self_cond_cfg_scale in itertools.product(
        steps_list, cfg_list, self_cond_cfg_scales_list
    ):
        log_for_0(f"\n--- Method: {sampling_method}, Steps: {num_sampling_steps}, CFG Scale: {cfg_scale}, "
                  f"SC-CFG: {self_cond_cfg_scale} ---")

        all_generated = []
        generation_time = 0.0
        decode_time = 0.0

        pbar = tqdm(total=len(dataloader), desc="Generating samples (cond)", disable=(rank != 0))
        for batch in dataloader:
            bsz = batch["input_ids"].shape[0]
            input_ids = torch.from_numpy(batch["input_ids"]).to(device).long()
            encoder_attention_mask = torch.from_numpy(batch["encoder_attention_mask"]).to(device).float()
            cond_seq_mask_arr = torch.from_numpy(batch["cond_seq_mask"]).to(device).float()
            t_steps = _sampling_steps(sampling_config, num_sampling_steps, config, device, param_dtype)

            cond_seq = encode_text(
                input_ids=input_ids, attention_mask=encoder_attention_mask,
                encoder=encoder, latent_mean=config.latent_mean, latent_std=config.latent_std,
                use_bf16=config.use_bf16,
            ).to(param_dtype)

            z = (torch.randn((bsz, config.max_length, d_model),
                             generator=generator, dtype=param_dtype)
                 * config.denoiser_noise_scale).to(device)

            gen_start = time.time()
            latent = _generate_samples_single_batch(
                model=model, generator=generator, z=z,
                num_steps=num_sampling_steps, t_steps=t_steps,
                cond_seq=cond_seq, cond_seq_mask=cond_seq_mask_arr,
                config=config, sampling_config=sampling_config,
                cfg_scale=cfg_scale, self_cond_cfg_scale=self_cond_cfg_scale,
            )
            generation_time += time.time() - gen_start

            cond_len_per_sample = cond_seq_mask_arr.to(torch.int32).sum(dim=1)

            dec_start = time.time()
            predicted_ids = _dlm_decode_batch(
                z=latent, model=model,
                config=config, self_cond_cfg_scale=self_cond_cfg_scale,
            )
            predicted_ids = shift_left(predicted_ids, cond_len_per_sample, pad_token_id)
            predicted_ids = mask_after_eos(predicted_ids, eos_token_id=eos_token_id, pad_token_id=pad_token_id)
            decode_time += time.time() - dec_start

            for i in range(bsz):
                all_generated.append({
                    "id": sample_indices[len(all_generated)],
                    "input": batch["input"][i],
                    "target": batch["target"][i],
                    "generated": tokenizer.decode(predicted_ids[i].detach().cpu().numpy(), skip_special_tokens=True),
                })
            pbar.update(1)
        pbar.close()
        if world > 1:
            gathered = [None] * world
            dist.all_gather_object(gathered, all_generated)
            all_generated = sorted((r for shard in gathered for r in shard), key=lambda r: r["id"])

        log_for_0(f"Generation: {generation_time:.2f}s ({num_sampling_steps} steps) | Decode: {decode_time:.2f}s")
        log_for_0("-" * 70)
        if rank != 0:
            continue

        name = _build_run_name(sampling_config, num_sampling_steps, cfg_scale, self_cond_cfg_scale, suffix="cond")
        run_dir = os.path.join(config.output_dir, name)
        os.makedirs(run_dir, exist_ok=True)
        out_path = os.path.join(run_dir, f"all_generated_{int(state.epoch)}_{int(state.step)}.jsonl")
        _write_jsonl(out_path, all_generated)
        log_for_0(f"Saved {len(all_generated)} generated texts to {out_path}")

        metrics = {}
        if config.online_eval and all_generated:
            from utils.gsm_exec import compute_accuracy
            metrics = compute_accuracy(
                [r["generated"] for r in all_generated], [r["target"] for r in all_generated],
            )
            log_for_0(f"Accuracy: {metrics['accuracy']:.2f}")
            _write_jsonl(os.path.join(run_dir, "metrics.jsonl"), [
                {"epoch": int(state.epoch), "step": int(state.step), "seed": config.seed, **metrics},
            ], mode="a")
        results[name] = (metrics, all_generated[:NUM_LOGGED_SAMPLES])

    log_for_0("=" * 70 + "\n")
    return results
