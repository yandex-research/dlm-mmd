"""DMax OPUT inputs and objectives, with a native PyTorch training loop."""

import json
import math
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from mmd import MASK_TOKEN_ID, MMDLoss, evaluating
from vendor.dmax_data_transform import process_mdm_sft_example


def block_diffusion_mask(length, block_size, device=None):
    """[xt|x0]: xt sees its noisy block and strictly earlier clean blocks."""
    pos = torch.arange(2 * length, device=device)
    clean = pos >= length
    block = (pos % length) // block_size
    same_half_block = (clean[:, None] == clean[None, :]) & (block[:, None] == block[None, :])
    noisy_to_clean = (~clean[:, None]) & clean[None, :] & (block[:, None] > block[None, :])
    clean_causal = clean[:, None] & clean[None, :] & (block[:, None] >= block[None, :])
    return (same_half_block | noisy_to_clean | clean_causal)[None, None]


def attention_bias(mask):
    """Additive attention mask: 0 where attention is allowed, -inf elsewhere."""
    return torch.zeros_like(mask, dtype=torch.float32).masked_fill(~mask, float("-inf"))


class OPUTDataset(Dataset):
    """Original chat transform; raw trajectories are duplicated with both flags."""

    def __init__(self, records, tokenizer, max_seq_len=2048, noise_probability=0.75):
        self.records = records
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.noise_probability = noise_probability
        keys = records.column_names if hasattr(records, "column_names") else records[0].keys()
        self.prepared = "messages" in keys and "flag" in keys
        if not self.prepared and not {"question", "answer"}.issubset(keys):
            raise ValueError("Data needs question/answer or upstream messages/flag columns")

    def __len__(self):
        return len(self.records) * (1 if self.prepared else 2)

    def __getitem__(self, index):
        row = self.records[index if self.prepared else index // 2]
        if not self.prepared:
            row = {
                "messages": [
                    {"role": "user", "content": row["question"]},
                    {"role": "assistant", "content": row["answer"]},
                ],
                "flag": bool(index % 2),
            }
        result = process_mdm_sft_example(
            row,
            self.tokenizer,
            self.max_seq_len,
            noise_range=(self.noise_probability, self.noise_probability),
        )[0]
        # Upstream produces no labels for overlong prompts, then divides by zero.
        # Reject that input rather than silently changing the training population.
        if not (result["labels"] != -100).any():
            raise ValueError(f"Example {index} has no response targets within max_seq_len")
        return result


def load_training_data(config, tokenizer):
    """Load a Hub dataset, a `save_to_disk` directory, or a JSON/JSONL file."""
    from datasets import load_dataset, load_from_disk

    cfg = config.dataset
    path = Path(str(cfg.train_path)).expanduser()
    if path.is_dir():
        records = load_from_disk(str(path))
        if hasattr(records, "keys"):
            records = records[str(cfg.split)]
    elif path.is_file():
        records = load_dataset("json", data_files=str(path), split="train")
    else:
        records = load_dataset(str(cfg.train_path), split=str(cfg.split), revision=cfg.revision)
    return OPUTDataset(
        records,
        tokenizer,
        int(config.training.max_seq_len),
        float(config.training.noise_probability),
    )


def correction_loss(model, batch, attention, mmd):
    """Build original DMax OPUT inputs, then apply token RBF MMD.

    Each row contains a noisy half and a clean half. Masked-view rows supervise
    originally masked response tokens. Prediction-view rows first replace masks
    with detached argmax predictions and supervise every response token.
    """
    noisy, clean = batch["noisy_input_ids"], batch["input_ids"]
    prediction_view = batch["flag"].bool().reshape(-1)
    selected = (batch["labels"] != -100) & (prediction_view[:, None] | noisy.eq(MASK_TOKEN_ID))
    labels = batch["labels"].masked_fill(~selected, -100)
    length = noisy.shape[1]
    inputs = torch.cat((noisy, clean), dim=1)
    positions = (
        torch.arange(length, device=inputs.device).repeat(2)[None].expand(inputs.shape[0], -1)
    )

    # Both views execute this forward so distributed ranks keep the same number
    # of collectives. Only prediction-view rows consume its argmax tokens.
    with evaluating(model), torch.no_grad():
        rollout = model(
            input_ids=inputs,
            attention_mask=attention,
            position_ids=positions,
            logits_to_keep=(0, length),
        ).logits
        predicted = rollout[:, :length].argmax(dim=-1)
        del rollout
        replace = inputs[:, :length].eq(MASK_TOKEN_ID) & prediction_view[:, None]
        inputs[:, :length] = torch.where(replace, predicted, inputs[:, :length])

    return mmd(model, inputs, attention, positions, labels)


def start_wandb(config, context):
    """Optional DMax-style W&B run on rank 0."""
    if not config.training.use_wandb or context.rank != 0:
        return None
    import wandb
    from omegaconf import OmegaConf

    return wandb.init(
        project=str(config.training.wandb_project),
        name=str(config.training.wandb_name or Path(config.output_dir).name),
        config=OmegaConf.to_container(config, resolve=True),
    )


def train(config, model, tokenizer, context):
    """Run `max_steps` AdamW updates of the MMD loss, then export `output_dir/final_model`."""
    from distributed import clip_grad_norm, save_checkpoint
    from model_loading import load_feature_extractor

    cfg = config.training
    torch.manual_seed(int(cfg.seed) + context.rank)
    run = start_wandb(config, context)
    mmd = MMDLoss(config, load_feature_extractor(config, context))
    dataset = load_training_data(config, tokenizer)
    sampler = DistributedSampler(
        dataset,
        num_replicas=context.world_size,
        rank=context.rank,
        shuffle=True,
        seed=int(cfg.seed),
        drop_last=True,
    )
    if not len(sampler):
        raise ValueError("Training dataset must have at least one example per rank")
    loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        drop_last=True,
        num_workers=int(cfg.num_workers),
        pin_memory=context.device.type == "cuda",
    )
    global_batch = int(cfg.global_batch_size)
    if global_batch < context.world_size or global_batch % context.world_size:
        raise ValueError(
            "global_batch_size must be a positive multiple of world_size (microbatch=1)"
        )
    accumulation = global_batch // context.world_size

    opt = config.optimizer
    # Fused AdamW updates the FP32 masters and moments in place; the default
    # foreach implementation allocates an extra FP32 copy of the parameter shard.
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=float(opt.lr),
        betas=(float(opt.beta1), float(opt.beta2)),
        weight_decay=float(opt.weight_decay),
        **({"fused": True} if context.device.type == "cuda" else {}),
    )
    max_steps = int(cfg.max_steps)
    warmup = int(max_steps * float(opt.warmup_ratio))

    def lr_factor(step):
        if warmup and step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, max_steps - warmup)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)

    attention = attention_bias(
        block_diffusion_mask(int(cfg.max_seq_len), int(cfg.block_size), context.device)
    )
    epoch = 0
    iterator = iter(loader)

    def next_batch():
        nonlocal epoch, iterator
        try:
            batch = next(iterator)
        except StopIteration:
            epoch += 1
            sampler.set_epoch(epoch)
            iterator = iter(loader)
            batch = next(iterator)
        return {key: value.to(context.device, non_blocking=True) for key, value in batch.items()}

    model.train()
    for step in range(1, max_steps + 1):
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        total_loss = torch.zeros((), device=context.device)
        total_reward = torch.zeros((), device=context.device)
        total_reward_count = torch.zeros((), device=context.device)
        for _ in range(accumulation):
            batch = next_batch()
            # FSDP reduces each microstep to avoid retaining full unsharded gradients.
            # Equal microbatch and accumulation counts give the global example mean.
            amp = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if context.device.type == "cuda"
                else nullcontext()
            )
            with amp:
                loss = correction_loss(model, batch, attention, mmd) / accumulation
            loss.backward()
            total_loss += loss.detach()
            count = mmd.metrics["mmd_reward_count"]
            total_reward += mmd.metrics["mmd_reward"] * count
            total_reward_count += count
        norm = clip_grad_norm(model, float(cfg.max_grad_norm))
        lr = optimizer.param_groups[0]["lr"]
        optimizer.step()
        scheduler.step()
        if dist.is_initialized():
            dist.all_reduce(total_loss)
            total_loss /= context.world_size
            dist.all_reduce(total_reward)
            dist.all_reduce(total_reward_count)
        if context.rank == 0:
            record = {
                "step": step,
                "loss": total_loss.item(),
                "grad_norm": float(norm),
                "lr": lr,
                "sec_per_step": time.perf_counter() - started,
                "peak_mem_gib": (
                    torch.cuda.max_memory_allocated(context.device) / 2**30
                    if context.device.type == "cuda"
                    else 0.0
                ),
                "mmd_reward": (total_reward / total_reward_count.clamp_min(1)).item(),
            }
            print(json.dumps(record), flush=True)
            with (Path(config.output_dir) / "training.jsonl").open("a") as f:
                f.write(json.dumps(record) + "\n")
            if run is not None:
                run.log(
                    {f"training/{key}": value for key, value in record.items() if key != "step"},
                    step=step,
                )

    # Free the AdamW state before gathering the full model for export.
    optimizer.zero_grad(set_to_none=True)
    optimizer.state.clear()
    checkpoint = Path(config.output_dir) / "final_model"
    save_checkpoint(model, tokenizer, checkpoint, context)
    if run is not None:
        run.finish()
    return str(checkpoint)
