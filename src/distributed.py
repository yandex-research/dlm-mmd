"""Distributed setup for torchrun launches, FSDP wrapping and checkpoint export."""

import functools
import json
import os
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist


@dataclass
class DistributedContext:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    def barrier(self):
        if dist.is_initialized():
            dist.barrier()


def initialize_distributed():
    """Read the torchrun environment. On CUDA every run, even with one GPU, uses FSDP.

    Without CUDA a single plain process runs (useful for small CPU checks).
    """
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        # Generous timeout: on a first run the other ranks wait while rank 0
        # downloads the 16B checkpoint from the Hugging Face Hub.
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world,
            timeout=timedelta(hours=2),
            device_id=device,
        )
        torch.backends.cuda.matmul.allow_tf32 = True
    elif world > 1:
        raise RuntimeError("Multi-process training requires CUDA")
    return DistributedContext(rank, world, local, device)


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def unwrap_model(model):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    while isinstance(model, FSDP):
        model = model.module
    return model


def distribute(model, context):
    """Shard with FSDP on CUDA; on CPU, simply move the model to the device."""
    if context.device.type != "cuda":
        return model.to(context.device)
    from torch.distributed.fsdp import BackwardPrefetch, MixedPrecision, ShardingStrategy
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    from vendor.llada2.modeling_llada2_moe import LLaDA2MoeDecoderLayer

    # Rank 0 holds the loaded weights; the other ranks hold meta tensors that
    # FSDP materializes and fills from rank 0 (sync_module_states).
    def materialize(module):
        module.to_empty(device=context.device, recurse=False)

    wrapped = FSDP(
        model,
        auto_wrap_policy=functools.partial(
            transformer_auto_wrap_policy, transformer_layer_cls={LLaDA2MoeDecoderLayer}
        ),
        sharding_strategy=ShardingStrategy.HYBRID_SHARD,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=None,
            # Preserve integer token/position IDs and the explicit attention mask.
            cast_root_forward_inputs=False,
        ),
        use_orig_params=True,
        device_id=context.device,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        limit_all_gathers=True,
        sync_module_states=True,
        param_init_fn=materialize,
    )
    # The router's expert_bias buffer must come from rank 0 on every rank.
    for buffer in wrapped.buffers():
        dist.broadcast(buffer, src=0)
    # RoPE must remain fp32: bf16 rounding of inv_freq changes long-context positions.
    unwrap_model(wrapped).model.rotary_emb.rebuild(context.device)
    return wrapped


def clip_grad_norm(model, max_norm):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    if isinstance(model, FSDP):
        return model.clip_grad_norm_(max_norm)
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def save_checkpoint(model, tokenizer, path, context):
    """Collectively export a complete original-HF-layout model (no optimizer)."""
    from torch.distributed.fsdp import FullStateDictConfig, StateDictType
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    from vendor.llada2.grouped_moe import reference_state_dict

    inner = unwrap_model(model)
    if isinstance(model, FSDP):
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            state = model.state_dict()
    else:
        state = inner.state_dict()
    if context.rank == 0:
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        state = {key.replace("_fsdp_wrapped_module.", ""): value for key, value in state.items()}
        # Compute-dtype export is what upstream decoder expects; optimizer
        # master precision remains in memory during training.
        buffers = {name.replace("_fsdp_wrapped_module.", "") for name, _ in inner.named_buffers()}
        state = {
            key: value.detach().to(device="cpu", dtype=torch.bfloat16)
            if value.is_floating_point() and key not in buffers
            else value.detach().cpu()
            for key, value in state.items()
        }
        state = reference_state_dict(state)
        # Transformers 5 may consume/pop entries from the supplied dict.
        export_keys = list(state)
        export_bytes = sum(t.numel() * t.element_size() for t in state.values())
        inner.save_pretrained(
            target, state_dict=state, safe_serialization=True, max_shard_size="4GB"
        )
        # Original dInfer's loader requires an index even for a single shard.
        index_path = target / "model.safetensors.index.json"
        if (target / "model.safetensors").is_file():
            index_path.write_text(
                json.dumps(
                    {
                        "metadata": {"total_size": export_bytes},
                        "weight_map": {key: "model.safetensors" for key in export_keys},
                    },
                    indent=2,
                )
                + "\n"
            )
        tokenizer.save_pretrained(target)
        # Transformers >= 5 saves the class as "TokenizersBackend", which the evaluation
        # environment (transformers 4.57) cannot load; the tokenizer itself is unchanged.
        tokenizer_config = target / "tokenizer_config.json"
        settings = json.loads(tokenizer_config.read_text())
        if settings.get("tokenizer_class") == "TokenizersBackend":
            settings["tokenizer_class"] = "PreTrainedTokenizerFast"
            tokenizer_config.write_text(json.dumps(settings, indent=2, ensure_ascii=False) + "\n")
    context.barrier()
    return str(path)
