import logging
import os
import random
import re
from typing import Any, Optional, Tuple

import torch
import numpy as np

from utils.logging_utils import log_for_0, _process_index
from utils.train_utils import unwrap_model


def _local_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def upload_output_dir_to_hf(output_dir: str, hf_repo_id: Optional[str], reason: str = "artifacts"):
    if not hf_repo_id or _process_index() != 0:
        return
    folder_path = _local_path(output_dir)
    if not os.path.isdir(folder_path):
        log_for_0(f"HF upload skipped; output directory does not exist: {folder_path}",
                  level=logging.WARNING)
        return
    try:
        from huggingface_hub import HfApi
        repo_id = hf_repo_id.strip("/")
        api = HfApi()
        api.create_repo(repo_id, repo_type="model", exist_ok=True)
        log_for_0(f"Uploading {reason} to HF: {repo_id}")
        api.upload_folder(repo_id=repo_id, folder_path=folder_path, repo_type="model")
        log_for_0(f"Uploaded {reason} to HF: {repo_id}")
    except Exception as e:
        log_for_0(f"Failed to upload {reason} to HF: {e}", level=logging.WARNING)


def _split_hf_path(path: str, min_parts: int) -> Optional[Tuple[str, str]]:
    if "://" in path:
        return None
    if path.startswith(("/", ".", "~")):
        return None
    if os.path.exists(_local_path(path)):
        return None
    parts = path.split("/")
    if len(parts) < min_parts:
        return None
    return "/".join(parts[:2]), "/".join(parts[2:])


def _gather_optimizer_state(optimizer):
    """Collect Muon momentum from its owning ranks; Adam state is replicated."""
    state_dict = optimizer.state_dict()
    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    if not distributed or not any(g.get("use_muon") for g in state_dict["param_groups"]):
        return state_dict

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    owned_states = {}
    for group in state_dict["param_groups"]:
        if not group.get("use_muon"):
            continue
        # A resumed rank may also hold stale copies of another rank's momentum.
        for param_id in group["params"][rank::world_size]:
            if param_id in state_dict["state"]:
                owned_states[param_id] = {
                    key: value.detach().cpu() if torch.is_tensor(value) else value
                    for key, value in state_dict["state"][param_id].items()
                }
    gathered_states = [None] * world_size if rank == 0 else None
    torch.distributed.gather_object(owned_states, gathered_states, dst=0)
    if rank == 0:
        for group in state_dict["param_groups"]:
            if group.get("use_muon"):
                for param_id in group["params"]:
                    state_dict["state"].pop(param_id, None)
        for rank_states in gathered_states:
            state_dict["state"].update(rank_states)
    return state_dict


def save_checkpoint(state, output_dir: str, step: int, hf_repo_id: str = None):
    """Save model, optimizer, and per-rank RNG state; call on every training rank."""
    rng_state = {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    if distributed:
        rng_states = [None] * torch.distributed.get_world_size() if _process_index() == 0 else None
        torch.distributed.gather_object(rng_state, rng_states, dst=0)
    else:
        rng_states = [rng_state]
    optimizer_state = _gather_optimizer_state(state.optimizer)
    if _process_index() != 0:
        return
    ckpt_dir = _local_path(output_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    inner_model = unwrap_model(state.model)
    payload = {
        "params": inner_model.state_dict(),
        "ema_params1": state.ema_params1,
        "opt_state": optimizer_state,
        "lr_scheduler": state.lr_scheduler.state_dict() if state.lr_scheduler is not None else None,
        "step": int(state.step),
        "epoch": int(state.epoch),
        "rng_states": rng_states,
    }
    out_path = os.path.join(ckpt_dir, f"checkpoint_{step}")
    log_for_0(f"Saving checkpoint to {out_path}")
    torch.save(payload, out_path)
    log_for_0(f"Checkpoint written to {out_path}")
    upload_output_dir_to_hf(output_dir, hf_repo_id, reason="checkpoint")


def _checkpoint_step(checkpoint_name: str) -> int:
    """Extract the trailing checkpoint step from a name; -1 if absent."""
    match = re.search(r"(\d+)$", checkpoint_name)
    return int(match.group(1)) if match else -1


def find_latest_checkpoint(ckpt_dir: str):
    """Return the local `checkpoint_<step>` path with the highest step, or None."""
    ckpt_dir = _local_path(ckpt_dir)
    if not os.path.isdir(ckpt_dir):
        return None
    names = [f for f in os.listdir(ckpt_dir) if f.startswith("checkpoint_")]
    return os.path.join(ckpt_dir, max(names, key=_checkpoint_step)) if names else None


def _download_hf_checkpoint(checkpoint_path: str) -> Optional[str]:
    """Download `org/repo[/sub_path]` from the HF Hub; sub_path may be a file or a folder."""
    hf_path = _split_hf_path(checkpoint_path, min_parts=2)
    if hf_path is None:
        return None
    repo_id, sub_path = hf_path
    from huggingface_hub import snapshot_download
    log_for_0(f"Downloading checkpoint from HF: {repo_id}" + (f"/{sub_path}" if sub_path else ""))
    local_dir = snapshot_download(
        repo_id=repo_id, repo_type="model",
        allow_patterns=[sub_path, f"{sub_path}/**"] if sub_path else None,
    )
    return os.path.join(local_dir, sub_path) if sub_path else local_dir


def _restore_checkpoint(checkpoint_path: str) -> Any:
    """Restore a checkpoint from a file or directory (latest inside dir)."""
    local = _local_path(checkpoint_path)
    resolved = local
    if os.path.isdir(local):
        latest = find_latest_checkpoint(local)
        if latest is not None and os.path.isfile(latest):
            resolved = latest
    if os.path.isfile(resolved):
        return torch.load(resolved, map_location="cpu", weights_only=False)
    return None


def _validate_checkpoint(ckpt: Any):
    if ckpt is None:
        raise ValueError("checkpoint restore returned None")
    required_keys = ("params", "opt_state", "step", "epoch")
    missing_keys = [key for key in required_keys if key not in ckpt]
    if missing_keys:
        raise ValueError(f"checkpoint restore missing keys: {missing_keys}")


def load_checkpoint(checkpoint_path: str, state) -> Tuple[Any, int]:
    """Load an ELF checkpoint.

    Uses an existing local path first; otherwise tries HF and then local fallback.
    """
    log_for_0(f"Loading ELF checkpoint from {checkpoint_path}...")
    ckpt, loaded_from = None, None
    errors = []

    local_path = _local_path(checkpoint_path)
    if os.path.exists(local_path):
        try:
            log_for_0(f"Loading local checkpoint from {local_path}...")
            ckpt = _restore_checkpoint(local_path)
            _validate_checkpoint(ckpt)
            loaded_from = "local"
        except Exception as e:
            errors.append(f"local: {e}")

    if ckpt is None:
        try:
            hf_path = _download_hf_checkpoint(checkpoint_path)
            if hf_path:
                log_for_0(f"Loading HF checkpoint from {hf_path}...")
                ckpt = _restore_checkpoint(hf_path)
                _validate_checkpoint(ckpt)
                loaded_from = "HF"
        except Exception as e:
            errors.append(f"HF: {e}")

    if ckpt is None:
        raise ValueError(
            f"Failed to load checkpoint from {checkpoint_path}. Tried: {'; '.join(errors)}"
        )

    log_for_0(f"Loaded checkpoint keys: {list(ckpt.keys())}")

    rng_states = ckpt.get("rng_states")
    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    world_size = torch.distributed.get_world_size() if distributed else 1
    if rng_states is not None and len(rng_states) != world_size:
        raise ValueError(
            f"Checkpoint has RNG states for {len(rng_states)} ranks; resume requires {world_size}"
        )

    inner_model = unwrap_model(state.model)
    inner_model.load_state_dict(ckpt["params"])
    ema_src = ckpt.get("ema_params1", ckpt["params"])
    device_map = {n: p.device for n, p in inner_model.named_parameters()}
    for n, b in inner_model.named_buffers():
        device_map.setdefault(n, b.device)
    fallback_device = next(iter(device_map.values()), torch.device("cpu"))
    state.ema_params1 = {
        n: t.to(device_map.get(n, fallback_device)) for n, t in ema_src.items()
    }
    optimizer_state = ckpt["opt_state"]
    missing_muon = sum(
        param_id not in optimizer_state["state"]
        for group in optimizer_state["param_groups"] if group.get("use_muon")
        for param_id in group["params"]
    )
    if missing_muon and ckpt["step"] > 0:
        log_for_0(
            f"Checkpoint is missing {missing_muon} Muon momentum states; "
            "those states will restart on resume.",
            level=logging.WARNING,
        )
    state.optimizer.load_state_dict(optimizer_state)
    if state.lr_scheduler is not None and ckpt.get("lr_scheduler") is not None:
        state.lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
    state.step = int(ckpt["step"])
    state.epoch = int(ckpt["epoch"])
    if rng_states is not None:
        rng = rng_states[_process_index() if distributed else 0]
        torch.set_rng_state(rng["torch"])
        if torch.cuda.is_available() and rng["cuda"] is not None:
            torch.cuda.set_rng_state(rng["cuda"])
        np.random.set_state(rng["numpy"])
        random.setstate(rng["python"])

    step = int(ckpt["step"])
    log_for_0(f"Loaded {loaded_from} checkpoint from step {step} (epoch {state.epoch})")
    return state, step


def load_pretrained_elf_weights(checkpoint_path: str, model) -> dict:
    """Load ELF weights without training state and return the saved {"step", "epoch"}.

    `checkpoint_path` may be a checkpoint file, a directory (ckpt.pt, or its latest
    `checkpoint_<step>`) or a Hugging Face model repo with an optional
    sub-path, e.g. `embedded-language-flows/ELF-B-owt-torch`. EMA weights are
    laid over the raw ones, and loading is strict: the configured architecture
    must match the checkpoint. Plain tensor state_dict exports are also accepted;
    they already contain inference weights and have no training step metadata.
    """
    resolved = _local_path(checkpoint_path)
    if not os.path.exists(resolved):
        resolved = _download_hf_checkpoint(checkpoint_path)
    if resolved is not None and os.path.isdir(resolved):
        export_path = os.path.join(resolved, "ckpt.pt")
        resolved = export_path if os.path.isfile(export_path) else find_latest_checkpoint(resolved)
    if resolved is None or not os.path.isfile(resolved):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path} (local path or HF repo)")

    checkpoint = torch.load(resolved, map_location="cpu", weights_only=False)
    if (isinstance(checkpoint, dict) and checkpoint
            and all(isinstance(key, str) and torch.is_tensor(value)
                    for key, value in checkpoint.items())):
        weights = checkpoint
        metadata = {"step": 0, "epoch": 0}
    else:
        weights = dict(checkpoint["params"])
        weights.update(checkpoint["ema_params1"])
        metadata = {"step": int(checkpoint.get("step", 0)), "epoch": int(checkpoint.get("epoch", 0))}
    unwrap_model(model).load_state_dict(weights, strict=True)
    log_for_0(f"Loaded {len(weights)} ELF tensors from {resolved}")
    return metadata
