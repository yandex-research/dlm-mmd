from pathlib import Path

import torch
import numpy as np


def get_encoder(model_name, dtype=torch.float32, encoder_family="t5"):
    """Load a frozen T5 or GPT-2 encoder and return its config and module."""
    if encoder_family == "gpt2":
        from modules.gpt2_encoder import GPT2Encoder

        encoder = GPT2Encoder(model_name, dtype=dtype)
        config = encoder.config
    else:
        from modules.t5_encoder import get_encoder as get_t5_encoder

        config, encoder = get_t5_encoder(model_name, dtype)
    encoder.requires_grad_(False).eval()
    return config, encoder


@torch.no_grad()
def encode_text(
    input_ids,
    attention_mask,
    encoder,
    latent_mean,
    latent_std,
    use_bf16=True,
):
    """Encoder pass from text to latent with normalization."""
    autocast_enabled = bool(use_bf16) and input_ids.is_cuda
    with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=autocast_enabled):
        latents = encoder(input_ids=input_ids, attention_mask=attention_mask, deterministic=True)
    if torch.is_tensor(latent_mean):
        latent_mean = latent_mean.to(device=latents.device, dtype=latents.dtype)
    if torch.is_tensor(latent_std):
        latent_std = latent_std.to(device=latents.device, dtype=latents.dtype)
    return (latents - latent_mean) / latent_std


def load_latent_stats(path, max_length, device):
    """Load local or org/repo/file statistics, with optional position-zero overrides."""
    if isinstance(path, str) and (
        path.startswith("hf://") or (
            not path.startswith(("/", ".", "~")) and "://" not in path
            and path.count("/") >= 2 and not Path(path).exists()
        )
    ):
        parts = path.removeprefix("hf://").split("/", 2)
        if len(parts) != 3 or not all(parts) or parts[2].endswith("/"):
            raise ValueError("Hugging Face statistics paths must use org/repo/path/to/file")
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id="/".join(parts[:2]), filename=parts[2], repo_type="model")
    else:
        path = Path(path).expanduser()
    stats = torch.load(path, map_location="cpu", weights_only=True)
    mean = torch.as_tensor(stats["mean"], dtype=torch.float32)
    std = torch.as_tensor(stats["std"], dtype=torch.float32)
    if "mean_pos0" in stats:
        mean = mean.expand(max_length, -1).clone()
        std = std.expand(max_length, -1).clone()
        mean[0] = torch.as_tensor(stats["mean_pos0"], dtype=torch.float32)
        std[0] = torch.as_tensor(stats["std_pos0"], dtype=torch.float32)
    return mean.to(device), std.to(device)


def resolve_latent_stats(config, device):
    """Replace a statistics path in `config.latent_mean` with the loaded mean and std tensors."""
    if isinstance(config.latent_mean, str):
        config.latent_mean, config.latent_std = load_latent_stats(config.latent_mean, config.max_length, device)


def build_self_attn_cond_masks(is_cond, is_valid):
    """Build self-attention conditioning masks from cond/valid token flags."""
    encoder_attention_mask = (
        (is_cond[:, :, None] & is_cond[:, None, :]) |
        (~is_cond[:, :, None] & is_valid[:, None, :])
    ).astype(np.float32)
    attention_mask = is_valid.astype(np.float32)
    cond_seq_mask = is_cond.astype(np.float32)
    return encoder_attention_mask, attention_mask, cond_seq_mask
