from typing import Optional

import torch
import torch.nn as nn

from configs.config import Config, SamplingConfig
from utils.sampling_utils import restore_cond, _ode_step, _sde_step, iterative_refinement


# ============================================
# Generation utilities
# ============================================

def mask_after_eos(predicted_ids: torch.Tensor, eos_token_id: int, pad_token_id: int) -> torch.Tensor:
    """Mask everything at/after first EOS token per sequence."""
    eos_mask = (predicted_ids == eos_token_id)
    keep_mask = (eos_mask.to(torch.int32).cumsum(dim=1) == 0)
    return torch.where(keep_mask, predicted_ids, torch.full_like(predicted_ids, pad_token_id))


def shift_left(x: torch.Tensor, shift_per_sample: torch.Tensor, pad_value=0) -> torch.Tensor:
    """Shift each (batch, length) row left by its own amount; pad emptied positions."""
    seq_len = x.shape[1]
    gather_idx = shift_per_sample.to(device=x.device, dtype=torch.long)[:, None] + torch.arange(seq_len, device=x.device)
    shifted = torch.gather(x, 1, gather_idx.clamp(max=seq_len - 1))
    return torch.where(gather_idx < seq_len, shifted, torch.full_like(shifted, pad_value))


# ============================================
# Single-batch sampling (PyTorch)
# ============================================

@torch.no_grad()
def _generate_samples_single_batch(
    model: nn.Module,
    generator: torch.Generator,
    z: torch.Tensor,
    num_steps: int,
    t_steps: Optional[torch.Tensor],
    cond_seq: Optional[torch.Tensor],
    cond_seq_mask: Optional[torch.Tensor],
    config: Config,
    sampling_config: SamplingConfig,
    cfg_scale: float,
    self_cond_cfg_scale: float,
) -> torch.Tensor:
    """Generate samples for a single batch (iterative refinement, or Euler / SDE rollout)."""
    method = sampling_config.sampling_method
    if method == "iterative_refinement":
        return iterative_refinement(
            model, z, cond_seq, cond_seq_mask, config, num_steps,
            self_cond_cfg_scale=self_cond_cfg_scale, cfg_scale=cfg_scale,
            resample_z=sampling_config.resample_z, generator=generator,
        )
    batch_size, max_length, d_model = z.shape
    if cond_seq is None:
        cond_seq = torch.zeros((batch_size, max_length, d_model), dtype=z.dtype, device=z.device)
        cond_seq_mask = torch.zeros((batch_size, max_length), dtype=z.dtype, device=z.device)

    step_kwargs = dict(
        model=model, config=config,
        cfg_scale=cfg_scale, self_cond_cfg_scale=self_cond_cfg_scale,
        cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
    )

    z = restore_cond(z, cond_seq, cond_seq_mask)
    x_pred = restore_cond(torch.zeros_like(z), cond_seq, cond_seq_mask)

    n = t_steps.shape[0]
    sde_gamma = sampling_config.sde_gamma

    use_bf16 = config.use_bf16 and z.is_cuda
    with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
        for i in range(n - 2):
            t = t_steps[i].item()
            t_next = t_steps[i + 1].item()
            if method == "sde":
                z, x_pred = _sde_step(
                    z=z, t=t, t_next=t_next, x_pred_prev=x_pred,
                    gamma=sde_gamma, generator=generator, **step_kwargs,
                )
            elif method == "ode":
                z, x_pred = _ode_step(z=z, t=t, t_next=t_next, x_pred_prev=x_pred, **step_kwargs)
            else:
                raise ValueError(f"Invalid sampling method: {method}")

        # Last step always with ODE.
        t = t_steps[-2].item()
        t_next = t_steps[-1].item()
        z, x_pred = _ode_step(z=z, t=t, t_next=t_next, x_pred_prev=x_pred, **step_kwargs)
    return z


@torch.no_grad()
def _dlm_decode_batch(z: torch.Tensor, model: nn.Module, config, self_cond_cfg_scale: float) -> torch.Tensor:
    """Decode clean latents z -> tokens with the DLM decoder head (at t=1)."""
    batch_size = z.shape[0]
    t_final = torch.ones((batch_size,), dtype=z.dtype, device=z.device)
    sc_batch = torch.full((batch_size,), float(self_cond_cfg_scale), dtype=z.dtype, device=z.device)
    z_input = torch.cat([z, torch.zeros_like(z)], dim=-1)
    use_bf16 = config.use_bf16 and z.is_cuda
    with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
        _, decoder_logits = model(
            z_input, t_final, deterministic=True,
            self_cond_cfg_scale=sc_batch,
            decoder_step_active=True,
        )
    return decoder_logits.argmax(dim=-1)


def _build_run_name(sampling_config, num_sampling_steps, cfg_scale, self_cond_cfg_scale, suffix):
    """Name the output directory of one sampling setting, e.g. `sde-steps32-cfg1-sccfg3-ts_logit_normal-gamma1.5-uncond`."""
    method = sampling_config.sampling_method
    name = f"{method}-steps{num_sampling_steps}-cfg{cfg_scale}"
    if self_cond_cfg_scale != 1.0:
        name += f"-sccfg{self_cond_cfg_scale}"
    if sampling_config.resample_z:
        name += "-resample_z"
    if method in ("ode", "sde"):
        name += f"-ts_{sampling_config.time_schedule}"
        if sampling_config.time_schedule == "shift":
            name += f"{sampling_config.time_shift:g}"
    if method == "sde":
        name += f"-gamma{sampling_config.sde_gamma}"
    return f"{name}-{suffix}"
