"""Post-training steps: MMD and iterative refinement distillation (IRD).

A loaded batch holds distillation rows followed by decoder rows. Both
objectives make one generator pass over the whole batch: distillation rows at
t=0, and decoder rows at t=1, where the token decoder is trained with
cross-entropy. The two losses are weighted by their row counts.
"""

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from configs.config import parse_seeds
from utils.encoder_utils import encode_text
from utils.mmd_utils import rbf_mmd
from utils.sampling_utils import add_noise, iterative_refinement, restore_cond, sample_cfg_scale


# ============================================
# Shared by both objectives
# ============================================

def encode_batch(model, encoder, batch, config, repeats=1):
    """Encode tokens to normalized latents and sample SC-CFG scales.

    TinyGSM MMD loads one reference per prompt; `repeats` copies it once per
    generated response. Decoder rows are never repeated.
    """
    dtype = next(model.parameters()).dtype
    ids = batch["input_ids"].long()
    cond_mask = batch["cond_seq_mask"].float()
    attention_mask = batch["attention_mask"].float()
    clean = encode_text(
        ids, batch["encoder_attention_mask"].float(), encoder,
        config.latent_mean, config.latent_std, use_bf16=config.use_bf16,
    ).to(dtype)
    sc_cfg = sample_cfg_scale(
        ids.shape[0], config.self_cond_cfg_min, config.self_cond_cfg_max,
        dtype=dtype, device=ids.device,
    )
    tensors = (clean, ids, cond_mask, attention_mask, sc_cfg)
    if repeats > 1:
        num_references = config.distill_batch_size // repeats
        tensors = tuple(
            torch.cat((x[:num_references].repeat_interleave(repeats, dim=0), x[num_references:]))
            for x in tensors
        )
    return tensors


def sample_noise(clean, cond_mask, config):
    """Starting noise for every row, with clean prompt tokens."""
    noise = torch.randn_like(clean) * config.denoiser_noise_scale
    return restore_cond(noise, clean, cond_mask)


def generator_forward(model, clean, z, self_cond, sc_cfg, config):
    """Predict latents for the distillation rows and token logits for the decoder rows.

    `z` and `self_cond` cover the distillation rows; decoder rows get noised
    clean latents at a per-token logit-normal level and no self-conditioning.
    """
    n = z.shape[0]
    decoder_clean = clean[n:]
    mix = torch.sigmoid(torch.randn_like(decoder_clean[:, :, :1]) * config.decoder_p_std + config.decoder_p_mean)
    decoder_z = mix * decoder_clean + (1 - mix) * torch.randn_like(decoder_clean) * config.decoder_noise_scale
    z = torch.cat((z, decoder_z))
    self_cond = torch.cat((self_cond, torch.zeros_like(decoder_clean)))
    # Decoder rows run at t=1 with the model-mode tokens switched on.
    is_decoder_row = (torch.arange(clean.shape[0], device=clean.device) >= n).to(clean.dtype)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=config.use_bf16 and clean.is_cuda):
        prediction, logits = model(
            torch.cat((z, self_cond), dim=-1), is_decoder_row,
            deterministic=False, self_cond_cfg_scale=sc_cfg, decoder_step_active=is_decoder_row,
        )
    return prediction[:n], logits[n:]


def decoder_ce_loss(logits, ids, cond_mask, attention_mask, config):
    """Mean token cross-entropy over response positions (and non-PAD ones when padding with PAD)."""
    token_ce = F.cross_entropy(logits.float().transpose(1, 2), ids, reduction="none")
    mask = 1 - cond_mask
    if config.pad_token == "pad":
        mask = mask * attention_mask
    return (token_ce * mask).sum() / mask.sum().clamp_min(1)


def weighted_loss(distill_loss, ce_loss, num_distill, num_decoder):
    return (distill_loss * num_distill + ce_loss * num_decoder) / (num_distill + num_decoder)


# ============================================
# MMD
# ============================================

def sample_bootstrap_steps(config, step):
    """0 passes with probability no_bootstrap_prob, else uniform in 1..max_bootstrap_steps.

    Seeded by the step, so every rank draws the same count and a resumed run repeats it.
    """
    rng = torch.Generator().manual_seed(parse_seeds(config.seed)[0] * 1_000_003 + step)
    if torch.rand((), generator=rng).item() < config.no_bootstrap_prob:
        return 0
    return torch.randint(1, config.max_bootstrap_steps + 1, (), generator=rng).item()


def mmd_loss(feature_model, real, fake, noise, cond_mask, sc_cfg, config, conditional):
    """MMD between frozen-model features of real and generated latents noised to t = 1 - t_eps.

    Generated latents reuse the generator's input `noise`; real latents get fresh noise.
    Each kernel row holds mmd_batch_size generated samples: consecutive OWT
    sequences, or the responses to one TinyGSM prompt, whose reference is kept once.
    """
    group = config.mmd_batch_size
    fake_mask = cond_mask
    t = real.new_full((real.shape[0],), 1 - config.t_eps)
    fake_noised = restore_cond(t[:, None, None] * fake + config.t_eps * noise, real, cond_mask)
    if conditional:
        real, cond_mask, real_sc_cfg = real[::group], cond_mask[::group], sc_cfg[::group]
    else:
        real_sc_cfg = sc_cfg
    real_noised = add_noise(
        real, torch.randn_like(real), t[:real.shape[0]], config, cond_seq_mask=cond_mask.unsqueeze(-1),
    )

    def features(z, self_cond, scale):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=config.use_bf16 and z.is_cuda):
            return feature_model.forward_features(
                torch.cat((z, self_cond), dim=-1), t[:z.shape[0]], config.feature_layer,
                deterministic=True, self_cond_cfg_scale=scale,
            ).float()

    with torch.no_grad():
        real_features = features(real_noised, real, real_sc_cfg)
    fake_features = features(fake_noised, fake, sc_cfg)

    batch, length, width = fake_features.shape
    rows = batch // group
    fake_features = fake_features.reshape(rows, group * length, width)
    fake_weight = (1 - fake_mask).reshape(rows, group * length)
    sample_ids = torch.arange(group, device=fake.device).repeat_interleave(length).expand(rows, -1)
    real_weight = 1 - cond_mask
    if not conditional:
        real_features = real_features.reshape(rows, group * length, width)
        real_weight = real_weight.reshape(rows, group * length)

    return rbf_mmd(
        real_features, fake_features, real_weight, fake_weight, config.sigma,
        fake_groups=sample_ids, real_groups=None if conditional else sample_ids,
        fixed_real=conditional, unbiased=config.unbiased_rbf, block_size=config.mmd_block_size,
    )


def mmd_train_step(state, encoder, batch, config, feature_model):
    """Bootstrap self-conditioning, make one generator pass, and match frozen features."""
    conditional = config.task == "tinygsm"
    repeats = config.mmd_batch_size if conditional else 1
    clean, ids, cond_mask, attention_mask, sc_cfg = encode_batch(state.model, encoder, batch, config, repeats)
    n = config.distill_batch_size
    noise = sample_noise(clean, cond_mask, config)[:n]
    real, real_mask, real_sc_cfg = clean[:n], cond_mask[:n], sc_cfg[:n]

    # Gradient-free passes build the self-conditioning input. They bypass DDP
    # bookkeeping but still use the compiled module.
    num_bootstrap = sample_bootstrap_steps(config, state.step)
    generator = state.model.module if isinstance(state.model, DDP) else state.model
    self_cond = iterative_refinement(
        generator, noise, real, real_mask, config, num_bootstrap, self_cond_cfg_scale=real_sc_cfg,
    )

    prediction, logits = generator_forward(state.model, clean, noise, self_cond, sc_cfg, config)
    fake = restore_cond(prediction, real, real_mask)
    mmd = mmd_loss(feature_model, real, fake, noise, real_mask, real_sc_cfg, config, conditional)
    ce = decoder_ce_loss(logits, ids[n:], cond_mask[n:], attention_mask[n:], config)
    loss = weighted_loss(mmd, ce, n, logits.shape[0])
    metrics = {"loss": loss.detach(), "mmd_loss": mmd.detach(), "ce_loss": ce.detach()}
    return loss, metrics


# ============================================
# IRD
# ============================================

def ird_train_step(state, encoder, batch, config, target_model):
    """Fit one cold-start generator pass to ird_steps refinement passes of the frozen model."""
    clean, ids, cond_mask, attention_mask, sc_cfg = encode_batch(state.model, encoder, batch, config)
    n = config.distill_batch_size
    noise = sample_noise(clean, cond_mask, config)[:n]
    real, real_mask, real_sc_cfg = clean[:n], cond_mask[:n], sc_cfg[:n]

    target = iterative_refinement(
        target_model, noise, real, real_mask, config, config.ird_steps, self_cond_cfg_scale=real_sc_cfg,
    )
    zero_sc = restore_cond(torch.zeros_like(real), real, real_mask)
    prediction, logits = generator_forward(state.model, clean, noise, zero_sc, sc_cfg, config)

    response = 1 - real_mask
    token_mse = (prediction.float() - target.float()).square().mean(-1)
    mse = (token_mse * response).sum() / response.sum()
    ce = decoder_ce_loss(logits, ids[n:], cond_mask[n:], attention_mask[n:], config)
    loss = weighted_loss(mse, ce, n, logits.shape[0])
    return loss, {"loss": loss.detach(), "mse_loss": mse.detach(), "ce_loss": ce.detach()}
