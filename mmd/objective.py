"""Frozen MDLM features and grouped REINFORCE objectives for OWT and TinyGSM.

G independent candidates share one student prediction. Each candidate contains
B sampled sequences and receives one feature-distribution reward. Only sampled
token log probabilities carry gradients; the feature extractor is frozen.
"""

import math

import torch


@torch.no_grad()
def _kernel_sum(left, right, selected, alpha, block_size, exclude_position=False):
    """Masked RBF sum per row, without a full length-by-length allocation."""
    total = left.new_zeros(left.shape[0])
    left2 = left.square().sum(-1)
    right2 = right.square().sum(-1)
    length = left.shape[1]
    for i in range(0, length, block_size):
        for j in range(0, length, block_size):
            a, b = left[:, i:i + block_size], right[:, j:j + block_size]
            kernel = torch.bmm(a, b.transpose(1, 2))
            kernel.mul_(-2).add_(left2[:, i:i + block_size, None])
            kernel.add_(right2[:, None, j:j + block_size]).clamp_min_(0)
            kernel.mul_(-alpha).exp_()
            if exclude_position and i == j:
                kernel.diagonal(dim1=1, dim2=2).zero_()
            total += torch.bmm(selected[:, None, i:i + block_size], torch.bmm(
                kernel, selected[:, j:j + block_size, None])).reshape(-1)
    return total


@torch.no_grad()
def feature_reward(real, fake, selected, *, kernel='token_rbf', alpha=2e-5,
                   block_size=0, exclude_same_position=True):
    """Return rewards and active rows from real (N,L,D), fake (N,B,L,D).

    Token RBF excludes all same-sequence fake pairs and optionally matching
    token positions across sequences. The real-real constant is omitted.
    """
    if kernel != 'token_rbf':
        raise ValueError(f'Only token_rbf is supported, got {kernel}')
    if real.ndim != 3 or fake.ndim != 4 or fake.shape[0] != real.shape[0] \
            or fake.shape[2:] != real.shape[1:] or selected.shape != real.shape[:2]:
        raise ValueError('Expected real (N,L,D), fake (N,B,L,D), selected (N,L)')
    draws = fake.shape[1]
    if draws < 2:
        raise ValueError('token_rbf needs B >= 2')
    if block_size < 0 or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError('block_size must be nonnegative and alpha positive')
    if block_size == 0:
        block_size = max(64, min(real.shape[1], int(((1 << 26) / max(real.shape[0], 1)) ** 0.5)))
    real, fake = real.detach().float(), fake.detach().float()
    selected = selected.bool()
    weight = selected.float()
    count = weight.sum(-1)
    with torch.autocast(device_type=real.device.type, enabled=False):
        cross = torch.zeros_like(count)
        within = torch.zeros_like(count)
        for i in range(draws):
            cross += _kernel_sum(real, fake[:, i], weight, alpha, block_size)
            for j in range(i + 1, draws):
                within.add_(_kernel_sum(
                    fake[:, i], fake[:, j], weight, alpha, block_size,
                    exclude_position=exclude_same_position), alpha=2)
        reward = 2 * cross / (draws * count.square()).clamp_min(1)
        pairs = count * (count - 1) if exclude_same_position else count.square()
        reward -= within / (draws * (draws - 1) * pairs).clamp_min(1)
        active = count >= (2 if exclude_same_position else 1)
    return torch.where(active, reward, torch.zeros_like(reward)), active


class _FeaturesReady(Exception):
    pass


class FeatureReward:
    """Read one zero-based block from a frozen local or HF MDLM backbone.

    This plain Python holder does not register an extra copy of the backbone in
    a Lightning state dict. The caller owns, checkpoints and moves the backbone.
    ``feature_batch_size`` limits physical extractor forwards without changing
    the batch or the MMD statistic. Inputs are clean; sigma is identically zero.
    """

    def __init__(self, backbone, layer=3, kernel='token_rbf', alpha=2e-5,
                 block_size=0, feature_batch_size=None, exclude_same_position=True):
        core = getattr(backbone, 'backbone', backbone)
        blocks = getattr(core, 'blocks', None)
        if blocks is None or not 0 <= layer < len(blocks):
            raise ValueError('Feature layer must be a zero-based MDLM block index')
        self.backbone = backbone
        self.block = blocks[layer]
        self.kernel = kernel
        self.alpha = alpha
        self.block_size = block_size
        self.feature_batch_size = feature_batch_size
        self.exclude_same_position = exclude_same_position
        if feature_batch_size is not None and feature_batch_size < 1:
            raise ValueError('feature_batch_size must be positive or None')
        backbone.requires_grad_(False)
        backbone.eval()

    @torch.no_grad()
    def features(self, tokens):
        captured = []

        def capture(_module, _inputs, output):
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            captured.append(hidden.detach())
            # No downstream layer or vocabulary-sized output is used by MMD.
            raise _FeaturesReady

        self.backbone.eval()
        handle = self.block.register_forward_hook(capture)
        try:
            size = self.feature_batch_size or tokens.shape[0]
            for start in range(0, tokens.shape[0], size):
                chunk = tokens[start:start + size]
                try:
                    self.backbone(chunk, torch.zeros(chunk.shape[0], device=chunk.device))
                except _FeaturesReady:
                    continue
                raise RuntimeError('The selected MDLM feature block did not run')
        finally:
            handle.remove()
        return torch.cat(captured)

    @torch.no_grad()
    def __call__(self, x0, draws, selected):
        """Score one candidate: draws has shape (B,N,L)."""
        # The reference consumes this critic-mask draw even at noise_level=0.
        # Preserve its RNG stream so removing unused branches keeps sampling parity.
        torch.rand(x0.shape, device=x0.device)
        n, length = x0.shape
        features = self.features(torch.cat([x0, draws.reshape(-1, length)]))
        real = features[:n]
        fake = features[n:].reshape(draws.shape[0], n, length, -1).transpose(0, 1)
        return feature_reward(real, fake, selected, kernel=self.kernel, alpha=self.alpha,
                              block_size=self.block_size,
                              exclude_same_position=self.exclude_same_position)


def reinforce_loss(student_probs, x0, selected, reward_fn, *, group_size=4,
                   draws_per_candidate=2, average_candidates=True):
    """Return (loss, stats) for probabilities (N,L,V) and clean tokens (N,L).

    ``reward_fn(x0, draws, selected)`` returns detached rewards and active rows;
    draws is (B,N,L). G candidates use a leave-one-out group baseline. G=1
    uses no baseline, matching the archived group-size ablation. Set
    ``average_candidates=False`` for the historical TinyGSM sum over G.
    """
    if group_size < 1 or draws_per_candidate < 1:
        raise ValueError('group_size and draws_per_candidate must be positive')
    if student_probs.shape[:2] != x0.shape or selected.shape != x0.shape:
        raise ValueError('Student probabilities, tokens and mask must share (N,L)')
    selected = selected.bool()
    if not selected.any():
        zero = student_probs.sum() * 0
        return zero, {'mmd/reward': zero.detach(), 'mmd/advantage_absmean': zero.detach(),
                      'reward_count': zero.detach()}
    with torch.no_grad():
        draws = torch.multinomial(student_probs.reshape(-1, student_probs.shape[-1]).float(),
                                  group_size * draws_per_candidate, replacement=True)
        draws = draws.view(*x0.shape, -1).permute(2, 0, 1)
        samples = torch.where(selected[None], draws, x0[None])
        rewards = []
        for g in range(group_size):
            start = g * draws_per_candidate
            reward, active = reward_fn(x0, samples[start:start + draws_per_candidate], selected)
            rewards.append(reward)
        rewards = torch.stack(rewards, -1)
        if group_size > 1:
            advantages = rewards - (rewards.sum(-1, keepdim=True) - rewards) / (group_size - 1)
        else:
            advantages = rewards
        advantages = advantages.masked_fill(~active[:, None], 0)
    logp = student_probs.clamp_min(1e-12).log()
    sampled_logp = torch.stack([logp.gather(-1, sample[..., None]).squeeze(-1) for sample in samples])
    seq_logp = (sampled_logp * selected[None]).sum(-1)
    seq_logp = seq_logp.reshape(group_size, draws_per_candidate, -1).sum(1).transpose(0, 1)
    group_divisor = group_size if average_candidates else 1
    denominator = selected.sum().clamp_min(1) * group_divisor * draws_per_candidate
    loss = -(advantages * seq_logp).sum() / denominator
    return loss, {
        'mmd/reward': rewards[active].mean() if active.any() else rewards.new_zeros(()),
        'reward_count': active.sum(),
        'mmd/advantage_absmean': advantages.abs().mean(),
    }
