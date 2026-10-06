import math
from contextlib import contextmanager

import torch
import torch.nn.functional as F

# All released checkpoints use the LLaDA2 tokenizer.
MASK_TOKEN_ID = 156895


@contextmanager
def evaluating(model):
    """Temporarily switch every module to eval mode, restoring each module's own mode after."""
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        yield
    finally:
        for module, mode in modes:
            module.training = mode


def block_points(features, selected, block_size):
    """Split tokens into blocks, padding the final partial block with zero weight."""
    length = selected.shape[1]
    padded_length = math.ceil(length / block_size) * block_size
    points = F.pad(features, (0, 0, 0, padded_length - length))
    mask = F.pad(selected, (0, padded_length - length))
    return (
        points.reshape(len(points), -1, block_size, points.shape[-1]),
        mask.reshape(len(points), -1, block_size),
    )


def block_rewards(reference, generated, selected, block_size, sigma):
    """Return one reward per example, block and sample group.

    Features have shapes [batch, length, hidden] for the reference response and
    [batch, G, B, length, hidden] for generated samples. Each of the G groups
    gives one MMD estimate over its B samples.
    """
    batch, group_size, mmd_batch_size, length, hidden = generated.shape
    generated, mask = block_points(
        generated.reshape(batch * group_size * mmd_batch_size, length, hidden),
        selected[:, None, None]
        .expand(batch, group_size, mmd_batch_size, length)
        .reshape(batch * group_size * mmd_batch_size, length),
        block_size,
    )
    blocks, width = generated.shape[1:3]
    generated = generated.reshape(batch, group_size, mmd_batch_size, blocks, width, hidden).permute(
        0, 3, 1, 2, 4, 5
    )
    mask = mask.reshape(batch, group_size, mmd_batch_size, blocks, width).permute(0, 3, 1, 2, 4)
    reference, _ = block_points(reference, selected, block_size)
    reference = reference[:, :, None, None].expand_as(generated)
    return token_mmd_rewards(reference, generated, mask, sigma)


def _score_surrogate(log_probs, selected, rewards, block_size):
    """Negative advantage times joint score, averaged over active blocks and groups.

    Sum over the B samples of a group: averaging their scores would shrink the
    desired gradient by B. A leave-one-group-out baseline reduces variance
    without introducing dependence on the scored group's actions.
    """
    batch, group_size, mmd_batch_size, length = log_probs.shape
    blocks = rewards.shape[1]
    log_probs = log_probs.masked_fill(~selected[:, None, None], 0)
    scores = F.pad(log_probs, (0, blocks * block_size - length))
    scores = scores.reshape(batch, group_size, mmd_batch_size, blocks, block_size).sum(-1)
    scores = scores.permute(0, 3, 1, 2).sum(-1)
    advantage = rewards.detach()
    if group_size > 1:
        advantage = advantage - (advantage.sum(-1, keepdim=True) - advantage) / (group_size - 1)
    active = F.pad(selected, (0, blocks * block_size - length))
    active = active.reshape(batch, blocks, block_size).any(-1)
    return (-(advantage * scores).mean(-1).sum(-1) / active.sum(-1).clamp_min(1)).mean()


def _gather_log_probs(logits, indices, mask_id):
    logits = logits.float()
    if mask_id < logits.shape[-1]:
        logits = logits.index_fill(-1, indices.new_tensor([mask_id]), -torch.inf)
    return logits.log_softmax(-1).gather(-1, indices)


class _SampledLogProbs(torch.autograd.Function):
    """Retain raw logits, recomputing bounded FP32 score chunks in backward.

    One custom backward writes into a single vocabulary-gradient buffer. Using
    separate checkpointed slices would allocate that full buffer per slice.
    """

    chunk_size = 64

    @staticmethod
    def forward(ctx, logits, indices, selected, mask_id):
        rows = selected.flatten().nonzero().flatten()
        ctx.save_for_backward(logits, indices, rows)
        ctx.mask_id = mask_id
        length = logits.shape[1]
        actions = indices.reshape(-1, indices.shape[-1])
        result = torch.zeros(indices.shape, device=logits.device, dtype=torch.float32)
        scores = result.reshape_as(actions)
        for start in range(0, len(rows), _SampledLogProbs.chunk_size):
            keep = rows[start : start + _SampledLogProbs.chunk_size]
            local = logits[keep // length, keep % length]
            scores[keep] = _gather_log_probs(local, actions[keep], mask_id)
        return result

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        logits, indices, rows = ctx.saved_tensors
        length = logits.shape[1]
        actions = indices.reshape(-1, indices.shape[-1])
        upstream = grad_output.reshape_as(actions)
        gradient = torch.zeros(logits.shape, device=logits.device, dtype=logits.dtype)
        flat_gradient = gradient.reshape(-1, logits.shape[-1])
        for start in range(0, len(rows), _SampledLogProbs.chunk_size):
            keep = rows[start : start + _SampledLogProbs.chunk_size]
            with torch.enable_grad():
                local = logits[keep // length, keep % length].detach().requires_grad_(True)
                scores = _gather_log_probs(local, actions[keep], ctx.mask_id)
                (local_grad,) = torch.autograd.grad(scores, local, upstream[keep])
            flat_gradient[keep] = local_grad
        return gradient, None, None, None


def sampled_policy_surrogate(logits, samples, selected, rewards, block_size, mask_id):
    """Score sampled token IDs without retaining a full FP32 log-softmax tensor."""
    batch, group_size, mmd_batch_size, length = samples.shape
    indices = samples.permute(0, 3, 1, 2).reshape(batch, length, group_size * mmd_batch_size)
    log_probs = _SampledLogProbs.apply(logits, indices, selected, mask_id)
    log_probs = log_probs.reshape(batch, length, group_size, mmd_batch_size).permute(0, 2, 3, 1)
    return _score_surrogate(log_probs, selected, rewards, block_size)


class MMDLoss:
    """Match features of sampled one-step corrections to features of the reference response."""

    def __init__(self, config, feature_extractor):
        from distributed import unwrap_model

        self.block_size = int(config.training.block_size)
        self.cfg = config.training.mmd
        self.feature_extractor = feature_extractor
        self.layer = unwrap_model(feature_extractor).model.layers[int(self.cfg.mmd_feature_layer)]
        self.metrics = {}

    @torch.no_grad()
    def features(self, candidates, reference, attention):
        """Read candidate features under the policy's blockwise clean-prefix context.

        Each [candidate | reference] row uses the same mask and position grid as OPUT.
        Candidate tokens see their own block and strictly earlier reference blocks;
        they cannot see the reference answer for their current block. One row per
        forward bounds activation memory for the 16B frozen model.
        """
        length = reference.shape[1]
        positions = torch.arange(length, device=reference.device).repeat(2)[None]
        result = []
        with evaluating(self.feature_extractor):
            for candidate, target in zip(candidates.split(1), reference.split(1)):
                rows = torch.cat((candidate, target), -1)
                captured = []

                def hook(module, inputs, output):
                    hidden = output[0] if isinstance(output, (tuple, list)) else output
                    captured.append(hidden.detach())

                handle = self.layer.register_forward_hook(hook)
                try:
                    self.feature_extractor(
                        input_ids=rows,
                        attention_mask=attention,
                        position_ids=positions,
                        logits_to_keep=(0, 0),
                    )
                finally:
                    handle.remove()
                result.append(captured[0][:, :length].float().clone())
        return torch.cat(result)

    def __call__(self, model, inputs, attention, positions, labels):
        batch, length = labels.shape
        group_size = int(self.cfg.group_size)
        mmd_batch_size = int(self.cfg.mmd_batch_size)
        selected = labels != -100
        kwargs = dict(
            input_ids=inputs,
            attention_mask=attention,
            position_ids=positions,
            logits_to_keep=(0, length),
        )
        # Sampling and replay have the same distribution because LLaDA2 has no dropout.
        # Replay stays in train mode so native gradient checkpointing remains on.
        with evaluating(model), torch.no_grad():
            logits = model(**kwargs).logits[:, :length].float()
            if MASK_TOKEN_ID < logits.shape[-1]:
                # Out-of-place: .float() may alias an already-FP32 model output.
                logits = logits.index_fill(-1, inputs.new_tensor([MASK_TOKEN_ID]), -torch.inf)
            draws = torch.multinomial(
                logits[selected].softmax(-1), group_size * mmd_batch_size, replacement=True
            )
            del logits

        reference = inputs[:, length:]
        context = inputs[:, :length]
        # Only selected positions differ between the reference and the generated samples.
        # Preserve all other noisy/predicted tokens, including ignored tails.
        samples = context[:, None, None].expand(batch, group_size, mmd_batch_size, length).clone()
        samples.permute(0, 3, 1, 2)[selected] = draws.reshape(-1, group_size, mmd_batch_size)
        reference_features = self.features(
            torch.where(selected, reference, context), reference, attention
        )
        generated = []
        repeated_reference = (
            reference[:, None].expand(batch, mmd_batch_size, length).reshape(-1, length)
        )
        for group in range(group_size):
            candidates = samples[:, group : group + 1].reshape(batch * mmd_batch_size, length)
            generated.append(
                self.features(candidates, repeated_reference, attention).reshape(
                    batch, 1, mmd_batch_size, length, -1
                )
            )
        rewards = block_rewards(
            reference_features,
            torch.cat(generated, 1),
            selected,
            self.block_size,
            float(self.cfg.mmd_rbf_sigma),
        )
        del generated, reference_features
        with torch.enable_grad():
            raw_logits = model(**kwargs).logits[:, :length]
            loss = sampled_policy_surrogate(
                raw_logits, samples, selected, rewards, self.block_size, MASK_TOKEN_ID
            )
        active = F.pad(selected, (0, rewards.shape[1] * self.block_size - length))
        active = active.reshape(batch, -1, self.block_size).any(-1)
        values = rewards[active]
        self.metrics = {
            "mmd_reward": (values.sum() / max(values.numel(), 1)).item(),
            "mmd_reward_count": values.numel(),
        }
        return loss


def token_mmd_rewards(reference, generated, token_mask, sigma):
    """Compute token RBF rewards in FP32, independent of model autocast."""
    # .float() alone cannot prevent autocast from downcasting the dot products.
    with torch.autocast(device_type=reference.device.type, enabled=False):
        return _token_mmd_rewards_fp32(reference, generated, token_mask, sigma)


def _token_mmd_rewards_fp32(reference, generated, token_mask, sigma):
    """Return 2 * k(generated, reference) - k(generated, generated) for each block and group.

    Inputs have shape [batch, blocks, G, B, tokens, hidden]. The reference
    features repeat across the B samples. Each selected token has equal weight
    within its sample. Generated-generated pairs exclude the entire same-sample
    token cluster; matching token positions across different samples remain
    included. With B = 1 there is no generated-generated term. The
    action-independent reference-reference term is omitted, and the final reward
    is never clipped.
    """
    batch, blocks, group_size, mmd_batch_size, width, hidden = reference.shape
    mask = token_mask.bool()
    reference, generated = reference.float(), generated.float()
    reference = reference[..., 0, :, :]
    weights = mask.float() / mask.sum(-1, keepdim=True).clamp_min(1)
    reference_weights = weights[..., 0, :]
    active = mask[..., 0, :].any(-1)
    generated_points = generated.flatten(-3, -2)
    generated_weights = weights.flatten(-2)

    def weighted_kernel_sum(x, y, x_weights, y_weights, exclude_same_sample=False):
        # Chunk both leading rows and token pairs. Norms and a dot product avoid
        # materializing [rows, x_tokens, y_tokens, hidden] feature differences.
        x_count, y_count = x.shape[-2], y.shape[-2]
        x_rows = x.reshape(-1, x_count, hidden)
        y_rows = y.reshape(-1, y_count, hidden)
        x_weight_rows = x_weights.reshape(-1, x_count)
        y_weight_rows = y_weights.reshape(-1, y_count)
        values = []
        for row in range(0, x_rows.shape[0], 16):
            x_chunk, y_chunk = x_rows[row : row + 16], y_rows[row : row + 16]
            accumulated = x_chunk.new_zeros(x_chunk.shape[0])
            for i in range(0, x_count, 64):
                xi = x_chunk[:, i : i + 64]
                for j in range(0, y_count, 64):
                    yj = y_chunk[:, j : j + 64]
                    squared_distance = (
                        xi.square().sum(-1).unsqueeze(-1)
                        + yj.square().sum(-1).unsqueeze(-2)
                        - 2 * torch.matmul(xi, yj.transpose(-1, -2))
                    ) / hidden
                    # Clamp only roundoff below zero, not the final reward.
                    kernel = torch.exp(-squared_distance.clamp_min(0) / (2 * sigma**2))
                    pair_weight = x_weight_rows[row : row + 16, i : i + 64].unsqueeze(
                        -1
                    ) * y_weight_rows[row : row + 16, j : j + 64].unsqueeze(-2)
                    if exclude_same_sample:
                        x_sample = torch.arange(i, min(i + 64, x_count), device=x.device) // width
                        y_sample = torch.arange(j, min(j + 64, y_count), device=x.device) // width
                        pair_weight = pair_weight * x_sample[:, None].ne(y_sample[None, :])
                    accumulated = accumulated + (kernel * pair_weight).sum((-1, -2))
            values.append(accumulated)
        return torch.cat(values).reshape(batch, blocks, group_size)

    generated_generated = reference.new_zeros((batch, blocks, group_size))
    if mmd_batch_size > 1:
        generated_generated = weighted_kernel_sum(
            generated_points, generated_points, generated_weights, generated_weights, True
        )
        generated_generated = generated_generated / (mmd_batch_size * (mmd_batch_size - 1))
    generated_reference = (
        weighted_kernel_sum(generated_points, reference, generated_weights, reference_weights)
        / mmd_batch_size
    )
    return torch.where(
        active, 2 * generated_reference - generated_generated, torch.zeros_like(generated_generated)
    )
