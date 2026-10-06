"""RBF-kernel MMD between token features, computed in tiles to bound memory.

Features have shape (rows, tokens, channels); each row is one independent
kernel estimate, e.g. one prompt with its generated responses. Token weights
mask out padding and prompt positions. Backward recomputes kernel tiles
instead of storing the (tokens x tokens) kernel matrix.
"""

import torch


# ============================================
# Kernel tiles
# ============================================

def _kernel_tile(a, b, weight_a, weight_b, alpha, group_a=None, group_b=None, same_tile=False):
    """Weighted exp(-alpha * |a_i - b_j|^2) for one tile; pairs from the same group are zeroed."""
    distance = (a.square().sum(-1, keepdim=True)
                + b.square().sum(-1).unsqueeze(1)
                - 2 * torch.bmm(a, b.transpose(1, 2)))
    kernel = distance.clamp_min_(0).mul_(-alpha).exp_()
    if same_tile:
        # The expansion above leaves round-off on the diagonal; k(x, x) is exactly 1.
        kernel.diagonal(dim1=1, dim2=2).fill_(1)
    kernel.mul_(weight_a.unsqueeze(2)).mul_(weight_b.unsqueeze(1))
    if group_a is not None:
        kernel.masked_fill_(group_a.unsqueeze(2) == group_b.unsqueeze(1), 0)
    return kernel


def _tiles(num_tokens, block_size):
    return [slice(start, start + block_size) for start in range(0, num_tokens, block_size)]


def _default_block_size(x):
    # Keep one (rows, block, block) kernel tile at about 4M elements.
    return max(1, min(x.shape[1], int((2 ** 22 / x.shape[0]) ** 0.5)))


class _SelfKernelSum(torch.autograd.Function):
    """sum_ij w_i w_j k(x_i, x_j) per row, optionally skipping pairs in the same group."""

    @staticmethod
    def forward(ctx, x, weight, groups, alpha, block_size):
        ctx.save_for_backward(x, weight, groups, alpha)
        ctx.block_size = block_size
        g = groups if groups.numel() else None
        total = x.new_zeros(x.shape[0])
        tiles = _tiles(x.shape[1], block_size)
        # The sum is symmetric: visit each off-diagonal tile once and count it twice.
        for a, i in enumerate(tiles):
            for j in tiles[a:]:
                kernel = _kernel_tile(
                    x[:, i], x[:, j], weight[:, i], weight[:, j], alpha,
                    g[:, i] if g is not None else None, g[:, j] if g is not None else None,
                    same_tile=i == j,
                )
                total += (1 if i == j else 2) * kernel.sum((1, 2))
        return total

    @staticmethod
    def backward(ctx, grad_total):
        x, weight, groups, alpha = ctx.saved_tensors
        g = groups if groups.numel() else None
        grad = torch.zeros_like(x)
        scale = -4 * alpha * grad_total[:, None, None]
        tiles = _tiles(x.shape[1], ctx.block_size)
        # d/dx_i sum_ij K_ij = -4 alpha sum_j K_ij (x_i - x_j).
        for a, i in enumerate(tiles):
            for j in tiles[a:]:
                kernel = _kernel_tile(
                    x[:, i], x[:, j], weight[:, i], weight[:, j], alpha,
                    g[:, i] if g is not None else None, g[:, j] if g is not None else None,
                    same_tile=i == j,
                )
                grad[:, i] += scale * (kernel.sum(2, keepdim=True) * x[:, i] - torch.bmm(kernel, x[:, j]))
                if i != j:
                    kernel = kernel.transpose(1, 2)
                    grad[:, j] += scale * (kernel.sum(2, keepdim=True) * x[:, j] - torch.bmm(kernel, x[:, i]))
        return grad, None, None, None, None


class _CrossKernelSum(torch.autograd.Function):
    """sum_ij u_i w_j k(y_i, x_j) per row; gradients flow to x only."""

    @staticmethod
    def forward(ctx, y, x, weight_y, weight_x, alpha, block_size):
        ctx.save_for_backward(y, x, weight_y, weight_x, alpha)
        ctx.block_size = block_size
        total = x.new_zeros(x.shape[0])
        for i in _tiles(y.shape[1], block_size):
            for j in _tiles(x.shape[1], block_size):
                total += _kernel_tile(y[:, i], x[:, j], weight_y[:, i], weight_x[:, j], alpha).sum((1, 2))
        return total

    @staticmethod
    def backward(ctx, grad_total):
        y, x, weight_y, weight_x, alpha = ctx.saved_tensors
        grad = torch.zeros_like(x)
        scale = 2 * alpha * grad_total[:, None, None]
        # d/dx_j sum_ij K_ij = 2 alpha sum_i K_ij (y_i - x_j).
        for i in _tiles(y.shape[1], ctx.block_size):
            for j in _tiles(x.shape[1], ctx.block_size):
                kernel = _kernel_tile(y[:, i], x[:, j], weight_y[:, i], weight_x[:, j], alpha).transpose(1, 2)
                grad[:, j] += scale * (torch.bmm(kernel, y[:, i]) - kernel.sum(2, keepdim=True) * x[:, j])
        return None, grad, None, None, None, None


def self_kernel_sum(x, weight, alpha, groups=None, block_size=0):
    groups = groups if groups is not None else torch.empty(0, dtype=torch.long, device=x.device)
    return _SelfKernelSum.apply(x, weight, groups, alpha, block_size or _default_block_size(x))


def cross_kernel_sum(y, x, weight_y, weight_x, alpha, block_size=0):
    return _CrossKernelSum.apply(y.detach(), x, weight_y, weight_x, alpha, block_size or _default_block_size(x))


# ============================================
# MMD estimate
# ============================================

def drop_masked_tokens(x, weight, groups=None):
    """Move weighted tokens to the front of each row and trim the all-masked tail."""
    width = int((weight > 0).sum(1).max().item())
    if width == weight.shape[1]:
        return x, weight, groups
    columns = torch.arange(weight.shape[1], device=weight.device).expand_as(weight)
    order = columns.masked_fill(weight <= 0, weight.shape[1]).argsort(1)[:, :width]
    x = x.gather(1, order.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
    groups = groups.gather(1, order) if groups is not None else None
    return x, weight.gather(1, order), groups


def _pair_count(weight, groups=None):
    """Number of token pairs (i != j), or of pairs from different groups."""
    count = weight.sum(1)
    if groups is None:
        return count * (count - 1)
    group_counts = weight.new_zeros(weight.shape[0], int(groups.max()) + 1).scatter_add_(1, groups, weight)
    return count.square() - group_counts.square().sum(1)


def rbf_mmd(real, fake, real_weight, fake_weight, sigma, *, fake_groups, real_groups=None,
            fixed_real=False, unbiased=False, block_size=0):
    """Squared MMD per row between real and generated token features, averaged over rows.

    `fake_groups` labels the generated sample each token comes from (likewise
    `real_groups`). `unbiased` drops kernel pairs within one sample, so a sample
    is never compared with itself. With `fixed_real` (conditional data), the
    real row is a single reference whose term uses all its token pairs.
    """
    real = real.detach().float()
    fake = fake.float()
    real, real_weight, real_groups = drop_masked_tokens(real, real_weight.float(), real_groups)
    fake, fake_weight, fake_groups = drop_masked_tokens(fake, fake_weight.float(), fake_groups)
    alpha = real.new_tensor(1 / (2 * sigma ** 2))
    n_real, n_fake = real_weight.sum(1), fake_weight.sum(1)

    def pair_sum(x, weight, groups):
        """Kernel sum and count over token pairs i != j, or over pairs from different groups."""
        if groups is not None:
            return self_kernel_sum(x, weight, alpha, groups, block_size), _pair_count(weight, groups)
        total = self_kernel_sum(x, weight, alpha, block_size=block_size)
        return total - weight.sum(1), _pair_count(weight)  # k(x_i, x_i) = 1 for every token

    # Real-real term: a constant of the loss, so no gradient is needed.
    with torch.no_grad():
        if fixed_real:
            xx = self_kernel_sum(real, real_weight, alpha, block_size=block_size) / n_real.square().clamp_min(1)
        else:
            xx_sum, xx_pairs = pair_sum(real, real_weight, real_groups if unbiased else None)
            xx = xx_sum / xx_pairs.clamp_min(1)

    # Fake-fake (repulsion) and real-fake (attraction) terms.
    yy_sum, yy_pairs = pair_sum(fake, fake_weight, fake_groups if unbiased else None)
    yy = yy_sum / yy_pairs.clamp_min(1)
    xy = cross_kernel_sum(real, fake, real_weight, fake_weight, alpha, block_size) / (n_real * n_fake).clamp_min(1)

    # Denominators are clamped so empty rows stay finite; they are dropped here.
    usable = (yy_pairs > 0) & (n_real > 0)
    if not usable.any():
        raise ValueError("MMD needs at least two generated tokens in some row")
    return (xx + yy - 2 * xy)[usable].mean()
