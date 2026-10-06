"""Grouped-GEMM expert packing for full fine-tuning of LLaDA2 MoE layers; see NOTICE.md."""
from types import MethodType
import torch
from torch import nn
from .modeling_llada2_moe import LLaDA2MoeSparseMoeBlock


def torch_grouped_mm(a, b, sizes):
    return torch._grouped_mm(a, b, offs=sizes.cumsum(0).to(torch.int32))


def megablocks_grouped_mm(a, b, sizes):
    import grouped_gemm
    return grouped_gemm.ops.gmm(a, b, sizes.to("cpu", torch.long))


def choose_backend(device, dtype=torch.bfloat16):
    if device.type != "cuda":
        return None
    # Training needs gradients for BOTH operands; a forward-only probe is unsafe.
    for backend in (torch_grouped_mm, megablocks_grouped_mm):
        try:
            a = torch.randn(6, 16, device=device, dtype=dtype, requires_grad=True)
            b = torch.randn(2, 16, 16, device=device, dtype=dtype, requires_grad=True)
            out = backend(a, b, torch.tensor([4, 2], device=device))
            da, db = torch.autograd.grad(out.float().square().sum(), (a, b))
            if torch.isfinite(da).all() and torch.isfinite(db).all():
                return backend
        except (ImportError, RuntimeError, AttributeError, NotImplementedError):
            continue
    return None


def grouped_forward(self, hidden_states):
    bsz, seq_len, hidden = hidden_states.shape
    indices, weights, _ = self.gate(hidden_states)
    x = hidden_states.reshape(-1, hidden)
    counts = indices.new_zeros((x.shape[0], self._num_experts))
    counts.scatter_(1, indices, 1)
    sizes = counts.sum(0)
    order = indices.reshape(-1).argsort()
    routed = x[order // indices.shape[1]]
    gate = self._grouped_backend(routed, self._w_gate, sizes)
    up = self._grouped_backend(routed, self._w_up, sizes)
    outputs = self._grouped_backend(self._act_fn(gate) * up, self._w_down, sizes)
    restored = torch.empty_like(outputs)
    restored[order] = outputs
    result = (restored.view(*indices.shape, -1).to(weights.dtype)
              * weights.unsqueeze(-1)).sum(1).to(outputs.dtype).view(bsz, seq_len, hidden)
    if self.shared_experts is not None:
        result = result + self.shared_experts(hidden_states)
    return result


def pack_experts(model, backend):
    if backend is None:
        return model
    for block in model.modules():
        if not isinstance(block, LLaDA2MoeSparseMoeBlock) or block.experts is None:
            continue
        experts = block.experts
        count = len(experts)
        first = experts[0]
        hidden, inter = first.gate_proj.weight.shape[1], first.gate_proj.weight.shape[0]
        opts = dict(dtype=first.gate_proj.weight.dtype, device=first.gate_proj.weight.device)
        tensors = [torch.empty(count, hidden, inter, **opts),
                   torch.empty(count, hidden, inter, **opts),
                   torch.empty(count, inter, hidden, **opts)]
        for i, expert in enumerate(experts):
            for packed, attr in zip(tensors, ("gate_proj", "up_proj", "down_proj")):
                with torch.no_grad():
                    packed[i].copy_(getattr(expert, attr).weight.t())
                setattr(expert, attr, None)
        block._w_gate, block._w_up, block._w_down = [nn.Parameter(t) for t in tensors]
        block._num_experts = count
        block._act_fn = first.act_fn
        block._grouped_backend = backend
        block.experts = None
        block.forward = MethodType(grouped_forward, block)
    return model


def reference_state_dict(state_dict):
    """De-stack a FULL (CPU) FSDP state dict to original separate expert keys."""
    result = {}
    names = {"_w_gate": "gate_proj", "_w_up": "up_proj", "_w_down": "down_proj"}
    for key, tensor in state_dict.items():
        key = key.replace("_fsdp_wrapped_module.", "")
        prefix, _, last = key.rpartition(".")
        if last in names:
            prefix = prefix + "." if prefix else ""
            for i in range(tensor.shape[0]):
                result[f"{prefix}experts.{i}.{names[last]}.weight"] = tensor[i].t().contiguous()
        else:
            result[key] = tensor
    return result
