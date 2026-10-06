# coding=utf-8
# Copyright 2025 Antgroup and The HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""DMax LLaDA2 training layers; see NOTICE.md for upstream revision and adaptations.

Explicit 4D attention masks are passed unchanged to SDPA (True means attend).
"""
import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers.activations import ACT2FN
from transformers.modeling_outputs import MoeCausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from .configuration_llada2_moe import LLaDA2MoeConfig

class LLaDA2MoeRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LLaDA2MoeRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin):
    """Rotate the configured head dimensions at the supplied logical positions."""
    # RoPE values are [batch, token, rotary_dim]; broadcast over attention heads.
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    # Keep half or full tensor for later concatenation
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    # Apply rotary embeddings on the first half or full tensor
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    # Concatenate back to full shape
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed


class LLaDA2MoeMLP(nn.Module):
    def __init__(self, config: LLaDA2MoeConfig, intermediate_size: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class LLaDA2MoeGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts

        self.n_group = config.n_group
        self.topk_group = config.topk_group

        # topk selection algorithm
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty((self.num_experts, self.gating_dim)))
        self.routed_scaling_factor = config.routed_scaling_factor

        self.register_buffer("expert_bias", torch.zeros((self.num_experts)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init as init

        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def group_limited_topk(
        self,
        scores: torch.Tensor,
    ):
        num_tokens, _ = scores.size()
        # Organize the experts into groups
        group_scores = scores.view(num_tokens, self.n_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)

        # Mask the experts based on selection groups
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(num_tokens, self.n_group, self.num_experts // self.n_group)
            .reshape(num_tokens, -1)
        )

        masked_scores = scores.masked_fill(~score_mask.bool(), float('-inf'))
        probs, top_indices = torch.topk(masked_scores, k=self.top_k, dim=-1)

        return probs, top_indices

    def forward(self, hidden_states):
        # compute gating score
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32))

        scores = torch.sigmoid(logits.float()).type_as(logits)

        scores_for_routing = scores + self.expert_bias
        _, topk_idx = self.group_limited_topk(scores_for_routing)

        scores = torch.gather(scores, dim=1, index=topk_idx).type_as(logits)

        topk_weight = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20) if self.top_k > 1 else scores
        topk_weight = topk_weight * self.routed_scaling_factor

        return topk_idx, topk_weight, logits


class LLaDA2MoeRotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        scaling = config.rope_scaling or {}
        if scaling.get("rope_type", scaling.get("type", "default")) != "default":
            raise ValueError("Only the released LLaDA2 default RoPE is supported")
        self.rebuild(torch.device("cpu"))

    def rebuild(self, device):
        dim = int(self.config.head_dim * self.config.partial_rotary_factor)
        inv_freq = 1.0 / (self.config.rope_theta ** (
            torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        with torch.autocast(device_type=x.device.type, enabled=False):
            freqs = (inv @ position_ids[:, None, :].float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


class LLaDA2MoeSparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok
        self.experts = nn.ModuleList([
            LLaDA2MoeMLP(config, config.moe_intermediate_size)
            for _ in range(config.num_experts)])
        self.gate = LLaDA2MoeGate(config)
        self.shared_experts = (LLaDA2MoeMLP(
            config, config.moe_intermediate_size * config.num_shared_experts)
            if config.num_shared_experts else None)

    def forward(self, hidden_states):
        # Same expert and combine equations as upstream _forward; autograd also
        # works in eval mode, so temporary mode changes cannot freeze experts.
        shape = hidden_states.shape
        indices, weights, router_logits = self.gate(hidden_states)
        flat = hidden_states.reshape(-1, shape[-1])
        expanded = flat.repeat_interleave(self.num_experts_per_tok, dim=0)
        result = torch.empty_like(expanded)
        for i, expert in enumerate(self.experts):
            selected = indices.reshape(-1) == i
            # Unwrapped/DDP FP32 masters may produce BF16 expert activations
            # under autocast. Indexed assignment does not promote dtypes.
            result[selected] = expert(expanded[selected]).to(result.dtype)
        result = (result.view(*weights.shape, -1) * weights.unsqueeze(-1)).sum(1)
        result = result.to(hidden_states.dtype).view(shape)
        if self.shared_experts is not None:
            result = result + self.shared_experts(hidden_states)
        return result


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# Copied from transformers.models.llama.modeling_llama.LlamaAttention with Llama->LLaDA2Moe
class LLaDA2MoeAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LLaDA2MoeConfig, layer_idx=None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim or self.hidden_size // self.num_heads
        partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
        self.rope_dim = int(self.head_dim * partial_rotary_factor)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = False

        self.query_key_value = nn.Linear(
            self.hidden_size,
            (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim,
            bias=config.use_qkv_bias,
        )

        self.query_layernorm = LLaDA2MoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.key_layernorm = LLaDA2MoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.dense = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.use_bias)

    def forward(self, hidden_states, attention_mask, position_embeddings):
        bsz, q_len, _ = hidden_states.shape
        qkv = self.query_key_value(hidden_states).view(
            bsz, q_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        query, key, value = qkv.split(
            [self.num_heads, self.num_key_value_heads, self.num_key_value_heads], dim=-2)
        query = self.query_layernorm(query.transpose(1, 2))
        key = self.key_layernorm(key.transpose(1, 2))
        value = value.transpose(1, 2)
        query, key = apply_rotary_pos_emb(query, key, *position_embeddings)
        key, value = repeat_kv(key, self.num_key_value_groups), repeat_kv(value, self.num_key_value_groups)
        out = F.scaled_dot_product_attention(
            query.contiguous(), key.contiguous(), value.contiguous(),
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False)
        out = self.dense(out.transpose(1, 2).contiguous().reshape(bsz, q_len, -1))
        return out


class LLaDA2MoeDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attention = LLaDA2MoeAttention(config, layer_idx)
        self.mlp = (LLaDA2MoeSparseMoeBlock(config)
                    if config.num_experts is not None and layer_idx >= config.first_k_dense_replace
                    else LLaDA2MoeMLP(config, config.intermediate_size))
        self.input_layernorm = LLaDA2MoeRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = LLaDA2MoeRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, hidden_states, attention_mask, position_embeddings):
        attention_output = self.attention(self.input_layernorm(hidden_states), attention_mask,
                                          position_embeddings)
        hidden_states = hidden_states + attention_output
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states


class LLaDA2MoeModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([LLaDA2MoeDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = LLaDA2MoeRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = LLaDA2MoeRotaryEmbedding(config)
        self.gradient_checkpointing = False

    def forward(self, input_ids, attention_mask, position_ids, inputs_embeds=None):
        hidden = self.word_embeddings(input_ids) if inputs_embeds is None else inputs_embeds
        positions = self.rotary_emb(hidden, position_ids)
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden = checkpoint(layer, hidden, attention_mask, positions, use_reentrant=False)
            else:
                hidden = layer(hidden, attention_mask, positions)
        return self.norm(hidden)


class LLaDA2MoeModelLM(PreTrainedModel):
    config_class = LLaDA2MoeConfig
    base_model_prefix = "model"
    _no_split_modules = ["LLaDA2MoeDecoderLayer"]
    _supports_sdpa = True
    supports_gradient_checkpointing = True
    _tied_weights_keys = {"lm_head.weight": "model.word_embeddings.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.model = LLaDA2MoeModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if getattr(module, "bias", None) is not None:
                module.bias.data.zero_()
            if getattr(module, "padding_idx", None) is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, LLaDA2MoeGate):
            module.reset_parameters()
        elif isinstance(module, LLaDA2MoeRMSNorm):
            module.weight.data.fill_(1.0)

    def get_input_embeddings(self):
        return self.model.word_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.model.gradient_checkpointing = True

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                inputs_embeds=None, logits_to_keep=None):
        """Apply the explicit OPUT grid and project only the requested loss positions."""
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Supply exactly one of input_ids and inputs_embeds")
        shape = input_ids.shape if input_ids is not None else inputs_embeds.shape[:2]
        if attention_mask is None or attention_mask.ndim != 4:
            raise ValueError("Supply an explicit 4D block attention mask")
        if attention_mask.shape[-2:] != (shape[1], shape[1]):
            raise ValueError("Attention mask does not match the input grid")
        if position_ids is None:
            raise ValueError("Supply original DMax position_ids explicitly")
        hidden = self.model(input_ids, attention_mask, position_ids, inputs_embeds)
        if logits_to_keep is not None:
            # The student needs only candidate logits; the frozen feature extractor needs
            # no vocabulary projection when its decoder hook collects features.
            hidden = hidden[:, logits_to_keep if isinstance(logits_to_keep, slice) else slice(*logits_to_keep)]
        return MoeCausalLMOutputWithPast(logits=self.lm_head(hidden))
