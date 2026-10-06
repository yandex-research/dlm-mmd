"""Frozen GPT-2 embeddings for OWT and TinyGSM."""

from typing import Optional

import torch
from torch import nn


class GPT2Encoder(nn.Module):
    """Return GPT-2's final normalized hidden states using causal attention."""

    def __init__(self, model_name: str, dtype: torch.dtype = torch.float32):
        super().__init__()
        from transformers import GPT2Model

        self.model = GPT2Model.from_pretrained(model_name, torch_dtype=dtype)
        self.model.requires_grad_(False).eval()
        self.config = self.model.config
        self.config.d_model = self.config.n_embd

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        deterministic: bool = True,
    ) -> torch.Tensor:
        """Accept a (batch, length) padding mask; GPT-2 supplies causality."""
        if attention_mask is not None and attention_mask.ndim != 2:
            raise ValueError("GPT-2 requires a two-dimensional padding mask")
        self.model.eval()
        return self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
