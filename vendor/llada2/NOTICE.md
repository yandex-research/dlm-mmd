# LLaDA2 training model provenance

The model equations and parameter names derive from DMax commit
`82bc29ec433d94b3e53dfcf1626f34d62bf5a885`, file
`dFactory/models/llada2_moe/modeling_llada2_moe.py` and its configuration.
Upstream: https://github.com/czg1225/DMax
The original Antgroup/Hugging Face Apache-2.0 header is preserved.

Local adaptations: use explicit-mask SDPA, remove VeOmni/Liger,
use original separate-expert parameter names, recompute deterministic RoPE after
loading/meta initialization, and use nonreentrant activation checkpointing.
The same native expert equations retain autograd in both train and eval modes.
The training backbone has no decoding or KV-cache path. dInfer uses its own
model implementation for evaluation. The LM head returns compute-dtype logits;
the external objective handles fp32 cross entropy. `logits_to_keep=(start, stop)`
slices hidden states before the vocabulary projection to bound training memory.
An empty slice skips that projection when a hook on the frozen feature extractor reads features.
The standard `inputs_embeds` interface remains available for independent
input-gradient and attention-leakage checks.

The untouched upstream decoder definitions used for output/gradient parity tests
are retained in `tests/reference_llada2.py`, under their original Apache-2.0 header.

For training, the experts of each MoE layer are stacked for grouped GEMM
(`grouped_moe.py`). The checkpoint exporter undoes expert stacking and writes standard
separate-expert Hugging Face weights.
