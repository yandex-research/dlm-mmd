"""Pretrained LLaDA2/DMax loading."""

import logging

import torch

from distributed import distribute

logger = logging.getLogger(__name__)


def model_classes():
    from vendor.llada2.configuration_llada2_moe import LLaDA2MoeConfig
    from vendor.llada2.modeling_llada2_moe import LLaDA2MoeModelLM

    # HF writes the local model/config sources into complete exported checkpoints.
    LLaDA2MoeConfig.register_for_auto_class()
    LLaDA2MoeModelLM.register_for_auto_class("AutoModelForCausalLM")
    return LLaDA2MoeConfig, LLaDA2MoeModelLM


def load_pretrained_model(pretrained, revision=None, *, meta=False):
    """Load bf16 weights, or build an empty meta-device model that FSDP fills from rank 0."""
    config_class, model_class = model_classes()
    config = config_class.from_pretrained(pretrained, revision=revision)
    config._attn_implementation = "sdpa"
    config.use_cache = False
    config.fuse_cross_entropy = False
    # Do not retain an upstream bare-AutoModel entry: the local bare backbone
    # is an nn.Module; checkpoint consumers use the complete LM class.
    config.auto_map = {
        "AutoConfig": "configuration_llada2_moe.LLaDA2MoeConfig",
        "AutoModelForCausalLM": "modeling_llada2_moe.LLaDA2MoeModelLM",
    }
    if meta:
        with torch.device("meta"):
            model = model_class(config)
    else:
        model, information = model_class.from_pretrained(
            pretrained,
            revision=revision,
            config=config,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            output_loading_info=True,
        )
        # A missing or renamed weight would otherwise be silently randomly initialized.
        mismatches = {
            name: information[name]
            for name in ("missing_keys", "unexpected_keys", "mismatched_keys")
            if information.get(name)
        }
        if mismatches:
            raise RuntimeError(
                f"Checkpoint does not match the LLaDA2 parameter layout: {mismatches}"
            )
    # Never trust uninitialized nonpersistent buffers left by HF meta loading.
    model.model.rotary_emb.rebuild(torch.device("cpu"))
    return model


def load_model(config, context):
    """The trainable policy: FP32 master weights, bf16 compute under FSDP and autocast."""
    from transformers import AutoTokenizer

    from vendor.llada2.grouped_moe import choose_backend, pack_experts

    pretrained = str(config.model.pretrained_model)
    revision = config.model.revision
    if context.rank == 0:
        print(f"[model] Student: {pretrained}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(pretrained, revision=revision, trust_remote_code=True)
    # Under FSDP only rank 0 reads the 16B weights; FSDP broadcasts them to the other ranks.
    model = load_pretrained_model(
        pretrained, revision, meta=context.device.type == "cuda" and context.rank != 0
    )
    backend = choose_backend(context.device)
    logger.info(
        "Grouped MoE backend: %s", backend.__name__ if backend else "differentiable reference loop"
    )
    pack_experts(model, backend)
    if config.training.gradient_checkpointing_enable:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    # Buffers on parameterless modules are not materialized by FSDP's param_init_fn.
    model.model.rotary_emb.rebuild(context.device)
    return distribute(model.float(), context), tokenizer


def truncate_to_feature_layer(model, feature_layer):
    """Keep only the decoder prefix that produces the MMD features."""
    # Keep the original config depth; the forward iterates the actual ModuleList.
    model.model.layers = model.model.layers[: feature_layer + 1]
    model.lm_head = torch.nn.Identity()


def load_feature_extractor(config, context):
    """A frozen bf16 copy of the initial model, truncated after the MMD feature layer."""
    from vendor.llada2.grouped_moe import choose_backend, pack_experts

    pretrained = config.model.pretrained_model
    feature_layer = int(config.training.mmd.mmd_feature_layer)
    if context.rank == 0:
        print(f"[model] Feature extractor: {pretrained}, layer {feature_layer}", flush=True)
    # Loading must not change the policy's sampling RNG stream.
    devices = [context.device] if context.device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        model = load_pretrained_model(
            pretrained,
            config.model.revision,
            meta=context.device.type == "cuda" and context.rank != 0,
        )
        truncate_to_feature_layer(model, feature_layer)
        # Meta constructors default to fp32; all ranks must have the same frozen
        # parameter dtype before FSDP broadcasts module states.
        model.to(dtype=torch.bfloat16)
        model.model.rotary_emb.rebuild(context.device)
        pack_experts(model, choose_backend(context.device))
        model.requires_grad_(False).eval()
        return distribute(model, context).eval()
