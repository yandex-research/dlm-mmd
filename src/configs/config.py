import math
from pathlib import Path

import yaml


class SamplingConfig:
    """Sampling configuration for generation."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise ValueError(f"Unknown sampling config field: {k}")
            setattr(self, k, v)

    def __repr__(self):
        fields = {k: getattr(self, k) for k in self.__class__.__annotations__}
        items = ", ".join(f"{k}={v!r}" for k, v in fields.items())
        return f"SamplingConfig({items})"

    sampling_method: str = "iterative_refinement"  # "iterative_refinement", "ode" or "sde"
    num_sampling_steps: list = [1, 2, 4, 8]
    cfgs: list = [1]
    self_cond_cfg_scales: list = [1.0]
    resample_z: bool = False  # Iterative refinement only: draw fresh noise before every pass after the first.
    time_schedule: str = "logit_normal"  # ODE/SDE only: "logit_normal", "uniform" or "shift".
    time_shift: float = 1.0  # Shift schedule only; values above 1 put more steps near the noisy end.
    sde_gamma: float = 0.0  # Per-step SDE churn fraction; 0.0 -> pure ODE. Used when sampling_method == "sde".


# ============================================
# Configuration
# ============================================
class Config:
    # Post-training
    objective: str = "mmd"  # "mmd" or "ird"
    task: str = "owt"  # "owt" (unconditional) or "tinygsm" (conditional)
    teacher_checkpoint: str = None  # Initializes the generator and the frozen model; local path or HF repo id.
    max_iters: int = 20000

    # MMD: bootstrap self-conditioning, then match frozen intermediate features.
    no_bootstrap_prob: float = 0.25  # Probability of a step without bootstrap passes.
    max_bootstrap_steps: int = 3  # Otherwise, draw the number of passes uniformly from 1..max_bootstrap_steps.
    feature_layer: int = 5  # Zero-based block of the frozen model whose outputs are compared.
    mmd_batch_size: int = 1  # Samples per kernel: OWT sequences per group, TinyGSM responses per prompt.
    sigma: float = None  # RBF bandwidth; required for MMD.
    unbiased_rbf: bool = False  # Exclude kernel pairs within one sample.
    mmd_block_size: int = 0  # Kernel tile size; 0 picks one from the batch shape.

    # IRD: match k self-conditioning passes of the frozen model with one generator pass.
    ird_steps: int = 8

    # Dataset
    data_path: str = None
    data_split: str = None  # Exact split name; None requires a single-split dataset.
    eval_data_path: str = None
    eval_data_split: str = None
    max_length: int = 128
    max_input_length: int = None  # Max length for conditioning input (e.g., prompt or encoder input); None = no limit
    pad_token: str = "pad"  # "pad" or "eos" - which token to use for padding

    # Tokenizer
    tokenizer_name: str = None  # Defaults to encoder_model_name if not set

    # Encoder
    encoder_family: str = "t5"  # "t5" or "gpt2"
    encoder_model_name: str = "t5-small"
    latent_mean: float | str = 0.0  # A scalar, local stats path, or org/repo/file on HF.
    latent_std: float = 1.0  # Ignored when latent_mean is a statistics file.

    # Model architecture
    model: str = "ELF-B"
    bottleneck_dim: int = 128  # Bottleneck dimension for text projection
    decoder_head_norm: bool = False  # RMSNorm before the decoder head (TinyGSM teacher).
    num_time_tokens: int = 4  # Number of in-context time conditioning tokens
    num_self_cond_cfg_tokens: int = 4  # Number of in-context self-cond CFG tokens
    num_model_mode_tokens: int = 4  # If > 0, prepend learnable model-mode tokens that signal decoding mode
    attn_dropout: float = 0.0
    proj_dropout: float = 0.0

    # Denoiser (noise scale for post-training; time distribution for ODE/SDE sampling)
    denoiser_p_mean: float = 0.8
    denoiser_p_std: float = 0.8
    denoiser_noise_scale: float = 2.0
    t_eps: float = 5e-2

    # Decoder objective
    decoder_prob: float = 0.2  # Fraction of each batch spent on decoder (CE) rows.
    decoder_noise_scale: float = 1.0  # Scale of noise in logit-normal-noised latent for CE branch
    decoder_p_mean: float = 0.8  # Mean for logit-normal noise schedule in decoder objective
    decoder_p_std: float = 0.8  # Std for logit-normal noise schedule in decoder objective

    # Conditioning / CFG
    self_cond_cfg_min: float = 0.5
    self_cond_cfg_max: float = 5.0

    # Training (optimizer + schedule)
    warmup_steps: int = 100
    global_batch_size: int = 128  # Distillation samples per optimizer update, over all devices.
    lr: float = 5e-5
    min_lr: float = 0.0
    lr_schedule: str = "constant"
    weight_decay: float = 0.0
    optimizer: str = "muon"  # "adamw" or "muon"
    adam_b1: float = 0.9
    adam_b2: float = 0.95
    use_bf16: bool = True  # Use CUDA BF16 autocast for training/eval forward passes.
    use_compile: bool = True  # Compile the training models and the eval model.
    gradient_checkpointing: bool = False  # Save activation memory by recomputing ELF blocks during backward.

    # EMA
    ema_decay1: float = 0.9999

    # Sampling
    sampling_configs_path: str = None
    # Sampling configs sweep (list of SamplingConfig objects, loaded from YAML)
    sampling_configs: list = [SamplingConfig()]
    num_samples: int = 100

    # Evaluation
    checkpoint_path: str = None  # Standalone evaluation default; --checkpoint_path takes precedence.
    online_eval: bool = True  # Score generated samples (OWT: GPT-2 Large PPL and entropy; TinyGSM: accuracy).
    eval_ppl_model: str = "gpt2-large"  # Model for PPL evaluation
    eval_ppl_batch_size: int = 64  # Batch size for PPL evaluation
    eval_ppl_max_length: int = 1024  # Max sequence length for PPL evaluation

    # Logging & Checkpointing (all intervals count optimizer updates)
    log_freq: int = 100
    eval_freq: int = 1000
    late_eval_start: int = 0  # After this step, evaluate every late_eval_freq steps instead.
    late_eval_freq: int = 0
    save_freq: int = 1000

    # Output
    output_dir: str = "./output_dir"
    hf_repo_id: str = None  # Optional HF repo id to mirror local outputs/checkpoints.
    resume: str = None

    # Wandb
    use_wandb: bool = False
    wandb_project: str = "elf-posttrain"
    wandb_entity: str = None
    wandb_run_name: str = None  # Defaults to the output directory name.
    wandb_tag: str = None  # Comma-separated tags.

    # Misc
    seed: int | str | list = 0  # "42, 43, 44": the first seed trains; every seed is used for sampling.
    num_workers: int = 8


def parse_seeds(value):
    """Read one seed, a comma-separated string, or a list."""
    if isinstance(value, str):
        value = value.split(",")
    if not isinstance(value, list):
        value = [value]
    seeds = [int(seed) for seed in value]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError(f"seed must list distinct integers, got {value!r}")
    return seeds


def _coerce(name, value):
    """Convert a YAML value to the field's declared type where PyYAML falls short."""
    if name == "sampling_configs":
        return [SamplingConfig(**entry) for entry in value]
    # PyYAML reads values such as 5e-5 as strings.
    if Config.__annotations__.get(name) is float and value is not None:
        return float(value)
    return value


def load_config_from_yaml(path: str) -> Config:
    """Load a YAML config and override defaults in Config."""
    config = Config()
    with open(path, "r") as f:
        cfg_dict = yaml.safe_load(f) or {}

    for key, value in cfg_dict.items():
        if not hasattr(config, key):
            raise ValueError(f"Unknown config field: {key}")
        setattr(config, key, _coerce(key, value))

    if config.sampling_configs_path:
        config.sampling_configs = load_sampling_configs(config.sampling_configs_path)
    if (isinstance(config.latent_mean, str) and not config.latent_mean.startswith("hf://")
            and not Path(config.latent_mean).is_file()):
        bundled_stats = Path(path).resolve().parent / config.latent_mean
        if bundled_stats.is_file():
            config.latent_mean = str(bundled_stats)
    return config


def apply_config_overrides(config: Config, overrides: list) -> Config:
    """Apply command-line overrides of the form field_name=value (values are parsed as YAML)."""
    for override in overrides or []:
        field_name, sep, value_str = override.partition("=")
        field_name = field_name.strip()
        if not sep:
            raise ValueError(f"Invalid override format: '{override}'. Expected 'field_name=value'")
        if not hasattr(config, field_name):
            raise ValueError(f"Config has no field named '{field_name}'")
        value = None if value_str.strip().lower() == "none" else yaml.safe_load(value_str)
        setattr(config, field_name, _coerce(field_name, value))
        if field_name == "sampling_configs":
            config.sampling_configs_path = None
        elif field_name == "sampling_configs_path" and value:
            config.sampling_configs = load_sampling_configs(value)
    return config


def load_sampling_configs(sampling_configs_path: str):
    """Return sampling configs, loading from sampling_configs_path if set."""
    with open(sampling_configs_path, "r") as f:
        entries = yaml.safe_load(f)
    return [SamplingConfig(**entry) for entry in entries]


def config_to_dict(config):
    """Return the declared settings as plain YAML-friendly values."""
    values = {name: getattr(config, name) for name in Config.__annotations__}
    values["sampling_configs"] = [
        {name: getattr(sc, name) for name in SamplingConfig.__annotations__}
        for sc in config.sampling_configs
    ]
    return values


def validate_config(config, *, training=True):
    """Reject settings the code does not support; everything else is trusted."""
    parse_seeds(config.seed)
    if config.task not in ("owt", "tinygsm"):
        raise ValueError("task must be owt or tinygsm")
    if config.encoder_family not in ("t5", "gpt2"):
        raise ValueError("encoder_family must be t5 or gpt2")
    if config.task == "tinygsm" and config.encoder_family != "gpt2":
        raise ValueError("TinyGSM requires the GPT-2 encoder")
    for sc in config.sampling_configs:
        if sc.sampling_method not in ("iterative_refinement", "ode", "sde"):
            raise ValueError(f"Unknown sampling method: {sc.sampling_method}")
        if sc.resample_z and sc.sampling_method != "iterative_refinement":
            raise ValueError("resample_z is only supported by iterative refinement")
    evaluates = not training or config.eval_freq or config.late_eval_freq
    if config.task == "tinygsm" and evaluates and not config.eval_data_path:
        raise ValueError("TinyGSM evaluation needs eval_data_path; set eval_freq=0 to train without it")
    if not training:
        return

    if config.objective not in ("mmd", "ird"):
        raise ValueError("objective must be mmd or ird")
    if not config.teacher_checkpoint:
        raise ValueError("teacher_checkpoint is required, including when resuming")
    if not 0 < config.decoder_prob < 1:
        raise ValueError("decoder_prob must lie in (0, 1)")
    if config.objective == "mmd" and (
        config.sigma is None or not math.isfinite(config.sigma) or config.sigma <= 0
    ):
        raise ValueError("MMD needs a finite positive sigma")
    if config.objective == "mmd" and config.unbiased_rbf and config.mmd_batch_size < 2:
        raise ValueError("unbiased_rbf needs mmd_batch_size >= 2")
