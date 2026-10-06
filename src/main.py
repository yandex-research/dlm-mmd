"""Train DMax with MMD and export a Hugging Face checkpoint for dInfer evaluation."""

import sys
from pathlib import Path

from omegaconf import OmegaConf


def load_config(argv):
    """Resolve YAML inheritance (`_base_`), apply key=value overrides and reject unknown keys."""
    overrides = OmegaConf.from_dotlist(argv)
    path = overrides.pop("config", Path(__file__).parent / "configs/dmax_math_mmd.yaml")

    def read(path):
        path = Path(path).resolve()
        config = OmegaConf.load(path)
        base = config.pop("_base_", None)
        return OmegaConf.merge(read(path.parent / base), config) if base else config

    schema = OmegaConf.load(Path(__file__).resolve().parent / "configs/base.yaml")
    OmegaConf.set_struct(schema, True)
    try:
        config = OmegaConf.merge(schema, read(path), overrides)
    except Exception as exc:
        raise ValueError(f"Invalid configuration: {exc}") from exc
    OmegaConf.resolve(config)
    return config


def main(config):
    from distributed import cleanup_distributed, initialize_distributed
    from model_loading import load_model
    from training import train

    context = initialize_distributed()
    try:
        if context.rank == 0:
            output = Path(config.output_dir)
            output.mkdir(parents=True, exist_ok=True)
            OmegaConf.save(config, output / "resolved_config.yaml")
        model, tokenizer = load_model(config, context)
        checkpoint = train(config, model, tokenizer, context)
        if context.rank == 0:
            print(f"Saved final model to {checkpoint}", flush=True)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main(load_config(sys.argv[1:]))
