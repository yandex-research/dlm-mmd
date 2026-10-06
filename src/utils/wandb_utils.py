"""Weights & Biases logging. Only rank 0 logs; every function is a no-op when use_wandb is off."""

import os

from configs.config import config_to_dict
from utils.logging_utils import _process_index, log_for_0

try:
    import wandb
except ImportError:
    wandb = None


def init_wandb(config, job_type: str = "train"):
    """Start (or, when resuming training, continue) the run for `config.output_dir`.

    The run id is kept in `<output_dir>/wandb_run_id.txt`, so `resume=...`
    appends to the same W&B run instead of starting a new one.
    """
    if not config.use_wandb or _process_index() != 0:
        return None
    if wandb is None:
        raise ImportError("use_wandb=true needs the wandb package: pip install wandb")
    os.makedirs(config.output_dir, exist_ok=True)
    id_path = os.path.join(config.output_dir, f"wandb_{job_type}_run_id.txt")
    run_id = None
    if config.resume and os.path.isfile(id_path):
        with open(id_path) as f:
            run_id = f.read().strip()
    run = wandb.init(
        project=config.wandb_project, entity=config.wandb_entity,
        name=config.wandb_run_name or os.path.basename(os.path.normpath(config.output_dir)),
        tags=config.wandb_tag.split(",") if config.wandb_tag else None,
        config=config_to_dict(config), dir=config.output_dir, job_type=job_type,
        id=run_id, resume="must" if run_id else None,
    )
    with open(id_path, "w") as f:
        f.write(run.id)
    # Training metrics use the optimizer step; evaluation metrics are logged against the same axis.
    wandb.define_metric("train/step")
    wandb.define_metric("*", step_metric="train/step")
    log_for_0(f"W&B run: {run.url}")
    return run


def log_wandb(metrics: dict, step: int):
    if wandb is not None and wandb.run is not None:
        wandb.log({**metrics, "train/step": step})


def log_wandb_samples(key: str, columns: list, rows: list, step: int):
    """Log a table of generated samples."""
    if wandb is not None and wandb.run is not None:
        wandb.log({key: wandb.Table(columns=columns, data=rows), "train/step": step})


def finish_wandb():
    if wandb is not None and wandb.run is not None:
        wandb.finish()
