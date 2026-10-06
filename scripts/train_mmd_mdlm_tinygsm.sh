set -e

python - <<'PY'
from omegaconf import OmegaConf
import dataloader

config = OmegaConf.create({
    "data": OmegaConf.to_container(OmegaConf.load("configs/data/tiny-gsm.yaml")),
    "model": {"length": 512},
    "loader": {"num_workers": 4},
})
config.data.separator = r"\n"
tokenizer = dataloader.get_tokenizer(config)
dataloader.get_tiny_gsm_dataset(config, tokenizer)
PY

python -u -m main \
  data="tiny-gsm" \
  'data.separator="\n"' \
  model.length=512 \
  algo.backbone=dit \
  mmd.feature_layer=6 \
  mmd.rbf_alpha=6e-5 \
  mmd.average_candidates=False \
  sampling.steps=32 \
  loader.batch_size=32 \
  loader.num_workers=4 \
  strategy.find_unused_parameters=True \
  optim.lr=1e-5 \
  eval.generate_samples=False \
  eval.gsm8k_at_validation=True \
  eval.validate_before_train=True \
  'eval.gsm8k_pareto_thresholds=[0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.98]' \
  trainer.devices=8 \
  trainer.precision=bf16-mixed \
  trainer.gradient_clip_val=null \
  trainer.max_steps=8000 \
  trainer.num_sanity_val_steps=0 \
  trainer.val_check_interval=500 \
  trainer.limit_val_batches=500 \
  callbacks.checkpoint_every_n_steps.save_top_k=0 \
  callbacks.checkpoint_every_n_steps.save_last=False \
  checkpointing.resume_from_ckpt=False \
  checkpointing.use_periodic_checkpoint=True
