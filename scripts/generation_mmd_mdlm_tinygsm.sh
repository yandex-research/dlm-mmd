export TOKENIZERS_PARALLELISM=false

python -u -m main \
  mode=sample_eval \
  data=tiny-gsm \
  model=small model.length=512 \
  algo=mdlm algo.name=mmd algo.backbone=dit \
  eval.checkpoint_path=hf://yresearch/MDLM-MMD-TinyGSM/model.ckpt \
  trainer.devices=1 \
  sampling.predictor=ancestral_cache \
  eval.gsm8k_batch_size=16 \
  eval.gsm8k_pareto_temperature=0.0 \
  'eval.gsm8k_pareto_thresholds=[0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.98]' \
  eval.generated_samples_path=tinygsm_eval.json
