python -u -m main \
  mode=sample_eval \
  data=openwebtext-split \
  model=small model.length=1024 \
  algo=mdlm algo.name=mmd algo.backbone=hf_dit \
  eval.checkpoint_path=hf://yresearch/MDLM-MMD-OWT/model.ckpt \
  loader.eval_batch_size=8 \
  trainer.devices=1 \
  sampling.predictor=ancestral_cache \
  sampling.noise_removal=ancestral \
  eval.num_samples=1000 \
  eval.owt_pareto_seeds=5 \
  'eval.owt_pareto_steps=[8,16,32]' \
  'eval.pareto_temperatures=[0.75,0.80,0.85,0.875,0.90,0.925,0.95,0.975,1.00,1.05,1.10,1.20]' \
  eval.generated_samples_path=owt_eval.json
