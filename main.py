import json
import os

import fsspec
import hydra
from huggingface_hub import hf_hub_download
import lightning as L
from lightning.pytorch.loops.training_epoch_loop import _TrainingEpochLoop
import omegaconf
import rich.syntax
import rich.tree
import torch

import algo
from mmd.checkpoint_utils import _CheckpointPickle, load_weights, MMDCheckpointIO
import dataloader
import utils

torch.autograd.set_detect_anomaly(True)

omegaconf.OmegaConf.register_new_resolver(
  'cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver(
  'device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver(
  'eval', eval)
omegaconf.OmegaConf.register_new_resolver(
  'div_up', lambda x, y: (x + y - 1) // y)


class GlobalBatchTrainingEpochLoop(_TrainingEpochLoop):
  """Schedule validation by optimizer updates, including across epoch boundaries."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.last_validation_step = 0

  def _should_check_val_fx(self, data_fetcher):
    step = self.global_step
    return (self.trainer.enable_validation and step > 0
            and not self._should_accumulate()
            and step != self.last_validation_step
            and step % self.trainer.val_check_interval == 0)

  def on_advance_end(self, data_fetcher):
    super().on_advance_end(data_fetcher)
    # A resumed checkpoint can run pending validation without an optimizer step.
    self._batches_that_stepped = self.global_step

  def on_save_checkpoint(self):
    state = super().on_save_checkpoint()
    state['last_validation_step'] = self.last_validation_step
    return state

  def on_load_checkpoint(self, state):
    super().on_load_checkpoint(state)
    self.last_validation_step = state.get('last_validation_step', 0)


class GlobalBatchValidation(L.Callback):
  def on_validation_end(self, trainer, pl_module):
    if not trainer.sanity_checking:
      # Lightning runs ordinary callbacks before ModelCheckpoint saves last.ckpt.
      trainer.fit_loop.epoch_loop.last_validation_step = trainer.global_step


class PeriodicStepCheckpoint(L.Callback):
  """Save checkpoints every N global batches (optimizer steps)."""

  def __init__(self, dirpath, every_n_train_steps):
    self.dirpath = os.fspath(dirpath)
    self.every_n_train_steps = int(every_n_train_steps)
    self._last_saved_step = 0

  def on_train_start(self, trainer, pl_module):
    self._last_saved_step = trainer.global_step

  def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
    if self.every_n_train_steps <= 0:
      return

    save_step = trainer.global_step
    if (save_step <= self._last_saved_step
        or save_step % self.every_n_train_steps != 0):
      return

    os.makedirs(self.dirpath, exist_ok=True)
    ckpt_path = os.path.join(self.dirpath, f'step={save_step:08d}.ckpt')
    # All DDP ranks must enter save_checkpoint because Lightning synchronizes
    # inside the strategy; only rank zero writes the file.
    trainer.save_checkpoint(ckpt_path)
    self._last_saved_step = save_step
    if trainer.is_global_zero:
      print(
        f'Saved periodic checkpoint at global_batch={save_step}: {ckpt_path}',
        flush=True)


def _load_from_checkpoint(diffusion_model, config, tokenizer):
  checkpoint = None
  source = config.eval.checkpoint_path
  if source.startswith('hf://'):
    parts = source[len('hf://'):].split('/', 2)
    if len(parts) != 3 or not all(parts):
      raise ValueError('Expected hf://owner/repository/checkpoint-file.')
    path = hf_hub_download(repo_id='/'.join(parts[:2]), filename=parts[2])
  else:
    path = hydra.utils.to_absolute_path(os.path.expanduser(source))
  model_config = omegaconf.OmegaConf.create(config)
  omegaconf.OmegaConf.set_struct(model_config, False)
  if os.path.isfile(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False,
                            pickle_module=_CheckpointPickle)
    saved = checkpoint.get('hyper_parameters', {}).get('config')
    if saved is None:
      raise ValueError('Checkpoint must contain hyper_parameters.config.')
    model_config = omegaconf.OmegaConf.merge(model_config, saved)
    pretrained = model_config.eval.checkpoint_path
    model_config.eval = config.eval
    model_config.eval.checkpoint_path = pretrained
    model_config.sampling = config.sampling
    model_config.loader = config.loader
  elif path.endswith('.ckpt'):
    raise FileNotFoundError(path)
  elif os.path.isdir(path):
    model_config.eval.checkpoint_path = path
  if config.eval.pretrained_path:
    pretrained = os.path.expanduser(config.eval.pretrained_path)
    local = hydra.utils.to_absolute_path(pretrained)
    model_config.eval.checkpoint_path = local if os.path.isdir(local) else pretrained
  if config.eval.tokenizer_path:
    tokenizer_path = os.path.expanduser(config.eval.tokenizer_path)
    local = hydra.utils.to_absolute_path(tokenizer_path)
    model_config.data.tokenizer_name_or_path = local if os.path.isdir(local) else tokenizer_path
  if checkpoint is not None or config.eval.tokenizer_path:
    tokenizer = dataloader.get_tokenizer(model_config)
  model_config.algo.name = 'mmd'
  model_config.training.ema = 0
  model = diffusion_model(model_config, tokenizer=tokenizer, inference_only=True)
  if checkpoint is not None:
    load_weights(model, checkpoint, use_ema=not config.eval.disable_ema)
  device = 'cpu' if config.trainer.accelerator == 'cpu' else 'cuda'
  return model.eval().to(device)


@L.pytorch.utilities.rank_zero_only
def _print_config(
  config: omegaconf.DictConfig,
  resolve: bool = True,
  save_cfg: bool = True) -> None:
  """Prints content of DictConfig using Rich library and its tree structure.
  
  Args:
    config (DictConfig): Configuration composed by Hydra.
    resolve (bool): Whether to resolve reference fields of DictConfig.
    save_cfg (bool): Whether to save the configuration tree to a file.
  """

  style = 'dim'
  tree = rich.tree.Tree('CONFIG', style=style, guide_style=style)

  fields = config.keys()
  for field in fields:
    branch = tree.add(field, style=style, guide_style=style)

    config_section = config.get(field)
    branch_content = str(config_section)
    if isinstance(config_section, omegaconf.DictConfig):
      branch_content = omegaconf.OmegaConf.to_yaml(
        config_section, resolve=resolve)

    branch.add(rich.syntax.Syntax(branch_content, 'yaml'))
  rich.print(tree)
  if save_cfg:
    with fsspec.open(
      '{}/config_tree.txt'.format(
        config.checkpointing.save_dir), 'w') as fp:
      rich.print(tree, file=fp)


@L.pytorch.utilities.rank_zero_only
def _print_batch(train_ds, valid_ds, tokenizer, k=64):
  for dl_type, dl in [
    ('train', train_ds), ('valid', valid_ds)]:
    print(f'Printing {dl_type} dataloader batch.')
    batch = next(iter(dl))
    print('Batch input_ids.shape', batch['input_ids'].shape)
    first = batch['input_ids'][0, :k]
    last = batch['input_ids'][0, -k:]
    print(f'First {k} tokens:', tokenizer.decode(first))
    print('ids:', first)
    print(f'Last {k} tokens:', tokenizer.decode(last))
    print('ids:', last)


def _generate_samples(diffusion_model, config, logger,
                      tokenizer):
  logger.info('Starting Sample Eval.')
  model = _load_from_checkpoint(
    diffusion_model=diffusion_model,
    config=config,
    tokenizer=tokenizer)
  if config.algo.name == 'mmd':
    if model.config.data.train == 'tiny_gsm':
      if model.config.algo.backbone != 'dit' or model.num_tokens != 512:
        raise ValueError('TinyGSM evaluation requires a native DiT with length 512.')
      from mmd.tinygsm_validation import evaluate
      evaluate(model, config)
      return
    if config.eval.owt_pareto_steps or config.eval.pareto_temperatures:
      from mmd.owt_pareto import evaluate
      evaluate(model, config)
      return
  model.metrics.gen_ppl.reset()
  model.metrics.sample_entropy.reset()
  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None
  stride_length = config.sampling.stride_length
  num_strides = config.sampling.num_strides
  all_samples = []
  for _ in range(config.sampling.num_sample_batches):
    if config.sampling.semi_ar:
      _, intermediate_samples, _ = model.restore_model_and_semi_ar_sample(
        stride_length=stride_length,
        num_strides=num_strides,
        dt=1 / config.sampling.steps)
      text_samples = intermediate_samples[-1]
      # Note: Samples generated using semi-ar method
      # need to to be processed before computing generative perplexity
      # since these samples contain numerous <|endoftext|> tokens
      # and diffusion.compute_generative_perplexity() discards
      # any text after the first EOS token.
    else:
      samples = model.restore_model_and_sample(
        num_steps=config.sampling.steps)
      model.metrics.record_entropy(samples)
      text_samples = model.tokenizer.batch_decode(samples)
      model.metrics.record_generative_perplexity(
        text_samples, model.num_tokens, device=model.device)
      all_samples.extend(list(text_samples))
  generative_ppl = 0.
  entropy = 0.
  if not config.sampling.semi_ar:
    generative_ppl = model.metrics.gen_ppl.compute().item()
    entropy = model.metrics.sample_entropy.compute().item()
    print('Generative perplexity:', generative_ppl)
    print('Sample entropy:', entropy)
  samples_path = config.eval.generated_samples_path
  if '://' not in samples_path:
    samples_path = hydra.utils.to_absolute_path(os.path.expanduser(samples_path))
    os.makedirs(os.path.dirname(samples_path), exist_ok=True)
  with fsspec.open(samples_path, 'w') as f:
    json.dump({'generative_ppl': generative_ppl,
               'entropy': entropy,
               'generated_seqs': all_samples}, f, indent=4)
  print('Samples saved at:', samples_path)

def _eval_ppl(diffusion_model, config, logger, tokenizer):
  logger.info('Starting Perplexity Eval.')

  model = _load_from_checkpoint(
    diffusion_model=diffusion_model,
    config=config,
    tokenizer=tokenizer)
  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None

  loggers = hydra.utils.instantiate(config.logger)
  callbacks = []
  if 'callbacks' in config:
    for _, callback in config.callbacks.items():
      callbacks.append(hydra.utils.instantiate(callback))
  trainer = hydra.utils.instantiate(
    config.trainer,
    default_root_dir=os.getcwd(),
    callbacks=callbacks,
    strategy=hydra.utils.instantiate(config.strategy),
    logger=loggers)
  _, valid_ds = dataloader.get_dataloaders(
    config, tokenizer, skip_train=True, valid_seed=config.seed)
  trainer.validate(model, valid_ds)


def _train(diffusion_model, config, logger, tokenizer):
  logger.info('Starting Training.')
  if (type(config.trainer.val_check_interval) is not int
      or config.trainer.val_check_interval <= 0):
    raise ValueError('trainer.val_check_interval must be a positive integer '
                     'number of global batches (optimizer steps).')
  loggers = hydra.utils.instantiate(config.logger)

  if (config.checkpointing.resume_from_ckpt
      and config.checkpointing.resume_ckpt_path is not None
      and utils.fsspec_exists(
        config.checkpointing.resume_ckpt_path)):
    ckpt_path = config.checkpointing.resume_ckpt_path
  else:
    ckpt_path = None

  # Lightning callbacks
  callbacks = [GlobalBatchValidation()]
  if 'callbacks' in config:
    for _, callback in config.callbacks.items():
      instance = hydra.utils.instantiate(callback)
      if isinstance(instance, L.pytorch.callbacks.ModelCheckpoint):
        # last.ckpt must contain training state, not link to an EMA-only file.
        instance.save_weights_only = False
        if instance.save_last == 'link':
          instance.save_last = True
      callbacks.append(instance)
    if config.checkpointing.get('use_periodic_checkpoint', False):
      callbacks.append(PeriodicStepCheckpoint(
        os.path.join(os.fspath(config.checkpointing.save_dir), 'checkpoints'),
        config.callbacks.checkpoint_every_n_steps.every_n_train_steps))

  train_ds, valid_ds = dataloader.get_dataloaders(
    config, tokenizer)
  _print_batch(train_ds, valid_ds, tokenizer)

  model = diffusion_model(config, tokenizer=tokenizer)
  if ckpt_path is None and config.data.train == 'tiny_gsm':
    model.initialize_from_checkpoint(hf_hub_download(
      repo_id='jdeschena/s-flm', filename='tinygsm/mdlm.ckpt'))

  trainer = hydra.utils.instantiate(
    config.trainer,
    default_root_dir=os.getcwd(),
    callbacks=callbacks,
    strategy=hydra.utils.instantiate(config.strategy),
    logger=loggers,
    plugins=MMDCheckpointIO(model),
    check_val_every_n_epoch=None,
    enable_progress_bar=False if config.debug else True)
  trainer.fit_loop.epoch_loop = GlobalBatchTrainingEpochLoop(
    trainer, min_steps=trainer.min_steps, max_steps=trainer.max_steps)
  trainer.fit(model, train_ds, valid_ds, ckpt_path=ckpt_path)


@hydra.main(version_base=None, config_path='configs',
            config_name='config')
def main(config):
  """Main entry point for training."""
  if config.data.cache_dir:
    config.data.cache_dir = hydra.utils.to_absolute_path(config.data.cache_dir)
  L.seed_everything(config.seed)
  _print_config(config, resolve=True, save_cfg=True)
  
  logger = utils.get_logger(__name__)
  tokenizer = dataloader.get_tokenizer(config)
  if config.algo.name == 'mmd':
    diffusion_model = algo.MMD
  else:
    raise ValueError(
      f'Invalid algorithm name: {config.algo.name}')
  kwargs = {'diffusion_model': diffusion_model,
            'config': config,
            'tokenizer': tokenizer,
            'logger': logger}
  if config.mode == 'sample_eval':
    _generate_samples(**kwargs)
  elif config.mode == 'ppl_eval':
    _eval_ppl(**kwargs)
  else:
    _train(**kwargs)


if __name__ == '__main__':
  main()
