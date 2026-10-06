"""Checkpoint loading and EMA-only evaluation checkpoint saving."""

import os
import pickle
import re

import torch
from lightning.pytorch.plugins.io import TorchCheckpointIO


class MMDCheckpointIO(TorchCheckpointIO):
  """Save EMA inference weights or full last checkpoints."""

  def __init__(self, model):
    super().__init__()
    parameters = [p for p in model._get_parameters() if p.requires_grad]
    indices = {id(parameter): index for index, parameter in enumerate(parameters)}
    self.ema_parameter_count = len(parameters)
    self.ema_parameter_indices = {
      name: indices[id(parameter)]
      for name, parameter in model.named_parameters(remove_duplicate=False)
      if id(parameter) in indices}

  def _ema_checkpoint(self, checkpoint):
    if 'ema' not in checkpoint:
      raise ValueError('EMA checkpoint saving requires training.ema > 0.')
    shadows = checkpoint['ema']['shadow_params']
    if len(shadows) != self.ema_parameter_count:
      raise ValueError('EMA parameter count does not match the model.')
    # Keep student buffers as well as parameters, without changing the full payload.
    state = {key: value for key, value in checkpoint['state_dict'].items()
             if not key.startswith(('teacher.', 'fake_model.'))}
    for name, index in self.ema_parameter_indices.items():
      if state[name].shape != shadows[index].shape:
        raise ValueError(f'EMA parameter shape does not match {name}.')
      state[name] = shadows[index]
    result = {key: checkpoint[key] for key in (
      'epoch', 'global_step', 'pytorch-lightning_version') if key in checkpoint}
    result.update(
      state_dict=state, checkpoint_weights='ema',
      hyper_parameters={'config': checkpoint['hyper_parameters']['config']})
    return result

  def save_checkpoint(self, checkpoint, path, storage_options=None):
    # The strategy calls checkpoint I/O on global rank zero, then synchronizes.
    if not re.fullmatch(r'last(?:-v\d+)?\.ckpt', os.path.basename(os.fspath(path))):
      checkpoint = self._ema_checkpoint(checkpoint)
    super().save_checkpoint(checkpoint, path, storage_options=storage_options)


class _TokenizerPlaceholder:
  """Ignore serialized Rust tokenizer state; rebuild from the saved config."""

  def __init__(self, *args, **kwargs):
    pass

  def __setstate__(self, state):
    pass


class _CheckpointUnpickler(pickle.Unpickler):
  def find_class(self, module, name):
    if module.split('.')[0] == 'tokenizers':
      return _TokenizerPlaceholder
    return super().find_class(module, name)


class _CheckpointPickle:
  Unpickler = _CheckpointUnpickler
  load = pickle.load


def load_weights(model, checkpoint, use_ema=True):
  """Load inference weights, excluding known frozen/auxiliary training models."""
  ema_only = checkpoint.get('checkpoint_weights') == 'ema'
  if ema_only and not use_ema:
    raise ValueError('This checkpoint contains only EMA weights; use '
                     'eval.disable_ema=false, or load last.ckpt for live weights.')
  state = {key: value for key, value in checkpoint['state_dict'].items()
           if not key.startswith(('teacher.', 'fake_model.'))}
  model.load_state_dict(state, strict=True)
  if use_ema and not ema_only:
    if 'ema' not in checkpoint:
      raise ValueError('Checkpoint has no EMA weights; set eval.disable_ema=true for live weights.')
    parameters = [p for p in model._get_parameters() if p.requires_grad]
    shadows = checkpoint['ema']['shadow_params']
    if len(parameters) != len(shadows):
      raise ValueError('EMA parameter count does not match the inference model.')
    if any(p.shape != s.shape for p, s in zip(parameters, shadows)):
      raise ValueError('EMA parameter shapes do not match the inference model.')
    with torch.no_grad():
      for parameter, shadow in zip(parameters, shadows):
        parameter.copy_(shadow)
  # EMA has already been applied. Never swap in newly initialized EMA weights.
  model.ema = None
