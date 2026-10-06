import copy

import torch
import torch.nn.functional as F

import trainer_base
from mmd.checkpoint_utils import _CheckpointPickle
from mmd.objective import FeatureReward, reinforce_loss


def answer_content_mask(tokens, valid_tokens, tokenizer):
  """Score answer tokens through the first answer EOS, excluding padding."""
  selected = valid_tokens.bool()
  if tokenizer.pad_token_id is not None:
    selected = selected & tokens.ne(tokenizer.pad_token_id)
  if tokenizer.eos_token_id is not None:
    eos = tokens.eq(tokenizer.eos_token_id) & selected
    seen = F.pad(eos.cumsum(-1)[:, :-1], (1, 0))
    selected = selected & seen.eq(0)
  return selected


class MMD(trainer_base.AbsorbingState):
  """Representation-MMD training, or student-only checkpoint inference."""

  def __init__(self, config, tokenizer, *, inference_only=False):
    if (not inference_only
        and config.data.train not in {'openwebtext-train', 'tiny_gsm'}):
      raise ValueError('MMD reproduction supports OpenWebText and TinyGSM')
    super().__init__(config, tokenizer)
    self._validate_configuration()
    if inference_only:
      return
    self.teacher = copy.deepcopy(self.backbone).requires_grad_(False).eval()
    # Hugging Face returns an eval-mode model; enable the student's dropout.
    self.backbone.train()
    cfg = config.mmd
    self.reward = FeatureReward(
      self.teacher, layer=cfg.feature_layer, kernel=cfg.kernel,
      alpha=cfg.rbf_alpha, block_size=cfg.kernel_block_size,
      feature_batch_size=cfg.feature_batch_size,
      exclude_same_position=cfg.exclude_same_position)

  def _validate_configuration(self):
    assert self.sampler == 'ancestral_cache'

  def _process_model_output(self, model_output, xt, sigma):
    del sigma
    model_output[:, :, self.mask_index] += self.neg_infinity
    model_output = model_output - torch.logsumexp(
      model_output, dim=-1, keepdim=True)
    # Unmasked tokens retain their observed values with probability one.
    unmasked_indices = (xt != self.mask_index)
    model_output[unmasked_indices] = self.neg_infinity
    model_output[unmasked_indices, xt[unmasked_indices]] = 0
    return model_output

  def nll_per_token(self, log_x_theta, xt, x0, alpha_t,
                    dalpha_t, low_var=False):
    del xt
    log_p_theta = torch.gather(
      input=log_x_theta,
      dim=-1,
      index=x0[:, :, None]).squeeze(-1)
    return log_p_theta * dalpha_t / (1 - alpha_t)

  def initialize_from_checkpoint(self, path):
    """Warm start from raw MDLM weights and its saved EMA history.

    Student and frozen teacher use the live weights. The reference retains
    the pretrained EMA shadows and update counter, but starts a fresh
    optimizer and scheduler. Full MMD resumes instead use Trainer.fit.
    """
    checkpoint = torch.load(path, map_location='cpu', weights_only=False,
                            pickle_module=_CheckpointPickle)
    state = {k: v for k, v in checkpoint['state_dict'].items()
             if not k.startswith(('teacher.', 'fake_model.'))}
    state.update({'teacher.' + k[len('backbone.'):]: v
                  for k, v in list(state.items()) if k.startswith('backbone.')})
    self.load_state_dict(state, strict=True)
    if self.ema is not None:
      ema = checkpoint.get('ema')
      parameters = [p for p in self._get_parameters() if p.requires_grad]
      if ema is None or len(ema['shadow_params']) != len(parameters):
        raise ValueError('Pretrained checkpoint must contain matching EMA parameters')
      if any(p.shape != s.shape for p, s in zip(parameters, ema['shadow_params'])):
        raise ValueError('Pretrained checkpoint EMA shapes do not match the model')
      self.ema.load_state_dict(copy.deepcopy(ema))

  def train(self, mode=True):
    super().train(mode)
    if hasattr(self, 'teacher'):
      self.teacher.eval()
    return self

  def on_train_start(self):
    batch = self.config.loader.batch_size * self.trainer.world_size
    effective_batch = batch * self.trainer.accumulate_grad_batches
    if self.config.loader.global_batch_size != effective_batch:
      raise ValueError('global_batch_size must equal batch_size * world_size * '
                       'accumulate_grad_batches for the antithetic time sampler')
    self._mmd_metric_sum = None
    self._mmd_metric_step = self.global_step
    super().on_train_start()

  def training_step(self, batch, batch_idx):
    cfg = self.config.mmd
    x0 = batch['input_ids']
    accumulation = batch_idx % self.trainer.accumulate_grad_batches
    t = self._sample_t(x0.shape[0], accumulation)
    _, alpha = self.noise(t)
    alpha = alpha.unsqueeze(-1)
    valid = batch['attention_mask'] if self.config.data.train == 'tiny_gsm' else None
    # TinyGSM corrupts answer and padding, but scores only answer content.
    xt = self.q_xt(x0, alpha, valid_tokens=valid)
    selected = xt == self.mask_index
    if valid is not None:
      selected &= answer_content_mask(x0, valid, self.tokenizer)
    probabilities = self(xt, self._sigma_from_alphat(alpha)).exp()
    loss, stats = reinforce_loss(
      probabilities, x0, selected, self.reward,
      group_size=cfg.group_size, draws_per_candidate=cfg.draws_per_candidate,
      average_candidates=cfg.get('average_candidates', True))
    # Pool metric sums/counts across accumulation; reward counts active examples.
    counts = torch.stack([stats['reward_count'],
                          stats['reward_count'].new_tensor(x0.shape[0])]).float()
    values = torch.stack([stats['mmd/reward'].detach(),
                          stats['mmd/advantage_absmean'].detach()]).float()
    values = torch.cat([values * counts, counts])
    if self._mmd_metric_sum is None:
      self._mmd_metric_sum = values
    else:
      self._mmd_metric_sum += values
    return loss

  def on_train_batch_end(self, outputs, batch, batch_idx):
    if self.global_step == self._mmd_metric_step:
      return
    totals = self.trainer.strategy.reduce(self._mmd_metric_sum, reduce_op='sum')
    values = totals[:2] / totals[2:].clamp_min(1)
    self.log_dict({'mmd/reward': values[0],
                   'mmd/advantage_absmean': values[1],
                   # Lightning consumes this key as the logging axis.
                   'step': self.global_step},
                  on_step=True, on_epoch=False, sync_dist=False)
    self._mmd_metric_sum = None
    self._mmd_metric_step = self.global_step
