"""GSM8K confidence-threshold evaluation and training-validation logging."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import warnings

import datasets
from filelock import FileLock
from hydra.utils import to_absolute_path
import torch
import torch.distributed as dist

THRESHOLDS = [i / 100 for i in range(35, 96, 5)] + [.98]


def load_examples(path, tokenizer, separator, length, limit=None):
    """Use the paper's BOS + stripped question + literal separator prompt."""
    records = json.loads(Path(path).expanduser().read_text())
    if not isinstance(records, list) or not records:
        raise ValueError('GSM8K data must be a nonempty JSON list.')
    separator_ids = tokenizer(separator, add_special_tokens=False).input_ids
    examples = []
    for index, record in enumerate(records[:limit]):
        question, answer = record.get('prompt'), record.get('response_ground_truth')
        if not isinstance(question, str) or not question.strip() or not isinstance(answer, str) or not answer.strip():
            raise ValueError(f'Example {index} requires prompt and response_ground_truth strings.')
        tokens = [tokenizer.bos_token_id] + tokenizer(
            question.strip(), add_special_tokens=False).input_ids + separator_ids
        if len(tokens) >= length:
            raise ValueError(f'Example {index} leaves no answer space at sequence length {length}.')
        examples.append({'input_ids': tokens, 'answer': answer})
    return examples


@torch.no_grad()
def generate_samples(model, prompts, threshold, temperature=0., max_steps=None):
    """Paper-run low_confidence_dynamic sampler, including the capped fallback.

    Confidence uses untempered probabilities, the comparison is strictly `>`,
    and a stalled row commits its most confident token. EOS does not stop a row:
    the reference finishes all 512 positions, including padding after the answer.
    """
    from trainer_base import sample_categorical

    if not prompts or any(not 0 < len(p) < model.num_tokens for p in prompts):
        raise ValueError('Each prompt must leave at least one answer position.')
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Threshold must be finite and in [0, 1].')
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError('Temperature must be finite and nonnegative.')
    if max_steps is not None and max_steps <= 0:
        raise ValueError('max_steps must be positive or None.')
    x = model.prior_sample(len(prompts), model.num_tokens)
    for row, prompt in enumerate(prompts):
        prefix = torch.as_tensor(prompt, dtype=torch.long, device=model.device)
        if bool((prefix == model.mask_index).any()):
            raise ValueError('Prompt cannot contain the diffusion mask token.')
        x[row, :len(prefix)] = prefix
    per_row = torch.zeros(len(prompts), dtype=torch.long, device=model.device)
    nfe = 0
    cap = max_steps or model.num_tokens
    for step in range(cap + 1):
        candidates = x == model.mask_index
        unfinished = candidates.any(dim=-1)
        if not bool(unfinished.any()):
            break
        per_row += unfinished.long()
        nfe += 1
        if step == cap:
            t = torch.full((len(prompts), 1), 1e-5, device=model.device)
        else:
            # Invert the original linear schedule from the current mask fraction.
            alpha = 1 - candidates.float().mean(dim=-1, keepdim=True)
            t = ((1 - alpha) / (1 - model.noise.eps)).clamp(1e-5, 1.)
        probs = model.forward(x, model._sigma_from_alphat(model.noise(t)[1])).exp()
        if temperature == 0:
            proposed = probs.argmax(dim=-1)
        elif temperature == 1:
            proposed = sample_categorical(probs)
        else:
            proposed = sample_categorical(torch.softmax(
                probs.clamp_min(1e-30).log() / temperature, dim=-1))
        confidence = probs.gather(-1, proposed.unsqueeze(-1)).squeeze(-1).float()
        confidence = confidence.masked_fill(~candidates, -float('inf'))
        if step == cap:
            transfer = candidates
        else:
            transfer = (confidence > threshold) & candidates
            stalled = unfinished & ~transfer.any(dim=-1)
            forced = torch.zeros_like(transfer)
            forced.scatter_(-1, confidence.argmax(dim=-1, keepdim=True), True)
            transfer = torch.where(stalled.unsqueeze(-1), forced, transfer)
        x = torch.where(transfer, proposed, x)
    if bool((x == model.mask_index).any()):
        raise ValueError('Decoder returned masked tokens after final noise removal.')
    return x, {'nfe': nfe, 'nfe_per_row': per_row}


def grade_sample(sample, answer, timeout_s=5.):
    """Run reference scoring outside the model process with an outer timeout."""
    worker = Path(__file__).resolve().with_name('sandbox_gsm8k.py')
    request = json.dumps({'sample': sample, 'answer': answer, 'timeout_s': timeout_s})
    try:
        result = subprocess.run([sys.executable, '-I', str(worker)], input=request,
                                text=True, capture_output=True, timeout=timeout_s + 5,
                                check=True)
    except subprocess.TimeoutExpired:
        return False
    # Infrastructure failures should stop evaluation, not silently lower accuracy.
    value = json.loads(result.stdout)
    if not isinstance(value, bool):
        raise ValueError('Execution worker returned a non-boolean grade.')
    return value


def resolve_data_path(config):
    """Reuse or download GSM8K test data; an explicit path never falls back."""
    explicit = config.eval.get('gsm8k_data_path')
    if explicit:
        path = Path(to_absolute_path(str(Path(explicit).expanduser())))
        if not path.is_file():
            raise FileNotFoundError(f'GSM8K evaluation data not found: {path}')
        return path
    candidates = []
    if config.data.get('cache_dir'):
        candidates.append(Path(config.data.cache_dir) / 'gsm8k_test.json')
    candidates.append(Path(__file__).resolve().parents[1] / 'data' / 'gsm8k_test.json')
    for candidate in candidates:
        path = Path(to_absolute_path(str(candidate.expanduser())))
        if path.is_file():
            return path
    path = Path(to_absolute_path(str(candidates[0].expanduser())))
    path.parent.mkdir(parents=True, exist_ok=True)
    # DDP ranks can reach the first evaluation together. Publish only complete JSON.
    with FileLock(str(path) + '.lock'):
        if not path.is_file():
            print(f'Preparing GSM8K test data at {path}', flush=True)
            samples = datasets.load_dataset(
                'openai/gsm8k', 'main', split='test', cache_dir=str(path.parent))
            records = [{'prompt': row['question'], 'response_ground_truth': row['answer']}
                       for row in samples]
            temporary = path.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(records, indent=2), encoding='utf-8')
            temporary.replace(path)
    return path


def _raise_if_failed(error):
    """Reach the same error boundary on every rank before reducing metrics."""
    if dist.is_available() and dist.is_initialized():
        errors = [None] * dist.get_world_size()
        dist.all_gather_object(errors, None if error is None else repr(error))
        if any(item is not None for item in errors):
            raise RuntimeError(f'GSM8K evaluation failed across ranks: {errors}') from error
    elif error is not None:
        raise error


def _settings(cfg, thresholds):
    thresholds = sorted(set(float(t) for t in thresholds))
    batch_size = int(cfg.get('gsm8k_batch_size', 16))
    workers = int(cfg.get('gsm8k_grading_workers', 4))
    cap = cfg.get('gsm8k_max_steps')
    temperature = float(cfg.get('gsm8k_pareto_temperature', 0.))
    timeout = float(cfg.get('tinygsm_exec_timeout', 5.))
    if min(batch_size, workers) <= 0 or (cap is not None and cap <= 0):
        raise ValueError('GSM8K batch size, grading workers and cap must be positive.')
    if not thresholds or any(not math.isfinite(t) or not 0 <= t <= 1 for t in thresholds):
        raise ValueError('GSM8K thresholds must be finite and in [0, 1].')
    if not math.isfinite(temperature) or temperature < 0 or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('GSM8K temperature must be nonnegative and timeout positive; both finite.')
    return thresholds, batch_size, workers, cap, temperature, timeout


def _evaluate_threshold(model, examples, threshold, pool, batch_size, temperature, cap, timeout):
    correct = count = total_nfe = batch_nfe = 0
    for start in range(0, len(examples), batch_size):
        batch = examples[start:start + batch_size]
        prompts = [example['input_ids'] for example in batch]
        samples, metadata = generate_samples(model, prompts, threshold, temperature, cap)
        texts = [model.tokenizer.decode(sample[len(prompt):].tolist(), skip_special_tokens=True)
                 for sample, prompt in zip(samples.cpu(), prompts)]
        pairs = zip(texts, [example['answer'] for example in batch])
        correct += sum(pool.map(lambda pair: grade_sample(*pair, timeout_s=timeout), pairs))
        count += len(batch)
        total_nfe += int(metadata['nfe_per_row'].sum())
        batch_nfe += metadata['nfe']
    return correct, count, total_nfe, batch_nfe


@torch.no_grad()
def evaluate(model, config):
    """Standalone GSM8K evaluation through main.py mode=sample_eval."""
    cfg = config.eval
    settings = _settings(cfg, cfg.get('gsm8k_pareto_thresholds') or THRESHOLDS)
    thresholds, batch_size, workers, cap, temperature, timeout = settings
    limit = cfg.get('gsm8k_num_examples')
    if limit is not None and limit <= 0:
        raise ValueError('eval.gsm8k_num_examples must be positive or null.')
    path = resolve_data_path(config)
    examples = load_examples(path, model.tokenizer, model.config.data.separator, model.num_tokens, limit)
    results = []
    payload = {
        'checkpoint': cfg.checkpoint_path, 'weights': 'live' if cfg.disable_ema else 'ema',
        'data_path': str(path), 'data_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'sequence_length': model.num_tokens, 'separator': model.config.data.separator,
        'batch_size': batch_size, 'temperature': temperature, 'seed': config.seed,
        'max_steps': cap or model.num_tokens,
        'sampler': 'low_confidence_dynamic', 'scoring': 'S-FLM simple_math_problem execution',
        'timeout_seconds': timeout,
        'nfe_definition': 'mean active forwards per example, including capped final removal',
        'results': results,
    }
    output = Path(to_absolute_path(str(Path(cfg.generated_samples_path).expanduser())))
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for threshold in thresholds:
            correct, count, total_nfe, batch_nfe = _evaluate_threshold(
                model, examples, threshold, pool, batch_size, temperature, cap, timeout)
            point = {'threshold': threshold, 'accuracy': correct / count,
                     'correct': correct, 'count': count,
                     'nfe': total_nfe / count, 'total_batch_forwards': batch_nfe}
            results.append(point)
            temporary = output.with_suffix(output.suffix + '.tmp')
            temporary.write_text(json.dumps(payload, indent=2) + '\n')
            temporary.replace(output)
            print(f'threshold={threshold:g}: {correct}/{count} '
                  f'({100 * point["accuracy"]:.2f}%) at {point["nfe"]:.2f} NFE/example', flush=True)
    print(f'Results saved to {output}', flush=True)
    return results


@torch.no_grad()
def sweep(model, examples):
    """Pool counts across rank-strided shards, including ranks with no examples."""
    cfg = model.config.eval
    error = None
    try:
        if not examples:
            raise ValueError('GSM8K examples must be nonempty.')
        settings = _settings(cfg, cfg.get('gsm8k_pareto_thresholds', THRESHOLDS))
        thresholds, batch_size, workers, cap, temperature, timeout = settings
    except Exception as caught:
        error = caught
    _raise_if_failed(error)
    distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if distributed else model.trainer.global_rank
    world_size = dist.get_world_size() if distributed else getattr(model.trainer, 'world_size', 1)
    local_examples = examples[rank::world_size]
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    curve = []
    with torch.random.fork_rng(devices=devices), ThreadPoolExecutor(max_workers=workers) as pool:
        for threshold in thresholds:
            torch.manual_seed(int(model.config.get('seed', 1)) + 104729 * rank)
            error = None
            try:
                correct, count, total_nfe, _ = _evaluate_threshold(
                    model, local_examples, threshold, pool, batch_size, temperature, cap, timeout)
            except Exception as caught:
                error = caught
            _raise_if_failed(error)
            totals = torch.tensor([correct, count, total_nfe], dtype=torch.float64, device=model.device)
            if distributed:
                dist.all_reduce(totals, op=dist.ReduceOp.SUM)
            correct, count, total_nfe = [int(value) for value in totals.tolist()]
            point = {'threshold': threshold, 'accuracy': correct / count,
                     'nfe': total_nfe / count, 'correct': correct, 'count': count}
            curve.append(point)
            if rank == 0:
                print(f'[gsm8k-pareto] step {model.global_step} thr {threshold:g}: '
                      f'{100 * point["accuracy"]:.2f}% ({correct}/{count}) '
                      f'at {point["nfe"]:.2f} NFE/example', flush=True)
    return curve


def log_validation_gsm8k(model, initial=False):
    """Called on every rank with EMA weights and evaluation mode already active."""
    cfg = model.config.eval
    every = int(cfg.get('gsm8k_every_n_validations', 1))
    if (model.config.data.train != 'tiny_gsm' or model.trainer.sanity_checking
            or not cfg.get('gsm8k_at_validation', False) or every <= 0
            or (initial and (not cfg.get('validate_before_train', False) or model.global_step != 0))):
        return
    count = getattr(model, '_gsm8k_validation_count', 0)
    model._gsm8k_validation_count = count + 1
    if count % every:
        return
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        error = None
        try:
            if getattr(model, '_gsm8k_examples', None) is None:
                limit = cfg.get('gsm8k_num_examples')
                if limit is not None and limit <= 0:
                    raise ValueError('eval.gsm8k_num_examples must be positive or null.')
                path = resolve_data_path(model.config)
                model._gsm8k_examples = load_examples(
                    path, model.tokenizer, model.config.data.separator, model.num_tokens, limit)
                model._gsm8k_protocol = {
                    'data_path': str(path), 'data_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                    'examples': len(model._gsm8k_examples), 'sequence_length': model.num_tokens,
                    'separator': model.config.data.separator,
                    'weights': 'ema' if getattr(model, 'ema', None) is not None else 'live',
                    'batch_size': cfg.get('gsm8k_batch_size', 16),
                    'world_size': getattr(model.trainer, 'world_size', 1),
                    'seed': model.config.get('seed', 1), 'rank_seed_offset': 104729,
                    'sampler': 'low_confidence_dynamic', 'temperature': cfg.get('gsm8k_pareto_temperature', 0.),
                    'max_steps': cfg.get('gsm8k_max_steps') or model.num_tokens,
                    'timeout_seconds': cfg.get('tinygsm_exec_timeout', 5.),
                    'nfe_definition': 'mean active forwards per example, including capped final removal'}
        except Exception as caught:
            error = caught
        _raise_if_failed(error)
        curve = sweep(model, model._gsm8k_examples)
        error = None
        try:
            publish(model, curve, initial=initial)
        except Exception as caught:
            error = caught
        _raise_if_failed(error)


def publish(model, curve, initial=False):
    """Log threshold metrics and comparison curves under the Pareto namespace."""
    logger = model.trainer.logger
    metrics = {}
    for point in curve:
        tag = f'{point["threshold"]:g}'.replace('.', '')
        for suffix, key in [('acc', 'accuracy'), ('nfe', 'nfe')]:
            name = f'pareto/gsm8k_{suffix}_thr{tag}'
            metrics[name] = point[key]
            if not initial:
                model.log(name, torch.tensor(point[key], device=model.device, dtype=torch.float32),
                          on_epoch=True, on_step=False, sync_dist=False, rank_zero_only=True)
    if model.trainer.global_rank != 0:
        return
    step = int(model.global_step)
    if initial and logger is not None:
        for destination in model.trainer.loggers:
            destination.log_metrics(metrics, step=step)
    path = Path(model.config.checkpointing.save_dir) / 'pareto_history.json'
    if getattr(model, '_gsm8k_pareto_history', None) is None:
        history = {}
        if step > 0 and model.trainer.ckpt_path is not None and path.is_file():
            try:
                stored = json.loads(path.read_text())
                # A new run may reuse a directory, or resume an earlier checkpoint.
                history = {int(k): v for k, v in stored['curves'].items() if int(k) <= step}
            except (OSError, ValueError, KeyError, TypeError) as error:
                warnings.warn(f'Could not restore GSM8K Pareto history: {error}')
        model._gsm8k_pareto_history = history
    history = model._gsm8k_pareto_history
    history[step] = curve
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps({'curves': history,
                                     'protocol': getattr(model, '_gsm8k_protocol', {})}, indent=2) + '\n')
    temporary.replace(path)
    if logger is None:
        return
    for logger in model.trainer.loggers:
        for point in sorted(curve, key=lambda p: p['threshold']):
            logger.log_metrics({f'pareto/gsm8k_step_{step}': point['accuracy']},
                               step=int(round(point['nfe'] * 100)))
