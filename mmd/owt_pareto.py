"""OWT validation sweeps and the reference run-comparison logging format."""

import json
import math
from pathlib import Path
import statistics
import warnings

from hydra.utils import to_absolute_path
import torch


@torch.no_grad()
def sweep(model, temperatures, budgets, seeds=1, num_batches=2, *, rank=0,
          num_samples=None, on_point=None):
    """Paired temperatures; each rank contributes the configured sample count."""
    temperatures = sorted(set(float(t) for t in temperatures))
    budgets = sorted(set(int(b) for b in budgets))
    batch_size = int(model.config.loader.eval_batch_size)
    sample_count = batch_size * num_batches if num_samples is None else num_samples
    if (not temperatures or not budgets or min(budgets) <= 0
            or min(seeds, num_batches, batch_size, sample_count) <= 0
            or any(not math.isfinite(t) or t <= 0 for t in temperatures)):
        raise ValueError('OWT Pareto temperatures and counts must be positive and finite')
    original_temperature = model.sampling_temperature
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    curves = {}
    # Includes lazy reference-LM initialization, which may consume random numbers.
    with torch.random.fork_rng(devices=devices):
        try:
            for temperature in temperatures:
                model.sampling_temperature = temperature
                curve = []
                for budget in budgets:
                    entropies, perplexities, repeats = [], [], []
                    for repeat in range(seeds):
                        seed = 9973 * (repeat + 1) + budget + 104729 * rank
                        torch.manual_seed(seed)
                        # Preserve NLL metrics already logged by ordinary validation.
                        model.metrics.gen_ppl.reset()
                        model.metrics.sample_entropy.reset()
                        for start in range(0, sample_count, batch_size):
                            tokens = model.generate_samples(
                                num_samples=min(batch_size, sample_count - start),
                                num_steps=budget)
                            model.metrics.record_entropy(tokens)
                            texts = model.tokenizer.batch_decode(tokens)
                            model.metrics.record_generative_perplexity(
                                texts, model.num_tokens, device=model.device)
                        # All ranks participate: torchmetrics pools sums and weights.
                        entropies.append(float(model.metrics.sample_entropy.compute()))
                        perplexities.append(float(model.metrics.gen_ppl.compute()))
                        repeats.append({'seed': seed, 'gen_ppl': perplexities[-1],
                                        'entropy': entropies[-1]})
                    if not all(math.isfinite(v) for v in entropies + perplexities):
                        raise ValueError('Non-finite OWT Pareto metrics')
                    point = {'steps': budget, 'seeds': seeds}
                    for key, values in [('entropy', entropies), ('gen_ppl', perplexities)]:
                        point[key] = statistics.mean(values)
                        point[key + '_std'] = statistics.stdev(values) if seeds > 1 else 0.
                    if num_samples is not None:
                        point['repeats'] = repeats
                    curve.append(point)
                    if on_point is not None:
                        on_point(temperature, point)
                    if rank == 0:
                        print(f'[owt-pareto] T={temperature:g} steps={budget}: '
                              f'gen_ppl={point["gen_ppl"]:.4f} '
                              f'entropy={point["entropy"]:.4f}', flush=True)
                curves[temperature] = curve
        finally:
            model.sampling_temperature = original_temperature
    return curves


def log_validation_pareto(model):
    """Called while the validation EMA weights are active, on every rank."""
    cfg = model.config.eval
    every = int(cfg.get('owt_pareto_temp_every_n_validations', 0))
    if (every <= 0 or model.config.data.train != 'openwebtext-train'
            or model.trainer.sanity_checking
            or not cfg.get('pareto_temperatures') or not cfg.get('owt_pareto_steps')):
        return
    count = getattr(model, '_owt_pareto_temp_count', 0)
    model._owt_pareto_temp_count = count + 1
    if count % every or not cfg.compute_generative_perplexity:
        return
    curves = sweep(
        model, cfg.pareto_temperatures, cfg.owt_pareto_steps,
        seeds=int(cfg.owt_pareto_seeds),
        num_batches=int(cfg.owt_pareto_num_batches or model.config.sampling.num_sample_batches),
        rank=model.trainer.global_rank)
    publish(model, curves)


def evaluate(model, config):
    """Run standalone sample_eval with the same sampler and metrics as validation."""
    if model.num_tokens != 1024 or config.sampling.semi_ar:
        raise ValueError('OWT sweeps require full-sequence sampling with length 1024.')
    cfg = config.eval
    num_samples = cfg.num_samples
    if num_samples is None:
        num_samples = config.loader.eval_batch_size * config.sampling.num_sample_batches
    results = []
    checkpoint = Path(to_absolute_path(str(Path(cfg.checkpoint_path).expanduser())))
    is_checkpoint = cfg.checkpoint_path.startswith('hf://') or checkpoint.is_file()
    payload = {
        'checkpoint': cfg.checkpoint_path,
        'weights': ('live' if cfg.disable_ema else 'ema') if is_checkpoint else 'pretrained',
        'num_samples_per_seed': num_samples, 'seeds': int(cfg.owt_pareto_seeds),
        'batch_size': config.loader.eval_batch_size, 'sequence_length': model.num_tokens,
        'ppl_model': cfg.gen_ppl_eval_model_name_or_path,
        'seed_formula': '9973 * (repeat + 1) + steps; single process',
        'steps_exclude_final_noise_removal': True, 'reference_entropy': 5.43,
        'results': results,
    }
    output = Path(to_absolute_path(str(Path(cfg.generated_samples_path).expanduser())))
    output.parent.mkdir(parents=True, exist_ok=True)
    def save_point(temperature, point):
        results.append(dict(temperature=temperature, **{k: v for k, v in point.items()
                                                        if k != 'seeds'}))
        payload['nearest_entropy'] = [
            min((p for p in results if p['steps'] == budget),
                key=lambda p: abs(p['entropy'] - 5.43))
            for budget in sorted({p['steps'] for p in results})]
        temporary = output.with_suffix(output.suffix + '.tmp')
        temporary.write_text(json.dumps(payload, indent=2) + '\n')
        temporary.replace(output)

    sweep(model, cfg.pareto_temperatures or [config.sampling.temperature],
          cfg.owt_pareto_steps or [config.sampling.steps],
          seeds=int(cfg.owt_pareto_seeds), num_samples=int(num_samples),
          on_point=save_point)
    print(f'Results saved to {output}', flush=True)
    return results


def publish(model, curves):
    """Log sweep metrics and comparison curves under the Pareto namespace."""
    for temperature, curve in curves.items():
        for point in curve:
            tag = f's{point["steps"]:g}_t{float(temperature):g}'
            for metric in ('gen_ppl', 'entropy'):
                model.log(f'pareto/{metric}_{tag}',
                          torch.tensor(point[metric], device=model.device, dtype=torch.float32),
                          on_epoch=True, on_step=False, sync_dist=False, rank_zero_only=True)
    if model.trainer.global_rank != 0:
        return
    step = int(model.global_step)
    logger = model.trainer.logger
    path = Path(model.config.checkpointing.save_dir) / 'owt_pareto_temp_history.json'
    if getattr(model, '_owt_pareto_temp_history', None) is None:
        history = {}
        if step > 0 and model.trainer.ckpt_path is not None and path.is_file():
            try:
                stored = json.loads(path.read_text())
                # A new run may reuse a directory, or resume an earlier checkpoint.
                history = {int(k): v for k, v in stored['curves'].items() if int(k) <= step}
            except (OSError, ValueError, KeyError, TypeError) as error:
                warnings.warn(f'Could not restore OWT Pareto history: {error}')
        model._owt_pareto_temp_history = history
    history = model._owt_pareto_temp_history
    history[step] = {f'{float(t):g}': curve for t, curve in curves.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps({'curves': history}, indent=2) + '\n')
    temporary.replace(path)
    if logger is None:
        return
    by_budget = {}
    for temperature, curve in sorted(curves.items(), key=lambda pair: float(pair[0])):
        for point in sorted(curve, key=lambda p: p['steps']):
            by_budget.setdefault(point['steps'], []).append(point)
    for logger in model.trainer.loggers:
        for budget, points in sorted(by_budget.items()):
            for point in sorted(points, key=lambda p: p['entropy']):
                logger.log_metrics({f'pareto/temp_step_{step}_s{budget}': point['gen_ppl']},
                                   step=int(round(point['entropy'] * 1000)))
