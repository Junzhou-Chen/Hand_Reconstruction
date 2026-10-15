#!/usr/bin/env python3
"""Remote frozen-checkpoint wrist sensitivity/gradient diagnostics; no updates."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('code', 'run', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if not Path('/sciclone/home/jchen57').is_dir():
        parser.error('Run numerical diagnostics in the authorized remote Pod')
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('0', '1'):
        parser.error('Explicitly select one GPU')
    if args.output.exists():
        parser.error('Choose a new diagnostic receipt')
    sys.path[:0] = [str(args.code/'scripts'), str(args.code/'src')]
    from train_probabilistic_twm import (
        torch, ProbabilisticTWM, ProbabilisticTWMConfig, ForecastTrajectoryWindows,
        prepare_batch, objective, utc,
    )
    torch.set_num_threads(2)
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    meta = json.loads((args.run/'run_metadata.json').read_text())
    for name, expected in meta['source_sha256'].items():
        assert hashlib.sha256((args.code/name).read_bytes()).hexdigest() == expected, name
    checkpoint = torch.load(args.run/'latest.pt', map_location='cpu', weights_only=False)
    config = checkpoint['config']
    assert config == meta['config'] and checkpoint['code_version'] == meta['code_version']
    digest = hashlib.sha256()
    for name, value in checkpoint['model'].items():
        if value.is_floating_point():
            assert bool(torch.isfinite(value).all())
        digest.update((name+str(value.dtype)+str(tuple(value.shape))).encode())
        digest.update(value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    assert all(state['model_state_sha256'] == digest.hexdigest() for state in checkpoint['rank_states'])
    dataset = ForecastTrajectoryWindows(
        meta['manifest'], 'val', history=config['history_frames'],
        forecast_frames=config['model']['forecast_horizon'],
        min_future_frames=config['min_future_frames'], window_stride=config['window_stride'],
        cache_sequences=64, cache_bytes=128*1024**2)
    chosen = {source: indices[::max(1, len(indices)//8)][:8]
              for source, indices in dataset.fixed_indices_by_dataset(256).items()}
    samples = [dataset[index] for indices in chosen.values() for index in indices]
    batch = {key: torch.stack([sample[key] for sample in samples]).to(device) for key in samples[0]}
    model = ProbabilisticTWM(ProbabilisticTWMConfig(**config['model'])).to(device)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()
    state_before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    rng = torch.Generator(device=device).manual_seed(config['seed']+123)
    history, raw, candidate, kwargs, target, future = prepare_batch(
        batch, rng, 'noisy', condition_on_beta=config['condition_on_beta'],
        corruption_profile=config['corruption_profile'])

    def wrist_error(position):
        return float((position-target[:, :3]).norm(dim=-1).mean()*1000)

    baselines = {'candidate_mm': wrist_error(candidate[:, :3]),
                 'history_hold_mm': wrist_error(history[:, -1, :3])}
    for amount in (.25, .5, .75):
        baselines[f'fixed_history_hold_blend_{amount:g}_mm'] = wrist_error(
            (1-amount)*candidate[:, :3]+amount*history[:, -1, :3])
    times = batch['dt'].cumsum(-1)
    times -= times[:, -1:].clone()
    for length in (3, 5):
        t, p = times[:, -length-1:-1], history[:, -length:, :3]
        tc = t-t.mean(-1, keepdim=True)
        velocity = (tc[..., None]*(p-p.mean(1, keepdim=True))).sum(1)/tc.square().sum(-1, keepdim=True).clamp_min(1e-10)
        estimate = p.mean(1)-velocity*t.mean(-1, keepdim=True)
        baselines[f'past_{length}_linear_extrapolation_mm'] = wrist_error(estimate)
        baselines[f'fixed_half_candidate_half_past_{length}_mm'] = wrist_error(.5*candidate[:, :3]+.5*estimate)
    precision = {}
    projection = {}

    def capture_projection(module, inputs, output):
        projection['root'] = output[:, 0].detach().clone()

    hook = model.current_projection.register_forward_hook(capture_projection)
    try:
        for mode in ('bf16', 'fp32'):
            try:
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=mode == 'bf16'):
                    base = model(history, raw, candidate, **kwargs)
                    base_projection = projection['root']
                    delta = base['delta'][:, :3]
                    rows = []
                    for millimetres in (1., 10.):
                        for axis in range(3):
                            differences, projection_changes = [], []
                            for sign in (-1., 1.):
                                offset = torch.zeros(3, device=device)
                                offset[axis] = sign*millimetres/1000
                                changed_raw, changed_candidate = raw.clone(), candidate.clone()
                                changed_raw[:, :3] += offset
                                changed_candidate[:, :3] += offset
                                shifted = model(history, changed_raw, changed_candidate, **kwargs)
                                # -1 means perfect counteraction, 0 means unchanged residual.
                                slope = ((shifted['delta'][:, axis]-delta[:, axis])/offset[axis]).mean()
                                differences.append(float(slope))
                                projection_changes.append(float((projection['root'] != base_projection).float().mean()))
                            rows.append({'offset_mm': millimetres, 'axis': axis,
                                'mean_residual_response_slope': sum(differences)/2,
                                'mean_projection_changed_fraction': sum(projection_changes)/2})
                    precision[mode] = {'wrist_mm': wrist_error(base['state'][:, :3]),
                        'mean_correction_norm_mm': float(delta.norm(dim=-1).mean()*1000),
                        'correction_coordinate_mean_mm': (delta.mean(0)*1000).tolist(),
                        'correction_coordinate_std_mm': (delta.std(0)*1000).tolist(),
                        'projection_dtype': str(base_projection.dtype), 'sensitivity': rows}
            except RuntimeError as error:
                precision[mode] = {'error': str(error)[:1000]}
                if mode == 'bf16':
                    raise
    finally:
        hook.remove()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        prediction = model(history, raw, candidate, **kwargs)
    _, terms = objective(prediction, target, future, batch['future_valid'], batch['future_dt'], config)
    weights = {'restoration_loss': 1., 'current_nll_per_dof': config['current_nll_weight'],
               'forecast_nll_per_dof': config['forecast_nll_weight'],
               'best_trajectory_geometry_loss': config['forecast_geometry_weight'],
               'velocity_loss': config['velocity_weight']}
    gradients, gradient_rows = {}, {}
    for name, weight in weights.items():
        gradient, = torch.autograd.grad(terms[name]*weight, model.translation_head.weight,
                                       retain_graph=True, allow_unused=True)
        if gradient is not None:
            assert bool(torch.isfinite(gradient).all())
            gradients[name] = gradient.detach().float().flatten()
        gradient_rows[name] = {'term_value': float(terms[name].detach()), 'weight': weight,
                              'translation_head_grad_norm': float(gradient.norm()) if gradient is not None else 0.}
    reference = gradients['restoration_loss']
    for name, gradient in gradients.items():
        gradient_rows[name]['cosine_to_restoration'] = float(
            (gradient*reference).sum()/(gradient.norm()*reference.norm()).clamp_min(1e-12))
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in state_before.items())
    report = {'utc': utc(), 'run': str(args.run), 'checkpoint_step': checkpoint['step'],
        'checkpoint_model_sha256': digest.hexdigest(), 'training_code_version': checkpoint['code_version'],
        'manifest_sha256': meta['manifest_sha256'], 'source_sha256_verified': meta['source_sha256'],
        'selected_indices': chosen, 'samples': len(samples), 'corruption_seed': config['seed']+123,
        'corruption_mode': 'noisy', 'baselines': baselines, 'precision': precision,
        'loss_gradients_on_translation_head': gradient_rows,
        'head_weight_norm': float(model.translation_head.weight.norm()),
        'head_bias': model.translation_head.bias.detach().tolist(), 'parameters_unchanged': True,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'scope': 'Small fixed synthetic diagnostic sample, eval-mode frozen checkpoint. No optimizer, no production run mutation, not an accuracy benchmark or causal proof of a training failure.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
