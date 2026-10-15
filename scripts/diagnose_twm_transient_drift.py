#!/usr/bin/env python3
"""Remote frozen TWM tests for temporal wrist bursts and sampler event coverage."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('code', 'run', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    if not Path('/sciclone/home/jchen57').is_dir() or os.environ.get('CUDA_VISIBLE_DEVICES') not in ('0', '1'):
        parser.error('Run remotely with one explicitly selected GPU')
    if args.output.exists():
        parser.error('Choose a fresh output')
    sys.path[:0] = [str(args.code/'scripts'), str(args.code/'src')]
    from train_probabilistic_twm import torch, g, ProbabilisticTWM, ProbabilisticTWMConfig, ForecastTrajectoryWindows, prepare_batch, utc
    torch.set_num_threads(2)
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    meta = json.loads((args.run/'run_metadata.json').read_text())
    for name, expected in meta['source_sha256'].items():
        assert hashlib.sha256((args.code/name).read_bytes()).hexdigest() == expected, name
    ckpt = torch.load(args.run/'latest.pt', map_location='cpu', weights_only=False)
    config = ckpt['config']
    assert config == meta['config']
    digest = hashlib.sha256()
    for name, value in ckpt['model'].items():
        if value.is_floating_point():
            assert bool(torch.isfinite(value).all())
        digest.update((name+str(value.dtype)+str(tuple(value.shape))).encode())
        digest.update(value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    assert all(state['model_state_sha256'] == digest.hexdigest() for state in ckpt['rank_states'])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def dataset(split):
        return ForecastTrajectoryWindows(meta['manifest'], split, history=config['history_frames'],
            forecast_frames=config['model']['forecast_horizon'], min_future_frames=config['min_future_frames'],
            window_stride=config['window_stride'], cache_sequences=64, cache_bytes=256*1024**2)

    def collate(samples):
        return {key: torch.stack([sample[key] for sample in samples]).to(device) for key in samples[0]}

    def distribution(values):
        values = values.detach().float().flatten()
        return {'mean': float(values.mean()), 'p50': float(values.quantile(.5)), 'p95': float(values.quantile(.95))}

    train = dataset('train')
    train_ids = train.fixed_indices_by_dataset(1)
    base = collate([train[ids[0]] for ids in train_ids.values()])
    rows = torch.arange(256, device=device) % len(train_ids)
    batch = {key: value[rows] for key, value in base.items()}
    rng = torch.Generator(device=device).manual_seed(config['seed']+711)
    audit_counts, clean_enclosed, case_counts, example = {}, 0, torch.zeros(5, device=device, dtype=torch.long), None
    total = 0
    with torch.no_grad():
        for _ in range(16):
            prepared, diagnostics = prepare_batch(batch, rng, 'mixed', condition_on_beta=config['condition_on_beta'],
                corruption_profile=config['corruption_profile'], return_diagnostics=True)
            clean = batch['past_current']
            actual = torch.cat((diagnostics['history_world'], diagnostics['raw_world'][:, None]), 1)
            error = (actual[..., :3]-clean[..., :3]).norm(dim=-1)*1000
            bad = error > 6.
            historical_bad_count = bad[:, :-1].sum(-1)
            last_ten_bad_count = bad[:, -10:].sum(-1)
            for label, values in (('historical_bad_frame_count', historical_bad_count), ('last_ten_bad_frame_count', last_ten_bad_count)):
                hist = torch.bincount(values, minlength=31).cpu().tolist()
                for index, count in enumerate(hist):
                    if count:
                        audit_counts.setdefault(label, {})[str(index)] = audit_counts.setdefault(label, {}).get(str(index), 0)+count
            # Exactly 3/4 bad frames surrounded by good observed history on both sides.
            for length in (3, 4):
                for start in range(1, 30-length):
                    event = (~bad[:, start-1]) & bad[:, start:start+length].all(-1) & (~bad[:, start+length])
                    clean_enclosed += int(event.sum())
            case_counts += torch.bincount(diagnostics['case_id'], minlength=5)
            if example is None:
                candidates = ((diagnostics['history_level'] == 3) & (diagnostics['case_id'] == 2)).nonzero().flatten()
                if len(candidates):
                    i = int(candidates[0])
                    example = {'case': 'ordinary_with_medium_history', 'error_norm_mm': error[i].cpu().tolist()}
            total += len(batch['dt'])
    audit = {'samples': total, 'threshold_mm': 6., 'train_reference_indices': train_ids,
        'histograms': audit_counts, 'enclosed_three_or_four_frame_runs': clean_enclosed,
        'case_counts': case_counts.cpu().tolist(), 'example': example,
        'scope': 'Independent same-sampler draws, not replay of production batches. Pre-fill wrist noise relative to source reference; historical availability is always true.'}
    del train, batch, base, prepared, diagnostics
    print(json.dumps({'phase': 'sampler_audited', 'audit': {key: value for key, value in audit.items() if key != 'example'}}), flush=True)

    val = dataset('val')
    selected, samples, sources = {}, [], []
    for source, indices in val.fixed_indices_by_dataset(256).items():
        selected[source] = []
        for target_index in torch.linspace(0, len(indices)-1, 16).round().long().tolist():
            for attempt in range(len(indices)):
                index = indices[(target_index+attempt) % len(indices)]
                if index in selected[source]:
                    continue
                sample = val[index]
                if bool(sample['future_valid'][:9].all()):
                    selected[source].append(index)
                    samples.append(sample)
                    sources.append(source)
                    break
            else:
                raise RuntimeError('Source cannot supply 16 distinct nine-frame continuation windows')
    data = collate(samples)
    del val, samples
    truth = torch.cat((data['past_current'], data['future'][:, :9]), 1)
    future_steps = torch.diff(data['future_dt'][:, :9], prepend=torch.zeros_like(data['future_dt'][:, :1]), dim=-1)
    full_dt = torch.cat((data['dt'], future_steps), 1)
    assert truth.shape[1] == 40 and bool((full_dt > 0).all())
    timeline = full_dt.cumsum(-1)
    b, h = len(truth), config['history_frames']
    motion = {}
    for source in selected:
        mask = torch.tensor([name == source for name in sources], device=device)
        step = (truth[mask, 1:, :3]-truth[mask, :-1, :3]).norm(dim=-1)*1000
        motion[source] = {'step_mm': distribution(step), 'dt_seconds': distribution(full_dt[mask, 1:]),
            'step_over_10mm_fraction': float((step > 10).float().mean()),
            'step_over_30mm_fraction': float((step > 30).float().mean())}
    model = ProbabilisticTWM(ProbabilisticTWMConfig(**config['model'])).to(device)
    model.load_state_dict(ckpt['model'], strict=True)
    model.eval()
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    direction_rng = torch.Generator(device=device).manual_seed(config['seed']+901)
    direction = torch.randn(b, 3, device=device, generator=direction_rng)
    direction /= direction.norm(dim=-1, keepdim=True)

    def to_anchor(state, anchor):
        rot = g.state_to_rotmat(state)
        ra = g.state_to_rotmat(anchor)[:, 0]
        extra = state.ndim-2
        r = ra.reshape(len(state), *([1]*extra), 3, 3)
        p = anchor[:, :3].reshape(len(state), *([1]*extra), 3)
        pos = (r.transpose(-1, -2) @ (state[..., :3]-p)[..., None]).squeeze(-1)
        rot = rot.clone()
        rot[..., 0, :, :] = r.transpose(-1, -2) @ rot[..., 0, :, :]
        return g.state_from_rotmat(pos, rot)

    @torch.no_grad()
    def infer(history, current, dt, source_rows):
        results = []
        for start in range(0, len(current), 128):
            end = min(start+128, len(current))
            past, raw_world, times = history[start:end], current[start:end], dt[start:end]
            anchor = past[:, -1]
            raw = to_anchor(raw_world, anchor)
            kwargs = {'history_valid': torch.ones(end-start, h, device=device, dtype=torch.bool),
                'history_availability': torch.ones(end-start, h, 17, device=device, dtype=torch.bool),
                'availability': torch.ones(end-start, 17, device=device, dtype=torch.bool),
                'candidate_availability': torch.ones(end-start, 17, device=device, dtype=torch.bool),
                'dt': times, 'beta': data['beta'][source_rows[start:end]],
                'beta_valid': torch.zeros(end-start, device=device, dtype=torch.bool),
                'handedness': data['handedness'][source_rows[start:end]],
                'quality': -torch.ones(end-start, 17, device=device), 'history_quality': None,
                'forecast_dt': data['forecast_dt'][source_rows[start:end]]}
            with torch.autocast('cuda', dtype=torch.bfloat16):
                output = model(to_anchor(past, anchor), raw, raw.clone(), **kwargs)['state']
            ra = g.state_to_rotmat(anchor)[:, 0]
            p = (ra @ output[:, :3, None]).squeeze(-1)+anchor[:, :3]
            rot = g.state_to_rotmat(output)
            rot[:, 0] = ra @ rot[:, 0]
            results.append(g.state_from_rotmat(p, rot))
        return torch.cat(results)

    @torch.no_grad()
    def evaluate(observed, event_steps, mode):
        if mode == 'observed_history':
            history = torch.cat([observed[:, t:t+h] for t in range(10)])
            current = torch.cat([observed[:, h+t] for t in range(10)])
            dt = torch.cat([full_dt[:, t:t+h+1] for t in range(10)])
            states = infer(history, current, dt, torch.arange(b, device=device).repeat(10)).reshape(10, b, 99).transpose(0, 1)
        else:
            completed, output = observed[:, :h].clone(), []
            for t in range(10):
                state = infer(completed[:, -h:], observed[:, h+t], full_dt[:, t:t+h+1], torch.arange(b, device=device))
                output.append(state)
                completed = torch.cat((completed, state[:, None]), 1)
            states = torch.stack(output, 1)
        inp = observed[:, h:, :3]
        target = truth[:, h:, :3]
        errors = {'input': (inp-target).norm(dim=-1)*1000,
                  'model': (states[..., :3]-target).norm(dim=-1)*1000}
        # Causal baselines always use observed states, never future/GT endpoints.
        estimates = []
        for t in range(10):
            past = observed[:, t:t+h, :3]
            ts = timeline[:, t:t+h]-timeline[:, h+t:h+t+1]
            tc = ts[:, -5:]-ts[:, -5:].mean(-1, keepdim=True)
            velocity = (tc[..., None]*(past[:, -5:]-past[:, -5:].mean(1, keepdim=True))).sum(1)/tc.square().sum(-1, keepdim=True).clamp_min(1e-10)
            linear = past[:, -5:].mean(1)-velocity*ts[:, -5:].mean(-1, keepdim=True)
            estimates.append((past[:, -1], linear,
                torch.cat((past[:, -4:], inp[:, t:t+1]), 1).median(1).values))
        for j, name in enumerate(('hold', 'linear5', 'median5')):
            estimate = torch.stack([entry[j] for entry in estimates], 1)
            errors[name] = (estimate-target).norm(dim=-1)*1000
        errors['half_current_half_linear5'] = (.5*inp+.5*torch.stack([row[1] for row in estimates], 1)-target).norm(dim=-1)*1000
        phases = {'during': event_steps}
        if event_steps and min(event_steps) > 0:
            phases['before'] = list(range(min(event_steps)))
        if event_steps and max(event_steps) < 9:
            phases['after'] = list(range(max(event_steps)+1, 10))
        if not event_steps:
            phases = {'all': list(range(10))}
        summary = {phase: {name: float(values[:, steps].mean()) for name, values in errors.items()}
                   for phase, steps in phases.items()}
        return {'phase_mean_error_mm': summary,
            'trace_mean_error_mm': {name: values.mean(0).cpu().tolist() for name, values in errors.items()},
            'per_source_during_mm': {source: {name: float(values[torch.tensor([s == source for s in sources], device=device)][:, event_steps or list(range(10))].mean()) for name, values in errors.items()} for source in selected},
            'positions': states[..., :3].cpu().tolist()}

    cases = [('clean', 0., [], False, False)]
    for amplitude in (10., 30.):
        cases += [(name, amplitude, steps, ramp, common) for name, steps, ramp, common in (
            ('single', [5], False, False), ('scattered3', [2, 5, 8], False, False),
            ('block3', [4, 5, 6], False, False), ('block4', [3, 4, 5, 6], False, False),
            ('ramp4', [3, 4, 5, 6], True, False), ('common_bias', list(range(10)), False, True))]
    outcomes = {}
    for name, amplitude, steps, ramp, common in cases:
        key = f'{name}_{amplitude:g}mm'
        observed = truth.clone()
        if common:
            observed[..., :3] += direction[:, None]*(amplitude/1000)
        else:
            for i, step in enumerate(steps):
                observed[:, h+step, :3] += direction*(amplitude/1000)*((i+1)/len(steps) if ramp else 1.)
        modes = ('observed_history', 'corrected_history') if name in ('clean', 'block4') else ('observed_history',)
        outcomes[key] = {'corrupt_steps_zero_based': steps, 'amplitude_mm': amplitude,
                         'history_before_segment_clean': not common}
        for mode in modes:
            outcomes[key][mode] = evaluate(observed, steps, mode)
        print(json.dumps({'phase': 'scenario_complete', 'case': key,
            'result': {mode: outcomes[key][mode]['phase_mean_error_mm'] for mode in modes}}), flush=True)
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in before.items())
    report = {'utc': utc(), 'run': str(args.run), 'checkpoint_step': ckpt['step'],
        'checkpoint_model_sha256': digest.hexdigest(), 'training_code_version': ckpt['code_version'],
        'manifest_sha256': meta['manifest_sha256'], 'source_sha256_verified': meta['source_sha256'],
        'noise_audit': audit, 'selected_val_indices': selected, 'samples': b, 'source_order': sources,
        'reference_motion': motion, 'scenarios': outcomes,
        'truth_segment_positions_m': truth[:, h:, :3].cpu().tolist(),
        'offset_directions_world': direction.cpu().tolist(), 'parameters_unchanged': True,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'scope': 'Frozen causal one-step current restoration on synthetic wrist-only bursts. Observed-history control and selected closed-loop corrected-history rollouts. No images, no future observations as inputs, no test labels, no optimizer. Not real drift accuracy or a proof of why training failed.'}
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'phase': 'complete', 'output': str(args.output), 'checkpoint_step': ckpt['step'],
        'samples': b, 'parameters_unchanged': True, 'peak_allocated_bytes': report['peak_allocated_bytes']}), flush=True)


if __name__ == '__main__':
    main()
