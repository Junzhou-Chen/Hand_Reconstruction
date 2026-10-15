#!/usr/bin/env python3
"""Summarize saved TWM forecast validation remotely without running a model."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path


def forecast_metrics(values):
    return {key: value for key, value in values.items()
            if key.startswith(('future_', 'forecast_')) or key in (
                'mode_entropy', 'samples', 'counts',
                'calibration/future_xyz_90pct_marginal',
                'candidate/wrist_mm', 'restored/wrist_mm')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not Path('/sciclone/home/jchen57').is_dir():
        parser.error('Run this analysis in the authorized remote compute Pod')
    read = lambda path: json.loads(path.read_text())
    metadata = read(args.run / 'run_metadata.json')
    status = read(args.run / 'status.json')
    latest_bytes = (args.run / 'latest_validation.json').read_bytes()
    latest = json.loads(latest_bytes)
    events = {}
    with (args.run / 'train.log').open() as stream:
        for line in stream:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (isinstance(event, dict) and 'validation' in event
                    and event.get('step', latest['step'] + 1) <= latest['step']):
                events[event['step']] = event
    initial = read(args.run / 'initial_validation.json')
    if initial['step'] <= latest['step']:
        events[initial['step']] = initial
    events[latest['step']] = latest
    values = latest['validation']
    sources = values['by_dataset']
    modes = [key for key, item in values.items()
             if key != 'by_dataset' and isinstance(item, dict)]
    horizons = metadata['config']['eval_horizons_seconds']
    macro = {mode: forecast_metrics(values[mode]) for mode in modes}
    for mode in modes:
        for key, value in macro[mode].items():
            if not isinstance(value, (float, int)):
                continue
            contributing = [item[mode][key] for item in sources.values() if key in item[mode]]
            assert math.isclose(sum(contributing) / len(contributing), value,
                                rel_tol=1e-6, abs_tol=1e-8), (mode, key)
    curve = []
    for step, event in sorted(events.items()):
        for source in sources:
            for mode in ('noisy', 'clean'):
                assert (event['validation']['by_dataset'][source][mode]['counts']
                        == sources[source][mode]['counts']), (step, source, mode)
        curve.append({'step': step, 'utc': event['utc'],
                      'macro': {mode: forecast_metrics(event['validation'][mode])
                                for mode in ('noisy', 'clean')}})
    comparisons = {}
    for mode in modes:
        comparisons[mode] = {}
        for seconds in horizons:
            label = f'{seconds:g}s'
            metrics = {}
            for metric in ('wrist_mm', 'root_deg', 'finger_deg'):
                prediction = values[mode][f'future_{label}/map/{metric}']
                hold = values[mode][f'future_{label}/restored_hold/{metric}']
                cv = values[mode][f'future_{label}/constant_velocity/{metric}']
                metrics[metric] = {'prediction': prediction, 'restored_hold': hold,
                                   'constant_velocity': cv,
                                   'reduction_vs_hold_percent': 100 * (hold - prediction) / hold}
            metrics['counts_by_dataset'] = {
                source: item[mode]['counts'][label] for source, item in sources.items()}
            comparisons[mode][label] = metrics
    report = {
        'utc': datetime.now(timezone.utc).isoformat(), 'run': str(args.run),
        'code_version': metadata['code_version'], 'parameters': metadata['parameters'],
        'metadata': {key: metadata[key] for key in (
            'manifest_sha256', 'config_file_sha256', 'resolved_config_sha256',
            'source_sha256', 'GT', 'position_units', 'rotation6d', 'beta_conditioning',
            'sampling', 'val_samples_by_dataset', 'train_windows_by_dataset',
            'val_windows_by_dataset', 'world_size', 'global_batch_size', 'per_gpu_batch_size')},
        'config': metadata['config'], 'running_snapshot': status,
        'latest_validation': {'step': latest['step'], 'utc': latest['utc'], 'macro': macro,
            'by_dataset': {source: {mode: forecast_metrics(item[mode]) for mode in modes}
                           for source, item in sources.items()}},
        'horizon_comparisons': comparisons, 'fixed_validation_curve': curve,
        'provenance': {'latest_validation_sha256': hashlib.sha256(latest_bytes).hexdigest(),
            'analysis_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'forecast_source_macro_checked': True, 'fixed_horizon_counts_checked': True,
            'train_log': str(args.run / 'train.log')},
        'definitions': {
            'horizons': 'Endpoint nearest the requested physical timestamp; eligible within 0.6 * current dt. Available-source equal mean; populations differ by horizon.',
            'map': 'Whole trajectory branch with highest mixture weight; no future GT selection.',
            'baselines': 'Both start at restored current state. Constant velocity uses last two causally filled observed history frames.',
            'forecast_ade': 'Per-source mean over all valid future points, then equal-source mean; variable lengths.',
            'minade': 'GT oracle best wrist-ADE branch over each available sequence, then mean over windows and sources. Different weighting from forecast_ade; their difference is not pure branch-selection loss.',
            'motion_groups': 'Moving iff maximum valid-future wrist displacement >=20mm or finger-rotation change >=10deg from clean current reference. Group metrics average valid future points. Low motion may include root-only motion.',
            'coverage': 'Mean per-coordinate central 90% mixture interval coverage over valid future points; not xyz joint, trajectory joint, or rotation coverage.',
            'entropy': 'Natural-log entropy of four trajectory mixture weights, not MoE routing; does not establish useful mode diversity.',
            'scope': 'Saved fixed synthetic validation, source-fitted MANO references, recording-disjoint validation. One observed history to a full open-loop future; no real HaWoR, no continuous visual feedback or independent sensor accuracy.'},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'snapshot': {
        key: status[key] for key in ('utc', 'step', 'phase')},
        'validation_step': latest['step'], 'noisy': comparisons['noisy'],
        'clean': comparisons['clean'],
        'noisy_other_metrics': {key: value for key, value in macro['noisy'].items()
            if not key.startswith('future_') or key.startswith('future_motion_')},
        'curve': [{'step': row['step'], 'wrist_1s': row['macro']['noisy']['future_1s/map/wrist_mm'],
                  'wrist_2s': row['macro']['noisy']['future_2s/map/wrist_mm'],
                  'forecast_nll': row['macro']['noisy']['forecast_nll_per_dof']}
                  for row in curve]}, indent=2))


if __name__ == '__main__':
    main()
