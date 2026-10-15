#!/usr/bin/env python3
"""Read saved TWM metrics remotely; no inference, optimization, or run mutation."""
import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path


CURRENT_KEYS = (
    'restoration_loss', 'current_nll_per_dof', 'current_xyz_std_mm',
    'candidate/wrist_mm', 'restored/wrist_mm', 'history_hold/wrist_mm',
    'candidate/root_deg', 'restored/root_deg', 'history_hold/root_deg',
    'candidate/finger_deg', 'restored/finger_deg', 'history_hold/finger_deg',
    'wrist_degradation_over_1mm', 'calibration/current_xyz_90pct_ellipsoid',
)


def compact_metrics(row):
    keys = set(CURRENT_KEYS)
    for prefix in ('candidate', 'restored'):
        for metric in ('wrist_mm', 'root_deg', 'finger_deg'):
            keys.update((f'{prefix}/{metric}_p50', f'{prefix}/{metric}_p95'))
    for horizon in ('0.1s', '0.5s', '1s', '2s'):
        for baseline in ('map', 'restored_hold', 'constant_velocity'):
            for metric in ('wrist_mm', 'root_deg', 'finger_deg'):
                keys.add(f'future_{horizon}/{baseline}/{metric}')
    keys.update(('samples', 'counts', 'forecast_nll_per_dof', 'mode_entropy',
                 'calibration/future_xyz_90pct_marginal'))
    return {key: row[key] for key in sorted(keys) if key in row}


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
    maximum_step = latest['step']
    # Validation events are stdout records; metrics.jsonl contains training only.
    validation = {}
    skipped = 0
    with (args.run / 'train.log').open() as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            if isinstance(row, dict) and 'validation' in row and row.get('step', maximum_step+1) <= maximum_step:
                validation[row['step']] = row
    initial = read(args.run / 'initial_validation.json')
    if initial['step'] <= maximum_step:
        validation[initial['step']] = initial
    validation[maximum_step] = latest
    weights = metadata['config']['best_selection_weights']
    assert math.isclose(sum(weights.values()), 1.0)
    curve = []
    for step, event in sorted(validation.items()):
        values = event['validation']
        score = sum(weight * values[mode]['restoration_loss'] for mode, weight in weights.items())
        assert math.isfinite(score)
        row = {'step': step, 'utc': event['utc'], 'weighted_current_score': score,
               'best_score_so_far': event['best_validation_loss']}
        row.update({mode: {key: item[key] for key in CURRENT_KEYS if key in item}
                    for mode,item in values.items() if mode!='by_dataset' and isinstance(item,dict)})
        row['forecast_1s'] = {key: value for key, value in values['noisy'].items() if key.startswith('future_1s/')}
        curve.append(row)
    values = latest['validation']
    by_source = values['by_dataset']
    assert set(by_source) == set(metadata['dataset_names'])
    metrics = {mode: compact_metrics(row) for mode, row in values.items() if isinstance(row, dict) and mode != 'by_dataset'}
    # Check that the reported headline is the stated equal-source mean.
    for mode in metrics:
        for key in CURRENT_KEYS:
            if key in values[mode]:
                mean = sum(by_source[source][mode][key] for source in by_source) / len(by_source)
                assert math.isclose(mean, values[mode][key], rel_tol=1e-6, abs_tol=1e-8), (mode, key)
    report = {
        'utc': datetime.now(timezone.utc).isoformat(), 'run': str(args.run),
        'code_version': metadata['code_version'], 'parameters': metadata['parameters'],
        'metadata': {key: metadata[key] for key in (
            'initialization', 'world_size', 'global_batch_size', 'per_gpu_batch_size',
            'manifest_sha256', 'config_file_sha256', 'resolved_config_sha256', 'source_sha256',
            'GT', 'rotation6d', 'position_units', 'beta_conditioning', 'sampling',
            'val_samples_by_dataset', 'train_windows_by_dataset', 'val_windows_by_dataset')},
        'config': metadata['config'], 'running_snapshot': status,
        'latest_validation': {'step': maximum_step, 'utc': latest['utc'],
            'actual_weighted_current_score': curve[-1]['weighted_current_score'],
            'best_score_so_far': latest['best_validation_loss'],
            'macro': metrics,
            'by_dataset': {source: {mode: compact_metrics(row) for mode, row in modes.items()}
                           for source, modes in by_source.items()}},
        'fixed_validation_curve': curve,
        'provenance': {'train_log': str(args.run / 'train.log'),
            'latest_validation_bytes_sha256': hashlib.sha256(latest_bytes).hexdigest(),
            'ignored_non_json_log_lines': skipped,
            'source_aggregation_checked': True},
        'scope': 'Recorded fixed synthetic validation only; source-fitted MANO references, equal-source macro means. No new model evaluation, no test labels, no real HaWoR or independent sensor accuracy claim.',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'snapshot_step': status.get('step'),
        'validation_step': maximum_step, 'curve': [
            {'step': row['step'], 'score': row['weighted_current_score'],
             'wrist_mm': row['noisy']['restored/wrist_mm'],
             'root_deg': row['noisy']['restored/root_deg'],
             'finger_deg': row['noisy']['restored/finger_deg']} for row in curve],
        'latest_macro': {mode: {key: row[key] for key in CURRENT_KEYS if key in row}
                         for mode, row in metrics.items() if mode in ('noisy', 'clean', 'reanchor')},
        'scope': report['scope']}, indent=2))


if __name__ == '__main__':
    main()
