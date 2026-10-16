#!/usr/bin/env python3
"""Remote-only, paired hand/source audit of pretrained and jointly tuned TWM.

This loads only numeric TWM tensors from joint checkpoints. No image frontend,
adapter, optimizer, training update, or future reference is a model input.
All windows, corruption realizations and mini-batch boundaries are identical
across checkpoints. This is a synthetic held-out motion diagnostic, not an
independent evaluation of the complete visual reconstruction system.

Run inside the ds compute Pod with PYTHONPATH=src, for example:
  python scripts/diagnose_twm_handedness.py \
    --run-metadata /remote/D37/run_metadata.json \
    --pretrained /remote/D37/best.pt \
    --joint best300=/remote/joint/best.pt \
    --joint last1000=/remote/joint/latest.pt \
    --output /remote/evaluations/twm_hands --device cuda:0

The complete indices and per-window CSV stay remote. summary.json contains
only small aggregates, paired input hashes and provenance for source control.
"""
from __future__ import annotations

import argparse
import bisect
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
from typing import Any


MODES = ('clean', 'noisy', 'reanchor', 'position_burst4', 'missing')
METRICS = ('wrist_mm', 'root_deg', 'finger_deg')
JOINT_PREFIX = 'pipeline.feedback.twm.'
NUMERIC_SOURCES = (
    'scripts/train_probabilistic_twm.py', 'scripts/train_twm.py',
    'src/handloop/geometry.py', 'src/handloop/models/probabilistic_twm.py',
    'src/handloop/models/structured_mamba.py', 'src/handloop/models/twm.py',
    'src/handloop/data/trajectory.py', 'src/handloop/data/forecast_trajectory.py',
    'src/handloop/data/corruption.py', 'src/handloop/data/temporal_corruption.py',
)


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2,
                                    allow_nan=False) + '\n')
    temporary.replace(path)


def label_path(value: str) -> tuple[str, Path]:
    if '=' not in value:
        raise argparse.ArgumentTypeError('Use LABEL=CHECKPOINT')
    label, raw = value.split('=', 1)
    if not label or not all(ch.isalnum() or ch in '_-' for ch in label):
        raise argparse.ArgumentTypeError('Labels use letters, digits, _ and -')
    return label, Path(raw).expanduser().resolve()


def tensor_hash(value: Any) -> str:
    """Include dtype, shape, names and raw bytes, including bf16 tensors."""
    import torch
    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(canonical({'shape': list(tensor.shape),
                                     'dtype': str(tensor.dtype)}).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item):
                digest.update(canonical(key).encode()); visit(item[key])
        elif isinstance(item, (tuple, list)):
            for slot, child in enumerate(item):
                digest.update(str(slot).encode()); visit(child)
        else:
            digest.update(canonical(item).encode())
    visit(value)
    return digest.hexdigest()


def select_windows(dataset: Any, requested: int, seed: int) -> tuple[dict, dict]:
    """Cap each source equally by hand; allocate windows evenly to recordings.

    Recording allocations differ by at most one until a recording exhausts
    its available windows. Quantile midpoints spread centers within each
    recording. No pose/reference/error values enter the selection.
    """
    grouped: dict = defaultdict(lambda: defaultdict(list))
    for seq_id, seq in enumerate(dataset.sequences):
        if seq['handedness'] not in (0, 1):
            raise ValueError('A required source contains unknown handedness')
        grouped[(seq['dataset'], seq['handedness'])][seq['recording']].append(seq_id)
    selections, coverage = {}, {}
    for source in dataset.dataset_names:
        counts = {hand: sum(len(dataset.sequences[sid]['centers'])
                           for seqs in grouped[(source, hand)].values() for sid in seqs)
                  for hand in (0, 1)}
        count = min(requested, counts[0], counts[1])
        if count < 1:
            raise ValueError(f'{source} lacks both hands in validation: {counts}')
        for hand in (0, 1):
            key = f'{source}/{"left" if hand == 0 else "right"}'
            records = grouped[(source, hand)]
            ordered = sorted(records)
            salt = int(hashlib.sha256(f'{source}/{hand}'.encode()).hexdigest()[:12], 16)
            random.Random(seed + salt).shuffle(ordered)
            capacity = {rec: sum(len(dataset.sequences[sid]['centers']) for sid in records[rec])
                        for rec in ordered}
            allocations = dict.fromkeys(ordered, 0)
            remaining = count
            while remaining:
                for rec in ordered:
                    if allocations[rec] < capacity[rec]:
                        allocations[rec] += 1; remaining -= 1
                        if not remaining:
                            break
            chosen = []
            for rec in sorted(ordered):
                number = allocations[rec]
                if not number:
                    continue
                seqs = sorted(records[rec], key=lambda sid: str(dataset.sequences[sid]['_path']))
                ends, total = [], 0
                for sid in seqs:
                    total += len(dataset.sequences[sid]['centers']); ends.append(total)
                # Integer midpoint quantiles are distinct for number <= total.
                for j in range(number):
                    offset = ((2 * j + 1) * total) // (2 * number)
                    slot = bisect.bisect_right(ends, offset)
                    sid = seqs[slot]
                    local = offset - (ends[slot - 1] if slot else 0)
                    seq = dataset.sequences[sid]
                    global_index = (dataset.ends[sid - 1] if sid else 0) + local
                    chosen.append({'global_index': global_index, 'sequence_index': sid,
                                   'center': int(seq['centers'][local]), 'recording': rec,
                                   'path': str(seq['_path']), 'source_manifest': seq['source_manifest'],
                                   'source': source, 'handedness': hand, 'cell': key})
            chosen.sort(key=lambda row: (row['recording'], row['path'], row['center']))
            if len(chosen) != count or len({r['global_index'] for r in chosen}) != count:
                raise AssertionError('Selection is incomplete or repeats windows')
            selections[key] = chosen
            coverage[key] = {'available_windows': counts[hand], 'selected_windows': count,
                             'available_recordings': len(records),
                             'selected_recordings': sum(v > 0 for v in allocations.values()),
                             'windows_by_recording': {r: v for r, v in allocations.items() if v},
                             'hand_matching_cap': count,
                             'population_includes_all_windows': count == counts[hand]}
    return selections, coverage


def _summary(rows: list[dict]) -> dict:
    """Per-window means; future averages use eligible-window counts only."""
    numerical = sorted({key for row in rows for key, value in row.items()
                        if isinstance(value, (int, float)) and key not in
                        ('global_index', 'center', 'handedness', 'sequence_index')})
    result = {'samples': len(rows), 'metrics': {}, 'counts': {}}
    for key in numerical:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        if values:
            if not all(math.isfinite(v) for v in values):
                raise FloatingPointError(f'Nonfinite diagnostic: {key}')
            result['metrics'][key] = sum(values) / len(values)
            result['counts'][key] = len(values)
    for key in [f'{group}/{metric}' for group in ('input', 'output') for metric in METRICS]:
        values = sorted(float(row[key]) for row in rows)
        def quantile(q):
            slot = (len(values) - 1) * q
            low = int(slot); high = min(low + 1, len(values) - 1)
            return values[low] * (high - slot) + values[high] * (slot - low) if low != high else values[low]
        result['metrics'][key + '_p50'] = quantile(.5)
        result['metrics'][key + '_p95'] = quantile(.95)
    return result


def macro(summaries: list[dict]) -> dict:
    keys = sorted({key for value in summaries for key in value['metrics']})
    return {'aggregation': 'equal-cell means, skipping cells without eligible future targets',
            'cells': len(summaries), 'metrics': {
                key: sum(value['metrics'][key] for value in summaries if key in value['metrics']) /
                sum(key in value['metrics'] for value in summaries) for key in keys},
            'contributing_cells': {key: sum(key in value['metrics'] for value in summaries) for key in keys}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-metadata', type=Path, required=True)
    parser.add_argument('--pretrained', type=Path, required=True)
    parser.add_argument('--joint', type=label_path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--windows-per-cell', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=20261016)
    args = parser.parse_args()
    if not Path('/sciclone/home/jchen57').is_dir():
        parser.error('Run model evaluation only in the authorized remote ds Pod')
    if args.windows_per_cell < 1 or args.batch_size < 1:
        parser.error('Positive windows and batch sizes required')
    if args.output.exists():
        parser.error('Use a fresh output directory')
    labels = ['pretrained'] + [label for label, _ in args.joint]
    if len(set(labels)) != len(labels):
        parser.error('Checkpoint labels must be unique and cannot be pretrained')

    import csv
    import torch
    from torch.utils.data import DataLoader, Subset
    from handloop.data.forecast_trajectory import ForecastTrajectoryWindows
    from handloop.data.temporal_corruption import causal_fill_history
    from handloop.models.probabilistic_twm import ProbabilisticTWM, ProbabilisticTWMConfig
    from train_probabilistic_twm import prepare_batch, state_errors, mixture_translation_uncertainty

    torch.set_num_threads(4)
    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        parser.error('A remote CUDA device is required for the original bf16 evaluation')
    metadata = json.loads(args.run_metadata.read_text())
    config = metadata['config']
    if config.get('condition_on_beta') is not False:
        raise ValueError('This audit requires D37 beta conditioning to remain false')
    if config['model'].get('position_feature_scale') != .05:
        raise ValueError('Expected the actual D37 position_feature_scale=0.05m')
    profile = config.get('corruption_profile')
    if not profile or profile.get('version') != 3:
        raise ValueError('Run metadata must embed the actual D37 v3 corruption profile')
    manifest = Path(metadata['manifest']).resolve()
    if file_hash(manifest) != metadata['manifest_sha256']:
        raise ValueError('The D37 training collection manifest changed')
    root = Path(__file__).resolve().parents[1]
    source_hashes = {name: file_hash(root / name) for name in NUMERIC_SOURCES}
    for name, digest in source_hashes.items():
        if digest != metadata['source_sha256'][name]:
            raise ValueError(f'Numeric/evaluation source differs from D37: {name}')
    args.output.mkdir(parents=True)
    save_json(args.output / 'status.json', {'phase': 'indexing', 'utc': utc()})
    dataset = ForecastTrajectoryWindows(manifest, 'val', history=config['history_frames'],
        forecast_frames=config['model']['forecast_horizon'],
        min_future_frames=config['min_future_frames'], window_stride=config['window_stride'],
        cache_sequences=128, cache_bytes=512 * 1024**2)
    if set(dataset.dataset_names) != set(config['required_datasets']):
        raise ValueError('Required source set changed')
    selections, coverage = select_windows(dataset, args.windows_per_cell, args.seed)
    index_path = args.output / 'selection.json'
    save_json(index_path, {'seed': args.seed, 'split': 'val', 'manifest': str(manifest),
        'manifest_sha256': metadata['manifest_sha256'], 'counts': coverage,
        'selection_policy': 'equal source-hand target; recording-balanced allocations, quantile-spaced centers; no labels/errors consulted',
        'selections': selections})

    pretrained_path = args.pretrained.expanduser().resolve()
    pretrained_file_hash = file_hash(pretrained_path)
    models = [('pretrained', pretrained_path)] + args.joint
    result = {'schema': 'handloop.twm_paired_handedness.v1', 'started_utc': utc(),
        'run_metadata': str(args.run_metadata.resolve()),
        'run_metadata_sha256': file_hash(args.run_metadata),
        'training_code_version': metadata['code_version'],
        'manifest': str(manifest), 'manifest_sha256': metadata['manifest_sha256'],
        'selection_path_remote': str(index_path.resolve()),
        'selection_sha256': file_hash(index_path), 'selection_counts': coverage,
        'source_sha256': source_hashes, 'script_sha256': file_hash(Path(__file__)),
        'device': str(device), 'torch': torch.__version__, 'config': config,
        'batch_size': args.batch_size, 'selection_seed': args.seed,
        'corruption_seed': config['seed'] + 123, 'corruption_modes': MODES,
        'protocol': {'split': 'existing recording-disjoint D37 val, not new test',
            'input': 'same original prepare_batch v3, reset training eval seed per cell/mode, identical batches across checkpoints',
            'precision': 'torch.autocast cuda bfloat16, matching D37 evaluate',
            'beta': 'source values read by original sampler but all beta_valid false; model masks them to zero',
            'future': 'future states and masks are targets only; forecast_dt query locations are model inputs',
            'current_baseline': 'causally filled candidate, matching D37 evaluator',
            'future_baseline': 'hold each model restored current state, matching D37 evaluator',
            'future_selection': 'single highest-weight trajectory, no oracle branch selection',
            'future_horizons': '1s/2s nearest valid source timestamp within 0.6*last observed dt',
            'degradation_thresholds': {'wrist_mm': 1., 'root_deg': .1, 'finger_deg': .1},
            'units': 'anatomical wrist mm; SO(3) geodesic degrees; finger averages 15 joints',
            'reference': 'prepared source fitted MANO parameters; no mesh/MoCap independence claim',
            'read_only': 'no optimizer, eval+no_grad, all tensor and checkpoint hashes before/after',
            'scope': 'numeric TWM alone under synthetic corruption; not whole visual system or closed loop'},
        'checkpoints': {}, 'paired_input_hashes': {}, 'by_checkpoint': {},
        'limitations': ['Windows can overlap; within-cell windows are not independent recordings.',
            'Source and hand motion populations can differ despite recording-balanced sampling.',
            'Only the requested bounded val population is evaluated; no causal attribution from this alone.',
            'Future hold inherits each model current restoration, so gains combine restoration and forecasting.']}
    try:
        result['evaluation_code_version'] = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        result['evaluation_code_version'] = 'unavailable'
    paired_rows = {}
    for label, checkpoint_path in models:
        save_json(args.output / 'status.json', {'phase': 'loading', 'checkpoint': label, 'utc': utc()})
        before_file = pretrained_file_hash if label == 'pretrained' else file_hash(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False, mmap=True)
        if label == 'pretrained':
            if canonical(checkpoint['config']) != canonical(config):
                raise ValueError('D37 checkpoint config differs from its run metadata')
            if checkpoint['run']['manifest_sha256'] != metadata['manifest_sha256']:
                raise ValueError('D37 checkpoint uses a different manifest')
            state = checkpoint['model']
        else:
            if canonical(checkpoint['replay_config']) != canonical(config):
                raise ValueError(f'{label}: joint replay changed the original resolved D37 config')
            source = checkpoint['twm_source']
            if source['sha256'] != pretrained_file_hash:
                raise ValueError(f'{label}: different D37 source checkpoint')
            if canonical(source['model_config']) != canonical(config['model']):
                raise ValueError(f'{label}: joint TWM architecture or position scaling changed')
            if checkpoint['identity']['motion_manifest_sha256'] != metadata['manifest_sha256']:
                raise ValueError(f'{label}: joint motion manifest changed')
            state = {name.removeprefix(JOINT_PREFIX): value
                     for name, value in checkpoint['model'].items() if name.startswith(JOINT_PREFIX)}
            if not state:
                raise ValueError('Actual JointTaskModel pipeline.feedback.twm. prefix not found')
        model = ProbabilisticTWM(ProbabilisticTWMConfig(**config['model']))
        model.load_state_dict(state, strict=True)
        checkpoint_step = int(checkpoint['step'])
        checkpoint_code = checkpoint['code_version']
        del checkpoint, state
        model.to(device).eval().requires_grad_(False)
        before_model = tensor_hash(model.state_dict())
        all_cells, rows_for_pair = {}, {}
        csv_path = args.output / f'{label}_per_window.csv'
        csv_stream = csv_path.open('w', newline='')
        writer = None
        try:
            with torch.no_grad():
                for cell, selected in selections.items():
                    all_cells[cell] = {}
                    loader = DataLoader(Subset(dataset, [r['global_index'] for r in selected]),
                        batch_size=args.batch_size, shuffle=False, num_workers=0)
                    for mode in MODES:
                        save_json(args.output / 'status.json', {'phase': 'evaluating',
                            'checkpoint': label, 'cell': cell, 'mode': mode, 'utc': utc()})
                        generator = torch.Generator(device=device).manual_seed(config['seed'] + 123)
                        cell_rows, batch_hashes, target_hashes = [], [], []
                        cursor = 0
                        for batch in loader:
                            batch = {key: value.to(device) for key, value in batch.items()}
                            history, raw, candidate, kwargs, target, future = prepare_batch(
                                batch, generator, mode, condition_on_beta=False,
                                corruption_profile=profile)
                            if bool(kwargs['beta_valid'].any()):
                                raise AssertionError('Unexpected beta conditioning')
                            batch_hashes.append(tensor_hash({'history': history, 'raw': raw,
                                'candidate': candidate, 'kwargs': kwargs}))
                            target_hashes.append(tensor_hash({'current': target, 'future': future,
                                'future_valid': batch['future_valid']}))
                            with torch.autocast('cuda', dtype=torch.bfloat16):
                                output = model(history, raw, candidate, **kwargs)
                            current = state_errors(output['state'], target)
                            inputs = state_errors(candidate, target)
                            raw_errors = state_errors(raw, target)
                            filled = causal_fill_history(history, kwargs['history_availability'])
                            history_errors = state_errors(filled[:, -1], target)
                            forecast_errors = state_errors(output['forecast'], future)
                            hold_errors = state_errors(output['state'][:, None], future)
                            _, future_std = mixture_translation_uncertainty(output)
                            standardized = (future[:, None, :, :3] - output['forecast_modes'][..., :3]) / output['forecast_translation_std']
                            component_cdf = .5 * (1 + torch.erf(standardized / math.sqrt(2)))
                            mixture_cdf = (component_cdf * output['mixture_weights'][:, :, None, None]).sum(1)
                            future_coverage = ((mixture_cdf >= .05) & (mixture_cdf <= .95)).float().mean(-1).cpu()
                            current_std = output['state_translation_std'].float()
                            mahal = ((target[:, :3] - output['state'][:, :3]) / current_std).square().sum(-1)
                            n = len(target)
                            slots = {}
                            for seconds in (1., 2.):
                                difference = (batch['future_dt'] - seconds).abs().masked_fill(
                                    ~batch['future_valid'], float('inf'))
                                distance, slot = difference.min(-1)
                                eligible = distance <= batch['dt'][:, -1] * .6
                                slots[seconds] = (slot.cpu(), eligible.cpu())
                            # Bulk copies avoid thousands of scalar CUDA synchronizations.
                            current = {key: value.float().cpu() for key, value in current.items()}
                            inputs = {key: value.float().cpu() for key, value in inputs.items()}
                            raw_errors = {key: value.float().cpu() for key, value in raw_errors.items()}
                            history_errors = {key: value.float().cpu() for key, value in history_errors.items()}
                            forecast_errors = {key: value.float().cpu() for key, value in forecast_errors.items()}
                            hold_errors = {key: value.float().cpu() for key, value in hold_errors.items()}
                            current_std, mahal = current_std.cpu(), mahal.cpu()
                            future_std = future_std.cpu()
                            concentrations = output['state_rotation_concentration'].float().cpu()
                            for i in range(n):
                                row = {**selected[cursor + i], 'mode': mode}
                                # IDs remain separate from numeric metric keys.
                                for group, values in [('input', inputs), ('output', current),
                                                      ('raw', raw_errors), ('history_hold', history_errors)]:
                                    row.update({f'{group}/{key}': float(values[key][i]) for key in METRICS})
                                for key, threshold in [('wrist_mm', 1.), ('root_deg', .1), ('finger_deg', .1)]:
                                    row[f'degraded/{key}'] = float(current[key][i] > inputs[key][i] + threshold)
                                    row[f'gain/{key}'] = float(inputs[key][i] - current[key][i])
                                row['current/std_mm'] = float(current_std[i].mean() * 1000)
                                row['current/90pct_xyz_ellipsoid_coverage'] = float(mahal[i] <= 6.251389)
                                concentration = concentrations[i]
                                row['current/root_concentration'] = float(concentration[0])
                                row['current/finger_concentration'] = float(concentration[1:].mean())
                                # Preserve fields even where every row lacks a 2s target (e.g. Dex).
                                for seconds, (slots_at_time, eligible) in slots.items():
                                    prefix = f'future_{seconds:g}s'
                                    row[prefix + '/eligible'] = float(eligible[i])
                                    slot_i = int(slots_at_time[i])
                                    for group, errors in [('map', forecast_errors), ('restored_hold', hold_errors)]:
                                        for key in METRICS:
                                            row[f'{prefix}/{group}/{key}'] = float(errors[key][i, slot_i]) if eligible[i] else None
                                    row[prefix + '/total_xyz_std_mm'] = float(future_std[i, slot_i].mean() * 1000) if eligible[i] else None
                                    row[prefix + '/90pct_xyz_marginal_coverage'] = float(future_coverage[i, slot_i]) if eligible[i] else None
                                cell_rows.append(row)
                            cursor += n
                        if cursor != len(selected):
                            raise AssertionError('Not all paired windows evaluated')
                        hashes = {'input_batch_sha256': batch_hashes, 'target_batch_sha256': target_hashes,
                                  'combined_input_sha256': hashlib.sha256(canonical(batch_hashes).encode()).hexdigest(),
                                  'combined_target_sha256': hashlib.sha256(canonical(target_hashes).encode()).hexdigest()}
                        pair_key = cell + '/' + mode
                        if label == 'pretrained':
                            result['paired_input_hashes'][pair_key] = hashes
                        elif result['paired_input_hashes'][pair_key] != hashes:
                            raise AssertionError(f'Paired inputs/targets changed: {label}/{pair_key}')
                        rows_for_pair[pair_key] = cell_rows
                        all_cells[cell][mode] = _summary(cell_rows)
                        for row in cell_rows:
                            if writer is None:
                                writer = csv.DictWriter(csv_stream, fieldnames=list(row))
                                writer.writeheader()
                            writer.writerow(row)
        finally:
            csv_stream.close()
        after_model = tensor_hash(model.state_dict())
        del model
        torch.cuda.empty_cache()
        after_file = file_hash(checkpoint_path)
        if before_model != after_model or before_file != after_file:
            raise AssertionError('Read-only audit failed: model or checkpoint changed')
        result['checkpoints'][label] = {'path': str(checkpoint_path), 'step': checkpoint_step,
            'code_version': checkpoint_code, 'file_sha256_before': before_file,
            'file_sha256_after': after_file, 'numeric_twm_sha256_before': before_model,
            'numeric_twm_sha256_after': after_model, 'read_only_verified': True,
            'strict_state_dict': True, 'joint_prefix': JOINT_PREFIX if label != 'pretrained' else None,
            'per_window_csv_remote': str(csv_path.resolve())}
        by_hand = {hand: {mode: macro([all_cells[cell][mode] for cell in all_cells
                                     if cell.endswith('/' + hand)]) for mode in MODES}
                   for hand in ('left', 'right')}
        by_recording = {}
        for pair_key, rows in rows_for_pair.items():
            by_recording[pair_key] = {}
            record_metrics = [f'{group}/{metric}' for group in ('input', 'output') for metric in METRICS]
            record_metrics += ['current/std_mm', 'current/90pct_xyz_ellipsoid_coverage']
            for recording in sorted({row['recording'] for row in rows}):
                chosen = [row for row in rows if row['recording'] == recording]
                by_recording[pair_key][recording] = {'samples': len(chosen),
                    'metrics': {key: sum(row[key] for row in chosen) / len(chosen) for key in record_metrics}}
        result['by_checkpoint'][label] = {'by_source_hand': all_cells, 'equal_source_by_hand': by_hand,
                                         'by_recording': by_recording}
        if label == 'pretrained':
            paired_rows = rows_for_pair
        else:
            paired = {}
            for pair_key, rows in rows_for_pair.items():
                reference = paired_rows[pair_key]
                deltas = []
                for a, b in zip(rows, reference, strict=True):
                    if a['global_index'] != b['global_index']:
                        raise AssertionError('Paired row identity changed')
                    delta = {'global_index': a['global_index']}
                    for key, value in a.items():
                        if key.startswith(('output/', 'current/', 'future_')) and isinstance(value, (int, float)):
                            if b.get(key) is not None:
                                delta[key] = float(value) - float(b[key])
                    deltas.append(delta)
                paired[pair_key] = {'samples': len(deltas),
                    'joint_minus_pretrained': {key: sum(row[key] for row in deltas if key in row) /
                        sum(key in row for row in deltas) for key in sorted({key for row in deltas for key in row
                                                                                 if key != 'global_index'})}}
            result['by_checkpoint'][label]['paired_vs_pretrained'] = paired
        save_json(args.output / 'summary.json', result)
        print(json.dumps({'utc': utc(), 'checkpoint': label, 'step': checkpoint_step,
                          'complete': True, 'paired_input_verified': True}), flush=True)
    result['completed_utc'] = utc()
    result['all_paired_inputs_verified'] = True
    save_json(args.output / 'summary.json', result)
    save_json(args.output / 'status.json', {'phase': 'complete', 'utc': utc(),
                                          'summary': str((args.output / 'summary.json').resolve())})


if __name__ == '__main__':
    main()
