#!/usr/bin/env python3
"""Remote-only reproducible paired summary of visual state and MANO geometry.

Example: --baseline BASE --joint best=BEST --joint last=LAST
         --geometry geometry.json --output comparison.json
Positive baseline-minus-final differences mean lower error after correction.
Bootstrap units are complete recordings, never independently sampled frames.
This script does not select a checkpoint, run a model, or modify evaluation data.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess


STATE_METRICS = ('wrist_mm', 'root_deg', 'finger_deg')
GEOMETRY_METRICS = ('mpjpe21_mm', 'wrist_aligned_mpjpe21_mm',
                    'mpvpe778_mm', 'wrist_aligned_mpvpe778_mm')
METRICS = STATE_METRICS + GEOMETRY_METRICS


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_csv(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def identity(row):
    return str(row['recording']), int(row['clip_index']), int(row['reference_index'])


def numeric(row, name):
    value = row.get(name)
    if value is None or value == '':
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError('Nonfinite input field: ' + name)
    return result


def indexed(rows):
    result = {}
    for row in rows:
        key = identity(row)
        if key in result:
            raise ValueError('Duplicate evaluation frame: ' + repr(key))
        result[key] = row
    if not result:
        raise ValueError('Empty evaluation CSV')
    return result


def parse_joint(value):
    if '=' not in value:
        raise argparse.ArgumentTypeError('Use LABEL=EVALUATION_DIRECTORY')
    label, path = value.split('=', 1)
    if not label or label == 'baseline' or not all(c.isalnum() or c in '_-' for c in label):
        raise argparse.ArgumentTypeError('Use a unique label of letters/digits/_/- other than baseline')
    return label, Path(path).expanduser().resolve()


def weighted_metric(rows, branch, metric):
    values, count = 0., 0
    for row in rows:
        value = row['errors'][branch].get(metric)
        if value is None:
            continue
        weight = int(row['errors'][branch]['finger_count']) if metric == 'finger_deg' else 1
        if weight < 1:
            raise ValueError('Eligible finger mean must have positive rotation count')
        values += value * weight
        count += weight
    return {'mean': values / count if count else None, 'count': count,
            'count_unit': 'finger_rotations' if metric == 'finger_deg' else 'complete_reference_states'}


def summarize(rows, branches):
    return {'source_frames': len(rows),
            'complete_reference_states': sum(row['complete_reference_valid'] for row in rows),
            'branches': {branch: {metric: weighted_metric(rows, branch, metric) for metric in METRICS}
                         for branch in branches}}


def grouped(rows, branches, field):
    groups = {}
    for row in rows:
        groups.setdefault(str(row[field]), []).append(row)
    by_group = {key: summarize(values, branches) for key, values in sorted(groups.items())}
    macro = {}
    for branch in branches:
        macro[branch] = {}
        for metric in METRICS:
            eligible = [value['branches'][branch][metric]['mean'] for value in by_group.values()
                        if value['branches'][branch][metric]['count']]
            macro[branch][metric] = {'mean': sum(eligible) / len(eligible) if eligible else None,
                                     'eligible_groups': len(eligible)}
    return {'by_group': by_group, 'macro_equal_group': macro}


def compare(rows, base, final):
    result = {}
    for metric in METRICS:
        eligible = [row for row in rows if row['errors'][base].get(metric) is not None
                    and row['errors'][final].get(metric) is not None
                    and (metric != 'finger_deg' or
                         row['errors'][base]['finger_count'] == row['errors'][final]['finger_count'] == 15)]
        n = len(eligible)
        base_mean = sum(row['errors'][base][metric] for row in eligible) / n if n else None
        final_mean = sum(row['errors'][final][metric] for row in eligible) / n if n else None
        gain = base_mean - final_mean if n else None
        result[metric] = {'paired_states': n, 'baseline_mean': base_mean, 'final_mean': final_mean,
                          'absolute_error_reduction': gain,
                          'relative_error_reduction_pct': 100 * gain / base_mean if n and base_mean > 0 else None}
        if metric.endswith('_mm'):
            worse = sum(row['errors'][final][metric] - row['errors'][base][metric] > 1. for row in eligible)
            result[metric].update(degradation_over_1mm_count=worse,
                                  degradation_over_1mm_fraction=worse / n if n else None)
    return {'baseline': base, 'final': final, 'metrics': result}


def bootstrap_recordings(rows, base, final, metric, *, samples, seed):
    # Imports/numerical execution occur only after main's remote-only guard.
    import numpy as np
    recordings = sorted({row['recording'] for row in rows})
    sums, counts = [], []
    for recording in recordings:
        eligible = [row for row in rows if row['recording'] == recording
                    and row['errors'][base].get(metric) is not None
                    and row['errors'][final].get(metric) is not None]
        sums.append(sum(row['errors'][base][metric] - row['errors'][final][metric] for row in eligible))
        counts.append(len(eligible))
    sums, counts = np.asarray(sums, dtype=np.float64), np.asarray(counts, dtype=np.int64)
    clusters = int((counts > 0).sum())
    report = {'metric': metric, 'direction': 'baseline_minus_final_positive_is_improvement',
              'recordings': len(recordings), 'eligible_recordings': clusters,
              'paired_states': int(counts.sum()), 'requested_resamples': samples, 'seed': seed,
              'estimate': float(sums.sum() / counts.sum()) if counts.sum() else None,
              'ci95_percentile': None, 'valid_resamples': 0,
              'limitation': 'Recording-cluster percentile bootstrap from one trained run and a bounded dataset; small recording counts limit precision. No independent-seed, causal or population-wide significance claim.'}
    if clusters < 2:
        report['reason'] = 'At least two eligible recordings are required'
        return report
    indices = np.random.default_rng(seed).integers(0, len(recordings), size=(samples, len(recordings)))
    totals, amounts = sums[indices].sum(1), counts[indices].sum(1)
    estimates = totals[amounts > 0] / amounts[amounts > 0]
    report.update(valid_resamples=int(len(estimates)),
                  ci95_percentile=[float(v) for v in np.quantile(estimates, [.025, .975])])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--joint', action='append', type=parse_joint, required=True)
    parser.add_argument('--geometry', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bootstrap-samples', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=20261016)
    args = parser.parse_args()
    if not Path('/sciclone/home/jchen57').is_dir():
        parser.error('Run numerical evaluation summaries only in the authorized remote ds Pod')
    if args.bootstrap_samples < 100:
        parser.error('--bootstrap-samples must be at least 100')
    if len({label for label, _ in args.joint}) != len(args.joint):
        parser.error('Joint labels must be unique')
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError('Use a fresh output JSON filename')
    inputs = {'baseline': args.baseline.expanduser().resolve(), **dict(args.joint)}
    evidence, provenance, reports, state = {}, {}, {}, {}
    for label, directory in inputs.items():
        paths = {key: directory / name for key, name in
                 (('provenance', 'provenance.json'), ('report', 'report.json'), ('csv', 'per_frame.csv'))}
        for path in paths.values():
            if not path.is_file():
                raise FileNotFoundError(path)
        provenance[label], reports[label] = read_json(paths['provenance']), read_json(paths['report'])
        state[label] = indexed(read_csv(paths['csv']))
        evidence[label] = {key: {'path': str(path), 'sha256': sha256(path)} for key, path in paths.items()}
        expected_mode = 'baseline' if label == 'baseline' else 'joint'
        if provenance[label]['mode'] != expected_mode or reports[label]['mode'] != expected_mode:
            raise ValueError('Input mode does not match label role: ' + label)
        if provenance[label]['source_manifest_sha256'] != reports[label]['source_manifest_sha256']:
            raise ValueError('Evaluation report/provenance manifest mismatch')
        if provenance[label]['evaluated_indices'] != reports[label]['evaluated_indices']:
            raise ValueError('Evaluation report/provenance clip coverage mismatch')
        if label != 'baseline':
            p = provenance[label]
            if (not p.get('checkpoint_sha256') or p['checkpoint_step'] != reports[label]['checkpoint_step']
                    or p['training_identity']['mano_asset_sha256'] != p['mano_asset_sha256']
                    or p['training_identity']['hawor_checkpoint_sha256'] != p['official_hawor_checkpoint_sha256']
                    or p['training_identity']['hawor_config_sha256'] != p['hawor_config_sha256']):
                raise ValueError('Joint checkpoint/assets provenance is inconsistent: ' + label)
            rank_hashes = p.get('saved_rank_model_sha256', [])
            if rank_hashes and len(set(rank_hashes)) != 1:
                raise ValueError('Saved rank model hashes differ')
    anchor = provenance['baseline']
    keys = set(state['baseline'])
    for label, p in provenance.items():
        for key in ('source_manifest_sha256', 'mano_asset_sha256', 'official_hawor_checkpoint_sha256',
                    'hawor_config_sha256', 'evaluated_indices', 'training_fixed_indices', 'history_policy'):
            if p[key] != anchor[key]:
                raise ValueError(f'Unpaired evaluation provenance {key}: {label}')
        for key in ('clip_length', 'precision', 'split'):
            if p['arguments'][key] != anchor['arguments'][key]:
                raise ValueError(f'Unpaired inference setting {key}: {label}')
        if set(state[label]) != keys:
            raise ValueError('State CSV frame coverage differs: ' + label)
    geometry_path = args.geometry.expanduser().resolve()
    geometry = read_json(geometry_path)
    if (geometry.get('schema') != 'handloop.visual_geometry_evaluation.v1'
            or geometry['source_manifest_sha256'] != anchor['source_manifest_sha256']
            or geometry['mano_asset_sha256'] != anchor['mano_asset_sha256']):
        raise ValueError('Geometry report manifest/assets differ from state evaluation')
    expected_branches = ['baseline.raw'] + [label + '.' + branch for label, _ in args.joint
                                          for branch in ('raw', 'candidate', 'final')]
    if set(geometry['branches']) != set(expected_branches) or len(geometry['branches']) != len(expected_branches):
        raise ValueError('Geometry labels/branches must exactly match baseline and requested joint labels')
    if set(geometry['inputs']) != set(inputs):
        raise ValueError('Geometry input labels differ')
    for label, source in geometry['inputs'].items():
        if Path(source['path']).resolve() != inputs[label]:
            raise ValueError('Geometry predictions path differs from state input: ' + label)
    if not geometry.get('reference_fk_audit') or not all(v['passed'] for v in geometry['reference_fk_audit'].values()):
        raise ValueError('Reference FK audit must pass for every recording')
    geometry_csv = Path(geometry['per_frame_csv'])
    if not geometry_csv.is_absolute():
        geometry_csv = geometry_path.parent / geometry_csv
    geometric = indexed(read_csv(geometry_csv))
    if set(geometric) != keys:
        raise ValueError('Geometry CSV and state CSV must have identical frame coverage')
    evidence['geometry'] = {'report': {'path': str(geometry_path), 'sha256': sha256(geometry_path)},
                            'csv': {'path': str(geometry_csv), 'sha256': sha256(geometry_csv)}}
    rows = []
    for key in sorted(keys):
        raw = state['baseline'][key]
        geo = geometric[key]
        frame_id, hand = int(raw['frame_id']), raw['hand']
        complete = bool(int(raw['complete_reference_valid']))
        if int(geo['frame_id']) != frame_id or geo['handedness'] != hand:
            raise ValueError('Geometry/state frame identity mismatch')
        row = {'recording': key[0], 'clip_index': key[1], 'reference_index': key[2],
               'frame_id': frame_id, 'hand': hand, 'complete_reference_valid': complete, 'errors': {}}
        for label, values in state.items():
            local = values[key]
            if (int(local['frame_id']) != frame_id or local['hand'] != hand
                    or bool(int(local['complete_reference_valid'])) != complete
                    or int(local['frame_valid']) != int(raw['frame_valid'])
                    or local['target_valid_17'] != raw['target_valid_17']
                    or local['raw_available_17'] != raw['raw_available_17']
                    or abs(float(local['timestamp_seconds']) - float(raw['timestamp_seconds'])) > 1e-9
                    or bool(int(geo[label + '_complete_reference_valid'])) != complete):
                raise ValueError('Paired frame metadata/masks disagree: ' + label)
            for branch in (('raw',) if label == 'baseline' else ('raw', 'candidate', 'final')):
                name = label + '.' + branch
                errors = {metric: numeric(local, branch + '_' + metric) for metric in STATE_METRICS}
                errors['finger_count'] = numeric(local, branch + '_finger_count') or 0
                errors.update({metric: numeric(geo, name + '_' + metric) for metric in GEOMETRY_METRICS})
                if not complete and any(errors[k] is not None for k in METRICS):
                    raise ValueError('Invalid references must not contribute numerical metrics')
                row['errors'][name] = errors
        rows.append(row)
    overall = summarize(rows, expected_branches)
    # Confirm exported rows recompute the exact state means/counts in each input report.
    for label, source in reports.items():
        for branch in (('raw',) if label == 'baseline' else ('raw', 'candidate', 'final')):
            for metric in STATE_METRICS:
                item = overall['branches'][label + '.' + branch][metric]
                saved_key = branch + '/' + metric
                if item['count'] != source['all']['metric_counts'].get(saved_key, 0):
                    raise ValueError('Per-frame/report state metric count mismatch')
                if item['count'] and not math.isclose(item['mean'], source['all'][saved_key], abs_tol=1e-3, rel_tol=1e-5):
                    raise ValueError('Per-frame/report state metric mean mismatch')
    for branch in expected_branches:
        source = geometry['all']['branches'][branch]
        for metric in GEOMETRY_METRICS:
            item = overall['branches'][branch][metric]
            if item['count'] != source['valid_states'] or (item['count'] and not math.isclose(item['mean'], source[metric], abs_tol=1e-5, rel_tol=1e-7)):
                raise ValueError('Per-frame/report geometric metric mismatch')
    comparisons, bootstrap = {}, {}
    for label, _ in args.joint:
        for base in ('baseline.raw', label + '.raw'):
            final = label + '.final'
            name = final + '_vs_' + base
            comparisons[name] = compare(rows, base, final)
            bootstrap[name] = {metric: bootstrap_recordings(rows, base, final, metric,
                                samples=args.bootstrap_samples, seed=args.seed)
                               for metric in ('wrist_mm', 'mpjpe21_mm')}
    fixed = set(anchor['training_fixed_indices'])
    fixed_rows = [row for row in rows if row['clip_index'] in fixed]
    try:
        revision = subprocess.check_output(['git', '-C', str(Path(__file__).resolve().parents[1]),
                                            'rev-parse', 'HEAD'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = 'unavailable'
    report = {'schema': 'handloop.visual_paired_comparison.v1',
        'completed_utc': datetime.now(timezone.utc).isoformat(),
        'code_version': revision, 'script_sha256': sha256(Path(__file__)),
        'source_manifest_sha256': anchor['source_manifest_sha256'], 'split': anchor['split'],
        'mano_asset_sha256': anchor['mano_asset_sha256'], 'branches': expected_branches,
        'checkpoint_identities': {label: {key: p.get(key) for key in
            ('checkpoint', 'checkpoint_sha256', 'checkpoint_step', 'checkpoint_code_version', 'saved_rank_model_sha256')}
            for label, p in provenance.items()},
        'inputs': evidence, 'all': overall, 'paired_comparisons': comparisons,
        'recording_cluster_bootstrap': bootstrap,
        'recording': grouped(rows, expected_branches, 'recording'),
        'hand': grouped(rows, expected_branches, 'hand'),
        'fixed_subset': {'indices': sorted(fixed), 'recomputed': summarize(fixed_rows, expected_branches),
                         'input_reports': {label: {'complete': r['fixed_subset_complete'], 'metrics': r['fixed_subset']}
                                           for label, r in reports.items()}},
        'uncertainty_and_selection_metrics': {label: {key: value for key, value in source['all'].items()
            if key not in ('metric_counts',) and '/' not in key} for label, source in reports.items()},
        'protocol': {'join': '(recording, clip_index, reference_index), exact input frame coverage and masks',
            'state_metrics': 'Anatomical wrist L2 mm; global root and 15 local finger SO(3) geodesic degrees; finger rotations pooled by actual count.',
            'geometry_metrics': geometry['metric_definition'],
            'paired_improvement': 'Matched eligible states; baseline mean minus final mean. Percentage divides by matched baseline mean; >1mm degradation is final-baseline >1.',
            'finger_pairing': 'Requires all15 finger rotations in both branches for same-group paired comparison.',
            'bootstrap': f'{args.bootstrap_samples} resamples of complete recordings with replacement; every selected recording contributes its paired error-difference sum and count, pooled before division; percentile 2.5/97.5.',
            'checkpoint_policy': 'Named checkpoint inputs are evaluated as supplied. No test score is used to select, replace, or rank a checkpoint.',
            'limitations': 'Single-run bounded fixed-camera/GT-crop fitted-MANO evaluation; reference mesh is fitted, not an independent scan. Bootstrap reflects recording resampling only, with limited precision for few recordings; it does not establish causality, seed robustness, detector/SLAM performance or population-wide significance.'}}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.tmp')
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(output)
    print(json.dumps({'complete': True, 'output': str(output), 'all': overall}, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
