#!/usr/bin/env python3
"""Select explicitly favorable held-out-test showcase windows from saved poses.

Run only in the authorized compute Pod. This reads fixed baseline.raw and
validation-selected best.final archives, never performs neural inference or
changes a checkpoint. All shards are coarsely ranked with existing 3D metrics;
only the disclosed candidate pool is decoded with each prediction's own beta.
Final ranking emphasizes sustained calibrated 2D improvement, without aligning
or modifying the geometry that the renderer will show.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for part in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(part)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def safe(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Path escapes root')
    return path


def metrics(result, branch):
    return {key: value['mean'] for key, value in
            result['summary']['all']['branches'].get(branch, {}).get('metrics', {}).items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'source-root', 'evaluation-root', 'mano-models', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--cpu-threads', type=int, default=2)
    parser.add_argument('--decode-batch-size', type=int, default=256)
    parser.add_argument('--per-side-per-ranking', type=int, default=24)
    parser.add_argument('--window-frames', type=int, default=45)
    parser.add_argument('--minimum-frames', type=int, default=30)
    parser.add_argument('--segments', type=int, default=3)
    parser.add_argument('--max-best-projected-px', type=float, default=10.)
    args = parser.parse_args()
    if not Path('/sciclone/data10/jchen57').is_dir():
        parser.error('Numerical data processing is remote compute Pod only')
    if not 30 <= args.minimum_frames <= args.window_frames or args.segments < 2:
        parser.error('Use at least 30 consecutive frames and at least two segments')
    root = args.root.resolve()
    for name in ('source_root', 'evaluation_root', 'mano_models', 'output'):
        path = getattr(args, name).resolve()
        if not path.is_relative_to(root):
            parser.error(name + ' must be inside root')
        setattr(args, name, path)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'selection.json').exists():
        raise FileExistsError('Refusing to replace an existing selection')
    os.environ.setdefault('OMP_NUM_THREADS', str(args.cpu_threads))
    os.environ.setdefault('OPENBLAS_NUM_THREADS', str(args.cpu_threads))
    os.environ.setdefault('MKL_NUM_THREADS', str(args.cpu_threads))
    import numpy as np
    import torch
    torch.set_num_threads(args.cpu_threads)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from dexycb_test_metrics import dex_joints, pa_joint_errors_mm
    from handloop.visual.codec import MANOStateCodec

    started = time.monotonic()
    index = read(args.source_root / 'shard_index.json')
    if (index['phase'] != 'complete' or index['expected_shards'] != 1280
            or index['completed_shards'] != 1280 or not index['eval_only']
            or index['training_allowed']):
        raise ValueError('Require completed eval-only full s0 test')
    source_selection = read(args.source_root / 'selection.json')
    metadata = {}
    for subject in source_selection['source_index_selection']:
        for entry in subject['selected']:
            metadata[subject['subject'] + '/' + entry['name']] = entry['meta']
    coarse = []
    models = {}
    for number, indexed in enumerate(index['shards']):
        key = indexed['key']
        branches = {}
        for label, branch in (('baseline', 'raw'), ('best', 'final')):
            result = read(args.evaluation_root / label / 'shards' / key / 'result.json')
            if (result['phase'] != 'complete' or result['frames'] != indexed['frames']
                    or result['source_manifest_sha256'] != indexed['manifest_sha256']):
                raise ValueError('Source/result identity mismatch: ' + key)
            if label in models and models[label] != result['identity']:
                raise ValueError('Model identity changed within evaluation')
            models[label] = result['identity']
            branches[label] = metrics(result, branch)
        item = {**indexed, 'coarse_metrics': branches}
        # Hand is determined from source metadata, not from the checkpoint.
        physical = indexed['physical_recording']
        meta = metadata[physical]
        item['hand'] = meta['mano_sides'][0]
        item['object_ids'] = meta.get('ycb_ids', [])
        item['grasped_object_index'] = meta.get('ycb_grasp_ind')
        grasp_index = item['grasped_object_index']
        item['grasped_object_id'] = (item['object_ids'][grasp_index]
                                    if isinstance(grasp_index, int)
                                    and 0 <= grasp_index < len(item['object_ids']) else None)
        if indexed['frames'] >= args.minimum_frames:
            coarse.append(item)
        if (number + 1) % 160 == 0:
            print(json.dumps({'phase': 'coarse_scan', 'shards': number + 1}), flush=True)
    def improvement(item, metric):
        return item['coarse_metrics']['baseline'][metric] - item['coarse_metrics']['best'][metric]
    def abs_rank(item):
        delta = improvement(item, 'mpjpe21_mm')
        return delta * max(delta / item['coarse_metrics']['baseline']['mpjpe21_mm'], 0)
    ranks = {'absolute_reduction_times_relative': abs_rank,
             'wrist_aligned_reduction': lambda x: improvement(x, 'wrist_aligned_mpjpe21_mm'),
             'pa_reduction': lambda x: improvement(x, 'pa_mpjpe21_mm'),
             'absolute_and_local': lambda x: abs_rank(x) * max(improvement(x, 'wrist_aligned_mpjpe21_mm'), .1)}
    selected, pool_reasons = {}, {}
    for side in ('left', 'right'):
        eligible = [item for item in coarse if item['hand'] == side
                    and improvement(item, 'mpjpe21_mm') > 10]
        for label, rank in ranks.items():
            # Keep several views for projection diversity, with a bounded pool.
            ranked = sorted(eligible, key=lambda x: (-rank(x), x['key']))
            recording_count, count = {}, 0
            for item in ranked:
                physical = item['physical_recording']
                if recording_count.get(physical, 0) >= 2:
                    continue
                recording_count[physical] = recording_count.get(physical, 0) + 1
                selected[item['key']] = item
                pool_reasons.setdefault(item['key'], []).append(label + ':' + side)
                count += 1
                if count == args.per_side_per_ranking:
                    break
    coarse_report = {'schema': 'handloop.showcase_coarse_scan.v1',
        'scanned_shards': len(index['shards']), 'source_frames': index['frames'],
        'eligible_shards_at_least_minimum_frames': len(coarse),
        'decode_pool_shards': len(selected), 'decode_pool_policy': {
            'rankings': list(ranks), 'per_side_per_ranking': args.per_side_per_ranking,
            'maximum_views_per_physical_recording_per_ranking': 2,
            'minimum_absolute_improvement_mm': 10},
        'models': models, 'decode_pool': [dict(item, pool_reasons=pool_reasons[item['key']])
                                        for item in sorted(selected.values(), key=lambda x: x['key'])],
        'top_absolute_shards': sorted(coarse, key=lambda x: (-abs_rank(x), x['key']))[:30]}
    write(args.output / 'coarse_scan.json', coarse_report)
    print(json.dumps({'phase': 'decode_pool', 'shards': len(selected)}), flush=True)
    codec = MANOStateCodec(args.mano_models / 'MANO_RIGHT.pkl', args.mano_models / 'MANO_LEFT.pkl',
                           left_shapedirs_fix=True).to(args.device).eval()
    windows, decoded_frames = [], 0
    for number, item in enumerate(sorted(selected.values(), key=lambda x: x['key'])):
        key = item['key']
        manifest_path = safe(args.source_root, item['manifest'])
        if sha(manifest_path) != item['manifest_sha256']:
            raise ValueError('Manifest changed')
        manifest = read(manifest_path)
        if manifest['phase'] != 'complete' or len(manifest['records']) != 1:
            raise ValueError('Require one complete record')
        record = manifest['records'][0]
        if record['split'] != 'test':
            raise ValueError('Showcases must use held-out test')
        reference_path = safe(manifest_path.parent, record['reference_file'])
        if sha(reference_path) != record['reference_sha256']:
            raise ValueError('Reference changed')
        with np.load(reference_path, allow_pickle=False) as archive:
            source = {name: archive[name].copy() for name in ('frame_ids', 'joint_3d',
                'joint_3d_valid', 'camera_intrinsics', 'camera_to_world', 'handedness', 'bbox_xyxy')}
        ids = source['frame_ids']
        target = torch.as_tensor(source['joint_3d'], device=args.device)
        side = torch.as_tensor(source['handedness'].astype(bool), device=args.device)
        c2w, intrinsic = source['camera_to_world'], source['camera_intrinsics']
        def project(points):
            camera = np.einsum('nkj,nij->nki', points - c2w[:, None, :3, 3],
                               c2w[:, :3, :3].transpose(0, 2, 1))
            homogeneous = np.einsum('nkj,nij->nki', camera, intrinsic)
            return homogeneous[..., :2] / camera[..., 2:3], camera[..., 2]
        target_uv, target_depth = project(source['joint_3d'])
        values, prediction_sources = {}, {}
        for label, branch in (('baseline', 'raw'), ('best', 'final')):
            prediction_path = args.evaluation_root / label / 'shards' / key / 'predictions.npz'
            result = read(prediction_path.parent / 'result.json')
            digest = sha(prediction_path)
            if digest != result['predictions_sha256']:
                raise ValueError('Prediction archive changed')
            with np.load(prediction_path, allow_pickle=False) as archive:
                if not np.array_equal(archive['frame_ids'], ids):
                    raise ValueError('Prediction/reference frame mismatch')
                state, beta, available = (archive[name].copy() for name in (branch, 'beta', 'availability'))
            if not np.isfinite(state).all() or not np.isfinite(beta).all():
                raise ValueError('Nonfinite predicted state/beta')
            geometry = []
            for begin in range(0, len(ids), args.decode_batch_size):
                end = begin + args.decode_batch_size
                with torch.inference_mode():
                    decoded = codec.geometry(torch.as_tensor(state[begin:end], device=args.device),
                        torch.as_tensor(beta[begin:end], device=args.device), side[begin:end])
                    geometry.append(dex_joints(decoded, side[begin:end]))
            joints = torch.cat(geometry)
            delta = joints - target
            uv, depth = project(joints.cpu().numpy())
            with torch.inference_mode():
                pa = pa_joint_errors_mm(joints, target).mean(-1).cpu().numpy()
            values[label] = {
                'absolute_mpjpe_mm': delta.norm(dim=-1).mean(-1).cpu().numpy() * 1000,
                'wrist_aligned_mpjpe_mm': (delta-delta[:, :1]).norm(dim=-1).mean(-1).cpu().numpy() * 1000,
                'pa_mpjpe_mm': pa,
                'projected_joint_error_px': np.linalg.norm(uv-target_uv, axis=-1).mean(-1),
                'eligible': available.all(-1) & source['joint_3d_valid'].all(-1)
                    & (depth > 1e-5).all(-1) & (target_depth > 1e-5).all(-1),
                'in_image_joints': ((uv[..., 0] >= 0) & (uv[..., 0] < 640)
                    & (uv[..., 1] >= 0) & (uv[..., 1] < 480)).sum(-1)}
            prediction_sources[label] = {'branch': branch, 'path': str(prediction_path), 'sha256': digest}
        decoded_frames += len(ids)
        # Evaluate fixed-size sliding windows inside complete consecutive runs.
        boundaries = np.r_[0, np.flatnonzero(np.diff(ids) != 1) + 1, len(ids)]
        shard_windows = []
        for low, high in zip(boundaries[:-1], boundaries[1:]):
            length = int(min(args.window_frames, high - low))
            if length < args.minimum_frames:
                continue
            for begin in range(low, high-length+1):
                end = begin + length
                b = {name: value[begin:end] for name, value in values['baseline'].items()}
                h = {name: value[begin:end] for name, value in values['best'].items()}
                if not b['eligible'].all() or not h['eligible'].all() or h['in_image_joints'].min() < 20:
                    continue
                summary = {name: {'baseline': float(b[name].mean()), 'handloop': float(h[name].mean()),
                                 'improvement': float((b[name]-h[name]).mean()),
                                 'improved_frame_fraction': float((h[name] < b[name]).mean())}
                           for name in ('absolute_mpjpe_mm', 'wrist_aligned_mpjpe_mm',
                                        'pa_mpjpe_mm', 'projected_joint_error_px')}
                absolute, projected = summary['absolute_mpjpe_mm'], summary['projected_joint_error_px']
                if (absolute['improvement'] < 10 or absolute['handloop'] > .8 * absolute['baseline']
                        or absolute['improved_frame_fraction'] < .8 or projected['improvement'] < 2
                        or projected['handloop'] > .75 * projected['baseline']
                        or projected['handloop'] > args.max_best_projected_px
                        or projected['improved_frame_fraction'] < .8):
                    continue
                local_bonus = 1 + .15 * max(min(summary['wrist_aligned_mpjpe_mm']['improvement'], 5), 0)
                score = projected['improvement'] * projected['improved_frame_fraction'] * local_bonus
                # A low final projection error makes visible contact more legible.
                score *= 1 / (1 + projected['handloop'] / 10)
                candidate = {'key': key, 'manifest': item['manifest'],
                    'start_frame_id': int(ids[begin]), 'end_frame_id': int(ids[end-1]), 'max_frames': length,
                    'hand': item['hand'], 'subject': record['subject'],
                    'physical_recording': record['physical_recording'], 'camera_serial': record['serial'],
                    'object_ids': item['object_ids'], 'grasped_object_index': item['grasped_object_index'],
                    'grasped_object_id': item['grasped_object_id'],
                    'manifest_sha256': item['manifest_sha256'], 'reference_sha256': record['reference_sha256'],
                    'prediction_sources': prediction_sources, 'selection_metrics': summary, 'score': float(score),
                    'pool_reasons': pool_reasons[key],
                    'error_summary': {'metric': 'Absolute MPJPE', 'unit': 'mm',
                        'baseline': absolute['baseline'], 'handloop': absolute['handloop'], 'frames': length}}
                shard_windows.append(candidate)
        # One best sustained window per camera; this avoids near-identical top entries.
        if shard_windows:
            windows.append(max(shard_windows, key=lambda x: (x['score'], -x['start_frame_id'])))
        if (number + 1) % 16 == 0:
            print(json.dumps({'phase': 'decoding', 'shards': number + 1, 'frames': decoded_frames,
                              'qualified_windows': len(windows)}), flush=True)
    ranked = sorted(windows, key=lambda x: (-x['score'], x['key'], x['start_frame_id']))
    write(args.output / 'candidate_windows.json', {'scanned_shards': len(index['shards']),
        'decoded_shards': len(selected), 'decoded_frames': decoded_frames,
        'qualifying_windows': ranked})
    chosen = []
    # Ensure both hands are visible in the presentation before adding variety.
    for side_name in ('right', 'left'):
        available = [item for item in ranked if item['hand'] == side_name]
        if not available:
            raise ValueError('No sustained qualifying window for ' + side_name + '; inspect thresholds')
        chosen.append(available[0])
    for item in ranked:
        if len(chosen) >= args.segments:
            break
        if (item['physical_recording'] in {x['physical_recording'] for x in chosen}
                or item['subject'] in {x['subject'] for x in chosen}
                or (item['grasped_object_id'] is not None
                    and item['grasped_object_id'] in {x['grasped_object_id'] for x in chosen})):
            continue
        chosen.append(item)
    if len(chosen) < args.segments:
        for item in ranked:
            if len(chosen) >= args.segments:
                break
            if item['physical_recording'] not in {x['physical_recording'] for x in chosen}:
                chosen.append(item)
    if len(chosen) < args.segments:
        raise ValueError('Insufficient distinct physical recordings')
    for number, item in enumerate(chosen):
        item['title'] = f"Case {number+1}: {item['hand']} hand / {item['subject']}"
    report = {'schema': 'handloop.favorable_test_showcase_selection.v1', 'phase': 'complete',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'script_sha256': sha(__file__),
        'source_root': str(args.source_root), 'evaluation_root': str(args.evaluation_root),
        'source_index_sha256': sha(args.source_root / 'shard_index.json'),
        'scanned_shards': len(index['shards']), 'source_frames': index['frames'],
        'decoded_candidate_shards': len(selected), 'decoded_candidate_frames': decoded_frames,
        'qualified_shards': len(ranked), 'models': models,
        'disclosure': 'Deliberately favorable examples selected by held-out-test reference errors; not representative averages. No checkpoint selection or neural inference. The 2D ranking covers only the disclosed 3D-ranked candidate pool, not every test window.',
        'geometry': 'Own saved predicted beta; original calibrated K and inverse camera_to_world; no PA/GT wrist/rotation/scale correction of displayed poses. PA is diagnostic only. Source GT is fitted DexYCB MANO21, not independent MoCap.',
        'protocol': 'Existing GT-crop full-test saved predictions; 16-frame bidirectional blocks / 32-frame feedback resets.',
        'selection_rule': {'window_frames': args.window_frames, 'minimum_consecutive_frames': args.minimum_frames,
            'minimum_absolute_improvement_mm': 10, 'minimum_absolute_relative_improvement': .2,
            'minimum_projected_improvement_px': 2, 'minimum_projected_relative_improvement': .25,
            'maximum_handloop_projected_error_px': args.max_best_projected_px,
            'minimum_improved_frame_fraction_absolute_and_projected': .8,
            'minimum_handloop_in_image_joints_every_frame': 20,
            'score': 'projected improvement * projected improved-frame fraction * (1+.15*clamp(wrist-aligned improvement,0,5)) / (1+HandLoop projected error/10)',
            'diversity': 'best qualifying right and left; subsequent different subjects, grasped objects and physical recordings when available'},
        'entries': chosen, 'top_qualifying_windows': ranked[:50],
        'elapsed_seconds': time.monotonic()-started}
    write(args.output / 'selection.json', report)
    print(json.dumps({'phase': 'complete', 'decoded_shards': len(selected), 'entries':
        [{k: item[k] for k in ('key', 'start_frame_id', 'end_frame_id', 'hand', 'selection_metrics')}
         for item in chosen], 'output': str(args.output / 'selection.json')}), flush=True)


if __name__ == '__main__':
    main()
