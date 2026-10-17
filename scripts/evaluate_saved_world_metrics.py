#!/usr/bin/env python3
"""Remote-only trajectory metrics from immutable full-DexYCB saved predictions.

No neural inference, image loading, checkpoint selection, or shape substitution.
The trajectory alignment operations follow HaWoR 66c7d410's HOT3D evaluator;
the observations remain DexYCB supplied Dex21 landmarks and its static camera
gauge. Both complete coverage and the official >=16-frame run filter are shown.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time


BRANCHES = {'baseline': 'raw', 'best': 'final', 'last': 'final'}
OFFICIAL_COMMIT = '66c7d4108d58a716deccd192cb7645170cdc7bd7'
SOURCE_FIELDS = ('frame_ids', 'timestamps', 'target_world', 'target_valid',
                 'joint_3d', 'joint_3d_valid', 'handedness', 'beta', 'camera_to_world')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def safe(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Path escapes root: ' + str(relative))
    return path


def similarity(predicted, target, *, scale=True, proper=True, allow_degenerate=False):
    """Return row-vector pred->GT Sim(3), supporting optional batch axes.

    The fit uses every point in the penultimate dimension. Degenerate rigid
    trajectories have a valid translation-only optimum; a degenerate Sim(3)
    hand cloud is an error rather than a successful zero-valued sample.
    """
    import numpy as np

    x, y = np.asarray(predicted, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if x.shape != y.shape or x.shape[-1] != 3 or x.shape[-2] < 1:
        raise ValueError('Similarity requires matching nonempty [..., points, 3]')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise FloatingPointError('Nonfinite alignment input')
    xm, ym = x.mean(axis=-2, keepdims=True), y.mean(axis=-2, keepdims=True)
    xc, yc = x - xm, y - ym
    variance = (xc * xc).sum(axis=(-2, -1))
    degenerate = variance <= 1e-20
    if np.any(degenerate) and (scale or not allow_degenerate):
        raise ValueError('Degenerate predicted cloud for alignment')
    u, singular, vh = np.linalg.svd(np.swapaxes(xc, -2, -1) @ yc)
    reflection = np.linalg.det(u @ vh) < 0
    signs = np.ones_like(singular)
    if proper:
        signs[..., -1] = np.where(reflection, -1., 1.)
    rotation = (u * signs[..., None, :]) @ vh
    factor = ((singular * signs).sum(axis=-1) / np.where(degenerate, 1., variance)
              if scale else np.ones_like(variance))
    translation = ym - factor[..., None, None] * (xm @ rotation)
    return {'rotation': rotation, 'scale': factor, 'translation': translation,
            'reflection_unconstrained': reflection, 'degenerate_prediction': degenerate}


def transform(points, fit):
    import numpy as np

    return (np.asarray(points, dtype=np.float64) @ fit['rotation']) * fit['scale'][..., None, None] + fit['translation']


def acceleration_errors(predicted, target, fps):
    """GT acceleration error per center in m/s²; no coordinate alignment."""
    import numpy as np

    delta = np.asarray(predicted, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    second = delta[2:] - 2 * delta[1:-1] + delta[:-2]
    return np.linalg.norm(second, axis=-1).mean(axis=-1) * fps ** 2


def audit_one_segment(gt_joints, pred_joints, gt_trans, pred_trans, fps=30., chunk_length=100):
    """Unreduced arrays for the separate, official-source acceptance harness."""
    import numpy as np

    target, predicted = np.asarray(gt_joints, dtype=np.float64), np.asarray(pred_joints, dtype=np.float64)
    gt_trans, pred_trans = np.asarray(gt_trans, dtype=np.float64), np.asarray(pred_trans, dtype=np.float64)
    w, wa = [], []
    for a in range(0, len(target), chunk_length):
        x, y = predicted[a:a + chunk_length], target[a:a + chunk_length]
        first = similarity(x[:2].reshape(-1, 3), y[:2].reshape(-1, 3))
        whole = similarity(x.reshape(-1, 3), y.reshape(-1, 3))
        w.extend(np.linalg.norm(transform(x, first) - y, axis=-1).mean(-1) * 1000)
        wa.extend(np.linalg.norm(transform(x, whole) - y, axis=-1).mean(-1) * 1000)
    proper_pa = np.linalg.norm(transform(predicted, similarity(predicted, target)) - target, axis=-1).mean(-1) * 1000
    path = float(np.linalg.norm(np.diff(gt_trans, axis=0), axis=-1).sum())
    rigid = similarity(pred_trans, gt_trans, scale=False, allow_degenerate=True)
    rte = (np.linalg.norm(transform(pred_trans, rigid) - gt_trans, axis=-1) / path * 100
           if path > 1e-12 else np.full(len(target), np.nan))
    return {'w_mpjpe_mm': np.asarray(w), 'wa_mpjpe_mm': np.asarray(wa),
            'pa_mpjpe_so3_mm': proper_pa,
            'accel_official_mps2': acceleration_errors(predicted, target, fps)[1:-1],
            'rte_percent': rte}


def runs_from_ids(ids):
    import numpy as np

    bounds = np.r_[0, np.flatnonzero(np.diff(ids) != 1) + 1, len(ids)]
    return [(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]


def new_stat(unit):
    return {'unit': unit, 'frame_weighted': {'sum': 0., 'count': 0},
            'equal_block_weighted': {'sum': 0., 'count': 0},
            'equal_run_weighted': {'sum': 0., 'count': 0}}


def add_stat(bucket, name, values, *, unit, block_values=None):
    """One call represents one continuous run; values are per-frame/center."""
    import numpy as np

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError('Metric values must be finite one-dimensional samples: ' + name)
    metric = bucket.setdefault('metrics', {}).setdefault(name, new_stat(unit))
    if not len(values):
        return
    micro, macro = metric['frame_weighted'], metric['equal_run_weighted']
    micro['sum'] += float(values.sum(dtype=np.float64))
    micro['count'] += len(values)
    macro['sum'] += float(values.mean())
    macro['count'] += 1
    blocks = [values.mean()] if block_values is None else block_values
    metric['equal_block_weighted']['sum'] += float(np.asarray(blocks).sum(dtype=np.float64))
    metric['equal_block_weighted']['count'] += len(blocks)


def finalize(bucket):
    result = json.loads(json.dumps(bucket))
    for metric in result.get('metrics', {}).values():
        for kind in ('frame_weighted', 'equal_block_weighted', 'equal_run_weighted'):
            value = metric[kind]
            value['mean'] = value['sum'] / value['count'] if value['count'] else None
    return result


def compute_run(bucket, predicted, target, pred_transl, gt_transl, *, fps=30.,
                chunk_length=100, reset_length=32, near_zero_path_m=.001):
    """Accumulate one gap-free track, retaining all protocol denominators."""
    import numpy as np

    length, joints = target.shape[:2]
    coverage = bucket.setdefault('coverage', Counter())
    coverage['frames'] += length
    coverage['joint_observations'] += length * joints
    coverage['continuous_runs'] += 1
    coverage['runs_under_2_frames'] += int(length < 2)
    coverage['runs_under_3_frames'] += int(length < 3)
    coverage['runs_under_5_frames'] += int(length < 5)
    coverage['runs_under_16_frames'] += int(length < 16)
    chunks = [(a, min(a + chunk_length, length)) for a in range(0, length, chunk_length)]
    coverage['alignment_blocks'] += len(chunks)
    coverage['alignment_tail_blocks'] += sum(b - a < chunk_length for a, b in chunks)
    coverage['alignment_singleton_blocks'] += sum(b - a == 1 for a, b in chunks)
    pa_o3_fit = similarity(predicted, target, proper=False)
    pa_so3_fit = similarity(predicted, target, proper=True)
    pa_o3 = np.linalg.norm(transform(predicted, pa_o3_fit) - target, axis=-1).mean(-1) * 1000
    pa_so3 = np.linalg.norm(transform(predicted, pa_so3_fit) - target, axis=-1).mean(-1) * 1000
    coverage['pa_unconstrained_reflection_frames'] += int(pa_o3_fit['reflection_unconstrained'].sum())
    absolute = np.linalg.norm(predicted - target, axis=-1).mean(-1) * 1000
    wrist = np.linalg.norm((predicted - predicted[:, :1]) - (target - target[:, :1]), axis=-1).mean(-1) * 1000
    for name, values in (('absolute_mpjpe21_mm', absolute), ('wrist_aligned_mpjpe21_mm', wrist),
                         ('pa_freihand_o3_mpjpe21_mm', pa_o3), ('pa_hawor_so3_mpjpe21_mm', pa_so3)):
        add_stat(bucket, name, values, unit='mm', block_values=[values[a:b].mean() for a, b in chunks])
    w_values, wa_values, w_blocks, wa_blocks = [], [], [], []
    for a, b in chunks:
        x, y = predicted[a:b], target[a:b]
        first_fit = similarity(x[:2].reshape(-1, 3), y[:2].reshape(-1, 3))
        global_fit = similarity(x.reshape(-1, 3), y.reshape(-1, 3))
        coverage['w_unconstrained_reflection_blocks'] += int(first_fit['reflection_unconstrained'])
        coverage['wa_unconstrained_reflection_blocks'] += int(global_fit['reflection_unconstrained'])
        w = np.linalg.norm(transform(x, first_fit) - y, axis=-1).mean(-1) * 1000
        wa = np.linalg.norm(transform(x, global_fit) - y, axis=-1).mean(-1) * 1000
        w_values.extend(w.tolist())
        wa_values.extend(wa.tolist())
        w_blocks.append(float(w.mean()))
        wa_blocks.append(float(wa.mean()))
    add_stat(bucket, 'w_mpjpe21_mm', w_values, unit='mm', block_values=w_blocks)
    add_stat(bucket, 'wa_mpjpe21_mm', wa_values, unit='mm', block_values=wa_blocks)
    all_accel = acceleration_errors(predicted, target, fps)
    centers = np.arange(1, length - 1, dtype=np.int64)
    crossing_reset = ((centers % reset_length) == 0) | ((centers % reset_length) == reset_length - 1)
    official_accel = all_accel[1:-1]
    official_crossing = crossing_reset[1:-1]
    coverage['accel_all_valid_triples'] += len(all_accel)
    coverage['accel_all_reset_crossing_triples'] += int(crossing_reset.sum())
    coverage['accel_hawor_trimmed_centers'] += len(official_accel)
    coverage['accel_hawor_trimmed_reset_crossing_centers'] += int(official_crossing.sum())
    coverage['accel_hawor_trimmed_excluding_reset_centers'] += int((~official_crossing).sum())
    add_stat(bucket, 'accel_hawor_m_s2', official_accel, unit='m/s^2')
    add_stat(bucket, 'accel_hawor_excluding_reset_m_s2', official_accel[~official_crossing], unit='m/s^2')
    add_stat(bucket, 'accel_all_valid_triples_m_s2', all_accel, unit='m/s^2')
    add_stat(bucket, 'accel_all_valid_excluding_reset_m_s2', all_accel[~crossing_reset], unit='m/s^2')
    # Official RTE is native MANO transl, aligned by SE(3) without scale. It
    # divides every frame by this run's entire GT path, not per-step distance.
    path = float(np.linalg.norm(np.diff(gt_transl, axis=0), axis=-1).sum())
    pred_path = float(np.linalg.norm(np.diff(pred_transl, axis=0), axis=-1).sum())
    coverage['rte_runs'] += 1
    coverage['rte_gt_zero_path_runs'] += int(path <= 1e-12)
    coverage['rte_gt_zero_path_frames'] += length * int(path <= 1e-12)
    coverage['rte_gt_near_zero_nonzero_path_runs'] += int(1e-12 < path < near_zero_path_m)
    coverage['rte_pred_zero_path_runs'] += int(pred_path <= 1e-12)
    coverage['rte_gt_path_sum_m'] += path
    coverage['rte_pred_path_sum_m'] += pred_path
    rigid_fit = similarity(pred_transl, gt_transl, scale=False, allow_degenerate=True)
    coverage['rte_degenerate_pred_cloud_runs'] += int(rigid_fit['degenerate_prediction'])
    coverage['rte_unconstrained_reflection_runs'] += int(rigid_fit['reflection_unconstrained'])
    translation_errors = np.linalg.norm(transform(pred_transl, rigid_fit) - gt_transl, axis=-1)
    add_stat(bucket, 'rte_aligned_transl_error_mm', translation_errors * 1000, unit='mm')
    if path > 1e-12:
        values = translation_errors / path * 100
        add_stat(bucket, 'rte_percent', values, unit='%')
        coverage['rte_defined_frames'] += length
        coverage['rte_defined_runs'] += 1


def self_test(official_path=None):
    """Independent SciPy oracles plus physically interpretable motion fixtures."""
    import numpy as np
    from scipy.linalg import orthogonal_procrustes
    from scipy.spatial.transform import Rotation

    checks = {}

    def check(name, condition, **details):
        checks[name] = {'passed': bool(condition), **details}
        if not condition:
            raise AssertionError(name + ': ' + repr(details))

    rng = np.random.default_rng(20261017)
    cloud = rng.normal(size=(11, 21, 3)) * .05
    rotation = Rotation.from_rotvec([.3, -.5, .2]).as_matrix()
    x = 1.7 * (cloud @ rotation) + np.array([.2, -.3, .8])
    fit = similarity(x, cloud)
    residual = float(np.abs(transform(x, fit) - cloud).max())
    check('known_similarity_inverse', residual < 1e-12,
          max_error_m=residual, max_scale_error=float(abs(fit['scale'] - 1 / 1.7).max()),
          min_rotation_determinant=float(np.linalg.det(fit['rotation']).min()))
    # SciPy align_vectors solves weighted proper Kabsch, independently of our
    # SVD bookkeeping and row-vector convention. Scale is fitted afterwards.
    oracle = []
    for a, b in zip(x, cloud):
        ac, bc = a - a.mean(0), b - b.mean(0)
        r, _ = Rotation.align_vectors(bc, ac)
        ar = r.apply(ac)
        s = float((ar * bc).sum() / (ac * ac).sum())
        oracle.append(s * ar + b.mean(0))
    difference = float(np.abs(np.stack(oracle) - transform(x, fit)).max())
    check('proper_similarity_matches_scipy_kabsch', difference < 1e-12, max_error_m=difference)
    reflected = cloud.copy()
    reflected[..., 0] *= -1
    free = transform(reflected, similarity(reflected, cloud, proper=False))
    constrained = transform(reflected, similarity(reflected, cloud))
    free_error = float(np.linalg.norm(free - cloud, axis=-1).mean())
    proper_error = float(np.linalg.norm(constrained - cloud, axis=-1).mean())
    check('reflection_is_permitted_only_in_freihand_diagnostic', free_error < 1e-12 and proper_error > .001,
          free_error_m=free_error, proper_error_m=proper_error)
    independent = []
    noisy = x + rng.normal(size=x.shape) * .003
    for a, b in zip(noisy, cloud):
        ac, bc = a - a.mean(0), b - b.mean(0)
        r, singular_sum = orthogonal_procrustes(ac, bc)
        independent.append((singular_sum / (ac * ac).sum()) * (ac @ r) + b.mean(0))
    diff = float(np.abs(np.stack(independent) - transform(noisy, similarity(noisy, cloud, proper=False))).max())
    check('free_similarity_matches_scipy_orthogonal_procrustes', diff < 1e-12, max_error_m=diff)
    anisotropic = cloud * [1., 2., 3.]
    check('anisotropic_scaling_does_not_disappear',
          np.linalg.norm(transform(anisotropic, similarity(anisotropic, cloud)) - cloud) > .001)
    fps = 30.
    t = np.arange(80) / fps
    stationary = np.broadcast_to(cloud[0], (80, 21, 3)).copy()
    velocity = np.array([.2, -.1, .3])
    constant_speed = stationary + t[:, None, None] * velocity
    speed_acc = acceleration_errors(constant_speed, stationary, fps)
    check('constant_velocity_has_zero_acceleration_error', float(speed_acc.max()) < 1e-10,
          max_error_m_s2=float(speed_acc.max()))
    accel = np.array([.4, -.3, .2])
    accelerated = stationary + .5 * t[:, None, None] ** 2 * accel
    measured = acceleration_errors(accelerated, stationary, fps)
    check('known_constant_acceleration', float(abs(measured - np.linalg.norm(accel)).max()) < 1e-10,
          expected_m_s2=float(np.linalg.norm(accel)), max_error_m_s2=float(abs(measured - np.linalg.norm(accel)).max()))
    jumping = stationary.copy()
    jumping[32:] += [.05, 0, 0]
    bucket = {}
    compute_run(bucket, jumping, stationary, np.zeros((80, 3)), np.zeros((80, 3)))
    check('reset_jump_affects_two_centers_and_exclusion_removes_it',
          int((acceleration_errors(jumping, stationary, fps) > 1e-8).sum()) == 2
          and bucket['metrics']['accel_hawor_excluding_reset_m_s2']['frame_weighted']['sum'] < 1e-8,
          trimmed_centers=bucket['coverage']['accel_hawor_trimmed_centers'],
          trimmed_reset_crossing=bucket['coverage']['accel_hawor_trimmed_reset_crossing_centers'])
    check('stationary_rte_is_explicitly_undefined', bucket['coverage']['rte_gt_zero_path_runs'] == 1
          and 'rte_percent' not in bucket['metrics'])
    check('gap_splits_temporal_runs', runs_from_ids(np.array([0, 1, 2, 5, 6, 9])) == [(0, 3), (3, 5), (5, 6)])
    rigid = similarity(constant_speed[:, 0] @ rotation + [.1, .4, -.2],
                       constant_speed[:, 0], scale=False, allow_degenerate=True)
    check('native_translation_rigid_invariance',
          np.max(np.abs(transform(constant_speed[:, 0] @ rotation + [.1, .4, -.2], rigid)
                        - constant_speed[:, 0])) < 1e-11)
    singleton = {}
    compute_run(singleton, cloud[:1], cloud[:1], np.zeros((1, 3)), np.zeros((1, 3)))
    check('singleton_tail_retained_without_temporal_metric',
          singleton['coverage']['alignment_singleton_blocks'] == 1
          and singleton['metrics']['w_mpjpe21_mm']['frame_weighted']['count'] == 1
          and singleton['metrics']['accel_hawor_m_s2']['frame_weighted']['count'] == 0)
    if official_path is not None:
        checks['official_source_recorded'] = {'passed': True, 'path': str(official_path),
                                              'sha256': sha(official_path)}
    return {'schema': 'handloop.saved_world_metrics_self_test.v1', 'passed': True, 'checks': checks}


def load_shard(entry, source_root, evaluation_root, identities, selection_sha):
    """Bounded prefetch task: one source read and all three saved branches."""
    import numpy as np

    key = entry['key']
    manifest_path = safe(source_root, entry['manifest'])
    if sha(manifest_path) != entry['manifest_sha256']:
        raise ValueError('Changed source manifest: ' + key)
    manifest = read(manifest_path)
    if (manifest['phase'] != 'complete' or not manifest['eval_only'] or manifest['training_allowed']
            or manifest['selection_sha256'] != selection_sha or manifest['shard_id'] != key):
        raise ValueError('Invalid source shard identity: ' + key)
    results = {}
    for label in BRANCHES:
        directory = safe(evaluation_root / label / 'shards', key)
        result = read(directory / 'result.json')
        if (result['phase'] != 'complete' or result['identity'] != identities[label]
                or result['source_manifest_sha256'] != entry['manifest_sha256']
                or result['key'] != key or result['frames'] != entry['frames']):
            raise ValueError('Model/source result mismatch: ' + label + '/' + key)
        results[label] = result
    if not entry['frames']:
        if manifest['records'] or any(result['predictions_sha256'] is not None for result in results.values()):
            raise ValueError('Empty shard has unexpected records/predictions')
        return entry, manifest, None, {}, results
    if len(manifest['records']) != 1 or manifest['records'][0]['split'] != 'test':
        raise ValueError('Require exactly one held-out recording/view: ' + key)
    record = manifest['records'][0]
    if record['timestamp_source'] != 'nominal_30hz_frame_index_not_measured' or abs(record['nominal_dt'] - 1 / 30) > 1e-12:
        raise ValueError('Unexpected timestamp protocol')
    path = safe(manifest_path.parent, record['reference_file'])
    if sha(path) != record['reference_sha256']:
        raise ValueError('Changed reference archive: ' + key)
    with np.load(path, allow_pickle=False) as archive:
        source = {name: archive[name].copy() for name in SOURCE_FIELDS}
    ids, n = source['frame_ids'], entry['frames']
    if (ids.shape != (n,) or source['joint_3d'].shape != (n, 21, 3)
            or source['target_world'].shape != (n, 99)
            or source['handedness'].shape != (n,)
            or source['target_valid'].shape != (n, 17)
            or source['joint_3d_valid'].shape != (n, 21)
            or np.any(np.diff(ids) <= 0) or not source['target_valid'].all()
            or not source['joint_3d_valid'].all()):
        raise ValueError('Incomplete source reference: ' + key)
    if (not np.isfinite(source['target_world']).all() or not np.isfinite(source['joint_3d']).all()
            or not np.isfinite(source['beta']).all()
            or not np.allclose(source['timestamps'], ids / 30., rtol=0, atol=1e-10)
            or not np.array_equal(source['camera_to_world'], np.broadcast_to(np.eye(4), (n, 4, 4)))):
        raise ValueError('Nonfinite or non-static-camera source: ' + key)
    if not np.all(source['handedness'] == source['handedness'][0]) or source['handedness'][0] not in (0, 1):
        raise ValueError('Expected one constant hand per view')
    ids_hash = hashlib.sha256(np.asarray(ids, dtype='<i8').tobytes()).hexdigest()
    values = {}
    for label, branch in BRANCHES.items():
        path = safe(evaluation_root / label / 'shards', key) / 'predictions.npz'
        result = results[label]
        if sha(path) != result['predictions_sha256'] or ids_hash != result['frame_ids_sha256']:
            raise ValueError('Prediction or frame-ID hash mismatch: ' + label + '/' + key)
        with np.load(path, allow_pickle=False) as archive:
            if not np.array_equal(archive['frame_ids'], ids):
                raise ValueError('Prediction/reference ID mismatch: ' + label + '/' + key)
            values[label] = {name: archive[name].copy() for name in (branch, 'beta', 'availability')}
        value = values[label]
        if (value[branch].shape != (n, 99) or value['beta'].shape != (n, 10)
                or value['availability'].shape != (n, 17)
                or not np.isfinite(value[branch]).all() or not np.isfinite(value['beta']).all()
                or not value['availability'].all()):
            raise ValueError('Missing/nonfinite eligible prediction: ' + label + '/' + key)
    return entry, manifest, source, values, results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'source-root', 'evaluation-root', 'mano-models', 'output'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--cpu-threads', type=int, default=2)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--decode-batch-size', type=int, default=512)
    parser.add_argument('--geometry-grouping', choices=('shard', 'original-clips32'), default='shard',
                        help='Decode full view batches or strictly replay original32-frame clip boundaries.')
    parser.add_argument('--chunk-length', type=int, default=100)
    parser.add_argument('--near-zero-path-m', type=float, default=.001)
    parser.add_argument('--official-eval-utils', type=Path)
    parser.add_argument('--max-shards', type=int, default=0,
                        help='Separate-output smoke: first N nonempty preselected shards; zero means full1280.')
    parser.add_argument('--dump-audit-fixture', type=Path,
                        help='Smoke only: NPZ of actual decoded continuous runs for independent checking.')
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--self-test-only', action='store_true')
    args = parser.parse_args()
    if not Path('/sciclone/data10/jchen57').is_dir():
        parser.error('Numerical processing is restricted to the authorized compute Pod')
    if args.output is None:
        parser.error('--output is required')
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError('Refusing to replace existing report: ' + str(args.output))
    if (args.cpu_threads < 1 or not 1 <= args.workers <= 16 or args.decode_batch_size < 1
            or args.chunk_length < 1 or args.near_zero_path_m <= 0 or args.max_shards < 0):
        parser.error('Invalid execution parameter')
    if args.dump_audit_fixture is not None and not args.max_shards:
        parser.error('--dump-audit-fixture is restricted to explicitly named smoke runs')
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ.setdefault(name, str(args.cpu_threads))
    if args.self_test_only:
        write(args.output, self_test(args.official_eval_utils))
        return
    if any(getattr(args, name) is None for name in ('root', 'source_root', 'evaluation_root', 'mano_models')):
        parser.error('--root, --source-root, --evaluation-root and --mano-models are required')
    args.root = args.root.resolve()
    for name in ('source_root', 'evaluation_root', 'mano_models', 'output'):
        path = getattr(args, name).resolve()
        if not path.is_relative_to(args.root):
            parser.error(name + ' must be inside root')
        setattr(args, name, path)
    import numpy as np
    import torch
    from dexycb_test_metrics import dex_joints
    from handloop.visual.codec import MANOStateCodec

    torch.set_num_threads(args.cpu_threads)
    if torch.device(args.device).type == 'cuda':
        torch.cuda.set_device(args.device)
        # The completed full-test evaluator enabled TF32 before MANO metric
        # decoding. Keep that precision convention explicit: disabling it
        # changes PA by about0.00026mm on the fixed8-shard probe.
        torch.backends.cuda.matmul.allow_tf32 = True
    started = time.monotonic()
    index_path = args.source_root / 'shard_index.json'
    index = read(index_path)
    if (index['phase'] != 'complete' or index['expected_shards'] != 1280
            or index['completed_shards'] != 1280 or len(index['shards']) != 1280
            or index['frames'] != 78768 or not index['eval_only'] or index['training_allowed']):
        raise ValueError('Require immutable completed full1280-shard/78768-frame s0 test')
    keys = [entry['key'] for entry in index['shards']]
    if len(set(keys)) != len(keys) or sum(entry['frames'] for entry in index['shards']) != index['frames']:
        raise ValueError('Duplicate or inconsistent source index')
    entries = index['shards']
    if args.max_shards:
        entries = [entry for entry in entries if entry['frames']][:args.max_shards]
        if not entries:
            raise ValueError('No nonempty shard available for smoke')
    expected_frames = sum(entry['frames'] for entry in entries)
    freeze_path = args.evaluation_root / 'frozen_models.json'
    frozen, freeze_sha = read(freeze_path), sha(freeze_path)
    identities, existing_reports = {}, {}
    for label in BRANCHES:
        report_path = args.evaluation_root / label / 'report.json'
        report = read(report_path)
        identity = report['identity']
        if (report['phase'] != 'complete' or report['completed_shards'] != 1280
                or report['selection_sha256'] != index['selection_sha256']
                or identity['label'] != label or identity['checkpoint_sha256'] != frozen['models'][label]['sha256']
                or identity['freeze_manifest_sha256'] != freeze_sha
                or Path(identity['source_root']).resolve() != args.source_root
                or identity['clip_length'] != 32 or identity['smoke_shards'] != 0):
            raise ValueError('Frozen model/report identity mismatch: ' + label)
        identities[label], existing_reports[label] = identity, report
    selection = read(args.source_root / 'selection.json')
    selection_digest = hashlib.sha256(json.dumps(selection, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    if selection_digest != index['selection_sha256']:
        raise ValueError('Source selection canonical checksum differs from shard index')
    assets = {side: sha(args.mano_models / ('MANO_' + side.upper() + '.pkl')) for side in ('left', 'right')}
    if any(assets[side] != selection['mano_decoder']['assets'][side]['sha256'] for side in assets):
        raise ValueError('MANO asset checksum differs from audited source')
    prediction_codec = MANOStateCodec(args.mano_models / 'MANO_RIGHT.pkl',
        args.mano_models / 'MANO_LEFT.pkl', left_shapedirs_fix=True).to(args.device).eval()
    reference_codec = MANOStateCodec(args.mano_models / 'MANO_RIGHT.pkl',
        args.mano_models / 'MANO_LEFT.pkl', left_shapedirs_fix=False).to(args.device).eval()
    scopes = {name: {group: {label: {} for label in BRANCHES} for group in ('all', 'left', 'right')}
              for name in ('all_contiguous_runs', 'hawor_min16_contiguous_runs')}
    coverage = {'completed_shards': 0, 'empty_shards': 0, 'frames': 0, 'clips32': 0,
                'left_frames': 0, 'right_frames': 0, 'run_length_histogram': Counter()}
    coverage_digest = hashlib.sha256()
    audit_rows = []
    fixture = {}
    fixture_segments = []
    check_results = self_test(args.official_eval_utils) if args.self_test else None
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        iterator = iter(entries)
        pending = deque()
        def submit_next():
            entry = next(iterator, None)
            if entry is not None:
                pending.append(executor.submit(load_shard, entry, args.source_root,
                    args.evaluation_root, identities, index['selection_sha256']))
        for _ in range(args.workers * 2):
            submit_next()
        while pending:
            entry, manifest, source, values, results = pending.popleft().result()
            submit_next()
            coverage['completed_shards'] += 1
            coverage['clips32'] += results['baseline']['clips']
            if any(result['clips'] != results['baseline']['clips'] for result in results.values()):
                raise ValueError('Model clip counts disagree')
            coverage_digest.update((entry['key'] + '\0').encode())
            coverage_digest.update(results['baseline']['frame_ids_sha256'].encode())
            audit_rows.append({'key': entry['key'], 'frames': entry['frames'],
                'source_manifest_sha256': entry['manifest_sha256'],
                'frame_ids_sha256': results['baseline']['frame_ids_sha256'],
                'predictions_sha256': {label: result['predictions_sha256'] for label, result in results.items()}})
            if source is None:
                coverage['empty_shards'] += 1
                continue
            n = len(source['frame_ids'])
            coverage['frames'] += n
            side_name = 'right' if source['handedness'][0] else 'left'
            coverage[side_name + '_frames'] += n
            run_ranges = runs_from_ids(source['frame_ids'])
            decode_ranges = []
            geometry_units = ([(0, n)] if args.geometry_grouping == 'shard' else
                              [(a, min(a + 32, high)) for low, high in run_ranges
                               for a in range(low, high, 32)])
            for low, high in geometry_units:
                decode_ranges.extend((a, min(a + args.decode_batch_size, high))
                                     for a in range(low, high, args.decode_batch_size))
            coverage['run_length_histogram'].update(str(b - a) for a, b in run_ranges)
            expected_clips = sum((b - a + 31) // 32 for a, b in run_ranges)
            if expected_clips != results['baseline']['clips']:
                raise ValueError('Reconstructed temporal reset differs from inference windows')
            target = source['joint_3d'].astype(np.float64)
            side = torch.as_tensor(source['handedness'].astype(bool), device=args.device)
            gt_beta = source['beta']
            if gt_beta.shape == (10,):
                gt_beta = np.broadcast_to(gt_beta, (n, 10)).copy()
            if gt_beta.shape != (n, 10):
                raise ValueError('Invalid reference beta shape')
            with torch.inference_mode():
                gt_transl = reference_codec.to_mano(torch.as_tensor(source['target_world'], device=args.device),
                    torch.as_tensor(gt_beta, device=args.device), side)['transl'].cpu().numpy().astype(np.float64)
            for label, branch in BRANCHES.items():
                value = values[label]
                joint_pieces, transl_pieces = [], []
                for a, b in decode_ranges:
                    state = torch.as_tensor(value[branch][a:b], device=args.device)
                    beta = torch.as_tensor(value['beta'][a:b], device=args.device)
                    with torch.inference_mode():
                        joint_pieces.append(dex_joints(prediction_codec.geometry(state, beta, side[a:b]), side[a:b]).cpu().numpy())
                        transl_pieces.append(prediction_codec.to_mano(state, beta, side[a:b])['transl'].cpu().numpy())
                predicted = np.concatenate(joint_pieces).astype(np.float64)
                pred_transl = np.concatenate(transl_pieces).astype(np.float64)
                if not np.isfinite(predicted).all() or not np.isfinite(pred_transl).all():
                    raise FloatingPointError('Nonfinite decoded prediction')
                if args.dump_audit_fixture is not None:
                    for a, b in run_ranges:
                        # One segment per model/run, enough source diversity for
                        # numerical oracles while the fixture remains small.
                        if len(fixture_segments) >= 24:
                            break
                        segment = 'segment_' + str(len(fixture_segments)).zfill(3)
                        fixture[segment + '_gt_joints'] = target[a:b]
                        fixture[segment + '_pred_joints'] = predicted[a:b]
                        fixture[segment + '_gt_trans'] = gt_transl[a:b]
                        fixture[segment + '_pred_trans'] = pred_transl[a:b]
                        fixture[segment + '_frame_ids'] = source['frame_ids'][a:b]
                        fixture_segments.append({'prefix': segment, 'key': entry['key'], 'model': label,
                                                 'frames': b - a, 'side': side_name})
                for a, b in run_ranges:
                    for scope in ('all_contiguous_runs', 'hawor_min16_contiguous_runs'):
                        if scope.startswith('hawor') and b - a < 16:
                            continue
                        for group in ('all', side_name):
                            compute_run(scopes[scope][group][label], predicted[a:b], target[a:b],
                                pred_transl[a:b], gt_transl[a:b], chunk_length=args.chunk_length,
                                near_zero_path_m=args.near_zero_path_m)
            if coverage['completed_shards'] % 80 == 0:
                update = {'phase': 'running', 'completed_shards': coverage['completed_shards'],
                          'frames': coverage['frames'], 'elapsed_seconds': time.monotonic() - started}
                print(json.dumps(update), flush=True)
                write(args.output.with_suffix('.progress.json'), update)
    if (coverage['frames'] != expected_frames or coverage['completed_shards'] != len(entries)
            or (not args.max_shards and coverage['clips32'] != 2992)):
        raise ValueError('Final whole-test coverage mismatch')
    crosscheck = {}
    for label, branch in BRANCHES.items():
        summary = existing_reports[label]['metrics']['all']['branches'][branch]['metrics']
        if args.max_shards:
            # Compare the same selected shard reductions, never a smoke subset
            # against the full-test mean.
            summary = {}
            for row in audit_rows:
                prior = read(args.evaluation_root / label / 'shards' / row['key'] / 'result.json')
                for name, value in prior['summary']['all']['branches'][branch]['metrics'].items():
                    total = summary.setdefault(name, {'mean': 0., 'count': 0})
                    total['mean'] += value['mean'] * value['count']
                    total['count'] += value['count']
            for value in summary.values():
                value['mean'] /= value['count']
        calculated = scopes['all_contiguous_runs']['all'][label]['metrics']
        checks = {}
        for new, old in (('pa_freihand_o3_mpjpe21_mm', 'pa_mpjpe21_mm'),
                         ('pa_hawor_so3_mpjpe21_mm', 'pa_proper_mpjpe21_mm'),
                         ('absolute_mpjpe21_mm', 'mpjpe21_mm'),
                         ('wrist_aligned_mpjpe21_mm', 'wrist_aligned_mpjpe21_mm')):
            sample = calculated[new]['frame_weighted']
            actual, expected = sample['sum'] / sample['count'], summary[old]['mean']
            tolerance = 1e-4 if 'absolute' in new or new.startswith('wrist') else 1e-5
            difference = actual - expected
            checks[new] = {'recomputed': actual, 'existing': expected, 'difference': difference,
                           'tolerance_mm': tolerance, 'count': sample['count'], 'passed': abs(difference) < tolerance}
            if sample['count'] != expected_frames or not checks[new]['passed']:
                raise ValueError('Existing geometry crosscheck failed: ' + label + '/' + new + ': ' + str(checks[new]))
        crosscheck[label] = checks
    report = {'schema': 'handloop.saved_dexycb_world_metrics.v1',
        'phase': 'smoke_complete' if args.max_shards else 'complete',
        'max_shards': args.max_shards,
        'completed_utc': datetime.now(timezone.utc).isoformat(), 'elapsed_seconds': time.monotonic() - started,
        'script_sha256': sha(__file__), 'source_index_sha256': sha(index_path),
        'source_root': str(args.source_root), 'evaluation_root': str(args.evaluation_root),
        'source_selection_sha256': index['selection_sha256'], 'freeze_manifest_sha256': freeze_sha,
        'formal_branches': BRANCHES, 'identities': identities, 'mano_assets_sha256': assets,
        'prediction_left_shapedirs_fix': True, 'reference_left_shapedirs_fix': False,
        'geometry_execution': {'cuda_matmul_allow_tf32': bool(torch.backends.cuda.matmul.allow_tf32),
            'device': args.device, 'grouping': args.geometry_grouping,
            'decode_batch_size_upper_cap': args.decode_batch_size,
            'precision_rationale': 'Match completed full-test MANO geometry precision; fixed8-shard probe distinguished TF32 convention from small batch-size roundoff.'},
        'predicted_beta_policy': 'Each branch decodes its own saved predicted beta; GT beta only restores GT native transl.',
        'coverage': coverage, 'coverage_key_and_frame_hash_sha256': coverage_digest.hexdigest(),
        'protocol': {'source': 'HaWoR ' + OFFICIAL_COMMIT + ' lib/eval_utils/eval_utils.py and scripts/scripts_eval/eval_hawor_hot3d.py',
            'joint_set': 'DexYCB supplied Dex21, 21 landmarks; Dex tip vertices differ from official HOT3D OpenPose21 tips.',
            'world_gauge': 'Each calibrated static camera defines an independent identity world gauge; not HOT3D SLAM evaluation.',
            'fps': 30, 'timestamps': 'frame_id/nominal30Hz; not measured capture timestamps',
            'continuous_runs': 'Split on diff(frame_ids)!=1; never bridge omitted invalid-GT frames.',
            'w_mpjpe': 'Proper SO(3)+positive scale+translation fitted to first min(2,T) complete frames of each block.',
            'wa_mpjpe': 'Proper SO(3)+positive scale+translation fitted jointly to all frame/joint points in each block.',
            'alignment_chunk_frames': args.chunk_length, 'alignment_tail': 'Retained, including a one-frame tail.',
            'pa_hawor': 'Per-frame proper SO(3)+scale+translation; separate from existing FreiHAND unrestricted O(3).',
            'accel': 'Absolute joints, ||pred_second_difference-GT_second_difference||, mean21, multiplied by30^2; official additionally trims first/last valid second differences, leaving T-4 centers.',
            'reset_policy': 'Primary includes independent32-frame inference resets; diagnostic excludes centers at run-relative indices32k-1 and32k.',
            'rte': 'Per continuous run native MANO transl=p3-rest_wrist(own beta); rigid SE(3), scale fixed1; each frame error divided by entire GT transl path length, times100.',
            'rte_zero_path': 'GT path<=1e-12m explicitly undefined and counted; no epsilon denominator.',
            'rte_near_zero_threshold_m': args.near_zero_path_m,
            'weighting': 'All frame/center weighted, equal100-frame-block weighted, and equal-continuous-run weighted reported; Accel and RTE blocks equal whole runs.',
            'scopes': 'Full78768-frame coverage plus a separately named official-min16-run sensitivity subset.'},
        'existing_geometry_crosscheck': crosscheck, 'synthetic_acceptance': check_results,
        'scopes': {scope: {group: {label: finalize(bucket) for label, bucket in models.items()}
                          for group, models in groups.items()} for scope, groups in scopes.items()}}
    audit_path = args.output.with_suffix('.coverage_audit.json')
    write(audit_path, {'schema': 'handloop.saved_world_coverage_audit.v1', 'rows': audit_rows})
    report['coverage_audit'] = {'path': str(audit_path), 'sha256': sha(audit_path)}
    if args.dump_audit_fixture is not None:
        fixture_path = args.dump_audit_fixture.resolve()
        if not fixture_path.is_relative_to(args.root):
            raise ValueError('Audit fixture must remain inside remote root')
        fixture_path.parent.mkdir(parents=True, exist_ok=True)
        if fixture_path.exists():
            raise FileExistsError('Refusing to replace existing fixture')
        with fixture_path.open('wb') as stream:
            np.savez(stream, **fixture)
        report['audit_fixture'] = {'path': str(fixture_path), 'sha256': sha(fixture_path),
                                   'segments': fixture_segments}
    write(args.output, report)
    print(json.dumps({'phase': 'complete', 'frames': coverage['frames'], 'report': str(args.output),
                      'elapsed_seconds': report['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
