#!/usr/bin/env python3
"""Remote-only independent numerical acceptance for full DexYCB evaluation.

Run in the authorized compute Pod, not on the writing laptop. This checker
does not train a model or access test images. It checks alignment, pooled PCK,
stream/resume/merge, masks, own-beta decoding and actual MANO conventions.
An optional already-prepared source reference adds real Dex21/FK acceptance.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mano-models', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--official-eval-util', type=Path,
                        help='Reviewed local official FreiHAND utils/eval_util.py')
    parser.add_argument('--reference', type=Path, action='append', default=[],
                        help='Optional actual full-test reference NPZ; repeatable')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if not Path('/sciclone/home/jchen57').is_dir():
        parser.error('Run numerical/MANO acceptance only in the authorized remote ds Pod')
    if args.output.exists():
        raise FileExistsError('Use a fresh acceptance report path')

    import numpy as np
    from scipy.linalg import orthogonal_procrustes
    from scipy.spatial.transform import Rotation
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
    from handloop import geometry as g
    from handloop.visual.codec import MANOStateCodec
    import dexycb_test_metrics as helper

    torch.set_num_threads(2)
    torch.manual_seed(20261016)
    rng = np.random.default_rng(20261016)
    device = torch.device(args.device)
    pred_codec = MANOStateCodec(args.mano_models / 'MANO_RIGHT.pkl',
                               args.mano_models / 'MANO_LEFT.pkl', left_shapedirs_fix=True).to(device)
    ref_codec = MANOStateCodec(args.mano_models / 'MANO_RIGHT.pkl',
                              args.mano_models / 'MANO_LEFT.pkl', left_shapedirs_fix=False).to(device)
    checks = {}
    report = {'schema': 'handloop.dexycb_full_test_acceptance.v1',
              'started_utc': datetime.now(timezone.utc).isoformat(),
              'helper_sha256': sha256(helper.__file__), 'checker_sha256': sha256(__file__),
              'mano_sha256': {side: sha256(args.mano_models / f'MANO_{side}.pkl')
                             for side in ('LEFT', 'RIGHT')},
              'device': str(device), 'checks': checks, 'passed': False}
    report['fixture_decode_policy'] = (
        'Fixed one-pose FP32 MANO decoding for fixture references, independent '
        'oracles and accumulators. Resume/merge tests therefore measure the '
        'reduction itself, without conflating batch-shape-dependent GEMM roundoff '
        'with exact-zero PCK at threshold0. Real-source FK keeps its 1mm bound.')

    def check(name, test, **evidence):
        if not bool(test):
            raise AssertionError(name + ': ' + repr(evidence))
        checks[name] = {'passed': True, **evidence}

    def new():
        return helper.StreamingDexYCBMetrics(pred_codec, ref_codec, device=device,
                                            batch_size=1, group_by=('hand', 'subject', 'view', 'recording'))

    def decode_fixture(codec, states, shapes, handedness):
        pieces = [codec.geometry(states[i:i + 1], shapes[i:i + 1], handedness[i:i + 1])
                  for i in range(len(states))]
        return {name: torch.cat([value[name] for value in pieces]) for name in ('joints', 'vertices')}

    def metric(snapshot, name, branch='raw'):
        return snapshot['all']['branches'][branch]['metrics'][name]

    def assert_same_numeric(a, b, tolerance=1e-9, path='root'):
        if isinstance(a, dict):
            if set(a) != set(b):
                raise AssertionError('Keys differ at ' + path)
            for name in a:
                assert_same_numeric(a[name], b[name], tolerance, path + '.' + str(name))
        elif isinstance(a, list):
            if len(a) != len(b):
                raise AssertionError('Lengths differ at ' + path)
            for i, (x, y) in enumerate(zip(a, b)):
                assert_same_numeric(x, y, tolerance, path + '[' + str(i) + ']')
        elif isinstance(a, (int, float)) and not isinstance(a, bool):
            if not np.isclose(a, b, rtol=0, atol=tolerance):
                raise AssertionError(f'Value differs at {path}: {a} != {b}')
        elif a != b:
            raise AssertionError(f'Value differs at {path}: {a!r} != {b!r}')

    def official_align(target, predicted):
        # Independent NumPy/SciPy implementation of FreiHAND align_w_scale.
        first, second = target.astype(np.float64).copy(), predicted.astype(np.float64).copy()
        t1, t2 = first.mean(0), second.mean(0)
        first -= t1
        second -= t2
        s1, s2 = np.linalg.norm(first) + 1e-8, np.linalg.norm(second) + 1e-8
        first /= s1
        second /= s2
        rotation, scale = orthogonal_procrustes(first, second)
        aligned = second @ rotation.T * scale * s1 + t1
        return np.linalg.norm(aligned - target, axis=-1)

    try:
        thresholds = np.linspace(0., 50., 100)
        check('thresholds_exact_official_linspace', np.array_equal(helper.THRESHOLDS_MM, thresholds))
        cloud = rng.normal(size=(5, 21, 3)) * .035
        # Keep the synthetic similarity in float64. The production geometry
        # helpers intentionally cast rotations to float32; using those here
        # introduces a real ~1e-6 mm residual from non-orthogonality and would
        # make a strict double-precision invariance test test its own fixture.
        rotation = Rotation.from_rotvec(np.array([.3, -.2, .5], dtype=np.float64)).as_matrix()
        orthogonality = float(np.max(np.abs(rotation.T @ rotation - np.eye(3))))
        check('synthetic_similarity_is_double_precision_so3', orthogonality < 1e-13
              and abs(np.linalg.det(rotation) - 1.) < 1e-13,
              orthogonality_error=orthogonality, determinant=float(np.linalg.det(rotation)))
        transformed = 1.7 * (cloud @ rotation) + np.array([.2, -.1, .5])
        transformed[1, :, 0] *= -1  # A negative determinant must remain auditable.
        measured, alignment_info = helper.pa_joint_errors_mm(
            torch.from_numpy(transformed), torch.from_numpy(cloud), return_info=True)
        independent = np.stack([official_align(y * 1000, x * 1000)
                                for x, y in zip(transformed, cloud)])
        max_pa_delta = float(np.max(np.abs(measured.numpy() - independent)))
        check('pa_matches_independent_freihand_formula', max_pa_delta < 1e-5,
              max_error_mm=max_pa_delta)
        reflection = np.asarray(alignment_info['reflection'].cpu())
        proper = alignment_info['proper_errors_mm'].cpu().numpy()
        check('reflection_primary_and_proper_diagnostic', bool(reflection[1])
              and measured[1].max().item() < 1e-7 and proper[1].mean() > 1.,
              primary_reflected_mean_mm=float(measured[1].mean()),
              proper_reflected_mean_mm=float(proper[1].mean()), reflection_count=int(reflection.sum()))
        check('pa_scale_translation_rotation_invariance', measured.max().item() < 1e-7,
              max_error_mm=float(measured.max()))
        try:
            helper.pa_joint_errors_mm(torch.zeros(1, 21, 3), torch.zeros(1, 21, 3))
        except ValueError:
            checks['degenerate_pa_raises'] = {'passed': True}
        else:
            raise AssertionError('degenerate_pa_raises')

        # Test threshold inclusivity and per-joint pooling independently of FK.
        errors = np.resize(np.array([0., 5., 10., 20., 30., 49.99, 50., 50.0001, 100., thresholds[25]]), (11, 21))
        hits = (errors[..., None] <= thresholds).sum(0)
        state = new().state_dict()
        bucket = state['groups']['all']['all']
        bucket['source_frames'] = bucket['complete_reference_states'] = len(errors)
        bucket['branches']['boundary'] = {'scalars': {}, 'pck': {'absolute': {
            'joint_counts': [len(errors)] * 21, 'hits': hits.tolist(),
            'pck_at_hits': {str(t): int((errors <= t).sum()) for t in (5, 10, 20, 30, 50)}}}}
        synthetic = new().load_state_dict(state).snapshot()['all']['branches']['boundary']['pck_auc']['absolute']
        expected_curve = (errors[..., None] <= thresholds).mean(0).mean(0)
        expected_auc = float(((expected_curve[1:] + expected_curve[:-1]) * .5
                              * np.diff(thresholds)).sum() / 50.)
        check('pooled_pck_exact_threshold_inclusive', np.array_equal(synthetic['pck_curve'], expected_curve)
              and abs(synthetic['auc'] - expected_auc) < 1e-15, auc=synthetic['auc'])
        if args.official_eval_util:
            # NumPy 2 removed the deprecated spelling used by the reviewed
            # historical EvalUtil. Alias only within this checker process;
            # the official file and the production metric helper stay intact.
            alias_added = not hasattr(np, 'trapz')
            if alias_added:
                np.trapz = np.trapezoid
            report['official_evalutil_numpy_compatibility'] = {
                'numpy_version': np.__version__, 'trapz_alias_added': alias_added,
                'alias': 'numpy.trapz = numpy.trapezoid' if alias_added else None,
                'official_source_modified': False}
            spec = importlib.util.spec_from_file_location('reviewed_official_eval_util', args.official_eval_util)
            official = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(official)
            evaluator = official.EvalUtil(num_kp=21)
            for distances in errors:
                evaluator.feed(np.zeros((21, 3)), np.ones(21, dtype=bool),
                               np.column_stack((distances, np.zeros((21, 2)))))
            measures = evaluator.get_measures(0., 50., 100)
            official_auc, official_curve, official_thresholds = measures[2:5]
            check('actual_official_evalutil_boundary_agreement',
                  abs(synthetic['auc'] - float(official_auc)) < 1e-15
                  and np.array_equal(synthetic['pck_curve'], official_curve)
                  and np.array_equal(thresholds, official_thresholds),
                  official_sha256=sha256(args.official_eval_util), auc=float(official_auc))
        else:
            report['official_evalutil_comparison'] = 'not_requested; independent formula only'

        n = 8
        position = torch.zeros(n, 3, device=device)
        position[:, 0] = torch.arange(n, device=device) * .015
        position[:, 2] = .6
        rotations = g.axis_angle_to_matrix(torch.tensor(rng.normal(size=(n, 16, 3)) * .15,
                                                        dtype=torch.float32, device=device))
        target = g.state_from_rotmat(position, rotations)
        right = torch.ones(n, dtype=torch.bool, device=device)
        beta = torch.zeros(n, 10, device=device)
        target_valid = torch.ones(n, 17, dtype=torch.bool, device=device)
        joints = helper.dex_joints(decode_fixture(ref_codec, target, beta, right), right).detach()
        metadata = {'subject': 'fixture_subject', 'view': 'fixture_view', 'recording': 'fixture_recording'}
        exact = new()
        exact.update({'raw': target}, beta, target, beta, joints, right, target_valid, group=metadata)
        exact_report = exact.snapshot()
        check('actual_mano_zero_error', metric(exact_report, 'mpjpe21_mm')['mean'] < 1e-5
              and metric(exact_report, 'mpvpe778_mm')['mean'] < 1e-5
              and exact_report['all']['branches']['raw']['pck_auc']['absolute']['auc'] == 1.,
              mpjpe21_mm=metric(exact_report, 'mpjpe21_mm')['mean'])
        shifted = target.clone()
        shifted[:, 0] += .01
        translated = new()
        translated.update({'raw': shifted}, beta, target, beta, joints, right, target_valid, group=metadata)
        translated_report = translated.snapshot()
        check('actual_mano_10mm_translation', abs(metric(translated_report, 'mpjpe21_mm')['mean'] - 10.) < .001
              and metric(translated_report, 'wrist_aligned_mpjpe21_mm')['mean'] < .001
              and metric(translated_report, 'pa_mpjpe21_mm')['mean'] < .001,
              absolute_mm=metric(translated_report, 'mpjpe21_mm')['mean'],
              wrist_aligned_mm=metric(translated_report, 'wrist_aligned_mpjpe21_mm')['mean'],
              pa_mm=metric(translated_report, 'pa_mpjpe21_mm')['mean'])

        # Deliberately invalid labels/predictions contain NaNs: they cannot be
        # converted into successful zeros or poison other eligible samples.
        masked_target, masked_joints, masked_beta = target.clone(), joints.clone(), beta.clone()
        masked_prediction, valid = shifted.clone(), target_valid.clone()
        valid[1] = False
        masked_target[1] = masked_joints[1] = masked_beta[1] = torch.nan
        masked_prediction[1] = torch.nan
        frame_valid = torch.ones(n, device=device, dtype=torch.bool)
        frame_valid[2] = False
        masked_target[2] = masked_joints[2] = masked_beta[2] = torch.nan
        masked_prediction[2] = torch.nan
        available = torch.ones(n, 17, dtype=torch.bool, device=device)
        available[3, 2] = False
        # An unavailable rotation can carry NaN; available groups stay valid.
        masked_prediction[3, 9:15] = torch.nan
        masked = new()
        arguments = (masked_prediction, beta, masked_target, masked_beta, masked_joints, right, valid, available, frame_valid)

        def update_take(accumulator, indices):
            p, b, t, tb, j, h, tv, av, fv = (value[indices] for value in arguments)
            accumulator.update({'raw': p}, b, t, tb, j, h, tv, availability=av,
                               frame_valid=fv, group=metadata)

        update_take(masked, slice(None))
        masked_report = masked.snapshot()
        check('mask_counts_and_nonfinite_exclusion',
              masked_report['all']['source_frames'] == 7
              and masked_report['all']['complete_reference_states'] == 6
              and metric(masked_report, 'mpjpe21_mm')['count'] == 5
              and metric(masked_report, 'wrist_mm')['count'] == 6
              and metric(masked_report, 'finger_deg')['count'] == 6 * 15 - 1,
              source_frames=masked_report['all']['source_frames'],
              complete_states=masked_report['all']['complete_reference_states'],
              geometry_count=metric(masked_report, 'mpjpe21_mm')['count'])
        first, second = new(), new()
        update_take(first, slice(0, 4))
        restored = new().load_state_dict(json.loads(json.dumps(first.state_dict(), allow_nan=False)))
        update_take(restored, slice(4, None))
        update_take(second, slice(4, None))
        merged = new().load_state_dict(first.state_dict()).merge_state_dict(second.state_dict())
        # Updates differ only by batching, so compare every actual statistic.
        for accumulated in (restored, merged):
            snapshot = accumulated.snapshot()
            snapshot['updates'] = masked_report['updates']
            assert_same_numeric(masked_report, snapshot, tolerance=1e-7)
        checks['resume_merge_matches_pooled'] = {'passed': True}

        mixed_right = torch.tensor([True, False] * 4, device=device)
        source_beta = torch.tensor(rng.normal(size=(n, 10)) * .8, dtype=torch.float32, device=device)
        own_beta = torch.tensor(rng.normal(size=(n, 10)) * .6, dtype=torch.float32, device=device)
        source_geometry = decode_fixture(ref_codec, target, source_beta, mixed_right)
        mixed_joints = helper.dex_joints(source_geometry, mixed_right).detach()
        own_geometry = decode_fixture(pred_codec, target, own_beta, mixed_right)
        direct = (helper.dex_joints(own_geometry, mixed_right) - mixed_joints).norm(dim=-1).double().mean().item() * 1000
        wrong = (helper.dex_joints(decode_fixture(pred_codec, target, source_beta, mixed_right), mixed_right)
                 - mixed_joints).norm(dim=-1).double().mean().item() * 1000
        own = new()
        own.update({'final': target}, own_beta, target, source_beta, mixed_joints,
                   mixed_right, target_valid, group=metadata)
        own_report = own.snapshot()
        own_measured = metric(own_report, 'mpjpe21_mm', 'final')['mean']
        check('prediction_own_beta_and_left_shape_convention', abs(own_measured - direct) < 1e-5
              and abs(direct - wrong) > 1e-3
              and own_report['groups']['hand']['left']['branches']['final']['metrics']['mpjpe21_mm']['count'] == 4,
              accumulated_mm=own_measured, direct_mm=direct, wrong_gt_beta_mm=wrong)

        joint_valid = torch.ones(n, 21, dtype=torch.bool, device=device)
        joint_valid[0, 4] = False
        try:
            new().update({'raw': target}, beta, target, beta, joints, right,
                         target_valid, joint_valid=joint_valid)
        except ValueError:
            checks['complete_state_partial_joint_mask_raises'] = {'passed': True}
        else:
            raise AssertionError('complete_state_partial_joint_mask_raises')

        real_results = []
        for path in args.reference:
            with np.load(path, allow_pickle=False) as archive:
                arrays = {key: archive[key].copy() for key in archive.files}
            take = np.flatnonzero(arrays['target_valid'].all(-1))
            take = take[np.linspace(0, len(take) - 1, min(12, len(take))).astype(int)] if len(take) else take
            if not len(take):
                raise ValueError('Real reference has no complete states: ' + str(path))
            source_beta = arrays['beta']
            if source_beta.shape != (10,):
                source_beta = source_beta[take]
            actual = new()
            actual.update({'raw': arrays['target_world'][take]},
                          np.zeros((len(take), 10), dtype=np.float32),
                          arrays['target_world'][take], source_beta, arrays['joint_3d'][take],
                          arrays['handedness'][take], arrays['target_valid'][take],
                          joint_valid=arrays.get('joint_3d_valid', np.ones((len(arrays['target_world']), 21), dtype=bool))[take])
            actual_report = actual.snapshot()
            real_results.append({'path': str(path.resolve()), 'sha256': sha256(path),
                                 'selected_indices': take.tolist(), 'fk': actual_report['reference_fk_audit'],
                                 'complete_states': actual_report['all']['complete_reference_states']})
        report['real_reference_checks'] = real_results
        report['passed'] = True
    except Exception as error:
        report.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report['completed_utc'] = datetime.now(timezone.utc).isoformat()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + '.tmp')
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')
        temporary.replace(args.output)
        print(json.dumps({'passed': report['passed'], 'checks': len(checks),
                          'report': str(args.output.resolve())}), flush=True)


if __name__ == '__main__':
    main()
