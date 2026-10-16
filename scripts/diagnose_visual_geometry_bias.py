#!/usr/bin/env python3
"""Remote-only, read-only decomposition of saved visual reconstruction errors.

No HaWoR/TWM forward, training, checkpoint selection, or official metric change
occurs here. Saved prediction beta is used by the normal prediction decoder.
Explicit oracle controls use source states/beta *only after prediction*.

Example (inside the compute Pod):
  python scripts/diagnose_visual_geometry_bias.py \
    --predictions baseline=/remote/evaluation/test_baseline \
    --predictions best=/remote/evaluation/test_best \
    --predictions last=/remote/evaluation/test_last \
    --source-manifest /remote/sources/source_manifest.json \
    --mano-models /remote/licensed/MANO --device cuda:0 \
    --output /remote/evaluation/test_geometry_bias.json

The companion long-form per-frame CSV stays remote. JSON contains only grouped
summaries and recording comparisons. Camera x/y/z are the saved original fixed
camera axes; z is optical depth. This is not a moving-camera/world benchmark.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

from evaluate_visual_geometry import DEX_ORDER, parse_input, prediction_paths, save_json


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Frame-count-weighted arithmetic means; square roots are applied last."""
    if not rows:
        return {'valid_states': 0}
    names = [name for name, value in rows[0].items()
             if isinstance(value, (float, int)) and name not in
             ('clip_index', 'reference_index', 'frame_id')]
    result = {'valid_states': len(rows), 'recordings': len({r['recording'] for r in rows}),
              **{name: sum(r[name] for r in rows) / len(rows) for name in names}}
    joint_names = [name for name in names if name.startswith('perjoint')]
    result['relative_error_by_Dex21_joint'] = [
        {'joint_index': index,
         **{suffix: result.pop(f'perjoint{index:02d}_{suffix}') for suffix in
            ('relative_x_mm', 'relative_y_mm', 'relative_z_mm', 'relative_norm_mm')}}
        for index in range(21)] if joint_names else []
    for axis in ('x', 'y', 'z'):
        result['wrist_' + axis + '_rmse_mm'] = math.sqrt(result['wrist_' + axis + '_sq_mm2'])
    for name in ('wrist', 'joint_absolute', 'joint_relative'):
        source = 'translation_mse_mm2' if name == 'wrist' else name + '_mse_mm2'
        result[name + '_rmse_mm'] = math.sqrt(max(result[source], 0.0))
    total = result['joint_absolute_mse_mm2']
    result['mse_decomposition_ratios'] = {
        key: result[key] / total if total else None for key in
        ('translation_mse_mm2', 'joint_relative_mse_mm2', 'cross_mse_mm2')}
    squared_group_bias = sum(result['wrist_' + axis + '_bias_mm'] ** 2 for axis in ('x', 'y', 'z'))
    result['wrist_bias_norm_mm'] = math.sqrt(squared_group_bias)
    result['oracle_wrist_rmse_after_group_constant_bias_removed_mm'] = math.sqrt(
        max(0.0, result['translation_mse_mm2'] - squared_group_bias))
    result['wrist_squared_group_bias_fraction_of_translation_MSE'] = (
        squared_group_bias / result['translation_mse_mm2'] if result['translation_mse_mm2'] else None)
    result['axis_convention'] = 'signed predicted-minus-reference; fixed camera x/y/z, z optical depth'
    return result


def grouped(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row[field]), []).append(row)
    return {key: summarize(value) for key, value in sorted(groups.items())}


def paired(rows: list[dict[str, Any]], before: str, after: str) -> dict[str, Any]:
    def key(row):
        return row['recording'], row['clip_index'], row['reference_index']
    left = {key(row): row for row in rows if row['branch'] == before}
    right = {key(row): row for row in rows if row['branch'] == after}
    matches = sorted(left.keys() & right.keys())
    result = {'before': before, 'after': after, 'paired_valid_states': len(matches)}
    for side in ('all', 'left', 'right'):
        take = [k for k in matches if side == 'all' or left[k]['handedness'] == side]
        if not take:
            result[side] = {'valid_states': 0}
            continue
        entry = {'valid_states': len(take)}
        for metric in ('wrist_mm', 'mpjpe21_mm', 'wrist_aligned_mpjpe21_mm',
                       'joint_absolute_mse_mm2', 'translation_mse_mm2',
                       'joint_relative_mse_mm2', 'cross_mse_mm2'):
            differences = [right[k][metric] - left[k][metric] for k in take]
            entry[metric + '_after_minus_before'] = sum(differences) / len(take)
            if metric.endswith('_mm'):
                entry[metric + '_worse_gt1mm_fraction'] = sum(v > 1 for v in differences) / len(take)
        result[side] = entry
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--predictions', action='append', type=parse_input, required=True)
    parser.add_argument('--source-manifest', type=Path, required=True)
    parser.add_argument('--mano-models', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--cpu-threads', type=int, default=2)
    args = parser.parse_args()
    if not Path('/sciclone/data10/jchen57').is_dir():
        raise RuntimeError('Numerical/model diagnostics must run inside the remote compute Pod')
    if args.batch_size < 1 or args.cpu_threads < 1:
        parser.error('Batch size and CPU threads must be positive')
    labels = [name for name, _ in args.predictions]
    if len(labels) != len(set(labels)):
        parser.error('Prediction labels must be unique')
    output = args.output.expanduser().resolve()
    csv_path = output.with_name(output.stem + '_per_frame.csv')
    if output.exists() or csv_path.exists():
        raise FileExistsError('Use a fresh diagnostic output path')
    output.parent.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
    from handloop.data.visual_clips import file_sha256
    from handloop.visual.codec import MANOStateCodec, mirror_rotations_x
    from handloop import geometry as g

    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    manifest_path = args.source_manifest.expanduser().resolve()
    manifest_sha = file_sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get('schema') != 'handloop.visual_feedback_sources.v1'
            or manifest.get('phase') != 'complete' or manifest.get('dataset') != 'dexycb'):
        raise ValueError('Expected a complete, audited DexYCB source manifest')
    records = {record['recording']: record for record in manifest['records']}
    if len(records) != len(manifest['records']):
        raise ValueError('Manifest recording identities must be unique')
    assets = {side: args.mano_models / f'MANO_{side}.pkl' for side in ('LEFT', 'RIGHT')}
    asset_sha = {side: file_sha256(path) for side, path in assets.items()}
    prediction_codec = MANOStateCodec(assets['RIGHT'], assets['LEFT'], left_shapedirs_fix=True).to(device)
    reference_codec = MANOStateCodec(assets['RIGHT'], assets['LEFT'], left_shapedirs_fix=False).to(device)

    def tensor(value, *, dtype=torch.float32):
        return torch.as_tensor(value, dtype=dtype, device=device)

    def dex_joints(geometry, right):
        tip_ids = torch.tensor([745, 317, 444, 556, 673], device=device).expand(len(right), -1).clone()
        tip_ids[~right.bool(), 2] = 445
        tips = geometry['vertices'].gather(1, tip_ids[..., None].expand(-1, -1, 3))
        return torch.cat((geometry['joints'], tips), 1)[:, DEX_ORDER]

    def decode(codec, state, beta, right):
        geometry = codec.geometry(state, beta, right)
        # J0 is an anatomical wrist in the 99D contract, irrespective of shape.
        maximum = float((geometry['joints'][:, 0] - state[:, :3]).abs().max())
        if maximum > 2e-6:
            raise ValueError(f'MANO J0/state identity failed: {maximum} m')
        return dex_joints(geometry, right)

    references, reference_audit = {}, {}

    def load_reference(recording):
        if recording in references:
            return references[recording]
        record = records[recording]
        if record.get('split') not in ('val', 'test'):
            raise ValueError('Only held-out val/test references are accepted')
        coordinate = record.get('coordinate_frame', '')
        if not (coordinate.startswith('fixed_camera_') and coordinate.endswith('_as_world_gauge')):
            raise ValueError('Only explicit fixed-camera identity world gauge is supported')
        path = (manifest_path.parent / record['reference_file']).resolve()
        if not path.is_relative_to(manifest_path.parent):
            raise ValueError('Reference path escapes its source directory')
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        n = int(record['frames'])
        required = {'target_world': (n, 99), 'target_valid': (n, 17),
                    'joint_3d': (n, 21, 3), 'frame_ids': (n,), 'timestamps': (n,),
                    'handedness': (n,), 'camera_to_world': (n, 4, 4)}
        for key, shape in required.items():
            if key not in arrays or arrays[key].shape != shape or not np.isfinite(arrays[key]).all():
                raise ValueError(f'Invalid source {recording}: {key}')
        if not np.allclose(arrays['camera_to_world'], np.eye(4), atol=1e-7, rtol=0):
            raise ValueError('Camera must coincide with saved world gauge for axis diagnostics')
        side = arrays['handedness']
        if not np.isin(side, [0, 1]).all() or not np.all(side == int(record['handedness'])):
            raise ValueError('Inconsistent manifest/reference handedness')
        beta = arrays.get('beta')
        if beta is None or beta.shape not in ((10,), (n, 10)) or not np.isfinite(beta).all():
            raise ValueError('Source beta must be finite10 or Nx10')
        if beta.shape == (10,):
            beta = np.broadcast_to(beta, (n, 10)).copy()
        arrays['beta'] = beta
        arrays['complete_valid'] = arrays['target_valid'].astype(bool).all(-1)
        take = np.flatnonzero(arrays['complete_valid'])
        if not len(take):
            raise ValueError('No complete reference state for FK audit')
        fk_errors, convention_errors, zero_errors, mirror_errors = [], [], [], []
        for begin in range(0, len(take), args.batch_size):
            indices = take[begin:begin + args.batch_size]
            state, source_beta = tensor(arrays['target_world'][indices]), tensor(beta[indices])
            right = tensor(side[indices], dtype=torch.bool)
            decoded = decode(reference_codec, state, source_beta, right)
            supplied = tensor(arrays['joint_3d'][indices])
            fk_errors.append((decoded - supplied).norm(dim=-1).detach().cpu())
            convention = decode(prediction_codec, state, source_beta, right)
            convention_errors.append((convention - decoded).norm(dim=-1).detach().cpu())
            zero = torch.zeros_like(source_beta)
            zero_errors.append((decode(prediction_codec, state, zero, right)
                                - decode(reference_codec, state, zero, right)).norm(dim=-1).detach().cpu())
            rotations = g.state_to_rotmat(state)
            mirrored_twice = mirror_rotations_x(mirror_rotations_x(rotations, right), right)
            mirror_errors.append((mirrored_twice - rotations).abs().detach().cpu().reshape(-1))
        fk = torch.cat(fk_errors)
        zero = torch.cat(zero_errors)
        convention = torch.cat(convention_errors)
        mirror = torch.cat(mirror_errors)
        entry = {'reference_file': str(path), 'reference_file_sha256': file_sha256(path),
                 'complete_states_audited': len(take), 'handedness': 'right' if side[0] else 'left',
                 'reference_fk_mean_mm': float(fk.mean()) * 1000,
                 'reference_fk_max_mm': float(fk.max()) * 1000,
                 'same_GT99D_same_numeric_sourcebeta_cross_codec_mpjpe_mm': float(convention.mean()) * 1000,
                 'zero_beta_cross_codec_max_mm': float(zero.max()) * 1000,
                 'mirror_twice_rotation_max_abs': float(mirror.max()),
                 'reference_fk_tolerance_mm': 1.0,
                 'zero_beta_cross_codec_tolerance_mm': 0.01}
        entry['passed'] = (entry['reference_fk_max_mm'] <= 1.0
                           and entry['zero_beta_cross_codec_max_mm'] <= 0.01
                           and entry['mirror_twice_rotation_max_abs'] <= 1e-7)
        reference_audit[recording] = entry
        if not entry['passed']:
            save_json(output.with_name(output.stem + '_audit_failed.json'),
                      {'reference_audit': reference_audit, 'source_manifest_sha256': manifest_sha})
            raise ValueError(f'Codec/source identity audit failed: {recording}')
        references[recording] = arrays
        return arrays

    rows, coverage, provenance, branch_order = [], {}, {}, {}
    valid_coverage = {}
    paired_protocol = None
    decomposition_max_abs_residual_mm2 = 0.0
    with torch.no_grad():
        for label, path in args.predictions:
            paths = prediction_paths(path)
            adjacent = path / 'provenance.json' if path.is_dir() else path.parent.parent / 'provenance.json'
            if path.is_dir() and path.name == 'clips':
                adjacent = path.parent / 'provenance.json'
            if not adjacent.is_file():
                raise ValueError('Require adjacent evaluation provenance, including MANO/source hashes')
            proof = json.loads(adjacent.read_text())
            if (proof.get('schema') != 'handloop.visual_joint_evaluation.v1'
                    or proof.get('source_manifest_sha256') != manifest_sha
                    or proof.get('mano_asset_sha256') != asset_sha):
                raise ValueError('Evaluation manifest/MANO identities differ')
            protocol = {key: proof.get(key) for key in
                        ('split', 'official_hawor_checkpoint_sha256', 'hawor_config_sha256',
                         'history_policy', 'input_contract', 'evaluated_indices')}
            if paired_protocol is None:
                paired_protocol = protocol
            elif protocol != paired_protocol:
                raise ValueError('Paired saved inputs used different evaluation/frontend protocols')
            provenance[label] = {'path': str(path), 'evaluation_provenance': str(adjacent),
                                 'evaluation_provenance_sha256': file_sha256(adjacent),
                                 **{key: proof.get(key) for key in
                                    ('mode', 'checkpoint', 'checkpoint_sha256', 'checkpoint_step',
                                     'evaluation_code_revision', 'evaluation_script_sha256', 'training_identity')},
                                 'clips': []}
            coverage[label], valid_coverage[label] = set(), set()
            expected_branches = None
            for prediction_path in paths:
                payload = torch.load(prediction_path, map_location='cpu', weights_only=False)
                if (payload.get('source_manifest_sha256') != manifest_sha
                        or not isinstance(payload.get('source_manifest'), str)
                        or Path(payload['source_manifest']).resolve() != manifest_path):
                    raise ValueError('Saved clip source identity differs from the supplied manifest')
                record = payload['source_record']
                recording = record['recording']
                if recording not in records or record != records[recording]:
                    raise ValueError('Saved source record differs from the audited manifest')
                reference = load_reference(recording)
                matching_sources = [identity for identity in proof.get('reference_identities', {}).values()
                                    if identity.get('recording') == recording]
                if (len(matching_sources) != 1 or matching_sources[0].get('sha256')
                        != reference_audit[recording]['reference_file_sha256']):
                    raise ValueError('Source reference bytes differ from saved evaluation provenance')
                count, clip = int(payload['unpadded_frames']), int(payload['clip_index'])
                take = torch.as_tensor(payload['reference_indices'], dtype=torch.long).numpy()
                if (count < 1 or take.shape != (count,) or take.min() < 0
                        or take.max() >= len(reference['frame_ids']) or not np.all(np.diff(take) == 1)
                        or take[0] != int(payload['reference_start'])):
                    raise ValueError('Prediction source indices do not exactly describe a contiguous clip')
                if (not np.array_equal(torch.as_tensor(payload['source_frame_ids']).numpy(), reference['frame_ids'][take])
                        or not np.all(np.diff(reference['frame_ids'][take]) == 1)):
                    raise ValueError('Saved source frame identity differs, or a clip crosses a gap')
                frame_valid = torch.as_tensor(payload['frame_valid']).bool()
                if frame_valid.ndim != 2 or frame_valid.shape[0] != 1:
                    raise ValueError('Expected [1,T] saved frame validity')
                length = frame_valid.shape[1]
                if count > length or bool(frame_valid[0, count:].any()):
                    raise ValueError('Padded states cannot be valid')
                target = torch.as_tensor(payload['target_world']).float()
                target_valid = torch.as_tensor(payload['target_valid']).bool()
                expected_valid = torch.from_numpy(reference['target_valid'][take]).bool() & frame_valid[0, :count, None]
                if (target.shape != (1, length, 99)
                        or not torch.allclose(target[0, :count], torch.from_numpy(reference['target_world'][take]).float(), atol=1e-6, rtol=0)
                        or target_valid.shape != (1, length, 17)
                        or not torch.equal(target_valid[0, :count], expected_valid)
                        or bool(target_valid[0, count:].any())):
                    raise ValueError('Saved targets/masks differ from exact source states')
                hand = torch.as_tensor(payload['handedness'])
                if hand.shape[:2] != (1, length) or not np.array_equal(hand[0, :count].numpy(), reference['handedness'][take]):
                    raise ValueError('Saved handedness differs from source')
                beta = torch.as_tensor(payload['beta']).float()
                available = torch.as_tensor(payload['availability']).bool()
                if beta.shape != (1, length, 10) or available.shape != (1, length, 17):
                    raise ValueError('Saved predicted beta/availability have invalid dimensions')
                states = {'raw': torch.as_tensor(payload['raw_world']).float()}
                if 'state_world' in payload:
                    candidates = torch.as_tensor(payload['refinement_candidates_world']).float()
                    refined = torch.as_tensor(payload['refinement_states_world']).float()
                    if (candidates.ndim != 4 or candidates.shape[:2] != (1, length)
                            or candidates.shape[-1] != 99 or candidates.shape != refined.shape):
                        raise ValueError('Refinement states/candidates must be matching [1,T,K,99]')
                    for iteration in range(candidates.shape[2]):
                        states[f'round{iteration + 1}_candidate'] = candidates[:, :, iteration]
                        states[f'round{iteration + 1}_state'] = refined[:, :, iteration]
                    states.update(candidate=torch.as_tensor(payload['candidate_world']).float(),
                                  final=torch.as_tensor(payload['state_world']).float())
                    if not (torch.allclose(states['candidate'], candidates[:, :, -1], atol=1e-6, rtol=0)
                            and torch.allclose(states['final'], refined[:, :, -1], atol=1e-6, rtol=0)):
                        raise ValueError('Saved final/candidate do not match the final refinement round')
                if any(state.shape != (1, length, 99) for state in states.values()):
                    raise ValueError('Every saved state branch must be [1,T,99]')
                if expected_branches is None:
                    expected_branches = list(states)
                    branch_order[label] = [label + '.' + name for name in states]
                elif list(states) != expected_branches:
                    raise ValueError('Saved clips within an input must share refinement branches')
                valid = expected_valid.all(-1) & frame_valid[0, :count]
                for local, source_index in enumerate(take):
                    key = recording, clip, int(source_index)
                    if key in coverage[label]:
                        raise ValueError('Duplicate source frame within saved input')
                    coverage[label].add(key)
                    if valid[local]:
                        valid_coverage[label].add(key)
                for branch_name, state in states.items():
                    branch_valid = valid & (available[0, :count].all(-1) if branch_name == 'raw' else True)
                    selected = branch_valid.nonzero().flatten()
                    for begin in range(0, len(selected), args.batch_size):
                        select = selected[begin:begin + args.batch_size]
                        indices = take[select.numpy()]
                        state_batch, beta_batch = state[0, select].to(device), beta[0, select].to(device)
                        if not bool(torch.isfinite(state_batch).all() and torch.isfinite(beta_batch).all()):
                            raise FloatingPointError('Nonfinite saved valid state/beta')
                        right = tensor(reference['handedness'][indices], dtype=torch.bool)
                        gt_state, gt_beta = tensor(reference['target_world'][indices]), tensor(reference['beta'][indices])
                        gt_joints = tensor(reference['joint_3d'][indices])
                        predicted = decode(prediction_codec, state_batch, beta_batch, right)
                        delta = (predicted - gt_joints).double() * 1000
                        translation = delta[:, 0]
                        relative = delta - translation[:, None]
                        abs_mse = delta.square().sum(-1).mean(-1)
                        translation_mse = translation.square().sum(-1)
                        relative_mse = relative.square().sum(-1).mean(-1)
                        cross = 2 * (translation[:, None] * relative).sum(-1).mean(-1)
                        residual = abs_mse - translation_mse - relative_mse - cross
                        decomposition_max_abs_residual_mm2 = max(decomposition_max_abs_residual_mm2,
                                                                float(residual.abs().max()))
                        if not torch.allclose(abs_mse, translation_mse + relative_mse + cross, atol=1e-6, rtol=1e-10):
                            raise ValueError('Exact joint MSE decomposition failed')
                        # Post-hoc oracle controls, never used as model inputs.
                        shape_only = decode(prediction_codec, gt_state, beta_batch, right)
                        pose_source_shape = decode(reference_codec, state_batch, gt_beta, right)
                        gt_numeric_beta_pred_codec = decode(prediction_codec, state_batch, gt_beta, right)
                        shape_delta = (shape_only - gt_joints).double() * 1000
                        pose_delta = (pose_source_shape - gt_joints).double() * 1000
                        numeric_delta = (gt_numeric_beta_pred_codec - gt_joints).double() * 1000
                        values = {
                            'wrist_mm': translation.norm(dim=-1), 'mpjpe21_mm': delta.norm(dim=-1).mean(-1),
                            'wrist_aligned_mpjpe21_mm': relative.norm(dim=-1).mean(-1),
                            'joint_absolute_mse_mm2': abs_mse, 'translation_mse_mm2': translation_mse,
                            'joint_relative_mse_mm2': relative_mse, 'cross_mse_mm2': cross,
                            'mse_identity_residual_mm2': residual,
                            'oracle_GT99D_predbeta_predictioncodec_mpjpe_mm': shape_delta.norm(dim=-1).mean(-1),
                            'oracle_pred99D_sourcebeta_referencecodec_mpjpe_mm': pose_delta.norm(dim=-1).mean(-1),
                            'oracle_pred99D_sourcebeta_referencecodec_wrist_aligned_mpjpe_mm': (pose_delta - pose_delta[:, :1]).norm(dim=-1).mean(-1),
                            'control_pred99D_same_numeric_sourcebeta_predictioncodec_mpjpe_mm': numeric_delta.norm(dim=-1).mean(-1),
                            'gt_depth_mm': gt_state[:, 2].double() * 1000,
                            'pred_depth_mm': state_batch[:, 2].double() * 1000,
                        }
                        for axis_index, axis in enumerate(('x', 'y', 'z')):
                            values['wrist_' + axis + '_bias_mm'] = translation[:, axis_index]
                            values['wrist_' + axis + '_mae_mm'] = translation[:, axis_index].abs()
                            values['wrist_' + axis + '_sq_mm2'] = translation[:, axis_index].square()
                            values['joint_relative_' + axis + '_mean_mm'] = relative[:, :, axis_index].mean(-1)
                            values['joint_absolute_' + axis + '_mean_mm'] = delta[:, :, axis_index].mean(-1)
                            values['cross_' + axis + '_mm2'] = 2 * translation[:, axis_index] * relative[:, :, axis_index].mean(-1)
                            for joint_index in range(21):
                                values[f'perjoint{joint_index:02d}_relative_{axis}_mm'] = relative[:, joint_index, axis_index]
                        for joint_index in range(21):
                            values[f'perjoint{joint_index:02d}_relative_norm_mm'] = relative[:, joint_index].norm(dim=-1)
                        cpu_values = {name: value.detach().cpu().tolist() for name, value in values.items()}
                        for offset, source_index in enumerate(indices):
                            row = {'input': label, 'branch': label + '.' + branch_name,
                                   'recording': recording, 'clip_index': clip,
                                   'reference_index': int(source_index), 'frame_id': int(reference['frame_ids'][source_index]),
                                   'handedness': 'right' if reference['handedness'][source_index] else 'left',
                                   'coordinate_frame': record['coordinate_frame']}
                            row.update({name: value[offset] for name, value in cpu_values.items()})
                            if not all(math.isfinite(v) for k, v in row.items() if isinstance(v, float)):
                                raise FloatingPointError('Nonfinite derived diagnostic')
                            rows.append(row)
                provenance[label]['clips'].append({'path': str(prediction_path), 'sha256': file_sha256(prediction_path),
                                                    'recording': recording, 'clip_index': clip, 'source_frames': count})
            print(json.dumps({'input': label, 'clips': len(paths), 'source_frames': len(coverage[label])}), flush=True)

    if any(keys != coverage[labels[0]] for keys in coverage.values()):
        raise ValueError('Inputs must cover exactly the same source frames/clips')
    if any(keys != valid_coverage[labels[0]] for keys in valid_coverage.values()):
        raise ValueError('Inputs disagree on reference validity')
    branches = [branch for label in labels for branch in branch_order[label]]
    transitions = []
    for label in labels:
        ordered = [name for name in branch_order[label] if not name.endswith(('.candidate', '.final'))]
        transitions.extend(paired(rows, before, after) for before, after in zip(ordered[:-1], ordered[1:]))
        if label + '.final' in branch_order[label]:
            transitions.append(paired(rows, label + '.raw', label + '.final'))
    comparisons = []
    baseline = labels[0] + '.raw'
    for branch in branches:
        if branch != baseline:
            comparisons.append(paired(rows, baseline, branch))
    per_branch = {}
    for branch in branches:
        selected = [row for row in rows if row['branch'] == branch]
        per_branch[branch] = {'all': summarize(selected), 'handedness': grouped(selected, 'handedness'),
                              'recording': grouped(selected, 'recording')}
    rows.sort(key=lambda row: (row['recording'], row['clip_index'], row['reference_index'], row['branch']))
    with csv_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    try:
        revision = subprocess.check_output(['git', '-C', str(Path(__file__).resolve().parents[1]),
                                            'rev-parse', 'HEAD'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = 'unavailable'
    report = {
        'schema': 'handloop.visual_geometry_bias_diagnosis.v1',
        'completed_utc': datetime.now(timezone.utc).isoformat(), 'code_revision': revision,
        'script_sha256': file_sha256(Path(__file__)), 'source_manifest': str(manifest_path),
        'source_manifest_sha256': manifest_sha, 'mano_asset_sha256': asset_sha,
        'paired_protocol': paired_protocol,
        'inputs': provenance, 'branches': per_branch, 'stage_transitions': transitions,
        'comparisons_to_first_input_raw': comparisons, 'reference_and_codec_audit': reference_audit,
        'prediction_codec': prediction_codec.provenance(), 'reference_codec': reference_codec.provenance(),
        'paired_source_frames': len(coverage[labels[0]]), 'complete_reference_states': len(valid_coverage[labels[0]]),
        'maximum_mse_identity_residual_mm2': decomposition_max_abs_residual_mm2,
        'definitions': {
            'normal_geometry': 'Every normal branch uses its own saved predicted beta and official HaWoR prediction codec (left shapedirs flip=True); reference is source beta/codec (flip=False). No PA, scale, or rotation fit.',
            'MSE': 'For each frame/joint, delta=t+r; t=predicted J0-reference J0; r=(predicted J-predicted J0)-(reference J-reference J0). mean_j ||delta||^2=||t||^2+mean_j||r||^2+2*t dot mean_j(r). All three terms in mm^2. Cross term may be negative; ratios are not independent causal contributions.',
            'reduction': 'Per-frame means over all21 joints; grouped arithmetic mean over valid complete states. Raw additionally requires all17 availability flags. RMSE is sqrt of grouped mean squared error, not mean Euclidean distance. No macro reweighting.',
            'oracle_GT99D_predbeta_predictioncodec': 'Post-hoc GT99D state with this input predicted beta in prediction codec, compared with source reference. Shape-related geometric error plus any prediction/reference codec convention mismatch; not pure physical shape error and not a model output.',
            'oracle_pred99D_sourcebeta_referencecodec': 'Post-hoc saved predicted99D with source beta in source reference codec. Pose/translation diagnostic under source fitted shape; never fed to prediction. Does not equal an achievable loss improvement, because pose and shape can compensate.',
            'control_same_numeric_sourcebeta_predictioncodec': 'Saved predicted99D with numerical source beta substituted into prediction codec. For left hands source/prediction shapedirs differ, so this is an explicit convention control, NOT a physically matched GT-shape oracle.',
            'same_GT_state_beta_codec_control': 'Decode identical GT99D and numerical source beta in both codecs. Nonzero left differences isolate this decoder convention operation; they do not prove a bug or explain the whole reconstruction error. At zero beta the codecs must agree.',
            'axes': 'Original fixed-camera x/y/z; signed errors predicted minus reference. z is optical depth. Group bias can cancel within or between recordings; inspect MAE/RMSE and per-recording entries.',
            'constant_bias_control': 'Oracle subtracts the group signed wrist mean after seeing all its reference frames. RMSE residual=sqrt(E||t||^2-||E(t)||^2). A high bias fraction describes a common offset within that group, not an identifiable correction available to a trajectory-only deployment model.',
        },
        'limitations': [
            'Read-only post-hoc diagnosis on previously saved predictions; no forward, gradient, retraining, or checkpoint selection.',
            'Beta coefficients from different left-shapedirs conventions are not compared by L2. Geometry mixes pose and shape; oracle error differences are not additive or causal.',
            'Source geometry is fitted MANO, not independently measured MoCap/scanned surface. Same-source FK audit checks implementation consistency only.',
            'Last candidate already includes earlier TWM rounds and self-history. Round deltas are conditional stages, not whole-module causal ablations.',
            'Handedness is confounded with subject, recording, action, and imagery; per-side averages alone cannot establish a mirror bug or a training cause.',
            'GT crop, fixed camera, bounded clips with resets and original HaWoR lookahead remain the evaluation scope; no full official benchmark or long online claim.',
        ],
        'per_frame_csv': str(csv_path), 'torch_version': torch.__version__, 'device': str(device),
    }
    save_json(output, report)
    print(json.dumps({'complete': True, 'report': str(output), 'per_frame_csv': str(csv_path),
                      'branches': branches, 'source_frames': report['paired_source_frames']}), flush=True)


if __name__ == '__main__':
    main()
