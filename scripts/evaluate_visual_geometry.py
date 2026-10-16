#!/usr/bin/env python3
"""Remote-only MANO geometry evaluation of saved visual-model predictions.

This script never loads HaWoR/TWM or performs prediction. It decodes each
saved branch with that prediction's own beta. Source beta is used only to
construct the fitted reference mesh and to audit the reference joint mapping.

Example (run inside the compute Pod):
  python scripts/evaluate_visual_geometry.py \
    --predictions baseline=/remote/evaluation/baseline \
    --predictions joint=/remote/evaluation/joint \
    --source-manifest /remote/sources/source_manifest.json \
    --mano-models /remote/licensed/MANO --device cuda:0 \
    --output /remote/evaluation/geometry.json

Each input is an evaluate_visual_joint.py output directory containing
clips/*.pt, a directory of such .pt files, or one clip .pt. Only fixed-camera
DexYCB source manifests with identity camera_to_world are supported. Absolute
and wrist-translation-aligned metrics are always reported; --include-pa adds
per-frame similarity-aligned joint error for comparison with pose papers.
--include-auc additionally reports normalized PCK AUC over 0--50 mm (100
thresholds), for absolute, wrist-aligned, and PA-aligned joint distances.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


METRICS = ('mpjpe21_mm', 'wrist_aligned_mpjpe21_mm',
           'mpvpe778_mm', 'wrist_aligned_mpvpe778_mm')
DEX_ORDER = (0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18,
             10, 11, 12, 19, 7, 8, 9, 20)


def pa_joint_errors_mm(predicted, target):
    """Fit s*X@R+t to Y per frame, enforcing det(R)=+1 (no reflection).

    Double precision keeps the small hand cloud stable after camera offsets
    are removed. This is an evaluation alignment using reference joints,
    never a correction available during deployment.
    """
    import torch

    x, y = predicted.double(), target.double()
    x_mean, y_mean = x.mean(1, keepdim=True), y.mean(1, keepdim=True)
    xc, yc = x - x_mean, y - y_mean
    variance = xc.square().sum((1, 2))
    if bool((variance <= 1e-20).any()):
        raise ValueError('Cannot similarity-align a degenerate predicted hand')
    u, singular, vh = torch.linalg.svd(xc.transpose(1, 2) @ yc)
    signs = torch.ones_like(singular)
    signs[:, -1] = torch.where(torch.linalg.det(u @ vh) < 0, -1.0, 1.0)
    rotation = (u * signs[:, None, :]) @ vh
    scale = (singular * signs).sum(-1) / variance
    aligned = scale[:, None, None] * (xc @ rotation) + y_mean
    return (aligned - y).norm(dim=-1) * 1000


def pa_mpjpe_mm(predicted, target):
    return pa_joint_errors_mm(predicted, target).mean(-1)


def pck_auc(errors_mm):
    """FreiHAND EvalUtil semantics for complete [N,21] joint observations.

    PCK uses <= threshold. Integrate each joint's PCK over 100 inclusive
    thresholds from 0 to 50 mm, normalize by 50, then average 21 joints.
    Never threshold a per-frame mean error in place of individual joints.
    """
    import numpy as np

    errors = np.asarray(errors_mm, dtype=np.float64)
    if (errors.ndim != 2 or errors.shape[1] != 21 or not len(errors)
            or not np.isfinite(errors).all() or (errors < 0).any()):
        raise ValueError('AUC requires nonempty finite nonnegative [N,21] errors in mm')
    thresholds = np.linspace(0.0, 50.0, 100)
    per_joint_curve = (errors[..., None] <= thresholds).mean(axis=0)
    area = ((per_joint_curve[:, 1:] + per_joint_curve[:, :-1]) * 0.5
            * np.diff(thresholds)).sum(axis=-1) / 50.0
    auc = float(area.mean())
    return {'auc': auc, 'auc_percent': auc * 100.0,
            'valid_states': len(errors), 'joint_observations': int(errors.size),
            'thresholds_mm': thresholds.tolist(),
            'pck_curve': per_joint_curve.mean(axis=0).tolist(),
            'pck_at_mm': {str(t): float((errors <= t).mean()) for t in (5, 10, 20, 30, 50)}}


def save_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True,
                                    allow_nan=False) + '\n')
    temporary.replace(path)


def parse_input(value: str) -> tuple[str, Path]:
    if '=' not in value:
        raise argparse.ArgumentTypeError('Use LABEL=EVALUATION_OUTPUT_OR_CLIP_PT')
    label, name = value.split('=', 1)
    if not label or not all(c.isalnum() or c in '_-' for c in label):
        raise argparse.ArgumentTypeError('Input labels use letters, digits, _ or -')
    return label, Path(name).expanduser().resolve()


def prediction_paths(path: Path) -> list[Path]:
    if path.is_file() and path.suffix == '.pt':
        return [path]
    directory = path / 'clips' if (path / 'clips').is_dir() else path
    paths = sorted(directory.glob('*.pt'))
    if not paths:
        raise ValueError(f'No saved clip predictions at {path}; use --save-predictions during evaluation')
    return paths


def summarize(rows: list[dict[str, Any]], branches: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {'frames': len(rows), 'branches': {}}
    for branch in branches:
        eligible = [row for row in rows if row.get(branch + '_' + METRICS[0]) is not None]
        result['branches'][branch] = {
            'valid_states': len(eligible),
            **{key: sum(row[branch + '_' + key] for row in eligible) / len(eligible)
               for key in METRICS if eligible},
        }
    return result


def grouped(rows: list[dict[str, Any]], field: str,
            branches: list[str]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row[field]), []).append(row)
    return {name: summarize(values, branches) for name, values in sorted(groups.items())}


def main() -> None:
    global METRICS
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--predictions', action='append', type=parse_input, required=True,
                        help='Repeat LABEL=output_directory (or LABEL=one_clip.pt)')
    parser.add_argument('--source-manifest', type=Path, required=True)
    parser.add_argument('--mano-models', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', type=Path, required=True, help='New JSON report filename')
    parser.add_argument('--batch-size', type=int, default=64, help='MANO decode batch size')
    parser.add_argument('--cpu-threads', type=int, default=2)
    parser.add_argument('--include-pa', action='store_true',
                        help='Also report per-frame translation/rotation/scale-aligned MPJPE21')
    parser.add_argument('--include-auc', action='store_true',
                        help='Also report absolute/wrist-aligned/PA PCK AUC; implies --include-pa')
    args = parser.parse_args()
    args.include_pa = args.include_pa or args.include_auc
    if args.include_pa:
        METRICS = (*METRICS, 'pa_mpjpe21_mm')
    if not Path('/sciclone/data10/jchen57').is_dir():
        raise RuntimeError('Run geometry/model evaluation in the remote compute Pod')
    if args.batch_size < 1 or args.cpu_threads < 1:
        parser.error('--batch-size and --cpu-threads must be positive')
    labels = [label for label, _ in args.predictions]
    if len(labels) != len(set(labels)):
        parser.error('Input labels must be unique')
    output = args.output.expanduser().resolve()
    csv_path = output.with_name(output.stem + '_per_frame.csv')
    if output.exists() or csv_path.exists():
        raise FileExistsError('Use a fresh geometry output filename')
    output.parent.mkdir(parents=True, exist_ok=True)

    # Keep heavy runtime imports behind the remote-only guard.
    import numpy as np
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
    from handloop.data.visual_clips import file_sha256
    from handloop.visual.codec import MANOStateCodec

    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    manifest_path = args.source_manifest.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    manifest_sha = file_sha256(manifest_path)
    if (manifest.get('schema') != 'handloop.visual_feedback_sources.v1'
            or manifest.get('phase') != 'complete' or manifest.get('dataset') != 'dexycb'):
        raise ValueError('Use a complete audited DexYCB visual-feedback source manifest')
    records = {record['recording']: record for record in manifest['records']}
    if len(records) != len(manifest['records']):
        raise ValueError('Source recording identities must be unique')
    assets = {side: args.mano_models / f'MANO_{side}.pkl' for side in ('LEFT', 'RIGHT')}
    asset_sha = {side: file_sha256(path) for side, path in assets.items()}
    prediction_codec = MANOStateCodec(assets['RIGHT'], assets['LEFT'], left_shapedirs_fix=True).to(device)
    reference_codec = MANOStateCodec(assets['RIGHT'], assets['LEFT'], left_shapedirs_fix=False).to(device)

    def tensor(value: Any, *, dtype=torch.float32):
        return torch.as_tensor(value, dtype=dtype, device=device)

    def dex_joints(geometry, right):
        vertices, joints = geometry['vertices'], geometry['joints']
        tip_ids = torch.tensor([745, 317, 444, 556, 673], device=device).expand(len(right), -1).clone()
        tip_ids[~right.bool(), 2] = 445
        tips = vertices.gather(1, tip_ids[..., None].expand(-1, -1, 3))
        return torch.cat((joints, tips), dim=1)[:, DEX_ORDER]

    references: dict[str, dict[str, Any]] = {}
    audit: dict[str, dict[str, Any]] = {}

    def load_reference(recording: str) -> dict[str, Any]:
        if recording in references:
            return references[recording]
        record = records[recording]
        if record.get('split') not in ('val', 'test'):
            raise ValueError('Geometry evaluation excludes training recordings')
        coordinate = record.get('coordinate_frame', '')
        if not (coordinate.startswith('fixed_camera_') and coordinate.endswith('_as_world_gauge')):
            raise ValueError('This evaluator only supports the explicit fixed-camera world gauge')
        path = (manifest_path.parent / record['reference_file']).resolve()
        if not path.is_relative_to(manifest_path.parent):
            raise ValueError('Reference path escapes the source manifest directory')
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        n = int(record['frames'])
        for key, shape in {'target_world': (n, 99), 'target_valid': (n, 17),
                           'joint_3d': (n, 21, 3), 'frame_ids': (n,),
                           'timestamps': (n,), 'handedness': (n,),
                           'camera_to_world': (n, 4, 4)}.items():
            if key not in arrays or arrays[key].shape != shape or not np.isfinite(arrays[key]).all():
                raise ValueError(f'Invalid source reference {key}: {recording}')
        if not np.allclose(arrays['camera_to_world'], np.eye(4), rtol=0, atol=1e-7):
            raise ValueError('Source joint_3d camera frame is not the saved prediction world gauge')
        side = arrays['handedness']
        if not np.isin(side, [0, 1]).all() or not np.all(side == side[0]):
            raise ValueError('A reference recording must contain one consistent hand identity')
        if int(record['handedness']) != int(side[0]):
            raise ValueError('Manifest/reference handedness mismatch')
        beta = arrays.get('beta')
        if beta is None or beta.shape not in ((10,), (n, 10)) or not np.isfinite(beta).all():
            raise ValueError('Source reference beta must be static10 or per-frame10')
        if beta.shape == (10,):
            beta = np.broadcast_to(beta, (n, 10)).copy()
        valid = arrays['target_valid'].astype(bool).all(-1)
        vertices = torch.full((n, 778, 3), torch.nan)
        errors = []
        indices = np.flatnonzero(valid)
        if not len(indices):
            raise ValueError('No complete reference states for geometry auditing')
        for begin in range(0, len(indices), args.batch_size):
            take = indices[begin:begin + args.batch_size]
            right = tensor(side[take], dtype=torch.bool)
            geometry = reference_codec.geometry(tensor(arrays['target_world'][take]), tensor(beta[take]), right)
            decoded = dex_joints(geometry, right)
            difference = (decoded - tensor(arrays['joint_3d'][take])).norm(dim=-1)
            if not bool(torch.isfinite(difference).all()):
                raise FloatingPointError('Reference FK produced nonfinite joints')
            errors.append(difference.detach().cpu())
            vertices[take] = geometry['vertices'].detach().cpu()
        error = torch.cat(errors)
        max_mm = float(error.max()) * 1000
        audit[recording] = {'reference_file': str(path), 'sha256': file_sha256(path),
            'frames': n, 'audited_complete_states': len(indices),
            'fk_vs_supplied_joint21_mean_mm': float(error.mean()) * 1000,
            'fk_vs_supplied_joint21_max_mm': max_mm, 'tolerance_mm': 1.0,
            'passed': max_mm <= 1.0}
        if max_mm > 1.0:
            failure = output.with_name(output.stem + '_reference_audit_failed.json')
            save_json(failure, {'source_manifest_sha256': manifest_sha,
                               'reference_audit': audit, 'error': 'Reference FK exceeds 1 mm'})
            raise ValueError(f'Reference FK/Dex21 disagreement in {recording}: {max_mm:.6f} mm > 1 mm')
        arrays['reference_vertices'] = vertices
        arrays['complete_valid'] = valid
        references[recording] = arrays
        return arrays

    rows: dict[tuple[str, int, int], dict[str, Any]] = {}
    branches: list[str] = []
    input_provenance: dict[str, Any] = {}
    coverage: dict[str, set[tuple[str, int, int]]] = {}
    auc_samples: dict[str, list[dict[str, Any]]] = {}
    with torch.no_grad():
        for label, path in args.predictions:
            paths = prediction_paths(path)
            eval_provenance = path / 'provenance.json' if path.is_dir() else path.parent.parent / 'provenance.json'
            if path.is_dir() and path.name == 'clips' and not eval_provenance.is_file():
                eval_provenance = path.parent / 'provenance.json'
            verified_evaluation_provenance = False
            if eval_provenance.is_file():
                source_provenance = json.loads(eval_provenance.read_text())
                if source_provenance.get('source_manifest_sha256') != manifest_sha:
                    raise ValueError('Evaluation provenance used another source manifest')
                if source_provenance.get('mano_asset_sha256') != asset_sha:
                    raise ValueError('Evaluation predictions used different MANO assets')
                verified_evaluation_provenance = True
            input_provenance[label] = {'path': str(path), 'clips': [],
                                       'evaluation_provenance': str(eval_provenance) if eval_provenance.is_file() else None,
                                       'legacy_payload_sha_fallback_clips': []}
            coverage[label] = set()
            label_state_keys = None
            for prediction_path in paths:
                payload = torch.load(prediction_path, map_location='cpu', weights_only=False)
                payload_manifest = payload.get('source_manifest')
                if (not isinstance(payload_manifest, str) or not Path(payload_manifest).is_absolute()
                        or Path(payload_manifest).resolve() != manifest_path):
                    raise ValueError(f'Prediction/source manifest absolute path mismatch: {prediction_path}')
                payload_sha = payload.get('source_manifest_sha256')
                if 'source_manifest_sha256' not in payload:
                    if not verified_evaluation_provenance:
                        raise ValueError('Legacy prediction without manifest SHA requires verified adjacent evaluation provenance')
                    input_provenance[label]['legacy_payload_sha_fallback_clips'].append(str(prediction_path))
                elif payload_sha != manifest_sha:
                    raise ValueError(f'Prediction/source manifest identity mismatch: {prediction_path}')
                source_record = payload['source_record']
                recording = source_record['recording']
                if recording not in records or source_record != records[recording]:
                    raise ValueError('Saved source record differs from the supplied manifest')
                reference = load_reference(recording)
                count, clip_index = int(payload['unpadded_frames']), int(payload['clip_index'])
                take = torch.as_tensor(payload['reference_indices'], dtype=torch.long).numpy()
                if (take.shape != (count,) or not count or take.min() < 0
                        or take.max() >= int(source_record['frames'])
                        or not np.all(np.diff(take) == 1)
                        or int(take[0]) != int(payload['reference_start'])):
                    raise ValueError('Saved reference indices must exactly cover the unpadded contiguous clip')
                if not np.array_equal(torch.as_tensor(payload['source_frame_ids']).numpy(), reference['frame_ids'][take]):
                    raise ValueError('Prediction/source frame identity mismatch')
                if len(take) > 1 and not np.all(np.diff(reference['frame_ids'][take]) == 1):
                    raise ValueError('A prediction clip must not cross a source frame gap')
                frame_valid = torch.as_tensor(payload['frame_valid']).bool()
                length = frame_valid.shape[-1]
                if frame_valid.shape != (1, length) or count > length or bool(frame_valid[0, count:].any()):
                    raise ValueError('Padding must remain invalid in saved predictions')
                expected_target = torch.from_numpy(reference['target_world'][take]).float()
                saved_target = torch.as_tensor(payload['target_world']).float()
                if saved_target.shape != (1, length, 99) or not torch.allclose(saved_target[0, :count], expected_target, rtol=0, atol=1e-6):
                    raise ValueError('Saved target states differ from the exact source frames')
                if not np.array_equal(torch.as_tensor(payload['handedness'])[0, :count].numpy(), reference['handedness'][take].astype(bool)):
                    raise ValueError('Prediction/source hand identity mismatch')
                target_valid = torch.as_tensor(payload['target_valid']).bool()
                source_valid = torch.from_numpy(reference['target_valid'][take]).bool()
                expected_valid = source_valid & frame_valid[0, :count, None]
                if target_valid.shape != (1, length, 17) or not torch.equal(target_valid[0, :count], expected_valid) or bool(target_valid[0, count:].any()):
                    raise ValueError('Saved target-valid masks must match source masks with padding excluded')
                valid = target_valid[0, :count].all(-1) & frame_valid[0, :count]
                beta = torch.as_tensor(payload['beta']).float()
                available = torch.as_tensor(payload['availability']).bool()
                if beta.shape != (1, length, 10) or available.shape != (1, length, 17):
                    raise ValueError('Invalid predicted beta or raw-availability dimensions')
                state_keys = [('raw', 'raw_world')]
                if 'state_world' in payload:
                    state_keys += [('candidate', 'candidate_world'), ('final', 'state_world')]
                if label_state_keys is None:
                    label_state_keys = state_keys
                    branches.extend(label + '.' + name for name, _ in state_keys)
                elif label_state_keys != state_keys:
                    raise ValueError('All clips of one input must contain the same output branches')
                local_rows = []
                for local, source_index in enumerate(take):
                    key = (recording, clip_index, int(source_index))
                    if key in coverage[label]:
                        raise ValueError('Duplicate prediction state in an input')
                    coverage[label].add(key)
                    row = rows.setdefault(key, {'recording': recording, 'clip_index': clip_index,
                        'clip_key': recording + '::' + str(clip_index),
                        'reference_index': int(source_index), 'frame_id': int(reference['frame_ids'][source_index]),
                        'timestamp_seconds': float(reference['timestamps'][source_index]),
                        'handedness': 'right' if reference['handedness'][source_index] else 'left',
                        'coordinate_frame': source_record['coordinate_frame']})
                    row[label + '_complete_reference_valid'] = int(valid[local])
                    local_rows.append(row)
                for branch_name, state_key in state_keys:
                    branch = label + '.' + branch_name
                    state = torch.as_tensor(payload[state_key]).float()
                    if state.shape != (1, length, 99):
                        raise ValueError('Predicted branch state must be [1,T,99]')
                    branch_valid = valid & (available[0, :count].all(-1) if branch_name == 'raw' else True)
                    selected = branch_valid.nonzero().flatten()
                    for row in local_rows:
                        row.update({branch + '_' + metric: None for metric in METRICS})
                    for begin in range(0, len(selected), args.batch_size):
                        select = selected[begin:begin + args.batch_size]
                        source_index = take[select.numpy()]
                        predicted_state, predicted_beta = state[0, select], beta[0, select]
                        if not bool(torch.isfinite(predicted_state).all() and torch.isfinite(predicted_beta).all()):
                            raise FloatingPointError('Nonfinite valid prediction state/beta')
                        right = tensor(reference['handedness'][source_index], dtype=torch.bool)
                        geometry = prediction_codec.geometry(predicted_state.to(device), predicted_beta.to(device), right)
                        predicted_joints = dex_joints(geometry, right)
                        target_joints = tensor(reference['joint_3d'][source_index])
                        target_vertices = reference['reference_vertices'][source_index].to(device)
                        joint_delta = predicted_joints - target_joints
                        vertex_delta = geometry['vertices'] - target_vertices
                        wrist_delta = joint_delta[:, :1]
                        measured = {
                            'mpjpe21_mm': joint_delta.norm(dim=-1).mean(-1) * 1000,
                            'wrist_aligned_mpjpe21_mm': (joint_delta - wrist_delta).norm(dim=-1).mean(-1) * 1000,
                            'mpvpe778_mm': vertex_delta.norm(dim=-1).mean(-1) * 1000,
                            'wrist_aligned_mpvpe778_mm': (vertex_delta - wrist_delta).norm(dim=-1).mean(-1) * 1000,
                        }
                        if args.include_pa:
                            pa_errors = pa_joint_errors_mm(predicted_joints, target_joints)
                            measured['pa_mpjpe21_mm'] = pa_errors.mean(-1)
                        if args.include_auc:
                            auc_samples.setdefault(branch, []).append({
                                'handedness': right.detach().cpu().numpy(),
                                'absolute': joint_delta.norm(dim=-1).detach().cpu().numpy() * 1000,
                                'wrist_aligned': (joint_delta - wrist_delta).norm(dim=-1).detach().cpu().numpy() * 1000,
                                'pa': pa_errors.detach().cpu().numpy(),
                            })
                        if not all(bool(torch.isfinite(value).all()) for value in measured.values()):
                            raise FloatingPointError('Nonfinite decoded geometric error')
                        for metric, values in measured.items():
                            for local, value in zip(select.tolist(), values.detach().cpu().tolist()):
                                local_rows[local][branch + '_' + metric] = value
                input_provenance[label]['clips'].append({'file': str(prediction_path),
                    'sha256': file_sha256(prediction_path), 'clip_index': clip_index,
                    'recording': recording, 'unpadded_frames': count})
            print(json.dumps({'input': label, 'clips': len(paths), 'source_frames': len(coverage[label])}), flush=True)

    if any(keys != coverage[labels[0]] for keys in coverage.values()):
        raise ValueError('Paired inputs must cover exactly the same clips and source frames')
    ordered_rows = [rows[key] for key in sorted(rows)]
    fields = list(dict.fromkeys(field for row in ordered_rows for field in row))
    with csv_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(ordered_rows)
    try:
        revision = subprocess.check_output(['git', '-C', str(Path(__file__).resolve().parents[1]),
                                            'rev-parse', 'HEAD'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = 'unavailable'
    report = {'schema': 'handloop.visual_geometry_evaluation.v1',
        'completed_utc': datetime.now(timezone.utc).isoformat(),
        'evaluation_code_revision': revision,
        'evaluation_script_sha256': file_sha256(Path(__file__)),
        'source_manifest': str(manifest_path), 'source_manifest_sha256': manifest_sha,
        'mano_asset_sha256': asset_sha, 'inputs': input_provenance, 'branches': branches,
        'reference_fk_audit': audit,
        'prediction_codec': prediction_codec.provenance(),
        'reference_codec': reference_codec.provenance(),
        'joint_mapping': {'mano_joint_count': 16, 'right_tip_vertices': [745, 317, 444, 556, 673],
                          'left_tip_vertices': [745, 317, 445, 556, 673], 'dex21_order': list(DEX_ORDER)},
        'metric_definition': {
            'mpjpe21_mm': 'Mean Euclidean predicted MANO/Dex21 joint distance to source joint_3d, unaligned, millimetres.',
            'wrist_aligned_mpjpe21_mm': 'Subtract each hand own J0 translation before joint distance; no rotation/scale/PA alignment.',
            'mpvpe778_mm': 'Mean corresponding MANO778 vertex distance to source-state/source-beta fitted reference mesh, unaligned.',
            'wrist_aligned_mpvpe778_mm': 'Subtract each hand own anatomical wrist translation before vertex distance; no rotation/scale/PA alignment.'},
        'reduction': 'Per-frame mean over all21 joints or778 vertices; arithmetic mean over valid complete states. Raw requires all17 availability flags; padding excluded. Counts reported per branch.',
        'reference_mesh_scope': 'Same-source fitted MANO surface, not independently measured/scanned mesh. Reference J0 is source joint_3d wrist as in prepared99D state.',
        'shape_policy': 'Every prediction branch uses only its own saved frontend beta. Source beta is used only by the reference decoder; never fed to a prediction forward.',
        'coordinate_scope': 'Original fixed camera is explicitly asserted identity world gauge; these errors do not establish moving-camera SLAM/world accuracy.',
        'all': summarize(ordered_rows, branches),
        'clip': grouped(ordered_rows, 'clip_key', branches),
        'recording': grouped(ordered_rows, 'recording', branches),
        'handedness': grouped(ordered_rows, 'handedness', branches),
        'per_frame_csv': str(csv_path), 'torch_version': torch.__version__,
        'device': str(device), 'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None}
    if args.include_pa:
        report['metric_definition']['pa_mpjpe21_mm'] = (
            'Per-frame least-squares similarity fit of all21 predicted joints to reference '
            '(translation, proper rotation det+1, uniform scale), then mean Euclidean '
            'joint error in millimetres. Same subset/mask as absolute metrics; not the full paper benchmark.')
    if args.include_auc:
        auc_groups: dict[str, Any] = {'all': {}, 'left': {}, 'right': {}}
        for branch, samples in auc_samples.items():
            right = np.concatenate([sample['handedness'] for sample in samples]).astype(bool)
            for alignment in ('absolute', 'wrist_aligned', 'pa'):
                distances = np.concatenate([sample[alignment] for sample in samples])
                for name, mask in (('all', np.ones_like(right)), ('left', ~right), ('right', right)):
                    if mask.any():
                        auc_groups[name].setdefault(branch, {})[alignment] = pck_auc(distances[mask])
        report['pck_auc'] = {
            'protocol': 'Per-joint PCK(error<=threshold), 100 inclusive linspace thresholds 0--50 mm; trapezoidal integral / 50, then mean over21 joints. AUC in [0,1], auc_percent in [0,100]; higher is better.',
            'reference': 'https://github.com/lmb-freiburg/freihand/blob/master/utils/eval_util.py',
            'groups': auc_groups,
            'scope': 'Same complete-state subset and prediction-valid masks as the geometric metrics; no occlusion grouping or full benchmark claim.'}
    save_json(output, report)
    print(json.dumps({'complete': True, 'report': str(output), 'all': report['all']}, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
