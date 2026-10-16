#!/usr/bin/env python3
"""Remote-only inverse audit of weak-perspective camera depth from saved states.

No RGB/network forward occurs. For raw outputs the bridge contract gives
MANO transl = saved anatomical wrist - shaped rest wrist(predicted beta).
The official get_trans map is invertible away from zero depth/scale, so its
implied camera [s,tx,ty] can be recovered and replayed without a camera head.
For candidate/final this is only an effective scale: they have no camera head.

  python scripts/diagnose_camera_depth_failures.py \
    --predictions baseline=/remote/test_baseline \
    --predictions best=/remote/test_best --predictions last=/remote/test_last \
    --source-manifest /remote/source_manifest.json --mano-models /remote/MANO \
    --hawor-root /remote/official_HaWoR --device cpu \
    --output /remote/test_camera_depth_diagnosis.json

This only diagnoses saved predictions. Inversion cannot identify why a learned
camera head changed or show that a different loss/constraint would fix it.
"""
from __future__ import annotations

import argparse
import ast
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

from evaluate_visual_geometry import parse_input, prediction_paths, save_json


def extract_function(path: Path, name: str, namespace: dict[str, Any]):
    tree = ast.parse(path.read_text())
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one official {name} function in {path}')
    function = matches[0]
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace[name], {'source_file': str(path), 'first_line': function.lineno,
                             'last_line': function.end_lineno}


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    import numpy as np
    if not rows:
        return {'states': 0}
    metrics = ('wrist_mm', 'wrist_depth_error_mm', 'mano_transl_z_mm', 'GT_mano_transl_z_mm',
               'implied_camera_s', 'GT_implied_camera_s', 's_over_GT_s', 'depth_over_GT_depth',
               'depth_derivative_abs_m_per_scale', 'crop_b_pixels', 'weak_camera_sensitivity_mm_per_1pct_s')
    result = {'states': len(rows), 'recordings': len({r['recording'] for r in rows})}
    for key in metrics:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        result[key] = {'mean': float(values.mean()), 'min': float(values.min()),
                       'p05': float(np.quantile(values, .05)), 'median': float(np.median(values)),
                       'p95': float(np.quantile(values, .95)), 'max': float(values.max())}
    result['diagnostic_threshold_counts'] = {
        'negative_or_zero_MANO_depth': sum(row['mano_transl_z_mm'] <= 0 for row in rows),
        'abs_MANO_depth_gt2000mm': sum(abs(row['mano_transl_z_mm']) > 2000 for row in rows),
        'wrist_error_gt500mm': sum(row['wrist_mm'] > 500 for row in rows),
        'abs_implied_s_over_GT_s_lt0p25': sum(abs(row['s_over_GT_s']) < .25 for row in rows),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--predictions', action='append', type=parse_input, required=True)
    parser.add_argument('--source-manifest', type=Path, required=True)
    parser.add_argument('--mano-models', type=Path, required=True)
    parser.add_argument('--hawor-root', type=Path, required=True)
    parser.add_argument('--device', default='cpu', help='Only a shaped-rest-wrist tensor calculation; no model forward')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--top', type=int, default=10)
    args = parser.parse_args()
    if not Path('/sciclone/data10/jchen57').is_dir():
        raise RuntimeError('Run numerical diagnostics inside the remote compute Pod')
    if args.top < 1:
        parser.error('--top must be positive')
    labels = [label for label, _ in args.predictions]
    if len(labels) != len(set(labels)):
        parser.error('Input labels must be unique')
    output = args.output.expanduser().resolve()
    csv_path = output.with_name(output.stem + '_per_frame.csv')
    if output.exists() or csv_path.exists():
        raise FileExistsError('Use a fresh diagnostic filename')
    output.parent.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
    from handloop.data.visual_clips import file_sha256
    from handloop.visual.codec import MANOStateCodec

    torch.set_num_threads(2)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    hawor_root = args.hawor_root.expanduser().resolve()
    get_trans_path = hawor_root / 'lib/models/hawor.py'
    boxes_path = hawor_root / 'lib/utils/imutils.py'
    get_trans, camera_source = extract_function(get_trans_path, 'get_trans', {'torch': torch})
    boxes_to_cs, box_source = extract_function(boxes_path, 'boxes_2_cs', {'np': np})
    source_files = {str(path): file_sha256(path) for path in
                    (get_trans_path, boxes_path, hawor_root / 'lib/datasets/track_dataset.py',
                     hawor_root / 'lib/models/modules.py')}
    assets = {side: args.mano_models / f'MANO_{side}.pkl' for side in ('LEFT', 'RIGHT')}
    asset_sha = {side: file_sha256(path) for side, path in assets.items()}
    prediction_codec = MANOStateCodec(assets['RIGHT'], assets['LEFT'], left_shapedirs_fix=True).to(device)
    reference_codec = MANOStateCodec(assets['RIGHT'], assets['LEFT'], left_shapedirs_fix=False).to(device)
    manifest_path = args.source_manifest.expanduser().resolve()
    manifest_sha = file_sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get('schema') != 'handloop.visual_feedback_sources.v1'
            or manifest.get('phase') != 'complete' or manifest.get('dataset') != 'dexycb'):
        raise ValueError('Use a completed audited DexYCB source manifest')
    records = {record['recording']: record for record in manifest['records']}
    if len(records) != len(manifest['records']):
        raise ValueError('Source recording identities must be unique')
    references, reference_sha = {}, {}

    def tensor(value, *, dtype=torch.float32):
        return torch.as_tensor(value, device=device, dtype=dtype)

    def load_reference(recording):
        if recording in references:
            return references[recording]
        record = records[recording]
        if record['split'] not in ('val', 'test'):
            raise ValueError('Only held-out val/test are accepted')
        coordinate = record.get('coordinate_frame', '')
        if not (coordinate.startswith('fixed_camera_') and coordinate.endswith('_as_world_gauge')):
            raise ValueError('Camera depth analysis requires the fixed original camera gauge')
        path = (manifest_path.parent / record['reference_file']).resolve()
        if not path.is_relative_to(manifest_path.parent):
            raise ValueError('Reference path escapes manifest directory')
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
        n = int(record['frames'])
        for name, shape in {'target_world': (n, 99), 'target_valid': (n, 17),
                            'handedness': (n,), 'frame_ids': (n,), 'bbox_xyxy': (n, 4),
                            'camera_intrinsics': (n, 3, 3), 'camera_to_world': (n, 4, 4)}.items():
            if name not in arrays or arrays[name].shape != shape or not np.isfinite(arrays[name]).all():
                raise ValueError(f'Invalid reference array {recording}: {name}')
        if not np.allclose(arrays['camera_to_world'], np.eye(4), rtol=0, atol=1e-7):
            raise ValueError('Saved world gauge is not the original camera')
        if not np.all(arrays['handedness'] == int(record['handedness'])):
            raise ValueError('Manifest/reference handedness disagreement')
        beta = arrays.get('beta')
        if beta is None or beta.shape not in ((10,), (n, 10)) or not np.isfinite(beta).all():
            raise ValueError('Source beta must be static10 or per-frame10')
        if beta.shape == (10,):
            beta = np.broadcast_to(beta, (n, 10)).copy()
        arrays['beta'] = beta
        center, scale = boxes_to_cs(arrays['bbox_xyxy'].copy())
        # This matches VisualFeedbackRGBDataset._crop: TrackDatasetEval dilate=1.2.
        arrays['crop_center_unflipped'] = center
        arrays['crop_scale'] = scale * 1.2
        arrays['official_focal'] = (arrays['camera_intrinsics'][:, 0, 0]
                                    + arrays['camera_intrinsics'][:, 1, 1]) * .5
        if np.any(arrays['crop_scale'] <= 0) or np.any(arrays['official_focal'] <= 0):
            raise ValueError('Positive crop size/focal are required')
        references[recording], reference_sha[recording] = arrays, file_sha256(path)
        return arrays

    rows, provenance, coverage = [], {}, {}
    protocol = None
    maximum_replay_error_m = 0.0
    with torch.no_grad():
        for label, directory in args.predictions:
            paths = prediction_paths(directory)
            proof_path = directory / 'provenance.json' if directory.is_dir() else directory.parent.parent / 'provenance.json'
            if directory.is_dir() and directory.name == 'clips':
                proof_path = directory.parent / 'provenance.json'
            proof = json.loads(proof_path.read_text())
            if proof.get('source_manifest_sha256') != manifest_sha or proof.get('mano_asset_sha256') != asset_sha:
                raise ValueError('Saved predictions used different source/MANO assets')
            current_protocol = {name: proof.get(name) for name in
                                ('split', 'history_policy', 'evaluated_indices', 'hawor_config_sha256',
                                 'official_hawor_checkpoint_sha256')}
            if protocol is None:
                protocol = current_protocol
            elif protocol != current_protocol:
                raise ValueError('Paired inputs used different evaluation protocols')
            provenance[label] = {'evaluation_provenance': str(proof_path),
                                  'evaluation_provenance_sha256': file_sha256(proof_path),
                                  **{key: proof.get(key) for key in
                                     ('mode', 'checkpoint', 'checkpoint_sha256', 'checkpoint_step')}, 'clips': []}
            coverage[label] = set()
            for path in paths:
                payload = torch.load(path, map_location='cpu', weights_only=False)
                if (payload.get('source_manifest_sha256') != manifest_sha
                        or Path(payload['source_manifest']).resolve() != manifest_path):
                    raise ValueError('Saved clip/source identity mismatch')
                record = payload['source_record']
                recording = record['recording']
                if record != records.get(recording):
                    raise ValueError('Saved clip record differs from source manifest')
                arrays = load_reference(recording)
                identities = [r for r in proof.get('reference_identities', {}).values() if r.get('recording') == recording]
                if len(identities) != 1 or identities[0]['sha256'] != reference_sha[recording]:
                    raise ValueError('Source reference bytes changed after saved evaluation')
                count, clip_index = int(payload['unpadded_frames']), int(payload['clip_index'])
                take = torch.as_tensor(payload['reference_indices'], dtype=torch.long).numpy()
                if (take.shape != (count,) or not count or take.min() < 0 or take.max() >= len(arrays['frame_ids'])
                        or take[0] != int(payload['reference_start']) or not np.all(np.diff(take) == 1)
                        or not np.all(np.diff(arrays['frame_ids'][take]) == 1)
                        or not np.array_equal(torch.as_tensor(payload['source_frame_ids']).numpy(), arrays['frame_ids'][take])):
                    raise ValueError('Saved frame/index identity is inconsistent')
                frame_valid = torch.as_tensor(payload['frame_valid']).bool()
                target_valid = torch.as_tensor(payload['target_valid']).bool()
                if frame_valid.ndim != 2 or frame_valid.shape[0] != 1:
                    raise ValueError('Expected saved [1,T] frame validity')
                length = frame_valid.shape[1]
                expected_valid = torch.from_numpy(arrays['target_valid'][take]).bool() & frame_valid[0, :count, None]
                if (count > length or bool(frame_valid[0, count:].any()) or target_valid.shape != (1, length, 17)
                        or not torch.equal(target_valid[0, :count], expected_valid) or bool(target_valid[0, count:].any())
                        or not torch.allclose(torch.as_tensor(payload['target_world'])[0, :count].float(),
                                              torch.from_numpy(arrays['target_world'][take]).float(), atol=1e-6, rtol=0)
                        or not np.array_equal(torch.as_tensor(payload['handedness'])[0, :count].numpy(), arrays['handedness'][take])):
                    raise ValueError('Saved target/mask/hand identity differs from source')
                beta = torch.as_tensor(payload['beta']).float()
                available = torch.as_tensor(payload['availability']).bool()
                if beta.shape != (1, length, 10) or available.shape != (1, length, 17):
                    raise ValueError('Invalid saved beta/availability dimensions')
                selected = (expected_valid.all(-1) & available[0, :count].all(-1)).nonzero().flatten()
                indices = take[selected.numpy()]
                for source_index in indices:
                    key = recording, clip_index, int(source_index)
                    if key in coverage[label]:
                        raise ValueError('Duplicate complete source state')
                    coverage[label].add(key)
                if not len(indices):
                    continue
                right = tensor(arrays['handedness'][indices], dtype=torch.bool)
                predicted_beta = beta[0, selected].to(device)
                predicted_rest = prediction_codec.rest_wrist(predicted_beta, right).double()
                target_state = tensor(arrays['target_world'][indices], dtype=torch.float64)
                target_rest = reference_codec.rest_wrist(tensor(arrays['beta'][indices]), right).double()
                gt_translation = target_state[:, :3] - target_rest
                center = tensor(arrays['crop_center_unflipped'][indices]).double()
                scale = tensor(arrays['crop_scale'][indices]).double()
                focal = tensor(arrays['official_focal'][indices]).double()
                principal = tensor(arrays['camera_intrinsics'][indices, :2, 2]).double()
                b = scale * 200
                if bool((gt_translation[:, 2].abs() < 1e-8).any()):
                    raise ValueError('Reference zero MANO depth cannot be inverted')
                gt_s = 2 * focal / (b * gt_translation[:, 2])
                states = {'raw': payload['raw_world']}
                if 'state_world' in payload:
                    states.update(candidate=payload['candidate_world'], final=payload['state_world'])
                for branch, value in states.items():
                    state = torch.as_tensor(value).float()
                    if state.shape != (1, length, 99):
                        raise ValueError('Saved state dimensions must be [1,T,99]')
                    state = state[0, selected].to(device).double()
                    translation = state[:, :3] - predicted_rest
                    if not bool(torch.isfinite(state).all() and torch.isfinite(predicted_beta).all()):
                        raise FloatingPointError('Nonfinite saved prediction')
                    z = translation[:, 2]
                    if bool((z.abs() < 1e-8).any()):
                        raise ValueError('Zero MANO depth cannot be inverted; preserve offending saved clip')
                    s = 2 * focal / (b * z)
                    shift = (center - principal) / focal[:, None] * z[:, None]
                    post_unflip_tx_ty = translation[:, :2] - shift
                    camera = torch.cat((s[:, None], post_unflip_tx_ty), -1)
                    replay = get_trans(None, camera, center, scale, focal, principal).squeeze(1)
                    replay_error = float((replay - translation).abs().max())
                    maximum_replay_error_m = max(maximum_replay_error_m, replay_error)
                    if replay_error > 1e-8:
                        raise ValueError('Inverse formula does not replay the supplied official get_trans')
                    wrist_delta = state[:, :3] - target_state[:, :3]
                    for offset, source_index in enumerate(indices):
                        row = {'input': label, 'branch': label + '.' + branch, 'recording': recording,
                               'clip_index': clip_index, 'reference_index': int(source_index),
                               'frame_id': int(arrays['frame_ids'][source_index]), 'handedness': 'right' if right[offset] else 'left',
                               'scale_interpretation': 'inverse_raw_camera_head_s' if branch == 'raw' else 'effective_state_s_not_camera_head',
                               'wrist_mm': float(wrist_delta[offset].norm()) * 1000,
                               'wrist_depth_error_mm': float(wrist_delta[offset, 2]) * 1000,
                               'wrist_z_mm': float(state[offset, 2]) * 1000,
                               'GT_wrist_z_mm': float(target_state[offset, 2]) * 1000,
                               'pred_rest_wrist_z_mm': float(predicted_rest[offset, 2]) * 1000,
                               'source_rest_wrist_z_mm': float(target_rest[offset, 2]) * 1000,
                               'mano_transl_z_mm': float(z[offset]) * 1000,
                               'GT_mano_transl_z_mm': float(gt_translation[offset, 2]) * 1000,
                               'implied_camera_s': float(s[offset]), 'GT_implied_camera_s': float(gt_s[offset]),
                               's_over_GT_s': float(s[offset] / gt_s[offset]),
                               'depth_over_GT_depth': float(z[offset] / gt_translation[offset, 2]),
                               'implied_post_unflip_tx_m': float(post_unflip_tx_ty[offset, 0]),
                               'implied_post_unflip_ty_m': float(post_unflip_tx_ty[offset, 1]),
                               'implied_pre_unflip_tx_m': float(post_unflip_tx_ty[offset, 0]) * (1 if right[offset] else -1),
                               'crop_b_pixels': float(b[offset]), 'crop_center_x_pixels': float(center[offset, 0]),
                               'crop_center_y_pixels': float(center[offset, 1]), 'focal_pixels': float(focal[offset]),
                               'principal_x_pixels': float(principal[offset, 0]), 'principal_y_pixels': float(principal[offset, 1]),
                               'bbox_x1': float(arrays['bbox_xyxy'][source_index, 0]),
                               'bbox_y1': float(arrays['bbox_xyxy'][source_index, 1]),
                               'bbox_x2': float(arrays['bbox_xyxy'][source_index, 2]),
                               'bbox_y2': float(arrays['bbox_xyxy'][source_index, 3]),
                               'depth_derivative_abs_m_per_scale': abs(float(z[offset] / s[offset])),
                               'weak_camera_sensitivity_mm_per_1pct_s': abs(float(z[offset])) * 10,
                               'get_trans_replay_max_abs_m': replay_error}
                        if not all(math.isfinite(v) for v in row.values() if isinstance(v, float)):
                            raise FloatingPointError('Nonfinite inverse camera diagnostic')
                        rows.append(row)
                provenance[label]['clips'].append({'path': str(path), 'sha256': file_sha256(path)})
            print(json.dumps({'input': label, 'complete_paired_states': len(coverage[label])}), flush=True)
    if any(keys != coverage[labels[0]] for keys in coverage.values()):
        raise ValueError('Inputs do not cover identical complete raw-observed reference states')
    summaries = {}
    for branch in dict.fromkeys(row['branch'] for row in rows):
        selected = [row for row in rows if row['branch'] == branch]
        summary = {'all': aggregate(selected), 'handedness': {}, 'recording': {}}
        for side in ('left', 'right'):
            subset = [row for row in selected if row['handedness'] == side]
            summary['handedness'][side] = {**aggregate(subset),
                'largest_wrist_errors': sorted(subset, key=lambda row: row['wrist_mm'], reverse=True)[:args.top]}
        for recording in sorted({row['recording'] for row in selected}):
            subset = [row for row in selected if row['recording'] == recording]
            summary['recording'][recording] = {**aggregate(subset),
                'largest_wrist_error': max(subset, key=lambda row: row['wrist_mm'])}
        summaries[branch] = summary
    rows.sort(key=lambda row: (row['recording'], row['clip_index'], row['reference_index'], row['branch']))
    with csv_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    try:
        code_revision = subprocess.check_output(['git', '-C', str(Path(__file__).resolve().parents[1]),
                                                 'rev-parse', 'HEAD'], text=True).strip()
        hawor_revision = subprocess.check_output(['git', '-C', str(hawor_root), 'rev-parse', 'HEAD'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        code_revision = hawor_revision = 'unavailable'
    report = {'schema': 'handloop.camera_depth_failure_diagnosis.v1',
        'completed_utc': datetime.now(timezone.utc).isoformat(), 'code_revision': code_revision,
        'script_sha256': file_sha256(Path(__file__)), 'source_manifest': str(manifest_path),
        'source_manifest_sha256': manifest_sha, 'source_reference_sha256': reference_sha,
        'mano_asset_sha256': asset_sha, 'inputs': provenance, 'paired_protocol': protocol,
        'paired_complete_states': len(coverage[labels[0]]), 'branches': summaries,
        'official_hawor_revision': hawor_revision, 'official_source_file_sha256': source_files,
        'official_functions': {'get_trans': camera_source, 'boxes_2_cs': box_source},
        'prediction_codec': prediction_codec.provenance(), 'reference_codec': reference_codec.provenance(),
        'maximum_official_get_trans_replay_error_m': maximum_replay_error_m,
        'formula': {
            'crop': 'b=200*scale=1.2*max(bbox_width,bbox_height); center is original unflipped bbox center; f=mean(fx,fy).',
            'forward': 'Tz=2*f/(b*s); Tx=tx+2*(cx-c0x)/(b*s); Ty=ty+2*(cy-c0y)/(b*s).',
            'wrist': 'raw anatomical J0 = MANO transl + shaped rest J0(predicted beta), in the original fixed camera gauge.',
            'inverse': 'T=raw_J0-rest_J0; s=2*f/(b*Tz); tx=Tx-(cx-c0x)*Tz/f; ty=Ty-(cy-c0y)*Tz/f.',
            'derivative': 'dTz/ds=-2*f/(b*s^2)=-Tz/s; for relative small ds/s, dTz/Tz≈-ds/s. s→0 produces unbounded depth. Crop-center offsets also multiply the same inverse denominator.',
            'left_mirror': 'The bridge unflips tx only, then restores pose with SRS. Camera s and depth are not negated by left-image mirroring. The codec left-shapedirs fix changes x components only; predicted beta can still affect all rest-wrist axes.',
        },
        'limitations': [
            'The original pred_cam tensors were not saved. Raw s is recovered by algebra under the checked bridge/get_trans contract; it is not an independently observed head tensor.',
            'Candidate/final implied s is only an effective reparameterization of their saved states; those branches are not camera-head outputs.',
            'GT implied s uses fitted reference99D/source beta and is an algebraic reference, not an independently measured weak-camera label.',
            'Official get_trans has no denominator clamp and the released linear camera readout has no positive scale activation. This explains a possible amplification mechanism, not why optimization produced a particular scale.',
            'Head-tail sign, left mirroring, and beta conventions must not be blamed solely from a depth outlier. Test whether recovered scales and axes actually support the hypothesis.',
            '1.2 crop dilation follows the current VisualFeedbackRGBDataset._crop implementation; no RGB is read, no model forward/optimizer/checkpoint change occurs.',
            'Threshold counts are descriptive diagnostics, not modified official metrics, filtering, or a decision to remove test frames.',
        ], 'per_frame_csv': str(csv_path), 'device': str(device), 'torch_version': torch.__version__}
    save_json(output, report)
    print(json.dumps({'complete': True, 'report': str(output), 'paired_complete_states': report['paired_complete_states'],
                      'replay_max_error_m': maximum_replay_error_m}), flush=True)


if __name__ == '__main__':
    main()
