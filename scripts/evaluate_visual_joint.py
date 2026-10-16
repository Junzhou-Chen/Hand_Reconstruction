#!/usr/bin/env python3
"""Remote-only paired state evaluation for official HaWoR and joint checkpoints.

Run baseline and joint in separate processes/output directories. Both use the
same official crops/16-frame blocks and 32-frame reset policy as training.
Default evaluation traverses every validation clip and additionally reports
the exact fixed-eight training subset, without repeating model inference.
These are wrist-position/local-rotation metrics, NOT MPJPE or mesh errors.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
from contextlib import nullcontext
import csv
from datetime import datetime, timezone
import gc
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from torch.utils.data import default_collate

from handloop import geometry as g
from handloop.data.visual_clips import file_sha256
from handloop.data.visual_rgb import SOURCE_SCHEMA, VisualFeedbackRGBDataset
from handloop.models.probabilistic_twm import (
    ProbabilisticTWM, ProbabilisticTWMConfig, current_distribution_log_prob,
)
from handloop.models.visual_feedback import (
    VisualFeedbackConfig, VisualFeedbackSystem, inverse_rigid, transform_state,
)
from handloop.visual.hawor import FrozenHaWoRBridge
from handloop.visual.pipeline import HandLoopFeedbackPipeline
from train_probabilistic_twm import state_geometry_loss
from train_visual_feedback import metrics


def utc():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    tmp.replace(path)


class EvaluationRGBDataset(VisualFeedbackRGBDataset):
    """Evaluation-only split reader; training's deliberate test ban stays intact.

    Crop, reference validation, padding, gap segmentation and reset behavior
    reuse the actual training loader. Only split parsing also accepts test.
    No manifest is rewritten and no target or GT beta is added to inputs.
    """
    def __init__(self, source_manifest, split, *, clip_length, hawor_root):
        if split not in ('val', 'test'):
            raise ValueError('Evaluation only accepts val or held-out test')
        if clip_length < 16 or clip_length % 16:
            raise ValueError('clip_length must be a positive multiple of 16')
        self.path = Path(source_manifest).expanduser().resolve()
        self.manifest = json.loads(self.path.read_text())
        if self.manifest.get('schema') != SOURCE_SCHEMA or self.manifest.get('phase') != 'complete':
            raise ValueError('Use a completed visual-feedback source manifest')
        if self.manifest.get('focal_convention') != 'official HaWoR single focal uses mean(fx,fy); real K retained for geometry':
            raise ValueError('Source focal convention must preserve actual K and mean(fx,fy)')
        self.manifest_sha256 = file_sha256(self.path)
        self.clip_length = self.stride = int(clip_length)
        self.hawor_root = str(Path(hawor_root).expanduser().resolve())
        self.max_open_records = 4
        self.records, self.windows = [], []
        self._arrays, self._reference_identities = OrderedDict(), {}
        seen = set()
        for record in self.manifest.get('records', []):
            recording = record.get('recording')
            if not recording or recording in seen or record.get('split') not in ('train', 'val', 'test'):
                raise ValueError('Source recordings must be unique and split-disjoint')
            seen.add(recording)
            if not record.get('coordinate_frame') or not record.get('timestamp_source'):
                raise ValueError('Source coordinate/timestamp provenance is required')
            if record['split'] != split:
                continue
            n = int(record['frames'])
            if n < 1 or len(record['image_paths']) != n:
                raise ValueError('Source RGB/reference frame counts disagree')
            index = len(self.records)
            self.records.append(record)
            arrays = self._load_reference(index, initial=True)
            boundaries = np.concatenate(([0], np.flatnonzero(np.diff(arrays['frame_ids']) != 1) + 1, [n]))
            for begin, end in zip(boundaries[:-1], boundaries[1:]):
                for start in range(int(begin), int(end), self.stride):
                    self.windows.append((index, start, min(self.clip_length, int(end) - start)))
        if not self.windows:
            raise ValueError(f'No {split} RGB clips')


def fixed_indices(length, count):
    n = min(length, count)
    return sorted(set(round(i * (length - 1) / max(1, n - 1)) for i in range(n)))


def observe(frontend, batch):
    # Identical explicit frontend allowlist and official block order to pipeline.
    names = ('img', 'center', 'center_unflipped', 'scale', 'img_focal', 'img_center',
             'is_right', 'camera_to_world', 'detected', 'bbox_valid', 'frame_valid', 'image_valid')
    parts = {}
    for start in range(0, batch['img'].shape[1], 16):
        for key, value in frontend({key: batch[key][:, start:start + 16] for key in names}).items():
            if key in ('raw_state_world', 'raw_available', 'frame_valid', 'beta'):
                parts.setdefault(key, []).append(value)
    return {key: torch.cat(values, 1) for key, values in parts.items()}


def raw_metrics(raw, availability, frame_valid, target, target_valid):
    valid = frame_valid.bool() & target_valid.bool().all(-1)
    result, counts = {}, {}
    safe = g.identity_state(*valid.shape, device=raw.device)
    safe[..., :3] = torch.where(availability[..., :1], raw[..., :3], 0.)
    safe[..., 3:] = torch.where(availability[..., 1:, None], raw[..., 3:].reshape(*valid.shape, 16, 6),
                                g.identity_rotation_6d(device=raw.device)).flatten(-2)
    angles = g.geodesic_angle(g.state_to_rotmat(safe), g.state_to_rotmat(target)) * (180 / math.pi)
    values = {'raw/wrist_mm': ((safe[..., :3] - target[..., :3]).norm(dim=-1) * 1000,
                              valid & availability[..., 0]),
              'raw/root_deg': (angles[..., 0], valid & availability[..., 1]),
              'raw/finger_deg': (angles[..., 1:], valid[..., None] & availability[..., 2:])}
    for key, (value, mask) in values.items():
        counts[key] = int(mask.sum())
        if counts[key]:
            result[key] = float(value[mask].mean())
    return {**result, 'metric_counts': counts, 'valid_states': int(valid.sum())}


def summarize(clips):
    sums, counts = {}, {}
    for clip in clips:
        value = clip['metrics']
        for key, count in value['metric_counts'].items():
            if count and key in value:
                counts[key] = counts.get(key, 0) + count
                sums[key] = sums.get(key, 0.) + value[key] * count
    return {**{key: sums[key] / counts[key] for key in sums},
            'metric_counts': counts, 'valid_states': sum(c['metrics']['valid_states'] for c in clips),
            'clips': len(clips)}


def grouped_summary(clips, field):
    groups = {}
    for clip in clips:
        groups.setdefault(str(clip[field]), []).append(clip)
    by_group = {name: summarize(values) for name, values in groups.items()}
    keys = set().union(*(set(v['metric_counts']) for v in by_group.values())) if by_group else set()
    macro, eligible = {}, {}
    for key in sorted(keys):
        values = [v[key] for v in by_group.values() if v['metric_counts'].get(key, 0)]
        if values:
            macro[key], eligible[key] = sum(values) / len(values), len(values)
    return {'by_group': by_group, 'macro_equal_group': macro, 'macro_eligible_groups': eligible}


def load_model(args, device):
    checkpoint = None
    if args.mode == 'joint':
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
        if not {'identity', 'twm_source', 'replay_config', 'model', 'config', 'step'} <= checkpoint.keys():
            raise ValueError('Expected a full joint checkpoint')
        if args.clip_length != checkpoint['config']['clip_length']:
            raise ValueError('Keep the trained clip length for a paired comparison')
        identity = checkpoint['identity']
        if args.split == 'val' and file_sha256(args.source_manifest) != identity['source_manifest_sha256']:
            raise ValueError('Validation manifest must match the source used for checkpoint selection')
        if file_sha256(args.hawor_checkpoint) != identity['hawor_checkpoint_sha256']:
            raise ValueError('Official initialization checkpoint differs from training')
        if file_sha256(args.hawor_config) != identity['hawor_config_sha256']:
            raise ValueError('HaWoR architecture config differs from training')
        for side in ('LEFT', 'RIGHT'):
            if file_sha256(args.mano_models / f'MANO_{side}.pkl') != identity['mano_asset_sha256'][side]:
                raise ValueError('MANO assets differ from training')
    # CPU construction avoids a second official model or initialization state on GPU.
    frontend = FrozenHaWoRBridge.from_pretrained(args.hawor_root, args.hawor_checkpoint,
        args.hawor_config, args.mano_models, device='cpu').eval()
    info = {'frontend_provenance': frontend.provenance()}
    if checkpoint is None:
        return frontend.to(device).eval().requires_grad_(False), info
    twm = ProbabilisticTWM(ProbabilisticTWMConfig(**checkpoint['twm_source']['model_config']))
    feedback = VisualFeedbackSystem(twm, VisualFeedbackConfig(**checkpoint['config']['adapter']))
    model = HandLoopFeedbackPipeline(frontend, feedback)
    prefix = 'pipeline.'
    if not all(name.startswith(prefix) for name in checkpoint['model']):
        raise ValueError('Unexpected full joint parameter namespace')
    model.load_state_dict({name[len(prefix):]: value for name, value in checkpoint['model'].items()}, strict=True)
    info.update(checkpoint_step=int(checkpoint['step']), training_config=checkpoint['config'],
                training_identity=checkpoint['identity'], twm_source=checkpoint['twm_source'],
                replay_config=checkpoint['replay_config'], checkpoint_code_version=checkpoint.get('code_version'),
                best_geometry_score=checkpoint.get('best_geometry_score'),
                saved_rank_model_sha256=[r.get('model_sha256') for r in checkpoint.get('rank_states', [])])
    del checkpoint
    gc.collect()
    return model.to(device).eval().requires_grad_(False), info


def frame_rows(batch, output, clip, arrays, start, count):
    device = batch['target_world'].device
    target, valid = batch['target_world'], batch['target_valid'].all(-1) & output['frame_valid']
    states = {'raw': output['raw_world']}
    if 'state_world' in output:
        states.update(candidate=output['candidate_world'], final=output['state_world'])
    values = {}
    for label, state in states.items():
        available = output['availability'] if label == 'raw' else torch.ones_like(output['availability'])
        safe = state.clone()
        safe[..., :3] = torch.where(available[..., :1], state[..., :3], 0.)
        safe[..., 3:] = torch.where(available[..., 1:, None], state[..., 3:].reshape(*valid.shape, 16, 6),
                                    g.identity_rotation_6d(device=device)).flatten(-2)
        angles = g.geodesic_angle(g.state_to_rotmat(safe), g.state_to_rotmat(target)) * (180 / math.pi)
        for key, number, mask in (
            ('wrist_mm', (safe[..., :3] - target[..., :3]).norm(dim=-1) * 1000, valid & available[..., 0]),
            ('root_deg', angles[..., 0], valid & available[..., 1]),
            ('finger_deg', (angles[..., 1:] * available[..., 2:]).sum(-1) / available[..., 2:].sum(-1).clamp_min(1),
             valid & available[..., 2:].any(-1)),
            ('finger_count', available[..., 2:].sum(-1), valid & available[..., 2:].any(-1))):
            values[label + '_' + key] = (number[0].detach().cpu(), mask[0].detach().cpu())
    if 'state_world' in output and bool(valid.any()):
        target_anchor = transform_state(target, inverse_rigid(output['anchor_world']))
        standardized = (target_anchor[..., :3] - output['state'][..., :3]) / output['state_translation_std']
        mahal = standardized.square().sum(-1)
        distributions = {key: output[key][valid] for key in ('state', 'state_translation_std', 'state_rotation_concentration')}
        nll = torch.zeros_like(mahal)
        nll[valid] = -current_distribution_log_prob(distributions, target_anchor[valid])['total']
        geometry = state_geometry_loss(output['state_world'], target)
        for key, number in {'current_nll': nll, 'geometry_score': geometry,
                            'mahalanobis_sq': mahal,
                            'xyz_90pct_ellipsoid_coverage': (mahal <= 6.251388631170325).float(),
                            'xyz_std_mm': output['state_translation_std'].mean(-1) * 1000,
                            'rotation_concentration_mean': output['state_rotation_concentration'].mean(-1)}.items():
            values[key] = (number[0].detach().cpu(), valid[0].detach().cpu())
        for axis, name in enumerate(('x', 'y', 'z')):
            values['anchor_std_' + name + '_mm'] = (output['state_translation_std'][0, :, axis].detach().cpu() * 1000,
                                                   valid[0].detach().cpu())
    rows = []
    for index in range(count):
        row = {'clip_index': clip['clip_index'], 'recording': clip['recording'], 'hand': clip['hand'],
               'reference_index': start + index, 'frame_id': int(arrays['frame_ids'][start + index]),
               'timestamp_seconds': float(arrays['timestamps'][start + index]),
               'frame_valid': int(output['frame_valid'][0, index]), 'complete_reference_valid': int(valid[0, index]),
               'target_valid_17': ''.join(str(int(v)) for v in batch['target_valid'][0, index].tolist()),
               'raw_available_17': ''.join(str(int(v)) for v in output['availability'][0, index].tolist())}
        for key, (number, mask) in values.items():
            row[key] = float(number[index]) if bool(mask[index]) else None
            if row[key] is not None and not math.isfinite(row[key]):
                raise FloatingPointError('Nonfinite exported metric: ' + key)
        if 'final_wrist_mm' in row:
            row['final_vs_raw_wrist_degradation_over_1mm'] = (
                int(row['final_wrist_mm'] - row['raw_wrist_mm'] > 1.)
                if row['final_wrist_mm'] is not None and row['raw_wrist_mm'] is not None else None)
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('baseline', 'joint'), required=True)
    for key in ('source-manifest', 'hawor-root', 'hawor-checkpoint', 'hawor-config', 'mano-models', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--training-manifest', type=Path,
                        help='Required for test: audit recording disjointness against actual joint training source')
    parser.add_argument('--split', choices=('val', 'test'), default='val')
    parser.add_argument('--clip-length', type=int, default=32)
    parser.add_argument('--fixed-clips', type=int, default=8)
    parser.add_argument('--indices', help='Optional comma-separated clip indices, e.g. 0 for a smoke check')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--precision', choices=('bf16', 'fp32'), default='bf16')
    parser.add_argument('--cpu-threads', type=int, default=2)
    parser.add_argument('--save-predictions', action='store_true', help='Remote numerical tensors only; never RGB/features')
    args = parser.parse_args()
    if not Path('/sciclone/home/jchen57').is_dir():
        parser.error('Run model evaluation only in the authorized remote ds Pod')
    if args.mode == 'joint' and not args.checkpoint:
        parser.error('--mode joint requires --checkpoint')
    if args.mode == 'baseline' and args.checkpoint:
        parser.error('Baseline uses official HaWoR weights only; omit --checkpoint')
    if args.split == 'test' and not args.training_manifest:
        parser.error('--split test requires --training-manifest for a leakage audit')
    if args.fixed_clips < 1:
        parser.error('--fixed-clips must be positive')
    if args.output.exists():
        raise ValueError('Use a new output directory')
    args.output.mkdir(parents=True)
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
    if args.precision == 'bf16' and device.type != 'cuda':
        raise ValueError('BF16 evaluation requires CUDA; CPU use --precision fp32')
    dataset = EvaluationRGBDataset(args.source_manifest, args.split, clip_length=args.clip_length, hawor_root=args.hawor_root)
    fixed = fixed_indices(len(dataset), args.fixed_clips)
    indices = list(range(len(dataset))) if args.indices is None else [int(v) for v in args.indices.split(',')]
    if len(set(indices)) != len(indices) or not indices or min(indices) < 0 or max(indices) >= len(dataset):
        raise ValueError('Clip indices must be unique and in range')
    model, model_info = load_model(args, device)
    training_split_audit = None
    if args.training_manifest:
        training = json.loads(args.training_manifest.read_text())
        digest = file_sha256(args.training_manifest)
        if args.mode == 'joint' and digest != model_info['training_identity']['source_manifest_sha256']:
            raise ValueError('Training manifest does not match checkpoint identity')
        forbidden = {r['recording'] for r in training['records'] if r['split'] == 'train' or (args.split == 'test' and r['split'] == 'val')}
        overlap = forbidden & {r['recording'] for r in dataset.records}
        if overlap:
            raise ValueError('Evaluation recordings overlap training/selection recordings: ' + repr(sorted(overlap)))
        training_split_audit = {'manifest': str(args.training_manifest.resolve()), 'sha256': digest,
                                'forbidden_recordings': sorted(forbidden), 'overlap': []}
    try:
        revision = subprocess.check_output(['git', '-C', str(Path(__file__).resolve().parents[1]), 'rev-parse', 'HEAD'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = 'unavailable'
    provenance = {'schema': 'handloop.visual_joint_evaluation.v1', 'started_utc': utc(),
        'mode': args.mode, 'split': args.split, 'source_manifest': str(args.source_manifest.resolve()),
        'source_manifest_sha256': dataset.manifest_sha256,
        'training_split_audit': training_split_audit,
        'checkpoint': str(args.checkpoint.resolve()) if args.checkpoint else None,
        'checkpoint_sha256': file_sha256(args.checkpoint) if args.checkpoint else None,
        'official_hawor_checkpoint_sha256': file_sha256(args.hawor_checkpoint),
        'hawor_config_sha256': file_sha256(args.hawor_config),
        'mano_asset_sha256': {s: file_sha256(args.mano_models / f'MANO_{s}.pkl') for s in ('LEFT', 'RIGHT')},
        'evaluation_code_revision': revision, 'evaluation_script_sha256': file_sha256(Path(__file__)),
        'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'dataset_clips': len(dataset), 'evaluated_indices': indices, 'training_fixed_indices': fixed,
        'reference_identities': {str(i): {'recording': dataset.records[i]['recording'], 'size_bytes': v[0], 'sha256': v[2]}
                                 for i, v in dataset._reference_identities.items()},
        'records': [{k: v for k, v in r.items() if k != 'image_paths'} for r in dataset.records],
        'units': {'state_position': 'metres', 'wrist_error': 'millimetres', 'rotation_error': 'degrees'},
        'metric_definition': 'Unaligned anatomical wrist L2; global root geodesic; 15 local finger geodesics. Not MPJPE.',
        'metric_reduction': 'Complete-reference states only; raw per-group availability; finger rotations counted individually.',
        'history_policy': 'Empty history per clip; official bidirectional 16-frame blocks; final-only chronological history commit within clip.',
        'input_contract': 'Only predicted HaWoR beta and explicit RGB/camera/mask inputs; GT targets used exclusively after forward.',
        'coverage_definition': '90% anchor-space Gaussian xyz ellipsoid, chi-square(3) threshold 6.251388631170325; descriptors are not calibrated q.',
        'scope': 'Uses supplied source crops/cameras and fitted MANO references; not an automatic detector/SLAM or official paper benchmark.',
        'torch_version': torch.__version__, 'cuda_version': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        **model_info}
    atomic_json(args.output / 'provenance.json', provenance)
    clips, rows = [], []
    (args.output / 'clips').mkdir()
    with torch.no_grad():
        for index in indices:
            record_index, start, count = dataset.windows[index]
            record, arrays = dataset.records[record_index], dataset._load_reference(record_index)
            batch = default_collate([dataset[index]])
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            context = torch.autocast('cuda', dtype=torch.bfloat16) if args.precision == 'bf16' else nullcontext()
            with context:
                if args.mode == 'joint':
                    output = model(batch, return_forecast=False)
                    measured = metrics(output, {**batch, 'raw_world': output['raw_world'], 'availability': output['availability']})
                else:
                    observed = observe(model, batch)
                    output = {'raw_world': observed['raw_state_world'], 'availability': observed['raw_available'],
                              'frame_valid': observed['frame_valid'], 'beta': observed['beta']}
                    measured = raw_metrics(output['raw_world'], output['availability'], output['frame_valid'], batch['target_world'], batch['target_valid'])
                clip = {'clip_index': index, 'recording': record['recording'], 'record_index': record_index,
                        'hand': 'right' if arrays['handedness'][start] else 'left', 'reference_start': start,
                        'unpadded_frames': count, 'frame_ids': [int(v) for v in arrays['frame_ids'][start:start + count]],
                        'coordinate_frame': record['coordinate_frame'], 'timestamp_source': record['timestamp_source'],
                        'training_fixed_subset': index in fixed, 'metrics': measured}
                local_rows = frame_rows(batch, output, clip, arrays, start, count)
                if args.mode == 'joint':
                    valid_rows = [r for r in local_rows if r['complete_reference_valid']]
                    for key in ('current_nll', 'geometry_score'):
                        if valid_rows:
                            measured[key] = sum(r[key] for r in valid_rows) / len(valid_rows)
                            measured['metric_counts'][key] = len(valid_rows)
            # Float32/CPU JSON exports are numerical only; keep RGB and features out.
            rows.extend(local_rows)
            clips.append(clip)
            atomic_json(args.output / 'clips' / f'{index:05d}.json', clip)
            if args.save_predictions:
                keys = ('raw_world', 'candidate_world', 'state_world', 'beta', 'availability', 'frame_valid',
                        'state_translation_std', 'state_translation_covariance_world', 'state_rotation_concentration',
                        'anchor_world', 'refinement_states_world', 'refinement_candidates_world')
                payload = {k: output[k].detach().cpu() for k in keys if k in output}
                payload.update(target_world=batch['target_world'].detach().cpu(),
                               target_valid=batch['target_valid'].detach().cpu(),
                               handedness=batch['is_right'].detach().cpu(),
                               source_manifest=str(dataset.path), source_manifest_sha256=dataset.manifest_sha256,
                               source_record=dict(record),
                               record_index=record_index, clip_index=index, reference_start=start,
                               reference_indices=torch.arange(start, start + count, dtype=torch.int64),
                               source_frame_ids=torch.as_tensor(arrays['frame_ids'][start:start + count].copy()),
                               unpadded_frames=count)
                torch.save(payload, args.output / 'clips' / f'{index:05d}.pt')
            atomic_json(args.output / 'status.json', {'updated_utc': utc(), 'phase': 'evaluating', 'completed_clips': len(clips), 'total_clips': len(indices)})
            print(json.dumps({'clip': index, 'completed': len(clips), 'of': len(indices), 'valid_states': measured['valid_states']}, sort_keys=True), flush=True)
            del output, batch
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (args.output / 'per_frame.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    fixed_clips = [clip for clip in clips if clip['training_fixed_subset']]
    report = {'schema': provenance['schema'], 'completed_utc': utc(), 'mode': args.mode,
              'checkpoint_step': model_info.get('checkpoint_step'),
              'source_manifest_sha256': dataset.manifest_sha256, 'evaluated_indices': indices,
              'all': summarize(clips), 'fixed_subset': summarize(fixed_clips),
              'fixed_subset_complete': {c['clip_index'] for c in fixed_clips} == set(fixed),
              'recording': grouped_summary(clips, 'recording'), 'hand': grouped_summary(clips, 'hand'),
              'clips': clips, 'per_frame_csv': str((args.output / 'per_frame.csv').resolve())}
    atomic_json(args.output / 'report.json', report)
    atomic_json(args.output / 'status.json', {'updated_utc': utc(), 'phase': 'complete', 'completed_clips': len(clips), 'total_clips': len(indices)})
    print(json.dumps({'complete': True, 'report': str(args.output / 'report.json'), 'all': report['all']}, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
