"""Bounded-memory DexYCB metrics for remote full-test inference.

This module has no eager numerical/model imports. Execute numerical methods in
the compute Pod. Prediction beta and source beta always use separate codecs:
HaWoR's left shapedirs fix is enabled for predictions, disabled for references.
Only compact sums/counts/PCK threshold hits are serialized; no images, features,
meshes or trajectories are retained. The caller owns unit identity/deduplication
when resuming or merging recording/view shards.
"""
from __future__ import annotations

import copy
import math
from typing import Any


DEX_ORDER = (0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18,
             10, 11, 12, 19, 7, 8, 9, 20)
ALIGNMENTS = ('absolute', 'wrist_aligned', 'pa')
SCHEMA = 'handloop.dexycb_streaming_metrics.v1'
THRESHOLDS_MM = [index * (50.0 / 99) for index in range(100)]
THRESHOLDS_MM[-1] = 50.0


def dex_joints(geometry, right):
    """Exactly the handedness/tip mapping used by evaluate_visual_geometry.py."""
    import torch

    vertices, joints = geometry['vertices'], geometry['joints']
    tips = torch.tensor([745, 317, 444, 556, 673], device=vertices.device)
    tips = tips.expand(len(right), -1).clone()
    tips[~right.bool(), 2] = 445
    return torch.cat((joints, vertices.gather(1, tips[..., None].expand(-1, -1, 3))), 1)[:, DEX_ORDER]


def pa_joint_errors_mm(predicted, target, *, proper=False, return_info=False):
    """Per-frame similarity alignment; default matches FreiHAND's free O(3).

    Official ``align_w_scale`` uses scipy orthogonal_procrustes and permits a
    reflection. The older small-subset evaluator constrained det(R)=+1; those
    1,015 baseline fits happened to need no reflection. Full-test primary PA
    follows the official unrestricted fit, reporting reflection counts plus a
    separate proper-rotation MPJPE diagnostic instead of silently conflating it.
    """
    import torch

    x, y = predicted.double(), target.double()
    xm, ym = x.mean(1, keepdim=True), y.mean(1, keepdim=True)
    xc, yc = x - xm, y - ym
    variance = xc.square().sum((1, 2))
    if bool((variance <= 1e-20).any()):
        raise ValueError('Cannot similarity-align a degenerate predicted hand')
    u, singular, vh = torch.linalg.svd(xc.transpose(1, 2) @ yc)
    reflection = torch.linalg.det(u @ vh) < 0

    def errors(enforce_proper):
        signs = torch.ones_like(singular)
        if enforce_proper:
            signs[:, -1] = torch.where(reflection, -1.0, 1.0)
        rotation = (u * signs[:, None, :]) @ vh
        scale = (singular * signs).sum(-1) / variance
        aligned = scale[:, None, None] * (xc @ rotation) + ym
        return (aligned - y).norm(dim=-1) * 1000

    result = errors(proper)
    if return_info:
        return result, {'reflection': reflection,
                        'proper_errors_mm': result if proper else errors(True)}
    return result


def _new_bucket():
    return {'source_frames': 0, 'complete_reference_states': 0, 'branches': {}}


class StreamingDexYCBMetrics:
    """Stream exact MPJPE/MPVPE/PCK-AUC and state-error sufficient statistics.

    ``group`` in update contains scalar recording/subject/view metadata. Hand
    groups are computed from the per-frame handedness, never inferred from a
    filename. ``group_by`` can omit recording to keep a global report small;
    resume shards may each keep their recording-level report separately.
    Missing predictions are counted, never converted to successful zero error.
    Geometry/PA use complete17 reference and complete17 prediction availability.
    State metrics retain per-group availability; finger means weight rotations.
    """

    def __init__(self, prediction_codec, reference_codec, device='cpu', *,
                 batch_size=64, group_by=('hand', 'subject', 'view', 'recording'),
                 reference_tolerance_mm=1.0):
        if batch_size < 1 or reference_tolerance_mm <= 0:
            raise ValueError('batch_size and reference_tolerance_mm must be positive')
        if (getattr(prediction_codec, 'left_shapedirs_fix', None) is not True
                or getattr(reference_codec, 'left_shapedirs_fix', None) is not False):
            raise ValueError('Use prediction left_shapedirs_fix=True and reference False')
        self.prediction_codec, self.reference_codec = prediction_codec, reference_codec
        self.device, self.batch_size = device, int(batch_size)
        self.group_by = tuple(group_by)
        if len(set(self.group_by)) != len(self.group_by):
            raise ValueError('group_by fields must be unique')
        self.reference_tolerance_mm = float(reference_tolerance_mm)
        self._state = self._empty_state()

    def _empty_state(self):
        return {'schema': SCHEMA, 'thresholds_mm': THRESHOLDS_MM[:],
                'group_by': list(self.group_by), 'updates': 0,
                'groups': {'all': {'all': _new_bucket()}},
                'reference_fk_audit': {'states': 0, 'joint_observations': 0,
                    'sum_error_mm': 0.0, 'max_error_mm': 0.0,
                    'tolerance_mm': self.reference_tolerance_mm}}

    def state_dict(self):
        """JSON-serializable sufficient statistics for an atomic shard save."""
        return copy.deepcopy(self._state)

    def load_state_dict(self, state):
        self._validate_state(state)
        self._state = copy.deepcopy(state)
        return self

    def _validate_state(self, state):
        if (state.get('schema') != SCHEMA or state.get('thresholds_mm') != THRESHOLDS_MM
                or state.get('group_by') != list(self.group_by)
                or state.get('reference_fk_audit', {}).get('tolerance_mm') != self.reference_tolerance_mm):
            raise ValueError('Incompatible streaming metric protocol/state')
        if state['reference_fk_audit']['max_error_mm'] > self.reference_tolerance_mm:
            raise ValueError('Cannot import a failed reference FK audit')

    def merge_state_dict(self, state):
        """Merge DISJOINT shards only; their sample identities belong to caller."""
        self._validate_state(state)
        self._state['updates'] += state['updates']
        audit, incoming = self._state['reference_fk_audit'], state['reference_fk_audit']
        for key in ('states', 'joint_observations', 'sum_error_mm'):
            audit[key] += incoming[key]
        audit['max_error_mm'] = max(audit['max_error_mm'], incoming['max_error_mm'])
        for field, groups in state['groups'].items():
            for name, source in groups.items():
                destination = self._state['groups'].setdefault(field, {}).setdefault(name, _new_bucket())
                for key in ('source_frames', 'complete_reference_states'):
                    destination[key] += source[key]
                for branch, values in source['branches'].items():
                    target = destination['branches'].setdefault(branch, {'scalars': {}, 'pck': {}})
                    for metric, stats in values['scalars'].items():
                        total = target['scalars'].setdefault(metric, {'sum': 0.0, 'count': 0})
                        total['sum'] += stats['sum']
                        total['count'] += stats['count']
                    for alignment, stats in values['pck'].items():
                        total = target['pck'].setdefault(alignment, self._new_pck())
                        for joint in range(21):
                            total['joint_counts'][joint] += stats['joint_counts'][joint]
                            total['hits'][joint] = [a + b for a, b in zip(total['hits'][joint], stats['hits'][joint])]
                        for threshold, hits in stats['pck_at_hits'].items():
                            total['pck_at_hits'][threshold] += hits
        return self

    @staticmethod
    def _new_pck():
        return {'joint_counts': [0] * 21, 'hits': [[0] * 100 for _ in range(21)],
                'pck_at_hits': {str(t): 0 for t in (5, 10, 20, 30, 50)}}

    def update(self, predictions, beta, target_state, target_beta, target_joints,
               handedness, target_valid, availability=None, group=None, *,
               frame_valid=None, joint_valid=None, valid_by_branch=None):
        """Add one clip/batch; all tensors have an unpadded leading N dimension.

        predictions: dict[label, N x 99]; beta: N x 10 or dict[label,N x 10].
        Reference target_beta is static10 or N x10. availability applies only
        labels ``raw``/``baseline``. valid_by_branch overrides with N or N x17.
        Unknown/invalid geometries on eligible states raise rather than drop.
        Returns a compact report for JUST this update, useful in unit progress.
        """
        import numpy as np
        import torch
        from handloop import geometry as g

        if not predictions:
            raise ValueError('At least one prediction branch is required')
        device = torch.device(self.device)
        tensor = lambda x, dtype=torch.float32: torch.as_tensor(x, dtype=dtype, device=device)
        target = tensor(target_state)
        n = len(target)
        if target.shape != (n, 99) or n < 1:
            raise ValueError('Expected nonempty target_state[N,99]')
        joints, side = tensor(target_joints), tensor(handedness, torch.bool)
        target_mask = tensor(target_valid, torch.bool)
        if joints.shape != (n, 21, 3) or side.shape != (n,) or target_mask.shape != (n, 17):
            raise ValueError('Invalid target_joints/handedness/target_valid shape')
        valid = torch.ones(n, device=device, dtype=torch.bool) if frame_valid is None else tensor(frame_valid, torch.bool)
        if valid.shape != (n,):
            raise ValueError('frame_valid must be [N]')
        complete = valid & target_mask.all(-1)
        if joint_valid is not None:
            jv = tensor(joint_valid, torch.bool)
            if jv.shape != (n, 21) or bool((complete & ~jv.all(-1)).any()):
                raise ValueError('Complete DexYCB states require all21 supplied reference joints')
        reference_beta = tensor(target_beta)
        if reference_beta.shape == (10,):
            reference_beta = reference_beta.expand(n, -1)
        if reference_beta.shape != (n, 10):
            raise ValueError('target_beta must be [10] or [N,10]')
        for value in (target[complete], reference_beta[complete], joints[complete]):
            if not bool(torch.isfinite(value).all()):
                raise FloatingPointError('Nonfinite eligible reference')
        group = group or {}
        masks = [('all', 'all', valid.detach().cpu().numpy())]
        for field in self.group_by:
            if field == 'hand':
                right = side.detach().cpu().numpy()
                masks += [('hand', 'left', (~right) & valid.detach().cpu().numpy()),
                          ('hand', 'right', right & valid.detach().cpu().numpy())]
            elif field in group:
                masks.append((field, str(group[field]), valid.detach().cpu().numpy()))
        local = self._empty_state()
        local['updates'] = 1
        complete_np = complete.detach().cpu().numpy()
        for field, name, mask in masks:
            if mask.any():
                bucket = local['groups'].setdefault(field, {}).setdefault(name, _new_bucket())
                bucket['source_frames'] = int(mask.sum())
                bucket['complete_reference_states'] = int((mask & complete_np).sum())

        def record(branch, scalar_values, scalar_masks, distances, geometry_mask):
            for field, name, group_mask in masks:
                if not group_mask.any():
                    continue
                bucket = local['groups'][field][name]['branches'].setdefault(branch, {'scalars': {}, 'pck': {}})
                for key, values in scalar_values.items():
                    use = scalar_masks[key] & group_mask
                    selected = values[use]
                    if not np.isfinite(selected).all():
                        raise FloatingPointError('Nonfinite eligible metric: ' + key)
                    bucket['scalars'][key] = {'sum': float(selected.sum(dtype=np.float64)), 'count': int(selected.size)}
                use = geometry_mask & group_mask
                for alignment, errors in distances.items():
                    selected = errors[use]
                    if not len(selected):
                        continue
                    stats = self._new_pck()
                    stats['joint_counts'] = [len(selected)] * 21
                    stats['hits'] = (selected[..., None] <= np.asarray(THRESHOLDS_MM)).sum(axis=0).tolist()
                    stats['pck_at_hits'] = {str(t): int((selected <= t).sum()) for t in (5, 10, 20, 30, 50)}
                    bucket['pck'][alignment] = stats

        with torch.no_grad():
            reference_vertices = torch.empty((n, 778, 3), device=device)
            indices = complete.nonzero().flatten()
            for begin in range(0, len(indices), self.batch_size):
                take = indices[begin:begin + self.batch_size]
                geometry = self.reference_codec.geometry(target[take], reference_beta[take], side[take])
                error = (dex_joints(geometry, side[take]) - joints[take]).norm(dim=-1).double() * 1000
                if not bool(torch.isfinite(error).all()):
                    raise FloatingPointError('Nonfinite source-beta reference FK audit')
                maximum = float(error.max())
                if maximum > self.reference_tolerance_mm:
                    raise ValueError(f'Reference FK/Dex21 mismatch {maximum:.6f} mm > {self.reference_tolerance_mm} mm; group={group}')
                audit = local['reference_fk_audit']
                audit['states'] += len(take)
                audit['joint_observations'] += error.numel()
                audit['sum_error_mm'] += float(error.sum())
                audit['max_error_mm'] = max(audit['max_error_mm'], maximum)
                reference_vertices[take] = geometry['vertices']

            for label, value in predictions.items():
                state = tensor(value)
                shape = tensor(beta[label] if isinstance(beta, dict) else beta)
                if state.shape != (n, 99) or shape.shape != (n, 10):
                    raise ValueError('Predictions and own beta must be [N,99] and [N,10]')
                branch_mask = (valid_by_branch or {}).get(label)
                if branch_mask is None and label in ('raw', 'baseline'):
                    branch_mask = availability
                available = (torch.ones((n, 17), dtype=torch.bool, device=device)
                             if branch_mask is None else tensor(branch_mask, torch.bool))
                if available.shape == (n,):
                    available = available[:, None].expand(-1, 17)
                if available.shape != (n, 17):
                    raise ValueError('Branch availability must be [N] or [N,17]')
                geometry_mask = complete & available.all(-1)
                safe = state.clone()
                safe[..., :3] = torch.where(available[..., :1], state[..., :3], 0.)
                safe[..., 3:] = torch.where(available[..., 1:, None], state[..., 3:].reshape(n, 16, 6),
                                          g.identity_rotation_6d(device=device)).flatten(-2)
                angles = g.geodesic_angle(g.state_to_rotmat(safe), g.state_to_rotmat(target)) * (180 / math.pi)
                scalar_values = {
                    'wrist_mm': (safe[..., :3] - target[..., :3]).norm(dim=-1) * 1000,
                    'root_deg': angles[..., 0], 'finger_deg': angles[..., 1:]}
                scalar_masks = {'wrist_mm': complete & available[..., 0],
                    'root_deg': complete & available[..., 1],
                    'finger_deg': complete[:, None] & available[..., 2:]}
                # Flatten finger samples only after group selection, preserving
                # per-finger weighting while all geometric metrics count states.
                distances = {name: np.full((n, 21), np.nan) for name in ALIGNMENTS}
                for key in ('mpjpe21_mm', 'wrist_aligned_mpjpe21_mm', 'pa_mpjpe21_mm',
                            'pa_proper_mpjpe21_mm', 'pa_reflection_fraction',
                            'mpvpe778_mm', 'wrist_aligned_mpvpe778_mm'):
                    scalar_values[key] = torch.full((n,), torch.nan, device=device, dtype=torch.float64)
                    scalar_masks[key] = geometry_mask
                selected = geometry_mask.nonzero().flatten()
                for begin in range(0, len(selected), self.batch_size):
                    take = selected[begin:begin + self.batch_size]
                    if not bool(torch.isfinite(state[take]).all() and torch.isfinite(shape[take]).all()):
                        raise FloatingPointError('Nonfinite eligible prediction state/beta: ' + label)
                    geometry = self.prediction_codec.geometry(state[take], shape[take], side[take])
                    predicted_joints = dex_joints(geometry, side[take])
                    delta = predicted_joints - joints[take]
                    vertex_delta = geometry['vertices'] - reference_vertices[take]
                    wrist_delta = delta[:, :1]
                    pa_errors, pa_info = pa_joint_errors_mm(predicted_joints, joints[take], return_info=True)
                    errors = {'absolute': delta.norm(dim=-1) * 1000,
                              'wrist_aligned': (delta - wrist_delta).norm(dim=-1) * 1000,
                              'pa': pa_errors}
                    for alignment, metric in zip(ALIGNMENTS, ('mpjpe21_mm', 'wrist_aligned_mpjpe21_mm', 'pa_mpjpe21_mm')):
                        values = errors[alignment]
                        if not bool(torch.isfinite(values).all()):
                            raise FloatingPointError('Nonfinite decoded geometry: ' + label)
                        scalar_values[metric][take] = values.double().mean(-1)
                        distances[alignment][take.cpu().numpy()] = values.cpu().numpy()
                    scalar_values['mpvpe778_mm'][take] = vertex_delta.norm(dim=-1).double().mean(-1) * 1000
                    scalar_values['wrist_aligned_mpvpe778_mm'][take] = (vertex_delta - wrist_delta).norm(dim=-1).double().mean(-1) * 1000
                    scalar_values['pa_proper_mpjpe21_mm'][take] = pa_info['proper_errors_mm'].mean(-1)
                    scalar_values['pa_reflection_fraction'][take] = pa_info['reflection'].double()
                values_np = {key: value.double().cpu().numpy() for key, value in scalar_values.items()}
                masks_np = {key: value.cpu().numpy() for key, value in scalar_masks.items()}
                # record() broadcasts group membership over the finger axis.
                for field, name, mask in masks:
                    if not mask.any():
                        continue
                    bucket = local['groups'][field][name]['branches'].setdefault(label, {'scalars': {}, 'pck': {}})
                    finger_use = masks_np['finger_deg'] & mask[:, None]
                    samples = values_np['finger_deg'][finger_use]
                    if not np.isfinite(samples).all():
                        raise FloatingPointError('Nonfinite eligible finger metric')
                    bucket['scalars']['finger_deg'] = {'sum': float(samples.sum(dtype=np.float64)), 'count': int(samples.size)}
                del values_np['finger_deg'], masks_np['finger_deg']
                record(label, values_np, masks_np, distances, geometry_mask.cpu().numpy())
        self.merge_state_dict(local)
        return self._snapshot(local)

    @staticmethod
    def _snapshot(state):
        import numpy as np

        report = {'schema': SCHEMA, 'updates': state['updates'], 'groups': {},
                  'auc_protocol': '21 per-joint PCK(error<=threshold); 100 inclusive thresholds 0--50mm; trapezoid /50 then equal-joint mean.',
                  'pa_protocol': 'Primary pa permits reflections, matching FreiHAND scipy orthogonal_procrustes; pa_proper_mpjpe21_mm is det+1 diagnostic. Reflection fits counted explicitly.',
                  'reference_mesh_scope': 'Source MANO fitted surface; not an independent scan.',
                  'reference_fk_audit': copy.deepcopy(state['reference_fk_audit'])}
        audit = report['reference_fk_audit']
        audit['mean_error_mm'] = audit['sum_error_mm'] / audit['joint_observations'] if audit['joint_observations'] else None
        audit['passed'] = audit['max_error_mm'] <= audit['tolerance_mm']
        thresholds = np.asarray(THRESHOLDS_MM, dtype=np.float64)
        for field, groups in state['groups'].items():
            report['groups'][field] = {}
            for name, bucket in groups.items():
                result = {'source_frames': bucket['source_frames'],
                          'complete_reference_states': bucket['complete_reference_states'], 'branches': {}}
                for branch, values in bucket['branches'].items():
                    scalars = {key: {'mean': stats['sum'] / stats['count'] if stats['count'] else None,
                                     'count': stats['count'],
                                     'count_unit': 'finger_rotations' if key == 'finger_deg' else 'states'}
                               for key, stats in values['scalars'].items()}
                    aucs = {}
                    for alignment, stats in values['pck'].items():
                        counts = np.asarray(stats['joint_counts'], dtype=np.int64)
                        if (counts <= 0).any():
                            raise ValueError('All21 joints must have observations in complete-state Dex evaluation')
                        curve = np.asarray(stats['hits'], dtype=np.float64) / counts[:, None]
                        area = ((curve[:, 1:] + curve[:, :-1]) * .5 * np.diff(thresholds)).sum(-1) / 50
                        total = int(counts.sum())
                        aucs[alignment] = {'auc': float(area.mean()), 'auc_percent': float(area.mean() * 100),
                            'valid_states': int(counts[0]), 'joint_observations': total,
                            'thresholds_mm': THRESHOLDS_MM[:], 'pck_curve': curve.mean(0).tolist(),
                            'pck_at_mm': {key: hits / total for key, hits in stats['pck_at_hits'].items()}}
                    geometry_count = scalars.get('mpjpe21_mm', {}).get('count', 0)
                    result['branches'][branch] = {'metrics': scalars, 'pck_auc': aucs,
                        'covered_counts': {'geometry_states': geometry_count,
                            'missing_geometry_states': bucket['complete_reference_states'] - geometry_count,
                            'pa_reflection_fits': int(round(values['scalars'].get('pa_reflection_fraction', {}).get('sum', 0)))}}
                report['groups'][field][name] = result
        report['all'] = report['groups']['all']['all']
        return report

    def snapshot(self):
        return self._snapshot(self._state)

    def wandb_scalars(self, prefix='test'):
        """Flat all-sample means/counts for ordinary W&B history logging."""
        # Avoid rendering hundreds of recording curves at every W&B update.
        all_only = {**self._state, 'groups': {'all': self._state['groups']['all']}}
        all_values = self._snapshot(all_only)['all']
        result = {f'{prefix}/source_frames': all_values['source_frames'],
                  f'{prefix}/complete_reference_states': all_values['complete_reference_states']}
        for branch, values in all_values['branches'].items():
            for metric, stats in values['metrics'].items():
                if stats['mean'] is not None:
                    result[f'{prefix}/{branch}/{metric}'] = stats['mean']
                    result[f'{prefix}/{branch}/{metric}_count'] = stats['count']
            for alignment, stats in values['pck_auc'].items():
                result[f'{prefix}/{branch}/{alignment}_auc_percent'] = stats['auc_percent']
            for key, count in values['covered_counts'].items():
                result[f'{prefix}/{branch}/{key}'] = count
        return result
