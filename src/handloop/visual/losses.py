"""External, metre-based visual supervision; no reference enters inference.

The existing full-state final NLL is unchanged. Two independently configurable
auxiliaries supervise raw anatomical wrist J0 and final absolute landmarks.
Reference beta is deliberately not an argument: predictions decode with their
own beta and the prediction codec's left-hand convention.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import Tensor
import torch.distributed as dist

from handloop.models.visual_feedback import feedback_current_nll
from .codec import MANOStateCodec, SUPPORTED_REFERENCE_JOINT_ORDERS


@dataclass(frozen=True)
class VisualGeometryLossConfig:
    """Geometric metre distances divide by explicit scales before weighting.

    Zeros preserve the previous training objective. Positive weights are
    experiment settings requiring validation, not calibrated physical values.
    No uncertainty weighting, alignment, robust clipping, or source-beta L2 is
    performed. Defaults retain the old NLL coefficient and metre units.
    """
    nll_weight: float = 1.0
    raw_wrist_weight: float = 0.0
    final_joint_weight: float = 0.0
    raw_wrist_scale_m: float = 1.0
    final_joint_scale_m: float = 1.0

    def __post_init__(self):
        for name, value in vars(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f'{name} must be a finite numeric value')
            if (value <= 0 if name.endswith('_scale_m') else value < 0):
                raise ValueError(f'{name} must be positive (scales) or nonnegative (weights)')

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None):
        return cls(**(value or {}))


def _empty_sum(value: Tensor) -> Tensor:
    # Unlike sum()*0, an empty selection cannot turn masked NaNs into NaNs.
    return value.reshape(-1)[:0].float().sum()


def visual_loss_coefficients(config: VisualGeometryLossConfig) -> dict[str, float]:
    """Effective scalar multipliers exposed for train-only gradient probes."""
    return {'current_nll': config.nll_weight,
            'raw_wrist_geometry_m': config.raw_wrist_weight / config.raw_wrist_scale_m,
            'final_mpjpe21_m': config.final_joint_weight / config.final_joint_scale_m}


def local_visual_term_means(terms: dict[str, tuple[Tensor, Tensor]]) -> dict[str, Tensor]:
    """Unweighted local means for no-optimizer, per-term autograd probes.

    These are deliberately not an alternative DDP training reduction. A
    caller can inspect named parameter subsets without taking an update.
    """
    return {name: total / count if int(count) else total
            for name, (total, count) in terms.items()}


def _finite_selected(value: Tensor, name: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f'Nonfinite available {name}')


def visual_loss_terms(output: dict, batch: dict, codec: MANOStateCodec,
                      config: VisualGeometryLossConfig) -> dict[str, tuple[Tensor, Tensor]]:
    """Return differentiable local sums and exact counts, including empty ranks.

    NLL is a per-complete-state mean. Raw position is a per-labelled-wrist
    mean. Final geometry is a per-labelled-joint mean (equivalent to standard
    MPJPE for complete 21-point references). A partial label never turns into
    a fabricated full MANO target. Selection precedes subtraction, density,
    rotations, and MANO, so masked NaNs cannot poison gradients.
    """
    state = output['state_world']
    if state.ndim != 3 or state.shape[-1] != 99:
        raise ValueError('Visual outputs require [B,T,99] world states')
    shape, device = state.shape[:-1], state.device
    frame_valid = output['frame_valid'].bool()
    target_valid = batch['target_valid'].bool()
    if frame_valid.shape != shape or target_valid.shape != (*shape, 17):
        raise ValueError('State/frame/group validity shapes disagree')
    if batch['target_world'].shape != state.shape:
        raise ValueError('Reference 99D shape differs from the prediction')
    complete = frame_valid & target_valid.all(-1)
    count = complete.sum().detach()
    nll = (feedback_current_nll(output, batch['target_world'], batch['target_valid']) * count
           if bool(complete.any()) else _empty_sum(state))
    result = {'current_nll': (nll, count)}
    if config.raw_wrist_weight > 0:
        raw = output['raw_world']
        available = output['availability'].bool()
        if raw.shape != state.shape or available.shape != (*shape, 17):
            raise ValueError('Raw state/availability shape differs from final state')
        valid = frame_valid & target_valid[..., 0] & available[..., 0]
        prediction, reference = raw[..., :3][valid].float(), batch['target_world'][..., :3][valid].float()
        _finite_selected(prediction, 'raw J0 prediction')
        _finite_selected(reference, 'raw J0 reference')
        result['raw_wrist_geometry_m'] = (torch.linalg.vector_norm(prediction-reference, dim=-1).sum(),
                                           valid.sum().detach())
    if config.final_joint_weight > 0:
        reference, valid = batch['reference_joints_world'], batch['reference_joints_valid'].bool()
        if reference.shape != (*shape, 21, 3) or valid.shape != (*shape, 21):
            raise ValueError('Reference geometry must be [B,T,21,3] with [B,T,21] validity')
        valid = valid & frame_valid[..., None]
        beta, right = output['beta'], batch['is_right'].bool()
        if beta.shape != (*shape, 10) or right.shape != shape:
            raise ValueError('Prediction beta/handedness shape differs from final state')
        metadata = batch.get('reference_geometry_metadata', {})
        orders = metadata.get('joint_order')
        if not isinstance(orders, (list, tuple)) or len(orders) != shape[0]:
            raise ValueError('One explicit source joint-order metadata item is required per clip')
        if any(order not in SUPPORTED_REFERENCE_JOINT_ORDERS for order in orders):
            raise ValueError('Unaudited geometry joint mapping')
        if metadata.get('units') != ['metres'] * shape[0]:
            raise ValueError('Loss reference landmarks must be explicitly in metres')
        if metadata.get('loss_coordinate_frame') != ['world'] * shape[0]:
            raise ValueError('Loss reference landmarks must be converted into the state world frame')
        # Each clip has one source identity, but a batch can contain sources
        # with different fingertip definitions. Decode each mapping separately.
        total = _empty_sum(state)
        for order in sorted(set(orders)):
            clip_mask = torch.tensor([value == order for value in orders], device=device)
            frames = valid.any(-1) & clip_mask[:, None]
            if not bool(frames.any()):
                continue
            selected_state, selected_beta = state[frames], beta[frames]
            _finite_selected(selected_state, 'final state prediction')
            _finite_selected(selected_beta, 'prediction beta')
            joints = codec.joints21(selected_state, selected_beta, right[frames], joint_order=order)
            joint_mask = valid[frames]
            prediction, target = joints[joint_mask], reference[frames][joint_mask].float()
            _finite_selected(prediction, 'decoded final landmarks')
            _finite_selected(target, 'reference landmarks')
            total = total + torch.linalg.vector_norm(prediction-target, dim=-1).sum()
        result['final_mpjpe21_m'] = (total, valid.sum().detach())
    return result


def reduce_visual_loss_terms(terms: dict[str, tuple[Tensor, Tensor]],
                             config: VisualGeometryLossConfig, *, distributed: bool
                             ) -> tuple[Tensor, dict[str, float | int]]:
    """Global masked means with gradients correct under DDP gradient averaging.

    Only detached counts/metric numerators are all-reduced. On each rank the
    differentiable local sum is multiplied by world_size/global_count. After
    DDP averages, its gradient equals that of the global valid-label mean,
    even if another rank has no labels for this term. Rank-zero-only validation
    must pass distributed=False to avoid collectives with training peers.
    """
    use_dist = distributed and dist.is_available() and dist.is_initialized()
    world = dist.get_world_size() if use_dist else 1
    names = list(terms)
    packed = torch.stack([torch.stack((terms[name][0].detach().double(),
                                      terms[name][1].detach().double())) for name in names])
    if use_dist:
        dist.all_reduce(packed)
    weights = visual_loss_coefficients(config)
    objective = _empty_sum(terms[names[0]][0])
    measurements: dict[str, float | int] = {}
    supervised = False
    for index, name in enumerate(names):
        total, count = packed[index]
        measurements[name + '/count'] = int(count)
        if int(count):
            objective = objective + weights[name] * terms[name][0] * (world / count.float())
            measurements[name] = float(total / count)
            supervised = supervised or weights[name] > 0
        else:
            objective = objective + terms[name][0] * 0
    if not supervised:
        raise ValueError('No valid labels for any enabled visual objective on any rank')
    if not bool(torch.isfinite(objective)):
        raise FloatingPointError('Nonfinite reduced visual loss')
    return objective, measurements
