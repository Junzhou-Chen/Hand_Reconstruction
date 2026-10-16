#!/usr/bin/env python3
"""Remote-only contracts for differentiable multi-round visual feedback.

Uses a small explicit motion fixture and camera-joint fixture to isolate update
ordering, masks and gradients. This does not replace actual D37/MANO/HaWoR
integration checks and makes no reconstruction or physical accuracy claim.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from handloop import geometry as g
from handloop.models.visual_feedback import (
    VisualFeedbackConfig, VisualFeedbackSystem, feedback_current_nll, state_anchor_world,
)


class MotionFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.translation_head = nn.Linear(9, 3)
        self.rotation_head = nn.Linear(9, 3)
        self.state_translation_scale_head = nn.Linear(9, 3)
        self.state_rotation_concentration_head = nn.Linear(9, 1)
        self.future_fixture = nn.Linear(9, 3)
        self.calls = []

    def forward(self, history, raw, candidate, **kwargs):
        features = candidate[..., :9, None].transpose(-1, -2).expand(-1, 16, -1)
        dp = self.translation_head(features[:, 0]) * .001
        dp = dp + dp.new_tensor([.005, 0., 0.])
        dr = self.rotation_head(features) * .001
        state = g.state_apply_delta(candidate, torch.cat((dp, dr.flatten(-2)), -1))
        if state.requires_grad:
            state.retain_grad()
        self.calls.append({'history': history.detach().clone(), 'raw': raw.detach().clone(),
                           'candidate': candidate.detach().clone(),
                           'availability': kwargs['availability'].detach().clone(),
                           'history_valid': kwargs['history_valid'].detach().clone(),
                           'dt': kwargs['dt'].detach().clone(), 'state': state})
        return {'state': state,
                'state_translation_std': torch.nn.functional.softplus(
                    self.state_translation_scale_head(features[:, 0])) * .01 + .01,
                'state_rotation_concentration': torch.nn.functional.softplus(
                    self.state_rotation_concentration_head(features).squeeze(-1)) + 10.,
                'moe_load': state.new_ones(1), 'moe_balance_loss': state.new_zeros(())}


def joint_fixture(state_camera, beta, handedness):
    # Geometry fixture is deliberately not described as a MANO reconstruction.
    offset = state_camera.new_zeros(state_camera.shape[0], 16, 3)
    offset[..., 0] = torch.linspace(-.04, .04, 16, device=state_camera.device)
    offset[..., 1] = torch.linspace(-.06, .06, 16, device=state_camera.device)
    root = g.state_to_rotmat(state_camera)[..., 0, :, :]
    return ((root[:, None] @ offset[..., None]).squeeze(-1)
            + state_camera[:, None, :3] + beta[:, None, :1] * 0.)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if not Path('/sciclone/home/jchen57').exists():
        raise RuntimeError('Numerical checks must run on the remote compute Pod')
    torch.set_num_threads(1)
    torch.manual_seed(934)
    device = torch.device(args.device)
    config = VisualFeedbackConfig(image_feature_dim=16, patch_grid_height=4,
        patch_grid_width=4, d_model=32, nhead=4, num_layers=1,
        dim_feedforward=64, history_length=3, refinement_steps=2,
        candidate_guided_queries=True)
    motion = MotionFixture()
    system = VisualFeedbackSystem(motion, config, trainable_scope='full').to(device).train()
    assert motion.training and all(p.requires_grad for p in motion.parameters())
    # Identity initialization intentionally has no patch gradient on update one.
    # Perturb the existing last projection to test the active visual path.
    with torch.no_grad():
        system.adapter.translation_head.weight.normal_(0., .005)
        system.adapter.rotation_head.weight.normal_(0., .005)
    raw = g.identity_state(1, device=device)
    raw[:, 2] = .6
    raw.requires_grad_()
    raw_before = raw.detach().clone()
    patches = torch.randn(1, 16, 16, device=device, requires_grad=True)
    eye = torch.eye(4, device=device)[None]
    intrinsic = eye[:, :3, :3].clone()
    intrinsic[:, 0, 0] = intrinsic[:, 1, 1] = 300.
    intrinsic[:, 0, 2] = intrinsic[:, 1, 2] = 128.
    crop = intrinsic.new_tensor([[[2 / 256., 0., -1.],
                                  [0., 2 / 256., -1.], [0., 0., 1.]]])
    kwargs = {'timestamps': torch.zeros(1, device=device, dtype=torch.float64),
              'frame_valid': torch.ones(1, device=device, dtype=torch.bool),
              'availability': torch.ones(1, 17, device=device, dtype=torch.bool),
              'image_valid': torch.ones(1, device=device, dtype=torch.bool),
              'handedness': torch.ones(1, device=device, dtype=torch.long),
              'world_to_camera': eye, 'camera_intrinsics': intrinsic,
              'crop_transform': crop, 'beta': torch.zeros(1, 10, device=device),
              'joint_geometry': joint_fixture}
    output, history = system.step(patches, raw, **kwargs)
    assert len(motion.calls) == 2
    for key in ('history', 'raw', 'availability', 'history_valid', 'dt'):
        assert torch.equal(motion.calls[0][key], motion.calls[1][key]), key
    assert torch.equal(raw, raw_before)
    assert torch.allclose(output['anchor_world'], state_anchor_world(raw).detach())
    assert output['refinement_states_world'].shape == (1, 2, 99)
    assert output['refinement_joint_crop_coords'].shape == (1, 2, 16, 2)
    assert bool(output['refinement_joint_projection_valid'].all())
    query_change = (output['refinement_joint_crop_coords'][:, 1]
                    - output['refinement_joint_crop_coords'][:, 0]).abs().max()
    assert float(query_change) > 1e-5
    assert int(history.valid.sum()) == 1  # Two rounds still submit one frame.
    assert not history.state_world.requires_grad
    assert torch.allclose(history.state_world[:, -1], output['state_world'])
    target = raw.detach().clone()
    target[:, 0] += .015
    loss = feedback_current_nll(output, target, kwargs['frame_valid'])
    loss.backward()
    assert raw.grad is not None and float(raw.grad.norm()) > 0
    assert patches.grad is not None and float(patches.grad.norm()) > 0
    assert motion.calls[0]['state'].grad is not None
    assert float(motion.calls[0]['state'].grad.norm()) > 0
    assert motion.translation_head.weight.grad is not None
    assert motion.future_fixture.weight.grad is None  # Requires separate replay.
    gradient_norms = {'raw': float(raw.grad.norm()), 'patches': float(patches.grad.norm()),
                      'first_round_state': float(motion.calls[0]['state'].grad.norm())}
    system.configure_trainable('adapter').train()
    assert not motion.training and not any(p.requires_grad for p in motion.parameters())
    system.configure_trainable('current_readout').train()
    expected = ('translation_head.', 'rotation_head.', 'state_translation_scale_head.',
                'state_rotation_concentration_head.')
    assert all(p.requires_grad == name.startswith(expected) for name, p in motion.named_parameters())
    assert not motion.training
    system.configure_trainable('full').eval()
    with torch.no_grad():
        missing = dict(kwargs, availability=torch.zeros_like(kwargs['availability']),
                       image_valid=torch.zeros_like(kwargs['image_valid']))
        inferred, missing_history = system.step(torch.full_like(patches, float('nan')),
            torch.full_like(raw, float('nan')), **missing)
        assert bool(torch.isfinite(inferred['state_world']).all())
        assert not bool(inferred['candidate_availability'].any())
        assert not bool(inferred['refinement_joint_projection_valid'].any())
        padding = dict(kwargs, frame_valid=torch.zeros_like(kwargs['frame_valid']),
                       timestamps=kwargs['timestamps'] + 1 / 30.)
        _, padded_history = system.step(patches, raw, history=history, **padding)
        for name in ('state_world', 'valid', 'availability', 'timestamps', 'uncertainty'):
            assert torch.equal(getattr(history, name), getattr(padded_history, name)), name
    try:
        system.step(patches, raw, **dict(kwargs, joint_geometry=lambda s, b, h:
                                       s.new_zeros(s.shape[0], 21, 3)))
    except ValueError as error:
        assert 'MANO16' in str(error)
    else:
        raise AssertionError('A 21-joint order was silently truncated')
    report = {'schema': 'handloop.visual_refinement_contract.v1', 'passed': True,
              'scope': 'ordering/masks/gradient fixtures only; not actual MANO/HaWoR accuracy',
              'rounds': 2, 'commits_per_frame': 1,
              'query_change_max': float(query_change), 'gradient_norms': gradient_norms,
              'fixed_context': ['raw', 'history', 'availability', 'dt', 'anchor'],
              'missing_warmup_finite_unavailable': True, 'padding_never_commits': True,
              'legacy_scopes_freeze_twm': True, 'full_scope_trainable': True,
              'joint21_rejected': True, 'future_needs_separate_objective': True}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
