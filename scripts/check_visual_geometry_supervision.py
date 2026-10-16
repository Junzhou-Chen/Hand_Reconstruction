#!/usr/bin/env python3
"""Remote-only MANO, missing-label and distributed geometry-loss contracts.

No optimizer is constructed and no accuracy claim is made. With torchrun,
also exercises actual DDP gradient averaging with uneven/empty per-rank masks.
Licensed MANO assets remain external.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from handloop import geometry as g
from handloop.data.visual_rgb import reference_geometry_metadata
from handloop.visual.codec import MANOStateCodec, MANO21_THUMB_FIRST_ORDER
from handloop.visual.losses import (VisualGeometryLossConfig, visual_loss_terms,
    reduce_visual_loss_terms, local_visual_term_means, visual_loss_coefficients)
from handloop.models.visual_feedback import feedback_current_nll


def close(obtained, expected, name, atol=2e-6):
    if not torch.allclose(obtained.float(), expected.float(), atol=atol, rtol=1e-5):
        raise AssertionError(f'{name}: expected {expected}, obtained {obtained}')


def fixture(codec, device):
    target = g.identity_state(2, 3, device=device)
    target[..., :3] = torch.tensor([.03, -.02, .7], device=device)
    state = target.clone()
    state[..., :3] += torch.tensor([.008, -.004, .003], device=device)
    state.requires_grad_()
    beta = torch.full((2, 3, 10), .15, device=device, requires_grad=True)
    raw = target.clone()
    raw[..., :3] += torch.tensor([-.012, .009, .016], device=device)
    raw.requires_grad_()
    right = torch.tensor([[False]*3, [True]*3], device=device)
    frame = torch.tensor([[True, True, False], [True, True, False]], device=device)
    groups = frame[..., None].expand(-1, -1, 17).clone()
    groups[0, 1, 1:] = False
    target[0, 1, 3:] = float('nan')
    target[~frame] = float('nan')
    joints = torch.zeros(2, 3, 21, 3, device=device)
    valid = frame[..., None].expand(-1, -1, 21).clone()
    valid[0, 1, 1::2] = False
    with torch.no_grad():
        clean_state = g.identity_state(3, device=device)
        clean_state[..., :3] = torch.tensor([.03, -.02, .7], device=device)
        for index, order in enumerate(('dex21', 'meta21')):
            joints[index] = codec.joints21(clean_state, torch.zeros(3, 10, device=device),
                                           right[index], joint_order=order)
    joints[~valid] = float('nan')
    availability = torch.ones(2, 3, 17, dtype=torch.bool, device=device)
    availability[1, 1, 0] = False
    output = {'state_world': state, 'state': state,
              'state_translation_std': torch.full((2, 3, 3), .03, device=device),
              'state_rotation_concentration': torch.full((2, 3, 16), 20., device=device),
              'anchor_world': torch.eye(4, device=device).expand(2, 3, 4, 4),
              'raw_world': raw, 'availability': availability, 'frame_valid': frame, 'beta': beta}
    batch = {'target_world': target, 'target_valid': groups, 'is_right': right,
             'reference_joints_world': joints, 'reference_joints_valid': valid,
             'beta': torch.full_like(beta, float('nan')),  # Must never be used.
             'reference_geometry_metadata': {'joint_order': ['dex21', 'meta21'],
                 'units': ['metres']*2, 'loss_coordinate_frame': ['world']*2}}
    config = VisualGeometryLossConfig(raw_wrist_weight=.7, final_joint_weight=1.3,
                                      raw_wrist_scale_m=.05, final_joint_scale_m=.05)
    terms = visual_loss_terms(output, batch, codec, config)
    means = local_visual_term_means(terms)
    close(means['current_nll'], feedback_current_nll(output, target, groups), 'unchanged NLL')
    raw_mask = frame & groups[..., 0] & availability[..., 0]
    close(means['raw_wrist_geometry_m'], (raw[..., :3][raw_mask]-target[..., :3][raw_mask]).norm(dim=-1).mean(), 'raw J0 mean')
    decoded = torch.stack([codec.joints21(state[i], beta[i], right[i], joint_order=order)
                           for i, order in enumerate(('dex21', 'meta21'))])
    close(means['final_mpjpe21_m'], (decoded[valid]-joints[valid]).norm(dim=-1).mean(), 'joint masked mean')
    for index, order in enumerate(('dex21', 'meta21')):
        geometry = codec.geometry(state[index], beta[index], right[index])
        ids = ([745, 317, 445, 556, 673] if index == 0 else [744, 320, 443, 554, 671])
        manual = torch.cat((geometry['joints'], geometry['vertices'][:, ids]), 1)[:, MANO21_THUMB_FIRST_ORDER]
        close(decoded[index], manual, 'explicit side/tip mapping '+order)
        close(decoded[index, :, 0], state[index, :, :3], 'anatomical J0 identity')
    # Check both sides for both mappings, including Dex's side-specific middle tip.
    for order in ('dex21', 'meta21'):
        geometry = codec.geometry(state[:, 0], beta[:, 0], right[:, 0])
        ids = torch.tensor(([745, 317, 444, 556, 673] if order == 'dex21'
                            else [744, 320, 443, 554, 671]), device=device).expand(2, -1).clone()
        if order == 'dex21':
            ids[0, 2] = 445
        tips = geometry['vertices'].gather(1, ids[..., None].expand(-1, -1, 3))
        manual = torch.cat((geometry['joints'], tips), 1)[:, MANO21_THUMB_FIRST_ORDER]
        close(codec.joints21(state[:, 0], beta[:, 0], right[:, 0], joint_order=order),
              manual, 'both sides '+order)
    loss, measurements = reduce_visual_loss_terms(terms, config, distributed=False)
    loss.backward()
    for name, value in (('raw', raw), ('final', state), ('prediction_beta', beta)):
        if value.grad is None or not bool(torch.isfinite(value.grad).all()) or not float(value.grad.norm()) > 0:
            raise AssertionError('Missing/nonfinite '+name+' gradient')
    if bool(state.grad[~frame].any()) or bool(raw.grad[~frame].any()) or bool(beta.grad[~frame].any()):
        raise AssertionError('Padding acquired geometry gradients')
    # The original NLL-only configuration is numerically preserved.
    nll_only = visual_loss_terms(output, batch, codec, VisualGeometryLossConfig())
    old_loss, _ = reduce_visual_loss_terms(nll_only, VisualGeometryLossConfig(), distributed=False)
    close(old_loss.detach(), feedback_current_nll(output, target, groups).detach(), 'default objective parity')
    return {'masked_NaN_labels': True, 'source_beta_unused': True, 'padding_zero_gradient': True,
            'left_right_landmarks': True, 'prediction_beta_gradient': float(beta.grad.norm()),
            'measurements': measurements, 'config': asdict(config)}


class ReductionFixture(nn.Module):
    def __init__(self, rank, device, config):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(.7, device=device))
        self.rank, self.config = rank, config

    def forward(self):
        rank = self.rank
        # Rank zero has no complete state; other terms also have unequal counts.
        counts = (0 if rank == 0 else 2, rank+1, 3*rank+1)
        coefficients = (rank+1., rank+2., rank+3.)
        terms = {name: (self.value.square()*coefficient*count,
                        torch.tensor(count, device=self.value.device))
                 for name, coefficient, count in zip(
                     ('current_nll', 'raw_wrist_geometry_m', 'final_mpjpe21_m'), coefficients, counts)}
        loss, _ = reduce_visual_loss_terms(terms, self.config, distributed=True)
        return loss


def reduction_contract(rank, world, device):
    config = VisualGeometryLossConfig(nll_weight=.8, raw_wrist_weight=.7, final_joint_weight=1.3,
                                      raw_wrist_scale_m=.05, final_joint_scale_m=.1)
    module = ReductionFixture(rank, device, config)
    model = DDP(module, device_ids=[device.index] if device.type == 'cuda' else None) if world > 1 else module
    model().backward()
    weights = list(visual_loss_coefficients(config).values())
    expected = 0.
    for term, weight in enumerate(weights):
        counts = [(0 if r == 0 else 2, r+1, 3*r+1)[term] for r in range(world)]
        if sum(counts):
            expected += weight * sum((r+term+1)*n for r, n in enumerate(counts))/sum(counts)
    expected *= 2*.7
    close(module.value.grad, module.value.new_tensor(expected), 'DDP global gradient', atol=2e-5)
    zero = module.value.reshape(-1)[:0].sum()
    try:
        reduce_visual_loss_terms({'current_nll': (zero, torch.tensor(0, device=device))}, config, distributed=True)
    except ValueError:
        pass
    else:
        raise AssertionError('An entirely unlabelled global batch must fail')
    return {'world_size': world, 'uneven_rank_counts': True, 'empty_rank_nll': True,
            'actual_DDP_gradient': float(module.value.grad), 'expected_global_gradient': expected,
            'globally_empty_rejected': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mano-models', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if not Path('/sciclone/data10/jchen57').is_dir():
        raise RuntimeError('Run geometry numerical contracts in the remote compute Pod')
    rank, world, local = (int(os.environ.get(name, default)) for name, default in
                           (('RANK', 0), ('WORLD_SIZE', 1), ('LOCAL_RANK', 0)))
    device = torch.device('cuda', local) if args.device == 'cuda' else torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group('nccl' if device.type == 'cuda' else 'gloo')
    torch.set_num_threads(1)
    report = {'schema': 'handloop.visual_geometry_contracts.v1', 'passed': False,
              'utc': datetime.now(timezone.utc).isoformat(), 'rank': rank,
              'scope': 'implementation/gradient contracts, no optimizer or accuracy claim'}
    destination = args.report if rank == 0 else args.report.with_name(args.report.stem+f'_rank{rank}.json')
    try:
        codec = MANOStateCodec(args.mano_models/'MANO_RIGHT.pkl', args.mano_models/'MANO_LEFT.pkl').to(device)
        report['geometry'] = fixture(codec, device)
        report['reduction'] = reduction_contract(rank, world, device)
        for overrides in ({'dataset': 'unknown'}, {'dataset': 'dexycb', 'joint_3d_order': 'unknown'},
                          {'dataset': 'dexycb', 'joint_3d_order': 'dex21', 'joint_3d_units': 'mm'}):
            try:
                reference_geometry_metadata(overrides, {'coordinate_frame': 'world'})
            except ValueError:
                continue
            raise AssertionError('Unknown source/order/units must be rejected')
        report['source_contract_rejections'] = True
        report['passed'] = True
    except Exception as exc:
        report['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        raise
    finally:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        if dist.is_initialized():
            dist.destroy_process_group()
    print(json.dumps({'passed': report['passed'], 'report': str(destination)}))


if __name__ == '__main__':
    main()
