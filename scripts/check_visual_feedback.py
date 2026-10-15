#!/usr/bin/env python3
"""Remote-only numerical contracts for the D33 visual feedback integration.

This is a short real-cache/checkpoint integration check, not a benchmark or
accuracy claim. It deliberately tests feedback ordering and differentiability
with the actual pretrained motion model. No downloads or training-run edits.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import traceback

import torch

from handloop import geometry as g
from handloop.data.visual_clips import VisualFeedbackClipDataset, file_sha256
from handloop.models.probabilistic_twm import ProbabilisticTWM, ProbabilisticTWMConfig
from handloop.models.visual_feedback import (FeedbackHistory, VisualFeedbackConfig,
    VisualFeedbackSystem, feedback_current_nll)


INPUTS = ('timestamps', 'frame_valid', 'availability', 'image_valid', 'handedness',
          'world_to_camera', 'camera_intrinsics', 'crop_transform', 'beta', 'reset')


def module_hash(module):
    digest = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def clipped(batch, start, stop):
    return {key: value[:, start:stop].clone() for key, value in batch.items()
            if isinstance(value, torch.Tensor)}


def run_clip(system, batch, **kwargs):
    return system.forward_clip(batch['patch_features'], batch['raw_world'],
        **{key: batch[key] for key in INPUTS if key in batch}, **kwargs)


def run_step(system, batch, *, index=0, history=None, **kwargs):
    return system.step(batch['patch_features'][:, index], batch['raw_world'][:, index],
        **{key: batch[key][:, index] for key in INPUTS if key in batch},
        history=history, **kwargs)


def close(a, b, *, atol=3e-5, message='tensors disagree'):
    if a.dtype == torch.bool or not a.is_floating_point():
        assert torch.equal(a, b), message
        return 0.0
    error = float((a.float() - b.float()).abs().max()) if a.numel() else 0.0
    assert torch.allclose(a, b, atol=atol, rtol=1e-5, equal_nan=False), f'{message}: max_abs={error}'
    return error


def check_history(a: FeedbackHistory, b: FeedbackHistory):
    for key in a.__dataclass_fields__:
        close(getattr(a, key), getattr(b, key), message='history ' + key)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-manifest', type=Path, required=True)
    parser.add_argument('--twm-checkpoint', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-twm-parameters', type=int, default=13_530_032)
    args = parser.parse_args()
    report = {'schema': 'handloop.visual_feedback_contracts.v1',
              'started_utc': datetime.now(timezone.utc).isoformat(), 'passed': False,
              'checks': {}, 'scope': 'short numerical integration; no accuracy/physical claim'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    checks = report['checks']
    try:
        torch.set_num_threads(1)
        torch.manual_seed(1933)
        device = torch.device(args.device)
        if device.type == 'cuda':
            torch.cuda.set_device(device)
        source = torch.load(args.twm_checkpoint, map_location='cpu', weights_only=False)
        twm_config = ProbabilisticTWMConfig(**source['config']['model'])
        twm = ProbabilisticTWM(twm_config)
        twm.load_state_dict(source['model'], strict=True)
        parameters = sum(parameter.numel() for parameter in twm.parameters())
        assert parameters == args.expected_twm_parameters, parameters
        system = VisualFeedbackSystem(twm, VisualFeedbackConfig()).to(device).eval()
        frozen_before = module_hash(system.twm)
        report.update({'device': str(device), 'twm_parameters': parameters,
                       'adapter_parameters': sum(p.numel() for p in system.adapter.parameters()),
                       'twm_checkpoint': str(args.twm_checkpoint.resolve()),
                       'twm_checkpoint_sha256': file_sha256(args.twm_checkpoint),
                       'twm_step': source.get('step'), 'adapter_config': asdict(system.config),
                       'cache_sha256': file_sha256(args.cache_manifest)})
        dataset = VisualFeedbackClipDataset(args.cache_manifest, 'train', clip_length=3)
        sample = None
        for index in range(len(dataset)):
            candidate = dataset[index]
            if (bool(candidate['frame_valid'].all()) and bool(candidate['availability'].all())
                    and bool(candidate['image_valid'].all()) and bool(candidate['target_valid'].all())):
                sample = candidate
                break
        assert sample is not None, 'Need one three-frame real image/full-MANO reference cache window'
        report['recording'] = sample['recording']
        report['start_frame'] = sample['start_frame']
        batch = {key: value[None].to(device) for key, value in sample.items() if isinstance(value, torch.Tensor)}
        checks['actual_checkpoint_strict_load'] = True

        with torch.no_grad():
            full = run_clip(system, clipped(batch, 0, 1), return_forecast=True)
            current = run_clip(system, clipped(batch, 0, 1))
            equivalence = {key: close(full[key], current[key], atol=1e-7, message='current/full ' + key)
                           for key in ('state', 'state_translation_std', 'state_rotation_concentration')}
            assert full['forecast_modes'].shape == (1, 1, twm_config.num_modes, twm_config.forecast_horizon, 99)
            checks['current_only_matches_full'] = equivalence
            rotation_world = full['anchor_world'][..., :3, :3]
            covariance = full['state_translation_covariance_world']
            close(covariance, rotation_world @ torch.diag_embed(full['state_translation_std'].square())
                  @ rotation_world.transpose(-1, -2), atol=1e-8)
            assert bool(torch.isfinite(covariance).all()) and bool((torch.linalg.eigvalsh(covariance) > 0).all())
            future_rotation_world = rotation_world[:, :, None, None]
            future_covariance = full['forecast_modes_translation_covariance_world']
            close(future_covariance, future_rotation_world
                  @ torch.diag_embed(full['forecast_translation_std'].square())
                  @ future_rotation_world.transpose(-1, -2), atol=1e-8)
            assert bool(torch.isfinite(future_covariance).all())
            assert bool((torch.linalg.eigvalsh(future_covariance) > 0).all())
            checks['world_translation_covariance_transform_and_positive_definite'] = True
            output = run_clip(system, batch)
            bypass = run_clip(system, batch, use_adapter=False)
            checks['zero_init_candidate_raw_max_abs'] = close(output['candidate_world'],
                g.normalize_state(batch['raw_world']), message='zero candidate/raw')
            checks['zero_init_final_bypass_max_abs'] = close(output['state_world'], bypass['state_world'])
            assert torch.equal(output['availability'], batch['availability'])
            assert torch.equal(output['candidate_availability'], batch['availability'])
            checks['raw_and_candidate_full_masks_preserved'] = True
            rotation = g.state_to_rotmat(output['state_world'])
            orthogonal = rotation.transpose(-1, -2) @ rotation
            close(orthogonal, torch.eye(3, device=device).expand_as(orthogonal), atol=2e-5)
            assert bool((torch.det(rotation) > .9999).all())
            checks['legal_rotations'] = True

            first = run_clip(system, clipped(batch, 0, 1))
            rest = run_clip(system, clipped(batch, 1, 3), initial_history=first['final_history'])
            close(torch.cat((first['state_world'], rest['state_world']), 1), output['state_world'])
            check_history(rest['final_history'], output['final_history'])
            assert not output['final_history'].state_world.requires_grad
            checks['streaming_matches_clip_and_detaches'] = True

            pad = clipped(batch, 0, 3)
            for key in ('frame_valid', 'image_valid', 'availability', 'target_valid'):
                pad[key][:, 2:] = False
            for key in ('patch_features', 'raw_world', 'target_world', 'world_to_camera',
                        'camera_intrinsics', 'crop_transform', 'timestamps'):
                pad[key][:, 2:] = float('nan')
            pad['reset'][:, 2:] = False
            padded = run_clip(system, pad)
            two = run_clip(system, clipped(batch, 0, 2))
            check_history(padded['final_history'], two['final_history'])
            checks['padding_never_commits_and_nan_placeholders_safe'] = True

            missing = clipped(batch, 0, 1)
            missing['availability'][:] = False
            missing['image_valid'][:] = False
            missing['raw_world'][:] = float('nan')
            missing['patch_features'][:] = float('nan')
            inferred = run_clip(system, missing)
            assert bool(torch.isfinite(inferred['state_world']).all())
            assert int(inferred['history_count'][0, 0]) == 0
            assert not bool(inferred['candidate_availability'].any())
            checks['empty_missing_warm_start_finite_explicit_masks'] = True

            gap = clipped(batch, 2, 3)
            gap['reset'][:] = False
            gap['timestamps'][:] = two['final_history'].timestamps[:, -1:] + system.config.max_gap_seconds * 2
            gap_output = run_clip(system, gap, initial_history=two['final_history'])
            assert bool(gap_output['history_reset'].all()) and not bool(gap_output['history_count'].any())
            explicit = clipped(batch, 2, 3)
            explicit['reset'][:] = True
            reset_output = run_clip(system, explicit, initial_history=two['final_history'])
            assert bool(reset_output['history_reset'].all()) and not bool(reset_output['history_count'].any())
            hand = clipped(batch, 2, 3)
            hand['reset'][:] = False
            hand['handedness'][:] = 1 - two['final_history'].handedness[:, None]
            hand_output = run_clip(system, hand, initial_history=two['final_history'])
            assert bool(hand_output['history_reset'].all()) and not bool(hand_output['history_count'].any())
            checks['gap_explicit_and_hand_switch_reset'] = True

        system.train()
        assert not system.twm.training and system.adapter.training
        captured = []
        def capture_candidate(module, args, kwargs):
            args[2].retain_grad()
            captured.append(args[2])
        handle = system.twm.register_forward_pre_hook(capture_candidate, with_kwargs=True)
        optimizer = torch.optim.AdamW([p for p in system.parameters() if p.requires_grad], lr=1e-4)
        losses = []
        candidate_gradients = []
        for unused in range(2):
            optimizer.zero_grad(set_to_none=True)
            step, _ = run_step(system, batch)
            loss = feedback_current_nll(step, batch['target_world'][:, 0], batch['target_valid'][:, 0])
            assert bool(torch.isfinite(loss))
            loss.backward()
            norm = float(captured[-1].grad.norm())
            assert norm > 0 and bool(torch.isfinite(captured[-1].grad).all())
            assert all(p.grad is None for p in system.twm.parameters())
            gradients = [p.grad for p in system.adapter.parameters() if p.grad is not None]
            assert gradients and all(bool(torch.isfinite(value).all()) for value in gradients)
            assert float(system.adapter.translation_head.weight.grad.norm()
                         + system.adapter.rotation_head.weight.grad.norm()) > 0
            losses.append(float(loss.detach())); candidate_gradients.append(norm)
            optimizer.step()
        handle.remove()
        assert module_hash(system.twm) == frozen_before
        checks['candidate_gradient_frozen_weights_unchanged'] = {'losses': losses,
            'candidate_gradient_norm': candidate_gradients, 'twm_hash': frozen_before}

        # GT selection happens before geometry; a masked NaN frame cannot
        # contaminate either the proper NLL or the plugin's backward pass.
        optimizer.zero_grad(set_to_none=True)
        nan_gt = batch['target_world'].clone()
        target_valid = batch['target_valid'].clone()
        nan_gt[:, 1:] = float('nan'); target_valid[:, 1:] = False
        final = run_clip(system, batch)
        masked_loss = feedback_current_nll(final, nan_gt, target_valid)
        masked_loss.backward()
        assert bool(torch.isfinite(masked_loss))
        assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in system.adapter.parameters())
        checks['masked_nan_gt_nll_and_backward_finite'] = float(masked_loss.detach())

        system.eval()
        with torch.no_grad():
            normal = run_clip(system, batch)
            altered = clipped(batch, 0, 3)
            altered['patch_features'][:, 2] = torch.randn_like(altered['patch_features'][:, 2]) * 10
            altered['raw_world'][:, 2, :3] += 2
            later = run_clip(system, altered)
            close(normal['state_world'][:, :2], later['state_world'][:, :2], atol=1e-7)
            checks['later_core_cache_cannot_change_earlier_output'] = True
            seen_adapter, seen_twm = [], []
            def capture_adapter(module, args, kwargs):
                seen_adapter.append(kwargs['history_valid'].clone())
            def capture_motion(module, args, kwargs):
                seen_twm.append(kwargs['history_valid'].clone())
            ha = system.adapter.register_forward_pre_hook(capture_adapter, with_kwargs=True)
            hw = system.twm.register_forward_pre_hook(capture_motion, with_kwargs=True)
            no_feedback = run_clip(system, batch, adapter_history=False)
            ha.remove(); hw.remove()
            assert all(not bool(value.any()) for value in seen_adapter)
            assert [int(value.sum()) for value in seen_twm] == [0, 1, 2]
            assert torch.equal(no_feedback['history_count'], normal['history_count'])
            checks['no_feedback_ablation_preserves_motion_history'] = True
        system.configure_trainable('current_readout')
        expected_heads = ('translation_head.', 'rotation_head.', 'state_translation_scale_head.',
                          'state_rotation_concentration_head.')
        assert all(parameter.requires_grad == name.startswith(expected_heads)
                   for name, parameter in system.twm.named_parameters())
        system.configure_trainable('adapter')
        checks['optional_current_readout_scope'] = True
        report['passed'] = True
    except Exception as error:
        report['failure'] = {'type': type(error).__name__, 'message': str(error),
                             'traceback': traceback.format_exc()}
        raise
    finally:
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')
        print(json.dumps({'passed': report['passed'], 'output': str(args.output),
                          'completed_checks': len(checks)}))


if __name__ == '__main__':
    main()
