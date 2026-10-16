#!/usr/bin/env python3
"""Remote-only, read-only gradient diagnostics for the completed joint pilot.

No optimizer is constructed. The fixed train RGB selection and motion replay
inputs are written before any gradient measurement. Gradients are evaluated
with dropout disabled. ViT/IAM are treated as fixed upstream feature producers
to bound diagnostic cost; all HaWoR head/PAM, adapter and TWM parameters remain
differentiable. Their partial derivatives are unchanged by this restriction.
This is NOT an estimate of the missing ViT/IAM gradient norm or a retraining.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import socket

import torch
from torch.utils.data import default_collate

from handloop import geometry as g
from handloop.data.forecast_trajectory import ForecastTrajectoryWindows
from handloop.data.visual_clips import file_sha256
from handloop.data.visual_rgb import VisualFeedbackRGBDataset
from handloop.models.probabilistic_twm import current_distribution_log_prob
from handloop.models.visual_feedback import feedback_current_nll, transform_state, inverse_rigid
from handloop.visual.codec import mirror_rotations_x
from evaluate_visual_joint import load_model
from train_probabilistic_twm import prepare_batch, objective, state_geometry_loss


def utc():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    tmp.replace(path)


def parameter_digest(module):
    digest = hashlib.sha256()
    for name, parameter in module.named_parameters():
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def selected_parameter_digest(named_parameters):
    digest = hashlib.sha256()
    for name, parameter in named_parameters:
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def selected_groups(pipeline):
    """Disjoint groups, complete TWM/adapter plus HaWoR head and PAM only."""
    groups = {}
    for name, parameter in pipeline.named_parameters():
        if name.startswith('frontend.model.mano_head.'):
            tail = name.removeprefix('frontend.model.mano_head.').lower()
            if 'deccam' in tail or tail.startswith('camera'):
                group = 'hawor_camera_readout'
            elif 'decpose' in tail:
                group = 'hawor_pose_readout'
            elif 'decshape' in tail or 'decbeta' in tail:
                group = 'hawor_beta_readout'
            else:
                group = 'hawor_head_shared'
        elif name.startswith('frontend.model.motion_module.'):
            group = 'hawor_pam'
        elif name.startswith('feedback.adapter.'):
            group = 'adapter'
        elif name.startswith('feedback.twm.'):
            tail = name.removeprefix('feedback.twm.')
            if tail.startswith('encoder.'):
                group = 'twm_encoder'
            elif tail.startswith('translation_head.'):
                group = 'twm_translation_head'
            elif tail.startswith('rotation_head.'):
                group = 'twm_rotation_head'
            elif tail.startswith('state_translation_scale_head.'):
                group = 'twm_translation_uncertainty'
            elif tail.startswith('state_rotation_concentration_head.'):
                group = 'twm_rotation_uncertainty'
            elif tail.startswith(('future_', 'mixture_head.')):
                group = 'twm_future'
            else:
                group = 'twm_input_other'
        else:
            continue
        groups.setdefault(group, []).append((name, parameter))
    selected = {id(p) for entries in groups.values() for _, p in entries}
    for parameter in pipeline.parameters():
        parameter.requires_grad_(id(parameter) in selected)
    return groups


def gradient(loss, named_parameters, *, retain_graph=False):
    values = torch.autograd.grad(loss, [p for _, p in named_parameters],
                                 allow_unused=True, retain_graph=retain_graph)
    # CPU summaries preserve None so disconnected objectives stay distinguishable.
    return {name: None if value is None else value.detach().float().cpu()
            for (name, _), value in zip(named_parameters, values)}


def comparison(one, two, names):
    first, second, dot = 0., 0., 0.
    tensors_a, tensors_b, finite = 0, 0, True
    for name in names:
        a, b = one.get(name), two.get(name)
        if a is not None:
            first += float(a.double().square().sum()); tensors_a += 1
            finite &= bool(torch.isfinite(a).all())
        if b is not None:
            second += float(b.double().square().sum()); tensors_b += 1
            finite &= bool(torch.isfinite(b).all())
        if a is not None and b is not None:
            dot += float((a.double() * b.double()).sum())
    norm_a, norm_b = first ** .5, second ** .5
    return {'first_norm': norm_a, 'second_norm': norm_b,
            'cosine': dot / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else None,
            'first_connected_tensors': tensors_a, 'second_connected_tensors': tensors_b,
            'finite': finite}


def group_comparison(one, two, groups):
    return {name: comparison(one, two, [key for key, _ in entries])
            for name, entries in groups.items()}


def clipping_summary(gradients, groups, max_norm):
    result = {}
    for label, predicate, complete in (
        ('twm', lambda key: key.startswith('feedback.twm.'), True),
        ('adapter', lambda key: key.startswith('feedback.adapter.'), True),
        ('hawor_head_pam_subset', lambda key: key.startswith('frontend.model.'), False),
    ):
        names = [key for entries in groups.values() for key, _ in entries if predicate(key)]
        norm = comparison(gradients, gradients, names)['first_norm']
        scale = min(1., max_norm / (norm + 1e-6))
        per_group = {}
        for name, entries in groups.items():
            members = [key for key, _ in entries if predicate(key)]
            if not members:
                continue
            value = comparison(gradients, gradients, members)['first_norm']
            per_group[name] = {'norm': value, 'norm_after_clip': value * scale,
                               'squared_norm_fraction': value ** 2 / norm ** 2 if norm > 0 else None}
        result[label] = {'complete_training_module': complete, 'norm': norm,
                         'clip_scale': scale, 'norm_after_clip': norm * scale,
                         'by_group': per_group,
                         'interpretation': 'exact training-module clip factor' if complete else
                         'head/PAM subset only; actual whole-HaWoR clip factor unavailable'}
    return result


def mirror_gradient_sanity(device):
    torch.manual_seed(20261016)
    matrix = torch.randn(2, 16, 3, 3, device=device, requires_grad=True)
    weight = torch.randn_like(matrix)
    sides = torch.tensor([False, True], device=device)
    result = mirror_rotations_x(matrix, sides)
    obtained, = torch.autograd.grad((result * weight).sum(), matrix)
    reflection = matrix.new_tensor([-1., 1., 1.])
    expected = weight.clone()
    expected[0] *= reflection[:, None] * reflection[None, :]
    tx = torch.randn(2, device=device, requires_grad=True)
    recovered = torch.where(sides, tx, -tx)
    tx_gradient, = torch.autograd.grad(recovered.sum(), tx)
    return {'matrix_srs_max_gradient_error': float((obtained - expected).abs().max()),
            'tx_unflip_derivative_left_right': tx_gradient.tolist(),
            'matrix_srs_involution_max_error': float((mirror_rotations_x(result, sides) - matrix).abs().max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'source-manifest', 'motion-manifest', 'hawor-root',
                 'hawor-checkpoint', 'hawor-config', 'mano-models', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--clips-per-side', type=int, default=2)
    parser.add_argument('--frames', type=int, default=16)
    parser.add_argument('--seed', type=int, default=20261016)
    parser.add_argument('--motion-samples-per-source', type=int, default=2)
    args = parser.parse_args()
    if not str(Path.cwd()).startswith('/sciclone/') or not torch.cuda.is_available():
        raise RuntimeError('Numerical diagnostics must execute in the remote SciClone GPU workspace')
    if args.output.exists():
        raise ValueError('Use a new output JSON; do not overwrite diagnostic evidence')
    if args.frames not in (16, 32) or args.clips_per_side < 1:
        raise ValueError('Use 16/32 frame prefixes and positive fixed clips per side')
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    args.mode, args.split, args.clip_length = 'joint', 'val', 32
    pipeline, info = load_model(args, device)
    pipeline.frontend.trainable = True  # retain eval mode, permit autograd at bridge boundary
    pipeline.feedback.trainable_scope = 'full'
    pipeline.eval()
    groups = selected_groups(pipeline)
    named_parameters = [entry for entries in groups.values() for entry in entries]
    dataset = VisualFeedbackRGBDataset(args.source_manifest, 'train', clip_length=32, hawor_root=args.hawor_root)
    selected = {0: [], 1: []}
    seen_recordings = set()
    for index, (record_index, start, count) in enumerate(dataset.windows):
        record = dataset.records[record_index]
        arrays = dataset._load_reference(record_index)
        hand = int(arrays['handedness'][start])
        if count >= args.frames and record['recording'] not in seen_recordings and len(selected[hand]) < args.clips_per_side:
            selected[hand].append({'index': index, 'recording': record['recording'], 'record_index': record_index,
                                   'start': start, 'frames': args.frames, 'handedness': hand})
            seen_recordings.add(record['recording'])
    if any(len(value) != args.clips_per_side for value in selected.values()):
        raise ValueError('Insufficient distinct full-length train recordings for balanced diagnostics')
    replay_config = info['replay_config']
    if file_sha256(args.motion_manifest) != info['training_identity']['motion_manifest_sha256']:
        raise ValueError('Motion train manifest differs from actual joint training')
    motion_dataset = ForecastTrajectoryWindows(args.motion_manifest, 'train', history=replay_config['history_frames'],
        forecast_frames=replay_config['model']['forecast_horizon'], min_future_frames=replay_config['min_future_frames'],
        window_stride=replay_config['window_stride'], cache_sequences=16, cache_bytes=134217728)
    motion_indices = motion_dataset.fixed_indices_by_dataset(args.motion_samples_per_source)
    selection = {'rgb_train': selected, 'motion_train_indices_by_source': motion_indices,
                 'seed': args.seed, 'frames': args.frames, 'checkpoint_step': info['checkpoint_step'],
                 'source_manifest_sha256': file_sha256(args.source_manifest),
                 'motion_manifest_sha256': file_sha256(args.motion_manifest)}
    atomic_json(args.output.with_name(args.output.stem + '_selection.json'), selection)
    before = parameter_digest(pipeline.feedback.twm)
    selected_before = selected_parameter_digest(named_parameters)
    motion_batch = default_collate([motion_dataset[index] for indices in motion_indices.values() for index in indices])
    motion_batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in motion_batch.items()}
    generator = torch.Generator(device=device).manual_seed(args.seed + 901)
    h, raw, candidate, kwargs, target, future = prepare_batch(motion_batch, generator, 'mixed',
        condition_on_beta=replay_config.get('condition_on_beta', False), corruption_profile=replay_config.get('corruption_profile'))
    with torch.autocast('cuda', dtype=torch.bfloat16):
        motion_output = pipeline.feedback.twm(h, raw, candidate, **kwargs)
        motion_loss, motion_terms = objective(motion_output, target, future, motion_batch['future_valid'],
                                             motion_batch['future_dt'], replay_config)
    motion_grad = gradient(motion_loss, named_parameters)
    report = {'schema': 'handloop.joint_gradient_diagnostic.v1', 'created_utc': utc(), 'host': socket.gethostname(),
        'checkpoint': str(args.checkpoint), 'checkpoint_step': info['checkpoint_step'], 'selection': selection,
        'source_sha256': {name: file_sha256(Path(__file__).resolve().parents[1] / name) for name in
            ('scripts/diagnose_joint_gradients.py', 'scripts/train_visual_joint.py',
             'scripts/train_probabilistic_twm.py', 'src/handloop/models/visual_feedback.py',
             'src/handloop/models/probabilistic_twm.py', 'src/handloop/visual/hawor.py',
             'src/handloop/visual/codec.py')},
        'actual_training_configuration': info['training_config'],
        'motion_objective_coefficients': {key: replay_config[key] for key in
            ('current_nll_weight', 'forecast_nll_weight', 'forecast_geometry_weight',
             'velocity_weight', 'moe_balance_weight')},
        'mode': 'eval/dropout off; autograd on; no optimizer; balanced fixed train prefixes',
        'parameter_scope': {name: {'parameters': sum(p.numel() for _, p in entries),
                                  'names': [key for key, _ in entries]} for name, entries in groups.items()},
        'scope_limit': 'ViT/IAM frozen as upstream feature producers; selected head/PAM/TWM/adapter partial derivatives preserved; no whole-HaWoR clip factor',
        'motion': {'loss': float(motion_loss.detach()), 'terms': {key: float(value.detach()) for key, value in motion_terms.items()},
                   'handedness': motion_batch['handedness'].detach().cpu().tolist(),
                   'clipping': clipping_summary(motion_grad, groups, info['training_config']['grad_clip'])},
        'mirror_autograd_sanity': mirror_gradient_sanity(device), 'clips': [],
        'interpretation': 'Gradient scale/cosine is a local diagnostic at frozen parameters. Different visual/motion content is not a matched loss ablation. Dividing visual NLL by 51 changes no cosine; this is a unit diagnostic, not a proposed training change.'}
    cross_groups = {name: entries for name, entries in groups.items() if
        name in ('hawor_camera_readout', 'hawor_pose_readout', 'hawor_beta_readout', 'adapter')
        or name.startswith('twm_')}
    cross_groups['twm_all'] = [entry for name, entries in groups.items()
                               if name.startswith('twm_') for entry in entries]
    cross_names = {key for entries in cross_groups.values() for key, _ in entries}
    side_sums = {0: {}, 1: {}}
    side_clipped_sums = {0: {}, 1: {}}
    side_counts = {0: 0, 1: 0}
    del motion_output, motion_loss, motion_batch, motion_terms
    gc.collect(); torch.cuda.empty_cache()
    for hand in (0, 1):
        for selection_item in selected[hand]:
            batch = default_collate([dataset[selection_item['index']]])
            batch = {key: value[:, :args.frames].to(device) if isinstance(value, torch.Tensor) and value.ndim >= 2
                     else value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            captured = []
            def capture(module, inputs, kwargs):
                captured.append({'raw_anchor': inputs[1], 'candidate_anchor': inputs[2]})
            hook = pipeline.feedback.twm.register_forward_pre_hook(capture, with_kwargs=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                observed = pipeline._observe(batch)
                output = pipeline.feedback.forward_clip(observed['image_features'], observed['raw_state_world'],
                    timestamps=batch['timestamps'], frame_valid=observed['frame_valid'], availability=observed['raw_available'],
                    image_valid=observed['image_valid'] & observed['bbox_valid'], handedness=observed['is_right'].long(),
                    world_to_camera=inverse_rigid(observed['camera_to_world']), camera_intrinsics=batch['intrinsics'],
                    crop_transform=batch['crop_transform'], beta=observed['beta'], reset=batch.get('reset'),
                    return_forecast=False, joint_geometry=pipeline.joint_geometry)
                valid = output['frame_valid'] & batch['target_valid'].all(-1)
                final_nll = feedback_current_nll(output, batch['target_world'], batch['target_valid'])
                density = current_distribution_log_prob({key: output[key][valid] for key in
                    ('state', 'state_translation_std', 'state_rotation_concentration')},
                    transform_state(batch['target_world'][valid], inverse_rigid(output['anchor_world'][valid])))
                density_terms = {'translation_nll': float(-density['translation'].mean().detach()),
                                 'root_nll': float(-density['rotation'][..., 0].mean().detach()),
                                 'finger_nll_sum15': float(-density['rotation'][..., 1:].sum(-1).mean().detach())}
                final_geometry = state_geometry_loss(output['state_world'][valid], batch['target_world'][valid]).mean()
                raw_geometry = state_geometry_loss(observed['raw_state_world'][valid], batch['target_world'][valid]).mean()
                raw_wrist = ((observed['raw_state_world'][valid][..., :3] - batch['target_world'][valid][..., :3]) / .05).square().mean()
            hook.remove()
            visual_grad = gradient(final_nll, named_parameters, retain_graph=True)
            geometry_grad = gradient(final_geometry, named_parameters, retain_graph=True)
            raw_geometry_grad = gradient(raw_geometry, named_parameters, retain_graph=True)
            raw_wrist_grad = gradient(raw_wrist, named_parameters, retain_graph=True)
            last = args.frames - 1
            local_jacobian = {'raw_world_same_frame_position': [], 'last_twm_candidate_anchor_position': [],
                              'raw_world_prior_frames_position_max_abs': []}
            for axis in range(3):
                raw_derivative, candidate_derivative = torch.autograd.grad(output['state_world'][0, last, axis],
                    [observed['raw_state_world'], captured[-1]['candidate_anchor']], retain_graph=axis < 2,
                    allow_unused=True)
                local_jacobian['raw_world_same_frame_position'].append(raw_derivative[0, last, :3].detach().cpu().tolist())
                local_jacobian['raw_world_prior_frames_position_max_abs'].append(float(raw_derivative[0, :last, :3].abs().max()))
                local_jacobian['last_twm_candidate_anchor_position'].append(candidate_derivative[0, :3].detach().cpu().tolist())
            scaled_visual = {name: None if value is None else value / 51 for name, value in visual_grad.items()}
            entry = {**selection_item, 'valid_states': int(valid.sum()),
                'loss': {'visual_full_nll': float(final_nll.detach()), 'visual_nll_div51': float(final_nll.detach()) / 51,
                         **density_terms,
                         'final_geometry': float(final_geometry.detach()), 'raw_geometry': float(raw_geometry.detach()),
                         'raw_wrist_mse_normalized_005m': float(raw_wrist.detach())},
                'visual_vs_motion': group_comparison(visual_grad, motion_grad, groups),
                'visual_div51_vs_motion': group_comparison(scaled_visual, motion_grad, groups),
                'visual_vs_final_geometry': group_comparison(visual_grad, geometry_grad, groups),
                'visual_vs_raw_geometry': group_comparison(visual_grad, raw_geometry_grad, groups),
                'visual_vs_raw_wrist': group_comparison(visual_grad, raw_wrist_grad, groups),
                'visual_clipping': clipping_summary(visual_grad, groups, info['training_config']['grad_clip']),
                'visual_div51_clipping': clipping_summary(scaled_visual, groups, info['training_config']['grad_clip']),
                'local_position_jacobian': local_jacobian}
            for name in cross_names:
                value = visual_grad.get(name)
                if value is None:
                    continue
                if name not in side_sums[hand]:
                    side_sums[hand][name] = value.clone()
                else:
                    side_sums[hand][name].add_(value)
                if name.startswith('feedback.'):
                    module = 'twm' if name.startswith('feedback.twm.') else 'adapter'
                    clipped = value * entry['visual_clipping'][module]['clip_scale']
                    if name not in side_clipped_sums[hand]:
                        side_clipped_sums[hand][name] = clipped
                    else:
                        side_clipped_sums[hand][name].add_(clipped)
            side_counts[hand] += 1
            report['clips'].append(entry)
            atomic_json(args.output.with_name(args.output.stem + '_progress.json'), report)
            print(json.dumps({'utc': utc(), 'checkpoint_step': info['checkpoint_step'], 'hand': hand,
                              'recording': selection_item['recording'], 'loss': entry['loss'],
                              'twm_visual_norm': entry['visual_clipping']['twm']['norm'],
                              'twm_clip_scale': entry['visual_clipping']['twm']['clip_scale']}), flush=True)
            del output, observed, captured, batch, visual_grad, geometry_grad, raw_geometry_grad, raw_wrist_grad, scaled_visual, density
            gc.collect(); torch.cuda.empty_cache()
    means = {hand: {name: value / side_counts[hand] for name, value in values.items()}
             for hand, values in side_sums.items()}
    clipped_means = {hand: {name: value / side_counts[hand] for name, value in values.items()}
                     for hand, values in side_clipped_sums.items()}
    report['left_right_mean_visual_gradients'] = {
        'clips_by_side': side_counts, 'reduction': 'equal mean of selected train clips; each contains same valid frame count',
        'first': 'left', 'second': 'right',
        'unclipped': group_comparison(means[0], means[1], cross_groups),
        'per_clip_training_module_clipped_twm_adapter': group_comparison(clipped_means[0], clipped_means[1],
            {name: entries for name, entries in cross_groups.items() if name == 'adapter' or name.startswith('twm_')}),
        'interpretation': 'Different selected left/right recordings are not mirror-matched actions. Negative cosine is local optimization tension, not proof of the cause of test disparity. HaWoR clipping unavailable outside head/PAM subset.'}
    report['twm_parameter_sha256_before'] = before
    report['twm_parameter_sha256_after'] = parameter_digest(pipeline.feedback.twm)
    report['selected_parameter_sha256_before'] = selected_before
    report['selected_parameter_sha256_after'] = selected_parameter_digest(named_parameters)
    report['parameters_unchanged'] = (report['twm_parameter_sha256_before'] == report['twm_parameter_sha256_after'] and
                                    report['selected_parameter_sha256_before'] == report['selected_parameter_sha256_after'])
    if not report['parameters_unchanged']:
        raise RuntimeError('Read-only diagnostic unexpectedly changed TWM parameters')
    report['complete_utc'] = utc()
    atomic_json(args.output, report)
    print(json.dumps({'complete': str(args.output), 'parameters_unchanged': True}), flush=True)


if __name__ == '__main__':
    main()
