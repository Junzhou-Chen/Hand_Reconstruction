#!/usr/bin/env python3
"""Remote-only real mixed-source RGB forward/backward contract, no optimizer.

Exactly two train sources contribute one short clip each to the same collated
batch. The complete visual backbone, adapter and TWM retain their trainability
scope; this checks integration rather than fitting or evaluating accuracy.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

import torch
from torch.utils.data import default_collate

from handloop.data.visual_clips import file_sha256
from handloop.data.visual_rgb import VisualFeedbackRGBDataset
from handloop.visual.losses import VisualGeometryLossConfig, visual_loss_terms, reduce_visual_loss_terms
from train_visual_joint import build_pipeline, move, gradient_groups
from diagnose_joint_gradients import selected_parameter_digest


def gradient_summary(module):
    gradients = [parameter.grad.detach().float() for parameter in module.parameters()
                 if parameter.requires_grad and parameter.grad is not None]
    return {'connected_tensors': len(gradients),
            'norm': float(torch.stack([value.square().sum() for value in gradients]).sum().sqrt()) if gradients else 0.,
            'finite': all(bool(torch.isfinite(value).all()) for value in gradients)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-manifest', type=Path, action='append', required=True)
    parser.add_argument('--source-handedness', type=int, action='append', choices=(0, 1))
    for name in ('twm-checkpoint', 'hawor-root', 'hawor-checkpoint', 'hawor-config', 'mano-models', 'config', 'report'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--frames', type=int, default=8)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=20261016)
    args = parser.parse_args()
    if not Path('/sciclone/data10/jchen57').is_dir() or not torch.cuda.is_available():
        raise RuntimeError('Run real mixed-source RGB/model checks in the remote GPU Pod')
    if len(args.source_manifest) != 2 or len(set(args.source_manifest)) != 2 or not 1 <= args.frames <= 16:
        raise ValueError('Use exactly two distinct source manifests and 1..16 real frames per clip')
    hands = args.source_handedness or [0, 1]
    if len(hands) != 2:
        raise ValueError('Provide one explicit handedness per source')
    if args.report.exists():
        raise FileExistsError('Preserve previous contract reports; use a new filename')
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    config = json.loads(args.config.read_text())
    loss_config = VisualGeometryLossConfig.from_dict(config.get('visual_geometry_supervision'))
    if loss_config.raw_wrist_weight <= 0 or loss_config.final_joint_weight <= 0:
        raise ValueError('Mixed geometry integration requires both geometric terms enabled')
    identities = {name: {'path': str(path), 'sha256': file_sha256(path),
                         'bytes': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns}
                  for name, path in (('twm', args.twm_checkpoint), ('hawor', args.hawor_checkpoint))}
    source = torch.load(args.twm_checkpoint, map_location='cpu', weights_only=False, mmap=True)
    pipeline = build_pipeline(args, config, device, source)
    del source
    # Match actual training mode/checkpointing; unlike earlier local-gradient
    # probes, ViT and IAM remain trainable here. No optimizer exists.
    pipeline.train()
    named = [(name, parameter) for name, parameter in pipeline.named_parameters() if parameter.requires_grad]
    parameters_before = selected_parameter_digest(named)
    report = {'schema': 'handloop.mixed_visual_geometry_contract.v1', 'passed': False,
              'utc': datetime.now(timezone.utc).isoformat(), 'seed': args.seed,
              'scope': 'real two-source train-only same-batch full-backbone forward/backward; no optimizer or accuracy claim; distributed reduction verified separately',
              'visual_geometry_supervision': asdict(loss_config), 'source_checkpoints': identities,
              'trainable_parameter_count': sum(parameter.numel() for _, parameter in named), 'selection': []}
    try:
        samples, datasets = [], []
        for path, hand in zip(args.source_manifest, hands):
            dataset = VisualFeedbackRGBDataset(path, 'train', clip_length=16, hawor_root=args.hawor_root)
            selected = next(((record_index, start, args.frames) for record_index, start, count in dataset.windows
                             if count >= args.frames and int(dataset.records[record_index]['handedness']) == hand), None)
            if selected is None:
                raise ValueError('Source has no eligible train clip for the requested side')
            dataset.windows = [selected]
            sample = dataset[0]
            if samples and set(sample) != set(samples[0]):
                raise AssertionError('Mixed source samples have different default_collate keys')
            samples.append(sample)
            datasets.append(dataset)
            report['selection'].append({'source_manifest': str(path), 'source_manifest_sha256': dataset.manifest_sha256,
                'recording': sample['recording'], 'handedness': hand, 'split': 'train',
                'start': selected[1], 'real_frames': args.frames,
                'reference_geometry_metadata': sample['reference_geometry_metadata']})
        batch = move(default_collate(samples), device)
        if len(set(batch['reference_geometry_metadata']['dataset'])) != 2:
            raise AssertionError('This integration check must include two different datasets')
        report['collated_tensor_shapes'] = {key: list(value.shape) for key, value in batch.items() if isinstance(value, torch.Tensor)}
        with torch.autocast('cuda', dtype=torch.bfloat16):
            output = pipeline(batch, return_forecast=False)
            for key in ('raw_world', 'state_world', 'beta'):
                output[key].retain_grad()
            terms = visual_loss_terms(output, batch, pipeline.frontend.codec, loss_config)
            loss, measurements = reduce_visual_loss_terms(terms, loss_config, distributed=False)
        loss.backward()
        report['loss'] = float(loss.detach())
        report['measurements'] = measurements
        report['gradients'] = {name: gradient_summary(module) for name, module in gradient_groups(pipeline).items()
                               if module is not None}
        report['readout_gradients'] = {}
        for label, token in (('camera', 'deccam'), ('beta', 'decshape')):
            selected = [(name, parameter) for name, parameter in pipeline.frontend.model.mano_head.named_parameters()
                        if token in name]
            if not selected:
                raise AssertionError('Expected official '+label+' readout absent')
            gradients = [parameter.grad for _, parameter in selected if parameter.grad is not None]
            norm = float(torch.stack([value.detach().float().square().sum() for value in gradients]).sum().sqrt()) if gradients else 0.
            if norm <= 0:
                raise AssertionError('Mixed loss did not reach '+label+' readout')
            report['readout_gradients'][label] = {'norm': norm, 'connected_tensors': len(gradients)}
        connected, disconnected = [], []
        for name, parameter in named:
            if parameter.grad is None:
                disconnected.append(name)
            else:
                if not bool(torch.isfinite(parameter.grad).all()):
                    raise FloatingPointError('Nonfinite parameter gradient '+name)
                connected.append(name)
        for name in ('hawor_vit', 'hawor_iam', 'hawor_head', 'hawor_pam', 'adapter', 'twm_encoder', 'twm_current', 'twm_rotation'):
            if name not in report['gradients'] or report['gradients'][name]['norm'] <= 0:
                raise AssertionError('Missing mixed-source gradient '+name)
        report['output_gradients'] = {}
        for key in ('raw_world', 'state_world', 'beta'):
            gradient = output[key].grad
            if gradient is None or not bool(torch.isfinite(gradient).all()) or float(gradient.norm()) <= 0:
                raise AssertionError('Missing/nonfinite mixed-source output gradient '+key)
            report['output_gradients'][key] = float(gradient.norm())
        report['connected_parameter_tensors'] = len(connected)
        report['disconnected_parameter_names'] = disconnected
        parameters_after = selected_parameter_digest(named)
        if parameters_after != parameters_before:
            raise AssertionError('A no-optimizer check unexpectedly changed trainable parameters')
        for name, path in (('twm', args.twm_checkpoint), ('hawor', args.hawor_checkpoint)):
            after = {'path': str(path), 'sha256': file_sha256(path),
                     'bytes': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns}
            if identities[name] != after:
                raise AssertionError('Source checkpoint changed: '+name)
        if any(file_sha256(dataset.path) != dataset.manifest_sha256 for dataset in datasets):
            raise AssertionError('Source manifest changed during the mixed check')
        report.update(passed=True, parameters_unchanged=True, source_checkpoints_unchanged=True,
                      all_present_parameter_gradients_finite=True, mixed_default_collate=True,
                      source_manifests_unchanged=True, peak_allocated_bytes=torch.cuda.max_memory_allocated(device))
    except Exception as exc:
        report['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        raise
    finally:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'passed': report['passed'], 'report': str(args.report)}))


if __name__ == '__main__':
    main()
