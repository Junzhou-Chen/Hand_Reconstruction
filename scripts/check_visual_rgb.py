#!/usr/bin/env python3
"""Remote-only online-RGB crop, label-boundary and padding contracts.

Compare the new differentiable training loader with the existing official
extract_clip preprocessing on real RGB, then run one padded feedback clip.
No optimizer, checkpoint mutation, dataset preparation, or accuracy claim.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import torch
from torch.utils.data import default_collate

from handloop.data.visual_clips import file_sha256
from handloop.data.visual_rgb import VisualFeedbackRGBDataset
from handloop.models.probabilistic_twm import ProbabilisticTWM, ProbabilisticTWMConfig
from handloop.models.visual_feedback import VisualFeedbackConfig, VisualFeedbackSystem
from handloop.visual.hawor import FrozenHaWoRBridge
from handloop.visual.pipeline import HandLoopFeedbackPipeline, effective_crop_transform


def close(a, b, name, atol=2e-5):
    difference = float((a.float().cpu() - b.float().cpu()).abs().max())
    if not torch.allclose(a.float().cpu(), b.float().cpu(), atol=atol, rtol=1e-5):
        raise AssertionError(f'{name}: max abs error {difference}')
    return difference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-manifest', type=Path, required=True)
    parser.add_argument('--twm-checkpoint', type=Path, required=True)
    parser.add_argument('--hawor-root', type=Path, required=True)
    parser.add_argument('--hawor-checkpoint', type=Path, required=True)
    parser.add_argument('--hawor-config', type=Path, required=True)
    parser.add_argument('--mano-models', type=Path, required=True)
    parser.add_argument('--train-config', type=Path)
    parser.add_argument('--split', choices=('train', 'val'), default='val')
    parser.add_argument('--real-frames', type=int, default=4)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--autocast-bf16', action='store_true')
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if not Path('/sciclone/data10/jchen57').is_dir():
        raise RuntimeError('Run RGB/model contracts in the remote compute Pod')
    if not 1 <= args.real_frames < 16:
        raise ValueError('Use 1..15 real frames to exercise one padded official block')
    report = {'schema': 'handloop.visual_rgb_contracts.v1', 'passed': False,
              'utc': datetime.now(timezone.utc).isoformat(), 'checks': {},
              'scope': 'real RGB implementation contracts only; no accuracy or optimization claim'}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.set_num_threads(1)
        torch.manual_seed(20261016)
        device = torch.device(args.device)
        if args.autocast_bf16 and device.type != 'cuda':
            raise ValueError('The BF16 production contract requires a CUDA device')
        if device.type == 'cuda':
            torch.cuda.set_device(device)
        dataset = VisualFeedbackRGBDataset(args.source_manifest, args.split,
            clip_length=16, hawor_root=args.hawor_root)
        record_index, start, available = dataset.windows[0]
        count = min(args.real_frames, available)
        # Bound a normal source window in memory solely for a deterministic
        # tail contract. No source files or stored window metadata are edited.
        dataset.windows = [(record_index, start, count)]
        sample = dataset[0]
        arrays, record = dataset._load_reference(record_index), dataset.records[record_index]
        for key in ('frame_valid', 'detected', 'bbox_valid', 'image_valid', 'target_valid'):
            if bool(sample[key][count:].any()):
                raise AssertionError(f'Padded {key} must be false')
        if not bool(sample['reset'][0]) or bool(sample['reset'][1:].any()):
            raise AssertionError('Only the first frame of this contiguous clip resets history')
        report['checks']['tail_masks'] = True
        frontend = FrozenHaWoRBridge.from_pretrained(args.hawor_root, args.hawor_checkpoint,
            args.hawor_config, args.mano_models, device=device)
        captured = []
        def capture(_module, positional):
            block = positional[0]
            if 'target_world' in block or 'target_valid' in block:
                raise AssertionError('Ground truth entered the official HaWoR forward')
            captured.append({key: value.detach().cpu().clone() for key, value in block.items()})
        handle = frontend.register_forward_pre_hook(capture)
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                             enabled=args.autocast_bf16):
            old = frontend.extract_clip(
                [dataset.path.parent / p for p in record['image_paths'][start:start + count]],
                arrays['bbox_xyxy'][start:start + count], arrays['handedness'][start:start + count],
                arrays['camera_intrinsics'][start:start + count], arrays['camera_to_world'][start:start + count],
                detected=sample['detected'][:count], bbox_valid=sample['bbox_valid'][:count],
                image_valid=sample['image_valid'][:count])
        expected = {key: torch.cat([block[key][0] for block in captured], dim=0)[:count]
                    for key in ('img', 'center', 'center_unflipped', 'scale', 'img_focal', 'img_center')}
        report['checks']['crop_errors'] = {
            key: close(sample[key][:count], value, key) for key, value in expected.items()}
        report['checks']['crop_errors']['normalized_crop_transform'] = close(
            sample['crop_transform'][:count], effective_crop_transform(old['crop_transform']),
            'effective crop transform')
        source = torch.load(args.twm_checkpoint, map_location='cpu', weights_only=False)
        twm = ProbabilisticTWM(ProbabilisticTWMConfig(**source['config']['model']))
        twm.load_state_dict(source['model'], strict=True)
        adapter = ({**json.loads(args.train_config.read_text())['adapter']}
                   if args.train_config else {'refinement_steps': 2, 'candidate_guided_queries': True})
        feedback = VisualFeedbackSystem(twm, VisualFeedbackConfig(**adapter)).to(device).eval()
        pipeline = HandLoopFeedbackPipeline(frontend, feedback).eval()
        batch = default_collate([sample])
        batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
        # Deliberately poisonous supervision must have no effect on inference.
        batch['target_world'].fill_(float('nan'))
        batch['target_valid'].zero_()
        captured.clear()
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                             enabled=args.autocast_bf16):
            output = pipeline(batch, return_forecast=False)
        handle.remove()
        if len(captured) != 1:
            raise AssertionError('Expected one official 16-frame forward')
        report['checks']['no_gt_forward'] = True
        report['checks']['online_crop_vs_extract_clip'] = True
        report['checks']['raw_world_parity_mixed_units'] = close(
            output['raw_world'][0, :count], old['raw_state_world'], 'raw 99D')
        if not bool(torch.isfinite(output['state_world']).all()):
            raise AssertionError('Poisoned loss-only GT affected numerical prediction')
        expected_count = min(count, feedback.config.history_length)
        if int(output['final_history'].valid.sum()) != expected_count:
            raise AssertionError('Padded frames or intermediate refinement states entered history')
        if not torch.equal(output['history_count'][0, :count].cpu(), torch.arange(count)):
            raise AssertionError('History must commit exactly once per completed real frame')
        if not bool((output['history_count'][0, count:] == expected_count).all()):
            raise AssertionError('Padded frames advanced history')
        report['checks']['padding_not_committed'] = True
        report['checks']['one_commit_per_real_frame'] = True
        report.update({'passed': True, 'real_frames': count, 'padded_frames': 16-count,
                       'recording': record['recording'], 'split': args.split,
                       'source_manifest_sha256': dataset.manifest_sha256,
                       'twm_checkpoint_sha256': file_sha256(args.twm_checkpoint),
                       'twm_step': source.get('step'), 'refinement_steps': feedback.config.refinement_steps,
                       'autocast_bf16': args.autocast_bf16,
                       'state_shape': list(output['state_world'].shape),
                       'final_history_valid': int(output['final_history'].valid.sum()),
                       'frontend': frontend.provenance()})
    except Exception as exc:
        report['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        raise
    finally:
        args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'passed': report['passed'], 'report': str(args.report)}))


if __name__ == '__main__':
    main()
