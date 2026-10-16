#!/usr/bin/env python3
"""Remote-only train-split calibration of separate visual geometry terms.

No optimizer or parameter update. ViT/IAM are upstream feature producers;
head/PAM, complete adapter and TWM remain differentiable. Norms for HaWoR are
therefore the explicitly selected head/PAM subset, not the full backbone.
Probe weights are engineering settings, not evidence of optimal coefficients.
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
from handloop.visual.losses import (VisualGeometryLossConfig, visual_loss_terms,
    reduce_visual_loss_terms, local_visual_term_means, visual_loss_coefficients)
from train_visual_joint import build_pipeline, move
from diagnose_joint_gradients import selected_groups, gradient, comparison, selected_parameter_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-manifest', 'twm-checkpoint', 'hawor-root', 'hawor-checkpoint',
                 'hawor-config', 'mano-models', 'config', 'report'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--frames', type=int, default=4)
    parser.add_argument('--clips-per-hand', type=int, default=1)
    parser.add_argument('--seed', type=int, default=20261016)
    for name in ('nll-weight', 'raw-wrist-weight', 'final-joint-weight', 'raw-wrist-scale-m', 'final-joint-scale-m'):
        parser.add_argument('--'+name, type=float)
    args = parser.parse_args()
    if not Path('/sciclone/data10/jchen57').is_dir() or not torch.cuda.is_available():
        raise RuntimeError('Run actual RGB gradient probes in the remote GPU Pod')
    if not 1 <= args.frames <= 16 or args.clips_per_hand < 1:
        raise ValueError('Use 1..16 real frames and positive clips per hand')
    if args.report.exists():
        raise ValueError('Preserve prior diagnostic reports; use a new output filename')
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    config = json.loads(args.config.read_text())
    # The probe enables both terms if absent. It never writes a train config.
    loss_values = {'raw_wrist_weight': 1., 'final_joint_weight': 1.,
                   'raw_wrist_scale_m': .05, 'final_joint_scale_m': .05,
                   **config.get('visual_geometry_supervision', {})}
    for name in ('nll_weight', 'raw_wrist_weight', 'final_joint_weight', 'raw_wrist_scale_m', 'final_joint_scale_m'):
        if getattr(args, name) is not None:
            loss_values[name] = getattr(args, name)
    loss_config = VisualGeometryLossConfig.from_dict(loss_values)
    if loss_config.raw_wrist_weight <= 0 or loss_config.final_joint_weight <= 0:
        raise ValueError('A calibration probe requires both geometry terms enabled')
    source_hash = file_sha256(args.twm_checkpoint)
    source_stat = args.twm_checkpoint.stat()
    checkpoint = torch.load(args.twm_checkpoint, map_location='cpu', weights_only=False, mmap=True)
    pipeline = build_pipeline(args, config, device, checkpoint)
    del checkpoint
    pipeline.eval()
    groups = selected_groups(pipeline)
    named = [entry for entries in groups.values() for entry in entries]
    before = selected_parameter_digest(named)
    group_names = {name: [key for key, _ in entries] for name, entries in groups.items()}
    group_names.update({name: [key for key, _ in named if key.startswith(prefix)]
                        for name, prefix in (('twm_complete', 'feedback.twm.'),
                                             ('adapter_complete', 'feedback.adapter.'),
                                             ('hawor_head_PAM_subset', 'frontend.model.'))})
    dataset = VisualFeedbackRGBDataset(args.source_manifest, 'train', clip_length=16, hawor_root=args.hawor_root)
    chosen, seen, side_counts = [], set(), {0: 0, 1: 0}
    for record_index, start, count in dataset.windows:
        record = dataset.records[record_index]
        hand = int(record['handedness'])
        if count >= args.frames and side_counts[hand] < args.clips_per_hand and record['recording'] not in seen:
            chosen.append((record_index, start, args.frames))
            seen.add(record['recording'])
            side_counts[hand] += 1
    if min(side_counts.values()) < args.clips_per_hand:
        raise ValueError('Insufficient distinct train recordings for both hands')
    report = {'schema': 'handloop.visual_geometry_gradient_probe.v1', 'passed': False,
              'utc': datetime.now(timezone.utc).isoformat(), 'seed': args.seed,
              'visual_geometry_supervision': asdict(loss_config),
              'effective_coefficients': visual_loss_coefficients(loss_config),
              'scope': 'train-only, eval/dropout-off, ViT/IAM frozen upstream; head/PAM+full adapter/TWM; no optimizer, not actual training-batch or Adam update',
              'source_manifest_sha256': dataset.manifest_sha256, 'twm_checkpoint_sha256': source_hash,
              'parameter_group_counts': {name: sum(parameter.numel() for _, parameter in entries) for name, entries in groups.items()},
              'clips': []}
    captured = []
    def no_labels(_module, positional):
        if any(key.startswith(('target_', 'reference_')) or key == 'beta' for key in positional[0]):
            raise AssertionError('Loss-only ground truth entered HaWoR')
        captured.append(True)
    hook = pipeline.frontend.register_forward_pre_hook(no_labels)
    try:
        for record_index, start, count in chosen:
            dataset.windows = [(record_index, start, count)]
            batch = move(default_collate([dataset[0]]), device)
            # Poisoned source beta is deliberately present in the outer batch.
            batch['beta'] = torch.full((1, 16, 10), float('nan'), device=device)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                output = pipeline(batch, return_forecast=False)
                terms = visual_loss_terms(output, batch, pipeline.frontend.codec, loss_config)
                loss, measurements = reduce_visual_loss_terms(terms, loss_config, distributed=False)
            means = local_visual_term_means(terms)
            gradients = {name: gradient(value, named, retain_graph=True) for name, value in means.items()}
            combined = gradient(loss, named)
            values = {name: float(value.detach()) for name, value in means.items()}
            norms, cosines = {}, {}
            for label, names in group_names.items():
                norms[label] = {}
                for term, gradients_one in gradients.items():
                    summary = comparison(gradients_one, gradients_one, names)
                    if not summary['finite']:
                        raise FloatingPointError('Nonfinite gradient '+label+'/'+term)
                    norms[label][term] = {'unweighted_norm': summary['first_norm'],
                        'effective_weighted_norm': summary['first_norm'] * visual_loss_coefficients(loss_config)[term],
                        'connected_tensors': summary['first_connected_tensors']}
                norms[label]['combined'] = comparison(combined, combined, names)
                cosines[label] = {}
                labels = list(gradients)
                for one, first in enumerate(labels):
                    for second in labels[one+1:]:
                        cosines[label][first+' vs '+second] = comparison(gradients[first], gradients[second], names)['cosine']
            for label in ('hawor_camera_readout', 'hawor_beta_readout'):
                if label not in norms:
                    raise AssertionError('Expected official readout group absent: '+label)
            if norms['hawor_camera_readout']['raw_wrist_geometry_m']['unweighted_norm'] <= 0:
                raise AssertionError('Raw-J0 supervision failed to reach camera readout')
            if norms['hawor_beta_readout']['final_mpjpe21_m']['unweighted_norm'] <= 0:
                raise AssertionError('Final geometry supervision failed to reach own predicted beta')
            report['clips'].append({'recording': dataset.records[record_index]['recording'], 'split': 'train',
                'handedness': int(dataset.records[record_index]['handedness']), 'start': start, 'real_frames': count,
                'loss_means_unweighted': values, 'measurements': measurements,
                'per_group_gradient_norms': norms, 'per_group_term_cosines': cosines})
            del gradients, combined, output, terms, means, loss, batch
        after = selected_parameter_digest(named)
        if before != after or file_sha256(args.twm_checkpoint) != source_hash:
            raise AssertionError('Read-only probe modified source weights/parameters')
        stat_after = args.twm_checkpoint.stat()
        if (source_stat.st_size, source_stat.st_mtime_ns) != (stat_after.st_size, stat_after.st_mtime_ns):
            raise AssertionError('Source checkpoint file changed')
        report.update(passed=True, parameters_unchanged=True, no_reference_forward=True,
                      source_checkpoint_unchanged=True, official_forward_calls=len(captured))
    except Exception as exc:
        report['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        raise
    finally:
        hook.remove()
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'passed': report['passed'], 'report': str(args.report)}))


if __name__ == '__main__':
    main()
