#!/usr/bin/env python3
"""Train the visual feedback readout with ONE final-current state NLL.

Run on the remote compute Pod. The cache contains genuine frozen-HaWoR image
features. Reference states enter only the objective, never the model forward.
Each sampled clip starts with empty history and rolls out its own predictions.
This is an integration trainer, not a reproduction of HaWoR's original train.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import random
import signal
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from handloop import geometry as g
from handloop.data.visual_clips import VisualFeedbackClipDataset, file_sha256
from handloop.models.probabilistic_twm import ProbabilisticTWM, ProbabilisticTWMConfig
from handloop.models.visual_feedback import (VisualFeedbackConfig, VisualFeedbackSystem,
                                             feedback_current_nll)
from handloop.tracking import WandbTracker

INPUT_KEYS = ('timestamps', 'frame_valid', 'availability', 'image_valid', 'handedness',
              'world_to_camera', 'camera_intrinsics', 'crop_transform', 'beta', 'reset')


def utc():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def module_hash(module):
    digest = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def build_system(checkpoint_path, adapter_config, device, *, train_current_readout=False):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model_config = checkpoint['config']['model']
    twm = ProbabilisticTWM(ProbabilisticTWMConfig(**model_config))
    twm.load_state_dict(checkpoint['model'], strict=True)
    system = VisualFeedbackSystem(twm, VisualFeedbackConfig(**adapter_config),
                                  train_current_readout=train_current_readout).to(device)
    provenance = {'path': str(Path(checkpoint_path).resolve()),
                  'sha256': file_sha256(checkpoint_path), 'step': checkpoint.get('step'),
                  'code_version': checkpoint.get('code_version'), 'model_config': model_config}
    return system, provenance


def load_feedback_system(checkpoint_path, device):
    """Self-contained feedback checkpoint; the old motion file is not required."""
    saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    twm = ProbabilisticTWM(ProbabilisticTWMConfig(**saved['twm_source']['model_config']))
    system = VisualFeedbackSystem(twm, VisualFeedbackConfig(**saved['config']['adapter']),
        train_current_readout=saved['config'].get('train_current_readout', False)).to(device)
    system.load_state_dict(saved['model'], strict=True)
    return system.eval(), saved


def move_batch(batch, device):
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}


def forward_batch(system, batch, *, use_adapter=True, adapter_history=True, return_forecast=False):
    # Deliberate allowlist: target_world/target_valid are NEVER forward kwargs.
    return system.forward_clip(batch['patch_features'], batch['raw_world'],
        **{key: batch[key] for key in INPUT_KEYS if key in batch},
        use_adapter=use_adapter, adapter_history=adapter_history, return_forecast=return_forecast)


@torch.no_grad()
def metrics(output, batch):
    valid = batch['frame_valid'].bool() & batch['target_valid'].bool().all(-1)
    if not bool(valid.any()):
        raise ValueError('A clip batch has no complete valid current reference')
    target = torch.where(valid[..., None], batch['target_world'].float(),
                         g.identity_state(*valid.shape, device=valid.device))
    result = {'valid_states': int(valid.sum()), 'metric_counts': {}}
    def add(name, values, mask):
        count = int(mask.sum())
        result['metric_counts'][name] = count
        if count:
            result[name] = float(values[mask].mean())
    for name, state in (('raw', batch['raw_world']), ('candidate', output['candidate_world']),
                        ('final', output['state_world'])):
        groups = (batch['availability'].bool() if name == 'raw' else
                  torch.ones_like(batch['availability'], dtype=torch.bool))
        # Missing raw values may be NaN; they are not observations or metrics.
        position = torch.where(groups[..., :1], state[..., :3], 0.0)
        rotation = torch.where(groups[..., 1:, None], state[..., 3:].reshape(*valid.shape, 16, 6),
                               g.identity_rotation_6d(device=state.device))
        state = torch.cat((position, rotation.flatten(-2)), -1)
        angles = g.geodesic_angle(g.state_to_rotmat(state), g.state_to_rotmat(target)) * (180 / math.pi)
        add(name + '/wrist_mm', (state[..., :3] - target[..., :3]).norm(dim=-1) * 1000,
            valid & groups[..., 0])
        add(name + '/root_deg', angles[..., 0], valid & groups[..., 1])
        add(name + '/finger_deg', angles[..., 1:], valid[..., None] & groups[..., 2:])
    xyz_error = torch.linalg.vector_norm(output['state_world'][..., :3] - target[..., :3], dim=-1)
    raw_error = torch.linalg.vector_norm(batch['raw_world'][..., :3] - target[..., :3], dim=-1)
    add('wrist_degradation_over_1mm', ((xyz_error - raw_error) > .001).float(),
        valid & batch['availability'][..., 0].bool())
    # Evaluate the same anchor-space distribution that defines the objective.
    from handloop.models.visual_feedback import transform_state, inverse_rigid
    target_anchor = transform_state(target, inverse_rigid(output['anchor_world']))
    standardized = (target_anchor[..., :3] - output['state'][..., :3]) / output['state_translation_std']
    add('xyz_90pct_ellipsoid_coverage',
        (standardized.square().sum(-1) <= 6.251388631170325).float(), valid)
    add('xyz_std_mm', output['state_translation_std'].mean(-1) * 1000, valid)
    return result


@torch.no_grad()
def evaluate(system, loader, device, max_batches, *, use_adapter=True, adapter_history=True):
    system.eval()
    totals, metric_counts, count = {}, {}, 0
    for index, batch in enumerate(loader):
        if index >= max_batches:
            break
        batch = move_batch(batch, device)
        output = forward_batch(system, batch, use_adapter=use_adapter, adapter_history=adapter_history)
        value = metrics(output, batch)
        n = value.pop('valid_states')
        counts = value.pop('metric_counts')
        value['current_nll'] = float(feedback_current_nll(output, batch['target_world'], batch['target_valid']))
        counts['current_nll'] = n
        for key, eligible in counts.items():
            metric_counts[key] = metric_counts.get(key, 0) + eligible
        for key, number in value.items():
            totals[key] = totals.get(key, 0.) + number * counts[key]
        count += n
    if not count:
        raise ValueError('Validation is empty')
    return {**{key: number / metric_counts[key] for key, number in totals.items()},
            'valid_states': count, 'metric_counts': metric_counts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--twm-checkpoint', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('config/visual_feedback_v1.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--stop-after-updates', type=int,
                        help='Execution-only bound; does not change the resumable training config')
    parser.add_argument('--tracking-mode', choices=('online', 'offline', 'disabled'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--code-version', default='unrecorded')
    parser.add_argument('--stop-before-utc')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.steps is not None:
        config['steps'] = args.steps
    if args.tracking_mode:
        config['tracking']['enabled'] = args.tracking_mode != 'disabled'
        config['tracking']['mode'] = args.tracking_mode
        config['tracking']['require_online'] = args.tracking_mode == 'online'
    if config['steps'] < 1 or config['batch_size'] < 1:
        raise ValueError('Positive steps and batch size required')
    deadline = datetime.fromisoformat(args.stop_before_utc) if args.stop_before_utc else None
    if deadline is not None and deadline.utcoffset() is None:
        raise ValueError('Execution deadline requires an explicit timezone')
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'run_metadata.json').exists() and not args.resume:
        raise ValueError('Use a fresh output directory or explicit resume')
    if (args.output / 'STOP').exists():
        raise ValueError('Archive the STOP request explicitly before resuming')
    torch.set_num_threads(config.get('cpu_threads', 1))
    seed = config['seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device); torch.cuda.manual_seed(seed)
    train_set = VisualFeedbackClipDataset(args.manifest, 'train', clip_length=config['clip_length'])
    val_set = VisualFeedbackClipDataset(args.manifest, 'val', clip_length=config['clip_length'])
    generator = torch.Generator().manual_seed(seed)
    options = {'batch_size': config['batch_size'], 'num_workers': config.get('num_workers', 0),
               'pin_memory': device.type == 'cuda'}
    train_loader = DataLoader(train_set, shuffle=True, generator=generator, **options)
    val_loader = DataLoader(val_set, shuffle=False, **options)
    system, source = build_system(args.twm_checkpoint, config['adapter'], device,
                                  train_current_readout=config.get('train_current_readout', False))
    parameters = [p for p in system.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config['learning_rate'], weight_decay=config['weight_decay'])
    frozen_before = module_hash(system.twm)
    start_step, best = 0, float('inf')
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        if checkpoint['config'] != config or checkpoint['cache_sha256'] != train_set.manifest_sha256:
            raise ValueError('Resume must retain config and immutable cache manifest')
        if checkpoint['twm_source']['sha256'] != source['sha256']:
            raise ValueError('Resume TWM source changed')
        system.load_state_dict(checkpoint['model'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_step, best = checkpoint['step'], checkpoint['best_validation_nll']
        torch.set_rng_state(checkpoint['rng']['torch'])
        generator.set_state(checkpoint['rng']['loader'])
        random.setstate(checkpoint['rng']['python']); np.random.set_state(checkpoint['rng']['numpy'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state(checkpoint['rng']['cuda'], device=device)
    # No loader-cursor promise: resume starts a fresh sampled epoch.
    metadata = {'schema': 'handloop.visual_feedback_run.v1', 'started_utc': utc(),
                'code_version': args.code_version, 'config': config, 'twm_source': source,
                'cache_manifest': str(args.manifest.resolve()), 'cache_sha256': train_set.manifest_sha256,
                'cache_provenance': {k:v for k,v in train_set.manifest.items() if k != 'records'},
                'train_recordings': [x['recording'] for x in train_set.records],
                'validation_recordings': [x['recording'] for x in val_set.records],
                'trainable_parameters': sum(p.numel() for p in parameters),
                'twm_parameters': sum(p.numel() for p in system.twm.parameters()),
                'objective': 'mean final-current Gaussian x SO(3)^16 matrix-Fisher NLL; no auxiliary losses',
                'history': 'own completed final states; detached between frames; reset at sampled clip start',
                'resume_policy': 'restore parameters/optimizer/RNG; no exact loader-cursor replay',
                'scope': 'real cached image integration; small smoke is not benchmark accuracy or physical validation'}
    tracker = WandbTracker(config.get('tracking'), args.output, metadata, resume=bool(args.resume))
    metadata['tracking'] = tracker.metadata
    atomic_json(args.output / 'run_metadata.json', metadata)
    stop = {'requested': False}
    def request_stop(*unused):
        stop['requested'] = True
    signal.signal(signal.SIGTERM, request_stop); signal.signal(signal.SIGINT, request_stop)
    def save(step, *, best_checkpoint=False):
        payload = {'step': step, 'model': system.state_dict(), 'optimizer': optimizer.state_dict(),
                   'config': config, 'twm_source': source, 'cache_sha256': train_set.manifest_sha256,
                   'best_validation_nll': best, 'code_version': args.code_version,
                   'rng': {'torch': torch.get_rng_state(), 'loader': generator.get_state(),
                           'python': random.getstate(), 'numpy': np.random.get_state(),
                           'cuda': torch.cuda.get_rng_state(device) if device.type == 'cuda' else None}}
        destination = args.output / ('best.pt' if best_checkpoint else 'latest.pt')
        temporary = destination.with_suffix('.pt.tmp')
        torch.save(payload, temporary); temporary.replace(destination)
    started, step = time.monotonic(), start_step
    exit_code = 1
    try:
        baseline_path = args.output / 'baseline_validation.json'
        if args.resume:
            if not baseline_path.exists():
                raise ValueError('Resume needs the original saved baseline_validation.json')
            baseline = json.loads(baseline_path.read_text())
        else:
            baseline = evaluate(system, val_loader, device, config['eval_batches'], use_adapter=False)
        tracker.log({'baseline_validation': baseline}, step)
        if not args.resume:
            atomic_json(baseline_path, baseline)
        iterator = iter(train_loader)
        while step < config['steps']:
            if args.stop_after_updates is not None and step - start_step >= args.stop_after_updates:
                break
            if stop['requested'] or (args.output / 'STOP').exists() or (deadline and datetime.now(timezone.utc) >= deadline):
                break
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader); batch = next(iterator)
            batch = move_batch(batch, device)
            system.train(); optimizer.zero_grad(set_to_none=True)
            output = forward_batch(system, batch)
            loss = feedback_current_nll(output, batch['target_world'], batch['target_valid'])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Nonfinite final-current NLL')
            loss.backward()
            if not config.get('train_current_readout', False) and any(p.grad is not None for p in system.twm.parameters()):
                raise RuntimeError('Frozen TWM unexpectedly accumulated parameter gradients')
            grad = torch.nn.utils.clip_grad_norm_(parameters, config['grad_clip'], error_if_nonfinite=True)
            if not float(grad) > 0:
                raise RuntimeError('Feedback plugin has no usable final-output gradient')
            optimizer.step(); step += 1
            training = {'current_nll': float(loss.detach()), 'grad_norm': float(grad),
                        **metrics(output, batch), 'elapsed_seconds': time.monotonic()-started}
            validation = None
            if step % config['eval_every'] == 0 or step == config['steps']:
                validation = evaluate(system, val_loader, device, config['eval_batches'])
                if validation['current_nll'] < best:
                    best = validation['current_nll']; save(step, best_checkpoint=True)
                tracker.log({'validation': validation}, step)
            if step % config['log_every'] == 0 or step == start_step + 1:
                tracker.log({'train': training}, step)
                print(json.dumps({'utc': utc(), 'step': step, 'train': training, 'validation': validation}), flush=True)
            if step % config['save_every'] == 0:
                save(step)
            atomic_json(args.output/'status.json', {'utc': utc(), 'phase': 'training', 'step': step,
                        'training': training, 'validation': validation, 'wandb_url': tracker.url})
        save(step)
        frozen_after = module_hash(system.twm)
        if not config.get('train_current_readout', False) and frozen_before != frozen_after:
            raise RuntimeError('Frozen TWM weights changed')
        final = {'normal': evaluate(system, val_loader, device, config['eval_batches']),
                 'adapter_without_history': evaluate(system, val_loader, device, config['eval_batches'], adapter_history=False),
                 'adapter_bypass': baseline}
        atomic_json(args.output/'final_validation.json', final)
        phase = 'complete' if step == config['steps'] else 'stopped'
        atomic_json(args.output/'status.json', {'utc': utc(), 'phase': phase, 'step': step,
                    'validation': final, 'twm_hash_unchanged': frozen_before == frozen_after,
                    'wandb_url': tracker.url})
        tracker.log({'final_validation': final}, step)
        exit_code = 0
    except Exception as error:
        atomic_json(args.output/'status.json', {'utc': utc(), 'phase': 'failed', 'step': step,
                    'error_type': type(error).__name__, 'wandb_url': tracker.url})
        raise
    finally:
        tracker.finish(exit_code)


if __name__ == '__main__':
    main()
