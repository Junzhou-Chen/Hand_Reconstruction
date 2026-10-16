#!/usr/bin/env python3
"""Remote-only online RGB joint adaptation of HaWoR, adapter and motion TWM.

Visual updates retain final-current NLL and optionally add absolute raw wrist
and final landmark supervision. Separate motion replay updates retain the
source checkpoint's original current/future training objective.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import time
import traceback

import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset

from handloop.data.visual_clips import file_sha256
from handloop.data.visual_rgb import VisualFeedbackRGBDataset, reference_geometry_metadata
from handloop.data.forecast_trajectory import ForecastTrajectoryWindows
from handloop.data.trajectory import SourceBalancedSampler
from handloop.models.probabilistic_twm import ProbabilisticTWM, ProbabilisticTWMConfig
from handloop.models.visual_feedback import VisualFeedbackConfig, VisualFeedbackSystem, feedback_current_nll
from handloop.visual.hawor import FrozenHaWoRBridge
from handloop.visual.pipeline import HandLoopFeedbackPipeline
from handloop.visual.losses import (
    VisualGeometryLossConfig, visual_loss_terms, reduce_visual_loss_terms,
)
from handloop.tracking import WandbTracker
from train_probabilistic_twm import prepare_batch, objective, state_geometry_loss, state_errors
from train_visual_feedback import metrics


def utc():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    tmp.replace(path)


def move(batch, device):
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def resume_config_changes(previous, current, *, allow_rgb_batch_size_change=False):
    changes = {key: {'previous': previous.get(key), 'current': current.get(key)}
               for key in previous.keys() | current.keys()
               if key not in previous or key not in current or previous[key] != current[key]}
    allowed = {'per_gpu_rgb_batch_size'} if allow_rgb_batch_size_change else set()
    if changes.keys() - allowed:
        raise ValueError('Resume config differs outside explicitly allowed RGB batch size')
    if 'per_gpu_rgb_batch_size' in changes:
        for value in changes['per_gpu_rgb_batch_size'].values():
            if type(value) is not int or value < 1:
                raise ValueError('RGB batch size must be a positive integer')
    return changes


def tensor_hash(tensor):
    value = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
    return hashlib.sha256(value.tobytes()).hexdigest()


def model_hash(module):
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update((name + str(tensor.dtype) + str(tuple(tensor.shape))).encode())
        digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def gradient_groups(pipeline):
    return {
        'hawor_vit': pipeline.frontend.model.backbone,
        'hawor_iam': pipeline.frontend.model.st_module,
        'hawor_head': pipeline.frontend.model.mano_head,
        'hawor_pam': pipeline.frontend.model.motion_module,
        'adapter': pipeline.feedback.adapter,
        'twm_encoder': pipeline.feedback.twm.encoder,
        'twm_current': pipeline.feedback.twm.translation_head,
        'twm_rotation': pipeline.feedback.twm.rotation_head,
        'twm_future': pipeline.feedback.twm.future_decoder,
        'twm_future_velocity': pipeline.feedback.twm.future_translation_velocity_head,
        'twm_future_scale': pipeline.feedback.twm.future_translation_scale_head,
        'twm_future_rotation': pipeline.feedback.twm.future_rotation_velocity_head,
        'twm_future_concentration': pipeline.feedback.twm.future_rotation_concentration_head,
        'twm_mixture': pipeline.feedback.twm.mixture_head,
    }


def gradient_stats(pipeline):
    result = {}
    for name, module in gradient_groups(pipeline).items():
        if module is None:
            continue
        values = [p.grad.detach().float() for p in module.parameters() if p.requires_grad and p.grad is not None]
        result[name] = {
            'tensors': len(values),
            'norm': float(torch.stack([v.square().sum() for v in values]).sum().sqrt()) if values else 0.,
            'finite': bool(all(torch.isfinite(v).all() for v in values)),
        }
    return result


class JointTaskModel(nn.Module):
    def __init__(self, pipeline, replay_config, visual_loss_config=None):
        super().__init__()
        self.pipeline = pipeline
        self.replay_config = replay_config
        self.visual_loss_config = (visual_loss_config if isinstance(visual_loss_config, VisualGeometryLossConfig)
                                   else VisualGeometryLossConfig.from_dict(visual_loss_config))

    def forward(self, batch, *, task, corruption_generator=None):
        # Labels are consumed by objectives only. Pipeline and numeric TWM
        # receive explicit input allowlists, never the future/target tensors.
        if task == 'visual':
            output = self.pipeline(batch, return_forecast=False)
            valid = output['frame_valid'].bool() & batch['target_valid'].bool().all(-1)
            terms = visual_loss_terms(output, batch, self.pipeline.frontend.codec, self.visual_loss_config)
            loss, objective_metrics = reduce_visual_loss_terms(terms, self.visual_loss_config, distributed=True)
            # A rank may have only partial labels. It still participates in
            # global objective/count collectives without inventing rotations.
            measured = (metrics(output, {**batch, 'raw_world': output['raw_world'],
                                          'availability': output['availability']})
                        if bool(valid.any()) else {'valid_states': 0})
            measured.update(objective_metrics)
            if bool(valid.any()):
                measured['local_current_nll'] = float((terms['current_nll'][0] / terms['current_nll'][1]).detach())
                measured['geometry_score'] = float(state_geometry_loss(
                    output['state_world'][valid], batch['target_world'][valid]).mean().detach())
                measured['beta_norm'] = float(output['beta'].detach().norm(dim=-1)[valid].mean()) if 'beta' in output else 0.
            measured.pop('metric_counts', None)
            return {'loss': loss, 'measurements': measured}
        if task != 'motion':
            raise ValueError('Unknown task')
        history, raw, candidate, kwargs, target, future = prepare_batch(
            batch, corruption_generator, 'mixed',
            condition_on_beta=self.replay_config.get('condition_on_beta', False),
            corruption_profile=self.replay_config.get('corruption_profile'))
        output = self.pipeline.feedback.twm(history, raw, candidate, **kwargs)
        loss, values = objective(output, target, future, batch['future_valid'],
                                 batch['future_dt'], self.replay_config)
        measured = {k: float(v.detach()) for k, v in values.items()}
        measured.update({'current/' + k: float(v.mean()) for k, v in state_errors(output['state'], target).items()})
        return {'loss': loss, 'measurements': measured}


def optimizer_groups(pipeline, config):
    groups = []
    used = set()
    for label, module, lr in (
        ('hawor', pipeline.frontend.model, config['hawor_learning_rate']),
        ('adapter', pipeline.feedback.adapter, config['adapter_learning_rate']),
        ('twm', pipeline.feedback.twm, config['twm_learning_rate']),
    ):
        for decay in (True, False):
            params = []
            for name, parameter in module.named_parameters():
                if not parameter.requires_grad:
                    continue
                apply_decay = (parameter.ndim >= 2 and not getattr(parameter, '_no_weight_decay', False)
                               and 'pos_embed' not in name)
                if apply_decay == decay:
                    if id(parameter) in used:
                        raise ValueError('Optimizer parameter appears twice')
                    used.add(id(parameter)); params.append(parameter)
            if params:
                groups.append({'params': params, 'lr': lr, 'base_lr': lr,
                               'weight_decay': config['weight_decay'] if decay else 0.,
                               'name': label + ('_decay' if decay else '_no_decay')})
    expected = {id(p) for p in pipeline.parameters() if p.requires_grad}
    if expected != used:
        raise RuntimeError('Some trainable parameters are outside optimizer groups')
    return groups


@torch.no_grad()
def evaluate_rgb(pipeline, dataset, device, batches, *, use_adapter=True, visual_loss_config=None):
    pipeline.eval()
    n = min(len(dataset), batches)
    indices = sorted(set(round(i * (len(dataset)-1) / max(1, n-1)) for i in range(n)))
    loader = DataLoader(Subset(dataset, indices), batch_size=1, num_workers=0)
    totals, counts = {}, {}
    loss_config = (visual_loss_config if isinstance(visual_loss_config, VisualGeometryLossConfig)
                   else VisualGeometryLossConfig.from_dict(visual_loss_config))
    for batch in loader:
        batch = move(batch, device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            output = pipeline(batch, return_forecast=False, use_adapter=use_adapter)
            valid = output['frame_valid'].bool() & batch['target_valid'].bool().all(-1)
            amount = int(valid.sum())
            value = (metrics(output, {**batch, 'raw_world': output['raw_world'], 'availability': output['availability']})
                     if amount else {'valid_states': 0, 'metric_counts': {}})
            terms = visual_loss_terms(output, batch, pipeline.frontend.codec, loss_config)
            _, objective_metrics = reduce_visual_loss_terms(terms, loss_config, distributed=False)
            for name in terms:
                if objective_metrics[name + '/count']:
                    value[name] = objective_metrics[name]
            if amount:
                value['geometry_score'] = float(state_geometry_loss(output['state_world'][valid], batch['target_world'][valid]).mean())
        local_counts = value.pop('metric_counts')
        value.pop('valid_states')
        local_counts.update({name: objective_metrics[name + '/count'] for name in terms})
        if amount:
            local_counts['geometry_score'] = amount
        for key, number in value.items():
            if not math.isfinite(number):
                raise FloatingPointError('Nonfinite validation metric: ' + key)
            count = local_counts[key]
            totals[key] = totals.get(key, 0.) + number * count
            counts[key] = counts.get(key, 0) + count
    return {**{k: totals[k]/counts[k] for k in totals if counts[k]},
            'valid_states': counts.get('current_nll', 0), 'clips': len(indices), 'metric_counts': counts}


@torch.no_grad()
def evaluate_motion(twm, dataset, device, config, samples=16):
    """Small fixed pure-motion retention set; not visual future evaluation."""
    twm.eval()
    result = {}
    for source, indices in dataset.fixed_indices_by_dataset(samples).items():
        loader = DataLoader(Subset(dataset, indices), batch_size=samples, num_workers=0)
        result[source] = {}
        for mode in ('noisy', 'clean'):
            generator = torch.Generator(device=device).manual_seed(config['seed'] + 123)
            total, counts = {}, {}
            def add(key, values, valid=None):
                selected = values.float().flatten() if valid is None else values.float()[valid]
                if selected.numel():
                    total[key] = total.get(key, 0.) + float(selected.sum())
                    counts[key] = counts.get(key, 0) + selected.numel()
            for batch in loader:
                batch = move(batch, device)
                h, raw, candidate, kwargs, target, future = prepare_batch(batch, generator, mode,
                    condition_on_beta=config.get('condition_on_beta', False), corruption_profile=config.get('corruption_profile'))
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    output = twm(h, raw, candidate, **kwargs)
                for key, values in state_errors(output['state'], target).items(): add('current/' + key, values)
                future_errors = state_errors(output['forecast'], future)
                hold_errors = state_errors(output['state'][:, None], future)
                for key, values in future_errors.items(): add('future_ade/' + key, values, batch['future_valid'])
                for key, values in hold_errors.items(): add('future_hold/' + key, values, batch['future_valid'])
                rows = torch.arange(len(target), device=device)
                for seconds in (.1, .5, 1., 2.):
                    difference = (batch['future_dt']-seconds).abs().masked_fill(~batch['future_valid'], float('inf'))
                    distance, slot = difference.min(-1)
                    eligible = distance <= batch['dt'][:, -1]*.6
                    for label, errors in (('map', future_errors), ('hold', hold_errors)):
                        for key, values in errors.items():
                            add(f'future_{seconds:g}s/{label}/{key}', values[rows, slot], eligible)
            result[source][mode] = {**{k: total[k]/counts[k] for k in total}, 'metric_counts': counts}
    return result


def build_pipeline(args, config, device, checkpoint):
    frontend = FrozenHaWoRBridge.from_pretrained(
        repo_root=args.hawor_root, checkpoint_path=args.hawor_checkpoint,
        config_path=args.hawor_config, mano_model_dir=args.mano_models, device=device,
        trainable=True, gradient_checkpointing=config.get('gradient_checkpointing', True))
    twm_config = checkpoint['config']['model']
    twm = ProbabilisticTWM(ProbabilisticTWMConfig(**twm_config))
    if 'model' in checkpoint:
        twm.load_state_dict(checkpoint['model'], strict=True)
    feedback = VisualFeedbackSystem(twm, VisualFeedbackConfig(**config['adapter']), trainable_scope='full')
    return HandLoopFeedbackPipeline(frontend, feedback).to(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-manifest', 'motion-manifest', 'twm-checkpoint', 'hawor-root', 'hawor-checkpoint',
                 'hawor-config', 'mano-models', 'config', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--allow-rgb-batch-size-change', action='store_true')
    parser.add_argument('--code-version', default='unrecorded')
    parser.add_argument('--stop-after-updates', type=int)
    parser.add_argument('--tracking-mode', choices=('online', 'offline', 'disabled'))
    parser.add_argument('--stop-before-utc')
    args = parser.parse_args()
    if args.allow_rgb_batch_size_change and not args.resume:
        parser.error('--allow-rgb-batch-size-change requires --resume')
    config = json.loads(args.config.read_text())
    visual_loss_config = VisualGeometryLossConfig.from_dict(config.get('visual_geometry_supervision'))
    if args.tracking_mode:
        config['tracking'].update(enabled=args.tracking_mode != 'disabled', mode=args.tracking_mode,
                                  require_online=args.tracking_mode == 'online')
    world, rank, local = (int(os.environ.get(k, d)) for k, d in
                          (('WORLD_SIZE', '1'), ('RANK', '0'), ('LOCAL_RANK', '0')))
    if config['world_size'] != world:
        raise ValueError('Explicit launch world size differs from config')
    if config['clip_length'] % 16 or config['clip_length'] < 16:
        raise ValueError('RGB clip must contain complete official 16-frame blocks')
    device = torch.device('cuda', local); torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group('nccl', timeout=timedelta(hours=2), device_id=device)
    torch.set_num_threads(config.get('cpu_threads', 2))
    torch.backends.cuda.matmul.allow_tf32 = True
    seed = config['seed'] + rank
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        if (args.output/'run_metadata.json').exists() and not args.resume:
            raise ValueError('Existing run requires explicit resume')
        if (args.output/'STOP').exists():
            raise ValueError('Archive the STOP request explicitly before resume')
    if world > 1:
        dist.barrier()
    deadline = datetime.fromisoformat(args.stop_before_utc or config['stop_before_utc'])
    if deadline.utcoffset() is None:
        raise ValueError('Deadline must contain timezone')
    resumed = torch.load(args.resume, map_location='cpu', weights_only=False, mmap=True) if args.resume else None
    if resumed:
        source, replay_config = resumed['twm_source'], resumed['replay_config']
        # Build the architecture from embedded config. Trained weights below
        # replace every numeric tensor; no old D37 reference file is opened.
        source_checkpoint = {'config': {'model': source['model_config']}}
    else:
        source_checkpoint = torch.load(args.twm_checkpoint, map_location='cpu', weights_only=False)
        source = {'path': str(args.twm_checkpoint), 'sha256': file_sha256(args.twm_checkpoint),
                  'step': source_checkpoint['step'], 'code_version': source_checkpoint['code_version'],
                  'model_config': source_checkpoint['config']['model']}
        replay_config = source_checkpoint['config'].copy()
    if not replay_config.get('corruption_profile'):
        profile_path = Path(replay_config['corruption_profile_path'])
        if not profile_path.is_absolute(): profile_path = Path(__file__).resolve().parents[1]/profile_path
        replay_config['corruption_profile'] = json.loads(profile_path.read_text())
    expected_motion_hash = (source_checkpoint.get('run') or {}).get('manifest_sha256')
    if expected_motion_hash and file_sha256(args.motion_manifest) != expected_motion_hash:
        raise ValueError('Motion replay manifest differs from source pretraining')
    pipeline = build_pipeline(args, config, device, source_checkpoint)
    del source_checkpoint
    train = VisualFeedbackRGBDataset(args.source_manifest, 'train', clip_length=config['clip_length'],
                                     hawor_root=args.hawor_root)
    val = VisualFeedbackRGBDataset(args.source_manifest, 'val', clip_length=config['clip_length'],
                                   hawor_root=args.hawor_root) if rank == 0 else None
    sampler = DistributedSampler(train, num_replicas=world, rank=rank, shuffle=True, seed=config['seed'])
    loader = DataLoader(train, batch_size=config['per_gpu_rgb_batch_size'], sampler=sampler,
                        num_workers=config.get('num_workers', 0), pin_memory=True,
                        generator=torch.Generator().manual_seed(seed))
    motion = ForecastTrajectoryWindows(args.motion_manifest, 'train', history=replay_config['history_frames'],
        forecast_frames=replay_config['model']['forecast_horizon'], min_future_frames=replay_config['min_future_frames'],
        window_stride=replay_config['window_stride'], cache_sequences=128, cache_bytes=536870912)
    if set(motion.dataset_names) != set(replay_config['required_datasets']):
        raise ValueError('Motion replay source set differs from pretraining')
    motion_val = ForecastTrajectoryWindows(args.motion_manifest, 'val', history=replay_config['history_frames'],
        forecast_frames=replay_config['model']['forecast_horizon'], min_future_frames=replay_config['min_future_frames'],
        window_stride=replay_config['window_stride'], cache_sequences=32, cache_bytes=268435456) if rank == 0 else None
    motion_sampler = SourceBalancedSampler(motion, num_samples=config['steps']*config['per_gpu_motion_batch_size'], seed=seed)
    motion_loader = DataLoader(motion, batch_size=config['per_gpu_motion_batch_size'], sampler=motion_sampler,
                               num_workers=0, pin_memory=True)
    model = JointTaskModel(pipeline, replay_config, visual_loss_config)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(optimizer_groups(pipeline, config), foreach=False)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step:
        min(1., (step+1)/max(1, config['warmup_steps'])) *
        (.1 + .9*.5*(1+math.cos(math.pi*min(step, config['steps'])/config['steps']))))
    identity = {'source_manifest_sha256': file_sha256(args.source_manifest),
                'motion_manifest_sha256': file_sha256(args.motion_manifest),
                'twm_source_sha256': source['sha256'], 'hawor_checkpoint_sha256': file_sha256(args.hawor_checkpoint),
                'hawor_config_sha256': file_sha256(args.hawor_config),
                'mano_asset_sha256': {side: file_sha256(args.mano_models/f'MANO_{side}.pkl') for side in ('LEFT', 'RIGHT')}}
    start, best = 0, float('inf')
    generator = torch.Generator(device=device).manual_seed(seed + 901)
    resume_changes = {}
    if args.resume:
        resume_changes = resume_config_changes(resumed['config'], config,
            allow_rgb_batch_size_change=args.allow_rgb_batch_size_change)
        if resumed['identity'] != identity or resumed['world_size'] != world:
            raise ValueError('Resume requires identical inputs, sources, assets and world size')
        model.load_state_dict(resumed['model'], strict=True)
        optimizer.load_state_dict(resumed['optimizer']); scheduler.load_state_dict(resumed['scheduler'])
        start, best = resumed['step'], resumed['best_geometry_score']
        rng = resumed['rank_states'][rank]
        torch.set_rng_state(rng['torch']); torch.cuda.set_rng_state(rng['cuda'], device)
        generator.set_state(rng['corruption']); random.setstate(rng['python']); np.random.set_state(rng['numpy'])
        del resumed
    ddp = DDP(model, device_ids=[local], broadcast_buffers=False, find_unused_parameters=True,
              gradient_as_bucket_view=True) if world > 1 else model
    run = {'schema': 'handloop.visual_joint.v2', 'started_utc': utc(), 'code_version': args.code_version,
        'config': config, 'identity': identity, 'twm_source': source, 'world_size': world,
        'hawor_provenance': pipeline.frontend.provenance(),
        'trainable_parameters': sum(p.numel() for p in parameters),
        'optimizer_groups': [{k:v for k,v in group.items() if k != 'params'} |
                             {'parameters': sum(p.numel() for p in group['params'])} for group in optimizer.param_groups],
        'train_rgb_clips': len(train), 'validation_rgb_clips': len(val) if val is not None else None,
        'train_recordings': [r['recording'] for r in train.records],
        'validation_recordings': [r['recording'] for r in val.records] if val is not None else [],
        'motion_windows_by_dataset': motion.windows_by_dataset,
        'visual_objective': 'final-current NLL (default coefficient1) plus explicitly enabled absolute raw-J0/final-MANO21 Euclidean metre losses divided by configured physical scales; separate global masked means',
        'visual_geometry_supervision': asdict(visual_loss_config),
        'geometry_reference_policy': 'source joints and masks are loss-only; prediction uses own beta/HaWoR left codec; no GT beta, PA alignment or uncertainty reweighting',
        'visual_source_datasets': sorted({reference_geometry_metadata(train.manifest, record)['dataset'] for record in train.records}),
        'visual_reference_contracts': [dict(values) for values in sorted({tuple(sorted(reference_geometry_metadata(train.manifest, record).items()))
                                                                       for record in train.records})],
        'motion_objective': 'separate replay updates using unchanged source D37 six-term objective',
        'motion_sampling': 'equal source then uniform windows with replacement, independent rank seeds; not D37 full-coverage order',
        'best_selection': 'validation geometry score from existing D37 state_geometry_loss; NLL/coverage reported separately',
        'baseline_definition': 'initialized joint two-round system and fixed original HaWoR raw; separate initialized adapter-bypass baseline',
        'best_scope': 'best among trained checkpoints; initialized step0 metrics retained separately',
        'history': 'own completed final states; detached across physical frames; only final refinement committed',
        'resume_limit': 'weights/optimizer/schedule/rank RNG restored; RGB and motion loader cursors restart',
        'resume': {'checkpoint': str(args.resume) if args.resume else None,
                   'start_step': start, 'config_changes': resume_changes},
        'scope': train.manifest.get('input_protocol', 'source-declared RGB/calibration/reference protocol; not a detector/SLAM benchmark')}
    tracker = WandbTracker(config.get('tracking'), args.output, run, rank=rank, resume=bool(args.resume))
    if rank == 0:
        run['tracking'] = tracker.metadata
        execution_tag = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        if (args.output/'run_metadata.json').exists():
            atomic_json(args.output/'execution_metadata'/f'{execution_tag}_previous.json',
                        json.loads((args.output/'run_metadata.json').read_text()))
        atomic_json(args.output/'run_metadata.json', run)
        atomic_json(args.output/'execution_metadata'/f'{execution_tag}.json', run)
        tracker.log({'runtime': {'per_gpu_rgb_batch_size': config['per_gpu_rgb_batch_size'],
                                'per_gpu_motion_batch_size': config['per_gpu_motion_batch_size'],
                                'resume_start_step': start}}, start)
    watched = {name: next((p for p in module.parameters() if p.requires_grad), None)
               for name,module in gradient_groups(pipeline).items() if module is not None}
    before = {name: tensor_hash(p) for name,p in watched.items() if p is not None}
    grad_seen = {name: False for name in watched}
    stop = {'requested': False}
    def request_stop(*unused): stop['requested'] = True
    signal.signal(signal.SIGTERM, request_stop); signal.signal(signal.SIGINT, request_stop)
    last_saved = {'step': None, 'path': None}

    def save(step, is_best=False):
        destination = args.output/('best.pt' if is_best else 'latest.pt')
        if last_saved['step'] == step:
            # Evaluation/break can request the same ~8GB state several times.
            # Share the immutable inode until a later atomic replacement.
            if rank == 0 and destination != last_saved['path']:
                same_inode = destination.exists() and os.path.samefile(destination, last_saved['path'])
                if not same_inode:
                    tmp = destination.with_suffix('.pt.tmp')
                    if tmp.exists(): tmp.unlink()
                    os.link(last_saved['path'], tmp); tmp.replace(destination)
            if world > 1: dist.barrier()
            last_saved.update(step=step, path=destination)
            return
        digest = model_hash(model)
        state = {'rank': rank, 'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state(device),
                 'corruption': generator.get_state(), 'python': random.getstate(), 'numpy': np.random.get_state(),
                 'model_sha256': digest}
        states = [state]
        if world > 1:
            states = [None]*world; dist.all_gather_object(states, state)
        if len({x['model_sha256'] for x in states}) != 1:
            raise RuntimeError('DDP model replicas differ')
        if rank == 0:
            # Includes learned HaWoR + TWM + adapter: no dependence on an old
            # motion checkpoint for restoring their trained state tensors.
            payload = {'step': step, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(), 'config': config, 'identity': identity,
                'twm_source': source, 'replay_config': replay_config, 'world_size': world,
                'resume_transition': run['resume'],
                'best_geometry_score': best, 'rank_states': states, 'code_version': args.code_version,
                'frontend_assets': {'hawor_root': str(args.hawor_root), 'hawor_config': str(args.hawor_config),
                                    'mano_models': str(args.mano_models)}}
            path = destination; tmp = path.with_suffix('.pt.tmp')
            torch.save(payload, tmp); tmp.replace(path)
            atomic_json(args.output/'replica_checkpoint_audit.json',
                {'utc': utc(), 'step': step, 'byte_equal': True, 'model_sha256': digest})
        if world > 1: dist.barrier()
        last_saved.update(step=step, path=destination)

    epoch, step, started, exit_code = 0, start, time.monotonic(), 1
    sampler.set_epoch(epoch); rgb_iterator = iter(loader); motion_iterator = iter(motion_loader)
    try:
        if rank == 0 and not args.resume:
            baseline = evaluate_rgb(pipeline, val, device, config['eval_batches'], visual_loss_config=visual_loss_config)
            atomic_json(args.output/'baseline_validation.json', baseline)
            atomic_json(args.output/'baseline_adapter_bypass.json',
                        evaluate_rgb(pipeline, val, device, config['eval_batches'], use_adapter=False,
                                     visual_loss_config=visual_loss_config))
            retention = evaluate_motion(pipeline.feedback.twm, motion_val, device, replay_config,
                                       config.get('motion_eval_samples_per_source', 16))
            atomic_json(args.output/'baseline_motion_validation.json', retention)
            tracker.log({'baseline_validation': baseline}, step)
            tracker.log({'baseline_motion_validation': retention}, step)
        if world > 1: dist.barrier()
        while step < config['steps']:
            requested = (stop['requested'] or (args.output/'STOP').exists() or datetime.now(timezone.utc) >= deadline
                         or (args.stop_after_updates is not None and step-start >= args.stop_after_updates))
            flag = torch.tensor(int(requested), device=device)
            if world > 1: dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            if int(flag): break
            task = 'motion' if (step+1) % config['motion_replay_every'] == 0 else 'visual'
            if task == 'visual':
                try: batch = next(rgb_iterator)
                except StopIteration:
                    epoch += 1; sampler.set_epoch(epoch); rgb_iterator = iter(loader); batch = next(rgb_iterator)
            else:
                try: batch = next(motion_iterator)
                except StopIteration:
                    motion_iterator = iter(motion_loader); batch = next(motion_iterator)
            batch = move(batch, device)
            ddp.train(); optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                result = ddp(batch, task=task, corruption_generator=generator)
                loss = result['loss']
            if not bool(torch.isfinite(loss)): raise FloatingPointError('Nonfinite ' + task + ' loss')
            loss.backward()
            groups = gradient_stats(pipeline)
            for name, item in groups.items():
                if not item['finite']: raise FloatingPointError('Nonfinite gradient ' + name)
                grad_seen[name] |= item['norm'] > 0
            # Clip each pretrained/new module separately so a large visual
            # residual cannot suppress the small-LR motion update entirely.
            gradient_norms = {}
            for name, module in (('hawor', pipeline.frontend.model), ('adapter', pipeline.feedback.adapter),
                                 ('twm', pipeline.feedback.twm)):
                gradient_norms[name] = float(torch.nn.utils.clip_grad_norm_(
                    [p for p in module.parameters() if p.requires_grad], config['grad_clip'], error_if_nonfinite=True))
            optimizer.step(); scheduler.step(); step += 1
            changed = {name: tensor_hash(p) != before[name] for name,p in watched.items() if p is not None} if step-start <= 8 else None
            if step-start <= 8:
                atomic_json(args.output/f'gradient_audit_rank{rank}.json',
                    {'utc': utc(), 'step': step, 'execution_start_step': start, 'task': task, 'gradients': groups,
                     'per_gpu_rgb_batch_size': config['per_gpu_rgb_batch_size'],
                     'nonzero_gradient_seen': grad_seen, 'representative_parameters_changed': changed,
                     'peak_allocated_bytes': torch.cuda.max_memory_allocated(device)})
            training = {**result['measurements'], 'gradient_norms': gradient_norms,
                        'per_gpu_rgb_batch_size': config['per_gpu_rgb_batch_size'],
                        'elapsed_seconds': time.monotonic()-started,
                        'peak_allocated_gb': torch.cuda.max_memory_allocated(device)/2**30}
            if rank == 0 and (step % config['log_every'] == 0 or step-start <= 8):
                tracker.log({'train_' + task: training}, step)
                print(json.dumps({'utc': utc(), 'step': step, 'task': task, 'training': training}), flush=True)
            validation = None
            if step % config['eval_every'] == 0 or step == config['steps']:
                if world > 1: dist.barrier()
                if rank == 0:
                    validation = evaluate_rgb(pipeline, val, device, config['eval_batches'], visual_loss_config=visual_loss_config)
                    retention = evaluate_motion(pipeline.feedback.twm, motion_val, device, replay_config,
                                               config.get('motion_eval_samples_per_source', 16))
                    tracker.log({'validation': validation}, step)
                    tracker.log({'motion_validation': retention}, step)
                    atomic_json(args.output/'latest_motion_validation.json', {'step': step, 'values': retention})
                    improved = validation['geometry_score'] < best
                    if improved: best = validation['geometry_score']
                    decision = [improved, best]
                else: decision = [None, None]
                if world > 1: dist.broadcast_object_list(decision, src=0, device=device)
                improved, best = decision
                if improved: save(step, is_best=True)
            if step % config['save_every'] == 0 or step == start+config.get('initial_checkpoint_step', 4):
                save(step)
            if rank == 0:
                status = {'utc': utc(), 'phase': 'training', 'step': step, 'target_steps': config['steps'],
                          'task': task, 'training': training, 'wandb_url': tracker.url}
                if validation is not None: status['validation'] = validation
                atomic_json(args.output/'status.json', status)
            del result, loss, batch
        save(step)
        if rank == 0:
            final = evaluate_rgb(pipeline, val, device, config['eval_batches'], visual_loss_config=visual_loss_config)
            atomic_json(args.output/'final_validation.json', final)
            atomic_json(args.output/'status.json', {'utc': utc(), 'phase': 'complete' if step == config['steps'] else 'stopped',
                'step': step, 'target_steps': config['steps'], 'validation': final, 'wandb_url': tracker.url})
        if world > 1: dist.barrier()
        exit_code = 0
    except Exception as error:
        traceback.print_exc()
        if rank == 0:
            atomic_json(args.output/'status.json', {'utc': utc(), 'phase': 'failed', 'step': step,
                         'error_type': type(error).__name__, 'error': str(error), 'wandb_url': tracker.url})
        # A peer may be waiting at a baseline/evaluation barrier. Do not block
        # the exception behind a graceful NCCL destroy; torchrun terminates the
        # remaining workers when this worker exits with a nonzero status.
        if dist.is_initialized():
            tracker.finish(1)
            os._exit(1)
        raise
    finally:
        tracker.finish(exit_code)
        if dist.is_initialized(): dist.destroy_process_group()


if __name__ == '__main__':
    main()
