#!/usr/bin/env python3
"""Remote-only synchronous multi-GPU TWM training and audited batch migration.

Launch with torchrun. batch_size is GLOBAL; equal-size local mean losses and
DDP gradient averaging give the global mean objective. Validation retains a
separate batch size so changing training batch does not change its protocol.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset

from train_twm import (TWM, TWMConfig, TrajectoryWindows, SourceBalancedSampler,
                       atomic_json, evaluate_sources, sample_batch, state_loss, utc)


EXECUTION_KEYS = frozenset({'batch_size', 'eval_batch_size', 'world_size', 'cpu_threads'})


def verify_migration(checkpoint, config, manifest):
    """Only authorized execution/batch changes; no silent model/data changes."""
    previous = checkpoint['run']
    before = {k:v for k,v in checkpoint['config'].items() if k not in EXECUTION_KEYS}
    after = {k:v for k,v in config.items() if k not in EXECUTION_KEYS}
    if before != after:
        raise ValueError('Migration only permits batch/evaluation/device/thread execution changes')
    old_eval_batch = checkpoint['config'].get('eval_batch_size', checkpoint['config']['batch_size'])
    if config.get('eval_batch_size', config['batch_size']) != old_eval_batch:
        raise ValueError('Migration must retain original validation batch/corruption protocol')
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if previous['manifest'] != str(manifest) or previous['manifest_sha256'] != digest:
        raise ValueError('Migration requires the exact original collection and leaf-manifest protocol')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'config', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--code-version', required=True)
    initial = parser.add_mutually_exclusive_group()
    initial.add_argument('--migrate-from', type=Path)
    initial.add_argument('--resume', type=Path)
    parser.add_argument('--stop-after-updates', type=int, help='bounded smoke; preserves full scheduler config')
    args = parser.parse_args()
    if not Path('/sciclone/home/jchen57').is_dir():
        parser.error('Run inside the authorized remote compute Pod')
    config = json.loads(args.config.read_text())
    rank, local_rank, world = (int(os.environ[k]) for k in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE'))
    if world != config.get('world_size', world) or config['batch_size'] % world:
        raise ValueError('GLOBAL batch must divide evenly across configured world_size')
    if args.stop_after_updates is not None and args.stop_after_updates < 1:
        raise ValueError('stop-after-updates must be positive')
    local_batch = config['batch_size'] // world
    eval_batch = config.get('eval_batch_size', 64)
    torch.set_num_threads(config.get('cpu_threads', 4))
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    seed = config['seed'] + rank * 100003
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    generator = torch.Generator(device=device).manual_seed(seed)
    dist.init_process_group('nccl', timeout=timedelta(minutes=30), device_id=device)
    # NCCL is actually exercised before model/loader initialization.
    probe = torch.tensor(float(rank+1), device=device)
    dist.all_reduce(probe)
    if probe.item() != world*(world+1)/2:
        raise RuntimeError('NCCL all-reduce probe failed')
    stop = {'requested':False}
    def request_stop(signum, frame): stop['requested'] = True
    signal.signal(signal.SIGTERM, request_stop); signal.signal(signal.SIGINT, request_stop)
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        if (args.output/'run_metadata.json').exists() and not args.resume:
            raise ValueError('Fresh migration/smoke requires a new output directory')
        if (args.output/'STOP').exists():
            raise ValueError('Archive the previous STOP request before resume; ds launcher does this explicitly')
        atomic_json(args.output/'status.json', {'utc':utc(), 'phase':'initializing_ddp', 'world_size':world})
    dist.barrier()
    model = TWM(TWMConfig(**config['model'])).to(device)
    decay = [p for p in model.parameters() if not getattr(p, '_no_weight_decay', False)]
    no_decay = [p for p in model.parameters() if getattr(p, '_no_weight_decay', False)]
    optimizer = torch.optim.AdamW([
        {'params':decay, 'weight_decay':config['weight_decay']},
        {'params':no_decay, 'weight_decay':0.0}], lr=config['learning_rate'])
    start_step, best, checkpoint = 0, float('inf'), None
    source_checkpoint = args.resume or args.migrate_from
    if source_checkpoint:
        checkpoint = torch.load(source_checkpoint, map_location='cpu', weights_only=False)
        if args.migrate_from:
            verify_migration(checkpoint, config, args.manifest)
        else:
            verify_migration(checkpoint, config, args.manifest)
            if checkpoint['config'] != config or checkpoint['run']['world_size'] != world:
                raise ValueError('DDP resume requires exactly matching config and world size')
        model.load_state_dict(checkpoint['model'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_step, best = checkpoint['step'], checkpoint['best_validation_loss']
    if start_step >= config['steps']:
        raise ValueError('Source checkpoint already reached configured steps; do not silently retrain')
    # DDP initial synchronization makes replicas identical, including buffers.
    ddp = DDP(model, device_ids=[local_rank], output_device=local_rank,
              broadcast_buffers=False, gradient_as_bucket_view=True)
    data_options = {k:config[k] for k in ('cache_sequences', 'cache_bytes') if k in config}
    train = TrajectoryWindows(args.manifest, 'train', config['history_frames'],
                              config['forecast_offset_frames'], config['window_stride'], **data_options)
    required = set(config['required_datasets'])
    if set(train.dataset_names) != required or config['source_sampling'] != 'equal_source':
        raise ValueError('This DDP entry requires the declared sources and equal_source protocol')
    sampler = SourceBalancedSampler(train, num_samples=config['steps']*local_batch, seed=seed)
    loader = DataLoader(train, batch_size=local_batch, sampler=sampler, num_workers=0,
                        pin_memory=True, drop_last=True, generator=torch.Generator().manual_seed(seed))
    val_loaders = {}
    if rank == 0:
        val = TrajectoryWindows(args.manifest, 'val', config['history_frames'],
                                config['forecast_offset_frames'], config['window_stride'], **data_options)
        if set(val.dataset_names) != required:
            raise ValueError('Validation sources differ')
        val_loaders = {name:DataLoader(Subset(val, indices), batch_size=eval_batch,
                         num_workers=0, pin_memory=True)
                       for name,indices in val.fixed_indices_by_dataset(eval_batch*config['eval_batches']).items()}
    run = {'started_utc':utc(), 'code_version':args.code_version, 'pid':os.getpid(),
           'world_size':world, 'backend':'nccl', 'global_batch_size':config['batch_size'],
           'per_gpu_batch_size':local_batch, 'eval_batch_size':eval_batch,
           'config':config, 'manifest':str(args.manifest),
           'manifest_sha256':hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
           'parameters':sum(p.numel() for p in model.parameters()), 'torch':torch.__version__,
           'gpu':torch.cuda.get_device_name(device), 'dataset_names':train.dataset_names,
           'train_windows_by_dataset':train.windows_by_dataset, 'source_sampling':'equal_source',
           'sampling':'independent rank-seeded source/window draws WITH REPLACEMENT; overlaps are possible by design',
           'initialization':'resume' if args.resume else ('single-GPU batch/DDP migration' if args.migrate_from else 'from scratch'),
           'parent_checkpoint':str(source_checkpoint) if source_checkpoint else None,
           'parent_step':start_step, 'parent_code_version':checkpoint.get('code_version') if checkpoint else None,
           'parent_checkpoint_sha256':hashlib.sha256(source_checkpoint.read_bytes()).hexdigest() if source_checkpoint else None,
           'beta_conditioning':config.get('condition_on_beta', True),
           'scheduler':'original absolute optimizer-step schedule retained; no automatic LR scaling',
           'batch_change':'larger global batch increases sample exposure per remaining optimizer step',
           'scope':'synthetic-corruption pure-trajectory restoration; no RGB/real HaWoR or physical guarantee',
           'resume_limit':'per-rank RNG saved, but prefetched sampler chunk/loader cursor not exactly replayed'}
    if args.resume:
        state = checkpoint['rank_states'][rank]
        generator.set_state(state['corruption_rng'].cpu()); torch.set_rng_state(state['torch_rng'].cpu())
        torch.cuda.set_rng_state(state['cuda_rng'].cpu(), device)
        random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng'])
        sampler.load_state_dict(state['sampler'])
    elif args.migrate_from:
        # A topology/batch migration deliberately begins new per-rank sampling
        # and corruption streams; model, optimizer and absolute step survive.
        pass
    checkpoint = None
    if rank == 0:
        run.update(val_windows_by_dataset=val.windows_by_dataset,
                   val_samples_by_dataset={name:len(loader.dataset) for name,loader in val_loaders.items()})
        atomic_json(args.output/'run_metadata.json', run)
        print(json.dumps(run), flush=True)

    def save(step, best_checkpoint=False):
        digest = hashlib.sha256()
        for name, tensor in model.state_dict().items():
            digest.update((name+str(tensor.dtype)+str(tuple(tensor.shape))).encode())
            digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        state = {'rank':rank, 'corruption_rng':generator.get_state().cpu(),
                 'torch_rng':torch.get_rng_state(), 'cuda_rng':torch.cuda.get_rng_state(device).cpu(),
                 'python_rng':random.getstate(), 'numpy_rng':np.random.get_state(), 'sampler':sampler.state_dict(),
                 'model_state_sha256':digest.hexdigest()}
        rank_states = [None]*world
        dist.all_gather_object(rank_states, state)
        if len({state['model_state_sha256'] for state in rank_states}) != 1:
            raise RuntimeError('DDP model replicas differ byte-for-byte at checkpoint boundary')
        if rank == 0:
            atomic_json(args.output/'replica_checkpoint_audit.json', {'utc':utc(), 'step':step,
                        'world_size':world, 'all_model_replicas_byte_equal':True,
                        'model_state_sha256':digest.hexdigest()})
            value = {'step':step, 'model':model.state_dict(), 'optimizer':optimizer.state_dict(),
                     'config':config, 'code_version':args.code_version, 'best_validation_loss':best,
                     'rank_states':rank_states, 'run':run}
            path = args.output/('best.pt' if best_checkpoint else 'latest.pt')
            tmp = path.with_suffix('.pt.tmp'); torch.save(value, tmp); tmp.replace(path)
        dist.barrier()

    def validate(step, initial=False):
        nonlocal best
        decision = torch.zeros(1, device=device)
        values = None
        # Only unwrapped forward on rank0; no DDP collectives inside validation.
        if rank == 0:
            values = evaluate_sources(model, val_loaders, device, config)
            score = (values['noisy']['loss']+values['missing']['loss'])/2
            if score < best:
                best = score; decision.fill_(1)
            event = {'utc':utc(), 'step':step, 'validation':values, 'best_validation_loss':best}
            atomic_json(args.output/('initial_validation.json' if initial else 'latest_validation.json'), event)
            print(json.dumps(event), flush=True)
            if not initial:
                with (args.output/'metrics.jsonl').open('a') as stream: stream.write(json.dumps(event)+'\n')
        dist.broadcast(decision, 0)
        best_tensor = torch.tensor(best, device=device, dtype=torch.float64)
        dist.broadcast(best_tensor, 0); best = best_tensor.item()
        if decision.item(): save(step, True)
        model.train()
        return values

    latest_val = validate(start_step, True)
    save(start_step)
    iterator = iter(loader); rolling = []; sampled = torch.zeros(len(train.dataset_names), device=device, dtype=torch.long)
    start = time.monotonic(); step = start_step
    end_step = min(config['steps'], start_step+args.stop_after_updates) if args.stop_after_updates else config['steps']
    while step < end_step:
        stopping = torch.tensor(int(stop['requested'] or (args.output/'STOP').exists()), device=device)
        dist.all_reduce(stopping, op=dist.ReduceOp.MAX)
        if stopping.item(): stop['requested'] = True; break
        try: batch = next(iterator)
        except StopIteration: iterator = iter(loader); batch = next(iterator)
        batch = {k:v.to(device, non_blocking=True) for k,v in batch.items()}
        sampled += torch.bincount(batch['dataset_id'], minlength=len(train.dataset_names))
        history,raw,candidate,kwargs,target,future = sample_batch(batch, generator,
                      condition_on_beta=config.get('condition_on_beta', True))
        step += 1
        multiplier = (step/config['warmup_steps'] if step <= config['warmup_steps'] else
                      0.1+0.9*0.5*(1+math.cos(math.pi*(step-config['warmup_steps'])/max(1,config['steps']-config['warmup_steps']))))
        for group in optimizer.param_groups: group['lr'] = config['learning_rate']*multiplier
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16): output = ddp(history,raw,candidate,**kwargs)
        restore = state_loss(output['state'], target); forecast = state_loss(output['forecast'][:,0], future)
        loss = restore + config['forecast_weight']*forecast
        finite = torch.tensor(int(torch.isfinite(loss)), device=device)
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not finite.item(): raise FloatingPointError(f'Nonfinite rank loss at step {step}')
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step(); rolling.append(loss.detach())
        if step % config['log_every'] == 0 or step == end_step:
            scalars = torch.stack((torch.stack(rolling).mean(), restore.detach(), forecast.detach(), grad.detach()))
            dist.all_reduce(scalars); scalars /= world
            counts = sampled.clone(); dist.all_reduce(counts)
            # Includes actual process/device, per-rank samples and replica checksum.
            checksum = torch.stack([p.detach().float().sum() for p in model.parameters()]).sum()
            local = {'rank':rank, 'pid':os.getpid(), 'device':local_rank,
                     'peak_allocated_gb':torch.cuda.max_memory_allocated(device)/1e9,
                     'peak_reserved_gb':torch.cuda.max_memory_reserved(device)/1e9,
                     'model_sum':float(checksum), 'samples_by_dataset':dict(zip(train.dataset_names,sampled.cpu().tolist()))}
            ranks = [None]*world; dist.all_gather_object(ranks, local)
            if rank == 0:
                elapsed = time.monotonic()-start
                event = {'utc':utc(), 'phase':'training', 'step':step, 'updates_since_migration':step-start_step,
                         'loss':float(scalars[0]), 'restoration_loss':float(scalars[1]),
                         'forecast_loss':float(scalars[2]), 'grad_norm':float(scalars[3]),
                         'lr':optimizer.param_groups[0]['lr'], 'elapsed_seconds':elapsed,
                         'steps_per_second':(step-start_step)/elapsed,
                         'samples_per_second':(step-start_step)*config['batch_size']/elapsed,
                         'global_batch_size':config['batch_size'], 'per_gpu_batch_size':local_batch,
                         'world_size':world, 'ranks':ranks,
                         'samples_by_dataset':dict(zip(train.dataset_names,counts.cpu().tolist()))}
                with (args.output/'metrics.jsonl').open('a') as stream: stream.write(json.dumps(event)+'\n')
                atomic_json(args.output/'status.json', event); print(json.dumps(event), flush=True)
            rolling = []
        if step % config['eval_every'] == 0: latest_val = validate(step)
        if step % config['save_every'] == 0: save(step)
    save(step)
    if rank == 0:
        phase = 'stopped' if stop['requested'] else ('smoke_complete' if args.stop_after_updates else 'complete')
        atomic_json(args.output/'status.json', {'utc':utc(), 'step':step, 'phase':phase,
                    'world_size':world, 'global_batch_size':config['batch_size'],
                    'parent_step':start_step, 'best_validation_loss':best, 'validation':latest_val})
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
