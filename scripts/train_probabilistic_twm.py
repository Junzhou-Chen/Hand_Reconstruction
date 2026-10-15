#!/usr/bin/env python3
"""Remote-only current restoration and probabilistic long-trajectory training.

The future sequence is a loss target, never an encoder input. Mixture identity
is shared over the entire trajectory. Rotation likelihoods use normalized Haar
measure on SO(3); their densities/NLLs are not directly comparable to xyz NLLs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
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
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset

from train_twm import atomic_json, sample_batch, utc
from handloop import geometry as g
from handloop.data.trajectory import SourceBalancedSampler
from handloop.data.forecast_trajectory import ForecastTrajectoryWindows
from handloop.data.corruption import sample_mixed_batch
from handloop.models.probabilistic_twm import (
    ProbabilisticTWM, ProbabilisticTWMConfig,
    current_distribution_log_prob, forecast_distribution_log_prob,
)
from handloop.tracking import WandbTracker


def prepare_batch(batch, generator, mode='mixed', *, condition_on_beta=False,
                  corruption_profile=None, return_diagnostics=False):
    """Select an explicit corruption version without changing historical sampling."""
    valid = batch['future_valid']
    if valid.dtype != torch.bool or not bool(valid.any(-1).all()):
        raise ValueError('Every window needs at least one valid future target')
    if not bool(batch['future_gt_mask'][valid].all()):
        raise ValueError('Valid future targets require complete 17-group MANO GT')
    # Legacy preparation already handles arbitrary leading future dimensions.
    # Its full-GT guard must not mistake padding for partial MANO supervision.
    checked = {key:value for key,value in batch.items() if key != 'future_gt_mask'}
    if corruption_profile is not None and not mode.startswith('legacy_'):
        return sample_mixed_batch(checked, generator, mode,
            condition_on_beta=condition_on_beta, profile=corruption_profile,
            return_diagnostics=return_diagnostics)
    legacy_mode = mode.removeprefix('legacy_')
    prepared = sample_batch(checked, generator, legacy_mode, condition_on_beta=condition_on_beta)
    return (prepared, {}) if return_diagnostics else prepared


def state_geometry_loss(pred, target):
    """Per-state loss; accepts [B,K,F,99] and broadcast-compatible targets."""
    pos = F.smooth_l1_loss(pred[..., :3]/0.05,
                          target[..., :3].expand_as(pred[..., :3])/0.05,
                          reduction='none', beta=0.5).mean(-1)
    chordal = (g.state_to_rotmat(pred)-g.state_to_rotmat(target)).square().sum((-1,-2))/2
    return pos+2*chordal[..., 0]+chordal[..., 1:].mean(-1)


def target_velocity(current, future, times):
    previous = torch.cat((current[:, None], future[:, :-1]), dim=1)
    delta_time = torch.diff(times, dim=-1, prepend=torch.zeros_like(times[:, :1])).clamp_min(1e-5)
    dp = (future[..., :3]-previous[..., :3])/delta_time[..., None]
    rotations = g.state_to_rotmat(future) @ g.state_to_rotmat(previous).transpose(-1,-2)
    dr = g.so3_log_map(rotations).flatten(-2)/delta_time[..., None]
    return torch.cat((dp, dr), -1)


def mixture_translation_uncertainty(output):
    weights = output['mixture_weights'].float()[:, :, None, None]
    means = output['forecast_modes'][..., :3].float()
    component_variance = output['forecast_translation_std'].float().square()
    mean = (weights*means).sum(1,keepdim=True)
    within = (weights*component_variance).sum(1)
    total = (weights*(component_variance+(means-mean).square())).sum(1)
    return within.sqrt(), total.sqrt()


def objective(output, current, future, valid, times, config):
    """Exact mixture trajectory likelihood, normalized AFTER logsumexp.

    Conditional independence of xyz/rotations/timesteps given a mode is an
    explicit modeling assumption. Means are coupled through causal latent
    dynamics and integration. The 51-dimensional scaling changes loss units,
    not the probability distribution used for mode responsibilities.
    """
    restore = state_geometry_loss(output['state'], current).mean()
    current_lp = current_distribution_log_prob(output, current)
    if isinstance(current_lp, dict): current_lp = current_lp['total']
    current_nll = -current_lp.mean()/51
    lp = forecast_distribution_log_prob(output, future)['total']
    mask = valid[:, None].to(lp.dtype)
    steps = valid.sum(-1).to(lp.dtype)
    component_lp = (lp*mask).sum(-1)
    logits = F.log_softmax(output['mixture_logits'].float(), -1)
    trajectory_lp = torch.logsumexp(logits+component_lp, -1)
    future_nll = (-trajectory_lp/(51*steps)).mean()
    geometric = state_geometry_loss(output['forecast_modes'], future[:, None])
    per_mode = (geometric*mask).sum(-1)/steps[:, None]
    best_geometry = per_mode.min(-1).values.mean()
    responsibilities = (logits+component_lp).softmax(-1).detach()
    velocity_gt = target_velocity(current, future, times)
    velocity_error = output['forecast_velocity'].float()-velocity_gt[:, None]
    scales = velocity_error.new_tensor([0.5]*3+[4.0]*48)
    velocity = F.smooth_l1_loss(velocity_error/scales, torch.zeros_like(velocity_error),
                                beta=0.5, reduction='none').mean(-1)
    per_mode_velocity = (velocity*mask).sum(-1)/steps[:, None]
    velocity_loss = (per_mode_velocity*responsibilities).sum(-1).mean()
    balance = output['moe_balance_loss'].float()
    within_std,total_std = mixture_translation_uncertainty(output)
    loss = (restore+config['current_nll_weight']*current_nll
            +config['forecast_nll_weight']*future_nll
            +config['forecast_geometry_weight']*best_geometry
            +config['velocity_weight']*velocity_loss
            +config['moe_balance_weight']*balance)
    return loss, {'loss':loss, 'restoration_loss':restore, 'current_nll_per_dof':current_nll,
                  'forecast_nll_per_dof':future_nll, 'best_trajectory_geometry_loss':best_geometry,
                  'velocity_loss':velocity_loss, 'moe_balance_loss':balance,
                  'mode_entropy':-(logits.exp()*logits).sum(-1).mean(),
                  'current_xyz_std_mm':output['state_translation_std'].float().mean()*1000,
                  'future_within_mode_xyz_std_mm':(within_std.mean(-1)*valid).sum()/valid.sum()*1000,
                  'future_total_xyz_std_mm':(total_std.mean(-1)*valid).sum()/valid.sum()*1000}


def state_errors(pred, target):
    angle = g.geodesic_angle(g.state_to_rotmat(pred),g.state_to_rotmat(target))*180/math.pi
    return {'wrist_mm':(pred[..., :3]-target[..., :3]).norm(dim=-1)*1000,
            'root_deg':angle[..., 0], 'finger_deg':angle[..., 1:].mean(-1)}


def flatten_numbers(value, prefix=''):
    output = {}
    for key,item in value.items():
        name = prefix+'/'+str(key) if prefix else str(key)
        if isinstance(item, dict): output.update(flatten_numbers(item, name))
        elif isinstance(item, (float,int)) and not isinstance(item,bool) and math.isfinite(item): output[name]=item
    return output


@torch.no_grad()
def evaluate(model, loaders, device, config):
    model.eval()
    by_source = {}
    modes = config.get('eval_corruption_modes', ['noisy','missing','clean'])
    for source,loader in loaders.items():
        by_source[source] = {}
        for corruption in modes:
            generator = torch.Generator(device=device).manual_seed(config['seed']+123)
            totals,counts,tails = {},{},{}
            def add(key,values,valid=None):
                values = values.float()
                chosen = values.flatten() if valid is None else values[valid]
                if chosen.numel():
                    totals[key] = totals.get(key,0.)+float(chosen.sum())
                    counts[key] = counts.get(key,0)+chosen.numel()
            for index,batch in enumerate(loader):
                if index >= config['eval_batches']: break
                batch = {k:v.to(device,non_blocking=True) for k,v in batch.items()}
                history,raw,candidate,kwargs,target,future = prepare_batch(batch,generator,corruption,
                    condition_on_beta=config.get('condition_on_beta',False),
                    corruption_profile=config.get('corruption_profile'))
                with torch.autocast('cuda',dtype=torch.bfloat16): result = model(history,raw,candidate,**kwargs)
                valid = batch['future_valid']
                _,losses = objective(result,target,future,valid,batch['future_dt'],config)
                n = len(target)
                for key,value in losses.items(): add(key,value.expand(n))
                for name,state in [('candidate',candidate),('history_hold',history[:,-1]),('restored',result['state'])]:
                    for key,value in state_errors(state,target).items():
                        add(name+'/'+key,value)
                        tails.setdefault(name+'/'+key,[]).append(value.detach().float().cpu())
                input_wrist = state_errors(candidate,target)['wrist_mm']
                restored_wrist = state_errors(result['state'],target)['wrist_mm']
                add('wrist_degradation_over_1mm',(restored_wrist>input_wrist+1).float())
                last_delta_time = batch['dt'][:, -2].clamp_min(1e-5)
                hist_rotations = g.state_to_rotmat(history)
                hist_dp = (history[:,-1,:3]-history[:,-2,:3])/last_delta_time[:,None]
                hist_dr = g.so3_log_map(hist_rotations[:,-1]@hist_rotations[:,-2].transpose(-1,-2)).flatten(-2)/last_delta_time[:,None]
                velocity = torch.cat((hist_dp,hist_dr),-1)
                constant_velocity = g.state_apply_delta(result['state'][:,None],velocity[:,None]*batch['future_dt'][...,None])
                forecast_errors = state_errors(result['forecast'],future)
                hold_errors = state_errors(result['state'][:,None],future)
                cv_errors = state_errors(constant_velocity,future)
                for key,value in forecast_errors.items(): add('forecast_ade/'+key,value,valid)
                target_motion = state_errors(future,target[:,None])
                moving = ((target_motion['wrist_mm'].masked_fill(~valid,0).amax(-1)>=20)
                          |(target_motion['finger_deg'].masked_fill(~valid,0).amax(-1)>=10))
                for motion_group,rows_mask in [('moving',moving),('low_motion',~moving)]:
                    for key,value in forecast_errors.items():
                        add('future_motion_'+motion_group+'/map/'+key,value,valid&rows_mask[:,None])
                    for key,value in hold_errors.items():
                        add('future_motion_'+motion_group+'/restored_hold/'+key,value,valid&rows_mask[:,None])
                mode_errors = state_errors(result['forecast_modes'],future[:,None])
                mode_ade = (mode_errors['wrist_mm']*valid[:,None]).sum(-1)/valid.sum(-1)[:,None]
                add('forecast_minade_wrist_mm',mode_ade.min(-1).values)
                # Translation calibration diagnostics; empirical coverage is not a guarantee.
                current_mahal = ((target[:,:3]-result['state'][:,:3])/result['state_translation_std']).square().sum(-1)
                add('calibration/current_xyz_90pct_ellipsoid', (current_mahal <= 6.251389).float())
                standardized = (future[:,None,:,:3]-result['forecast_modes'][...,:3])/result['forecast_translation_std']
                component_cdf = 0.5*(1+torch.erf(standardized/math.sqrt(2)))
                mixture_cdf = (component_cdf*result['mixture_weights'][:,:,None,None]).sum(1)
                within_std,total_std = mixture_translation_uncertainty(result)
                add('calibration/future_xyz_90pct_marginal',((mixture_cdf>=.05)&(mixture_cdf<=.95)).float().mean(-1),valid)
                for seconds in config['eval_horizons_seconds']:
                    difference = (batch['future_dt']-seconds).abs().masked_fill(~valid,float('inf'))
                    distance,slot = difference.min(-1)
                    eligible = distance <= batch['dt'][:,-1]*0.6
                    rows = torch.arange(n,device=device)
                    label = f'{seconds:g}s'
                    for group,errors in [('map',forecast_errors),('restored_hold',hold_errors),('constant_velocity',cv_errors)]:
                        for key,value in errors.items(): add(f'future_{label}/{group}/{key}',value[rows,slot],eligible)
                    add(f'future_{label}/within_mode_xyz_std_mm',within_std.mean(-1)[rows,slot]*1000,eligible)
                    add(f'future_{label}/total_xyz_std_mm',total_std.mean(-1)[rows,slot]*1000,eligible)
            values = {key:totals[key]/counts[key] for key in totals}
            if not all(math.isfinite(value) for value in values.values()):
                raise FloatingPointError(f'Nonfinite validation measurement: {source}/{corruption}')
            for key,chunks in tails.items():
                joined=torch.cat(chunks)
                values[key+'_p50']=float(joined.quantile(0.5))
                values[key+'_p95']=float(joined.quantile(0.95))
            values['samples'] = counts.get('restored/wrist_mm',0)
            values['counts'] = {f'{seconds:g}s':counts.get(f'future_{seconds:g}s/map/wrist_mm',0)
                                for seconds in config['eval_horizons_seconds']}
            by_source[source][corruption] = values
    result = {'by_dataset':by_source,'aggregation':'equal-source mean over sources with eligible targets at each horizon; p50/p95 macro values are source-quantile means, not pooled quantiles'}
    for corruption in modes:
        keys = set().union(*(item[corruption].keys() for item in by_source.values()))-{'counts','samples'}
        result[corruption] = {key:sum(item[corruption][key] for item in by_source.values() if key in item[corruption])
                             /sum(key in item[corruption] for item in by_source.values()) for key in keys}
    model.train()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','config','output'): parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--code-version',required=True)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--stop-after-updates',type=int)
    parser.add_argument('--batch-size',type=int,help='explicit fresh-run/smoke global batch override')
    parser.add_argument('--eval-batches',type=int)
    parser.add_argument('--tracking-mode',choices=('online','offline','disabled'))
    parser.add_argument('--stop-before-utc',help='Execution-only deadline override for a new allocation; ISO timestamp or none')
    args = parser.parse_args()
    if not Path('/sciclone/home/jchen57').is_dir(): parser.error('Run inside the authorized remote compute Pod')
    config = json.loads(args.config.read_text())
    if config.get('corruption_profile_path'):
        profile_path=Path(__file__).resolve().parents[1]/config['corruption_profile_path']
        config['corruption_profile']=json.loads(profile_path.read_text())
        config['corruption_profile_sha256']=hashlib.sha256(profile_path.read_bytes()).hexdigest()
        if config['corruption_profile'].get('status')!='active':
            raise ValueError('Training needs an explicitly active corruption profile')
    if args.batch_size: config['batch_size']=args.batch_size
    if args.eval_batches: config['eval_batches']=args.eval_batches
    if args.tracking_mode:
        config['tracking']['mode']=args.tracking_mode
        config['tracking']['require_online']=args.tracking_mode=='online'
    deadline_value=args.stop_before_utc if args.stop_before_utc is not None else config.get('stop_before_utc')
    deadline=None if deadline_value in (None,'none') else datetime.fromisoformat(deadline_value)
    if deadline is not None and deadline.utcoffset() is None:
        raise ValueError('Execution deadline needs an explicit UTC offset')
    rank,local_rank,world = (int(os.environ[k]) for k in ('RANK','LOCAL_RANK','WORLD_SIZE'))
    if world != config['world_size'] or config['batch_size']%world: raise ValueError('Invalid world/global batch')
    local_batch = config['batch_size']//world
    torch.set_num_threads(config.get('cpu_threads',4)); torch.cuda.set_device(local_rank)
    device = torch.device('cuda',local_rank)
    seed = config['seed']+rank*100003
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    generator = torch.Generator(device=device).manual_seed(seed)
    dist.init_process_group('nccl',timeout=timedelta(minutes=30),device_id=device)
    probe = torch.tensor(float(rank+1),device=device); dist.all_reduce(probe)
    if probe.item()!=world*(world+1)/2: raise RuntimeError('NCCL probe failed')
    stop = {'requested':False}
    def request_stop(signum,frame): stop['requested']=True
    signal.signal(signal.SIGTERM,request_stop); signal.signal(signal.SIGINT,request_stop)
    if rank==0:
        args.output.mkdir(parents=True,exist_ok=True)
        if (args.output/'run_metadata.json').exists() and not args.resume: raise ValueError('Use a fresh output directory')
        if (args.output/'STOP').exists(): raise ValueError('Archive previous STOP explicitly before resume')
        atomic_json(args.output/'status.json',{'utc':utc(),'phase':'initializing_ddp'})
    dist.barrier()
    model = ProbabilisticTWM(ProbabilisticTWMConfig(**config['model'])).to(device)
    optimizer = torch.optim.AdamW([
        {'params':[p for p in model.parameters() if not getattr(p,'_no_weight_decay',False)],'weight_decay':config['weight_decay']},
        {'params':[p for p in model.parameters() if getattr(p,'_no_weight_decay',False)],'weight_decay':0.0},
    ],lr=config['learning_rate'])
    start_step,best,checkpoint = 0,float('inf'),None
    manifest_hash = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    if args.resume:
        checkpoint = torch.load(args.resume,map_location='cpu',weights_only=False)
        if checkpoint['config']!=config or checkpoint['run']['manifest_sha256']!=manifest_hash or checkpoint['run']['world_size']!=world:
            raise ValueError('Resume requires unchanged configuration, source manifests and world size')
        model.load_state_dict(checkpoint['model'],strict=True); optimizer.load_state_dict(checkpoint['optimizer'])
        start_step,best = checkpoint['step'],checkpoint['best_validation_loss']
    ddp = DDP(model,device_ids=[local_rank],output_device=local_rank,broadcast_buffers=False,gradient_as_bucket_view=True)
    data_kwargs = {k:config[k] for k in ('cache_sequences','cache_bytes')}
    data_kwargs.update(history=config['history_frames'],forecast_frames=config['model']['forecast_horizon'],
                       min_future_frames=config['min_future_frames'],window_stride=config['window_stride'])
    train = ForecastTrajectoryWindows(args.manifest,'train',**data_kwargs)
    if set(train.dataset_names)!=set(config['required_datasets']): raise ValueError('Required training sources differ')
    if any(train.windows_by_dataset[name]<1 for name in config['required_datasets']):
        raise ValueError('Every required source needs usable train windows')
    sampler = SourceBalancedSampler(train,num_samples=config['steps']*local_batch,seed=seed)
    loader = DataLoader(train,batch_size=local_batch,sampler=sampler,num_workers=0,pin_memory=True,drop_last=True,
                        generator=torch.Generator().manual_seed(seed))
    loaders = {}
    if rank==0:
        val = ForecastTrajectoryWindows(args.manifest,'val',**data_kwargs)
        if set(val.dataset_names)!=set(train.dataset_names): raise ValueError('Validation sources differ')
        if any(val.windows_by_dataset[name]<1 for name in config['required_datasets']):
            raise ValueError('Every required source needs usable validation windows')
        loaders = {name:DataLoader(Subset(val,indices),batch_size=config['eval_batch_size'],num_workers=0,pin_memory=True)
                   for name,indices in val.fixed_indices_by_dataset(config['eval_batch_size']*config['eval_batches']).items()}
    if checkpoint:
        state=checkpoint['rank_states'][rank]
        generator.set_state(state['corruption_rng']); torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state(state['cuda_rng'],device); random.setstate(state['python_rng'])
        np.random.set_state(state['numpy_rng']); sampler.load_state_dict(state['sampler'])
    run = {'started_utc':utc(),'code_version':args.code_version,'pid':os.getpid(),'world_size':world,
           'global_batch_size':config['batch_size'],'per_gpu_batch_size':local_batch,'config':config,
           'effective_stop_before_utc':deadline.isoformat() if deadline else None,
           'deadline_override':args.stop_before_utc,
           'config_file_sha256':hashlib.sha256(args.config.read_bytes()).hexdigest(),
           'resolved_config_sha256':hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),
           'manifest':str(args.manifest),'manifest_sha256':manifest_hash,'source_revisions':train.source_revisions,
           'parameters':sum(p.numel() for p in model.parameters()),'torch':torch.__version__,
           'gpu':torch.cuda.get_device_name(device),'dataset_names':train.dataset_names,
           'train_windows_by_dataset':train.windows_by_dataset,'initialization':'resume' if args.resume else 'from scratch',
           'train_forecast_contract':train.forecast_contract,
           'rotation6d':g.ROTATION_6D_CONVENTION,'position_units':'metres in noisy-history wrist anchor frame',
           'sampling':'equal source then uniform windows, independent rank RNG, with replacement',
           'GT':'source fitted MANO references; unchanged recording-disjoint four-source splits; no test labels',
           'beta_conditioning':config['condition_on_beta'],'future_targets':'next up to 60 contiguous source frames, mask tails/gaps; physical timestamps',
           'scope':'synthetic corruption pure-motion current restoration and multi-modal trajectory forecast; no visual correction evidence yet',
           'resume_limit':'rank RNG saved; sampler prefetched chunks/loader cursor not bitwise replay',
           'likelihood':'xyz diagonal Gaussian in m; isotropic matrix Fisher on SO3 normalized Haar; trajectory-wide categorical mixture'}
    source_root=Path(__file__).resolve().parents[1]
    run['source_sha256']={name:hashlib.sha256((source_root/name).read_bytes()).hexdigest() for name in (
        'scripts/train_probabilistic_twm.py','scripts/train_twm.py','src/handloop/geometry.py',
        'src/handloop/models/probabilistic_twm.py','src/handloop/models/structured_mamba.py',
        'src/handloop/data/trajectory.py','src/handloop/data/forecast_trajectory.py','src/handloop/tracking.py',
        'src/handloop/data/corruption.py')}
    tracker = WandbTracker(config=config['tracking'],output_dir=args.output,metadata=run,rank=rank,resume=bool(args.resume))
    if rank==0:
        run.update(wandb=tracker.metadata,val_windows_by_dataset=val.windows_by_dataset,
                   val_forecast_contract=val.forecast_contract,
                   val_samples_by_dataset={name:len(item.dataset) for name,item in loaders.items()})
        atomic_json(args.output/'run_metadata.json',run); print(json.dumps(run),flush=True)
    checkpoint = None

    def save(step,best_checkpoint=False):
        digest=hashlib.sha256()
        for name,tensor in model.state_dict().items():
            digest.update((name+str(tensor.dtype)+str(tuple(tensor.shape))).encode())
            digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        rank_state = {'rank':rank,'corruption_rng':generator.get_state().cpu(),'torch_rng':torch.get_rng_state(),
                      'cuda_rng':torch.cuda.get_rng_state(device).cpu(),'python_rng':random.getstate(),
                      'numpy_rng':np.random.get_state(),'sampler':sampler.state_dict(),'model_state_sha256':digest.hexdigest()}
        states=[None]*world; dist.all_gather_object(states,rank_state)
        if len({state['model_state_sha256'] for state in states})!=1: raise RuntimeError('DDP replicas differ at save')
        if rank==0:
            atomic_json(args.output/'replica_checkpoint_audit.json',{'utc':utc(),'step':step,'byte_equal':True,'sha256':digest.hexdigest()})
            path=args.output/('best.pt' if best_checkpoint else 'latest.pt'); tmp=path.with_suffix('.pt.tmp')
            torch.save({'step':step,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'config':config,
                        'code_version':args.code_version,'best_validation_loss':best,'rank_states':states,'run':run},tmp)
            tmp.replace(path)
        dist.barrier()

    def validate(step,initial=False):
        nonlocal best
        decision=torch.zeros(1,device=device); values=None
        if rank==0:
            values=evaluate(model,loaders,device,config)
            selection=config.get('best_selection_weights',{'noisy':0.5,'missing':0.5})
            if (any(weight<0 for weight in selection.values())
                    or not math.isclose(sum(selection.values()),1.0)):
                raise ValueError('Best-checkpoint validation weights must be a probability mixture')
            score=sum(weight*values[mode]['restoration_loss'] for mode,weight in selection.items())
            if score<best: best=score; decision.fill_(1)
            event={'utc':utc(),'step':step,'validation':values,'best_validation_loss':best}
            atomic_json(args.output/('initial_validation.json' if initial else 'latest_validation.json'),event)
            tracker.log(flatten_numbers(values,'validation'),step=step)
            print(json.dumps(event),flush=True)
        dist.broadcast(decision,0)
        best_value=torch.tensor(best,device=device,dtype=torch.float64); dist.broadcast(best_value,0); best=best_value.item()
        if decision.item(): save(step,True)
        model.train(); return values

    latest_val=validate(start_step,True); save(start_step)
    iterator=iter(loader); step=start_step; start=time.monotonic()
    sampled=torch.zeros(len(train.dataset_names),device=device,dtype=torch.long)
    corruption_counts=torch.zeros(5,device=device,dtype=torch.long)
    end_step=min(config['steps'],start_step+args.stop_after_updates) if args.stop_after_updates else config['steps']
    while step<end_step:
        stopping=torch.tensor(int(stop['requested'] or (args.output/'STOP').exists() or
            (deadline is not None and datetime.now(timezone.utc)>=deadline)),device=device)
        dist.all_reduce(stopping,op=dist.ReduceOp.MAX)
        if stopping.item(): stop['requested']=True; break
        try: batch=next(iterator)
        except StopIteration: iterator=iter(loader); batch=next(iterator)
        batch={k:v.to(device,non_blocking=True) for k,v in batch.items()}
        sampled+=torch.bincount(batch['dataset_id'],minlength=len(train.dataset_names))
        prepared,diagnostics=prepare_batch(batch,generator,condition_on_beta=config['condition_on_beta'],
            corruption_profile=config.get('corruption_profile'),return_diagnostics=True)
        history,raw,candidate,kwargs,target,future=prepared
        if 'case_id' in diagnostics:
            corruption_counts+=torch.bincount(diagnostics['case_id'],minlength=5)
        step+=1
        multiplier=step/config['warmup_steps'] if step<=config['warmup_steps'] else .1+.9*.5*(1+math.cos(math.pi*(step-config['warmup_steps'])/max(1,config['steps']-config['warmup_steps'])))
        for group in optimizer.param_groups: group['lr']=config['learning_rate']*multiplier
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16): output=ddp(history,raw,candidate,**kwargs)
        loss,losses=objective(output,target,future,batch['future_valid'],batch['future_dt'],config)
        finite=torch.tensor(int(torch.isfinite(loss)),device=device); dist.all_reduce(finite,op=dist.ReduceOp.MIN)
        if not finite.item(): raise FloatingPointError(f'Nonfinite loss at step {step}')
        loss.backward(); grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
        optimizer.step()
        if step%config['log_every']==0 or step==end_step:
            keys=list(losses); scalars=torch.stack([losses[k].detach() for k in keys]+[grad.detach()])
            dist.all_reduce(scalars); scalars/=world
            counts=sampled.clone(); dist.all_reduce(counts)
            case_counts=corruption_counts.clone(); dist.all_reduce(case_counts)
            loads=output['moe_load'].detach().float(); dist.all_reduce(loads); loads/=world
            local={'rank':rank,'pid':os.getpid(),'device':local_rank,
                   'peak_allocated_gb':torch.cuda.max_memory_allocated(device)/1e9,
                   'peak_reserved_gb':torch.cuda.max_memory_reserved(device)/1e9}
            ranks=[None]*world; dist.all_gather_object(ranks,local)
            if rank==0:
                elapsed=time.monotonic()-start
                event={'utc':utc(),'phase':'training','step':step,**dict(zip(keys,scalars[:-1].cpu().tolist())),
                       'grad_norm':float(scalars[-1]),'lr':optimizer.param_groups[0]['lr'],'elapsed_seconds':elapsed,
                       'samples_per_second':(step-start_step)*config['batch_size']/elapsed,'ranks':ranks,
                       'samples_by_dataset':dict(zip(train.dataset_names,counts.cpu().tolist())),
                       'corruption_samples_by_case':dict(zip(
                           ('clean_motion','current_reanchors_history','ordinary_restoration','partial_missing','rare_local_failure'),
                           case_counts.cpu().tolist())),
                       'moe_load':{str(i):float(value) for i,value in enumerate(loads)},'wandb_url':tracker.url}
                atomic_json(args.output/'status.json',event)
                with (args.output/'metrics.jsonl').open('a') as stream: stream.write(json.dumps(event)+'\n')
                tracker.log(flatten_numbers(event,'train'),step=step); print(json.dumps(event),flush=True)
        if step%config['eval_every']==0: latest_val=validate(step)
        if step%config['save_every']==0 or step==start_step+config['log_every']: save(step)
    save(step)
    if rank==0:
        phase='stopped' if stop['requested'] else ('smoke_complete' if args.stop_after_updates else 'complete')
        atomic_json(args.output/'status.json',{'utc':utc(),'step':step,'phase':phase,'best_validation_loss':best,
                    'wandb_url':tracker.url,'validation':latest_val})
        tracker.finish()
    dist.destroy_process_group()


if __name__=='__main__': main()
