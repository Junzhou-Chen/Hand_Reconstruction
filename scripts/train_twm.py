#!/usr/bin/env python3
"""Remote-only OakInk2 pilot: current restoration plus short forecast.

Metrics are wrist xyz (mm) and root/finger SO(3) error (degrees), against
source fitted MANO reference. No joint MPJPE/mesh result without MANO assets.
Validation corruption uses a fixed RNG seed and original recording split.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from handloop import geometry as g
from handloop.data.trajectory import TrajectoryWindows
from handloop.models import TWM, TWMConfig


def utc(): return datetime.now(timezone.utc).isoformat()


def atomic_json(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n')
    tmp.replace(path)


def sample_batch(batch, generator, mode='mixed'):
    clean=batch['past_current'].float()
    b,h1,_=clean.shape
    h=h1-1
    device=clean.device
    def rand(*shape): return torch.rand(*shape,device=device,generator=generator)
    def normal(*shape): return torch.randn(*shape,device=device,generator=generator)
    if mode=='mixed':
        draw=rand(b)
        clean_case=draw<0.15
        missing_case=(draw>=0.15)&(draw<0.55)
    else:
        clean_case=torch.full((b,),mode=='clean',device=device,dtype=torch.bool)
        missing_case=torch.full((b,),mode=='missing',device=device,dtype=torch.bool)
    active=(~clean_case).float()
    hist_delta=normal(b,h,51)
    hist_delta[...,:3]*=0.004
    hist_delta[...,3:]*=math.radians(1.5)
    hist_delta*=active[:,None,None]
    hist_world=g.state_apply_delta(clean[:,:h],hist_delta)
    anchor=hist_world[:,-1]
    anchor_r=g.state_to_rotmat(anchor)[:,0]
    def to_anchor(x):
        rotations=g.state_to_rotmat(x)
        extra=x.ndim-2
        ra=anchor_r.reshape(b,*([1]*extra),3,3)
        pa=anchor[:,:3].reshape(b,*([1]*extra),3)
        p=(ra.transpose(-1,-2)@(x[...,:3]-pa)[...,None]).squeeze(-1)
        rotations=rotations.clone()
        rotations[...,0,:,:]=ra.transpose(-1,-2)@rotations[...,0,:,:]
        return g.state_from_rotmat(p,rotations)
    history=to_anchor(hist_world)
    target=to_anchor(clean[:,-1])
    future=to_anchor(batch['future'].float())
    delta=normal(b,51)
    delta[:,:3]*=0.025
    delta[:,3:6]*=math.radians(12)
    delta[:,6:]*=math.radians(8)
    severe=(rand(b)<0.1)&~clean_case
    delta*=torch.where(severe,3.,1.)[:,None]
    delta*=active[:,None]
    raw=g.state_apply_delta(target,delta)
    available=torch.ones(b,17,device=device,dtype=torch.bool)
    available &= ~((rand(b,17)<0.2)&missing_case[:,None])
    whole_finger=(rand(b)<0.5)&missing_case
    finger=(rand(b)*5).long()
    groups=torch.arange(17,device=device)[None]
    finger_slots=(groups>=2+finger[:,None]*3)&(groups<2+(finger[:,None]+1)*3)
    available &= ~(whole_finger[:,None]&finger_slots)
    full_missing=(rand(b)<0.1)&missing_case
    available[full_missing]=False
    # Candidate fills unavailable slots ONLY from completed noisy history.
    base=history[:,-1]
    candidate=raw.clone()
    candidate[:,:3]=torch.where(available[:,:1],raw[:,:3],base[:,:3])
    raw_r=raw[:,3:].reshape(b,16,6)
    candidate[:,3:]=torch.where(available[:,1:,None],raw_r,base[:,3:].reshape(b,16,6)).flatten(1)
    raw[:,:3]=torch.where(available[:,:1],raw[:,:3],0.)
    raw[:,3:]=torch.where(available[:,1:,None],raw_r,g.identity_rotation_6d(device=device)).flatten(1)
    # Noisy, occasionally false-high quality hints, NOT calibrated GT error.
    quality=0.55+rand(b,17)*0.3
    quality[clean_case]=1.
    false_high=(rand(b,17)<0.2)&~clean_case[:,None]
    quality[false_high]=0.98
    quality=torch.where(available,quality,0.)
    quality=torch.where((rand(b,17)<0.2)&available,-1.,quality)
    kwargs={'history_valid':torch.ones(b,h,device=device,dtype=torch.bool),
            'history_availability':torch.ones(b,h,17,device=device,dtype=torch.bool),
            'availability':available,'candidate_availability':torch.ones_like(available),
            'dt':batch['dt'],'beta':batch['beta'],'handedness':batch['handedness'],
            'quality':quality,'history_quality':None,'forecast_dt':batch['forecast_dt']}
    return history,raw,candidate,kwargs,target,future


def state_loss(pred,target):
    position=F.smooth_l1_loss(pred[:,:3]/0.05,target[:,:3]/0.05,beta=0.5)
    rotations=g.state_to_rotmat(pred)
    true_rotations=g.state_to_rotmat(target)
    chordal=(rotations-true_rotations).square().sum((-1,-2))/2
    return position+2*chordal[:,0].mean()+chordal[:,1:].mean()


def metrics(pred,target):
    angles=g.geodesic_angle(g.state_to_rotmat(pred),g.state_to_rotmat(target))*180/math.pi
    return {'wrist_mm':float((pred[:,:3]-target[:,:3]).norm(dim=-1).mean()*1000),
            'root_deg':float(angles[:,0].mean()),'finger_deg':float(angles[:,1:].mean())}


@torch.no_grad()
def evaluate(model,loader,device,config):
    model.eval()
    result={}
    for mode in ('noisy','missing','clean'):
        generator=torch.Generator(device=device).manual_seed(config['seed']+123)
        totals={}
        count=0
        for index,batch in enumerate(loader):
            if index>=config['eval_batches']: break
            batch={k:v.to(device,non_blocking=True) for k,v in batch.items()}
            history,raw,candidate,kwargs,target,future=sample_batch(batch,generator,mode)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                output=model(history,raw,candidate,**kwargs)
            n=len(target)
            values={'loss':float(state_loss(output['state'],target))}
            values.update({'baseline_'+k:v for k,v in metrics(candidate,target).items()})
            values.update({'restored_'+k:v for k,v in metrics(output['state'],target).items()})
            values.update({'forecast_'+k:v for k,v in metrics(output['forecast'][:,0],future).items()})
            for key,value in values.items(): totals[key]=totals.get(key,0.)+value*n
            count+=n
        result[mode]={k:v/count for k,v in totals.items()}
        result[mode]['samples']=count
    model.train()
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--code-version',required=True)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--steps',type=int)
    args=parser.parse_args()
    if not Path('/sciclone/home/jchen57').is_dir(): parser.error('Run training in the remote compute Pod')
    config=json.loads(args.config.read_text())
    if args.steps is not None: config['steps']=args.steps
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    device=torch.device('cuda:0')
    generator=torch.Generator(device=device).manual_seed(config['seed'])
    model=TWM(TWMConfig(**config['model'])).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    train=TrajectoryWindows(args.manifest,'train',config['history_frames'],config['forecast_offset_frames'],config['window_stride'])
    val=TrajectoryWindows(args.manifest,'val',config['history_frames'],config['forecast_offset_frames'],config['window_stride'])
    loader=DataLoader(train,batch_size=config['batch_size'],shuffle=True,num_workers=0,pin_memory=True,drop_last=True,
                      generator=torch.Generator().manual_seed(config['seed']))
    val_ids=np.linspace(0,len(val)-1,min(len(val),config['batch_size']*config['eval_batches']),dtype=int).tolist()
    val_loader=DataLoader(Subset(val,val_ids),batch_size=config['batch_size'],num_workers=0,pin_memory=True)
    if not len(loader): raise RuntimeError('Training dataset smaller than one batch')
    stop={'requested':False}
    def request_stop(signum,frame): stop['requested']=True
    signal.signal(signal.SIGTERM,request_stop); signal.signal(signal.SIGINT,request_stop)
    start_step=0; best=float('inf')
    if args.resume:
        checkpoint=torch.load(args.resume,map_location=device,weights_only=False)
        model.load_state_dict(checkpoint['model']); optimizer.load_state_dict(checkpoint['optimizer'])
        start_step=checkpoint['step']; best=checkpoint.get('best_validation_loss',best)
        if 'rng_state' in checkpoint: generator.set_state(checkpoint['rng_state'].cpu())
        if 'torch_rng_state' in checkpoint: torch.set_rng_state(checkpoint['torch_rng_state'].cpu())
        if 'cuda_rng_states' in checkpoint: torch.cuda.set_rng_state_all([v.cpu() for v in checkpoint['cuda_rng_states']])
    run={'started_utc':utc(),'phase':'starting','pid':os.getpid(),'code_version':args.code_version,
         'config':config,'manifest':str(args.manifest),'dataset_revision':train.metadata['source_revision'],
         'train_windows':len(train),'val_windows':len(val),'train_trajectories':len(train.sequences),'val_trajectories':len(val.sequences),
         'parameters':sum(p.numel() for p in model.parameters()),'torch':torch.__version__,
         'gpu':torch.cuda.get_device_name(device),'rotation6d':g.ROTATION_6D_CONVENTION,
         'metrics':'source wrist xyz mm; root/finger geodesic degrees; NO MPJPE/mesh; fitted MANO reference',
         'scope':'OakInk2 recording-disjoint pilot; synthetic corruption only; no RGB, no real HaWoR adaptation'}
    atomic_json(args.output/'run_metadata.json',run)
    def save(step,best_checkpoint=False):
        checkpoint={'step':step,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                    'config':config,'code_version':args.code_version,'best_validation_loss':best,
                    'rng_state':generator.get_state(),'torch_rng_state':torch.get_rng_state(),
                    'cuda_rng_states':torch.cuda.get_rng_state_all(),'run':run}
        path=args.output/('best.pt' if best_checkpoint else 'latest.pt')
        tmp=path.with_suffix('.pt.tmp'); torch.save(checkpoint,tmp); tmp.replace(path)
    print(json.dumps(run),flush=True)
    initial=evaluate(model,val_loader,device,config)
    atomic_json(args.output/'initial_validation.json',initial)
    print(json.dumps({'step':start_step,'validation':initial}),flush=True)
    save(start_step)
    model.train(); iterator=iter(loader); start=time.monotonic(); rolling=[]; latest_val=initial
    step=start_step
    with (args.output/'metrics.jsonl').open('a',buffering=1) as stream:
        while step<config['steps'] and not stop['requested']:
            try: batch=next(iterator)
            except StopIteration: iterator=iter(loader); batch=next(iterator)
            batch={k:v.to(device,non_blocking=True) for k,v in batch.items()}
            history,raw,candidate,kwargs,target,future=sample_batch(batch,generator)
            step+=1
            if step<=config['warmup_steps']: multiplier=step/config['warmup_steps']
            else: multiplier=0.1+0.9*0.5*(1+math.cos(math.pi*(step-config['warmup_steps'])/max(1,config['steps']-config['warmup_steps'])))
            for group in optimizer.param_groups: group['lr']=config['learning_rate']*multiplier
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                output=model(history,raw,candidate,**kwargs)
            restore=state_loss(output['state'],target)
            forecast=state_loss(output['forecast'][:,0],future)
            loss=restore+config['forecast_weight']*forecast
            if not torch.isfinite(loss): raise FloatingPointError(f'Nonfinite loss at step {step}')
            loss.backward()
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
            optimizer.step()
            rolling.append(float(loss.detach()))
            if step%config['log_every']==0:
                event={'utc':utc(),'step':step,'phase':'training','loss':sum(rolling)/len(rolling),
                       'restoration_loss':float(restore.detach()),'forecast_loss':float(forecast.detach()),
                       'grad_norm':float(grad),'lr':optimizer.param_groups[0]['lr'],
                       'elapsed_seconds':round(time.monotonic()-start,1),
                       'steps_per_second':(step-start_step)/max(1e-6,time.monotonic()-start),
                       'gpu_memory_gb':torch.cuda.max_memory_allocated()/1e9}
                stream.write(json.dumps(event)+'\n'); print(json.dumps(event),flush=True)
                atomic_json(args.output/'status.json',event); rolling=[]
            if step%config['eval_every']==0:
                latest_val=evaluate(model,val_loader,device,config)
                score=(latest_val['noisy']['loss']+latest_val['missing']['loss'])/2
                if score<best: best=score; save(step,True)
                event={'utc':utc(),'step':step,'validation':latest_val,'best_validation_loss':best}
                stream.write(json.dumps(event)+'\n'); print(json.dumps(event),flush=True)
                atomic_json(args.output/'latest_validation.json',event)
            if step%config['save_every']==0: save(step)
        save(step)
        atomic_json(args.output/'status.json',{'utc':utc(),'step':step,'phase':'stopped' if stop['requested'] else 'complete',
                                             'best_validation_loss':best,'validation':latest_val})


if __name__=='__main__': main()
