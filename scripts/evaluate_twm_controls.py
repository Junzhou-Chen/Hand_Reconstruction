#!/usr/bin/env python3
"""Fixed-window input controls: history only and correct current/noisy history.

These are synthetic diagnostic interventions, not visual-frontend results.
All cases share the same noisy past and target. No future state enters model.
"""
import argparse
import hashlib
import json
from pathlib import Path
from train_twm import (torch,TWM,TWMConfig,TrajectoryWindows,sample_batch,metrics,atomic_json,utc)
from torch.utils.data import DataLoader,Subset


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','manifest','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--code-version',required=True)
    args=parser.parse_args()
    if not Path('/sciclone/home/jchen57').is_dir():parser.error('Remote Pod only')
    if args.output.exists():parser.error('Fresh output JSON required')
    torch.set_num_threads(4);device=torch.device('cuda:0')
    ckpt=torch.load(args.checkpoint,map_location='cpu',weights_only=False);cfg=ckpt['config']
    if hashlib.sha256(args.manifest.read_bytes()).hexdigest()!=ckpt['run']['manifest_sha256']:raise ValueError('Manifest changed')
    model=TWM(TWMConfig(**cfg['model'])).to(device);model.load_state_dict(ckpt['model']);model.eval()
    val=TrajectoryWindows(args.manifest,'val',cfg['history_frames'],cfg['forecast_offset_frames'],cfg['window_stride'],cache_sequences=8192,cache_bytes=4*1024**3)
    batch_size=cfg.get('eval_batch_size',cfg['batch_size'])
    by_source={};indices=val.fixed_indices_by_dataset(batch_size*cfg['eval_batches'])
    for source,ids in indices.items():
        generator=torch.Generator(device=device).manual_seed(cfg['seed']+123)
        totals={case:{} for case in ('normal_noisy','history_only','correct_current_noisy_history')};count=0
        for batch in DataLoader(Subset(val,ids),batch_size=batch_size,pin_memory=True):
            batch={k:v.to(device) for k,v in batch.items()}
            history,raw,candidate,kwargs,target,_=sample_batch(batch,generator,'noisy',condition_on_beta=cfg.get('condition_on_beta',True))
            normal=None
            for case in totals:
                options=kwargs.copy();r=raw;c=candidate
                if case=='history_only':
                    r=torch.zeros_like(raw);r[:,3:]=torch.tensor([1,0,0,0,1,0],device=device).repeat(16)
                    c=history[:,-1].clone();options['availability']=torch.zeros_like(kwargs['availability']);options['quality']=torch.zeros_like(kwargs['quality'])
                elif case=='correct_current_noisy_history':
                    r=target.clone();c=target.clone();options['availability']=torch.ones_like(kwargs['availability']);options['quality']=torch.ones_like(kwargs['quality'])
                with torch.autocast('cuda',dtype=torch.bfloat16):prediction=model(history,r,c,**options)['state']
                if normal is None:normal=prediction
                values=metrics(prediction,target)
                values.update({'input_'+k:v for k,v in metrics(c,target).items()})
                values.update({'change_from_normal_'+k:v for k,v in metrics(prediction,normal).items()})
                for k,v in values.items():totals[case][k]=totals[case].get(k,0.)+v*len(target)
            count+=len(target)
        by_source[source]={case:{**{k:v/count for k,v in values.items()},'samples':count} for case,values in totals.items()}
        print(json.dumps({'dataset':source,'results':by_source[source]}),flush=True)
    aggregate={}
    for case in next(iter(by_source.values())):
        aggregate[case]={k:sum(v[case][k] for v in by_source.values())/len(by_source) for k in next(iter(by_source.values()))[case] if k!='samples'}
        aggregate[case]['samples']=sum(v[case]['samples'] for v in by_source.values())
    result={'phase':'complete','utc':utc(),'code_version':args.code_version,'checkpoint_step':ckpt['step'],
            'checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            'manifest_sha256':ckpt['run']['manifest_sha256'],'batch_size':batch_size,'corruption_seed':cfg['seed']+123,
            'by_dataset':by_source,'equal_source_mean':aggregate,'optimizer_updates':0,
            'scope':'Fixed training-val subset; synthetic interventions with identical noisy history. Correct-current uses oracle current target and quality=1 as a diagnostic, not deployable extra GT or proof of physics.',
            'indices_sha256_by_dataset':{k:hashlib.sha256(json.dumps(v).encode()).hexdigest() for k,v in indices.items()}}
    args.output.parent.mkdir(parents=True,exist_ok=True);atomic_json(args.output,result)


if __name__=='__main__':main()
