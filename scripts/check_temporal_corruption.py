#!/usr/bin/env python3
"""Remote semantic checks for temporal augmentation and finite epoch coverage."""
import argparse
import json
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not Path('/sciclone/home/jchen57').is_dir(): parser.error('Remote Pod only')
    root=Path(__file__).resolve().parents[1]
    sys.path[:0]=[str(root/'src'),str(root/'scripts')]
    import torch
    from handloop import geometry as g
    from handloop.data.forecast_trajectory import ForecastTrajectoryWindows
    from handloop.data.epoch_sampling import CoverageEpochSampler
    from handloop.models.twm import TWM,TWMConfig
    from train_probabilistic_twm import prepare_batch
    torch.set_num_threads(2)
    device='cuda:0'; torch.cuda.set_device(0)
    profile=json.loads((root/'config/twm_corruption_transient_v3.json').read_text())
    train=ForecastTrajectoryWindows(args.manifest,'train',history=30,forecast_frames=60,
        min_future_frames=3,window_stride=4,cache_sequences=16,cache_bytes=128*1024**2)
    samples=[train[ids[0]] for ids in train.fixed_indices_by_dataset(1).values()]
    batch={k:torch.stack([s[k] for s in samples]).to(device) for k in samples[0]}
    batch={k:v[torch.arange(1024,device=device)%4] for k,v in batch.items()}
    def sample(data,mode):
        rng=torch.Generator(device=device).manual_seed(20261015)
        return prepare_batch(data,rng,mode,corruption_profile=profile,return_diagnostics=True)
    checks={}; cases={}
    for mode in ('mixed','clean','reanchor','noisy','missing','strong','single','burst3','burst4','position_burst4','scattered3','ramp4','history_only'):
        inputs,d=sample(batch,mode); history,raw,candidate,kwargs,target,future=inputs
        assert all(torch.isfinite(v).all() for v in (history,raw,candidate,target,future))
        actual=torch.cat((d['history_world'],d['raw_world'][:,None]),1)
        errors=(actual[...,:3]-batch['past_current'][...,:3]).norm(dim=-1)
        assert torch.allclose(errors,d['sampled_amplitudes'][...,0],atol=3e-6),mode
        angles=g.geodesic_angle(g.state_to_rotmat(actual),g.state_to_rotmat(batch['past_current']))
        assert torch.allclose(angles,d['sampled_amplitudes'][...,1:],atol=.002),mode
        outside=~d['event_groups']
        assert bool((d['sampled_amplitudes'][outside]<=.018).all())  # max near rotation 1deg
        assert not bool((kwargs['quality'][kwargs['availability']]>=0).any())
        assert kwargs['history_quality'] is None and not bool(kwargs['beta_valid'].any())
        if mode=='clean': assert errors.max()<3e-6 and torch.allclose(candidate,target,atol=3e-6)
        if mode=='reanchor': assert not bool(d['episode_mask'][:,-1].any()) and torch.allclose(candidate,target,atol=.0011)
        if mode=='position_burst4':
            assert bool((d['event_groups'][...,1:]==False).all())
            selected=d['episode_mask']
            vectors=actual[...,:3]-batch['past_current'][...,:3]
            assert torch.allclose(vectors[selected],(d['directions'][...,0,:]*d['sampled_amplitudes'][...,0,None])[selected],atol=3e-6)
        # Changing future values must not change any model input or current label.
        changed={**batch,'future':g.state_apply_delta(batch['future'],torch.randn_like(batch['future'][...,:51])*.01)}
        other,_=sample(changed,mode)
        for a,c in zip(inputs[:3]+(target,),other[:3]+(other[4],)): assert torch.equal(a,c)
        for key,value in kwargs.items():
            if torch.is_tensor(value): assert torch.equal(value,other[3][key]),key
        if mode=='missing':
            absent=~d['availability']; delta=torch.randn(*absent.shape,3,device=device)*.01
            delta*=absent[...,None]
            changed={**batch,'past_current':g.state_apply_delta(batch['past_current'],delta.flatten(-2))}
            other,_=sample(changed,mode)
            for a,c in zip(inputs[:3],other[:3]): assert torch.allclose(a,c,atol=5e-6)
            for key in ('history_availability','availability'): assert torch.equal(kwargs[key],other[3][key])
        cases[mode]={'cases':torch.bincount(d['case_id'],minlength=5).cpu().tolist(),
            'patterns':torch.bincount(d['pattern_id'],minlength=4).cpu().tolist(),
            'current_event_age_histogram':torch.bincount(d['event_age_at_current'].clamp_min(0)).cpu().tolist(),
            'history_missing_groups':int((~kwargs['history_availability']).sum()),
            'input_wrist_mm':float((candidate[:,:3]-target[:,:3]).norm(dim=-1).mean()*1000)}
    checks['norm_angle_bounds_future_isolation_unknown_quality_causal_missing_fill']=True
    # A real-size epoch enumerates every index once; changed epoch reshuffles.
    samplers=[CoverageEpochSampler(train,replicas=2,rank=r,seed=17,batch_size=512) for r in range(2)]
    ranks=[list(s) for s in samplers]; all_ids=ranks[0]+ranks[1]
    assert len(all_ids)==len(train) and len(set(all_ids))==len(train)
    assert not set(ranks[0]).intersection(ranks[1])
    schedule={'windows':len(train),'by_source':train.windows_by_dataset,
        'updates_per_epoch':(len(ranks[0])+511)//512,'last_per_gpu_batch':len(ranks[0])%512}
    for r,s in enumerate(samplers):
        s.batch_cursor=3; saved=s.state_dict()
        restored=CoverageEpochSampler(train,replicas=2,rank=r,seed=17,batch_size=512)
        restored.load_state_dict(saved); assert list(restored)==ranks[r][1536:]
        restored.set_epoch(1); restored.batch_cursor=0; assert list(restored)!=ranks[r]
    # Exercise partial final batch without loading any real data.
    short=[CoverageEpochSampler(range(1030),replicas=2,rank=r,seed=17,batch_size=512) for r in range(2)]
    assert len(list(short[0]))==515 and len(set(list(short[0])+list(short[1])))==1030
    checks['full_epoch_coverage_disjoint_ranks_partial_batch_and_resume_cursor']=True
    old=TWM(TWMConfig(d_model=16,nhead=4,num_layers=1,dropout=0,forecast_horizon=0)).to(device)
    new=TWM(TWMConfig(d_model=16,nhead=4,num_layers=1,dropout=0,forecast_horizon=0,position_feature_scale=.05)).to(device)
    mask=torch.ones(1024,17,device=device,dtype=torch.bool)
    states=batch['past_current'][:,-1]
    old_tokens,old_clean=old._state_tokens(states,mask); tokens,clean=new._state_tokens(states,mask)
    assert torch.equal(clean,old_clean) and torch.allclose(tokens[:,0,:3],old_tokens[:,0,:3]*20)
    assert torch.equal(tokens[...,3:],old_tokens[...,3:])
    checks['encoder_only_position_scale_preserves_public_metre_state_and_legacy_default']=True
    result={'checks':checks,'cases':cases,'epoch_schedule':schedule,
        'scope':'1024 draws per mode using four real train reference windows; sampler semantics, not trained accuracy'}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
