"""Window-local temporal error events; masks/severities are diagnostic only.

Unlike v2, a history is not assigned one severity for its entire duration.
Events share direction/peak over a short interval, with clean/near boundaries.
This is synthetic augmentation, not a calibrated HaWoR error distribution.
"""
from __future__ import annotations
import math
import torch
from handloop import geometry as g
from .corruption import CASE_NAMES, LEVEL_NAMES

PATTERN_NAMES = ('jitter', 'plateau', 'ramp', 'scattered')
SPATIAL_NAMES = ('translation_only', 'wrist_pose', 'fingers', 'whole_hand')


def causal_fill_history(states, available):
    """Carry forward each group's most recent observed past value.

    Caller provides a fully observed initial seed. No future values or
    uncertainty labels are consulted. Also used by validation baselines.
    """
    if not bool(available[:,0].all()): raise ValueError('Observed initial seed required')
    b=states.shape[0]
    filled=[]; previous=states[:,0]
    for t in range(states.shape[1]):
        now=states[:,t]; mask=available[:,t]
        previous=torch.cat((torch.where(mask[:,:1],now[:,:3],previous[:,:3]),
            torch.where(mask[:,1:,None],now[:,3:].reshape(b,16,6),previous[:,3:].reshape(b,16,6)).flatten(1)),-1)
        filled.append(previous)
    return torch.stack(filled,1)


def sample_temporal_batch(batch, generator, mode='mixed', *, condition_on_beta=False,
                          profile, return_diagnostics=False):
    clean = batch['past_current'].float()
    b, length, width = clean.shape
    h, device = length-1, clean.device
    if width != 99 or h < 10 or not bool(batch['past_current_gt_mask'].all()):
        raise ValueError('Temporal corruption needs full-reference [B,H+1,99], H>=10')
    dt = batch['dt'].float()
    if dt.shape != (b,length) or not bool(torch.isfinite(dt).all() and (dt>0).all()):
        raise ValueError('Positive physical dt required')
    cases={entry['name']:entry['probability'] for entry in profile['window_cases']}
    if set(cases)!=set(CASE_NAMES): raise ValueError('Expected five explicit window cases')
    probabilities=[cases[name] for name in CASE_NAMES]
    range_rows=[]
    for level in LEVEL_NAMES:
        row=[]
        for key,scale,ceiling in (('translation_norm_mm',.001,math.inf),
                                  ('root_angle_deg',math.pi/180,180),
                                  ('finger_angle_deg',math.pi/180,180)):
            bounds=profile['levels'][level][key]
            if len(bounds)!=2 or not all(math.isfinite(x) for x in bounds) or not 0<=bounds[0]<=bounds[1]<=ceiling:
                raise ValueError(f'Invalid amplitude bounds: {level}.{key}')
            row.append([x*scale for x in bounds])
        range_rows.append(row[:2]+[row[2]]*15)
    ranges=torch.tensor(range_rows,device=device,dtype=torch.float32)
    spec = profile['events']
    for key in ('jitter_frame_probability','medium_probability','near_background_probability'):
        if not 0<=spec[key]<=1: raise ValueError(f'Invalid {key}')
    def rand(*shape): return torch.rand(*shape, device=device, generator=generator)
    def cat(prob):
        p = torch.tensor(prob,device=device,dtype=torch.float32)
        if not bool((p>=0).all()) or abs(float(p.sum())-1)>1e-6: raise ValueError('Invalid probabilities')
        return (rand(b,1)>=p.cumsum(0)[None]).sum(-1).clamp_max(len(prob)-1)
    case = cat(probabilities)
    forced = {'clean':0,'reanchor':1,'noisy':2,'missing':3,'strong':4,
              'history_only':2,'single':2,'burst3':2,'burst4':2,
              'position_burst4':2,'scattered3':2,'ramp4':2}
    if mode != 'mixed':
        if mode not in forced: raise ValueError(f'Unknown temporal corruption mode: {mode}')
        case.fill_(forced[mode])
    pattern = cat(spec['pattern_probabilities'])
    duration = cat(spec['duration_probabilities'])+1
    spatial = cat(spec['spatial_probabilities'])
    # Present all causal phases: onset, middle/last bad frame, recovered current.
    # Reanchor explicitly ends in the recent history; ordinary current is bad.
    age = (rand(b)*duration).long()
    recovery_gap = 1+(rand(b)*4).long()
    end = torch.where(case==1,h-recovery_gap,h+duration-1-age)
    start = end-duration+1
    if mode in ('single','burst3','burst4','position_burst4','scattered3','ramp4'):
        duration.fill_(1 if mode=='single' else 3 if mode in ('burst3','scattered3') else 4)
        age=(rand(b)*duration).long(); start=h-age; end=start+duration-1
        pattern.fill_(3 if mode=='scattered3' else 2 if mode=='ramp4' else 1)
        if mode=='position_burst4': spatial.fill_(0)
    frames=torch.arange(length,device=device)[None]
    event=(frames>=start[:,None]) & (frames<=end[:,None])
    # Scattered events have 1..4 active frames in the recent ten, including
    # current for ordinary/missing/strong; recovery excludes current entirely.
    eligible_end=torch.where(case==1,h-recovery_gap,h)
    recent=frames-(eligible_end[:,None]-9)
    # Reserve end separately, so the requested count is exact.
    end_mask=frames==eligible_end[:,None]
    ranks_end=rand(b,10); ranks_end[:,-1]=2
    ranks=ranks_end.argsort(-1).argsort(-1)
    scattered=((recent>=0)&(recent<10)&(ranks.gather(1,recent.clamp(0,9))<(duration-1)[:,None]))|end_mask
    jitter=(rand(b,length)<spec['jitter_frame_probability']) & (frames>=h-9)
    jitter=torch.where(case[:,None]==1,jitter & (frames<h),jitter| (frames==h))
    event=torch.where(pattern[:,None]==0,jitter,torch.where(pattern[:,None]==3,scattered,event))
    event &= (case[:,None]!=0) & (frames>0)
    group=torch.arange(17,device=device)[None]
    finger_ranks=rand(b,5).argsort(-1).argsort(-1)
    finger_count=1+(rand(b)<.5).long()
    fingers=(finger_ranks.gather(1,((group-2)//3).clamp(0,4).expand(b,17))<finger_count[:,None])&(group>=2)
    affected=torch.where(spatial[:,None]==0,group==0,
               torch.where(spatial[:,None]==1,group<2,
               torch.where(spatial[:,None]==2,fingers,torch.ones(b,17,device=device,dtype=torch.bool))))
    # No all-hand strong case; strong remains a rare local wrist/finger failure.
    affected=torch.where((case==4)[:,None],torch.where((spatial<2)[:,None],group<2,fingers),affected)
    severity=torch.where(rand(b)<spec['medium_probability'],3,2).long()
    severity=torch.where(case==4,4,severity)
    group_idx=torch.arange(17,device=device)[None]
    bounds=ranges[severity[:,None],group_idx]
    peak=bounds[...,0]+rand(b,17)*(bounds[...,1]-bounds[...,0])
    directions=torch.randn(b,17,3,device=device,generator=generator)
    directions=directions/directions.norm(dim=-1,keepdim=True).clamp_min(1e-12)
    shared=directions[:,None].expand(b,length,17,3)
    independent=torch.randn(b,length,17,3,device=device,generator=generator)
    independent=independent/independent.norm(dim=-1,keepdim=True).clamp_min(1e-12)
    directions=torch.where((pattern==0)[:,None,None,None],independent,shared)
    ramp=((frames-start[:,None]+1)/duration[:,None]).clamp(0,1)
    factor=torch.where((pattern==2)[:,None],ramp,torch.ones_like(ramp))
    background=spec['near_background_probability']>rand(b)
    bg=ranges[1,:,1][None,None]*rand(b,length,17)*background[:,None,None]
    bg=torch.where((case==0)[:,None,None],0.,bg)
    if mode=='position_burst4': bg.zero_()
    event_groups=event[:,:,None]&affected[:,None]
    amplitude=torch.where(event_groups,peak[:,None]*factor[:,:,None],bg)
    corrupted=g.state_apply_delta(clean,(directions*amplitude[...,None]).flatten(-2))
    available=torch.ones(b,length,17,device=device,dtype=torch.bool)
    available &= ~((case==3)[:,None,None]&event_groups)
    if mode=='history_only': available[:,-1]=False
    # All groups have an observed initial seed. Causal fill never reads a
    # hidden/missing reference, including when selecting the wrist anchor.
    filled=causal_fill_history(corrupted,available)
    anchor=filled[:,h-1]; ra=g.state_to_rotmat(anchor)[:,0]
    def to_anchor(states):
        extra=states.ndim-2
        rotation=ra.reshape(b,*([1]*extra),3,3)
        pa=anchor[:,:3].reshape(b,*([1]*extra),3)
        p=(rotation.transpose(-1,-2)@(states[...,:3]-pa)[...,None]).squeeze(-1)
        r=g.state_to_rotmat(states).clone(); r[...,0,:,:]=rotation.transpose(-1,-2)@r[...,0,:,:]
        return g.state_from_rotmat(p,r)
    observations=to_anchor(corrupted)
    # Missing history/raw values are placeholders; only candidate is filled.
    observations[...,:3]=torch.where(available[...,:1],observations[...,:3],0.)
    observations[...,3:]=torch.where(available[...,1:,None],observations[...,3:].reshape(b,length,16,6),g.identity_rotation_6d(device=device)).flatten(-2)
    history,raw=observations[:,:h],observations[:,-1]
    candidate=to_anchor(filled[:,-1])
    kwargs={'history_valid':torch.ones(b,h,device=device,dtype=torch.bool),
        'history_availability':available[:,:h],'availability':available[:,-1],
        'candidate_availability':torch.ones_like(available[:,-1]),
        'dt':dt,'beta':batch['beta'],'handedness':batch['handedness'],
        'quality':torch.where(available[:,-1],-1.,0.),'history_quality':None,
        'forecast_dt':batch['forecast_dt'],
        'beta_valid':batch['beta_valid'] if condition_on_beta else torch.zeros(b,device=device,dtype=torch.bool)}
    prepared=history,raw,candidate,kwargs,to_anchor(clean[:,-1]),to_anchor(batch['future'].float())
    diagnostics={'case_id':case,'case_names':CASE_NAMES,'pattern_id':pattern,'pattern_names':PATTERN_NAMES,
        'spatial_id':spatial,'spatial_names':SPATIAL_NAMES,'episode_mask':event,
        'event_groups':event_groups,'duration_frames':duration,'sampled_amplitudes':amplitude,
        'peak_amplitudes':peak,'directions':directions,'history_world':corrupted[:,:h],
        'raw_world':corrupted[:,-1],'anchor_world':anchor,'availability':available,
        'event_age_at_current':h-start,'recovery_gap':recovery_gap,'severity':severity}
    return (prepared,diagnostics) if return_diagnostics else prepared
