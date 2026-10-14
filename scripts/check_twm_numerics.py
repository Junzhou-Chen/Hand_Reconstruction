#!/usr/bin/env python3
"""Remote-only numerical checks for 99D geometry and model gradients."""
from pathlib import Path
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
import torch
from handloop import geometry as g
from handloop.models import TWM, TWMConfig


def main():
    if not Path("/sciclone/home/jchen57").is_dir():
        raise RuntimeError("Run numerical/model checks on the remote Pod")
    torch.set_num_threads(2)
    torch.manual_seed(17)
    device="cuda:0"
    q=torch.randn(128,4,device=device)
    r=g.quaternion_to_matrix(q)
    back=g.rotation_6d_to_matrix(g.matrix_to_rotation_6d(r))
    assert torch.max(torch.abs(r-back)) < 2e-6
    assert torch.max(torch.abs(g.quaternion_to_matrix(-q)-r)) < 1e-6
    bad=torch.tensor([[0.,0,0,0,0,0],[1,0,0,2,0,0]],device=device)
    fallback=g.rotation_6d_to_matrix(bad)
    assert torch.max(torch.abs(fallback.transpose(-1,-2)@fallback-torch.eye(3,device=device))) < 1e-6
    aa=torch.tensor([[0.,0,0],[torch.pi-1e-5,0,0]],device=device,requires_grad=True)
    result=g.axis_angle_to_matrix(aa)
    result.square().sum().backward()
    assert torch.isfinite(aa.grad).all()
    identity=g.identity_state(2,device=device)
    assert torch.allclose(g.state_apply_delta(identity,torch.zeros(2,51,device=device)),identity)
    model=TWM(TWMConfig(d_model=32,num_layers=2,nhead=4,dim_feedforward=96,dropout=0)).to(device)
    history=g.identity_state(2,4,device=device)
    kwargs=dict(history_valid=torch.ones(2,4,device=device,dtype=torch.bool),
                history_availability=torch.ones(2,4,17,device=device,dtype=torch.bool),
                availability=torch.ones(2,17,device=device,dtype=torch.bool),
                candidate_availability=torch.ones(2,17,device=device,dtype=torch.bool),
                dt=torch.full((2,5),1/30,device=device),beta=torch.zeros(2,10,device=device),
                handedness=torch.tensor([0,1],device=device),quality=None,
                forecast_dt=torch.full((2,1),0.1,device=device))
    output=model(history,identity,identity,**kwargs)
    target=g.state_apply_delta(identity,torch.randn(2,51,device=device)*0.02)
    loss=(output['state']-target).square().mean()+(output['forecast']-target[:,None]).square().mean()
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    model.eval()
    with torch.no_grad():
        first=model(history,identity,identity,**kwargs)['state']
        kwargs['forecast_dt']*=2
        second=model(history,identity,identity,**kwargs)['state']
        assert torch.equal(first,second),"forecast horizon contaminated current restoration"
        kwargs['history_valid'].zero_()
        padded=model(history,identity,identity,**kwargs)['state']
        assert torch.isfinite(padded).all()
    print(json.dumps({'passed':True,'device':device,'torch':torch.__version__,
                      'rotation_roundtrip_max':float((r-back).abs().max()),
                      'gradient_loss':float(loss.detach()),'tests':['quaternion_sign','6d_roundtrip','degenerate_6d','zero_pi_gradients','state_zero_delta','model_backward','forecast_no_current_leak','padded_history']}))


if __name__=='__main__': main()
