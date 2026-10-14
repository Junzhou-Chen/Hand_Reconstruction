#!/usr/bin/env python3
"""Remote Mamba fused CUDA forward/backward checks; no training is started."""
from pathlib import Path
import sys,json,datetime,importlib.metadata,traceback
import argparse
if not Path('/sciclone/home/jchen57').is_dir():
    raise RuntimeError('Run GPU checks on the remote Pod only')
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,default=Path('/sciclone/data10/jchen57/Hand_Reconstruction/runtime/twm_mamba_torch210_v1/gpu_smoke_receipt.json'))
args=parser.parse_args()
args.output.parent.mkdir(parents=True,exist_ok=True)
result={'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'python':sys.executable,'status':'running','checks':[]}
try:
    import torch,mamba_ssm,causal_conv1d,selective_scan_cuda
    from mamba_ssm.modules import mamba_simple
    result['versions']={n:importlib.metadata.version(n) for n in ('torch','triton','mamba-ssm','causal-conv1d','numpy')}
    result['torch_cuda']=torch.version.cuda
    result['cxx11_abi']=torch._C._GLIBCXX_USE_CXX11_ABI
    result['arch_list']=torch.cuda.get_arch_list()
    result['modules']={'torch':torch.__file__,'mamba_ssm':mamba_ssm.__file__,'causal_conv1d':causal_conv1d.__file__,'selective_scan_cuda':selective_scan_cuda.__file__}
    assert torch.__version__.startswith('2.10.0+cu130'),torch.__version__
    assert mamba_simple.causal_conv1d_fn is not None
    original=mamba_simple.mamba_inner_fn
    calls=[]
    def tracked(*args,**kwargs):
        calls.append(1)
        return original(*args,**kwargs)
    mamba_simple.mamba_inner_fn=tracked
    for gpu in [0]:
        torch.manual_seed(20261014)
        model=mamba_simple.Mamba(d_model=64,d_state=64,d_conv=4,expand=2,use_fast_path=True).to('cuda:'+str(gpu))
        x=torch.randn(2,8,64,device='cuda:'+str(gpu),requires_grad=True)
        before=len(calls)
        y=model(x); loss=y.float().square().mean();loss.backward();torch.cuda.synchronize(gpu)
        assert len(calls)>before,'fused path was not invoked'
        assert y.shape==x.shape and torch.isfinite(y).all()
        assert x.grad is not None and torch.isfinite(x.grad).all()
        grads=[v.grad for v in model.parameters() if v.requires_grad]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        with torch.no_grad():
            fast=model(x.detach());model.use_fast_path=False;slow=model(x.detach());model.use_fast_path=True
            maxdiff=float((fast-slow).abs().max().item())
            assert torch.allclose(fast,slow,rtol=2e-4,atol=3e-5),maxdiff
        result['checks'].append({'gpu':gpu,'name':torch.cuda.get_device_name(gpu),'capability':torch.cuda.get_device_capability(gpu),'dtype':'float32','shape':list(y.shape),'loss':float(loss.item()),'fused_calls':len(calls)-before,'fast_slow_max_abs_diff':maxdiff,'forward_backward_finite':True})
        model.zero_grad(set_to_none=True);x.grad=None
        before=len(calls)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            y=model(x);loss=y.float().square().mean()
        loss.backward();torch.cuda.synchronize(gpu)
        assert len(calls)>before and torch.isfinite(y).all() and torch.isfinite(x.grad).all()
        assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters() if v.requires_grad)
        result['checks'].append({'gpu':gpu,'dtype':'bfloat16_autocast','shape':list(y.shape),'loss':float(loss.item()),'fused_calls':len(calls)-before,'forward_backward_finite':True})
        del model,x,y,fast,slow;torch.cuda.empty_cache()
    result['status']='passed'
except BaseException as e:
    result['status']='failed';result['error']=repr(e);result['traceback']=traceback.format_exc()
finally:
    result['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)
if result['status']!='passed': raise SystemExit(1)
