#!/usr/bin/env python3
"""One-shot ds launcher for the authorized two-GPU probabilistic TWM; cooperative STOP file."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('start','status','stop'))
    parser.add_argument('--pod', default='jchen57-debug-mpd5p')
    parser.add_argument('--namespace', default='jchen57')
    parser.add_argument('--code', default='/sciclone/home/jchen57/project/Hand_Reconstruction')
    parser.add_argument('--python', default='/sciclone/data10/jchen57/Hand_Reconstruction/runtime/twm_mamba_torch210_v1/bin/python')
    parser.add_argument('--manifest', default='/sciclone/data10/jchen57/Hand_Reconstruction/prepared/twm_four_sources_99d_v1/manifest.json')
    parser.add_argument('--config', default='config/twm_four_sources_moe_mixed_v2_b512_10k.json')
    parser.add_argument('--output', default='/sciclone/data10/jchen57/Hand_Reconstruction/runs/twm_four_sources_moe_mixed_v2_b512_10k_20261015')
    parser.add_argument('--gpus', default='0,1')
    parser.add_argument('--wandb-key-file', default='/sciclone/data10/jchen57/Hand_Reconstruction/private/wandb_api_key',
                        help='Private remote file; only passed through the child environment')
    initial=parser.add_mutually_exclusive_group()
    initial.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after-updates', type=int)
    parser.add_argument('--stop-before-utc', help='Execution deadline for this allocation; ISO timestamp or none')
    args=parser.parse_args()
    config=vars(args)
    config['code_version']=subprocess.check_output(['git','-C',args.code,'rev-parse','HEAD'],text=True).strip()
    program='CONFIG='+repr(config)+'\n'+r'''
import json,os,subprocess
from pathlib import Path
from datetime import datetime,timezone
out=Path(CONFIG['output']); receipt=out/'process.json'
launch=json.loads(receipt.read_text()) if receipt.exists() else {}
pid=launch.get('pid'); proc=Path(f'/proc/{pid}/cmdline') if pid else None
script=str(Path(CONFIG['code'])/'scripts/train_probabilistic_twm.py')
alive=bool(proc and proc.exists() and script.encode() in proc.read_bytes().split(b'\0')
           and str(out).encode() in proc.read_bytes().split(b'\0'))
if CONFIG['action']=='start' and not alive:
    if (out/'run_metadata.json').exists() and not CONFIG['resume']:
        raise RuntimeError('Existing run: explicit resume or fresh output required')
    if launch and not CONFIG['resume']:
        raise RuntimeError('Previous launch receipt exists; inspect it and choose a new output')
    cfg=Path(CONFIG['code'])/CONFIG['config']; protocol=json.loads(cfg.read_text())
    devices=CONFIG['gpus'].split(',')
    if len(devices)!=protocol['world_size'] or len(set(devices))!=len(devices):
        raise ValueError('Explicit GPU list must match world_size and have no repeats')
    if not Path(CONFIG['manifest']).is_file(): raise RuntimeError('Prepared collection missing')
    out.mkdir(parents=True,exist_ok=True)
    if (out/'STOP').exists():
        if not CONFIG['resume']: raise RuntimeError('Fresh run has a STOP request')
        (out/'STOP').rename(out/('STOP.resumed-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')))
    command=[CONFIG['python'],'-m','torch.distributed.run','--standalone','--nnodes=1',
             '--nproc-per-node='+str(len(devices)),script,'--manifest',CONFIG['manifest'],
             '--config',str(cfg),'--output',str(out),'--code-version',CONFIG['code_version']]
    if CONFIG['resume']: command+=['--resume',str(out/'latest.pt')]
    if CONFIG['stop_after_updates']: command+=['--stop-after-updates',str(CONFIG['stop_after_updates'])]
    if CONFIG.get('stop_before_utc') is not None: command+=['--stop-before-utc',CONFIG['stop_before_utc']]
    env=os.environ.copy(); threads=str(protocol.get('cpu_threads',4))
    env.update(CUDA_VISIBLE_DEVICES=CONFIG['gpus'],OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS=threads,OMP_NUM_THREADS=threads)
    secret=Path(CONFIG['wandb_key_file']) if CONFIG.get('wandb_key_file') else None
    if secret:
        if not secret.is_file() or secret.stat().st_mode & 0o077:
            raise RuntimeError('W&B credential file must exist with private 600 permissions')
        env['WANDB_API_KEY']=secret.read_text().strip()
    with (out/'train.log').open('ab',buffering=0) as log:
        process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                                 env=env,cwd=CONFIG['code'],start_new_session=True)
    launch={'pid':process.pid,'command':command,'code_version':CONFIG['code_version'],
            'gpus':devices,'pod':CONFIG['pod'],'started_utc':datetime.now(timezone.utc).isoformat()}
    receipt.write_text(json.dumps(launch,indent=2)+'\n'); alive=True
elif CONFIG['action']=='stop':
    if not alive: raise RuntimeError('No verified DDP launcher process to stop')
    # Both ranks see this at a collective boundary and save before exiting.
    (out/'STOP').write_text(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),
                       'reason':'explicit ds_train_twm_ddp stop request'})+'\n')
def read(name):
    path=out/name
    return json.loads(path.read_text()) if path.exists() else None
print(json.dumps({'process_alive':alive,'output':str(out),'launch':launch,'status':read('status.json'),
      'checkpoints':{name:(out/name).stat().st_size for name in ('latest.pt','best.pt') if (out/name).exists()},
      'log_tail':(out/'train.log').read_text()[-2500:] if (out/'train.log').exists() else None}),flush=True)
'''
    subprocess.run(['kubectl','exec','-i',args.pod,'-n',args.namespace,'--',args.python,'-'],
                   input=program,text=True,check=True)


if __name__=='__main__': main()
