#!/usr/bin/env python3
"""ds login-host launcher/status for the authorized persistent GPU Pod.

Does not create/delete Kubernetes jobs, alter the source data or auto-restart
completed training. Training runs inside the Pod, with persistent checkpoints.
"""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('start','status'))
    parser.add_argument('--pod',default='jchen57-debug-mpd5p')
    parser.add_argument('--namespace',default='jchen57')
    parser.add_argument('--code',default='/sciclone/home/jchen57/project/Hand_Reconstruction')
    parser.add_argument('--python',default='/sciclone/data10/jchen57/Hand_Reconstruction/runtime/twm_mamba_torch210_v1/bin/python')
    parser.add_argument('--manifest',default='/sciclone/data10/jchen57/Hand_Reconstruction/prepared/oakink2_99d_pilot_v1/manifest.json')
    parser.add_argument('--config',default='config/twm_oakink2_structured_mamba_pilot.json')
    parser.add_argument('--output',default='/sciclone/data10/jchen57/Hand_Reconstruction/runs/twm_oakink2_structured_mamba_20261014')
    parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    config=vars(args)
    config['code_version']=subprocess.check_output(['git','-C',args.code,'rev-parse','HEAD'],text=True).strip()
    program='CONFIG='+repr(config)+'\n'+r'''
import json,os,subprocess
from pathlib import Path
from datetime import datetime,timezone
out=Path(CONFIG['output'])
process_path=out/'process.json'
launch=json.loads(process_path.read_text()) if process_path.exists() else {}
pid=launch.get('pid')
script=str(Path(CONFIG['code'])/'scripts/train_twm.py')
proc=Path(f'/proc/{pid}/cmdline') if pid else None
alive=bool(proc and proc.exists() and script.encode() in proc.read_bytes().split(b'\0')
           and str(out).encode() in proc.read_bytes().split(b'\0'))
if CONFIG['action']=='start' and not alive:
    if not Path(CONFIG['manifest']).is_file(): raise RuntimeError('Prepared manifest missing')
    if (out/'run_metadata.json').exists() and not CONFIG['resume']:
        raise RuntimeError('Existing run: use explicit --resume or a fresh output path')
    out.mkdir(parents=True,exist_ok=True)
    cmd=[CONFIG['python'],'-u',script,'--manifest',CONFIG['manifest'],
         '--config',str(Path(CONFIG['code'])/CONFIG['config']),
         '--output',str(out),'--code-version',CONFIG['code_version']]
    if CONFIG['resume']:
        checkpoint=out/'latest.pt'
        if not checkpoint.is_file(): raise RuntimeError('No latest.pt to resume')
        cmd+=['--resume',str(checkpoint)]
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(CONFIG['gpu']),OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='4',OMP_NUM_THREADS='4')
    with (out/'train.log').open('ab',buffering=0) as log:
        p=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                           cwd=CONFIG['code'],env=env,start_new_session=True)
    launch={'pid':p.pid,'pod':CONFIG['pod'],'gpu':CONFIG['gpu'],'command':cmd,
            'code_version':CONFIG['code_version'],'started_utc':datetime.now(timezone.utc).isoformat()}
    process_path.write_text(json.dumps(launch,indent=2)+'\n')
    alive=True
def read(name):
    p=out/name
    return json.loads(p.read_text()) if p.exists() else None
status=read('status.json')
print(json.dumps({'process_alive':alive,'pid':launch.get('pid'),'output':str(out),
                  'launch':launch,'status':status,'validation':read('latest_validation.json'),
                  'checkpoints':{name:(out/name).stat().st_size for name in ('latest.pt','best.pt') if (out/name).exists()},
                  'log_tail':(out/'train.log').read_text()[-3500:] if (out/'train.log').exists() else None}),flush=True)
'''
    prefix=['kubectl','exec','-i',args.pod,'-n',args.namespace,'--',args.python,'-']
    subprocess.run(prefix,input=program,text=True,check=True)


if __name__=='__main__': main()
