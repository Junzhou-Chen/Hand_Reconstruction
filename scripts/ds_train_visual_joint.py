#!/usr/bin/env python3
"""Run inside the ds login host; all RGB/model work executes in the GPU Pod."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('start', 'status', 'stop'))
    parser.add_argument('--pod', default='jchen57-fineturn-tl4zr')
    parser.add_argument('--namespace', default='jchen57')
    base = '/sciclone/data10/jchen57/Hand_Reconstruction'
    parser.add_argument('--code', default=base + '/code/visual_feedback_joint_v2_20261016')
    parser.add_argument('--python', default=base + '/runtime/twm_mamba_torch210_v1/bin/python')
    parser.add_argument('--source-manifest', default=base + '/prepared/visual_feedback_dex_joint_sources_v2_20261016/source_manifest.json')
    parser.add_argument('--motion-manifest', default=base + '/prepared/twm_four_sources_99d_v1/manifest.json')
    parser.add_argument('--twm-checkpoint', default=base + '/model_assets/twm_d37_final_20261016/best.pt')
    parser.add_argument('--hawor-root', default=base + '/model_assets/hawor_v1/source')
    parser.add_argument('--hawor-checkpoint', default=base + '/model_assets/hawor_v1/weights/hawor.ckpt')
    parser.add_argument('--hawor-config', default=base + '/model_assets/hawor_v1/weights/model_config.yaml')
    parser.add_argument('--mano-models', default=base + '/model_assets/mano_v1_2/prepared_v1/derived')
    parser.add_argument('--config', default='config/visual_joint_v2.json')
    parser.add_argument('--output', default=base + '/runs/visual_feedback_joint_v2_20261016')
    parser.add_argument('--gpus', default='0,1')
    parser.add_argument('--wandb-key-file', default=base + '/private/wandb_api_key')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after-updates', type=int)
    parser.add_argument('--tracking-mode', choices=('online', 'offline', 'disabled'))
    args = parser.parse_args()
    config = vars(args)
    config['code_version'] = subprocess.check_output(['git', '-C', args.code, 'rev-parse', 'HEAD'], text=True).strip()
    program = 'CONFIG=' + repr(config) + '\n' + r'''
from pathlib import Path
import datetime,json,os,subprocess
out=Path(CONFIG['output']); receipt=out/'process.json'
launch=json.loads(receipt.read_text()) if receipt.exists() else {}
pid=launch.get('pid'); proc=Path(f'/proc/{pid}/cmdline') if pid else None
script=str(Path(CONFIG['code'])/'scripts/train_visual_joint.py')
alive=bool(proc and proc.exists() and script.encode() in proc.read_bytes().split(b'\0')
           and str(out).encode() in proc.read_bytes().split(b'\0'))
if CONFIG['action']=='start' and not alive:
    if (out/'run_metadata.json').exists() and not CONFIG['resume']:
        raise RuntimeError('Existing run requires explicit resume or a fresh output')
    if launch and not CONFIG['resume']:
        raise RuntimeError('Previous launch receipt exists; inspect it and choose fresh output')
    cfg=Path(CONFIG['config']); cfg=cfg if cfg.is_absolute() else Path(CONFIG['code'])/cfg
    protocol=json.loads(cfg.read_text()); devices=CONFIG['gpus'].split(',')
    if len(devices)!=protocol['world_size'] or len(set(devices))!=len(devices):
        raise ValueError('Explicit GPU list must match protocol world size')
    required=['source_manifest','motion_manifest','hawor_checkpoint','hawor_config']
    if not CONFIG['resume']: required.append('twm_checkpoint')
    for key in required:
        if not Path(CONFIG[key]).is_file(): raise FileNotFoundError(key)
    out.mkdir(parents=True,exist_ok=True)
    if (out/'STOP').exists():
        if not CONFIG['resume']: raise RuntimeError('Fresh run has STOP')
        (out/'STOP').rename(out/('STOP.resumed-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S')))
    command=[CONFIG['python'],'-m','torch.distributed.run','--standalone','--nnodes=1',
             '--nproc-per-node='+str(len(devices)),script,'--config',str(cfg),'--output',str(out),
             '--code-version',CONFIG['code_version']]
    for key in ('source_manifest','motion_manifest','twm_checkpoint','hawor_root','hawor_checkpoint','hawor_config','mano_models'):
        command+=['--'+key.replace('_','-'),CONFIG[key]]
    if CONFIG['resume']: command+=['--resume',str(out/'latest.pt')]
    if CONFIG['stop_after_updates']: command+=['--stop-after-updates',str(CONFIG['stop_after_updates'])]
    if CONFIG['tracking_mode']: command+=['--tracking-mode',CONFIG['tracking_mode']]
    env=os.environ.copy(); threads=str(protocol.get('cpu_threads',2))
    root=Path(CONFIG['hawor_root']).parents[2]
    env.update(CUDA_VISIBLE_DEVICES=CONFIG['gpus'],OMP_NUM_THREADS=threads,MKL_NUM_THREADS=threads,
        OPENBLAS_NUM_THREADS='1',PYTHONPATH=str(Path(CONFIG['code'])/'src')+':'+str(root/'runtime/hawor_extra/site-packages')+':'+CONFIG['hawor_root'],
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    secret=Path(CONFIG['wandb_key_file'])
    if protocol.get('tracking',{}).get('enabled') and CONFIG.get('tracking_mode')!='disabled':
        if not secret.is_file() or secret.stat().st_mode & 0o077:
            raise RuntimeError('W&B private 600 credential file required')
        env['WANDB_API_KEY']=secret.read_text().strip()
    with (out/'train.log').open('ab',buffering=0) as log:
        p=subprocess.Popen(command,cwd=CONFIG['code'],env=env,stdin=subprocess.DEVNULL,
                           stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    launch={'pid':p.pid,'command':command,'code_version':CONFIG['code_version'],'gpus':devices,
            'pod':CONFIG['pod'],'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    receipt.write_text(json.dumps(launch,indent=2)+'\n');alive=True
    previous=json.loads((out/'status.json').read_text()) if (out/'status.json').exists() else {}
    if previous:
        (out/'previous_execution_status.json').write_text(json.dumps(previous,indent=2)+'\n')
    (out/'status.json').write_text(json.dumps({'utc':launch['started_utc'],'phase':'initializing',
        'step':previous.get('step',0),'target_steps':protocol['steps'],'resume':CONFIG['resume'],
        'wandb_url':previous.get('wandb_url')},indent=2)+'\n')
elif CONFIG['action']=='stop':
    if not alive: raise RuntimeError('No verified training launcher to stop')
    (out/'STOP').write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                      'reason':'explicit joint training stop request'})+'\n')
status=json.loads((out/'status.json').read_text()) if (out/'status.json').exists() else None
print(json.dumps({'alive':alive,'output':str(out),'launch':launch,'status':status,
  'checkpoint_bytes':{name:(out/name).stat().st_size for name in ('latest.pt','best.pt') if (out/name).exists()},
  'log_tail':(out/'train.log').read_text()[-3000:] if (out/'train.log').exists() else None}),flush=True)
'''
    subprocess.run(['kubectl', 'exec', '-i', args.pod, '-n', args.namespace, '--', args.python, '-'],
                   input=program, text=True, check=True)


if __name__ == '__main__':
    main()
