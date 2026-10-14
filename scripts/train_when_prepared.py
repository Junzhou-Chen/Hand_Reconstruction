#!/usr/bin/env python3
"""One-shot remote coordinator: validate four complete pools, then launch TWM.

No recurring scheduler, automatic restart, Job creation, or partial-data run.
Preprocessing is launched separately. This process keeps its status on the
persistent filesystem and starts training exactly once after all four pools
pass the collection inspection. It never resumes an existing training run.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

SOURCES=('oakink2','hot3d','umetrack','dexycb')


def utc(): return datetime.now(timezone.utc).isoformat()


def atomic_json(path,value):
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in SOURCES: parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--collection',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--status-dir',type=Path,required=True)
    parser.add_argument('--code-version',required=True)
    parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--max-wait-seconds',type=float,default=21600)
    args=parser.parse_args()
    if not Path('/sciclone/home/jchen57').is_dir():
        parser.error('Run in the existing remote compute Pod')
    code=Path(__file__).resolve().parents[1]
    if (args.output/'run_metadata.json').exists() or (args.output/'process.json').exists():
        raise FileExistsError('Choose a fresh training output; this coordinator never resumes/restarts')
    args.status_dir.mkdir(parents=True,exist_ok=True)
    path=args.status_dir/'status.json'
    started=time.monotonic()
    protocol={'code_version':args.code_version,'config_sha256':hashlib.sha256(args.config.read_bytes()).hexdigest(),
              'trainer_sha256':hashlib.sha256((code/'scripts/train_twm.py').read_bytes()).hexdigest()}
    try:
        while True:
            states={}
            for name in SOURCES:
                manifest=getattr(args,name)
                if manifest.is_file():
                    value=json.loads(manifest.read_text())
                    states[name]=value.get('phase','missing phase')
                    if states[name]=='complete' and value.get('dataset')!=name:
                        raise ValueError(f'Unexpected dataset identity: {name}')
                else:
                    status=manifest.parent/'preparation_status.json'
                    value=json.loads(status.read_text()) if status.is_file() else {}
                    states[name]=value.get('phase','not yet published')
                    if states[name]=='complete':
                        raise RuntimeError(f'{name}: complete receipt exists without its prepared manifest')
                if states[name]=='failed': raise RuntimeError(f'{name}: preprocessing failed')
            atomic_json(path,{'utc':utc(),'phase':'waiting_for_complete_prepared_data','sources':states,
                              'elapsed_seconds':round(time.monotonic()-started,1),**protocol})
            if all(value=='complete' for value in states.values()): break
            if time.monotonic()-started>args.max_wait_seconds:
                raise TimeoutError('Prepared data wait exceeded the explicit limit; no training launched')
            time.sleep(30)
        if (hashlib.sha256(args.config.read_bytes()).hexdigest()!=protocol['config_sha256'] or
                hashlib.sha256((code/'scripts/train_twm.py').read_bytes()).hexdigest()!=protocol['trainer_sha256']):
            raise RuntimeError('Trainer/config changed while waiting; inspect and relaunch with an explicit version')
        command=[sys.executable,str(code/'scripts/prepare_twm_collection.py')]
        for name in SOURCES:command+=['--'+name,str(getattr(args,name))]
        command+=['--output',str(args.collection),'--code-version',args.code_version]
        atomic_json(path,{'utc':utc(),'phase':'inspecting_complete_collection',**protocol})
        subprocess.run(command,check=True)
        args.output.mkdir(parents=True,exist_ok=True)
        command=[sys.executable,'-u',str(code/'scripts/train_twm.py'),'--manifest',str(args.collection),
                 '--config',str(args.config),'--output',str(args.output),'--code-version',args.code_version]
        environment=os.environ.copy()
        environment.update(CUDA_VISIBLE_DEVICES=str(args.gpu),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
        with (args.output/'train.log').open('ab',buffering=0) as log:
            process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                                     cwd=code,env=environment,start_new_session=True)
        launch={'pid':process.pid,'command':command,'code_version':args.code_version,'gpu':args.gpu,
                'pod':'jchen57-debug-mpd5p','started_utc':utc(),'coordinator':'one-shot train_when_prepared.py'}
        atomic_json(args.output/'process.json',launch)
        atomic_json(path,{'utc':utc(),'phase':'training_launched','launch':launch,**protocol})
        print(json.dumps(launch),flush=True)
    except BaseException as error:
        atomic_json(path,{'utc':utc(),'phase':'failed','error':repr(error),**protocol})
        raise


if __name__=='__main__':main()
