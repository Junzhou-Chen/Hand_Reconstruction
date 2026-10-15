#!/usr/bin/env python3
"""Remote DDP integration check: partial batches, epoch boundary, resume, STOP.

Copies 42 contiguous source frames from one train and one val sequence per
source to a private remote test directory. Formal manifests are untouched.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--code-version',required=True)
    args=parser.parse_args()
    if not Path('/sciclone/home/jchen57').is_dir(): parser.error('Remote only')
    if args.output.exists(): parser.error('Use a fresh smoke directory')
    root=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(root/'src'))
    import numpy as np
    from handloop.data.trajectory import _read_manifest,_load_arrays,_storage_keys_and_state_shape
    _,entries=_read_manifest(args.manifest)
    args.output.mkdir(parents=True)
    selections=[]; sequences=[]
    for source in ('oakink2','hot3d','umetrack','dexycb'):
        for split in ('train','val'):
            for entry in entries:
                if entry['dataset']!=source or entry['split']!=split: continue
                keys,shape=_storage_keys_and_state_shape(entry['_path'])
                if shape[0]<42: continue
                times=_load_arrays(entry['_path'],['timestamps'])['timestamps']
                dt=float(entry.get('nominal_dt',np.median(np.diff(times))))
                bounds=np.r_[0,np.flatnonzero(np.diff(times)>=dt*1.5)+1,len(times)]
                starts=[int(a) for a,z in zip(bounds[:-1],bounds[1:]) if z-a>=42]
                if not starts: continue
                begin=starts[0]; arrays=_load_arrays(entry['_path'],list(keys))
                arrays={key:np.array(value[begin:begin+42] if value.ndim and value.shape[0]==shape[0] else value,copy=True)
                        for key,value in arrays.items()}
                path=args.output/f'{source}_{split}.npz'; np.savez(path,**arrays)
                copied={k:v for k,v in entry.items() if k not in ('_path','source_manifest')}
                copied.update(path=path.name,frames=42); sequences.append(copied)
                selections.append({'source':source,'split':split,'path':str(entry['_path']),'frame_begin':begin,'frames':42})
                break
            else: raise RuntimeError(f'No contiguous 42-frame {source} {split} reference')
    manifest=args.output/'manifest.json'
    manifest.write_text(json.dumps({'rotation6d':'columns_c0xyz_c1xyz','sequences':sequences},indent=2)+'\n')
    config=json.loads((root/'config/twm_four_sources_moe_transient_v3_epoch2.json').read_text())
    config.update(batch_size=8,eval_batch_size=4,eval_batches=1,log_every=1,save_every=1)
    config['tracking'].update(mode='disabled',require_online=False)
    cfg=args.output/'config.json'; cfg.write_text(json.dumps(config,indent=2)+'\n')
    env=os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES='0,1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONPATH=str(root/'src'))
    command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=2',str(root/'scripts/train_probabilistic_twm.py'),
        '--manifest',str(manifest),'--config',str(cfg),'--code-version',args.code_version]
    def run(out,extra=(),stop=False):
        out.mkdir()
        return launch(out,extra,stop)
    def launch(out,extra,stop):
        print(json.dumps({'phase':'launching_smoke','output':str(out),'args':list(extra)}),flush=True)
        with (out/'train.log').open('ab') as log:
            proc=subprocess.Popen(command+['--output',str(out)]+list(extra),env=env,stdout=log,stderr=subprocess.STDOUT)
            requested=False; end=time.monotonic()+300
            while proc.poll() is None:
                if stop and not requested and (out/'status.json').exists():
                    status=json.loads((out/'status.json').read_text())
                    if status.get('phase')=='training' and status.get('step',0)>=1:
                        (out/'STOP').write_text('integration test cooperative save\n'); requested=True
                if time.monotonic()>end:
                    proc.terminate(); raise TimeoutError(str(out))
                time.sleep(.02)
            if proc.returncode:
                print((out/'train.log').read_text()[-9000:],flush=True); raise RuntimeError('DDP smoke failed')
        status=json.loads((out/'status.json').read_text())
        if stop: assert requested and status['phase']=='stopped' and 0<status['step']<4,status
        audit=args.output/(out.name+f'_audit_{status["step"]}.json')
        subprocess.run([sys.executable,str(root/'scripts/audit_probabilistic_checkpoint.py'),'--run',str(out),'--output',str(audit)],check=True,stdout=subprocess.DEVNULL)
        result=json.loads(audit.read_text()); result={k:result[k] for k in ('step','model_and_optimizer_finite','model_state_sha256','all_saved_rank_hashes_equal_cpu_checkpoint','world_size')}
        result['progress']={k:status.get(k) for k in ('epoch_progress','completed_epochs','windows_seen')}
        print(json.dumps({'phase':'smoke_saved','audit':result}),flush=True)
        return result
    out=args.output/'resume_run'
    first=run(out,['--stop-after-updates','1'])
    resumed=launch(out,['--resume',str(out/'latest.pt')],False)
    assert first['step']==1 and resumed['step']==4 and resumed['progress']['completed_epochs']==2
    assert resumed['progress']['windows_seen']==24 and first['model_state_sha256']!=resumed['model_state_sha256']
    last=json.loads((out/'metrics.jsonl').read_text().splitlines()[-1])
    initial=json.loads((out/'metrics.jsonl').read_text().splitlines()[0])
    assert all(last['samples_by_dataset'][name]+initial['samples_by_dataset'][name]==6 for name in ('oakink2','hot3d','umetrack','dexycb')),last['samples_by_dataset']
    # Counts reset on resume: 24 total exposures minus the initial eight.
    stopped=run(args.output/'stop_run',stop=True)
    result={'code_version':args.code_version,'source_selections':selections,'first':first,'resumed':resumed,'stopped':stopped,
        'checks':{'two_rank_updates_finite_sync':True,'resume_across_epoch_boundary':True,
                  'last_partial_batch':True,'two_epochs_24_window_exposures':True,'cooperative_stop_saved':True},
        'scope':'tiny real-reference integration check; no quality claim; formal data/splits untouched'}
    (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
