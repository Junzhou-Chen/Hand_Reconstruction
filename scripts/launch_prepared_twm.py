#!/usr/bin/env python3
"""Launch an already audited TWM collection inside the authorized compute Pod.

One-shot launcher, not a scheduler. Refuses missing sources, changed manifests,
busy devices, stale deadlines and existing output. Credentials stay in the
child environment. A launch receipt is not evidence of a completed update.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code', type=Path, required=True)
    parser.add_argument('--code-version', required=True)
    parser.add_argument('--python', required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpus', default='0,1')
    parser.add_argument('--wandb-key-file', type=Path, required=True)
    args = parser.parse_args()
    if not Path('/sciclone/home/jchen57').is_dir():
        parser.error('Launch training only inside the authorized remote Pod')
    if any(args.output.glob('*')):
        raise RuntimeError('Training output is not empty; inspect it and use a fresh path')
    code = args.code.resolve()
    head = subprocess.check_output(['git', '-C', str(code), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(code), 'status', '--porcelain',
                                     '--untracked-files=no'], text=True).strip()
    if head != args.code_version or dirty:
        raise RuntimeError('Require the exact clean committed training code')
    config = json.loads(args.config.read_text())
    manifest = json.loads(args.manifest.read_text())
    expected = set(config['required_datasets'])
    if manifest.get('phase') != 'complete' or set(manifest['required_datasets']) != expected:
        raise ValueError('Completed collection must match every configured dataset')
    for entry in manifest['manifests']:
        path = (args.manifest.parent / entry['path']).resolve()
        if sha(path) != entry['manifest_sha256']:
            raise ValueError(f'Prepared source changed: {path}')
    if not manifest.get('original_source_preservation'):
        raise ValueError('Collection must certify preservation of the previous four sources')
    deadline = datetime.fromisoformat(config['stop_before_utc'].replace('Z', '+00:00'))
    if deadline.tzinfo is None or deadline <= datetime.now(timezone.utc):
        raise ValueError('Allocation execution deadline has passed')
    devices = args.gpus.split(',')
    if len(devices) != config['world_size'] or len(set(devices)) != len(devices):
        raise ValueError('Distinct GPU list must match configured world size')
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                                   '--format=csv,noheader,nounits'], text=True)
    device_status = {line.split(',')[0].strip(): [int(x.strip()) for x in line.split(',')[1:]]
                     for line in gpu.strip().splitlines()}
    if any(device_status.get(d, [10**9, 100])[0] > 1024 or
           device_status.get(d, [10**9, 100])[1] > 5 for d in devices):
        raise RuntimeError('Requested GPU is in use; no training process launched')
    secret = args.wandb_key_file
    if not secret.is_file() or secret.stat().st_mode & 0o077:
        raise ValueError('W&B credential must be a private existing file')
    env = os.environ.copy()
    threads = str(config.get('cpu_threads', 4))
    env.update(CUDA_VISIBLE_DEVICES=args.gpus, PYTHONPATH=str(code / 'src'),
               OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads,
               OPENBLAS_NUM_THREADS='1', WANDB_API_KEY=secret.read_text().strip())
    command = [args.python, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
               '--nproc-per-node=' + str(len(devices)), str(code / 'scripts/train_probabilistic_twm.py'),
               '--manifest', str(args.manifest), '--config', str(args.config),
               '--output', str(args.output), '--code-version', head]
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'train.log').open('ab', buffering=0) as log:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, cwd=code, env=env, start_new_session=True)
    receipt = {'pid': process.pid, 'command': command, 'code_version': head,
               'started_utc': datetime.now(timezone.utc).isoformat(),
               'manifest_sha256': sha(args.manifest), 'config_sha256': sha(args.config),
               'gpus': devices, 'status': 'launcher_created; inspect status.json and online W&B for actual updates'}
    (args.output / 'process.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
