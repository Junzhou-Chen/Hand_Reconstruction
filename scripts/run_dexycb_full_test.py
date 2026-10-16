#!/usr/bin/env python3
"""Persistent Pod controller: freeze paused weights, smoke, then all3 tests."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('/sciclone/data10/jchen57/Hand_Reconstruction'))
    p.add_argument('--code', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--training-run', type=Path, required=True)
    p.add_argument('--training-manifest', type=Path, required=True)
    args = p.parse_args()
    if not Path('/sciclone/home/jchen57').is_dir():
        p.error('Run this controller only inside the authorized compute Pod')
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output/'controller.lock').open('a+')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    root, code, out, train = args.root, args.code, args.output, args.training_run
    py = root/'runtime/twm_mamba_torch210_v1/bin/python'
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        PYTHONPATH=str(code/'src')+':'+str(root/'runtime/hawor_extra/site-packages')+':'+str(root/'model_assets/hawor_v1/source'))
    private = root/'private/wandb_api_key'
    if private.stat().st_mode & 0o077:
        raise ValueError('W&B credential must remain private600')
    env['WANDB_API_KEY'] = private.read_text().strip()
    def status(phase, **fields):
        value = {'phase': phase, 'utc': utc(), **fields}
        write(out/'controller_status.json', value)
        print(json.dumps(value), flush=True)
    def run_audit(label, checkpoint):
        target = out/(label+'_checkpoint_audit.json')
        if not target.exists():
            with (out/(label+'_checkpoint_audit.log')).open('w') as log:
                result = subprocess.run([str(py), str(code/'scripts/check_visual_joint_checkpoint.py'),
                    '--checkpoint', str(checkpoint), '--output', str(target)], cwd=code,
                    env=env, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(label+' checkpoint audit failed; see log')
        result = read(target)
        if not result['passed']:
            raise ValueError(label+' checkpoint audit did not pass')
        return result
    try:
        training_status = read(train/'status.json')
        if training_status['phase'] not in ('stopped', 'complete'):
            raise ValueError('Wait for cooperative training stop and complete save')
        if not read(out/'metric_acceptance_passed.json')['passed']:
            raise ValueError('Independent metrics acceptance must pass first')
        freeze_path = out/'frozen_models.json'
        if not freeze_path.exists():
            status('freezing_paused_checkpoints', paused_step=training_status['step'])
            models = out/'models'
            models.mkdir(exist_ok=True)
            freeze = {'schema': 'handloop.dexycb_frozen_three_models.v1', 'frozen_utc': utc(),
                'training_run': str(train), 'training_status': training_status,
                'models': {'baseline': {'checkpoint': str(root/'model_assets/hawor_v1/weights/hawor.ckpt'),
                    'sha256': sha(root/'model_assets/hawor_v1/weights/hawor.ckpt'), 'kind': 'official pretrained HaWoR raw'}}}
            for label, original in (('best', 'best.pt'), ('last', 'latest.pt')):
                target = models/(label+'.pt')
                if not target.exists():
                    os.link(train/original, target)
                result = run_audit(label, target)
                freeze['models'][label] = {'checkpoint': str(target), 'sha256': sha(target),
                    'step': result['step'], 'model_sha256': result['model_sha256'],
                    'training_code_version': result['code_version'],
                    'selection': 'validation geometry score' if label == 'best' else 'user-requested safe stop'}
            if freeze['models']['last']['step'] != training_status['step']:
                raise ValueError('Paused last checkpoint step mismatch')
            write(freeze_path, freeze)
        frozen = read(freeze_path)
        def launch(label, *, smoke=False):
            output = out/('smoke_'+label if smoke else label)
            if (output/'report.json').exists():
                return None
            name = ('smoke_' if smoke else '')+label
            receipt = out/(name+'_process.json')
            if receipt.exists():
                old = read(receipt)
                procfile = Path('/proc')/str(old['pid'])/'cmdline'
                if procfile.exists() and str(output).encode() in procfile.read_bytes():
                    raise RuntimeError('Existing evaluation worker is alive; refusing duplicate '+name)
            cmd = [str(py), str(code/'scripts/evaluate_dexycb_full.py'), '--label', label,
                '--source-root', str(args.source_root), '--training-manifest', str(args.training_manifest),
                '--freeze-manifest', str(freeze_path), '--hawor-root', str(root/'model_assets/hawor_v1/source'),
                '--hawor-checkpoint', str(root/'model_assets/hawor_v1/weights/hawor.ckpt'),
                '--hawor-config', str(root/'model_assets/hawor_v1/weights/model_config.yaml'),
                '--mano-models', str(root/'model_assets/mano_v1_2/prepared_v1/derived'),
                '--output', str(output), '--device', 'cuda:0', '--cpu-threads', '1',
                '--wandb-group', 'dexycb-s0-full-smoke-20261016' if smoke else 'dexycb-s0-full-20261016']
            if label != 'baseline':
                cmd += ['--checkpoint', frozen['models'][label]['checkpoint']]
            if smoke:
                cmd += ['--smoke-shards', '2']
            worker_env = env.copy()
            worker_env['CUDA_VISIBLE_DEVICES'] = '1' if label == 'best' else '0'
            with (out/(name+'.log')).open('ab', buffering=0) as log:
                process = subprocess.Popen(cmd, env=worker_env, cwd=code, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            write(out/(name+'_process.json'), {'pid': process.pid, 'command': cmd,
                'gpu': worker_env['CUDA_VISIBLE_DEVICES'], 'started_utc': utc()})
            return process
        status('real_rgb_smoke', frozen_steps={k:v.get('step') for k,v in frozen['models'].items()})
        smoke_workers = {label: launch(label, smoke=True) for label in ('baseline', 'best', 'last')}
        for label, process in smoke_workers.items():
            if process is not None and process.wait():
                raise RuntimeError('RealRGB smoke failed '+label+'; inspect its W&B/log')
            smoke = read(out/('smoke_'+label)/'report.json')
            if smoke['phase'] != 'smoke_complete' or smoke['completed_shards'] != 2:
                raise ValueError('Smoke did not finish two complete views '+label)
        # Compare exact model input identities, not just equal output counts.
        smoke_coverage = [read(out/('smoke_'+label)/'accumulator.json')['completed'] for label in ('baseline','best','last')]
        if any(value != smoke_coverage[0] for value in smoke_coverage[1:]):
            raise ValueError('Smoke model coverage differs')
        write(out/'real_rgb_smoke_acceptance.json', {'passed': True, 'utc': utc(),
            'models': ['baseline','best','last'], 'same_frame_ids_and_manifest_hashes': True,
            'completed': smoke_coverage[0]})
        workers = {label: launch(label) for label in ('baseline', 'best', 'last')}
        status('testing', models=['baseline','best','last'], scope='full s0 all1280 views; no frame cap')
        while any(proc is not None and proc.poll() is None for proc in workers.values()):
            states = {}
            for label, proc in workers.items():
                path = out/label/'status.json'
                states[label] = read(path) if path.exists() else {'phase': 'initializing'}
                if proc is not None and proc.poll() not in (None, 0):
                    states[label]['process_exit_code'] = proc.returncode
            status('testing', models=states)
            time.sleep(30)
        for label, process in workers.items():
            if process is not None and process.returncode:
                raise RuntimeError(label+' full test failed; other model results preserved')
        reports = {label: read(out/label/'report.json') for label in workers}
        coverage = {label: read(out/label/'accumulator.json')['completed'] for label in workers}
        if any(value != coverage['baseline'] for value in coverage.values()):
            raise ValueError('Final exact frame/shard coverage differs between models')
        if any(report['phase'] != 'complete' or report['completed_shards'] != 1280 for report in reports.values()):
            raise ValueError('Full1280-view coverage incomplete')
        write(out/'comparison.json', {'schema': 'handloop.dexycb_s0_full_comparison.v1',
            'completed_utc': utc(), 'exact_coverage_matches': True, 'frozen_models': frozen,
            'models': {label: report['metrics'] for label, report in reports.items()},
            'wandb': {label: read(out/label/'wandb_run.json') for label in workers}})
        status('complete', comparison=str(out/'comparison.json'))
    except BaseException as exc:
        status('failed', error_type=type(exc).__name__, error=str(exc))
        raise


if __name__ == '__main__':
    main()
