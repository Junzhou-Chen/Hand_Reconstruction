#!/usr/bin/env python3
"""Remote resumable full-s0 evaluation worker with online W&B progress.

One long-lived process per model follows atomically published recording/view
shards. Models are loaded once. Only a completed shard enters cumulative test
metrics; restarting an incomplete shard cannot double count its frames.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def physical(record):
    return record.get('physical_recording', '/'.join(record['recording'].split('/')[:2]))


class OnlineEvaluation:
    def __init__(self, args, config):
        import wandb

        self.path = args.output / 'wandb_run.json'
        previous = read(self.path) if self.path.exists() else None
        settings = wandb.Settings(console='off', disable_code=True, disable_git=True,
                                  save_code=False, init_timeout=90)
        self.run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
            id=previous['id'] if previous else None, resume='allow' if previous else None,
            name='DexYCB-s0-full-' + args.label, group=args.wandb_group,
            job_type='full-test', mode='online', dir=str(args.output),
            config=config, settings=settings, tags=['dexycb', 's0-test', 'all8views', args.label])
        if self.run is None or self.run.settings.mode != 'online':
            raise RuntimeError('Online W&B is required for this user-requested test')
        self.event = int(previous.get('last_event', 0)) if previous else 0
        self.meta = {'id': self.run.id, 'url': self.run.url, 'project': args.wandb_project,
                     'entity': args.wandb_entity, 'state': 'running', 'last_event': self.event}
        self.run.define_metric('progress/completed_shards')
        self.run.define_metric('test/*', step_metric='progress/completed_shards')
        write(self.path, self.meta)

    def log(self, values):
        self.event += 1
        self.run.log(values, step=self.event)
        self.meta['last_event'] = self.event
        self.meta['last_log_utc'] = utc()
        write(self.path, self.meta)

    def finish(self, success):
        self.run.finish(exit_code=0 if success else 1)
        self.meta.update(state='finished' if success else 'failed', finished_utc=utc())
        write(self.path, self.meta)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-root', 'training-manifest', 'freeze-manifest', 'hawor-root',
                 'hawor-checkpoint', 'hawor-config', 'mano-models', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--label', choices=('baseline', 'best', 'last'), required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--precision', choices=('bf16', 'fp32'), default='bf16')
    parser.add_argument('--clip-length', type=int, default=32)
    parser.add_argument('--cpu-threads', type=int, default=1)
    parser.add_argument('--poll-seconds', type=float, default=20)
    parser.add_argument('--wandb-project', default='handloop-twm')
    parser.add_argument('--wandb-entity', default='black-ant')
    parser.add_argument('--wandb-group', default='dexycb-s0-full-20261016')
    parser.add_argument('--smoke-shards', type=int, default=0,
                        help='Explicit separate-output smoke only; zero means all1280 shards')
    args = parser.parse_args()
    if not Path('/sciclone/data10/jchen57').is_dir():
        parser.error('Run all inference/metrics inside the authorized compute Pod')
    if args.label == 'baseline' and args.checkpoint or args.label != 'baseline' and not args.checkpoint:
        parser.error('Only best/last take a frozen joint checkpoint')
    if args.clip_length != 32 or args.poll_seconds <= 0 or args.poll_seconds > 60:
        parser.error('Keep trained clip32 and poll interval within (0,60]')
    args.output.mkdir(parents=True, exist_ok=True)
    import numpy as np
    import torch
    from torch.utils.data import default_collate
    from evaluate_visual_joint import EvaluationRGBDataset, load_model, observe
    from dexycb_test_metrics import StreamingDexYCBMetrics
    from handloop.visual.codec import MANOStateCodec

    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
    if args.precision == 'bf16' and device.type != 'cuda':
        raise ValueError('BF16 evaluation requires CUDA')
    frozen = read(args.freeze_manifest)
    item = frozen['models'][args.label]
    if args.checkpoint and (str(args.checkpoint.resolve()) != item['checkpoint']
                            or sha(args.checkpoint) != item['sha256']):
        raise ValueError('Joint checkpoint differs from the pre-test frozen identity')
    if sha(args.hawor_checkpoint) != frozen['models']['baseline']['sha256']:
        raise ValueError('Official baseline checkpoint changed')
    training = read(args.training_manifest)
    train_sha = sha(args.training_manifest)
    forbidden = {physical(r) for r in training['records'] if r['split'] in ('train', 'val')}
    identity = {'label': args.label, 'checkpoint_sha256': item['sha256'],
                'training_manifest_sha256': train_sha, 'freeze_manifest_sha256': sha(args.freeze_manifest),
                'source_root': str(args.source_root.resolve()), 'clip_length': 32,
                'precision': args.precision, 'smoke_shards': args.smoke_shards,
                'evaluator_sha256': sha(__file__),
                'metrics_sha256': sha(Path(__file__).with_name('dexycb_test_metrics.py'))}
    tracker = OnlineEvaluation(args, {**identity, 'checkpoint_step': item.get('step'),
        'planned_views': 1280, 'planned_physical_recordings': 160, 'cameras': 8,
        'scope': 'GT crop, original fixed camera gauge, full s0 test, no prefix limit; '
                 'official16 bidirectional blocks, independent32-frame feedback clips; not detector/SLAM evaluation',
        'training_run': frozen['training_run'], 'checkpoint_selection': 'validation-only best; last at user-requested safe stop'})
    def status(phase, **values):
        write(args.output / 'status.json', {'utc': utc(), 'phase': phase,
            'label': args.label, 'wandb_url': tracker.run.url, **values})
    try:
        prior_completed = len(read(args.output/'accumulator.json')['completed']) if (args.output/'accumulator.json').exists() else 0
        status('initializing', completed_shards=prior_completed, expected_shards=1280)
        tracker.log({'progress/completed_shards': prior_completed, 'progress/expected_shards': 1280,
                     'progress/initializing': 1})
        # These argparse fields are consumed by the audited shared model loader.
        args.mode = 'baseline' if args.label == 'baseline' else 'joint'
        args.split = 'test'
        args.source_manifest = args.source_root / 'source_manifest.json'
        model, model_info = load_model(args, device)
        if args.label != 'baseline':
            if (model_info['training_identity']['source_manifest_sha256'] != train_sha
                    or model_info['checkpoint_step'] != item['step']):
                raise ValueError('Checkpoint training source/step differs from frozen run')
        pred_codec = model.codec if args.label == 'baseline' else model.frontend.codec
        ref_codec = MANOStateCodec(args.mano_models / 'MANO_RIGHT.pkl',
                                  args.mano_models / 'MANO_LEFT.pkl', left_shapedirs_fix=False).to(device)
        def accumulator():
            return StreamingDexYCBMetrics(pred_codec, ref_codec, device,
                batch_size=64, group_by=('hand', 'subject', 'view'))
        total = accumulator()
        progress_path = args.output / 'accumulator.json'
        completed, completed_clips, selection_sha = {}, 0, None
        if progress_path.exists():
            saved = read(progress_path)
            if saved['identity'] != identity:
                raise ValueError('Resume requires identical evaluator, weights, and inputs')
            total.load_state_dict(saved['metrics'])
            completed, completed_clips = saved['completed'], saved['completed_clips']
            selection_sha = saved['selection_sha256']
        write(args.output / 'provenance.json', {'identity': identity, 'model': model_info,
            'frozen_models': frozen, 'started_or_resumed_utc': utc(),
            'geometry': {'prediction': pred_codec.provenance(), 'reference': ref_codec.provenance()},
            'history_policy': 'Reset each32-frame clip; no padding commits; gaps split clips',
            'frame_ids': 'all preselected official s0 test valid frames; no prefix or recording cap'})
        expected = 1280
        while True:
            index_path = args.source_root / 'shard_index.json'
            if not index_path.exists():
                status('waiting_for_sources', completed_shards=len(completed), expected_shards=expected)
                tracker.log({'progress/completed_shards': len(completed), 'progress/waiting_for_sources': 1})
                time.sleep(args.poll_seconds)
                continue
            index = read(index_path)
            if index.get('phase') == 'failed':
                raise RuntimeError('Full test preparation failed: ' + str(index.get('error')))
            preparation_status = args.source_root / 'preparation_status.json'
            if preparation_status.exists() and read(preparation_status).get('phase') == 'failed':
                raise RuntimeError('Preparation failed; see ' + str(preparation_status))
            if index['expected_shards'] != expected or not index.get('eval_only') or index.get('training_allowed'):
                raise ValueError('Require full eval-only 1280-view source selection')
            if selection_sha is None:
                selection_sha = index['selection_sha256']
            elif selection_sha != index['selection_sha256']:
                raise ValueError('Source selection changed during evaluation')
            keys = [entry['key'] for entry in index['shards']]
            if len(keys) != len(set(keys)) or len(keys) > expected:
                raise ValueError('Duplicate/excess prepared shard identities')
            indexed_entries = {entry['key']: entry for entry in index['shards']}
            for key, saved in completed.items():
                if (key not in indexed_entries or saved['source_manifest_sha256'] !=
                        indexed_entries[key]['manifest_sha256'] or saved['frames'] != indexed_entries[key]['frames']):
                    raise ValueError('Previously completed source shard changed during resume')
            ready = [entry for entry in index['shards'] if entry['key'] not in completed]
            for entry in ready:
                key = entry['key']
                if not key or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_+-' for c in key):
                    raise ValueError('Unsafe source shard key')
                if entry['physical_recording'] in forbidden:
                    raise ValueError('Test physical recording overlaps train/selection: ' + key)
                manifest_path = (args.source_root / entry['manifest']).resolve()
                if not manifest_path.is_relative_to(args.source_root.resolve()) or sha(manifest_path) != entry['manifest_sha256']:
                    raise ValueError('Source shard manifest path/hash mismatch')
                unit_dir = args.output / 'shards' / key
                result_path = unit_dir / 'result.json'
                unit_dir.mkdir(parents=True, exist_ok=True)
                if result_path.exists():
                    result = read(result_path)
                    if (result['identity'] != identity or result['source_manifest_sha256'] != entry['manifest_sha256']):
                        raise ValueError('Committed shard result has another input/model identity')
                else:
                    local = accumulator()
                    clip_count, processed, frames_hash = 0, 0, hashlib.sha256()
                    arrays_to_save = {'frame_ids': [], 'beta': [], 'availability': []}
                    if entry['frames']:
                        dataset = EvaluationRGBDataset(manifest_path, 'test', clip_length=32, hawor_root=args.hawor_root)
                        if len(dataset.records) != 1 or physical(dataset.records[0]) != entry['physical_recording']:
                            raise ValueError('One immutable camera trajectory is required per shard')
                        record = dataset.records[0]
                        arrays = dataset._load_reference(0)
                        reference_beta = arrays['beta']
                        if reference_beta.shape == (10,):
                            reference_beta = np.broadcast_to(reference_beta, (int(record['frames']), 10)).copy()
                        seen_ids = []
                        for clip_index, (ri, start, count) in enumerate(dataset.windows):
                            if ri != 0:
                                raise ValueError('Unexpected second recording in a view shard')
                            batch = default_collate([dataset[clip_index]])
                            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                            context = torch.autocast('cuda', dtype=torch.bfloat16) if args.precision == 'bf16' else nullcontext()
                            with torch.inference_mode(), context:
                                if args.mode == 'joint':
                                    output = model(batch, return_forecast=False)
                                    predictions = {'raw': output['raw_world'][0, :count],
                                                   'candidate': output['candidate_world'][0, :count],
                                                   'final': output['state_world'][0, :count]}
                                else:
                                    observed = observe(model, batch)
                                    output = {'raw_world': observed['raw_state_world'],
                                        'availability': observed['raw_available'],
                                        'frame_valid': observed['frame_valid'], 'beta': observed['beta']}
                                    predictions = {'raw': output['raw_world'][0, :count]}
                            with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
                                local.update(predictions, output['beta'][0, :count],
                                    batch['target_world'][0, :count], reference_beta[start:start+count],
                                    arrays['joint_3d'][start:start+count], batch['is_right'][0, :count],
                                    batch['target_valid'][0, :count], availability=output['availability'][0, :count],
                                    frame_valid=output['frame_valid'][0, :count],
                                    joint_valid=arrays.get('joint_3d_valid', np.ones((int(record['frames']), 21), dtype=bool))[start:start+count],
                                    group={'recording': key, 'subject': entry['subject'], 'view': entry['serial']})
                            ids = arrays['frame_ids'][start:start+count]
                            seen_ids.extend(int(i) for i in ids)
                            frames_hash.update(np.asarray(ids, dtype='<i8').tobytes())
                            arrays_to_save['frame_ids'].append(np.array(ids, copy=True))
                            arrays_to_save['beta'].append(output['beta'][0, :count].float().cpu().numpy())
                            arrays_to_save['availability'].append(output['availability'][0, :count].cpu().numpy())
                            for label, state in predictions.items():
                                arrays_to_save.setdefault(label, []).append(state.float().detach().cpu().numpy())
                            clip_count += 1
                            processed += count
                            if clip_count % 5 == 0:
                                status('evaluating', completed_shards=len(completed), expected_shards=expected,
                                    current_shard=key, current_shard_clips=clip_count,
                                    completed_frames=total.snapshot()['all']['source_frames'])
                                tracker.log({'progress/completed_shards': len(completed),
                                    'progress/current_shard_clips': clip_count, 'progress/current_shard_frames': processed})
                            del output, batch, predictions
                        if (processed != entry['frames'] or len(set(seen_ids)) != processed
                                or seen_ids != arrays['frame_ids'].tolist()):
                            raise ValueError('Model windows do not exactly cover all manifest frame identities')
                        tmp = unit_dir / 'predictions.npz.tmp'
                        with tmp.open('wb') as f:
                            np.savez(f, **{k: np.concatenate(v) for k, v in arrays_to_save.items()})
                        tmp.replace(unit_dir / 'predictions.npz')
                        del dataset, arrays, arrays_to_save
                        gc.collect()
                    snap = local.snapshot()
                    if snap['all']['source_frames'] != entry['frames']:
                        raise ValueError('Metrics did not cover all source frames')
                    for branch, value in snap['all']['branches'].items():
                        if value['covered_counts']['missing_geometry_states']:
                            raise ValueError('Eligible prediction missing; fail rather than silently omit test frames')
                    result = {'phase': 'complete', 'completed_utc': utc(), 'identity': identity,
                        'source_manifest_sha256': entry['manifest_sha256'], 'key': key,
                        'frames': processed, 'clips': clip_count, 'frame_ids_sha256': frames_hash.hexdigest(),
                        'metrics_state': local.state_dict(), 'summary': snap,
                        'predictions_sha256': sha(unit_dir/'predictions.npz') if entry['frames'] else None}
                    write(result_path, result)
                total.merge_state_dict(result['metrics_state'])
                completed[key] = {'frames': result['frames'], 'clips': result['clips'],
                    'frame_ids_sha256': result['frame_ids_sha256'], 'source_manifest_sha256': entry['manifest_sha256']}
                completed_clips += result['clips']
                write(progress_path, {'identity': identity, 'selection_sha256': selection_sha,
                    'completed': completed, 'completed_clips': completed_clips, 'metrics': total.state_dict()})
                values = total.wandb_scalars()
                values.update({'progress/completed_shards': len(completed), 'progress/expected_shards': expected,
                    'progress/completed_clips': completed_clips,
                    'progress/completed_frames': total.snapshot()['all']['source_frames'],
                    'progress/prepared_shards': index['completed_shards']})
                tracker.log(values)
                status('evaluating', completed_shards=len(completed), expected_shards=expected,
                    completed_clips=completed_clips, completed_frames=total.snapshot()['all']['source_frames'])
                write(args.output/'running_metrics.json', total.snapshot())
                print(json.dumps({'utc': utc(), 'label': args.label, 'completed_shards': len(completed),
                    'frames': total.snapshot()['all']['source_frames']}), flush=True)
                if args.smoke_shards and len(completed) >= args.smoke_shards:
                    break
            smoke_done = bool(args.smoke_shards and len(completed) >= args.smoke_shards)
            if smoke_done or index['phase'] == 'complete' and len(completed) == expected:
                if not smoke_done:
                    if (len(keys) != expected or set(keys) != set(completed)
                            or total.snapshot()['all']['source_frames'] != index['frames']):
                        raise ValueError('Final complete-model coverage differs from full source selection')
                report = {'phase': 'smoke_complete' if smoke_done else 'complete',
                    'completed_utc': utc(), 'identity': identity, 'selection_sha256': selection_sha,
                    'completed_shards': len(completed), 'expected_shards': expected,
                    'completed_clips': completed_clips, 'metrics': total.snapshot()}
                write(args.output/'report.json', report)
                tracker.log({**total.wandb_scalars(), 'progress/completed_shards': len(completed),
                             'progress/complete': int(not smoke_done), 'progress/smoke_complete': int(smoke_done)})
                status(report['phase'], completed_shards=len(completed), expected_shards=expected,
                    completed_frames=total.snapshot()['all']['source_frames'])
                tracker.finish(True)
                return
            if not ready:
                status('waiting_for_sources', completed_shards=len(completed), expected_shards=expected,
                       completed_frames=total.snapshot()['all']['source_frames'])
                tracker.log({'progress/completed_shards': len(completed), 'progress/prepared_shards': index['completed_shards'],
                             'progress/waiting_for_sources': 1})
                time.sleep(args.poll_seconds)
    except BaseException as exc:
        status('failed', error_type=type(exc).__name__, error=str(exc))
        try:
            tracker.log({'progress/failed': 1})
        except Exception:
            traceback.print_exc()
        try:
            tracker.finish(False)
        except Exception:
            traceback.print_exc()
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
