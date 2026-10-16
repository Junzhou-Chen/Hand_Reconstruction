#!/usr/bin/env python3
"""Read-only TWM interventions on already saved visual states, remote only.

These are controlled numeric interventions, not newly trained baselines or a
replacement for the original whole-pipeline evaluation. Oracle past states
are explicitly diagnostic and never claimed as deployable observations.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import sys


def digest_file(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def model_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        value = value.detach().cpu().contiguous()
        digest.update((name + str(value.dtype) + str(tuple(value.shape))).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def load_twm(path, device):
    saved = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    if 'twm_source' in saved:
        config = saved['twm_source']['model_config']
        prefix = 'pipeline.feedback.twm.'
        state = {key[len(prefix):]: value for key, value in saved['model'].items() if key.startswith(prefix)}
    else:
        config, state = saved['config']['model'], saved['model']
    model = ProbabilisticTWM(ProbabilisticTWMConfig(**config))
    model.load_state_dict(state, strict=True)
    info = {'path': str(path.resolve()), 'step': int(saved['step']), 'config': config,
            'file_bytes': path.stat().st_size, 'file_mtime_ns': path.stat().st_mtime_ns,
            'source_code': saved.get('code_version')}
    del saved, state
    gc.collect()
    return model.to(device).eval().requires_grad_(False), info


def collect(prediction_dir, baseline_dir, manifest):
    source = json.loads(manifest.read_text())
    source_sha = digest_file(manifest)
    records = {r['recording']: r for r in source['records']}
    clips, frames = [], []
    for path in sorted((prediction_dir / 'clips').glob('*.pt')):
        payload = torch.load(path, map_location='cpu', weights_only=False)
        baseline = torch.load(baseline_dir / 'clips' / path.name, map_location='cpu', weights_only=False)
        for item in (payload, baseline):
            if item['source_manifest_sha256'] != source_sha:
                raise ValueError('Prediction manifest mismatch')
        if payload['source_record']['recording'] != baseline['source_record']['recording']:
            raise ValueError('Paired recording mismatch')
        if not torch.equal(payload['source_frame_ids'], baseline['source_frame_ids']):
            raise ValueError('Paired frame mismatch')
        for key in ('reference_indices', 'handedness', 'target_world', 'target_valid', 'frame_valid'):
            if not torch.equal(payload[key], baseline[key]):
                raise ValueError('Paired target/identity mismatch: ' + key)
        record = records[payload['source_record']['recording']]
        with np.load(manifest.parent / record['reference_file'], allow_pickle=False) as archive:
            reference_times = torch.from_numpy(archive['timestamps'].copy()).double()
        n = int(payload['unpadded_frames'])
        indices = payload['reference_indices'][:n].long()
        timestamps = reference_times[indices]
        if not bool(payload['target_valid'][0, :n].all()):
            raise ValueError('This diagnostic needs complete paired reference states')
        if not bool(payload['availability'][0, :n].all()) or not bool(baseline['availability'][0, :n].all()):
            raise ValueError('Explicitly extend handling before using unavailable observations')
        clip = {'payload': payload, 'baseline': baseline, 'times': timestamps, 'frames': n}
        clips.append(clip)
        for t in range(n):
            frames.append({'clip': len(clips) - 1, 'offset': t,
                           'recording': record['recording'], 'hand': int(payload['handedness'][0, t]),
                           'clip_index': int(payload['clip_index']), 'frame_id': int(payload['source_frame_ids'][t])})
    if not clips:
        raise ValueError('No saved predictions')
    return clips, frames, source_sha


def prepare_inputs(clips, frames, control, device):
    histories, raws, candidates, anchors, valids, dts = [], [], [], [], [], []
    targets, hands, stored = [], [], []
    for frame in frames:
        clip = clips[frame['clip']]
        p, base, t = clip['payload'], clip['baseline'], frame['offset']
        raw = (base if control.startswith('baseline_') else p)['raw_world'][0, t].float()
        candidate = raw if control.startswith('baseline_') else p['candidate_world'][0, t].float()
        hist = g.identity_state(30, device=torch.device('cpu'))
        valid = torch.zeros(30, dtype=torch.bool)
        hist_time = torch.zeros(30, dtype=torch.float64)
        if control.endswith('oracle_history'):
            past = p['target_world'][0]
        elif control.startswith('baseline_'):
            past = base['raw_world'][0]
        else:
            past = p['state_world'][0]
        n = 0 if control == 'joint_empty_history' else min(t, 30)
        if n:
            hist[-n:] = past[t - n:t]
            hist_time[-n:] = clip['times'][t - n:t]
            valid[-n:] = True
        anchor = state_anchor_world(hist[-1] if n else raw)
        if control == 'joint_saved_history':
            saved_anchor = p['anchor_world'][0, t].float()
            if not torch.allclose(anchor, saved_anchor, atol=2e-5, rtol=0):
                raise ValueError('Saved anchor cannot be reconstructed from completed history')
            anchor = saved_anchor
        hist_dt = torch.full((30,), 1 / 30)
        hist_dt[1:] = torch.where(valid[:-1] & valid[1:],
                                 (hist_time[1:] - hist_time[:-1]).float(), 1 / 30)
        hist_dt = torch.where(valid, hist_dt, 0.)
        current_dt = float(clip['times'][t] - hist_time[-1]) if n else 1 / 30
        histories.append(hist); raws.append(raw); candidates.append(candidate)
        anchors.append(anchor); valids.append(valid)
        dts.append(torch.cat((hist_dt, torch.tensor([current_dt]))))
        targets.append(p['target_world'][0, t]); hands.append(frame['hand'])
        stored.append(p['state_world'][0, t])
    anchors = torch.stack(anchors).to(device)
    to_anchor = inverse_rigid(anchors)
    hist_valid = torch.stack(valids).to(device)
    result = {'history': transform_state(torch.stack(histories).to(device), to_anchor),
              'raw': transform_state(torch.stack(raws).to(device), to_anchor),
              'candidate': transform_state(torch.stack(candidates).to(device), to_anchor),
              'history_valid': hist_valid, 'history_availability': hist_valid[..., None].expand(-1, -1, 17),
              'availability': torch.ones(len(frames), 17, dtype=torch.bool, device=device),
              'candidate_availability': torch.ones(len(frames), 17, dtype=torch.bool, device=device),
              'dt': torch.stack(dts).to(device), 'handedness': torch.tensor(hands, device=device),
              'beta': None, 'return_forecast': False}
    return result, anchors, torch.stack(targets).to(device), torch.stack(stored).to(device)


def errors(pred, target):
    angles = g.geodesic_angle(g.state_to_rotmat(pred), g.state_to_rotmat(target)) * (180 / np.pi)
    return {'wrist_mm': (pred[..., :3] - target[..., :3]).norm(dim=-1) * 1000,
            'root_deg': angles[..., 0], 'finger_deg': angles[..., 1:].mean(-1)}


def summarize(states, targets, candidates, frames):
    values, before = errors(states, targets), errors(candidates, targets)
    result = {}
    for side, code in [('left', 0), ('right', 1), ('all', None)]:
        mask = torch.tensor([code is None or f['hand'] == code for f in frames], device=states.device)
        result[side] = {'states': int(mask.sum()),
                        'output': {k: float(v[mask].mean()) for k, v in values.items()},
                        'candidate': {k: float(v[mask].mean()) for k, v in before.items()},
                        'wrist_degraded_over_1mm': float((values['wrist_mm'][mask] > before['wrist_mm'][mask] + 1).float().mean())}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--joint-predictions', type=Path, required=True)
    parser.add_argument('--baseline-predictions', type=Path, required=True)
    parser.add_argument('--source-manifest', type=Path, required=True)
    parser.add_argument('--checkpoint', action='append', required=True, help='LABEL=PATH; labels pretrain,best,last')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    if not Path('/sciclone/home/jchen57').is_dir():
        parser.error('Run inside the authorized SciClone Pod')
    if args.output.exists():
        raise FileExistsError('Use a fresh output directory')
    global torch, np, g, ProbabilisticTWM, ProbabilisticTWMConfig, transform_state, inverse_rigid, state_anchor_world
    import numpy as np
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
    from handloop import geometry as g
    from handloop.models.probabilistic_twm import ProbabilisticTWM, ProbabilisticTWMConfig
    from handloop.models.visual_feedback import transform_state, inverse_rigid, state_anchor_world
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.cuda.set_device(args.device)
    torch.manual_seed(20261016)
    device = torch.device(args.device)
    args.output.mkdir(parents=True)
    clips, frames, source_sha = collect(args.joint_predictions, args.baseline_predictions, args.source_manifest)
    report = {'schema': 'handloop.real_state_twm_interventions.v1',
              'started_utc': datetime.now(timezone.utc).isoformat(), 'source_manifest_sha256': source_sha,
              'script_sha256': digest_file(Path(__file__)), 'frames': len(frames),
              'control_definition': {'joint_saved_history': 'fixed saved joint raw/candidate/completed history; saved checkpoint replay parity',
                  'joint_oracle_history': 'same joint raw/candidate; oracle reference past; recomputed anchor',
                  'joint_empty_history': 'same joint raw/candidate; empty history; current raw anchor',
                  'baseline_raw_history': 'released HaWoR raw as raw/candidate and past observations, no visual adapter',
                  'baseline_oracle_history': 'released HaWoR raw as raw/candidate; oracle reference past, no adapter'},
              'scope': 'Numeric post-hoc interventions on frozen weights; oracle controls not deployable; not whole-pipeline retrained ablations; original fixed-camera GT-crop fitted-MANO limits retained.',
              'models': {}}
    original_config = None
    for specification in args.checkpoint:
        label, raw_path = specification.split('=', 1)
        path = Path(raw_path)
        model, info = load_twm(path, device)
        if original_config is None:
            original_config = info['config']
        elif info['config'] != original_config:
            raise ValueError('Checkpoint TWM architectures/configs differ')
        before_hash = model_hash(model)
        result = {'identity': info, 'hash_before': before_hash, 'controls': {}}
        for control in report['control_definition']:
            inputs, anchors, targets, saved = prepare_inputs(clips, frames, control, device)
            outputs = []
            with torch.no_grad():
                for start in range(0, len(frames), args.batch_size):
                    end = start + args.batch_size
                    batch = {k: (v[start:end] if torch.is_tensor(v) else v) for k, v in inputs.items()}
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        output = model(**batch)
                    outputs.append(transform_state(output['state'], anchors[start:end]))
            states = torch.cat(outputs)
            candidate_world = transform_state(inputs['candidate'], anchors)
            measured = summarize(states, targets, candidate_world, frames)
            if control == 'joint_saved_history':
                difference = errors(states, saved)
                measured['saved_replay_difference'] = {k: {'mean': float(v.mean()), 'max': float(v.max())} for k, v in difference.items()}
                if label == 'best' and (float(difference['wrist_mm'].max()) > 1.0 or float(difference['root_deg'].max()) > .25):
                    raise ValueError('Saved best replay parity failed; do not interpret interventions')
            result['controls'][control] = measured
            torch.save({'state_world': states.cpu(), 'frames': frames, 'control': control,
                        'source_manifest_sha256': source_sha, 'checkpoint_label': label}, args.output / f'{label}_{control}.pt')
            print(json.dumps({'model': label, 'control': control, 'left': measured['left'], 'right': measured['right']}), flush=True)
        result['hash_after'] = model_hash(model)
        if before_hash != result['hash_after']:
            raise ValueError('Evaluation modified TWM weights')
        if (info['file_bytes'], info['file_mtime_ns']) != (path.stat().st_size, path.stat().st_mtime_ns):
            raise ValueError('Source checkpoint changed')
        report['models'][label] = result
        (args.output / 'report.incomplete.json').write_text(json.dumps(report, indent=2) + '\n')
        del model
        gc.collect(); torch.cuda.empty_cache()
    report['completed_utc'] = datetime.now(timezone.utc).isoformat()
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'phase': 'complete', 'output': str(args.output / 'report.json')}), flush=True)


if __name__ == '__main__':
    main()
