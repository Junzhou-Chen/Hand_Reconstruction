#!/usr/bin/env python3
"""Frozen-checkpoint synthetic restoration validation; no optimizer or training.

Fixed validation exactly retains the training batch64/window-selection/RNG
protocol. Full validation covers all held-out windows in the prepared source
splits. Last noisy history is a causal hold baseline, not independent GT.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

from train_twm import (torch, TWM, TWMConfig, TrajectoryWindows, atomic_json,
                       sample_batch, state_loss, metrics, utc)
from torch.utils.data import DataLoader, Subset


@torch.no_grad()
def evaluate_source(model, loader, config, device, output, protocol, source):
    result = {}
    model.eval()
    for mode in ('noisy', 'missing', 'clean'):
        generator = torch.Generator(device=device).manual_seed(config['seed']+123)
        totals, count = {}, 0
        for index, batch in enumerate(loader):
            batch = {k:v.to(device, non_blocking=True) for k,v in batch.items()}
            history,raw,candidate,kwargs,target,future = sample_batch(batch, generator, mode,
                            condition_on_beta=config.get('condition_on_beta', True))
            with torch.autocast('cuda', dtype=torch.bfloat16):
                prediction = model(history,raw,candidate,**kwargs)
            values = {'loss':float(state_loss(prediction['state'], target))}
            for label,value in (('baseline',candidate), ('history_hold',history[:,-1]),
                                ('restored',prediction['state'])):
                values.update({label+'_'+k:v for k,v in metrics(value,target).items()})
            values.update({'forecast_'+k:v for k,v in metrics(prediction['forecast'][:,0],future).items()})
            if not all(math.isfinite(v) for v in values.values()):
                raise FloatingPointError(f'Nonfinite evaluation at {protocol}/{source}/{mode}/{index}')
            n = len(target); count += n
            for key,value in values.items(): totals[key] = totals.get(key,0.)+value*n
            if index % 100 == 0:
                atomic_json(output/'status.json', {'utc':utc(), 'phase':'evaluating',
                            'protocol':protocol, 'dataset':source, 'mode':mode,
                            'batches_done':index+1, 'windows_done':count, 'windows_total':len(loader.dataset)})
        if count != len(loader.dataset): raise RuntimeError('Validation coverage mismatch')
        result[mode] = {k:v/count for k,v in totals.items()}
        result[mode]['samples'] = count
        print(json.dumps({'protocol':protocol, 'dataset':source, 'mode':mode,
                          'metrics':result[mode]}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--code-version', required=True)
    parser.add_argument('--protocol', choices=('fixed','full','both'), default='both')
    args = parser.parse_args()
    if not Path('/sciclone/home/jchen57').is_dir(): parser.error('Remote compute Pod only')
    if (args.output/'report.json').exists(): parser.error('Choose a fresh evaluation output')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    device = torch.device('cuda:0'); torch.cuda.set_device(device)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = checkpoint['config']; manifest_hash = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    if checkpoint['run']['manifest_sha256'] != manifest_hash:
        raise ValueError('Checkpoint data manifest differs; no silent evaluation dataset change')
    if any(not bool(torch.isfinite(t).all()) for t in checkpoint['model'].values() if t.is_floating_point()):
        raise FloatingPointError('Checkpoint contains nonfinite weights')
    model = TWM(TWMConfig(**config['model'])).to(device)
    model.load_state_dict(checkpoint['model'], strict=True); model.eval()
    report = {'phase':'evaluating', 'started_utc':utc(), 'pid':os.getpid(),
              'code_version':args.code_version, 'evaluator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'trainer_sha256':hashlib.sha256(Path(__file__).with_name('train_twm.py').read_bytes()).hexdigest(),
              'checkpoint':str(args.checkpoint), 'checkpoint_step':checkpoint['step'],
              'checkpoint_code_version':checkpoint['code_version'],
              'checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              'manifest':str(args.manifest), 'manifest_sha256':manifest_hash, 'config':config,
              'python':os.sys.executable, 'torch':torch.__version__, 'gpu':torch.cuda.get_device_name(device),
              'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
              'optimizer_updates':0, 'model_training':False, 'beta_conditioning':config.get('condition_on_beta',True),
              'metrics':'actual source wrist mm; root/finger geodesic degrees against fitted MANO reference',
              'limitations':['Synthetic current corruption and GT plus synthetic noisy history; no real HaWoR errors or own-output rollout.',
                            'Source-fitted MANO state reference, not independent native sensor/MPJPE/mesh or physical correctness.',
                            'Overlapping held-out windows; full validation is not an independent test or confidence interval.',
                            'Fixed and full window sets differ; their corruption samples have the same distribution, not per-frame identical assignments.',
                            'Last-history hold uses only past input, with the same synthetic history noise as the model.'],
              'protocols':{}}
    del checkpoint
    data_options = {k:config[k] for k in ('cache_sequences','cache_bytes') if k in config}
    val = TrajectoryWindows(args.manifest,'val',config['history_frames'],
                            config['forecast_offset_frames'],config['window_stride'],**data_options)
    if set(val.dataset_names) != set(config['required_datasets']): raise ValueError('Required source mismatch')
    report.update(source_revisions=val.source_revisions, val_windows_by_dataset=val.windows_by_dataset,
                  val_sequences_by_dataset=val.sequence_counts_by_dataset)
    batch = config.get('eval_batch_size', config['batch_size'])
    protocols = ('fixed','full') if args.protocol=='both' else (args.protocol,)
    start = time.monotonic()
    for protocol in protocols:
        maximum = batch*config['eval_batches'] if protocol=='fixed' else max(val.windows_by_dataset.values())
        indices = val.fixed_indices_by_dataset(maximum)
        by_dataset = {}
        for source,ids in indices.items():
            loader = DataLoader(Subset(val,ids),batch_size=batch,num_workers=0,pin_memory=True)
            by_dataset[source] = evaluate_source(model,loader,config,device,args.output,protocol,source)
            atomic_json(args.output/f'{protocol}_{source}.json', by_dataset[source])
        aggregate = {}
        for mode in ('noisy','missing','clean'):
            keys = next(iter(by_dataset.values()))[mode]
            aggregate[mode] = {key:sum(values[mode][key] for values in by_dataset.values())/len(by_dataset)
                               for key in keys if key!='samples'}
            aggregate[mode]['samples'] = sum(v[mode]['samples'] for v in by_dataset.values())
        report['protocols'][protocol] = {'by_dataset':by_dataset,'equal_source_mean':aggregate,
              'batch_size':batch, 'windows_by_dataset':{k:len(v) for k,v in indices.items()},
              'indices_sha256_by_dataset':{k:hashlib.sha256(json.dumps(v).encode()).hexdigest() for k,v in indices.items()},
              'corruption_seed':config['seed']+123, 'aggregation':'equal source mean; within-source mean weighted by sample count'}
        atomic_json(args.output/'report.partial.json',report)
    report.update(phase='complete', completed_utc=utc(), elapsed_seconds=time.monotonic()-start)
    atomic_json(args.output/'report.json', report)
    atomic_json(args.output/'status.json', {'utc':utc(), 'phase':'complete', 'checkpoint_step':report['checkpoint_step'],
               'protocols':list(report['protocols']), 'elapsed_seconds':report['elapsed_seconds']})
    print(json.dumps({'phase':'complete', 'checkpoint_step':report['checkpoint_step'],
                     'report':str(args.output/'report.json'), 'elapsed_seconds':report['elapsed_seconds']}),flush=True)


if __name__=='__main__': main()
