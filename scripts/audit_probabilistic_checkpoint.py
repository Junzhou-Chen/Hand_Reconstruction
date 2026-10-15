#!/usr/bin/env python3
"""Remote CPU checkpoint audit without instantiating or evaluating a model."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not Path('/sciclone/home/jchen57').is_dir(): parser.error('Run inside the authorized remote Pod')
    import torch
    path=args.run/'latest.pt'
    # Atomic replacement means this open/load refers to one complete version.
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    model_float=[v for v in checkpoint['model'].values() if v.is_floating_point()]
    optimizer_float=[v for s in checkpoint['optimizer']['state'].values() for v in s.values()
                     if torch.is_tensor(v) and v.is_floating_point()]
    if not all(bool(torch.isfinite(v).all()) for v in model_float+optimizer_float):
        raise FloatingPointError('Nonfinite checkpoint model or optimizer')
    digest=hashlib.sha256()
    for name,value in checkpoint['model'].items():
        digest.update((name+str(value.dtype)+str(tuple(value.shape))).encode())
        digest.update(value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    ranks=checkpoint['rank_states']
    if not all(v['model_state_sha256']==digest.hexdigest() for v in ranks):
        raise ValueError('CPU checkpoint bytes differ from saved replica audit')
    steps=sorted({float(v['step']) for v in checkpoint['optimizer']['state'].values() if 'step' in v})
    if steps!=[float(checkpoint['step'])]: raise ValueError('AdamW steps differ from checkpoint step')
    result={'step':checkpoint['step'],'code_version':checkpoint['code_version'],
            'model_float_tensors':len(model_float),'adamw_float_tensors':len(optimizer_float),
            'model_and_optimizer_finite':True,'model_state_sha256':digest.hexdigest(),
            'all_saved_rank_hashes_equal_cpu_checkpoint':True,'world_size':len(ranks),
            'adamw_steps':steps,'run':str(args.run),'config':checkpoint['config'],
            'manifest_sha256':checkpoint['run']['manifest_sha256'],
            'wandb':checkpoint['run'].get('wandb'),'parameters':checkpoint['run']['parameters'],
            'scope':'CPU checkpoint numerical/synchronization audit, not model quality evidence'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
