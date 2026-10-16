#!/usr/bin/env python3
"""Remote CPU audit of a saved full joint checkpoint, without model execution."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not Path('/sciclone/data10/jchen57').is_dir():
        raise RuntimeError('Audit experiment tensors on the remote compute Pod')
    torch.set_num_threads(2)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    result = {'utc': datetime.now(timezone.utc).isoformat(), 'checkpoint': str(args.checkpoint),
              'bytes': args.checkpoint.stat().st_size, 'step': saved['step'], 'code_version': saved['code_version']}
    digest, count, finite = hashlib.sha256(), 0, True
    for name, tensor in sorted(saved['model'].items()):
        if tensor.is_floating_point():
            count += 1; finite &= bool(torch.isfinite(tensor).all())
        digest.update((name + str(tensor.dtype) + str(tuple(tensor.shape))).encode())
        digest.update(tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    result.update(model_floating_tensors=count, model_finite=finite, model_sha256=digest.hexdigest())
    state_count, finite, steps = 0, True, set()
    for state in saved['optimizer']['state'].values():
        for tensor in state.values():
            if isinstance(tensor, torch.Tensor) and tensor.is_floating_point():
                state_count += 1; finite &= bool(torch.isfinite(tensor).all())
        if 'step' in state: steps.add(int(state['step']))
    result.update(optimizer_floating_tensors=state_count, optimizer_finite=finite,
                  optimizer_parameter_states=len(saved['optimizer']['state']), optimizer_distinct_steps=sorted(steps),
                  optimizer_groups=[{k:v for k,v in x.items() if k != 'params'} for x in saved['optimizer']['param_groups']],
                  source_twm_step=saved['twm_source']['step'], source_twm_sha256=saved['twm_source']['sha256'],
                  position_feature_scale=saved['twm_source']['model_config']['position_feature_scale'])
    hashes = {item['model_sha256'] for item in saved['rank_states']}
    result['cpu_matches_all_saved_ranks'] = hashes == {result['model_sha256']}
    result['passed'] = result['model_finite'] and result['optimizer_finite'] and result['cpu_matches_all_saved_ranks']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k:result[k] for k in ('passed', 'step', 'bytes', 'model_floating_tensors',
                     'optimizer_floating_tensors', 'cpu_matches_all_saved_ranks')}))
    if not result['passed']: raise RuntimeError('Saved joint checkpoint audit failed')


if __name__ == '__main__':
    main()
