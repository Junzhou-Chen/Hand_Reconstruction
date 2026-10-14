#!/usr/bin/env python3
"""Build and inspect a complete four-source prepared trajectory collection.

Does not read test labels, alter source splits, fit missing MANO states, or
claim equal source sampling is optimal. Run in the authorized compute Pod.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from handloop.data.trajectory import TrajectoryWindows

SOURCES = ('oakink2', 'hot3d', 'umetrack', 'dexycb')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in SOURCES: parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--code-version',required=True)
    args=parser.parse_args()
    if not Path('/sciclone/home/jchen57').is_dir():
        parser.error('Run collection preparation in the remote compute Pod')
    references=[]
    for name in SOURCES:
        path=getattr(args,name).resolve()
        data=json.loads(path.read_text())
        if data.get('phase')!='complete' or data.get('dataset')!=name:
            raise ValueError(f'{name}: manifest must be complete and explicitly identify its dataset')
        if not data.get('sequences'):
            raise ValueError(f'{name}: no prepared trajectories')
        references.append({'path':str(path),'dataset':name,
                           'manifest_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    collection={'phase':'complete','prepared_utc':datetime.now(timezone.utc).isoformat(),
                'code_version':args.code_version,'rotation6d':'columns_c0xyz_c1xyz',
                'manifests':references,'required_datasets':list(SOURCES),
                'scope':'four-source full-MANO train/validation; source splits and coordinates retained',
                'sampling_hypothesis':'equal source probability, uniform windows within source; not validated as optimal',
                'position':'actual wrist J0 in metres; source world or fixed camera coordinates, anchored per sample',
                'partial_gt_policy':'not fitted or imputed; this trainer requires full MANO supervision'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    pending=args.output.with_suffix('.candidate.json')
    pending.write_text(json.dumps(collection,indent=2)+'\n')
    try:
        counts={}
        for split in ('train','val'):
            dataset=TrajectoryWindows(pending,split)
            if set(dataset.dataset_names)!=set(SOURCES):
                raise ValueError('Collection must expose all four sources')
            for name in SOURCES:
                if not dataset.windows_by_dataset[name]:
                    raise ValueError(f'{name}: no contiguous {split} windows')
            # Materialize one real window from every source. All labels are
            # checked finite when each of these sequences enters the cache.
            for ids in dataset.fixed_indices_by_dataset(1).values():
                dataset[ids[0]]
            counts[split]={'windows':dataset.windows_by_dataset,
                           'sequences':dataset.sequence_counts_by_dataset,
                           'source_revisions':dataset.source_revisions}
        collection['loader_inspection']=counts
        pending.write_text(json.dumps(collection,indent=2)+'\n')
        pending.replace(args.output)
    except Exception:
        pending.unlink(missing_ok=True)
        raise
    print(json.dumps({'manifest':str(args.output),'inspection':counts}),flush=True)


if __name__=='__main__': main()
