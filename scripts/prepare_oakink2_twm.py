#!/usr/bin/env python3
"""Remote-only OakInk2 quaternion-to-99D preparation, without MANO assets.

Source decoder: OakInk2 502a028..., manotorch use_pca=False,
flat_hand_mean=True, center_idx=0. Its wrist-centered FK output is translated
by tsl, in metres; hence tsl is actual world wrist xyz. Quaternion0 is root,
remaining fifteen are parent-local MANO rotations. This checks numerical
rotation/translation conventions, NOT full MANO FK/mesh accuracy.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from audit_twm_dataset_samples import AnnotationUnpickler, array
from download_twm_data import now, save_json


def quat_to_matrix(q):
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                     2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                     2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), -1).reshape(q.shape[:-1]+(3, 3))


def convert(job):
    root, output, recording, split, stride, source_hash = job
    source = Path(root) / "anno_preview" / (recording.replace("/", "++")+".pkl")
    with source.open("rb") as stream:
        value = AnnotationUnpickler(stream).load()
    records = value["raw_mano"]
    ids = sorted(records, key=int)
    ids = [i for i in ids if (int(i)-int(ids[0])) % stride == 0]
    entries = []
    for side, handedness in (("lh", 0), ("rh", 1)):
        states, times, betas = [], [], []
        norm_error, ortho_error, det_error = 0.0, 0.0, 0.0
        for index in ids:
            frame = records[index]
            keys = [side+"__pose_coeffs", side+"__tsl", side+"__betas"]
            if not all(k in frame for k in keys):
                continue
            q = array(frame[keys[0]]).reshape(16, 4).astype(np.float64)
            p = array(frame[keys[1]]).reshape(3).astype(np.float64)
            beta = array(frame[keys[2]]).reshape(10).astype(np.float64)
            if not all(np.isfinite(v).all() for v in (q, p, beta)):
                continue
            norms = np.linalg.norm(q, axis=-1)
            if np.any(norms < 1e-6) or np.max(np.abs(norms-1)) > 1e-3:
                raise ValueError(f"Invalid source quaternion: {recording}/{index}/{side}")
            rotation = quat_to_matrix(q)
            r6 = np.concatenate((rotation[..., :, 0], rotation[..., :, 1]), -1)
            states.append(np.concatenate((p, r6.reshape(96))).astype(np.float32))
            times.append(int(index)/120.0)  # official FPS_MOCAP; not RGB frame IDs.
            betas.append(beta)
            norm_error = max(norm_error, float(np.max(np.abs(norms-1))))
            ortho_error = max(ortho_error, float(np.max(np.abs(rotation.swapaxes(-1, -2)@rotation-np.eye(3)))))
            det_error = max(det_error, float(np.max(np.abs(np.linalg.det(rotation)-1))))
        if len(states) < 40:
            continue
        betas = np.asarray(betas)
        span = float(np.max(np.ptp(betas, axis=0)))
        if span > 1e-5:
            raise ValueError(f"Shape changes across recording {recording}/{side}: {span}")
        safe = recording.replace("/", "++")+"__"+side+".npz"
        relative = split+"/"+safe
        path = Path(output)/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".npz.tmp")
        with temporary.open("wb") as stream:
            np.savez(stream, state=np.asarray(states), timestamps=np.asarray(times),
                     beta=betas[0].astype(np.float32))
        temporary.replace(path)
        entries.append({"path": relative, "split": split, "recording": recording,
                        "handedness": handedness, "frames": len(states),
                        "source_sha256": source_hash, "beta_span": span,
                        "source_quaternion_norm_error": norm_error,
                        "rotation_orthogonality_error": ortho_error,
                        "rotation_determinant_error": det_error})
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-recordings", type=int, default=64)
    parser.add_argument("--val-recordings", type=int, default=8)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261014)
    args = parser.parse_args()
    if not Path("/sciclone/home/jchen57").is_dir():
        parser.error("Preprocess on the remote Pod, not the local machine")
    args.output.mkdir(parents=True, exist_ok=True)
    all_splits = {name: [s.strip() for s in (args.root/"splits"/(name+".txt")).read_text().splitlines() if s.strip()]
                  for name in ("train", "val", "test")}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if set(all_splits[a]) & set(all_splits[b]):
            raise ValueError("Official recording splits overlap")
        if {s.rsplit("/", 1)[-1] for s in all_splits[a]} & {s.rsplit("/", 1)[-1] for s in all_splits[b]}:
            raise ValueError("Recording hash/timestamp overlaps across splits")
    rng = np.random.default_rng(args.seed)
    selected = {"train": sorted(rng.choice(all_splits["train"], min(args.train_recordings, len(all_splits["train"])), replace=False).tolist()),
                "val": sorted(rng.choice(all_splits["val"], min(args.val_recordings, len(all_splits["val"])), replace=False).tolist())}
    source_manifest = json.loads((args.root/"download_manifest.json").read_text())
    hashes = {f["path"]: f["hash"] for f in source_manifest["files"]}
    jobs = [(str(args.root), str(args.output), name, split, args.stride,
             hashes["anno_preview/"+name.replace("/", "++")+".pkl"])
            for split, names in selected.items() for name in names]
    metadata = {"utc": now(), "phase": "preparing", "source": "OakInk2 official annotations / fitted MANO from MoCap",
                "source_revision": source_manifest["revision"], "source_fps": 120,
                "effective_fps": 120/args.stride, "rotation6d": "columns_c0xyz_c1xyz",
                "position": "actual world wrist xyz in metres; source centered MANO + tsl",
                "shape": "static beta10 separately stored", "full_fk_verified": False,
                "official_split_counts": {k: len(v) for k, v in all_splits.items()},
                "selected_recordings": selected, "seed": args.seed, "sequences": [],
                "scope": "recording-disjoint pilot subset; no test labels read"}
    save_json(args.output/"preparation_status.json", {"phase": "preparing", "done": 0, "total": len(jobs), "utc": now()})
    start = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(convert, job) for job in jobs]
        for done, future in enumerate(as_completed(futures), 1):
            metadata["sequences"].extend(future.result())
            status = {"phase": "preparing", "done": done, "total": len(jobs), "utc": now(),
                      "elapsed_seconds": round(time.monotonic()-start, 1)}
            save_json(args.output/"preparation_status.json", status)
            print(json.dumps(status), flush=True)
    metadata["phase"] = "complete"
    metadata["sequences"].sort(key=lambda e: e["path"])
    metadata["completed_utc"] = now()
    metadata["frames_by_split"] = {s: sum(e["frames"] for e in metadata["sequences"] if e["split"] == s) for s in selected}
    if not all(metadata["frames_by_split"].values()):
        raise RuntimeError("Empty train/val preparation")
    save_json(args.output/"manifest.json", metadata)
    save_json(args.output/"preparation_status.json", {"phase": "complete", "recordings": len(jobs), "frames": metadata["frames_by_split"], "utc": now()})
    print(json.dumps({"phase": "complete", "frames": metadata["frames_by_split"], "manifest": str(args.output/"manifest.json")}), flush=True)


if __name__ == "__main__":
    main()
