#!/usr/bin/env python3
"""Remote-only OakInk2 quaternion-to-99D preparation, optionally checking FK.

Source decoder: OakInk2 502a028..., manotorch use_pca=False,
flat_hand_mean=True, center_idx=0. Its wrist-centered FK output is translated
by tsl, in metres; hence tsl is actual world wrist xyz. Quaternion0 is root,
remaining fifteen are parent-local MANO rotations. Default NPZ and 64/8
recording selection preserve the original pilot. Use --storage npy and
--train-recordings 0 --val-recordings 0 for all official train/val recordings.

Assets are optional for this full-quaternion source. --mano-assets enables
actual joint/mesh FK roundtrip checks on up to 32 frames per hand sequence,
using shared official smplx layers and Oak's wrist-centering convention.
This checks geometry preservation through 99D serialization; it is NOT an
independent comparison with original manotorch outputs or physical ground truth.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from audit_twm_dataset_samples import AnnotationUnpickler, array
from download_twm_data import now, save_json


_FK_DECODERS = {}
_FK_MAX_SAMPLES = 32
_SOURCE_CONVENTION = {
    "implementation": "OakInk2 source manotorch ManoLayer",
    "source_revision": "502a02809b50b7f5d0037f92a1ccd2a479051ad1",
    "source_url": "https://github.com/oakink/OakInk2/blob/502a02809b50b7f5d0037f92a1ccd2a479051ad1/src/oakink2_preview/launch/viz/seg_3d.py",
    "rotation_mode": "quat", "quaternion": "wxyz", "use_pca": False,
    "flat_hand_mean": True, "center_idx": 0,
    "translation": "source tsl added after subtracting decoded wrist J0",
    "left_shapedirs_fix": False,
    "timestamp": "numeric raw_mano frame id / official FPS_MOCAP 120",
    "reference_quality": "source MoCap-fitted MANO; not independent sensor ground truth",
}


def check_fk_roundtrip(mano_assets, records, retained_ids, side, states, betas, tolerance_m):
    """Use actual MANO skinning for source-quaternion and serialized-state paths.

    Both paths share the same asset and official smplx implementation, so this
    is deliberately recorded as a roundtrip, not independent FK validation.
    The source path uses each frame's beta; the stored-state path uses beta[0],
    matching the existing static-beta export and its explicit span check.
    """
    import torch
    from handloop import geometry as g
    from handloop.data.meta_mano import MANOReferenceDecoder

    key = str(Path(mano_assets).resolve())
    if key not in _FK_DECODERS:
        torch.set_num_threads(1)
        _FK_DECODERS[key] = MANOReferenceDecoder(key, device="cpu", fix_left_shapedirs=False)
    decoder = _FK_DECODERS[key]
    name = "left" if side == "lh" else "right"
    chosen = np.linspace(0, len(states)-1, min(len(states), _FK_MAX_SAMPLES), dtype=int)
    quaternions = np.stack([array(records[retained_ids[i]][side+"__pose_coeffs"]).reshape(16, 4)
                            for i in chosen])
    translations = np.stack([array(records[retained_ids[i]][side+"__tsl"]).reshape(3)
                             for i in chosen])
    with torch.no_grad():
        source_pose = g.quaternion_to_axis_angle(torch.as_tensor(quaternions, dtype=torch.float32)).reshape(-1, 48)
        state = torch.as_tensor(states[chosen], dtype=torch.float32)
        stored_pose = g.matrix_to_axis_angle(g.state_to_rotmat(state)).reshape(-1, 48)
        source_beta = torch.as_tensor(betas[chosen], dtype=torch.float32)
        stored_beta = torch.as_tensor(betas[0], dtype=torch.float32).expand(len(chosen), -1)
        source_wrist = torch.as_tensor(translations, dtype=torch.float32)

        def decode(pose, beta, wrist):
            # Shared layers have flat_hand_mean=False; subtract that mean
            # before calling them so their added mean yields Oak's full pose.
            output = decoder.layers[name](
                betas=beta, global_orient=pose[:, :3],
                hand_pose=pose[:, 3:]-decoder.means[name],
                transl=torch.zeros_like(wrist), return_verts=True, return_full_pose=True,
            )
            if not torch.allclose(output.full_pose, pose, rtol=1e-5, atol=1e-6):
                raise RuntimeError("Oak flat-mean convention was not preserved by shared decoder")
            joints = output.joints[:, :16]
            center = joints[:, :1]
            return joints-center+wrist[:, None], output.vertices-center+wrist[:, None]

        source_joints, source_vertices = decode(source_pose, source_beta, source_wrist)
        stored_joints, stored_vertices = decode(stored_pose, stored_beta, state[:, :3])
        joint_errors = (source_joints-stored_joints).norm(dim=-1)
        vertex_errors = (source_vertices-stored_vertices).norm(dim=-1)
        if not (torch.isfinite(joint_errors).all() and torch.isfinite(vertex_errors).all()):
            raise ValueError("Nonfinite MANO geometry in Oak FK roundtrip")
        joint_max, vertex_max = float(joint_errors.max()), float(vertex_errors.max())
        if max(joint_max, vertex_max) > tolerance_m:
            raise ValueError(f"Oak FK roundtrip exceeds {tolerance_m} m: joints={joint_max}, mesh={vertex_max}")
    return {
        "checked": True, "frames_checked": len(chosen),
        "sampling": f"up to {_FK_MAX_SAMPLES} evenly spaced available frames per hand sequence",
        "frame_ids": [int(retained_ids[i]) for i in chosen],
        "joint_max_m": joint_max, "joint_mean_m": float(joint_errors.mean()),
        "vertex_max_m": vertex_max, "vertex_mean_m": float(vertex_errors.mean()),
        "tolerance_m": tolerance_m, "decoder_provenance": decoder.provenance(),
        "scope": "actual shared-smplx FK, source quaternion vs exported99D; not independent source/GT agreement",
        "wrist_note": "wrist=tsl is imposed by prescribed centering, not independently calibrated here",
    }


def save_arrays(path, arrays, storage):
    """Atomic NPZ or individual NPY writes; publish the manifest only at end."""
    if storage == "npz":
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".npz.tmp")
        with temporary.open("wb") as stream:
            np.savez(stream, **arrays)
        temporary.replace(path)
    else:
        path.mkdir(parents=True, exist_ok=True)
        for name, value in arrays.items():
            target = path / f"{name}.npy"
            temporary = path / f"{name}.npy.tmp-{os.getpid()}"
            with temporary.open("wb") as stream:
                np.save(stream, value, allow_pickle=False)
            temporary.replace(target)


def quat_to_matrix(q):
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                     2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                     2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), -1).reshape(q.shape[:-1]+(3, 3))


def convert(job):
    root, output, recording, split, stride, source_hash, storage, mano_assets, fk_tolerance = job
    source = Path(root) / "anno_preview" / (recording.replace("/", "++")+".pkl")
    with source.open("rb") as stream:
        value = AnnotationUnpickler(stream).load()
    records = value["raw_mano"]
    ids = sorted(records, key=int)
    if not ids:
        return []
    ids = [i for i in ids if (int(i)-int(ids[0])) % stride == 0]
    entries = []
    for side, handedness in (("lh", 0), ("rh", 1)):
        states, times, betas, retained_ids = [], [], [], []
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
            retained_ids.append(index)
            norm_error = max(norm_error, float(np.max(np.abs(norms-1))))
            ortho_error = max(ortho_error, float(np.max(np.abs(rotation.swapaxes(-1, -2)@rotation-np.eye(3)))))
            det_error = max(det_error, float(np.max(np.abs(np.linalg.det(rotation)-1))))
        if len(states) < 40:
            continue
        betas = np.asarray(betas)
        span = float(np.max(np.ptp(betas, axis=0)))
        if span > 1e-5:
            raise ValueError(f"Shape changes across recording {recording}/{side}: {span}")
        states = np.asarray(states)
        fk = check_fk_roundtrip(mano_assets, records, retained_ids, side, states, betas,
                                fk_tolerance) if mano_assets else {"checked": False}
        safe = recording.replace("/", "++")+"__"+side+(".npz" if storage == "npz" else "")
        relative = split+"/"+safe
        path = Path(output)/relative
        save_arrays(path, {
            "state": states, "timestamps": np.asarray(times), "beta": betas[0].astype(np.float32),
            "gt_mask": np.ones((len(states), 17), dtype=np.bool_),
            "beta_valid": np.asarray(True, dtype=np.bool_),
            "frame_ids": np.asarray([int(index) for index in retained_ids], dtype=np.int64),
        }, storage)
        entries.append({"path": relative, "split": split, "recording": recording,
                        "dataset": "oakink2", "source": "OakInk2 official raw_mano / MoCap-fitted MANO",
                        "coordinate_frame": "world", "nominal_dt": stride/120.0,
                        "storage": storage, "gt_complete": True, "partial_gt": False,
                        "handedness": handedness, "frames": len(states),
                        "source_sha256": source_hash, "beta_span": span,
                        "source_path": str(source), "source_convention": _SOURCE_CONVENTION,
                        "fk_roundtrip": fk, "full_fk_verified": False,
                        "source_quaternion_norm_error": norm_error,
                        "rotation_orthogonality_error": ortho_error,
                        "rotation_determinant_error": det_error})
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-recordings", type=int, default=64, help="0 selects all official train recordings")
    parser.add_argument("--val-recordings", type=int, default=8, help="0 selects all official val recordings")
    parser.add_argument("--storage", choices=("npz", "npy"), default="npz", help="npy is a memory-mappable directory per sequence")
    parser.add_argument("--mano-assets", type=Path, help="optional directory containing licensed MANO_LEFT/RIGHT.pkl for actual FK roundtrip")
    parser.add_argument("--fk-tolerance-m", type=float, default=1e-5)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261014)
    args = parser.parse_args()
    if not Path("/sciclone/home/jchen57").is_dir():
        parser.error("Preprocess on the remote Pod, not the local machine")
    if (args.train_recordings < 0 or args.val_recordings < 0 or args.stride < 1 or
            args.workers < 1 or not np.isfinite(args.fk_tolerance_m) or args.fk_tolerance_m <= 0):
        parser.error("recording counts must be >=0; stride/workers/tolerance must be positive")
    if args.mano_assets:
        for hand in ("LEFT", "RIGHT"):
            if not (args.mano_assets / f"MANO_{hand}.pkl").is_file():
                parser.error(f"Missing licensed MANO_{hand}.pkl in --mano-assets")
    args.output.mkdir(parents=True, exist_ok=True)
    all_splits = {name: [s.strip() for s in (args.root/"splits"/(name+".txt")).read_text().splitlines() if s.strip()]
                  for name in ("train", "val", "test")}
    if any(len(values) != len(set(values)) for values in all_splits.values()):
        raise ValueError("Duplicate recording in an official split list")
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if set(all_splits[a]) & set(all_splits[b]):
            raise ValueError("Official recording splits overlap")
        if {s.rsplit("/", 1)[-1] for s in all_splits[a]} & {s.rsplit("/", 1)[-1] for s in all_splits[b]}:
            raise ValueError("Recording hash/timestamp overlaps across splits")
    rng = np.random.default_rng(args.seed)
    def select(name, count):
        if count == 0:
            return sorted(all_splits[name])
        return sorted(rng.choice(all_splits[name], min(count, len(all_splits[name])), replace=False).tolist())
    selected = {"train": select("train", args.train_recordings), "val": select("val", args.val_recordings)}
    full_selection = {name: len(values) == len(all_splits[name]) for name, values in selected.items()}
    source_manifest = json.loads((args.root/"download_manifest.json").read_text())
    hashes = {f["path"]: f["hash"] for f in source_manifest["files"]}
    jobs = [(str(args.root), str(args.output), name, split, args.stride,
             hashes["anno_preview/"+name.replace("/", "++")+".pkl"], args.storage,
             str(args.mano_assets.resolve()) if args.mano_assets else None, args.fk_tolerance_m)
            for split, names in selected.items() for name in names]
    metadata = {"utc": now(), "phase": "preparing", "dataset": "oakink2",
                "source": "OakInk2 official annotations / fitted MANO from MoCap",
                "source_revision": source_manifest["revision"], "source_fps": 120,
                "effective_fps": 120/args.stride, "rotation6d": "columns_c0xyz_c1xyz",
                "coordinate_frame": "world", "nominal_dt": args.stride/120.0,
                "storage": args.storage, "gt_complete": True, "partial_gt": False,
                "gt_mask_groups": "translation, root rotation, 15 MANO local rotations",
                "position": "actual world wrist xyz in metres; source centered MANO + tsl",
                "shape": "static beta10 separately stored", "full_fk_verified": False,
                "source_convention": _SOURCE_CONVENTION,
                "fk_roundtrip_requested": bool(args.mano_assets), "fk_tolerance_m": args.fk_tolerance_m,
                "official_split_counts": {k: len(v) for k, v in all_splits.items()},
                "full_official_selection": full_selection,
                "selected_recordings": selected, "seed": args.seed, "sequences": [],
                "scope": ("full official train+val recording selection; no test labels read"
                          if all(full_selection.values()) else
                          "recording-disjoint selected train/val subset (full flags recorded); no test labels read")}
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
    checked = [entry["fk_roundtrip"] for entry in metadata["sequences"] if entry["fk_roundtrip"]["checked"]]
    metadata["fk_roundtrip_checked"] = bool(checked) and len(checked) == len(metadata["sequences"])
    metadata["fk_roundtrip_summary"] = {
        "sequences_checked": len(checked), "frames_checked": sum(value["frames_checked"] for value in checked),
        "joint_max_m": max((value["joint_max_m"] for value in checked), default=None),
        "vertex_max_m": max((value["vertex_max_m"] for value in checked), default=None),
        "independent_source_fk_verified": False,
    }
    save_json(args.output/"manifest.json", metadata)
    save_json(args.output/"preparation_status.json", {"phase": "complete", "recordings": len(jobs), "frames": metadata["frames_by_split"], "utc": now()})
    print(json.dumps({"phase": "complete", "frames": metadata["frames_by_split"], "manifest": str(args.output/"manifest.json")}), flush=True)


if __name__ == "__main__":
    main()
