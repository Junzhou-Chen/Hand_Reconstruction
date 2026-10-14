#!/usr/bin/env python3
"""Remote CPU preparation of HOT3D/UmeTrack real-train MANO trajectories.

Read only JSON labels/shape/info from plain TARs; image payloads are never
decoded or extracted. Meta PCA15 is decoded using licensed MANO assets and
official smplx through MANOReferenceDecoder. Write immutable per-hand NPY
directories suitable for mmap, or one uncompressed NPZ per hand to reduce
small-file metadata traffic. Missing MANO rows are omitted while their
original times and independent native/MANO availability remain auditable.

The internal train/val split is an engineering choice, NOT an official split:
HOT groups by source sequence_id, UmeTrack by subject, with seeded ~10% val.
The complete grouping is established before applying an optional smoke limit.
HOT's HaWoR evaluation parents are checked before any labels are read.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tarfile
import time
import traceback
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from handloop.data.meta_mano import MANOReferenceDecoder


SOURCES = {
    "hot3d": ("bop-benchmark/hot3d", "30fe9674782f32e1e5edba98476b6ff4300132c5"),
    "umetrack": ("facebook/hand_tracking_challenge_umetrack", "19f6b2cad9eb17be4ebedfe4503203980a25e016"),
}
FRAME_FILE = re.compile(r"(\d+)\.(hands|info|cameras)\.json\Z")
UME_FILE = re.compile(r"subject_(\d+)_(separate_hand|hand_hand)_(\d+)\.tar\Z")
_DECODER = None


def now():
    return datetime.now(timezone.utc).isoformat()


def save_json(path: Path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def file_hash(path: Path, kind="sha256"):
    digest = hashlib.sha1() if kind == "git_blob_sha1" else hashlib.sha256()
    if kind == "git_blob_sha1":
        digest.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_metadata(root, entries, name):
    entry = entries[name]
    path = root / name
    if (path.stat().st_size != entry["size"]
            or file_hash(path, entry["hash_kind"]) != entry["hash"]):
        raise ValueError(f"Pinned metadata checksum mismatch: {name}")
    return path


def make_plan(root: Path, dataset: str, seed: int, val_fraction: float, limit: int):
    source_manifest = json.loads((root / "download_manifest.json").read_text())
    repo, revision = SOURCES[dataset]
    if (source_manifest["dataset"], source_manifest["repo"], source_manifest["revision"]) != (dataset, repo, revision):
        raise ValueError("Dataset does not match the audited pinned release")
    receipt = json.loads((root / "download_status.json").read_text())
    if (receipt["phase"] != "complete" or receipt["counts"]["failed"] != 0
            or receipt["counts"]["verified"] != len(source_manifest["files"])):
        raise ValueError("Require the completed, verified source download receipt")
    entries = {entry["path"]: entry for entry in source_manifest["files"]}
    if len(entries) != len(source_manifest["files"]):
        raise ValueError("Duplicate source paths in download manifest")
    excluded, definitions, official_train = [], {}, set()
    if dataset == "hot3d":
        definitions = json.loads(checked_metadata(root, entries, "clip_definitions.json").read_text())
        splits = json.loads(checked_metadata(root, entries, "clip_splits.json").read_text())
        official_train = {int(cid) for values in splits["train"].values() for cid in values}
        excluded = source_manifest["split_summary"]["hawor_excluded_recordings"]
        readme = checked_metadata(root, entries, "provenance/hawor_README.md").read_text()
        command = readme.split("--sequence_name", 1)[1].split("```", 1)[0]
        readme_excluded = set(re.findall(r"P\d{4}_[0-9a-f]{8}", command))
        if len(readme_excluded) != 27 or readme_excluded != set(excluded):
            raise ValueError("HaWoR evaluation parent exclusion list disagrees with pinned source")

    jobs, timestamps_seen = [], set()
    for path, entry in sorted(entries.items()):
        if not path.endswith(".tar"):
            continue
        safe = PurePosixPath(path)
        if safe.is_absolute() or ".." in safe.parts:
            raise ValueError(f"Unsafe source path: {path}")
        if entry["hash_kind"] != "sha256":
            raise ValueError(f"Expected official LFS SHA256 for TAR {path}")
        if not (root / path).is_file() or (root / path).stat().st_size != entry["size"]:
            raise ValueError(f"Source TAR missing or size changed after download: {path}")
        if dataset == "hot3d":
            if safe.parent.as_posix() not in ("train_aria", "train_quest3"):
                raise ValueError(f"Only official train TARs are allowed: {path}")
            clip_id = int(safe.stem.split("-")[-1])
            definition = definitions[str(clip_id)]
            parent, device = definition["sequence_id"], definition["device"]
            expected_device = "Aria" if safe.parent.as_posix() == "train_aria" else "Quest3"
            if (clip_id not in official_train or parent in excluded or device != expected_device
                    or entry.get("sequence_id") != parent or entry.get("device") != device):
                raise ValueError(f"HOT source/split/exclusion metadata mismatch: {path}")
            group, subject = parent, parent.split("_", 1)[0]
            reference_stream = "214-1" if device == "Aria" else "1201-1"
            expected_times = [frame[reference_stream] for frame in definition["per_frame_timestamps_ns"]]
            if len(expected_times) != 150 or any(b <= a for a, b in zip(expected_times, expected_times[1:])):
                raise ValueError(f"Invalid HOT timestamp definition: {path}")
            for timestamp in expected_times:
                key = (parent, int(timestamp))
                if key in timestamps_seen:
                    raise ValueError(f"Duplicate physical frame across HOT clips: {key}")
                timestamps_seen.add(key)
        else:
            match = UME_FILE.fullmatch(safe.name)
            if safe.parent.as_posix() != "train" or match is None:
                raise ValueError(f"Only audited Ume real training TARs are allowed: {path}")
            subject = "subject_" + match[1]
            parent, group, device, expected_times = safe.stem, subject, None, None
        jobs.append({"dataset": dataset, "source_path": path, "recording": parent,
                     "subject": subject, "split_group": group, "device": device,
                     "source_sha256": entry["hash"], "source_bytes": entry["size"],
                     "expected_timestamps_ns": expected_times})
    groups = sorted({job["split_group"] for job in jobs})
    if len(groups) < 2:
        raise ValueError("Need at least two independent groups for internal train/val")
    nval = max(1, min(len(groups) - 1, int(round(len(groups) * val_fraction))))
    val_groups = set(np.random.default_rng(seed).choice(groups, nval, replace=False).tolist())
    for job in jobs:
        job["split"] = "val" if job["split_group"] in val_groups else "train"
    summary = {"rule": "engineering split of official train; not official validation",
               "group_key": "source sequence_id" if dataset == "hot3d" else "subject prefix",
               "seed": seed, "requested_val_fraction": val_fraction,
               "train_groups": [g for g in groups if g not in val_groups],
               "val_groups": sorted(val_groups), "all_train_tars": len(jobs),
               "all_tars_by_split": dict(Counter(job["split"] for job in jobs)),
               "hawor_excluded_recordings": sorted(excluded),
               "hot_unique_reference_frames": len(timestamps_seen) if dataset == "hot3d" else None,
               "cross_dataset_motion_deduplication": "not established; no cross-dataset disjointness claim"}
    if limit:
        # A bounded two-clip smoke should exercise both splits. Full grouping
        # and all leak checks above are invariant to this diagnostic limit.
        sides = [[job for job in jobs if job["split"] == split] for split in ("train", "val")]
        interleaved = []
        for i in range(max(map(len, sides))):
            for subset in sides:
                if i < len(subset):
                    interleaved.append(subset[i])
        jobs = interleaved[:limit]
    summary["selected_tars"] = len(jobs)
    summary["selected_tars_by_split"] = dict(Counter(job["split"] for job in jobs))
    return source_manifest, receipt, jobs, summary


def init_worker(mano_models, asset_version):
    global _DECODER
    import torch
    torch.set_num_threads(1)
    _DECODER = MANOReferenceDecoder(mano_models, device="cpu", fix_left_shapedirs=True,
                                   asset_version=asset_version)


def read_json(archive, member, digest):
    if not member.isfile() or member.size > 8 * 1024 * 1024:
        raise ValueError(f"Unexpected JSON member type/size: {member.name}")
    payload = archive.extractfile(member).read()
    digest.update(member.name.encode() + b"\0" + len(payload).to_bytes(8, "big") + payload)
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {member.name}")
    return value


def native_valid(pose):
    """Schema/finite availability only, not an accuracy or FK quality score."""
    if not isinstance(pose, dict):
        return False
    try:
        transform = pose["T_world_from_wrist"]
        angles = np.asarray(pose["joint_angles"], dtype=np.float64)
        translation = np.asarray(transform["translation_xyz"], dtype=np.float64)
        quaternion = np.asarray(transform["quaternion_wxyz"], dtype=np.float64)
        return bool(angles.shape in ((20,), (22,)) and translation.shape == (3,)
                    and quaternion.shape == (4,) and np.isfinite(angles).all()
                    and np.isfinite(translation).all() and np.isfinite(quaternion).all()
                    and abs(np.linalg.norm(quaternion) - 1) < 1e-3)
    except (KeyError, TypeError, ValueError):
        return False


def save_arrays(directory: Path, arrays):
    directory.mkdir(parents=True, exist_ok=False)
    checksums = {}
    for name, value in arrays.items():
        destination = directory / (name + ".npy")
        temporary = directory / (name + ".npy.tmp")
        with temporary.open("wb") as stream:
            np.save(stream, value, allow_pickle=False)
        temporary.replace(destination)
        checksums[name] = {"sha256": file_hash(destination), "bytes": destination.stat().st_size,
                           "shape": list(np.asarray(value).shape), "dtype": str(np.asarray(value).dtype)}
    return checksums


def save_npz(destination: Path, arrays, *, reuse_equivalent_files=False):
    """Save all arrays, or retain an existing file only after exact verification."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    exists = destination.exists()
    if exists and not reuse_equivalent_files:
        raise FileExistsError(destination)
    checksums = {}
    for name, value in arrays.items():
        value = np.asarray(value)
        if value.dtype.hasobject:
            raise ValueError(f"Object arrays are forbidden: {name}")
        checksums[name] = {
            "npz_member": name + ".npy",
            "content_sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
            "content_hash_encoding": "C-order ndarray data bytes; dtype and shape recorded separately",
            "nbytes": value.nbytes, "shape": list(value.shape), "dtype": str(value.dtype)}
    if exists:
        with np.load(destination, allow_pickle=False) as stored:
            if len(stored.files) != len(arrays) or set(stored.files) != set(arrays):
                raise ValueError(f"Existing NPZ keys differ: {destination}")
            for name, value in arrays.items():
                value, previous = np.asarray(value), stored[name]
                if (previous.dtype != value.dtype or previous.shape != value.shape
                        or previous.tobytes(order="C") != value.tobytes(order="C")):
                    raise ValueError(f"Existing NPZ array differs: {destination}:{name}")
        with zipfile.ZipFile(destination) as archive:
            if any(member.compress_type != zipfile.ZIP_STORED for member in archive.infolist()):
                raise ValueError(f"Existing NPZ is not uncompressed: {destination}")
    else:
        temporary = destination.with_name(destination.name + f".tmp.{os.getpid()}")
        with temporary.open("xb") as stream:
            np.savez(stream, **arrays)
        temporary.replace(destination)
    artifact = {"sha256": file_hash(destination), "bytes": destination.stat().st_size,
                "format": "npz", "compression": "ZIP_STORED",
                "members": [name + ".npy" for name in arrays],
                "reused_existing_equivalent": exists}
    return checksums, artifact


def convert(job):
    if _DECODER is None:
        raise RuntimeError("Worker decoder was not initialized")
    source = Path(job["root"]) / job["source_path"]
    digest = hashlib.sha256()
    with tarfile.open(source, "r:") as archive:
        members, files = {}, {kind: {} for kind in ("hands", "info", "cameras")}
        for member in archive.getmembers():
            match = FRAME_FILE.fullmatch(member.name)
            if match:
                frame_id, kind = int(match[1]), match[2]
                if frame_id in files[kind]:
                    raise ValueError(f"Duplicate {kind} frame {frame_id}: {source}")
                files[kind][frame_id] = member
            if member.name == "__hand_shapes.json__":
                if "shape" in members:
                    raise ValueError(f"Duplicate hand shape: {source}")
                members["shape"] = member
        if "shape" not in members:
            raise ValueError(f"Missing MANO beta10 shape: {source}")
        shape = read_json(archive, members["shape"], digest)
        beta = np.asarray(shape["mano"], dtype=np.float32)
        if beta.shape != (10,) or not np.isfinite(beta).all():
            raise ValueError(f"Invalid shared MANO beta10: {source}")
        ids = sorted(set().union(*(set(items) for items in files.values())))
        if not ids:
            raise ValueError(f"No source frame IDs in TAR: {source}")
        all_ids = np.asarray(ids, dtype=np.int64)
        times, hands, absolute_ns = [], [], []
        for frame_id in ids:
            hands.append(read_json(archive, files["hands"][frame_id], digest)
                         if frame_id in files["hands"] else {})
            if job["dataset"] == "hot3d":
                if frame_id not in files["info"]:
                    raise ValueError(f"HOT frame has no exact timestamp: {source}/{frame_id}")
                info = read_json(archive, files["info"][frame_id], digest)
                if (info["sequence_id"] != job["recording"] or info["device"] != job["device"]
                        or info["participant_id"] != job["subject"]):
                    raise ValueError(f"HOT frame parent/device mismatch: {source}/{frame_id}")
                timestamp = info["ref_timestamp_ns"]
                if isinstance(timestamp, bool) or not isinstance(timestamp, int):
                    raise ValueError("ref_timestamp_ns must be an exact integer")
                expected = job["expected_timestamps_ns"]
                if not 0 <= frame_id < len(expected) or timestamp != expected[frame_id]:
                    raise ValueError(f"HOT actual/definition timestamp mismatch: {source}/{frame_id}")
                absolute_ns.append(timestamp)
                times.append(timestamp * 1e-9)
            else:
                times.append(frame_id / 30.0)
        all_times = np.asarray(times, dtype=np.float64)
        if not np.isfinite(all_times).all() or np.any(np.diff(all_times) <= 0):
            raise ValueError(f"Non-monotonic source time: {source}")

    output, entries, audits = Path(job["output"]), [], []
    for side_index, side in enumerate(("left", "right")):
        present, valid, native_present, native_ok = [], [], [], []
        theta, wrist, selected = [], [], []
        reasons = Counter()
        for index, hand in enumerate(hands):
            label = hand.get(side)
            if label is not None and not isinstance(label, dict):
                raise ValueError(f"Unexpected hand label type: {source}/{ids[index]}/{side}")
            label = label or {}
            pose = label.get("mano_pose")
            present.append(pose is not None)
            native = label.get("umetrack_pose")
            native_present.append(native is not None)
            native_ok.append(native_valid(native))
            if pose is None:
                valid.append(False)
                reasons["mano_absent"] += 1
                continue
            if not isinstance(pose, dict):
                raise ValueError(f"Unexpected MANO label type: {source}/{ids[index]}/{side}")
            angles = np.asarray(pose.get("thetas", []), dtype=np.float32)
            transform = np.asarray(pose.get("wrist_xform", []), dtype=np.float32)
            if angles.shape != (15,) or transform.shape != (6,):
                raise ValueError(f"Expected Meta PCA15/xform6: {source}/{ids[index]}/{side}")
            finite = bool(np.isfinite(angles).all() and np.isfinite(transform).all())
            valid.append(finite)
            if not finite:
                reasons["mano_nonfinite"] += 1
                continue
            theta.append(angles)
            wrist.append(transform)
            selected.append(index)
        mano_present = np.asarray(present, dtype=np.bool_)
        mano_valid_mask = np.asarray(valid, dtype=np.bool_)
        native_present = np.asarray(native_present, dtype=np.bool_)
        native_valid_mask = np.asarray(native_ok, dtype=np.bool_)
        audit = {"source_path": job["source_path"], "recording": job["recording"],
                 "handedness": side_index, "source_frames": len(ids),
                 "mano_present": int(mano_present.sum()), "mano_valid": int(mano_valid_mask.sum()),
                 "native_present": int(native_present.sum()), "native_valid": int(native_valid_mask.sum()),
                 "mano_valid_native_absent": int(np.count_nonzero(mano_valid_mask & ~native_present)),
                 "mano_valid_native_invalid": int(np.count_nonzero(mano_valid_mask & ~native_valid_mask)),
                 "skipped": dict(reasons)}
        audits.append(audit)
        if not selected:
            audit["prepared"] = False
            continue
        selected = np.asarray(selected, dtype=np.int64)
        theta, wrist = np.stack(theta), np.stack(wrist)
        chunks, orthogonality_error, determinant_error = [], 0.0, 0.0
        wrist_offset_min, wrist_offset_max = float("inf"), 0.0
        for begin in range(0, len(selected), job["batch_size"]):
            end = min(begin + job["batch_size"], len(selected))
            decoded = _DECODER.decode_meta(theta[begin:end], wrist[begin:end], beta,
                                           np.full(end - begin, side_index, dtype=np.int64))
            state = decoded.state.cpu().numpy()
            chunks.append(state)
            columns = state[:, 3:].reshape(-1, 16, 6)
            rotation = np.stack((columns[..., :3], columns[..., 3:],
                                 np.cross(columns[..., :3], columns[..., 3:])), axis=-1)
            orthogonality_error = max(orthogonality_error, float(np.abs(rotation.swapaxes(-1, -2) @ rotation - np.eye(3)).max()))
            determinant_error = max(determinant_error, float(np.abs(np.linalg.det(rotation) - 1).max()))
            offsets = np.linalg.norm(state[:, :3] - wrist[begin:end, 3:], axis=-1)
            wrist_offset_min, wrist_offset_max = min(wrist_offset_min, float(offsets.min())), max(wrist_offset_max, float(offsets.max()))
        if max(orthogonality_error, determinant_error) > 1e-4:
            raise ValueError(f"Decoded rotation numerical check failed: {source}/{side}")
        relative = job["split"] + "/" + Path(job["source_path"]).stem + "__" + side
        selected_ids, selected_times = all_ids[selected], all_times[selected]
        arrays = {"state": np.concatenate(chunks).astype(np.float32),
                  "timestamps": selected_times, "beta": beta,
                  "gt_mask": np.ones((len(selected), 17), dtype=np.bool_),
                  "beta_valid": np.bool_(True), "frame_ids": selected_ids,
                  "native_available": native_valid_mask[selected],
                  "source_frame_ids": all_ids, "source_timestamps": all_times,
                  "source_mano_present": mano_present, "source_mano_valid": mano_valid_mask,
                  "source_native_present": native_present, "source_native_valid": native_valid_mask}
        if absolute_ns:
            arrays["source_ref_timestamps_ns"] = np.asarray(absolute_ns, dtype=np.int64)
            arrays["ref_timestamps_ns"] = arrays["source_ref_timestamps_ns"][selected]
        storage = job.get("storage", "npy")
        artifact = None
        if storage == "npz":
            relative += ".npz"
            hashes, artifact = save_npz(output / relative, arrays,
                                       reuse_equivalent_files=job.get("reuse_equivalent_files", False))
        elif storage == "npy":
            hashes = save_arrays(output / relative, arrays)
        else:
            raise ValueError(f"Unknown storage format: {storage}")
        entries.append({"path": relative, "storage": "npz" if storage == "npz" else "npy_directory", "dataset": job["dataset"],
                        "source": "Meta marker-based native trajectory fitted to MANO",
                        "split": job["split"], "split_group": job["split_group"],
                        "recording": job["recording"], "subject": job["subject"],
                        "handedness": side_index, "frames": len(selected),
                        "source_path": job["source_path"], "source_sha256": job["source_sha256"],
                        "source_bytes": job["source_bytes"], "annotation_payload_sha256": digest.hexdigest(),
                        "source_frames": len(ids), "source_shape_keys": sorted(shape),
                        "coordinate_frame": "source_recording_world", "units": "metres",
                        "nominal_dt": 1 / 30, "gt_complete": True, "partial_gt": False,
                        "timestamp_source": "ref_timestamp_ns" if absolute_ns else "frame_id/30_nominal",
                        "frame_id_minmax": [int(selected_ids[0]), int(selected_ids[-1])],
                        "frame_gaps": int(np.count_nonzero(np.diff(selected_ids) != 1)),
                        "time_gaps_over_1_5_nominal_dt": int(np.count_nonzero(np.diff(selected_times) >= 1.5 / 30)),
                        "beta_span": 0.0, "availability": audit.copy(), "arrays": hashes,
                        "rotation_orthogonality_error": orthogonality_error,
                        "rotation_determinant_error": determinant_error,
                        "wrist_minus_mano_transl_norm_range_m": [wrist_offset_min, wrist_offset_max]})
        if artifact is not None:
            entries[-1]["storage_artifact"] = artifact
        audit["prepared"] = True
    return {"source_path": job["source_path"], "sequences": entries, "hands": audits,
            "source_frames": len(ids), "source_frame_id_gaps": int(np.count_nonzero(np.diff(all_ids) != 1)),
            "annotation_payload_sha256": digest.hexdigest(),
            "decoder_provenance": _DECODER.provenance()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=SOURCES, required=True)
    parser.add_argument("--root", type=Path, required=True, help="dataset directory containing download_manifest.json")
    parser.add_argument("--output", type=Path, required=True, help="fresh prepared output directory")
    parser.add_argument("--mano-models", type=Path, required=True)
    parser.add_argument("--mano-asset-version", default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--storage", choices=("npy", "npz"), default="npy",
                        help="npy: per-array mmap files; npz: one uncompressed file per hand")
    parser.add_argument("--reuse-equivalent-files", action="store_true",
                        help="NPZ only: recompute every TAR/FK and retain existing files only when every array is bitwise identical")
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=0, help="total TARs after full split; 2 selects one train and one val; 0=all")
    args = parser.parse_args()
    if not Path("/sciclone/home/jchen57").is_dir():
        parser.error("Data preparation runs on the existing remote ds Pod, not the local machine")
    if args.workers < 1 or args.batch_size < 1 or args.limit < 0 or not 0 < args.val_fraction < 1:
        parser.error("positive workers/batch-size, limit>=0 and 0<val-fraction<1 required")
    if args.reuse_equivalent_files and args.storage != "npz":
        parser.error("--reuse-equivalent-files requires --storage npz")
    args.root, args.output = args.root.resolve(), args.output.resolve()
    args.mano_models = args.mano_models.resolve()
    output_nonempty = args.output.exists() and any(args.output.iterdir())
    if output_nonempty and not args.reuse_equivalent_files:
        parser.error("Choose a fresh output directory to preserve immutable preparation provenance")
    source, receipt, jobs, split = make_plan(args.root, args.dataset, args.seed, args.val_fraction, args.limit)
    # Fail before creating outputs or workers when assets/runtime are missing.
    init_worker(str(args.mano_models), args.mano_asset_version)
    provenance = _DECODER.provenance()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"utc": now(), "phase": "preparing", "dataset": args.dataset,
                "source": "Meta marker-based native trajectory fitted to MANO; not direct MANO sensor measurements",
                "source_revision": source["revision"], "source_repo": source["repo"],
                "source_manifest_sha256": file_hash(args.root / "download_manifest.json"),
                "source_download_receipt": receipt,
                "source_archive_hash_policy": "pinned manifest SHA256 + completed download receipt; size rechecked, full TAR not rehashed",
                "coordinate_frame": "source_recording_world", "units": "metres", "nominal_dt": 1 / 30,
                "rotation6d": "columns_c0xyz_c1xyz", "position": "actual MANO wrist joints[:,0] in source world metres",
                "pose": "hand-specific PCA15 projection + nonflat hand mean exactly once; root axis-angle directly from source",
                "shape": "beta10 per TAR, shared by both sides and frames, separately stored",
                "timestamp_policy": "HOT actual ref_timestamp_ns; Ume original frame_id/30 nominal relative time; missing frames do not compress clock",
                "gt_mask_policy": "17 true groups only for finite complete MANO labels; native validity independent; no imputed GT",
                "availability_policy": "source_* arrays cover all discovered frame IDs; native_valid tests schema/finite values only, not accuracy",
                "native_supervision": "not consumed by this MANO-only preparation; source profiles/poses remain in immutable original TAR",
                "split": split, "bounded_limit": args.limit,
                "reuse_equivalent_files": args.reuse_equivalent_files,
                "reuse_policy": "Every selected source TAR is reread and full FK/arrays recomputed; existing NPZ reused only after exact key/dtype/shape/C-order-byte comparison; mismatch fails",
                "workers": args.workers, "batch_size": args.batch_size,
                "storage": args.storage,
                "storage_policy": ("immutable per-hand NPY directory; per-file SHA256"
                                   if args.storage == "npy" else
                                   "immutable per-hand uncompressed NPZ; all arrays retained; container SHA256 plus per-array C-order content SHA256"),
                "decoder_provenance": provenance, "full_fk_executed": False,
                "independent_fk_equivalence_verified": False,
                "validation_boundary": "official smplx FK and numerical rotations; no native/MANO error-equivalence or empirical model accuracy claim",
                "source_urls": [f"https://huggingface.co/datasets/{source['repo']}/tree/{source['revision']}",
                                "https://github.com/facebookresearch/hand_tracking_toolkit/tree/" + provenance["meta_toolkit_revision"],
                                "https://arxiv.org/html/2211.00099v1#S4"],
                "preparation_script_sha256": file_hash(Path(__file__)),
                "decoder_script_sha256": file_hash(Path(sys.modules[MANOReferenceDecoder.__module__].__file__)),
                "preparation_python": sys.version, "preparation_numpy": np.__version__,
                "sequences": [], "clips": []}
    if output_nonempty:
        previous_plan = args.output / "preparation_plan.json"
        if not previous_plan.is_file():
            raise ValueError("Reuse requires the previous preparation_plan.json provenance")
        previous = json.loads(previous_plan.read_text())
        for key in ("dataset", "source_repo", "source_revision", "source_manifest_sha256", "split",
                    "storage", "decoder_provenance", "decoder_script_sha256"):
            if previous.get(key) != manifest[key]:
                raise ValueError(f"Reuse provenance differs for {key}; choose a fresh output directory")
        # Retain prior plan/status/complete manifest before publishing this run.
        # A preparation-script change is recorded, never a reason to skip FK.
        history = args.output / "previous_preparations" / f"{time.time_ns()}_{os.getpid()}"
        history.mkdir(parents=True, exist_ok=False)
        snapshots = {}
        for name in ("preparation_plan.json", "preparation_status.json", "manifest.json"):
            original = args.output / name
            if original.is_file():
                destination = history / name
                shutil.copyfile(original, destination)
                snapshots[name] = {"path": str(destination.relative_to(args.output)),
                                   "sha256": file_hash(destination)}
        manifest["previous_metadata_snapshots"] = snapshots
    save_json(args.output / "preparation_plan.json", manifest)
    for job in jobs:
        job.update(root=str(args.root), output=str(args.output), batch_size=args.batch_size,
                   storage=args.storage, reuse_equivalent_files=args.reuse_equivalent_files)
    started, last_report = time.monotonic(), 0.0
    try:
        with ProcessPoolExecutor(max_workers=args.workers,
                                 mp_context=multiprocessing.get_context("spawn"),
                                 initializer=init_worker,
                                 initargs=(str(args.mano_models), args.mano_asset_version)) as pool:
            futures = [pool.submit(convert, job) for job in jobs]
            try:
                for done, future in enumerate(as_completed(futures), 1):
                    result = future.result()
                    if result.pop("decoder_provenance") != provenance:
                        raise ValueError("Worker asset/runtime provenance differs")
                    manifest["sequences"].extend(result.pop("sequences"))
                    manifest["clips"].append(result)
                    status = {"phase": "preparing", "done": done, "total": len(jobs),
                              "elapsed_seconds": round(time.monotonic() - started, 1), "utc": now()}
                    save_json(args.output / "preparation_status.json", status)
                    if done == 1 or done == len(jobs) or time.monotonic() - last_report > 20:
                        print(json.dumps(status), flush=True)
                        last_report = time.monotonic()
            except BaseException:
                for pending in futures:
                    pending.cancel()
                raise
    except BaseException as exc:
        save_json(args.output / "preparation_status.json",
                  {"phase": "failed", "utc": now(), "error": repr(exc), "traceback": traceback.format_exc()})
        raise
    manifest["sequences"].sort(key=lambda entry: entry["path"])
    manifest["clips"].sort(key=lambda entry: entry["source_path"])
    if not manifest["sequences"]:
        save_json(args.output / "preparation_status.json",
                  {"phase": "failed", "utc": now(), "error": "No finite complete MANO trajectories prepared"})
        raise RuntimeError("No finite complete MANO trajectories prepared")
    train = {entry["split_group"] for entry in manifest["sequences"] if entry["split"] == "train"}
    val = {entry["split_group"] for entry in manifest["sequences"] if entry["split"] == "val"}
    if train & val or any(entry["recording"] in split["hawor_excluded_recordings"] for entry in manifest["sequences"]):
        save_json(args.output / "preparation_status.json",
                  {"phase": "failed", "utc": now(), "error": "Prepared group leakage"})
        raise RuntimeError("Prepared group leakage")
    manifest["phase"], manifest["completed_utc"] = "complete", now()
    manifest["full_fk_executed"] = True
    reused = sum(entry.get("storage_artifact", {}).get("reused_existing_equivalent", False)
                 for entry in manifest["sequences"])
    manifest["revalidation_counts"] = {
        "source_tars_reread_and_fully_decoded": len(manifest["clips"]),
        "hand_sequences_recomputed": len(manifest["sequences"]),
        "arrays_recomputed": sum(len(entry["arrays"]) for entry in manifest["sequences"]),
        "existing_npz_sequences_verified_and_retained": reused,
        "new_hand_sequence_artifacts_written": len(manifest["sequences"]) - reused}
    manifest["frames_by_split"] = {s: sum(entry["frames"] for entry in manifest["sequences"] if entry["split"] == s)
                                   for s in ("train", "val")}
    all_audits = [audit for clip in manifest["clips"] for audit in clip["hands"]]
    manifest["availability_counts"] = {key: sum(audit[key] for audit in all_audits)
                                       for key in ("source_frames", "mano_present", "mano_valid", "native_present", "native_valid",
                                                   "mano_valid_native_absent", "mano_valid_native_invalid")}
    manifest["availability_counts"]["source_hand_frames"] = manifest["availability_counts"].pop("source_frames")
    manifest["source_time_frames"] = sum(clip["source_frames"] for clip in manifest["clips"])
    save_json(args.output / "manifest.json", manifest)
    status = {"phase": "complete", "utc": now(), "clips": len(jobs),
              "sequences": len(manifest["sequences"]), "frames": manifest["frames_by_split"],
              "manifest": str(args.output / "manifest.json")}
    save_json(args.output / "preparation_status.json", status)
    print(json.dumps(status), flush=True)


if __name__ == "__main__":
    main()
