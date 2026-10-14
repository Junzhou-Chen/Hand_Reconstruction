#!/usr/bin/env python3
"""Remote CPU preparation of single-view DexYCB s0 train/val trajectories.

Read metadata and spool opaque NPZ bytes in one TAR pass. Establish official
lexicographic sequence indices before parsing train/val labels; never treat the eight views
as eight motions. Licensed MANO assets are required, with PCA45 + non-flat mean
decoded using the official manopth convention. Every valid frame is checked
against its supplied 21 camera-space joints before any sequence is accepted.

Source: NVlabs/dex-ycb-toolkit commit 64551b001d360ad83bc383157a559ec248fb9100;
dex_ycb.py (s0 split), examples/visualize_pose.py (MANO), hpe_eval.py (metres).
DexYCB paper section 2.1 specifies synchronized 30 fps. Frame IDs/30 below are
nominal relative timestamps, not measured capture timestamps or world poses.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import io
import json
import multiprocessing
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tarfile
import tempfile

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from handloop.data.meta_mano import MANOReferenceDecoder
from download_twm_data import now, save_json


REVISION = "64551b001d360ad83bc383157a559ec248fb9100"
SUBJECTS = (
    "20200709-subject-01", "20200813-subject-02", "20200820-subject-03",
    "20200903-subject-04", "20200908-subject-05", "20200918-subject-06",
    "20200928-subject-07", "20201002-subject-08", "20201015-subject-09",
    "20201022-subject-10",
)
SERIALS = ("836212060125", "839512060362", "840412060917", "841412060263",
           "932122060857", "932122060861", "932122061900", "932122062010")
LABEL_PATTERN = re.compile(r"labels_(\d{6})\.npz\Z")


def s0_split(subject: str, sequence_index: int) -> str:
    """Official s0 is recording-disjoint, although subjects overlap."""
    if subject not in SUBJECTS or not 0 <= sequence_index < 100:
        raise ValueError("Invalid official subject or sequence index")
    if sequence_index % 5 != 4:
        return "train"
    return "val" if SUBJECTS.index(subject) < 2 else "test"


def read_yaml(archive: tarfile.TarFile, member: tarfile.TarInfo) -> dict:
    if member.size > 1_000_000:
        raise ValueError(f"Unexpectedly large YAML metadata: {member.name}")
    value = yaml.safe_load(archive.extractfile(member).read())
    if not isinstance(value, dict):
        raise ValueError(f"YAML metadata is not a mapping: {member.name}")
    return value


def index_subject(root: Path, output: Path, subject: str, serial: str,
                  spool=None, max_frames: int = 0) -> tuple[dict, dict]:
    """One gzip pass: metadata plus opaque fixed-view NPZ bytes only.

    Spooling keeps peak RAM bounded and lets us establish all 100 sequence
    indices BEFORE parsing train/val labels. Held-out NPZ bytes are never
    opened with NumPy. Images/depth/segmentation arrays are never decoded.
    """
    archive_path = root / "archives" / (subject + ".tar.gz")
    stat = archive_path.stat()
    identity = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    receipts_path = root / "download_receipts.json"
    receipt = json.loads(receipts_path.read_text()).get("archives/" + archive_path.name, {}) if receipts_path.is_file() else {}
    if receipt and receipt.get("size") != stat.st_size:
        raise ValueError(f"Archive size differs from verified download receipt: {archive_path}")
    cache = output / "source_indices" / (subject + ".json")
    if cache.exists() and spool is None:
        indexed = json.loads(cache.read_text())
        if indexed.get("archive_identity") == identity:
            return indexed, {}
    sequences, label_records = {}, {}
    with tarfile.open(archive_path, "r|gz") as archive:
        for member in archive:
            parts = PurePosixPath(member.name).parts
            if member.isfile() and len(parts) == 3 and parts[0] == subject and parts[2] == "meta.yml":
                if parts[1] in sequences:
                    raise ValueError(f"Duplicate metadata for {subject}/{parts[1]}")
                sequences[parts[1]] = read_yaml(archive, member)
            elif (spool is not None and member.isfile() and len(parts) == 4
                  and parts[0] == subject and parts[2] == serial):
                match = LABEL_PATTERN.fullmatch(parts[3])
                if match is None:
                    continue
                frame_id = int(match[1])
                if max_frames and frame_id >= max_frames:
                    continue
                key = (parts[1], frame_id)
                if key in label_records:
                    raise ValueError(f"Duplicate label frame: {member.name}")
                offset = spool.tell()
                data = archive.extractfile(member).read()
                spool.write(data)
                label_records[key] = (offset, len(data))
    if len(sequences) != 100:
        raise ValueError(f"Official indexing requires all 100 sequences of {subject}, found {len(sequences)}")
    indexed = {"subject": subject, "archive": str(archive_path), "archive_identity": identity,
               "source_sha256": receipt.get("sha256"),
               "sequences": [{"name": name, "index": index, "split": s0_split(subject, index),
                              "meta": sequences[name]}
                             for index, name in enumerate(sorted(sequences))]}
    cache.parent.mkdir(parents=True, exist_ok=True)
    save_json(cache, indexed)
    return indexed, label_records


def load_calibrations(root: Path) -> dict:
    values = {}
    with tarfile.open(root / "archives/calibration.tar.gz", "r:gz") as archive:
        for member in archive:
            parts = PurePosixPath(member.name).parts
            if member.isfile() and len(parts) == 3 and parts[0] == "calibration" and parts[2] == "mano.yml":
                values[parts[1]] = np.asarray(read_yaml(archive, member)["betas"], dtype=np.float64)
    if not values:
        raise ValueError("No MANO shape calibration found")
    for key, beta in values.items():
        if beta.shape != (10,) or not np.isfinite(beta).all():
            raise ValueError(f"Invalid beta10 calibration {key}")
    return values


def save_sequence_arrays(output: Path, stem: str, arrays: dict, storage: str) -> str:
    """Publish either a real mmap-capable NPY directory or an ordinary NPZ."""
    if storage not in ("npy", "npz"):
        raise ValueError(f"Unknown trajectory storage: {storage}")
    if any(np.asarray(value).dtype.hasobject for value in arrays.values()):
        raise ValueError("Object arrays cannot be written as prepared trajectories")
    relative = stem if storage == "npy" else stem + ".npz"
    destination = output / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Prepared trajectory already exists: {destination}")
    if storage == "npy":
        temporary = Path(tempfile.mkdtemp(prefix=destination.name + ".tmp-", dir=destination.parent))
        try:
            for key, value in arrays.items():
                np.save(temporary / (key + ".npy"), value, allow_pickle=False)
            temporary.replace(destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    else:
        temporary = destination.with_suffix(".npz.tmp")
        try:
            with temporary.open("wb") as stream:
                np.savez(stream, **arrays)
            temporary.replace(destination)
        finally:
            if temporary.exists():
                temporary.unlink()
    return relative


def convert_subject(job: dict) -> dict:
    root, output = Path(job["root"]), Path(job["output"])
    if job["index_only"]:
        indexed, _ = index_subject(root, output, job["subject"], job["serial"])
        all_counts = dict(Counter(entry["split"] for entry in indexed["sequences"]))
        return {"subject": job["subject"], "indexed_split_counts": all_counts, "sequences": []}
    # Each subject gets one temporary annotation spool; no full image extraction.
    with tempfile.TemporaryFile(dir=output, prefix="dex_labels_") as spool:
        indexed, records = index_subject(root, output, job["subject"], job["serial"],
                                         spool, job["max_frames_per_sequence"])
        return decode_subject(job, indexed, records, spool)


def decode_subject(job: dict, indexed: dict, records: dict, spool) -> dict:
    root, output = Path(job["root"]), Path(job["output"])
    all_counts = dict(Counter(entry["split"] for entry in indexed["sequences"]))
    chosen = {}
    for split in ("train", "val"):
        entries = [entry for entry in indexed["sequences"] if entry["split"] == split]
        if job["max_sequences_per_split"]:
            entries = entries[:job["max_sequences_per_split"]]
        chosen.update((entry["name"], entry) for entry in entries)
    calibrations = load_calibrations(root)
    import torch
    torch.set_num_threads(1)
    decoder = MANOReferenceDecoder(job["mano_models"], device="cpu", fix_left_shapedirs=False,
                                   asset_version=job["mano_asset_version"])
    provenance = decoder.provenance()
    for entry in chosen.values():
        meta = entry["meta"]
        if len(meta["mano_sides"]) != 1 or len(meta["mano_calib"]) != 1:
            raise ValueError(f"Expected exactly one hand: {job['subject']}/{entry['name']}")
        if meta["mano_sides"][0] not in ("left", "right"):
            raise ValueError(f"Unknown hand side: {meta['mano_sides']}")
        if job["serial"] not in meta["serials"]:
            raise ValueError(f"Fixed camera {job['serial']} absent from {entry['name']}")
    labels = {name: {} for name in chosen}
    skipped = {name: Counter() for name in chosen}
    for (sequence, frame_id), (offset, size) in sorted(records.items()):
        if sequence not in chosen:
            continue  # Held-out NPZ bytes remain opaque and leave with the temporary spool.
        meta = chosen[sequence]["meta"]
        if not 0 <= frame_id < meta["num_frames"]:
            raise ValueError(f"Frame ID outside metadata range: {sequence}/{frame_id}")
        spool.seek(offset)
        with np.load(io.BytesIO(spool.read(size)), allow_pickle=False) as label:
            pose = label["pose_m"].copy()
            joints = label["joint_3d"].copy()
        if pose.shape != (1, 51) or joints.shape != (1, 21, 3):
            raise ValueError(f"Unexpected label shape in {sequence}/{frame_id}: {pose.shape}, {joints.shape}")
        if np.all(pose == 0):
            skipped[sequence]["pose_all_zero"] += 1
            continue
        if np.any(np.all(joints == -1, axis=-1)):
            skipped[sequence]["joint_sentinel"] += 1
            continue
        if not np.isfinite(pose).all() or not np.isfinite(joints).all():
            skipped[sequence]["nonfinite"] += 1
            continue
        labels[sequence][frame_id] = (pose[0], joints[0])
    results, rejected = [], []
    for sequence in sorted(chosen):
        entry, frames = chosen[sequence], labels[sequence]
        if not frames:
            rejected.append({"recording": job["subject"] + "/" + sequence,
                             "reason": "no_valid_frames", "skipped": dict(skipped[sequence])})
            continue
        ids = np.asarray(sorted(frames), dtype=np.int64)
        pose = np.stack([frames[int(index)][0] for index in ids])
        joints = np.stack([frames[int(index)][1] for index in ids])
        side = entry["meta"]["mano_sides"][0]
        calibration = "mano_" + entry["meta"]["mano_calib"][0]
        beta = calibrations[calibration]
        states, fk_errors, wrist_errors, rotation_errors = [], [], [], []
        for begin in range(0, len(ids), 64):
            batch_pose = pose[begin:begin + 64]
            decoded = decoder.decode_pca(batch_pose[:, 3:48], beta, batch_pose[:, :3],
                                         batch_pose[:, 48:51], np.full(len(batch_pose), int(side == "right")))
            decoded_state = decoded.state.cpu().numpy()
            # Reproduce manopth's DexYCB 21-joint order and its side-specific tips.
            tips = [745, 317, 444 if side == "right" else 445, 556, 673]
            all_joints = np.concatenate((decoded.joints_m.cpu().numpy(),
                                         decoded.vertices_m[:, tips].cpu().numpy()), axis=1)
            order = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
            decoded_joints = all_joints[:, order]
            reference = joints[begin:begin + 64]
            error = np.linalg.norm(decoded_joints - reference, axis=-1)
            fk_errors.extend(error.ravel().tolist())
            wrist_errors.extend(error[:, 0].tolist())
            if float(error.max()) > job["fk_tolerance_m"]:
                raise ValueError(f"MANO/Dex label disagreement {job['subject']}/{sequence}/{side}: "
                                 f"max {error.max():.8f} m exceeds {job['fk_tolerance_m']} m; "
                                 "check original MANO v1.2 asset, PCA mean/basis, side and units")
            six = decoded_state[:, 3:].reshape(-1, 16, 6)
            first, second = six[..., :3], six[..., 3:]
            rotations = np.stack((first, second, np.cross(first, second)), axis=-1)
            rotation_errors.append(float(np.max(np.abs(rotations.swapaxes(-1, -2) @ rotations - np.eye(3)))))
            # Joint 0 is the anatomical wrist. pose[:,48:51] alone is NOT it.
            decoded_state[:, :3] = reference[:, 0]
            states.append(decoded_state)
        stem = entry["split"] + "/" + job["subject"] + "++" + sequence + "__" + side
        arrays = {"state": np.concatenate(states).astype(np.float32),
                  "timestamps": ids.astype(np.float64) / 30.0, "beta": beta.astype(np.float32),
                  "frame_ids": ids, "joint_3d": joints.astype(np.float32),
                  "gt_mask": np.ones((len(ids), 17), dtype=np.bool_), "beta_valid": np.bool_(True),
                  "joint_valid": np.ones((len(ids), 21), dtype=np.bool_)}
        relative = save_sequence_arrays(output, stem, arrays, job["storage"])
        results.append({"path": relative, "storage": job["storage"], "split": entry["split"], "source": "dexycb",
                        "recording": job["subject"] + "/" + sequence, "subject": job["subject"],
                        "sequence_index": entry["index"], "handedness": int(side == "right"),
                        "source_archive_sha256": indexed.get("source_sha256"),
                        "frames": len(ids), "serial": job["serial"], "frame_id_minmax": [int(ids[0]), int(ids[-1])],
                        "frame_gaps": int(np.count_nonzero(np.diff(ids) != 1)),
                        "source_num_frames": entry["meta"]["num_frames"], "skipped": dict(skipped[sequence]),
                        "mano_calibration": calibration, "extrinsics_calibration": entry["meta"].get("extrinsics"),
                        "coordinate_frame": "fixed_camera_" + job["serial"], "units": "metres",
                        "nominal_dt": 1 / 30, "gt_complete": True,
                        "timestamp_source": "nominal_30hz_frame_index", "beta_span": 0.0,
                        "mano_asset_sha256": provenance["assets"][side]["sha256"], "fk_joints_max_m": max(fk_errors),
                        "fk_joints_mean_m": float(np.mean(fk_errors)), "fk_wrist_max_m": max(wrist_errors),
                        "rotation_orthogonality_error": max(rotation_errors)})
    return {"subject": job["subject"], "indexed_split_counts": all_counts, "sequences": results,
            "rejected": rejected, "decoder_provenance": provenance}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="downloaded dexycb directory with archives/")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mano-models", type=Path, help="licensed original MANO v1.2 model directory")
    parser.add_argument("--mano-asset-version", default=None, help="user-verified release label; otherwise unknown")
    parser.add_argument("--subjects", nargs="+", choices=SUBJECTS, default=list(SUBJECTS))
    parser.add_argument("--serial", choices=SERIALS, default=SERIALS[0])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--storage", choices=("npy", "npz"), default="npy",
                        help="npy: per-sequence directory for real mmap; npz: optional compatibility format")
    parser.add_argument("--index-only", action="store_true", help="index metadata without reading pose labels or requiring MANO")
    parser.add_argument("--max-sequences-per-split", type=int, default=0, help="per subject, 0 means all; bounded smoke only")
    parser.add_argument("--max-frames-per-sequence", type=int, default=0, help="first numeric frame IDs, 0 means all")
    parser.add_argument("--fk-tolerance-m", type=float, default=0.001, help="fail if any decoded/reference joint differs by more")
    args = parser.parse_args()
    if not Path("/sciclone/home/jchen57").is_dir():
        parser.error("Run data preprocessing only in the authorized remote ds Pod")
    if not args.index_only and args.mano_models is None:
        parser.error("--mano-models is required: PCA45 coefficients cannot be treated as axis-angle")
    if (args.workers < 1 or min(args.max_sequences_per_split, args.max_frames_per_sequence) < 0
            or args.fk_tolerance_m <= 0 or len(set(args.subjects)) != len(args.subjects)):
        parser.error("invalid worker/limit/tolerance/subject arguments")
    if not args.index_only:
        # Fail on missing assets/runtime BEFORE inflating any large subject TAR.
        import torch
        torch.set_num_threads(1)
        MANOReferenceDecoder(args.mano_models, device="cpu", fix_left_shapedirs=False,
                             asset_version=args.mano_asset_version)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "manifest.json").exists():
        parser.error("output already has manifest.json; choose a fresh output to preserve provenance")
    manifest = {"utc": now(), "phase": "preparing", "dataset": "dexycb", "setup": "s0",
                "storage": args.storage,
                "source": "official multiview RGB-D fitted MANO annotations; not independent MoCap",
                "source_revision": REVISION, "nominal_fps": 30, "rotation6d": "columns_c0xyz_c1xyz",
                "archive_sha256_source": "existing verified download receipts; not recomputed during preparation",
                "source_urls": ["https://github.com/NVlabs/dex-ycb-toolkit/tree/" + REVISION,
                                "https://dex-ycb.github.io/assets/chao_cvpr2021.pdf"],
                "position": "actual anatomical wrist J0 in metres in one fixed camera frame",
                "shape": "static subject/calibration beta10; original side-specific MANO without extra shape fixes",
                "timestamp_source": "numeric label frame index / nominal 30Hz; not measured timestamp",
                "serial": args.serial, "official_full_sequence_counts": {"train": 800, "val": 40, "test": 160},
                "split_rule": "sorted 100 sequences/subject: i%5!=4 train; remaining subject01/02 val, subject03-10 test",
                "subjects": args.subjects, "scope": "s0 train/val only; test labels never decoded; one camera per motion",
                "fk_tolerance_m": args.fk_tolerance_m, "fk_verified_against": "provided 21 fitted joints, not independent GT",
                "bounded_limits": {"max_sequences_per_split_per_subject": args.max_sequences_per_split,
                                   "max_frames_per_sequence": args.max_frames_per_sequence},
                "sequences": [], "subjects_summary": [], "decoder_provenance": None}
    jobs = [{"root": str(args.root), "output": str(args.output), "subject": subject,
             "mano_models": str(args.mano_models) if args.mano_models else None, "serial": args.serial,
             "mano_asset_version": args.mano_asset_version,
             "storage": args.storage,
             "index_only": args.index_only, "max_sequences_per_split": args.max_sequences_per_split,
             "max_frames_per_sequence": args.max_frames_per_sequence, "fk_tolerance_m": args.fk_tolerance_m}
            for subject in args.subjects]
    status_path = args.output / "preparation_status.json"
    save_json(status_path, {"phase": "indexing" if args.index_only else "preparing", "done": 0,
                            "total": len(jobs), "utc": now()})
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(convert_subject, job) for job in jobs]
        for done, future in enumerate(as_completed(futures), 1):
            try:
                result = future.result()
            except Exception as exc:
                save_json(status_path, {"phase": "failed", "done": done - 1, "total": len(jobs),
                                        "error_type": type(exc).__name__, "error": str(exc), "utc": now()})
                for pending in futures:
                    pending.cancel()
                raise
            manifest["sequences"].extend(result.pop("sequences"))
            provenance = result.pop("decoder_provenance", None)
            if provenance is not None:
                if manifest["decoder_provenance"] not in (None, provenance):
                    raise RuntimeError("Decoder/asset provenance changed between subjects")
                manifest["decoder_provenance"] = provenance
            manifest["subjects_summary"].append(result)
            status = {"phase": "indexing" if args.index_only else "preparing", "done": done,
                      "total": len(jobs), "last_subject": result["subject"], "utc": now()}
            save_json(status_path, status)
            print(json.dumps(status), flush=True)
    manifest["sequences"].sort(key=lambda entry: entry["path"])
    manifest["subjects_summary"].sort(key=lambda entry: entry["subject"])
    train = {entry["recording"] for entry in manifest["sequences"] if entry["split"] == "train"}
    val = {entry["recording"] for entry in manifest["sequences"] if entry["split"] == "val"}
    if train & val or any(entry["split"] == "test" for entry in manifest["sequences"]):
        raise RuntimeError("Split leakage detected")
    if not args.index_only and not manifest["sequences"]:
        raise RuntimeError("No valid prepared trajectories")
    manifest["phase"] = "indexed" if args.index_only else "complete"
    manifest["full_fk_verified"] = not args.index_only
    manifest["completed_utc"] = now()
    manifest["frames_by_split"] = {split: sum(entry["frames"] for entry in manifest["sequences"]
                                             if entry["split"] == split) for split in ("train", "val")}
    target = args.output / ("index_manifest.json" if args.index_only else "manifest.json")
    save_json(target, manifest)
    summary = {"phase": manifest["phase"], "subjects": len(jobs), "frames": manifest["frames_by_split"],
               "sequences": len(manifest["sequences"]), "manifest": str(target), "utc": now()}
    save_json(status_path, summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
