#!/usr/bin/env python3
"""Remote CPU preparation of TACO V1 complete MANO trajectories.

Only official train recordings are decoded. Optional engineering validation is
selected deterministically at ORIGINAL sequence level within official train;
the two hands always remain together. Official test_1..test_4 are never decoded.

TACO's vendored manopth uses use_pca=False, flat_hand_mean=True, center_idx=0,
no left shapedirs correction. The source loader adds hand_trans AFTER centering
on joint zero and converting millimetres back to metres. Thus hand_trans is the
actual wrist J0, NOT native uncentred MANO transl. Pose48 is full axis-angle.
Preserve the source static world frame, both hands' original shapes, numeric
frame IDs, and gaps. Time = original frame ID / nominal 30 Hz, not sensor time.

The required licensed MANO assets enable sampled source-AA -> state99D -> AA
FK roundtrips with shared smplx. This is geometry-preservation verification,
not an independent comparison against TACO's original manopth or physical GT.
No RGB, object annotations, fitting, interpolation or native-label imputation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import io
import json
import multiprocessing
from pathlib import Path
import re
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from audit_twm_dataset_samples import AnnotationUnpickler, array
from download_twm_data import now, save_json
from prepare_dexycb_twm import save_sequence_arrays


HF_REVISION = "fa8ed8b4cf5d46949b90ef852128811a2854066a"
HAND_ARCHIVE_SHA256 = "db649d185ae818e63e9f14577067c1318d0751d35b213de8a06688637a1ece51"
RECORDING_PATTERN = re.compile(r"\d{8}_\d+\Z")
FRAME_PATTERN = re.compile(r"(?:hand_)?(\d+)(?:\.pkl)?\Z")
EXPECTED_SOURCE_RECORDINGS = 2317
ADAPTER_VERSION = 1
SOURCE_CONVENTION = {
    "implementation": "TACO-Instructions dataset_utils/hand_pose_loader.py and vendored manopth",
    "use_pca": False, "flat_hand_mean": True, "center_idx": 0,
    "left_shapedirs_fix": False, "rotation_mode": "full48_axis_angle_radians",
    "translation": "hand_trans in metres added after wrist-J0 centering; already actual wrist",
    "coordinate_frame": "source_static_world", "source_units": "metres",
    "timestamp_source": "original numeric frame key / nominal synchronized capture 30 Hz",
    "reference_quality": "multi-view visual/contact/temporal MANO fit, not hand-marker MoCap GT",
}
_DECODERS = {}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_official_lists(split_file: Path, sequence_list: Path) -> tuple[dict, dict]:
    """Read metadata only, before opening any hand pickle, including test data."""
    splits = {}
    for number, raw in enumerate(split_file.read_text().splitlines(), 1):
        raw = raw.strip()
        if not raw:
            continue
        fields = raw.split(",")
        if len(fields) != 2:
            raise ValueError(f"Invalid official split line {number}: {raw!r}")
        recording, split = (item.strip() for item in fields)
        if not RECORDING_PATTERN.fullmatch(recording) or split not in (
                "train", "test_1", "test_2", "test_3", "test_4"):
            raise ValueError(f"Unknown official sequence/split at line {number}: {raw!r}")
        if recording in splits:
            raise ValueError(f"Duplicate official split entry: {recording}")
        splits[recording] = split
    triplets = {}
    for number, raw in enumerate(sequence_list.read_text().splitlines(), 1):
        raw = raw.strip()
        if not raw:
            continue
        try:
            triplet, recording = raw.rsplit(maxsplit=1)
        except ValueError as exc:
            raise ValueError(f"Invalid official sequence-list line {number}") from exc
        if not RECORDING_PATTERN.fullmatch(recording) or not (triplet.startswith("(") and triplet.endswith(")")):
            raise ValueError(f"Invalid official sequence-list line {number}: {raw!r}")
        if "/" in triplet or "\\" in triplet or recording in triplets:
            raise ValueError(f"Unsafe/duplicate official recording: {recording}")
        triplets[recording] = triplet
    if set(triplets) != set(splits) or len(splits) != EXPECTED_SOURCE_RECORDINGS:
        raise ValueError("Expected the matching, complete 2317-entry official TACO V1 lists")
    return splits, triplets


def engineering_split(recording: str, fraction: float, seed: int) -> str:
    """Stable recording grouping, independent of annotations, length or outcome."""
    value = hashlib.sha256(f"taco-v1-validation:{seed}:{recording}".encode()).digest()
    return "val" if int.from_bytes(value[:8], "big") / 2**64 < fraction else "train"


def load_trusted_pickle(path: Path):
    """TACO tensors may have been serialized on CUDA; load them onto CPU only."""
    payload = path.read_bytes()
    return AnnotationUnpickler(io.BytesIO(payload)).load(), hashlib.sha256(payload).hexdigest()


def numeric_frame_id(key) -> int:
    if isinstance(key, (int, np.integer)) and not isinstance(key, (bool, np.bool_)):
        result = int(key)
    elif isinstance(key, str) and (match := FRAME_PATTERN.fullmatch(key)):
        result = int(match[1])
    else:
        raise ValueError(f"Unsupported original frame key {key!r}; do not infer time from sorted row order")
    if result < 0:
        raise ValueError("Negative frame index")
    return result


def get_decoder(models: str, asset_version: str | None):
    from handloop.data.meta_mano import MANOReferenceDecoder
    key = (str(Path(models).resolve()), asset_version)
    if key not in _DECODERS:
        import torch
        torch.set_num_threads(1)
        _DECODERS[key] = MANOReferenceDecoder(models, device="cpu", fix_left_shapedirs=False,
                                             asset_version=asset_version)
    return _DECODERS[key]


def fk_roundtrip(decoder, side: str, pose: np.ndarray, beta: np.ndarray,
                 state: np.ndarray, ids: np.ndarray, tolerance: float,
                 sample_count: int) -> dict:
    """Real FK, with full source pose and exactly one wrist-centering operation."""
    import torch
    from handloop import geometry as g
    chosen = np.unique(np.linspace(0, len(state) - 1, min(len(state), sample_count), dtype=int))
    source_pose = torch.as_tensor(pose[chosen], dtype=torch.float32)
    serialized = torch.as_tensor(state[chosen], dtype=torch.float32)
    reconstructed_pose = g.matrix_to_axis_angle(g.state_to_rotmat(serialized)).reshape(-1, 48)
    expanded_beta = torch.as_tensor(beta, dtype=torch.float32).expand(len(chosen), -1)

    def decode(full_pose, wrist):
        # Shared layers add a non-flat hand mean. Remove that mean before
        # calling them so their full_pose equals TACO's flat-mean pose48.
        output = decoder.layers[side](
            betas=expanded_beta, global_orient=full_pose[:, :3],
            hand_pose=full_pose[:, 3:] - decoder.means[side],
            transl=torch.zeros_like(wrist), return_verts=True, return_full_pose=True)
        if not torch.allclose(output.full_pose, full_pose, atol=1e-6, rtol=1e-5):
            raise RuntimeError("Shared MANO decoder did not preserve TACO full flat-mean pose")
        joints = output.joints[:, :16]
        return (joints - joints[:, :1] + wrist[:, None],
                output.vertices - joints[:, :1] + wrist[:, None])

    with torch.no_grad():
        source_joints, source_vertices = decode(source_pose, serialized[:, :3])
        stored_joints, stored_vertices = decode(reconstructed_pose, serialized[:, :3])
        joint_error = (source_joints - stored_joints).norm(dim=-1)
        vertex_error = (source_vertices - stored_vertices).norm(dim=-1)
        if not all(torch.isfinite(value).all() for value in
                   (source_joints, source_vertices, stored_joints, stored_vertices)):
            raise ValueError("Nonfinite FK geometry")
        maximum = max(float(joint_error.max()), float(vertex_error.max()))
        if maximum > tolerance:
            raise ValueError(f"TACO state99D FK roundtrip {maximum:.8g}m exceeds {tolerance}m")
    return {"checked": True, "frames_checked": len(chosen),
            "sampled_frame_ids": ids[chosen].tolist(), "joint_max_m": float(joint_error.max()),
            "joint_mean_m": float(joint_error.mean()), "vertex_max_m": float(vertex_error.max()),
            "vertex_mean_m": float(vertex_error.mean()), "tolerance_m": tolerance,
            "scope": "shared smplx source full-AA versus exported99D; not independent source/manopth agreement",
            "wrist_note": "J0=hand_trans by official source centering; not independently calibrated here"}


def convert_recording(job: dict) -> dict:
    import torch
    from handloop import geometry as g
    output, directory = Path(job["output"]), Path(job["directory"])
    decoder = get_decoder(job["mano_models"], job["mano_asset_version"])
    result = {"recording": job["recording"], "split": job["split"], "sequences": [], "rejected": []}
    for side in ("left", "right"):
        pose_file = directory / f"{side}_hand.pkl"
        shape_file = directory / f"{side}_hand_shape.pkl"
        if not pose_file.is_file() or not shape_file.is_file():
            raise FileNotFoundError(f"Missing real TACO pose/shape; no zero-beta imputation: {directory}/{side}")
        metadata_path = output / "sequence_receipts" / f"{job['recording']}__{side}.json"
        identity = {"adapter_version": ADAPTER_VERSION, "source_revision": job["source_revision"],
                    "source_hf_revision": HF_REVISION, "split": job["split"],
                    "source_pose": {"path": str(pose_file), "size": pose_file.stat().st_size,
                                    "mtime_ns": pose_file.stat().st_mtime_ns},
                    "source_shape": {"path": str(shape_file), "size": shape_file.stat().st_size,
                                     "mtime_ns": shape_file.stat().st_mtime_ns},
                    "decoder_provenance": decoder.provenance(), "storage": job["storage"],
                    "max_frames": job["max_frames"], "min_frames": job["min_frames"],
                    "fk_samples": job["fk_samples"], "fk_tolerance_m": job["fk_tolerance_m"]}
        if job["resume"] and metadata_path.is_file():
            receipt = json.loads(metadata_path.read_text())
            if receipt.get("identity") != identity:
                raise ValueError(f"Resume provenance differs: {metadata_path}")
            entry = receipt.get("entry")
            if entry is None:
                result["rejected"].append(receipt["rejected"])
                continue
            target = output / entry["path"]
            if not target.exists():
                raise FileNotFoundError(f"Resume receipt has missing trajectory: {target}")
            result["sequences"].append(entry)
            continue
        if metadata_path.exists():
            raise FileExistsError(f"Existing trajectory receipt; use --resume or a fresh output: {metadata_path}")
        source, pose_sha = load_trusted_pickle(pose_file)
        shape, shape_sha = load_trusted_pickle(shape_file)
        if not isinstance(source, dict) or not isinstance(shape, dict) or "hand_shape" not in shape:
            raise ValueError(f"Unexpected TACO pickle structure: {directory}/{side}")
        beta = np.asarray(array(shape["hand_shape"]), dtype=np.float64)
        if beta.size != 10 or not np.isfinite(beta).all():
            raise ValueError(f"Missing/invalid true beta10: {shape_file}")
        beta = beta.reshape(10)
        frame_map = {}
        for key, value in source.items():
            frame_id = numeric_frame_id(key)
            if frame_id in frame_map:
                raise ValueError(f"Frame keys collide after numeric parse: {pose_file}/{key}")
            frame_map[frame_id] = value
        original_ids = sorted(frame_map)
        selected_ids = original_ids[:job["max_frames"]] if job["max_frames"] else original_ids
        poses, wrists, ids, skipped = [], [], [], Counter()
        for frame_id in selected_ids:
            frame = frame_map[frame_id]
            if not isinstance(frame, dict) or not all(key in frame for key in ("hand_pose", "hand_trans")):
                skipped["missing_complete_pose_or_translation"] += 1
                continue
            pose = np.asarray(array(frame["hand_pose"]), dtype=np.float64)
            wrist = np.asarray(array(frame["hand_trans"]), dtype=np.float64)
            if pose.size != 48 or wrist.size != 3:
                raise ValueError(f"Expected pose48/trans3, got {pose.shape}/{wrist.shape}: {pose_file}/{frame_id}")
            if not np.isfinite(pose).all() or not np.isfinite(wrist).all():
                skipped["nonfinite_complete_pose_or_translation"] += 1
                continue
            poses.append(pose.reshape(48))
            wrists.append(wrist.reshape(3))
            ids.append(frame_id)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        if len(ids) < job["min_frames"]:
            rejection = {"recording": job["recording"], "side": side, "reason": "too_few_valid_frames",
                         "frames": len(ids), "minimum": job["min_frames"], "skipped": dict(skipped),
                         "source_pose_sha256": pose_sha, "source_shape_sha256": shape_sha}
            save_json(metadata_path, {"identity": identity, "entry": None, "rejected": rejection})
            result["rejected"].append(rejection)
            continue
        poses, wrists, ids = np.asarray(poses), np.asarray(wrists), np.asarray(ids, dtype=np.int64)
        with torch.no_grad():
            rotations = g.axis_angle_to_matrix(torch.as_tensor(poses, dtype=torch.float32).reshape(-1, 16, 3))
            states = g.state_from_rotmat(torch.as_tensor(wrists, dtype=torch.float32), rotations).numpy()
        ortho = float((rotations.transpose(-1, -2) @ rotations - torch.eye(3)).abs().max())
        determinant = float((torch.linalg.det(rotations) - 1).abs().max())
        if ortho > 1e-5 or determinant > 1e-5 or not np.isfinite(states).all():
            raise ValueError(f"Invalid SO(3)/state conversion: {pose_file}")
        fk = fk_roundtrip(decoder, side, poses, beta, states, ids, job["fk_tolerance_m"], job["fk_samples"])
        arrays = {"state": states.astype(np.float32), "timestamps": ids.astype(np.float64) / 30.0,
                  "frame_ids": ids, "beta": beta.astype(np.float32), "beta_valid": np.bool_(True),
                  "gt_mask": np.ones((len(ids), 17), dtype=np.bool_)}
        boundaries = np.concatenate(([0], np.flatnonzero(np.diff(ids) != 1) + 1, [len(ids)]))
        run_lengths = np.diff(boundaries)
        # Descriptive source QC only; no arbitrary motion/velocity filtering.
        contiguous = np.diff(ids) == 1
        wrist_steps = np.linalg.norm(np.diff(wrists, axis=0)[contiguous], axis=-1)
        stem = f"{job['split']}/{job['recording']}__{side}"
        relative = save_sequence_arrays(output, stem, arrays, job["storage"])
        entry = {"path": relative, "storage": job["storage"], "dataset": "taco", "source": "taco_v1",
                 "split": job["split"], "official_split": "train", "recording": job["recording"],
                 "triplet": job["triplet"], "handedness": int(side == "right"), "frames": len(ids),
                 "source_pose_path": str(pose_file), "source_shape_path": str(shape_file),
                 "source_pose_sha256": pose_sha, "source_shape_sha256": shape_sha,
                 "source_frame_count": len(original_ids), "frame_id_minmax": [int(ids[0]), int(ids[-1])],
                 "frame_gaps": int(np.count_nonzero(np.diff(ids) != 1)), "skipped": dict(skipped),
                 "contiguous_runs": len(run_lengths), "longest_contiguous_frames": int(run_lengths.max()),
                 "frames_in_runs_ge_91": int(run_lengths[run_lengths >= 91].sum()),
                 "wrist_axis_min_m": wrists.min(axis=0).tolist(), "wrist_axis_max_m": wrists.max(axis=0).tolist(),
                 "contiguous_wrist_step_max_m": float(wrist_steps.max()) if len(wrist_steps) else None,
                 "coordinate_frame": "source_static_world", "units": "metres", "nominal_dt": 1 / 30,
                 "timestamp_source": SOURCE_CONVENTION["timestamp_source"], "beta_span": 0.0,
                 "gt_complete": True, "partial_gt": False, "source_convention": SOURCE_CONVENTION,
                 "rotation_orthogonality_error": ortho, "rotation_determinant_error": determinant,
                 "fk_roundtrip": fk, "full_fk_verified": False,
                 "source_decoder_agreement_verified": False}
        save_json(metadata_path, {"identity": identity, "entry": entry})
        result["sequences"].append(entry)
    result["decoder_provenance"] = decoder.provenance()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="contains Hand_Poses/, or Hand_Poses itself")
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--sequence-list", type=Path, required=True)
    parser.add_argument("--source-revision", required=True, help="fixed TACO-Instructions Git commit SHA")
    parser.add_argument("--source-code", type=Path, help="optional checked-out official source for decoder source hashes")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mano-models", type=Path, required=True)
    parser.add_argument("--mano-asset-version", default=None)
    parser.add_argument("--storage", choices=("npy", "npz"), default="npy")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--validation-fraction", type=float, default=0.1,
                        help="engineering recording-level holdout ONLY from official train; 0 keeps all train")
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--max-recordings-per-split", type=int, default=0, help="bounded smoke only; 0 all")
    parser.add_argument("--max-frames", type=int, default=0, help="first original frame IDs per hand; 0 all")
    parser.add_argument("--min-frames", type=int, default=34)
    parser.add_argument("--fk-samples", type=int, default=32)
    parser.add_argument("--fk-tolerance-m", type=float, default=1e-5)
    parser.add_argument("--index-only", action="store_true", help="metadata/filesystem only, never opens hand labels")
    parser.add_argument("--resume", action="store_true", help="reuse completed immutable sequence receipts with identical provenance")
    args = parser.parse_args()
    if not Path("/sciclone/home/jchen57").is_dir():
        parser.error("Data preparation is authorized only in the remote ds Pod")
    if (args.workers < 1 or args.min_frames < 1 or args.fk_samples < 1
            or min(args.max_recordings_per_split, args.max_frames) < 0
            or not 0 <= args.validation_fraction < 1
            or not np.isfinite(args.fk_tolerance_m) or args.fk_tolerance_m <= 0
            or not re.fullmatch(r"[0-9a-f]{40}", args.source_revision)):
        parser.error("Invalid worker/frame/validation/FK/source revision arguments")
    splits, triplets = read_official_lists(args.split_file, args.sequence_list)
    directory = args.root if args.root.name == "Hand_Poses" else args.root / "Hand_Poses"
    if not directory.is_dir():
        parser.error("--root must contain the extracted Hand_Poses directory")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "manifest.json").exists():
        parser.error("Output already has a final manifest; use a fresh directory")
    assigned = {recording: engineering_split(recording, args.validation_fraction, args.seed)
                for recording in splits if splits[recording] == "train"}
    selected = []
    for split in ("train", "val"):
        recordings = sorted(recording for recording, value in assigned.items() if value == split)
        if args.max_recordings_per_split:
            recordings = recordings[:args.max_recordings_per_split]
        selected.extend((recording, split) for recording in recordings)
    if not selected or not any(split == "train" for _, split in selected):
        raise ValueError("No selected training recordings")
    source_hashes = {}
    if args.source_code:
        for name in ("dataset_utils/hand_pose_loader.py", "dataset_utils/manopth/manopth/manolayer.py"):
            path = args.source_code / name
            source_hashes[name] = digest(path)
    decoder = None if args.index_only else get_decoder(str(args.mano_models), args.mano_asset_version)
    manifest = {"utc": now(), "phase": "indexed" if args.index_only else "preparing", "dataset": "taco",
                "adapter_version": ADAPTER_VERSION, "source_revision": args.source_revision,
                "source_hf_revision": HF_REVISION, "source_decoder_files_sha256": source_hashes,
                "official_hand_archive_expected_sha256": HAND_ARCHIVE_SHA256,
                "archive_hash_note": "expected public LFS hash only; source file hashes computed by this adapter",
                "source_urls": ["https://github.com/leolyliu/TACO-Instructions/tree/" + args.source_revision,
                                "https://huggingface.co/datasets/leolyliu/TACO/tree/" + HF_REVISION,
                                "https://arxiv.org/abs/2401.08399"],
                "source_convention": SOURCE_CONVENTION, "rotation6d": "columns_c0xyz_c1xyz",
                "coordinate_frame": "source_static_world", "units": "metres", "nominal_dt": 1 / 30,
                "position": "anatomical wrist J0=source hand_trans after prescribed wrist centering",
                "shape": "source per-hand per-recording beta10; no missing-beta fallback",
                "source_quality": "multiview/contact/temporal visual fitting; not independent hand MoCap",
                "official_split_counts": dict(Counter(splits.values())),
                "official_train_assignments": assigned, "selected_recording_counts": dict(Counter(x[1] for x in selected)),
                "split_policy": {"official_test_policy": "all test_1..test_4 excluded before pose decode",
                                 "validation": "engineering original-recording SHA256 holdout within official train",
                                 "validation_fraction": args.validation_fraction, "seed": args.seed,
                                 "subject_disjoint": False, "both_hands_grouped": True},
                "split_file_sha256": digest(args.split_file), "sequence_list_sha256": digest(args.sequence_list),
                "bounded_limits": {"max_recordings_per_split": args.max_recordings_per_split,
                                   "max_frames_per_hand": args.max_frames},
                "full_fk_verified": False, "source_decoder_agreement_verified": False,
                "fk_policy": f"sampled {args.fk_samples} evenly spaced frames per accepted hand sequence, shared-smplx roundtrip",
                "decoder_provenance": decoder.provenance() if decoder else None,
                "sequences": [], "recordings_summary": []}
    jobs = []
    for recording, split in selected:
        source_directory = directory / triplets[recording] / recording
        # Filenames/stat only: test labels are not consulted, nor any RGB.
        for side in ("left", "right"):
            for suffix in ("_hand.pkl", "_hand_shape.pkl"):
                if not (source_directory / (side + suffix)).is_file():
                    raise FileNotFoundError(source_directory / (side + suffix))
        jobs.append({"root": str(args.root), "directory": str(source_directory), "recording": recording,
                     "triplet": triplets[recording], "split": split, "output": str(args.output),
                     "source_revision": args.source_revision, "mano_models": str(args.mano_models),
                     "mano_asset_version": args.mano_asset_version, "storage": args.storage,
                     "max_frames": args.max_frames, "min_frames": args.min_frames,
                     "fk_samples": args.fk_samples, "fk_tolerance_m": args.fk_tolerance_m, "resume": args.resume})
    if args.index_only:
        manifest["selected_recordings"] = [{key: job[key] for key in ("recording", "triplet", "split", "directory")} for job in jobs]
        save_json(args.output / "index_manifest.json", manifest)
        print(json.dumps({"phase": "indexed", "selected": len(jobs), "split_counts": manifest["selected_recording_counts"]}), flush=True)
        return
    status_path = args.output / "preparation_status.json"
    save_json(status_path, {"phase": "preparing", "done": 0, "total": len(jobs), "utc": now()})
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(convert_recording, job) for job in jobs]
        for done, future in enumerate(as_completed(futures), 1):
            try:
                result = future.result()
            except Exception as exc:
                save_json(status_path, {"phase": "failed", "done": done - 1, "total": len(jobs),
                                        "error_type": type(exc).__name__, "error": str(exc), "utc": now()})
                for pending in futures:
                    pending.cancel()
                raise
            if result.pop("decoder_provenance") != manifest["decoder_provenance"]:
                raise RuntimeError("MANO asset/decoder provenance changed between workers")
            manifest["sequences"].extend(result.pop("sequences"))
            manifest["recordings_summary"].append(result)
            status = {"phase": "preparing", "done": done, "total": len(jobs),
                      "last_recording": result["recording"], "utc": now()}
            save_json(status_path, status)
            print(json.dumps(status), flush=True)
    manifest["sequences"].sort(key=lambda item: item["path"])
    manifest["recordings_summary"].sort(key=lambda item: item["recording"])
    if not manifest["sequences"]:
        raise RuntimeError("No accepted full99D trajectories")
    assignments = {}
    for entry in manifest["sequences"]:
        if splits[entry["recording"]] != "train":
            raise RuntimeError("Official held-out label leaked into prepared pool")
        if assignments.setdefault(entry["recording"], entry["split"]) != entry["split"]:
            raise RuntimeError("Original recording crosses train/validation")
    manifest.update(phase="complete", completed_utc=now(), sampled_fk_roundtrip_verified=True)
    manifest["frames_by_split"] = {split: sum(entry["frames"] for entry in manifest["sequences"] if entry["split"] == split)
                                   for split in ("train", "val")}
    manifest["sequences_by_split"] = dict(Counter(entry["split"] for entry in manifest["sequences"]))
    save_json(args.output / "manifest.json", manifest)
    summary = {"phase": "complete", "recordings": len(jobs), "sequences": len(manifest["sequences"]),
               "frames": manifest["frames_by_split"], "manifest": str(args.output / "manifest.json"), "utc": now()}
    save_json(status_path, summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
