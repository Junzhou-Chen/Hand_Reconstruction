#!/usr/bin/env python3
"""Expand remote DexYCB RGB training without changing the held-out protocol.

Use every reference-valid frame of metadata-selected official s0 train
recordings, across all ten subjects by default. Keep the previous visual
validation recordings, frames, images and references byte-identical. The
complete single-view prepared reference has already been MANO/FK audited;
this script never opens source test labels or runs a neural network.

The output remains a GT-crop/fixed-camera pilot, not ego SLAM or detector
evaluation. Run all extraction, array processing and checks in the ds Pod.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
from pathlib import Path
import re
import shutil
import sys
import tarfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from handloop.data.visual_clips import file_sha256
from prepare_dexycb_twm import REVISION, SUBJECTS, s0_split
from prepare_visual_feedback_cache import SOURCE_SCHEMA, SERIAL, now, read_intrinsics, save_json


FOCAL_CONVENTION = "official HaWoR single focal uses mean(fx,fy); real K retained for geometry"
ARRAY_KEYS = ("state", "timestamps", "frame_ids", "joint_3d", "gt_mask", "beta")
REFERENCE_METADATA = {
    "dataset": "dexycb", "reference_source": "official fitted MANO and 21 joints; not independent MoCap",
    "reference_joint_order": "dex21", "reference_units": "metres",
    "reference_coordinate_frame": "world", "reference_left_shapedirs_fix": False,
    "joint_3d_order": "dex21", "joint_3d_units": "metres", "joint_3d_coordinate_frame": "world",
}


def identity(path: Path) -> dict:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": file_sha256(path)}


def inside(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Manifest path escapes source directory: {name}")
    return path


def checked_source_index(args, subject: str) -> dict:
    path = args.prepared / "source_indices" / (subject + ".json")
    value = json.loads(path.read_text())
    archive = args.dex_root / "archives" / (subject + ".tar.gz")
    stat = archive.stat()
    if (value.get("subject") != subject or value.get("archive_identity") !=
            {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}):
        raise ValueError(f"Source archive/index identity mismatch: {subject}")
    entries = value.get("sequences", [])
    names = [entry["name"] for entry in entries]
    if len(entries) != 100 or names != sorted(set(names)):
        raise ValueError(f"Require 100 unique sorted official recordings: {subject}")
    for index, entry in enumerate(entries):
        if (entry["index"] != index or entry["split"] != s0_split(subject, index)
                or not re.fullmatch(r"[A-Za-z0-9_-]+", entry["name"])):
            raise ValueError(f"Invalid official s0 recording index: {subject}")
    receipt = json.loads((args.dex_root / "download_receipts.json").read_text()).get(
        "archives/" + archive.name, {})
    digest = value.get("source_sha256") or receipt.get("sha256")
    if (not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest)
            or receipt.get("size") != stat.st_size or receipt.get("sha256") != digest):
        raise ValueError(f"Verified download receipt/index SHA mismatch: {subject}")
    return {**value, "source_index_identity": identity(path), "verified_archive_sha256": digest}


def select_training(args, prepared: dict, forbidden: set[str]) -> tuple[list[dict], dict]:
    """Select complete recording identities before loading any target arrays."""
    all_entries = prepared["sequences"]
    recordings = [entry["recording"] for entry in all_entries]
    if len(recordings) != len(set(recordings)):
        raise ValueError("Duplicate recording in prepared reference")
    selected, audits = [], {}
    for subject in args.subjects:
        source_index = checked_source_index(args, subject)
        indexed = {entry["name"]: entry for entry in source_index["sequences"]}
        eligible = sorted((entry for entry in all_entries if entry["subject"] == subject
                           and entry["split"] == "train" and entry["frames"] >= args.min_frames),
                          key=lambda entry: entry["sequence_index"])
        if not eligible:
            raise ValueError(f"No prepared official s0 train recording: {subject}")
        if args.records_per_subject and len(eligible) > args.records_per_subject:
            # Cover the recording-index range; do not select by predictions or errors.
            indices = np.linspace(0, len(eligible) - 1, args.records_per_subject, dtype=np.int64)
            eligible = [eligible[int(index)] for index in indices]
        for entry in eligible:
            parent, sequence = entry["recording"].split("/")
            original = indexed[sequence]
            if (parent != subject or original["index"] != entry["sequence_index"]
                    or original["split"] != "train" or s0_split(subject, entry["sequence_index"]) != "train"
                    or args.serial not in original["meta"]["serials"]
                    or entry["serial"] != args.serial or entry["storage"] != "npy"
                    or not entry.get("gt_complete") or entry["source_archive_sha256"] !=
                    source_index["verified_archive_sha256"]):
                raise ValueError(f"Prepared/index train metadata disagreement: {entry['recording']}")
            if entry["recording"] in forbidden:
                raise ValueError(f"Train recording overlaps held-out validation/test: {entry['recording']}")
            side = original["meta"]["mano_sides"]
            if side != ["right" if entry["handedness"] else "left"]:
                raise ValueError("Prepared/index hand side disagreement")
            selected.append(entry)
        audits[subject] = {"source_index": source_index["source_index_identity"],
                           "source_archive_identity": source_index["archive_identity"],
                           "source_archive_sha256": source_index["verified_archive_sha256"],
                           "selected_train_recordings": [entry["recording"] for entry in eligible]}
    return selected, audits


def prepare_subject(job: dict) -> dict:
    """CPU-only source arrays and one streamed JPEG pass per subject archive."""
    output, prepared_root = Path(job["output"]), Path(job["prepared"])
    intrinsics = np.asarray(job["intrinsics"], dtype=np.float32)
    requested, records = {}, []
    for entry in job["entries"]:
        source = inside(prepared_root, entry["path"])
        paths = {key: source / (key + ".npy") for key in ARRAY_KEYS}
        identities = {key: identity(path) for key, path in paths.items()}
        arrays = {key: np.load(path, mmap_mode="r", allow_pickle=False) for key, path in paths.items()}
        count = int(entry["frames"])
        expected = {"state": (count, 99), "timestamps": (count,), "frame_ids": (count,),
                    "joint_3d": (count, 21, 3), "gt_mask": (count, 17), "beta": (10,)}
        for key, shape in expected.items():
            if arrays[key].shape != shape or not np.isfinite(arrays[key]).all():
                raise ValueError(f"Invalid prepared {key}: {entry['recording']}")
        ids = np.asarray(arrays["frame_ids"], dtype=np.int64)
        times = np.asarray(arrays["timestamps"], dtype=np.float64)
        if (np.any(ids < 0) or np.any(np.diff(ids) <= 0) or np.any(np.diff(times) <= 0)
                or not np.allclose(times, ids / 30, atol=1e-10, rtol=0)
                or not arrays["gt_mask"].all()):
            raise ValueError(f"Invalid ordered full-valid reference: {entry['recording']}")
        joints = np.asarray(arrays["joint_3d"], dtype=np.float32)
        wrist_error = np.linalg.norm(arrays["state"][:, :3] - joints[:, 0], axis=-1)
        if float(wrist_error.max()) > 1e-5 or np.any(joints[..., 2] <= 0):
            raise ValueError("Reference anatomical J0/depth consistency failed")
        projected = joints @ intrinsics.T
        projected = projected[..., :2] / projected[..., 2:3]
        bbox = np.concatenate((projected.min(axis=1), projected.max(axis=1)), axis=-1)
        if np.any(bbox[:, 2:] - bbox[:, :2] < 4):
            raise ValueError(f"Degenerate reference-projected hand crop: {entry['recording']}")
        subject, sequence = entry["recording"].split("/")
        stem = "train/" + subject + "++" + sequence
        destination = output / stem
        destination.mkdir(parents=True, exist_ok=False)
        images = []
        for frame in ids:
            name = f"color_{int(frame):06d}.jpg"
            member = f"{subject}/{sequence}/{job['serial']}/{name}"
            if member in requested:
                raise ValueError(f"Duplicate requested physical RGB frame: {member}")
            relative = stem + "/images/" + name
            requested[member] = relative
            images.append(relative)
        reference_path = destination / "reference.npz"
        with reference_path.open("xb") as stream:
            np.savez(stream, target_world=np.asarray(arrays["state"], dtype=np.float32),
                     target_valid=np.asarray(arrays["gt_mask"], dtype=np.bool_), timestamps=times,
                     frame_ids=ids, bbox_xyxy=bbox.astype(np.float32), joint_3d=joints,
                     joint_3d_valid=np.ones((count, 21), dtype=np.bool_),
                     beta=np.asarray(arrays["beta"], dtype=np.float32),
                     camera_intrinsics=np.broadcast_to(intrinsics, (count, 3, 3)).copy(),
                     camera_to_world=np.broadcast_to(np.eye(4, dtype=np.float32), (count, 4, 4)).copy(),
                     handedness=np.full(count, int(entry["handedness"]), dtype=np.int64))
        if any(identity(paths[key]) != original for key, original in identities.items()):
            raise ValueError("Prepared source arrays changed during extraction")
        records.append({**REFERENCE_METADATA, "recording": entry["recording"], "subject": subject,
                        "split": "train", "path": stem, "frames": count, "image_paths": images,
                        "serial": job["serial"], "handedness": int(entry["handedness"]),
                        "sequence_index": entry["sequence_index"],
                        "coordinate_frame": f"fixed_camera_{job['serial']}_as_world_gauge",
                        "timestamp_source": "nominal_30hz_frame_index_not_measured", "nominal_dt": 1 / 30,
                        "source_archive_sha256": entry["source_archive_sha256"],
                        "source_archive_identity": job["source_archive_identity"],
                        "source_reference_path": str(source), "source_array_identities": identities,
                        "reference_file": stem + "/reference.npz", "reference_sha256": file_sha256(reference_path),
                        "frame_id_minmax": [int(ids[0]), int(ids[-1])],
                        "frame_gaps": int(np.count_nonzero(np.diff(ids) != 1)),
                        "source_num_frames": entry["source_num_frames"], "selection": "all_prepared_valid_frames",
                        "joint_3d_valid_policy": "explicit_mask",
                        "source_fk_joints_max_m": entry["fk_joints_max_m"], "gt_complete": True})
    archive = Path(job["archive"])
    stat = archive.stat()
    archive_identity = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if archive_identity != job["source_archive_identity"]:
        raise ValueError("Archive changed after recording selection")
    remaining, image_hashes = set(requested), {}
    with tarfile.open(archive, "r|gz") as handle:
        for member in handle:
            if member.name not in requested:
                continue
            if member.name not in remaining or not member.isfile() or not 0 < member.size <= 20_000_000:
                raise ValueError(f"Duplicate or invalid JPEG member: {member.name}")
            relative = requested[member.name]
            destination = inside(output, relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            data = handle.extractfile(member).read()
            if len(data) != member.size:
                raise ValueError(f"Truncated JPEG member: {member.name}")
            destination.write_bytes(data)
            image_hashes[relative] = hashlib.sha256(data).hexdigest()
            remaining.remove(member.name)
            if not remaining:
                break
    if remaining:
        raise FileNotFoundError(f"Missing {len(remaining)} requested JPEGs: {sorted(remaining)[:3]}")
    stat = archive.stat()
    if {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns} != archive_identity:
        raise ValueError("Archive changed during RGB extraction")
    for record in records:
        record["image_sha256"] = {path: image_hashes[path] for path in record["image_paths"]}
    return {"subject": job["subject"], "records": records, "images": len(image_hashes)}


def copy_validation(args, manifest: dict) -> list[dict]:
    records = []
    for entry in manifest["records"]:
        if entry["split"] != "val":
            continue
        subject, _ = entry["recording"].split("/")
        if s0_split(subject, int(entry["sequence_index"])) != "val":
            raise ValueError("Preserved validation disagrees with official s0")
        root = args.validation_manifest.parent
        paths = [entry["reference_file"], *entry["image_paths"]]
        original_identities = {}
        for relative in paths:
            source, target = inside(root, relative), inside(args.output, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise FileExistsError("Validation destination collides with training source")
            original = identity(source)
            shutil.copy2(source, target)
            if file_sha256(target) != original["sha256"] or identity(source) != original:
                raise ValueError("Validation reference/image changed during byte-preserving copy")
            original_identities[relative] = original
        record = {**entry, **REFERENCE_METADATA,
                  "joint_3d_valid_policy": "legacy_complete_mano21",
                  "preserved_validation_manifest_sha256": file_sha256(args.validation_manifest),
                  "preserved_reference_sha256": original_identities[entry["reference_file"]]["sha256"],
                  "reference_sha256": original_identities[entry["reference_file"]]["sha256"],
                  "image_sha256": {path: original_identities[path]["sha256"] for path in entry["image_paths"]},
                  "validation_protocol": "byte-identical recording/frame/reference/RGB selection retained"}
        records.append(record)
    if not records:
        raise ValueError("Preserved source contains no validation recording")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dex-root", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--forbidden-eval-manifest", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True, help="Fresh immutable output directory")
    parser.add_argument("--subjects", nargs="+", choices=SUBJECTS, default=list(SUBJECTS))
    parser.add_argument("--records-per-subject", type=int, default=0,
                        help="0 uses all official train recordings; positive cap spans sequence-index range")
    parser.add_argument("--min-frames", type=int, default=1)
    parser.add_argument("--serial", choices=(SERIAL,), default=SERIAL)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--code-version", default="unrecorded")
    args = parser.parse_args()
    if not Path("/sciclone/home/jchen57").is_dir():
        parser.error("Run all RGB/reference preprocessing only in the authorized remote ds Pod")
    if (args.workers < 1 or args.min_frames < 1 or args.records_per_subject < 0
            or len(set(args.subjects)) != len(args.subjects)):
        parser.error("Invalid worker/minimum/recording cap/subjects")
    for key in ("dex_root", "prepared", "validation_manifest", "output"):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    prepared_path = args.prepared / "manifest.json"
    prepared = json.loads(prepared_path.read_text())
    previous = json.loads(args.validation_manifest.read_text())
    if (prepared.get("phase") != "complete" or prepared.get("setup") != "s0"
            or not prepared.get("full_fk_verified") or prepared.get("serial") != args.serial
            or prepared.get("source_revision") != REVISION
            or prepared.get("bounded_limits", {}).get("max_frames_per_sequence") != 0
            or prepared.get("bounded_limits", {}).get("max_sequences_per_split_per_subject") != 0):
        raise ValueError("Require complete, uncapped, FK-audited single-view s0 reference")
    if (previous.get("schema") != SOURCE_SCHEMA or previous.get("phase") != "complete"
            or previous.get("dataset") != "dexycb" or previous.get("focal_convention") != FOCAL_CONVENTION
            or any(record.get("split") not in ("train", "val") for record in previous.get("records", []))):
        raise ValueError("Require completed DexYCB train/val visual source to preserve validation")
    decoder = prepared["decoder_provenance"]
    if decoder.get("left_shapedirs_fix_applied") or decoder.get("left_shapedirs_fix_requested"):
        raise ValueError("Keep original Dex MANO reference without left shapedirs fix")
    validation_identity, prepared_identity = identity(args.validation_manifest), identity(prepared_path)
    if (previous.get("reference_manifest_sha256") and
            previous["reference_manifest_sha256"] != prepared_identity["sha256"]):
        raise ValueError("Preserved validation and expanded training must use the same audited source")
    forbidden = {record["recording"] for record in previous["records"] if record["split"] == "val"}
    extra_holdouts = []
    for path in args.forbidden_eval_manifest:
        path = path.expanduser().resolve()
        value = json.loads(path.read_text())
        if value.get("phase") != "complete":
            raise ValueError("Held-out evaluation manifest must be complete")
        if any(record.get("split") not in ("val", "test") for record in value.get("records", [])):
            raise ValueError("Forbidden evaluation manifests must contain val/test only")
        forbidden.update(record["recording"] for record in value.get("records", []))
        extra_holdouts.append(identity(path))
    selected, source_audits = select_training(args, prepared, forbidden)
    if args.output.exists():
        raise FileExistsError("Use a fresh output directory; never overwrite an existing source")
    intrinsics = read_intrinsics(args.dex_root, args.serial)
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {"schema": SOURCE_SCHEMA, "phase": "extracting", "created_utc": now(),
                "dataset": "dexycb", "setup": "s0_expanded_train_preserved_visual_val",
                "code_version": args.code_version, "reference_manifest": prepared_identity,
                "decoder_provenance": decoder, "validation_source": validation_identity,
                "extra_held_out_sources": extra_holdouts, "forbidden_held_out_recordings": sorted(forbidden),
                "reference_source": REFERENCE_METADATA["reference_source"],
                "reference_joint_order": "dex21", "reference_units": "metres",
                "reference_coordinate_frame": "world", "reference_left_shapedirs_fix": False,
                "joint_3d_order": "dex21", "joint_3d_units": "metres", "joint_3d_coordinate_frame": "world",
                "crop_protocol": "project all 21 fitted reference joints with original color K; min/max GT box; official HaWoR square 1.2 expansion",
                "input_protocol": "GT crop on all selected reference-valid train frames; preserved validation; no detector/depth input; single fixed camera",
                "focal_convention": FOCAL_CONVENTION,
                "coordinate_convention": "column vectors; original camera is identity world gauge; metres",
                "selection_protocol": {"train": "official s0 train, metadata-selected before source array reads",
                                       "subjects": args.subjects, "records_per_subject_cap": args.records_per_subject,
                                       "cap_selection": "evenly spaced official sequence indices, never prediction/error selection",
                                       "temporal_scope": "all prepared reference-valid frames; no prefix/time cap",
                                       "min_reference_valid_frames": args.min_frames,
                                       "val": "byte-identical original visual validation frames and reference",
                                       "test": "excluded by official split; no test label read"},
                "source_index_audits": source_audits, "records": [], "completed_subjects": []}
    manifest["records"].extend(copy_validation(args, previous))
    status_path = args.output / "source_manifest.incomplete.json"
    save_json(status_path, manifest)
    jobs = [{"subject": subject, "output": str(args.output), "prepared": str(args.prepared),
             "entries": [entry for entry in selected if entry["subject"] == subject],
             "serial": args.serial, "intrinsics": intrinsics.tolist(),
             "archive": str(args.dex_root / "archives" / (subject + ".tar.gz")),
             "source_archive_identity": source_audits[subject]["source_archive_identity"]}
            for subject in args.subjects]
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(prepare_subject, job) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            manifest["records"].extend(result["records"])
            manifest["completed_subjects"].append(result["subject"])
            save_json(status_path, manifest)
            print(json.dumps({"phase": "extracting", "completed_subject": result["subject"],
                              "train_recordings": len(result["records"]), "train_images": result["images"]}), flush=True)
    train = {record["recording"] for record in manifest["records"] if record["split"] == "train"}
    val = {record["recording"] for record in manifest["records"] if record["split"] == "val"}
    if train & (val | forbidden) or len(train) != len(selected):
        raise ValueError("Final official train/held-out recording disjointness/count failed")
    if identity(args.validation_manifest) != validation_identity or identity(prepared_path) != prepared_identity:
        raise ValueError("Input manifests changed during expanded RGB preparation")
    manifest["records"].sort(key=lambda record: (record["split"], record["recording"]))
    manifest["completed_subjects"].sort()
    manifest["phase"], manifest["completed_utc"] = "complete", now()
    manifest["split_records"] = dict(Counter(record["split"] for record in manifest["records"]))
    manifest["split_frames"] = {split: sum(record["frames"] for record in manifest["records"]
                                           if record["split"] == split) for split in ("train", "val")}
    manifest["split_hand_frames"] = {split: {side: sum(record["frames"] for record in manifest["records"]
        if record["split"] == split and record["handedness"] == hand)
        for hand, side in enumerate(("left", "right"))} for split in ("train", "val")}
    save_json(args.output / "source_manifest.json", manifest)
    status_path.unlink()
    print(json.dumps({"phase": "complete", "manifest": str(args.output / "source_manifest.json"),
                      "split_records": manifest["split_records"], "split_frames": manifest["split_frames"]}), flush=True)


if __name__ == "__main__":
    main()
