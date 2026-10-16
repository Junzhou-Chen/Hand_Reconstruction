#!/usr/bin/env python3
"""Prepare the complete official DexYCB s0 test split on the remote ds Pod.

All 160 test recordings, all eight calibrated RGB cameras, and every raw frame
are selected from the audited metadata before test annotations are opened.
Each recording/camera becomes an immutable, independently consumable source
manifest. ``shard_index.json`` is updated while extraction runs, so evaluation
can consume completed shards without waiting for every subject TAR to finish.

This retains the existing disclosed GT-crop protocol and camera-as-world
gauge. It is neither detector evaluation nor moving-camera world evaluation.
No model predictions or evaluation scores control membership. Test sources
are explicitly evaluation-only. All array processing must run remotely.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import fcntl
import hashlib
import io
import json
import multiprocessing
from pathlib import Path, PurePosixPath
import queue
import re
import sys
import tarfile
import tempfile
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from handloop.data.meta_mano import MANOReferenceDecoder
from handloop.data.visual_clips import file_sha256
from prepare_dexycb_twm import (REVISION, SUBJECTS, SERIALS, LABEL_PATTERN,
                                load_calibrations, read_yaml, s0_split)
from prepare_visual_eval_sources import JPEG_PATTERN, path_identity, write_json
from prepare_visual_feedback_cache import SOURCE_SCHEMA, now, read_intrinsics


REFERENCE_METADATA = {
    "dataset": "dexycb", "reference_source": "official fitted MANO and 21 joints; not independent MoCap",
    "reference_joint_order": "dex21", "reference_units": "metres",
    "reference_coordinate_frame": "world", "reference_left_shapedirs_fix": False,
    "joint_3d_order": "dex21", "joint_3d_units": "metres", "joint_3d_coordinate_frame": "world",
    "joint_3d_valid_policy": "explicit_mask",
}


def digest_json(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def shard_name(subject: str, sequence: str, serial: str) -> str:
    return subject + "++" + sequence + "++" + serial


def select_subject(args, subject: str, forbidden: set[str]) -> dict:
    """Fail-closed source index/receipt checks, before any test labels are read."""
    path = args.prepared_source.parent / "source_indices" / (subject + ".json")
    indexed = json.loads(path.read_text())
    archive = args.dex_root / "archives" / (subject + ".tar.gz")
    stat = archive.stat()
    identity = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if indexed.get("subject") != subject or indexed.get("archive_identity") != identity:
        raise ValueError(f"Audited metadata/archive identity mismatch: {subject}")
    entries = indexed.get("sequences", [])
    names = [entry["name"] for entry in entries]
    if len(entries) != 100 or names != sorted(set(names)):
        raise ValueError(f"Expected 100 unique lexicographically indexed recordings: {subject}")
    for index, entry in enumerate(entries):
        if (entry["index"] != index or entry["split"] != s0_split(subject, index)
                or not re.fullmatch(r"[A-Za-z0-9_-]+", entry["name"])):
            raise ValueError(f"Invalid official s0 recording ordering/split: {subject}")
    selected = [entry for entry in entries if entry["split"] == "test"]
    if len(selected) != 20:
        raise ValueError(f"Expected all 20 official test recordings of {subject}")
    for entry in selected:
        recording, meta = subject + "/" + entry["name"], entry["meta"]
        if recording in forbidden:
            raise ValueError(f"Test/training or validation leakage: {recording}")
        if (len(meta["mano_sides"]) != 1 or len(meta["mano_calib"]) != 1
                or meta["mano_sides"][0] not in ("left", "right")
                or set(meta["serials"]) != set(SERIALS) or len(meta["serials"]) != 8
                or int(meta["num_frames"]) < 1):
            raise ValueError(f"Full eight-view recording metadata unavailable: {recording}")
    receipt = json.loads((args.dex_root / "download_receipts.json").read_text()).get(
        "archives/" + archive.name, {})
    sha = indexed.get("source_sha256") or receipt.get("sha256")
    if (not isinstance(sha, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", sha)
            or receipt.get("size") != stat.st_size or receipt.get("sha256") != sha):
        raise ValueError(f"Verified download receipt/archive SHA mismatch: {subject}")
    return {"subject": subject, "archive": str(archive), "archive_identity": identity,
            "source_archive_sha256": sha, "source_index": path_identity(path),
            "selected": selected, "official_split_counts": dict(Counter(e["split"] for e in entries))}


def safe_relative(root: Path, relative: str) -> Path:
    target = (root / relative).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError(f"Manifest path escapes its shard: {relative}")
    return target


def verify_completed(path: Path, selection_sha: str, recording: str) -> dict:
    manifest = json.loads(path.read_text())
    if (manifest.get("phase") != "complete" or manifest.get("selection_sha256") != selection_sha
            or manifest.get("evaluation_recording") != recording or not manifest.get("eval_only")
            or manifest.get("training_allowed") is not False):
        raise ValueError(f"Resume shard protocol/provenance mismatch: {path}")
    for record in manifest["records"]:
        reference = safe_relative(path.parent, record["reference_file"])
        if file_sha256(reference) != record["reference_sha256"]:
            raise ValueError(f"Resume reference checksum mismatch: {path}")
        for relative, sha in record["image_sha256"].items():
            if file_sha256(safe_relative(path.parent, relative)) != sha:
                raise ValueError(f"Resume RGB checksum mismatch: {relative}")
    return manifest


def shard_event(root: Path, path: Path, manifest: dict) -> dict:
    return {"key": manifest["shard_id"], "shard_id": manifest["shard_id"], "manifest": str(path.relative_to(root)),
            "manifest_sha256": file_sha256(path), "subject": manifest["subject"],
            "physical_recording": manifest["physical_recording"],
            "evaluation_recording": manifest["evaluation_recording"], "serial": manifest["serial"],
            "records": len(manifest["records"]), "frames": sum(r["frames"] for r in manifest["records"]),
            "raw_frames": manifest["frame_audit"]["raw_frames"],
            "invalid_gt_frames": manifest["frame_audit"]["invalid_joint_gt_frames"]}


def prepare_view(job: dict, entry: dict, serial: str, label_members: dict, image_members: dict,
                 spool, decoder, calibrations: dict) -> tuple[Path, dict]:
    """Decode one full recording/view; never silently remove valid official GT."""
    output, subject, name = Path(job["output"]), job["subject"], entry["name"]
    physical = subject + "/" + name
    evaluation_recording = physical + "/" + serial
    sid = shard_name(subject, name, serial)
    root = output / "shards" / sid
    manifest_path = root / "source_manifest.json"
    if manifest_path.exists():
        return manifest_path, verify_completed(manifest_path, job["selection_sha256"], evaluation_recording)
    expected = set(range(int(entry["meta"]["num_frames"])))
    missing_labels = sorted(expected - set(label_members))
    missing_images = sorted(expected - set(image_members))
    if missing_labels or missing_images or set(label_members) - expected or set(image_members) - expected:
        raise ValueError(f"Incomplete complete-frame coverage for {evaluation_recording}: "
                         f"missing labels={missing_labels[:8]}, RGB={missing_images[:8]}")
    valid, invalid_ids, ineligible = {}, [], []
    for frame_id in sorted(expected):
        offset, size = label_members[frame_id]
        spool.seek(offset)
        with np.load(io.BytesIO(spool.read(size)), allow_pickle=False) as label:
            joints = label["joint_3d"].copy()
            pose = label["pose_m"].copy()
        if joints.shape != (1, 21, 3) or pose.shape != (1, 51):
            raise ValueError(f"Unexpected reference shape: {evaluation_recording}/{frame_id}")
        # Official hpe_eval.py excludes only an entirely -1 joint array.
        if np.all(joints == -1):
            invalid_ids.append(frame_id)
            continue
        reason = None
        if not np.isfinite(joints).all() or np.any(np.all(joints == -1, axis=-1)):
            reason = "official_hand_valid_but_incomplete_or_nonfinite_joints"
        elif not np.isfinite(pose).all() or np.all(pose == 0):
            reason = "official_joint_valid_but_missing_or_nonfinite_mano_pose"
        elif np.any(joints[..., 2] <= 0):
            reason = "official_joint_valid_but_nonpositive_depth_for_gt_crop"
        if reason:
            ineligible.append({"frame_id": frame_id, "reason": reason})
        else:
            valid[frame_id] = (pose[0], joints[0])
    audit = {"physical_recording": physical, "evaluation_recording": evaluation_recording,
             "serial": serial, "raw_frames": len(expected), "label_members": len(label_members),
             "jpeg_members": len(image_members), "missing_label_ids": missing_labels,
             "missing_rgb_ids": missing_images, "invalid_joint_gt_frames": len(invalid_ids),
             "invalid_joint_gt_frame_ids": invalid_ids, "official_joint_gt_valid_frames": len(expected) - len(invalid_ids),
             "eligible_valid_frames": len(valid), "ineligible_official_joint_gt_valid": ineligible,
             "official_joint_gt_validity": "not np.all(joint_3d == -1), official hpe_eval.py pinned revision",
             "frame_range": "all raw frame IDs 0 <= id < source num_frames; no cap"}
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "frame_audit.json", audit)
    if ineligible:
        raise ValueError(f"Cannot represent {len(ineligible)} official jointGT-valid frames "
                         f"as 99D/crop for {evaluation_recording}; see {root / 'frame_audit.json'}")
    manifest = {**job["shared_manifest"], "phase": "complete", "completed_utc": now(),
                "selection_sha256": job["selection_sha256"], "shard_id": sid,
                "subject": subject, "physical_recording": physical,
                "evaluation_recording": evaluation_recording, "serial": serial,
                "frame_audit": audit, "records": [], "rejected": []}
    if not valid:
        manifest["rejected"] = [{"recording": evaluation_recording,
                                  "reason": "all_official_joint_references_invalid", "replacement": False}]
        write_json(manifest_path, manifest)
        return manifest_path, manifest
    ids = np.asarray(sorted(valid), dtype=np.int64)
    poses = np.stack([valid[int(i)][0] for i in ids])
    joints = np.stack([valid[int(i)][1] for i in ids]).astype(np.float32)
    side = entry["meta"]["mano_sides"][0]
    calibration = "mano_" + entry["meta"]["mano_calib"][0]
    beta = calibrations[calibration]
    states, fk_max, fk_sum, fk_n, wrist_max, orth_max = [], 0., 0., 0, 0., 0.
    for begin in range(0, len(ids), 64):
        pose = poses[begin:begin + 64]
        decoded = decoder.decode_pca(pose[:, 3:48], beta, pose[:, :3], pose[:, 48:51],
                                     np.full(len(pose), int(side == "right")))
        state = decoded.state.cpu().numpy()
        tips = [745, 317, 444 if side == "right" else 445, 556, 673]
        mano21 = np.concatenate((decoded.joints_m.cpu().numpy(),
                                 decoded.vertices_m[:, tips].cpu().numpy()), axis=1)
        order = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
        errors = np.linalg.norm(mano21[:, order] - joints[begin:begin + 64], axis=-1)
        fk_max, wrist_max = max(fk_max, float(errors.max())), max(wrist_max, float(errors[:, 0].max()))
        fk_sum, fk_n = fk_sum + float(errors.sum()), fk_n + errors.size
        if fk_max > job["fk_tolerance_m"]:
            raise ValueError(f"MANO/Dex FK disagreement {evaluation_recording}: {fk_max:.8g} m")
        six = state[:, 3:].reshape(-1, 16, 6)
        rotations = np.stack((six[..., :3], six[..., 3:], np.cross(six[..., :3], six[..., 3:])), axis=-1)
        orth_max = max(orth_max, float(np.max(np.abs(rotations.swapaxes(-1, -2) @ rotations - np.eye(3)))))
        state[:, :3] = joints[begin:begin + 64, 0]
        states.append(state)
    intrinsics = np.asarray(job["intrinsics"][serial], dtype=np.float32)
    projected = joints @ intrinsics.T
    projected = projected[..., :2] / projected[..., 2:3]
    bbox = np.concatenate((projected.min(axis=1), projected.max(axis=1)), axis=-1)
    if np.any(bbox[:, 2:] - bbox[:, :2] < 4):
        raise ValueError(f"Official valid GT has degenerate GT crop: {evaluation_recording}")
    stem = "test/" + sid
    reference_path = root / stem / "reference.npz"
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = reference_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, target_world=np.concatenate(states).astype(np.float32),
                 target_valid=np.ones((len(ids), 17), dtype=np.bool_), timestamps=ids.astype(np.float64) / 30,
                 frame_ids=ids, bbox_xyxy=bbox.astype(np.float32), joint_3d=joints,
                 joint_3d_valid=np.ones((len(ids), 21), dtype=np.bool_), beta=beta.astype(np.float32),
                 camera_intrinsics=np.broadcast_to(intrinsics, (len(ids), 3, 3)).copy(),
                 camera_to_world=np.broadcast_to(np.eye(4, dtype=np.float32), (len(ids), 4, 4)).copy(),
                 handedness=np.full(len(ids), int(side == "right"), dtype=np.int64))
    temporary.replace(reference_path)
    paths = [image_members[int(i)][0] for i in ids]
    record = {**REFERENCE_METADATA, "recording": evaluation_recording, "physical_recording": physical,
              "subject": subject, "split": "test", "path": stem, "frames": len(ids), "image_paths": paths,
              "serial": serial, "camera_serial": serial, "handedness": int(side == "right"),
              "sequence_index": entry["index"], "coordinate_frame": f"fixed_camera_{serial}_as_world_gauge",
              "timestamp_source": "nominal_30hz_frame_index_not_measured", "nominal_dt": 1 / 30,
              "source_archive_sha256": job["source_archive_sha256"], "source_archive_identity": job["archive_identity"],
              "source_reference_path": "official source TAR test annotations; independently decoded for evaluation only",
              "reference_file": stem + "/reference.npz", "reference_sha256": file_sha256(reference_path),
              "image_sha256": {image_members[int(i)][0]: image_members[int(i)][1] for i in ids},
              "frame_id_minmax": [int(ids[0]), int(ids[-1])], "frame_gaps": int(np.count_nonzero(np.diff(ids) != 1)),
              "source_num_frames": int(entry["meta"]["num_frames"]), "selection": "all_official_joint_gt_valid_frames",
              "mano_calibration": calibration, "extrinsics_calibration": entry["meta"].get("extrinsics"),
              "gt_complete": True, "fk_joints_max_m": fk_max, "fk_joints_mean_m": fk_sum / fk_n,
              "fk_wrist_max_m": wrist_max, "rotation_orthogonality_error": orth_max}
    manifest["records"] = [record]
    manifest["split_records"] = {"test": 1}
    write_json(manifest_path, manifest)
    return manifest_path, manifest


def prepare_subject(job: dict, events) -> dict:
    import torch
    torch.set_num_threads(1)
    root, subject = Path(job["output"]), job["subject"]
    chosen = {entry["name"]: entry for entry in job["selected"]}
    pending, completed = {}, []
    for name, entry in chosen.items():
        for serial in SERIALS:
            path = root / "shards" / shard_name(subject, name, serial) / "source_manifest.json"
            if path.is_file():
                manifest = verify_completed(path, job["selection_sha256"], subject + "/" + name + "/" + serial)
                event = shard_event(root, path, manifest)
                events.put({"type": "shard", **event})
                completed.append(event)
            else:
                pending[(name, serial)] = entry
    if not pending:
        return {"subject": subject, "shards": len(completed), "resumed_all": True}
    labels, images = {key: {} for key in pending}, {key: {} for key in pending}
    checked_meta, scanned, last_progress = set(), 0, time.monotonic()
    decoder = MANOReferenceDecoder(job["mano_models"], device="cpu", fix_left_shapedirs=False,
                                   asset_version=job["mano_asset_version"])
    if decoder.provenance() != job["shared_manifest"]["decoder_provenance"]:
        raise ValueError("MANO decoder provenance changed during preparation")
    calibrations = load_calibrations(Path(job["dex_root"]))
    published = set()
    with tempfile.TemporaryFile(dir=root, prefix="full_test_labels_" + subject + "_") as spool:
        def publish_ready(name: str) -> None:
            if name not in checked_meta:
                return
            for serial in SERIALS:
                key = (name, serial)
                if key not in pending or key in published:
                    continue
                count = int(chosen[name]["meta"]["num_frames"])
                if len(labels[key]) != count or len(images[key]) != count:
                    continue
                events.put({"type": "subject", "subject": subject, "phase": "decoding",
                            "physical_recording": subject + "/" + name, "serial": serial, "utc": now()})
                # Decoding seeks inside the spool. Restore its append position
                # before the next opaque NPZ write from the ongoing TAR pass.
                append_position = spool.seek(0, 2)
                path, manifest = prepare_view(job, chosen[name], serial, labels[key], images[key],
                                               spool, decoder, calibrations)
                spool.seek(append_position)
                stat = Path(job["archive"]).stat()
                if {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns} != job["archive_identity"]:
                    raise ValueError(f"Source archive changed before shard publication: {subject}")
                event = shard_event(root, path, manifest)
                events.put({"type": "shard", **event})
                completed.append(event)
                published.add(key)

        with tarfile.open(job["archive"], "r|gz") as archive:
            for member in archive:
                scanned += 1
                if time.monotonic() - last_progress >= 30:
                    events.put({"type": "subject", "subject": subject, "phase": "extracting",
                                "tar_members_scanned": scanned, "labels_spooled": sum(map(len, labels.values())),
                                "rgb_extracted": sum(map(len, images.values())), "utc": now()})
                    last_progress = time.monotonic()
                parts = PurePosixPath(member.name).parts
                if not member.isfile() or len(parts) < 3 or parts[0] != subject or parts[1] not in chosen:
                    continue
                name = parts[1]
                if len(parts) == 3 and parts[2] == "meta.yml":
                    if name in checked_meta or read_yaml(archive, member) != chosen[name]["meta"]:
                        raise ValueError(f"Source TAR metadata differs from audited index: {subject}/{name}")
                    checked_meta.add(name)
                    publish_ready(name)
                    continue
                if len(parts) != 4 or (name, parts[2]) not in pending:
                    continue
                serial, key = parts[2], (name, parts[2])
                label_match, image_match = LABEL_PATTERN.fullmatch(parts[3]), JPEG_PATTERN.fullmatch(parts[3])
                match = label_match or image_match
                if match is None:
                    continue
                frame_id = int(match[1])
                if not 0 <= frame_id < int(chosen[name]["meta"]["num_frames"]):
                    raise ValueError(f"Source frame outside complete metadata range: {member.name}")
                destination = labels[key] if label_match else images[key]
                if frame_id in destination or not 0 < member.size <= (5_000_000 if label_match else 20_000_000):
                    raise ValueError(f"Duplicate or invalid source member: {member.name}")
                data = archive.extractfile(member).read()
                if len(data) != member.size:
                    raise ValueError(f"Truncated source member: {member.name}")
                if label_match:
                    offset = spool.tell()
                    spool.write(data)
                    destination[frame_id] = (offset, len(data))
                else:
                    sid = shard_name(subject, name, serial)
                    relative = "test/" + sid + "/images/" + parts[3]
                    path = root / "shards" / sid / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = path.with_suffix(".jpg.tmp")
                    temporary.write_bytes(data)
                    temporary.replace(path)
                    destination[frame_id] = (relative, hashlib.sha256(data).hexdigest())
                publish_ready(name)
        if checked_meta != set(chosen):
            raise ValueError(f"Missing selected TAR metadata: {subject}")
        stat = Path(job["archive"]).stat()
        if {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns} != job["archive_identity"]:
            raise ValueError(f"Source archive changed during extraction: {subject}")
        for (name, serial), entry in sorted(pending.items()):
            if (name, serial) in published:
                continue
            events.put({"type": "subject", "subject": subject, "phase": "decoding",
                        "physical_recording": subject + "/" + name, "serial": serial, "utc": now()})
            path, manifest = prepare_view(job, entry, serial, labels[(name, serial)], images[(name, serial)],
                                           spool, decoder, calibrations)
            event = shard_event(root, path, manifest)
            events.put({"type": "shard", **event})
            completed.append(event)
    return {"subject": subject, "shards": len(completed), "tar_members_scanned": scanned,
            "raw_rgb_extracted": sum(map(len, images.values())), "resumed_all": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dex-root", type=Path, required=True)
    parser.add_argument("--prepared-source", type=Path, required=True,
                        help="Complete FK-audited s0 train/val manifest, adjacent to source_indices/")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mano-models", type=Path, required=True)
    parser.add_argument("--mano-asset-version", default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--fk-tolerance-m", type=float, default=.001)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--code-version", required=True)
    args = parser.parse_args()
    if not Path("/sciclone/home/jchen57").is_dir():
        parser.error("Run all data preprocessing only in the authorized remote ds Pod")
    if args.workers < 1 or args.fk_tolerance_m <= 0:
        parser.error("Positive worker count and FK tolerance required")
    for field in ("dex_root", "prepared_source", "output", "mano_models"):
        setattr(args, field, getattr(args, field).expanduser().resolve())
    prepared = json.loads(args.prepared_source.read_text())
    if (prepared.get("phase") != "complete" or prepared.get("setup") != "s0"
            or not prepared.get("full_fk_verified") or not prepared.get("sequences")
            or any(e["split"] not in ("train", "val") for e in prepared["sequences"])):
        raise ValueError("Require complete, full-FK-audited s0 train/val reference manifest")
    forbidden = {entry["recording"] for entry in prepared["sequences"]}
    selections = [select_subject(args, subject, forbidden) for subject in SUBJECTS[2:]]
    import torch
    torch.set_num_threads(1)
    decoder = MANOReferenceDecoder(args.mano_models, device="cpu", fix_left_shapedirs=False,
                                   asset_version=args.mano_asset_version)
    provenance, previous = decoder.provenance(), prepared["decoder_provenance"]
    if (previous.get("left_shapedirs_fix_applied") or previous.get("left_shapedirs_fix_requested")
            or any(previous["assets"][side]["sha256"] != provenance["assets"][side]["sha256"]
                   for side in ("left", "right"))):
        raise ValueError("Evaluation MANO assets/shape convention differ from audited training source")
    intrinsics = {serial: read_intrinsics(args.dex_root, serial).tolist() for serial in SERIALS}
    shared = {"schema": SOURCE_SCHEMA, "dataset": "dexycb", "setup": "s0_full_test_eval_only",
              "eval_only": True, "training_allowed": False, "purpose": "evaluation_only",
              "created_utc": now(), "code_version": args.code_version, "source_revision": REVISION,
              "decoder_provenance": provenance, "serials": list(SERIALS), "subjects": list(SUBJECTS[2:]),
              "expected_physical_recordings": 160, "expected_recording_views": 1280,
              "selection_rule": "all official s0 test recordings and all eight views; no frame/record cap",
              "split_rule": "sorted100 recordings/subject, index%5==4, subject03--10",
              "reference_source": "official fitted MANO and 21 joints, not independent MoCap",
              "crop_protocol": "project all21 fitted reference joints using original color K; min/max GT box; official HaWoR square1.2 expansion",
              "input_protocol": "GT crop, calibrated original RGB only; eight separately evaluated static cameras; no detector/depth",
              "focal_convention": "official HaWoR single focal uses mean(fx,fy); real K retained for geometry",
              "coordinate_convention": "column vectors; each original static camera defines its own identity world gauge; metres",
              "timestamp_source": "frame_id / nominal30Hz, not measured capture timestamp",
              "gt_shape_convention": "source static calibration beta10; no left shapedirs fix; never model input",
              "geometry_role": "test reference and disclosed GT crop only; predictions never control sample selection",
              "fk_tolerance_m": args.fk_tolerance_m,
              "archive_sha256_source": "previous verified receipt and audited source index; not rehashed this run",
              "license": {"dataset_terms_url": "https://dex-ycb.github.io/",
                          "mano": "Previously caller-supplied licensed assets; no downloading or redistribution"}}
    selection = {"schema": "handloop.dexycb_full_test_selection.v1", "code_version": args.code_version,
                 "script_sha256": file_sha256(Path(__file__)), "prepared_source": path_identity(args.prepared_source),
                 "calibration_archive": path_identity(args.dex_root / "archives/calibration.tar.gz"),
                 "mano_decoder": provenance, "source_index_selection": selections,
                 "excluded_prepared_recordings": sorted(forbidden), "serials": list(SERIALS),
                 "camera_intrinsics": intrinsics, "fk_tolerance_m": args.fk_tolerance_m,
                 "expected_physical_recordings": 160, "expected_recording_views": 1280,
                 "raw_frame_caps": None, "record_caps": None,
                 "joint_gt_validity": "not np.all(joint_3d == -1); every such frame must decode99D and yield crop, else fail"}
    selection_sha = digest_json(selection)
    selection_path = args.output / "selection.json"
    if args.output.exists():
        if not args.resume:
            raise FileExistsError("Existing output requires explicit --resume")
        if not selection_path.is_file() or digest_json(json.loads(selection_path.read_text())) != selection_sha:
            raise ValueError("Resume selection/source/code fingerprint differs; use fresh output")
    else:
        args.output.mkdir(parents=True)
        write_json(selection_path, selection)
    # Repeated launch commands after a disconnected SSH must not race writes.
    lock_stream = (args.output / ".preparation.lock").open("a")
    try:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("Another full-test preparation process owns this output") from exc
    expected_ids = {shard_name(item["subject"], entry["name"], serial)
                    for item in selections for entry in item["selected"] for serial in SERIALS}
    events_by_id, subject_progress, subject_results = {}, {}, []
    started = now()

    def publish(phase="preparing", error=None):
        shards = sorted(events_by_id.values(), key=lambda item: item["shard_id"])
        index = {"schema": "handloop.dexycb_full_test_shards.v1", "phase": phase,
                 "eval_only": True, "training_allowed": False, "selection_sha256": selection_sha,
                 "expected_shards": 1280, "completed_shards": len(shards), "shards": shards,
                 "frames": sum(s["frames"] for s in shards), "raw_frames": sum(s["raw_frames"] for s in shards),
                 "invalid_joint_gt_frames": sum(s["invalid_gt_frames"] for s in shards), "updated_utc": now()}
        write_json(args.output / "shard_index.json", index)
        status = {**index, "started_utc": started, "subject_progress": subject_progress,
                  "completed_subjects": len(subject_results), "error": error}
        status.pop("shards")
        write_json(args.output / "preparation_status.json", status)

    publish()
    jobs = [{**selection, "output": str(args.output), "dex_root": str(args.dex_root),
             "mano_models": str(args.mano_models), "mano_asset_version": args.mano_asset_version,
             "intrinsics": intrinsics, "fk_tolerance_m": args.fk_tolerance_m,
             "selection_sha256": selection_sha, "shared_manifest": shared} for selection in selections]
    context = multiprocessing.get_context("spawn")
    try:
        with context.Manager() as manager:
            events = manager.Queue()
            with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
                pending = {pool.submit(prepare_subject, job, events) for job in jobs}
                while pending:
                    finished, pending = wait(pending, timeout=2, return_when=FIRST_COMPLETED)
                    changed = False
                    while True:
                        try:
                            event = events.get_nowait()
                        except queue.Empty:
                            break
                        if event.pop("type") == "shard":
                            sid = event["shard_id"]
                            if sid not in expected_ids:
                                raise ValueError(f"Unexpected completed test shard: {sid}")
                            if sid in events_by_id and events_by_id[sid] != event:
                                raise ValueError(f"Conflicting duplicate shard: {sid}")
                            events_by_id[sid] = event
                        else:
                            subject_progress[event["subject"]] = event
                        changed = True
                    for future in finished:
                        try:
                            result = future.result()
                        except Exception as exc:
                            # Publish failure immediately, before executor cleanup
                            # waits for any other subject workers to unwind.
                            publish("failed", {"type": type(exc).__name__, "message": str(exc)})
                            for unfinished in pending:
                                unfinished.cancel()
                            raise
                        subject_results.append(result)
                        subject_progress[result["subject"]] = {**result, "phase": "complete", "utc": now()}
                        changed = True
                    if changed:
                        publish()
                        print(json.dumps({"phase": "preparing", "completed_shards": len(events_by_id),
                                          "expected_shards": 1280, "frames": sum(e["frames"] for e in events_by_id.values()),
                                          "completed_subjects": len(subject_results), "utc": now()}), flush=True)
                # A completed Future implies every producer event has been enqueued.
                while True:
                    try:
                        event = events.get_nowait()
                    except queue.Empty:
                        break
                    if event.pop("type") == "shard":
                        events_by_id[event["shard_id"]] = event
        if set(events_by_id) != expected_ids:
            raise ValueError(f"Incomplete full test coverage: {len(events_by_id)}/1280 view shards")
        records, rejected, coverage = [], [], []
        for event in sorted(events_by_id.values(), key=lambda value: value["shard_id"]):
            path = args.output / event["manifest"]
            if file_sha256(path) != event["manifest_sha256"]:
                raise ValueError(f"Shard changed after publish: {path}")
            manifest = json.loads(path.read_text())
            prefix = str(path.parent.relative_to(args.output)) + "/"
            for record in manifest["records"]:
                records.append({**record, "path": prefix + record["path"],
                                "reference_file": prefix + record["reference_file"],
                                "image_paths": [prefix + p for p in record["image_paths"]],
                                "image_sha256": {prefix + p: sha for p, sha in record["image_sha256"].items()}})
            rejected.extend(manifest["rejected"])
            coverage.append(manifest["frame_audit"])
        unique = [record["recording"] for record in records]
        if len(unique) != len(set(unique)):
            raise ValueError("Duplicate recording/view in global test manifest")
        if any(record["physical_recording"] in forbidden for record in records):
            raise ValueError("Full test manifest overlaps audited train/val recording")
        final = {**shared, "phase": "complete", "completed_utc": now(), "selection_sha256": selection_sha,
                 "selection_file": "selection.json", "shard_index": "shard_index.json",
                 "records": records, "rejected": rejected, "frame_coverage": coverage,
                 "subject_audits": subject_results, "split_records": {"test": len(records)},
                 "frames": sum(record["frames"] for record in records),
                 "raw_frames": sum(audit["raw_frames"] for audit in coverage),
                 "official_joint_gt_valid_frames": sum(audit["official_joint_gt_valid_frames"] for audit in coverage),
                 "invalid_joint_gt_frames": sum(audit["invalid_joint_gt_frames"] for audit in coverage)}
        if final["frames"] != final["official_joint_gt_valid_frames"]:
            raise ValueError("Some official jointGT-valid frames were silently omitted")
        write_json(args.output / "source_manifest.json", final)
        publish("complete")
        print(json.dumps({"phase": "complete", "manifest": str(args.output / "source_manifest.json"),
                          "shards": 1280, "records": len(records), "frames": final["frames"], "utc": now()}), flush=True)
    except Exception as exc:
        publish("failed", {"type": type(exc).__name__, "message": str(exc)})
        raise


if __name__ == "__main__":
    main()
