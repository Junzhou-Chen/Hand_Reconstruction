#!/usr/bin/env python3
"""Remote CPU-only DexYCB s0 test sources, isolated from all training loaders.

Select recordings from audited metadata BEFORE opening their test labels.
Default: first two official test recordings of each subject03--10, with the
raw frame-id prefix 0 <= frame_id < 128 (NOT the first 128 valid frames).
Each subject TAR is streamed once for selected labels, JPEGs and metadata.
No model prediction, validation score, or replacement recording enters the
selection. Official fitted MANO/joints and GT crops are evaluation references;
they are not independent MoCap or an automatic detector/SLAM benchmark.
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
import sys
import tarfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from handloop.data.meta_mano import MANOReferenceDecoder
from handloop.data.visual_clips import file_sha256
from prepare_dexycb_twm import (REVISION, SUBJECTS, LABEL_PATTERN,
                                load_calibrations, read_yaml, s0_split)
from prepare_visual_feedback_cache import SOURCE_SCHEMA, SERIAL, now, read_intrinsics


JPEG_PATTERN = re.compile(r"color_(\d{6})\.jpg\Z")


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def path_identity(path: Path) -> dict:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": file_sha256(path)}


def selected_index(args, prepared: dict, subject: str, forbidden: set[str]) -> dict:
    """Metadata-only, fail-closed selection from a complete, unchanged index."""
    index_path = args.prepared_source.parent / "source_indices" / (subject + ".json")
    indexed = json.loads(index_path.read_text())
    archive_path = args.dex_root / "archives" / (subject + ".tar.gz")
    stat = archive_path.stat()
    identity = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if indexed.get("subject") != subject or indexed.get("archive_identity") != identity:
        raise ValueError(f"Audited source index/archive identity mismatch: {subject}")
    entries = indexed.get("sequences", [])
    names = [entry["name"] for entry in entries]
    if len(entries) != 100 or names != sorted(set(names)):
        raise ValueError(f"Expected exactly 100 unique sorted indexed recordings: {subject}")
    for index, entry in enumerate(entries):
        if (entry["index"] != index or entry["split"] != s0_split(subject, index)
                or not re.fullmatch(r"[A-Za-z0-9_-]+", entry["name"])):
            raise ValueError(f"Invalid official source-index ordering/split: {subject}")
    chosen = [entry for entry in entries if entry["split"] == "test"][:args.records_per_subject]
    if len(chosen) != args.records_per_subject:
        raise ValueError(f"Insufficient metadata-selected test recordings: {subject}")
    for entry in chosen:
        recording = subject + "/" + entry["name"]
        if recording in forbidden:
            raise ValueError(f"Test recording overlaps prepared train/val: {recording}")
        meta = entry["meta"]
        if (len(meta["mano_sides"]) != 1 or len(meta["mano_calib"]) != 1
                or meta["mano_sides"][0] not in ("left", "right")
                or args.serial not in meta["serials"] or int(meta["num_frames"]) < 1):
            raise ValueError(f"Unsupported selected recording metadata: {recording}")
    receipt_path = args.dex_root / "download_receipts.json"
    receipt = json.loads(receipt_path.read_text()).get("archives/" + archive_path.name, {})
    source_sha = indexed.get("source_sha256") or receipt.get("sha256")
    if (not isinstance(source_sha, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", source_sha)
            or (receipt.get("size") is not None and receipt["size"] != stat.st_size)
            or (receipt.get("sha256") and receipt["sha256"] != source_sha)):
        raise ValueError(f"Verified download receipt/index SHA mismatch: {subject}")
    return {"subject": subject, "archive": str(archive_path), "archive_identity": identity,
            "source_archive_sha256": source_sha,
            "source_index": path_identity(index_path), "selected": chosen,
            "official_split_counts": dict(Counter(entry["split"] for entry in entries))}


def prepare_subject(job: dict) -> dict:
    import torch
    torch.set_num_threads(1)
    output, root = Path(job["output"]), Path(job["dex_root"])
    subject, serial = job["subject"], job["serial"]
    chosen = {entry["name"]: entry for entry in job["selected"]}
    labels = {name: {} for name in chosen}
    images = {name: {} for name in chosen}
    selected_meta = set()
    raw_limits = {name: min(job["raw_prefix_frames"], int(entry["meta"]["num_frames"]))
                  for name, entry in chosen.items()}
    wanted_members = sum(raw_limits.values())
    metadata_count, scanned = 0, 0
    with tarfile.open(job["archive"], "r|gz") as archive:
        for member in archive:
            scanned += 1
            parts = PurePosixPath(member.name).parts
            if not member.isfile() or len(parts) < 3 or parts[0] != subject or parts[1] not in chosen:
                continue
            name = parts[1]
            if len(parts) == 3 and parts[2] == "meta.yml":
                if name in selected_meta or read_yaml(archive, member) != chosen[name]["meta"]:
                    raise ValueError(f"Selected TAR metadata differs from audited index: {subject}/{name}")
                selected_meta.add(name)
                metadata_count += 1
            elif len(parts) == 4 and parts[2] == serial:
                label_match = LABEL_PATTERN.fullmatch(parts[3])
                image_match = JPEG_PATTERN.fullmatch(parts[3])
                match = label_match or image_match
                if match is None or int(match[1]) >= raw_limits[name]:
                    continue
                frame_id = int(match[1])
                destination = labels[name] if label_match else images[name]
                if frame_id in destination or member.size > (5_000_000 if label_match else 20_000_000):
                    raise ValueError(f"Duplicate or oversized source member: {member.name}")
                data = archive.extractfile(member).read()
                if len(data) != member.size:
                    raise ValueError(f"Truncated source member: {member.name}")
                if label_match:
                    destination[frame_id] = data
                else:
                    relative = Path("test") / (subject + "++" + name) / "images" / parts[3]
                    path = output / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(data)
                    destination[frame_id] = str(relative)
            if (metadata_count == len(chosen) and sum(map(len, labels.values())) == wanted_members
                    and sum(map(len, images.values())) == wanted_members):
                break
    if selected_meta != set(chosen):
        raise ValueError(f"Missing selected TAR metadata: {subject}")
    archive_stat = Path(job["archive"]).stat()
    if {"bytes": archive_stat.st_size, "mtime_ns": archive_stat.st_mtime_ns} != job["archive_identity"]:
        raise ValueError(f"Source archive changed during extraction: {subject}")
    decoder = MANOReferenceDecoder(job["mano_models"], device="cpu", fix_left_shapedirs=False,
                                   asset_version=job["mano_asset_version"])
    provenance = decoder.provenance()
    if provenance != job["decoder_provenance"]:
        raise ValueError("MANO decoder provenance changed between selection and extraction")
    calibrations, intrinsics = load_calibrations(root), np.asarray(job["intrinsics"], dtype=np.float32)
    records, rejected, frame_audit = [], [], []
    for name, entry in sorted(chosen.items()):
        skipped, valid = Counter(), {}
        for frame_id, data in sorted(labels[name].items()):
            with np.load(io.BytesIO(data), allow_pickle=False) as label:
                pose, joints = label["pose_m"].copy(), label["joint_3d"].copy()
            if pose.shape != (1, 51) or joints.shape != (1, 21, 3):
                raise ValueError(f"Unexpected label shape: {subject}/{name}/{frame_id}")
            if np.all(pose == 0):
                skipped["pose_all_zero"] += 1
            elif np.any(np.all(joints == -1, axis=-1)):
                skipped["joint_sentinel"] += 1
            elif not np.isfinite(pose).all() or not np.isfinite(joints).all():
                skipped["nonfinite"] += 1
            else:
                valid[frame_id] = (pose[0], joints[0])
        missing_labels = sorted(set(range(raw_limits[name])) - set(labels[name]))
        audit = {"recording": subject + "/" + name, "raw_prefix_limit_exclusive": raw_limits[name],
                 "label_members": len(labels[name]), "jpeg_members": len(images[name]),
                 "missing_label_ids": missing_labels, "skipped": dict(skipped), "valid_frames": len(valid)}
        frame_audit.append(audit)
        if not valid:
            rejected.append({**audit, "reason": "no_valid_frames_in_fixed_raw_prefix", "replacement": False})
            continue
        ids = np.asarray(sorted(valid), dtype=np.int64)
        if any(int(frame_id) not in images[name] for frame_id in ids):
            raise ValueError(f"JPEG missing for reference-valid frame: {subject}/{name}")
        pose = np.stack([valid[int(i)][0] for i in ids])
        joints = np.stack([valid[int(i)][1] for i in ids]).astype(np.float32)
        side = entry["meta"]["mano_sides"][0]
        calibration = "mano_" + entry["meta"]["mano_calib"][0]
        beta = calibrations[calibration]
        states, fk_errors, wrist_errors, rotation_errors = [], [], [], []
        for begin in range(0, len(ids), 64):
            batch_pose = pose[begin:begin + 64]
            decoded = decoder.decode_pca(batch_pose[:, 3:48], beta, batch_pose[:, :3],
                                         batch_pose[:, 48:51], np.full(len(batch_pose), int(side == "right")))
            state = decoded.state.cpu().numpy()
            tips = [745, 317, 444 if side == "right" else 445, 556, 673]
            all_joints = np.concatenate((decoded.joints_m.cpu().numpy(),
                                         decoded.vertices_m[:, tips].cpu().numpy()), axis=1)
            order = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
            error = np.linalg.norm(all_joints[:, order] - joints[begin:begin + 64], axis=-1)
            if float(error.max()) > job["fk_tolerance_m"]:
                raise ValueError(f"MANO/Dex FK disagreement: {subject}/{name}, max={error.max():.9g} m")
            fk_errors.extend(error.ravel().tolist())
            wrist_errors.extend(error[:, 0].tolist())
            six = state[:, 3:].reshape(-1, 16, 6)
            first, second = six[..., :3], six[..., 3:]
            rotations = np.stack((first, second, np.cross(first, second)), axis=-1)
            rotation_errors.append(float(np.max(np.abs(rotations.swapaxes(-1, -2) @ rotations - np.eye(3)))))
            state[:, :3] = joints[begin:begin + 64, 0]
            states.append(state)
        if np.any(joints[..., 2] <= 0):
            raise ValueError(f"GT crop requires positive-depth reference joints: {subject}/{name}")
        projected = joints @ intrinsics.T
        projected = projected[..., :2] / projected[..., 2:3]
        bbox = np.concatenate((projected.min(axis=1), projected.max(axis=1)), axis=-1)
        if np.any(bbox[:, 2:] - bbox[:, :2] < 4):
            raise ValueError(f"Degenerate GT crop: {subject}/{name}")
        stem = "test/" + subject + "++" + name
        reference_path = output / stem / "reference.npz"
        reference_path.parent.mkdir(parents=True, exist_ok=True)
        with reference_path.open("wb") as stream:
            np.savez(stream, target_world=np.concatenate(states).astype(np.float32),
                     target_valid=np.ones((len(ids), 17), dtype=np.bool_), timestamps=ids.astype(np.float64) / 30,
                     frame_ids=ids, bbox_xyxy=bbox.astype(np.float32), joint_3d=joints,
                     beta=beta.astype(np.float32),
                     camera_intrinsics=np.broadcast_to(intrinsics, (len(ids), 3, 3)).copy(),
                     camera_to_world=np.broadcast_to(np.eye(4, dtype=np.float32), (len(ids), 4, 4)).copy(),
                     handedness=np.full(len(ids), int(side == "right"), dtype=np.int64))
        image_paths = [images[name][int(i)] for i in ids]
        records.append({"recording": subject + "/" + name, "subject": subject, "split": "test", "path": stem,
                        "frames": len(ids), "image_paths": image_paths, "serial": serial,
                        "handedness": int(side == "right"), "sequence_index": entry["index"],
                        "coordinate_frame": f"fixed_camera_{serial}_as_world_gauge",
                        "timestamp_source": "nominal_30hz_frame_index_not_measured", "nominal_dt": 1 / 30,
                        "source_archive_sha256": job["source_archive_sha256"],
                        "source_archive_identity": job["archive_identity"],
                        "source_reference_path": "official source TAR test annotations; independently decoded for eval only",
                        "reference_file": stem + "/reference.npz", "reference_sha256": file_sha256(reference_path),
                        "image_sha256": {path: file_sha256(output / path) for path in image_paths},
                        "frame_id_minmax": [int(ids[0]), int(ids[-1])],
                        "frame_gaps": int(np.count_nonzero(np.diff(ids) != 1)),
                        "source_num_frames": int(entry["meta"]["num_frames"]),
                        "raw_prefix_limit_exclusive": raw_limits[name], "skipped": dict(skipped),
                        "mano_calibration": calibration,
                        "extrinsics_calibration": entry["meta"].get("extrinsics"), "gt_complete": True,
                        "fk_joints_max_m": max(fk_errors), "fk_joints_mean_m": float(np.mean(fk_errors)),
                        "fk_wrist_max_m": max(wrist_errors), "rotation_orthogonality_error": max(rotation_errors)})
    return {"subject": subject, "records": records, "rejected": rejected, "frame_audit": frame_audit,
            "tar_members_scanned": scanned, "extracted_prefix_jpegs": sum(map(len, images.values())),
            "referenced_valid_jpegs": sum(record["frames"] for record in records)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dex-root", type=Path, required=True)
    parser.add_argument("--prepared-source", type=Path, required=True,
                        help="Audited complete s0 train/val manifest.json, with adjacent source_indices/")
    parser.add_argument("--output", type=Path, required=True, help="Fresh directory; existing paths are refused")
    parser.add_argument("--mano-models", type=Path, required=True)
    parser.add_argument("--mano-asset-version", default=None)
    parser.add_argument("--mano-provenance", type=Path)
    parser.add_argument("--license-file", type=Path, action="append", default=[])
    parser.add_argument("--subjects", nargs="+", choices=SUBJECTS[2:], default=list(SUBJECTS[2:]))
    parser.add_argument("--serial", choices=(SERIAL,), default=SERIAL)
    parser.add_argument("--records-per-subject", type=int, default=2)
    parser.add_argument("--raw-prefix-frames", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--fk-tolerance-m", type=float, default=.001)
    parser.add_argument("--code-version", default="unrecorded")
    args = parser.parse_args()
    if not Path("/sciclone/home/jchen57").is_dir():
        parser.error("Run all data preprocessing only in the authorized remote ds Pod")
    if (not 1 <= args.records_per_subject <= 20 or args.raw_prefix_frames < 1 or args.workers < 1
            or args.fk_tolerance_m <= 0 or len(set(args.subjects)) != len(args.subjects)):
        parser.error("Invalid subject/count/prefix/worker/tolerance arguments")
    args.dex_root = args.dex_root.expanduser().resolve()
    args.prepared_source = args.prepared_source.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.mano_models = args.mano_models.expanduser().resolve()
    prepared = json.loads(args.prepared_source.read_text())
    if (prepared.get("phase") != "complete" or prepared.get("setup") != "s0"
            or prepared.get("serial") != args.serial or not prepared.get("full_fk_verified")):
        raise ValueError("Require complete, FK-audited s0 source manifest with the same serial")
    if not prepared.get("sequences") or any(entry["split"] not in ("train", "val")
                                             for entry in prepared["sequences"]):
        raise ValueError("Reference prepared source must contain train/val only")
    forbidden = {entry["recording"] for entry in prepared["sequences"]}
    import torch
    torch.set_num_threads(1)
    decoder = MANOReferenceDecoder(args.mano_models, device="cpu", fix_left_shapedirs=False,
                                   asset_version=args.mano_asset_version)
    provenance = decoder.provenance()
    original = prepared["decoder_provenance"]
    for side in ("left", "right"):
        if original["assets"][side]["sha256"] != provenance["assets"][side]["sha256"]:
            raise ValueError("Evaluation MANO assets differ from audited source decoder")
    if original.get("left_shapedirs_fix_applied") or original.get("left_shapedirs_fix_requested"):
        raise ValueError("Dex source MANO shape convention must have no left shapedirs fix")
    indices = [selected_index(args, prepared, subject, forbidden) for subject in args.subjects]
    if args.output.exists():
        raise FileExistsError("Use a fresh eval-only output directory")
    intrinsic = read_intrinsics(args.dex_root, args.serial)
    args.output.mkdir(parents=True, exist_ok=False)
    optional_provenance = args.mano_provenance or args.mano_models.parent / "provenance.json"
    manifest = {"schema": SOURCE_SCHEMA, "phase": "extracting", "created_utc": now(),
                "dataset": "dexycb", "setup": "s0_test_eval_only", "eval_only": True,
                "training_allowed": False, "records": [], "rejected": [], "subject_audits": [],
                "code_version": args.code_version, "source_revision": REVISION,
                "prepared_source": path_identity(args.prepared_source), "decoder_provenance": provenance,
                "source_index_selection": indices, "excluded_prepared_recordings": sorted(forbidden),
                "selection_rule": "first N official s0 test recordings per chosen subject, fixed from metadata before opening labels; no replacement",
                "raw_prefix_frames_exclusive": args.raw_prefix_frames,
                "prefix_semantics": "0 <= raw frame_id < limit; not first N valid frames; invalid references filtered without extending prefix",
                "subjects": args.subjects, "records_per_subject": args.records_per_subject,
                "serial": args.serial, "split_rule": "sorted 100 recordings/subject, index%5==4 and subject03--10 => test",
                "archive_sha256_source": "previous verified download receipt and audited metadata index; not rehashed this run",
                "license": {"dataset_terms_url": "https://dex-ycb.github.io/",
                            "mano": "Caller-supplied licensed MANO assets; no downloading or redistribution",
                            "mano_provenance": path_identity(optional_provenance) if optional_provenance.is_file() else None,
                            "provided_license_files": [path_identity(path) for path in args.license_file]},
                "reference_source": "official fitted MANO and joints, not independent MoCap",
                "crop_protocol": "project all 21 fitted reference joints with original color K; min/max GT box; official HaWoR adds its own square 1.2 expansion",
                "input_protocol": "GT crop on reference-valid selected frames; no detector; single fixed camera; no depth input",
                "focal_convention": "official HaWoR single focal uses mean(fx,fy); real K retained for geometry",
                "coordinate_convention": "column vectors; original camera is identity world gauge; metres",
                "timestamp_source": "numeric frame index / nominal 30Hz; not measured capture timestamps",
                "fk_tolerance_m": args.fk_tolerance_m, "fk_reference": "provided 21 fitted joints, not independent GT",
                "gt_shape_convention": "static source calibration beta10; source MANO no left shapedirs fix; never supplied as model input",
                "geometry_role": "test target and disclosed GT crop only; no model prediction controls selection",
                "calibration_archive": path_identity(args.dex_root / "archives/calibration.tar.gz")}
    write_json(args.output / "selection.json", manifest)
    write_json(args.output / "source_manifest.incomplete.json", manifest)
    jobs = [{**item, "output": str(args.output), "dex_root": str(args.dex_root), "serial": args.serial,
             "raw_prefix_frames": args.raw_prefix_frames, "mano_models": str(args.mano_models),
             "mano_asset_version": args.mano_asset_version, "decoder_provenance": provenance,
             "intrinsics": intrinsic.tolist(), "fk_tolerance_m": args.fk_tolerance_m} for item in indices]
    try:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = [pool.submit(prepare_subject, job) for job in jobs]
            for done, future in enumerate(as_completed(futures), 1):
                result = future.result()
                manifest["records"].extend(result.pop("records"))
                manifest["rejected"].extend(result.pop("rejected"))
                manifest["subject_audits"].append(result)
                write_json(args.output / "source_manifest.incomplete.json", manifest)
                status = {"phase": "extracting", "done_subjects": done, "total_subjects": len(jobs),
                          "last_subject": result["subject"], "valid_frames": sum(r["frames"] for r in manifest["records"]),
                          "utc": now()}
                write_json(args.output / "preparation_status.json", status)
                print(json.dumps(status), flush=True)
        manifest["records"].sort(key=lambda record: record["recording"])
        manifest["subject_audits"].sort(key=lambda item: item["subject"])
        if not manifest["records"] or any(r["recording"] in forbidden or r["split"] != "test" for r in manifest["records"]):
            raise ValueError("Empty evaluation set or train/val overlap")
        manifest.update(phase="complete", completed_utc=now(), split_records={"test": len(manifest["records"])},
                        extraction={"extracted_prefix_jpegs": sum(s["extracted_prefix_jpegs"] for s in manifest["subject_audits"]),
                                    "referenced_valid_jpegs": sum(r["frames"] for r in manifest["records"])})
        write_json(args.output / "source_manifest.json", manifest)
        (args.output / "source_manifest.incomplete.json").unlink()
        status = {"phase": "complete", "manifest": str(args.output / "source_manifest.json"),
                  "records": len(manifest["records"]), "rejected": len(manifest["rejected"]),
                  "frames": manifest["extraction"]["referenced_valid_jpegs"], "utc": now()}
        write_json(args.output / "preparation_status.json", status)
        print(json.dumps(status), flush=True)
    except Exception as exc:
        write_json(args.output / "preparation_status.json", {"phase": "failed", "utc": now(),
                   "error_type": type(exc).__name__, "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
