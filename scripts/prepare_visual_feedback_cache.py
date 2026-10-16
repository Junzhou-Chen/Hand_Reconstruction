#!/usr/bin/env python3
"""Remote-only real DexYCB RGB/reference preparation and frozen HaWoR caching.

The ``sources`` stage extracts only selected single-camera RGB JPEGs from
already-downloaded subject TARs, reusing the audited full99D prepared reference.
No test labels are opened, and recordings are assigned official s0 splits
before image extraction. The ``cache`` stage runs the official pretrained
HaWoR bridge, storing post-IAM image features and post-PAM numeric states.

This is a GT-crop, fixed-camera engineering experiment. It is not the original
HaWoR full test protocol, a detector evaluation, or egocentric world accuracy.
Do not run preprocessing or model inference on the development laptop.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from handloop.data.visual_clips import CACHE_SCHEMA, CORE_KEYS, file_sha256


SOURCE_SCHEMA = "handloop.visual_feedback_sources.v1"
SERIAL = "836212060125"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def read_intrinsics(dex_root: Path, serial: str) -> np.ndarray:
    import yaml
    class CalibrationLoader(yaml.SafeLoader):
        pass
    # Official intrinsics have an unrelated tuple-valued depth extrinsic.
    # Handle only this data container tag; never use the executable YAML loader.
    CalibrationLoader.add_constructor("tag:yaml.org,2002:python/tuple",
                                      lambda loader, node: tuple(loader.construct_sequence(node)))
    name = f"calibration/intrinsics/{serial}_640x480.yml"
    with tarfile.open(dex_root / "archives/calibration.tar.gz", "r:gz") as archive:
        value = yaml.load(archive.extractfile(name).read(), Loader=CalibrationLoader)["color"]
    matrix = np.array([[value["fx"], 0, value["ppx"]],
                       [0, value["fy"], value["ppy"]], [0, 0, 1]], dtype=np.float32)
    if not np.isfinite(matrix).all() or min(matrix[0, 0], matrix[1, 1]) <= 0:
        raise ValueError("Invalid color intrinsics")
    return matrix


def official_s0_split(subject: str, sequence_index: int) -> str:
    # Keep exactly the existing audited source rule and fail closed on test.
    from prepare_dexycb_twm import s0_split
    return s0_split(subject, sequence_index)


def prepare_sources(args: argparse.Namespace) -> None:
    prepared_path = args.prepared / "manifest.json"
    prepared = json.loads(prepared_path.read_text())
    if prepared.get("phase") != "complete" or prepared.get("setup") != "s0":
        raise ValueError("Use the complete audited s0 prepared reference")
    if prepared.get("serial") != args.serial:
        raise ValueError("Do not combine a prepared reference with another camera")
    if (args.output / "source_manifest.json").exists():
        raise FileExistsError("Use a fresh output directory; source manifests are immutable")
    args.output.mkdir(parents=True, exist_ok=True)
    intrinsics = read_intrinsics(args.dex_root, args.serial)
    chosen = []
    for split in ("train", "val"):
        eligible = sorted((r for r in prepared["sequences"]
                           if r["split"] == split and r["subject"] in args.subject
                           and r["frames"] >= args.min_frames
                           and (args.handedness == "both" or
                                int(r["handedness"]) == int(args.handedness == "right"))),
                          key=lambda r: (r["subject"], r["sequence_index"]))
        per_subject = getattr(args, 'records_per_subject_' + split, 0)
        if per_subject:
            for subject in args.subject:
                selected = [r for r in eligible if r['subject'] == subject][:per_subject]
                if not selected:
                    raise ValueError(f'No eligible {split} recording for {subject}')
                chosen.extend(selected)
        else:
            chosen.extend(eligible[:args.records_per_split])
        if not eligible:
            raise ValueError(f"No eligible {split} recording")
    requested: dict[str, dict[str, Path]] = {}
    records = []
    for entry in chosen:
        subject, sequence = entry["recording"].split("/")
        if official_s0_split(subject, int(entry["sequence_index"])) != entry["split"]:
            raise ValueError("Prepared reference disagrees with official recording split")
        if entry["storage"] != "npy" or not entry.get("gt_complete"):
            raise ValueError("This bounded preparation requires complete NPY references")
        source = args.prepared / entry["path"]
        arrays = {key: np.load(source / f"{key}.npy", mmap_mode="r", allow_pickle=False)
                  for key in ("state", "timestamps", "frame_ids", "joint_3d", "gt_mask", "beta")}
        count = min(len(arrays["state"]), args.max_frames) if args.max_frames else len(arrays["state"])
        if count < args.min_frames:
            raise ValueError("Chosen sequence is shorter than requested minimum")
        frame_ids = np.array(arrays["frame_ids"][:count], dtype=np.int64)
        joints = np.array(arrays["joint_3d"][:count], dtype=np.float32)
        if not np.isfinite(joints).all() or np.any(joints[..., 2] <= 0):
            raise ValueError("GT crop requires finite camera-space joints with positive depth")
        projected = joints @ intrinsics.T
        projected = projected[..., :2] / projected[..., 2:3]
        bbox = np.concatenate((projected.min(axis=1), projected.max(axis=1)), axis=-1)
        if np.any(bbox[:, 2:] - bbox[:, :2] < 4):
            raise ValueError("Degenerate projected GT hand box")
        stem = entry["split"] + "/" + subject + "++" + sequence
        destination = args.output / stem
        destination.mkdir(parents=True, exist_ok=False)
        images = []
        for frame_id in frame_ids:
            member = f"{subject}/{sequence}/{args.serial}/color_{int(frame_id):06d}.jpg"
            image_path = destination / "images" / f"color_{int(frame_id):06d}.jpg"
            requested.setdefault(subject, {})[member] = image_path
            images.append(str(image_path.relative_to(args.output)))
        with (destination / "reference.npz").open("wb") as stream:
            np.savez(stream, target_world=np.array(arrays["state"][:count], dtype=np.float32),
                     target_valid=np.array(arrays["gt_mask"][:count], dtype=np.bool_),
                     timestamps=np.array(arrays["timestamps"][:count], dtype=np.float64),
                     frame_ids=frame_ids, bbox_xyxy=bbox.astype(np.float32),
                     joint_3d=joints, beta=np.array(arrays["beta"], dtype=np.float32),
                     camera_intrinsics=np.broadcast_to(intrinsics, (count, 3, 3)).copy(),
                     camera_to_world=np.broadcast_to(np.eye(4, dtype=np.float32), (count, 4, 4)).copy(),
                     handedness=np.full(count, int(entry["handedness"]), dtype=np.int64))
        records.append({"recording": entry["recording"], "split": entry["split"], "path": stem,
                        "frames": count, "image_paths": images, "serial": args.serial,
                        "handedness": int(entry["handedness"]), "sequence_index": entry["sequence_index"],
                        "coordinate_frame": f"fixed_camera_{args.serial}_as_world_gauge",
                        "timestamp_source": "nominal_30hz_frame_index_not_measured",
                        "nominal_dt": 1 / 30, "source_archive_sha256": entry["source_archive_sha256"],
                        "source_reference_path": str(source), "reference_file": stem + "/reference.npz",
                        "frame_id_minmax": [int(frame_ids[0]), int(frame_ids[-1])],
                        "frame_gaps": int(np.count_nonzero(np.diff(frame_ids) != 1))})
    manifest = {"schema": SOURCE_SCHEMA, "phase": "extracting", "created_utc": now(),
                "dataset": "dexycb", "setup": "s0_train_val_bounded", "records": records,
                "reference_manifest": str(prepared_path),
                "reference_manifest_sha256": file_sha256(prepared_path),
                "reference_source": "official fitted MANO and joints, not independent MoCap",
                "crop_protocol": "project all 21 fitted reference joints with original color K; min/max GT box; official HaWoR adds its own square 1.2 expansion",
                "input_protocol": "GT crop on reference-valid selected frames; no detector; single fixed camera; no depth input",
                "focal_convention": "official HaWoR single focal uses mean(fx,fy); real K retained for geometry",
                "coordinate_convention": "column vectors; original camera is identity world gauge; metres",
                "extraction": {"requested_images": sum(map(len, requested.values())), "extracted_images": 0}}
    save_json(args.output / "source_manifest.incomplete.json", manifest)
    for subject, wanted in requested.items():
        remaining = set(wanted)
        with tarfile.open(args.dex_root / "archives" / f"{subject}.tar.gz", "r|gz") as archive:
            for member in archive:
                if member.name not in wanted:
                    continue
                if not member.isfile() or member.size > 20_000_000:
                    raise ValueError(f"Unexpected JPEG member {member.name}")
                target = wanted[member.name]
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("wb") as stream:
                    shutil.copyfileobj(archive.extractfile(member), stream)
                remaining.remove(member.name)
                manifest["extraction"]["extracted_images"] += 1
                if not remaining:
                    break
        if remaining:
            raise FileNotFoundError(f"Selected source JPEGs missing ({len(remaining)}): {sorted(remaining)[:3]}")
    manifest["phase"] = "complete"
    manifest["completed_utc"] = now()
    manifest["split_records"] = dict(Counter(r["split"] for r in records))
    save_json(args.output / "source_manifest.json", manifest)
    (args.output / "source_manifest.incomplete.json").unlink()
    print(json.dumps({"phase": "complete", "manifest": str(args.output / "source_manifest.json"),
                      "split_records": manifest["split_records"], "images": manifest["extraction"]}))


def numpy_value(value):
    import torch
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def prepare_cache(args: argparse.Namespace) -> None:
    # Model import stays lazy so source preparation does not load CUDA/models.
    from handloop.visual.hawor import FrozenHaWoRBridge
    source_path = args.source_manifest.expanduser().resolve()
    sources = json.loads(source_path.read_text())
    if sources.get("schema") != SOURCE_SCHEMA or sources.get("phase") != "complete":
        raise ValueError("Use a completed source manifest")
    if (args.output / "manifest.json").exists():
        raise FileExistsError("Do not replace an existing frozen cache")
    args.output.mkdir(parents=True, exist_ok=True)
    bridge = FrozenHaWoRBridge.from_pretrained(
        repo_root=args.hawor_root, checkpoint_path=args.hawor_checkpoint,
        config_path=args.hawor_config, mano_model_dir=args.mano_models, device=args.device)
    records = []
    manifest = {"schema": CACHE_SCHEMA, "phase": "extracting", "created_utc": now(), "records": records,
                "source_manifest": str(source_path), "source_manifest_sha256": file_sha256(source_path),
                "dataset": sources["dataset"], "setup": sources["setup"],
                "crop_protocol": sources["crop_protocol"], "input_protocol": sources["input_protocol"],
                "coordinate_convention": sources["coordinate_convention"],
                "reference_source": sources["reference_source"],
                "focal_convention": sources["focal_convention"],
                "crop_transform_convention": "source_pixels_to_hawor_192x256_normalized",
                "hawor_checkpoint_sha256": file_sha256(args.hawor_checkpoint),
                "hawor_config_sha256": file_sha256(args.hawor_config),
                "hawor_bridge_provenance": bridge.provenance(),
                "mano_asset_sha256": {side: file_sha256(args.mano_models / f"MANO_{side.upper()}.pkl")
                                      for side in ("left", "right")},
                "feature_source": "official pretrained post-IAM patches; post-PAM raw state; bidirectional nonoverlapping 16-frame chunks",
                "feature_dtype": args.feature_dtype}
    save_json(args.output / "manifest.incomplete.json", manifest)
    for entry in sources["records"]:
        if entry["split"] not in ("train", "val"):
            raise ValueError("Held-out test cannot enter this cache")
        with np.load(source_path.parent / entry["reference_file"], allow_pickle=False) as archive:
            reference = {key: archive[key].copy() for key in archive.files}
        images = [str(source_path.parent / p) for p in entry["image_paths"]]
        # Invalid reference frames were excluded by the audited source adapter.
        # Never let a HaWoR temporal block silently bridge the remaining gaps.
        boundaries = np.concatenate(([0], np.flatnonzero(np.diff(reference["frame_ids"]) != 1) + 1,
                                     [len(images)]))
        pieces: dict[str, list[np.ndarray]] = {}
        feature_chunks = []
        for begin, end in zip(boundaries[:-1], boundaries[1:]):
            result_piece = bridge.extract_clip(image_paths=images[begin:end],
                    bbox_xyxy=reference["bbox_xyxy"][begin:end],
                    is_right=reference["handedness"][begin:end],
                    intrinsics=reference["camera_intrinsics"][begin:end],
                    camera_to_world=reference["camera_to_world"][begin:end],
                    detected=np.ones(end - begin, dtype=np.bool_), focal_convention="mean_fx_fy")
            for key, value in result_piece.items():
                if hasattr(value, "shape"):
                    pieces.setdefault(key, []).append(numpy_value(value))
            feature_chunks.extend((int(start), int(min(start + 16, end)))
                                  for start in range(begin, end, 16))
        result = {key: np.concatenate(values) for key, values in pieces.items()}
        # Support the bridge's leading batch dimension only when it is exactly 1.
        for key in ("image_features", "raw_state_world", "raw_available", "beta", "crop_transform"):
            value = result[key]
            expected_ndim = {"image_features": 3, "raw_state_world": 2,
                             "raw_available": 2, "beta": 2, "crop_transform": 3}[key]
            if value.ndim == expected_ndim + 1 and value.shape[0] == 1:
                result[key] = value[0]
        n = len(images)
        # The official crop is 256 square; ViT drops 32 pixels on each x side.
        # Core query positions live in the remaining 192x256 region's [-1,1].
        crop_to_patch = np.array([[2 / 192, 0, -1 - 63 / 192],
                                  [0, 2 / 256, -1 + 1 / 256], [0, 0, 1]], dtype=np.float32)
        crop_transform = crop_to_patch @ result["crop_transform"]
        reset = np.concatenate(([True], np.diff(reference["frame_ids"]) != 1)).astype(np.bool_)
        arrays = {"patch_features": result["image_features"].astype(args.feature_dtype),
                  "raw_world": result["raw_state_world"].astype(np.float32),
                  "availability": result["raw_available"].astype(np.bool_),
                  "timestamps": reference["timestamps"].astype(np.float64),
                  "frame_valid": result["frame_valid"].astype(np.bool_),
                  "image_valid": (result["image_valid"].astype(np.bool_) &
                                  result["bbox_valid"].astype(np.bool_) &
                                  result["frame_valid"].astype(np.bool_)),
                  "handedness": reference["handedness"].astype(np.int64),
                  "world_to_camera": np.linalg.inv(reference["camera_to_world"]).astype(np.float32),
                  "camera_intrinsics": reference["camera_intrinsics"].astype(np.float32),
                  "crop_transform": crop_transform.astype(np.float32),
                  "beta": result["beta"].astype(np.float32), "reset": reset,
                  "target_world": reference["target_world"].astype(np.float32),
                  "target_valid": reference["target_valid"].astype(np.bool_)}
        if set(arrays) != set(CORE_KEYS) or any(len(v) != n for v in arrays.values()):
            raise ValueError("Bridge returned incompatible frame count/schema")
        for key, value in arrays.items():
            if not np.isfinite(value).all():
                raise ValueError(f"Nonfinite frozen cache tensor {key}")
        destination = args.output / entry["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=destination.name + ".tmp-", dir=destination.parent))
        try:
            for key, value in arrays.items():
                np.save(temporary / f"{key}.npy", value, allow_pickle=False)
            temporary.replace(destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        metadata = {key: value for key, value in entry.items() if key != "image_paths"}
        metadata["tensor_sha256"] = {key: file_sha256(destination / f"{key}.npy") for key in CORE_KEYS}
        metadata["tensor_bytes"] = {key: (destination / f"{key}.npy").stat().st_size for key in CORE_KEYS}
        metadata["source_reference_sha256"] = file_sha256(source_path.parent / entry["reference_file"])
        metadata["feature_chunks"] = feature_chunks
        records.append(metadata)
        save_json(args.output / "manifest.incomplete.json", manifest)
        print(json.dumps({"cached_recording": entry["recording"], "split": entry["split"], "frames": n}), flush=True)
    manifest["phase"] = "complete"
    manifest["completed_utc"] = now()
    save_json(args.output / "manifest.json", manifest)
    (args.output / "manifest.incomplete.json").unlink()
    print(json.dumps({"phase": "complete", "manifest": str(args.output / "manifest.json"), "records": len(records)}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("sources", "cache"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dex-root", type=Path)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--subject", action="append", default=None)
    parser.add_argument("--serial", default=SERIAL)
    parser.add_argument("--records-per-split", type=int, default=1)
    parser.add_argument("--records-per-subject-train", type=int, default=0,
                        help="If positive, select this many train recordings per requested subject")
    parser.add_argument("--records-per-subject-val", type=int, default=0,
                        help="If positive, select this many validation recordings per requested subject")
    parser.add_argument("--min-frames", type=int, default=32)
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--handedness", choices=("both", "right", "left"), default="right")
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--hawor-root", type=Path)
    parser.add_argument("--hawor-checkpoint", type=Path)
    parser.add_argument("--hawor-config", type=Path)
    parser.add_argument("--mano-models", type=Path)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--feature-dtype", choices=("float32", "float16"), default="float16")
    args = parser.parse_args()
    args.subject = args.subject or ["20200709-subject-01"]
    if (args.records_per_split < 1 or args.min_frames < 1 or args.max_frames < 0
            or args.records_per_subject_train < 0 or args.records_per_subject_val < 0):
        parser.error("Invalid bounded source sizes")
    required = (("dex_root", "prepared") if args.stage == "sources" else
                ("source_manifest", "hawor_root", "hawor_checkpoint", "hawor_config", "mano_models"))
    if any(getattr(args, key) is None for key in required):
        parser.error("Missing required --stage options: " + ", ".join(required))
    prepare_sources(args) if args.stage == "sources" else prepare_cache(args)


if __name__ == "__main__":
    main()
