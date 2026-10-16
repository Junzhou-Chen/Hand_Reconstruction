#!/usr/bin/env python3
"""Compose immutable RGB/reference sources without mixing geometry conventions.

Publish a fresh self-contained manifest tree. Prefer hard links to immutable
source images/reference files; copy only across filesystems. Symlinks are not
used because the RGB loader intentionally disallows paths resolving outside
its own manifest root. All published files retain the exact source SHA256.

Keep each dataset's own joint order, units, reference codec and parent split.
Do not sample data, inspect predictions, change validation labels or construct
new train/val boundaries. This command runs only in the authorized remote Pod.
"""
from __future__ import annotations

import argparse
from collections import Counter
import errno
import json
import os
from pathlib import Path
import re
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from handloop.data.visual_clips import file_sha256
from prepare_visual_feedback_cache import SOURCE_SCHEMA, now, save_json
from prepare_visual_joint_sources import FOCAL_CONVENTION, identity, inside


GEOMETRY_KEYS = ("dataset", "reference_source", "reference_joint_order", "reference_units",
                 "reference_coordinate_frame", "reference_left_shapedirs_fix", "joint_3d_order",
                 "joint_3d_units", "joint_3d_coordinate_frame", "joint_3d_valid_policy",
                 "reference_joint_missing_policy")


def publish(source: Path, target: Path, expected: str | None) -> dict:
    original = identity(source)
    if expected is not None and expected != original["sha256"]:
        raise ValueError(f"Source SHA differs from its completed manifest: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if file_sha256(target) != original["sha256"]:
            raise ValueError("Shared source image maps to different target content")
        return {"mode": "deduplicated_existing_exact_file", "sha256": original["sha256"], "bytes": original["bytes"]}
    try:
        os.link(source, target)
        mode = "hardlink_immutable_source"
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
        shutil.copy2(source, target)
        mode = "cross_device_byte_copy"
    if file_sha256(target) != original["sha256"] or identity(source) != original:
        raise ValueError("Source changed while publishing an exact collection member")
    return {"mode": mode, "sha256": original["sha256"], "bytes": original["bytes"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-version", default="unrecorded")
    args = parser.parse_args()
    if not Path("/sciclone/home/jchen57").is_dir():
        parser.error("Compose data only in the authorized remote ds Pod")
    args.output = args.output.expanduser().resolve()
    paths = [path.expanduser().resolve() for path in args.source_manifest]
    if len(paths) < 2 or len(paths) != len(set(paths)):
        parser.error("Specify two or more distinct completed visual source manifests")
    sources, source_identities, record_ids, physical_splits = [], [], set(), {}
    for index, path in enumerate(paths):
        source_identity = identity(path)
        source = json.loads(path.read_text())
        if (source.get("schema") != SOURCE_SCHEMA or source.get("phase") != "complete"
                or source.get("focal_convention") != FOCAL_CONVENTION or not source.get("records")
                or source.get("eval_only") or source.get("training_allowed") is False):
            raise ValueError("Require completed training-allowed visual source manifests with explicit focal convention")
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", source.get("dataset", "")):
            raise ValueError("Collection source dataset names must be safe identifiers")
        for record in source["records"]:
            dataset, split = record.get("dataset", source.get("dataset")), record["split"]
            if not isinstance(dataset, str) or not re.fullmatch(r"[a-zA-Z0-9_-]+", dataset) or split not in ("train", "val"):
                raise ValueError("Require named train/val datasets; test sources are forbidden")
            physical = record.get("parent_recording", record.get("split_group", record["recording"]))
            key = (dataset, physical)
            if key in physical_splits and physical_splits[key] != split:
                raise ValueError(f"Physical parent recording crosses train/val: {key}")
            physical_splits[key] = split
            if record["recording"] in record_ids:
                raise ValueError("Recording IDs must already be unique across collection sources")
            record_ids.add(record["recording"])
            if not record.get("coordinate_frame") or not record.get("timestamp_source"):
                raise ValueError("Every source must preserve coordinate/time provenance")
            count = int(record["frames"])
            if count < 1 or len(record["image_paths"]) != count:
                raise ValueError("Source RGB/reference counts disagree")
            for key in GEOMETRY_KEYS:
                value = record.get(key, source.get(key))
                if key in ("dataset", "reference_source", "joint_3d_order", "joint_3d_units", "joint_3d_coordinate_frame") and value is None:
                    raise ValueError(f"Explicit geometry provenance required before combining sources: {key}")
        sources.append((path, source, f"source{index:02d}_" + source["dataset"]))
        source_identities.append(source_identity)
    if args.output.exists():
        raise FileExistsError("Use a fresh collection directory; existing sources remain immutable")
    args.output.mkdir(parents=True, exist_ok=False)
    output = {"schema": SOURCE_SCHEMA, "phase": "composing", "created_utc": now(), "dataset": "mixed",
        "setup": "dataset_specific_existing_train_val_splits", "code_version": args.code_version,
        "focal_convention": FOCAL_CONVENTION,
        "reference_source": "dataset-specific official fitted references; see every record/source codec, not independent MoCap",
        "crop_protocol": "dataset-specific GT crop protocols preserved; see sources",
        "input_protocol": "RGB only; each source retains camera metadata and its GT-crop pilot limitations",
        "coordinate_convention": "column vectors; metres; per-record camera/world gauge and joint order preserved",
        "source_manifests": source_identities, "source_protocols": [], "records": [],
        "publication": {"policy": "hardlink immutable source; copy only for EXDEV; no symlink", "counts": {}, "unique_files": 0}}
    published, modes = {}, Counter()
    incomplete = args.output / "source_manifest.incomplete.json"
    save_json(incomplete, output)
    for path, source, prefix in sources:
        original_manifest_sha256 = next(item["sha256"] for item in source_identities if item["path"] == str(path))
        output["source_protocols"].append({key: value for key, value in source.items() if key not in
            ("records", "clip_audits", "source_index_audits", "completed_subjects")})
        for record in source["records"]:
            result = dict(record)
            for key in GEOMETRY_KEYS:
                if key not in result and key in source:
                    result[key] = source[key]
            result["origin_source_manifest"] = str(path)
            result["origin_source_manifest_sha256"] = original_manifest_sha256
            result["origin_reference_file"] = record["reference_file"]
            result["path"] = prefix + "/" + record["path"]
            reference = prefix + "/" + record["reference_file"]
            if reference not in published:
                receipt = publish(inside(path.parent, record["reference_file"]), inside(args.output, reference),
                                  record.get("reference_sha256"))
                published[reference] = receipt
                modes[receipt["mode"]] += 1
            result["reference_file"], result["reference_sha256"] = reference, published[reference]["sha256"]
            image_paths, image_hashes, source_jpeg_hashes = [], {}, {}
            for original in record["image_paths"]:
                relative = prefix + "/" + original
                if relative not in published:
                    receipt = publish(inside(path.parent, original), inside(args.output, relative),
                                      record.get("image_sha256", {}).get(original))
                    published[relative] = receipt
                    modes[receipt["mode"]] += 1
                image_paths.append(relative)
                image_hashes[relative] = published[relative]["sha256"]
                if original in record.get("source_jpeg_sha256", {}):
                    source_jpeg_hashes[relative] = record["source_jpeg_sha256"][original]
            result["image_paths"], result["image_sha256"] = image_paths, image_hashes
            if source_jpeg_hashes:
                result["source_jpeg_sha256"] = source_jpeg_hashes
            output["records"].append(result)
        save_json(incomplete, output)
        print(json.dumps({"phase": "composing", "completed_source": str(path),
                          "hand_records": len(output["records"]), "unique_files": len(published)}), flush=True)
    if any(identity(path) != original for path, original in zip(paths, source_identities)):
        raise ValueError("Input source manifest changed during collection publication")
    output["phase"], output["completed_utc"] = "complete", now()
    output["records"].sort(key=lambda record: (record["split"], record["dataset"], record["recording"]))
    output["publication"]["counts"], output["publication"]["unique_files"] = dict(modes), len(published)
    output["split_records"] = dict(Counter(record["split"] for record in output["records"]))
    output["dataset_split_frames"] = {dataset: {split: sum(record["frames"] for record in output["records"]
        if record["dataset"] == dataset and record["split"] == split) for split in ("train", "val")}
        for dataset in sorted({record["dataset"] for record in output["records"]})}
    output["checks"] = {"physical_parent_train_val_disjoint": True, "test_records_absent": True,
                        "all_image_reference_bytes_preserved": True, "per_record_geometry_protocol_preserved": True,
                        "source_manifests_unchanged": True}
    save_json(args.output / "source_manifest.json", output)
    incomplete.unlink()
    print(json.dumps({"phase": "complete", "manifest": str(args.output / "source_manifest.json"),
        "split_records": output["split_records"], "dataset_split_frames": output["dataset_split_frames"]}), flush=True)


if __name__ == "__main__":
    main()
