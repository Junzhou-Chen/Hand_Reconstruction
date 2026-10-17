#!/usr/bin/env python3
"""Publish and inspect an expanded complete99D TWM collection remotely.

Source manifests remain immutable; train/val assignments and coordinates are
retained exactly. The actual ForecastTrajectoryWindows trainer loader performs
window construction and future coverage inspection. No interpolation, gap
bridging, tail repetition or motion/visual test relabeling is performed.

Durations use exported timestamp intervals within contiguous segments. Some
source adapters use an explicit nominal capture cadence rather than measured
sensor timestamps; that evidence and uncertainty remain in source metadata.
Hand-seconds may count two hands; the recording-time union is also reported so
they cannot be confused. This command does not choose weights or train.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from handloop.data.forecast_trajectory import ForecastTrajectoryWindows
from handloop.data.trajectory import (_gt_mask, _load_arrays, _read_manifest,
                                      _storage_keys_and_state_shape)


ORIGINAL_SOURCES = ("oakink2", "hot3d", "umetrack", "dexycb")
ADDITIONAL_SOURCES = ("interhand", "taco", "gigahands", "arctic", "ho3d")
SOURCES = ORIGINAL_SOURCES + ADDITIONAL_SOURCES
PREVIOUS_COLLECTION = Path("/sciclone/data10/jchen57/Hand_Reconstruction/prepared/twm_four_sources_99d_v1/manifest.json")


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def parse_sources(values: list[str], expected: list[str], parser) -> dict[str, Path]:
    if (len(expected) != len(set(expected)) or not set(expected).issubset(SOURCES)
            or not set(ORIGINAL_SOURCES).issubset(expected)
            or not (set(expected) & set(ADDITIONAL_SOURCES))):
        parser.error("--expected-sources must contain every original source and at least one supported addition")
    selected = {}
    for value in values:
        if "=" not in value:
            parser.error("Each --source must have the form NAME=/absolute/path/manifest.json")
        name, raw_path = value.split("=", 1)
        if name not in SOURCES or name in selected or not raw_path.strip():
            parser.error(f"Unknown/duplicate/empty --source: {name!r}")
        selected[name] = Path(raw_path).expanduser().resolve()
    if set(selected) != set(expected):
        parser.error("--source must identify EVERY expected source exactly; unavailable additions cannot be silently dropped: " + ", ".join(expected))
    if len(set(selected.values())) != len(expected):
        parser.error("Every source must reference its own distinct completed manifest")
    return selected


def merged_interval_seconds(intervals: list[tuple[float, float]]) -> float:
    """Union observed intervals for one original recording, across both hands."""
    if not intervals:
        return 0.0
    begin, end = sorted(intervals)[0]
    total = 0.0
    for next_begin, next_end in sorted(intervals)[1:]:
        if next_begin <= end:
            end = max(end, next_end)
        else:
            total += end - begin
            begin, end = next_begin, next_end
    return float(total + end - begin)


def check_original_sources(previous: Path, selected: dict[str, Path]) -> dict:
    """User contract: retain all four prior sources and their EXACT manifests."""
    previous = previous.expanduser().resolve()
    metadata = json.loads(previous.read_text())
    if metadata.get("phase") != "complete":
        raise ValueError("Previous training collection must be complete")
    references = metadata.get("manifests", [])
    original = {entry.get("dataset"): entry for entry in references if isinstance(entry, dict)}
    if set(original) != set(ORIGINAL_SOURCES) or len(references) != len(ORIGINAL_SOURCES):
        raise ValueError("Previous collection must identify the original Oak/HOT/Ume/Dex manifests exactly")
    preserved = {}
    for name in ORIGINAL_SOURCES:
        entry = original[name]
        source = (previous.parent / entry["path"]).resolve()
        expected = entry.get("manifest_sha256")
        if not expected or selected[name] != source or sha256(source) != expected:
            raise ValueError(f"{name}: require the EXACT unchanged source manifest from previous pretraining")
        preserved[name] = {"path": str(source), "manifest_sha256": expected}
    return {"user_requirement": "retain every dataset used by previous TWM pretraining; add new sources without removing/replacing old data",
            "previous_collection": str(previous), "previous_collection_sha256": sha256(previous),
            "preserved_original_sources": preserved,
            "old_source_splits_or_sequences_changed": False,
            "verification": "same resolved source manifest paths and prior pinned SHA256 values"}


def inspect_prepared_source(name: str, path: Path, gap_factor: float = 1.5) -> dict:
    """Scan timestamp/mask metadata for ALL exported sequences, including short ones.

    State values stay lazy here, as in the trainer. All prepared state headers
    must identify [N,99]; every sequence must have an explicit complete17 mask.
    One actual window/source/split is separately materialized by the loader.
    """
    metadata, entries = _read_manifest(path)
    if metadata.get("phase") != "complete" or metadata.get("dataset") != name:
        raise ValueError(f"{name}: require an explicitly identified, completed source manifest")
    if not entries or metadata.get("partial_gt") or metadata.get("gt_complete") is False:
        raise ValueError(f"{name}: empty or partial source manifest")
    results = {}
    original_splits, storage_paths = {}, set()
    for split in ("train", "val"):
        selected = [entry for entry in entries if entry["split"] == split]
        stats = {"hand_sequences": len(selected), "hand_states": 0, "original_recordings": 0,
                 "observed_contiguous_hand_seconds": 0.0, "observed_recording_union_seconds": 0.0,
                 "contiguous_segments": 0, "timestamp_gap_count": 0,
                 "source_timestamp_min_s": None, "source_timestamp_max_s": None,
                 "contiguous_segment_duration_min_s": None, "contiguous_segment_duration_max_s": None,
                 "nominal_dt_values_s": [], "coordinate_frames": [], "storage_formats": {},
                 "timestamp_sources": [],
                 "explicit_complete17_gt_masks": True}
        recording_intervals = defaultdict(list)
        coordinates, nominal_values, storage, timestamp_sources = set(), set(), Counter(), set()
        for entry in selected:
            if entry["dataset"] != name or entry.get("partial_gt") or entry.get("gt_complete") is False:
                raise ValueError(f"{name}: per-entry dataset identity or complete GT differs")
            recording = entry["recording"]
            if original_splits.setdefault(recording, split) != split:
                raise ValueError(f"{name}: original recording crosses train/val: {recording}")
            source = entry["_path"]
            if source in storage_paths:
                raise ValueError(f"{name}: duplicate storage path: {source}")
            storage_paths.add(source)
            keys, shape = _storage_keys_and_state_shape(source)
            if len(shape) != 2 or shape[1] != 99 or shape[0] < 1:
                raise ValueError(f"{source}: require nonempty [N,99] state")
            if not {"timestamps", "gt_mask"}.issubset(keys):
                raise ValueError(f"{source}: explicit timestamps and complete17 gt_mask required")
            frames = shape[0]
            if entry.get("frames", frames) != frames:
                raise ValueError(f"{source}: manifest/state frame count differs")
            small = _load_arrays(source, ("timestamps", "gt_mask"))
            times = np.asarray(small["timestamps"], dtype=np.float64)
            mask = _gt_mask(small["gt_mask"], frames, f"{source}:gt_mask")
            if times.shape != (frames,) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
                raise ValueError(f"{source}: invalid original timestamps")
            if mask.shape != (frames, 17) or not mask.all():
                raise ValueError(f"{source}: full-state trainer requires all 17 groups at every frame")
            nominal = float(entry.get("nominal_dt", np.median(np.diff(times)) if frames > 1 else 0.0))
            if not np.isfinite(nominal) or nominal <= 0:
                raise ValueError(f"{source}: positive nominal interval required")
            intervals = np.diff(times)
            gaps = np.flatnonzero(intervals >= nominal * gap_factor) + 1
            boundaries = np.concatenate(([0], gaps, [frames]))
            durations = []
            for begin, end in zip(boundaries[:-1], boundaries[1:]):
                low, high = float(times[begin]), float(times[end - 1])
                durations.append(high - low)
                recording_intervals[recording].append((low, high))
            stats["hand_states"] += frames
            stats["observed_contiguous_hand_seconds"] += float(sum(durations))
            stats["contiguous_segments"] += len(durations)
            stats["timestamp_gap_count"] += len(gaps)
            for minimum, maximum, low, high in (
                    ("source_timestamp_min_s", "source_timestamp_max_s", float(times[0]), float(times[-1])),
                    ("contiguous_segment_duration_min_s", "contiguous_segment_duration_max_s", min(durations), max(durations))):
                stats[minimum] = low if stats[minimum] is None else min(stats[minimum], low)
                stats[maximum] = high if stats[maximum] is None else max(stats[maximum], high)
            coordinates.add(str(entry.get("coordinate_frame", "unspecified")))
            nominal_values.add(nominal)
            timestamp_sources.add(str(entry.get("timestamp_source", metadata.get("timestamp_source", "unspecified"))))
            storage["npy_directory" if source.is_dir() else "npz"] += 1
        stats["original_recordings"] = len(recording_intervals)
        stats["observed_recording_union_seconds"] = sum(merged_interval_seconds(value) for value in recording_intervals.values())
        stats["observed_contiguous_hand_hours"] = stats["observed_contiguous_hand_seconds"] / 3600
        stats["observed_recording_union_hours"] = stats["observed_recording_union_seconds"] / 3600
        stats["coordinate_frames"], stats["nominal_dt_values_s"] = sorted(coordinates), sorted(nominal_values)
        stats["storage_formats"] = dict(storage)
        stats["timestamp_sources"] = sorted(timestamp_sources)
        results[split] = stats
    if any(entry["split"] not in ("train", "val") for entry in entries):
        raise ValueError(f"{name}: collection rejects test/unknown prepared entries")
    if any(results[split]["hand_sequences"] < 1 for split in ("train", "val")):
        raise ValueError(f"{name}: both train and val need prepared hand sequences")
    return {"source_manifest": str(path), "source_manifest_sha256": sha256(path),
            "source_revision": metadata.get("source_revision"), "by_split": results,
            "duration_definition": "sum last-first exported timestamp intervals per contiguous segment; no endpoint padding; exported times may derive from a documented nominal cadence",
            "hand_duration_note": "two hands count twice in hand-seconds; recording union removes temporal overlap within recording IDs",
            "timing_evidence": {key: metadata.get(key) for key in ("timestamp_source", "nominal_fps",
                "capture_fps_independently_verified", "nominal_fps_evidence", "nominal_fps_evidence_limit")},
            "finite_state_audit_scope": "source preparation provenance plus materialized one window per source per split; no new exhaustive numerical state scan"}


def inspect_materialized_window(dataset, name: str, index: int) -> dict:
    sample = dataset[index]
    h, f = dataset.history, dataset.forecast_frames
    if tuple(sample["past_current"].shape) != (h + 1, 99) or tuple(sample["future"].shape) != (f, 99):
        raise ValueError(f"{name}: loader shapes do not match current forecast trainer")
    if tuple(sample["past_current_gt_mask"].shape) != (h + 1, 17) or not sample["past_current_gt_mask"].all():
        raise ValueError(f"{name}: history lacks complete17 supervision")
    valid = sample["future_valid"]
    if tuple(valid.shape) != (f,) or int(valid.sum()) < dataset.min_future_frames:
        raise ValueError(f"{name}: insufficient available contiguous future")
    masks = sample["future_gt_mask"]
    if tuple(masks.shape) != (f, 17) or not masks[valid].all() or masks[~valid].any():
        raise ValueError(f"{name}: available versus padded future supervision differs")
    for key in ("past_current", "future", "dt", "future_dt", "forecast_dt", "beta"):
        if not torch.isfinite(sample[key]).all():
            raise ValueError(f"{name}: nonfinite materialized {key}")
    if int(sample["dataset_id"]) != dataset.dataset_to_id[name]:
        raise ValueError(f"{name}: materialized source ID differs")
    return {"global_window_index": index, "dataset_id": int(sample["dataset_id"]),
            "history_seconds": float(sample["dt"][1:].sum()),
            "available_future_frames": int(valid.sum()),
            "available_future_max_seconds": float(sample["future_dt"][valid][-1]),
            "future_tail_masked": bool((~valid).any()), "beta_valid": bool(sample["beta_valid"]),
            "complete17_history_and_available_future": True, "finite_materialized_tensors": True}


def inspect_forecast_loader(candidate: Path, args) -> dict:
    inspection = {}
    for split in ("train", "val"):
        dataset = ForecastTrajectoryWindows(candidate, split, history=args.history_frames,
            forecast_frames=args.forecast_horizon, min_future_frames=args.min_future_frames,
            window_stride=args.window_stride, require_complete_gt=True,
            cache_sequences=1, cache_bytes=128 * 1024**2)
        if set(dataset.dataset_names) != set(args.expected_sources):
            raise ValueError("Actual trainer loader must expose every expected source exactly")
        for name in args.expected_sources:
            if dataset.windows_by_dataset[name] < 1:
                raise ValueError(f"{name}: no contiguous {split} windows under actual trainer settings")
        samples = {name: inspect_materialized_window(dataset, name, indices[0])
                   for name, indices in dataset.fixed_indices_by_dataset(1).items()}
        by_source = {}
        steps = sorted({step for step in (1, args.min_future_frames, 15, 30, args.forecast_horizon)
                        if 1 <= step <= args.forecast_horizon})
        for name in args.expected_sources:
            stats = dataset.future_coverage[name]
            windows = dataset.windows_by_dataset[name]
            by_source[name] = {
                "windows": windows, "eligible_hand_sequences": dataset.sequence_counts_by_dataset[name],
                "available_by_future_step": {str(step): {
                    "windows": stats["valid_windows_by_step"][step - 1],
                    "fraction": stats["valid_windows_by_step"][step - 1] / windows} for step in steps},
                "available_by_seconds": {seconds: {"windows": count, "fraction": count / windows}
                                         for seconds, count in stats["windows_reaching_seconds"].items()},
                "all_requested_future_frames_fraction": stats["valid_windows_by_step"][-1] / windows,
                "two_second_target_available_fraction": stats["windows_reaching_seconds"]["2.0"] / windows,
                "some_windows_shorter_than_two_seconds": stats["windows_reaching_seconds"]["2.0"] < windows,
                "maximum_observed_future_seconds": stats["maximum_observed_seconds"],
                "minimum_observed_future_seconds": stats["minimum_observed_seconds"],
                "materialized_window": samples[name]}
        inspection[split] = {"total_windows": len(dataset), "windows": dataset.windows_by_dataset,
                             "sequences": dataset.sequence_counts_by_dataset,
                             "source_revisions": dataset.source_revisions,
                             "by_source": by_source, "forecast_contract": dataset.forecast_contract}
        print(json.dumps({"phase": "inspecting", "split": split, "windows": dataset.windows_by_dataset}), flush=True)
    return inspection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True, metavar="NAME=MANIFEST")
    parser.add_argument("--expected-sources", nargs="+", default=list(SOURCES), choices=SOURCES,
                        help="all selected sources, default all nine; no selected source may be omitted")
    parser.add_argument("--code-version", required=True)
    parser.add_argument("--output", type=Path, required=True, help="new collection manifest JSON path")
    parser.add_argument("--previous-collection", type=Path, default=PREVIOUS_COLLECTION,
                        help="prior four-source collection; requires exact unchanged old source manifests")
    parser.add_argument("--history-frames", type=int, default=30)
    parser.add_argument("--forecast-horizon", type=int, default=60)
    parser.add_argument("--min-future-frames", type=int, default=3)
    parser.add_argument("--window-stride", type=int, default=4)
    args = parser.parse_args()
    if not Path("/sciclone/home/jchen57").is_dir():
        parser.error("Prepare and inspect data only in the authorized remote ds Pod")
    if (args.history_frames < 1 or args.window_stride < 1
            or not 1 <= args.min_future_frames <= args.forecast_horizon):
        parser.error("Require positive history/stride and 1 <= minimum future <= forecast horizon")
    selected = parse_sources(args.source, args.expected_sources, parser)
    preservation = check_original_sources(args.previous_collection, selected)
    args.output = args.output.expanduser().resolve()
    if args.output.exists():
        raise FileExistsError("Use a fresh collection JSON; existing manifests are immutable")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    candidate = args.output.with_name(args.output.name + f".candidate-{os.getpid()}.json")
    if candidate.exists():
        raise FileExistsError(candidate)
    references, source_statistics = [], {}
    for name in args.expected_sources:
        path = selected[name]
        source_statistics[name] = inspect_prepared_source(name, path)
        if (name in ORIGINAL_SOURCES and source_statistics[name]["source_manifest_sha256"]
                != preservation["preserved_original_sources"][name]["manifest_sha256"]):
            raise ValueError(f"{name}: previous source manifest changed during metadata inspection")
        references.append({"path": str(path), "dataset": name,
                           "manifest_sha256": source_statistics[name]["source_manifest_sha256"]})
        print(json.dumps({"phase": "source_metadata_checked", "source": name,
            "states": {split: source_statistics[name]["by_split"][split]["hand_states"] for split in ("train", "val")}}), flush=True)
    collection = {"phase": "complete", "prepared_utc": utc(), "code_version": args.code_version,
        "rotation6d": "columns_c0xyz_c1xyz", "manifests": references, "required_datasets": list(args.expected_sources),
        "scope": "retain ALL prior Oak/HOT/Ume/Dex data via exact immutable source manifests, then ADD every explicitly selected ready dataset; source splits/coordinates retained",
        "selected_additional_sources": [name for name in args.expected_sources if name not in ORIGINAL_SOURCES],
        "unavailable_selected_source_policy": "fail; do not silently drop any selected dataset",
        "original_source_preservation": preservation,
        "position": "actual anatomical wrist J0, metres, source world/fixed-camera gauge anchored per sample",
        "partial_gt_policy": "explicit17 complete masks only; no fitted missing states or imputed GT",
        "sampling_policy": "collection does not select weights; actual trainer/config must explicitly choose sampling",
        "temporal_policy": "actual exported timestamp offsets, with measured versus nominal source evidence preserved; future masks end at first gap or sequence boundary; no extrapolation",
        "loader_arguments": {"history": args.history_frames, "forecast_frames": args.forecast_horizon,
            "min_future_frames": args.min_future_frames, "window_stride": args.window_stride,
            "require_complete_gt": True, "gap_factor": 1.5},
        "source_statistics": source_statistics}
    try:
        atomic_json(candidate, collection)
        collection["loader_inspection"] = inspect_forecast_loader(candidate, args)
        for reference in references:
            if sha256(Path(reference["path"])) != reference["manifest_sha256"]:
                raise ValueError("A source manifest changed while the collection was being inspected")
        if sha256(Path(preservation["previous_collection"])) != preservation["previous_collection_sha256"]:
            raise ValueError("Previous collection changed during expanded collection preparation")
        collection["checks"] = {"exact_expected_sources": True, "expected_source_count": len(args.expected_sources), "all_source_manifests_complete": True,
            "all_prepared_masks_explicit_complete17": True, "test_entries_absent": True,
            "recording_train_val_disjoint_within_source": True,
            "every_source_positive_train_val_windows": True,
            "real_window_materialized_per_source_and_split": True,
            "source_manifest_hashes_unchanged": True, "actual_forecast_loader_used": True,
            "all_previous_pretraining_sources_retained_exactly": True,
            "all_state_values_exhaustively_rescanned": False}
        collection["completed_utc"] = utc()
        atomic_json(candidate, collection)
        if args.output.exists():
            raise FileExistsError("Collection target appeared during inspection; refusing replacement")
        candidate.replace(args.output)
    except Exception:
        candidate.unlink(missing_ok=True)
        raise
    print(json.dumps({"phase": "complete", "manifest": str(args.output),
        "manifest_sha256": sha256(args.output),
        "windows": {split: collection["loader_inspection"][split]["windows"] for split in ("train", "val")}}), flush=True)


if __name__ == "__main__":
    main()
