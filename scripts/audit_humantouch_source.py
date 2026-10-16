#!/usr/bin/env python3
"""Read-only HumanTouch native-label audit; run on the remote data host only.

This does not fit MANO, create 99D GT, assign a subject split, or admit data to
training. The sample is a pinned public episode selected before inspecting its
geometry. Pose/world-frame calibration still requires a separate acceptance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


SOURCE = "chuqiaoLyu/Xspark-HumanTouch"
REVISION = "81822a10cac5def2638db546eaeced581d71a8b5"
SOURCE_PATHS = {
    "info": "X001/meta/info.json",
    "episode": "X001/meta/humantouch/episodes/episode_000001.json",
    "parquet": "X001/data/chunk-000/episode_000001.parquet",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stats(values: np.ndarray) -> dict:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = flat[np.isfinite(flat)]
    result = {"count": int(flat.size), "finite_count": int(finite.size)}
    if finite.size:
        result.update(
            mean=float(finite.mean()),
            minimum=float(finite.min()),
            p50=float(np.percentile(finite, 50)),
            p95=float(np.percentile(finite, 95)),
            maximum=float(finite.max()),
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--info", required=True, type=Path)
    parser.add_argument("--episode", required=True, type=Path)
    parser.add_argument("--parquet", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    paths = {name: getattr(args, name) for name in SOURCE_PATHS}
    before = {name: digest(path) for name, path in paths.items()}
    info = json.loads(args.info.read_text())
    episode = json.loads(args.episode.read_text())
    table = pq.read_table(args.parquet)
    rows = table.num_rows
    issues: list[str] = []
    warnings: list[str] = []

    def array(name: str, tail: tuple[int, ...], dtype=np.float64) -> np.ndarray:
        if name not in table.column_names:
            raise ValueError(f"Required native field missing: {name}")
        result = np.asarray(table[name].to_pylist(), dtype=dtype)
        expected = (rows, *tail)
        # LeRobot's built-in scalar columns may use physical scalar storage
        # although meta/info.json advertises shape [1]. Custom scalar fields
        # may instead retain a length-one list; both encode one scalar.
        if tail == (1,) and result.shape == (rows,):
            result = result.reshape(rows, 1)
        if result.shape != expected:
            raise ValueError(f"{name}: shape {result.shape}, expected {expected}")
        return result

    report = {
        "schema": "handloop_humantouch_native_source_audit_v1",
        "source": SOURCE,
        "revision": REVISION,
        "inputs": {
            name: {
                "local_path": str(path.resolve()),
                "source_path": SOURCE_PATHS[name],
                "url": f"https://huggingface.co/datasets/{SOURCE}/resolve/{REVISION}/{SOURCE_PATHS[name]}",
                "sha256": before[name],
                "size_bytes": path.stat().st_size,
            }
            for name, path in paths.items()
        },
        "rows": rows,
        "parquet_columns": table.column_names,
        "declared_fps": info.get("fps"),
        "session": episode.get("session"),
        "source_episode_key": episode.get("source_episode_key"),
        "data_contract": episode.get("data_contract"),
        "joint_names": episode.get("joint_names"),
        "camera_calibration": episode.get("camera_calibration"),
        "hands": {},
        "unresolved": [
            "No MANO/native mapping or anatomical wrist offset has been accepted.",
            "Native positions/orientations are sensor-solved labels, not independent per-joint ground truth.",
            "World declarations and proximity statistics do not establish the tracker-to-skeleton transform.",
            "Subject identities and an official train/val/test split are not established by this sample.",
            "Camera projection and head-tracker extrinsics require image-overlay acceptance before visual training.",
            "This report does not admit the sample to the complete-17-group 99D trainer.",
        ],
    }
    contract = episode.get("data_contract", {}).get("data", {})
    for field, value in {
        "position_unit": "meter",
        "orientation_order": "xyzw",
        "coordinate_frame": "world",
    }.items():
        if contract.get(field) != value:
            issues.append(f"Unexpected declared {field}: {contract.get(field)!r}")
    declared_rows = episode.get("session", {}).get("num_frames_60")
    if rows != declared_rows:
        issues.append(f"Parquet rows {rows} != episode declaration {declared_rows}")
    source_valid = array("observation.source_valid", (1,), bool).reshape(rows)
    pose_valid = array("observation.human.pose.valid", (3,), bool)
    frame = array("frame_index", (1,), np.int64).reshape(rows)
    session_ns = array("observation.session_time_ns", (1,), np.int64).reshape(rows)
    timestamp = array("timestamp", (1,)).reshape(rows)
    dt = np.diff(session_ns).astype(np.float64) * 1e-9
    if np.any(np.diff(frame) != 1):
        issues.append("Frame indexes are not consecutive; do not bridge gaps.")
    if np.any(dt <= 0):
        issues.append("Session timestamps are not strictly increasing.")
    if not np.all(np.isfinite(timestamp)):
        issues.append("Non-finite float timestamp.")
    report["time"] = {
        "frame_first": int(frame[0]) if rows else None,
        "frame_last": int(frame[-1]) if rows else None,
        "session_time_first_ns": int(session_ns[0]) if rows else None,
        "session_time_last_ns": int(session_ns[-1]) if rows else None,
        "dt_seconds": stats(dt),
        "source_valid_count": int(source_valid.sum()),
        "float_timestamp_minus_relative_ns_seconds": stats(
            timestamp - ((session_ns - session_ns[0]) * 1e-9) if rows else []
        ),
    }
    # The 25-node MANUS names are source-native. This checks native segments;
    # it deliberately does not declare any node to be a MANO joint.
    chains = {
        "thumb": [0, 1, 2, 3, 4],
        "index": [0, 5, 6, 7, 8, 9],
        "middle": [0, 10, 11, 12, 13, 14],
        "ring": [0, 15, 16, 17, 18, 19],
        "pinky": [0, 20, 21, 22, 23, 24],
    }
    for side, pose_index in [("left", 1), ("right", 2)]:
        prefix = f"observation.human.hand_skeleton.{side}"
        positions = array(f"{prefix}.position", (25, 3))
        orientations = array(f"{prefix}.orientation", (25, 4))
        valid = array(f"{prefix}.valid", (25,), bool) & source_valid[:, None]
        wrist = array(f"observation.human.pose.wrist_{side}", (7,))
        wrist_valid = pose_valid[:, pose_index] & source_valid
        if not np.all(np.isfinite(positions[valid])):
            issues.append(f"{side}: non-finite native positions marked valid")
        if not np.all(np.isfinite(orientations[valid])):
            issues.append(f"{side}: non-finite native orientations marked valid")
        if not np.all(np.isfinite(wrist[wrist_valid])):
            issues.append(f"{side}: non-finite wrist pose marked valid")
        adjacent = wrist_valid[1:] & wrist_valid[:-1] & (dt > 0)
        delta = np.linalg.norm(np.diff(wrist[:, :3], axis=0), axis=-1)
        shared_root_valid = wrist_valid & valid[:, 0]
        segments = {}
        for finger, nodes in chains.items():
            segments[finger] = []
            for parent, child in zip(nodes[:-1], nodes[1:]):
                mask = valid[:, parent] & valid[:, child]
                segments[finger].append({
                    "source_parent": parent,
                    "source_child": child,
                    "length_m": stats(np.linalg.norm(positions[:, child] - positions[:, parent], axis=-1)[mask]),
                })
        report["hands"][side] = {
            "wrist_valid_count": int(wrist_valid.sum()),
            "joint_valid_counts": valid.sum(axis=0).tolist(),
            "native_quaternion_norm": stats(np.linalg.norm(orientations[valid], axis=-1)),
            "wrist_quaternion_norm": stats(np.linalg.norm(wrist[wrist_valid, 3:], axis=-1)),
            "native_position_values_m": stats(positions[valid]),
            "native_segments": segments,
            "wrist_displacement_per_frame_m": stats(delta[adjacent]),
            "wrist_speed_m_s": stats(delta[adjacent] / dt[adjacent]),
            "root_to_declared_wrist_m_if_same_frame": stats(
                np.linalg.norm(positions[:, 0] - wrist[:, :3], axis=-1)[shared_root_valid]
            ),
            "native_root_norm_m_if_local": stats(np.linalg.norm(positions[valid[:, 0], 0], axis=-1)),
            "root_comparison_interpretation": "Diagnostic hypotheses only; neither establishes a calibrated common coordinate frame.",
        }
    if "subject_id" not in episode and "subject" not in episode:
        warnings.append("No explicit subject identity in selected episode metadata.")
    if not episode.get("camera_calibration"):
        warnings.append("No camera calibration in selected episode metadata.")
    after = {name: digest(path) for name, path in paths.items()}
    report.update(
        structural_pass=not issues,
        ready_for_training=False,
        issues=issues,
        warnings=warnings,
        source_files_unchanged=before == after,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "rows": rows, "structural_pass": not issues, "ready_for_training": False, "source_files_unchanged": before == after}))
    if issues or before != after:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
