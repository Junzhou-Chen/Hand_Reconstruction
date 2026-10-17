#!/usr/bin/env python3
"""Inspect/acquire public DexWild human numerical data through HTTP ranges.

The pinned release wraps each human HDF5 in an uncompressed, split TAR.
Only explicitly selected HDF5 numerical arrays may be copied; downloading is
not native-MANO conversion, world-coordinate validation, or training admission.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tarfile
import time

import h5py
import requests

from download_twm_data import hf_entry, now, save_json
from hf_range_file import HFConcatRangeFile

REPO = "boardd/dexwild-dataset"
REVISION = "df5cd775f985e79036ff23ad579abbe7752b8df1"
TASKS = ("clothes", "florist", "pour", "spray", "toy")
POLICY = "native-manus-trackers-timestamps-no-media-no-leap-v1"
ALLOWED_GROUPS = {"left_manus", "right_manus", "left_tracker", "right_tracker",
                  "zed", "timesteps", "intergripper", "metadata", "calibration",
                  "config", "validity", "tracking_lost_ratios"}
MEDIA_SUFFIXES = (".jpg", ".jpeg", ".png", ".gif", ".mp4", ".avi")


def source_parts(task):
    url = f"https://huggingface.co/api/datasets/{REPO}/revision/{REVISION}"
    response = requests.get(url, params={"blobs": "true"}, timeout=(20, 90))
    response.raise_for_status()
    source = response.json()
    if source.get("sha") != REVISION or source.get("gated") or source.get("private"):
        raise ValueError("Pinned source identity/access mismatch")
    regex = rf"{task}_data/human/human_{task}_data\.part_(\d+)"
    rows = [(int(m[1]), hf_entry(REPO, REVISION, item))
            for item in source["siblings"]
            if (m := re.fullmatch(regex, item["rfilename"]))]
    rows.sort(key=lambda row: row[0])
    if not rows or [row[0] for row in rows] != list(range(len(rows))):
        raise ValueError("Missing or duplicated human split archive parts")
    return [entry for _, entry in rows]


def open_source(parts, task):
    with HFConcatRangeFile(REPO, REVISION, parts) as archive:
        header = archive.read(512)
    member = tarfile.TarInfo.frombuf(header, "utf-8", "surrogateescape")
    if member.name != f"human_{task}_data.hdf5" or not member.isfile():
        raise ValueError("Unexpected first TAR member; inspect source before proceeding")
    if member.size + 512 > sum(row["size"] for row in parts):
        raise ValueError("HDF5 window extends beyond split archive")
    return HFConcatRangeFile(REPO, REVISION, parts, base=512, size=member.size)


def probe(task, directory):
    parts = source_parts(task)
    result = {"queried_utc": now(), "repo": REPO, "revision": REVISION,
              "task": task, "parts": parts, "training_ready": False, "tree": []}
    with open_source(parts, task) as remote:
        with h5py.File(remote, "r") as source:
            def inspect(group, depth):
                count = len(group)
                keys = []
                limit = 1 if count > 100 else 30
                def collect(name):
                    keys.append(name.decode("utf-8"))
                    return 1 if len(keys) >= limit else None
                group.id.links.iterate(collect)
                print(json.dumps({"inspecting": group.name, "children_count": count,
                                  "first_children": keys}), flush=True)
                result["tree"].append({"path": group.name, "kind": "group",
                                       "children_count": count, "first_children": keys,
                                       "attribute_keys": list(group.attrs.keys())})
                for name in keys[:1] if len(keys) > 100 else keys[:30]:
                    if name.lower().endswith(MEDIA_SUFFIXES):
                        continue
                    if group.name != "/" and group.name.count("/") == 1 and name not in ALLOWED_GROUPS:
                        continue
                    obj = group[name]
                    if isinstance(obj, h5py.Dataset):
                        result["tree"].append({"path": obj.name, "kind": "dataset",
                                               "shape": obj.shape, "dtype": str(obj.dtype),
                                               "chunks": obj.chunks, "compression": obj.compression})
                    elif depth:
                        inspect(obj, depth - 1)
            inspect(source, 4)
        result["network_bytes"] = remote.network_bytes
        result["network_requests"] = remote.network_requests
    save_json(directory / f"{task}_source_probe.json", result)
    print(json.dumps({key: value for key, value in result.items() if key != "parts"}), flush=True)


def plan(directory):
    path = directory / "download_manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text())
        if (manifest.get("revision"), manifest.get("policy"), manifest.get("repo")) != (REVISION, POLICY, REPO):
            raise ValueError("Existing output contains a conflicting source or extraction policy")
        return manifest
    tasks = {task: source_parts(task) for task in TASKS}
    if {task: len(rows) for task, rows in tasks.items()} != {"clothes": 12, "florist": 13, "pour": 5, "spray": 28, "toy": 29}:
        raise ValueError("Pinned human archive part counts differ from audited source")
    total_bytes = sum(e["size"] for rows in tasks.values() for e in rows)
    if total_bytes != 1_817_035_653_120:
        raise ValueError("Pinned human source-container byte count changed")
    manifest = {"schema": "handloop.dexwild_range_trajectories.v1", "repo": REPO,
                "revision": REVISION, "policy": POLICY, "created_utc": now(),
                "tasks": tasks, "source_container_bytes": total_bytes,
                "training_ready": False,
                "scope": "Human MANUS, tracker, ZED pose/timestamps and metadata arrays only; no media arrays or LEAP robot actions.",
                "reference_limit": "Audited pour first episode has timestamp plus 10 native points (71 columns), not 25-point full MANUS or full MANO GT.",
                "integrity": "Pinned HF revision, source part sizes/LFS hashes recorded; strict HTTP ranges and TAR header checksum. Extracted HDF5 files have local SHA256 and read-back checks. Full source-part SHA256 is NOT verified because the parts are not fully downloaded."}
    save_json(path, manifest)
    return manifest


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def copy_attributes(source, destination):
    for key, value in source.attrs.items():
        destination.attrs[key] = value


def copy_numerical_group(source, destination, records, skipped):
    import numpy as np
    copy_attributes(source, destination)
    for name in source.keys():
        if name.lower().endswith(MEDIA_SUFFIXES):
            skipped.append(source.name + "/" + name)
            continue
        value = source[name]
        if isinstance(value, h5py.Group):
            copy_numerical_group(value, destination.create_group(name), records, skipped)
            continue
        is_string = h5py.check_string_dtype(value.dtype) is not None
        if value.dtype.kind not in "biufcSU" and not is_string:
            raise ValueError(f"Unsupported non-numerical source field: {value.name}")
        if value.dtype.kind in "biu" and value.ndim >= 3 and value.shape[-1] in (1, 3, 4):
            raise ValueError(f"Unexpected image-like array in numerical group: {value.name}")
        if value.size * max(1, value.dtype.itemsize) > 128 * 1024 * 1024:
            raise ValueError(f"Unexpectedly large numerical dataset: {value.name}")
        # Bounded blocks avoid large individual HDF5 fileobj reads. Object
        # fields are source UTF8/ASCII timestamps/metadata, never pickle loads.
        result = destination.create_dataset(name, shape=value.shape, dtype=value.dtype)
        copy_attributes(value, result)
        if value.ndim == 0:
            data = value[()]
            result[()] = data
            if result[()] != data:
                raise ValueError("Scalar source read-back differs")
        elif value.shape[0]:
            row_bytes = max(1, value.dtype.itemsize * int(np.prod(value.shape[1:])))
            step = max(1, (4 * 1024 * 1024) // row_bytes)
            for start in range(0, value.shape[0], step):
                section = slice(start, min(start + step, value.shape[0]))
                data = value[section]
                result[section] = data
                if not np.array_equal(result[section], data, equal_nan=value.dtype.kind in "fc"):
                    raise ValueError("Numerical source read-back differs")
        records.append({"source_path": value.name, "shape": list(value.shape), "dtype": str(value.dtype),
                        "logical_bytes": int(value.size * value.dtype.itemsize)})


def episode_verified(target, receipt_path, task, episode):
    if not receipt_path.exists():
        return False
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get("revision"), receipt.get("policy"), receipt.get("task"), receipt.get("episode")) != (REVISION, POLICY, task, episode):
        raise ValueError("Existing episode provenance differs; refusing overwrite")
    if not target.exists():
        return False
    return target.stat().st_size == receipt["file_bytes"] and sha256(target) == receipt["sha256"]


def extract_task(task, parts, output, limit_episodes=None):
    directory = Path(output)
    task_dir = directory / task
    task_dir.mkdir(exist_ok=True)
    state = {"task": task, "phase": "opening_source", "started_utc": now(),
             "episodes_total": None, "verified_episodes": 0, "file_bytes": 0,
             "logical_bytes": 0, "network_bytes": 0, "errors": {}, "training_ready": False}
    status_path = directory / f"{task}_status.json"
    def report(remote=None, **updates):
        state.update(updates, updated_utc=now())
        if remote is not None:
            state.update(network_bytes=remote.network_bytes, network_requests=remote.network_requests)
        save_json(status_path, state)
    report()
    try:
        with open_source(parts, task) as remote:
            with h5py.File(remote, "r") as source:
                episodes = sorted(name for name in source.keys() if re.fullmatch(r"ep_\d+", name))
                if len(episodes) != len(source):
                    raise ValueError("Unexpected non-episode root entries; inspect before proceeding")
                state["source_episodes_total"] = len(episodes)
                if limit_episodes is not None:
                    episodes = episodes[:limit_episodes]
                report(remote, phase="extracting", episodes_total=len(episodes))
                last_report = time.monotonic()
                for episode in episodes:
                    target = task_dir / (episode + ".h5")
                    receipt_path = task_dir / (episode + ".receipt.json")
                    if episode_verified(target, receipt_path, task, episode):
                        receipt = json.loads(receipt_path.read_text())
                    else:
                        if target.exists() and receipt_path.exists():
                            raise ValueError(f"Existing verified episode is corrupt: {task}/{episode}")
                        temporary = target.with_name(target.name + ".partial")
                        records, skipped = [], []
                        with h5py.File(temporary, "w") as result:
                            copy_attributes(source, result)
                            group = source[episode]
                            copy_attributes(group, result)
                            selected = sorted(set(group.keys()) & ALLOWED_GROUPS)
                            if not any(name.endswith("_manus") for name in selected):
                                raise ValueError(f"No native MANUS group: {task}/{episode}")
                            for name in selected:
                                obj = group[name]
                                if not isinstance(obj, h5py.Group):
                                    raise ValueError("Expected numerical group changed type")
                                copy_numerical_group(obj, result.create_group(name), records, skipped)
                            result.attrs["handloop_source_revision"] = REVISION
                            result.attrs["handloop_source_episode"] = episode
                            result.attrs["handloop_training_ready"] = False
                        receipt = {"task": task, "episode": episode, "revision": REVISION,
                                   "policy": POLICY, "completed_utc": now(), "source_datasets": records,
                                   "skipped_media_entries": skipped, "file_bytes": temporary.stat().st_size,
                                   "logical_bytes": sum(row["logical_bytes"] for row in records),
                                   "sha256": sha256(temporary), "source_part_sha256_verified": False,
                                   "training_ready": False}
                        # Replacing a complete orphan is safe: it can only be
                        # from this unfinished extraction; a conflicting receipt
                        # has already been rejected above.
                        temporary.replace(target)
                        save_json(receipt_path, receipt)
                    state["verified_episodes"] += 1
                    state["file_bytes"] += receipt["file_bytes"]
                    state["logical_bytes"] += receipt["logical_bytes"]
                    if state["verified_episodes"] <= 2 or time.monotonic() - last_report >= 20:
                        report(remote, current_episode=episode)
                        last_report = time.monotonic()
                report(remote, phase="sample_complete" if limit_episodes is not None else "complete")
    except Exception as error:
        # Requests' signed URLs are sanitized by the range helper. Other error
        # classes are retained without arbitrary exception text.
        report(phase="failed", errors={"exception": type(error).__name__})
        raise
    return state


def download(directory, workers):
    manifest = plan(directory)
    started = now()
    def report(phase):
        states = {}
        for task in TASKS:
            path = directory / f"{task}_status.json"
            states[task] = json.loads(path.read_text()) if path.exists() else {"phase": "pending"}
        summary = {"dataset": "dexwild_human_trajectories", "phase": phase,
                   "started_utc": started, "updated_utc": now(), "workers": workers,
                   "tasks": states, "training_ready": False,
                   "source_container_bytes": manifest["source_container_bytes"],
                   "verified_episodes": sum(row.get("verified_episodes", 0) for row in states.values()),
                   "file_bytes": sum(row.get("file_bytes", 0) for row in states.values()),
                   "network_bytes": sum(row.get("network_bytes", 0) for row in states.values())}
        save_json(directory / "download_status.json", summary)
        print(json.dumps(summary), flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(extract_task, task, manifest["tasks"][task], str(directory)) for task in TASKS]
        while not all(future.done() for future in futures):
            report("running")
            time.sleep(20)
        errors = []
        for future in futures:
            try:
                future.result()
            except Exception as error:
                errors.append(type(error).__name__)
        report("failed" if errors else "complete")
        if errors:
            raise RuntimeError("Task extraction failed: " + ",".join(errors))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("probe", "plan", "download", "smoke"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", choices=TASKS, default="pour")
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    directory = args.output.resolve()
    guard = Path("/sciclone/data10/jchen57")
    if not guard.is_dir() or guard not in directory.parents:
        parser.error("Dataset inspection/acquisition runs in the authorized remote Pod")
    directory.mkdir(parents=True, exist_ok=True)
    if not 1 <= args.workers <= 5:
        parser.error("workers must be between 1 and 5")
    with (directory / ".download.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another process holds this dataset's download lock")
        if args.action == "probe":
            probe(args.task, directory)
        elif args.action == "smoke":
            print(json.dumps(extract_task(args.task, source_parts(args.task), str(directory), limit_episodes=1)), flush=True)
        elif args.action == "plan":
            manifest = plan(directory)
            print(json.dumps({key: value for key, value in manifest.items() if key != "tasks"}), flush=True)
        else:
            download(directory, args.workers)


if __name__ == "__main__":
    main()
