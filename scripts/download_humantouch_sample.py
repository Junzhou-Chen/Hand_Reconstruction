#!/usr/bin/env python3
"""Fetch a checksum-verified, pinned native glove episode on remote storage.

Only three annotation/metadata files are downloaded. No RGB, MANO fitting,
split assignment or training admission is performed by this script.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests

from download_twm_data import fetch_file, hf_entry, now, save_json, verified


REPO = "chuqiaoLyu/Xspark-HumanTouch"
REVISION = "81822a10cac5def2638db546eaeced581d71a8b5"
PATHS = (
    "X001/meta/info.json",
    "X001/meta/humantouch/episodes/episode_000001.json",
    "X001/data/chunk-000/episode_000001.parquet",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not Path("/sciclone/data10/jchen57").is_dir():
        parser.error("Download project data only in the authorized remote Pod")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / "download_manifest.json"
    if destination.exists():
        manifest = json.loads(destination.read_text())
        if (manifest.get("repo"), manifest.get("revision"),
                tuple(item["path"] for item in manifest["files"])) != (REPO, REVISION, PATHS):
            raise ValueError("Existing output has a different source or selection")
    else:
        response = requests.get(
            f"https://huggingface.co/api/datasets/{REPO}/revision/{REVISION}",
            params={"blobs": "true"}, timeout=(30, 90))
        response.raise_for_status()
        metadata = response.json()
        if metadata.get("sha") != REVISION or metadata.get("gated") or metadata.get("private"):
            raise ValueError("Pinned public source identity/access changed")
        rows = {item["rfilename"]: item for item in metadata["siblings"]}
        selected = [hf_entry(REPO, REVISION, rows[path]) for path in PATHS]
        if any(item["size"] > 100_000_000 for item in selected):
            raise ValueError("Unexpectedly large sample; inspect scope before downloading")
        manifest = {"schema": "handloop.pinned_native_sample.v1", "dataset": "humantouch",
                    "repo": REPO, "revision": REVISION, "created_utc": now(),
                    "selection": "first named X001 episode, fixed before inspecting poses",
                    "scope": "native label audit only; no RGB and no training admission",
                    "license": "CC-BY-NC-4.0; official ModelScope summary",
                    "files": selected}
        save_json(destination, manifest)
    for entry in manifest["files"]:
        fetch_file(entry, root / entry["path"], attempts=3)
    if not all(verified(root / item["path"], item) for item in manifest["files"]):
        raise ValueError("Sample verification failed")
    status = {"phase": "complete", "completed_utc": now(), "verified": len(PATHS),
              "bytes": sum(item["size"] for item in manifest["files"]),
              "training_ready": False, "manifest": str(destination)}
    save_json(root / "download_status.json", status)
    print(json.dumps(status), flush=True)


if __name__ == "__main__":
    main()
