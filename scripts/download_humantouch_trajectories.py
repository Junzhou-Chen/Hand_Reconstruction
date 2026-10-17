#!/usr/bin/env python3
"""Download pinned HumanTouch numerical trajectories and metadata remotely.

Parquets include native hand/wrist/timestamp fields and embedded numerical touch
data. Independent RGB/depth media are excluded. Downloading is not MANO fitting,
split assignment, calibration acceptance, or admission to TWM training.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
from functools import partial
import json
from pathlib import Path
import re
import threading
import time
from urllib.parse import quote

import requests

from download_twm_data import download, fetch_file, hf_entry, now, save_json


REPO = "chuqiaoLyu/Xspark-HumanTouch"
REVISION = "81822a10cac5def2638db546eaeced581d71a8b5"
SCHEMA = "handloop.humantouch_trajectory_download.v1"
REMOTE_ROOT = Path("/sciclone/data10/jchen57")
PARQUET_PATTERN = r"X\d{3}/data/chunk-\d+/episode_\d+\.parquet"
META_PATTERN = r"X\d{3}/meta/.+\.(?:json|jsonl)"
EPISODE_META_PATTERN = (
    r"(?P<session>X\d{3})/meta/humantouch/episodes/episode_(?P<episode>\d+)\.json"
)
SELECTION = {
    "readme": "README.md",
    "parquet_fullmatch": PARQUET_PATTERN,
    "metadata_fullmatch": META_PATTERN,
    "standalone_rgb_and_depth": "excluded",
    "embedded_numeric_touch": "retained inside the source parquet",
}
EXPECTED = {
    "files": 35_246,
    "total_bytes": 110_417_888_793,
    "parquet_files": 17_576,
    "parquet_bytes": 108_985_108_170,
    "metadata_files": 17_669,
    "metadata_bytes": 1_432_776_863,
}


class RateLimitedGet:
    """Coordinate HF resolver requests across workers and honor reset headers."""
    def __init__(self, directory, requests_per_second=4.0):
        self.directory = directory
        self.interval = 1.0 / requests_per_second
        self.current_interval = self.interval
        self.next_request = 0.0
        self.paused_until = 0.0
        self.mutex = threading.Lock()
        self.rate_limit_responses = 0

    def __call__(self, *args, **kwargs):
        while True:
            while True:
                with self.mutex:
                    current = time.monotonic()
                    delay = max(self.next_request, self.paused_until) - current
                    if delay <= 0:
                        self.next_request = current + self.current_interval
                        break
                time.sleep(min(delay, 5.0))
            response = requests.get(*args, **kwargs)
            limits = re.search(r'r=(\d+);t=(\d+)', response.headers.get("RateLimit", ""))
            remaining, reset = map(int, limits.groups()) if limits else (None, None)
            with self.mutex:
                current = time.monotonic()
                if remaining is not None:
                    # Keep a quarter of the remaining quota available to other
                    # clients sharing the same egress address.
                    self.current_interval = max(self.interval, reset / max(remaining * 0.75, 1))
                    if remaining == 0:
                        self.paused_until = max(self.paused_until, current + reset + 3)
                if response.status_code == 429:
                    self.rate_limit_responses += 1
                    retry = response.headers.get("Retry-After", "")
                    wait = int(retry) if retry.isdigit() else reset if reset is not None else 300
                    self.paused_until = max(self.paused_until, current + max(wait, 5) + 3)
                save_json(self.directory / "http_rate_limit_status.json", {
                    "updated_utc": now(), "last_http_status": response.status_code,
                    "remaining_resolver_quota": remaining, "reset_seconds": reset,
                    "request_interval_seconds": self.current_interval,
                    "http_429_responses": self.rate_limit_responses,
                    "cooldown_seconds": max(0, self.paused_until - current),
                    "policy": "at most 4 resolver requests/s; shared-quota-aware throttle; honor 429 reset",
                })
            if response.status_code != 429:
                return response
            # Never spend a file's finite retry budget on server rate limiting.
            # Other HTTP/transport errors retain fetch_file's bounded retries.
            response.close()


def selected_path(path):
    return (path == "README.md" or re.fullmatch(PARQUET_PATTERN, path)
            or re.fullmatch(META_PATTERN, path))


def summarize(entries):
    """Describe source-tree coverage without interpreting pose payloads."""
    sessions = {}
    totals = {key: 0 for key in EXPECTED}
    totals["files"] = len(entries)
    for entry in entries:
        path, size = entry["path"], entry["size"]
        totals["total_bytes"] += size
        if path == "README.md":
            continue
        session = path.split("/", 1)[0]
        item = sessions.setdefault(session, {
            "parquet_files": 0, "parquet_bytes": 0,
            "metadata_files": 0, "metadata_bytes": 0,
            "parquet_episode_ids": set(), "episode_metadata_ids": set(),
        })
        if re.fullmatch(PARQUET_PATTERN, path):
            kind = "parquet"
            episode_id = int(Path(path).stem.removeprefix("episode_"))
            if episode_id in item["parquet_episode_ids"]:
                raise ValueError(f"Duplicate parquet episode in {session}: {episode_id}")
            item["parquet_episode_ids"].add(episode_id)
        else:
            kind = "metadata"
            match = re.fullmatch(EPISODE_META_PATTERN, path)
            if match:
                episode_id = int(match["episode"])
                if episode_id in item["episode_metadata_ids"]:
                    raise ValueError(f"Duplicate episode metadata in {session}: {episode_id}")
                item["episode_metadata_ids"].add(episode_id)
        item[f"{kind}_files"] += 1
        item[f"{kind}_bytes"] += size
        totals[f"{kind}_files"] += 1
        totals[f"{kind}_bytes"] += size
    if totals != EXPECTED:
        raise ValueError(f"Pinned source selection differs from audited counts: {totals}")
    gaps = {}
    for session, item in sorted(sessions.items()):
        parquet_ids = item.pop("parquet_episode_ids")
        metadata_ids = item.pop("episode_metadata_ids")
        item["episode_metadata_files"] = len(metadata_ids)
        item["total_bytes"] = item["parquet_bytes"] + item["metadata_bytes"]
        missing_parquet = sorted(metadata_ids - parquet_ids)
        missing_metadata = sorted(parquet_ids - metadata_ids)
        if missing_parquet or missing_metadata:
            gaps[session] = {
                "episode_metadata_without_parquet": missing_parquet,
                "parquet_without_episode_metadata": missing_metadata,
            }
    return totals, dict(sorted(sessions.items())), gaps


def validate_manifest(manifest):
    identity = (manifest.get("schema"), manifest.get("dataset"),
                manifest.get("repo"), manifest.get("revision"), manifest.get("selection"))
    if identity != (SCHEMA, "humantouch", REPO, REVISION, SELECTION):
        raise ValueError("Existing output has a different source/revision/selection; refusing overwrite")
    if manifest.get("training_ready") is not False:
        raise ValueError("Raw source manifest must not claim training readiness")
    entries = manifest.get("files", [])
    paths = [entry["path"] for entry in entries]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ValueError("Manifest paths must be sorted and unique")
    for entry in entries:
        if not selected_path(entry["path"]):
            raise ValueError("Manifest includes a path outside the exact numerical-data selection")
        if not isinstance(entry["size"], int) or entry["size"] < 0:
            raise ValueError("Invalid source file size")
        kind = entry["hash_kind"]
        length = {"sha256": 64, "git_blob_sha1": 40}.get(kind)
        if length is None or not re.fullmatch(rf"[0-9a-f]{{{length}}}", entry["hash"]):
            raise ValueError("Invalid source checksum")
        url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{quote(entry['path'])}"
        if entry["url"] != url:
            raise ValueError("Manifest URL differs from the pinned source")
    totals, sessions, gaps = summarize(entries)
    if (manifest.get("summary") != totals or manifest.get("total_bytes") != totals["total_bytes"]
            or manifest.get("sessions") != sessions or manifest.get("source_gaps") != gaps):
        raise ValueError("Existing manifest summary does not match its selected files")
    return manifest


def print_plan(manifest, reused):
    print(json.dumps({
        "dataset": "humantouch", "revision": REVISION,
        "reused_existing_manifest": reused, "summary": manifest["summary"],
        "sessions": manifest["sessions"], "source_gaps": manifest["source_gaps"],
        "training_ready": False,
    }), flush=True)


def plan(directory):
    destination = directory / "download_manifest.json"
    if destination.exists():
        manifest = validate_manifest(json.loads(destination.read_text()))
        print_plan(manifest, reused=True)
        return
    api_url = f"https://huggingface.co/api/datasets/{REPO}/revision/{REVISION}"
    # The full siblings listing contains tens of thousands of entries. No source
    # payloads are requested here; the longer read timeout covers its large JSON.
    response = requests.get(api_url, params={"blobs": "true"}, timeout=(30, 300))
    response.raise_for_status()
    source = response.json()
    if (source.get("sha") != REVISION or source.get("gated") or source.get("private")):
        raise ValueError("Pinned public source revision/access changed")
    entries = sorted(
        (hf_entry(REPO, REVISION, item) for item in source["siblings"]
         if selected_path(item["rfilename"])),
        key=lambda item: item["path"],
    )
    totals, sessions, gaps = summarize(entries)
    manifest = {
        "schema": SCHEMA, "dataset": "humantouch", "repo": REPO,
        "revision": REVISION, "created_utc": now(), "source_api_url": api_url,
        "selection": SELECTION,
        "scope": (
            "All released session parquets and JSON/JSONL metadata at the pinned revision; "
            "no standalone RGB/depth media. Native numerical touch fields remain embedded. "
            "No calibration/MANO mapping, split assignment or training admission performed."
        ),
        "training_ready": False,
        "source_gap_policy": (
            "Episode IDs are compared within each session. Metadata without source parquet "
            "are a release coverage gap, not a download failure or a usable trajectory. "
            "Selected files are the authoritative downloadable inventory; info.json claims "
            "have not been interpreted or accepted as actual trajectory counts."
        ),
        "summary": totals, "sessions": sessions, "source_gaps": gaps,
        "total_bytes": totals["total_bytes"], "files": entries,
    }
    validate_manifest(manifest)
    save_json(destination, manifest)
    print_plan(manifest, reused=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "download"))
    parser.add_argument("--output", type=Path, required=True,
                        help="Dataset directory under /sciclone/data10/jchen57")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("workers must be between 1 and 8")
    directory = args.output.resolve()
    if not REMOTE_ROOT.is_dir() or (directory != REMOTE_ROOT and REMOTE_ROOT not in directory.parents):
        parser.error("Download project data only in the authorized remote persistent storage")
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".download.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another process holds this dataset's download lock")
        if args.action == "plan":
            plan(directory)
        else:
            validate_manifest(json.loads((directory / "download_manifest.json").read_text()))
            # The shared downloader verifies complete HF SHA256/git-blob hashes,
            # resumes .partial files with Range, retries, and writes status every 20s.
            limiter = RateLimitedGet(directory)
            download(directory, args.workers, fetcher=partial(fetch_file, request_get=limiter))


if __name__ == "__main__":
    main()
