#!/usr/bin/env python3
"""Download pinned, public TWM sources on remote persistent storage.

No MANO assets, image-only OakInk2 archives, or HOT3D/UmeTrack test data.
Requires requests; planning and downloads must run remotely for this project.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import threading
import time
from urllib.parse import quote

import requests

SOURCES = {
    "oakink2": ("kelvin34501/OakInk-v2", "21705616140d726607027e70d58b7837f442ffd8"),
    "hot3d": ("bop-benchmark/hot3d", "30fe9674782f32e1e5edba98476b6ff4300132c5"),
    "umetrack": ("facebook/hand_tracking_challenge_umetrack", "19f6b2cad9eb17be4ebedfe4503203980a25e016"),
}
HAWOR_REV = "66c7d4108d58a716deccd192cb7645170cdc7bd7"
TAMF_REV = "4cb9c461b92c032715374bd252c7179c70a14bde"


def now():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def digest(path, git_blob=False):
    h = hashlib.sha1() if git_blob else hashlib.sha256()
    if git_blob:
        h.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verified(path, entry):
    return (path.is_file() and path.stat().st_size == entry["size"]
            and digest(path, entry["hash_kind"] == "git_blob_sha1") == entry["hash"])


def fetch_file(entry, destination, progress=lambda size: None, attempts=6):
    """Resume from a partial file; only verified bytes get the final name."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if verified(destination, entry):
        progress(entry["size"])
        return
    if destination.exists():
        raise RuntimeError(f"Existing final file failed verification: {entry['path']}")
    partial = destination.with_name(destination.name + ".partial")
    for attempt in range(attempts):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > entry["size"]:
                partial.replace(partial.with_name(partial.name + f".corrupt-{time.time_ns()}"))
                offset = 0
            if entry["size"] == 0:
                partial.touch()
            if offset < entry["size"]:
                headers = {"Range": f"bytes={offset}-"} if offset else {}
                # A fresh resolve request renews CDN signatures; never persist signed URLs.
                with requests.get(entry["url"], headers=headers, stream=True,
                                  params={"download": "true", "request_time": time.time_ns()},
                                  timeout=(30, 90)) as response:
                    response.raise_for_status()
                    if response.status_code == 206:
                        content_range = response.headers.get("Content-Range", "")
                        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
                        if not match or int(match[1]) != offset or int(match[3]) != entry["size"]:
                            raise ValueError("Unexpected HTTP Content-Range")
                    elif response.status_code == 200:
                        offset = 0  # Server ignored Range: overwrite rather than append.
                    else:
                        raise ValueError(f"Unexpected HTTP status {response.status_code}")
                    with partial.open("ab" if offset else "wb") as f:
                        progress(offset)
                        for chunk in response.iter_content(4 * 1024 * 1024):
                            f.write(chunk)
                            offset += len(chunk)
                            if offset > entry["size"]:
                                raise ValueError("HTTP payload exceeds manifest size")
                            progress(offset)
            if not verified(partial, entry):
                if partial.stat().st_size == entry["size"]:
                    partial.replace(partial.with_name(partial.name + f".corrupt-{time.time_ns()}"))
                raise ValueError("File size or checksum mismatch")
            partial.replace(destination)
            return
        except (requests.RequestException, OSError, ValueError) as error:
            if attempt + 1 == attempts:
                detail = type(error).__name__
                if isinstance(error, requests.HTTPError) and error.response is not None:
                    detail += f" HTTP {error.response.status_code}"
                elif isinstance(error, ValueError):
                    detail += f": {error}"
                raise RuntimeError(f"Download failed: {entry['path']} ({detail})") from None
            time.sleep(min(2 ** attempt * 2, 60))


def hf_entry(repo, revision, item):
    path = item["rfilename"]
    if PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts:
        raise ValueError("Unsafe source path")
    lfs = item.get("lfs")
    return {"path": path, "size": item["size"],
            "hash_kind": "sha256" if lfs else "git_blob_sha1",
            "hash": lfs["sha256"] if lfs else item["blobId"],
            "url": f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{quote(path)}"}


def github_text(url, path, directory):
    response = requests.get(url, timeout=(30, 90))
    response.raise_for_status()
    payload = response.content
    entry = {"path": path, "size": len(payload), "hash_kind": "sha256",
             "hash": hashlib.sha256(payload).hexdigest(), "url": url}
    target = directory / path
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not verified(target, entry):
        raise RuntimeError(f"Conflicting metadata: {path}")
    target.write_bytes(payload)
    return entry, response.text


def plan(dataset, directory):
    repo, revision = SOURCES[dataset]
    response = requests.get(f"https://huggingface.co/api/datasets/{repo}/revision/{revision}",
                            params={"blobs": "true"}, timeout=(30, 90))
    response.raise_for_status()
    source = response.json()
    if source.get("sha") != revision or source.get("gated"):
        raise RuntimeError("Source revision/access does not match the pinned public release")
    items = {x["rfilename"]: hf_entry(repo, revision, x) for x in source["siblings"]}
    auxiliary, excluded = [], []
    split_summary = {}
    if dataset == "oakink2":
        selected = [x for p, x in items.items() if p.startswith("anno_preview/") and p.endswith(".pkl")]
        selected += [items[p] for p in ("README.md", "program.tar", "program_extension.tar")]
        split_sets = {}
        for name in ("all.txt", "train.txt", "val.txt", "test.txt"):
            url = f"https://raw.githubusercontent.com/oakink/OakInk2-TaMF/{TAMF_REV}/asset/split/{name}"
            entry, text = github_text(url, "splits/" + name, directory)
            auxiliary.append(entry)
            names = {"anno_preview/" + line.strip().replace("/", "++") + ".pkl"
                     for line in text.splitlines() if line.strip()}
            if not names <= items.keys():
                raise RuntimeError("OakInk2 split contains missing annotations")
            split_summary[name] = {"records": len(names), "bytes": sum(items[p]["size"] for p in names)}
            if name != "all.txt":
                split_sets[name] = names
        if set(split_sets) != {"train.txt", "val.txt", "test.txt"}:
            raise RuntimeError("Expected OakInk2 official splits are missing")
        if any(split_sets[a] & split_sets[b] for a, b in (("train.txt", "val.txt"), ("train.txt", "test.txt"), ("val.txt", "test.txt"))):
            raise RuntimeError("OakInk2 record splits overlap")
        annotation_paths = {p for p in items if p.startswith("anno_preview/") and p.endswith(".pkl")}
        if set.union(*split_sets.values()) != annotation_paths:
            raise RuntimeError("OakInk2 splits do not cover annotation release")
        scope = "All pose annotations; TaMF train only for pretraining; val/test retained separately. No images/objects."
    elif dataset == "hot3d":
        for p in ("clip_definitions.json", "clip_splits.json"):
            fetch_file(items[p], directory / p)
        definitions = json.loads((directory / "clip_definitions.json").read_text())
        splits = json.loads((directory / "clip_splits.json").read_text())
        entry, readme = github_text(f"https://raw.githubusercontent.com/ThunderVVV/HaWoR/{HAWOR_REV}/README.md", "provenance/hawor_README.md", directory)
        auxiliary.append(entry)
        section = readme.split("--sequence_name", 1)[1].split("```", 1)[0]
        heldout = set(re.findall(r"P\d{4}_[0-9a-f]{8}", section))
        if len(heldout) != 27:
            raise RuntimeError("HaWoR evaluation sequence list changed or could not be parsed")
        available = {d["sequence_id"] for d in definitions.values()}
        if not heldout & available:
            raise RuntimeError("No HaWoR evaluation recordings found in clip definitions")
        # BOP clips are a subset of the original release; two listed recordings
        # are absent entirely. Exclude every match and record the absence explicitly.
        for prefix, device in (("train_aria/", "Aria"), ("train_quest3/", "Quest3")):
            directory_ids = {int(PurePosixPath(p).stem.split("-")[-1])
                             for p in items if p.startswith(prefix) and p.endswith(".tar")}
            if directory_ids != set(splits["train"][device]):
                raise RuntimeError(f"HOT3D {device} train directory disagrees with official split")
        selected = []
        for p, item in items.items():
            if p.startswith(("train_aria/", "train_quest3/")) and p.endswith(".tar"):
                clip_id = str(int(PurePosixPath(p).stem.split("-")[-1]))
                definition = definitions[clip_id]
                item["sequence_id"] = definition["sequence_id"]
                item["device"] = definition["device"]
                expected_device = "Aria" if p.startswith("train_aria/") else "Quest3"
                if definition["device"] != expected_device:
                    raise RuntimeError("HOT3D clip device disagrees with its train directory")
                if item["sequence_id"] in heldout:
                    excluded.append(item)
                else:
                    selected.append(item)
        selected += [items[p] for p in ("README.md", "clip_definitions.json", "clip_splits.json", "hot3d_dataset_license_agreement.pdf", "hot3d_base.zip")]
        split_summary = {"hawor_excluded_recordings": sorted(heldout),
                         "hawor_recordings_absent_from_bop_release": sorted(heldout - available),
                         "excluded_train_clips": len(excluded), "excluded_bytes": sum(x["size"] for x in excluded),
                         "official_train_split_and_devices_verified": True}
        scope = "Official train Aria/Quest3 mixed TARs, excluding all 27 HaWoR evaluation source recordings before windowing. Other benchmark overlaps still need audit."
    else:
        selected = [x for p, x in items.items() if p.startswith("train/") and p.endswith(".tar") and "_synthetic_" not in p]
        excluded = [x for p, x in items.items() if p.startswith("train/") and p.endswith(".tar") and "_synthetic_" in p]
        selected.append(items["README.md"])
        if len(selected) != 1021 or len(excluded) != 1021:
            raise RuntimeError("UmeTrack real/synthetic scope differs from audited release")
        split_summary = {"real_train_tars": 1020, "synthetic_train_tars_skipped": len(excluded)}
        scope = "HF official real train only; synthetic renders skipped to avoid redundant motion. Test data excluded; synthetic-only coverage not yet audited."
    entries = sorted(selected + auxiliary, key=lambda x: (not x["path"].endswith((".txt", ".md", ".json")), x["path"]))
    manifest = {"dataset": dataset, "created_utc": now(), "repo": repo, "revision": revision,
                "scope": scope, "split_summary": split_summary, "files": entries,
                "total_bytes": sum(x["size"] for x in entries),
                "excluded_files": excluded, "source_api_url": response.url}
    save_json(directory / "download_manifest.json", manifest)
    print(json.dumps({k: manifest[k] for k in ("dataset", "revision", "scope", "total_bytes", "split_summary")}), flush=True)
    print("selected_files", len(entries), flush=True)


def download(directory, workers):
    manifest = json.loads((directory / "download_manifest.json").read_text())
    mutex, stop = threading.Lock(), threading.Event()
    state = {x["path"]: {"status": "pending", "bytes": 0} for x in manifest["files"]}
    started = now()

    def update(path, **changes):
        with mutex:
            state[path].update(changes)

    def report(phase):
        with mutex:
            copy = {p: dict(v) for p, v in state.items()}
        counts = {name: sum(v["status"] == name for v in copy.values()) for name in ("pending", "downloading", "verified", "failed")}
        summary = {"dataset": manifest["dataset"], "started_utc": started, "updated_utc": now(),
                   "phase": phase, "total_bytes": manifest["total_bytes"],
                   "present_bytes": sum(v["bytes"] for v in copy.values()), "counts": counts,
                   "errors": {p: v["error"] for p, v in copy.items() if "error" in v}}
        save_json(directory / "download_status.json", summary)
        print(json.dumps(summary), flush=True)

    def reporter():
        while not stop.wait(20):
            report("running")

    def worker(entry):
        p = entry["path"]
        update(p, status="downloading")
        try:
            fetch_file(entry, directory / p, lambda size: update(p, bytes=size))
            update(p, status="verified", bytes=entry["size"])
        except Exception as error:
            # Avoid logging transient signed URLs or credentials from HTTP exceptions.
            detail = str(error) if isinstance(error, RuntimeError) else type(error).__name__
            update(p, status="failed", error=detail)

    report("running")
    thread = threading.Thread(target=reporter, daemon=True)
    thread.start()
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(worker, x) for x in manifest["files"]]
            for future in as_completed(futures):
                future.result()
    finally:
        stop.set()
        thread.join()
    failed = any(v["status"] == "failed" for v in state.values())
    report("failed" if failed else "complete")
    if failed:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "download"))
    parser.add_argument("dataset", choices=SOURCES)
    parser.add_argument("--root", type=Path, required=True, help="Remote persistent dataset root")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("workers must be between 1 and 8")
    directory = args.root / args.dataset
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".download.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.action == "plan":
            plan(args.dataset, directory)
        else:
            download(directory, args.workers)


if __name__ == "__main__":
    main()
