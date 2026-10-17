#!/usr/bin/env python3
"""Remote-only, resumable numerical releases for InterHand30fps/TACO/GigaHands.

Never downloads RGB. Official source hashes are distinguished from local
integrity receipts; Google Drive JSON is size- and parse-checked. Archive
extraction rejects links and traversal and preserves recording directories.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import threading
import time
import zipfile

import requests

HF_REV = "fa8ed8b4cf5d46949b90ef852128811a2854066a"
EASY_REV = "b61bf7148d4411bb81ddcf6350a3ef1427213649"
DRIVE = [
    ("InterHand2.6M_train_MANO_NeuralAnnot.json", "1pgPRPcL8usdb94aGOb-GF7CoQhy4kjOz", 142194147),
    ("InterHand2.6M_train_joint_3d.json", "1eqeYdJcm62gbuXxC_KQZba4F7cfPJ_Wi", 184442740),
    ("InterHand2.6M_val_MANO_NeuralAnnot.json", "1tL2UyAsPsZRPIuJfzKWLPwSOXdm5Ix_2", 29427783),
    ("InterHand2.6M_val_joint_3d.json", "1TEpAIOxitetYPpvq3L7qks6qypw4NHNU", None),
]


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, obj):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def get(url, **kwargs):
    for attempt in range(8):
        try:
            r = requests.get(url, timeout=(30, 120), **kwargs)
            r.raise_for_status()
            return r
        except requests.RequestException:
            if attempt == 7:
                raise
            time.sleep(min(60, 2 ** attempt))


def small(url, path):
    receipt = path.with_name(path.name + ".receipt.json")
    previous = json.loads(receipt.read_text()) if receipt.exists() else {}
    if path.exists() and previous.get("url") == url:
        if path.stat().st_size != previous.get("bytes") or sha(path) != previous.get("local_sha256"):
            raise ValueError(f"Metadata changed after download receipt: {path}")
        return previous
    else:
        r = get(url)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(r.content)
        tmp.replace(path)
    recorded = {"url": url, "path": str(path), "bytes": path.stat().st_size,
                "local_sha256": sha(path), "verified_utc": now()}
    save(receipt, recorded)
    return recorded


def stream(url, path, size, official_sha=None):
    receipt = path.with_name(path.name + ".receipt.json")
    if path.exists():
        if path.stat().st_size != size:
            raise ValueError(f"Final size mismatch: {path}")
        digest = sha(path)
        previous = json.loads(receipt.read_text()) if receipt.exists() else {}
        if previous.get("url", url) != url:
            raise ValueError(f"Source URL changed for existing archive: {path}")
        expected = official_sha or previous.get("local_sha256")
        if expected and digest != expected:
            raise ValueError(f"Final hash mismatch: {path}")
        save(receipt, {"url": url, "bytes": size, "local_sha256": digest,
                       "official_sha256": official_sha, "verified_utc": now()})
        return
    part = path.with_name(path.name + ".partial")
    for attempt in range(10):
        try:
            offset = part.stat().st_size if part.exists() else 0
            if offset > size:
                part.rename(part.with_name(part.name + ".bad-" + str(time.time_ns())))
                offset = 0
            if offset < size:
                headers = {"Range": f"bytes={offset}-"} if offset else {}
                with get(url, headers=headers, stream=True, params={"download": "true", "request_time": time.time_ns()}) as r:
                    if r.status_code == 206:
                        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", r.headers.get("Content-Range", ""))
                        if not match or int(match[1]) != offset or int(match[3]) != size:
                            raise ValueError("Invalid Content-Range")
                    elif r.status_code == 200:
                        offset = 0
                    else:
                        raise ValueError("Unexpected download status")
                    with part.open("ab" if offset else "wb") as f:
                        for chunk in r.iter_content(8 << 20):
                            f.write(chunk)
                            offset += len(chunk)
                            if offset > size:
                                raise ValueError("Oversized response")
            if part.stat().st_size != size:
                raise ValueError("Incomplete response")
            digest = sha(part)
            if official_sha and digest != official_sha:
                part.rename(part.with_name(part.name + ".bad-" + str(time.time_ns())))
                raise ValueError("Official SHA mismatch")
            part.replace(path)
            save(receipt, {"url": url, "bytes": size, "local_sha256": digest,
                           "official_sha256": official_sha, "verified_utc": now()})
            return
        except Exception:
            if attempt == 9:
                raise
            time.sleep(min(60, 2 ** attempt))


def target(base, name):
    p = PurePosixPath(name)
    if p.is_absolute() or ".." in p.parts:
        raise ValueError(f"Unsafe archive path: {name}")
    return base.joinpath(*p.parts)


def extract(archive, output):
    marker = output / ".extracted.json"
    identity = {"archive": str(archive), "archive_bytes": archive.stat().st_size,
                "archive_sha256": sha(archive)}
    if marker.exists():
        recorded = json.loads(marker.read_text())
        if all(recorded.get(key) == value for key, value in identity.items()):
            return
        # A marker from an older downloader lacks the archive fingerprint.
        # Re-extract rather than treating a stale marker as validated output.
        marker.unlink()
    output.mkdir(parents=True, exist_ok=True)
    count = 0
    seen = set()
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            for item in z.infolist():
                dst = target(output, item.filename)
                if item.is_dir():
                    dst.mkdir(parents=True, exist_ok=True)
                    continue
                if item.filename in seen:
                    raise ValueError(f"Duplicate archive member: {item.filename}")
                seen.add(item.filename)
                if (item.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError("Archive link rejected")
                dst.parent.mkdir(parents=True, exist_ok=True)
                tmp = dst.with_name(dst.name + ".tmp")
                with z.open(item) as src, tmp.open("wb") as f:
                    shutil.copyfileobj(src, f, 4 << 20)
                tmp.replace(dst)
                count += 1
    else:
        with tarfile.open(archive, "r:gz") as t:
            for item in t:
                dst = target(output, item.name)
                if item.isdir():
                    dst.mkdir(parents=True, exist_ok=True)
                    continue
                if item.name in seen:
                    raise ValueError(f"Duplicate archive member: {item.name}")
                seen.add(item.name)
                if not item.isfile():
                    raise ValueError("Non-regular archive member rejected")
                dst.parent.mkdir(parents=True, exist_ok=True)
                tmp = dst.with_name(dst.name + ".tmp")
                with t.extractfile(item) as src, tmp.open("wb") as f:
                    shutil.copyfileobj(src, f, 4 << 20)
                tmp.replace(dst)
                count += 1
    save(marker, {**identity, "files": count, "completed_utc": now()})


def validate_interhand_json(path, name):
    """Reject access pages, wrong annotation types and malformed/truncated JSON.

    These are format checks, not a claim of official hash authentication.
    A source ID with unknown official byte count (currently val joint_3d)
    must still resolve to the intended annotation schema and official split.
    """
    split = "train" if "_train_" in name else "val"
    with path.open() as f:
        value = json.load(f)
    if not isinstance(value, dict) or not value:
        raise ValueError(f"Expected nonempty official annotation JSON object: {name}")
    if not all(str(key).isdigit() and isinstance(frames, dict) for key, frames in value.items()):
        raise ValueError(f"Expected capture/frame dictionaries: {name}")
    captures = {int(key) for key in value}
    if ((split == "train" and (min(captures) < 0 or max(captures) > 26))
            or (split == "val" and captures != {0})):
        raise ValueError(f"Wrong capture IDs for official {split} source: {name}")
    checked = False
    for frames in value.values():
        for key, record in frames.items():
            if not str(key).isdigit() or not isinstance(record, dict):
                raise ValueError(f"Expected numeric frame IDs and annotation mappings: {name}")
            if name.endswith("joint_3d.json"):
                coords, valid = record.get("world_coord"), record.get("joint_valid")
                if (not isinstance(coords, list) or len(coords) != 42
                        or not all(isinstance(point, list) and len(point) == 3 for point in coords)
                        or not isinstance(valid, list) or len(valid) != 42
                        or not isinstance(record.get("seq"), str)):
                    raise ValueError(f"Wrong joint_3d annotation schema: {name}")
                checked = True
            else:
                for side in ("left", "right"):
                    hand = record.get(side)
                    if hand is None:
                        continue
                    if (not isinstance(hand, dict) or not isinstance(hand.get("pose"), list)
                            or len(hand["pose"]) != 48 or not isinstance(hand.get("shape"), list)
                            or len(hand["shape"]) != 10 or not isinstance(hand.get("trans"), list)
                            or len(hand["trans"]) != 3):
                        raise ValueError(f"Wrong MANO annotation schema: {name}")
                    checked = True
            if checked:
                break
        if checked:
            break
    if not checked:
        raise ValueError(f"No nonempty expected annotation record: {name}")
    return {"schema_checked": True, "schema_scope": "capture IDs plus first available expected label",
            "captures": sorted(captures), "frame_records": sum(len(frames) for frames in value.values())}


def interhand(root):
    import gdown
    receipt_path = root / "download_receipts.json"
    previous = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    # The first deployed version used names without InterHand2.6M_. Root may
    # rename those completed files; preserve their original hash receipts.
    previous_files = {}
    for entry in previous.get("files", []):
        name = Path(entry["path"]).name
        if not name.startswith("InterHand2.6M_"):
            name = "InterHand2.6M_" + name
        previous_files[name] = entry
    files = []
    for name, drive_id, size in DRIVE:
        path = root / name
        if not path.exists():
            part = path.with_name(path.name + ".partial")
            for attempt in range(8):
                try:
                    if part.exists() and size is not None and part.stat().st_size > size:
                        part.rename(part.with_name(part.name + ".bad-" + str(time.time_ns())))
                    result = gdown.download(id=drive_id, output=str(part), quiet=True, resume=True)
                    if not result:
                        raise RuntimeError(f"Drive download failed: {name}")
                    if size is not None and part.stat().st_size != size:
                        part.rename(part.with_name(part.name + ".bad-" + str(time.time_ns())))
                        raise ValueError(f"Drive size mismatch: {name}")
                    try:
                        validate_interhand_json(part, name)
                    except (ValueError, TypeError):
                        # A complete HTML/error response cannot be resumed into
                        # a good JSON. Retain it for diagnosis, restart clean.
                        part.rename(part.with_name(part.name + ".bad-" + str(time.time_ns())))
                        raise
                    part.replace(path)
                    break
                except Exception:
                    if attempt == 7:
                        raise
                    time.sleep(min(60, 2 ** attempt))
        if size is not None and path.stat().st_size != size:
            raise ValueError(f"Drive final size mismatch: {name}")
        schema = validate_interhand_json(path, name)
        digest = sha(path)
        prior = previous_files.get(name, {})
        if prior.get("drive_id", drive_id) != drive_id:
            raise ValueError(f"InterHand source Drive ID changed: {name}")
        if prior.get("local_sha256") and prior["local_sha256"] != digest:
            raise ValueError(f"InterHand file changed after previous integrity receipt: {name}")
        files.append({"path": str(path), "bytes": path.stat().st_size,
                      "drive_id": drive_id, "local_sha256": digest, "official_sha256": None,
                      "official_size_bytes": size, **schema})
        save(receipt_path, {"source": "InterHand2.6M-30fps", "files": files})


def mark_taco_trajectory_ready(root):
    """Hand trajectories do not depend on optional camera ZIP extraction."""
    archive = root / 'Hand_Poses.zip'
    receipt_path = root / 'Hand_Poses.zip.receipt.json'
    marker = root / 'extracted' / '.extracted.json'
    receipt = json.loads(receipt_path.read_text())
    expected = 'db649d185ae818e63e9f14577067c1318d0751d35b213de8a06688637a1ece51'
    if (archive.stat().st_size != 15231790511 or
            receipt.get('official_sha256') != expected or receipt.get('local_sha256') != expected):
        raise ValueError('TACO requires the verified official hand archive')
    extracted = json.loads(marker.read_text())
    if extracted.get('archive') != str(archive) or extracted.get('files', 0) != 9268:
        raise ValueError('TACO hand extraction is not complete')
    if not (root / 'source_revision.json').is_file():
        raise ValueError('TACO official split/source revision not yet available')
    save(root / 'trajectory_ready.json', {
        'phase': 'complete', 'dataset': 'taco', 'completed_utc': now(),
        'archive_sha256': expected, 'archive_receipt_sha256': sha(receipt_path),
        'extraction_marker_sha256': sha(marker), 'extracted_pose_files': extracted['files'],
        'scope': 'complete hand parameters and official split metadata; optional camera extraction is independent'})


def taco(root):
    api = get(f"https://huggingface.co/api/datasets/leolyliu/TACO/tree/{HF_REV}?recursive=true&expand=false&limit=1000").json()
    save(root / "hf_tree.json", api)
    files = []
    for name in ("Hand_Poses.zip", "Allocentric_Camera_Parameters.zip"):
        entry = next(x for x in api if x.get("path") == name)
        digest = entry["lfs"]["oid"]
        url = f"https://huggingface.co/datasets/leolyliu/TACO/resolve/{HF_REV}/{name}"
        stream(url, root / name, entry["size"], digest)
        files.append(name)
    repo = "leolyliu/TACO-Instructions"
    revision_path = root / "source_revision.json"
    if revision_path.exists():
        pinned = json.loads(revision_path.read_text())
        if pinned.get("github") != repo or pinned.get("hf_revision") != HF_REV:
            raise ValueError("Existing TACO source revisions differ from this downloader")
        revision = pinned["revision"]
    else:
        revision = get(f"https://api.github.com/repos/{repo}/commits/master").json()["sha"]
    for path in ("README.md", "data_lists/v1_overall_data_train_test_split.txt", "data_lists/v1_overall_data_sequences.txt"):
        small(f"https://raw.githubusercontent.com/{repo}/{revision}/{path}", root / "source" / path)
    save(revision_path, {"github": repo, "revision": revision, "hf_revision": HF_REV})
    for name in files:
        extract(root / name, root / ("extracted" if name == "Hand_Poses.zip" else "camera"))
        if name == 'Hand_Poses.zip':
            mark_taco_trajectory_ready(root)


def gigahands(root):
    base = "https://g-ad09a0.56197.5898.data.globus.org/"
    small(base + "annotations_v2.jsonl", root / "annotations_v2.jsonl")
    for name in ("body_model.py", "lbs.py"):
        small(f"https://raw.githubusercontent.com/zju3dv/EasyMocap/{EASY_REV}/easymocap/smplmodel/{name}",
              root / "source_easymocap" / "easymocap" / "smplmodel" / name)
    revision_path = root / "source_revision.json"
    if revision_path.exists():
        pinned = json.loads(revision_path.read_text())
        if pinned.get("easymocap_revision") != EASY_REV:
            raise ValueError("Existing GigaHands EasyMocap revision differs from this downloader")
        revision = pinned["github_revision"]
    else:
        revision = get("https://api.github.com/repos/brown-ivl/GigaHands/commits/main").json()["sha"]
    small(f"https://raw.githubusercontent.com/brown-ivl/GigaHands/{revision}/README.md", root / "source_README.md")
    save(revision_path, {"github_revision": revision, "easymocap_revision": EASY_REV})
    stream(base + "hand_poses.tar.gz", root / "hand_poses.tar.gz", 11304232382)
    extract(root / "hand_poses.tar.gz", root / "extracted")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", choices=("all", "interhand", "taco", "gigahands"), default="all")
    args = parser.parse_args()
    if not str(args.root).startswith("/sciclone/data10/jchen57/"):
        parser.error("Run on project remote persistent storage")
    args.root.mkdir(parents=True, exist_ok=True)
    with (args.root / ("invocation-" + args.source + ".lock")).open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        selected = ("interhand", "taco", "gigahands") if args.source == "all" else (args.source,)
        states = {name: {"phase": "starting"} for name in selected}
        mutex = threading.Lock()
        status_path = args.root / (args.source + "_status.json")

        def report():
            with mutex:
                save(status_path, {"updated_utc": now(), "sources": states})

        def run(name):
            directory = args.root / name
            directory.mkdir(exist_ok=True)
            try:
                # --source all and --source interhand/etc must share a lock,
                # otherwise they could resume/write the same partial file.
                with (args.root / (name + ".lock")).open("w") as source_lock:
                    fcntl.flock(source_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    with mutex:
                        states[name] = {"phase": "running", "root": str(directory), "started_utc": now()}
                    report()
                    globals()[name](directory)
                with mutex:
                    states[name].update(phase="complete", completed_utc=now())
            except Exception as e:
                with mutex:
                    states[name].update(phase="failed", error=f"{type(e).__name__}: {e}")
            report()

        with ThreadPoolExecutor(max_workers=len(selected)) as pool:
            futures = [pool.submit(run, name) for name in selected]
            while not all(f.done() for f in futures):
                report()
                time.sleep(10)
        report()
        print(status_path.read_text(), flush=True)
        if any(x["phase"] != "complete" for x in states.values()):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
