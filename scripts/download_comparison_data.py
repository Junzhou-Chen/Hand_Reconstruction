#!/usr/bin/env python3
"""Remote acquisition of official DexYCB and authenticated ARCTIC archives."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tarfile
import threading
import time
from urllib.parse import parse_qs, urlsplit

import requests

from download_twm_data import digest, now, save_json

ARCTIC_REV = "49f3eb4b7f01663172576357bb3d95695d7a7689"
ARCTIC_RAW = f"https://raw.githubusercontent.com/zc-alexfan/arctic/{ARCTIC_REV}/"
RUNTIME = Path("/sciclone/data10/jchen57/Hand_Reconstruction/download_runtime")
DEX_FILES = [
    ("20200709-subject-01", "1Ehh92wDE3CWAiKG7E9E73HjN2Xk2XfEk", 12412314463),
    ("20200813-subject-02", "1Uo7MLqTbXEa-8s7YQZ3duugJ1nXFEo62", 12004145048),
    ("20200820-subject-03", "1FkUxas8sv8UcVGgAzmSZlJw1eI5W5CXq", 12197037343),
    ("20200903-subject-04", "14up6qsTpvgEyqOQ5hir-QbjMB_dHfdpA", 12792618020),
    ("20200908-subject-05", "1NBA_FPyGWOQF5-X9ueAat5g8lDMz-EmS", 12815420651),
    ("20200918-subject-06", "1UWIN2-wOBZX2T0dkAi4ctAAW8KffkXMQ", 12610980973),
    ("20200928-subject-07", "1oWEYD_o3PVh39pLzMlJcArkDtMj4nzI0", 12818333033),
    ("20201002-subject-08", "1GTNZwhWbs7Mfez0krTgXwLPndvrw1Ztv", 12418522598),
    ("20201015-subject-09", "1j0BLkaCjIuwjakmywKdOO9vynHTWR0UH", 12493882548),
    ("20201022-subject-10", "1FvFlRfX-p5a5sAWoKEGc17zKJWwKaSB-", 12341740204),
    ("bop", "1CPqLjsaYNjE3xSJbuWmqaMsGvyGIxiKL", 1250767497),
    ("calibration", "1UAwVKT4Rgb1fLcFoa1o71_-0NtSvvLAQ", 14464),
    ("models", "1cAzlQBpcTatI5ykYQ8ziQiHLUG_a_UpM", 1455399753),
]


class AuthenticationRequired(RuntimeError):
    pass


def credentials(path):
    if path.is_file():
        try:
            value = json.loads(path.read_text())
            if not isinstance(value, dict):
                raise ValueError("Expected credential object")
            username, password = value.get("username"), value.get("password")
        except (OSError, ValueError):
            raise AuthenticationRequired("ARCTIC private credential file could not be read") from None
    else:
        username, password = os.environ.get("ARCTIC_USERNAME"), os.environ.get("ARCTIC_PASSWORD")
    if not isinstance(username, str) or not isinstance(password, str) or not username or not password:
        raise AuthenticationRequired("ARCTIC official username/password are required")
    if username == "YOUR_ARCTIC_EMAIL" or password == "YOUR_ARCTIC_PASSWORD":
        raise AuthenticationRequired("ARCTIC example placeholders are not credentials")
    return {"username": username, "password": password}


def plan(dataset, directory, images):
    if dataset == "dexycb":
        entries = [{"path": f"archives/{name}.tar.gz", "drive_id": file_id, "size": size,
                    "source_url": f"https://drive.google.com/file/d/{file_id}/view"}
                   for name, file_id, size in DEX_FILES]
        entries.sort(key=lambda x: x["size"])
        scope = "All 10 official subject archives, calibration, object models, BOP; preserve dataset splits."
        reference = "https://dex-ycb.github.io/"
        verification = "Drive metadata byte size + full gzip CRC + TAR header; locally computed SHA256 (no published author checksum)."
    else:
        checksum_response = requests.get(ARCTIC_RAW + "bash/assets/checksum.json", timeout=(30, 90))
        checksum_response.raise_for_status()
        checksums = checksum_response.json()
        entries = []
        for group in ("misc", "splits", "cropped_images" if images == "cropped" else "images"):
            response = requests.get(ARCTIC_RAW + f"bash/assets/urls/{group}.txt", timeout=(30, 90))
            response.raise_for_status()
            for url in response.text.splitlines():
                url = url.strip()
                if not url:
                    continue
                release_path = parse_qs(urlsplit(url).query)["sfile"][0]
                suffix = release_path.split("/data/", 1)[1]
                if group == "misc" and suffix not in ("raw_seqs.zip", "meta.zip", "splits_json.zip"):
                    continue
                if ".." in PurePosixPath(suffix).parts or suffix.startswith("/"):
                    raise ValueError("Unsafe ARCTIC source path")
                entries.append({"path": "archives/data/" + suffix, "url": url, "size": None,
                                "sha256": checksums["/data/" + suffix]})
        if len(entries) != 343:
            raise RuntimeError("ARCTIC pinned manifest has an unexpected archive count")
        reference = ARCTIC_RAW + "docs/data/README.md"
        scope = f"339 official {images} image archives + raw_seqs/meta/splits_json/splits; no body-model assets or baseline weights."
        verification = "Official pinned SHA256; reject login HTML. Exact archive bytes discovered after authentication."
    manifest = {"dataset": dataset, "created_utc": now(), "runner": "scripts/download_comparison_data.py",
                "scope": scope, "source": reference, "verification": verification,
                "source_revision": ARCTIC_REV if dataset == "arctic" else "dex-ycb-20210415 release / Drive IDs",
                "files": entries, "total_bytes": sum(x["size"] for x in entries) if dataset == "dexycb" else None}
    save_json(directory / "download_manifest.json", manifest)
    print(json.dumps({"dataset": dataset, "files": len(entries), "total_bytes": manifest["total_bytes"],
                      "scope": scope, "verification": verification}), flush=True)


def validate_gzip_tar(path, expected_size):
    if path.stat().st_size != expected_size:
        raise ValueError("File size differs from official Drive metadata")
    with gzip.open(path, "rb") as stream:
        for _ in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            pass  # Read through EOF to validate every gzip member's CRC and trailer.
    with tarfile.open(path, "r:gz") as archive:
        if archive.next() is None:
            raise ValueError("Expected a nonempty TAR archive")


def fetch_dex(entry, target, update):
    sys.path.insert(0, str(RUNTIME))
    import gdown
    if gdown.__version__ != "6.4.2":
        raise RuntimeError("Pinned isolated gdown 6.4.2 runtime is required")
    target.parent.mkdir(parents=True, exist_ok=True)
    receipt = target.with_name(target.name + ".receipt.json")
    if target.exists():
        update(bytes=target.stat().st_size, status="checking")
        if receipt.exists():
            known = json.loads(receipt.read_text())
            if target.stat().st_size == entry["size"] and digest(target) == known["sha256"]:
                return known
        validate_gzip_tar(target, entry["size"])
        result_hash = digest(target)
    else:
        hasher = hashlib.sha256()
        # gdown handles the normal public large-file confirmation and validates Range.
        # Its own retries preserve partials; quota/permission failures stop this file.
        result = gdown.download(id=entry["drive_id"], output=str(target), quiet=True,
                               use_cookies=False, resume=True, timeout=(30, 120), retries=3,
                               progress=lambda current, total: update(bytes=current, status="downloading"),
                               hasher=hasher)
        if not result:
            raise RuntimeError("Google Drive download did not produce an archive")
        update(status="checking")
        validate_gzip_tar(target, entry["size"])
        result_hash = hasher.hexdigest()
    result = {"sha256": result_hash, "size": entry["size"], "verified_utc": now(),
              "verification": "metadata size + full gzip CRC + TAR header; SHA256 computed locally"}
    save_json(receipt, result)
    return result


def fetch_arctic(entry, target, auth, update, attempts=4):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        update(bytes=target.stat().st_size, size=target.stat().st_size, status="checking")
        if digest(target) != entry["sha256"]:
            raise RuntimeError("Existing ARCTIC archive fails official SHA256")
        return {"sha256": entry["sha256"], "size": target.stat().st_size, "verified_utc": now()}
    partial = target.with_name(target.name + ".partial")
    if partial.exists() and digest(partial) == entry["sha256"]:
        partial.replace(target)
        update(bytes=target.stat().st_size, size=target.stat().st_size)
        return {"sha256": entry["sha256"], "size": target.stat().st_size, "verified_utc": now()}
    for attempt in range(attempts):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            with requests.post(entry["url"], data=auth, headers=headers, stream=True,
                               allow_redirects=True, timeout=(30, 120)) as response:
                if response.status_code in (401, 403):
                    raise AuthenticationRequired("ARCTIC rejected the provided account")
                if response.status_code == 416:
                    partial.replace(partial.with_name(partial.name + f".corrupt-{time.time_ns()}"))
                    raise ValueError("Invalid complete partial; restarting archive")
                response.raise_for_status()
                if "text/html" in response.headers.get("Content-Type", "").lower():
                    raise AuthenticationRequired("ARCTIC returned a login/permission page")
                if response.status_code == 206:
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                    if not match or int(match[1]) != offset:
                        raise ValueError("Invalid authenticated Content-Range")
                    size = int(match[3])
                elif response.status_code == 200:
                    offset = 0  # Never append if this endpoint ignores Range.
                    size = int(response.headers["Content-Length"]) if "Content-Length" in response.headers else None
                else:
                    raise ValueError("Unexpected ARCTIC HTTP status")
                update(bytes=offset, size=size, status="downloading")
                with partial.open("ab" if offset else "wb") as stream:
                    first = True
                    for chunk in response.iter_content(4 * 1024 * 1024):
                        if not chunk:
                            continue
                        if first and not offset and not chunk.startswith(b"PK"):
                            raise ValueError("Expected ZIP bytes, received a non-archive response")
                        first = False
                        stream.write(chunk)
                        offset += len(chunk)
                        update(bytes=offset)
            if size is not None and partial.stat().st_size != size:
                raise ValueError("ARCTIC response size mismatch")
            update(status="checking")
            if digest(partial) != entry["sha256"]:
                partial.replace(partial.with_name(partial.name + f".corrupt-{time.time_ns()}"))
                raise ValueError("ARCTIC official SHA256 mismatch")
            partial.replace(target)
            update(size=target.stat().st_size)
            return {"sha256": entry["sha256"], "size": target.stat().st_size, "verified_utc": now()}
        except AuthenticationRequired:
            raise
        except (requests.RequestException, OSError, ValueError):
            if attempt + 1 == attempts:
                raise RuntimeError("ARCTIC network, range, size or checksum validation failed") from None
            time.sleep(min(2 ** attempt * 5, 40))


def download(manifest, directory, worker_count, credential_path):
    state = {x["path"]: {"status": "pending", "bytes": 0, "size": x["size"]} for x in manifest["files"]}
    lock, stop = threading.Lock(), threading.Event()
    receipts = {}
    started = now()

    def report(phase):
        with lock:
            values = {p: dict(v) for p, v in state.items()}
            receipt_copy = dict(receipts)
        summary = {"dataset": manifest["dataset"], "phase": phase, "started_utc": started,
                   "updated_utc": now(), "total_bytes": manifest["total_bytes"],
                   "known_total_bytes": sum(v["size"] or 0 for v in values.values()),
                   "present_bytes": sum(v["bytes"] for v in values.values()),
                   "counts": {s: sum(v["status"] == s for v in values.values())
                              for s in ("pending", "downloading", "checking", "verified", "failed")},
                   "errors": {p: v["error"] for p, v in values.items() if "error" in v}}
        save_json(directory / "download_status.json", summary)
        save_json(directory / "download_receipts.json", receipt_copy)
        print(json.dumps(summary), flush=True)

    auth = None
    auth_lost = threading.Event()
    if manifest["dataset"] == "arctic":
        try:
            auth = credentials(credential_path)
        except AuthenticationRequired:
            report("waiting_credentials")
            print("ARCTIC needs its official account in the private credential file.", flush=True)
            raise SystemExit(2)
        # Verify the account on one small official archive before issuing image requests.
        smoke = next(x for x in manifest["files"] if x["path"].endswith("/splits_json.zip"))
        try:
            def smoke_update(**changes):
                with lock:
                    state[smoke["path"]].update(changes)
            result = fetch_arctic(smoke, directory / smoke["path"], auth, smoke_update, attempts=1)
            receipts[smoke["path"]] = result
            state[smoke["path"]].update(status="verified", bytes=result["size"], size=result["size"])
        except AuthenticationRequired:
            report("authentication_failed")
            print("ARCTIC account check failed; no image download was started.", flush=True)
            raise SystemExit(2)

    def worker(entry):
        def update(**changes):
            with lock:
                state[entry["path"]].update(changes)
        try:
            if auth_lost.is_set():
                raise AuthenticationRequired("ARCTIC authentication failed; further requests stopped")
            update(status="downloading")
            if manifest["dataset"] == "dexycb":
                result = fetch_dex(entry, directory / entry["path"], update)
            else:
                result = fetch_arctic(entry, directory / entry["path"], auth, update)
            with lock:
                receipts[entry["path"]] = result
                state[entry["path"]].update(status="verified", bytes=result["size"], size=result["size"])
        except Exception as error:
            if isinstance(error, AuthenticationRequired):
                auth_lost.set()
            # Never log request form contents, cookies or credential values.
            reason = type(error).__name__
            if isinstance(error, (RuntimeError, AuthenticationRequired)):
                reason += ": " + str(error)
            update(status="failed", error=reason)

    def reporter():
        while not stop.wait(20):
            report("running")

    report("running")
    thread = threading.Thread(target=reporter, daemon=True)
    thread.start()
    try:
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            for future in as_completed([pool.submit(worker, x) for x in manifest["files"]]):
                future.result()
    finally:
        stop.set()
        thread.join()
    failed = any(v["status"] == "failed" for v in state.values())
    if not failed and manifest["total_bytes"] is None:
        manifest["total_bytes"] = sum(v["size"] for v in state.values())
        save_json(directory / "download_manifest.json", manifest)
    report("failed" if failed else "complete")
    if failed:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "download"))
    parser.add_argument("dataset", choices=("dexycb", "arctic"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--images", choices=("cropped", "full"), default="cropped")
    parser.add_argument("--credentials", type=Path,
                        default=Path("/sciclone/home/jchen57/project/Hand_Reconstruction/config/arctic_credentials.local.json"))
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("workers must be 1–8")
    directory = args.root / args.dataset
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".download.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.action == "plan":
            plan(args.dataset, directory, args.images)
        else:
            download(json.loads((directory / "download_manifest.json").read_text()), directory,
                     args.workers, args.credentials)


if __name__ == "__main__":
    main()
