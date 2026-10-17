#!/usr/bin/env python3
"""Verify preparation code using clean Git HEAD or a deployment receipt.

The no-Git Pod must have CODE_ROOT/.codex/deployment.json, exported by the
login host from a clean Git worktree. This is an integrity bridge from that
trusted producer, not an independent signature/authentication of the producer.
The verified set is every src/**/*.py, scripts/**/*.py and config/**/*.json.
No datasets, weights, experiment arrays or licensed assets are inspected.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess


SCOPE = ("src", "scripts", "config")
PRODUCER = "login-host-git-clean-worktree"
SHA256 = re.compile(r"[0-9a-fA-F]{64}")
COMMIT = re.compile(r"[0-9a-fA-F]{40}")


def file_sha256(path):
    before = path.stat()
    if path.is_symlink() or not stat.S_ISREG(before.st_mode):
        raise ValueError(f"Code/receipt must be a regular non-symlink file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError(f"Code/receipt changed during verification: {path}")
    return digest.hexdigest()


def is_source(relative):
    path = PurePosixPath(relative)
    if (not isinstance(relative, str) or path.is_absolute() or len(path.parts) < 2
            or ".." in path.parts or "\\" in relative or path.as_posix() != relative):
        return False
    return ((path.parts[0] in ("src", "scripts") and path.suffix == ".py")
            or (path.parts[0] == "config" and path.suffix == ".json"))


def actual_source_files(root):
    files = {}
    for scope in SCOPE:
        directory = root / scope
        if directory.is_symlink():
            raise ValueError(f"Source directory symlink rejected: {directory}")
        if not directory.exists():
            continue
        if not directory.is_dir():
            raise ValueError(f"Source scope must be a directory: {directory}")
        for current, directories, names in os.walk(directory, followlinks=False):
            parent = Path(current)
            for name in directories + names:
                if (parent / name).is_symlink():
                    raise ValueError(f"Symlink within source scope rejected: {parent / name}")
            for name in sorted(names):
                path = parent / name
                relative = path.relative_to(root).as_posix()
                if is_source(relative):
                    files[relative] = file_sha256(path)
    if not files:
        raise ValueError("No Python/JSON source files found in the declared code scopes")
    return dict(sorted(files.items()))


def compare_files(actual, expected):
    missing, unknown = sorted(set(expected)-set(actual)), sorted(set(actual)-set(expected))
    if missing or unknown:
        raise ValueError("Source path set differs: " + json.dumps({"missing": missing, "unknown": unknown}))
    changed = sorted(path for path in actual if actual[path] != expected[path])
    if changed:
        raise ValueError("Source SHA256 differs: " + json.dumps(changed))


def git_command(binary, root, *arguments, input_bytes=None):
    try:
        result = subprocess.run([binary, "-C", str(root), *arguments], input=input_bytes,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ValueError(f"Git snapshot verification failed: {error}") from error
    if result.returncode:
        raise ValueError(f"Git {' '.join(arguments[:2])} failed: {result.stderr.decode(errors='replace').strip()}")
    return result.stdout


def clean_git_head(binary, root):
    head = git_command(binary, root, "rev-parse", "--verify", "HEAD").decode().strip().lower()
    if not COMMIT.fullmatch(head):
        raise ValueError("Git HEAD is not a full SHA1 commit")
    if git_command(binary, root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise ValueError("Code worktree is not clean at HEAD, including untracked files")
    return head


def git_source_files(binary, root):
    raw = git_command(binary, root, "ls-tree", "-r", "--full-tree", "-z", "HEAD", "--", *SCOPE)
    object_by_path = {}
    for record in raw.split(b"\0"):
        if not record:
            continue
        description, raw_name = record.split(b"\t", 1)
        mode, kind, oid = description.split()
        name = raw_name.decode()
        if mode not in (b"100644", b"100755") or kind != b"blob":
            raise ValueError(f"Non-regular Git tree entry within source scope: {name}")
        if is_source(name):
            object_by_path[name] = oid.decode()
    object_ids = sorted(set(object_by_path.values()))
    if not object_ids:
        raise ValueError("Git HEAD contains no source files in declared scopes")
    payload = git_command(binary, root, "cat-file", "--batch",
                          input_bytes="".join(oid + "\n" for oid in object_ids).encode())
    position, blob_hashes = 0, {}
    for requested in object_ids:
        end = payload.find(b"\n", position)
        if end < 0:
            raise ValueError("Truncated Git cat-file header")
        oid, kind, length = payload[position:end].split()
        size = int(length)
        position = end + 1
        data = payload[position:position+size]
        if oid.decode() != requested or kind != b"blob" or len(data) != size or payload[position+size:position+size+1] != b"\n":
            raise ValueError("Unexpected/truncated Git source blob")
        blob_hashes[requested] = hashlib.sha256(data).hexdigest()
        position += size + 1
    if position != len(payload):
        raise ValueError("Unexpected trailing Git source output")
    return {path: blob_hashes[oid] for path, oid in object_by_path.items()}


def read_deployment(root):
    directory = root / ".codex"
    path = directory / "deployment.json"
    if directory.is_symlink() or path.is_symlink():
        raise ValueError("Deployment receipt symlink rejected")
    if not path.is_file():
        raise ValueError("Git binary missing and CODE_ROOT/.codex/deployment.json is required")
    receipt_hash = file_sha256(path)
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"Duplicate deployment receipt key: {key}")
            value[key] = item
        return value
    value = json.loads(path.read_text(), object_pairs_hook=unique)
    if not isinstance(value, dict) or value.get("producer") != PRODUCER or value.get("scope") != list(SCOPE):
        raise ValueError("Deployment receipt producer/scope does not match the clean-login-host contract")
    commit = value.get("git_commit")
    if not isinstance(commit, str) or not COMMIT.fullmatch(commit):
        raise ValueError("Deployment receipt requires a full git_commit")
    try:
        created = datetime.fromisoformat(value["created_utc"].replace("Z", "+00:00"))
        if created.tzinfo is None or created.utcoffset().total_seconds() != 0:
            raise ValueError("created_utc must be UTC-aware")
    except (KeyError, TypeError, AttributeError, ValueError) as error:
        raise ValueError("Deployment receipt created_utc must be an ISO8601 UTC timestamp") from error
    files = value.get("files")
    if (not isinstance(files, dict) or not files
            or not all(isinstance(path, str) and is_source(path) and isinstance(digest, str)
                       and SHA256.fullmatch(digest) for path, digest in files.items())):
        raise ValueError("Deployment receipt contains missing/unknown source paths or invalid SHA256")
    return path, receipt_hash, commit.lower(), {key: digest.lower() for key, digest in files.items()}


def verify_code_snapshot(code_root, expected_commit=None):
    """Return verified effective HEAD and exact source-file fingerprints.

    expected_commit, when provided, must be a full 40-digit commit and match
    exactly. Missing/broken Git never permits trusting a recipe hash by itself.
    A present but failing Git is an error, not a fallback to a deployment file.
    """
    root = Path(code_root).expanduser().absolute()
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Code root must be a real, non-symlink directory")
    root = root.resolve()
    if expected_commit is not None and (not isinstance(expected_commit, str) or not COMMIT.fullmatch(expected_commit)):
        raise ValueError("expected_commit must be a full 40-digit Git commit")
    expected_commit = expected_commit.lower() if expected_commit else None
    binary = shutil.which("git")
    if binary:
        head = clean_git_head(binary, root)
        if expected_commit is not None and expected_commit != head:
            raise ValueError("Expected commit differs from clean Git HEAD")
        expected = git_source_files(binary, root)
        files = actual_source_files(root)
        compare_files(files, expected)
        if clean_git_head(binary, root) != head:
            raise ValueError("Git HEAD changed during source verification")
        mode, receipt_path, receipt_hash = "git_clean_head", None, None
    else:
        receipt_path, receipt_hash, head, expected = read_deployment(root)
        if expected_commit is not None and expected_commit != head:
            raise ValueError("Expected commit differs from deployment receipt git_commit")
        files = actual_source_files(root)
        compare_files(files, expected)
        if file_sha256(receipt_path) != receipt_hash:
            raise ValueError("Deployment receipt changed during verification")
        mode = "deployment_receipt_verified"
    aggregate = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"mode": mode, "effective_git_head": head, "expected_commit": expected_commit,
            "expected_commit_verified": expected_commit is not None,
            "git_binary_missing": binary is None, "git_binary": binary,
            "deployment_receipt": str(receipt_path) if receipt_path else None,
            "deployment_receipt_sha256": receipt_hash,
            "scope": list(SCOPE), "files": files, "file_count": len(files),
            "files_aggregate_sha256": aggregate,
            "trust_boundary": "clean local Git HEAD or exact hashes from trusted login-host deployment receipt"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", "--code-root", type=Path, required=True)
    parser.add_argument("--expected-commit")
    parser.add_argument("--include-files", action="store_true")
    args = parser.parse_args()
    try:
        result = verify_code_snapshot(args.root, args.expected_commit)
    except (OSError, ValueError, TypeError) as error:
        parser.exit(1, json.dumps({"verified": False, "error_type": type(error).__name__, "error": str(error)}) + "\n")
    if not args.include_files:
        result.pop("files")
    print(json.dumps({"verified": True, **result}, sort_keys=True))


if __name__ == "__main__":
    main()
