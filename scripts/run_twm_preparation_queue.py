#!/usr/bin/env python3
"""One-shot remote preparation queue; runs explicit argv lists without a shell.

Recipe keys: code_version, code_root, python, stages. Each stage has name and
command (argv list), optionally cwd, expected_manifest, requires (earlier
stage names), wait_for_json={path,source?,phase:"complete"}, and
adopt_existing_complete=true. {python}/{code_root} may appear in argv strings.
An optional recipe deadline_utc or --deadline-utc is an ISO8601 UTC deadline.

Only a matching prior queue receipt or a manifest declaring this code version
permits an automatic existing-output skip. To adopt an independently prepared
complete manifest whose producer version is missing/different, explicitly set
stage adopt_existing_complete=true or --adopt-existing-complete. The receipt
then states that version was NOT verified; it never attributes old output to
the new code. All skips record the actual manifest SHA256 and bounded limits.

STOP (by default OUTPUT/STOP), SIGINT/SIGTERM, or the deadline stops this queue
and its active child process group. Partial outputs and logs are retained.
Failures do not prevent later independent stages; requires prevents publishing
an aggregate after a prerequisite failed. This is not a scheduler/automation
and never invents a training command.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

from verify_code_snapshot import verify_code_snapshot


SUCCESS_PHASES = {"complete", "skipped"}
TERMINAL_SOURCE_FAILURES = {"failed", "error", "blocked", "cancelled", "canceled", "stopped", "terminated"}
_SIGNAL = None


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def parse_deadline(value):
    if value is None:
        return None
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError("deadline_utc must include UTC Z or +00:00")
    return parsed.timestamp()


def receive_signal(number, _frame):
    global _SIGNAL
    _SIGNAL = int(number)


def stopped_reason(stop_file, deadline):
    if _SIGNAL is not None:
        return f"signal_{_SIGNAL}"
    if stop_file.exists():
        return "STOP_file"
    if deadline is not None and time.time() >= deadline:
        return "deadline_utc_reached"
    return None


def code_snapshot(root, requested_version):
    """Verify clean HEAD or exact trusted deployment bytes, including config."""
    exact_git_request = bool(re.fullmatch(r"[0-9a-fA-F]{7,40}", requested_version))
    verified = verify_code_snapshot(root, requested_version if len(requested_version) == 40 and exact_git_request else None)
    if exact_git_request and not verified["effective_git_head"].startswith(requested_version.lower()):
        raise ValueError("recipe code_version differs from verified effective Git HEAD")
    return {"requested_code_version": requested_version,
            # Keep the old receipt field for readers; scope now explicitly
            # includes config JSON and is identified by the verifier below.
            "python_source_sha256": verified["files_aggregate_sha256"],
            "git_head": verified["effective_git_head"],
            "tracked_worktree_dirty": False if not verified["git_binary_missing"] else None,
            "requested_git_sha_matches_head": exact_git_request, **verified}


def read_manifest(path):
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or value.get("phase") != "complete":
        raise ValueError(f"Expected an explicitly complete JSON manifest: {path}")
    result = {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size,
              "phase": value["phase"]}
    for key in ("dataset", "source", "code_version", "producer_code_version", "source_code_version",
                "git_revision", "git_commit", "source_revision", "bounded_limits", "frames_by_split",
                "partial_gt", "gt_complete"):
        if key in value:
            result[key] = value[key]
    if isinstance(value.get("manifests"), list):
        result["referenced_manifests"] = value["manifests"]
    return result


def declared_producer_version(manifest):
    # source_revision often refers to a dataset/upstream decoder, not this
    # adapter. Never use it as proof of the preparation code version.
    for key in ("producer_code_version", "code_version", "source_code_version", "git_commit", "git_revision"):
        if isinstance(manifest.get(key), str):
            return manifest[key]
    return None


def source_wait_state(spec):
    path = Path(spec["path"])
    if not path.is_file():
        return {"ready": False, "path": str(path), "reason": "status_file_not_present"}
    try:
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise ValueError("status JSON is not an object")
    except (OSError, ValueError) as error:
        return {"ready": False, "path": str(path), "reason": "status_json_unreadable",
                "error": f"{type(error).__name__}: {error}"}
    selected = value
    source = spec.get("source")
    if source is not None and isinstance(value.get("sources"), dict):
        selected = value["sources"].get(source)
        if not isinstance(selected, dict):
            return {"ready": False, "path": str(path), "source": source, "reason": "source_not_yet_in_status"}
    elif source is not None and value.get("dataset", value.get("source")) not in (None, source):
        return {"ready": False, "failed": True, "path": str(path), "source": source,
                "reason": "status_source_identity_mismatch"}
    phase = selected.get("phase")
    state = {"ready": phase == spec.get("phase", "complete"), "path": str(path),
             "source": source, "observed_phase": phase, "wanted_phase": spec.get("phase", "complete")}
    if phase in TERMINAL_SOURCE_FAILURES:
        state.update(failed=True, reason="upstream_source_terminal_failure",
                     upstream_error=selected.get("error", selected.get("reason")))
    if state["ready"]:
        state["status_sha256"] = sha256(path)
    return state


def terminate_child(process, deadline=None, grace_seconds=10):
    """Terminate all spawned workers, with no grace beyond the hard deadline."""
    if process.poll() is not None:
        return process.returncode
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return process.wait()
    grace = float(grace_seconds)
    if deadline is not None:
        grace = min(grace, max(0.0, deadline-time.time()))
    try:
        return process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return process.wait()


def resolve_stage(stage, recipe, default_root):
    resolved = dict(stage)
    replacements = {"{python}": recipe["python"], "{code_root}": str(default_root)}
    def expand(value):
        for token, replacement in replacements.items():
            value = value.replace(token, replacement)
        return value
    command = [expand(argument) for argument in stage["command"]]
    if command[0] in ("python", "python3"):
        command[0] = recipe["python"]
    resolved["command"] = command
    cwd = Path(expand(stage.get("cwd", str(default_root)))).expanduser()
    if not cwd.is_absolute():
        cwd = default_root / cwd
    resolved["cwd"] = str(cwd.resolve())
    if not cwd.is_dir():
        raise ValueError(f"Stage cwd is not a directory: {cwd}")
    if stage.get("expected_manifest") is not None:
        path = Path(expand(stage["expected_manifest"])).expanduser()
        resolved["expected_manifest"] = str((path if path.is_absolute() else cwd / path).resolve())
    if stage.get("wait_for_json") is not None:
        waiting = dict(stage["wait_for_json"])
        path = Path(expand(waiting["path"])).expanduser()
        waiting["path"] = str((path if path.is_absolute() else cwd / path).resolve())
        resolved["wait_for_json"] = waiting
    return resolved


def validate_recipe(recipe):
    if not isinstance(recipe, dict):
        raise ValueError("Recipe must be a JSON object")
    for key in ("code_version", "code_root", "python"):
        if not isinstance(recipe.get(key), str) or not recipe[key].strip():
            raise ValueError(f"Recipe requires a nonempty string {key}")
    root = Path(recipe["code_root"]).expanduser().resolve()
    if not root.is_dir() or not Path(recipe["python"]).is_absolute() or not os.access(recipe["python"], os.X_OK):
        raise ValueError("Require existing code_root and an absolute executable python path")
    if not isinstance(recipe.get("stages"), list) or not recipe["stages"]:
        raise ValueError("Recipe requires a nonempty stages list")
    names, resolved = set(), []
    for stage in recipe["stages"]:
        if not isinstance(stage, dict):
            raise ValueError("Every stage must be a JSON object")
        name = stage.get("name")
        if (not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name)
                or name in names):
            raise ValueError(f"Invalid/duplicate stage name: {name!r}")
        command = stage.get("command")
        if not isinstance(command, list) or not command or not all(isinstance(v, str) and v for v in command):
            raise ValueError(f"{name}: command must be a nonempty argv string list")
        requires = stage.get("requires", [])
        if (not isinstance(requires, list) or not all(isinstance(v, str) for v in requires)
                or len(set(requires)) != len(requires) or not set(requires).issubset(names)):
            raise ValueError(f"{name}: requires must name distinct EARLIER stages")
        for key in ("cwd", "expected_manifest"):
            if key in stage and (not isinstance(stage[key], str) or not stage[key].strip()):
                raise ValueError(f"{name}: {key} must be a nonempty string when supplied")
        if "adopt_existing_complete" in stage and not isinstance(stage["adopt_existing_complete"], bool):
            raise ValueError(f"{name}: adopt_existing_complete must be boolean")
        waiting = stage.get("wait_for_json")
        if waiting is not None and (not isinstance(waiting, dict) or not isinstance(waiting.get("path"), str)
                or not waiting["path"] or waiting.get("phase", "complete") != "complete"):
            raise ValueError(f"{name}: wait_for_json requires path and phase='complete'")
        if waiting is not None and "source" in waiting and (not isinstance(waiting["source"], str) or not waiting["source"]):
            raise ValueError(f"{name}: wait_for_json.source must be a nonempty string")
        names.add(name)
        resolved.append(resolve_stage(stage, recipe, root))
    return root, resolved


def run_queue(args, recipe, stages, root, recipe_hash, deadline, stop_file):
    output = args.output.resolve()
    status_path = output / "queue_status.json"
    state = {"created_utc": utc(), "updated_utc": utc(), "phase": "running", "pid": os.getpid(),
             "recipe_path": str(args.recipe.resolve()), "recipe_sha256": recipe_hash,
             "requested_code_version": recipe["code_version"], "code_root": str(root),
             "python": recipe["python"], "deadline_utc": datetime.fromtimestamp(deadline, timezone.utc).isoformat() if deadline else None,
             "stop_file": str(stop_file), "stage": None, "child_pid": None, "returncode": None,
             "stages": {stage["name"]: {"phase": "pending", "pid": None, "returncode": None} for stage in stages}}
    def publish():
        state["updated_utc"] = utc()
        atomic_json(status_path, state)
    publish()
    global_stop = None
    for stage in stages:
        name = stage["name"]
        current = state["stages"][name]
        state["stage"] = name
        global_stop = global_stop or stopped_reason(stop_file, deadline)
        if global_stop:
            current.update(phase="not_started", reason=global_stop)
            publish()
            continue
        dependencies = {required: state["stages"][required]["phase"] for required in stage.get("requires", [])}
        if any(phase not in SUCCESS_PHASES for phase in dependencies.values()):
            current.update(phase="blocked", reason="prerequisite_stage_not_successful", requires=dependencies)
            publish()
            continue
        log_path = output / "logs" / (name + ".log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path = output / "receipts" / (name + ".json")
        current.update(phase="starting", started_utc=utc(), command=stage["command"], cwd=stage["cwd"],
                       log_path=str(log_path), receipt_path=str(receipt_path))
        publish()
        with log_path.open("ab", buffering=0) as log:
            def message(value):
                log.write((json.dumps({"utc": utc(), **value}, ensure_ascii=False) + "\n").encode())
            message({"queue_stage_start": name, "code_version": recipe["code_version"], "command": stage["command"]})
            process = None
            try:
                expected = Path(stage["expected_manifest"]) if stage.get("expected_manifest") else None
                waiting = stage.get("wait_for_json")
                # A completed prepared output does not need its historical
                # downloader to be running/available again. Its manifest and
                # producer receipt are checked below before accepting a skip.
                if waiting and not (expected is not None and expected.exists()):
                    current["phase"] = "waiting_for_source"
                    last_wait = None
                    next_poll = 0.0
                    while True:
                        reason = stopped_reason(stop_file, deadline)
                        if reason:
                            global_stop = reason
                            raise InterruptedError(reason)
                        if time.monotonic() >= next_poll:
                            observed = source_wait_state(waiting)
                            current["source_wait"] = observed
                            publish()
                            if observed != last_wait:
                                message({"source_wait": observed})
                                last_wait = observed
                            if observed.get("failed"):
                                raise RuntimeError("Source preparation prerequisite failed: " + json.dumps(observed))
                            if observed["ready"]:
                                break
                            next_poll = time.monotonic() + args.poll_seconds
                        time.sleep(0.5)
                snapshot = code_snapshot(root, recipe["code_version"])
                reason = stopped_reason(stop_file, deadline)
                if reason:
                    global_stop = reason
                    raise InterruptedError(reason)
                # Identical source bytes can belong to different commits or
                # freshly exported deployment receipts. Name the metadata by
                # its entire identity, so the saved provenance never points
                # silently at an older HEAD/receipt with the same source hash.
                snapshot_metadata_sha = hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest()
                snapshot_path = output / "code_snapshots" / (snapshot_metadata_sha + ".json")
                if not snapshot_path.exists():
                    atomic_json(snapshot_path, snapshot)
                current["code_snapshot_sha256"] = snapshot["python_source_sha256"]
                current["code_snapshot_path"] = str(snapshot_path)
                current["code_snapshot_metadata_sha256"] = snapshot_metadata_sha
                current["effective_git_head"] = snapshot["effective_git_head"]
                current["code_verification_mode"] = snapshot["mode"]
                current["git_binary_missing"] = snapshot["git_binary_missing"]
                current["deployment_receipt_sha256"] = snapshot["deployment_receipt_sha256"]
                if expected is not None and expected.exists():
                    existing = read_manifest(expected)
                    previous = json.loads(receipt_path.read_text()) if receipt_path.is_file() else {}
                    exact_previous = (previous.get("phase") in SUCCESS_PHASES
                        and previous.get("recipe_sha256") == recipe_hash
                        and previous.get("requested_code_version") == recipe["code_version"]
                        and previous.get("code_snapshot_sha256") == snapshot["python_source_sha256"]
                        and previous.get("command") == stage["command"]
                        and previous.get("cwd") == stage["cwd"]
                        and previous.get("manifest", {}).get("sha256") == existing["sha256"])
                    producer = declared_producer_version(existing)
                    version_claim_matches = producer == recipe["code_version"]
                    adopted = bool(args.adopt_existing_complete or stage.get("adopt_existing_complete"))
                    if not (exact_previous or version_claim_matches or adopted):
                        raise RuntimeError("Existing complete manifest producer/recipe is unverified; output preserved. "
                            "Use explicit adopt_existing_complete=true only after checking this source/configuration.")
                    current.update(phase="skipped", returncode=0, completed_utc=utc(), manifest=existing,
                        skip_reason="matching_prior_queue_receipt" if exact_previous else (
                            "manifest_declares_requested_code_version" if version_claim_matches else "explicit_existing_output_adoption"),
                        producer_version=previous.get("producer_version") if exact_previous else producer,
                        producer_code_version_verified=bool(version_claim_matches or (
                            exact_previous and previous.get("producer_code_version_verified", False))))
                    message({"queue_stage_skipped": name, "reason": current["skip_reason"], "manifest_sha256": existing["sha256"]})
                else:
                    reason = stopped_reason(stop_file, deadline)
                    if reason:
                        global_stop = reason
                        raise InterruptedError(reason)
                    environment = os.environ.copy()
                    environment.setdefault("PYTHONUNBUFFERED", "1")
                    process = subprocess.Popen(stage["command"], cwd=stage["cwd"], env=environment,
                        stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                        start_new_session=True, shell=False)
                    current.update(phase="running", pid=process.pid)
                    state["child_pid"] = process.pid
                    publish()
                    next_status = time.monotonic()
                    while process.poll() is None:
                        reason = stopped_reason(stop_file, deadline)
                        if reason:
                            global_stop = reason
                            current["returncode"] = terminate_child(process, deadline)
                            raise InterruptedError(reason)
                        if time.monotonic() >= next_status:
                            publish()
                            next_status = time.monotonic() + 5
                        # At a hard deadline do not wait for the normal 30s
                        # source poll; this loop observes it at <=0.5s cadence.
                        time.sleep(min(0.5, max(0.001, deadline-time.time())) if deadline else 0.5)
                    current["returncode"] = process.returncode
                    if process.returncode != 0:
                        raise RuntimeError(f"Stage child exited with returncode={process.returncode}; see {log_path}")
                    manifest = read_manifest(expected) if expected is not None else None
                    current.update(phase="complete", completed_utc=utc(), manifest=manifest,
                                   producer_version=recipe["code_version"], producer_code_version_verified=True)
                    message({"queue_stage_complete": name, "manifest_sha256": manifest["sha256"] if manifest else None})
            except InterruptedError as error:
                current.update(phase="stopped", reason=str(error), completed_utc=utc())
                message({"queue_stage_stopped": name, "reason": str(error)})
            except Exception as error:
                if process is not None and process.poll() is None:
                    current["returncode"] = terminate_child(process, deadline)
                current.update(phase="failed", completed_utc=utc(), error_type=type(error).__name__, error=str(error))
                message({"queue_stage_failed": name, "error_type": type(error).__name__, "error": str(error)})
            finally:
                state["child_pid"] = None
                receipt = {"stage": name, "recipe_sha256": recipe_hash,
                           "requested_code_version": recipe["code_version"], **current}
                atomic_json(receipt_path, receipt)
                publish()
    phases = [entry["phase"] for entry in state["stages"].values()]
    if global_stop or any(phase in ("stopped", "not_started") for phase in phases):
        state.update(phase="stopped", reason=global_stop, returncode=2)
    elif any(phase not in SUCCESS_PHASES for phase in phases):
        state.update(phase="failed", returncode=1)
    else:
        state.update(phase="complete", returncode=0)
    state["completed_utc"] = utc()
    state["stage"] = None
    publish()
    print(json.dumps({"phase": state["phase"], "returncode": state["returncode"],
                      "status_path": str(status_path), "stages": {name: entry["phase"] for name, entry in state["stages"].items()}}), flush=True)
    return state["returncode"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output", "--state-dir", type=Path, required=True)
    parser.add_argument("--deadline-utc", default=None)
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--adopt-existing-complete", action="store_true")
    args = parser.parse_args()
    if not Path("/sciclone/home/jchen57").is_dir():
        parser.error("Run this preparation queue only in the authorized remote ds Pod")
    if not 1 <= args.poll_seconds <= 60:
        parser.error("poll-seconds must be between 1 and 60")
    try:
        recipe = json.loads(args.recipe.read_text())
        root, stages = validate_recipe(recipe)
        deadline = parse_deadline(args.deadline_utc or recipe.get("deadline_utc"))
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    args.output.mkdir(parents=True, exist_ok=True)
    stop_file = (args.stop_file or args.output / "STOP").resolve()
    signal.signal(signal.SIGINT, receive_signal)
    signal.signal(signal.SIGTERM, receive_signal)
    with (args.output / "queue.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("This queue output already has a running owner")
        recipe_hash = sha256(args.recipe)
        atomic_json(args.output / "resolved_recipe.json", {"recipe_sha256": recipe_hash,
                    "requested_code_version": recipe["code_version"], "code_root": str(root),
                    "python": recipe["python"], "stages": stages})
        sys.exit(run_queue(args, recipe, stages, root, recipe_hash, deadline, stop_file))


if __name__ == "__main__":
    main()
