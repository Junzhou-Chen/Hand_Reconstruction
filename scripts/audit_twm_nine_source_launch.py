#!/usr/bin/env python3
"""Remote-only, read-only nine-source TWM launch acceptance.

Print one compact JSON result to stdout; the caller may redirect it to an
independent audit location. Never create STOP files, change checkpoints, write
tracking state, initialize a W&B run, or print credential values. A live
checkpoint is hashed and loaded through ONE open descriptor, so an atomic
latest.pt replacement cannot mix bytes from two checkpoint generations.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
import math
import numbers
import os
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True

BASE = Path("/sciclone/data10/jchen57/Hand_Reconstruction")
DEFAULT_RUN = BASE / "runs/twm_nine_sources_moe_transient_v3_b512_10k_20261017"
EXPECTED = {"oakink2", "hot3d", "umetrack", "dexycb", "interhand", "taco", "gigahands", "arctic", "ho3d"}


class AuditFailure(RuntimeError):
    pass


def utc():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise AuditFailure(message)


def read_json(path):
    payload = Path(path).read_bytes()
    return json.loads(payload), hashlib.sha256(payload).hexdigest()


def identity(stat):
    return {"device": stat.st_dev, "inode": stat.st_ino, "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns}


def tensor_model_hash(model, torch):
    """Exactly train_probabilistic_twm.py save(): ordered names/dtype/shape/bytes."""
    digest = hashlib.sha256()
    for name, tensor in model.items():
        require(isinstance(tensor, torch.Tensor), "Model state has an unexpected non-tensor entry")
        digest.update((name + str(tensor.dtype) + str(tuple(tensor.shape))).encode())
        tensor = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        for begin in range(0, tensor.numel(), 4 * 1024**2):
            digest.update(tensor[begin:begin + 4 * 1024**2].numpy().tobytes())
    return digest.hexdigest()


def recursive_finite(value, torch):
    """Bound temporary memory and report paths/counts, never tensor contents."""
    import numpy as np
    stats = {"all_tensors": 0, "floating_tensors": 0, "floating_elements": 0,
             "floating_numpy_arrays": 0, "python_float_values": 0,
             "all_tensor_devices_cpu": True, "nonfinite_paths": []}
    pending = [("root", value)]
    while pending:
        location, item = pending.pop()
        if isinstance(item, torch.Tensor):
            stats["all_tensors"] += 1
            if item.device.type != "cpu":
                stats["all_tensor_devices_cpu"] = False
            if item.is_floating_point() or item.is_complex():
                stats["floating_tensors"] += 1
                stats["floating_elements"] += item.numel()
                flattened = item.detach().reshape(-1)
                for begin in range(0, flattened.numel(), 262144):
                    if not bool(torch.isfinite(flattened[begin:begin + 262144]).all()):
                        if len(stats["nonfinite_paths"]) < 20:
                            stats["nonfinite_paths"].append(location)
                        break
        elif isinstance(item, np.ndarray) and item.dtype.kind in "fc":
            stats["floating_numpy_arrays"] += 1
            flattened = item.reshape(-1)
            for begin in range(0, flattened.size, 262144):
                if not np.isfinite(flattened[begin:begin + 262144]).all():
                    if len(stats["nonfinite_paths"]) < 20:
                        stats["nonfinite_paths"].append(location)
                    break
        elif isinstance(item, Mapping):
            pending.extend((f"{location}.{key}", nested) for key, nested in item.items())
        elif isinstance(item, (tuple, list)):
            pending.extend((f"{location}[{index}]", nested) for index, nested in enumerate(item))
        elif isinstance(item, numbers.Real) and not isinstance(item, (int, bool)):
            stats["python_float_values"] += 1
            if not math.isfinite(float(item)) and len(stats["nonfinite_paths"]) < 20:
                stats["nonfinite_paths"].append(location)
    stats["finite"] = not stats["nonfinite_paths"]
    return stats


def checkpoint_snapshot(path, torch):
    with Path(path).open("rb") as stream:
        first = identity(os.fstat(stream.fileno()))
        require(first["bytes"] > 0, "Checkpoint is empty")
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
        stream.seek(0)
        # This is the project's own trusted optimizer checkpoint, not downloaded
        # arbitrary model code. map_location prevents any CUDA allocation.
        checkpoint = torch.load(stream, map_location="cpu", weights_only=False)
        last = identity(os.fstat(stream.fileno()))
        require(first == last, "Opened checkpoint inode changed while hashing/loading")
        try:
            latest_path = identity(Path(path).stat())
        except FileNotFoundError:
            latest_path = None
        receipt = {"path": str(path), "opened_file": first, "file_sha256": digest.hexdigest(),
                   "same_open_fd_for_hash_and_load": True, "opened_inode_unchanged": True,
                   "live_path_still_same_inode": latest_path == first,
                   "race_policy": "an atomic replacement of live path is allowed; audit refers only to this immutable opened inode"}
    return checkpoint, receipt


def positive_source_counts(value, label):
    require(isinstance(value, Mapping) and set(value) == EXPECTED,
            f"{label} must identify exactly nine sources")
    counts = {name: int(value[name]) for name in sorted(EXPECTED)}
    require(all(count > 0 for count in counts.values()), f"{label} has a source with no usable windows")
    return counts


def inspect_source_contract(metadata, checkpoint):
    config = metadata["config"]
    require(set(metadata["dataset_names"]) == EXPECTED, "Run dataset names differ from nine-source requirement")
    require(set(config["required_datasets"]) == EXPECTED, "Training config does not require exactly nine sources")
    train = positive_source_counts(metadata["train_windows_by_dataset"], "run train windows")
    val = positive_source_counts(metadata["val_windows_by_dataset"], "run validation windows")
    manifest_path = Path(metadata["manifest"])
    collection, collection_hash = read_json(manifest_path)
    require(collection_hash == metadata["manifest_sha256"], "Live collection differs from pinned training manifest")
    require(collection.get("phase") == "complete", "Collection is not complete")
    require(set(collection["required_datasets"]) == EXPECTED, "Collection source requirement differs")
    require(checkpoint["run"]["manifest_sha256"] == collection_hash,
            "Checkpoint belongs to another training collection")
    require(set(checkpoint["config"]["required_datasets"]) == EXPECTED,
            "Checkpoint source configuration differs")
    source_receipts = {}
    for reference in collection["manifests"]:
        name = reference["dataset"]
        require(name in EXPECTED and name not in source_receipts, "Duplicate/unknown collection source")
        path = (manifest_path.parent / reference["path"]).resolve()
        source, digest = read_json(path)
        require(digest == reference["manifest_sha256"], f"Source manifest changed: {name}")
        require(source.get("phase") == "complete" and source.get("dataset") == name,
                f"Source identity/preparation incomplete: {name}")
        source_receipts[name] = {"path": str(path), "sha256": digest, "phase": source["phase"]}
    require(set(source_receipts) == EXPECTED, "Collection must retain all nine source manifests")
    for split, counts in (("train", train), ("val", val)):
        inspected = positive_source_counts(collection["loader_inspection"][split]["windows"],
                                           f"collection {split} windows")
        require(inspected == counts, f"Run and actual-loader collection counts differ: {split}")
        coverage = metadata[f"{split}_forecast_contract"]["future_coverage"]
        require(set(coverage) == EXPECTED, f"Run {split} forecast coverage omitted a source")
        require(all(int(coverage[name]["windows"]) == counts[name] for name in EXPECTED),
                f"Run {split} forecast window counts differ")
    samplers = []
    for rank_state in checkpoint["rank_states"]:
        sampler = rank_state["sampler"]
        if "dataset_names" in sampler:
            require(set(sampler["dataset_names"]) == EXPECTED, "Balanced sampler omits a source")
            require(positive_source_counts(sampler["windows_by_dataset"], "sampler windows") == train,
                    "Sampler counts differ from actual training population")
            samplers.append({"rank": rank_state["rank"], "kind": "source_balanced_with_replacement",
                             "source_count": 9, "num_samples": int(sampler["num_samples"])})
        else:
            require(sampler.get("replicas") == 2 and sampler.get("rank") == rank_state["rank"],
                    "Coverage sampler's rank/world configuration differs")
            require(int(sampler["num_samples"]) == math.ceil(sum(train.values()) / 2),
                    "Coverage sampler population differs from all-nine-source windows")
            samplers.append({key: sampler[key] for key in
                             ("rank", "replicas", "epoch", "batch_cursor", "num_samples", "batch_size")})
    return {"train_windows": train, "val_windows": val, "collection_manifest": str(manifest_path),
            "collection_manifest_sha256": collection_hash, "source_manifests": source_receipts,
            "all_nine_train_and_val_sources_positive": True, "samplers": samplers}


def wandb_server_snapshot(metadata, key_file, minimum, timeout, maximum_rows):
    tracking = metadata.get("wandb", {})
    require(tracking.get("mode") == "online" and tracking.get("enabled"), "Run is not configured for online W&B")
    fields = {name: tracking.get(name) for name in ("entity", "project", "id")}
    require(all(isinstance(value, str) and value for value in fields.values()), "W&B server identity missing from run metadata")
    key = Path(key_file).read_text().strip()
    require(bool(key), "W&B private key file is empty")
    import wandb
    api = wandb.Api(api_key=key, timeout=timeout)
    server = api.run("/".join(fields[name] for name in ("entity", "project", "id")))
    del key
    require(server.state in ("running", "finished"), "W&B server state is not running/finished")
    rows, visited = [], 0
    for row in server.scan_history(keys=["optimizer_step", "train/loss"], page_size=200):
        visited += 1
        step, loss = row.get("optimizer_step"), row.get("train/loss")
        if isinstance(step, numbers.Real) and isinstance(loss, numbers.Real):
            if float(step) >= minimum and math.isfinite(float(step)) and math.isfinite(float(loss)):
                rows.append({"optimizer_step": int(step), "train/loss": float(loss)})
                if len(rows) >= 5:
                    break
        if visited >= maximum_rows:
            break
    require(bool(rows), "W&B API has not yet delivered a same-row optimizer_step>=minimum and finite train/loss")
    summary = server.summary
    selected_summary = {}
    for name in ("optimizer_step", "train/loss"):
        value = summary.get(name)
        if isinstance(value, numbers.Real) and math.isfinite(float(value)):
            selected_summary[name] = float(value)
    return {"state": server.state, "url": tracking.get("url"), "identity": fields,
            "training_rows": rows, "summary": selected_summary,
            "server_history_verified": True, "private_credentials_printed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--min-updates", type=int, default=10)
    parser.add_argument("--key-file", type=Path, default=BASE / "private/wandb_api_key")
    parser.add_argument("--wandb-timeout", type=int, default=25)
    parser.add_argument("--max-history-rows", type=int, default=1000)
    args = parser.parse_args()
    result = {"checked_utc": utc(), "run_directory": str(args.run), "read_only": True,
              "required_minimum_updates": args.min_updates, "accepted": False}
    try:
        require(Path("/sciclone/home/jchen57").is_dir(), "Run acceptance only in the authorized remote Pod")
        require(args.min_updates >= 10, "Acceptance requires at least ten optimizer updates")
        import torch
        torch.set_num_threads(2)
        metadata, metadata_hash = read_json(args.run / "run_metadata.json")
        status, status_hash = read_json(args.run / "status.json")
        require(int(status.get("step", -1)) >= args.min_updates, "Status has not reached the required actual optimizer step")
        require(status.get("phase") in ("training", "complete"), "Training is not running or successfully complete")
        require(int(metadata.get("world_size", 0)) == 2, "Expected actual two-rank DDP run")
        result["metadata"] = {"sha256": metadata_hash, "code_version": metadata.get("code_version"),
                              "global_batch_size": metadata.get("global_batch_size"),
                              "per_gpu_batch_size": metadata.get("per_gpu_batch_size"),
                              "world_size": metadata["world_size"], "sampling": metadata.get("sampling")}
        result["status"] = {key: status.get(key) for key in ("utc", "phase", "step", "loss", "lr", "grad_norm", "samples_by_dataset")}
        result["status"]["sha256"] = status_hash
        if isinstance(status.get("loss"), numbers.Real):
            require(math.isfinite(float(status["loss"])), "Live status loss is nonfinite")
        checkpoint, receipt = checkpoint_snapshot(args.run / "latest.pt", torch)
        require(int(checkpoint.get("step", -1)) >= args.min_updates, "Latest immutable checkpoint is older than ten updates")
        require(checkpoint.get("code_version") == metadata.get("code_version"), "Checkpoint/run code versions differ")
        require(checkpoint["run"].get("world_size") == 2, "Checkpoint does not belong to two-rank training")
        model, optimizer = checkpoint["model"], checkpoint["optimizer"]
        require(isinstance(model, Mapping) and len(model) > 0, "Missing model state")
        require(isinstance(optimizer.get("state"), Mapping) and len(optimizer["state"]) > 0, "Optimizer has no trained state")
        model_finite, optimizer_finite = recursive_finite(model, torch), recursive_finite(optimizer, torch)
        require(model_finite["finite"] and optimizer_finite["finite"], "Nonfinite model/optimizer state")
        require(model_finite["all_tensor_devices_cpu"] and optimizer_finite["all_tensor_devices_cpu"], "Checkpoint was not fully mapped to CPU")
        parameter_steps = []
        for parameter_state in optimizer["state"].values():
            value = parameter_state.get("step")
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                parameter_steps.append(float(value.item()))
            elif isinstance(value, numbers.Real):
                parameter_steps.append(float(value))
        require(bool(parameter_steps) and max(parameter_steps) >= args.min_updates,
                "Adam state does not yet prove at least ten actual parameter updates")
        require(max(parameter_steps) <= int(checkpoint["step"]), "Optimizer parameter step exceeds checkpoint step")
        rank_states = checkpoint["rank_states"]
        require(len(rank_states) == 2 and {entry["rank"] for entry in rank_states} == {0, 1}, "Missing one of the two rank states")
        hashes = [entry.get("model_state_sha256") for entry in rank_states]
        require(all(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) for value in hashes), "Missing canonical per-rank model hashes")
        require(len(set(hashes)) == 1, "DDP model hashes disagree")
        recomputed = tensor_model_hash(model, torch)
        require(recomputed == hashes[0], "Checkpoint model tensor bytes differ from saved rank model hashes")
        receipt.update(step=int(checkpoint["step"]), model_state_sha256=recomputed,
                       rank_hashes_equal=True, checkpoint_model_hash_matches_ranks=True,
                       optimizer_parameter_step_min=min(parameter_steps),
                       optimizer_parameter_step_max=max(parameter_steps),
                       model_finite=model_finite, optimizer_finite=optimizer_finite)
        result["checkpoint"] = receipt
        result["source_coverage"] = inspect_source_contract(metadata, checkpoint)
        observed = status.get("samples_by_dataset")
        if isinstance(observed, Mapping):
            result["source_coverage"]["observed_training_samples"] = {name: int(observed.get(name, 0)) for name in sorted(EXPECTED)}
            result["source_coverage"]["early_sampling_note"] = "window availability is checked for all nine; zero early observed samples is not treated as a missing source"
        result["wandb_server"] = wandb_server_snapshot(metadata, args.key_file, args.min_updates,
                                                       args.wandb_timeout, args.max_history_rows)
        result.update(accepted=True, phase="accepted", completed_utc=utc())
    except AuditFailure as error:
        result.update(phase="not_yet_accepted", reason=str(error), completed_utc=utc())
    except Exception as error:
        # Remote auth/network exceptions may carry headers or credential-bearing
        # request details. Print ONLY a type, never the raw exception/repr/trace.
        result.update(phase="audit_error", error_type=type(error).__name__, completed_utc=utc())
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0 if result["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
