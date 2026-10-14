#!/usr/bin/env python3
"""Remote GPU1 training-step probe; repeated real windows are NOT a quality test.

The parent must set CUDA_VISIBLE_DEVICES=1. No production checkpoint is saved
or changed. --output accepts a new JSON file or a directory for receipt.json.
Numbers exclude data loading and DDP communication; no safe batch is inferred.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "config", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--batch-sizes", nargs="+", default=["128,256,512,768,1024"])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    if not Path("/sciclone/home/jchen57").is_dir():
        parser.error("Run this benchmark only in the authorized remote compute Pod")
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() != "1":
        parser.error("The parent must explicitly set CUDA_VISIBLE_DEVICES=1")
    try:
        sizes = sorted({int(x) for group in args.batch_sizes for x in group.split(",")})
    except ValueError:
        parser.error("--batch-sizes must contain positive integers separated by commas or spaces")
    if not sizes or min(sizes) < 1 or args.warmup < 0 or args.iterations < 1:
        parser.error("Positive batch sizes/iterations and nonnegative warmup required")
    output = args.output if args.output.suffix == ".json" else args.output / "receipt.json"
    if output.exists():
        parser.error("Receipt already exists; choose a fresh output")
    output.parent.mkdir(parents=True, exist_ok=True)

    # Import Torch/model code only after the remote and GPU selection guards.
    from train_twm import (torch, TWM, TWMConfig, TrajectoryWindows,
                           sample_batch, state_loss, atomic_json, utc)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one visible CUDA device is required")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This probe requires BF16 support")
    torch.set_num_threads(2)
    device = torch.device("cuda:0")  # Remapped physical GPU1 only.
    config = json.loads(args.config.read_text())
    dataset = TrajectoryWindows(args.manifest, "train", config["history_frames"],
                                config["forecast_offset_frames"], config["window_stride"],
                                cache_sequences=4, cache_bytes=256 * 1024**2)
    selected = dataset.fixed_indices_by_dataset(1)
    if set(selected) != set(config.get("required_datasets", dataset.dataset_names)):
        raise ValueError("Collection sources differ from the configured required datasets")
    if not selected or any(len(ids) != 1 for ids in selected.values()):
        raise ValueError("Every source must supply one real complete train window")
    samples = [dataset[ids[0]] for ids in selected.values()]
    base = {key: torch.stack([sample[key] for sample in samples]).to(device)
            for key in samples[0]}
    del dataset, samples
    weights, checkpoint_info = None, None
    if args.checkpoint:
        # One open file descriptor remains consistent if latest.pt is replaced.
        with args.checkpoint.open("rb") as stream:
            checkpoint = torch.load(stream, map_location="cpu", weights_only=False)
        if checkpoint.get("config", {}).get("model") != config["model"]:
            raise ValueError("Checkpoint model configuration differs from the probe")
        weights = checkpoint["model"]
        checkpoint_info = {"path": str(args.checkpoint.resolve()), "loaded_step": checkpoint["step"],
                           "code_version": checkpoint.get("code_version"), "optimizer_loaded": False}
        del checkpoint
    report = {"phase": "running", "started_utc": utc(), "manifest": str(args.manifest.resolve()),
              "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
              "config": config, "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "trainer_sha256": hashlib.sha256(Path(__file__).with_name("train_twm.py").read_bytes()).hexdigest(),
              "python": sys.executable, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(device),
              "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
              "total_vram_bytes": torch.cuda.get_device_properties(device).total_memory,
              "base_window_indices_by_source": selected, "checkpoint": checkpoint_info,
              "batch_sizes": sizes, "warmup": args.warmup, "iterations": args.iterations,
              "scope": "Single-GPU repeated-real-window corruption + BF16 network/FP32 geometry + backward/AdamW probe. Excludes loader, validation and DDP; not quality or a production-safe-batch claim.",
              "results": []}
    atomic_json(output, report)

    def probe(batch_size):
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        model = TWM(TWMConfig(**config["model"])).to(device).train()
        if weights is not None:
            model.load_state_dict(weights, strict=True)
        groups = [{"params": [p for p in model.parameters()
                              if bool(getattr(p, "_no_weight_decay", False)) == exempt],
                   "weight_decay": 0.0 if exempt else config["weight_decay"]}
                  for exempt in (False, True)]
        optimizer = torch.optim.AdamW(groups, lr=config["learning_rate"])
        index = torch.arange(batch_size, device=device) % len(selected)
        batch = {key: value.index_select(0, index) for key, value in base.items()}
        generator = torch.Generator(device=device).manual_seed(config["seed"])

        def step():
            history, raw, candidate, kwargs, target, future = sample_batch(
                batch, generator, condition_on_beta=config.get("condition_on_beta", True))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = model(history, raw, candidate, **kwargs)
            loss = (state_loss(prediction["state"], target)
                    + config["forecast_weight"] * state_loss(prediction["forecast"][:, 0], future))
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite loss")
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            return {"loss": float(loss.detach()), "grad_norm": float(grad)}

        for _ in range(args.warmup):
            step()
        measurements = []
        for _ in range(args.iterations):
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            values = step()
            torch.cuda.synchronize(device)
            measurements.append({**values, "seconds": time.perf_counter() - start})
        tensors = list(model.parameters()) + [v for state in optimizer.state.values()
                                              for v in state.values() if torch.is_tensor(v)]
        if any(not bool(torch.isfinite(t).all()) for t in tensors if t.is_floating_point()):
            raise FloatingPointError("Nonfinite parameter or optimizer state after measured steps")
        seconds = sum(item["seconds"] for item in measurements)
        return {"batch_size": batch_size, "status": "passed", "measurements": measurements,
                "steps_per_second": args.iterations / seconds,
                "samples_per_second": batch_size * args.iterations / seconds,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                "finite_loss_grad_parameters_optimizer": True,
                "no_weight_decay_parameters": sum(p.numel() for p in groups[1]["params"])}

    for size in sizes:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        free_before, _ = torch.cuda.mem_get_info(device)
        try:
            result = probe(size)
        except torch.cuda.OutOfMemoryError as exc:
            result = {"batch_size": size, "status": "oom", "error": str(exc),
                      "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                      "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}
        except Exception as exc:
            result = {"batch_size": size, "status": "failed", "error_type": type(exc).__name__, "error": str(exc)}
        result["free_vram_before_bytes"] = free_before
        report["results"].append(result)
        report["updated_utc"] = utc()
        atomic_json(output, report)
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except RuntimeError as exc:
            result["cleanup_error"] = str(exc)
            result["status"] = "failed"
        print(json.dumps(result), flush=True)
        if result["status"] != "passed":
            break  # Never try a larger batch after OOM or numerical/runtime failure.
    last = report["results"][-1]["status"]
    report["phase"] = "complete" if last == "passed" else "stopped_" + last
    report["completed_utc"] = utc()
    report["untested_batch_sizes"] = sizes[len(report["results"]):]
    atomic_json(output, report)
    if last == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
