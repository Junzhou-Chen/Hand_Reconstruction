#!/usr/bin/env python3
"""Remote-only real-window MoE/forecast forward-backward-AdamW batch probe.

Repeated real windows exercise the full objective, not generalization. Results
exclude data loading and DDP communication. No production checkpoint changes.
Set CUDA_VISIBLE_DEVICES explicitly to one physical GPU before running.
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
    parser.add_argument("--batch-sizes", nargs="+", default=["32,64,128,256"])
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=0)
    args = parser.parse_args()
    if not Path("/sciclone/home/jchen57").is_dir():
        parser.error("Run this numerical/model benchmark remotely in the authorized ds Pod")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible not in ("0", "1"):
        parser.error("Explicitly set CUDA_VISIBLE_DEVICES=0 or 1 to select a single physical GPU")
    try:
        sizes = sorted({int(item) for group in args.batch_sizes for item in group.split(",")})
    except ValueError:
        parser.error("Batch sizes must be positive integers, comma/space separated")
    if not sizes or min(sizes) < 1 or args.iterations < 1 or args.warmup < 0:
        parser.error("Positive batch sizes/iterations and nonnegative warmup required")
    output = args.output if args.output.suffix == ".json" else args.output / "receipt.json"
    if output.exists():
        parser.error("Choose a fresh receipt path")
    output.parent.mkdir(parents=True, exist_ok=True)

    # Numerical imports follow the remote guard, never running locally.
    from train_probabilistic_twm import (
        torch, ProbabilisticTWM, ProbabilisticTWMConfig, ForecastTrajectoryWindows,
        prepare_batch, objective, atomic_json, utc, g,
    )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one visible CUDA device required")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16-capable GPU required")
    torch.set_num_threads(2)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    config = json.loads(args.config.read_text())
    if config.get("corruption_profile_path"):
        profile_path = Path(__file__).resolve().parents[1] / config["corruption_profile_path"]
        config["corruption_profile"] = json.loads(profile_path.read_text())
        config["corruption_profile_sha256"] = hashlib.sha256(profile_path.read_bytes()).hexdigest()
        if config["corruption_profile"].get("status") != "active":
            raise ValueError("Batch probe needs an explicitly active corruption profile")
    dataset = ForecastTrajectoryWindows(
        args.manifest, "train", history=config["history_frames"],
        forecast_frames=config["model"]["forecast_horizon"],
        min_future_frames=config["min_future_frames"], window_stride=config["window_stride"],
        cache_sequences=4, cache_bytes=256 * 1024**2,
    )
    selected = dataset.fixed_indices_by_dataset(1)
    if set(selected) != set(config["required_datasets"]) or any(len(ids) != 1 for ids in selected.values()):
        raise ValueError("Every configured source must supply a real train window")
    samples = [dataset[ids[0]] for ids in selected.values()]
    base = {key:torch.stack([sample[key] for sample in samples]).to(device) for key in samples[0]}
    revisions = dataset.source_revisions
    del samples, dataset

    def new_model():
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        return ProbabilisticTWM(ProbabilisticTWMConfig(**config["model"])).to(device)

    @torch.no_grad()
    def mask_invariance():
        model = new_model().eval()
        original = {key:value.clone() for key,value in base.items()}
        # Enforce a nonempty masked suffix even for the longest source clip.
        original["future_valid"][:, config["min_future_frames"]:] = False
        altered = {key:value.clone() for key,value in original.items()}
        invalid = ~altered["future_valid"]
        delta = torch.randn(*altered["future"].shape[:-1], 51, device=device)
        delta[..., :3] *= 10  # Large SI translation perturbation at invalid labels.
        changed = g.state_apply_delta(altered["future"], delta)
        altered["future"][invalid] = changed[invalid]
        outputs, losses, diagnostics, prepared = [], [], [], []
        for batch in (original, altered):
            rng = torch.Generator(device=device).manual_seed(config["seed"] + 991)
            history,raw,candidate,kwargs,current,future = prepare_batch(
                batch,rng,condition_on_beta=config.get("condition_on_beta",False),
                corruption_profile=config.get("corruption_profile"))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = model(history,raw,candidate,**kwargs)
            loss, values = objective(prediction,current,future,batch["future_valid"],batch["future_dt"],config)
            outputs.append(prediction)
            losses.append(float(loss))
            diagnostics.append({key:float(value) for key,value in values.items()})
            prepared.append((history,raw,candidate,current))
        input_error = max(float((a-b).abs().max()) for a,b in zip(*prepared))
        current_error = float((outputs[0]["state"]-outputs[1]["state"]).abs().max())
        forecast_error = float((outputs[0]["forecast_modes"]-outputs[1]["forecast_modes"]).abs().max())
        loss_error = abs(losses[0]-losses[1])
        term_error = max(abs(diagnostics[0][key]-diagnostics[1][key]) for key in diagnostics[0])
        if max(input_error,current_error,forecast_error,loss_error,term_error) > 1e-6:
            raise AssertionError(f"Masked future labels leaked into full objective: {loss_error}, {term_error}")
        return {"passed":True,"masked_labels":int(invalid.sum()),"input_max_abs":input_error,
                "current_output_max_abs":current_error,"forecast_output_max_abs":forecast_error,
                "full_objective_abs_difference":loss_error,"objective_term_max_abs":term_error}

    report = {"phase":"running","started_utc":utc(),"python":sys.executable,"torch":torch.__version__,
              "gpu":torch.cuda.get_device_name(device),"cuda_visible_devices":visible,
              "total_vram_bytes":torch.cuda.get_device_properties(device).total_memory,
              "manifest":str(args.manifest),"manifest_sha256":hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
              "config":config,"source_revisions":revisions,"base_indices_by_source":selected,
              "base_valid_future_frames":base["future_valid"].sum(-1).tolist(),
              "batch_sizes":sizes,"iterations":args.iterations,"warmup":args.warmup,
              "scope":"Full probabilistic objective on repeated real windows; excludes loading/DDP; not quality/calibration evidence",
              "mask_invariance":mask_invariance(),"results":[]}
    atomic_json(output,report)

    def probe(size):
        model = new_model().train()
        groups = [{"params":[p for p in model.parameters() if bool(getattr(p,"_no_weight_decay",False)) == exempt],
                   "weight_decay":0. if exempt else config["weight_decay"]} for exempt in (False,True)]
        optimizer = torch.optim.AdamW(groups,lr=config["learning_rate"])
        rows = torch.arange(size,device=device) % len(selected)
        batch = {key:value.index_select(0,rows) for key,value in base.items()}
        generator = torch.Generator(device=device).manual_seed(config["seed"])

        def step():
            history,raw,candidate,kwargs,current,future = prepare_batch(
                batch,generator,condition_on_beta=config.get("condition_on_beta",False),
                corruption_profile=config.get("corruption_profile"))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.bfloat16):
                prediction = model(history,raw,candidate,**kwargs)
            loss,terms = objective(prediction,current,future,batch["future_valid"],batch["future_dt"],config)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite full objective")
            loss.backward()
            bad = [name for name,p in model.named_parameters()
                   if p.grad is None or not bool(torch.isfinite(p.grad).all())]
            if bad:
                raise FloatingPointError(f"Missing/nonfinite gradients: {bad}")
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            return {"loss":float(loss.detach()),"grad_norm":float(norm),
                    "terms":{key:float(value.detach()) for key,value in terms.items()}}

        for _ in range(args.warmup): step()
        measurements = []
        for _ in range(args.iterations):
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            values = step()
            torch.cuda.synchronize(device)
            measurements.append({**values,"seconds":time.perf_counter()-started})
        tensors = list(model.parameters()) + [value for state in optimizer.state.values()
                                              for value in state.values() if torch.is_tensor(value)]
        if any(not bool(torch.isfinite(value).all()) for value in tensors if value.is_floating_point()):
            raise FloatingPointError("Nonfinite learned parameter/AdamW state")
        seconds = sum(item["seconds"] for item in measurements)
        return {"batch_size":size,"status":"passed","measurements":measurements,
                "parameters":sum(p.numel() for p in model.parameters()),
                "samples_per_second":size*args.iterations/seconds,
                "peak_allocated_bytes":torch.cuda.max_memory_allocated(device),
                "peak_reserved_bytes":torch.cuda.max_memory_reserved(device),
                "all_gradients_used_finite":True,"parameters_optimizer_finite":True}

    for size in sizes:
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        free,_ = torch.cuda.mem_get_info(device)
        try:
            result = probe(size)
        except torch.cuda.OutOfMemoryError as exc:
            result = {"batch_size":size,"status":"oom","error":str(exc),
                      "peak_allocated_bytes":torch.cuda.max_memory_allocated(device),
                      "peak_reserved_bytes":torch.cuda.max_memory_reserved(device)}
        except Exception as exc:
            result = {"batch_size":size,"status":"failed","error_type":type(exc).__name__,"error":str(exc)}
        result["free_vram_before_bytes"] = free
        report["results"].append(result)
        report["updated_utc"] = utc()
        atomic_json(output,report)
        print(json.dumps(result),flush=True)
        gc.collect(); torch.cuda.empty_cache()
        if result["status"] != "passed": break
    report.update(phase="complete" if report["results"][-1]["status"] == "passed" else "stopped",
                  completed_utc=utc(),untested_batch_sizes=sizes[len(report["results"]):])
    atomic_json(output,report)
    if report["results"][-1]["status"] == "failed": raise SystemExit(1)


if __name__ == "__main__":
    main()
