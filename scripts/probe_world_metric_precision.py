#!/usr/bin/env python3
"""Remote-only diagnostic: TF32 versus MANO decode batch grouping.

Read the first eight nonempty frozen baseline prediction shards once. Decode
the same own-beta predictions in four precision/grouping configurations; use
the frozen full-test PA function for every reduction. No neural inference.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def safe(root, relative):
    root = Path(root).resolve()
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError("Path escapes expected root: " + str(relative))
    return candidate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--mano-models", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True,
                        help="Frozen original full-test code root containing src/ and scripts/.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-shards", type=int, default=8)
    args = parser.parse_args()
    if not Path("/sciclone/data10/jchen57").is_dir():
        parser.error("Numerical diagnostics must execute in the experiment Pod")
    if args.output.exists():
        raise FileExistsError("Refusing to replace existing diagnostic report")
    if not 1 <= args.max_shards <= 8:
        parser.error("This bounded probe accepts 1--8 nonempty shards")
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(variable, "2")
    sys.path.insert(0, str(args.code_root.resolve() / "src"))
    sys.path.insert(0, str(args.code_root.resolve() / "scripts"))
    import numpy as np
    import torch

    metric_module = importlib.import_module("dexycb_test_metrics")
    codec_module = importlib.import_module("handloop.visual.codec")
    torch.set_num_threads(2)
    device = torch.device(args.device)
    if device.type != "cuda":
        parser.error("TF32 diagnostic requires a CUDA device")
    torch.cuda.set_device(device)
    started = time.monotonic()
    index_path = args.source_root / "shard_index.json"
    index = read(index_path)
    if index["phase"] != "complete" or index["frames"] != 78768:
        raise ValueError("Expected the immutable completed full-test source")
    entries = [entry for entry in index["shards"] if entry["frames"]][:args.max_shards]
    data = []
    for entry in entries:
        key = entry["key"]
        manifest_path = safe(args.source_root, entry["manifest"])
        if sha(manifest_path) != entry["manifest_sha256"]:
            raise ValueError("Source manifest checksum mismatch: " + key)
        manifest = read(manifest_path)
        if len(manifest["records"]) != 1:
            raise ValueError("Expected one source recording/view per shard")
        record = manifest["records"][0]
        reference_path = safe(manifest_path.parent, record["reference_file"])
        if sha(reference_path) != record["reference_sha256"]:
            raise ValueError("Source reference checksum mismatch: " + key)
        directory = safe(args.evaluation_root / "baseline" / "shards", key)
        result = read(directory / "result.json")
        prediction_path = directory / "predictions.npz"
        if result["phase"] != "complete" or sha(prediction_path) != result["predictions_sha256"]:
            raise ValueError("Frozen baseline prediction mismatch: " + key)
        with np.load(reference_path, allow_pickle=False) as source:
            values = {name: source[name].copy() for name in ("frame_ids", "joint_3d", "handedness")}
        with np.load(prediction_path, allow_pickle=False) as predictions:
            if not np.array_equal(predictions["frame_ids"], values["frame_ids"]):
                raise ValueError("Prediction/source frame-ID mismatch: " + key)
            values["raw"] = predictions["raw"].copy()
            values["beta"] = predictions["beta"].copy()
            if not predictions["availability"].all():
                raise ValueError("Incomplete baseline availability: " + key)
        count = len(values["frame_ids"])
        frame_hash = hashlib.sha256(np.asarray(values["frame_ids"], dtype="<i8").tobytes()).hexdigest()
        if count != entry["frames"] or frame_hash != result["frame_ids_sha256"]:
            raise ValueError("Frame count/hash mismatch: " + key)
        summary = result["summary"]["all"]["branches"]["raw"]["metrics"]
        if summary["pa_mpjpe21_mm"]["count"] != count:
            raise ValueError("Prior PA denominator mismatch: " + key)
        bounds = np.r_[0, np.flatnonzero(np.diff(values["frame_ids"]) != 1) + 1, count]
        runs = [(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
        clip_ranges = [(a + offset, min(a + offset + 32, b))
                       for a, b in runs for offset in range(0, b - a, 32)]
        if len(clip_ranges) != result["clips"]:
            raise ValueError("Reconstructed reset windows differ from old inference: " + key)
        values.update(key=key, count=count, clip_ranges=clip_ranges, prior=summary,
                      manifest_sha256=entry["manifest_sha256"],
                      reference_sha256=record["reference_sha256"],
                      prediction_sha256=result["predictions_sha256"])
        data.append(values)
    codec = codec_module.MANOStateCodec(
        args.mano_models / "MANO_RIGHT.pkl", args.mano_models / "MANO_LEFT.pkl",
        left_shapedirs_fix=True).to(device).eval()
    initial_flags = {
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }
    configs = [(False, "all512"), (True, "all512"), (True, "run32"), (False, "run32")]
    outputs = []
    first_geometry, first_pa = {}, {}
    for tf32, grouping in configs:
        torch.backends.cuda.matmul.allow_tf32 = tf32
        rows = []
        total_count = 0
        totals = {"pa_mpjpe21_mm": 0.0, "pa_proper_mpjpe21_mm": 0.0,
                  "mpjpe21_mm": 0.0, "wrist_aligned_mpjpe21_mm": 0.0}
        prior_totals = {name: 0.0 for name in totals}
        max_geometry_change_mm, max_pa_change_mm = 0.0, 0.0
        for values in data:
            n, key = values["count"], values["key"]
            side = torch.as_tensor(values["handedness"].astype(bool), device=device)
            ranges = ([(a, min(a + 512, n)) for a in range(0, n, 512)]
                      if grouping == "all512" else values["clip_ranges"])
            joints_parts, pa_parts, proper_parts, absolute_parts, wrist_parts = [], [], [], [], []
            with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
                for a, b in ranges:
                    state = torch.as_tensor(values["raw"][a:b], dtype=torch.float32, device=device)
                    beta = torch.as_tensor(values["beta"][a:b], dtype=torch.float32, device=device)
                    target = torch.as_tensor(values["joint_3d"][a:b], dtype=torch.float32, device=device)
                    predicted = metric_module.dex_joints(codec.geometry(state, beta, side[a:b]), side[a:b])
                    pa, info = metric_module.pa_joint_errors_mm(predicted, target, return_info=True)
                    delta = predicted - target
                    joints_parts.append(predicted.cpu().numpy())
                    pa_parts.append(pa.mean(-1).cpu().numpy())
                    proper_parts.append(info["proper_errors_mm"].mean(-1).cpu().numpy())
                    absolute_parts.append((delta.norm(dim=-1) * 1000).double().mean(-1).cpu().numpy())
                    wrist_parts.append(((delta - delta[:, :1]).norm(dim=-1) * 1000).double().mean(-1).cpu().numpy())
            geometry = np.concatenate(joints_parts)
            metric_arrays = {
                "pa_mpjpe21_mm": np.concatenate(pa_parts),
                "pa_proper_mpjpe21_mm": np.concatenate(proper_parts),
                "mpjpe21_mm": np.concatenate(absolute_parts),
                "wrist_aligned_mpjpe21_mm": np.concatenate(wrist_parts),
            }
            if key not in first_geometry:
                first_geometry[key], first_pa[key] = geometry, metric_arrays["pa_mpjpe21_mm"]
            geometry_change = float(np.linalg.norm(geometry.astype(np.float64) - first_geometry[key], axis=-1).max() * 1000)
            pa_change = float(np.abs(metric_arrays["pa_mpjpe21_mm"] - first_pa[key]).max())
            max_geometry_change_mm = max(max_geometry_change_mm, geometry_change)
            max_pa_change_mm = max(max_pa_change_mm, pa_change)
            row = {"key": key, "frames": n, "decode_batches": len(ranges),
                   "max_per_joint_geometry_change_from_false512_mm": geometry_change,
                   "max_per_frame_pa_change_from_false512_mm": pa_change, "metrics": {}}
            for name, array in metric_arrays.items():
                actual = float(array.mean())
                prior = values["prior"][name]
                if prior["count"] != n:
                    raise ValueError("Prior denominator differs: " + key + "/" + name)
                totals[name] += float(array.sum(dtype=np.float64))
                prior_totals[name] += float(prior["mean"]) * n
                row["metrics"][name] = {"recomputed": actual, "prior": prior["mean"],
                                         "difference_mm": actual - prior["mean"]}
            total_count += n
            rows.append(row)
        summary = {
            "allow_tf32": tf32, "grouping": grouping, "frames": total_count,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "max_per_joint_geometry_change_from_false512_mm": max_geometry_change_mm,
            "max_per_frame_pa_change_from_false512_mm": max_pa_change_mm,
            "metrics": {name: {"recomputed": total / total_count,
                               "prior": prior_totals[name] / total_count,
                               "difference_mm": (total - prior_totals[name]) / total_count,
                               "max_abs_per_shard_difference_mm": max(abs(row["metrics"][name]["difference_mm"]) for row in rows)}
                        for name, total in totals.items()},
            "shards": rows,
        }
        outputs.append(summary)
        print(json.dumps({key: value for key, value in summary.items() if key != "shards"}), flush=True)
    torch.backends.cuda.matmul.allow_tf32 = initial_flags["matmul_allow_tf32"]
    report = {
        "schema": "handloop.geometry_precision_probe.v1", "phase": "complete",
        "frames": sum(value["count"] for value in data), "shards": len(data),
        "precision_scope": "FP32 MANO decode, frozen original double-precision PA implementation; own saved predicted beta.",
        "original_geometry_schedule": "Each gap-free run independently divided into <=32-frame clips; metric batch64 never merges clips.",
        "baseline_for_per_frame_changes": "allow_tf32=False, all512; no original per-frame geometry was saved.",
        "initial_flags": initial_flags, "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda, "gpu": torch.cuda.get_device_name(device),
        "environment": {key: os.environ.get(key) for key in ("NVIDIA_TF32_OVERRIDE", "CUBLAS_WORKSPACE_CONFIG")},
        "source_index_sha256": sha(index_path), "script_sha256": sha(__file__),
        "metric_module": {"path": metric_module.__file__, "sha256": sha(metric_module.__file__)},
        "codec_module": {"path": codec_module.__file__, "sha256": sha(codec_module.__file__)},
        "mano_assets_sha256": {side: sha(args.mano_models / f"MANO_{side.upper()}.pkl") for side in ("left", "right")},
        "source_shards": [{key: value[key] for key in ("key", "count", "manifest_sha256", "reference_sha256", "prediction_sha256")} for value in data],
        "configurations": outputs, "elapsed_seconds": time.monotonic() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "frames": report["frames"],
                      "elapsed_seconds": report["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
