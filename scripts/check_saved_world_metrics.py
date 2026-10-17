#!/usr/bin/env python3
"""Compare the saved-prediction evaluator with frozen HaWoR metric functions.

Run this lightweight numerical check in the experiment Pod, not on the laptop.
Only named function definitions are AST-extracted from the recorded official
source; its imports, plotting dependencies and top-level code never execute.
Real geometry fixtures remain in the remote evaluation directory.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import warnings

import numpy as np
import torch


OFFICIAL_FUNCTIONS = (
    "compute_error_accel", "batch_compute_similarity_transform_torch",
    "compute_jpe", "global_align_joints", "first_align_joints", "align_pcl",
    "compute_rte",
)
METRICS = (
    "w_mpjpe_mm", "wa_mpjpe_mm", "pa_mpjpe_so3_mm",
    "accel_official_mps2", "rte_percent",
)
ACCUMULATOR_NAMES = {
    "w_mpjpe_mm": "w_mpjpe21_mm",
    "wa_mpjpe_mm": "wa_mpjpe21_mm",
    "pa_mpjpe_so3_mm": "pa_hawor_so3_mpjpe21_mm",
    "accel_official_mps2": "accel_hawor_m_s2",
    "rte_percent": "rte_percent",
}
ABS_TOLERANCE = {
    "w_mpjpe_mm": 0.003,
    "wa_mpjpe_mm": 0.003,
    "pa_mpjpe_so3_mm": 0.003,
    "accel_official_mps2": 0.0005,
    "rte_percent": 0.003,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_official_functions(path: Path) -> dict:
    tree = ast.parse(path.read_text())
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                   and node.name in OFFICIAL_FUNCTIONS]
    found = {node.name for node in definitions}
    if found != set(OFFICIAL_FUNCTIONS):
        raise ValueError(f"missing official functions: {set(OFFICIAL_FUNCTIONS) - found}")
    namespace = {"torch": torch, "np": np}
    source = ast.Module(body=definitions, type_ignores=[])
    exec(compile(source, str(path), "exec"), namespace)
    return namespace


def load_implementation(path: Path):
    spec = importlib.util.spec_from_file_location("saved_world_metrics_under_test", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import implementation {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def official_segment(reference: dict, gt_joints, pred_joints, gt_trans, pred_trans,
                     fps: float = 30.0) -> dict[str, np.ndarray]:
    """Invoke unchanged official functions with the HOT evaluator's call order."""
    gt = torch.as_tensor(gt_joints, dtype=torch.float32)
    pred = torch.as_tensor(pred_joints, dtype=torch.float32)
    target_trans = torch.as_tensor(gt_trans, dtype=torch.float32)
    predicted_trans = torch.as_tensor(pred_trans, dtype=torch.float32)
    w, wa = [], []
    for start in range(0, len(gt), 100):
        target, estimate = gt[start:start + 100], pred[start:start + 100]
        w.append(reference["compute_jpe"](
            target, reference["first_align_joints"](target, estimate)) * 1000)
        wa.append(reference["compute_jpe"](
            target, reference["global_align_joints"](target, estimate)) * 1000)
    # A singleton call avoids the published batch function's shape[0] == 2/3
    # transpose ambiguity for synthetic short segments. Original HOT inputs
    # have T >= 16; per-frame PA mathematically has no temporal coupling.
    pa = []
    for target, estimate in zip(gt, pred):
        aligned = reference["batch_compute_similarity_transform_torch"](
            estimate[None], target[None])
        pa.append(reference["compute_jpe"](target[None], aligned)[0] * 1000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        acceleration = reference["compute_error_accel"](gt, pred)[1:-1] * fps ** 2
        rte = reference["compute_rte"](target_trans, predicted_trans) * 100
    return {
        "w_mpjpe_mm": np.concatenate(w),
        "wa_mpjpe_mm": np.concatenate(wa),
        "pa_mpjpe_so3_mm": np.asarray(pa),
        "accel_official_mps2": np.asarray(acceleration),
        "rte_percent": np.asarray(rte),
    }


def implementation_segment(module, gt_joints, pred_joints, gt_trans, pred_trans,
                           fps: float = 30.0) -> dict[str, np.ndarray]:
    result = module.audit_one_segment(
        gt_joints, pred_joints, gt_trans, pred_trans, fps=fps)
    return {name: np.asarray(result[name], dtype=np.float64) for name in METRICS}


def compare_arrays(actual: np.ndarray, expected: np.ndarray, name: str) -> dict:
    same_shape = actual.shape == expected.shape
    if not same_shape:
        return {"passed": False, "actual_shape": list(actual.shape),
                "expected_shape": list(expected.shape)}
    finite_actual, finite_expected = np.isfinite(actual), np.isfinite(expected)
    finite_masks_match = bool(np.array_equal(finite_actual, finite_expected))
    # Official RTE returns NaN/inf on zero-path inputs. The production evaluator
    # may explicitly return NaN to represent undefined; neither may become zero.
    finite = finite_actual & finite_expected
    delta = np.abs(actual[finite] - expected[finite])
    maximum = float(delta.max()) if delta.size else 0.0
    atol = ABS_TOLERANCE[name]
    close = np.allclose(actual[finite], expected[finite], atol=atol, rtol=2e-5)
    return {"passed": finite_masks_match and bool(close), "count": int(actual.size),
            "finite_count": int(finite.sum()), "undefined_count": int((~finite_expected).sum()),
            "finite_masks_match": finite_masks_match, "max_abs_difference": maximum,
            "absolute_tolerance": atol, "relative_tolerance": 2e-5}


def check_accumulator(module, fixture: dict, expected: dict) -> dict:
    bucket = {}
    module.compute_run(bucket, fixture["pred_joints"], fixture["gt_joints"],
                       fixture["pred_trans"], fixture["gt_trans"], fps=30.)
    checks = {}
    for name, stored_name in ACCUMULATOR_NAMES.items():
        values = expected[name]
        values = values[np.isfinite(values)]
        stored = bucket.get("metrics", {}).get(stored_name, {}).get(
            "frame_weighted", {"count": 0, "sum": 0.0})
        count_match = int(stored["count"]) == len(values)
        difference = abs(float(stored["sum"]) / stored["count"] - float(values.mean())) if len(values) and stored["count"] else 0.0
        tolerance = ABS_TOLERANCE[name] + (float(np.abs(values).max()) * 2e-5 if len(values) else 0.0)
        checks[name] = {"passed": count_match and difference <= tolerance,
                        "count": int(stored["count"]), "expected_count": len(values),
                        "mean_absolute_difference": difference, "tolerance": tolerance}
    return checks


def synthetic_fixtures() -> list[tuple[str, dict]]:
    rng = np.random.default_rng(20261016)
    local = rng.normal(size=(21, 3)) * np.array([0.04, 0.03, 0.02])
    rotation = np.array([[0.8, -0.6, 0.0], [0.6, 0.8, 0.0], [0.0, 0.0, 1.0]])
    fixtures = []
    for length in (1, 2, 3, 5, 15, 16, 32, 101, 205):
        time = np.arange(length, dtype=np.float64) / 30
        root = np.column_stack((0.02 * time, 0.01 * np.sin(time),
                                0.5 + 0.003 * time ** 2))
        gt = local[None] + root[:, None]
        pred = 1.25 * (gt @ rotation.T) + np.array([0.07, -0.03, 0.12])
        fixtures.append((f"known_sim3_T{length}", {
            "gt_joints": gt, "pred_joints": pred, "gt_trans": root,
            "pred_trans": 1.25 * (root @ rotation.T) + [0.07, -0.03, 0.12]}))
    length = 32
    time = np.arange(length, dtype=np.float64) / 30
    root = np.column_stack((0.02 * time, 0.01 * time, 0.5 + 0.03 * time))
    gt = local[None] + root[:, None]
    acceleration = np.array([0.2, -0.1, 0.3])
    displacement = 0.5 * time[:, None] ** 2 * acceleration
    fixtures.append(("constant_acceleration", {
        "gt_joints": gt, "pred_joints": gt + displacement[:, None],
        "gt_trans": root, "pred_trans": root + displacement}))
    fixtures.append(("rigid_trajectory", {
        "gt_joints": gt, "pred_joints": gt @ rotation.T + [0.07, -0.03, 0.12],
        "gt_trans": root, "pred_trans": root @ rotation.T + [0.07, -0.03, 0.12]}))
    reflected = gt.copy()
    reflected[..., 0] *= -1
    fixtures.append(("reflection_requires_proper_rotation", {
        "gt_joints": gt, "pred_joints": reflected,
        "gt_trans": root, "pred_trans": root}))
    stationary = np.tile(np.array([0.0, 0.0, 0.5]), (32, 1))
    fixtures.append(("stationary_rte_undefined", {
        "gt_joints": local[None] + stationary[:, None],
        "pred_joints": local[None] + stationary[:, None] + [0.02, 0.01, 0],
        "gt_trans": stationary, "pred_trans": stationary + [0.02, 0.01, 0]}))
    return fixtures


def load_fixture(path: Path) -> list[tuple[str, dict]]:
    """Load segment_NNN_{gt,pred}_{joints,trans} arrays, without pickle."""
    with np.load(path, allow_pickle=False) as data:
        prefixes = sorted(key[:-len("_gt_joints")] for key in data.files
                          if key.endswith("_gt_joints"))
        if not prefixes:
            raise ValueError("fixture has no segment_*_gt_joints arrays")
        result = []
        for prefix in prefixes:
            values = {name: data[f"{prefix}_{name}"].copy()
                      for name in ("gt_joints", "pred_joints", "gt_trans", "pred_trans")}
            frame_key = next((key for key in (f"{prefix}_frame_indices", f"{prefix}_frame_ids")
                              if key in data), None)
            if frame_key:
                frames = data[frame_key]
                if len(frames) != len(values["gt_joints"]) or np.any(np.diff(frames) != 1):
                    raise ValueError(f"{prefix}: fixture segment crosses a temporal gap")
            result.append((prefix, values))
        return result


def check_boundaries(module) -> dict:
    # Gaps break temporal derivatives. A model reset between consecutive source
    # frames is a real output boundary and stays in the main metric's run.
    ids = np.array([29, 30, 31, 32, 33, 36, 37, 38, 39, 63, 64, 65])
    expected = [(0, 5), (5, 9), (9, 12)]
    actual = module.runs_from_ids(ids)
    reset_transition_is_kept = any(ids[a] <= 31 and ids[b - 1] >= 32 for a, b in actual)
    reset_transition_is_kept &= any(ids[a] <= 63 and ids[b - 1] >= 64 for a, b in actual)
    no_gap_is_crossed = all(np.all(np.diff(ids[a:b]) == 1) for a, b in actual)
    counts = {
        "standard_acceleration_centers": sum(max(0, b - a - 2) for a, b in actual),
        "official_trimmed_acceleration_centers": sum(max(0, b - a - 4) for a, b in actual),
    }
    return {"passed": actual == expected and bool(reset_transition_is_kept) and bool(no_gap_is_crossed),
            "source_frame_ids": ids.tolist(), "expected_runs": expected, "actual_runs": actual,
            "reset_boundaries_remain_in_main_metric": bool(reset_transition_is_kept),
            "no_temporal_gap_crossed": bool(no_gap_is_crossed), **counts}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    reference = load_official_functions(args.reference)
    module = load_implementation(args.implementation)
    boundaries = check_boundaries(module)
    cases = synthetic_fixtures()
    if args.fixture:
        cases.extend(load_fixture(args.fixture))
    results = []
    for name, fixture in cases:
        expected = official_segment(reference, **fixture)
        actual = implementation_segment(module, **fixture)
        checks = {key: compare_arrays(actual[key], expected[key], key) for key in METRICS}
        accumulator_checks = check_accumulator(module, fixture, expected)
        length = len(fixture["gt_joints"])
        counts_pass = all(len(expected[key]) == length
                          for key in ("w_mpjpe_mm", "wa_mpjpe_mm", "pa_mpjpe_so3_mm", "rte_percent"))
        counts_pass &= len(expected["accel_official_mps2"]) == max(0, length - 4)
        known = {}
        if name.startswith("known_sim3"):
            known["w_wa_pa_zero"] = bool(all(np.max(expected[key]) < 0.003 for key in METRICS[:3]))
        if name == "constant_acceleration":
            target = float(np.linalg.norm([0.2, -0.1, 0.3]))
            known["acceleration_matches_known_mps2"] = bool(np.allclose(
                expected["accel_official_mps2"], target, atol=0.0005, rtol=0))
        if name == "stationary_rte_undefined":
            known["all_rte_undefined"] = bool(np.all(~np.isfinite(expected["rte_percent"])))
        if name == "reflection_requires_proper_rotation":
            known["pa_does_not_accept_reflection"] = bool(np.min(expected["pa_mpjpe_so3_mm"]) > 0.1)
        passed = bool(counts_pass and all(item["passed"] for item in checks.values())
                      and all(item["passed"] for item in accumulator_checks.values()) and all(known.values()))
        results.append({"name": name, "frames": length, "passed": passed,
                        "counts_pass": bool(counts_pass), "known_value_checks": known,
                        "metrics": checks, "production_accumulator_checks": accumulator_checks})
    report = {
        "passed": boundaries["passed"] and all(item["passed"] for item in results),
        "source": "HaWoR 66c7d4108d58a716deccd192cb7645170cdc7bd7",
        "reference_path": str(args.reference.resolve()),
        "reference_sha256": sha256(args.reference),
        "implementation_sha256": sha256(args.implementation),
        "checker_sha256": sha256(Path(__file__)),
        "fixture_sha256": sha256(args.fixture) if args.fixture else None,
        "ast_extracted_functions": list(OFFICIAL_FUNCTIONS),
        "reference_dtype": "float32, unchanged official functions",
        "boundary_checks": boundaries,
        "fixture_case_count": len(results), "cases": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": report["passed"], "cases": len(results),
                      "output": str(args.output)}, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
