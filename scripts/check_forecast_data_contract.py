#!/usr/bin/env python3
"""Remote-only checks for masked long-horizon trajectory targets.

Synthetic contract checks run first. Optionally audit a prepared collection's
timestamp-only coverage and a small, fixed sample from every required source.
This does not preprocess datasets, run a model, or claim predictive accuracy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from handloop.data.forecast_trajectory import ForecastTrajectoryWindows
from handloop.data.trajectory import SourceBalancedSampler, TrajectoryWindows


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check_sample(sample: dict, frames: int, minimum: int) -> dict:
    future, valid, dt = sample["future"], sample["future_valid"], sample["future_dt"]
    require(future.shape == (frames, 99), "future shape is not [F,99]")
    require(valid.shape == (frames,) and valid.dtype == torch.bool, "invalid future validity mask")
    count = int(valid.sum())
    require(count >= minimum, "window lacks its minimum future target")
    require(torch.equal(valid, torch.arange(frames) < count), "valid futures do not form a prefix")
    require(bool(torch.isfinite(future).all()), "non-finite future target")
    require(bool(torch.isfinite(dt).all() and (dt > 0).all() and (dt.diff() > 0).all()),
            "future query offsets are not finite, positive and increasing")
    require(sample["future_gt_mask"].shape == (frames, 17), "future GT mask shape")
    require(torch.equal(sample["future_gt_mask"], valid[:, None].expand(-1, 17)),
            "GT mask fails to exclude padded targets")
    require(bool(sample["past_current_gt_mask"].all()), "current/history GT contract changed")
    require(torch.equal(future, sample["future_states"]) and torch.equal(dt, sample["forecast_dt"]),
            "future aliases disagree")
    if count < frames:
        identity = torch.tensor([1., 0., 0., 0., 1., 0.]).repeat(16)
        require(bool((future[~valid, :3] == 0).all()), "padded translation is not zero")
        require(torch.equal(future[~valid, 3:], identity.expand(frames - count, -1)),
                "padded rotations are not identity")
    return {"valid_frames": count, "last_observed_seconds": float(dt[count - 1])}


def write_fixture(root: Path, future_change: float = 0., *, partial: bool = False) -> Path:
    root.mkdir(parents=True)
    frames = 20
    state = np.zeros((frames, 99), dtype=np.float32)
    state[:, 3:] = np.tile(np.asarray([1, 0, 0, 0, 1, 0], dtype=np.float32), 16)
    state[:, 0] = np.arange(frames, dtype=np.float32) / 100
    # First center is 4; changing future labels must leave that input unchanged.
    state[5:, 1] += future_change
    times = np.arange(frames, dtype=np.float64) / 30
    times[10:] += .5
    mask = np.ones((frames, 17), dtype=np.bool_)
    if partial:
        mask[5, 3] = False
    np.savez(root / "sequence.npz", state=state, timestamps=times, gt_mask=mask)
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"dataset": "fixture", "nominal_dt": 1 / 30,
        "rotation6d": "columns_c0xyz_c1xyz", "sequences": [{"path": "sequence.npz",
        "split": "train", "recording": "fixture_1", "handedness": 0, "frames": frames}]}))
    return manifest


def synthetic_contract() -> dict:
    with tempfile.TemporaryDirectory(prefix="handloop_forecast_contract_") as directory:
        root = Path(directory)
        first = write_fixture(root / "original")
        data = ForecastTrajectoryWindows(first, "train", history=4, forecast_frames=8,
                                         min_future_frames=3, window_stride=1)
        short = TrajectoryWindows(first, "train", history=4, forecast_offset=3, window_stride=1)
        require(len(data) == len(short) == 6, "minimum-offset window population changed")
        require(not data._cache and not data._validated_sequences,
                "initialization eagerly loaded sequence states")
        require(data.sequences[0]["centers"][:] == [4, 5, 6, 14, 15, 16],
                "history or minimum future crosses a timestamp gap")
        counts = []
        for index in range(len(data)):
            sample = data[index]
            require(torch.equal(sample["past_current"], short[index]["past_current"]),
                    "history/current changed from the original window contract")
            counts.append(check_sample(sample, 8, 3)["valid_frames"])
        require(counts == [5, 4, 3, 5, 4, 3], "futures crossed a gap or sequence boundary")
        require(data.future_coverage["fixture"]["valid_windows_by_step"] == [6, 6, 6, 4, 2, 0, 0, 0],
                "timestamp-only future coverage counts disagree with target masks")
        require(torch.equal(data[-1]["future"], data[len(data) - 1]["future"]),
                "negative indexing differs")
        changed = ForecastTrajectoryWindows(write_fixture(root / "changed", future_change=1.),
                                            "train", history=4, forecast_frames=8,
                                            min_future_frames=3, window_stride=1)
        original_sample, changed_sample = data[0], changed[0]
        for key in ("past_current", "dt", "handedness", "beta", "beta_valid",
                    "dataset_id", "past_current_gt_mask", "future_dt", "future_valid"):
            require(torch.equal(original_sample[key], changed_sample[key]),
                    f"future target contents contaminated input/metadata {key}")
        require(not torch.equal(original_sample["future"], changed_sample["future"]),
                "future target perturbation check is vacuous")
        try:
            ForecastTrajectoryWindows(write_fixture(root / "partial", partial=True), "train",
                                      history=4, forecast_frames=8, min_future_frames=3)
        except ValueError as error:
            require("partial GT" in str(error), "partial GT rejected for an unexpected reason")
        else:
            raise AssertionError("partial MANO GT entered a full-state trainer")
        sampler = SourceBalancedSampler(data, num_samples=12, seed=17)
        require(len(list(sampler)) == 12, "source-balanced sampler is incompatible")
        return {"passed": True, "windows": len(data), "valid_future_counts": counts,
                "checks": ["original_window_population", "metadata_only_initialization",
                           "gap_and_tail_masking", "identity_placeholders", "physical_query_times",
                           "coverage_matches_masks", "future_targets_do_not_change_inputs",
                           "partial_GT_rejection", "source_sampler_compatibility"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--forecast-frames", type=int, default=60)
    parser.add_argument("--history-frames", type=int, default=30)
    parser.add_argument("--min-future-frames", type=int, default=3)
    parser.add_argument("--window-stride", type=int, default=4)
    parser.add_argument("--samples-per-source", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not Path("/sciclone/home/jchen57").is_dir():
        raise RuntimeError("Run numerical/data checks on the remote ds Pod, never the local workstation")
    torch.set_num_threads(2)
    result = {"synthetic": synthetic_contract()}
    if args.manifest:
        result["collection"] = {}
        for split in ("train", "val"):
            data = ForecastTrajectoryWindows(args.manifest, split, args.history_frames,
                    args.forecast_frames, args.min_future_frames, args.window_stride)
            require(not data._cache and not data._validated_sequences,
                    "collection initialization eagerly loaded sequence states")
            samples = {}
            for name, indices in data.fixed_indices_by_dataset(args.samples_per_source).items():
                require(bool(indices), f"source has no {split} windows: {name}")
                samples[name] = [check_sample(data[index], args.forecast_frames, args.min_future_frames)
                                 for index in indices]
            result["collection"][split] = {"contract": data.forecast_contract,
                                             "fixed_sample_checks": samples}
    result["passed"] = True
    payload = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
