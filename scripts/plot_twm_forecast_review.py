#!/usr/bin/env python3
"""Plot recorded forecast validation remotely; no model or dataset execution."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


HORIZONS = (("0.1s", 0.1), ("0.5s", 0.5), ("1s", 1.0), ("2s", 2.0))
BASELINES = (
    ("map", "Highest-weight forecast", "#2867B2", "-"),
    ("restored_hold", "Restored-state hold", "#23836B", "--"),
    ("constant_velocity", "Constant velocity", "#D88324", ":"),
)


def measured(metrics, horizon, baseline, metric):
    key = f"future_{horizon}/{baseline}/{metric}"
    value = float(metrics[key])
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"Expected a finite nonnegative error: {key}={value}")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not Path("/sciclone/home/jchen57").is_dir():
        parser.error("Render this scientific figure in the remote compute Pod")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    review = json.loads(args.review.read_text())
    latest = review["latest_validation"]
    latest_step = int(latest["step"])
    latest_noisy = latest["macro"]["noisy"]
    rows = sorted(review["fixed_validation_curve"], key=lambda row: int(row["step"]))
    steps = [int(row["step"]) for row in rows]
    if not rows or len(set(steps)) != len(steps) or min(steps) < 0:
        raise ValueError("Recorded validation steps must be nonempty, unique and nonnegative")
    if max(steps) > latest_step:
        raise ValueError("Recorded curve extends beyond the stated latest validation")
    trained = [row for row in rows if int(row["step"]) > 0]
    if not trained:
        raise ValueError("At least one trained checkpoint is required for the trend panel")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titlesize": 12, "axes.labelsize": 10,
                         "legend.fontsize": 9})
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    offsets = [seconds for _, seconds in HORIZONS]
    panels = ((axes[0, 0], "wrist_mm", "A  Wrist position", "Endpoint error (mm)"),
              (axes[0, 1], "root_deg", "B  Root orientation", "Endpoint SO(3) error (degrees)"),
              (axes[1, 0], "finger_deg", "C  Finger orientations", "Mean endpoint SO(3) error (degrees)"))
    for axis, metric, title, ylabel in panels:
        for baseline, label, color, linestyle in BASELINES:
            values = [measured(latest_noisy, horizon, baseline, metric)
                      for horizon, _ in HORIZONS]
            axis.plot(offsets, values, label=label, color=color, linestyle=linestyle,
                      marker="o", markersize=5, linewidth=2)
        axis.set_title(title)
        axis.set_xlabel("Future offset (seconds)")
        axis.set_ylabel(ylabel)
        axis.set_xticks(offsets, ["0.1", "0.5", "1", "2"])
        axis.set_ylim(bottom=0)
        axis.legend(loc="best")

    axis = axes[1, 1]
    benefits = {}
    colors = ("#2867B2", "#23836B", "#D88324", "#8454AD")
    for (horizon, _), color in zip(HORIZONS, colors):
        values = []
        for row in trained:
            metrics = row["macro"]["noisy"]
            hold = measured(metrics, horizon, "restored_hold", "wrist_mm")
            forecast = measured(metrics, horizon, "map", "wrist_mm")
            if hold <= 0:
                raise ValueError(f"A positive hold error is required for relative improvement at step {row['step']}")
            values.append(100 * (hold - forecast) / hold)
        benefits[horizon] = values
        axis.plot([int(row["step"]) for row in trained], values, label=horizon,
                  color=color, marker="o", markersize=4, linewidth=1.8)
    axis.axhline(0, color="#596579", linewidth=1, linestyle="--", zorder=1)
    axis.set_title("D  Wrist prediction improvement over hold")
    axis.set_xlabel("Optimizer step (trained checkpoints only)")
    axis.set_ylabel("Wrist error reduction vs. hold (%)")
    axis.ticklabel_format(axis="x", style="plain", useOffset=False)
    axis.legend(title="Future offset", loc="best", ncol=2)

    for axis in axes.flat:
        axis.grid(color="#E4E9EF", linewidth=.7, zorder=0)
        axis.set_axisbelow(True)
    fig.suptitle(f"TWM future prediction: fixed noisy validation at step {latest_step:,}",
                 fontsize=16, fontweight="bold", y=.985)
    fig.text(.5, .070, "Fixed synthetic corruption | Equal mean of available-source means | 2 s excludes DexYCB",
             ha="center", fontsize=10, color="#46536B")
    fig.text(.5, .045, "A-C are errors at each future endpoint, not trajectory ADE; eligible windows differ across offsets.",
             ha="center", fontsize=9, color="#596579")
    fig.text(.5, .020, "Hold and constant velocity start at the restored current state; CV uses the noisy last-two-history velocity.",
             ha="center", fontsize=9, color="#596579")
    fig.tight_layout(rect=(0, .10, 1, .95), h_pad=2, w_pad=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    counts = {source: modes["noisy"].get("counts", {})
              for source, modes in latest["by_dataset"].items()}
    print(json.dumps({"output": str(args.output), "validation_step": latest_step,
                      "trained_steps": [int(row["step"]) for row in trained],
                      "wrist_improvement_percent": benefits,
                      "eligible_windows_by_source": counts,
                      "forecast_branch": "highest mixture weight for the whole trajectory; no oracle selection",
                      "point_errors_not_trajectory_ade": True}))


if __name__ == "__main__":
    main()
