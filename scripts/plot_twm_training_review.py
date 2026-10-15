#!/usr/bin/env python3
"""Render recorded TWM validation curves remotely; no model/data execution."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


SELECTION_WEIGHTS = {"clean": .20, "reanchor": .20, "noisy": .35,
                     "missing": .20, "strong": .05}


def measured(row, mode, key):
    value = float(row[mode][key])
    if not math.isfinite(value):
        raise ValueError(f"Nonfinite measurement at step {row['step']}: {mode}/{key}")
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
    rows = sorted(review["fixed_validation_curve"], key=lambda row: int(row["step"]))
    if not rows:
        raise ValueError("No recorded fixed validation measurements")
    steps = [int(row["step"]) for row in rows]
    if len(set(steps)) != len(steps) or min(steps) < 0:
        raise ValueError("Validation steps must be unique and nonnegative")
    # Recompute the current checkpoint score.  A running minimum conceals
    # regressions and must never be substituted for the measured curve.
    scores = [sum(weight * measured(row, mode, "restoration_loss")
                  for mode, weight in SELECTION_WEIGHTS.items()) for row in rows]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titlesize": 12, "axes.labelsize": 10,
                         "legend.fontsize": 9})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    blue, orange, green, purple = "#2867B2", "#D88324", "#23836B", "#8454AD"

    def curve(ax, values, label, color, linestyle="-"):
        trained = [index for index, step in enumerate(steps) if step > 0]
        if trained:
            ax.plot([steps[index] for index in trained], [values[index] for index in trained],
                    label=label, color=color, linestyle=linestyle,
                    marker="o", markersize=4, linewidth=1.8)
        if 0 in steps:
            index = steps.index(0)
            ax.scatter([0], [values[index]], marker="s", s=48, edgecolors=color,
                       facecolors="none", linewidths=1.5,
                       label=label if not trained else None, zorder=4)

    ax = axes[0, 0]
    curve(ax, scores, "Current checkpoint score", blue)
    ax.set_title("A  Weighted restoration score")
    ax.set_ylabel("Geometric loss (dimensionless)")
    ax.text(.03, .06, "0.20 clean + 0.20 reanchor + 0.35 noisy\n+ 0.20 missing + 0.05 strong",
            transform=ax.transAxes, va="bottom", fontsize=8, color="#46536B")
    ax.legend(loc="best")

    ax = axes[0, 1]
    curve(ax, [measured(row, "noisy", "candidate/wrist_mm") for row in rows],
          "Current candidate input", orange)
    curve(ax, [measured(row, "noisy", "restored/wrist_mm") for row in rows],
          "Restored current state", blue)
    ax.set_title("B  Noisy-condition wrist position")
    ax.set_ylabel("Mean wrist error (mm)")
    ax.legend(loc="best")

    ax = axes[1, 0]
    curve(ax, [measured(row, "noisy", "restored/root_deg") for row in rows],
          "Restored root rotation", blue)
    curve(ax, [measured(row, "noisy", "restored/finger_deg") for row in rows],
          "Restored finger rotations", green)
    ax.set_title("C  Noisy-condition orientation")
    ax.set_ylabel("Mean SO(3) error (degrees)")
    ax.legend(loc="best")

    ax = axes[1, 1]
    right = ax.twinx()
    right.spines["right"].set_visible(True)
    for mode, label, color in (("clean", "Clean history", green),
                               ("reanchor", "Noisy history", purple)):
        degradation = [measured(row, mode, "restored/wrist_mm")
                       - measured(row, mode, "candidate/wrist_mm") for row in rows]
        curve(ax, degradation, label + ": mean added error", color)
        probabilities = [measured(row, mode, "wrist_degradation_over_1mm") for row in rows]
        if any(value < 0 or value > 1 for value in probabilities):
            raise ValueError("wrist_degradation_over_1mm must be a fraction in [0,1]")
        curve(right, [100 * value for value in probabilities],
              label + ": >1 mm worsening", color, "--")
    ax.set_title("D  Preserving accurate current inputs")
    ax.set_ylabel("Added wrist error vs. candidate (mm)")
    right.set_ylabel("Windows worsened by >1 mm (%)")
    right.set_ylim(-2, 102)
    handles, labels = ax.get_legend_handles_labels()
    extra_handles, extra_labels = right.get_legend_handles_labels()
    ax.legend(handles + extra_handles, labels + extra_labels, loc="best", fontsize=8)

    for axis in axes.flat:
        axis.set_xlabel("Optimizer step")
        axis.grid(color="#E4E9EF", linewidth=.7, zorder=0)
        axis.set_axisbelow(True)
        if axis is not axes[1, 1]:
            axis.set_ylim(bottom=0)
        axis.ticklabel_format(axis="x", style="plain", useOffset=False)
    fig.suptitle("TWM training review: fixed synthetic validation", fontsize=16,
                 fontweight="bold", y=.985)
    fig.text(.5, .045, "Equal mean of four source means | Source-fitted MANO wrist and local rotations",
             ha="center", fontsize=10, color="#46536B")
    fig.text(.5, .022, "Hollow squares: step 0 copy-input initialization (untrained). Lines connect recorded trained checkpoints only.",
             ha="center", fontsize=9, color="#596579")
    fig.text(.5, .002, "Fixed held-out windows with synthetic corruption; these curves do not measure real HaWoR correction or independent sensor accuracy.",
             ha="center", fontsize=8, color="#596579")
    fig.tight_layout(rect=(0, .075, 1, .95), h_pad=2.0, w_pad=2.0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(json.dumps({"output": str(args.output), "recorded_steps": steps,
                      "actual_weighted_restoration_scores": scores,
                      "best_so_far_used_as_curve": False}))


if __name__ == "__main__":
    main()
