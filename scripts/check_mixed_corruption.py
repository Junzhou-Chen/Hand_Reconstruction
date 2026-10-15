#!/usr/bin/env python3
"""Remote-only semantic checks of the D31 mixed trajectory corruption.

No datasets or models are loaded. Measurements use synthetic, legal 99D
trajectories and the actual returned states, not claimed noise amplitudes.
These checks establish preparation semantics, not reconstruction accuracy.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch

from handloop import geometry as g
from handloop.data.corruption import sample_mixed_batch


CASE_NAMES = ("clean_motion", "current_reanchors_history", "ordinary_restoration",
              "partial_missing", "rare_local_failure")
LEVEL_NAMES = ("clean", "near", "light", "medium", "strong")


def require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def clone_tree(value):
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, dict):
        return {key: clone_tree(item) for key, item in value.items()}
    return copy.deepcopy(value)


def same_tree(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(
            same_tree(a[key], b[key]) for key in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(
            same_tree(left, right) for left, right in zip(a, b))
    return a == b


def finite_tree(value):
    if isinstance(value, torch.Tensor):
        require(torch.isfinite(value).all(), "nonfinite preparation output")
    elif isinstance(value, dict):
        for item in value.values():
            finite_tree(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            finite_tree(item)


def synthetic_batch(count, history=8, future=6):
    """Moving nonidentity world states, uneven dt, varying left/right and beta."""
    row = torch.arange(count, dtype=torch.float32)
    phase = row[:, None] * .127
    dt = (1 / 30) * (1 + .15 * torch.sin(phase + torch.arange(history + 1) * .71))
    past_time = dt.cumsum(-1)
    past_time = past_time - past_time[:, -1:]
    forecast_dt = torch.arange(1, future + 1)[None].expand(count, -1) / 30
    times = torch.cat((past_time, forecast_dt), -1)
    p = torch.stack((.3 + .03 * row[:, None] / max(count, 1) + .11 * times,
                     -.2 + .04 * torch.sin(times * 4 + phase),
                     .7 + .02 * torch.cos(times * 3 - phase)), -1)
    axis = torch.tensor([.4, -.2, .5])
    axis = axis / axis.norm()
    root_angle = .6 + times * .8 + phase * .1
    root = g.axis_angle_to_matrix(root_angle[..., None] * axis)
    slot = torch.arange(15)[None, None]
    joint_angle = .25 * torch.sin(times[..., None] * 3 + phase[..., None] + slot * .3)
    joint_axis = torch.tensor([.2, .9, -.3])
    joint_axis = joint_axis / joint_axis.norm()
    fingers = g.axis_angle_to_matrix(joint_angle[..., None] * joint_axis)
    state = g.state_from_rotmat(p, torch.cat((root[..., None, :, :], fingers), -3))
    return {"past_current": state[:, :history + 1], "future": state[:, history + 1:],
            "dt": dt, "forecast_dt": forecast_dt,
            "beta": torch.sin(row[:, None] + torch.arange(10)[None]) * .3,
            "beta_valid": torch.ones(count, dtype=torch.bool),
            "handedness": row.long() % 2,
            "past_current_gt_mask": torch.ones(count, history + 1, 17, dtype=torch.bool),
            "future_gt_mask": torch.ones(count, future, 17, dtype=torch.bool),
            "future_valid": torch.ones(count, future, dtype=torch.bool)}


def sample(batch, profile, seed=71231, mode="mixed"):
    return sample_mixed_batch(batch, torch.Generator().manual_seed(seed), mode,
                              condition_on_beta=False, profile=profile,
                              return_diagnostics=True)


def recover_world(prepared, batch):
    """Infer the shared transform from GT current and returned target only."""
    history, raw, candidate, kwargs, target, future = prepared
    true_current = batch["past_current"][:, -1]
    rotation = g.state_to_rotmat(true_current)[:, 0] @ g.state_to_rotmat(target)[:, 0].transpose(-1, -2)
    origin = true_current[:, :3] - (rotation @ target[:, :3, None]).squeeze(-1)

    def unanchor(value):
        leading = value.ndim - 2
        ra = rotation.reshape(len(value), *([1] * leading), 3, 3)
        pa = origin.reshape(len(value), *([1] * leading), 3)
        position = (ra @ value[..., :3, None]).squeeze(-1) + pa
        rotations = g.state_to_rotmat(value).clone()
        rotations[..., 0, :, :] = ra @ rotations[..., 0, :, :]
        return g.state_from_rotmat(position, rotations)

    anchor = g.state_from_rotmat(origin, torch.cat((rotation[:, None],
                 g.state_to_rotmat(true_current)[:, 1:]), 1))
    return unanchor(history), unanchor(raw), unanchor(candidate), unanchor(target), unanchor(future), anchor


def group_errors(prediction, target):
    xyz = (prediction[..., :3] - target[..., :3]).norm(dim=-1, keepdim=True)
    angles = g.geodesic_angle(g.state_to_rotmat(prediction), g.state_to_rotmat(target))
    return torch.cat((xyz, angles), -1)


def assert_states_close(a, b, message, tolerance=8e-6):
    require((a[..., :3] - b[..., :3]).abs().max() < tolerance, message + " (position)")
    require((g.state_to_rotmat(a) - g.state_to_rotmat(b)).abs().max() < tolerance,
            message + " (rotation)")


def check_so3(state):
    # Check serialized columns BEFORE Gram--Schmidt can conceal illegal inputs.
    columns = state[..., 3:].reshape(*state.shape[:-1], 16, 2, 3)
    require((columns.square().sum(-1) - 1).abs().max() < 3e-5, "6D columns not unit length")
    require((columns[..., 0, :] * columns[..., 1, :]).sum(-1).abs().max() < 3e-5,
            "6D columns not orthogonal")
    matrices = g.state_to_rotmat(state)
    require((torch.linalg.det(matrices) - 1).abs().max() < 3e-5, "SO3 determinant not one")


def assert_amplitudes(errors, level_ids, profile, label):
    if errors.ndim == 3 and level_ids.ndim == 2:
        level_ids = level_ids[:, None].expand_as(errors)
    bounds = []
    for name in LEVEL_NAMES:
        level = profile["levels"][name]
        bounds.append([level["translation_norm_mm"], level["root_angle_deg"],
                       level["finger_angle_deg"]])
    table = torch.tensor(bounds, dtype=errors.dtype)
    group_type = torch.tensor([0, 1] + [2] * 15)
    lower = table[level_ids, group_type, 0]
    upper = table[level_ids, group_type, 1]
    scale = errors.new_tensor([1000.] + [180 / math.pi] * 16)
    measured = errors * scale
    tolerance = errors.new_tensor([.01] + [.003] * 16)
    require(((measured >= lower - tolerance) & (measured <= upper + tolerance)).all(),
            label + ": measured geometric corruption outside selected level bounds")
    return {"max_wrist_mm": float(measured[..., 0].max()),
            "max_root_deg": float(measured[..., 1].max()),
            "max_finger_deg": float(measured[..., 2:].max())}


def check_semantics(batch, prepared, diagnostic, profile):
    finite_tree((prepared, diagnostic))
    history, raw, candidate, kwargs, target, future = prepared
    for state in (history, raw, candidate, target, future):
        check_so3(state)
    h_world, raw_world, candidate_world, target_world, future_world, anchor = recover_world(prepared, batch)
    assert_states_close(target_world, batch["past_current"][:, -1], "current GT altered")
    assert_states_close(future_world, batch["future"], "future GT altered")
    # Only root and wrist define the frame; fingers retain their actual local state.
    require(history[:, -1, :3].abs().max() < 3e-6, "anchor wrist is not zero")
    require((g.state_to_rotmat(history[:, -1])[:, 0] - torch.eye(3)).abs().max() < 3e-6,
            "anchor root is not identity")
    assert_states_close(h_world, diagnostic["history_world"], "history world/anchor inconsistency")
    require((anchor[:, :3] - diagnostic["anchor_world"][:, :3]).abs().max() < 3e-6,
            "anchor origin differs from corrupted last history")
    require((g.state_to_rotmat(anchor)[:, 0] - g.state_to_rotmat(diagnostic["anchor_world"])[:, 0]).abs().max() < 3e-6,
            "anchor rotation differs from corrupted last history")
    availability = kwargs["availability"]
    require(torch.equal(availability, diagnostic["availability"]), "availability diagnostic mismatch")
    require(availability.dtype == torch.bool and kwargs["candidate_availability"].all(),
            "raw/candidate availability conflated")
    quality = kwargs["quality"]
    require((quality[availability] == -1).all(), "available quality should be unknown, not oracle")
    require((quality[~availability] == 0).all(), "missing quality should be zero")
    require(kwargs["history_quality"] is None or (kwargs["history_quality"] == -1).all(),
            "history quality is not unknown")
    require(not kwargs["beta_valid"].any(), "beta conditioning unexpectedly enabled")
    require(torch.equal(kwargs["beta"], batch["beta"]), "stored source beta changed")
    require(torch.equal(kwargs["dt"], batch["dt"]), "dt not preserved")
    require(torch.equal(kwargs["forecast_dt"], batch["forecast_dt"]), "forecast_dt not preserved")
    require(torch.equal(kwargs["handedness"], batch["handedness"]), "handedness altered")
    raw_error = group_errors(raw_world, diagnostic["raw_world"])
    require((raw_error[availability] < 8e-6).all(), "available raw state differs from corrupt world observation")
    require((raw[:, :3][~availability[:, 0]] == 0).all(), "missing wrist raw placeholder is not zero")
    missing_rotations = raw[:, 3:].reshape(-1, 16, 6)[~availability[:, 1:]]
    require(torch.equal(missing_rotations, g.identity_rotation_6d().expand_as(missing_rotations)),
            "missing raw rotation placeholder is not identity")
    observed_candidate_error = group_errors(candidate_world, diagnostic["raw_world"])
    require((observed_candidate_error[availability] < 8e-6).all(), "available candidate differs from raw")
    held_error = group_errors(candidate, history[:, -1])
    require((held_error[~availability] < 8e-6).all(), "missing candidate not filled causally from last history")
    history_measure = assert_amplitudes(group_errors(h_world, batch["past_current"][:, :-1]),
        diagnostic["history_group_level"], profile, "history")
    current_measure = assert_amplitudes(group_errors(diagnostic["raw_world"], batch["past_current"][:, -1]),
        diagnostic["current_group_level"], profile, "current")
    case = diagnostic["case_id"]
    require(availability[case != 3].all(), "missing observations outside partial_missing case")
    missing_kind = diagnostic["missing_case_id"]
    require((missing_kind[case != 3] == -1).all(), "missing subtype outside partial_missing case")
    for kind in range(3):
        selected_missing = ~availability[(case == 3) & (missing_kind == kind)]
        if not len(selected_missing):
            continue
        count = selected_missing.sum(-1)
        if kind == 0:
            require(not selected_missing[:, :2].any(), "whole-finger missing also removes wrist")
            fingers = selected_missing[:, 2:].reshape(-1, 5, 3)
            require((count == 3).all() and (fingers.all(-1).sum(-1) == 1).all(),
                    "whole-finger missing does not remove exactly one complete finger")
        elif kind == 1:
            require(((count >= 1) & (count <= 3)).all(), "group dropout must remove one to three groups")
        else:
            require(selected_missing.all(), "whole-current missing retained a group")
    clean = case == 0
    if clean.any():
        assert_states_close(h_world[clean], batch["past_current"][clean, :-1], "clean history altered")
        assert_states_close(raw_world[clean], batch["past_current"][clean, -1], "clean current altered")
    # The D28 failure condition must be explicitly represented in training.
    reanchor = case == 1
    if reanchor.any():
        reanchor_error = group_errors(raw_world[reanchor], batch["past_current"][reanchor, -1])
        require((reanchor_error[:, 0] <= .00101).all(), "reanchor current has excessive position corruption")
        require((reanchor_error[:, 1:] <= math.radians(1.003)).all(), "reanchor current not near accurate")
    strong = diagnostic["strong_affected_groups"]
    require(not strong[case != 4].any(), "strong override outside rare local failure")
    require(torch.equal(strong, diagnostic["current_group_level"] == LEVEL_NAMES.index("strong")),
            "strong affected groups and actual selected levels disagree")
    if (case == 4).any():
        selected = strong[case == 4]
        wrist = selected[:, :2].all(-1)
        require(not selected[wrist, 2:].any(), "strong wrist override also corrupted fingers strongly")
        finger = selected[~wrist]
        require(not finger[:, :2].any(), "strong finger override also corrupted wrist strongly")
        whole_fingers = finger[:, 2:].reshape(-1, 5, 3)
        require((whole_fingers.any(-1) == whole_fingers.all(-1)).all(), "strong override splits selected finger")
        require(((whole_fingers.any(-1).sum(-1) >= 1) & (whole_fingers.any(-1).sum(-1) <= 2)).all(),
                "strong finger override must affect one or two fingers")
    return {"history": history_measure, "current": current_measure}


def check_repeat_and_no_leak(profile):
    batch = synthetic_batch(96)
    source = clone_tree(batch)
    first, diagnostic = sample(batch, profile)
    require(same_tree(batch, source), "preparation mutated source GT or metadata")
    repeat, repeated_diagnostic = sample(batch, profile)
    require(same_tree(first, repeat) and same_tree(diagnostic, repeated_diagnostic),
            "same seed/input did not reproduce exactly")
    original_interface = sample_mixed_batch(batch, torch.Generator().manual_seed(71231),
        "mixed", condition_on_beta=False, profile=profile)
    require(len(original_interface) == 6 and same_tree(first, original_interface),
            "default API no longer returns the original six-item preparation")
    altered = clone_tree(batch)
    delta = torch.zeros(*altered["future"].shape[:-1], 51)
    delta[..., :3] = .4
    delta[..., 3:] = .2
    altered["future"] = g.state_apply_delta(altered["future"], delta)
    changed, _ = sample(altered, profile)
    require(same_tree(first[:5], changed[:5]), "future label leaked into current inputs or target")
    require(not torch.equal(first[5], changed[5]), "future leakage intervention was vacuous")
    # Current labels cannot become part of historical corruption or frame choice.
    altered = clone_tree(batch)
    delta = torch.zeros(len(batch["past_current"]), 51)
    delta[:, :3] = .2
    delta[:, 3:6] = .3
    altered["past_current"][:, -1] = g.state_apply_delta(altered["past_current"][:, -1], delta)
    changed, _ = sample(altered, profile)
    require(torch.equal(first[0], changed[0]), "current state contaminated completed history")
    return {"same_seed_exact": True, "source_unchanged": True,
            "future_labels_not_inputs": True, "history_not_current_dependent": True}


def direction_correlation(states, reference):
    displacement = states[..., :3] - reference[..., :3]
    direction = displacement / displacement.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return float((direction[:, 1:] * direction[:, :-1]).sum(-1).mean())


def check_physical_time(profile):
    selected = copy.deepcopy(profile)
    selected["temporal_noise"]["affected_window_modes"] = {"independent": 0., "correlated": 1.}
    batch = synthetic_batch(1024)
    short = clone_tree(batch)
    long = clone_tree(batch)
    short["dt"].fill_(.001)
    long["dt"].fill_(1.)
    first, _ = sample(short, selected, mode="medium")
    second, _ = sample(long, selected, mode="medium")
    first_world = recover_world(first, short)[0]
    second_world = recover_world(second, long)[0]
    tiny_dt = direction_correlation(first_world, short["past_current"][:, :-1])
    large_dt = direction_correlation(second_world, long["past_current"][:, :-1])
    require(tiny_dt > large_dt + .35, "correlated corruption does not respond to physical dt")
    require(tiny_dt > .75 and abs(large_dt) < .15,
            "unexpected correlation measured from actual displacement directions")
    selected["temporal_noise"]["affected_window_modes"] = {"independent": 1., "correlated": 0.}
    independent_a, _ = sample(short, selected, mode="medium")
    independent_b, _ = sample(long, selected, mode="medium")
    require(torch.equal(independent_a[0], independent_b[0]) and torch.equal(independent_a[1], independent_b[1]),
            "independent corruption unexpectedly depends on dt")
    return {"actual_direction_lag1_at_1ms": tiny_dt,
            "actual_direction_lag1_at_1s": large_dt,
            "independent_noise_unchanged_except_dt_metadata": True}


def check_forced_modes(profile):
    batch = synthetic_batch(128)
    result = {}
    modes = ("clean", "reanchor", "light_clean_history", "light_noisy_history",
             "medium", "strong", "missing", "history_only", "noisy")
    for mode in modes:
        prepared, diagnostic = sample(batch, profile, mode=mode)
        finite_tree(prepared)
        if mode != "history_only":
            check_semantics(batch, prepared, diagnostic, profile)
        else:
            history, _, candidate, kwargs, _, _ = prepared
            require(not kwargs["availability"].any(), "history-only retained current observation")
            assert_states_close(candidate, history[:, -1], "history-only candidate not last history")
        if mode == "reanchor":
            restored_world = recover_world(prepared, batch)[1]
            assert_states_close(restored_world, batch["past_current"][:, -1],
                                "forced accurate-current/noisy-history is not accurate")
        result[mode] = True
    ordinary, _ = sample(batch, profile, mode="noisy")
    removed, _ = sample(batch, profile, mode="history_only")
    require(torch.equal(ordinary[0], removed[0]), "current-removal control changed history")
    require(torch.equal(ordinary[4], removed[4]) and torch.equal(ordinary[5], removed[5]),
            "current-removal control changed anchor or supervision")
    result["noisy_vs_history_only_paired_exactly"] = True
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=Path(__file__).resolve().parents[1]
                        / "config/twm_corruption_mixed_v2_proposal.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--chunk", type=int, default=512)
    args = parser.parse_args()
    if not Path("/sciclone/home/jchen57").is_dir():
        parser.error("Run this Torch check in the remote compute Pod, not locally")
    if args.samples < 2048 or args.chunk < 1:
        parser.error("Use at least 2048 mixture samples and a positive chunk size")
    torch.set_num_threads(2)
    profile = json.loads(args.profile.read_text())
    report = {"passed": False, "utc": datetime.now(timezone.utc).isoformat(),
              "torch": torch.__version__, "device": "cpu", "samples": args.samples,
              "profile": str(args.profile),
              "profile_sha256": hashlib.sha256(args.profile.read_bytes()).hexdigest(),
              "scope": "synthetic corruption semantics only; no accuracy claim"}
    try:
        report["determinism_and_leakage"] = check_repeat_and_no_leak(profile)
        counts = torch.zeros(5, dtype=torch.long)
        correlated = 0
        extrema = {"history": {}, "current": {}}
        for offset in range(0, args.samples, args.chunk):
            batch = synthetic_batch(min(args.chunk, args.samples - offset))
            unchanged = clone_tree(batch)
            prepared, diagnostic = sample(batch, profile, seed=81217 + offset)
            measured = check_semantics(batch, prepared, diagnostic, profile)
            require(same_tree(batch, unchanged), "mixture preparation mutated source batch")
            counts += torch.bincount(diagnostic["case_id"], minlength=5)
            correlated += int(diagnostic["correlated"].sum())
            for scope, values in measured.items():
                for key, value in values.items():
                    extrema[scope][key] = max(extrema[scope].get(key, 0.), value)
        probabilities = {item["name"]: item["probability"] for item in profile["window_cases"]}
        frequencies = {}
        for index, name in enumerate(CASE_NAMES):
            expected = probabilities[name]
            actual = int(counts[index]) / args.samples
            tolerance = max(.015, 6 * math.sqrt(expected * (1 - expected) / args.samples))
            require(abs(actual - expected) < tolerance, f"mixture frequency outside 6sigma tolerance: {name}")
            frequencies[name] = {"count": int(counts[index]), "fraction": actual,
                                 "expected": expected, "tolerance": tolerance}
        report["mixture"] = frequencies
        report["correlated_fraction"] = correlated / args.samples
        expected_correlation = profile["temporal_noise"]["affected_window_modes"]["correlated"]
        require(abs(correlated / args.samples - expected_correlation) <
                max(.015, 6 * math.sqrt(expected_correlation * (1 - expected_correlation) / args.samples)),
                "correlated/independent window frequency outside 6sigma tolerance")
        report["measured_geometric_extrema"] = extrema
        report["forced_validation_modes"] = check_forced_modes(profile)
        report["time_correlation"] = check_physical_time(profile)
        report["passed"] = True
    except Exception as error:
        report["error"] = str(error)
        report["traceback"] = traceback.format_exc()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(args.output)
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
