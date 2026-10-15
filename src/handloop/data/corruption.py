"""Bounded mixed corruption for pure-trajectory current-state restoration.

The public sampler preserves the original six-item trainer interface.  Only
history and current inputs are corrupted; current/future labels remain source
reference states expressed in the same history-derived anchor.  Magnitudes
are actual displacement norms and SO(3) angles, never per-axis deviations or
noise added to a serialized rotation6D vector.

State-group order is translation, root rotation, then MANO's internal index,
middle, pinky, ring, thumb order (three local rotations per finger).  This is
not the thumb-first visualization order.  Missing current groups are filled
in the candidate only from the last completed, possibly corrupted history.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import Tensor

from handloop import geometry as g


LEVEL_NAMES = ("clean", "near", "light", "medium", "strong")
CASE_NAMES = (
    "clean_motion", "current_reanchors_history", "ordinary_restoration",
    "partial_missing", "rare_local_failure",
)
GROUP_NAMES = ("wrist_translation", "root_rotation") + tuple(
    f"{finger}{joint}" for finger in ("index", "middle", "pinky", "ring", "thumb")
    for joint in (1, 2, 3)
)
_LEVEL_ID = {name: i for i, name in enumerate(LEVEL_NAMES)}
_MODES = (
    "mixed", "clean", "reanchor", "light_clean_history", "light_noisy_history",
    "medium", "strong", "missing", "history_only", "noisy",
)


def _probabilities(values: Mapping, names: tuple[str, ...], label: str) -> list[float]:
    if not isinstance(values, Mapping) or set(values) - set(names):
        raise ValueError(f"{label}: unsupported probability categories")
    probabilities = [float(values.get(name, 0.0)) for name in names]
    if (any(not math.isfinite(p) or p < 0 for p in probabilities)
            or abs(sum(probabilities) - 1.0) > 1e-6):
        raise ValueError(f"{label}: finite nonnegative probabilities must sum to one")
    return probabilities


def _profile_tables(profile: dict, device) -> tuple[list, Tensor, list, list, list, list, list]:
    """Validate the serializable profile without consulting any source motion."""
    levels = profile["levels"]
    ranges = []
    for name in LEVEL_NAMES:
        entry = levels[name]
        group_ranges = []
        for key, scale, ceiling in (
            ("translation_norm_mm", 0.001, math.inf),
            ("root_angle_deg", math.pi / 180.0, 180.0),
            ("finger_angle_deg", math.pi / 180.0, 180.0),
        ):
            bounds = entry[key]
            if (len(bounds) != 2 or not all(math.isfinite(float(x)) for x in bounds)
                    or not 0 <= float(bounds[0]) <= float(bounds[1]) <= ceiling):
                raise ValueError(f"Invalid {name}.{key} bounds")
            group_ranges.append([float(x) * scale for x in bounds])
        ranges.append([group_ranges[0], group_ranges[1]] + [group_ranges[2]] * 15)
    entries = profile["window_cases"]
    cases = {entry["name"]: entry for entry in entries}
    if len(entries) != len(CASE_NAMES) or set(cases) != set(CASE_NAMES):
        raise ValueError(f"window_cases must contain exactly {CASE_NAMES}")
    case_prob = _probabilities(
        {name: cases[name]["probability"] for name in CASE_NAMES}, CASE_NAMES, "window_cases")
    current_prob, history_prob = [], []
    for name in CASE_NAMES:
        current_prob.append(_probabilities(cases[name]["current_levels"], LEVEL_NAMES,
                                           name + ".current_levels"))
        history_prob.append(_probabilities(cases[name]["history_levels"], LEVEL_NAMES,
                                           name + ".history_levels"))
    missing_prob = _probabilities(cases[CASE_NAMES[3]]["missing_cases"],
                                 ("one_whole_finger", "one_to_three_state_groups", "whole_current_hand"),
                                 "missing_cases")
    strong_prob = _probabilities(cases[CASE_NAMES[4]]["strong_override"],
                                ("wrist_translation_and_root", "one_or_two_fingers"),
                                "strong_override")
    temporal_prob = _probabilities(profile["temporal_noise"]["affected_window_modes"],
                                  ("independent", "correlated"), "temporal modes")
    return (case_prob, torch.tensor(ranges, device=device, dtype=torch.float32),
            current_prob, history_prob, missing_prob, strong_prob, temporal_prob)


def sample_mixed_batch(batch, generator, mode="mixed", *, condition_on_beta=False,
                       profile: dict, return_diagnostics=False):
    """Return ``history, raw, candidate, kwargs, target, future``.

    ``mixed`` draws the profile's five cases (20/20/35/20/5 in the D31 profile).
    ``noisy`` forces the ordinary light/medium case.  Fixed diagnostic modes:

    * ``clean``: clean history/current;
    * ``reanchor``: medium history, exactly clean current;
    * ``light_clean_history`` / ``light_noisy_history``: light current with
      respectively clean / medium history;
    * ``medium``: medium history/current;
    * ``strong`` / ``missing``: their respective profile cases;
    * ``history_only``: the exact same history as ``noisy`` at the same RNG
      state, with completely absent current (a paired removal intervention).

    All modes consume the same random draw schedule, so equal RNG states give
    paired noise directions/amplitudes and matching histories when their
    history levels match.  Correlated directions are normalized stationary OU
    latents, with rho=exp(-actual_dt/tau), tau uniform in [0.1,0.3] seconds by
    default.  Radii/angles are separately uniform within their bounded range;
    temporal correlation therefore describes direction, not a fixed bias
    magnitude.  The reanchor case explicitly resets the current OU latent.

    If requested, return ``(prepared_tuple, diagnostics)``.  Diagnostics never
    enter model kwargs.  ``raw_world`` is the pre-missing-placeholder corrupted
    current, while model raw uses identity/zero for absent groups.  Amplitudes
    are metres for group zero and radians for the other sixteen groups.
    """
    if mode not in _MODES:
        raise ValueError(f"Unknown mixed corruption mode {mode!r}; expected {_MODES}")
    clean = batch["past_current"].float()
    if clean.ndim != 3 or clean.shape[-1] != 99 or clean.shape[1] < 2:
        raise ValueError("past_current must be [B,H+1,99] with at least one history frame")
    if "past_current_gt_mask" in batch and not bool(batch["past_current_gt_mask"].all()):
        raise ValueError("Mixed corruption requires complete past/current MANO GT")
    b, length, _ = clean.shape
    h, device = length - 1, clean.device
    dt = batch["dt"].float()
    if dt.shape != (b, length) or not bool(torch.isfinite(dt).all()) or bool((dt <= 0).any()):
        raise ValueError("dt must contain positive finite actual [B,H+1] intervals in seconds")
    (case_prob, range_table, current_prob, history_prob,
     missing_prob, strong_prob, temporal_prob) = _profile_tables(profile, device)

    def rand(*shape):
        return torch.rand(*shape, device=device, generator=generator)

    def normal(*shape):
        return torch.randn(*shape, device=device, generator=generator)

    def categorical(probabilities):
        cumulative = torch.tensor(probabilities, device=device, dtype=torch.float32).cumsum(0)
        cumulative[-1] = 1.0
        return (rand(b, 1) >= cumulative[None]).sum(-1)

    case_id = categorical(case_prob)
    forced_case = {
        "clean": 0, "reanchor": 1, "light_clean_history": 2,
        "light_noisy_history": 2, "medium": 2, "noisy": 2,
        "missing": 3, "history_only": 2, "strong": 4,
    }
    if mode != "mixed":
        case_id = torch.full_like(case_id, forced_case[mode])
    current_level, history_level = torch.zeros_like(case_id), torch.zeros_like(case_id)
    # Always draw both distributions for all cases.  Forced evaluation modes
    # consequently preserve the same RNG schedule instead of silently changing
    # the history noise when current availability/severity is changed.
    for index in range(len(CASE_NAMES)):
        selected = case_id == index
        current_level = torch.where(selected, categorical(current_prob[index]), current_level)
        history_level = torch.where(selected, categorical(history_prob[index]), history_level)
    forced_levels = {
        "clean": ("clean", "clean"), "reanchor": ("clean", "medium"),
        "light_clean_history": ("light", "clean"),
        "light_noisy_history": ("light", "medium"), "medium": ("medium", "medium"),
    }
    if mode in forced_levels:
        current_name, history_name = forced_levels[mode]
        current_level = torch.full_like(current_level, _LEVEL_ID[current_name])
        history_level = torch.full_like(history_level, _LEVEL_ID[history_name])
    current_group_level = current_level[:, None].expand(b, 17).clone()
    history_group_level = history_level[:, None].expand(b, 17).clone()
    group = torch.arange(17, device=device)[None]
    finger_group = ((group - 2) // 3).clamp_min(0)
    # Strong corruption is localized: one wrist/root pair OR one/two complete
    # fingers.  Other current groups retain their base (normally light) level.
    strong_kind = categorical(strong_prob)
    strong_fingers = rand(b, 5).argsort(-1).argsort(-1)
    strong_finger_count = 1 + (rand(b) < 0.5).long()
    selected_fingers = strong_fingers < strong_finger_count[:, None]
    finger_selected = selected_fingers.gather(1, finger_group.expand(b, 17)) & (group >= 2)
    strong_affected = ((strong_kind[:, None] == 0) & (group < 2)) | (
        (strong_kind[:, None] == 1) & finger_selected)
    strong_affected &= case_id[:, None] == 4
    current_group_level = torch.where(strong_affected, _LEVEL_ID["strong"], current_group_level)

    missing_kind = categorical(missing_prob)
    missing_finger = (rand(b) * 5).long()
    missing_group_count = 1 + (rand(b) * 3).long()
    missing_group_rank = rand(b, 17).argsort(-1).argsort(-1)
    one_finger = (group >= 2) & (finger_group == missing_finger[:, None])
    few_groups = missing_group_rank < missing_group_count[:, None]
    missing = ((missing_kind[:, None] == 0) & one_finger) | (
        (missing_kind[:, None] == 1) & few_groups) | (missing_kind[:, None] == 2)
    missing &= case_id[:, None] == 3
    if mode == "history_only":
        missing = torch.ones(b, 17, device=device, dtype=torch.bool)
        missing_kind = torch.full_like(missing_kind, 2)
    availability = ~missing

    correlated = categorical(temporal_prob) == 1
    tau_bounds = profile["temporal_noise"]["latent_direction_correlation_seconds"]
    if (len(tau_bounds) != 2 or not all(math.isfinite(float(x)) for x in tau_bounds)
            or not 0 < float(tau_bounds[0]) <= float(tau_bounds[1])):
        raise ValueError("Correlation times must be positive finite ordered seconds")
    tau = float(tau_bounds[0]) + rand(b) * (float(tau_bounds[1]) - float(tau_bounds[0]))
    innovations = normal(b, length, 17, 3)
    latent = [innovations[:, 0]]
    for index in range(1, length):
        rho = torch.exp(-dt[:, index] / tau)[:, None, None]
        value = rho * latent[-1] + (1.0 - rho.square()).clamp_min(0).sqrt() * innovations[:, index]
        # A reliable current observation may reject a corrupted history.  It
        # does not inherit the historical error direction in this case.
        if index == h:
            value = torch.where((case_id == 1)[:, None, None], innovations[:, index], value)
        latent.append(value)
    correlated_latent = torch.stack(latent, dim=1)
    directions = torch.where(correlated[:, None, None, None], correlated_latent, innovations)
    norm = directions.norm(dim=-1, keepdim=True)
    fallback = torch.zeros_like(directions)
    fallback[..., 0] = 1.0
    directions = torch.where(norm > 1e-12, directions / norm.clamp_min(1e-12), fallback)
    group_levels = torch.cat((history_group_level[:, None].expand(b, h, 17),
                              current_group_level[:, None]), dim=1)
    group_indices = torch.arange(17, device=device)[None, None]
    bounds = range_table[group_levels, group_indices]
    amplitudes = bounds[..., 0] + rand(b, length, 17) * (bounds[..., 1] - bounds[..., 0])
    delta = (directions * amplitudes[..., None]).flatten(-2)
    # Group zero is dp; groups 1..16 are full axis-angle vectors.  Legal SO(3)
    # composition keeps each actual geodesic angle equal to its sampled bound.
    corrupted = g.state_apply_delta(clean, delta)
    history_world, raw_world = corrupted[:, :h], corrupted[:, -1]
    anchor = history_world[:, -1]
    anchor_rotation = g.state_to_rotmat(anchor)[:, 0]

    def to_anchor(states: Tensor) -> Tensor:
        rotations = g.state_to_rotmat(states)
        extra = states.ndim - 2
        ra = anchor_rotation.reshape(b, *([1] * extra), 3, 3)
        pa = anchor[:, :3].reshape(b, *([1] * extra), 3)
        position = (ra.transpose(-1, -2) @ (states[..., :3] - pa)[..., None]).squeeze(-1)
        rotations = rotations.clone()
        rotations[..., 0, :, :] = ra.transpose(-1, -2) @ rotations[..., 0, :, :]
        return g.state_from_rotmat(position, rotations)

    history, raw = to_anchor(history_world), to_anchor(raw_world)
    # Placeholders are not observations.  The candidate's replacement values
    # come only from completed past input, never the current/future reference.
    base = history[:, -1]
    candidate = raw.clone()
    candidate[:, :3] = torch.where(availability[:, :1], raw[:, :3], base[:, :3])
    raw_rotations = raw[:, 3:].reshape(b, 16, 6)
    candidate[:, 3:] = torch.where(availability[:, 1:, None], raw_rotations,
                                    base[:, 3:].reshape(b, 16, 6)).flatten(1)
    raw[:, :3] = torch.where(availability[:, :1], raw[:, :3], 0.0)
    raw[:, 3:] = torch.where(availability[:, 1:, None], raw_rotations,
                             g.identity_rotation_6d(device=device)).flatten(1)
    kwargs = {
        "history_valid": torch.ones(b, h, device=device, dtype=torch.bool),
        "history_availability": torch.ones(b, h, 17, device=device, dtype=torch.bool),
        "availability": availability, "candidate_availability": torch.ones_like(availability),
        "dt": batch["dt"], "beta": batch["beta"], "handedness": batch["handedness"],
        "quality": torch.where(availability, -1.0, 0.0), "history_quality": None,
        "forecast_dt": batch["forecast_dt"],
    }
    if "beta_valid" in batch:
        kwargs["beta_valid"] = batch["beta_valid"]
    if not condition_on_beta:
        kwargs["beta_valid"] = torch.zeros(b, device=device, dtype=torch.bool)
    # Future content is first consulted here, after every input and random draw
    # is fixed.  Padded future GT/masks remain the forecast trainer's concern.
    target, future = to_anchor(clean[:, -1]), to_anchor(batch["future"].float())
    prepared = history, raw, candidate, kwargs, target, future
    if not return_diagnostics:
        return prepared
    diagnostics = {
        "case_id": case_id, "case_names": CASE_NAMES, "level_names": LEVEL_NAMES,
        "group_names": GROUP_NAMES, "current_level": current_level,
        "history_level": history_level, "current_group_level": current_group_level,
        "history_group_level": history_group_level, "correlated": correlated,
        "correlation_seconds": tau, "directions": directions,
        "sampled_amplitudes": amplitudes,
        "history_world": history_world, "raw_world": raw_world, "anchor_world": anchor,
        "availability": availability, "strong_affected_groups": strong_affected,
        "missing_case_id": (missing_kind if mode == "history_only" else
                            torch.where(case_id == 3, missing_kind, -1)),
    }
    return prepared, diagnostics
