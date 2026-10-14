#!/usr/bin/env python3
"""Remote GPU checks for the structured 99D TWM; emits one JSON result.

Run in the ds Pod, for example:
  python scripts/check_structured_twm_numerics.py --config config/<run>.json

These are synthetic numerical/interface checks, not reconstruction evaluation.
The default is the full six-block, width-256 configuration. A supplied config
must retain that capacity so a passing result includes the formal BF16 smoke
test. No datasets, checkpoints, or optimizer state are written.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields
import json
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch import nn

from handloop import geometry as g
from handloop.models import TWM, TWMConfig


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def random_state(*shape: int, device: torch.device) -> torch.Tensor:
    base = g.identity_state(*shape, device=device)
    delta = torch.randn(*shape, 51, device=device) * 0.12
    delta[..., :3] *= 0.2
    return g.state_apply_delta(base, delta)


def make_inputs(batch: int, history: int, horizon: int, device: torch.device) -> dict:
    return {
        "history": random_state(batch, history, device=device),
        "raw": random_state(batch, device=device),
        "candidate": random_state(batch, device=device),
        "history_valid": torch.ones(batch, history, dtype=torch.bool, device=device),
        "history_availability": torch.ones(batch, history, 17, dtype=torch.bool, device=device),
        "availability": torch.ones(batch, 17, dtype=torch.bool, device=device),
        "candidate_availability": torch.ones(batch, 17, dtype=torch.bool, device=device),
        "dt": torch.full((batch, history + 1), 1 / 30, device=device),
        "beta": torch.randn(batch, 10, device=device) * 0.1,
        "beta_valid": torch.ones(batch, dtype=torch.bool, device=device),
        "handedness": torch.arange(batch, device=device) % 2,
        "history_quality": torch.rand(batch, history, 17, device=device),
        "quality": torch.rand(batch, 17, device=device),
        "forecast_dt": torch.arange(1, horizon + 1, device=device)[None].expand(batch, -1) / 10,
    }


def clone_inputs(inputs: dict) -> dict:
    return {key: value.clone() for key, value in inputs.items()}


def check_rotations(state: torch.Tensor) -> dict:
    require(state.shape[-1] == 99, "public state is not 99D")
    require(state.dtype == torch.float32, "geometry output must remain float32")
    require(bool(torch.isfinite(state).all()), "non-finite state")
    # Check the emitted six numbers BEFORE the decoder can normalize them.
    encoded = state[..., 3:].reshape(-1, 16, 6)
    first, second = encoded[..., :3], encoded[..., 3:]
    norm_error = torch.stack((first.norm(dim=-1) - 1, second.norm(dim=-1) - 1)).abs().max()
    dot_error = (first * second).sum(-1).abs().max()
    matrix = torch.stack((first, second, torch.cross(first, second, dim=-1)), dim=-1)
    orthogonal_error = (matrix.transpose(-1, -2) @ matrix - torch.eye(3, device=state.device)).abs().max()
    determinant_error = (torch.linalg.det(matrix) - 1).abs().max()
    errors = {"column_norm": float(norm_error), "column_dot": float(dot_error),
              "orthogonality": float(orthogonal_error), "determinant": float(determinant_error)}
    require(max(errors.values()) < 3e-5, f"invalid emitted rotations: {errors}")
    return errors


def check_structure(model: TWM, inputs: dict) -> dict:
    encoder = model.encoder
    require(type(encoder).__name__ == "StructuredMambaEncoder", "structured encoder absent")
    expected_indices = {
        "parent_indices": [0, 1, 2, 3, 1, 5, 6, 1, 8, 9, 1, 11, 12, 1, 14, 15],
        "finger_indices": [0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5],
        "depth_indices": [0, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
    }
    for name, expected in expected_indices.items():
        require(getattr(encoder, name).tolist() == expected, f"incorrect MANO {name}")
    for name in ("parent_embedding", "finger_embedding", "depth_embedding"):
        require(isinstance(getattr(encoder, name), nn.Embedding), f"missing {name}")
    require(len(encoder.blocks) == model.config.num_layers, "wrong block count")
    events, handles = [], []

    def record(label):
        def hook(_module, args):
            events.append((label, list(args[0].shape)))
        return hook

    for index, block in enumerate(encoder.blocks):
        require(isinstance(block.spatial_attention, nn.MultiheadAttention), "missing joint attention")
        temporal_type = type(block.temporal)
        require(temporal_type.__module__.startswith("mamba_ssm.") and temporal_type.__name__ == "Mamba",
                f"expected actual mamba_ssm.Mamba, found {temporal_type}")
        require((block.temporal.d_state, block.temporal.d_conv, block.temporal.expand) ==
                (model.config.mamba_state_size, model.config.mamba_conv, model.config.mamba_expand),
                "Mamba hyperparameters differ from config")
        handles.append(block.spatial_attention.register_forward_pre_hook(record(f"joint_{index}")))
        handles.append(block.temporal.register_forward_pre_hook(record(f"time_{index}")))
    try:
        with torch.no_grad():
            model(**inputs)
    finally:
        for handle in handles:
            handle.remove()
    batch, history, _ = inputs["history"].shape
    frames, width = history + 1, model.config.d_model
    expected = []
    for index in range(model.config.num_layers):
        expected.extend([(f"joint_{index}", [batch * frames, 16, width]),
                         (f"time_{index}", [batch * 16, frames, width])])
    require(events == expected, f"wrong spatial/temporal execution: {events}")
    return {"blocks": len(encoder.blocks), "joint_input": expected[0][1],
            "temporal_input": expected[1][1], "temporal_class": str(temporal_type)}


def check_encoder_boundaries(model: TWM, device: torch.device) -> dict:
    encoder = model.encoder
    tokens = torch.randn(2, 7, 16, model.config.d_model, device=device)
    valid = torch.ones(2, 7, device=device, dtype=torch.bool)
    first = encoder(tokens, valid)
    changed = tokens.clone()
    changed[:, 4:] = torch.randn_like(changed[:, 4:]) * 20
    second = encoder(changed, valid)
    causal_error = float((first[:, :4] - second[:, :4]).abs().max())
    require(causal_error < 2e-5, f"later tokens contaminated earlier encoder features: {causal_error}")
    require(float((first[:, 4:] - second[:, 4:]).abs().max()) > 1e-4,
            "future perturbation had no effect; causal test is vacuous")

    # Include different mask patterns and different valid lengths in one batch.
    valid[0, [0, 2, 4]] = False
    valid[1, [1, 3]] = False
    padded = encoder(tokens, valid)
    placeholders = tokens.clone()
    placeholders[~valid] = float("nan")
    changed_padding = encoder(placeholders, valid)
    require(torch.equal(padded, changed_padding), "masked encoder token values influence valid outputs")
    require(bool((padded[~valid] == 0).all()), "invalid encoder slots must be zero")
    compact_error = 0.0
    for row in range(2):
        compact = tokens[row:row + 1, valid[row]]
        compact_valid = torch.ones(1, compact.shape[1], device=device, dtype=torch.bool)
        compact_out = encoder(compact, compact_valid)
        compact_error = max(compact_error, float((padded[row, valid[row]] - compact_out[0]).abs().max()))
    require(compact_error < 3e-5, f"padding changes the valid SSM sequence: {compact_error}")
    return {"causal_max_abs": causal_error, "padding_compaction_max_abs": compact_error}


def check_backward(model: TWM, inputs: dict, *, bf16: bool) -> dict:
    # Exercise the same SmoothL1-position/chordal-rotation objective as training.
    from train_twm import state_loss

    model.zero_grad(set_to_none=True)
    model.train()
    target = random_state(inputs["raw"].shape[0], device=inputs["raw"].device)
    attention_dtypes = []
    handle = model.encoder.blocks[0].spatial_attention.register_forward_hook(
        lambda _module, _args, output: attention_dtypes.append(str(output[0].dtype)))
    try:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bf16):
            output = model(**inputs)
    finally:
        handle.remove()
    future_target = target[:, None].expand(-1, model.config.forecast_horizon, -1)
    loss = (state_loss(output["state"], target)
            + 0.1 * state_loss(output["forecast"].reshape(-1, 99), future_target.reshape(-1, 99)))
    if bf16:
        require("torch.bfloat16" in attention_dtypes, "BF16 autocast did not reach joint attention")
    require(bool(torch.isfinite(loss)), "non-finite loss")
    loss.backward()
    gradients = {name: param.grad for name, param in model.named_parameters() if param.grad is not None}
    require(bool(gradients), "no gradients")
    bad = [name for name, gradient in gradients.items() if not bool(torch.isfinite(gradient).all())]
    require(not bad, f"non-finite gradients: {bad}")
    for name in ("encoder.parent_embedding.weight", "encoder.finger_embedding.weight",
                 "encoder.depth_embedding.weight", "translation_head.weight", "rotation_head.weight"):
        require(name in gradients and bool((gradients[name] != 0).any()), f"disconnected/zero gradient: {name}")
    for index in range(model.config.num_layers):
        for part in ("spatial_attention", "temporal"):
            prefix = f"encoder.blocks.{index}.{part}."
            require(any(name.startswith(prefix) and bool((gradient != 0).any())
                        for name, gradient in gradients.items()), f"no learning signal for {prefix}")
    geometry = check_rotations(output["state"])
    check_rotations(output["forecast"])
    return {"loss": float(loss.detach()), "gradient_tensors": len(gradients),
            "state_rotation_errors": geometry, "autocast": "bfloat16" if bf16 else "disabled",
            "attention_output_dtypes": sorted(set(attention_dtypes)),
            "valid_history_frames": inputs["history_valid"].sum(-1).tolist()}


def run() -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="training JSON with a model object, or model-only JSON")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2, help="synthetic batch size, at least 2")
    parser.add_argument("--history-frames", type=int, help="defaults to training config or 30")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if not Path("/sciclone/home/jchen57").is_dir():
        raise RuntimeError("Run this numerical/model check on the remote ds Pod, never the local workstation")
    require(args.device.startswith("cuda") and torch.cuda.is_available(), "a remote CUDA GPU is required")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    require(torch.cuda.is_bf16_supported(), "BF16 is not supported on this GPU")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    payload = json.loads(args.config.read_text()) if args.config else {}
    model_values = payload.get("model", payload)
    names = {field.name for field in fields(TWMConfig)}
    require(not (set(model_values) - names), f"unknown model config keys: {set(model_values) - names}")
    defaults = {"backbone": "structured_mamba", "d_model": 256, "num_layers": 6,
                "nhead": 4, "dim_feedforward": 512, "mamba_state_size": 64,
                "mamba_conv": 4, "mamba_expand": 2}
    defaults.update(model_values)
    defaults["check_finite"] = True  # Numerical checks deliberately keep input/output validation on.
    config = TWMConfig(**defaults)
    require(config.backbone == "structured_mamba", "this checker requires structured_mamba")
    require((config.num_layers, config.d_model, config.nhead, config.dim_feedforward,
             config.mamba_state_size, config.mamba_conv, config.mamba_expand) == (6, 256, 4, 512, 64, 4, 2),
            "formal BF16 check requires 6 blocks, d256, 4 heads, FF512, Mamba 64/4/2")
    require(config.forecast_horizon >= 1, "forecast timing check requires at least one forecast")
    history = args.history_frames if args.history_frames is not None else payload.get("history_frames", 30)
    require(args.batch_size >= 2 and history >= 2, "need batch >= 2 and history >= 2")

    model = TWM(config).to(device)
    # Zero-initialized heads would hide encoder contamination and zero encoder gradients.
    with torch.no_grad():
        for name in ("translation_head", "rotation_head", "forecast_translation_head", "forecast_rotation_head"):
            head = getattr(model, name)
            nn.init.normal_(head.weight, std=0.02)
            nn.init.normal_(head.bias, std=0.01)
    inputs = make_inputs(args.batch_size, history, config.forecast_horizon, device)
    model.eval()
    structure = check_structure(model, inputs)
    with torch.no_grad():
        boundaries = check_encoder_boundaries(model, device)
        first = model(**inputs)
        changed = clone_inputs(inputs)
        changed["forecast_dt"] *= 2.7
        later = model(**changed)
        require(torch.equal(first["state"], later["state"]), "forecast_dt contaminated current state")
        require(torch.equal(first["state_features"], later["state_features"]), "forecast_dt contaminated current features")
        require(not torch.equal(first["forecast"], later["forecast"]), "forecast timing test is vacuous")

        padded = clone_inputs(inputs)
        padded["history_valid"][:, ::2] = False
        padded["history_valid"][0] = False
        original_padding = model(**padded)
        masked = ~padded["history_valid"]
        padded["history"][masked] = float("nan")
        padded["history_availability"][masked] = False
        padded["history_quality"][masked] = 0.999
        history_dt = padded["dt"][:, :-1]
        history_dt[masked] = 12345.0
        new_padding = model(**padded)
        for key in ("state", "state_features", "forecast"):
            require(torch.equal(original_padding[key], new_padding[key]), f"masked history affects {key}")
        empty = make_inputs(args.batch_size, 0, config.forecast_horizon, device)
        empty_output = model(**empty)
        check_rotations(empty_output["state"])
        check_rotations(empty_output["forecast"])

    # Exercise both the dense and gather/scatter paths with nonzero heads.
    fp32 = check_backward(model, inputs, bf16=False)
    padding_backward = check_backward(model, padded, bf16=False)
    torch.cuda.reset_peak_memory_stats(device)
    bf16 = check_backward(model, inputs, bf16=True)
    torch.cuda.synchronize(device)
    return {"passed": True, "device": str(device), "gpu": torch.cuda.get_device_name(device),
            "torch": torch.__version__, "config": asdict(config), "seed": args.seed,
            "batch_size": args.batch_size, "history_frames": history,
            "parameters": sum(p.numel() for p in model.parameters()),
            "structure": structure, "encoder_boundaries": boundaries,
            "fp32_backward": fp32, "padding_backward": padding_backward, "bf16_backward": bf16,
            "bf16_peak_allocated_mib": round(torch.cuda.max_memory_allocated(device) / 2**20, 2),
            "tests": ["actual_structured_mamba", "joint_then_temporal_axes", "mano_tree_indices",
                      "nonzero_head_backward", "encoder_future_causality", "encoder_padding_compaction",
                      "masked_history_and_dt_inert", "padding_backward", "empty_history", "forecast_dt_no_current_leak",
                      "emitted_99d_rotations", "bf16_full_6x256_backward"]}


if __name__ == "__main__":
    try:
        print(json.dumps(run(), sort_keys=True))
    except Exception as exc:
        print(json.dumps({"passed": False, "error_type": type(exc).__name__, "error": str(exc)}), flush=True)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
