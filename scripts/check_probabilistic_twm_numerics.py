#!/usr/bin/env python3
"""Remote CUDA numerical checks for the MoE/trajectory-distribution TWM.

Synthetic interface, causal-boundary, gradient and density checks only; no
claim about reconstruction quality or uncertainty calibration is made.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch

from handloop import geometry as g
from handloop.models.probabilistic_twm import (
    ProbabilisticTWM, ProbabilisticTWMConfig, current_distribution_log_prob,
    forecast_distribution_log_prob, isotropic_matrix_fisher_log_normalizer,
)
from check_structured_twm_numerics import make_inputs, check_rotations, random_state, require


def check_density(device: torch.device) -> dict:
    # Independent radial quadrature of the SO(3) Haar integral. Its endpoint
    # resolution also resolves k=1e4's narrow mode near theta=0.
    k = torch.tensor([0., .1, 1., 40., 100., 1000., 1e4], device=device,
                     dtype=torch.float32, requires_grad=True)
    angle = torch.linspace(0, math.pi, 32769, device=device, dtype=torch.float64)
    energy = 2 - 2 * angle.cos()
    radial = (1 - angle.cos()) / math.pi
    log_z = isotropic_matrix_fisher_log_normalizer(k)
    integral = torch.trapezoid((-k.double()[:, None] * energy - log_z.double()[:, None]).exp()
                               * radial, angle, dim=-1)
    error = float((integral - 1).abs().max())
    require(error < 3e-6, f"SO3 density does not integrate to one: {error}")
    # A zero concentration endpoint is mathematically valid but not used by
    # training; special-function autograd at zero need not be differentiated.
    gradients, = torch.autograd.grad(log_z[1:].sum(), k)
    require(bool(torch.isfinite(gradients[1:]).all()), "nonfinite normalizer gradient")
    require(bool((gradients[1:] < 0).all()), "concentration normalizer should decrease")
    # Compare derivative to quadrature's exact expectation of negative energy.
    expected = -torch.trapezoid((-k.detach().double()[:, None] * energy
                                 - log_z.detach().double()[:, None]).exp()
                                * radial * energy, angle, dim=-1)
    derivative_error = float((gradients[1:].double() - expected[1:]).abs().max())
    require(derivative_error < 3e-5, f"incorrect density derivative: {derivative_error}")
    return {"normalization_max_abs": error, "derivative_max_abs": derivative_error,
            "concentrations": k.detach().tolist(), "integrals": integral.tolist()}


@torch.no_grad()
def check_causality(model: ProbabilisticTWM, inputs: dict) -> dict:
    model.eval()
    encoder = model.encoder
    tokens = torch.randn(2, 7, 16, model.config.d_model, device=inputs["raw"].device)
    valid = torch.ones(2, 7, dtype=torch.bool, device=tokens.device)
    first = encoder(tokens, valid)[0]
    changed = tokens.clone()
    changed[:, 4:] = torch.randn_like(changed[:, 4:]) * 5
    second = encoder(changed, valid)[0]
    causal_error = float((first[:, :4] - second[:, :4]).abs().max())
    require(causal_error < 3e-5, "future token perturbed past features")
    require(float((first[:, 4:] - second[:, 4:]).abs().max()) > 1e-4, "vacuous causality check")
    valid[0, [0, 2, 4]] = False
    valid[1, [1, 3]] = False
    padded = encoder(tokens, valid)[0]
    changed[~valid] = float("nan")
    changed[valid] = tokens[valid]
    padded_changed = encoder(changed, valid)[0]
    require(torch.equal(padded, padded_changed), "padding values changed encoder output")
    compaction_error = 0.
    for row in range(2):
        compact = tokens[row:row + 1, valid[row]]
        compact_valid = torch.ones(1, compact.shape[1], dtype=torch.bool, device=tokens.device)
        compact_output = encoder(compact, compact_valid)[0]
        compaction_error = max(compaction_error,
                               float((compact_output[0] - padded[row, valid[row]]).abs().max()))
    require(compaction_error < 3e-5, "padding changed compacted Mamba sequence")
    baseline = model(**inputs)
    changed_inputs = dict(inputs)
    changed_inputs["forecast_dt"] = inputs["forecast_dt"].clone()
    half = model.config.forecast_horizon // 2
    changed_inputs["forecast_dt"][:, half:] += 0.5
    future_changed = model(**changed_inputs)
    require(torch.equal(baseline["state"], future_changed["state"]),
            "forecast time query changed current restoration")
    prefix_error = float((baseline["forecast_modes"][:, :, :half]
                          - future_changed["forecast_modes"][:, :, :half]).abs().max())
    require(prefix_error < 3e-5, "later future queries changed earlier future predictions")
    require(torch.equal(baseline["mixture_logits"], future_changed["mixture_logits"]),
            "future time query changed context-only mixture weights")
    return {"encoder_causal_max_abs": causal_error, "padding_compaction_max_abs": compaction_error,
            "future_decoder_causal_max_abs": prefix_error}


def check_backward(model: ProbabilisticTWM, inputs: dict, bf16: bool) -> dict:
    model.train()
    model.zero_grad(set_to_none=True)
    b, f = len(inputs["raw"]), model.config.forecast_horizon
    target = random_state(b, device=inputs["raw"].device)
    future = random_state(b, f, device=inputs["raw"].device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
        output = model(**inputs)
    current = current_distribution_log_prob(output, target)
    terms = forecast_distribution_log_prob(output, future)
    # Use a partial target suffix in the second sample. Entire trajectory
    # mixture identity is held fixed over all valid future frames.
    mask = torch.ones(b, f, dtype=torch.bool, device=target.device)
    mask[1, f // 2:] = False
    log_likelihood = (terms["total"] * mask[:, None]).sum(-1)
    sequence_nll = -torch.logsumexp(output["mixture_logits"].log_softmax(-1) + log_likelihood, -1)
    loss = (-current["total"].mean() / 51
            + (sequence_nll / (mask.sum(-1) * 51)).mean()
            + .01 * output["moe_balance_loss"])
    require(bool(torch.isfinite(loss)), "nonfinite distribution training loss")
    loss.backward()
    bad = [name for name, p in model.named_parameters()
           if p.grad is None or not bool(torch.isfinite(p.grad).all())]
    require(not bad, f"unused/nonfinite model gradients, DDP would fail: {bad}")
    require(bool((model.encoder.mixtures[0].router.weight.grad != 0).any()), "router got no gradient")
    for index, expert in enumerate(model.encoder.mixtures[0].experts):
        require(bool((expert[0].weight.grad != 0).any()), f"expert {index} got no gradient")
    require(bool((model.mixture_head.weight.grad != 0).any()), "trajectory mixture got no gradient")
    require(output["forecast_modes"].shape == (b, model.config.num_modes, f, 99), "wrong future shape")
    require(output["forecast"].shape == (b, f, 99), "wrong MAP trajectory shape")
    check_rotations(output["state"])
    geometry = check_rotations(output["forecast_modes"])
    weights = output["mixture_weights"]
    require(float((weights.sum(-1) - 1).abs().max()) < 1e-6, "trajectory weights not normalized")
    require(bool((output["forecast_translation_std"] > 0).all()), "nonpositive translation scale")
    require(bool((output["forecast_rotation_concentration"] > 0).all()), "nonpositive concentration")
    # Different mode embeddings must actually produce distinct trajectories.
    diversity = float((output["forecast_modes"][:, 0] - output["forecast_modes"][:, 1]).abs().max())
    require(diversity > 1e-6, "identical future modes despite independent embeddings")
    # Invalid targets must make exactly zero contribution to the trajectory NLL.
    replacement = terms["total"].detach().clone()
    replacement[1, :, f // 2:] += 1e5
    masked_difference = float(((replacement * mask[:, None]).sum(-1)
                               - (terms["total"].detach() * mask[:, None]).sum(-1)).abs().max())
    require(masked_difference == 0, "masked future labels affected sequence likelihood")
    return {"loss": float(loss.detach()), "autocast_bf16": bf16,
            "parameter_tensors_with_finite_gradients": len(list(model.parameters())),
            "forecast_shape": list(output["forecast_modes"].shape),
            "initial_mode_difference": diversity, "moe_load": output["moe_load"].detach().tolist(),
            "rotation_errors": geometry}


def run() -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not Path("/sciclone/home/jchen57").is_dir():
        raise RuntimeError("Run model/numerical checks remotely in the ds Pod")
    require(torch.cuda.is_available() and args.device.startswith("cuda"), "remote CUDA required")
    torch.set_num_threads(2)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(29)
    payload = json.loads(args.config.read_text()) if args.config else {}
    config = ProbabilisticTWMConfig(**payload.get("model", {}))
    model = ProbabilisticTWM(config).to(device)
    inputs = make_inputs(2, payload.get("history_frames", 30), config.forecast_horizon, device)
    inputs["forecast_dt"] = torch.arange(1, config.forecast_horizon + 1, device=device)[None].expand(2, -1) / 30
    result = {"model_config": asdict(config), "parameters": sum(p.numel() for p in model.parameters()),
              "density": check_density(device), "causality": check_causality(model, inputs),
              "fp32_backward": check_backward(model, inputs, False),
              "bf16_backward": check_backward(model, inputs, True)}
    result["passed"] = True
    return result


if __name__ == "__main__":
    try:
        print(json.dumps(run(), indent=2))
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
