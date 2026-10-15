"""Motion-only restoration and coherent, probabilistic long-horizon forecasting.

The old deterministic TWM remains loadable. This model adds four DENSE MLP
experts after every spatial/temporal block, physical-time Fourier features,
and a causal latent future decoder. All experts run; this is not sparse top-k
execution. Routing sees a token's past/current features, never future labels.

One mixture variable selects a complete future trajectory, rather than a new
mode at every frame. Conditional on a mode, translation is diagonal Gaussian
and each rotation is isotropic matrix Fisher, normalized with respect to unit
Haar measure on SO(3). These are learned aleatoric distributions; neither MoE
weights nor predicted scales constitute an empirical calibration guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import torch
from torch import Tensor, nn

from handloop import geometry as g
from .structured_mamba import StructuredMambaEncoder
from .twm import TWM, TWMConfig


@dataclass
class ProbabilisticTWMConfig(TWMConfig):
    backbone: str = "structured_mamba"
    d_model: int = 256
    num_layers: int = 6
    nhead: int = 4
    dim_feedforward: int = 512
    forecast_horizon: int = 60
    num_experts: int = 4
    num_modes: int = 4
    future_decoder_layers: int = 1
    future_joint_width: int = 64
    time_frequencies_hz: tuple[float, ...] = (0.25, 0.5, 1., 2., 4., 8., 16.)
    translation_velocity_scale: float = 0.5  # metres / second
    rotation_velocity_scale: float = 3.0     # radians / second
    translation_std_min: float = 1e-3
    translation_std_max: float = 1.0
    current_translation_std_init: float = 0.05
    future_translation_std_init: float = 0.10
    rotation_concentration_min: float = 0.1
    rotation_concentration_max: float = 1e4
    rotation_concentration_init: float = 40.0


class PhysicalTimeEncoding(nn.Module):
    """Continuous seconds and elapsed dt, augmented by fixed Fourier bands."""

    def __init__(self, width: int, frequencies_hz: tuple[float, ...]):
        super().__init__()
        if not frequencies_hz or any(f <= 0 for f in frequencies_hz):
            raise ValueError("time frequencies must be nonempty and positive")
        self.register_buffer("frequencies", torch.tensor(frequencies_hz) * (2 * math.pi))
        self.projection = nn.Sequential(nn.Linear(3 + 2 * len(frequencies_hz), width),
                                        nn.SiLU(), nn.Linear(width, width))

    def forward(self, time_and_dt: Tensor) -> Tensor:
        # Same [relative_seconds, dt, log1p(dt)] contract as the original TWM.
        phase = time_and_dt[..., :1].float() * self.frequencies
        values = torch.cat((time_and_dt.float(), phase.sin(), phase.cos()), -1)
        return self.projection(values)


class DenseExpertMixture(nn.Module):
    """Input-conditioned dense mixture; all E experts receive gradients."""

    def __init__(self, width: int, hidden: int, experts: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.router = nn.Linear(width, experts)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Dropout(dropout),
                          nn.Linear(hidden, width)) for _ in range(experts)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: Tensor) -> tuple[Tensor, Tensor, int]:
        normalized = self.norm(values)
        weights = self.router(normalized).float().softmax(-1)
        # Accumulate without retaining a second stacked [..,E,D] tensor.
        mixed = sum(expert(normalized) * weights[..., i:i + 1].to(values.dtype)
                    for i, expert in enumerate(self.experts))
        count = weights.numel() // weights.shape[-1]
        return values + self.dropout(mixed), weights.reshape(-1, weights.shape[-1]).sum(0), count


class ProbabilisticStructuredMambaEncoder(StructuredMambaEncoder):
    def __init__(self, config: ProbabilisticTWMConfig):
        super().__init__(d_model=config.d_model, num_layers=config.num_layers,
                         nhead=config.nhead, dim_feedforward=config.dim_feedforward,
                         dropout=config.dropout, d_state=config.mamba_state_size,
                         d_conv=config.mamba_conv, expand=config.mamba_expand)
        self.mixtures = nn.ModuleList([
            DenseExpertMixture(config.d_model, config.dim_feedforward,
                               config.num_experts, config.dropout)
            for _ in range(config.num_layers)
        ])
        self._routing_stats: list[tuple[Tensor, int]] = []

    def _encode_valid(self, tokens: Tensor) -> Tensor:
        structure = (self.parent_embedding(self.parent_indices)
                     + self.finger_embedding(self.finger_indices)
                     + self.depth_embedding(self.depth_indices))
        tokens = tokens + structure[None, None].to(dtype=tokens.dtype)
        for block, mixture in zip(self.blocks, self.mixtures):
            tokens, load_sum, count = mixture(block(tokens))
            self._routing_stats.append((load_sum, count))
        return self.final_norm(tokens)

    def forward(self, tokens: Tensor, history_valid: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        self._routing_stats = []
        encoded = super().forward(tokens, history_valid)
        # Groups with different valid lengths contribute by token count. Each
        # layer is balanced separately; opposite collapsed layers must not
        # cancel each other in a seemingly balanced aggregate.
        layers = len(self.mixtures)
        per_layer = []
        for layer in range(layers):
            stats = self._routing_stats[layer::layers]
            per_layer.append(torch.stack([item[0] for item in stats]).sum(0)
                             / sum(item[1] for item in stats))
        layer_load = torch.stack(per_layer)
        load = layer_load.mean(0)
        # Zero at uniform load in every layer. This is a regularizer, not evidence
        # of physically meaningful expert specialization.
        balance = (load.numel() * layer_load.square().sum(-1) - 1).mean()
        self._routing_stats = []
        return encoded, balance, load


def isotropic_matrix_fisher_log_normalizer(concentration: Tensor) -> Tensor:
    """log Z(k) for exp[-k(3-tr(M^T R))] under unit SO(3) Haar measure.

    A uniform rotation's principal-angle density on [0,pi] is
    (1-cos(theta))/pi. Consequently
      Z(k) = exp(-2k) [I0(2k)-I1(2k)] = i0e(2k)-i1e(2k).
    Double precision avoids cancellation for the configured k<=1e4. The
    returned density and gradients are float32, consistent with geometry.
    """
    with torch.autocast(device_type=concentration.device.type, enabled=False):
        k = concentration.double()
        if bool((k < 0).any()) or not bool(torch.isfinite(k).all()):
            raise ValueError("matrix-Fisher concentration must be finite and nonnegative")
        z = 2 * k
        return (torch.special.i0e(z) - torch.special.i1e(z)).log().float()


def state_distribution_log_prob(mean: Tensor, target: Tensor, translation_std: Tensor,
                                rotation_concentration: Tensor) -> dict[str, Tensor]:
    """Broadcastable full-state log density; SI translation and Haar rotations.

    Translation result sums three scalar normal log densities. Rotation result
    preserves the sixteen anatomical rotation terms for separate diagnostics.
    The total is a proper product density, not a Gaussian on redundant 99D.
    """
    with torch.autocast(device_type=mean.device.type, enabled=False):
        mean, target = mean.float(), target.float()
        scale, k = translation_std.float(), rotation_concentration.float()
        translation = (-0.5 * ((target[..., :3] - mean[..., :3]) / scale).square()
                       - scale.log() - 0.5 * math.log(2 * math.pi)).sum(-1)
        rotations, true_rotations = g.state_to_rotmat(mean), g.state_to_rotmat(target)
        # Squared chordal distance is stable at exact equality and at pi.
        energy = (rotations - true_rotations).square().sum((-1, -2)) * 0.5
        rotation = -k * energy - isotropic_matrix_fisher_log_normalizer(k)
        return {"total": translation + rotation.sum(-1),
                "translation": translation, "rotation": rotation}


def current_distribution_log_prob(output: dict[str, Tensor], target: Tensor) -> dict[str, Tensor]:
    return state_distribution_log_prob(output["state"], target,
                                       output["state_translation_std"],
                                       output["state_rotation_concentration"])


def forecast_distribution_log_prob(output: dict[str, Tensor], target: Tensor) -> dict[str, Tensor]:
    return state_distribution_log_prob(output["forecast_modes"], target[:, None],
                                       output["forecast_translation_std"],
                                       output["forecast_rotation_concentration"])


class _FutureLatentBlock(nn.Module):
    def __init__(self, config: ProbabilisticTWMConfig, index: int):
        super().__init__()
        from mamba_ssm import Mamba
        self.norm = nn.LayerNorm(config.d_model)
        self.mamba = Mamba(d_model=config.d_model, d_state=config.mamba_state_size,
                           d_conv=config.mamba_conv, expand=config.mamba_expand,
                           layer_idx=config.num_layers + index)
        self.ffn_norm = nn.LayerNorm(config.d_model)
        self.ffn = nn.Sequential(nn.Linear(config.d_model, config.dim_feedforward), nn.GELU(),
                                 nn.Linear(config.dim_feedforward, config.d_model))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, values: Tensor) -> Tensor:
        values = values + self.dropout(self.mamba(self.norm(values).contiguous()))
        return values + self.dropout(self.ffn(self.ffn_norm(values)))


class ProbabilisticTWM(TWM):
    """99D current restoration and a K-mode distribution over F future states.

    Inputs exactly follow TWM.forward. forecast_dt[B,F] are increasing physical
    offsets in seconds, not labels. The decoder predicts incremental world-
    frame wrist velocity and parent-frame angular velocities. Exp(omega*dt)
    is LEFT-composed successively with each previous rotation. No future
    observed/teacher-forced state is ever an input.
    """

    def __init__(self, config: Optional[ProbabilisticTWMConfig] = None):
        config = config or ProbabilisticTWMConfig()
        if config.backbone != "structured_mamba":
            raise ValueError("ProbabilisticTWM requires the real structured Mamba backbone")
        if min(config.num_experts, config.num_modes, config.future_decoder_layers,
               config.future_joint_width, config.forecast_horizon) < 1:
            raise ValueError("experts, modes, decoder layers/width and horizon must be positive")
        if not (0 < config.translation_std_min <= config.current_translation_std_init
                <= config.translation_std_max and config.translation_std_min
                <= config.future_translation_std_init <= config.translation_std_max):
            raise ValueError("invalid translation uncertainty range/initialization")
        if not (0 < config.rotation_concentration_min <= config.rotation_concentration_init
                <= config.rotation_concentration_max):
            raise ValueError("invalid rotational uncertainty range/initialization")
        super().__init__(config)
        # Remove old deterministic one-point forecasting parameters entirely.
        del self.forecast_translation_head, self.forecast_rotation_head
        self.encoder = ProbabilisticStructuredMambaEncoder(config)
        self.time_projection = PhysicalTimeEncoding(config.d_model, tuple(config.time_frequencies_hz))
        self.state_translation_scale_head = nn.Linear(config.d_model, 3)
        self.state_rotation_concentration_head = nn.Linear(config.d_model, 1)
        self.mode_embedding = nn.Embedding(config.num_modes, config.d_model)
        nn.init.normal_(self.mode_embedding.weight, std=0.1)
        self.future_context = nn.Linear(config.d_model, config.d_model)
        self.future_decoder = nn.ModuleList([
            _FutureLatentBlock(config, i) for i in range(config.future_decoder_layers)
        ])
        self.future_norm = nn.LayerNorm(config.d_model)
        self.future_joint_projection = nn.Linear(config.d_model, config.future_joint_width)
        self.future_time_projection = nn.Linear(config.d_model, config.future_joint_width)
        self.future_joint_norm = nn.LayerNorm(config.future_joint_width)
        self.future_translation_velocity_head = nn.Linear(config.future_joint_width, 3)
        self.future_rotation_velocity_head = nn.Linear(config.future_joint_width, 3)
        self.future_translation_scale_head = nn.Linear(config.future_joint_width, 3)
        self.future_rotation_concentration_head = nn.Linear(config.future_joint_width, 1)
        self.mixture_head = nn.Linear(config.d_model, config.num_modes)
        nn.init.zeros_(self.mixture_head.weight)
        nn.init.zeros_(self.mixture_head.bias)
        for head, initial in (
            (self.state_translation_scale_head, config.current_translation_std_init),
            (self.future_translation_scale_head, config.future_translation_std_init),
            (self.state_rotation_concentration_head, config.rotation_concentration_init),
            (self.future_rotation_concentration_head, config.rotation_concentration_init),
        ):
            nn.init.zeros_(head.weight)
            nn.init.constant_(head.bias, math.log(initial))
        # Distinct mode features and small nonzero forecast heads break mixture
        # symmetry immediately. Current restoration retains its identity init.
        for head in (self.future_translation_velocity_head, self.future_rotation_velocity_head):
            nn.init.normal_(head.weight, std=0.01)
            nn.init.zeros_(head.bias)

    def _translation_std(self, values: Tensor) -> Tensor:
        c = self.config
        return values.float().clamp(math.log(c.translation_std_min),
                                     math.log(c.translation_std_max)).exp()

    def _concentration(self, values: Tensor) -> Tensor:
        c = self.config
        return values.float().clamp(math.log(c.rotation_concentration_min),
                                     math.log(c.rotation_concentration_max)).exp()

    def forward(self, history: Tensor, raw: Tensor, candidate: Tensor, *,
                history_valid: Tensor, history_availability: Tensor,
                availability: Tensor, candidate_availability: Tensor, dt: Tensor,
                beta: Optional[Tensor], handedness: Tensor,
                history_quality: Optional[Tensor] = None, quality: Optional[Tensor] = None,
                forecast_dt: Optional[Tensor] = None,
                beta_valid: Optional[Tensor] = None,
                return_forecast: bool = True) -> dict[str, Tensor]:
        """Restore current state; optionally skip only the future decoder.

        ``return_forecast=False`` preserves the exact current readout and its
        autograd graph. The default retains all existing training/inference
        outputs and does not change saved weights or their state-dict keys.
        """
        if history.ndim != 3 or history.shape[-1] != 99:
            raise ValueError("history must be (B,H,99)")
        b, h, _ = history.shape
        for value, expected, name in (
            (raw, (b, 99), "raw"), (candidate, (b, 99), "candidate"),
            (history_valid, (b, h), "history_valid"),
            (history_availability, (b, h, 17), "history_availability"),
            (availability, (b, 17), "availability"),
            (candidate_availability, (b, 17), "candidate_availability"),
            (dt, (b, h + 1), "dt"), (handedness, (b,), "handedness"),
        ):
            self._shape(value, expected, name)
        history_valid = history_valid.bool()
        hist_avail = history_availability.bool() & history_valid[..., None]
        raw_avail, cand_avail = availability.bool(), candidate_availability.bool()
        dt = dt.float()
        self._finite(dt, "dt")
        if (dt < 0).any() or (dt[:, -1] <= 0).any():
            raise ValueError("dt must be nonnegative, with positive current dt")
        if ((handedness < 0) | (handedness > 2)).any():
            raise ValueError("handedness must be 0=left, 1=right, 2=unknown")
        hist, _ = self._state_tokens(history, hist_avail)
        raw_tokens, _ = self._state_tokens(raw, raw_avail)
        cand_tokens, base = self._state_tokens(candidate, cand_avail)
        hq, hknown = self._quality(history_quality, hist_avail)
        q, known = self._quality(quality, raw_avail)
        hist = self.history_projection(torch.cat((hist, self._mask_to_tokens(hist_avail.float()),
                                                  hq, hknown), dim=-1))
        current = self.current_projection(torch.cat((raw_tokens, cand_tokens,
                                                     self._mask_to_tokens(raw_avail.float()),
                                                     self._mask_to_tokens(cand_avail.float()),
                                                     q, known), dim=-1))
        tokens = torch.cat((hist, current[:, None]), dim=1)
        if beta is None:
            shape_condition = torch.zeros(b, 11, device=raw.device)
        else:
            self._shape(beta, (b, 10), "beta")
            if beta_valid is None:
                beta_valid = torch.ones(b, device=raw.device, dtype=torch.bool)
            self._shape(beta_valid, (b,), "beta_valid")
            beta_value = torch.where(beta_valid.bool()[:, None], beta.float(), 0.0)
            self._finite(beta_value, "beta")
            shape_condition = torch.cat((beta_value, beta_valid.float()[:, None]), dim=-1)
        condition = (self.shape_projection(shape_condition)
                     + self.handedness_embedding(handedness.long()))
        dt = torch.cat((torch.where(history_valid, dt[:, :h], 0.0), dt[:, h:]), dim=1)
        times = dt.cumsum(-1)
        times = times - times[:, -1:]
        temporal = self.time_projection(torch.stack((times, dt, torch.log1p(dt)), -1))
        joints = self.joint_embedding(torch.arange(16, device=raw.device))
        tokens = tokens + temporal[:, :, None] + joints[None, None] + condition[:, None, None]
        frame_valid = torch.cat((history_valid,
                                 torch.ones(b, 1, device=raw.device, dtype=torch.bool)), dim=1)
        encoded, moe_balance, moe_load = self.encoder(tokens, frame_valid)
        features = encoded[:, -1]
        dp = self.translation_head(features[:, 0]).float() * self.config.translation_scale
        dr = self.rotation_head(features).float() * self.config.rotation_scale
        delta = torch.cat((dp, dr.flatten(-2)), dim=-1)
        state = g.state_apply_delta(base, delta)
        output = {"state": state, "state_features": features, "delta": delta,
                  "state_translation_std": self._translation_std(
                      self.state_translation_scale_head(features[:, 0])),
                  "state_rotation_concentration": self._concentration(
                      self.state_rotation_concentration_head(features).squeeze(-1)),
                  "moe_balance_loss": moe_balance, "moe_load": moe_load}
        self._finite(state, "restored state")
        if not return_forecast:
            return output

        c, f, k = self.config, self.config.forecast_horizon, self.config.num_modes
        if forecast_dt is None:
            forecast_dt = dt[:, -1:] * torch.arange(1, f + 1, device=dt.device)[None]
        self._shape(forecast_dt, (b, f), "forecast_dt")
        forecast_dt = forecast_dt.float()
        self._finite(forecast_dt, "forecast_dt")
        step_dt = torch.diff(forecast_dt, prepend=forecast_dt.new_zeros(b, 1), dim=-1)
        if (step_dt <= 0).any():
            raise ValueError("forecast_dt must contain strictly increasing positive offsets")
        future_time = self.time_projection(torch.stack(
            (forecast_dt, step_dt, torch.log1p(step_dt)), -1))
        pooled = features.mean(1)
        future = (self.future_context(pooled)[:, None, None]
                  + self.mode_embedding.weight[None, :, None]
                  + future_time[:, None]).reshape(b * k, f, c.d_model)
        for block in self.future_decoder:
            future = block(future)
        future = self.future_norm(future).reshape(b, k, f, c.d_model)
        # Joint conditioning uses a 64D factor rather than another full
        # [B,K,F,16,256] temporal backbone.
        joint_features = self.future_joint_norm(
            self.future_joint_projection(features)[:, None, None]
            + self.future_time_projection(future)[:, :, :, None])
        vp = self.future_translation_velocity_head(joint_features[..., 0, :]).float()
        vr = self.future_rotation_velocity_head(joint_features).float().flatten(-2)
        velocity = torch.cat((vp * c.translation_velocity_scale,
                              vr * c.rotation_velocity_scale), -1)
        forecast = self._integrate_forecast(state, velocity, step_dt)
        logits = self.mixture_head(pooled).float()
        selected = logits.argmax(-1)
        output.update({"forecast_modes": forecast,
                       "forecast": forecast[torch.arange(b, device=raw.device), selected],
                       "mixture_logits": logits, "mixture_weights": logits.softmax(-1),
                       "forecast_velocity": velocity,
                       "forecast_translation_std": self._translation_std(
                           self.future_translation_scale_head(joint_features[..., 0, :])),
                       "forecast_rotation_concentration": self._concentration(
                           self.future_rotation_concentration_head(joint_features).squeeze(-1))})
        self._finite(state, "restored state")
        self._finite(forecast, "forecast state")
        return output

    @staticmethod
    def _integrate_forecast(state: Tensor, velocity: Tensor, step_dt: Tensor) -> Tensor:
        # Quaternion products preserve the actual sequence of noncommuting
        # Exp increments. Summing axis-angle vectors would not do so.
        with torch.autocast(device_type=state.device.type, enabled=False):
            b, k, f, _ = velocity.shape
            increments = velocity.float() * step_dt[:, None, :, None]
            positions = state[:, None, None, :3] + increments[..., :3].cumsum(2)
            dq = g.axis_angle_to_quaternion(increments[..., 3:].reshape(b, k, f, 16, 3))
            quaternion = g.matrix_to_quaternion(g.state_to_rotmat(state))[:, None].expand(-1, k, -1, -1)
            rotations = []
            for index in range(f):
                quaternion = g.normalize_quaternion(g.quaternion_multiply(dq[:, :, index], quaternion))
                rotations.append(g.quaternion_to_matrix(quaternion))
            return g.state_from_rotmat(positions, torch.stack(rotations, dim=2))
