"""Motion-only TWM for CURRENT 99D restoration and auxiliary forecasting.

There are no RGB/features, camera features, contact assumptions or force heads.
The caller supplies one common coordinate frame for history/raw/candidate; this
module neither chooses an anchor nor overwrites raw observations/history.
Each frame has sixteen anatomical tokens (root + fifteen MANO local joints).
The root token additionally contains wrist J0 translation. A null context token
prevents all-masked attention for padded history; it contains no measurement.
Attention sees past/current frames only. Same-frame joints attend each other.
The structured_mamba option keeps joint and time axes separate and adds the
MANO tree encoding. The original flat_transformer remains for saved pilots.

Availability/quality use SEVENTEEN groups: translation, root rotation, then the
15 finger rotations. Availability is separate from confidence and history
padding. Missing quality is explicitly encoded, never equated to confidence=1.
Final-state reliability is NOT predicted by this minimal baseline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn

from handloop.geometry import normalize_state, state_apply_delta


@dataclass
class TWMConfig:
    backbone: str = "flat_transformer"
    d_model: int = 128
    num_layers: int = 4
    nhead: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1
    forecast_horizon: int = 1
    translation_scale: float = 0.1  # metres per unit restoration head output
    position_feature_scale: float = 1.0  # metres per encoder feature unit; legacy default
    rotation_scale: float = 0.5     # radians per unit restoration head output
    check_finite: bool = True
    mamba_state_size: int = 64
    mamba_conv: int = 4
    mamba_expand: int = 2


class TWM(nn.Module):
    """Shared left/right hand model. Handedness codes: 0=left, 1=right, 2=unknown.

    forward returns a dict:
      state: (B,99), restored CURRENT state, float32 legal rotations;
      forecast: (B,F,99), auxiliary predictions at forecast_dt after current;
      state_features: (B,16,D), current anatomical features;
      delta: (B,51), current tangent update (not the public state encoding).

    Heads initialize to exactly zero: current output starts at the normalized
    candidate; forecast starts with a constant-state baseline. The network
    must learn to change accurate/incorrect observations from supervision.
    """

    def __init__(self, config: Optional[TWMConfig] = None):
        super().__init__()
        self.config = config or TWMConfig()
        c = self.config
        if c.d_model <= 0 or c.nhead <= 0 or c.d_model % c.nhead:
            raise ValueError("d_model must be positive and divisible by nhead")
        if c.num_layers < 1 or c.forecast_horizon < 0:
            raise ValueError("num_layers >= 1 and forecast_horizon >= 0 required")
        if c.backbone not in ("flat_transformer", "structured_mamba"):
            raise ValueError(f"Unknown backbone: {c.backbone}")
        if not math.isfinite(c.position_feature_scale) or c.position_feature_scale <= 0:
            raise ValueError("position_feature_scale must be finite and positive")
        # Geometry9 + availability2 + quality2 + quality-known2.
        self.history_projection = nn.Linear(15, c.d_model)
        # raw9 + candidate9 + raw-mask2 + candidate-mask2 + quality2 + known2.
        self.current_projection = nn.Linear(26, c.d_model)
        self.joint_embedding = nn.Embedding(16, c.d_model)
        self.handedness_embedding = nn.Embedding(3, c.d_model)
        self.shape_projection = nn.Linear(11, c.d_model)  # beta10 + known flag
        self.time_projection = nn.Sequential(nn.Linear(3, c.d_model), nn.SiLU(),
                                             nn.Linear(c.d_model, c.d_model))
        if c.backbone == "structured_mamba":
            from .structured_mamba import StructuredMambaEncoder
            self.encoder = StructuredMambaEncoder(
                d_model=c.d_model, num_layers=c.num_layers, nhead=c.nhead,
                dim_feedforward=c.dim_feedforward, dropout=c.dropout,
                d_state=c.mamba_state_size, d_conv=c.mamba_conv,
                expand=c.mamba_expand,
            )
        else:
            self.null_token = nn.Parameter(torch.zeros(1, 1, c.d_model))
            layer = nn.TransformerEncoderLayer(
                d_model=c.d_model, nhead=c.nhead, dim_feedforward=c.dim_feedforward,
                dropout=c.dropout, activation="gelu", batch_first=True, norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, c.num_layers,
                                                 norm=nn.LayerNorm(c.d_model),
                                                 enable_nested_tensor=False)
        self.translation_head = nn.Linear(c.d_model, 3)
        self.rotation_head = nn.Linear(c.d_model, 3)
        self.forecast_translation_head = nn.Linear(c.d_model, 3)
        self.forecast_rotation_head = nn.Linear(c.d_model, 3)
        # TransformerEncoder clones its template with identical initial values;
        # initialize each attention/FF block independently.
        if c.backbone == "flat_transformer":
            for block in self.encoder.layers:
                nn.init.xavier_uniform_(block.self_attn.in_proj_weight)
                nn.init.xavier_uniform_(block.self_attn.out_proj.weight)
                nn.init.xavier_uniform_(block.linear1.weight)
                nn.init.xavier_uniform_(block.linear2.weight)
        for head in (self.translation_head, self.rotation_head,
                     self.forecast_translation_head, self.forecast_rotation_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    @staticmethod
    def _shape(x: Tensor, expected: tuple, name: str) -> None:
        if tuple(x.shape) != expected:
            raise ValueError(f"{name}: expected {expected}, got {tuple(x.shape)}")

    def _finite(self, x: Tensor, name: str) -> None:
        if self.config.check_finite and not torch.isfinite(x).all():
            raise ValueError(f"{name} contains non-finite available values")

    @staticmethod
    def _mask_to_tokens(group_values: Tensor) -> Tensor:
        """17 groups -> 16x2 channels (translation-only-at-root, rotation)."""
        p = torch.zeros_like(group_values[..., 1:])
        p[..., 0] = group_values[..., 0]
        return torch.stack((p, group_values[..., 1:]), dim=-1)

    def _state_tokens(self, state: Tensor, availability: Tensor) -> tuple[Tensor, Tensor]:
        """Suppress unavailable placeholders, then normalize rotation blocks."""
        state = state.float()
        p = torch.where(availability[..., :1], state[..., :3], 0.0)
        r = state[..., 3:].reshape(*state.shape[:-1], 16, 6)
        r = torch.where(availability[..., 1:, None], r, 0.0)
        clean = torch.cat((p, r.flatten(-2)), dim=-1)
        self._finite(clean, "state")
        clean = normalize_state(clean)
        token_p = torch.zeros(*clean.shape[:-1], 16, 3, device=clean.device)
        token_p[..., 0, :] = clean[..., :3] / self.config.position_feature_scale
        return torch.cat((token_p, clean[..., 3:].reshape(*clean.shape[:-1], 16, 6)), -1), clean

    def _quality(self, quality: Optional[Tensor], availability: Tensor) -> tuple[Tensor, Tensor]:
        if quality is None:
            q = torch.zeros_like(availability, dtype=torch.float32)
            known = torch.zeros_like(availability)
        else:
            self._shape(quality, tuple(availability.shape), "quality")
            q = quality.float()
            # Negative values and NaN mean explicitly unknown; infinity is invalid.
            if self.config.check_finite and (torch.isinf(q).any() or (q > 1).any()):
                raise ValueError("quality must be [0,1], negative/NaN for unknown")
            known = torch.isfinite(q) & (q >= 0) & availability
            q = torch.where(known, q, 0.0)
        return self._mask_to_tokens(q), self._mask_to_tokens(known.float())

    def forward(
        self,
        history: Tensor,
        raw: Tensor,
        candidate: Tensor,
        *,
        history_valid: Tensor,
        history_availability: Tensor,
        availability: Tensor,
        candidate_availability: Tensor,
        dt: Tensor,
        beta: Optional[Tensor],
        handedness: Tensor,
        history_quality: Optional[Tensor] = None,
        quality: Optional[Tensor] = None,
        forecast_dt: Optional[Tensor] = None,
        beta_valid: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        """Use ONLY completed past states and current numerical observations.

        history (B,H,99) is chronological; H can be zero. history_valid (B,H)
        marks real history slots, not correctness. Availability tensors are
        bool (B,H,17)/(B,17). Candidate availability describes filled/initialized
        slots independently of raw availability; neither mask is modified.

        dt (B,H+1), seconds: dt[:,i] is elapsed time into frame i. dt[:,0]
        does not affect relative timestamps, but remains a sample-rate feature.
        Current dt must be positive. Padding intervals must be finite >=0;
        structured_mamba ignores their intervals by setting them to zero. Real
        missed observations retain valid time slots with availability masks.
        beta (B,10) is sequence/person shape, or None when unknown. beta_valid
        (B,) distinguishes an actual zero shape from missing shape.
        forecast_dt (B,F) contains positive offsets AFTER current; these are
        time offsets only, never future states. Defaults to multiples of last dt.
        All state coordinates must already use the SAME world/anchor frame.
        """
        if history.ndim != 3 or history.shape[-1] != 99:
            raise ValueError("history must be (B,H,99)")
        b, h, _ = history.shape
        self._shape(raw, (b, 99), "raw")
        self._shape(candidate, (b, 99), "candidate")
        self._shape(history_valid, (b, h), "history_valid")
        self._shape(history_availability, (b, h, 17), "history_availability")
        self._shape(availability, (b, 17), "availability")
        self._shape(candidate_availability, (b, 17), "candidate_availability")
        self._shape(dt, (b, h+1), "dt")
        self._shape(handedness, (b,), "handedness")
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
        if self.config.backbone == "structured_mamba":
            dt = torch.cat((torch.where(history_valid, dt[:, :h], 0.0), dt[:, h:]), dim=1)
        times = dt.cumsum(-1)
        times = times - times[:, -1:]
        temporal = self.time_projection(torch.stack((times, dt, torch.log1p(dt)), -1))
        joints = self.joint_embedding(torch.arange(16, device=raw.device))
        tokens = tokens + temporal[:, :, None] + joints[None, None] + condition[:, None, None]
        if self.config.backbone == "structured_mamba":
            frame_valid = torch.cat((history_valid,
                                     torch.ones(b, 1, device=raw.device, dtype=torch.bool)), dim=1)
            encoded = self.encoder(tokens, frame_valid)
            features = encoded[:, -1]
        else:
            tokens = tokens.flatten(1, 2)
            tokens = torch.cat((self.null_token.expand(b, -1, -1), tokens), dim=1)
            frame_ids = torch.cat((torch.tensor([-1], device=raw.device),
                                   torch.arange(h+1, device=raw.device).repeat_interleave(16)))
            causal_mask = frame_ids[None, :] > frame_ids[:, None]
            padding = torch.cat((torch.zeros(b, 1, device=raw.device, dtype=torch.bool),
                                 (~history_valid).repeat_interleave(16, dim=1),
                                 torch.zeros(b, 16, device=raw.device, dtype=torch.bool)), dim=1)
            encoded = self.encoder(tokens, mask=causal_mask, src_key_padding_mask=padding)
            features = encoded[:, -16:]
        dp = self.translation_head(features[:, 0]).float() * self.config.translation_scale
        dr = self.rotation_head(features).float() * self.config.rotation_scale
        delta = torch.cat((dp, dr.flatten(-2)), dim=-1)
        state = state_apply_delta(base, delta)

        f = self.config.forecast_horizon
        if forecast_dt is None:
            forecast_dt = dt[:, -1:] * torch.arange(1, f+1, device=dt.device)[None]
        self._shape(forecast_dt, (b, f), "forecast_dt")
        forecast_dt = forecast_dt.float()
        self._finite(forecast_dt, "forecast_dt")
        if (forecast_dt <= 0).any():
            raise ValueError("forecast_dt offsets must be positive")
        horizon_features = self.time_projection(torch.stack(
            (forecast_dt, forecast_dt, torch.log1p(forecast_dt)), dim=-1))
        future_features = features[:, None] + horizon_features[:, :, None]
        future_dp = self.forecast_translation_head(future_features[:, :, 0]).float()
        future_dr = self.forecast_rotation_head(future_features).float().flatten(-2)
        # Auxiliary head predicts average tangent rate over each requested offset.
        future_delta = torch.cat((future_dp, future_dr), -1) * forecast_dt[..., None]
        forecast = state_apply_delta(state[:, None].expand(-1, f, -1), future_delta)
        self._finite(state, "restored state")
        self._finite(forecast, "forecast state")
        return {"state": state, "forecast": forecast,
                "state_features": features, "delta": delta}
