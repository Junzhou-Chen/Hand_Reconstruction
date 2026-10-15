"""D33 visual readout feedback around a frozen, numerical-only TWM.

The original HaWoR path runs once per image block, independently of feedback.
This module consumes its REAL cached patch features and immutable raw states,
then processes frames in chronological order. Only a completed final state is
committed, once, for use by later frames. No target/GT argument exists on the
forward path. Original HaWoR lookahead is retained: this is not an online claim.

Coordinates use column vectors, metres, anatomical wrist J0 and column 6D.
The visual adapter reads camera/crop geometry. TWM receives numerical states
in ONE previous-wrist anchor, with no image or camera feature tensors. Its
diagonal Gaussian and final NLL are defined in that anchor, not world axes.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Optional

import torch
from torch import Tensor, nn

from handloop import geometry as g
from .probabilistic_twm import ProbabilisticTWM, current_distribution_log_prob


@dataclass
class VisualFeedbackConfig:
    image_feature_dim: int = 1280
    patch_grid_height: int = 16
    patch_grid_width: int = 12
    d_model: int = 256
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 512
    dropout: float = 0.0
    history_length: int = 30
    max_gap_seconds: float = 0.25
    default_dt: float = 1.0 / 30.0
    translation_delta_scale: float = 0.1
    rotation_delta_scale: float = 0.5
    detach_history: bool = True
    condition_on_beta: bool = False


def transform_state(state: Tensor, transform: Tensor) -> Tensor:
    """Apply rigid output-frame <- input-frame transform; fingers stay local."""
    if state.shape[-1:] != (99,) or transform.shape[-2:] != (4, 4):
        raise ValueError("expected state[...,99] and transform[...,4,4]")
    with torch.autocast(device_type=state.device.type, enabled=False):
        transform = transform.float()
        while transform.ndim < state.ndim + 1:
            transform = transform.unsqueeze(-3)
        rotation, translation = transform[..., :3, :3], transform[..., :3, 3]
        position = (rotation @ state[..., :3].float().unsqueeze(-1)).squeeze(-1) + translation
        rotations = g.state_to_rotmat(state)
        root = rotation @ rotations[..., 0, :, :]
        return g.state_from_rotmat(position, torch.cat((root.unsqueeze(-3), rotations[..., 1:, :, :]), -3))


def inverse_rigid(transform: Tensor) -> Tensor:
    with torch.autocast(device_type=transform.device.type, enabled=False):
        rotation = transform[..., :3, :3].float().transpose(-1, -2)
        translation = -(rotation @ transform[..., :3, 3:4].float()).squeeze(-1)
        result = torch.zeros_like(transform, dtype=torch.float32)
        result[..., :3, :3] = rotation
        result[..., :3, 3] = translation
        result[..., 3, 3] = 1
        return result


def state_anchor_world(state: Tensor) -> Tensor:
    """World <- anchor from a completed wrist, or an explicit warm-start state."""
    result = torch.zeros(*state.shape[:-1], 4, 4, device=state.device, dtype=torch.float32)
    result[..., :3, :3] = g.state_to_rotmat(state)[..., 0, :, :]
    result[..., :3, 3] = state[..., :3]
    result[..., 3, 3] = 1
    return result


def _masked_state(state: Tensor, availability: Tensor) -> Tensor:
    """Ignore unavailable NaN placeholders before any geometric operation."""
    position = torch.where(availability[..., :1], state[..., :3].float(), 0.0)
    rotation = state[..., 3:].float().reshape(*state.shape[:-1], 16, 6)
    rotation = torch.where(availability[..., 1:, None], rotation, 0.0)
    clean = torch.cat((position, rotation.flatten(-2)), -1)
    if not bool(torch.isfinite(clean).all()):
        raise ValueError("available state groups must be finite")
    return g.normalize_state(clean)


def _state_tokens(state: Tensor, availability: Tensor) -> Tensor:
    state = _masked_state(state, availability)
    position = torch.zeros(*state.shape[:-1], 16, 3, device=state.device)
    position[..., 0, :] = state[..., :3]
    # Zero unavailable normalized rotation features rather than encoding the
    # deterministic identity placeholder as if it had been observed.
    rotation = state[..., 3:].reshape(*state.shape[:-1], 16, 6)
    rotation = torch.where(availability[..., 1:, None], rotation, 0.0)
    position = torch.where(availability[..., :1, None], position, 0.0)
    return torch.cat((position, rotation), -1)


def _group_tokens(values: Tensor) -> Tensor:
    position = torch.zeros_like(values[..., 1:])
    position[..., 0] = values[..., 0]
    return torch.stack((position, values[..., 1:]), -1)


@dataclass
class FeedbackHistory:
    """Only completed predictions; valid slots form a chronological suffix.

    uncertainty stores translation RMS standard deviation followed by sixteen
    rotation concentrations. It is a model distribution descriptor, not a
    calibrated correctness probability or TWM's observation quality input.
    """

    state_world: Tensor
    valid: Tensor
    availability: Tensor
    timestamps: Tensor
    uncertainty: Tensor
    uncertainty_known: Tensor
    handedness: Tensor

    @classmethod
    def empty(cls, batch_size: int, length: int, *, device: torch.device) -> "FeedbackHistory":
        return cls(g.identity_state(batch_size, length, device=device),
                   torch.zeros(batch_size, length, dtype=torch.bool, device=device),
                   torch.zeros(batch_size, length, 17, dtype=torch.bool, device=device),
                   torch.zeros(batch_size, length, dtype=torch.float64, device=device),
                   torch.zeros(batch_size, length, 17, device=device),
                   torch.zeros(batch_size, length, 17, dtype=torch.bool, device=device),
                   torch.full((batch_size,), 2, dtype=torch.long, device=device))

    def detach(self) -> "FeedbackHistory":
        return FeedbackHistory(**{field.name: getattr(self, field.name).detach() for field in fields(self)})

    def reset(self, reset: Tensor) -> "FeedbackHistory":
        values = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name == "handedness":
                replacement = torch.full_like(value, 2)
            elif field.name == "state_world":
                replacement = g.identity_state(*value.shape[:-1], device=value.device)
            else:
                replacement = torch.zeros_like(value)
            mask = reset.reshape(reset.shape[0], *([1] * (value.ndim - 1)))
            values[field.name] = torch.where(mask, replacement, value)
        return FeedbackHistory(**values)

    def commit(self, state_world: Tensor, timestamp: Tensor, translation_std: Tensor,
               rotation_concentration: Tensor, commit: Tensor, handedness: Tensor,
               *, detach: bool = True) -> "FeedbackHistory":
        rms = translation_std.float().square().mean(-1, keepdim=True).sqrt()
        uncertainty = torch.cat((rms, rotation_concentration.float()), -1)
        additions = {"state_world": state_world,
                     "valid": torch.ones_like(commit),
                     "availability": torch.ones(*commit.shape, 17, dtype=torch.bool, device=commit.device),
                     "timestamps": timestamp.double(), "uncertainty": uncertainty,
                     "uncertainty_known": torch.ones_like(uncertainty, dtype=torch.bool)}
        values = {}
        for name, value in additions.items():
            if detach:
                value = value.detach()
            old = getattr(self, name)
            shifted = torch.cat((old[:, 1:], value[:, None]), 1)
            mask = commit.reshape(commit.shape[0], *([1] * (old.ndim - 1)))
            values[name] = torch.where(mask, shifted, old)
        # Unknown labels do not erase the last known left/right identity.
        values["handedness"] = torch.where(commit & (handedness != 2),
                                            handedness.long(), self.handedness)
        return FeedbackHistory(**values)


class _ReadoutBlock(nn.Module):
    def __init__(self, config: VisualFeedbackConfig):
        super().__init__()
        d = config.d_model
        self.history_norm = nn.LayerNorm(d)
        self.history_attention = nn.MultiheadAttention(d, config.nhead, config.dropout, batch_first=True)
        self.image_norm = nn.LayerNorm(d)
        self.image_attention = nn.MultiheadAttention(d, config.nhead, config.dropout, batch_first=True)
        self.joint_norm = nn.LayerNorm(d)
        self.joint_attention = nn.MultiheadAttention(d, config.nhead, config.dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, config.dim_feedforward), nn.GELU(),
                                 nn.Dropout(config.dropout), nn.Linear(config.dim_feedforward, d))

    def forward(self, query: Tensor, history: Tensor, history_padding: Tensor,
                image: Tensor, image_padding: Tensor) -> Tensor:
        query = query + self.history_attention(self.history_norm(query), history, history,
                                               key_padding_mask=history_padding, need_weights=False)[0]
        query = query + self.image_attention(self.image_norm(query), image, image,
                                             key_padding_mask=image_padding, need_weights=False)[0]
        q = self.joint_norm(query)
        query = query + self.joint_attention(q, q, q, need_weights=False)[0]
        return query + self.ffn(self.ffn_norm(query))


class VisualFeedbackAdapter(nn.Module):
    """Sixteen state-conditioned queries over actual HaWoR crop patches."""

    def __init__(self, config: Optional[VisualFeedbackConfig] = None):
        super().__init__()
        self.config = config or VisualFeedbackConfig()
        c = self.config
        if c.d_model <= 0 or c.nhead <= 0 or c.d_model % c.nhead or c.num_layers < 1:
            raise ValueError("positive width divisible by heads and at least one layer required")
        if min(c.patch_grid_height, c.patch_grid_width, c.history_length) < 1:
            raise ValueError("patch grid and history length must be positive")
        self.joint_embedding = nn.Embedding(16, c.d_model)
        self.handedness_embedding = nn.Embedding(3, c.d_model)
        self.raw_projection = nn.Linear(11, c.d_model)
        self.history_projection = nn.Linear(15, c.d_model)
        # [relative seconds, dt, log1p(dt), sin/cos(2pi * t * {1,4})].
        self.time_projection = nn.Linear(7, c.d_model)
        # Crop-normalized wrist u/v, camera depth, and a projection-valid bit.
        self.camera_projection = nn.Linear(4, c.d_model)
        self.image_projection = nn.Sequential(nn.LayerNorm(c.image_feature_dim),
                                               nn.Linear(c.image_feature_dim, c.d_model))
        self.patch_position_projection = nn.Linear(2, c.d_model)
        self.null_history = nn.Parameter(torch.zeros(1, 1, c.d_model))
        self.null_image = nn.Parameter(torch.zeros(1, 1, c.d_model))
        rows, columns = torch.meshgrid((torch.arange(c.patch_grid_height) + 0.5) * (2 / c.patch_grid_height) - 1,
                                      (torch.arange(c.patch_grid_width) + 0.5) * (2 / c.patch_grid_width) - 1,
                                      indexing="ij")
        self.register_buffer("patch_positions", torch.stack((columns, rows), -1).reshape(-1, 2))
        self.blocks = nn.ModuleList([_ReadoutBlock(c) for _ in range(c.num_layers)])
        self.final_norm = nn.LayerNorm(c.d_model)
        self.translation_head = nn.Linear(c.d_model, 3)
        self.rotation_head = nn.Linear(c.d_model, 3)
        # Only the last projections are zero. There is no second zero gate.
        for head in (self.translation_head, self.rotation_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    @staticmethod
    def _crop_wrist(state_camera: Tensor, intrinsics: Tensor, crop_transform: Tensor,
                    valid: Tensor) -> Tensor:
        with torch.autocast(device_type=state_camera.device.type, enabled=False):
            xyz = state_camera[..., :3].float()
            pixel_h = (intrinsics.float() @ xyz.unsqueeze(-1)).squeeze(-1)
            projected = valid & (xyz[..., 2] > 1e-5) & torch.isfinite(pixel_h).all(-1)
            denominator = torch.where(projected, pixel_h[..., 2], 1.0)
            pixel_h = pixel_h / denominator.unsqueeze(-1)
            crop_h = (crop_transform.float() @ pixel_h.unsqueeze(-1)).squeeze(-1)
            denominator = torch.where(crop_h[..., 2].abs() > 1e-6, crop_h[..., 2], 1.0)
            crop = crop_h[..., :2] / denominator.unsqueeze(-1)
            # crop_transform maps source pixels to NORMALIZED crop coordinates
            # [-1,1], so coordinates agree with patch_positions above.
            descriptor = torch.cat((crop, xyz[..., 2:3], projected.float().unsqueeze(-1)), -1)
            return torch.where(projected.unsqueeze(-1), descriptor, 0.0)

    def forward(self, patch_features: Tensor, raw_camera: Tensor, history_camera: Tensor, *,
                availability: Tensor, history_availability: Tensor, history_valid: Tensor,
                history_uncertainty: Tensor, history_uncertainty_known: Tensor,
                relative_time: Tensor, history_dt: Tensor, image_valid: Tensor,
                handedness: Tensor, camera_intrinsics: Tensor,
                crop_transform: Tensor) -> Tensor:
        b, patches, channels = patch_features.shape
        c = self.config
        if (patches, channels) != (c.patch_grid_height * c.patch_grid_width, c.image_feature_dim):
            raise ValueError("patch_features must match configured HaWoR patch grid and width")
        safe_patches = torch.where(image_valid[:, None, None], patch_features.detach().float(), 0.0)
        if not bool(torch.isfinite(safe_patches).all()):
            raise ValueError("valid cached image patches must be finite")
        joint = self.joint_embedding.weight[None]
        query = self.raw_projection(torch.cat((_state_tokens(raw_camera, availability),
                                              _group_tokens(availability.float())), -1)) + joint
        query = query + self.handedness_embedding(handedness.long())[:, None]
        camera = self._crop_wrist(raw_camera, camera_intrinsics, crop_transform, availability[:, 0])
        query = query + self.camera_projection(camera)[:, None]
        known = history_uncertainty_known & history_availability & history_valid[..., None]
        uncertainty = torch.where(known, history_uncertainty, 1.0).clamp_min(1e-8).log()
        history = self.history_projection(torch.cat((
            _state_tokens(history_camera, history_availability),
            _group_tokens(history_availability.float()), _group_tokens(uncertainty),
            _group_tokens(known.float())), -1)) + joint[:, None]
        phase = relative_time[..., None] * relative_time.new_tensor([2 * torch.pi, 8 * torch.pi])
        time = torch.cat((relative_time[..., None], history_dt[..., None],
                          torch.log1p(history_dt)[..., None], phase.sin(), phase.cos()), -1)
        history = history + self.time_projection(time.float())[:, :, None]
        # Historical wrists are projected through the CURRENT camera and crop.
        camera_history = self._crop_wrist(history_camera,
                                          camera_intrinsics[:, None], crop_transform[:, None],
                                          history_valid & history_availability[..., 0])
        history = history + self.camera_projection(camera_history)[:, :, None]
        history = torch.cat((self.null_history.expand(b, -1, -1), history.flatten(1, 2)), 1)
        history_padding = torch.cat((torch.zeros(b, 1, dtype=torch.bool, device=history.device),
                                     (~history_valid).repeat_interleave(16, -1)), -1)
        image = self.image_projection(safe_patches) + self.patch_position_projection(self.patch_positions)[None]
        image = torch.cat((self.null_image.expand(b, -1, -1), image), 1)
        image_padding = torch.cat((torch.zeros(b, 1, dtype=torch.bool, device=image.device),
                                   (~image_valid)[:, None].expand(b, patches)), -1)
        for block in self.blocks:
            query = block(query, history, history_padding, image, image_padding)
        query = self.final_norm(query)
        delta = torch.cat((self.translation_head(query[:, 0]).float() * c.translation_delta_scale,
                           self.rotation_head(query).float().flatten(-2) * c.rotation_delta_scale), -1)
        # No actual image means no new visual update. History-filled candidate
        # and the TWM can still produce an explicitly inferred current state.
        return torch.where(image_valid[:, None], delta, 0.0)


class VisualFeedbackSystem(nn.Module):
    """Sequential post-PAM visual plugin + final probabilistic TWM readout."""

    def __init__(self, twm: ProbabilisticTWM, config: Optional[VisualFeedbackConfig] = None,
                 *, train_current_readout: bool = False):
        super().__init__()
        self.config = config or VisualFeedbackConfig()
        if self.config.default_dt <= 0 or self.config.max_gap_seconds <= 0:
            raise ValueError("default_dt and max_gap_seconds must be positive")
        self.adapter = VisualFeedbackAdapter(self.config)
        self.twm = twm
        self.configure_trainable("current_readout" if train_current_readout else "adapter")

    def configure_trainable(self, mode: str = "adapter") -> "VisualFeedbackSystem":
        """Keep motion core/future heads fixed, optionally unfreeze current heads."""
        if mode not in ("adapter", "current_readout"):
            raise ValueError("trainable mode must be adapter or current_readout")
        self.adapter.requires_grad_(True)
        self.twm.requires_grad_(False)
        if mode == "current_readout":
            for name in ("translation_head", "rotation_head", "state_translation_scale_head",
                         "state_rotation_concentration_head"):
                getattr(self.twm, name).requires_grad_(True)
        self.twm.eval()
        return self

    def train(self, mode: bool = True) -> "VisualFeedbackSystem":
        super().train(mode)
        # Freezing parameters alone would leave frozen dropout active.
        self.twm.eval()
        return self

    def empty_history(self, batch_size: int, *, device: torch.device) -> FeedbackHistory:
        return FeedbackHistory.empty(batch_size, self.config.history_length, device=device)

    def step(self, patch_features: Tensor, raw_world: Tensor, *, timestamps: Tensor,
             frame_valid: Tensor, availability: Tensor, image_valid: Tensor, handedness: Tensor,
             world_to_camera: Tensor, camera_intrinsics: Tensor, crop_transform: Tensor,
             beta: Optional[Tensor] = None, beta_valid: Optional[Tensor] = None,
             reset: Optional[Tensor] = None, history: Optional[FeedbackHistory] = None,
             return_forecast: bool = False, use_adapter: bool = True,
             adapter_history: bool = True) -> tuple[dict[str, Tensor], FeedbackHistory]:
        b = raw_world.shape[0]
        if raw_world.shape != (b, 99):
            raise ValueError("raw_world must be [B,99]")
        expected = ((timestamps, (b,), "timestamps"), (frame_valid, (b,), "frame_valid"),
                    (availability, (b, 17), "availability"), (image_valid, (b,), "image_valid"),
                    (handedness, (b,), "handedness"), (world_to_camera, (b, 4, 4), "world_to_camera"),
                    (camera_intrinsics, (b, 3, 3), "camera_intrinsics"),
                    (crop_transform, (b, 3, 3), "crop_transform"))
        for value, shape, name in expected:
            if tuple(value.shape) != shape:
                raise ValueError(f"{name} must be {shape}")
        if bool(((handedness < 0) | (handedness > 2)).any()):
            raise ValueError("handedness must be 0=left,1=right,2=unknown")
        active = frame_valid.bool()
        availability = availability.bool() & active[:, None]
        image_valid = image_valid.bool() & active
        history = history or self.empty_history(b, device=raw_world.device)
        if history.valid.shape != (b, self.config.history_length):
            raise ValueError("initial history has incompatible batch size/length")
        if not bool(torch.isfinite(torch.where(active, timestamps.double(), 0.0)).all()):
            raise ValueError("valid frame timestamps must be finite seconds")
        has_history = history.valid[:, -1]
        elapsed = timestamps.double() - history.timestamps[:, -1]
        changed_hand = ((handedness != history.handedness) & (handedness != 2)
                        & (history.handedness != 2))
        explicit_reset = torch.zeros_like(active) if reset is None else reset.bool()
        resets = active & (explicit_reset | changed_hand
                           | (has_history & (elapsed > self.config.max_gap_seconds)))
        if bool((active & has_history & (elapsed <= 0) & ~resets).any()):
            raise ValueError("completed-history timestamps must precede the current frame")
        history = history.reset(resets)
        has_history = history.valid[:, -1]
        timestamps = torch.where(active, timestamps.double(),
                                 history.timestamps[:, -1] + self.config.default_dt)
        raw_clean = _masked_state(raw_world.detach(), availability)
        last = history.state_world[:, -1]
        last_availability = history.availability[:, -1] & has_history[:, None]
        position = torch.where(availability[:, :1], raw_clean[:, :3], last[:, :3])
        raw_r = raw_clean[:, 3:].reshape(b, 16, 6)
        last_r = last[:, 3:].reshape(b, 16, 6)
        rotation = torch.where(availability[:, 1:, None], raw_r, last_r)
        base_world = torch.cat((position, rotation.flatten(-2)), -1)
        candidate_availability = availability | last_availability
        # Fixed throughout this frame; no current output changes its own anchor.
        anchor_state = torch.where(has_history[:, None], last, raw_clean)
        anchor_world = state_anchor_world(anchor_state).detach()
        world_to_anchor = inverse_rigid(anchor_world)
        raw_anchor = transform_state(raw_clean, world_to_anchor)
        base_anchor = transform_state(base_world, world_to_anchor)
        history_anchor = transform_state(history.state_world, world_to_anchor)
        # Padded camera metadata may be NaN; replace it before any multiplication.
        eye4 = torch.eye(4, device=raw_world.device).expand(b, -1, -1)
        eye3 = torch.eye(3, device=raw_world.device).expand(b, -1, -1)
        camera = torch.where(active[:, None, None], world_to_camera.float(), eye4)
        intrinsic = torch.where(active[:, None, None], camera_intrinsics.float(), eye3)
        crop = torch.where(active[:, None, None], crop_transform.float(), eye3)
        if not all(bool(torch.isfinite(value).all()) for value in (camera, intrinsic, crop)):
            raise ValueError("valid camera/crop metadata must be finite")
        hist_dt = torch.full_like(history.timestamps, self.config.default_dt)
        hist_dt[:, 1:] = torch.where(history.valid[:, :-1] & history.valid[:, 1:],
                                    history.timestamps[:, 1:] - history.timestamps[:, :-1],
                                    self.config.default_dt)
        hist_dt = torch.where(history.valid, hist_dt, 0.0).float()
        current_dt = torch.where(has_history, timestamps - history.timestamps[:, -1],
                                 self.config.default_dt).float()
        current_dt = torch.where(active, current_dt, self.config.default_dt)
        relative_time = torch.where(history.valid, history.timestamps - timestamps[:, None], 0.0).float()
        # Geometry is re-expressed for the actual current view. No RGB or camera
        # descriptors from this branch are passed into the motion model below.
        if use_adapter:
            # This ablation hides history ONLY from the visual addon; the TWM
            # still receives the exact same completed self-history below.
            addon_history_valid = history.valid if adapter_history else torch.zeros_like(history.valid)
            addon_history_avail = history.availability & addon_history_valid[..., None]
            delta = self.adapter(patch_features, transform_state(raw_clean, camera),
                                 transform_state(history.state_world, camera),
                                 availability=availability, history_availability=addon_history_avail,
                                 history_valid=addon_history_valid, history_uncertainty=history.uncertainty,
                                 history_uncertainty_known=history.uncertainty_known,
                                 relative_time=relative_time, history_dt=hist_dt,
                                 image_valid=image_valid, handedness=handedness,
                                 camera_intrinsics=intrinsic, crop_transform=crop)
        else:
            delta = raw_clean.new_zeros(b, 51)
        # Adapter dp/root updates are predicted in CURRENT CAMERA axes. Rotate
        # those two vectors into the fixed anchor; finger updates stay local.
        with torch.autocast(device_type=raw_world.device.type, enabled=False):
            camera_to_anchor = world_to_anchor[:, :3, :3] @ camera[:, :3, :3].transpose(-1, -2)
            dp = (camera_to_anchor @ delta[:, :3, None].float()).squeeze(-1)
            root_dr = (camera_to_anchor @ delta[:, 3:6, None].float()).squeeze(-1)
            delta_anchor = torch.cat((dp, root_dr, delta[:, 6:]), -1)
        candidate_anchor = g.state_apply_delta(base_anchor, delta_anchor)
        # A visual candidate may infer missing groups, but raw availability is
        # untouched. With neither image nor history, unavailable slots stay masked.
        candidate_availability = candidate_availability | (image_valid[:, None] & use_adapter)
        twm_beta = beta if self.config.condition_on_beta else None
        output = self.twm(history_anchor, raw_anchor, candidate_anchor,
                          history_valid=history.valid, history_availability=history.availability,
                          availability=availability, candidate_availability=candidate_availability,
                          dt=torch.cat((hist_dt, current_dt[:, None]), -1), beta=twm_beta,
                          beta_valid=beta_valid if self.config.condition_on_beta else None,
                          handedness=handedness, return_forecast=return_forecast)
        state_world = transform_state(output["state"], anchor_world)
        # A rotated diagonal Gaussian generally has correlated world axes.
        # Keep the anchor std for likelihoods and expose its FULL world-space
        # covariance explicitly; standard deviations are never rotated vectors.
        rotation_world = anchor_world[:, :3, :3]
        with torch.autocast(device_type=raw_world.device.type, enabled=False):
            translation_covariance_world = (rotation_world
                @ torch.diag_embed(output["state_translation_std"].square())
                @ rotation_world.transpose(-1, -2))
        output.update({"state_world": state_world, "anchor_world": anchor_world,
                       "state_translation_covariance_world": translation_covariance_world,
                       "raw_world": raw_world.detach(), "candidate_world": transform_state(candidate_anchor, anchor_world),
                       "adapter_delta": delta_anchor, "candidate_availability": candidate_availability,
                       "availability": availability, "frame_valid": active, "image_valid": image_valid,
                       "history_count": history.valid.sum(-1), "history_reset": resets})
        if return_forecast:
            output["forecast_world"] = transform_state(output["forecast"], anchor_world)
            output["forecast_modes_world"] = transform_state(output["forecast_modes"], anchor_world)
            future_rotation_world = rotation_world[:, None, None]
            with torch.autocast(device_type=raw_world.device.type, enabled=False):
                output["forecast_modes_translation_covariance_world"] = (future_rotation_world
                    @ torch.diag_embed(output["forecast_translation_std"].square())
                    @ future_rotation_world.transpose(-1, -2))
        updated = history.commit(state_world, timestamps, output["state_translation_std"],
                                 output["state_rotation_concentration"], active, handedness,
                                 detach=self.config.detach_history)
        return output, updated

    def forward_clip(self, patch_features: Tensor, raw_world: Tensor, *, timestamps: Tensor,
                     frame_valid: Tensor, availability: Tensor, image_valid: Tensor, handedness: Tensor,
                     world_to_camera: Tensor, camera_intrinsics: Tensor, crop_transform: Tensor,
                     beta: Optional[Tensor] = None, beta_valid: Optional[Tensor] = None,
                     reset: Optional[Tensor] = None, initial_history: Optional[FeedbackHistory] = None,
                     return_forecast: bool = False, use_adapter: bool = True,
                     adapter_history: bool = True) -> dict:
        """Process cached [B,T,...] clips; pad frames never enter history.

        crop_transform maps source-image pixels to normalized [-1,1] in the
        effective 192x256 region AFTER HaWoR removes 32 pixels at either side
        of its 256x256 crop. This matches the 16x12 patch grid.
        The caller owns reset flags for track/sequence boundaries. Returning
        final_history permits continuous feedback across separate HaWoR blocks.
        """
        if raw_world.ndim != 3 or raw_world.shape[-1] != 99:
            raise ValueError("raw_world must be [B,T,99]")
        b, length, _ = raw_world.shape
        if length < 1:
            raise ValueError("clip must contain at least one padded or real frame")
        if patch_features.shape[:2] != (b, length):
            raise ValueError("patch features and raw states must share batch/time axes")
        history = initial_history or self.empty_history(b, device=raw_world.device)
        results = []
        for index in range(length):
            hand = handedness if handedness.ndim == 1 else handedness[:, index]
            shape = None if beta is None else (beta if beta.ndim == 2 else beta[:, index])
            shape_valid = None if beta_valid is None else (beta_valid if beta_valid.ndim == 1 else beta_valid[:, index])
            output, history = self.step(
                patch_features[:, index], raw_world[:, index], timestamps=timestamps[:, index],
                frame_valid=frame_valid[:, index], availability=availability[:, index],
                image_valid=image_valid[:, index], handedness=hand,
                world_to_camera=world_to_camera[:, index], camera_intrinsics=camera_intrinsics[:, index],
                crop_transform=crop_transform[:, index], beta=shape, beta_valid=shape_valid,
                reset=None if reset is None else reset[:, index], history=history,
                return_forecast=return_forecast, use_adapter=use_adapter,
                adapter_history=adapter_history)
            results.append(output)
        # moe_load and its scalar balance are diagnostics with no batch axis.
        output = {key: torch.stack([result[key] for result in results], 1)
                  for key in results[0] if key not in ("moe_load", "moe_balance_loss")}
        output["moe_load"] = torch.stack([result["moe_load"] for result in results]).mean(0)
        output["moe_balance_loss"] = torch.stack([result["moe_balance_loss"] for result in results]).mean()
        output["final_history"] = history
        return output

    def forward(self, *args, **kwargs) -> dict:
        return self.forward_clip(*args, **kwargs)


def feedback_current_nll(output: dict, target_world: Tensor, target_valid: Tensor) -> Tensor:
    """One final-current proper NLL over complete MANO reference frames only.

    Targets are loss-only and never populate history. Partially available
    groups do not become invented full-state labels. Report their count in the
    caller; an entirely unsupervised batch is rejected instead of silent zero.
    """
    valid = target_valid.bool()
    if valid.ndim == output["frame_valid"].ndim + 1:
        if valid.shape[-1] != 17:
            raise ValueError("group target_valid must end in seventeen groups")
        valid = valid.all(-1)
    if valid.shape != output["frame_valid"].shape or target_world.shape != output["state_world"].shape:
        raise ValueError("target shapes must match output batch/time axes")
    valid = valid & output["frame_valid"]
    if not bool(valid.any()):
        raise ValueError("single full-state NLL requires at least one complete reference frame")
    # Select BEFORE geometry/log-density: masked NaN GT must not poison gradients.
    selected = {key: output[key][valid] for key in
                ("state", "state_translation_std", "state_rotation_concentration")}
    target = transform_state(target_world[valid], inverse_rigid(output["anchor_world"][valid]))
    if not bool(torch.isfinite(target).all()):
        raise ValueError("complete reference state must be finite")
    return -current_distribution_log_prob(selected, target)["total"].mean()
