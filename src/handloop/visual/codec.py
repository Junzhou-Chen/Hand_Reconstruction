"""Differentiable visual/state coordinate codec for the 99D MANO contract.

Rotations act on column vectors; state translation is anatomical wrist J0 in
metres. MANO's ``transl`` is NOT J0. Absolute rotation matrices already contain
the hand mean: it must never be added again. No image tensor enters this codec.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from handloop.geometry import state_from_rotmat, state_to_rotmat


def invert_transform(transform: Tensor) -> Tensor:
    """Invert an SE(3) matrix, differentiably, without a generic matrix inverse."""
    if transform.shape[-2:] != (4, 4):
        raise ValueError("transform must end in (4,4)")
    transform = transform.float()
    rotation = transform[..., :3, :3].transpose(-1, -2)
    translation = -(rotation @ transform[..., :3, 3:4]).squeeze(-1)
    upper = torch.cat((rotation, translation[..., None]), dim=-1)
    bottom = torch.zeros_like(transform[..., :1, :])
    bottom[..., 0, 3] = 1
    return torch.cat((upper, bottom), dim=-2)


def transform_state(state: Tensor, source_to_target: Tensor) -> Tensor:
    """Change only wrist position/root frame; parent-local fingers stay local.

    Prefix dimensions follow PyTorch broadcasting. For a history (...,K,99)
    and one transform per batch (...,4,4), pass transform[...,None,:,:].
    """
    if source_to_target.shape[-2:] != (4, 4):
        raise ValueError("source_to_target must end in (4,4)")
    rotations = state_to_rotmat(state)
    matrix = source_to_target.float()
    root = matrix[..., :3, :3] @ rotations[..., 0, :, :]
    position = (matrix[..., :3, :3] @ state[..., :3, None].float()).squeeze(-1)
    position = position + matrix[..., :3, 3]
    return state_from_rotmat(position, torch.cat((root[..., None, :, :],
                                                 rotations[..., 1:, :, :]), dim=-3))


def state_to_anchor(state_world: Tensor, anchor_to_world: Tensor) -> Tensor:
    return transform_state(state_world, invert_transform(anchor_to_world))


def state_from_anchor(state_anchor: Tensor, anchor_to_world: Tensor) -> Tensor:
    return transform_state(state_anchor, anchor_to_world)


def mirror_rotations_x(rotations: Tensor, is_right: Tensor) -> Tensor:
    """Undo HaWoR's left-image mirror: R_left = S R_right S, S=diag(-1,1,1).

    This equals negating axis-angle y/z, without an unstable log/exp roundtrip.
    ``is_right`` has the rotation batch prefix, excluding the 16-joint axis.
    """
    if rotations.shape[-3:] != (16, 3, 3):
        raise ValueError("rotations must end in (16,3,3)")
    reflection = rotations.new_tensor([-1., 1., 1.])
    mirrored = rotations * reflection[..., :, None] * reflection[..., None, :]
    return torch.where(is_right.bool()[..., None, None, None], rotations, mirrored)


def project_world_points(points_world: Tensor, camera_to_world: Tensor,
                         intrinsics: Tensor, full_to_crop: Tensor | None = None,
                         *, min_depth: float = 1e-5) -> tuple[Tensor, Tensor]:
    """Project (...,J,3) into full/crop pixel coordinates plus positive-depth mask.

    Coordinates are not clamped to the image. Off-crop points remain off-crop;
    callers must mask them rather than pretending border pixels are evidence.
    """
    world_to_camera = invert_transform(camera_to_world)
    camera = (world_to_camera[..., None, :3, :3] @ points_world[..., None]).squeeze(-1)
    camera = camera + world_to_camera[..., None, :3, 3]
    positive = camera[..., 2] > min_depth
    safe_z = camera[..., 2:3].clamp_min(min_depth)
    homogeneous = camera / safe_z
    pixels = (intrinsics.float()[..., None, :, :] @ homogeneous[..., None]).squeeze(-1)
    if full_to_crop is not None:
        pixels = (full_to_crop.float()[..., None, :, :] @ pixels[..., None]).squeeze(-1)
    return pixels[..., :2], positive


class MANOStateCodec(nn.Module):
    """Asset-gated MANO geometry with explicit visual left-shape convention.

    HaWoR's released camera/world renderer unconditionally flips
    ``left.shapedirs[:,0,:]``. ``left_shapedirs_fix=True`` reproduces that exact
    convention (unlike Meta's conditional fix). Source GT decoders may differ;
    do not reuse their beta as though the two conventions were interchangeable.
    Licensed assets are supplied by the caller and never downloaded here.
    """

    def __init__(self, mano_right_path: str | Path, mano_left_path: str | Path,
                 *, left_shapedirs_fix: bool = True) -> None:
        super().__init__()
        try:
            import smplx
        except ImportError as exc:
            raise ImportError("MANOStateCodec requires official smplx and licensed MANO files") from exc
        paths = {"right": Path(mano_right_path), "left": Path(mano_left_path)}
        for path in paths.values():
            if not path.is_file():
                raise FileNotFoundError(f"Licensed MANO asset missing: {path}")
        self.layers = nn.ModuleDict({side: smplx.MANOLayer(
            str(path), is_rhand=side == "right", use_pca=False,
            flat_hand_mean=True, num_betas=10) for side, path in paths.items()})
        self.left_shapedirs_fix = bool(left_shapedirs_fix)
        if self.left_shapedirs_fix:
            with torch.no_grad():
                self.layers["left"].shapedirs[:, 0, :].mul_(-1)
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True) -> "MANOStateCodec":
        super().train(False)
        return self

    def rest_wrist(self, beta: Tensor, is_right: Tensor) -> Tensor:
        """Shaped rest J0; MANO rotation occurs about this anatomical root."""
        result = torch.zeros(*beta.shape[:-1], 3, device=beta.device, dtype=torch.float32)
        for side, right in (("left", False), ("right", True)):
            layer = self.layers[side]
            shaped = layer.v_template.float() + torch.einsum(
                "...k,vck->...vc", beta.float(), layer.shapedirs[..., :10].float())
            root = torch.einsum("v,...vc->...c", layer.J_regressor[0].float(), shaped)
            result = torch.where((is_right.bool() == right)[..., None], root, result)
        return result

    def from_hawor(self, rotation_matrices: Tensor, mano_translation: Tensor,
                   beta: Tensor, is_right: Tensor, camera_to_world: Tensor) -> Tensor:
        """Canonicalize official post-PAM outputs after camera tx unflip.

        Matrices are still right-template predictions for mirrored left crops.
        ``mano_translation`` is already in the original unmirrored camera frame,
        matching HaWoR.forward_step/get_trans. The output is actual world J0.
        """
        rotations = mirror_rotations_x(rotation_matrices.float(), is_right)
        camera_wrist = mano_translation.float() + self.rest_wrist(beta, is_right)
        state_camera = state_from_rotmat(camera_wrist, rotations)
        return transform_state(state_camera, camera_to_world)

    def to_mano(self, state: Tensor, beta: Tensor, is_right: Tensor) -> dict[str, Tensor]:
        rotations = state_to_rotmat(state)
        return {"global_orient": rotations[..., :1, :, :],
                "hand_pose": rotations[..., 1:, :, :], "betas": beta.float(),
                "transl": state[..., :3].float() - self.rest_wrist(beta, is_right)}

    def geometry(self, state: Tensor, beta: Tensor, is_right: Tensor) -> dict[str, Tensor]:
        """Differentiable vertices and MANO16 joints, preserving state J0 exactly."""
        shape = state.shape[:-1]
        rotations = state_to_rotmat(state).reshape(-1, 16, 3, 3)
        beta_flat = beta.float().reshape(-1, 10)
        side_flat = is_right.bool().reshape(-1)
        vertices = state.new_zeros((rotations.shape[0], 778, 3), dtype=torch.float32)
        joints = state.new_zeros((rotations.shape[0], 16, 3), dtype=torch.float32)
        for side, right in (("left", False), ("right", True)):
            selected = side_flat == right
            if not selected.any():
                continue
            # MANOLayer consumes absolute matrices; it never adds the mean.
            output = self.layers[side](betas=beta_flat[selected],
                global_orient=rotations[selected, :1],
                hand_pose=rotations[selected, 1:], pose2rot=False)
            shift = state[..., :3].reshape(-1, 3)[selected] - output.joints[:, 0]
            vertices[selected] = output.vertices + shift[:, None]
            joints[selected] = output.joints[:, :16] + shift[:, None]
        return {"vertices": vertices.reshape(*shape, 778, 3),
                "joints": joints.reshape(*shape, 16, 3)}

    def provenance(self) -> dict[str, Any]:
        return {"state": "wrist_J0_metres+root_columns6D+15_local_columns6D",
                "hand_mean": "absolute rotation matrices; never added twice",
                "left_pose_unflip": "S R S with S=diag(-1,1,1)",
                "left_shapedirs_fix": self.left_shapedirs_fix,
                "left_shapedirs_fix_type": "unconditional official HaWoR renderer",
                "position_conversion": "J0=MANO_transl+J_regressor0(v_shaped)"}
