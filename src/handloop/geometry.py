"""Float32 rotation and 99D state utilities, with no MANO asset dependency.

Convention: matrices act on COLUMN vectors, ``v_parent = R @ v_local``.
Quaternions are scalar-first ``(w, x, y, z)`` and use the Hamilton product.
Rotation6D is the CONCATENATION OF THE FIRST TWO COLUMNS, i.e.
``[R00, R10, R20, R01, R11, R21]``. This is not PyTorch3D's two-row
serialization. Identity is ``[1, 0, 0, 0, 1, 0]``.

A state is ``[wrist_xyz(3), root_6d(6), MANO_local_finger_6d(15*6)]``.
Translation is the actual wrist J0 in metres, NOT MANO's transl parameter.
Shape beta and hand-mean/PCA expansion belong to the data adapter, not here.
All floating computations/outputs use float32; inputs must be finite.
Degenerate finite 6D/quaternion inputs have deterministic identity/orthogonal
fallbacks so missing-value placeholders do not produce singular matrices.
"""

from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F

STATE_DIM = 99
NUM_ROTATIONS = 16
DELTA_DIM = 51
ROTATION_6D_CONVENTION = "columns_c0xyz_c1xyz"


def _last_dim(x: Tensor, size: int, name: str) -> Tensor:
    if x.shape[-1:] != (size,):
        raise ValueError(f"{name} must end in {size}, got {tuple(x.shape)}")
    return x.float()


def _matrix(x: Tensor) -> Tensor:
    if x.shape[-2:] != (3, 3):
        raise ValueError(f"rotation matrix must end in (3, 3), got {tuple(x.shape)}")
    return x.float()


def identity_rotation_6d(*, device=None, dtype=torch.float32) -> Tensor:
    return torch.tensor([1, 0, 0, 0, 1, 0], device=device, dtype=dtype)


def identity_state(*batch_shape: int, device=None) -> Tensor:
    """Return zero wrist translation and sixteen identity rotations."""
    p = torch.zeros(*batch_shape, 3, device=device, dtype=torch.float32)
    r = identity_rotation_6d(device=device).repeat(NUM_ROTATIONS)
    return torch.cat((p, r.expand(*batch_shape, 96)), dim=-1)


def rotation_6d_to_matrix(d6: Tensor, eps: float = 1e-6) -> Tensor:
    """Gram--Schmidt on two columns; deterministic fallback for zero/collinear."""
    d6 = _last_dim(d6, 6, "rotation6d")
    a1, a2 = d6[..., :3], d6[..., 3:]
    n1 = torch.linalg.vector_norm(a1, dim=-1, keepdim=True)
    e1 = torch.zeros_like(a1)
    e1[..., 0] = 1.0
    b1 = torch.where(n1 > eps, a1 / n1.clamp_min(eps), e1)
    a2_perp = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    n2 = torch.linalg.vector_norm(a2_perp, dim=-1, keepdim=True)
    # Choose the coordinate axis least parallel to b1. Its orthogonal
    # projection has norm >= sqrt(2/3), including the zero-input fallback.
    axis = F.one_hot(b1.abs().argmin(-1), num_classes=3).to(b1.dtype)
    fallback = axis - (axis * b1).sum(-1, keepdim=True) * b1
    fallback = F.normalize(fallback, dim=-1, eps=eps)
    b2 = torch.where(n2 > eps, a2_perp / n2.clamp_min(eps), fallback)
    b3 = torch.linalg.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


def matrix_to_rotation_6d(matrix: Tensor) -> Tensor:
    """Serialize first column then second column; does not project onto SO(3)."""
    matrix = _matrix(matrix)
    return torch.cat((matrix[..., :, 0], matrix[..., :, 1]), dim=-1)


def normalize_quaternion(quaternion: Tensor, eps: float = 1e-8) -> Tensor:
    q = _last_dim(quaternion, 4, "quaternion")
    norm = torch.linalg.vector_norm(q, dim=-1, keepdim=True)
    identity = torch.zeros_like(q)
    identity[..., 0] = 1.0
    return torch.where(norm > eps, q / norm.clamp_min(eps), identity)


def standardize_quaternion(quaternion: Tensor) -> Tensor:
    """Choose nonnegative w; q and -q represent the same rotation."""
    q = _last_dim(quaternion, 4, "quaternion")
    return torch.where(q[..., :1] < 0, -q, q)


def quaternion_to_matrix(quaternion: Tensor) -> Tensor:
    w, x, y, z = normalize_quaternion(quaternion).unbind(-1)
    return torch.stack((
        1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y),
        2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x),
        2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y),
    ), dim=-1).reshape(*w.shape, 3, 3)


def _sqrt_positive_part(x: Tensor) -> Tensor:
    # Avoid sqrt'(0)=inf in unused quaternion candidates during backprop.
    positive = x > 0
    return torch.where(positive, torch.sqrt(torch.where(positive, x, torch.ones_like(x))),
                       torch.zeros_like(x))


def matrix_to_quaternion(matrix: Tensor) -> Tensor:
    """Stable largest-component conversion, including rotations near pi.

    Input must be a rotation matrix. This does not project arbitrary matrices.
    The quaternion branch is piecewise differentiable, as any such chart is.
    """
    m = _matrix(matrix)
    m00, m01, m02 = m[..., 0, :].unbind(-1)
    m10, m11, m12 = m[..., 1, :].unbind(-1)
    m20, m21, m22 = m[..., 2, :].unbind(-1)
    q_abs = _sqrt_positive_part(torch.stack((
        1+m00+m11+m22, 1+m00-m11-m22,
        1-m00+m11-m22, 1-m00-m11+m22,
    ), dim=-1))
    candidates = torch.stack((
        torch.stack((q_abs[..., 0]**2, m21-m12, m02-m20, m10-m01), -1),
        torch.stack((m21-m12, q_abs[..., 1]**2, m10+m01, m02+m20), -1),
        torch.stack((m02-m20, m10+m01, q_abs[..., 2]**2, m12+m21), -1),
        torch.stack((m10-m01, m20+m02, m21+m12, q_abs[..., 3]**2), -1),
    ), dim=-2)
    candidates = candidates / (2 * q_abs[..., :, None].clamp_min(0.1))
    index = q_abs.argmax(-1)[..., None, None].expand(*q_abs.shape[:-1], 1, 4)
    q = candidates.gather(-2, index).squeeze(-2)
    return standardize_quaternion(normalize_quaternion(q))


def quaternion_multiply(a: Tensor, b: Tensor) -> Tensor:
    """Hamilton product; R(a*b) = R(a) @ R(b) for unit quaternions."""
    aw, ax, ay, az = _last_dim(a, 4, "quaternion a").unbind(-1)
    bw, bx, by, bz = _last_dim(b, 4, "quaternion b").unbind(-1)
    return torch.stack((aw*bw-ax*bx-ay*by-az*bz,
                        aw*bx+ax*bw+ay*bz-az*by,
                        aw*by-ax*bz+ay*bw+az*bx,
                        aw*bz+ax*by-ay*bx+az*bw), dim=-1)


def quaternion_invert(quaternion: Tensor) -> Tensor:
    q = normalize_quaternion(quaternion)
    return torch.cat((q[..., :1], -q[..., 1:]), dim=-1)


def axis_angle_to_quaternion(axis_angle: Tensor) -> Tensor:
    aa = _last_dim(axis_angle, 3, "axis_angle")
    angle = torch.linalg.vector_norm(aa, dim=-1, keepdim=True)
    half = 0.5 * angle
    # sin(angle/2)/angle, with its analytic zero-angle limit.
    scale = 0.5 * torch.sinc(half / torch.pi)
    return torch.cat((torch.cos(half), aa * scale), dim=-1)


def quaternion_to_axis_angle(quaternion: Tensor, eps: float = 1e-6) -> Tensor:
    q = standardize_quaternion(normalize_quaternion(quaternion))
    vector = q[..., 1:]
    sin_half = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    angle = 2 * torch.atan2(sin_half, q[..., :1])
    ratio = angle / sin_half.clamp_min(eps)
    ratio = torch.where(sin_half > eps, ratio, 2 + sin_half.square()/3)
    return vector * ratio


def axis_angle_to_matrix(axis_angle: Tensor) -> Tensor:
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))


def matrix_to_axis_angle(matrix: Tensor) -> Tensor:
    """Principal SO(3) log vector in radians; discontinuous axis choice at pi."""
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))


def so3_exp_map(log_rotation: Tensor) -> Tensor:
    return axis_angle_to_matrix(log_rotation)


def so3_log_map(rotation: Tensor) -> Tensor:
    return matrix_to_axis_angle(rotation)


def geodesic_angle(a: Tensor, b: Tensor) -> Tensor:
    """SO(3) distance in [0, pi] radians, finite at equality and near pi.

    atan2 avoids acos' unbounded endpoint derivative and its usual positive
    error floor. The distance is not differentiable at exact equality/pi;
    torch's norm gives a finite zero subgradient at an exactly zero skew part.
    """
    a, b = _matrix(a), _matrix(b)
    with torch.autocast(device_type=a.device.type, enabled=False):
        relative = a.transpose(-1, -2) @ b
        skew = torch.stack((relative[..., 2, 1]-relative[..., 1, 2],
                            relative[..., 0, 2]-relative[..., 2, 0],
                            relative[..., 1, 0]-relative[..., 0, 1]), dim=-1)
        sin_angle = 0.5 * torch.linalg.vector_norm(skew, dim=-1)
        cos_angle = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1)-1)*0.5).clamp(-1, 1)
        return torch.atan2(sin_angle, cos_angle)


def state_to_rotmat(state: Tensor) -> Tensor:
    """Return sixteen rotations (...,16,3,3), root first, MANO order thereafter."""
    state = _last_dim(state, STATE_DIM, "state")
    return rotation_6d_to_matrix(state[..., 3:].reshape(*state.shape[:-1], 16, 6))


def state_from_rotmat(translation: Tensor, rotations: Tensor) -> Tensor:
    translation = _last_dim(translation, 3, "translation")
    if rotations.shape[-3:] != (16, 3, 3):
        raise ValueError("rotations must end in (16,3,3)")
    rotations6d = matrix_to_rotation_6d(rotations).flatten(-2)
    return torch.cat((translation, rotations6d), dim=-1)


def normalize_state(state: Tensor) -> Tensor:
    state = _last_dim(state, STATE_DIM, "state")
    return state_from_rotmat(state[..., :3], state_to_rotmat(state))


def state_apply_delta(base: Tensor, delta: Tensor) -> Tensor:
    """Apply (...,51) tangent update to (...,99), returning full legal 99D.

    delta = [dp(3), root_axis_angle(3), finger_axis_angles(15*3)].
    dp is in the common world/anchor frame. Root and each finger use LEFT
    composition: R_new=Exp(droot)@R and Q_new=Exp(dfinger)@Q. Finger increments
    therefore live in their MANO parent frames, not in world coordinates.
    This is a same-time inference update, not a physical-time integrator.
    """
    base = _last_dim(base, STATE_DIM, "base")
    delta = _last_dim(delta, DELTA_DIM, "delta")
    with torch.autocast(device_type=base.device.type, enabled=False):
        dr = axis_angle_to_matrix(delta[..., 3:].reshape(*delta.shape[:-1], 16, 3))
        rotations = dr @ state_to_rotmat(base)
        return state_from_rotmat(base[..., :3] + delta[..., :3], rotations)
