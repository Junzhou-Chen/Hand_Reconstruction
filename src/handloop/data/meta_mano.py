"""Asset-gated MANO reference decoding for Meta PCA15 and DexYCB PCA45.

This module does not download or redistribute MANO assets. Supply the user's
licensed ``MANO_LEFT.pkl`` and ``MANO_RIGHT.pkl`` files explicitly. Inputs are
finite, available labels only; callers preserve missing-label masks separately.

PCA coefficients are expanded with the corresponding hand's basis. The MANO
mean is added exactly once by official smplx, with its internal PCA disabled
even for K=45. State position is the actual decoded wrist J0 in metres, not
the source MANO ``transl`` parameter. No source-to-world transform is inferred.

Sources: Meta hand_tracking_toolkit bc628e9286c18444b47a0971a89d075ea6f00747,
hand_models/mano_hand_model.py; smplx 1265df7ba545e8b00f72e7c557c766e15c71632f,
body_models.py MANO and lbs.py. Meta applies the optional left shapedirs fix;
DexYCB's published manopth decoder does not apply that fix explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib import metadata
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from handloop.geometry import axis_angle_to_matrix, state_from_rotmat


META_TOOLKIT_REVISION = "bc628e9286c18444b47a0971a89d075ea6f00747"
SMPLX_AUDITED_REVISION = "1265df7ba545e8b00f72e7c557c766e15c71632f"
MANO_PARENTS = (-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14)
# Official Meta order: five tips, wrist, two thumb joints, then three joints
# of index/middle/ring/pinky. This is 20 points, not the common MANO21 order.
META_CANONICAL_MAPPING = (16, 17, 18, 19, 20, 0, 14, 15, 1, 2, 3,
                          4, 5, 6, 10, 11, 12, 7, 8, 9)
META_FINGERTIP_VERTICES = (744, 320, 443, 554, 671)


@dataclass(frozen=True)
class DecodedMANO:
    state: Tensor                 # (N,99): actual J0, root6D, local finger6D.
    beta: Tensor                  # (N,10), separately stored shape.
    joints_m: Tensor              # (N,16,3), unpermuted MANO kinematic joints.
    vertices_m: Tensor            # (N,778,3), source reference frame, metres.
    canonical_landmarks_m: Tensor # (N,20,3), official Meta canonical order.
    full_pose_axis_angle: Tensor  # (N,48), mean already included exactly once.


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class MANOReferenceDecoder:
    """Decode available MANO PCA labels using explicit, licensed model files.

    ``fix_left_shapedirs=True`` reproduces Meta's conditional correction. Use
    ``False`` for a source whose reference decoder does not make that change;
    the choice must be recorded in the dataset manifest. ``asset_version`` is
    a user-supplied release label; an absent label remains unknown, while file
    SHA256 hashes always identify the actual assets used.

    smplx loads the user-provided model pickle. Only pass trusted model files.
    This decoder is for reference preparation; it performs no fitting, temporal
    interpolation, validity imputation, or native-to-MANO skeleton mapping.
    """

    def __init__(self, model_dir: str | Path, *, device: str | torch.device = "cpu",
                 fix_left_shapedirs: bool = False,
                 asset_version: str | None = None) -> None:
        self.device = torch.device(device)
        self.model_dir = Path(model_dir).expanduser().resolve()
        files = {side: self.model_dir / f"MANO_{side.upper()}.pkl"
                 for side in ("left", "right")}
        missing = [str(path) for path in files.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Licensed MANO assets are required to decode PCA labels; "
                "this module never downloads them. Missing: " + ", ".join(missing))
        try:
            import smplx
        except ImportError as exc:
            raise ImportError("Install official smplx in the isolated preparation "
                              "runtime after supplying licensed MANO assets.") from exc

        self.layers: dict[str, Any] = {}
        self.components: dict[str, Tensor] = {}
        self.means: dict[str, Tensor] = {}
        for side, path in files.items():
            # PCA is ALWAYS expanded here. In particular, smplx switches PCA
            # off at num_pca_comps=45 regardless of a caller's use_pca=True.
            layer = smplx.create(str(path), "mano", is_rhand=side == "right",
                                 use_pca=False, num_pca_comps=45,
                                 flat_hand_mean=False, num_betas=10).to(self.device)
            layer.eval()
            layer.requires_grad_(False)
            if tuple(layer.parents.cpu().tolist()) != MANO_PARENTS:
                raise ValueError(f"Unexpected MANO kinematic tree in {path}")
            basis = torch.as_tensor(layer.np_hand_components, dtype=torch.float32,
                                    device=self.device).clone()
            mean = layer.hand_mean.detach().float().clone()
            if basis.shape != (45, 45) or mean.shape != (45,):
                raise ValueError(f"Unexpected MANO PCA basis/mean shape in {path}")
            if not torch.isfinite(basis).all() or not torch.isfinite(mean).all():
                raise ValueError(f"Non-finite MANO PCA basis/mean in {path}")
            if layer.use_pca or layer.flat_hand_mean:
                raise RuntimeError("smplx did not honor explicit decoding conventions")
            self.layers[side] = layer
            self.components[side], self.means[side] = basis, mean

        difference = (self.layers["left"].shapedirs[:, 0, :]
                      - self.layers["right"].shapedirs[:, 0, :]).abs().sum()
        self.left_shapedirs_fixed = bool(fix_left_shapedirs and difference.item() < 1)
        if self.left_shapedirs_fixed:
            with torch.no_grad():
                self.layers["left"].shapedirs[:, 0, :].mul_(-1)
        self._provenance = {
            "decoder": "MANOReferenceDecoder",
            "meta_toolkit_revision": META_TOOLKIT_REVISION,
            "smplx_audited_revision": SMPLX_AUDITED_REVISION,
            "smplx_installed_version": metadata.version("smplx"),
            "asset_version": asset_version,
            "assets": {side: {"filename": path.name, "sha256": _sha256(path),
                              "bytes": path.stat().st_size}
                       for side, path in files.items()},
            "left_shapedirs_fix_requested": bool(fix_left_shapedirs),
            "left_shapedirs_fix_applied": self.left_shapedirs_fixed,
            "left_right_shapedirs_difference_before_fix": float(difference.item()),
            "pca_expansion": "coefficients @ hand_specific_components[:K] + hand_mean",
            "position": "official smplx MANO output joints[:,0], metres",
            "rotation6d": "columns_c0xyz_c1xyz",
            "source_frame": "preserved; no inversion, reflection or extrinsic applied",
        }

    def provenance(self) -> dict[str, Any]:
        """Return JSON-serializable conventions and asset identifiers."""
        import copy
        return copy.deepcopy(self._provenance)

    def _float(self, value: Any, shape: tuple[int, ...], name: str) -> Tensor:
        tensor = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        if tuple(tensor.shape) != shape:
            raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} contains missing/non-finite labels; filter with its mask")
        return tensor

    @torch.no_grad()
    def decode_pca(self, pca_coeffs: Any, betas: Any, global_orient: Any,
                   translation: Any, handedness: Any) -> DecodedMANO:
        """Decode N available labels; handedness is 0=left, 1=right.

        Shapes: coefficients (N,K), 1<=K<=45; betas (10,), (1,10), or
        (N,10); global_orient (N,3) axis-angle radians; translation (N,3)
        MANO transl metres; handedness (N,). Meta uses K=15, DexYCB K=45.
        Output stays in the source annotation's reference frame.
        """
        coefficients = torch.as_tensor(pca_coeffs, dtype=torch.float32, device=self.device)
        if coefficients.ndim != 2 or not 1 <= coefficients.shape[1] <= 45:
            raise ValueError("pca_coeffs must have shape (N,K), 1<=K<=45")
        n, k = coefficients.shape
        if n == 0:
            raise ValueError("Decode only nonempty batches of available labels")
        coefficients = self._float(coefficients, (n, k), "pca_coeffs")
        beta = torch.as_tensor(betas, dtype=torch.float32, device=self.device)
        if beta.shape in (torch.Size([10]), torch.Size([1, 10])):
            beta = beta.reshape(1, 10).expand(n, -1)
        beta = self._float(beta, (n, 10), "betas")
        root = self._float(global_orient, (n, 3), "global_orient")
        transl = self._float(translation, (n, 3), "translation")
        sides = torch.as_tensor(handedness, device=self.device)
        if sides.shape != (n,) or not ((sides == 0) | (sides == 1)).all():
            raise ValueError("handedness must have shape (N,) with only 0=left or 1=right")

        joints = torch.empty(n, 16, 3, dtype=torch.float32, device=self.device)
        vertices = torch.empty(n, 778, 3, dtype=torch.float32, device=self.device)
        full_pose = torch.empty(n, 48, dtype=torch.float32, device=self.device)
        for value, side in enumerate(("left", "right")):
            mask = sides == value
            if not mask.any():
                continue
            # The projection is a 45D axis-angle OFFSET. Official smplx adds
            # the nonflat hand mean once, yielding output.full_pose.
            projected = coefficients[mask] @ self.components[side][:k]
            output = self.layers[side](betas=beta[mask], global_orient=root[mask],
                                       hand_pose=projected, transl=transl[mask],
                                       return_verts=True, return_full_pose=True)
            if output.joints.shape[1] < 16:
                raise RuntimeError("smplx returned fewer than sixteen MANO joints")
            joints[mask] = output.joints[:, :16]
            vertices[mask] = output.vertices
            full_pose[mask] = output.full_pose
            expected = projected + self.means[side]
            if not torch.allclose(output.full_pose[:, 3:], expected, rtol=1e-5, atol=1e-6):
                raise RuntimeError("smplx PCA/mean convention differs from audited source")

        state = state_from_rotmat(joints[:, 0],
                                  axis_angle_to_matrix(full_pose.reshape(n, 16, 3)))
        tips = vertices[:, list(META_FINGERTIP_VERTICES)]
        landmarks = torch.cat((joints, tips), dim=1)[:, list(META_CANONICAL_MAPPING)]
        if not all(torch.isfinite(x).all() for x in (state, joints, vertices, full_pose)):
            raise RuntimeError("MANO decoding produced non-finite geometry")
        return DecodedMANO(state, beta, joints, vertices, landmarks, full_pose)

    def decode_meta(self, theta15: Any, wrist_xform6: Any, betas: Any,
                    handedness: Any) -> DecodedMANO:
        """Thin wrapper for HOT3D/UmeTrack mano_pose fields; no native fallback."""
        theta = torch.as_tensor(theta15, dtype=torch.float32, device=self.device)
        if theta.ndim != 2 or theta.shape[1] != 15:
            raise ValueError("Meta thetas are (N,15) PCA coefficients, not axis angles")
        wrist = self._float(wrist_xform6, (theta.shape[0], 6), "wrist_xform6")
        return self.decode_pca(theta, betas, wrist[:, :3], wrist[:, 3:], handedness)
