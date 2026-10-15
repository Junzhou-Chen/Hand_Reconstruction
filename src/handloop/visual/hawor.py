"""Frozen official HaWoR bridge, pinned to 66c7d4108d58a716deccd192cb7645170cdc7bd7.

The pretrained 16-frame ViT/IAM/head/PAM path never receives feedback. Cached
IAM patches are actual image features, not projected model keypoints. This
module does not invoke the motion infiller and does not estimate camera poses.
"""

from __future__ import annotations

from pathlib import Path
import hashlib
import subprocess
import sys
from typing import Any, Sequence

import torch
from torch import Tensor, nn

from handloop.geometry import rotation_6d_to_matrix
from handloop.visual.codec import MANOStateCodec

HAWOR_REVISION = "66c7d4108d58a716deccd192cb7645170cdc7bd7"
HAWOR_CHECKPOINT_URL = "https://huggingface.co/ThunderVVV/HaWoR/resolve/main/hawor/checkpoints/hawor.ckpt"
HAWOR_CONFIG_URL = "https://huggingface.co/ThunderVVV/HaWoR/resolve/main/hawor/model_config.yaml"
_PINNED_GIT_BLOBS = {
    "_DATA/data/mano_mean_params.npz": "52792d92cd4c9183d8e7e306bfa660106de8430d",
    "lib/datasets/track_dataset.py": "076cf8897f87fd03f65eaea7d0fb9d8001eface0",
    "lib/models/backbones/__init__.py": "8288cfd84555de1720b81129fa8accf76948cb88",
    "lib/models/backbones/vit.py": "c56c71889cd441294f57ad687d0678d2443d1eed",
    "lib/models/components/pose_transformer.py": "ac04971407cb59637490cc4842f048b9bc4758be",
    "lib/models/hawor.py": "34a081cd63d0a3e3be894fda21375683f3e61d51",
    "lib/models/mano_wrapper.py": "4801c26e3410035db3d09f8dac70c637dd7b00f5",
    "lib/models/modules.py": "b0adbef1ddfb6f15e8cf0df6b99f924f9c409d6f",
    "lib/utils/geometry.py": "b3d65066ce383c744407f8d800a7342cd324142b",
    "lib/utils/imutils.py": "d38db0bdd4446245f4bfdafb43643d92125f9d8d",
}


def verify_official_source(root: Path) -> dict[str, str]:
    """Check Git blob IDs even when the source was obtained as a tarball."""
    for name, expected in _PINNED_GIT_BLOBS.items():
        content = (root / name).read_bytes()
        actual = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
        if actual != expected:
            raise ValueError(f"Official source differs from audited revision: {name}")
    return dict(_PINNED_GIT_BLOBS)


class FrozenHaWoRBridge(nn.Module):
    """Official image network plus explicit image/camera/MANO state conversion.

    ``forward`` takes ImageNet-normalized crops [B,16,3,256,256], matching
    official TrackDatasetEval. All metadata have [B,16] prefixes. ``center``
    is the center used for the crop (mirrored for left); ``center_unflipped``
    is required to recover original pixel coordinates without assuming cx=W/2.
    ``detected`` is the original detector/protocol availability and is never
    replaced by an interpolated bbox-valid flag. Padded frames may influence
    original bidirectional attention as in official inference, but are never
    committed as measurements/history by the caller.
    """

    def __init__(self, model: nn.Module, codec: MANOStateCodec,
                 *, provenance: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.model = model
        self.codec = codec
        self._provenance = provenance or {}
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True) -> "FrozenHaWoRBridge":
        super().train(False)
        return self

    @classmethod
    def from_pretrained(cls, repo_root: str | Path, checkpoint_path: str | Path,
                        config_path: str | Path, mano_model_dir: str | Path,
                        *, device: str | torch.device = "cuda:0") -> "FrozenHaWoRBridge":
        """Load public weights and caller-supplied licensed MANO assets.

        Repo import paths are explicit; no package stubs or rewritten official
        model code are used. Missing or unexpected learned weight keys fail.
        Only optional renderer/nonlearned asset buffers may be absent.
        """
        root = Path(repo_root).expanduser().resolve()
        checkpoint = Path(checkpoint_path).expanduser().resolve()
        config = Path(config_path).expanduser().resolve()
        mano_dir = Path(mano_model_dir).expanduser().resolve()
        for path in (checkpoint, config, mano_dir / "MANO_RIGHT.pkl", mano_dir / "MANO_LEFT.pkl"):
            if not path.is_file():
                raise FileNotFoundError(path)
        source_blobs = verify_official_source(root)
        revision = None
        if (root / ".git").exists():
            revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
            if revision != HAWOR_REVISION:
                raise ValueError(f"HaWoR source revision {revision} differs from audited {HAWOR_REVISION}")
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from hawor.configs import get_config
        from lib.models.hawor import HAWOR
        imported_model_source = Path(sys.modules[HAWOR.__module__].__file__).resolve()
        if imported_model_source != root / "lib/models/hawor.py":
            raise RuntimeError(f"Another HaWoR/lib module is already imported: {imported_model_source}")
        cfg = get_config(str(config), update_cachedir=False)
        cfg.defrost()
        if cfg.MODEL.BACKBONE.TYPE != "vit" or int(cfg.MODEL.IMAGE_SIZE) != 256:
            raise ValueError("This bridge is audited for the official ViT 256 crop release")
        if "BBOX_SHAPE" not in cfg.MODEL:
            cfg.MODEL.BBOX_SHAPE = [192, 256]
        cfg.MANO.MODEL_PATH = str(mano_dir / "MANO_RIGHT.pkl")
        # Mean head parameters are public auxiliary assets from the official
        # source bundle; resolve relative paths without changing process cwd.
        mean_path = Path(str(cfg.MANO.MEAN_PARAMS))
        if not mean_path.is_absolute():
            candidates = [root / mean_path, root / "_DATA" / mean_path,
                          config.parent / mean_path]
            mean_path = next((p for p in candidates if p.is_file()), candidates[0])
        if not mean_path.is_file():
            raise FileNotFoundError(f"Official MANO head mean_params asset missing: {mean_path}")
        cfg.MANO.MEAN_PARAMS = str(mean_path)
        # Backbone/checkpoint duplicate preloads may reference author-local
        # paths. The complete public checkpoint below supplies these weights.
        cfg.MODEL.BACKBONE.PRETRAINED_WEIGHTS = None
        cfg.MODEL.BACKBONE.TORCH_COMPILE = False
        if "LOAD_WEIGHTS" in cfg.MODEL:
            cfg.MODEL.LOAD_WEIGHTS = None
        cfg.freeze()
        model = HAWOR(cfg)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        weights = payload.get("state_dict", payload)
        incompatible = model.load_state_dict(weights, strict=False)
        learned = {name for name, _ in model.named_parameters()}
        missing_learned = sorted(set(incompatible.missing_keys) & learned)
        if missing_learned or incompatible.unexpected_keys:
            raise RuntimeError(f"HaWoR checkpoint mismatch: learned missing={missing_learned}; "
                               f"unexpected={incompatible.unexpected_keys}")
        codec = MANOStateCodec(mano_dir / "MANO_RIGHT.pkl", mano_dir / "MANO_LEFT.pkl",
                               left_shapedirs_fix=True)
        bridge = cls(model, codec, provenance={"official_revision": HAWOR_REVISION,
            "verified_git_revision": revision, "checkpoint_path": str(checkpoint),
            "verified_source_git_blobs": source_blobs,
            "config_path": str(config), "source_path": str(root),
            "missing_nonlearned_checkpoint_buffers": list(incompatible.missing_keys),
            "infiller": False, "center_unflip": "explicit original center, supports noncentral K"})
        return bridge.to(device)

    def provenance(self) -> dict[str, Any]:
        return {**self._provenance, "codec": self.codec.provenance(),
                "temporal_context": "16 bidirectional frames; last padding repeats final frame",
                "feature": "post-IAM 16x12 raster patches, each 1280 channels",
                "raw_pose": "post-PAM absolute rotation matrices",
                "rgb_enters_twm": False}

    @torch.no_grad()
    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        image = batch["img"]
        if image.ndim != 5 or tuple(image.shape[1:]) != (16, 3, 256, 256):
            raise ValueError("img must be [B,16,3,256,256] normalized official crops")
        b, t = image.shape[:2]
        required = ("center", "center_unflipped", "scale", "img_focal", "img_center",
                    "is_right", "camera_to_world", "detected", "bbox_valid", "frame_valid", "image_valid")
        for key in required:
            if key not in batch or batch[key].shape[:2] != (b, t):
                raise ValueError(f"{key} requires leading shape {(b,t)}")
        if not torch.isfinite(image).all():
            raise ValueError("Crops must be finite; use explicit image_valid for missing frames")
        center = batch["center"].flatten(0, 1).float()
        original_center = batch["center_unflipped"].flatten(0, 1).float()
        scale = batch["scale"].flatten(0, 1).float()
        focal = batch["img_focal"].flatten(0, 1).float()
        img_center = batch["img_center"].flatten(0, 1).float()
        if (scale <= 0).any() or (focal <= 0).any():
            raise ValueError("scale/focal must be positive, including finite missing-frame placeholders")
        bbox = self.model.bbox_est(center, scale, focal, img_center)
        feature = self.model.backbone(image.flatten(0, 1)[:, :, :, 32:-32]).float()
        if feature.shape[1:] != (1280, 16, 12):
            raise RuntimeError(f"Unexpected official ViT features {tuple(feature.shape)}")
        if self.model.st_module is not None:
            feature = torch.cat((feature, bbox[:, :, None, None].expand(-1, -1, 16, 12)), dim=1)
            feature = feature.reshape(b, t, -1, 16, 12).permute(0, 3, 4, 1, 2).reshape(b*16*12, t, -1)
            feature = self.model.st_module(feature)
            feature = feature.reshape(b, 16, 12, t, 1280).permute(0, 3, 4, 1, 2).reshape(b*t, 1280, 16, 12)
        patches = feature.flatten(2).transpose(1, 2).reshape(b, t, 192, 1280)
        pose, beta, camera = self.model.mano_head(feature)
        if self.model.motion_module is not None:
            pose = self.model.motion_module(torch.cat((pose.reshape(b, t, 96), bbox.reshape(b, t, 3)), dim=-1)).reshape(b*t, 96)
        # hawor.py aliases rot6d_to_rotmat_HMR2 (reshape(-1,2,3), transpose),
        # NOT geometry.py's older interleaved reshape(-1,3,2) converter. Thus
        # these six numbers are first column xyz followed by second column xyz.
        # check_hawor_bridge compares the resulting matrices with official out.
        rotations = rotation_6d_to_matrix(pose.reshape(b, t, 16, 6))
        # Mirror tx back into the original image frame. Unlike official code's
        # key-presence test, mixed left/right batches are handled per frame.
        right = batch["is_right"].bool()
        camera = camera.clone()
        camera[:, 1] = torch.where(right.flatten(), camera[:, 1], -camera[:, 1])
        translation = self.model.get_trans(camera, original_center, scale, focal, img_center).squeeze(1).reshape(b, t, 3)
        beta = beta.reshape(b, t, 10)
        raw_world = self.codec.from_hawor(rotations, translation, beta, right, batch["camera_to_world"])
        observed = (batch["detected"].bool() & batch["bbox_valid"].bool() &
                    batch["frame_valid"].bool() & batch["image_valid"].bool())
        return {"image_features": patches, "raw_state_world": raw_world,
                "raw_available": observed[..., None].expand(b, t, 17).clone(),
                "beta": beta, "pred_pose": pose.reshape(b, t, 96),
                "pred_rotmat_mirrored": rotations, "pred_cam": camera.reshape(b, t, 3),
                "mano_translation_camera": translation,
                **{key: batch[key].clone() for key in ("detected", "bbox_valid", "frame_valid", "image_valid", "is_right", "camera_to_world")}}

    @torch.no_grad()
    def extract_clip(self, image_paths: Sequence[str | Path], bbox_xyxy: Any,
                     is_right: Any, intrinsics: Any, camera_to_world: Any,
                     detected: Any, *, bbox_valid: Any | None = None,
                     image_valid: Any | None = None,
                     focal_convention: str = "mean_fx_fy") -> dict[str, Tensor]:
        """Extract arbitrary-length single-hand track; return CPU tensors [T,...].

        Cropping uses official TrackDatasetEval (dilate=1.2). Frames are divided
        into the same nonoverlapping 16-frame blocks as official inference.
        ``crop_transform`` maps original full pixels to 256x256 crop pixels,
        including the left reflection. Actual ViT width is crop x in [32,224].
        Intrinsics retain actual fx/fy for subsequent geometric projection;
        the official scalar-f network uses the explicitly recorded mean.
        """
        import numpy as np
        import cv2
        from lib.datasets.track_dataset import TrackDatasetEval
        from lib.utils.imutils import transform as official_transform
        from torch.utils.data import default_collate
        n = len(image_paths)
        if not n:
            raise ValueError("extract_clip requires at least one frame")
        boxes = np.asarray(bbox_xyxy, dtype=np.float32)
        right = np.broadcast_to(np.asarray(is_right, dtype=bool), (n,)).copy()
        if not np.all(right == right[0]):
            raise ValueError("Identity/handedness changes must start a new track")
        k = torch.as_tensor(intrinsics, dtype=torch.float32).expand(n, 3, 3).clone()
        c2w = torch.as_tensor(camera_to_world, dtype=torch.float32).expand(n, 4, 4).clone()
        if boxes.shape != (n, 4) or not np.isfinite(boxes).all():
            raise ValueError("bbox_xyxy must be finite [T,4], with explicit masks for placeholders")
        if focal_convention != "mean_fx_fy":
            raise ValueError("Only explicit mean_fx_fy scalar-f HaWoR approximation is supported")
        focal = (k[:, 0, 0] + k[:, 1, 1]) * 0.5
        det = torch.as_tensor(detected, dtype=torch.bool).expand(n).clone()
        bv = torch.ones(n, dtype=torch.bool) if bbox_valid is None else torch.as_tensor(bbox_valid, dtype=torch.bool).expand(n).clone()
        iv = torch.ones(n, dtype=torch.bool) if image_valid is None else torch.as_tensor(image_valid, dtype=torch.bool).expand(n).clone()
        device = next(self.model.parameters()).device
        items: list[dict[str, Tensor]] = []
        transforms: list[Tensor] = []
        image_sizes: list[Tensor] = []
        for idx, path in enumerate(image_paths):
            image = cv2.imread(str(path))
            if image is None:
                raise FileNotFoundError(f"Image cannot be decoded: {path}")
            height, width = image.shape[:2]
            # Fresh bbox copy avoids TrackDatasetEval's in-place center flip.
            dataset = TrackDatasetEval([str(path)], boxes[idx:idx+1].copy(),
                img_focal=float(focal[idx]), img_center=k[idx, :2, 2].tolist(),
                normalization=True, dilate=1.2, do_flip=not bool(right[idx]))
            item = dataset[0]
            item.pop("do_flip", None)
            original_center = item["center"].clone()
            if not right[idx]:
                original_center[0] = width - original_center[0] - 1
            item.update(center_unflipped=original_center, is_right=torch.tensor(right[idx]),
                camera_to_world=c2w[idx], detected=det[idx], bbox_valid=bv[idx],
                frame_valid=torch.tensor(True), image_valid=iv[idx])
            # Official crop first truncates the integer bounds, then skimage
            # resizes that raster. Cache its actual pixel-center transform,
            # rather than the ideal floating bbox formula used for its loss.
            crop_center = item["center"].numpy()
            crop_scale = float(item["scale"])
            ul = np.asarray(official_transform([1, 1], crop_center, crop_scale,
                                              [256, 256], invert=1)) - 1
            br = np.asarray(official_transform([257, 257], crop_center, crop_scale,
                                              [256, 256], invert=1)) - 1
            factor_x = 256 / float(br[0] - ul[0])
            factor_y = 256 / float(br[1] - ul[1])
            transform = torch.eye(3)
            transform[0, 0] = factor_x if right[idx] else -factor_x
            transform[1, 1] = factor_y
            transform[0, 2] = factor_x * (0.5 - float(ul[0])) - 0.5
            if not right[idx]:
                transform[0, 2] += factor_x * (width - 1)
            transform[1, 2] = factor_y * (0.5 - float(ul[1])) - 0.5
            transforms.append(transform)
            image_sizes.append(torch.tensor([height, width]))
            items.append(item)
        outputs: dict[str, list[Tensor]] = {}
        for start in range(0, n, 16):
            block = items[start:start+16]
            actual = len(block)
            padded = block + [block[-1]] * (16-actual)
            batch = {key: value.to(device).unsqueeze(0) for key, value in default_collate(padded).items() if isinstance(value, Tensor)}
            if actual < 16:
                batch["frame_valid"][:, actual:] = False
            for key, value in self(batch).items():
                outputs.setdefault(key, []).append(value[0, :actual].cpu())
        result = {key: torch.cat(values) for key, values in outputs.items()}
        result.update(crop_transform=torch.stack(transforms), intrinsics=k,
                      focal_for_hawor=focal, bbox_xyxy=torch.from_numpy(boxes),
                      image_size=torch.stack(image_sizes))
        return result
