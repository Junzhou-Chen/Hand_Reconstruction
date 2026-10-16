"""Recording-disjoint RGB clips for differentiable official HaWoR tuning.

Only image crops and camera/detection metadata enter the frontend. The fitted
MANO targets remain separate batch fields for an external loss. The existing
DexYCB source manifests use reference-projected GT boxes and a fixed-camera
world gauge; this loader does not turn those into detector/SLAM estimates.
No frozen image-feature cache is used during full visual-backbone tuning.
"""
from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .visual_clips import file_sha256


SOURCE_SCHEMA = "handloop.visual_feedback_sources.v1"


def _inside(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError("RGB/reference path escapes the source manifest directory")
    return path


class VisualFeedbackRGBDataset(Dataset):
    """Online official crops, with independent final-state supervision.

    Windows never cross a recording, split, frame-id gap, or hand identity.
    Length/stride are multiples of 16 to retain the official nonoverlapping
    temporal blocks. Each window starts a fresh feedback history. Tail images
    repeat only for HaWoR's temporal padding; frame/image/target/detection masks
    are false there, so no padded frame can become a completed observation.

    ``hawor_root`` supplies the pinned official Python source to DataLoader
    workers. Instantiate/load the official bridge separately in the parent;
    this dataset never constructs a model or opens a checkpoint.
    """

    def __init__(self, source_manifest: str | Path, split: str = "train", *,
                 clip_length: int = 32, stride: int | None = None,
                 include_tail: bool = True, hawor_root: str | Path | None = None,
                 max_open_records: int = 4) -> None:
        if split not in ("train", "val"):
            raise ValueError("RGB adaptation deliberately excludes held-out test")
        if clip_length < 16 or clip_length % 16:
            raise ValueError("clip_length must be a positive multiple of 16")
        stride = int(stride or clip_length)
        if stride < 16 or stride % 16:
            raise ValueError("stride must be a positive multiple of 16")
        self.path = Path(source_manifest).expanduser().resolve()
        self.manifest = json.loads(self.path.read_text())
        if (self.manifest.get("schema") != SOURCE_SCHEMA or
                self.manifest.get("phase") != "complete"):
            raise ValueError("Use a completed visual-feedback source manifest")
        if self.manifest.get("focal_convention") != "official HaWoR single focal uses mean(fx,fy); real K retained for geometry":
            raise ValueError("Source focal convention must explicitly preserve actual K and mean(fx,fy)")
        self.manifest_sha256 = file_sha256(self.path)
        self.clip_length, self.stride = int(clip_length), stride
        self.hawor_root = str(Path(hawor_root).expanduser().resolve()) if hawor_root else None
        self.max_open_records = max(1, int(max_open_records))
        self.records: list[dict[str, Any]] = []
        self.windows: list[tuple[int, int, int]] = []
        self._arrays: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()
        self._reference_identities: dict[int, tuple[int, int, str]] = {}
        seen: dict[str, str] = {}
        for record in self.manifest.get("records", []):
            recording, record_split = record.get("recording"), record.get("split")
            if record_split not in ("train", "val") or not recording or recording in seen:
                raise ValueError("Source recordings must be unique, disjoint train/val recordings")
            seen[recording] = record_split
            if not record.get("coordinate_frame") or not record.get("timestamp_source"):
                raise ValueError("Source coordinate/timestamp provenance is required")
            if record_split != split:
                continue
            n = int(record["frames"])
            if n < 1 or len(record["image_paths"]) != n:
                raise ValueError("Source RGB/reference frame counts disagree")
            index = len(self.records)
            self.records.append(record)
            arrays = self._load_reference(index, initial=True)
            # Original source selection may remove invalid reference frames.
            # HaWoR cannot treat the remaining temporal gaps as contiguous.
            boundaries = np.concatenate(([0], np.flatnonzero(np.diff(arrays['frame_ids']) != 1) + 1, [n]))
            for begin, end in zip(boundaries[:-1], boundaries[1:]):
                for start in range(int(begin), int(end), self.stride):
                    count = min(self.clip_length, int(end) - start)
                    if count == self.clip_length or include_tail:
                        self.windows.append((index, start, count))
        if not self.windows:
            raise ValueError(f"No {split} RGB clips")

    def __len__(self) -> int:
        return len(self.windows)

    def _load_reference(self, index: int, *, initial: bool = False) -> dict[str, np.ndarray]:
        record = self.records[index]
        path = _inside(self.path.parent, record['reference_file'])
        stat = path.stat()
        if not initial:
            size, modified, _ = self._reference_identities[index]
            if (stat.st_size, stat.st_mtime_ns) != (size, modified):
                raise ValueError("Source reference changed during training")
        if index in self._arrays:
            self._arrays.move_to_end(index)
            return self._arrays[index]
        digest = file_sha256(path)
        if initial:
            self._reference_identities[index] = (stat.st_size, stat.st_mtime_ns, digest)
        elif digest != self._reference_identities[index][2]:
            raise ValueError("Source reference failed SHA256 verification")
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        n = int(record['frames'])
        expected = {'target_world': (n, 99), 'target_valid': (n, 17),
                    'timestamps': (n,), 'frame_ids': (n,), 'bbox_xyxy': (n, 4),
                    'camera_intrinsics': (n, 3, 3), 'camera_to_world': (n, 4, 4),
                    'handedness': (n,)}
        for key, shape in expected.items():
            if key not in arrays or arrays[key].shape != shape or arrays[key].dtype.hasobject:
                raise ValueError(f"Invalid source {key} shape/type")
            if not np.isfinite(arrays[key]).all():
                raise ValueError(f"Nonfinite source {key}")
        if arrays['timestamps'].dtype != np.float64 or (n > 1 and not np.all(np.diff(arrays['timestamps']) > 0)):
            raise ValueError("Keep strictly increasing source timestamps in float64 seconds")
        if not np.all(arrays['handedness'] == arrays['handedness'][0]):
            raise ValueError("A hand identity change must start a new recording")
        if not np.isin(arrays['handedness'], [0, 1]).all():
            raise ValueError("handedness must be 0(left) or 1(right)")
        self._arrays[index] = arrays
        while len(self._arrays) > self.max_open_records:
            self._arrays.popitem(last=False)
        return arrays

    def _crop(self, path: Path, bbox: np.ndarray, right: bool, intrinsics: np.ndarray) -> dict[str, torch.Tensor]:
        if self.hawor_root and self.hawor_root not in sys.path:
            sys.path.insert(0, self.hawor_root)
        import cv2
        from lib.datasets.track_dataset import TrackDatasetEval
        from lib.utils.imutils import transform as official_transform

        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Image cannot be decoded: {path}")
        height, width = image.shape[:2]
        focal = float((intrinsics[0, 0] + intrinsics[1, 1]) * 0.5)
        if focal <= 0 or np.any(bbox[2:] <= bbox[:2]):
            raise ValueError("Invalid source focal length or hand crop")
        dataset = TrackDatasetEval([str(path)], bbox[None].copy(), img_focal=focal,
            img_center=intrinsics[:2, 2].tolist(), normalization=True, dilate=1.2, do_flip=not right)
        item = dataset[0]
        original_center = item['center'].clone()
        if not right:
            original_center[0] = width - original_center[0] - 1
        center = item['center'].numpy()
        scale = float(item['scale'])
        ul = np.asarray(official_transform([1, 1], center, scale, [256, 256], invert=1)) - 1
        br = np.asarray(official_transform([257, 257], center, scale, [256, 256], invert=1)) - 1
        factor_x, factor_y = 256 / float(br[0] - ul[0]), 256 / float(br[1] - ul[1])
        transform = torch.eye(3)
        transform[0, 0], transform[1, 1] = factor_x if right else -factor_x, factor_y
        transform[0, 2] = factor_x * (0.5 - float(ul[0])) - 0.5
        if not right:
            transform[0, 2] += factor_x * (width - 1)
        transform[1, 2] = factor_y * (0.5 - float(ul[1])) - 0.5
        crop_to_patch = torch.tensor([[2 / 192, 0, -1 - 63 / 192],
                                      [0, 2 / 256, -1 + 1 / 256], [0, 0, 1]])
        return {**{key: torch.as_tensor(item[key]) for key in ('img', 'center', 'scale', 'img_focal', 'img_center')},
                'center_unflipped': original_center, 'crop_transform': crop_to_patch @ transform}

    def __getitem__(self, index: int) -> dict[str, Any]:
        from torch.utils.data import default_collate

        record_index, start, count = self.windows[index]
        record, arrays = self.records[record_index], self._load_reference(record_index)
        cropped = [self._crop(_inside(self.path.parent, record['image_paths'][offset]),
                             arrays['bbox_xyxy'][offset], bool(arrays['handedness'][offset]),
                             arrays['camera_intrinsics'][offset]) for offset in range(start, start + count)]
        # Only the final official temporal block has repeated image padding.
        values = dict(default_collate(cropped + [cropped[-1]] * (self.clip_length - count)))
        for key, source_key in (('timestamps', 'timestamps'), ('intrinsics', 'camera_intrinsics'),
                                ('camera_to_world', 'camera_to_world'), ('is_right', 'handedness'),
                                ('target_world', 'target_world'), ('target_valid', 'target_valid')):
            value = np.array(arrays[source_key][start:start + count], copy=True)
            if count < self.clip_length:
                value = np.concatenate((value, np.repeat(value[-1:], self.clip_length - count, axis=0)))
            values[key] = torch.from_numpy(value)
        values['is_right'] = values['is_right'].bool()
        values['target_world'] = values['target_world'].float()
        values['target_valid'] = values['target_valid'].bool()
        # GT-crop sources have all-real source images/boxes. Future manifests
        # can retain original detector masks without treating filled boxes as detections.
        for key in ('detected', 'bbox_valid', 'image_valid', 'frame_valid'):
            original = arrays.get(key, np.ones(int(record['frames']), dtype=np.bool_))
            if original.shape != (int(record['frames']),):
                raise ValueError(f"Invalid source {key} mask")
            values[key] = torch.zeros(self.clip_length, dtype=torch.bool)
            values[key][:count] = torch.from_numpy(np.array(original[start:start + count], dtype=np.bool_, copy=True))
        values['target_valid'] &= values['frame_valid'][:, None]
        values['reset'] = torch.zeros(self.clip_length, dtype=torch.bool)
        values['reset'][0] = True
        if count < self.clip_length:
            dt = float(record.get('nominal_dt', 1 / 30))
            if not np.isfinite(dt) or dt <= 0:
                raise ValueError("Padding needs a positive nominal_dt")
            values['timestamps'][count:] = (values['timestamps'][count - 1] +
                torch.arange(1, self.clip_length - count + 1, dtype=torch.float64) * dt)
        values.update(recording=record['recording'], start_frame=start, valid_frames=count)
        return values

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state['_arrays'] = OrderedDict()
        return state
