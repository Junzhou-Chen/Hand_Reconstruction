"""Immutable, recording-disjoint frozen-HaWoR caches for visual feedback.

The files contain real image patch features only on the visual side. A caller
passes numeric states to TWM; this dataset never adds features to its interface.
Camera-space DexYCB uses its fixed camera as a named world gauge, not an
estimated moving/egocentric world frame. Capture times must keep their source
label: Dex frame index / 30 is nominal time, not a measured timestamp.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


CACHE_SCHEMA = "handloop.visual_feedback_cache.v1"
CORE_KEYS = (
    "patch_features", "raw_world", "timestamps", "frame_valid", "availability",
    "image_valid", "handedness", "world_to_camera", "camera_intrinsics",
    "crop_transform", "beta", "reset", "target_world", "target_valid",
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def read_cache_manifest(path: str | Path) -> dict[str, Any]:
    """Validate the split boundary before opening any cached tensor files."""
    manifest = json.loads(Path(path).read_text())
    if manifest.get("schema") != CACHE_SCHEMA:
        raise ValueError(f"Expected {CACHE_SCHEMA}, got {manifest.get('schema')!r}")
    if manifest.get("phase") != "complete":
        raise ValueError("An incomplete cache cannot be used for training")
    if manifest.get("crop_transform_convention") != "source_pixels_to_hawor_192x256_normalized":
        raise ValueError("Cache crop transforms must match the visual patch query coordinate convention")
    seen: dict[str, str] = {}
    paths: set[str] = set()
    for entry in manifest.get("records", []):
        split, recording = entry.get("split"), entry.get("recording")
        if split not in ("train", "val"):
            raise ValueError("This training cache accepts train/val only; test is excluded")
        if not recording or recording in seen:
            raise ValueError(f"Duplicate or cross-split recording: {recording}")
        seen[recording] = split
        if entry.get("path") in paths:
            raise ValueError("Two recordings share a tensor directory")
        paths.add(entry.get("path"))
        if entry.get("coordinate_frame") is None or entry.get("timestamp_source") is None:
            raise ValueError("Coordinate and timestamp provenance must be explicit")
        hashes, sizes = entry.get("tensor_sha256", {}), entry.get("tensor_bytes", {})
        if set(hashes) != set(CORE_KEYS) or set(sizes) != set(CORE_KEYS):
            raise ValueError("Every cached tensor requires its own SHA256 and byte count")
        if any(not isinstance(hashes[key], str) or len(hashes[key]) != 64 or
               not isinstance(sizes[key], int) or sizes[key] <= 0 for key in CORE_KEYS):
            raise ValueError("Invalid tensor identity in cache manifest")
    if not seen:
        raise ValueError("No cache records")
    return manifest


class VisualFeedbackClipDataset(Dataset):
    """Lazy NPY-backed clips; each returned clip starts with empty history.

    ``clip_length`` can span several original 16-frame HaWoR chunks: cached
    features preserve those chunk boundaries and the feedback model traverses
    the clip sequentially. No split is ever assigned after window extraction.
    Tail padding repeats numerical values but has frame/image/target masks
    false; it cannot create extra supervision or committed history.
    """

    def __init__(self, manifest: str | Path, split: str = "train", *,
                 clip_length: int = 32, stride: int | None = None,
                 include_tail: bool = True, max_open_records: int = 4) -> None:
        if split not in ("train", "val"):
            raise ValueError("Training loader deliberately excludes test")
        if clip_length < 1 or (stride is not None and stride < 1):
            raise ValueError("clip_length and stride must be positive")
        self.path = Path(manifest).expanduser().resolve()
        self.manifest = read_cache_manifest(self.path)
        self.manifest_sha256 = file_sha256(self.path)
        self.records = [r for r in self.manifest["records"] if r["split"] == split]
        self.clip_length = int(clip_length)
        self.stride = int(stride or clip_length)
        self.max_open_records = max(1, int(max_open_records))
        self._arrays: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()
        self._open_identities: dict[int, dict[str, tuple[int, int]]] = {}
        self.windows: list[tuple[int, int, int]] = []
        for index, record in enumerate(self.records):
            total = int(record["frames"])
            for start in range(0, total, self.stride):
                count = min(self.clip_length, total - start)
                if count == self.clip_length or include_tail:
                    self.windows.append((index, start, count))
        if not self.windows:
            raise ValueError(f"No {split} clips for clip_length={clip_length}")

    def __len__(self) -> int:
        return len(self.windows)

    def _load(self, index: int) -> dict[str, np.ndarray]:
        record = self.records[index]
        root = (self.path.parent / record["path"]).resolve()
        if not root.is_relative_to(self.path.parent):
            raise ValueError("Cache tensor path escapes manifest directory")
        identities = {}
        for key in CORE_KEYS:
            stat = (root / (key + ".npy")).stat()
            identities[key] = (stat.st_size, stat.st_mtime_ns)
        if index in self._arrays:
            if identities != self._open_identities[index]:
                raise ValueError(f"Cached tensors changed after verification: {record['recording']}")
            self._arrays.move_to_end(index)
            return self._arrays[index]
        for key in CORE_KEYS:
            if (identities[key][0] != record["tensor_bytes"][key] or
                    file_sha256(root / (key + ".npy")) != record["tensor_sha256"][key]):
                raise ValueError(f"Cached tensor failed SHA256/size verification: {record['recording']}/{key}")
        arrays = {key: np.load(root / (key + ".npy"), mmap_mode="r", allow_pickle=False)
                  for key in CORE_KEYS}
        n = int(record["frames"])
        expected = {"patch_features": (n, 192, 1280), "raw_world": (n, 99),
                    "timestamps": (n,), "frame_valid": (n,), "availability": (n, 17),
                    "image_valid": (n,), "handedness": (n,),
                    "world_to_camera": (n, 4, 4), "camera_intrinsics": (n, 3, 3),
                    "crop_transform": (n, 3, 3), "beta": (n, 10), "reset": (n,),
                    "target_world": (n, 99), "target_valid": (n, 17)}
        for key, shape in expected.items():
            if arrays[key].shape != shape or arrays[key].dtype.hasobject:
                raise ValueError(f"Invalid {key} cache shape/type: {arrays[key].shape}")
        if arrays["timestamps"].dtype != np.float64:
            raise ValueError("Keep timestamps in float64 seconds")
        if n > 1 and not np.all(np.diff(arrays["timestamps"]) > 0):
            raise ValueError("Timestamps must be strictly increasing within a track")
        self._arrays[index] = arrays
        self._open_identities[index] = identities
        while len(self._arrays) > self.max_open_records:
            evicted, _ = self._arrays.popitem(last=False)
            self._open_identities.pop(evicted, None)
        return arrays

    def __getitem__(self, index: int) -> dict[str, Any]:
        record_index, start, count = self.windows[index]
        arrays = self._load(record_index)
        values: dict[str, Any] = {}
        for key, array in arrays.items():
            value = np.array(array[start:start + count], copy=True)
            if count < self.clip_length:
                value = np.concatenate((value, np.repeat(value[-1:], self.clip_length - count, axis=0)))
            values[key] = torch.from_numpy(value)
        values["reset"][0] = True
        if count < self.clip_length:
            for key in ("frame_valid", "image_valid", "availability", "target_valid"):
                values[key][count:] = False
            # Distinct padded times remain unavailable, never repeated captures.
            dt = float(self.records[record_index].get("nominal_dt", 1 / 30))
            values["timestamps"][count:] = (values["timestamps"][count - 1]
                + torch.arange(1, self.clip_length - count + 1, dtype=torch.float64) * dt)
            values["reset"][count:] = False
        values["recording"] = self.records[record_index]["recording"]
        values["start_frame"] = start
        values["valid_frames"] = count
        return values

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_arrays"] = OrderedDict()
        state["_open_identities"] = {}
        return state
