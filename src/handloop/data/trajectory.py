"""Lazy, recording-disjoint trajectories and source-balanced sampling.

The original single-manifest NPZ format remains supported. A collection uses
``manifests: [{"path": "oak/manifest.json", "dataset": "oakink2"}, ...]``;
paths are relative to the containing manifest. A unified manifest can instead
contain all sequences with explicit dataset/source/coordinate_frame fields.

Storage is one NPZ per sequence, or a directory with state.npy, timestamps.npy,
and optional beta.npy, gt_mask.npy, beta_valid.npy. NPY arrays are memory mapped;
NPZ arrays use an on-demand bounded LRU (NPZ is NOT memory mapped). Initialization
reads timestamps and GT masks, not every state. Window indices are arithmetic
runs per sequence. Prepared files must remain immutable during dataset use.

GT masks have 17 groups: translation, root rotation, 15 local rotations. The
default rejects partial GT, preserving the old full-state training contract.
Explicit require_complete_gt=False exposes masks and finite placeholders for
a future mask-aware trainer; it does NOT produce 99D labels from raw joints or
implement partial/raw-joint supervision. Future is returned ONLY as a target.
"""
from __future__ import annotations

import bisect
import hashlib
from collections import OrderedDict
from collections.abc import Iterator, Sequence
import json
import operator
import os
from pathlib import Path
import zipfile

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


_ARRAY_KEYS = ("state", "timestamps", "beta", "gt_mask", "beta_valid")
_IDENTITY_6D = np.asarray([1, 0, 0, 0, 1, 0], dtype=np.float32)


def _npy_shape(stream) -> tuple[int, ...]:
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
    elif version == (2, 0):
        shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
    else:
        raise ValueError(f"Unsupported prepared NPY header version: {version}")
    if dtype.hasobject:
        raise ValueError("Object arrays are forbidden in prepared trajectories")
    return shape


def _storage_keys_and_state_shape(path: Path) -> tuple[set[str], tuple[int, ...]]:
    if path.is_dir():
        keys = {key for key in _ARRAY_KEYS if (path / f"{key}.npy").is_file()}
        state_path = path / "state.npy"
        if not state_path.is_file():
            raise ValueError(f"{path}: requires prepared state99D; raw joints are not full MANO GT")
        with state_path.open("rb") as stream:
            return keys, _npy_shape(stream)
    with zipfile.ZipFile(path) as archive:
        keys = {name[:-4] for name in archive.namelist() if name.endswith(".npy")}
        if "state" not in keys:
            raise ValueError(f"{path}: requires prepared state99D; raw joints are not full MANO GT")
        with archive.open("state.npy") as stream:
            return keys, _npy_shape(stream)


def _load_arrays(path: Path, keys: Sequence[str]) -> dict[str, np.ndarray]:
    if path.is_dir():
        return {key: np.load(path / f"{key}.npy", mmap_mode="r", allow_pickle=False)
                for key in keys}
    with np.load(path, allow_pickle=False) as sample:
        return {key: sample[key] for key in keys}


def _bool_array(value: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(value)
    if value.dtype != np.bool_ and not np.isin(value, (0, 1)).all():
        raise ValueError(f"{name} must contain only bool/0/1 values")
    return value.astype(np.bool_, copy=False)


def _gt_mask(value: np.ndarray | None, frames: int, name: str) -> np.ndarray:
    if value is None:
        return np.broadcast_to(np.ones(17, dtype=np.bool_), (frames, 17))
    mask = _bool_array(value, name)
    if mask.shape == (17,):
        return np.broadcast_to(mask, (frames, 17))
    if mask.shape != (frames, 17):
        raise ValueError(f"{name}: expected [17] or [{frames},17], got {mask.shape}")
    return mask


def _read_manifest(path: Path, overrides: dict | None = None,
                   ancestors: frozenset[Path] = frozenset()) -> tuple[dict, list[dict]]:
    path = path.resolve()
    if path in ancestors:
        raise ValueError(f"Recursive manifest reference: {path}")
    metadata = json.loads(path.read_text())
    defaults = {key: metadata[key] for key in
                ("dataset", "source", "coordinate_frame", "source_revision",
                 "partial_gt", "gt_complete", "nominal_dt") if key in metadata}
    defaults.update(overrides or {})
    if "dataset" not in defaults:
        # Compatibility with the original Oak manifest, which predates IDs.
        source = str(defaults.get("source", ""))
        defaults["dataset"] = "oakink2" if "oakink2" in source.lower() else "legacy"
    defaults.setdefault("source", defaults["dataset"])
    defaults.setdefault("coordinate_frame", "world" if "world wrist" in
                        str(metadata.get("position", "")) else "unspecified")
    if metadata.get("rotation6d", "columns_c0xyz_c1xyz") != "columns_c0xyz_c1xyz":
        raise ValueError(f"{path}: incompatible rotation6D convention")
    entries = []
    for raw in metadata.get("sequences", []):
        entry = {**defaults, **raw}
        for required in ("path", "split", "recording", "handedness"):
            if required not in entry:
                raise ValueError(f"{path}: sequence missing {required}")
        if not str(entry["dataset"]).strip() or not str(entry["recording"]).strip():
            raise ValueError(f"{path}: dataset and original recording must be nonempty")
        entry["dataset"] = str(entry["dataset"])
        entry["recording"] = str(entry["recording"])
        entry["_path"] = (path.parent / entry["path"]).resolve()
        entry["source_manifest"] = str(path)
        entries.append(entry)
    for reference in metadata.get("manifests", []):
        reference = {"path": reference} if isinstance(reference, str) else dict(reference)
        child_path = path.parent / reference.pop("path")
        expected_hash = reference.pop("manifest_sha256", None)
        if expected_hash is not None and hashlib.sha256(child_path.read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"Prepared child manifest changed: {child_path}")
        # Explicit fields on a reference override that leaf's defaults;
        # collection-level prose does not overwrite a leaf's provenance.
        _, children = _read_manifest(child_path, reference, ancestors | {path})
        entries.extend(children)
    if not entries:
        raise ValueError(f"{path}: no sequence entries or referenced manifests")
    return metadata, entries


class _WindowCenters(Sequence[int]):
    """Compact equivalent of the old list of every valid center index."""

    def __init__(self, times: np.ndarray, history: int, forecast: int,
                 stride: int, nominal_dt: float, gap_factor: float):
        self.stride = stride
        breaks = np.flatnonzero(np.diff(times) >= nominal_dt * gap_factor) + 1
        boundaries = np.concatenate(([0], breaks, [len(times)]))
        self.starts, self.ends = [], []
        total = 0
        for begin, end in zip(boundaries[:-1], boundaries[1:]):
            # Preserve range(history, N-forecast, stride) across timestamp gaps.
            first = history + ((int(begin) + stride - 1) // stride) * stride
            last = int(end) - forecast - 1
            if first <= last:
                total += (last - first) // stride + 1
                self.starts.append(first)
                self.ends.append(total)

    def __len__(self) -> int:
        return self.ends[-1] if self.ends else 0

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        segment = bisect.bisect_right(self.ends, index)
        before = self.ends[segment - 1] if segment else 0
        return self.starts[segment] + (index - before) * self.stride


class TrajectoryWindows(Dataset):
    def __init__(self, manifest: str | Path, split: str, history: int = 30,
                 forecast_offset: int = 3, window_stride: int = 4, *,
                 cache_sequences: int = 4, cache_bytes: int = 256 * 1024**2,
                 require_complete_gt: bool = True, gap_factor: float = 1.5):
        if (history < 0 or forecast_offset < 1 or window_stride < 1 or
                not np.isfinite(gap_factor) or gap_factor <= 1):
            raise ValueError("history>=0, forecast_offset/stride>=1 and gap_factor>1 required")
        if cache_sequences < 0 or cache_bytes < 0:
            raise ValueError("cache bounds must be nonnegative")
        self.history, self.forecast_offset = history, forecast_offset
        self.window_stride, self.split = window_stride, split
        self.root = Path(manifest).resolve().parent
        self.metadata, entries = _read_manifest(Path(manifest))
        self.require_complete_gt = require_complete_gt
        self.cache_sequences, self.cache_bytes = cache_sequences, cache_bytes
        self._cache: OrderedDict[int, tuple[dict, int]] = OrderedDict()
        self._cache_size = 0
        self._cache_pid = os.getpid()
        self._validated_sequences: set[int] = set()

        # Check original-recording assignment BEFORE selecting any windows.
        recording_splits, storage_paths = {}, set()
        for entry in entries:
            key = (entry["dataset"], entry["recording"])
            previous = recording_splits.setdefault(key, entry["split"])
            if previous != entry["split"]:
                raise ValueError(f"Recording crosses splits: {key}: {previous}/{entry['split']}")
            if entry["_path"] in storage_paths:
                raise ValueError(f"Duplicate prepared sequence path: {entry['_path']}")
            storage_paths.add(entry["_path"])
        self.dataset_names = tuple(sorted({entry["dataset"] for entry in entries}))
        self.dataset_to_id = {name: i for i, name in enumerate(self.dataset_names)}
        self.source_revisions = {
            name: sorted({str(e.get("source_revision", "unspecified")) for e in entries
                          if e["dataset"] == name}) for name in self.dataset_names
        }
        self.metadata.setdefault("source_revision", self.source_revisions)
        self.sequences, self.ends = [], []
        self._source_sequences = {name: [] for name in self.dataset_names}
        self._source_ends = {name: [] for name in self.dataset_names}
        self.windows_by_dataset = {name: 0 for name in self.dataset_names}
        self.sequence_counts_by_dataset = {name: 0 for name in self.dataset_names}
        total = 0
        for entry in entries:
            if entry["split"] != split:
                continue
            path = entry["_path"]
            keys, state_shape = _storage_keys_and_state_shape(path)
            if len(state_shape) != 2 or state_shape[1] != 99:
                raise ValueError(f"{path}: state must be [N,99], got {state_shape}")
            frames = state_shape[0]
            if "timestamps" not in keys:
                raise ValueError(f"{path}: timestamps are required")
            if entry.get("frames", frames) != frames:
                raise ValueError(f"{path}: manifest frame count disagrees with state")
            if entry.get("partial_gt", False) or entry.get("gt_complete") is False:
                if "gt_mask" not in keys:
                    raise ValueError(f"{path}: partial labels require an explicit gt_mask")
            small = _load_arrays(path, [key for key in ("timestamps", "gt_mask") if key in keys])
            times = np.asarray(small["timestamps"], dtype=np.float64)
            if times.shape != (frames,) or not np.isfinite(times).all():
                raise ValueError(f"{path}: invalid timestamps")
            if np.any(np.diff(times) <= 0):
                raise ValueError(f"{path}: timestamps must be strictly increasing")
            mask = _gt_mask(small.get("gt_mask"), frames, f"{path}:gt_mask")
            if require_complete_gt and not mask.all():
                raise ValueError(f"{path}: partial GT requires a mask-aware trainer; default is full99D")
            if frames < history + forecast_offset + 1:
                continue
            nominal = float(entry.get("nominal_dt", np.median(np.diff(times))))
            if not np.isfinite(nominal) or nominal <= 0:
                raise ValueError(f"{path}: invalid nominal frame interval")
            centers = _WindowCenters(times, history, forecast_offset, window_stride,
                                     nominal, gap_factor)
            if not len(centers):
                continue
            handedness = int(entry["handedness"])
            if handedness not in (0, 1, 2):
                raise ValueError(f"{path}: handedness must be 0=left,1=right,2=unknown")
            sequence = {**entry, "centers": centers, "frames": frames,
                        "keys": keys, "nominal_dt": nominal, "handedness": handedness,
                        "dataset_id": self.dataset_to_id[entry["dataset"]]}
            seq_id = len(self.sequences)
            self.sequences.append(sequence)
            total += len(centers)
            self.ends.append(total)
            name = entry["dataset"]
            self._source_sequences[name].append(seq_id)
            self.windows_by_dataset[name] += len(centers)
            self._source_ends[name].append(self.windows_by_dataset[name])
            self.sequence_counts_by_dataset[name] += 1
        if total == 0:
            raise ValueError(f"No contiguous {split} windows")

    def __len__(self) -> int:
        return self.ends[-1]

    def __getstate__(self):
        # Spawn workers must not pickle decompressed arrays or mmap handles.
        state = self.__dict__.copy()
        state.update(_cache=OrderedDict(), _cache_size=0, _cache_pid=None)
        return state

    def _arrays(self, seq_id: int) -> dict[str, np.ndarray]:
        if self._cache_pid != os.getpid():
            self._cache.clear()
            self._cache_size, self._cache_pid = 0, os.getpid()
        if seq_id in self._cache:
            self._cache.move_to_end(seq_id)
            return self._cache[seq_id][0]
        seq = self.sequences[seq_id]
        arrays = _load_arrays(seq["_path"], [key for key in _ARRAY_KEYS if key in seq["keys"]])
        state = arrays["state"]
        if state.shape != (seq["frames"], 99):
            raise ValueError(f"{seq['_path']}: state changed after window indexing")
        if seq_id not in self._validated_sequences:
            mask = _gt_mask(arrays.get("gt_mask"), len(state), "gt_mask")
            if self.require_complete_gt and not mask.all():
                raise ValueError(f"{seq['_path']}: incomplete GT")
            # Bound temporary memory while validating a large mapped sequence.
            for begin in range(0, len(state), 8192):
                chunk, available = state[begin:begin+8192], mask[begin:begin+8192]
                p_ok = np.isfinite(chunk[:, :3]) | ~available[:, :1]
                r_ok = np.isfinite(chunk[:, 3:].reshape(-1, 16, 6)) | ~available[:, 1:, None]
                if not (p_ok.all() and r_ok.all()):
                    raise ValueError(f"{seq['_path']}: non-finite available state GT")
            beta = arrays.get("beta")
            if beta is not None and beta.shape not in ((10,), (seq["frames"], 10)):
                raise ValueError(f"{seq['_path']}: beta must be [10] or [N,10]")
            if "beta_valid" in arrays:
                valid = _bool_array(arrays["beta_valid"], "beta_valid")
                if valid.shape not in ((), (seq["frames"],)):
                    raise ValueError(f"{seq['_path']}: beta_valid must be scalar or [N]")
                if beta is None and valid.any():
                    raise ValueError(f"{seq['_path']}: beta_valid cannot mark absent beta as known")
            self._validated_sequences.add(seq_id)
        size = sum(value.nbytes for value in arrays.values())
        if self.cache_sequences and size <= self.cache_bytes:
            while self._cache and (len(self._cache) >= self.cache_sequences or
                                   self._cache_size + size > self.cache_bytes):
                _, (_, removed_size) = self._cache.popitem(last=False)
                self._cache_size -= removed_size
            self._cache[seq_id] = (arrays, size)
            self._cache_size += size
        return arrays

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        seq_id = bisect.bisect_right(self.ends, index)
        seq = self.sequences[seq_id]
        before = self.ends[seq_id - 1] if seq_id else 0
        center = seq["centers"][index - before]
        begin, end = center - self.history, center + self.forecast_offset
        arrays = self._arrays(seq_id)
        times = np.asarray(arrays["timestamps"][begin:center+1], dtype=np.float64)
        first_dt = times[1] - times[0] if len(times) > 1 else seq["nominal_dt"]
        dt = np.diff(times, prepend=times[0]-first_dt).astype(np.float32)
        mask = _gt_mask(arrays.get("gt_mask"), seq["frames"], "gt_mask")

        def state_and_mask(selection):
            # Copies avoid writable Torch views into read-only mmap and keep
            # returned windows independent of cache eviction.
            value = np.array(arrays["state"][selection], dtype=np.float32, copy=True)
            valid = np.array(mask[selection], dtype=np.bool_, copy=True)
            value[..., :3] = np.where(valid[..., :1], value[..., :3], 0.0)
            value[..., 3:] = np.where(valid[..., 1:, None],
                                     value[..., 3:].reshape(*value.shape[:-1], 16, 6),
                                     _IDENTITY_6D).reshape(*value.shape[:-1], 96)
            return torch.from_numpy(value), torch.from_numpy(valid)

        past_current, history_mask = state_and_mask(slice(begin, center+1))
        future, future_mask = state_and_mask(end)
        beta, known = arrays.get("beta"), arrays.get("beta_valid")
        beta_valid = beta is not None and (known is None or bool(known if known.ndim == 0 else known[center]))
        beta_value = np.zeros(10, dtype=np.float32) if not beta_valid else np.array(
            beta if beta.ndim == 1 else beta[center], dtype=np.float32, copy=True)
        if not np.isfinite(beta_value).all():
            raise ValueError(f"{seq['_path']}: non-finite known beta")
        return {
            "past_current": past_current, "future": future, "dt": torch.from_numpy(dt),
            "forecast_dt": torch.tensor([float(arrays["timestamps"][end]-arrays["timestamps"][center])],
                                        dtype=torch.float32),
            "beta": torch.from_numpy(beta_value), "beta_valid": torch.tensor(beta_valid, dtype=torch.bool),
            "handedness": torch.tensor(seq["handedness"], dtype=torch.long),
            "dataset_id": torch.tensor(seq["dataset_id"], dtype=torch.long),
            "past_current_gt_mask": history_mask, "future_gt_mask": future_mask,
        }

    def _global_indices(self, dataset: str, offsets: np.ndarray) -> np.ndarray:
        offsets = np.asarray(offsets, dtype=np.int64)
        total = self.windows_by_dataset[dataset]
        if np.any(offsets < 0) or np.any(offsets >= total):
            raise IndexError(f"Window offset outside dataset {dataset}")
        ends = np.asarray(self._source_ends[dataset], dtype=np.int64)
        sequence_positions = np.searchsorted(ends, offsets, side="right")
        seq_ids = np.asarray(self._source_sequences[dataset], dtype=np.int64)[sequence_positions]
        source_begins = np.concatenate(([0], ends[:-1]))[sequence_positions]
        global_begins = np.asarray([0, *self.ends[:-1]], dtype=np.int64)[seq_ids]
        return global_begins + offsets - source_begins

    def fixed_indices_by_dataset(self, max_per_dataset: int) -> dict[str, list[int]]:
        """Fixed evenly spaced windows per source, including empty sources.

        Callers must reject absent required sources. These windows can overlap
        and are not an independent test set, as with the original validation.
        """
        if max_per_dataset < 1:
            raise ValueError("max_per_dataset must be positive")
        result = {}
        for name, count in self.windows_by_dataset.items():
            offsets = np.linspace(0, count-1, min(count, max_per_dataset), dtype=np.int64)
            result[name] = self._global_indices(name, offsets).tolist() if count else []
        return result


class SourceBalancedSampler(Sampler[int]):
    """Equal source probability, then uniform windows WITH REPLACEMENT.

    This is an initial engineering hypothesis, not an optimal mixture or a
    frame-proportional distribution across datasets. Recordings are not equally
    likely within each source. No per-window weight vector is allocated. RNG
    persists across iterations; set_epoch explicitly reproduces an epoch.
    Saving its RNG does not restore a partly consumed/prefetched loader iterator.
    """

    def __init__(self, dataset: TrajectoryWindows, num_samples: int | None = None,
                 seed: int = 0, datasets: Sequence[str] | None = None):
        self.dataset = dataset
        self.num_samples = len(dataset) if num_samples is None else int(num_samples)
        self.seed = int(seed)
        self.dataset_names = tuple(dataset.dataset_names if datasets is None else datasets)
        if self.num_samples < 1 or not self.dataset_names:
            raise ValueError("positive num_samples and at least one dataset required")
        if len(set(self.dataset_names)) != len(self.dataset_names):
            raise ValueError("duplicate dataset in sampler")
        for name in self.dataset_names:
            if dataset.windows_by_dataset.get(name, 0) < 1:
                raise ValueError(f"Required sampling source has no {dataset.split} windows: {name}")
        self.generator = torch.Generator().manual_seed(self.seed)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.generator.manual_seed(self.seed + int(epoch))

    def __iter__(self) -> Iterator[int]:
        for begin in range(0, self.num_samples, 1024):
            count = min(1024, self.num_samples-begin)
            source_ids = torch.randint(len(self.dataset_names), (count,), generator=self.generator).numpy()
            indices = np.empty(count, dtype=np.int64)
            for source_id, name in enumerate(self.dataset_names):
                positions = np.flatnonzero(source_ids == source_id)
                offsets = torch.randint(self.dataset.windows_by_dataset[name], (len(positions),),
                                        generator=self.generator).numpy()
                indices[positions] = self.dataset._global_indices(name, offsets)
            yield from indices.tolist()

    def state_dict(self) -> dict:
        return {"generator_state": self.generator.get_state(), "seed": self.seed,
                "num_samples": self.num_samples, "dataset_names": self.dataset_names,
                "windows_by_dataset": self.dataset.windows_by_dataset.copy()}

    def load_state_dict(self, state: dict) -> None:
        if (tuple(state["dataset_names"]) != self.dataset_names or
                state["windows_by_dataset"] != self.dataset.windows_by_dataset or
                state["num_samples"] != self.num_samples):
            raise ValueError("Sampler checkpoint does not match the current dataset/window schedule")
        self.seed = int(state["seed"])
        self.generator.set_state(state["generator_state"].cpu())
