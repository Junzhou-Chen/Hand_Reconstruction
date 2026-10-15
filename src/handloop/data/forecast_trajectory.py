"""Current restoration windows with a masked, contiguous future trajectory.

This leaves :class:`TrajectoryWindows` and its single-offset target unchanged.
The new dataset uses exactly its minimum-offset window population and sampling
indices, then exposes up to ``forecast_frames`` subsequent source frames.
Actual prepared timestamps, in seconds, are authoritative. No temporal
interpolation, extrapolated GT, gap bridging, or repeated tail labels are used.
"""
from __future__ import annotations

import bisect
import operator
from pathlib import Path

import numpy as np
import torch

from .trajectory import TrajectoryWindows, _gt_mask, _load_arrays, _IDENTITY_6D


class ForecastTrajectoryWindows(TrajectoryWindows):
    """Lazy full-MANO windows; futures may end before the requested horizon.

    Each sample adds ``future``/``future_states`` [F,99], ``future_dt``/
    ``forecast_dt`` [F], ``future_valid`` [F], and ``future_gt_mask`` [F,17].
    ``past_current`` and its metadata retain the original dataset contract.
    ``future`` contains targets only and must never be passed as model inputs.

    Valid future slots form a prefix ending at the first timestamp gap or the
    sequence boundary. Invalid targets use p=0 / rotation=identity and have
    false GT masks. Their query timestamps continue at the last valid interval
    to keep all F query times finite and increasing; these are query locations,
    not measured target times. Losses and metrics must use ``future_valid``.

    The default F=60 is approximately two seconds for the existing 30 Hz
    preparations. Physical offsets, rather than frame indices divided by an
    assumed FPS, are supplied to the network. Keeping min_future_frames=3
    retains the original short-target windows, including short DexYCB clips.
    """

    def __init__(self, manifest: str | Path, split: str, history: int = 30,
                 forecast_frames: int = 60, min_future_frames: int = 3,
                 window_stride: int = 4, *,
                 coverage_seconds: tuple[float, ...] = (0.1, 0.5, 1.0, 2.0),
                 **kwargs):
        if (not isinstance(forecast_frames, int) or not isinstance(min_future_frames, int)
                or not 1 <= min_future_frames <= forecast_frames):
            raise ValueError("1 <= min_future_frames <= forecast_frames required")
        if not kwargs.get("require_complete_gt", True):
            raise ValueError("ForecastTrajectoryWindows currently requires full 17-group MANO GT")
        coverage = np.asarray(coverage_seconds, dtype=np.float64)
        if (coverage.ndim != 1 or not len(coverage) or not np.isfinite(coverage).all()
                or np.any(coverage <= 0) or np.any(np.diff(coverage) <= 0)):
            raise ValueError("coverage_seconds must be finite, positive, and strictly increasing")
        self.forecast_frames = forecast_frames
        self.min_future_frames = min_future_frames
        super().__init__(manifest, split, history, min_future_frames, window_stride, **kwargs)
        gap_factor = float(kwargs.get("gap_factor", 1.5))
        self.future_coverage = {
            name: {"windows": self.windows_by_dataset[name],
                   "valid_windows_by_step": [0] * forecast_frames,
                   "windows_reaching_seconds": {str(float(t)): 0 for t in coverage},
                   "minimum_observed_seconds": None,
                   "maximum_observed_seconds": None}
            for name in self.dataset_names
        }
        # Timestamp-only audit. Do not open state arrays while enumerating the
        # collection, or retain decompressed sequence states in this metadata.
        for seq in self.sequences:
            times = np.asarray(_load_arrays(seq["_path"], ["timestamps"])["timestamps"],
                               dtype=np.float64)
            segment_ends = np.concatenate((
                np.flatnonzero(np.diff(times) >= seq["nominal_dt"] * gap_factor) + 1,
                [len(times)]))
            seq["future_segment_ends"] = segment_ends.astype(np.int64)
            stats = self.future_coverage[seq["dataset"]]
            histogram = np.zeros(forecast_frames + 1, dtype=np.int64)
            for begin in range(0, len(seq["centers"]), 8192):
                centers = np.fromiter(
                    (seq["centers"][i] for i in range(begin, min(begin + 8192, len(seq["centers"])))),
                    dtype=np.int64)
                ends = segment_ends[np.searchsorted(segment_ends, centers, side="right")]
                counts = np.minimum(forecast_frames, ends - centers - 1)
                if np.any(counts < min_future_frames):
                    raise ValueError("Minimum-offset indexing crossed a timestamp boundary")
                histogram += np.bincount(counts, minlength=forecast_frames + 1)
                observed = times[centers + counts] - times[centers]
                for seconds in coverage:
                    stats["windows_reaching_seconds"][str(float(seconds))] += int(
                        np.count_nonzero(observed >= seconds - 1e-6))
                low, high = float(observed.min()), float(observed.max())
                previous_low, previous_high = stats["minimum_observed_seconds"], stats["maximum_observed_seconds"]
                stats["minimum_observed_seconds"] = low if previous_low is None else min(previous_low, low)
                stats["maximum_observed_seconds"] = high if previous_high is None else max(previous_high, high)
            by_step = np.cumsum(histogram[::-1])[::-1][1:]
            stats["valid_windows_by_step"] = (
                np.asarray(stats["valid_windows_by_step"], dtype=np.int64) + by_step).tolist()
        self.forecast_contract = {
            "history_frames": history, "forecast_frames": forecast_frames,
            "min_future_frames": min_future_frames, "window_stride": window_stride,
            "timestamp_units": "seconds", "query_time": "actual source offsets; padded queries continue last valid cadence",
            "target_policy": "next contiguous frames, masked at first timestamp gap or sequence end",
            "gt_policy": "complete 17-group MANO only; padded targets have false GT masks",
            "coverage_seconds_comparison": "maximum valid target offset >= horizon minus 1e-6 seconds",
            "future_coverage": self.future_coverage,
        }

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        # Reuse exact history/current, shape, handedness and source handling.
        result = super().__getitem__(index)
        seq_id = bisect.bisect_right(self.ends, index)
        seq = self.sequences[seq_id]
        before = self.ends[seq_id - 1] if seq_id else 0
        center = seq["centers"][index - before]
        segment_end = int(seq["future_segment_ends"][np.searchsorted(
            seq["future_segment_ends"], center, side="right")])
        count = min(self.forecast_frames, segment_end - center - 1)
        arrays = self._arrays(seq_id)
        mask = _gt_mask(arrays.get("gt_mask"), seq["frames"], "gt_mask")
        target = np.zeros((self.forecast_frames, 99), dtype=np.float32)
        target[:, 3:] = np.tile(_IDENTITY_6D, 16)
        target[:count] = np.asarray(arrays["state"][center + 1:center + count + 1],
                                    dtype=np.float32)
        gt_mask = np.zeros((self.forecast_frames, 17), dtype=np.bool_)
        gt_mask[:count] = mask[center + 1:center + count + 1]
        if not gt_mask[:count].all():
            raise ValueError("Available future target lacks complete 17-group MANO GT")
        valid = np.arange(self.forecast_frames) < count
        times = np.asarray(arrays["timestamps"][center:center + count + 1], dtype=np.float64)
        offsets = np.empty(self.forecast_frames, dtype=np.float64)
        offsets[:count] = times[1:] - times[0]
        if count < self.forecast_frames:
            last_interval = times[-1] - times[-2]
            offsets[count:] = offsets[count - 1] + last_interval * np.arange(
                1, self.forecast_frames - count + 1)
        future = torch.from_numpy(target)
        future_dt = torch.from_numpy(offsets.astype(np.float32))
        result.update(future=future, future_states=future, future_dt=future_dt,
                      forecast_dt=future_dt, future_valid=torch.from_numpy(valid),
                      future_gt_mask=torch.from_numpy(gt_mask))
        return result
