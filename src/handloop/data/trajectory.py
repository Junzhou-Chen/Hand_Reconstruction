"""Contiguous past/current/future windows from auditable prepared trajectories.

Future states are returned ONLY as supervision; the trainer never sends them
to TWM. Every file belongs to one original recording/hand/split. Windows never
cross recording or timestamp gaps. Prepared states use columns rotation6D.
"""
from __future__ import annotations

import bisect
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class TrajectoryWindows(Dataset):
    def __init__(self, manifest: str | Path, split: str, history: int = 30,
                 forecast_offset: int = 3, window_stride: int = 4):
        self.history = history
        self.forecast_offset = forecast_offset
        self.root = Path(manifest).parent
        self.metadata = json.loads(Path(manifest).read_text())
        self.sequences, self.ends = [], []
        total = 0
        for entry in self.metadata["sequences"]:
            if entry["split"] != split:
                continue
            with np.load(self.root / entry["path"], allow_pickle=False) as sample:
                states = sample["state"].astype(np.float32)
                timestamps = sample["timestamps"].astype(np.float64)
                beta = sample["beta"].astype(np.float32)
            if len(states) < history + forecast_offset + 1:
                continue
            if states.shape[1:] != (99,) or not np.isfinite(states).all():
                raise ValueError(f"Invalid state in {entry['path']}")
            if np.any(np.diff(timestamps) <= 0):
                raise ValueError(f"Non-increasing time in {entry['path']}")
            nominal = float(np.median(np.diff(timestamps)))
            good = np.diff(timestamps) < nominal * 1.5
            centers = []
            for center in range(history, len(states)-forecast_offset, window_stride):
                if good[center-history:center+forecast_offset].all():
                    centers.append(center)
            if not centers:
                continue
            self.sequences.append({"state": torch.from_numpy(states),
                                   "time": timestamps, "beta": torch.from_numpy(beta),
                                   "handedness": entry["handedness"],
                                   "centers": centers, "recording": entry["recording"]})
            total += len(centers)
            self.ends.append(total)
        if total == 0:
            raise ValueError(f"No contiguous {split} windows")
        self.split = split

    def __len__(self):
        return self.ends[-1]

    def __getitem__(self, index):
        seq_id = bisect.bisect_right(self.ends, index)
        seq = self.sequences[seq_id]
        prior = self.ends[seq_id-1] if seq_id else 0
        center = seq["centers"][index-prior]
        begin = center - self.history
        times = seq["time"][begin:center+1]
        dt = np.diff(times, prepend=times[0]-(times[1]-times[0])).astype(np.float32)
        return {"past_current": seq["state"][begin:center+1],
                "future": seq["state"][center+self.forecast_offset],
                "dt": torch.from_numpy(dt),
                "forecast_dt": torch.tensor([seq["time"][center+self.forecast_offset]
                                             -seq["time"][center]], dtype=torch.float32),
                "beta": seq["beta"],
                "handedness": torch.tensor(seq["handedness"], dtype=torch.long)}
