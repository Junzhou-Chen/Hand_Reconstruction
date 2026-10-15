"""Rank-zero scalar experiment tracking without uploading models or datasets.

Authentication uses the normal W&B environment/netrc configuration. API keys
are never accepted in a tracked config or written into our run metadata. An
offline run is explicitly labelled offline: it does not supply a live URL.
"""
from __future__ import annotations

from collections.abc import Mapping
import json
import math
import numbers
import os
from pathlib import Path
from typing import Any
import uuid


def _public_config(value: Any) -> Any:
    """Sanitize nested experiment metadata before it leaves this process."""
    if isinstance(value, Mapping):
        return {
            str(key): ("[redacted]" if any(term in str(key).lower()
                       for term in ("api_key", "apikey", "password", "secret", "credential", "token"))
                       else _public_config(item))
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_public_config(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    # Avoid accidentally serializing a model, tensor, or loaded dataset.
    raise TypeError(f"W&B config must contain JSON-like metadata, got {type(value).__name__}")


def _scalar_metrics(metrics: Mapping[str, Any], prefix: str = "") -> dict[str, float | int]:
    """Flatten nested metric dictionaries; arrays and raw samples are rejected."""
    result: dict[str, float | int] = {}
    for key, value in metrics.items():
        name = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            result.update(_scalar_metrics(value, name))
        elif value is None:
            continue
        elif isinstance(value, numbers.Real):
            # JSON/W&B cannot reliably compare nonfinite values. Preserve a
            # visible failure counter instead of silently hiding a bad metric.
            if math.isfinite(float(value)):
                result[name] = float(value)
            else:
                result[name + "_nonfinite"] = 1
        else:
            raise TypeError(f"Metric {name!r} must be a scalar number or nested mapping")
    return result


class WandbTracker:
    """Optional W&B run owned by rank 0, with explicit online/offline status.

    Config fields: enabled (default False), mode (online/offline/disabled),
    project (default WANDB_PROJECT or handloop-twm), entity (optional), name,
    group, tags, require_online (default False), and init_timeout (seconds).
    Normal W&B credentials must already be configured outside the repository.
    On a checkpoint resume, pass resume=True and reuse the same output_dir.
    """

    def __init__(self, config: Mapping[str, Any] | None, output_dir: str | Path,
                 metadata: Mapping[str, Any] | None = None, *, rank: int = 0,
                 resume: bool = False):
        self.run = None
        self.url: str | None = None
        self.metadata: dict[str, Any] = {"enabled": False, "rank": rank, "mode": "disabled"}
        self._state_path: Path | None = None
        self._last_logged_step = -1
        options = dict(config or {})
        mode = options.get("mode", "online")
        if mode not in ("online", "offline", "disabled"):
            raise ValueError("W&B mode must be online, offline, or disabled")
        enabled = options.get("enabled", False) and mode != "disabled"
        if rank != 0:
            return
        if options.get("require_online", False) and (not enabled or mode != "online"):
            raise ValueError("require_online needs enabled=True and mode='online'")
        if not enabled:
            return
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        self._state_path = output / "wandb_run.json"
        previous = None
        if self._state_path.exists():
            if not resume:
                raise ValueError("W&B metadata already exists; use resume=True or a new run directory")
            previous = json.loads(self._state_path.read_text())
        run_id = previous["id"] if previous else uuid.uuid4().hex[:8]
        project = options.get("project") or os.environ.get("WANDB_PROJECT") or "handloop-twm"
        entity = options.get("entity") or os.environ.get("WANDB_ENTITY")
        # The SDK resolves an omitted entity to the authenticated account on
        # first init. Preserve that resolved account on subsequent resumes.
        if previous and entity is None:
            entity = previous.get("entity")
        if previous and (previous.get("project") != project or previous.get("entity") != entity):
            raise ValueError("W&B resume must retain project and entity")
        self.metadata = {"enabled": True, "id": run_id, "project": project,
                         "entity": entity, "mode": mode, "url": None,
                         "state": "initializing"}
        self._write_state()
        try:
            import wandb
        except ImportError:
            self.metadata["state"] = "missing_dependency"
            self._write_state()
            raise RuntimeError("Install wandb in the remote training runtime before enabling tracking") from None
        # Disable implicit source/diff/stdout uploads. Only config, scalar
        # metrics, and the SDK's normal system metrics are transmitted.
        settings = wandb.Settings(console="off", disable_code=True, disable_git=True,
                                  save_code=False,
                                  init_timeout=float(options.get("init_timeout", 45)))
        arguments = dict(project=project, entity=entity, name=options.get("name", output.name),
                         group=options.get("group"), tags=options.get("tags", []),
                         id=run_id, mode=mode, dir=str(output), save_code=False,
                         config=_public_config(dict(metadata or {})), settings=settings)
        if mode == "online":
            arguments["resume"] = "allow" if resume else "never"
            arguments["force"] = True  # Do not silently create an anonymous account.
        try:
            self.run = wandb.init(**arguments)
        except Exception as error:
            # Exception messages may include network headers or credentials;
            # store only the exception type in our own reports.
            self.metadata.update(state="initialization_failed", error_type=type(error).__name__)
            self._write_state()
            raise RuntimeError(f"W&B {mode} initialization failed ({type(error).__name__}); "
                               "check remote credentials and connectivity") from None
        actual_mode = str(getattr(self.run.settings, "mode", mode))
        if actual_mode != mode:
            self.run.finish(exit_code=1)
            self.run = None
            self.metadata.update(state="mode_mismatch", actual_mode=actual_mode)
            self._write_state()
            raise RuntimeError(f"W&B requested {mode}, but initialized {actual_mode}")
        self.url = self.run.url if mode == "online" else None
        self.metadata.update(entity=self.run.entity, project=self.run.project,
                             url=self.url, state="running")
        self._write_state()
        # Explicit metric axes let train and validation share a progress value
        # without W&B dropping one record as an out-of-order internal step.
        self.run.define_metric("optimizer_step")
        if metadata.get("config", {}).get("training_unit") == "epoch":
            self.run.define_metric("epoch_progress")
            self.run.define_metric("*", step_metric="epoch_progress")
        else:
            self.run.define_metric("*", step_metric="optimizer_step")

    def _write_state(self) -> None:
        if self._state_path is None:
            return
        temporary = self._state_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.metadata, indent=2, sort_keys=True) + "\n")
        temporary.replace(self._state_path)

    def log(self, metrics: Mapping[str, Any], step: int) -> None:
        if self.run is None:
            return
        if step < self._last_logged_step:
            raise ValueError("W&B optimizer steps must be nondecreasing")
        values = _scalar_metrics(metrics)
        values["optimizer_step"] = int(step)
        self.run.log(values)
        self._last_logged_step = int(step)
        self.metadata["last_optimizer_step"] = int(step)
        self._write_state()

    def finish(self, exit_code: int = 0) -> None:
        if self.run is None:
            return
        self.run.finish(exit_code=exit_code)
        self.metadata.update(state="finished" if exit_code == 0 else "failed",
                             exit_code=exit_code)
        self._write_state()
        self.run = None

