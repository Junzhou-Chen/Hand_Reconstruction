"""Check explicitly requested visual training sources before loading models."""
from __future__ import annotations

import json
from pathlib import Path


def require_visual_datasets(manifest_path: str | Path, required: list[str] | None) -> None:
    """A missing source must never silently become a different experiment.

    This checks source membership only. Dataset-specific split, reference,
    coordinate and image validation remains the RGB loader's responsibility.
    Historical configs without an explicit source requirement are unchanged.
    """
    if required is None:
        return
    if (not isinstance(required, list) or not required
            or any(not isinstance(name, str) or not name for name in required)
            or len(required) != len(set(required))):
        raise ValueError('required_visual_datasets must contain distinct source names')
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get('phase') != 'complete':
        raise ValueError('Visual source preparation is incomplete')
    actual = {record.get('dataset', manifest.get('dataset'))
              for record in manifest.get('records', [])
              if record.get('split') == 'train' and record.get('frames', 0) > 0}
    expected = set(required)
    if actual != expected:
        raise ValueError('Visual training sources differ from the requested experiment: '
                         f'missing={sorted(expected - actual)}, '
                         f'unexpected={sorted(str(name) for name in actual - expected)}')
