#!/usr/bin/env python3
"""Prepare the user's licensed MANO assets on the remote ds Pod only.

Verify the official ZIP SHA256, extract only MANO_LEFT/RIGHT.pkl and license
files, preserve those originals, and create a separate derived/ model pair.
Old Chumpy objects are evaluated to plain NumPy arrays in a child process with
process-local compatibility bridges. No original MANO webuser code is run.
A fresh child process blocks Chumpy imports while loading the derived pickles
and running a tiny CPU smplx check. Neither the ZIP nor raw models are replaced.

Typical usage (after supplying licensed assets and isolated-runtime packages):
  python scripts/prepare_mano_assets.py --zip /path/mano_v1_2.zip \
      --output /path/mano_v1_2/prepared_v1

The default expected hash identifies the official ZIP downloaded by the user
on 2026-10-14. A different authorized download requires its explicit hash.
The official smplx tools/clean_ch.py motivates array conversion; this script
adds Python 3/NumPy 2 compatibility, unchanged-value checks and provenance.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path, PurePosixPath
import pickle
import re
import stat
import subprocess
import sys
import zipfile


EXPECTED_SHA256 = "50976831790ea9657d8110e0c94e50e90eaf35cd76169f0b27e5d32f3fcd951f"
REQUIRED_FIELDS = ("v_template", "shapedirs", "posedirs", "J_regressor", "weights",
                   "kintree_table", "f", "hands_components", "hands_mean")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def versions(*packages: str) -> dict:
    result = {"python": sys.version, "executable": sys.executable}
    for package in packages:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def install_legacy_bridges() -> dict:
    """Only used by the short-lived conversion child, never by model training."""
    import builtins
    from collections import namedtuple
    import inspect
    import numpy as np

    replacements = {"bool": np.bool_, "int": builtins.int, "float": builtins.float,
                    "complex": builtins.complex, "object": builtins.object,
                    "unicode": builtins.str, "str": builtins.str}
    added = []
    for name, replacement in replacements.items():
        # Access __dict__ directly: NumPy's deprecated-attribute lookup warns.
        if name not in np.__dict__:
            setattr(np, name, replacement)
            added.append(name)
    inspect_bridge = not hasattr(inspect, "getargspec")
    if inspect_bridge:
        ArgSpec = namedtuple("ArgSpec", "args varargs keywords defaults")

        def getargspec(function):
            full = inspect.getfullargspec(function)
            return ArgSpec(full.args, full.varargs, full.varkw, full.defaults)

        inspect.getargspec = getargspec
    return {"numpy_aliases_added": added, "inspect_getargspec_bridge": inspect_bridge,
            "scope": "conversion subprocess only; no installed package files changed"}


def convert_one(raw_path: Path, destination: Path) -> dict:
    bridges = install_legacy_bridges()
    import numpy as np
    import scipy.sparse as sparse
    import chumpy

    replaced = []

    def plain(value, location):
        if isinstance(value, chumpy.Ch):
            replaced.append(location)
            value = np.asarray(value.r)
        if sparse.issparse(value):
            if value.dtype.hasobject or not np.isfinite(value.data).all():
                raise ValueError(f"Invalid sparse numeric data: {location}")
            return value.copy()
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                raise ValueError(f"Object array forbidden in converted MANO: {location}")
            if value.dtype.kind in "fc" and not np.isfinite(value).all():
                raise ValueError(f"Non-finite numeric array: {location}")
            return np.array(value, copy=True, subok=False)
        if isinstance(value, np.generic):
            return value.copy()
        if isinstance(value, dict):
            if not all(isinstance(key, (str, int)) for key in value):
                raise TypeError(f"Unsupported dictionary key: {location}")
            return {key: plain(item, f"{location}.{key}") for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            converted = [plain(item, f"{location}[{index}]") for index, item in enumerate(value)]
            return tuple(converted) if isinstance(value, tuple) else converted
        if value is None or isinstance(value, (str, bytes, int, float, bool)):
            return value
        raise TypeError(f"Unsupported object {type(value).__module__}.{type(value).__name__}: {location}")

    with raw_path.open("rb") as stream:
        original = pickle.load(stream, encoding="latin1")
    if not isinstance(original, dict):
        raise TypeError("MANO pickle must contain a dictionary")
    derived = plain(original, "model")
    fields = {}
    for key in REQUIRED_FIELDS:
        if key not in derived:
            raise ValueError(f"Required MANO field missing: {key}")
        source = original[key]
        if isinstance(source, chumpy.Ch):
            source = source.r
        source = source.toarray() if sparse.issparse(source) else np.asarray(source)
        target = derived[key]
        target = target.toarray() if sparse.issparse(target) else np.asarray(target)
        if source.dtype != target.dtype or not np.array_equal(source, target):
            raise ValueError(f"Conversion changed MANO values or dtype: {key}")
        fields[key] = {"shape": list(target.shape), "dtype": target.dtype.str,
                       "value_sha256": hashlib.sha256(np.ascontiguousarray(target).tobytes()).hexdigest(),
                       "sparse_preserved": bool(sparse.issparse(derived[key]))}
    if destination.exists():
        raise FileExistsError(f"Derived model already exists: {destination}")
    temporary = destination.with_suffix(".pkl.tmp")
    with temporary.open("wb") as stream:
        pickle.dump(derived, stream, protocol=4)
    temporary.replace(destination)
    return {"raw_sha256": sha256(raw_path), "derived_sha256": sha256(destination),
            "raw_bytes": raw_path.stat().st_size, "derived_bytes": destination.stat().st_size,
            "chumpy_objects_replaced": replaced, "numeric_fields": fields, "bridges": bridges,
            "environment": versions("numpy", "scipy", "chumpy")}


def verify_derived(directory: Path) -> dict:
    """Prove that the derived models no longer need Chumpy to unpickle."""
    import importlib.abc

    class BlockChumpy(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "chumpy" or fullname.startswith("chumpy."):
                raise ImportError("Verification intentionally blocks Chumpy imports")
            return None

    sys.meta_path.insert(0, BlockChumpy())
    import numpy as np
    import scipy.sparse as sparse
    import torch
    import smplx

    torch.set_num_threads(1)
    results = {}
    for side in ("left", "right"):
        path = directory / ("MANO_" + side.upper() + ".pkl")
        with path.open("rb") as stream:
            values = pickle.load(stream, encoding="latin1")
        for key in REQUIRED_FIELDS:
            value = values[key]
            if not isinstance(value, np.ndarray) and not sparse.issparse(value):
                raise TypeError(f"Derived field is not a NumPy/SciPy numeric array: {key}")
        layer = smplx.create(str(path), "mano", is_rhand=side == "right", use_pca=False,
                             num_pca_comps=45, flat_hand_mean=False, num_betas=10).cpu().eval()
        layer.requires_grad_(False)
        if layer.use_pca or tuple(layer.np_hand_components.shape) != (45, 45):
            raise ValueError("Unexpected smplx MANO PCA convention")
        with torch.no_grad():
            output = layer(betas=torch.zeros(2, 10), global_orient=torch.zeros(2, 3),
                           hand_pose=torch.zeros(2, 45), transl=torch.tensor([[0., 0., 0.], [.01, .02, .03]]),
                           return_verts=True, return_full_pose=True)
        if (output.vertices.shape != (2, 778, 3) or output.joints.shape[1] < 16
                or output.full_pose.shape != (2, 48)):
            raise ValueError("Unexpected smplx MANO output shapes")
        if not all(torch.isfinite(value).all() for value in (output.vertices, output.joints, output.full_pose)):
            raise ValueError("Non-finite smplx geometry")
        mean_error = float((output.full_pose[:, 3:] - layer.hand_mean).abs().max())
        translation_error = float((output.joints[1] - output.joints[0]
                                   - torch.tensor([.01, .02, .03])).abs().max())
        if mean_error > 1e-6 or translation_error > 1e-6:
            raise ValueError("smplx mean/translation check failed")
        results[side] = {"vertices_shape": list(output.vertices.shape), "joints_shape": list(output.joints.shape),
                         "zero_offset_mean_max_abs": mean_error, "translation_max_abs_m": translation_error}
    if any(name == "chumpy" or name.startswith("chumpy.") for name in sys.modules):
        raise RuntimeError("Chumpy unexpectedly loaded during clean verification")
    return {"passed": True, "chumpy_imports_blocked": True, "smplx_checks": results,
            "environment": versions("numpy", "scipy", "torch", "smplx"),
            "scope": "model loading/mean/translation only; no dataset FK agreement is claimed"}


def child_result(arguments: list[str]) -> dict:
    command = [sys.executable, str(Path(__file__).resolve()), *arguments]
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f"Asset subprocess failed ({arguments[0]}):\n{result.stdout}\n{result.stderr}")
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("Asset subprocess returned no report")
    report = json.loads(lines[-1])
    report["runtime_messages"] = lines[:-1]
    if result.stderr.strip():
        report["runtime_stderr"] = result.stderr.strip()
    return report


def prepare(archive_path: Path, output: Path, expected: str) -> dict:
    actual = sha256(archive_path)
    if actual != expected:
        raise ValueError(f"ZIP SHA256 mismatch: expected {expected}, found {actual}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Output directory is not empty; use a new directory to preserve raw/derived provenance")
    output.mkdir(parents=True, exist_ok=True)
    raw, derived = output / "raw", output / "derived"
    raw.mkdir()
    derived.mkdir()
    status_path = output / "preparation_status.json"
    save_json(status_path, {"phase": "extracting", "utc": now()})
    report = {"started_utc": now(), "source_zip": str(archive_path.resolve()),
              "source_zip_sha256": actual, "source_zip_bytes": archive_path.stat().st_size,
              "asset_release_label": "MANO v1.2 (user-supplied official archive)",
              "raw_dir": str(raw.resolve()), "derived_dir": str(derived.resolve()),
              "source_preserved": True, "selected_members": [], "models": {},
              "license": "User supplied licensed models; original terms apply equally to derived models",
              "conversion_reference": "https://github.com/vchoutas/smplx/blob/main/tools/clean_ch.py"}
    try:
        with zipfile.ZipFile(archive_path) as archive:
            infos = archive.infolist()
            models = {}
            for name in ("MANO_LEFT.pkl", "MANO_RIGHT.pkl"):
                matching = [info for info in infos if not info.is_dir() and PurePosixPath(info.filename).name == name]
                if len(matching) != 1:
                    raise ValueError(f"Expected exactly one {name} in archive, found {len(matching)}")
                models[name] = matching[0]
            licenses = [info for info in infos if not info.is_dir()
                        and re.search(r"licen[sc]e", PurePosixPath(info.filename).name, flags=re.IGNORECASE)]
            selected = [(info, raw / name, "model") for name, info in models.items()]
            selected += [(info, raw / "licenses" / info.filename, "license") for info in licenses]
            for info, target, role in selected:
                member = PurePosixPath(info.filename)
                if member.is_absolute() or ".." in member.parts or stat.S_ISLNK(info.external_attr >> 16):
                    raise ValueError(f"Unsafe selected ZIP member: {info.filename}")
                if info.file_size > 256 * 1024 * 1024:
                    raise ValueError(f"Unexpectedly large selected ZIP member: {info.filename}")
                if target.exists():
                    raise FileExistsError(f"Duplicate extraction destination: {target}")
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source, target.open("xb") as destination:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        destination.write(block)
                if target.stat().st_size != info.file_size:
                    raise RuntimeError(f"Incomplete extraction: {info.filename}")
                report["selected_members"].append({"archive_member": info.filename, "role": role,
                                                   "relative_path": str(target.relative_to(output)),
                                                   "bytes": info.file_size, "sha256": sha256(target)})
            report["license_member_count"] = len(licenses)
        save_json(status_path, {"phase": "converting", "utc": now()})
        for side in ("left", "right"):
            filename = "MANO_" + side.upper() + ".pkl"
            report["models"][side] = child_result(["_convert_one", str(raw / filename), str(derived / filename)])
        save_json(status_path, {"phase": "verifying", "utc": now()})
        report["verification"] = child_result(["_verify", str(derived)])
        # Confirm the conversion workers did not modify either original model.
        for member in report["selected_members"]:
            if sha256(output / member["relative_path"]) != member["sha256"]:
                raise RuntimeError("An extracted original changed during conversion")
        report["completed_utc"] = now()
        report["phase"] = "complete"
        save_json(output / "provenance.json", report)
        save_json(status_path, {"phase": "complete", "utc": now(), "derived_dir": str(derived.resolve())})
        return {"passed": True, "derived_dir": str(derived.resolve()),
                "provenance": str((output / "provenance.json").resolve()),
                "source_zip_sha256": actual, "license_member_count": report["license_member_count"]}
    except Exception as exc:
        report.update({"phase": "failed", "failed_utc": now(), "error_type": type(exc).__name__, "error": str(exc)})
        save_json(output / "provenance.failed.json", report)
        save_json(status_path, {"phase": "failed", "utc": now(), "error_type": type(exc).__name__, "error": str(exc)})
        raise


def main() -> None:
    if not Path("/sciclone/home/jchen57").is_dir():
        raise RuntimeError("Prepare/verify MANO assets only in the authorized remote ds Pod")
    if len(sys.argv) > 1 and sys.argv[1] == "_convert_one":
        print(json.dumps(convert_one(Path(sys.argv[2]), Path(sys.argv[3])), sort_keys=True))
        return
    if len(sys.argv) > 1 and sys.argv[1] == "_verify":
        print(json.dumps(verify_derived(Path(sys.argv[2])), sort_keys=True))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", dest="archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new directory; creates separate raw/ and derived/")
    parser.add_argument("--expected-sha256", default=EXPECTED_SHA256)
    args = parser.parse_args()
    expected = args.expected_sha256.lower()
    if re.fullmatch(r"[0-9a-f]{64}", expected) is None:
        parser.error("--expected-sha256 must contain exactly 64 hexadecimal characters")
    print(json.dumps(prepare(args.archive, args.output, expected), sort_keys=True))


if __name__ == "__main__":
    main()
