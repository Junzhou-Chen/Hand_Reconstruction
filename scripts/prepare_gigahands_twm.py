#!/usr/bin/env python3
"""Remote-only GigaHands original EasyMoCap MANO parameters to TWM99D.

Use original ``hand_poses/<session>/params/<clip>.json`` plus the original
``keypoints_3d/<clip>/chosen_frames_{left,right}.json``. Never use normalized
``keypoints_3d_mano`` as world trajectories. A pose row is mapped to its source
frame through the sorted union of chosen-frame IDs, with a mandatory exact
length check. Only a hand's own chosen frames are retained; missing hands are
not filled from the other side or temporally interpolated.

The official text-motion loader randomly splits annotation ROWS with seed42.
This adapter instead defaults to a documented SUBJECT-disjoint engineering
90/10 train/val split. A supplied split JSON can replace
that engineering split; all assignments are made before any smoke limits.

Reference decoder: GigaHands 8ad3ef9..., EasyMoCap submodule b61bf714....
MANO poses are 48 axis-angle offsets, nonflat mean added once. EasyMoCap
applies external Rh and Th after MANO LBS. Root = R(Rh) @ R(poses[:3]); wrist
is R(Rh) @ shaped_J0 + Th, rather than Th itself. Original beta10 is retained
per frame when it varies. No left-shapedirs correction is made by the source.

The pinned source EasyMoCap LBS is required for actual source-vertex versus
exported-state FK checks. --sample-frames 0 checks every retained frame;
positive values check evenly spaced frames and are NOT full-FK certification.
All numerical execution belongs in the authorized remote Pod.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.util
import json
import math
import multiprocessing
from pathlib import Path
import re
import shutil
import sys
import tempfile
import types

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from download_twm_data import now, save_json

GIGA_REVISION = "8ad3ef9a50157f44a7eef5645b0766362110a961"
EASY_REVISION = "b61bf7148d4411bb81ddcf6350a3ef1427213649"
SOURCE_HASHES = {
    "body_model.py": "4bf74b71d544516a2b58d9f14e5b1e32ab25788993087885a781d4725136f5dc",
    "lbs.py": "f819ebdac888ad364b2844d222a92668c767c6eadd728b264cec102afe74651a",
}
_RUNTIME = None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def resolve_root(root: Path) -> Path:
    candidates = [root, root / "hand_poses"]
    found = [candidate for candidate in candidates
             if candidate.is_dir() and any(candidate.glob("p*/params/*.json"))]
    if len(found) != 1:
        raise ValueError("--root must contain p*/params/*.json or hand_poses/p*/params/*.json")
    return found[0]


def annotation_index(path: Path) -> dict:
    """Keep each physical clip once, merging overlapping annotation intervals."""
    indexed = {}
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if str(value.get("clarify_annotation", "")).strip() in ("None", "Buggy"):
                continue
            sequence = value["sequence"]
            if isinstance(sequence, list):
                if not sequence:
                    raise ValueError(f"Empty annotation sequence at line {number}")
                # Match the pinned official loader: a sequence list uses its
                # first entry; do not invent a concatenation of source clips.
                sequence = sequence[0]
            session = str(value["scene"])
            clip = str(sequence)
            if not re.fullmatch(r"p\d+-[A-Za-z0-9_.-]+", session) or not clip.isdigit():
                raise ValueError(f"Unsafe/unrecognized recording identity at line {number}")
            begin, end = int(value["start_frame_id"]), int(value["end_frame_id"])
            if begin < 0 or end < -1 or (end != -1 and end < begin):
                raise ValueError(f"Invalid annotation row bounds at line {number}")
            key = session + "/" + str(int(clip)).zfill(3)
            indexed.setdefault(key, []).append([begin, end])
    if not indexed:
        raise ValueError("No usable original annotation identities")
    return indexed


def make_splits(recordings: list[dict], split_file: Path | None, seed: int) -> tuple[dict, dict]:
    """Split original identities before loading labels or limiting smoke jobs."""
    all_ids = {entry["recording"] for entry in recordings}
    if split_file is not None:
        value = read_json(split_file)
        mapping = {}
        for split in ("train", "val", "test"):
            for recording in value.get(split, []):
                if recording in mapping:
                    raise ValueError(f"Recording appears in multiple provided splits: {recording}")
                mapping[str(recording)] = split
        missing = all_ids - mapping.keys()
        if missing:
            raise ValueError(f"Provided split omits {len(missing)} original recording identities")
        # Enforce session-disjoint even if a caller's supplied split has row-level leakage.
        session_split = {}
        for entry in recordings:
            assigned = mapping[entry["recording"]]
            previous = session_split.setdefault(entry["session"], assigned)
            if previous != assigned:
                raise ValueError(f"One physical session crosses supplied splits: {entry['session']}")
        return mapping, {"kind": "provided_recording_lists_session_disjoint",
                         "path": str(split_file), "sha256": sha256(split_file)}
    subjects = sorted({entry["subject"] for entry in recordings},
                      key=lambda item: hashlib.sha256(f"{seed}:{item}".encode()).hexdigest())
    if len(subjects) < 2:
        raise ValueError("Subject split requires >=2 subjects; supply full annotation identities or an explicit split file")
    n_train = min(len(subjects)-1, max(1, math.floor(len(subjects) * 0.9)))
    by_subject = {subject: "train" if i < n_train else "val"
                  for i, subject in enumerate(subjects)}
    mapping = {entry["recording"]: by_subject[entry["subject"]] for entry in recordings}
    return mapping, {"kind": "engineering_subject_disjoint_ranked_sha256",
                     "seed": seed, "requested_fractions": {"train": 0.9, "val": 0.1},
                     "subject_assignments": by_subject,
                     "official_equivalence": False,
                     "official_loader_note": "official random_state=42 split operates on annotation rows; not reused"}


def source_runtime(mano_models: str, easy_root: str, asset_version: str | None):
    global _RUNTIME
    if _RUNTIME is not None:
        return _RUNTIME
    import torch
    from handloop.data.meta_mano import MANOReferenceDecoder
    torch.set_num_threads(1)
    decoder = MANOReferenceDecoder(mano_models, device="cpu", fix_left_shapedirs=False,
                                   asset_version=asset_version)
    directory = Path(easy_root) / "easymocap/smplmodel"
    for name, expected in SOURCE_HASHES.items():
        path = directory / name
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Pinned EasyMoCap file absent/changed: {path}; need {EASY_REVISION}")
    # Load only two pinned files in an isolated package; avoid renderer/cv2 imports.
    package_name = "_handloop_gigahands_easymocap_reference"
    package = types.ModuleType(package_name)
    package.__path__ = [str(directory)]
    sys.modules[package_name] = package
    for name in ("lbs", "body_model"):
        spec = importlib.util.spec_from_file_location(package_name + "." + name, directory / (name + ".py"))
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load pinned source module {name}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    layer_class = sys.modules[package_name + ".body_model"].SMPLlayer
    source_layers = {}
    for side in ("left", "right"):
        model = Path(mano_models) / f"MANO_{side.upper()}.pkl"
        layer = layer_class(str(model), model_type="mano", device="cpu", regressor_path=None,
                            use_joints=False, use_pose_blending=True, use_shape_blending=True,
                            use_pca=False, num_pca_comps=6, use_flat_mean=False)
        layer.eval().requires_grad_(False)
        if not torch.equal(layer.shapedirs, decoder.layers[side].shapedirs):
            raise ValueError("Source/shared assets or shapedirs differ")
        if not torch.allclose(layer.mHandsMean.reshape(45), decoder.means[side], atol=0, rtol=0):
            raise ValueError("Source/shared hand mean differ")
        source_layers[side] = layer
    _RUNTIME = decoder, source_layers
    return _RUNTIME


def rows(value, width: int, n: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 0 or array.shape[-1] != width:
        raise ValueError(f"Expected parameter rows ending in {width}, got {array.shape}")
    array = array.reshape(-1, width)
    if n is not None:
        if len(array) == 1:
            array = np.broadcast_to(array, (n, width)).copy()
        elif len(array) != n:
            raise ValueError(f"Parameter rows {len(array)} do not match poses rows {n}")
    return array


def chosen_ids(path: Path) -> np.ndarray:
    values = read_json(path)
    if not isinstance(values, list) or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in values):
        raise ValueError(f"Chosen frames must be nonnegative integer list: {path}")
    if len(set(values)) != len(values):
        raise ValueError(f"Duplicate chosen frame IDs: {path}")
    if values != sorted(values):
        raise ValueError(f"Chosen-frame source list is not chronological: {path}")
    return np.asarray(values, dtype=np.int64)


def triangulated_wrist_audit(path: Path, frame_ids: np.ndarray, wrists: np.ndarray) -> dict:
    """Read original dense per-video-frame 21x4 JSONL; no frame compaction.

    The official release's JSONL has rows for missing hands too, with zero
    confidence. Chosen-frame lists are subsets of these dense source IDs.
    Compare raw world wrist point 0 only; the MANO fit and triangulation are
    different estimates, so residuals are diagnostics, not an accuracy claim.
    """
    points, confidence = [], []
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for row, line in enumerate(stream):
            digest.update(line)
            if not line.strip():
                raise ValueError(f"Blank line breaks dense original-frame indexing: {path}:{row}")
            value = np.asarray(json.loads(line), dtype=np.float64)
            if value.shape != (21, 4):
                raise ValueError(f"Expected original JSONL row [21,4], got {value.shape}: {path}:{row}")
            points.append(value[0, :3])
            confidence.append(value[0, 3])
    points = np.asarray(points, dtype=np.float64)
    confidence = np.asarray(confidence, dtype=np.float64)
    if points.shape != (len(confidence), 3) or len(points) <= int(frame_ids[-1]):
        raise ValueError(f"Dense JSONL rows do not cover source frame IDs: {path}")
    available = np.isfinite(points).all(axis=-1) & np.isfinite(confidence) & (confidence > 0)
    diagnostics = {}
    for offset in (-2, -1, 0, 1, 2):
        candidate = frame_ids + offset
        in_range = (candidate >= 0) & (candidate < len(points))
        keep = np.flatnonzero(in_range)
        keep = keep[available[candidate[keep]]]
        if not len(keep):
            diagnostics[str(offset)] = {"frames": 0}
            continue
        distances = np.linalg.norm(wrists[keep].astype(np.float64)-points[candidate[keep]], axis=-1)
        diagnostics[str(offset)] = {"frames": len(keep), "mean_m": float(distances.mean()),
                                    "median_m": float(np.median(distances)), "max_m": float(distances.max()),
                                    "p95_m": float(np.percentile(distances, 95))}
    zero = diagnostics["0"]
    return {"checked": bool(zero["frames"]), "source_path": str(path), "source_sha256": digest.hexdigest(),
            "row_mapping": "dense original frame ID is zero-based JSONL line index; never index by chosen-frame row",
            "source_jsonl_frames": len(points), "retained_frames": len(frame_ids),
            "zero_or_invalid_confidence_in_source": int((~available).sum()),
            "retained_without_available_raw_wrist": len(frame_ids)-zero["frames"],
            "offset_diagnostics": diagnostics,
            "scope": "original triangulated wrist versus fitted MANO J0, no PA/centering/normalization; diagnostic only",
            "limitation": "neighbor residuals cannot prove exact clock synchronization or independent physical accuracy; no threshold invented"}


def save_arrays(output: Path, stem: str, arrays: dict, storage: str) -> str:
    destination = output / (stem + (".npz" if storage == "npz" else ""))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Prepared sequence already exists: {destination}")
    if storage == "npy":
        temporary = Path(tempfile.mkdtemp(prefix=destination.name + ".tmp-", dir=destination.parent))
        try:
            for key, value in arrays.items():
                np.save(temporary / (key + ".npy"), value, allow_pickle=False)
            temporary.replace(destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    else:
        temporary = destination.with_suffix(".npz.tmp")
        try:
            with temporary.open("wb") as stream:
                np.savez(stream, **arrays)
            temporary.replace(destination)
        finally:
            if temporary.exists():
                temporary.unlink()
    return str(destination.relative_to(output))


def convert(job: dict) -> dict:
    import torch
    from handloop import geometry as g
    decoder, source_layers = source_runtime(job["mano_models"], job["source_easymocap"], job["asset_version"])
    root, output = Path(job["root"]), Path(job["output"])
    session, clip = job["session"], job["clip"]
    source_path = root / session / "params" / (clip + ".json")
    camera_path = root / session / "optim_params.txt"
    if not camera_path.is_file():
        raise FileNotFoundError(f"Source world-camera provenance missing: {camera_path}")
    value = read_json(source_path)
    selected = {side: chosen_ids(root / session / "keypoints_3d" / clip / f"chosen_frames_{side}.json")
                for side in ("left", "right")}
    union = np.union1d(selected["left"], selected["right"])
    # The released renderer maps params rows through this unsorted set-union
    # expression. Accept its row map only when it equals chronological order;
    # neither reorder unknown pose rows nor create an invented time axis.
    official_order = np.asarray(list(set(selected["left"].tolist()) | set(selected["right"].tolist())), dtype=np.int64)
    if not np.array_equal(official_order, union):
        return {"recording": job["recording"], "sequences": [],
                "rejected": "official_set_union_order_not_chronological",
                "source_frame_count": len(union),
                "official_order_first20": official_order[:20].tolist(),
                "sorted_order_first20": union[:20].tolist()}
    if len(union) == 0:
        return {"recording": job["recording"], "sequences": [], "rejected": "no_chosen_frames"}
    entries, skipped = [], {}
    for side in ("left", "right"):
        params = value[side]
        pose = rows(params["poses"], 48)
        n = len(pose)
        if n != len(union):
            raise ValueError(f"Cannot prove pose-row/source-frame mapping {job['recording']}/{side}: "
                             f"{n} pose rows vs {len(union)} sorted chosen-frame union")
        rh, th, beta = (rows(params[key], width, n) for key, width in (("Rh", 3), ("Th", 3), ("shapes", 10)))
        in_annotation = np.zeros(n, dtype=np.bool_)
        for begin, end in job["annotation_ranges"]:
            if begin >= n or end >= n:
                raise ValueError(f"Annotation range exceeds source rows: {job['recording']} {begin}:{end}, n={n}")
            in_annotation[begin:n if end == -1 else end + 1] = True
        finite = np.isfinite(np.concatenate((pose, rh, th, beta), axis=-1)).all(axis=-1)
        own = np.isin(union, selected[side])
        indices = np.flatnonzero(in_annotation & finite & own)
        skipped[side] = {"outside_annotation": int((~in_annotation).sum()), "nonfinite": int((~finite).sum()),
                         "hand_not_chosen": int((~own).sum()), "source_pose_rows": n}
        if job["max_frames"]:
            indices = indices[:job["max_frames"]]
        if len(indices) < 2:
            continue
        pose, rh, th, beta = (torch.from_numpy(a[indices].copy()) for a in (pose, rh, th, beta))
        with torch.no_grad():
            layer = decoder.layers[side]
            full_pose = pose.clone()
            full_pose[:, 3:] += decoder.means[side]
            rotations = g.axis_angle_to_matrix(full_pose.reshape(-1, 16, 3))
            outer = g.axis_angle_to_matrix(rh)
            rotations[:, 0] = outer @ rotations[:, 0]
            shaped = layer.v_template[None] + torch.einsum("bi,vci->bvc", beta, layer.shapedirs)
            j0 = torch.einsum("v,bvc->bc", layer.J_regressor[0], shaped)
            wrist = torch.einsum("bij,bj->bi", outer, j0) + th
            state = g.state_from_rotmat(wrist, rotations)
            check = np.arange(len(indices)) if job["sample_frames"] == 0 else np.linspace(
                0, len(indices)-1, min(len(indices), job["sample_frames"]), dtype=int)
            joint_max = vertex_max = joint_sum = vertex_sum = 0.0
            joint_count = vertex_count = 0
            diameters = []
            for start in range(0, len(check), 64):
                take = check[start:start+64]
                source = source_layers[side](poses=pose[take], shapes=beta[take], Rh=rh[take], Th=th[take],
                                              return_verts=True, return_tensor=True)
                saved = state[take]
                saved_full_pose = g.matrix_to_axis_angle(g.state_to_rotmat(saved)).reshape(-1, 48)
                reconstructed = layer(betas=beta[take], global_orient=saved_full_pose[:, :3],
                                      hand_pose=saved_full_pose[:, 3:] - decoder.means[side],
                                      transl=torch.zeros(len(take), 3), return_verts=True, return_full_pose=True)
                target_vertices = reconstructed.vertices - reconstructed.joints[:, :1] + saved[:, None, :3]
                target_joints = reconstructed.joints[:, :16] - reconstructed.joints[:, :1] + saved[:, None, :3]
                # Kinematic J0 is not the regressed posed-mesh vertex wrist.
                source_full = source_layers[side].extend_pose(pose[take])
                _, source_joints = source_layers[side].lbs(
                    beta[take], source_full, source_layers[side].v_template,
                    source_layers[side].shapedirs, source_layers[side].posedirs,
                    source_layers[side].J_regressor, source_layers[side].parents,
                    source_layers[side].weights, pose2rot=True, dtype=torch.float32)
                source_joints = source_joints @ outer[take].transpose(1, 2) + th[take, None]
                ve = (source - target_vertices).norm(dim=-1)
                je = (source_joints[:, :16] - target_joints).norm(dim=-1)
                if not torch.isfinite(ve).all() or not torch.isfinite(je).all():
                    raise ValueError("Nonfinite source/exported FK geometry")
                if max(float(ve.max()), float(je.max())) > job["fk_tolerance_m"]:
                    raise ValueError(f"GigaHands source-FK mismatch {job['recording']}/{side}: "
                                     f"vertices={float(ve.max())} joints={float(je.max())} metres")
                vertex_max = max(vertex_max, float(ve.max()))
                joint_max = max(joint_max, float(je.max()))
                vertex_sum += float(ve.double().sum())
                joint_sum += float(je.double().sum())
                vertex_count += ve.numel()
                joint_count += je.numel()
                diameters.extend((source.amax(1)-source.amin(1)).norm(dim=-1).tolist())
            if not np.isfinite(diameters).all() or min(diameters) < 0.03 or max(diameters) > 0.5:
                raise ValueError("Source MANO scale is implausible for metre units; review source/assets")
        frames = union[indices]
        raw_wrist_audit = triangulated_wrist_audit(
            root / session / "keypoints_3d" / clip / (side + ".jsonl"), frames,
            state[:, :3].cpu().numpy())
        beta_array = beta.numpy()
        beta_span = float(np.ptp(beta_array, axis=0).max())
        arrays = {"state": state.numpy().astype(np.float32), "timestamps": frames.astype(np.float64)/job["fps"],
                  "beta": beta_array[0] if beta_span == 0 else beta_array,
                  "beta_valid": np.bool_(True), "gt_mask": np.ones((len(frames), 17), dtype=np.bool_),
                  "frame_ids": frames, "source_row_indices": indices.astype(np.int64)}
        stem = f"{job['split']}/{session}++{clip}__{side}"
        relative = save_arrays(output, stem, arrays, job["storage"])
        entries.append({"path": relative, "storage": job["storage"], "split": job["split"],
                        "dataset": "gigahands", "source": "GigaHands original multiview-fitted EasyMoCap MANO",
                        "recording": job["recording"], "session": session, "subject": job["subject"],
                        "handedness": int(side == "right"), "frames": len(frames), "gt_complete": True,
                        "partial_gt": False, "coordinate_frame": "source_multiview_calibrated_world_" + session,
                        "units": "metres", "nominal_dt": 1/job["fps"],
                        "frame_id_minmax": [int(frames[0]), int(frames[-1])],
                        "frame_gaps": int(np.count_nonzero(np.diff(frames) != 1)),
                        "source_path": str(source_path), "source_sha256": sha256(source_path),
                        "camera_parameters_path": str(camera_path), "camera_parameters_sha256": sha256(camera_path),
                        "chosen_frames_sha256": {name: sha256(root/session/"keypoints_3d"/clip/f"chosen_frames_{name}.json")
                                                  for name in ("left", "right")},
                        "source_frame_mapping": "official set-union expression equals sorted union and exact pose-row count; original side frame IDs retained",
                        "official_frame_order_checked": True,
                        "triangulated_wrist_audit": raw_wrist_audit,
                        "beta_span": beta_span, "beta_provenance": "source shapes, static only if exactly equal; per-frame otherwise",
                        "fk_roundtrip": {"checked": True, "all_frames": len(check) == len(indices),
                                         "frames_checked": len(check), "frame_ids": frames[check].tolist(),
                                         "vertex_max_m": vertex_max, "vertex_mean_m": vertex_sum / vertex_count,
                                         "joint_max_m": joint_max, "joint_mean_m": joint_sum / joint_count,
                                         "hand_bbox_diagonal_m": [min(diameters), max(diameters)],
                                         "tolerance_m": job["fk_tolerance_m"],
                                         "scope": "independent pinned source EasyMoCap LBS versus shared-smplx decoded exported99D; not physical GT accuracy"}})
    return {"recording": job["recording"], "sequences": entries, "skipped": skipped}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mano-models", type=Path)
    parser.add_argument("--mano-asset-version", default=None)
    parser.add_argument("--source-easymocap", type=Path)
    parser.add_argument("--split-json", type=Path, help="train/val/test original session/clip lists; session-disjoint required")
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--fps", type=float, default=30.0, help="nominal original video frames per second, not pose-row rate")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--storage", choices=("npy", "npz"), default="npy")
    parser.add_argument("--max-sequences", type=int, default=0, help="per exported split, after complete source split assignment")
    parser.add_argument("--max-frames", type=int, default=0, help="per hand sequence, smoke only")
    parser.add_argument("--sample-frames", type=int, default=32, help="FK audit per hand sequence; 0 checks all retained frames")
    parser.add_argument("--fk-tolerance-m", type=float, default=1e-5)
    parser.add_argument("--index-only", action="store_true")
    args = parser.parse_args()
    if not Path("/sciclone/home/jchen57").is_dir():
        parser.error("Run numerical preparation only inside the authorized remote ds Pod")
    if (min(args.max_sequences, args.max_frames, args.sample_frames) < 0 or args.workers < 1 or
            not np.isfinite(args.fps) or args.fps <= 0 or not np.isfinite(args.fk_tolerance_m) or args.fk_tolerance_m <= 0):
        parser.error("Invalid positive rate/worker/tolerance or nonnegative limits")
    if not args.index_only and (args.mano_models is None or args.source_easymocap is None):
        parser.error("--mano-models and --source-easymocap are required for source-FK checked conversion")
    root = resolve_root(args.root)
    annotations = annotation_index(args.annotations)
    recordings = []
    missing = []
    for recording, intervals in sorted(annotations.items()):
        session, clip = recording.split("/")
        path = root / session / "params" / (clip + ".json")
        if not path.is_file():
            missing.append(recording)
            continue
        recordings.append({"recording": recording, "session": session, "clip": clip,
                           "subject": session.split("-", 1)[0], "annotation_ranges": intervals})
    if not recordings:
        raise ValueError("No matching annotated original params")
    # Full official annotation identities stabilize subject assignment even for
    # an explicitly bounded smoke tree that contains only a few downloaded files.
    all_identities = [{"recording": identity, "session": identity.split("/")[0],
                       "subject": identity.split("/", 1)[0].split("-", 1)[0]}
                      for identity in sorted(annotations)]
    assignments, split_meta = make_splits(all_identities, args.split_json, args.seed)
    if missing and not (args.max_sequences or args.index_only):
        raise ValueError(f"Full conversion requires all annotated params; {len(missing)} absent. "
                         "A positive --max-sequences explicitly permits an incomplete smoke tree.")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "manifest.json").exists() or (args.output / "index_manifest.json").exists():
        parser.error("Output already has a published manifest; choose a fresh directory")
    all_split_counts = Counter(assignments[entry["recording"]] for entry in recordings)
    save_json(args.output / "source_split_assignment.json", {
        "utc": now(), "split_provenance": split_meta, "recording_split": assignments,
        "source_recording_counts": dict(all_split_counts), "missing_annotated_params": missing,
        "annotation_sha256": sha256(args.annotations), "split_before_limits": True})
    selected = []
    for split in ("train", "val"):
        candidates = [entry for entry in recordings if assignments[entry["recording"]] == split]
        if args.max_sequences:
            candidates = candidates[:args.max_sequences]
        selected.extend({**entry, "split": split} for entry in candidates)
    metadata = {"utc": now(), "phase": "indexed" if args.index_only else "preparing", "dataset": "gigahands",
                "source_revision": GIGA_REVISION, "source_easymocap_revision": EASY_REVISION,
                "source_decoder_sha256": SOURCE_HASHES, "source": "GigaHands multiview-fitted MANO, not independent sensor GT",
                "source_urls": [f"https://github.com/brown-ivl/GigaHands/tree/{GIGA_REVISION}",
                                f"https://github.com/zju3dv/EasyMocap/tree/{EASY_REVISION}"],
                "storage": args.storage, "rotation6d": "columns_c0xyz_c1xyz", "nominal_fps": args.fps,
                "timestamp_source": "official set-union equals chronological chosen-frame union, row count verified, frame_id/nominal_fps",
                "unsafe_frame_order_policy": "quarantine recording with explicit reason if source renderer set-union order differs from chronological order",
                "triangulated_wrist_reference": "original dense keypoints_3d/clip/{side}.jsonl, 21x4 xyz+confidence rows indexed by source frame ID",
                "source_conventions": {"use_pca": False, "use_flat_mean": False,
                                       "root": "R(Rh) @ R(poses[:3])", "wrist": "R(Rh) @ shaped_J0 + Th",
                                       "fingers": "poses[3:] + hand-specific mean exactly once",
                                       "left_shapedirs_fix": False, "coordinate_transform": "none; original source world preserved"},
                "scope": "train/val only, one original motion per session/clip/hand; no views/captions duplication",
                "annotation_sequence_list_rule": "first entry, matching pinned official dataset_hands.py; no clip concatenation",
                "split_provenance": split_meta, "source_recording_counts": dict(all_split_counts),
                "missing_annotated_params": missing, "source_params_complete": not missing,
                "annotation_sha256": sha256(args.annotations),
                "bounded_limits": {"max_sequences_per_split": args.max_sequences, "max_frames_per_hand": args.max_frames,
                                   "fk_sample_frames": args.sample_frames}, "sequences": [], "recordings_summary": []}
    status_path = args.output / "preparation_status.json"
    if args.index_only:
        metadata["indexed_recordings"] = selected
        save_json(args.output / "index_manifest.json", metadata)
        print(json.dumps({"phase": "indexed", "recordings": dict(all_split_counts), "missing": len(missing)}), flush=True)
        return
    decoder, _ = source_runtime(str(args.mano_models), str(args.source_easymocap), args.mano_asset_version)
    metadata["decoder_provenance"] = decoder.provenance()
    jobs = [{**entry, "root": str(root), "output": str(args.output), "mano_models": str(args.mano_models),
             "source_easymocap": str(args.source_easymocap), "asset_version": args.mano_asset_version,
             "storage": args.storage, "sample_frames": args.sample_frames, "max_frames": args.max_frames,
             "fps": args.fps, "fk_tolerance_m": args.fk_tolerance_m} for entry in selected]
    save_json(status_path, {"phase": "preparing", "done": 0, "total": len(jobs), "utc": now()})
    try:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = [pool.submit(convert, job) for job in jobs]
            for done, future in enumerate(as_completed(futures), 1):
                result = future.result()
                metadata["sequences"].extend(result.pop("sequences"))
                metadata["recordings_summary"].append(result)
                status = {"phase": "preparing", "done": done, "total": len(jobs),
                          "last_recording": result["recording"], "utc": now()}
                save_json(status_path, status)
                print(json.dumps(status), flush=True)
    except Exception as exc:
        save_json(status_path, {"phase": "failed", "error_type": type(exc).__name__, "error": str(exc), "utc": now()})
        raise
    metadata["sequences"].sort(key=lambda entry: entry["path"])
    metadata["recordings_summary"].sort(key=lambda entry: entry["recording"])
    if not metadata["sequences"]:
        raise ValueError("No usable converted hand trajectories")
    metadata["frames_by_split"] = {split: sum(entry["frames"] for entry in metadata["sequences"] if entry["split"] == split)
                                   for split in ("train", "val")}
    metadata["full_fk_verified"] = all(entry["fk_roundtrip"]["all_frames"] for entry in metadata["sequences"])
    metadata.update(phase="complete", completed_utc=now())
    save_json(args.output / "manifest.json", metadata)
    summary = {"phase": "complete", "utc": now(), "frames": metadata["frames_by_split"],
               "sequences": len(metadata["sequences"]), "full_fk_verified": metadata["full_fk_verified"],
               "manifest": str(args.output / "manifest.json")}
    save_json(status_path, summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
