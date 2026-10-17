#!/usr/bin/env python3
"""Prepare official ARCTIC raw world MANO to TWM99D on the remote Pod.

Official source 49f3eb4..., common/body_models.py build_mano_aa and
src/arctic/processing.py forward_gt_world: use_pca=False,
flat_hand_mean=False, rot3/pose45/trans3/beta10. Hand translation is already
metres (unlike ARCTIC object marker translations, which are millimetres).
The 99D wrist is actual decoded MANO J0, not the transl field. Personalized
subject_vtemplates are used by SMPL-X, not by the source MANO decoder.

Read only raw_seqs.zip, splits_json.zip and metadata identity; never unpack
images or decode held-out test labels. Official P1/P2 train must agree and
be disjoint from both protocols' val/test identities. Extra project/HaWoR
physical-recording exclusions can be supplied before any limits/windows.
The public HaWoR repository does not supply a separate ARCTIC eval list;
this script does not claim to reproduce one.

Source parameters -> smplx source FK -> 99D -> shared smplx FK checks preserve
geometry; this is a serialization/convention audit, not independent MoCap
accuracy measurement. --sample-frames 0 audits every retained frame.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import io
import json
import multiprocessing
from pathlib import Path, PurePosixPath
import re
import sys
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from download_twm_data import now, save_json
from prepare_dexycb_twm import save_sequence_arrays

REVISION = "49f3eb4b7f01663172576357bb3d95695d7a7689"
_RUNTIME = None


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def archive_path(root: Path, filename: str) -> Path:
    candidates = [root / "archives/data" / filename, root / "archives" / filename, root / filename]
    matches = [path for path in candidates if path.is_file()]
    if len(matches) != 1:
        raise ValueError(f"Require one official {filename} under --root or archives/data, found {matches}")
    return matches[0]


def exclusions(path: Path | None) -> set[str]:
    if path is None:
        return set()
    if path.suffix == ".json":
        value = json.loads(path.read_text())
        if isinstance(value, dict):
            value = value.get("arctic", value.get("recordings", []))
    else:
        value = [line.strip() for line in path.read_text().splitlines()
                 if line.strip() and not line.lstrip().startswith("#")]
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError("Exclusions must be a list of original sXX/sequence identities")
    result = set(value)
    if any(not re.fullmatch(r"s\d\d/[A-Za-z0-9_]+", item) for item in result):
        raise ValueError("Exclusions must use complete original sXX/sequence identities")
    return result


def read_splits(path: Path) -> tuple[dict, dict]:
    with zipfile.ZipFile(path) as archive:
        protocols = {name: json.loads(archive.read(f"splits_json/protocol_{name}.json"))
                     for name in ("p1", "p2")}
    for name, value in protocols.items():
        for split in ("train", "val", "test"):
            if not isinstance(value[split], list) or len(value[split]) != len(set(value[split])):
                raise ValueError(f"Invalid/duplicate official {name}/{split} identities")
            if any(not re.fullmatch(r"s\d\d/[A-Za-z0-9_]+", item) for item in value[split]):
                raise ValueError(f"Invalid official recording identity in {name}/{split}")
        for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
            if set(value[first]) & set(value[second]):
                raise ValueError(f"Official {name} recording split overlap")
    if set(protocols["p1"]["train"]) != set(protocols["p2"]["train"]):
        raise ValueError("P1/P2 train identities differ: reassess cross-protocol evaluation isolation")
    held_out = set().union(*(set(value[split]) for value in protocols.values() for split in ("val", "test")))
    if held_out & set(protocols["p1"]["train"]):
        raise ValueError("Training overlaps another protocol's held-out identities")
    expected = {"train": 267, "val": 34, "test": 38}
    if any({key: len(value[key]) for key in expected} != expected for value in protocols.values()):
        raise ValueError("Official ARCTIC release split counts changed; re-audit before preparation")
    return protocols, protocols["p1"]


def decoder(models: str, version: str | None):
    global _RUNTIME
    if _RUNTIME is None:
        import torch
        from handloop.data.meta_mano import MANOReferenceDecoder
        torch.set_num_threads(1)
        _RUNTIME = MANOReferenceDecoder(models, device="cpu", fix_left_shapedirs=False, asset_version=version)
    return _RUNTIME


def shape_rows(value, width: int, n: int, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape == (width,):
        array = np.broadcast_to(array, (n, width)).copy()
    elif array.shape == (1, width):
        array = np.broadcast_to(array, (n, width)).copy()
    if array.shape != (n, width):
        raise ValueError(f"{label}: expected [{n},{width}], got {array.shape}")
    return array


def convert(job: dict) -> dict:
    import torch
    from handloop import geometry as g
    reference = decoder(job["mano_models"], job["asset_version"])
    with zipfile.ZipFile(job["raw_zip"]) as archive:
        source_bytes = archive.read(job["member"])
    # Official trusted source NPY dictionaries contain pickle data, as documented
    # in ARCTIC's construct_loader. Do not pass third-party arbitrary files here.
    value = np.load(io.BytesIO(source_bytes), allow_pickle=True).item()
    if not isinstance(value, dict) or not {"left", "right"}.issubset(value):
        raise ValueError(f"Unexpected official MANO structure: {job['recording']}")
    entries, skipped = [], {}
    side_lengths = {side: len(np.asarray(value[side]["rot"])) for side in ("left", "right")}
    if len(set(side_lengths.values())) != 1:
        raise ValueError("Left/right recording time lengths disagree")
    n = side_lengths["left"]
    for side in ("left", "right"):
        source = value[side]
        root = shape_rows(source["rot"], 3, n, "rot")
        pose = shape_rows(source["pose"], 45, n, "pose")
        trans = shape_rows(source["trans"], 3, n, "trans")
        beta = shape_rows(source["shape"], 10, n, "shape")
        fit = np.asarray(source["fitting_err"], dtype=np.float64).reshape(-1)
        if fit.shape != (n,):
            raise ValueError(f"fitting_err shape differs from source length: {fit.shape}, n={n}")
        finite = np.isfinite(np.concatenate((root, pose, trans, beta), axis=-1)).all(axis=-1)
        valid = finite & np.isfinite(fit)
        ids = np.flatnonzero(valid)
        if job["max_frames"]:
            ids = ids[:job["max_frames"]]
        skipped[side] = {"source_frames": n, "nonfinite_parameters": int((~finite).sum()),
                         "nonfinite_fitting_error": int((~np.isfinite(fit)).sum())}
        if len(ids) < 2:
            continue
        sampled = np.arange(len(ids)) if job["sample_frames"] == 0 else np.linspace(
            0, len(ids)-1, min(len(ids), job["sample_frames"]), dtype=int)
        audit_rows = set(sampled.tolist())
        states = []
        joint_max = vertex_max = joint_sum = vertex_sum = 0.0
        joint_count = vertex_count = 0
        bbox_min, bbox_max = float("inf"), 0.0
        layer = reference.layers[side]
        with torch.no_grad():
            for begin in range(0, len(ids), 128):
                take = ids[begin:begin+128]
                tensors = [torch.from_numpy(array[take].copy()) for array in (root, pose, trans, beta)]
                root_t, pose_t, trans_t, beta_t = tensors
                source_fk = layer(global_orient=root_t, hand_pose=pose_t, betas=beta_t,
                                  transl=trans_t, return_verts=True, return_full_pose=True)
                full = torch.cat((root_t, pose_t + reference.means[side]), -1)
                if not torch.allclose(source_fk.full_pose, full, rtol=1e-5, atol=1e-6):
                    raise ValueError("ARCTIC nonflat hand mean was not added exactly once")
                state = g.state_from_rotmat(source_fk.joints[:, 0], g.axis_angle_to_matrix(full.reshape(-1, 16, 3)))
                if not torch.isfinite(state).all():
                    raise ValueError("Nonfinite 99D despite finite source parameters")
                states.append(state.cpu().numpy())
                chosen = [index for index in range(len(take)) if begin+index in audit_rows]
                if not chosen:
                    continue
                checked = state[chosen]
                restored_pose = g.matrix_to_axis_angle(g.state_to_rotmat(checked)).reshape(-1, 48)
                restored = layer(global_orient=restored_pose[:, :3], betas=beta_t[chosen],
                                 hand_pose=restored_pose[:, 3:] - reference.means[side],
                                 transl=torch.zeros(len(chosen), 3), return_verts=True)
                target_vertices = restored.vertices-restored.joints[:, :1]+checked[:, None, :3]
                target_joints = restored.joints[:, :16]-restored.joints[:, :1]+checked[:, None, :3]
                ve = (source_fk.vertices[chosen]-target_vertices).norm(dim=-1)
                je = (source_fk.joints[chosen, :16]-target_joints).norm(dim=-1)
                if not torch.isfinite(ve).all() or not torch.isfinite(je).all():
                    raise ValueError("Nonfinite source/state FK geometry")
                joint_max, vertex_max = max(joint_max, float(je.max())), max(vertex_max, float(ve.max()))
                if max(joint_max, vertex_max) > job["fk_tolerance_m"]:
                    raise ValueError(f"ARCTIC source/state FK mismatch {job['recording']}/{side}: "
                                     f"joint={joint_max}, vertex={vertex_max} metres")
                joint_sum += float(je.double().sum()); vertex_sum += float(ve.double().sum())
                joint_count += je.numel(); vertex_count += ve.numel()
                diagonal = (source_fk.vertices[chosen].amax(1)-source_fk.vertices[chosen].amin(1)).norm(dim=-1)
                bbox_min, bbox_max = min(bbox_min, float(diagonal.min())), max(bbox_max, float(diagonal.max()))
        if not 0.03 <= bbox_min <= bbox_max <= 0.5:
            raise ValueError("Implausible source hand scale for metre units; inspect assets and labels")
        beta_values = beta[ids]
        span = float(np.ptp(beta_values, axis=0).max())
        arrays = {"state": np.concatenate(states).astype(np.float32), "timestamps": ids.astype(np.float64)/30.0,
                  "beta": beta_values[0] if span == 0 else beta_values,
                  "beta_valid": np.bool_(True), "frame_ids": ids.astype(np.int64),
                  "fitting_err": fit[ids], "gt_mask": np.ones((len(ids), 17), dtype=np.bool_)}
        stem = job["split"] + "/" + job["recording"].replace("/", "++") + "__" + side
        relative = save_sequence_arrays(Path(job["output"]), stem, arrays, job["storage"])
        entries.append({"path": relative, "storage": job["storage"], "split": job["split"], "dataset": "arctic",
                        "source": "ARCTIC Vicon/MoSh++-fitted world MANO", "recording": job["recording"],
                        "subject": job["recording"].split("/")[0], "handedness": int(side == "right"),
                        "coordinate_frame": "world", "units": "metres", "nominal_dt": 1/30,
                        "frames": len(ids), "source_frames": n, "gt_complete": True, "partial_gt": False,
                        "source_member": job["member"], "source_member_sha256": hashlib.sha256(source_bytes).hexdigest(),
                        "beta_span": span, "beta_provenance": "source MANO shape; not personalized SMPL-X vtemplate",
                        "frame_id_minmax": [int(ids[0]), int(ids[-1])],
                        "frame_gaps": int(np.count_nonzero(np.diff(ids) != 1)),
                        "fitting_err_range": [float(fit[ids].min()), float(fit[ids].max())],
                        "fitting_error_note": "original scalar retained; units/absolute calibration not established; no invented QC threshold",
                        "fk_roundtrip": {"checked": True, "all_frames": len(sampled) == len(ids), "frames_checked": len(sampled),
                                         "frame_ids": ids[sampled].tolist(), "joint_max_m": joint_max,
                                         "joint_mean_m": joint_sum/joint_count, "vertex_max_m": vertex_max,
                                         "vertex_mean_m": vertex_sum/vertex_count, "tolerance_m": job["fk_tolerance_m"],
                                         "bbox_diagonal_m": [bbox_min, bbox_max],
                                         "scope": "source rot/pose/trans to serialized99D, both official smplx FK; not independent accuracy"}})
    return {"recording": job["recording"], "sequences": entries, "skipped": skipped}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mano-models", type=Path)
    parser.add_argument("--mano-asset-version", default=None)
    parser.add_argument("--exclude-recordings", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--storage", choices=("npy", "npz"), default="npy")
    parser.add_argument("--max-sequences", type=int, default=0, help="per train/val split; 0 all")
    parser.add_argument("--max-frames", type=int, default=0, help="per hand, smoke only")
    parser.add_argument("--sample-frames", type=int, default=32, help="FK samples per hand; 0 checks every retained frame")
    parser.add_argument("--fk-tolerance-m", type=float, default=1e-5)
    parser.add_argument("--index-only", action="store_true")
    args = parser.parse_args()
    if not Path("/sciclone/home/jchen57").is_dir():
        parser.error("Run preprocessing only inside authorized remote ds Pod")
    if min(args.max_sequences, args.max_frames, args.sample_frames) < 0 or args.workers < 1 or not (
            np.isfinite(args.fk_tolerance_m) and args.fk_tolerance_m > 0):
        parser.error("Invalid limits/workers/tolerance")
    if not args.index_only and args.mano_models is None:
        parser.error("--mano-models required to compute actual wrist J0 and audit source MANO FK")
    archives = {name: archive_path(args.root, name + ".zip") for name in ("raw_seqs", "splits_json", "meta")}
    archive_info = {name: {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
                    for name, path in archives.items()}
    # Verify against the original author's SHA256 already pinned by downloader.
    manifest_path = args.root / "download_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Official download_manifest.json required for archive authenticity checks")
    downloaded = json.loads(manifest_path.read_text())
    for name, info in archive_info.items():
        matches = [entry for entry in downloaded["files"] if PurePosixPath(entry["path"]).name == name + ".zip"]
        if len(matches) != 1 or matches[0].get("sha256") != info["sha256"]:
            raise ValueError(f"{name}.zip differs from pinned official download manifest SHA256")
    protocols, split = read_splits(archives["splits_json"])
    excluded = exclusions(args.exclude_recordings)
    with zipfile.ZipFile(archives["raw_seqs"]) as archive:
        members = {match.group(1): name for name in archive.namelist()
                   if (match := re.fullmatch(r"raw_seqs/(s\d\d/[A-Za-z0-9_]+)\.mano\.npy", name))}
    for name in ("train", "val"):
        missing = set(split[name]) - members.keys()
        if missing:
            raise ValueError(f"Missing official {name} raw MANO recordings: {sorted(missing)}")
    if set(split["test"]) & members.keys():
        raise ValueError("Unexpected held-out test MANO label release: audit before reading")
    jobs = []
    for name in ("train", "val"):
        chosen = sorted(set(split[name]) - excluded)
        if args.max_sequences:
            chosen = chosen[:args.max_sequences]
        jobs.extend({"recording": recording, "member": members[recording], "split": name,
                     "raw_zip": str(archives["raw_seqs"]), "output": str(args.output),
                     "mano_models": str(args.mano_models), "asset_version": args.mano_asset_version,
                     "max_frames": args.max_frames, "sample_frames": args.sample_frames,
                     "storage": args.storage, "fk_tolerance_m": args.fk_tolerance_m} for recording in chosen)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/"manifest.json").exists() or (args.output/"index_manifest.json").exists():
        parser.error("Prepared manifest already exists; use a fresh output directory")
    metadata = {"utc": now(), "phase": "indexed" if args.index_only else "preparing", "dataset": "arctic",
                "source_revision": REVISION, "source": "official ARCTIC MoCap-fitted MANO; not direct rotation measurements",
                "source_urls": [f"https://github.com/zc-alexfan/arctic/blob/{REVISION}/src/arctic/processing.py",
                                f"https://github.com/zc-alexfan/arctic/blob/{REVISION}/common/body_models.py"],
                "storage": args.storage, "rotation6d": "columns_c0xyz_c1xyz", "coordinate_frame": "world",
                "units": "metres", "nominal_fps": 30, "nominal_dt": 1/30, "gt_complete": True, "partial_gt": False,
                "source_archives": archive_info, "official_splits": protocols,
                "split_rule": "official P1 train/val; P1/P2 train equivalence and union-heldout exclusion verified",
                "excluded_recordings": sorted(excluded), "exclude_file_sha256": sha256(args.exclude_recordings) if args.exclude_recordings else None,
                "hawor_arctic_eval_list": "no separate public ARCTIC eval list in audited HaWoR repo; official val/test never in train",
                "timestamp_source": "raw synchronized row index / nominal30Hz; no measured clock timestamps; RGB sync offset not applied to world motion",
                "position": "actual decoded anatomical J0, not source MANO transl; no world-to-camera transform",
                "shape": "source hand-specific beta10 retained; subject_vtemplates apply only to SMPL-X, not MANO",
                "source_conventions": {"use_pca": False, "flat_hand_mean": False, "left_shapedirs_fix": False,
                                       "finger_full_pose": "source pose45 + original hand mean exactly once"},
                "bounded_limits": {"max_sequences_per_split": args.max_sequences, "max_frames_per_hand": args.max_frames,
                                   "fk_sample_frames": args.sample_frames}, "sequences": [], "recordings_summary": []}
    status_path = args.output / "preparation_status.json"
    if args.index_only:
        metadata["indexed_recordings"] = [{key: job[key] for key in ("recording", "member", "split")} for job in jobs]
        save_json(args.output/"index_manifest.json", metadata)
        print(json.dumps({"phase": "indexed", "recordings": dict(Counter(job["split"] for job in jobs))}), flush=True)
        return
    metadata["decoder_provenance"] = decoder(str(args.mano_models), args.mano_asset_version).provenance()
    save_json(status_path, {"phase": "preparing", "done": 0, "total": len(jobs), "utc": now()})
    try:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = [pool.submit(convert, job) for job in jobs]
            for done, future in enumerate(as_completed(futures), 1):
                result = future.result()
                metadata["sequences"].extend(result.pop("sequences"))
                metadata["recordings_summary"].append(result)
                status = {"phase": "preparing", "done": done, "total": len(jobs), "last_recording": result["recording"], "utc": now()}
                save_json(status_path, status)
                print(json.dumps(status), flush=True)
    except Exception as exc:
        save_json(status_path, {"phase": "failed", "error_type": type(exc).__name__, "error": str(exc), "utc": now()})
        raise
    metadata["sequences"].sort(key=lambda entry: entry["path"])
    metadata["recordings_summary"].sort(key=lambda entry: entry["recording"])
    if not metadata["sequences"]:
        raise ValueError("No prepared ARCTIC trajectories")
    metadata["frames_by_split"] = {name: sum(entry["frames"] for entry in metadata["sequences"] if entry["split"] == name)
                                   for name in ("train", "val")}
    metadata.update(phase="complete", completed_utc=now(),
                    full_fk_verified=all(entry["fk_roundtrip"]["all_frames"] for entry in metadata["sequences"]))
    save_json(args.output/"manifest.json", metadata)
    summary = {"phase": "complete", "utc": now(), "frames": metadata["frames_by_split"],
               "sequences": len(metadata["sequences"]), "full_fk_verified": metadata["full_fk_verified"],
               "manifest": str(args.output/"manifest.json")}
    save_json(status_path, summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
