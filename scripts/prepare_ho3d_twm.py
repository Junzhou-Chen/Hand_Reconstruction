#!/usr/bin/env python3
"""Remote CPU HO-3D v2 train-only complete99D trajectory preparation.

Official vis_HO3D.py sets MANO fullpose=handPose, trans=handTrans and
beta=handBeta with flat_hand_mean=True. handTrans is native MANO translation,
NOT anatomical wrist J0. Actual wrist is obtained with licensed MANO FK and
checked against supplied handJoints3D. HO3D's official setup_mano.py uses tip
vertices [744,333,444,555,672] in native MANO16+tips order (not DexYCB's tips).

Keep the source right-handed OpenGL camera gauge. Select ONE fixed camera per
physical recording to avoid counting views as independent motions, and group
all views before constructing engineering validation. Evaluation list is read
as metadata only to check physical-group leakage; evaluation labels never open.

HO3D v2 has no measured timestamps in this adapter. Frame ID / nominal 30 Hz
follows the public HMP lab implementation; capture cadence is NOT independently
verified by that precedent. This convention and its evidence remain explicit.
No RGB, depth, robot data, fitting, interpolation or missing-GT imputation.
"""
from __future__ import annotations

import argparse
import codecs
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib
import io
import json
import multiprocessing
from pathlib import Path
import re
import sys
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from audit_twm_dataset_samples import AnnotationUnpickler, array
from download_twm_data import now, save_json
from prepare_dexycb_twm import save_sequence_arrays
from prepare_taco_twm import get_decoder, digest


SOURCE_REVISION = "2fd6ee17dc1383d0b3922bc10595a67058fca776"
HMP_REVISION = "35d799f76b2b2bc1d1e945117b021014b099e7e6"
HMP_TIME_EVIDENCE = f"https://github.com/enesduran/HMP/blob/{HMP_REVISION}/configs/in_the_lab_sample_config.yaml"
MULTICAM_GROUPS = ("ABF1", "BB1", "GPMF1", "GSF1", "MDF1", "SB1", "ShSu1", "SiBF1", "SMu4", "MPM1", "AP1")
FINGERTIPS = (744, 333, 444, 555, 672)
ADAPTER_VERSION = 1
_ZIP_ARCHIVES = {}


def zip_source(path: str):
    """One read-only ZIP central directory per worker; no RGB decompression."""
    if path not in _ZIP_ARCHIVES:
        _ZIP_ARCHIVES[path] = zipfile.ZipFile(path, "r")
    return _ZIP_ARCHIVES[path]


class HO3DUnpickler(AnnotationUnpickler):
    """Restricted annotation loader with NumPy scalars used in HO3D metadata."""
    def find_class(self, module, name):
        if (module, name) == ("_codecs", "encode"):
            return codecs.encode  # Python2 NumPy pickle byte-string helper.
        if module in ("numpy.core.multiarray", "numpy._core.multiarray") and name == "scalar":
            return getattr(importlib.import_module(module), name)
        return super().find_class(module, name)


def parse_frame_list(path: Path, text: str | None = None) -> dict[str, dict[int, str]]:
    sequences = defaultdict(dict)
    for number, raw in enumerate((path.read_text() if text is None else text).splitlines(), 1):
        raw = raw.strip()
        if not raw:
            continue
        if not re.fullmatch(r"[A-Za-z0-9_-]+/\d+", raw):
            raise ValueError(f"Unexpected official image-list line {number}: {raw!r}")
        sequence, token = raw.split("/")
        frame_id = int(token)
        if frame_id in sequences[sequence]:
            raise ValueError(f"Duplicate official frame ID: {raw}")
        sequences[sequence][frame_id] = token
    if not sequences:
        raise ValueError(f"Empty official list: {path}")
    return dict(sequences)


def physical_group(sequence: str) -> str:
    matches = [group for group in MULTICAM_GROUPS if re.fullmatch(re.escape(group) + r"[0-9]", sequence)]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous known physical recording: {sequence}")
    return matches[0] if matches else sequence


def assign_split(group: str, fraction: float, seed: int) -> str:
    value = hashlib.sha256(f"ho3d-v2-physical-validation:{seed}:{group}".encode()).digest()
    return "val" if int.from_bytes(value[:8], "big") / 2**64 < fraction else "train"


def decode_full_pose(decoder, pose, beta, translation):
    """Faithful full-AA/native-transl path, without treating transl as wrist."""
    import torch
    layer = decoder.layers["right"]
    output = layer(betas=beta, global_orient=pose[:, :3],
        hand_pose=pose[:, 3:] - decoder.means["right"], transl=translation,
        return_verts=True, return_full_pose=True)
    if not torch.allclose(output.full_pose, pose, atol=1e-6, rtol=1e-5):
        raise RuntimeError("MANO mean convention differs from source full48 flat-hand pose")
    native_joints = output.joints[:, :16]
    source21 = torch.cat((native_joints, output.vertices[:, list(FINGERTIPS)]), dim=1)
    return native_joints, source21, output.vertices


def convert_sequence(job: dict) -> dict:
    import torch
    from handloop import geometry as g
    torch.set_num_threads(1)
    root, output = Path(job["root"]), Path(job["output"])
    sequence = job["sequence"]
    decoder = get_decoder(job["mano_models"], job["mano_asset_version"])
    poses, translations, betas, reference_joints, frame_ids, skipped = [], [], [], [], [], Counter()
    source_hash = hashlib.sha256()
    source_bytes, source_annotation_keys = 0, set()
    source_frames = sorted((int(index), token) for index, token in job["frame_tokens"].items())
    chosen = source_frames[:job["max_frames"]] if job["max_frames"] else source_frames
    for frame_id, token in chosen:
        relative = f"train/{sequence}/meta/{token}.pkl"
        path = root / relative
        payload = zip_source(job["archive"]).read(relative) if job["archive"] else path.read_bytes()
        source_bytes += len(payload)
        source_hash.update(relative.encode() + b"\0" + hashlib.sha256(payload).digest())
        annotation = HO3DUnpickler(io.BytesIO(payload), encoding="latin1").load()
        if not isinstance(annotation, dict):
            raise ValueError(f"Expected official annotation dict: {relative}")
        source_annotation_keys.update(str(key) for key in annotation)
        required = ("handPose", "handTrans", "handBeta", "handJoints3D")
        if any(annotation.get(key) is None for key in required):
            skipped["missing_complete_hand_annotation"] += 1
            continue
        pose, translation, beta, joints = [np.asarray(array(annotation[key]), dtype=np.float64) for key in required]
        if pose.size != 48 or translation.size != 3 or beta.size != 10 or joints.shape != (21, 3):
            raise ValueError(f"Unexpected pose/trans/beta/joints shape at {relative}: "
                             f"{pose.shape}/{translation.shape}/{beta.shape}/{joints.shape}")
        if not all(np.isfinite(value).all() for value in (pose, translation, beta, joints)):
            skipped["nonfinite_complete_hand_annotation"] += 1
            continue
        poses.append(pose.reshape(48)); translations.append(translation.reshape(3))
        betas.append(beta.reshape(10)); reference_joints.append(joints); frame_ids.append(frame_id)
    common = {"sequence": sequence, "physical_recording": job["physical_recording"], "split": job["split"],
              "source_annotated_frames": len(source_frames), "frames_consulted": len(chosen),
              "source_meta_hash_policy": "SHA256 ordered(relative_path NUL SHA256(file_bytes)); listed train labels only",
              "source_meta_sha256": source_hash.hexdigest(), "source_meta_bytes": source_bytes,
              "source_annotation_keys": sorted(source_annotation_keys), "skipped": dict(skipped)}
    if len(frame_ids) < job["min_frames"]:
        return {**common, "entry": None, "rejected": "too_few_valid_complete_frames", "frames": len(frame_ids),
                "decoder_provenance": decoder.provenance()}
    poses, translations, betas = np.asarray(poses), np.asarray(translations), np.asarray(betas)
    reference_joints, ids = np.asarray(reference_joints), np.asarray(frame_ids, dtype=np.int64)
    span = float(np.ptp(betas, axis=0).max())
    if span > 1e-5:
        raise ValueError(f"Source per-person beta varies in {sequence}: {span}; preserve/re-audit rather than averaging")
    states, errors = [], []
    wrist_offsets, roundtrip_joints, roundtrip_vertices = [], [], []
    ortho_error, determinant_error = 0.0, 0.0
    sample_ids = set(np.linspace(0, len(ids) - 1, min(len(ids), job["fk_samples"]), dtype=int).tolist())
    with torch.no_grad():
        for begin in range(0, len(ids), 64):
            end = min(begin + 64, len(ids))
            pose = torch.as_tensor(poses[begin:end], dtype=torch.float32)
            beta = torch.as_tensor(betas[begin:end], dtype=torch.float32)
            translation = torch.as_tensor(translations[begin:end], dtype=torch.float32)
            native, source21, vertices = decode_full_pose(decoder, pose, beta, translation)
            supplied = torch.as_tensor(reference_joints[begin:end], dtype=torch.float32)
            error = (source21 - supplied).norm(dim=-1)
            if not all(torch.isfinite(value).all() for value in (native, source21, vertices, error)):
                raise ValueError(f"Nonfinite MANO FK at {sequence}/{ids[begin]}")
            if float(error.max()) > job["fk_tolerance_m"]:
                raise ValueError(f"HO3Dv2 source-joint FK mismatch {sequence}: {float(error.max())}m > "
                                 f"{job['fk_tolerance_m']}m; check source assets, native joint order and tip vertices")
            rotation = g.axis_angle_to_matrix(pose.reshape(-1, 16, 3))
            state = g.state_from_rotmat(native[:, 0], rotation)
            ortho_error = max(ortho_error, float((rotation.transpose(-1, -2) @ rotation - torch.eye(3)).abs().max()))
            determinant_error = max(determinant_error, float((torch.linalg.det(rotation) - 1).abs().max()))
            states.append(state.numpy()); errors.append(error.numpy())
            wrist_offsets.extend((native[:, 0] - translation).norm(dim=-1).tolist())
            chosen_local = [index - begin for index in sorted(sample_ids) if begin <= index < end]
            if chosen_local:
                serialized = state[chosen_local]
                restored_pose = g.matrix_to_axis_angle(g.state_to_rotmat(serialized)).reshape(-1, 48)
                local_beta = beta[chosen_local]
                joints_zero, _, vertices_zero = decode_full_pose(decoder, restored_pose, local_beta,
                                                                  torch.zeros_like(translation[chosen_local]))
                offset = serialized[:, None, :3] - joints_zero[:, :1]
                restored_joints, restored_vertices = joints_zero + offset, vertices_zero + offset
                roundtrip_joints.extend((restored_joints - native[chosen_local]).norm(dim=-1).flatten().tolist())
                roundtrip_vertices.extend((restored_vertices - vertices[chosen_local]).norm(dim=-1).flatten().tolist())
    if max(roundtrip_joints + roundtrip_vertices) > job["roundtrip_tolerance_m"]:
        raise ValueError(f"HO3D actual-wrist state roundtrip exceeds tolerance: {sequence}")
    if max(ortho_error, determinant_error) > 1e-5:
        raise ValueError(f"SO(3) conversion failed: {sequence}")
    states, errors = np.concatenate(states), np.concatenate(errors)
    arrays = {"state": states.astype(np.float32), "timestamps": ids.astype(np.float64) / job["nominal_fps"],
              "frame_ids": ids, "beta": betas[0].astype(np.float32), "beta_valid": np.bool_(True),
              "gt_mask": np.ones((len(ids), 17), dtype=np.bool_),
              "joint_3d": reference_joints.astype(np.float32), "joint_valid": np.ones((len(ids), 21), dtype=np.bool_)}
    relative = save_sequence_arrays(output, f"{job['split']}/{sequence}__right", arrays, job["storage"])
    runs = np.diff(np.concatenate(([0], np.flatnonzero(np.diff(ids) != 1) + 1, [len(ids)])))
    entry = {"path": relative, "storage": job["storage"], "dataset": "ho3d", "source": "HO3D_v2",
             "split": job["split"], "official_split": "train", "recording": job["physical_recording"],
             "camera_sequence": sequence, "handedness": 1, "frames": len(ids), "beta_span": span,
             "coordinate_frame": f"source_OpenGL_fixed_camera_{sequence}", "units": "metres",
             "nominal_dt": 1 / job["nominal_fps"], "timestamp_source": "original frame IDs / explicit nominal FPS convention, not measured timestamps",
             "capture_fps_independently_verified": False, "nominal_fps_evidence": HMP_TIME_EVIDENCE,
             "gt_complete": True, "partial_gt": False, "frame_id_minmax": [int(ids[0]), int(ids[-1])],
             "frame_gaps": int(np.count_nonzero(np.diff(ids) != 1)), "longest_contiguous_frames": int(runs.max()),
             "source_meta_sha256": common["source_meta_sha256"], "source_meta_bytes": source_bytes,
             "source_num_listed_frames": len(source_frames), "skipped": dict(skipped),
             "fk_against_supplied_joints": {"checked": True, "all_retained_frames": True,
                 "joint_order": "unpermuted MANO16 followed by five official HO3D tips",
                 "tip_vertices": list(FINGERTIPS), "joint_max_m": float(errors.max()), "joint_mean_m": float(errors.mean()),
                 "wrist_max_m": float(errors[:, 0].max()), "tolerance_m": job["fk_tolerance_m"],
                 "independent_physical_gt": False},
             "fk_roundtrip": {"checked": True, "frames_checked": len(sample_ids),
                 "sampled_frame_ids": ids[sorted(sample_ids)].tolist(), "joint_max_m": max(roundtrip_joints),
                 "vertex_max_m": max(roundtrip_vertices), "tolerance_m": job["roundtrip_tolerance_m"],
                 "scope": "shared-smplx original native-transl vs exported actual-J0 state; not independent decoder"},
             "native_transl_to_actual_wrist_offset_m": {"min": min(wrist_offsets), "max": max(wrist_offsets)},
             "rotation_orthogonality_error": ortho_error, "rotation_determinant_error": determinant_error,
             "full_fk_verified": True}
    return {**common, "entry": entry, "decoder_provenance": decoder.provenance()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, help="extracted HO3Dv2 root, or directory containing HO3D_v2.zip")
    parser.add_argument("--archive", type=Path, help="read source HO3D_v2.zip directly, no RGB extraction")
    parser.add_argument("--archive-sha256", help="optional SHA from the caller's already-verified download receipt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-code", type=Path, help="official HO3D checkout for source code hashes")
    parser.add_argument("--source-revision", default=SOURCE_REVISION)
    parser.add_argument("--mano-models", type=Path, required=True)
    parser.add_argument("--mano-asset-version", default=None)
    parser.add_argument("--storage", choices=("npy", "npz"), default="npy")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--nominal-fps", type=float, default=30.0,
                        help="explicit nominal convention; source capture FPS not independently verified")
    parser.add_argument("--max-recordings-per-split", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--min-frames", type=int, default=34)
    parser.add_argument("--fk-samples", type=int, default=32)
    parser.add_argument("--fk-tolerance-m", type=float, default=0.001)
    parser.add_argument("--roundtrip-tolerance-m", type=float, default=1e-5)
    parser.add_argument("--index-only", action="store_true")
    parser.add_argument("--exclude-evaluation-physical-overlap", action="store_true",
                        help="explicitly exclude any train physical group also named in evaluation.txt; default fail")
    args = parser.parse_args()
    if not Path("/sciclone/home/jchen57").is_dir():
        parser.error("Prepare data only in the authorized remote ds Pod")
    if (args.workers < 1 or args.min_frames < 1 or args.fk_samples < 1
            or min(args.max_recordings_per_split, args.max_frames) < 0
            or not 0 <= args.validation_fraction < 1
            or not np.isfinite(args.nominal_fps) or args.nominal_fps <= 0
            or not np.isfinite(args.fk_tolerance_m) or args.fk_tolerance_m <= 0
            or not np.isfinite(args.roundtrip_tolerance_m) or args.roundtrip_tolerance_m <= 0
            or not re.fullmatch(r"[0-9a-f]{40}", args.source_revision)):
        parser.error("Invalid limits/FPS/tolerances/source revision")
    if args.root is None and args.archive is None:
        parser.error("Supply --archive or --root")
    if args.root is not None:
        args.root = args.root.expanduser().resolve()
    if args.archive is None and args.root is not None and not (args.root / "train.txt").is_file():
        candidate_archive = args.root / "HO3D_v2.zip"
        if candidate_archive.is_file():
            args.archive = candidate_archive
    if args.archive is not None:
        args.archive = args.archive.expanduser().resolve()
        if not args.archive.is_file():
            parser.error("--archive is not an existing file")
        args.root = args.root or args.archive.parent
    if args.archive_sha256 is not None and not re.fullmatch(r"[0-9a-f]{64}", args.archive_sha256):
        parser.error("--archive-sha256 must be a valid 64-character hexadecimal digest")
    if (args.output / "manifest.json").exists():
        parser.error("Use a fresh output directory; completed trajectories are immutable")
    train_file, evaluation_file = args.root / "train.txt", args.root / "evaluation.txt"
    if args.archive:
        archive = zip_source(str(args.archive))
        train_bytes, evaluation_bytes = archive.read("train.txt"), archive.read("evaluation.txt")
        train = parse_frame_list(train_file, train_bytes.decode("utf-8"))
        evaluation = parse_frame_list(evaluation_file, evaluation_bytes.decode("utf-8"))
        available_archive_members = set(archive.namelist())
        archive_identity = {"path": str(args.archive), "bytes": args.archive.stat().st_size,
                            "mtime_ns": args.archive.stat().st_mtime_ns, "sha256": args.archive_sha256,
                            "hash_provenance": "caller-supplied previously verified receipt; not recomputed by adapter"
                                if args.archive_sha256 else "no archive SHA supplied; selected train meta have independently computed file-content hashes"}
    else:
        train_bytes, evaluation_bytes = train_file.read_bytes(), evaluation_file.read_bytes()
        train, evaluation = parse_frame_list(train_file), parse_frame_list(evaluation_file)
        available_archive_members, archive_identity = None, None
    if set(train) & set(evaluation):
        raise ValueError("Original camera sequence appears in both official train/evaluation")
    grouped = defaultdict(list)
    for sequence in train:
        grouped[physical_group(sequence)].append(sequence)
    evaluation_groups = {physical_group(sequence) for sequence in evaluation}
    overlap = sorted(set(grouped) & evaluation_groups)
    if overlap and not args.exclude_evaluation_physical_overlap:
        raise ValueError(f"Physical recording overlaps official evaluation: {overlap}; inspect and explicitly exclude before pretraining")
    for group in overlap:
        del grouped[group]
    # Recording/camera selection is frozen from official list metadata only;
    # fit quality, predicted results and test labels never affect it.
    selected = {group: sorted(sequences, key=lambda sequence: (-len(train[sequence]), sequence))[0]
                for group, sequences in grouped.items()}
    assignments = {group: assign_split(group, args.validation_fraction, args.seed) for group in selected}
    jobs = []
    for split in ("train", "val"):
        groups = sorted(group for group in selected if assignments[group] == split)
        if args.max_recordings_per_split:
            groups = groups[:args.max_recordings_per_split]
        for group in groups:
            sequence = selected[group]
            if args.archive:
                for token in train[sequence].values():
                    if f"train/{sequence}/meta/{token}.pkl" not in available_archive_members:
                        raise FileNotFoundError(f"Listed train annotation absent from ZIP: {sequence}/{token}")
            elif not (args.root / "train" / sequence / "meta").is_dir():
                raise FileNotFoundError(args.root / "train" / sequence / "meta")
            jobs.append({"root": str(args.root), "output": str(args.output), "sequence": sequence,
                         "archive": str(args.archive) if args.archive else None,
                         "physical_recording": group, "split": split, "frame_tokens": train[sequence],
                         "mano_models": str(args.mano_models), "mano_asset_version": args.mano_asset_version,
                         "storage": args.storage, "max_frames": args.max_frames, "min_frames": args.min_frames,
                         "fk_samples": args.fk_samples, "fk_tolerance_m": args.fk_tolerance_m,
                         "roundtrip_tolerance_m": args.roundtrip_tolerance_m, "nominal_fps": args.nominal_fps})
    if not jobs or not any(job["split"] == "train" for job in jobs):
        raise ValueError("No selected original training recordings")
    args.output.mkdir(parents=True, exist_ok=True)
    source_hashes = {}
    if args.source_code:
        for relative in ("vis_HO3D.py", "setup_mano.py", "vis_pcl_all_cameras.py", "utils/vis_utils.py", "README.md"):
            source_hashes[relative] = digest(args.source_code / relative)
    decoder = None if args.index_only else get_decoder(str(args.mano_models), args.mano_asset_version)
    manifest = {"phase": "indexed" if args.index_only else "preparing", "utc": now(), "dataset": "ho3d",
        "release": "HO3D_v2", "adapter_version": ADAPTER_VERSION, "source_revision": args.source_revision,
        "source_urls": [f"https://github.com/shreyashampali/ho3d/tree/{args.source_revision}",
                        "https://arxiv.org/abs/1907.01481", HMP_TIME_EVIDENCE],
        "source_code_sha256": source_hashes, "rotation6d": "columns_c0xyz_c1xyz",
        "position": "actual MANO FK wrist J0, not native handTrans; metres in original fixed OpenGL camera gauge",
        "source_convention": {"full_pose": "48 full axis-angle radians; flat_hand_mean=True",
            "translation": "native MANO transl, uncentred source decoder", "handedness": "right only",
            "left_shapedirs_fix": False, "tip_vertices": list(FINGERTIPS), "world_transform_applied": False},
        "nominal_fps": args.nominal_fps, "capture_fps_independently_verified": False,
        "timestamp_source": "original numeric meta frame ID / explicit nominal FPS; not measured",
        "nominal_fps_evidence": HMP_TIME_EVIDENCE,
        "nominal_fps_evidence_limit": "HMP's lab config uses 30Hz for HO3Dv3; it does not establish original HO3Dv2 sensor timing",
        "source_quality": "RGB-D multiview/singleview temporal MANO fitting, not independent MoCap",
        "source_archive": archive_identity, "source_read_policy": "ZIP selected train/meta members only; no RGB/depth/evaluation annotations" if args.archive else "extracted train/meta only",
        "train_list_sha256": hashlib.sha256(train_bytes).hexdigest(), "evaluation_list_sha256": hashlib.sha256(evaluation_bytes).hexdigest(),
        "official_train_camera_sequences": len(train), "official_train_listed_images": sum(map(len, train.values())),
        "evaluation_list_metadata_only": True, "evaluation_labels_decoded": False,
        "physical_group_source": f"official vis_pcl_all_cameras.py@{args.source_revision} multiCamSeqs; single-view names preserved",
        "engineering_validation": {"fraction": args.validation_fraction, "seed": args.seed,
            "grouping": "physical recording, all known synchronized cameras grouped", "subject_disjoint": False,
            "not_author_official_validation": True},
        "camera_deduplication": "one fixed camera sequence per original recording, most official listed frames then lexicographic tie break",
        "physical_group_cameras": {group: sorted(value) for group, value in grouped.items()},
        "selected_camera_per_physical_group": selected, "physical_group_split_assignments": assignments,
        "excluded_evaluation_overlapping_groups": overlap,
        "bounded_limits": {"max_recordings_per_split": args.max_recordings_per_split, "max_frames": args.max_frames},
        "decoder_provenance": decoder.provenance() if decoder else None, "sequences": [], "recordings_summary": []}
    if args.index_only:
        manifest["selected_jobs"] = [{key: job[key] for key in ("sequence", "physical_recording", "split")} for job in jobs]
        save_json(args.output / "index_manifest.json", manifest)
        print(json.dumps({"phase": "indexed", "physical_recordings": len(jobs),
                          "split_counts": dict(Counter(job["split"] for job in jobs))}), flush=True)
        return
    status_path = args.output / "preparation_status.json"
    save_json(status_path, {"phase": "preparing", "done": 0, "total": len(jobs), "utc": now()})
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(convert_sequence, job) for job in jobs]
        for done, future in enumerate(as_completed(futures), 1):
            try:
                result = future.result()
            except Exception as exc:
                save_json(status_path, {"phase": "failed", "done": done - 1, "total": len(jobs),
                                        "error_type": type(exc).__name__, "error": str(exc), "utc": now()})
                for pending in futures:
                    pending.cancel()
                raise
            if result.pop("decoder_provenance") != manifest["decoder_provenance"]:
                raise RuntimeError("Decoder/asset provenance changed between workers")
            entry = result.pop("entry")
            if entry is not None:
                manifest["sequences"].append(entry)
            manifest["recordings_summary"].append(result)
            status = {"phase": "preparing", "done": done, "total": len(jobs),
                      "last_sequence": result["sequence"], "utc": now()}
            save_json(status_path, status)
            print(json.dumps(status), flush=True)
    if not manifest["sequences"]:
        raise ValueError("No complete validated HO3D trajectories")
    manifest["sequences"].sort(key=lambda entry: entry["path"])
    manifest["recordings_summary"].sort(key=lambda entry: entry["sequence"])
    manifest.update(phase="complete", completed_utc=now(), full_fk_verified=True,
        sampled_state_roundtrip_verified=True, gt_complete=True, partial_gt=False)
    manifest["frames_by_split"] = {split: sum(entry["frames"] for entry in manifest["sequences"] if entry["split"] == split)
                                   for split in ("train", "val")}
    manifest["sequences_by_split"] = dict(Counter(entry["split"] for entry in manifest["sequences"]))
    if args.archive and (args.archive.stat().st_size != archive_identity["bytes"]
                         or args.archive.stat().st_mtime_ns != archive_identity["mtime_ns"]):
        raise ValueError("Source ZIP changed during preparation")
    save_json(args.output / "manifest.json", manifest)
    summary = {"phase": "complete", "physical_recordings": len(jobs), "sequences": len(manifest["sequences"]),
               "frames": manifest["frames_by_split"], "manifest": str(args.output / "manifest.json"), "utc": now()}
    save_json(status_path, summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
