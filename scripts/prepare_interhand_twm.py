#!/usr/bin/env python3
"""Remote-only InterHand2.6M 30fps world-MANO -> anatomical-wrist 99D.

Only official train/val MANO_NeuralAnnot.json and joint_3d.json are read. No
images, camera duplicates, test labels, fitting, smoothing or interpolation
are used. ``pose`` is an axis-angle OFFSET from the nonflat MANO hand mean,
not PCA; the official renderer conditionally fixes the left shapedirs bug.

Raw frame indices are NOT universally 30Hz. The paper specifies 90Hz and
30Hz acquisition, while the release is nominally 30fps; train Capture10--26
are explicitly sparse. Default cadence inference is limited to the official
non-sparse captures, requires a strongly dominant index step of 1 or 3, and
records its inference. Sparse/ambiguous captures are quarantined unless an
explicit, evidence-bearing --timebase-json is provided. --index-only emits
cadence evidence without MANO decoding. Real capture timestamps are absent.

Timebase override format (keys are split/CaptureN, or split/CaptureN/seq):
{"train/Capture0": {"frame_index_hz": 90, "release_frame_step": 3,
                     "evidence": "URL / audited capture-specific basis"}}
The evidence string is provenance, not automatic verification of the claim.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


SOURCE_URL = "https://mks0601.github.io/InterHand2.6M/"
RENDERER_URL = "https://github.com/facebookresearch/InterHand2.6M/blob/main/tool/MANO_render/render.py"
PAPER_URL = "https://arxiv.org/html/2008.09309#S3.SS1"
# InterHand each hand: thumb tip->base, index tip->base, middle, ring,
# pinky, wrist. MANO16 order: wrist,index,middle,pinky,ring,thumb.
MANO16_TO_INTERHAND21 = (20, 7, 6, 5, 11, 10, 9, 19, 18, 17, 15, 14, 13, 3, 2, 1)
SPLITS = ("train", "val")


def now():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, value):
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    # Duplicate raw labels must fail rather than silently replacing a frame.
    def unique(pairs):
        output = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(f"Duplicate JSON key {key!r} in {path}")
            output[key] = value
        return output
    with path.open() as stream:
        return json.load(stream, object_pairs_hook=unique)


def source_path(root, split, suffix):
    name = f"InterHand2.6M_{split}_{suffix}.json"
    candidates = [root / "annotations" / split / name, root / split / name, root / name]
    found = [path for path in candidates if path.is_file()]
    if len(found) != 1:
        raise ValueError(f"Require exactly one {name} beneath --root, found {found}")
    return found[0]


def file_provenance(path):
    return {"path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": sha256(path)}


def group_records(split, mano, joints):
    """One physical recording per capture/seq, independent of camera views."""
    groups, missing = defaultdict(list), Counter()
    for capture, frames in sorted(joints.items(), key=lambda item: int(item[0])):
        if ((split == "train" and not 0 <= int(capture) <= 26)
                or (split == "val" and int(capture) != 0)):
            raise ValueError(f"Capture outside the official v1 {split} release: {capture}")
        if not isinstance(frames, dict):
            raise ValueError(f"Invalid joint capture mapping: {split}/{capture}")
        for frame_key, reference in sorted(frames.items(), key=lambda item: int(item[0])):
            frame = int(frame_key)
            if frame < 0 or str(frame) != frame_key:
                raise ValueError(f"Noncanonical numeric frame key {split}/{capture}/{frame_key}")
            sequence = reference.get("seq")
            if not isinstance(sequence, str) or not sequence.strip():
                missing["reference_missing_sequence"] += 1
                continue
            if "/" in sequence or "\\" in sequence or sequence in (".", ".."):
                raise ValueError(f"Unsafe source sequence name {sequence!r}")
            groups[(str(capture), sequence)].append(frame)
            if frame_key not in mano.get(capture, {}):
                missing["reference_without_mano_frame"] += 1
    # MANO-only labels cannot be assigned safely to a recording.
    for capture, frames in mano.items():
        missing["mano_without_reference_frame"] += sum(
            frame not in joints.get(capture, {}) for frame in frames)
    return dict(groups), dict(missing)


def cadence_inventory(split, groups):
    aggregate = defaultdict(Counter)
    counts = Counter()
    for (capture, _), ids in groups.items():
        counts[capture] += len(ids)
        aggregate[capture].update(int(b-a) for a, b in zip(ids, ids[1:]))
    return {f"{split}/Capture{capture}": {
        "reference_frames": counts[capture],
        "recordings": sum(capture == key[0] for key in groups),
        "frame_index_step_histogram": dict(sorted(aggregate[capture].items())),
        "official_sparse_release_warning": split == "train" and int(capture) >= 10,
    } for capture in sorted(counts, key=int)}


def resolve_timebase(split, capture, sequence, inventory, overrides,
                     minimum_intervals=32, dominant_fraction=0.90):
    key = f"{split}/Capture{capture}"
    explicit_key = key + "/" + sequence
    explicit_key = explicit_key if explicit_key in overrides else key
    if explicit_key in overrides:
        value = overrides[explicit_key]
        if not isinstance(value, dict):
            raise ValueError(f"Timebase {explicit_key} must be a mapping")
        hz = float(value["frame_index_hz"])
        step = int(value["release_frame_step"])
        evidence = value.get("evidence")
        if (not math.isfinite(hz) or hz <= 0 or step < 1 or
                not isinstance(evidence, str) or not evidence.strip()):
            raise ValueError(f"Invalid/unreferenced timebase override: {explicit_key}")
        if not math.isclose(hz / step, 30.0, abs_tol=1e-6):
            raise ValueError(f"Expected 30fps release cadence, got {hz}/{step}: {explicit_key}")
        return {"frame_index_hz": hz, "release_frame_step": step,
                "nominal_dt": step / hz, "mode": "explicit_capture_or_sequence_override",
                "override_key": explicit_key, "evidence": evidence,
                "measured_timestamps": False}
    info = inventory[key]
    if info["official_sparse_release_warning"]:
        return {"rejected": True, "reason": "official_sparse_capture_timebase_not_established",
                "capture_key": key}
    histogram = Counter({int(k): int(v) for k, v in info["frame_index_step_histogram"].items()})
    total = sum(histogram.values())
    if not histogram or total < minimum_intervals:
        return {"rejected": True, "reason": "insufficient_capture_cadence_evidence", "capture_key": key}
    step, support = histogram.most_common(1)[0]
    if step not in (1, 3) or support / total < dominant_fraction:
        return {"rejected": True, "reason": "ambiguous_capture_frame_index_cadence", "capture_key": key,
                "mode_step": step, "mode_support_fraction": support / total}
    return {"frame_index_hz": float(30 * step), "release_frame_step": step,
            "nominal_dt": 1 / 30, "mode": "inferred_from_nonsparse_30fps_release_cadence",
            "capture_key": key, "mode_support_fraction": support / total,
            "minimum_intervals": minimum_intervals, "required_dominant_fraction": dominant_fraction,
            "evidence": [SOURCE_URL + "#interhand26m-in-30-fps", PAPER_URL,
                         "within-sequence reference frame-index histogram of this capture"],
            "limitation": "inference from nominal release cadence, not a published capture-to-hardware-rate table",
            "measured_timestamps": False}


def save_arrays(output, stem, arrays, storage):
    relative = stem if storage == "npy" else stem + ".npz"
    destination = output / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Existing prepared sequence: {destination}")
    if any(np.asarray(value).dtype.hasobject for value in arrays.values()):
        raise ValueError("Prepared object arrays are forbidden")
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
    return relative


def make_decoders(model_dir, asset_version):
    import torch
    from handloop.data.meta_mano import MANOReferenceDecoder
    from handloop.visual.codec import MANOStateCodec
    torch.set_num_threads(1)
    source = MANOReferenceDecoder(model_dir, device="cpu", fix_left_shapedirs=True,
                                  asset_version=asset_version)
    # Reuse the exact resulting source shape convention, including already
    # fixed left assets. The general visual codec normally assumes HaWoR's
    # unconditional flip, which must NOT be imposed on InterHand blindly.
    matrix = MANOStateCodec(Path(model_dir) / "MANO_RIGHT.pkl",
                           Path(model_dir) / "MANO_LEFT.pkl",
                           left_shapedirs_fix=source.left_shapedirs_fixed)
    return source, matrix


def convert_hand(args, source_decoder, matrix_decoder, split, capture, sequence,
                 reference_ids, mano_frames, joint_frames, side, timebase):
    import torch
    from handloop import geometry as g
    ids, poses, betas, translations, references, masks = [], [], [], [], [], []
    rejected = Counter()
    offset = 0 if side == "right" else 21
    for frame_id in reference_ids:
        key = str(frame_id)
        hands = mano_frames.get(key)
        record = hands.get(side) if isinstance(hands, dict) else None
        if record is None:
            rejected["missing_mano_hand"] += 1
            continue
        reference = joint_frames[key]
        try:
            pose = np.asarray(record["pose"], dtype=np.float32)
            beta = np.asarray(record["shape"], dtype=np.float32)
            trans = np.asarray(record["trans"], dtype=np.float32)
            native = np.asarray(reference["world_coord"], dtype=np.float32)
            valid = np.asarray(reference["joint_valid"])
        except (KeyError, TypeError, ValueError):
            rejected["malformed_source_record"] += 1
            continue
        if not np.isin(valid, (0, 1)).all():
            rejected["nonbinary_joint_valid"] += 1
            continue
        if valid.shape == (42, 1):
            valid = valid[:, 0]
        # Some schema prose says Jx3; require all coordinate validity entries.
        if valid.shape == (42, 3):
            valid = np.all(valid == 1, axis=-1)
        if (pose.shape != (48,) or beta.shape != (10,) or trans.shape != (3,)
                or native.shape != (42, 3) or valid.shape != (42,)):
            rejected["source_shape_mismatch"] += 1
            continue
        valid = valid[offset:offset + 21].astype(bool)
        native = native[offset:offset + 21] / 1000.0
        if not valid[20] or int(valid.sum()) < args.min_valid_joints:
            rejected["insufficient_reference_joint_validity"] += 1
            continue
        if not (np.isfinite(pose).all() and np.isfinite(beta).all()
                and np.isfinite(trans).all() and np.isfinite(native[valid]).all()):
            rejected["nonfinite_available_source"] += 1
            continue
        ids.append(frame_id)
        poses.append(pose)
        betas.append(beta)
        translations.append(trans)
        # Invalid reference points are placeholders with their explicit mask.
        references.append(np.where(valid[:, None], native, 0))
        masks.append(valid)
    if args.sample_frames:
        ids, poses, betas, translations, references, masks = (
            values[:args.sample_frames] for values in (ids, poses, betas, translations, references, masks))
    if len(ids) < args.min_sequence_frames:
        return None, {"reason": "too_few_valid_frames", "frames": len(ids), "skipped": dict(rejected)}
    ids = np.asarray(ids, dtype=np.int64)
    poses, betas, translations = (np.stack(v) for v in (poses, betas, translations))
    references, masks = np.stack(references), np.stack(masks)
    states = np.empty((len(ids), 99), dtype=np.float32)
    fit_sum, fit_count, fit_max, fit_wrist_sum, fit_wrist_max = 0.0, 0, 0.0, 0.0, 0.0
    relative_fit_sum = 0.0
    source_wrist_offset_max = 0.0
    rotation_ortho_max, rotation_det_max = 0.0, 0.0
    roundtrip_joint_max, roundtrip_mesh_max, roundtrip_frames = 0.0, 0.0, 0
    selected = set(np.linspace(0, len(ids)-1, min(args.fk_sample_frames, len(ids)), dtype=int).tolist())
    layer = source_decoder.layers[side]
    mano_indices = np.asarray(MANO16_TO_INTERHAND21)
    with torch.no_grad():
        for begin in range(0, len(ids), args.decode_batch):
            end = min(len(ids), begin + args.decode_batch)
            pose, beta, trans = (torch.as_tensor(v[begin:end], dtype=torch.float32)
                                 for v in (poses, betas, translations))
            source = layer(global_orient=pose[:, :3], hand_pose=pose[:, 3:],
                           betas=beta, transl=trans, return_verts=True, return_full_pose=True)
            expected = pose[:, 3:] + source_decoder.means[side]
            if not torch.allclose(source.full_pose[:, 3:], expected, rtol=1e-5, atol=1e-6):
                raise ValueError("Source MANO nonflat mean was not added exactly once")
            rotations = g.axis_angle_to_matrix(source.full_pose.reshape(-1, 16, 3))
            state = g.state_from_rotmat(source.joints[:, 0], rotations)
            if not all(torch.isfinite(v).all() for v in (state, source.vertices, source.joints)):
                raise ValueError(f"Nonfinite source FK: {split}/{capture}/{sequence}/{side}")
            states[begin:end] = state.numpy()
            identity = torch.eye(3)
            rotation_ortho_max = max(rotation_ortho_max, float((rotations.transpose(-1,-2) @ rotations-identity).abs().max()))
            rotation_det_max = max(rotation_det_max, float((torch.linalg.det(rotations)-1).abs().max()))
            source_wrist_offset_max = max(source_wrist_offset_max, float((source.joints[:,0]-trans).norm(dim=-1).max()))
            native = references[begin:end, mano_indices]
            available = masks[begin:end, mano_indices]
            decoded_joints = source.joints[:, :16].numpy()
            errors = np.linalg.norm(decoded_joints-native, axis=-1)
            measured = errors[available]
            fit_sum += float(measured.sum(dtype=np.float64))
            fit_count += int(measured.size)
            fit_max = max(fit_max, float(measured.max()))
            wrist_error = errors[:, 0]
            fit_wrist_sum += float(wrist_error.sum(dtype=np.float64))
            fit_wrist_max = max(fit_wrist_max, float(wrist_error.max()))
            local_error = np.linalg.norm((decoded_joints-decoded_joints[:, :1])-(native-native[:, :1]), axis=-1)
            relative_fit_sum += float(local_error[available].sum(dtype=np.float64))
            batch_checked = [index-begin for index in sorted(selected) if begin <= index < end]
            if batch_checked:
                check = torch.as_tensor(batch_checked, dtype=torch.int64)
                geometry = matrix_decoder.geometry(state[check], beta[check],
                                                     torch.full((len(check),), side == "right"))
                joint_error = (geometry["joints"]-source.joints[check, :16]).norm(dim=-1)
                mesh_error = (geometry["vertices"]-source.vertices[check]).norm(dim=-1)
                if not (torch.isfinite(joint_error).all() and torch.isfinite(mesh_error).all()):
                    raise ValueError("Nonfinite 99D FK roundtrip")
                roundtrip_joint_max = max(roundtrip_joint_max, float(joint_error.max()))
                roundtrip_mesh_max = max(roundtrip_mesh_max, float(mesh_error.max()))
                roundtrip_frames += len(check)
    if max(roundtrip_joint_max, roundtrip_mesh_max) > args.fk_tolerance_m:
        raise ValueError(f"InterHand 99D FK roundtrip exceeds {args.fk_tolerance_m}m: "
                         f"joints={roundtrip_joint_max}, vertices={roundtrip_mesh_max}")
    times = ids.astype(np.float64) / timebase["frame_index_hz"]
    if not np.all(np.diff(times) > 0):
        raise ValueError("InterHand source timestamps are not strictly increasing")
    frame_gaps = int(np.count_nonzero(np.diff(times) >= timebase["nominal_dt"] * 1.5))
    stem = f"{split}/Capture{capture}++{sequence}__{side}"
    relative = save_arrays(args.output, stem, {
        "state": states, "timestamps": times, "beta": betas.astype(np.float32),
        "beta_valid": np.asarray(True), "gt_mask": np.ones((len(ids),17), dtype=np.bool_),
        "frame_ids": ids, "source_joint_3d_m": references.astype(np.float32),
        "source_joint_valid": masks,
    }, args.storage)
    return {"path": relative, "storage": args.storage, "split": split,
            "dataset": "interhand", "recording": f"{split}/Capture{capture}/{sequence}",
            "capture": int(capture), "sequence": sequence, "handedness": int(side == "right"),
            "frames": len(ids), "source": "InterHand2.6M 30fps NeuralAnnot world-MANO fits",
            "coordinate_frame": "world_capture_" + split + "_" + capture, "units": "metres",
            "nominal_dt": timebase["nominal_dt"], "timebase": timebase,
            "frame_id_minmax": [int(ids[0]), int(ids[-1])], "timestamp_gaps": frame_gaps,
            "gt_complete": True, "partial_gt": False,
            "reference_quality": "multiview visual triangulation / NeuralAnnot MANO fit; not independent MoCap",
            "beta_storage": "original per-frame beta10, no temporal averaging",
            "beta_span": float(np.ptp(betas.astype(np.float64), axis=0).max()),
            "source_transl_to_J0_max_m": source_wrist_offset_max,
            "rotation_orthogonality_error": rotation_ortho_max,
            "rotation_determinant_error": rotation_det_max,
            "fk_roundtrip": {"checked": True, "frames_checked": roundtrip_frames,
                "sampling": "evenly spaced retained source frames", "joint_max_m": roundtrip_joint_max,
                "vertex_max_m": roundtrip_mesh_max, "tolerance_m": args.fk_tolerance_m,
                "scope": "same assets and source shape convention, source smplx FK -> serialized99D -> matrix FK; not independent implementation/GT validation"},
            "source_fit_diagnostic": {"joints": "MANO16 kinematic joints mapped to corresponding InterHand points",
                "mano16_to_interhand21": list(MANO16_TO_INTERHAND21),
                "available_joint_instances": fit_count, "absolute_mean_m": fit_sum / fit_count,
                "absolute_max_m": fit_max, "wrist_aligned_mean_m": relative_fit_sum / fit_count,
                "wrist_mean_m": fit_wrist_sum / len(ids), "wrist_max_m": fit_wrist_max,
                "interpretation": "source fitting/reference discrepancy; NOT 99D conversion error; no error threshold selection"},
            "skipped": dict(rejected)}, None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mano-models", type=Path)
    parser.add_argument("--mano-asset-version", default=None)
    parser.add_argument("--source-code-revision", default=None, help="audited official renderer git revision, if pinned")
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--storage", choices=("npy", "npz"), default="npy")
    parser.add_argument("--timebase-json", type=Path)
    parser.add_argument("--index-only", action="store_true")
    parser.add_argument("--max-sequences-per-split", "--max-sequences", type=int, default=0,
                        help="bounded recording count per split, 0=all; split/cadence inventories use all metadata")
    parser.add_argument("--sample-frames", type=int, default=0,
                        help="bounded smoke: keep only first N valid frames per hand; 0=all (no downsampling)")
    parser.add_argument("--min-sequence-frames", type=int, default=40)
    parser.add_argument("--min-valid-joints", type=int, default=21,
                        help="source native21 required valid points; wrist always required")
    parser.add_argument("--decode-batch", type=int, default=128)
    parser.add_argument("--fk-sample-frames", type=int, default=32)
    parser.add_argument("--fk-tolerance-m", type=float, default=1e-5)
    args = parser.parse_args()
    if not Path("/sciclone/home/jchen57").is_dir():
        parser.error("Run data preparation only in the authorized remote ds Pod")
    if (min(args.max_sequences_per_split, args.sample_frames) < 0 or args.min_sequence_frames < 2
            or not 1 <= args.min_valid_joints <= 21 or min(args.decode_batch, args.fk_sample_frames) < 1
            or not math.isfinite(args.fk_tolerance_m) or args.fk_tolerance_m <= 0
            or len(set(args.splits)) != len(args.splits)):
        parser.error("invalid counts/splits/tolerance")
    if "5fps" in str(args.root).lower():
        parser.error("Only the official 30fps release is permitted; do not mix the 5fps duplicate")
    if not args.index_only and args.mano_models is None:
        parser.error("--mano-models required for source FK and anatomical J0 conversion")
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / ("index_manifest.json" if args.index_only else "manifest.json")
    if target.exists():
        parser.error("output manifest already exists; choose a fresh immutable output")
    overrides = load_json(args.timebase_json) if args.timebase_json else {}
    if not isinstance(overrides, dict):
        parser.error("timebase JSON must be a mapping")
    source_decoder, matrix_decoder = (None, None) if args.index_only else make_decoders(args.mano_models, args.mano_asset_version)
    manifest = {"utc": now(), "phase": "indexing" if args.index_only else "preparing",
        "dataset": "interhand", "source": "official InterHand2.6M 30fps v1 NeuralAnnot MANO fits",
        "source_urls": [SOURCE_URL, RENDERER_URL, PAPER_URL], "source_revision": args.source_code_revision,
        "storage": args.storage, "rotation6d": "columns_c0xyz_c1xyz",
        "position": "actual decoded anatomical J0 in capture-world metres; source MANO transl is NOT J0",
        "coordinate_frame": "capture_world", "shape": "per-frame source beta10 retained separately",
        "split_rule": "official train/val files only; never load test or duplicate 5fps release",
        "split_identity_note": "capture numbers are local to each official split",
        "source_convention": {"use_pca": False, "flat_hand_mean": False,
            "pose48": "axis-angle minus hand mean; source smplx adds hand mean exactly once",
            "left_shapedirs": "official conditional fix if sum(abs(left[:,0,:]-right[:,0,:]))<1",
            "trans": "MANO translation in metres, not actual wrist position",
            "joint_3d": "world millimetres; per-hand InterHand tip-to-base order ending with wrist"},
        "decoder_provenance": source_decoder.provenance() if source_decoder else None,
        "timestamp_policy": "raw numeric frame_idx divided by audited/inferred capture-specific index Hz; preserve gaps; no interpolation",
        "timebase_override_file": file_provenance(args.timebase_json) if args.timebase_json else None,
        "quality_filter": {"minimum_native_valid_joints": args.min_valid_joints,
            "wrist_required": True, "finite_available_parameters_required": True,
            "source_fit_error_cutoff": None},
        "bounded_limits": {"recordings_per_split": args.max_sequences_per_split,
            "first_valid_frames_per_hand": args.sample_frames},
        "source_files": [], "cadence_inventory": {}, "splits_summary": {}, "sequences": [], "rejected": []}
    status_path = args.output / "preparation_status.json"
    save_json(status_path, {"utc": now(), "phase": manifest["phase"], "step": "loading_source_json"})
    try:
        for split in args.splits:
            mano_path = source_path(args.root, split, "MANO_NeuralAnnot")
            joint_path = source_path(args.root, split, "joint_3d")
            manifest["source_files"].extend([file_provenance(mano_path), file_provenance(joint_path)])
            mano, joints = load_json(mano_path), load_json(joint_path)
            groups, missing = group_records(split, mano, joints)
            inventory = cadence_inventory(split, groups)
            manifest["cadence_inventory"].update(inventory)
            chosen, all_recordings = [], len(groups)
            timebase_rejected = Counter()
            for (capture, sequence), ids in sorted(groups.items(), key=lambda item: (int(item[0][0]), item[0][1])):
                timebase = resolve_timebase(split, capture, sequence, inventory, overrides)
                if timebase.get("rejected"):
                    timebase_rejected[timebase["reason"]] += 1
                    manifest["rejected"].append({"split": split, "capture": int(capture),
                        "sequence": sequence, "reference_frames": len(ids), **timebase})
                    continue
                chosen.append((capture, sequence, ids, timebase))
            if args.max_sequences_per_split:
                chosen = chosen[:args.max_sequences_per_split]
            manifest["splits_summary"][split] = {"official_recordings_in_joint_3d": all_recordings,
                "selected_recordings": len(chosen), "timebase_rejected_recordings": dict(timebase_rejected),
                "source_missing_counts": missing}
            save_json(args.output / "cadence_inventory.json", manifest["cadence_inventory"])
            for done, (capture, sequence, ids, timebase) in enumerate(chosen, 1):
                if not args.index_only:
                    for side in ("left", "right"):
                        entry, rejection = convert_hand(args, source_decoder, matrix_decoder, split, capture,
                            sequence, ids, mano.get(capture, {}), joints[capture], side, timebase)
                        if entry is not None:
                            manifest["sequences"].append(entry)
                        else:
                            manifest["rejected"].append({"split": split, "capture": int(capture),
                                "sequence": sequence, "side": side, **rejection})
                status = {"utc": now(), "phase": manifest["phase"], "split": split,
                    "recordings_done": done, "recordings_selected": len(chosen),
                    "prepared_hand_sequences": len(manifest["sequences"]),
                    "last_recording": f"Capture{capture}/{sequence}"}
                save_json(status_path, status)
                if done == 1 or done % 20 == 0 or done == len(chosen):
                    print(json.dumps(status), flush=True)
            del mano, joints
        if not args.index_only and not manifest["sequences"]:
            raise ValueError("No accepted trajectories; inspect cadence_inventory.json and source validity")
        manifest["sequences"].sort(key=lambda item: item["path"])
        manifest["phase"] = "indexed" if args.index_only else "complete"
        manifest["completed_utc"] = now()
        manifest["frames_by_split"] = {split: sum(entry["frames"] for entry in manifest["sequences"]
                                                  if entry["split"] == split) for split in args.splits}
        manifest["fk_roundtrip_verified"] = not args.index_only
        manifest["full_fk_verified"] = False
        manifest["fk_verification_note"] = "sampled same-decoder source->99D geometry roundtrip; source/native discrepancy reported separately"
        save_json(target, manifest)
        summary = {"utc": now(), "phase": manifest["phase"], "manifest": str(target),
            "sequences": len(manifest["sequences"]), "frames_by_split": manifest["frames_by_split"],
            "rejected_entries": len(manifest["rejected"])}
        save_json(status_path, summary)
        print(json.dumps(summary), flush=True)
    except Exception as exc:
        save_json(args.output / "partial_manifest.json", manifest)
        save_json(status_path, {"utc": now(), "phase": "failed", "error_type": type(exc).__name__,
                                "error": str(exc), "prepared_sequences": len(manifest["sequences"])})
        raise


if __name__ == "__main__":
    main()
