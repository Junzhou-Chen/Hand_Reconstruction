#!/usr/bin/env python3
"""Remote-only HOT3D Aria RGB/world-reference sources for joint adaptation.

Only downloaded official train_aria clips are eligible. Preserve the existing
pure-motion parent-recording split and all 27 HaWoR evaluation exclusions.
Decode the original source PCA15 with its own beta and Meta shape convention;
verify its 99D state against the immutable prepared reference. Source RGB is
fisheye: use the pinned official camera projection to warp to a pinhole camera,
then rotate pixels clockwise 90 degrees AND update K/camera_to_world together.
No network, pseudo-label fitting, test labels or test-derived selection occurs.

Meta's 20 published landmarks occupy thumb-first meta21 slots with thumb base
masked. GT-projected crops remain a controlled pilot; the moving camera pose
is source calibration, not an estimated deployment SLAM trajectory.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
import tarfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from handloop.data.meta_mano import MANOReferenceDecoder, META_TOOLKIT_REVISION, META_FINGERTIP_VERTICES
from handloop.data.visual_clips import file_sha256
from prepare_visual_feedback_cache import SOURCE_SCHEMA, now, save_json
from prepare_visual_joint_sources import FOCAL_CONVENTION, identity, inside


STREAM = "214-1"
ORDER = (0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20)
REFERENCE_METADATA = {
    "dataset": "hot3d", "reference_source": "official Meta marker-based trajectories fitted to MANO; source beta/decoder retained; not native joint sensor GT",
    "reference_joint_order": "meta21", "reference_units": "metres", "reference_coordinate_frame": "world",
    "reference_left_shapedirs_fix": True, "joint_3d_order": "meta21", "joint_3d_units": "metres",
    "joint_3d_coordinate_frame": "world", "joint_3d_valid_policy": "explicit_mask",
    "reference_joint_missing_policy": "published Meta20 landmarks; thumb-base meta21 slot1 false",
}
_DECODER = None
_CAMERA = None


def init_worker(model_dir: str, asset_version: str | None, toolkit: str):
    global _DECODER, _CAMERA
    import torch
    import cv2
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    sys.path.insert(0, toolkit)
    from hand_tracking_toolkit import camera
    _CAMERA = camera
    _DECODER = MANOReferenceDecoder(model_dir, device="cpu", fix_left_shapedirs=True,
                                   asset_version=asset_version)


def read_json(archive, name):
    member = archive.getmember(name)
    if not member.isfile() or not 0 < member.size < 8_000_000:
        raise ValueError(f"Unexpected source JSON: {name}")
    return json.load(archive.extractfile(member))


def read_reference(root: Path, entry: dict) -> tuple[dict, dict]:
    path = inside(root, entry["path"])
    if entry["storage"] != "npz":
        raise ValueError("Use complete audited HOT per-hand NPZ source references")
    original = identity(path)
    if original["sha256"] != entry["storage_artifact"]["sha256"]:
        raise ValueError("Prepared HOT reference container SHA changed")
    with np.load(path, allow_pickle=False) as stored:
        keys = ("state", "timestamps", "frame_ids", "beta", "gt_mask", "ref_timestamps_ns")
        arrays = {key: stored[key].copy() for key in keys}
    for key, value in arrays.items():
        expected = entry["arrays"][key]
        if (list(value.shape) != expected["shape"] or str(value.dtype) != expected["dtype"]
                or hashlib.sha256(value.tobytes(order="C")).hexdigest() != expected["content_sha256"]
                or not np.isfinite(value).all()):
            raise ValueError(f"Prepared HOT array identity changed: {key}")
    count = int(entry["frames"])
    if (arrays["state"].shape != (count, 99) or arrays["frame_ids"].shape != (count,)
            or arrays["gt_mask"].shape != (count, 17) or not arrays["gt_mask"].all()
            or arrays["beta"].shape != (10,) or np.any(np.diff(arrays["frame_ids"]) <= 0)
            or np.any(np.diff(arrays["ref_timestamps_ns"]) <= 0)):
        raise ValueError("Invalid complete/ordered HOT reference")
    if identity(path) != original:
        raise ValueError("Prepared HOT reference changed while loading")
    return arrays, original


def pinhole_parameters(raw: dict, image_size: int) -> tuple:
    """Official projection, plus an explicit center-consistent image scaling."""
    source = _CAMERA.from_json(raw)
    if (raw["calibration"]["projection_model_type"] != "CameraModelType.FISHEYE624"
            or len(raw["calibration"]["projection_params"]) != 15
            or source.width != source.height or "rgb" not in source.label.lower()):
        raise ValueError("Require audited square Aria FISHEYE624 RGB camera")
    scale = image_size / source.width
    f = np.asarray(source.f, dtype=np.float64) * scale
    c = (np.asarray(source.c, dtype=np.float64) + .5) * scale - .5
    pinhole = _CAMERA.PinholePlaneCameraModel(width=image_size, height=image_size,
        f=f, c=c, distort_coeffs=[], T_world_from_eye=source.T_world_from_eye)
    # Clockwise image rotation maps [u,v] -> [H-1-v,u]. Its positive-focal
    # camera coordinates become [-y,x,z], so T_new = T_old @ R_cw.T.
    clockwise = np.asarray([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    transform = np.asarray(source.T_world_from_eye, dtype=np.float64).copy()
    transform[:3, :3] = transform[:3, :3] @ clockwise.T
    intrinsic = np.asarray([[f[1], 0., image_size - 1 - c[1]],
                            [0., f[0], c[0]], [0., 0., 1.]], dtype=np.float64)
    if (np.max(np.abs(transform[:3, :3].T @ transform[:3, :3] - np.eye(3))) > 1e-5
            or not np.isfinite(intrinsic).all()):
        raise ValueError("Nonrigid/nonfinite rotated camera")
    return source, pinhole, intrinsic, transform


def remap_for_camera(source, pinhole) -> tuple[np.ndarray, np.ndarray, float]:
    """Same rays/projection/remap as toolkit.dataset.warp_image, cached per K.

    Extrinsics cancel because source and destination use the identical eye.
    Keeping the ray calculation in eye coordinates also avoids subtracting
    large float32 world translations when constructing the sampling map.
    """
    width, height = pinhole.width, pinhole.height
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    pixels = np.column_stack((xx.ravel(), yy.ravel()))
    rays = pinhole.window_to_eye(pixels)
    source_pixels = source.eye_to_window(rays)
    source_pixels[rays[:, 2] < 0] = -1
    if not np.isfinite(source_pixels).all():
        raise ValueError("Nonfinite fisheye-to-pinhole sampling map")
    # Independently check the equivalent official world-coordinate path on
    # a grid, in float64, without importing its unrelated training packages.
    grid = np.linspace(0, len(pixels) - 1, min(128, len(pixels)), dtype=np.int64)
    matrix = np.asarray(pinhole.T_world_from_eye, dtype=np.float64)
    world = rays[grid].astype(np.float64) @ matrix[:3, :3].T + matrix[:3, 3]
    # Use a true inverse for this check: source camera rotations are float32.
    inverse = np.linalg.inv(matrix)
    eye = world @ inverse[:3, :3].T + inverse[:3, 3]
    error = float(np.max(np.abs(source.eye_to_window(eye) - source_pixels[grid])))
    if error > 1e-5:
        raise ValueError(f"Cached and world-coordinate camera remaps disagree: {error}")
    return (source_pixels[:, 0].astype(np.float32).reshape(height, width),
            source_pixels[:, 1].astype(np.float32).reshape(height, width), error)


def project_world(joints: np.ndarray, transform: np.ndarray, intrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    inverse = np.linalg.inv(transform)
    camera_joints = joints.astype(np.float64) @ inverse[:3, :3].T + inverse[:3, 3]
    projected = camera_joints @ intrinsic.T
    return projected[..., :2] / projected[..., 2:], camera_joints


def prepare_clip(job: dict) -> dict:
    import cv2
    if _DECODER.provenance() != job["decoder_provenance"]:
        raise ValueError("Worker MANO codec/assets differ from original motion preparation")
    output, prepared_root = Path(job["output"]), Path(job["prepared"])
    tar_path = Path(job["hot_root"]) / job["source_path"]
    before = {"bytes": tar_path.stat().st_size, "mtime_ns": tar_path.stat().st_mtime_ns}
    if before["bytes"] != job["source_bytes"] or (job["verify_tar_sha"] and file_sha256(tar_path) != job["source_sha256"]):
        raise ValueError("HOT archive size/SHA differs from immutable download/preparation")
    records, rejected, metadata_digest = [], [], hashlib.sha256()
    states_max_error, rotation_pixel_max_error, map_max_error = 0., 0., 0.
    camera_data, image_metadata, map_cache = {}, {}, {}
    with tarfile.open(tar_path, "r:") as archive:
        names = [member.name for member in archive.getmembers() if member.isfile()]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate source TAR member")
        shape = read_json(archive, "__hand_shapes.json__")
        source_beta = np.asarray(shape["mano"], dtype=np.float32)
        frame_metadata = {}
        for entry in job["entries"]:
            reference, reference_identity = read_reference(prepared_root, entry)
            if not np.array_equal(source_beta, reference["beta"]):
                raise ValueError("Source/prepared beta10 differs")
            side = "right" if entry["handedness"] else "left"
            ids, thetas, wrists, times = reference["frame_ids"], [], [], []
            for position, frame_id in enumerate(ids):
                frame_key = f"{int(frame_id):06d}"
                if frame_key not in frame_metadata:
                    info = read_json(archive, frame_key + ".info.json")
                    hands = read_json(archive, frame_key + ".hands.json")
                    cameras = read_json(archive, frame_key + ".cameras.json")
                    if (info["sequence_id"] != entry["recording"] or info["device"] != "Aria"
                            or info["participant_id"] != entry["subject"] or STREAM not in cameras
                            or info["image_timestamps_ns"][STREAM] != info["ref_timestamp_ns"]):
                        raise ValueError("Source RGB/reference camera, participant or timestamp mismatch")
                    frame_metadata[frame_key] = (info, hands, cameras[STREAM])
                    metadata_digest.update(json.dumps((frame_key, info, cameras[STREAM]), sort_keys=True).encode())
                info, hands, camera = frame_metadata[frame_key]
                if int(info["ref_timestamp_ns"]) != int(reference["ref_timestamps_ns"][position]):
                    raise ValueError("Prepared and actual source nanosecond clock mismatch")
                pose = hands[side]["mano_pose"]
                thetas.append(pose["thetas"])
                wrists.append(pose["wrist_xform"])
                times.append(int(info["ref_timestamp_ns"]))
                if frame_key not in camera_data:
                    original, pinhole, intrinsic, transform = pinhole_parameters(camera, job["image_size"])
                    calibration_key = json.dumps(camera["calibration"], sort_keys=True)
                    if calibration_key not in map_cache:
                        map_x, map_y, map_error = remap_for_camera(original, pinhole)
                        map_max_error = max(map_max_error, map_error)
                        map_cache[calibration_key] = (map_x, map_y)
                    camera_data[frame_key] = (original, pinhole, intrinsic, transform, calibration_key)
            state_chunks, joint_chunks = [], []
            for begin in range(0, len(ids), 64):
                end = min(begin + 64, len(ids))
                decoded = _DECODER.decode_meta(np.asarray(thetas[begin:end], dtype=np.float32),
                    np.asarray(wrists[begin:end], dtype=np.float32), source_beta,
                    np.full(end - begin, int(entry["handedness"]), dtype=np.int64))
                state_chunks.append(decoded.state.cpu().numpy())
                all_joints = np.concatenate((decoded.joints_m.cpu().numpy(),
                    decoded.vertices_m[:, list(META_FINGERTIP_VERTICES)].cpu().numpy()), axis=1)
                joint_chunks.append(all_joints[:, ORDER])
            states, joints = np.concatenate(state_chunks), np.concatenate(joint_chunks)
            error = float(np.max(np.abs(states - reference["state"])))
            states_max_error = max(states_max_error, error)
            if error > 1e-5:
                raise ValueError(f"Source MANO and original prepared state differ: {error}")
            selected, boxes, intrinsics, transforms = [], [], [], []
            skipped = Counter()
            for position, frame_id in enumerate(ids):
                original, pinhole, intrinsic, transform, _ = camera_data[f"{int(frame_id):06d}"]
                projected, camera_joints = project_world(joints[position], transform, intrinsic)
                valid_slots = np.ones(21, dtype=np.bool_)
                valid_slots[1] = False
                if np.any(camera_joints[valid_slots, 2] <= 1e-4):
                    skipped["reference_landmark_nonpositive_pinhole_depth"] += 1
                    continue
                # Verify clockwise pixel rotation, K and world extrinsics
                # against a separate unrotated pinhole-camera projection.
                old = pinhole.world_to_window(joints[position].astype(np.float64))
                rotated = np.stack((job["image_size"] - 1 - old[:, 1], old[:, 0]), axis=-1)
                pixel_error = float(np.max(np.abs(projected[valid_slots] - rotated[valid_slots])))
                rotation_pixel_max_error = max(rotation_pixel_max_error, pixel_error)
                if pixel_error > .02:
                    raise ValueError(f"Rotated K/camera_to_world projection mismatch: {pixel_error}px")
                box = np.concatenate((projected[valid_slots].min(axis=0), projected[valid_slots].max(axis=0)))
                clipped = np.concatenate((np.maximum(box[:2], 0), np.minimum(box[2:], job["image_size"] - 1)))
                if not np.isfinite(box).all() or np.any(clipped[2:] - clipped[:2] < 4):
                    skipped["reference_bbox_has_less_than_4px_pinhole_intersection"] += 1
                    continue
                # Crop only the portion present in this pinhole image. This
                # avoids treating unobserved fisheye rays as an RGB detection.
                selected.append(position)
                boxes.append(clipped)
                intrinsics.append(intrinsic)
                transforms.append(transform)
            if not selected:
                rejected.append({"recording": entry["recording"], "source_path": job["source_path"],
                                 "side": side, "reason": "no_projectable_in_frame_source_reference", "skipped": dict(skipped)})
                continue
            selected = np.asarray(selected, dtype=np.int64)
            selected_ids = ids[selected].astype(np.int64)
            selected_times = np.asarray(times, dtype=np.int64)[selected]
            stem = entry["split"] + "/" + tar_path.stem + "__" + side
            destination = output / stem
            destination.mkdir(parents=True, exist_ok=False)
            image_paths = []
            for frame_id in selected_ids:
                frame_key = f"{int(frame_id):06d}"
                relative = "images/" + tar_path.stem + "/" + frame_key + ".jpg"
                image_paths.append(relative)
                if frame_key in image_metadata:
                    continue
                original, _, _, _, calibration_key = camera_data[frame_key]
                member = archive.getmember(frame_key + ".image_" + STREAM + ".jpg")
                if not member.isfile() or not 0 < member.size <= 20_000_000:
                    raise ValueError("Invalid source RGB member")
                encoded = archive.extractfile(member).read()
                image = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_UNCHANGED)
                if image is None or image.shape != (original.height, original.width, 3):
                    raise ValueError("Source RGB dimensions/channels disagree with calibrated Aria stream")
                map_x, map_y = map_cache[calibration_key]
                pinhole_image = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
                upright = np.ascontiguousarray(np.rot90(pinhole_image, k=3))
                path = inside(output, relative)
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists() or not cv2.imwrite(str(path), upright, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise ValueError("Refuse duplicate/failed pinhole image publication")
                image_metadata[frame_key] = {"path": relative, "sha256": file_sha256(path),
                    "source_jpeg_sha256": hashlib.sha256(encoded).hexdigest()}
            mask = np.ones((len(selected), 21), dtype=np.bool_)
            mask[:, 1] = False
            reference_path = destination / "reference.npz"
            with reference_path.open("xb") as stream:
                np.savez(stream, target_world=states[selected].astype(np.float32),
                    target_valid=np.ones((len(selected), 17), dtype=np.bool_),
                    timestamps=selected_times.astype(np.float64) * 1e-9,
                    ref_timestamps_ns=selected_times, frame_ids=selected_ids,
                    bbox_xyxy=np.asarray(boxes, dtype=np.float32), joint_3d=joints[selected].astype(np.float32),
                    joint_3d_valid=mask, beta=source_beta, camera_intrinsics=np.asarray(intrinsics, dtype=np.float32),
                    camera_to_world=np.asarray(transforms, dtype=np.float32),
                    handedness=np.full(len(selected), int(entry["handedness"]), dtype=np.int64))
            records.append({**REFERENCE_METADATA, "recording": "hot3d/" + entry["recording"] + "/" + tar_path.stem + "/" + side,
                "parent_recording": entry["recording"], "split_group": entry["split_group"],
                "subject": entry["subject"], "split": entry["split"], "path": stem,
                "frames": len(selected), "image_paths": image_paths, "serial": STREAM,
                "handedness": int(entry["handedness"]), "coordinate_frame": "source_recording_world",
                "timestamp_source": "exact source ref_timestamp_ns and matching RGB image_timestamps_ns['214-1']",
                "nominal_dt": 1 / 30, "source_path": job["source_path"], "source_archive_sha256": job["source_sha256"],
                "source_archive_identity": before, "source_reference_identity": reference_identity,
                "reference_file": stem + "/reference.npz", "reference_sha256": file_sha256(reference_path),
                "image_sha256": {path: image_metadata[Path(path).stem]["sha256"] for path in image_paths},
                "source_jpeg_sha256": {path: image_metadata[Path(path).stem]["source_jpeg_sha256"] for path in image_paths},
                "frame_id_minmax": [int(selected_ids[0]), int(selected_ids[-1])],
                "frame_gaps": int(np.count_nonzero(np.diff(selected_ids) != 1)), "source_num_frames": entry["source_frames"],
                "source_valid_mano_frames": len(ids), "skipped": dict(skipped), "gt_complete": True,
                "visibility_policy": "reference bbox has >=4px intersection and all published landmarks positive pinhole depth; not an occlusion visibility label",
                "pinhole_image_size": job["image_size"], "source_mano_redecode_max_abs_state_error": error})
    stat = tar_path.stat()
    if {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns} != before:
        raise ValueError("Source HOT TAR changed during preparation")
    return {"source_path": job["source_path"], "records": records, "rejected": rejected,
            "unique_rgb_images": len(image_metadata), "source_metadata_payload_sha256": metadata_digest.hexdigest(),
            "source_mano_redecode_max_abs_state_error": states_max_error,
            "rotated_pinhole_projection_max_error_px": rotation_pixel_max_error,
            "sampling_map_equivalence_max_error_px": map_max_error}


def select_clips(prepared: dict, download: dict, args) -> tuple[list[dict], dict]:
    source_files = {entry["path"]: entry for entry in download["files"]}
    grouped, parent_splits = defaultdict(list), {}
    excluded = set(prepared["split"]["hawor_excluded_recordings"])
    if len(excluded) != 27:
        raise ValueError("Require all 27 original HaWoR evaluation parent exclusions")
    train_parents, val_parents = set(prepared["split"]["train_groups"]), set(prepared["split"]["val_groups"])
    if train_parents & val_parents or (train_parents | val_parents) & excluded:
        raise ValueError("Original HOT parent split/exclusions overlap")
    for entry in prepared["sequences"]:
        source = Path(entry["source_path"])
        if source.parent.as_posix() != "train_aria":
            continue  # Quest3 is monochrome and does not satisfy the RGB task.
        parent, split = entry["recording"], entry["split"]
        if (parent in excluded or parent != entry["split_group"] or split not in ("train", "val")
                or (parent in val_parents) != (split == "val") or parent not in train_parents | val_parents
                or not entry.get("gt_complete") or entry.get("timestamp_source") != "ref_timestamp_ns"):
            raise ValueError("Prepared Aria split/clock/parent disagrees with original motion protocol")
        if parent in parent_splits and parent_splits[parent] != split:
            raise ValueError("Physical parent appears in both visual splits")
        parent_splits[parent] = split
        original = source_files[entry["source_path"]]
        if (original["hash_kind"] != "sha256" or original["hash"] != entry["source_sha256"]
                or original["size"] != entry["source_bytes"] or original["sequence_id"] != parent
                or original["device"] != "Aria"):
            raise ValueError("Prepared Aria source differs from pinned download metadata")
        grouped[entry["source_path"]].append(entry)
    selected = []
    for split in ("train", "val"):
        parents = sorted(parent for parent, value in parent_splits.items() if value == split)
        if args.parents_per_split and len(parents) > args.parents_per_split:
            parents = [parents[int(i)] for i in np.linspace(0, len(parents) - 1, args.parents_per_split, dtype=np.int64)]
        for parent in parents:
            clips = sorted(path for path, entries in grouped.items() if entries[0]["recording"] == parent)
            if args.clips_per_parent and len(clips) > args.clips_per_parent:
                clips = [clips[int(i)] for i in np.linspace(0, len(clips) - 1, args.clips_per_parent, dtype=np.int64)]
            for path in clips:
                entries = grouped[path]
                if len({entry["handedness"] for entry in entries}) != len(entries):
                    raise ValueError("Duplicate same-hand prepared clip")
                selected.append({"source_path": path, "source_sha256": entries[0]["source_sha256"],
                                 "source_bytes": entries[0]["source_bytes"], "entries": entries})
    if not selected or {job["entries"][0]["split"] for job in selected} != {"train", "val"}:
        raise ValueError("Need eligible official-train Aria clips in both original internal splits")
    return selected, {"rule": "preserve existing pure-motion parent recording split, no test-dependent selection",
        "motion_split_seed": prepared["split"]["seed"], "parents_per_split_cap": args.parents_per_split,
        "clips_per_parent_cap": args.clips_per_parent, "cap_selection": "evenly spaced sorted metadata IDs",
        "selected_clips": len(selected), "selected_parents": len({job["entries"][0]["recording"] for job in selected}),
        "source_stream": "Aria RGB214-1 only; Quest3 monochrome excluded",
        "hawor_excluded_recordings": sorted(excluded)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hot-root", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--mano-models", type=Path, required=True)
    parser.add_argument("--camera-toolkit", type=Path, required=True, help="Pinned official repository with SOURCE_REVISION")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=704)
    parser.add_argument("--parents-per-split", type=int, default=0, help="0 all; diagnostic cap before target reads")
    parser.add_argument("--clips-per-parent", type=int, default=0, help="0 all; diagnostic cap before target reads")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-tar-rehash", action="store_true", help="Record use of completed download SHA receipt plus size/stat only")
    parser.add_argument("--code-version", default="unrecorded")
    args = parser.parse_args()
    if not Path("/sciclone/home/jchen57").is_dir():
        parser.error("Run all RGB/reference preprocessing in the authorized remote ds Pod")
    if args.workers < 1 or args.image_size < 128 or min(args.parents_per_split, args.clips_per_parent) < 0:
        parser.error("Invalid worker/image size/metadata cap")
    for key in ("hot_root", "prepared", "mano_models", "camera_toolkit", "output"):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    toolkit_revision = (args.camera_toolkit / "SOURCE_REVISION").read_text().strip()
    if toolkit_revision != META_TOOLKIT_REVISION:
        raise ValueError("Camera toolkit must match the original audited Meta decoder revision")
    toolkit_files = {name: identity(args.camera_toolkit / "hand_tracking_toolkit" / name)
                     for name in ("camera.py", "camera_distortion.py", "math_utils.py")}
    prepared_path, download_path = args.prepared / "manifest.json", args.hot_root / "download_manifest.json"
    prepared_identity, download_identity = identity(prepared_path), identity(download_path)
    prepared, download = json.loads(prepared_path.read_text()), json.loads(download_path.read_text())
    if (prepared.get("phase") != "complete" or prepared.get("dataset") != "hot3d"
            or not prepared.get("full_fk_executed") or prepared.get("bounded_limit") != 0
            or prepared["source_manifest_sha256"] != download_identity["sha256"]
            or prepared["source_revision"] != download["revision"]):
        raise ValueError("Use original complete, uncapped, pinned HOT motion preparation")
    receipt = json.loads((args.hot_root / "download_status.json").read_text())
    if receipt.get("phase") != "complete" or receipt["counts"]["failed"] != 0:
        raise ValueError("HOT download receipt is not fully verified")
    selected, selection = select_clips(prepared, download, args)
    original_decoder = prepared["decoder_provenance"]
    init_worker(str(args.mano_models), original_decoder.get("asset_version"), str(args.camera_toolkit))
    decoder = _DECODER.provenance()
    if decoder != original_decoder or not decoder["left_shapedirs_fix_requested"]:
        raise ValueError("Keep exact source MANO assets, shape convention and decoder runtime")
    if args.output.exists():
        raise FileExistsError("Use a fresh immutable HOT visual output")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {"schema": SOURCE_SCHEMA, "phase": "preparing", "created_utc": now(), "dataset": "hot3d",
        "setup": "official_train_aria_original_motion_parent_split", "code_version": args.code_version,
        "reference_source": REFERENCE_METADATA["reference_source"], "reference_joint_order": "meta21",
        "reference_units": "metres", "reference_coordinate_frame": "world", "reference_left_shapedirs_fix": True,
        "joint_3d_order": "meta21", "joint_3d_units": "metres", "joint_3d_coordinate_frame": "world",
        "reference_manifest": prepared_identity, "download_manifest": download_identity, "decoder_provenance": decoder,
        "camera_toolkit_revision": toolkit_revision, "camera_toolkit_files": toolkit_files,
        "camera_protocol": "official FISHEYE624 projection; centered scaled pinhole; clockwise90 pixels with corresponding positive-focal K and T_world_from_camera rotation",
        "pinhole_image_size": args.image_size, "image_encoding": "JPEG quality95 after linear remap; no grayscale replication",
        "sampling_map_protocol": "pinhole rays to original calibrated fisheye eye_to_window; same eye as official warp_image; float64 world-path cross-check",
        "focal_convention": FOCAL_CONVENTION, "coordinate_convention": "column vectors; source recording world; metres; calibrated moving camera",
        "crop_protocol": "GT projection of published Meta20 landmarks in pinhole image; box clipped to image; >=4px intersection; official HaWoR square1.2 expansion",
        "input_protocol": "source RGB214-1 only; calibrated world/camera poses; GT crop pilot; no depth/detector/estimated SLAM; occlusion is not filtered",
        "selection_protocol": selection, "native_supervision": "not used; source fitted MANO geometry is the reference",
        "source_archive_hash_policy": "full TAR SHA recomputed" if not args.skip_tar_rehash else "completed download SHA receipt plus source size/stat; no full rehash",
        "records": [], "rejected": [], "clip_audits": []}
    incomplete = args.output / "source_manifest.incomplete.json"
    save_json(incomplete, manifest)
    jobs = [{**job, "output": str(args.output), "prepared": str(args.prepared), "hot_root": str(args.hot_root),
             "image_size": args.image_size, "decoder_provenance": decoder, "verify_tar_sha": not args.skip_tar_rehash}
            for job in selected]
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
            initializer=init_worker, initargs=(str(args.mano_models), original_decoder.get("asset_version"), str(args.camera_toolkit))) as pool:
        futures = [pool.submit(prepare_clip, job) for job in jobs]
        for count, future in enumerate(as_completed(futures), 1):
            result = future.result()
            manifest["records"].extend(result.pop("records"))
            manifest["rejected"].extend(result.pop("rejected"))
            manifest["clip_audits"].append(result)
            if count == 1 or count % 20 == 0 or count == len(jobs):
                save_json(incomplete, manifest)
                print(json.dumps({"phase": "preparing", "complete_clips": count, "total_clips": len(jobs),
                                  "hand_records": len(manifest["records"])}), flush=True)
    train = {entry["parent_recording"] for entry in manifest["records"] if entry["split"] == "train"}
    val = {entry["parent_recording"] for entry in manifest["records"] if entry["split"] == "val"}
    if train & val or (train | val) & set(selection["hawor_excluded_recordings"]):
        raise ValueError("Final HOT physical parent leakage/exclusion failure")
    if identity(prepared_path) != prepared_identity or identity(download_path) != download_identity:
        raise ValueError("Source manifests changed during HOT preparation")
    manifest["phase"], manifest["completed_utc"] = "complete", now()
    manifest["records"].sort(key=lambda record: (record["split"], record["recording"]))
    manifest["clip_audits"].sort(key=lambda record: record["source_path"])
    manifest["split_records"] = dict(Counter(record["split"] for record in manifest["records"]))
    manifest["split_frames"] = {split: sum(record["frames"] for record in manifest["records"] if record["split"] == split)
                                for split in ("train", "val")}
    manifest["split_hand_frames"] = {split: {side: sum(record["frames"] for record in manifest["records"]
        if record["split"] == split and record["handedness"] == hand)
        for hand, side in enumerate(("left", "right"))} for split in ("train", "val")}
    manifest["unique_rgb_images"] = sum(audit["unique_rgb_images"] for audit in manifest["clip_audits"])
    manifest["checks"] = {"original_motion_parent_split_preserved": True, "all_27_hawor_eval_parents_excluded": True,
                          "source_mano_states_redecoded_against_prepared": True,
                          "rgb_reference_nanosecond_timestamps_equal": True, "reference_j0_world_and_geometry_consistent": True,
                          "fisheye_pinhole_sampling_and_rotated_K_T_checks_passed": True}
    save_json(args.output / "source_manifest.json", manifest)
    incomplete.unlink()
    print(json.dumps({"phase": "complete", "manifest": str(args.output / "source_manifest.json"),
        "split_records": manifest["split_records"], "split_frames": manifest["split_frames"],
        "unique_rgb_images": manifest["unique_rgb_images"], "rejected_hand_clips": len(manifest["rejected"])}), flush=True)


if __name__ == "__main__":
    main()
