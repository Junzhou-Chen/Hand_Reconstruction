#!/usr/bin/env python3
"""Read-only remote audit of HOT JSON float64 vs toolkit float32 camera paths.

No RGB publication, neural-network forward, optimizer or source mutation.
Compare the original prepare script's transpose-vs-inverse check with exact
JSON-quaternion geometry and the proposed float32 export on in-frame points.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import tarfile

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from handloop.data.meta_mano import MANOReferenceDecoder, META_FINGERTIP_VERTICES
from prepare_hot3d_visual_sources import ORDER, STREAM, select_clips, read_json, read_reference
from prepare_visual_joint_sources import identity


def stable_project(points, rotation, translation, focal, center):
    eye = (points.astype(np.float64) - translation) @ rotation
    return eye[..., :2] / eye[..., 2:] * focal + center, eye


def original_inverse_projection(points, transform, intrinsic):
    """Freeze the old 41bf numerical path even after the prep fix is merged."""
    inverse = np.linalg.inv(transform)
    eye = points.astype(np.float64) @ inverse[:3, :3].T + inverse[:3, 3]
    homogeneous = eye @ intrinsic.T
    return homogeneous[:, :2] / homogeneous[:, 2:], eye


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("hot-root", "prepared", "mano-models", "camera-toolkit", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=704)
    parser.add_argument("--parents-per-split", type=int, default=0)
    parser.add_argument("--clips-per-parent", type=int, default=1)
    args = parser.parse_args()
    if not Path("/sciclone/home/jchen57").is_dir():
        parser.error("Remote Pod only")
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0, str(args.camera_toolkit))
    from hand_tracking_toolkit import camera
    prepared_path = args.prepared / "manifest.json"
    original_identity = identity(prepared_path)
    prepared = json.loads(prepared_path.read_text())
    download = json.loads((args.hot_root / "download_manifest.json").read_text())
    jobs, selection = select_clips(prepared, download, args)
    decoder = MANOReferenceDecoder(args.mano_models, device="cpu", fix_left_shapedirs=True,
                                   asset_version=prepared["decoder_provenance"].get("asset_version"))
    if decoder.provenance() != prepared["decoder_provenance"]:
        raise ValueError("Source decoder changed")
    clockwise = np.asarray([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    metrics, top, counts = {}, [], Counter()
    def update(name, value):
        metrics[name] = max(metrics.get(name, 0.), float(value))
    for j, job in enumerate(jobs):
        with tarfile.open(args.hot_root / job["source_path"], "r:") as archive:
            metadata = {}
            for entry in job["entries"]:
                arrays, _ = read_reference(args.prepared, entry)
                ids = arrays["frame_ids"]
                side = "right" if entry["handedness"] else "left"
                theta, wrist = [], []
                for frame_id in ids:
                    frame_key = f"{int(frame_id):06d}"
                    if frame_key not in metadata:
                        raw = read_json(archive, frame_key + ".cameras.json")[STREAM]
                        hand = read_json(archive, frame_key + ".hands.json")
                        metadata[frame_key] = raw, hand
                    raw, hand = metadata[frame_key]
                    theta.append(hand[side]["mano_pose"]["thetas"])
                    wrist.append(hand[side]["mano_pose"]["wrist_xform"])
                joint_chunks = []
                for begin in range(0, len(ids), 64):
                    end = min(begin + 64, len(ids))
                    decoded = decoder.decode_meta(np.asarray(theta[begin:end]), np.asarray(wrist[begin:end]),
                        arrays["beta"], np.full(end - begin, entry["handedness"]))
                    all_joints = np.concatenate((decoded.joints_m.cpu().numpy(),
                        decoded.vertices_m[:, META_FINGERTIP_VERTICES].cpu().numpy()), axis=1)
                    joint_chunks.append(all_joints[:, ORDER])
                joints = np.concatenate(joint_chunks)
                for position, frame_id in enumerate(ids):
                    raw, _ = metadata[f"{int(frame_id):06d}"]
                    source = camera.from_json(raw)
                    scale = args.image_size / source.width
                    focal = np.asarray(source.f, dtype=np.float64) * scale
                    center = (np.asarray(source.c, dtype=np.float64) + .5) * scale - .5
                    pinhole = camera.PinholePlaneCameraModel(width=args.image_size, height=args.image_size,
                        f=focal, c=center, distort_coeffs=[], T_world_from_eye=source.T_world_from_eye)
                    old_transform = np.asarray(source.T_world_from_eye, dtype=np.float64).copy()
                    old_transform[:3, :3] = old_transform[:3, :3] @ clockwise.T
                    intrinsic = np.asarray([[focal[1], 0., args.image_size - 1 - center[1]],
                                            [0., focal[0], center[0]], [0., 0., 1.]])
                    old_projected, old_eye = original_inverse_projection(joints[position], old_transform, intrinsic)
                    official_uv = pinhole.world_to_window(joints[position].astype(np.float64))
                    old_rotated = np.stack((args.image_size - 1 - official_uv[:, 1], official_uv[:, 0]), -1)
                    published = np.ones(21, dtype=bool)
                    published[1] = False
                    positive = published & (old_eye[:, 2] > 1e-4)
                    if not positive[published].all():
                        counts["nonpositive_frame"] += 1
                        continue
                    old_error = np.max(np.abs(old_projected - old_rotated), axis=-1)
                    update("old_transpose_vs_inverse_max_px_positive", old_error[positive].max())
                    quaternion = np.asarray(raw["T_world_from_camera"]["quaternion_wxyz"], dtype=np.float64)
                    rotation = Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()
                    translation = np.asarray(raw["T_world_from_camera"]["translation_xyz"], dtype=np.float64)
                    expected, eye64 = stable_project(joints[position], rotation, translation, focal, center)
                    expected_rotated = np.stack((args.image_size - 1 - expected[:, 1], expected[:, 0]), -1)
                    transform64 = np.eye(4)
                    transform64[:3, :3], transform64[:3, 3] = rotation @ clockwise.T, translation
                    exact_projected, _ = original_inverse_projection(joints[position], transform64, intrinsic)
                    algebra = np.abs(exact_projected - expected_rotated).max(axis=-1)
                    update("json_float64_rotation_algebra_max_px_positive", algebra[positive].max())
                    # Mirror the project's float32 rigid inverse + translation
                    # multiplication, rather than a generic matrix inverse.
                    points32 = torch.from_numpy(joints[position]).float()
                    matrix32 = torch.from_numpy(transform64).float()
                    inverse_rotation32 = matrix32[:3, :3].T
                    eye32 = (points32 @ inverse_rotation32.T
                             - (inverse_rotation32 @ matrix32[:3, 3])[None]).numpy()
                    intrinsic32 = intrinsic.astype(np.float32)
                    projected32 = eye32 @ intrinsic32.T
                    projected32 = projected32[:, :2] / projected32[:, 2:]
                    quant_error = np.abs(projected32 - expected_rotated).max(axis=-1)
                    inside = positive & np.all((expected_rotated >= 0) & (expected_rotated <= args.image_size - 1), axis=-1)
                    update("export_float32_max_px_positive", quant_error[positive].max())
                    if inside.any():
                        update("export_float32_max_px_in_frame", quant_error[inside].max())
                        update("old_transpose_vs_inverse_max_px_in_frame", old_error[inside].max())
                    counts["positive_frames"] += 1
                    counts["published_positive_landmarks"] += int(positive.sum())
                    counts["in_frame_landmarks"] += int(inside.sum())
                    if old_error[positive].max() > .02:
                        counts["old_check_failing_frames"] += 1
                        slot = int(np.argmax(np.where(positive, old_error, -1)))
                        case = {"source": job["source_path"], "recording": entry["recording"], "side": side,
                            "frame_id": int(frame_id), "slot": slot, "old_error_px": float(old_error[slot]),
                            "old_projected_uv": old_projected[slot].tolist(), "old_depth_m": float(old_eye[slot, 2]),
                            "json64_projected_uv": expected_rotated[slot].tolist(), "json64_depth_m": float(eye64[slot, 2]),
                            "json64_algebra_max_px": float(algebra[positive].max()),
                            "float32_export_in_frame_max_px": float(quant_error[inside].max()) if inside.any() else None,
                            "float32_export_at_failing_slot_px": float(quant_error[slot]),
                            "failing_slot_in_frame": bool(inside[slot]),
                            "source_world_camera_t": translation.tolist(),
                            "R32_orthogonality_error": float(np.max(np.abs(source.T_world_from_eye[:3, :3].T @ source.T_world_from_eye[:3, :3] - np.eye(3))))}
                        top.append(case)
                        top.sort(key=lambda value: value["old_error_px"], reverse=True)
                        top = top[:20]
        if (j + 1) % 10 == 0:
            print(json.dumps({"audited_clips": j + 1, "total_clips": len(jobs), "metrics": metrics, "counts": dict(counts)}), flush=True)
    if identity(prepared_path) != original_identity:
        raise ValueError("Source reference manifest changed")
    result = {"phase": "complete", "selection": selection, "counts": dict(counts), "metrics": metrics,
              "worst_old_check_cases": top, "reference_manifest": original_identity,
              "neural_network_or_optimizer_used": False, "reference_mano_decoder_used": True,
              "rgb_publication": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
