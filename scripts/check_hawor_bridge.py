#!/usr/bin/env python3
"""Remote-only numerical acceptance of licensed MANO codec and official bridge.

This checks implementation equivalence, not reconstruction accuracy. With a
checkpoint it uses random normalized crops to compare the frozen bridge with
official forward_step; no training, infiller, SLAM or dataset metric is run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from handloop.geometry import axis_angle_to_matrix, state_apply_delta, state_from_rotmat
from handloop.visual.codec import (MANOStateCodec, invert_transform,
    mirror_rotations_x, project_world_points, transform_state)
from handloop.visual.hawor import FrozenHaWoRBridge


def maximum(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a-b).abs().max().detach().cpu())


def require_close(name: str, value: float, tolerance: float = 2e-5) -> None:
    if not value <= tolerance:
        raise AssertionError(f"{name}: max error {value} > {tolerance}")


def check_codec(mano_model_dir: Path, device: str) -> dict:
    torch.manual_seed(1327)
    codec = MANOStateCodec(mano_model_dir / "MANO_RIGHT.pkl",
                            mano_model_dir / "MANO_LEFT.pkl").to(device)
    n = 8
    beta = torch.randn(n, 10, device=device) * 0.3
    rotations = axis_angle_to_matrix(torch.randn(n, 16, 3, device=device) * 0.35)
    sides = torch.arange(n, device=device) % 2 == 1
    translation = torch.randn(n, 3, device=device) * 0.1
    reference_vertices = torch.zeros(n, 778, 3, device=device)
    reference_joints = torch.zeros(n, 16, 3, device=device)
    for side, right in (("left", False), ("right", True)):
        selected = sides == right
        output = codec.layers[side](betas=beta[selected],
            global_orient=rotations[selected, :1], hand_pose=rotations[selected, 1:],
            transl=translation[selected], pose2rot=False)
        reference_vertices[selected] = output.vertices
        reference_joints[selected] = output.joints[:, :16]
    wrist = translation + codec.rest_wrist(beta, sides)
    state = state_from_rotmat(wrist, rotations)
    geometry = codec.geometry(state, beta, sides)
    errors = {"shaped_root_vs_MANOLayer_J0_m": maximum(wrist, reference_joints[:, 0]),
              "state_to_MANO_vertices_m": maximum(geometry["vertices"], reference_vertices),
              "state_to_MANO_joints_m": maximum(geometry["joints"], reference_joints),
              "MANO_translation_roundtrip_m": maximum(codec.to_mano(state, beta, sides)["transl"], translation)}
    transform = torch.eye(4, device=device).repeat(n, 1, 1)
    transform[:, :3, :3] = axis_angle_to_matrix(torch.randn(n, 3, device=device) * 0.5)
    transform[:, :3, 3] = torch.randn(n, 3, device=device)
    world = transform_state(state, transform)
    errors["SE3_state_roundtrip"] = maximum(transform_state(world, invert_transform(transform)), state)
    camera_positions = reference_joints
    expected_world = (transform[:, None, :3, :3] @ camera_positions[..., None]).squeeze(-1) + transform[:, None, :3, 3]
    errors["SE3_FK_commutation_m"] = maximum(codec.geometry(world, beta, sides)["joints"], expected_world)
    unflipped = mirror_rotations_x(rotations, sides)
    errors["left_SRS_involution"] = maximum(mirror_rotations_x(unflipped, sides), rotations)
    # Equivalent official axis-angle y/z reflection, checked independently of
    # codec's SRS implementation. The chosen logs stay away from the pi branch.
    from handloop.geometry import matrix_to_axis_angle
    aa = matrix_to_axis_angle(rotations)
    signs = torch.ones_like(aa)
    signs[~sides, :, 1:] = -1
    errors["left_SRS_vs_official_axis_angle"] = maximum(unflipped, axis_angle_to_matrix(aa * signs))
    for key, error in errors.items():
        require_close(key, error)
    # Candidate geometry and projection must remain differentiable even though
    # the MANO parameters are frozen. Avoid abs/norm at exact-zero residuals.
    base = state.clone()
    base[:, 2] += 1.5
    delta = torch.zeros(n, 51, device=device, requires_grad=True)
    candidate = state_apply_delta(base, delta)
    vertices = codec.geometry(candidate, beta, sides)["vertices"]
    intrinsic = torch.eye(3, device=device).repeat(n, 1, 1)
    intrinsic[:, 0, 0] = intrinsic[:, 1, 1] = 600
    intrinsic[:, 0, 2], intrinsic[:, 1, 2] = 320, 240
    identity = torch.eye(4, device=device).repeat(n, 1, 1)
    pixels, depth = project_world_points(vertices[:, :12], identity, intrinsic)
    (pixels.square().mean() + vertices.square().mean()).backward()
    gradient = delta.grad
    if gradient is None or not torch.isfinite(gradient).all() or not (gradient.abs().max() > 0):
        raise AssertionError("Frozen MANO candidate input gradient is absent/nonfinite/zero")
    return {"passed": True, "geometry_max_abs_errors": errors,
            "candidate_gradient_max_abs": float(gradient.abs().max()),
            "candidate_all_points_positive_depth": bool(depth.all()),
            "codec": codec.provenance()}


def check_official(bridge: FrozenHaWoRBridge, device: str) -> dict:
    model = bridge.model
    image = torch.randn(1, 16, 3, 256, 256, device=device) * 0.2
    results = {}
    for name, right, principal_x in (("right_central", True, 320.),
                                      ("left_central", False, 320.),
                                      ("left_noncentral", False, 331.)):
        original_center = torch.tensor([298., 215.], device=device).expand(1, 16, 2).clone()
        crop_center = original_center.clone()
        if not right:
            crop_center[..., 0] = 640 - crop_center[..., 0] - 1
        batch = {"img": image, "center": crop_center,
            "center_unflipped": original_center, "scale": torch.full((1, 16), 0.9, device=device),
            "img_focal": torch.full((1, 16), 600., device=device),
            "img_center": torch.tensor([principal_x, 240.], device=device).expand(1, 16, 2).clone(),
            "camera_to_world": torch.eye(4, device=device).expand(1, 16, 4, 4).clone(),
            "is_right": torch.full((1, 16), right, device=device, dtype=torch.bool),
            **{key: torch.ones(1, 16, dtype=torch.bool, device=device)
               for key in ("detected", "bbox_valid", "frame_valid", "image_valid")}}
        capture = {}
        handle = model.mano_head.register_forward_pre_hook(lambda _, args: capture.update(feature=args[0].detach().clone()))
        official_batch = {key: value.clone() for key, value in batch.items()}
        if not right:
            official_batch["do_flip"] = torch.ones(1, 16, device=device)
        with torch.no_grad():
            official = model.forward_step(official_batch)["out"]
        handle.remove()
        expected_feature = capture["feature"].flatten(2).transpose(1, 2).reshape(1, 16, 192, 1280)
        with torch.no_grad():
            actual = bridge(batch)
        errors = {"IAM_patch_max_abs": maximum(actual["image_features"], expected_feature),
                  "post_PAM_pose_max_abs": maximum(actual["pred_pose"], official["pred_pose"].reshape(1, 16, 96)),
                  "official_HMR2_rotation_matrix_max_abs": maximum(actual["pred_rotmat_mirrored"], official["pred_rotmat"].reshape(1, 16, 16, 3, 3)),
                  "beta_max_abs": maximum(actual["beta"], official["pred_shape"].reshape(1, 16, 10)),
                  "camera_max_abs": maximum(actual["pred_cam"], official["pred_cam"].reshape(1, 16, 3))}
        for key, error in errors.items():
            require_close(name+"_"+key, error, 2e-4)
        if right:
            with torch.no_grad():
                expected_wrist = model.mano.query(official).joints[:, 0] + official["trans_full"][:, 0]
            errors["wrist_J0_vs_official_MANO_m"] = maximum(actual["raw_state_world"][0, :, :3], expected_wrist)
            require_close("official_right_anatomical_wrist", errors["wrist_J0_vs_official_MANO_m"], 2e-5)
        translation_error = maximum(actual["mano_translation_camera"], official["trans_full"].reshape(1, 16, 3))
        if name != "left_noncentral":
            require_close(name+"_translation", translation_error)
        else:
            # Expected delta from restoring center using W rather than 2cx.
            expected_delta = 2 * (640 - 2 * principal_x) / (0.9 * 200 * actual["pred_cam"][..., 0])
            measured_delta = actual["mano_translation_camera"][..., 0] - official["trans_full"].reshape(1, 16, 3)[..., 0]
            require_close("noncentral_left_analytic_delta", maximum(measured_delta, expected_delta), 2e-5)
        results[name] = {**errors, "translation_difference_m": translation_error,
                         "noncentral_left_geometry_fix": name == "left_noncentral"}
    return {"passed": True, "cases": results, "feature_shape": [1, 16, 192, 1280],
            "interpretation": "Official forward equivalence only; random crops are not an accuracy benchmark",
            "provenance": bridge.provenance()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mano-model-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--config-path", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    report = {"torch": torch.__version__, "device": args.device, "passed": False}
    try:
        report["codec"] = check_codec(args.mano_model_dir, args.device)
        if args.checkpoint_path is not None:
            if args.repo_root is None or args.config_path is None:
                parser.error("checkpoint parity also requires --repo-root and --config-path")
            bridge = FrozenHaWoRBridge.from_pretrained(args.repo_root, args.checkpoint_path,
                args.config_path, args.mano_model_dir, device=args.device)
            report["official_bridge"] = check_official(bridge, args.device)
        report["passed"] = True
    except Exception as exc:
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n")
    print(json.dumps({"passed": True, "report": str(args.report),
                      "official_bridge_checked": "official_bridge" in report}))


if __name__ == "__main__":
    main()
