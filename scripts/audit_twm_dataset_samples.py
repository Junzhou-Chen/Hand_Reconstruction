#!/usr/bin/env python3
"""Remote-only bounded annotation audit; no images, model assets or training.

This checks sampled fields/finite values, not FK equivalence or label accuracy.
"""
import argparse
from collections import Counter
import importlib
import io
import json
from pathlib import Path
import pickle
import tarfile

import numpy as np
import torch

from download_twm_data import now, save_json


class AnnotationUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ("torch.storage", "_load_from_bytes"):
            return lambda value: torch.load(io.BytesIO(value), map_location="cpu", weights_only=True)
        if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
            return torch._utils._rebuild_tensor_v2
        allowed = {
            ("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
            ("numpy", "ndarray"), ("numpy", "dtype"),
            ("numpy.core.numeric", "_frombuffer"), ("numpy._core.numeric", "_frombuffer"),
            ("collections", "OrderedDict"),
        }
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f"Unsupported annotation class {module}.{name}")
        return getattr(importlib.import_module(module), name)


def array(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def array_summary(value):
    x = array(value)
    return {"shape": list(x.shape), "dtype": str(x.dtype), "finite": bool(np.isfinite(x).all())}


def oak_samples(root):
    names = ["anno_preview/" + s.replace("/", "++") + ".pkl"
             for s in (root / "splits/train.txt").read_text().splitlines() if s]
    result = []
    for index in sorted({0, len(names) // 2, len(names) - 1}):
        path = root / names[index]
        with path.open("rb") as stream:
            value = AnnotationUnpickler(stream).load()
        motion_ids = np.asarray(value["mocap_frame_id_list"], dtype=np.int64)
        video_ids = np.asarray(value["frame_id_list"], dtype=np.int64)
        records = value["raw_mano"]
        frames = [records[k] for k in sorted(records, key=int)[:20]]
        fields = sorted(frames[0])
        finite = all(np.isfinite(array(v)).all() for frame in frames for v in frame.values())
        quat_errors = []
        for frame in frames:
            for key, item in frame.items():
                if key.endswith("pose_coeffs"):
                    quat_errors.extend(np.abs(np.linalg.norm(array(item), axis=-1) - 1).ravel().tolist())
        result.append({"file": names[index], "train_record_index": index,
                       "mocap_records": len(records), "mocap_id_count": len(motion_ids),
                       "video_id_count": len(video_ids),
                       "mocap_id_steps": dict(Counter(np.diff(motion_ids).tolist())),
                       "video_id_steps": dict(Counter(np.diff(video_ids).tolist())),
                       "sampled_mano_frames": len(frames), "all_sampled_values_finite": bool(finite),
                       "quat_max_norm_error": max(quat_errors),
                       "first_frame_fields": {k: array_summary(frames[0][k]) for k in fields}})
    return result


def meta_tar_samples(root, dataset):
    manifest = json.loads((root / "download_manifest.json").read_text())
    files = [x["path"] for x in manifest["files"] if x["path"].endswith(".tar")]
    selected = []
    groups = ("train_aria/", "train_quest3/") if dataset == "hot3d" else ("separate_hand", "hand_hand")
    for group in groups:
        subset = [p for p in files if group in p]
        if subset:
            selected.extend([subset[0], subset[-1]])
    output = []
    for name in dict.fromkeys(selected):
        with tarfile.open(root / name, "r:") as archive:
            members = archive.getnames()
            hand_names = sorted(n for n in members if n.endswith(".hands.json"))
            dimensions, present, finite, quat_errors = Counter(), Counter(), True, []
            for label_name in hand_names:
                hand = json.load(archive.extractfile(label_name))
                for side in ("left", "right"):
                    label = hand.get(side)
                    if not label:
                        present[f"{side}_absent"] += 1
                        continue
                    present[f"{side}_native"] += int("umetrack_pose" in label)
                    present[f"{side}_mano"] += int("mano_pose" in label)
                    if "mano_pose" in label:
                        pose = label["mano_pose"]
                        dimensions[f"mano_theta_{len(pose['thetas'])}"] += 1
                        dimensions[f"mano_xform_{len(pose['wrist_xform'])}"] += 1
                        finite &= bool(np.isfinite(pose["thetas"]).all() and np.isfinite(pose["wrist_xform"]).all())
                    if "umetrack_pose" in label:
                        native = label["umetrack_pose"]
                        transform = native["T_world_from_wrist"]
                        dimensions[f"native_angles_{len(native['joint_angles'])}"] += 1
                        finite &= bool(np.isfinite(native["joint_angles"]).all()
                                       and np.isfinite(transform["translation_xyz"]).all())
                        q = np.asarray(transform["quaternion_wxyz"])
                        finite &= bool(np.isfinite(q).all())
                        quat_errors.append(float(abs(np.linalg.norm(q) - 1)))
            shape_name = next(n for n in members if "hand_shapes" in n)
            shape = json.load(archive.extractfile(shape_name))
            infos = sorted(n for n in members if n.endswith(".info.json"))
            timestamps = [json.load(archive.extractfile(n))["ref_timestamp_ns"] for n in infos]
            ids = [int(n.split(".", 1)[0]) for n in hand_names]
            output.append({"file": name, "frames": len(hand_names), "frame_ids_consecutive": bool(np.all(np.diff(ids) == 1)),
                           "fields": dict(dimensions), "availability": dict(present),
                           "all_pose_values_finite": bool(finite), "quat_max_norm_error": max(quat_errors) if quat_errors else None,
                           "beta_length": len(shape["mano"]), "has_native_profile": "umetrack" in shape,
                           "has_explicit_ref_timestamps": bool(timestamps),
                           "dt_ns_minmax": [int(min(np.diff(timestamps))), int(max(np.diff(timestamps)))] if len(timestamps) > 1 else None})
    return output


def dex_samples(root):
    path = root / "archives/20200709-subject-01.tar.gz"
    output, metadata = [], None
    with tarfile.open(path, "r|gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            if member.name.endswith("meta.yml") and metadata is None:
                metadata = {"path": member.name, "yaml": archive.extractfile(member).read().decode()}
            if member.name.endswith(".npz") and "labels_" in member.name:
                with np.load(io.BytesIO(archive.extractfile(member).read()), allow_pickle=False) as label:
                    output.append({"path": member.name, "arrays": {k: array_summary(label[k]) for k in ("pose_m", "joint_3d", "joint_2d")},
                                   "pose_all_zero": bool(np.all(label["pose_m"] == 0)),
                                   "joint_invalid_entries": int(np.count_nonzero(label["joint_3d"] == -1))})
                if len(output) == 6:
                    break
    calib = root / "archives/calibration.tar.gz"
    with tarfile.open(calib, "r:gz") as archive:
        name = next(n for n in archive.getnames() if n.endswith("mano.yml") and "subject-01_right" in n)
        text = archive.extractfile(name).read().decode()
    return {"subject_archive": path.name, "sequence_meta": metadata, "label_samples": output,
            "shape_calibration_member": name, "shape_calibration_yaml": text,
            "tar_order_is_time_order": False, "note": "Sort labels by numeric frame ID; do not treat TAR iteration order as motion time."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not Path("/sciclone/home/jchen57").is_dir():
        parser.error("Run this annotation audit on the remote ds Pod")
    report = {"utc": now(), "scope": "3 Oak train records (20 frames each), 4 HOT clips, 4 UmeTrack real sequences, 6 Dex labels + calibration; field checks only",
              "oakink2": oak_samples(args.root / "oakink2"),
              "hot3d": meta_tar_samples(args.root / "hot3d", "hot3d"),
              "umetrack": meta_tar_samples(args.root / "umetrack", "umetrack"),
              "dexycb": dex_samples(args.root / "dexycb")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
