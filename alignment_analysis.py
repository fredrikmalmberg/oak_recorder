"""Orchestration + analysis for reviewing a take's temporal alignment and
extrinsics stability, sitting on top of align_session.py (temporal sync)
and calibrate.py (ChArUco board detection/pose solving). Deliberately
headless -- no viser import, no GUI logic -- so this is importable both
from a standalone CLI viewer (view_take_alignment.py) and, later, from a
postprocess panel inside app.py without pulling in a second GUI concern
at import time. Every function here takes a take_dir (or already-loaded
data) and returns plain dicts/arrays; all widget/chart code lives in the
viewer script.
"""
import json
import math
import os

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

import align_session
import calibrate


def ensure_alignment_report(take_dir, force=False, **align_kwargs):
    """Run align_session.align_session(take_dir) if alignment_report.json
    doesn't exist yet (or force=True), otherwise just load the existing
    one. align_kwargs are forwarded to align_session() (align_threshold_ms,
    fps, align_host_only, ...).
    """
    report_path = os.path.join(take_dir, "alignment_report.json")
    if not force and os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return align_session.align_session(take_dir, **align_kwargs)


def ensure_aligned_grid_video(take_dir, report, force=False):
    """Build (or reuse) the ALIGNED grid video -- align_session.grid_mp4_path
    called with the take_dir itself lands at <take_dir>/compressed_video_grid.mp4,
    distinct from app.py's postprocess step, which writes the RAW (unaligned,
    index-tiled) grid to <take_dir>/processed/compressed_video_grid.mp4. Callers
    must use THIS function's return value, not app.py's path, to show alignment
    actually working.
    """
    output_path = align_session.grid_mp4_path(take_dir)
    if not force and os.path.exists(output_path):
        return output_path
    cam_labels = sorted(report["cameras"].keys())
    return align_session.convert_aligned_jpegs_to_grid_mp4(take_dir, cam_labels, report["fps"])


def match_calibration_to_take(calibration_path, take_dir, cfg=None):
    """Match this take's cameras to calibration_path's cameras by device_id
    -- never by camN label, which is independently (and, confirmed, often
    inconsistently) assigned per calibration run and per take. A take
    camera only counts as "matched" if its device has at least
    cfg["intrinsics"]["min_samples_before_calibrate"] real intrinsic
    samples in the calibration file -- calibrate.load_calibration_output
    alone isn't enough to detect this: it only drops cameras with null
    extrinsics, but a camera with zero real intrinsic samples still gets
    a (garbage, naive_chain) extrinsics entry written, so it would
    otherwise silently pass through with meaningless K/dist.

    Returns (matched: {cam_label: calib_entry}, skipped: {cam_label: reason}).
    """
    cfg = cfg or calibrate.load_config(calibrate.CONFIG_PATH_DEFAULT)
    min_samples = cfg["intrinsics"]["min_samples_before_calibrate"]

    with open(os.path.join(take_dir, "take_meta.json"), "r", encoding="utf-8") as f:
        take_meta = json.load(f)
    take_device_ids = {
        cam_label: info["device_id"] for cam_label, info in take_meta["cameras"].items()
    }

    with open(calibration_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    num_samples_by_device = {
        entry.get("device_id"): entry["intrinsics"].get("num_samples", 0)
        for entry in raw["cameras"].values()
    }

    calib_entries = calibrate.load_calibration_output(calibration_path)
    calib_by_device = {e["device_id"]: e for e in calib_entries.values() if e.get("device_id")}

    matched, skipped = {}, {}
    for cam_label, device_id in take_device_ids.items():
        entry = calib_by_device.get(device_id)
        num_samples = num_samples_by_device.get(device_id, 0)
        if entry is None:
            skipped[cam_label] = f"device {device_id} not present (or unposed) in {calibration_path}"
        elif num_samples < min_samples:
            skipped[cam_label] = (
                f"device {device_id} has only {num_samples} intrinsic sample(s) "
                f"(< {min_samples} required) in {calibration_path} -- intrinsics not trustworthy"
            )
        else:
            matched[cam_label] = entry
    return matched, skipped


def compute_board_consistency(take_dir, calibration_path, *, board_section="alignment_board",
                               target_hz=5.0, stride_slots=None, cfg=None, report=None):
    """Checks whether the stationary alignment board's pose, transformed
    into calibration_path's common ("world") frame via each camera's
    saved extrinsics, agrees across cameras (cross_camera_consistency) and
    stays constant over the take for each camera individually
    (per_camera_self_consistency) -- since the board is meant to stay
    physically stationary, drift in either signals poor/drifting
    extrinsics rather than a real board movement. Reuses
    align_session()'s already-solved cross-camera frame sync (reads
    aligned/<cam>/<slot>.jpg) instead of a second nearest-neighbor pass.

    Only cameras matched by match_calibration_to_take produce data; with
    most calibration files today that's at most 2 of 6 cameras (see
    match_calibration_to_take's docstring), so cross_camera_consistency
    may end up empty -- that's an expected, reported condition, not a bug.
    """
    cfg = cfg or calibrate.load_config(calibrate.CONFIG_PATH_DEFAULT)
    if report is None:
        report = ensure_alignment_report(take_dir)

    matched, skipped = match_calibration_to_take(calibration_path, take_dir, cfg=cfg)

    warnings = []
    if not matched:
        warnings.append("No camera has usable intrinsics for this calibration file -- nothing to check.")

    board, detector, board_points_3d = calibrate.build_board(cfg, section=board_section)
    min_corners = cfg["quality_gates"]["min_corners"]

    fps = report["fps"]
    num_slots = report["aligned_frame_count"]
    stride = stride_slots or max(1, round(fps / target_hz))
    sample_slots = list(range(0, num_slots, stride))

    aligned_dir = os.path.join(take_dir, "aligned")

    # slot -> {cam_label: (t_world[3], rotvec_world[3], reproj_err_px)}
    detections_by_slot = {}
    for cam_label, entry in matched.items():
        slot_offsets = report["cameras"][cam_label]["slot_offsets_ms"]
        T_cam_to_world = calibrate.invert_T(calibrate.rt_to_T(entry["R"], entry["t"]))
        for slot in sample_slots:
            if slot_offsets[slot] is None:
                continue  # blue placeholder -- no real frame at this slot for this camera
            img_path = os.path.join(aligned_dir, cam_label, f"{slot:06d}.jpg")
            img = cv2.imread(img_path)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            detection = calibrate.detect_charuco(detector, gray)
            if detection is None:
                continue
            corners2d, ids = detection
            if len(ids) < min_corners:
                continue
            pose = calibrate.solve_board_pose(board_points_3d[ids], corners2d, entry["K"], entry["dist"])
            if pose is None:
                continue
            R_bc, t_bc, reproj_err_px = pose
            T_board_to_world = T_cam_to_world @ calibrate.rt_to_T(R_bc, t_bc)
            t_world = T_board_to_world[:3, 3]
            rotvec_world = Rotation.from_matrix(T_board_to_world[:3, :3]).as_rotvec()
            detections_by_slot.setdefault(slot, {})[cam_label] = (t_world, rotvec_world, reproj_err_px)

    per_camera_self_consistency = {}
    for cam_label in matched:
        samples = []
        for slot, per_cam in sorted(detections_by_slot.items()):
            if cam_label in per_cam:
                t_world, rotvec_world, reproj_err_px = per_cam[cam_label]
                samples.append({
                    "slot": slot, "t_world": t_world.tolist(),
                    "rotvec_world": rotvec_world.tolist(), "reproj_err_px": reproj_err_px,
                })
        if samples:
            t_arr = np.array([s["t_world"] for s in samples])
            rv_arr = np.array([s["rotvec_world"] for s in samples])
            translation_stddev_m = float(np.linalg.norm(t_arr.std(axis=0)))
            rotation_stddev_deg = float(np.degrees(np.linalg.norm(rv_arr.std(axis=0))))
        else:
            translation_stddev_m = rotation_stddev_deg = None
        per_camera_self_consistency[cam_label] = {
            "num_detections": len(samples),
            "translation_stddev_m": translation_stddev_m,
            "rotation_stddev_deg": rotation_stddev_deg,
            "samples": samples,
        }

    # Cross-camera pairwise disagreement -- same rotation-angle convention
    # as calibrate.py's own PoseGraph._chain_consistency_error_deg.
    cross_samples = []
    for slot, per_cam in sorted(detections_by_slot.items()):
        cams = sorted(per_cam.keys())
        if len(cams) < 2:
            continue
        pair_trans, pair_rot = [], []
        for i, cam_a in enumerate(cams):
            for cam_b in cams[i + 1:]:
                t_a, rv_a, _ = per_cam[cam_a]
                t_b, rv_b, _ = per_cam[cam_b]
                pair_trans.append(float(np.linalg.norm(t_a - t_b)))
                R_a = Rotation.from_rotvec(rv_a).as_matrix()
                R_b = Rotation.from_rotvec(rv_b).as_matrix()
                pair_rot.append(math.degrees(Rotation.from_matrix(R_a.T @ R_b).magnitude()))
        cross_samples.append({
            "slot": slot, "cameras": cams,
            "mean_translation_disagreement_m": float(np.mean(pair_trans)),
            "max_translation_disagreement_m": float(np.max(pair_trans)),
            "mean_rotation_disagreement_deg": float(np.mean(pair_rot)),
            "max_rotation_disagreement_deg": float(np.max(pair_rot)),
        })

    if cross_samples:
        overall_mean_t = float(np.mean([s["mean_translation_disagreement_m"] for s in cross_samples]))
        overall_max_t = float(np.max([s["max_translation_disagreement_m"] for s in cross_samples]))
    else:
        overall_mean_t = overall_max_t = None

    return {
        "calibration_path": calibration_path,
        "board_section": board_section,
        "cameras_used": {cam: entry["device_id"] for cam, entry in matched.items()},
        "cameras_skipped": skipped,
        "stride_slots": stride,
        "num_slots_sampled": len(sample_slots),
        "per_camera_self_consistency": per_camera_self_consistency,
        "cross_camera_consistency": {
            "samples": cross_samples,
            "overall_mean_translation_disagreement_m": overall_mean_t,
            "overall_max_translation_disagreement_m": overall_max_t,
        },
        "warnings": warnings,
    }


def save_board_consistency_report(take_dir, report):
    path = os.path.join(take_dir, "board_consistency_report.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return path
