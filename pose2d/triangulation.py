"""3D triangulation from pose2d's extraction+tracking output (both hands +
body), reusing hand_pose.hand_multiview's RANSAC/DLT triangulation and
Kalman/RTS smoothing machinery -- the same primitives hand_pose/session_
triangulation.py already validated against this rig, just fed by pose2d's
richer (two-hand + body + occlusion/swap/body-trust-flagged) landmarks
instead of the older single-hand pipeline's.

Triangulates left hand, right hand, and body each as an independent
RANSAC-selected sequence -- triangulate_sequence_ransac doesn't care about
semantic meaning, only landmark_ids + landmarks/confidence/projection
matrices -- rather than one combined skeleton: a signer's left hand, right
hand, and torso don't need to share one per-frame camera-set decision, and
letting each part choose its own tends to use more of the rig's coverage
than forcing one shared set.

Usage:
    python -m pose2d.triangulation <take_dir> [--calib PATH] [--force]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import calibrate  # noqa: E402

from hand_pose import hand_multiview as hmv  # noqa: E402

LANDMARK_IDS_HAND = list(range(21))
REFERENCE_LANDMARK_ID_HAND = 0  # wrist

LANDMARK_IDS_BODY = list(range(13))  # COCO-17's upper body: nose/eyes/ears/shoulders/elbows/wrists/hips.
# Only knees/ankles (ids 13-16) are excluded -- hips (11-12) are kept as
# part of the upper body/torso. Legs are irrelevant for sign-language
# content and a source of unreliable detections (occluded by a desk,
# cropped out of frame) that add nothing but noise here. Confirmed via
# experiment that this doesn't change RANSAC camera selection for a take
# already dominated by good shoulder visibility (see extraction.py's
# confidence_body, which the same experiment showed excludes legs from its
# gating mean too), but there's no upside to triangulating/reporting them
# either, and it removes a class of bad data by construction rather than
# relying on that experiment's result holding for every future take.
REFERENCE_LANDMARK_ID_BODY = 5  # left_shoulder -- more consistently in-frame than hips for a torso-framed signer

DEFAULT_CALIB_PATH = os.path.join("output", "calibration", "20260903_153052_7cam.json")


def _try_load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_pose2d_take_data(take_dir, pose2d_dir="pose2d"):
    """pose2d_dir selects which detector's output to read: "pose2d"
    (MediaPipe, pose2d.run_pipeline's default) or "pose2d_dwpose" (DWPose,
    pose2d.run_pipeline_dwpose) -- same landmark/confidence schema either
    way, so nothing else here needs to know which detector produced it.
    """
    base = os.path.join(take_dir, "aligned", pose2d_dir)
    with open(os.path.join(base, "landmarks_left.json")) as f:
        landmarks_left = json.load(f)
    with open(os.path.join(base, "landmarks_right.json")) as f:
        landmarks_right = json.load(f)
    with open(os.path.join(base, "landmarks_body.json")) as f:
        landmarks_body = json.load(f)
    with open(os.path.join(base, "confidence_left.json")) as f:
        confidence_left = json.load(f)
    with open(os.path.join(base, "confidence_right.json")) as f:
        confidence_right = json.load(f)
    with open(os.path.join(base, "tracking_diagnostics.json")) as f:
        diagnostics = json.load(f)
    # confidence_body was computed by pose2d.tracking.run_tracking but
    # pose2d.run_pipeline never wrote it out on its own -- read it back from
    # the extraction cache instead of re-running the pipeline just for this.
    with open(os.path.join(base, "extraction.json")) as f:
        confidence_body = json.load(f)["confidence_body"]
    # Per-landmark confidence (e.g. DWPose's per-keypoint SimCC scores) --
    # optional, None when the detector/part doesn't have it (MediaPipe
    # Hands only ever gives one whole-hand score).
    landmark_confidence_left = _try_load(os.path.join(base, "landmark_confidence_left.json"))
    landmark_confidence_right = _try_load(os.path.join(base, "landmark_confidence_right.json"))
    landmark_confidence_body = _try_load(os.path.join(base, "landmark_confidence_body.json"))
    return {
        "landmarks_left": landmarks_left, "confidence_left": confidence_left,
        "landmarks_right": landmarks_right, "confidence_right": confidence_right,
        "landmarks_body": landmarks_body, "confidence_body": confidence_body,
        "landmark_confidence_left": landmark_confidence_left,
        "landmark_confidence_right": landmark_confidence_right,
        "landmark_confidence_body": landmark_confidence_body,
        "cam_ids": diagnostics["cam_ids"],
    }


def build_projection_matrices(calib, cam_ids):
    return {cam_id: hmv.build_projection_matrix(calib[cam_id]["K"], calib[cam_id]["R"], calib[cam_id]["t"])
            for cam_id in cam_ids}


def triangulate_part(
    landmarks, confidence, projection_matrices, width, height,
    landmark_ids, reference_landmark_id, reproj_thresh_px=15.0, min_confidence=0.5,
    landmark_confidence=None,
):
    """Runs the RANSAC-select -> triangulate -> smooth -> jitter-flag chain
    for one part (a hand or the body), independent of the other parts.

    landmark_confidence (optional): dict[cam_id][frame_key][lm_id] -> float,
    e.g. DWPose's per-keypoint SimCC scores. When given, triangulate_
    sequence_ransac's weighted DLT solve favors higher-confidence cameras'
    observations of each INDIVIDUAL landmark within the already-
    (geometrically) chosen inlier set -- the camera SET itself is still
    chosen purely by ransac_select_cameras_sequence's reprojection-error
    test, using `confidence` (the whole-part score) only for gating which
    cameras are even eligible. None (the default -- what MediaPipe Hands'
    single whole-hand score leaves you with) falls back to uniform
    weighting among inliers, same as before this parameter existed.

    Returns (raw_reconstruction, smoothed_reconstruction, selection,
    reproj_errors, jitter_flagged_frame_keys).
    """
    selection = hmv.ransac_select_cameras_sequence(
        landmarks, confidence, projection_matrices, width, height,
        reference_landmark_id=reference_landmark_id, reproj_thresh_px=reproj_thresh_px,
        min_confidence=min_confidence,
    )
    raw_reconstruction = hmv.triangulate_sequence_ransac(
        landmarks, projection_matrices, width, height, landmark_ids, selection,
        landmark_confidence=landmark_confidence,
    )
    frame_keys = sorted(selection.keys(), key=lambda k: int(os.path.splitext(k)[0]))
    n_inliers_by_frame = {f: sel["n_inliers"] for f, sel in selection.items()}
    smoothed_reconstruction = hmv.smooth_reconstruction_sequence(
        raw_reconstruction, frame_keys, landmark_ids, n_inliers_by_frame,
    )
    jitter_flags, _jitter_residuals = hmv.flag_jitter_frames(
        raw_reconstruction, smoothed_reconstruction, frame_keys, reference_landmark_id=reference_landmark_id,
    )
    reproj_errors = hmv.ransac_per_camera_reproj_error(
        landmarks, confidence, raw_reconstruction, projection_matrices, width, height,
        reference_landmark_id=reference_landmark_id, min_confidence=min_confidence,
    )
    return raw_reconstruction, smoothed_reconstruction, selection, reproj_errors, jitter_flags


def _frame_order(selection_or_frames):
    return sorted(selection_or_frames, key=lambda k: int(os.path.splitext(k)[0]))


def select_stable_camera_sets(
    landmarks, confidence, projection_matrices, width, height, reference_landmark_id,
    reproj_thresh_px=15.0, min_confidence=0.5, window=45, enter_px=10.0, leave_px=20.0,
):
    """Slowly-changing camera set. Per-frame RANSAC (ransac_select_cameras_
    sequence) gives a reference-landmark 3D point each frame; a camera's
    reprojection error against it, median-filtered over +-`window` frames,
    decides whether the camera is "in" the trusted set. Hysteresis: a camera
    enters below `enter_px` and only leaves above `leave_px`, so lateral
    cameras that wobble around the threshold don't toggle frame to frame.

    Returns (frame_keys, in_set) with in_set[cam_id] a bool array over
    frame_keys, plus the per-frame RANSAC selection (used as fallback).
    """
    ransac_sel = hmv.ransac_select_cameras_sequence(
        landmarks, confidence, projection_matrices, width, height,
        reference_landmark_id=reference_landmark_id, reproj_thresh_px=reproj_thresh_px,
        min_confidence=min_confidence,
    )
    ref_recon = hmv.triangulate_sequence_ransac(
        landmarks, projection_matrices, width, height, [reference_landmark_id], ransac_sel,
    )
    errs = hmv.ransac_per_camera_reproj_error(
        landmarks, confidence, ref_recon, projection_matrices, width, height,
        reference_landmark_id=reference_landmark_id, min_confidence=min_confidence,
    )
    frame_keys = _frame_order(ransac_sel.keys())
    T = len(frame_keys)
    in_set = {}
    for cam_id in projection_matrices:
        e = np.array([errs.get(cam_id, {}).get(f, np.nan) for f in frame_keys])
        state, flags = None, np.zeros(T, dtype=bool)
        for t in range(T):
            w = e[max(0, t - window): t + window + 1]
            w = w[~np.isnan(w)]
            if len(w) >= 5:
                m = np.median(w)
                if state is None:
                    state = m < enter_px
                elif state and m > leave_px:
                    state = False
                elif not state and m < enter_px:
                    state = True
            flags[t] = bool(state)
        in_set[cam_id] = flags
    return frame_keys, in_set, ransac_sel


def _max_ray_angle_deg(X, cams, projection_matrices):
    """Largest pairwise angle (deg) at X between the rays to the given cameras' centres."""
    rays = []
    for c in cams:
        P = projection_matrices[c]
        v = -np.linalg.inv(P[:, :3]) @ P[:, 3] - X
        rays.append(v / np.linalg.norm(v))
    best = 0.0
    for i in range(len(rays)):
        for j in range(i + 1, len(rays)):
            best = max(best, float(np.degrees(np.arccos(np.clip(rays[i] @ rays[j], -1.0, 1.0)))))
    return best


def triangulate_sequence_stable(
    landmarks, confidence, projection_matrices, width, height, landmark_ids,
    lm_sets, min_confidence=0.5, landmark_reproj_px=20.0, landmark_confidence=None,
    exclude=None, min_ray_angle_deg=20.0, stats=None, lm_fixed_cams=None, lm_min_cams=None,
    lm_fixed_reproj_px=None,
):
    """Triangulates each landmark from (its stable camera set) AND (cameras that
    detected it with part confidence >= min_confidence this frame), then
    rejects outliers PER LANDMARK: repeatedly drop the camera with the worst
    reprojection error while it exceeds `landmark_reproj_px` and more than 2
    cameras remain. Frames where a landmark's stable set leaves <2 usable
    cameras fall back to that frame's per-frame RANSAC camera set for it.

    exclude (optional): dict[cam_id][frame_key][lm_key] -> True marks that camera's observation of that
    landmark as untrustworthy (see sapiens2/make_sil_flags.py); it is never used, also not in the
    RANSAC fallback. Where a flag removed a camera, the result is only kept if the remaining cameras'
    rays at the point span at least `min_ray_angle_deg` (near-collinear cameras give an unconstrained
    depth); otherwise that landmark has no measurement this frame and the Kalman smoother bridges it.
    `stats` (optional dict) receives counts: excluded_observations, skipped_landmark_frames,
    skipped_by_few_cameras, skipped_by_ray_angle, fixed_cam_few_cameras, fixed_cam_residual.

    lm_fixed_cams (optional): dict[lm_id] -> cameras. Those landmarks ignore the stable/RANSAC sets and use exactly
    these cameras (still gated by part confidence). lm_min_cams: dict[lm_id] -> minimum cameras (default 2); a
    fixed-camera landmark with fewer cameras, or whose worst reprojection error still exceeds
    `lm_fixed_reproj_px` (default `landmark_reproj_px`; also the outlier-rejection threshold for these landmarks)
    once only that many remain, gets no measurement that frame (the Kalman smoother bridges it).

    lm_sets: dict[lm_id] -> (frame_keys, in_set, ransac_sel) from
    select_stable_camera_sets (one shared tuple for all landmarks = part-level
    set; one per landmark = per-landmark set).

    Returns (reconstruction, selection) -- selection has the same schema as
    ransac_select_cameras_sequence's ('cameras' = cameras that contributed to
    at least one landmark; 'n_inliers' = that count; 'mean_reproj_err').
    """
    reconstruction, used, errs_by_frame = {}, {}, {}
    for lm_id in landmark_ids:
        lm_key = str(lm_id)
        frame_keys, in_set, ransac_sel = lm_sets[lm_id]
        for t, frame in enumerate(frame_keys):
            fixed = lm_fixed_cams is not None and lm_id in lm_fixed_cams
            min_n = (lm_min_cams or {}).get(lm_id, 2)
            if fixed:
                base = [
                    c for c in lm_fixed_cams[lm_id]
                    if c in projection_matrices and (confidence is None or confidence.get(c, {}).get(frame, 0) >= min_confidence)
                ]
            else:
                base = [
                    c for c in projection_matrices
                    if in_set[c][t] and (confidence is None or confidence.get(c, {}).get(frame, 0) >= min_confidence)
                ]
                if len(base) < 2:
                    base = list(ransac_sel[frame]["cameras"])
            n_removed = 0
            if exclude is not None:
                kept = [c for c in base if not exclude.get(c, {}).get(frame, {}).get(lm_key)]
                n_removed = len(base) - len(kept)
                base = kept
                if stats is not None:
                    stats["excluded_observations"] = stats.get("excluded_observations", 0) + n_removed
            obs = {}
            for c in base:
                o = landmarks.get(c, {}).get(frame, {}).get(lm_key)
                if o is not None:
                    obs[c] = (o[0] * width, o[1] * height)
            while len(obs) >= min_n:
                weights = {c: (landmark_confidence.get(c, {}).get(frame, {}).get(lm_key, 1.0)
                               if landmark_confidence is not None else 1.0) for c in obs}
                X = hmv.triangulate_dlt_weighted(obs, projection_matrices, weights)
                if X is None:
                    break
                errs = {c: hmv._reproj_error_px(X, projection_matrices[c], obs[c]) for c in obs}
                worst = max(errs, key=errs.get)
                thresh = (lm_fixed_reproj_px or landmark_reproj_px) if fixed else landmark_reproj_px
                if errs[worst] > thresh and len(obs) > min_n:
                    del obs[worst]
                    continue
                if fixed and errs[worst] > thresh:
                    if stats is not None:
                        stats["fixed_cam_residual"] = stats.get("fixed_cam_residual", 0) + 1
                    break
                if n_removed and _max_ray_angle_deg(X, list(obs), projection_matrices) < min_ray_angle_deg:
                    if stats is not None:
                        stats["skipped_by_ray_angle"] = stats.get("skipped_by_ray_angle", 0) + 1
                        stats["skipped_landmark_frames"] = stats.get("skipped_landmark_frames", 0) + 1
                    break
                reconstruction.setdefault(frame, {})[lm_key] = X.tolist()
                used.setdefault(frame, set()).update(obs)
                errs_by_frame.setdefault(frame, []).extend(errs.values())
                break
            else:
                if fixed and stats is not None:
                    stats["fixed_cam_few_cameras"] = stats.get("fixed_cam_few_cameras", 0) + 1
                if n_removed and stats is not None and lm_key not in reconstruction.get(frame, {}):
                    stats["skipped_by_few_cameras"] = stats.get("skipped_by_few_cameras", 0) + 1
                    stats["skipped_landmark_frames"] = stats.get("skipped_landmark_frames", 0) + 1
    selection = {
        f: {"cameras": sorted(used[f]), "n_inliers": len(used[f]), "mean_reproj_err": float(np.mean(errs_by_frame[f]))}
        for f in reconstruction
    }
    return reconstruction, selection


def gate_hip_outliers(raw, frame_keys, shoulder_ids=(5, 6), hip_ids=(11, 12), vmax_mm=30.0, reset_frames=5,
                      torso_tol_mm=40.0, width_tol_mm=50.0, stats=None):
    """Rejects implausible raw hip points (before Kalman smoothing) and returns a gated copy of `raw`.

    1. Speed gate, per hip: a point more than vmax_mm per frame (x frames since the last accepted point, capped
       at 10) from the last accepted one is rejected. A run of >= reset_frames rejected points that are consistent
       with each other is a level shift, not a spike, so it is accepted retroactively.
    2. Bone gate: torso length (same-side shoulder to hip) deviating from its sequence median by more than
       torso_tol_mm rejects that hip; if both torsos pass but the hip width deviates by more than width_tol_mm,
       the hip with the larger torso deviation is rejected. Shoulders are never rejected.
    `stats` (optional dict) receives hip_speed_rejected, hip_level_shifts_accepted, hip_torso_rejected,
    hip_width_rejected.
    """
    def bump(k, n=1):
        if stats is not None:
            stats[k] = stats.get(k, 0) + n

    gated = {f: dict(pts) for f, pts in raw.items()}
    idx = {f: int(os.path.splitext(f)[0]) for f in frame_keys}
    for hip in hip_ids:
        key = str(hip)
        last, last_f, streak = None, None, []
        for f in frame_keys:
            p = raw.get(f, {}).get(key)
            if p is None:
                continue
            gated[f].pop(key, None)
            p = np.asarray(p, float)
            if last is not None:
                if np.linalg.norm(p - last) > vmax_mm * min(idx[f] - last_f, 10) / 1000.0:
                    consistent = streak and np.linalg.norm(p - streak[-1][1]) <= (
                        vmax_mm * min(idx[f] - idx[streak[-1][0]], 10) / 1000.0)
                    streak = streak + [(f, p)] if consistent else [(f, p)]
                    if len(streak) < reset_frames:
                        bump("hip_speed_rejected")
                        continue
                    for sf, sp in streak:
                        gated[sf][key] = sp.tolist()
                    bump("hip_speed_rejected", -(len(streak) - 1))
                    bump("hip_level_shifts_accepted")
                    last, last_f, streak = streak[-1][1], idx[streak[-1][0]], []
                    continue
            last, last_f, streak = p, idx[f], []
            gated[f][key] = p.tolist()

    def dist(f, a, b):
        pa, pb = gated.get(f, {}).get(str(a)), gated.get(f, {}).get(str(b))
        return None if pa is None or pb is None else float(np.linalg.norm(np.asarray(pa) - np.asarray(pb))) * 1000.0

    sh = dict(zip(hip_ids, shoulder_ids))
    med_torso = {h: np.median([d for f in frame_keys if (d := dist(f, sh[h], h)) is not None] or [np.nan]) for h in hip_ids}
    med_width = np.median([d for f in frame_keys if (d := dist(f, *hip_ids)) is not None] or [np.nan])
    for f in frame_keys:
        dev = {}
        for h in hip_ids:
            d = dist(f, sh[h], h)
            dev[h] = None if d is None else abs(d - med_torso[h])
        for h in hip_ids:
            if dev[h] is not None and dev[h] > torso_tol_mm:
                gated[f].pop(str(h), None)
                dev[h] = None
                bump("hip_torso_rejected")
        w = dist(f, *hip_ids)
        if w is not None and abs(w - med_width) > width_tol_mm:
            cand = [h for h in hip_ids if dev[h] is not None]
            if cand:
                gated[f].pop(str(max(cand, key=lambda h: dev[h])), None)
                bump("hip_width_rejected")
    return gated


def triangulate_part_stable(
    landmarks, confidence, projection_matrices, width, height,
    landmark_ids, reference_landmark_id, reproj_thresh_px=15.0, min_confidence=0.5,
    landmark_confidence=None, window=45, enter_px=10.0, leave_px=20.0, landmark_reproj_px=20.0,
    per_landmark_sets=True, lm_enter_px=20.0, lm_leave_px=40.0,
    exclude=None, min_ray_angle_deg=20.0, stats=None, lm_fixed_cams=None, lm_min_cams=None, gate=None,
    lm_fixed_reproj_px=None,
):
    """Stable-camera-set alternative to triangulate_part (same return tuple).
    per_landmark_sets=True derives a separate slowly-changing camera set for
    every landmark (its own RANSAC reference + windowed error + hysteresis,
    with the looser lm_enter_px/lm_leave_px since hips etc. reproject worse
    than the shoulder reference); False uses one part-level set from
    reference_landmark_id for all landmarks.
    lm_fixed_cams / lm_min_cams: see triangulate_sequence_stable. gate (dict of gate_hip_outliers kwargs, body
    only): rejects implausible raw hip points before smoothing; its counts go into `stats`.
    """
    sel_kw = dict(reproj_thresh_px=reproj_thresh_px, min_confidence=min_confidence, window=window)
    part_set = select_stable_camera_sets(
        landmarks, confidence, projection_matrices, width, height, reference_landmark_id,
        enter_px=enter_px, leave_px=leave_px, **sel_kw)
    if per_landmark_sets:
        lm_sets = {lm: select_stable_camera_sets(
            landmarks, confidence, projection_matrices, width, height, lm,
            enter_px=lm_enter_px, leave_px=lm_leave_px, **sel_kw) for lm in landmark_ids}
    else:
        lm_sets = {lm: part_set for lm in landmark_ids}
    raw, selection = triangulate_sequence_stable(
        landmarks, confidence, projection_matrices, width, height, landmark_ids, lm_sets,
        min_confidence=min_confidence, landmark_reproj_px=landmark_reproj_px,
        landmark_confidence=landmark_confidence,
        exclude=exclude, min_ray_angle_deg=min_ray_angle_deg, stats=stats,
        lm_fixed_cams=lm_fixed_cams, lm_min_cams=lm_min_cams, lm_fixed_reproj_px=lm_fixed_reproj_px,
    )
    sel_frames = _frame_order(selection.keys())
    if gate is not None:
        raw = gate_hip_outliers(raw, sel_frames, stats=stats, **gate)
    smoothed = hmv.smooth_reconstruction_sequence(
        raw, sel_frames, landmark_ids, {f: s["n_inliers"] for f, s in selection.items()},
    )
    jitter_flags, _ = hmv.flag_jitter_frames(
        raw, smoothed, sel_frames, reference_landmark_id=reference_landmark_id,
    )
    reproj_errors = hmv.ransac_per_camera_reproj_error(
        landmarks, confidence, raw, projection_matrices, width, height,
        reference_landmark_id=reference_landmark_id, min_confidence=min_confidence,
    )
    return raw, smoothed, selection, reproj_errors, jitter_flags

def summarize_part(name, total_frames, selection, reproj_errors, jitter_flags):
    inlier_sizes = [sel["n_inliers"] for sel in selection.values()]
    flat_reproj = [err for by_frame in reproj_errors.values() for err in by_frame.values()]
    metrics = {
        "total_frames": total_frames,
        "ransac_success_frames": len(selection),
        "ransac_success_rate": len(selection) / total_frames if total_frames else 0.0,
        "mean_inlier_set_size": float(np.mean(inlier_sizes)) if inlier_sizes else None,
        "reproj_error_px": {
            "mean": float(np.mean(flat_reproj)) if flat_reproj else None,
            "median": float(np.median(flat_reproj)) if flat_reproj else None,
            "p95": float(np.percentile(flat_reproj, 95)) if flat_reproj else None,
        },
        "jitter_flagged_frames": len(jitter_flags),
        "jitter_fraction": len(jitter_flags) / len(selection) if selection else 0.0,
    }
    print(f"\n  [{name}]")
    print(f"    RANSAC camera-set found: {metrics['ransac_success_frames']}/{total_frames} "
          f"({metrics['ransac_success_rate']:.1%})")
    if metrics["mean_inlier_set_size"] is not None:
        print(f"    mean inlier-set size: {metrics['mean_inlier_set_size']:.2f} cameras")
    rp = metrics["reproj_error_px"]
    if rp["mean"] is not None:
        print(f"    reprojection error (px): mean={rp['mean']:.2f} median={rp['median']:.2f} p95={rp['p95']:.2f}")
    print(f"    jitter-flagged frames: {metrics['jitter_flagged_frames']} ({metrics['jitter_fraction']:.1%})")
    return metrics


def run_take(take_dir, calib_path, force, out_dir=None, pose2d_dir="pose2d", camera_mode="ransac",
             exclude_flags=None, min_ray_angle_deg=20.0, hip_cams=None, hip_min_cams=3, gate_hips=False,
             hip_reproj_px=None, gate_vmax_mm=30.0, gate_torso_tol_mm=40.0, gate_width_tol_mm=50.0):
    """Reads pose2d's already-extracted 2D landmarks from `take_dir`'s
    `pose2d_dir` subfolder (never re-extracts -- calibration only changes
    the triangulation projection matrices, not the 2D detection). Pass
    pose2d_dir="pose2d_dwpose" to triangulate DWPose's output instead of
    MediaPipe's -- same schema either way (see load_pose2d_take_data).

    Writes reconstruction/diagnostics files under `out_dir` if given
    (default: take_dir's own aligned/<pose2d_dir>/, the normal in-place
    behavior, so DWPose and MediaPipe results coexist without collision) --
    so trying an alternate calibration doesn't clobber a take's real
    results either. `out_dir`, if given, is treated as a stand-in take_dir:
    its aligned/<pose2d_dir>/ subfolder gets the output, in the same
    reconstruction_<part>.json layout pose2d.visualize_triangulation
    already expects, so the viewer can point at it directly with no
    changes.

    exclude_flags (stable mode, body part only): path to a JSON {cam: {lm_id: [frame numbers]}} of
    camera observations to leave out of the triangulation (see triangulate_sequence_stable).

    hip_cams (stable mode, body only): list of camera ids that alone triangulate the hips (landmarks 11, 12), needing
    at least hip_min_cams of them. gate_hips: reject implausible raw hip points before smoothing (gate_hip_outliers).
    """
    if (exclude_flags or hip_cams or gate_hips) and camera_mode != "stable":
        raise ValueError("--exclude-flags, --hip-cams and --gate-hips need --camera-mode stable")
    if camera_mode == "stable":
        # Never clobber the per-frame-RANSAC results: default to a "stable" subfolder.
        out_dir = (os.path.join(out_dir, "aligned", pose2d_dir) if out_dir
                   else os.path.join(take_dir, "aligned", pose2d_dir, "stable"))
    else:
        out_dir = os.path.join(out_dir if out_dir else take_dir, "aligned", pose2d_dir)
    os.makedirs(out_dir, exist_ok=True)
    paths = {
        part: (os.path.join(out_dir, f"reconstruction_{part}.json"), os.path.join(out_dir, f"triangulation_diagnostics_{part}.json"))
        for part in ("left", "right", "body")
    }
    if not force and all(os.path.exists(p[0]) for p in paths.values()):
        print(f"Reconstructions already exist for {take_dir} -- pass --force to regenerate")
        return

    calib = calibrate.load_calibration_output(calib_path)
    data = load_pose2d_take_data(take_dir, pose2d_dir=pose2d_dir)
    cam_ids = [c for c in data["cam_ids"] if c in calib]
    if len(cam_ids) < 2:
        raise RuntimeError(f"{take_dir}: only {len(cam_ids)} cameras have real extrinsics in {calib_path}")

    projection_matrices = build_projection_matrices(calib, cam_ids)
    sample_frame = next(iter(data["landmarks_body"].get(cam_ids[0], {})), None)
    if sample_frame is None:
        sample_frame = next(iter(data["landmarks_left"].get(cam_ids[0], {})), None)
    width, height = hmv.get_image_size(take_dir, cam_ids[0], sample_frame)
    total_frames = len(hmv.discover_frames(take_dir, cam_ids[0]))

    print(f"\n=== {take_dir} (aligned/{pose2d_dir}) ===")
    print(f"  cameras: {len(cam_ids)}, frames per camera: {total_frames}")

    parts = {
        "left": (data["landmarks_left"], data["confidence_left"], LANDMARK_IDS_HAND, REFERENCE_LANDMARK_ID_HAND,
                  data["landmark_confidence_left"]),
        "right": (data["landmarks_right"], data["confidence_right"], LANDMARK_IDS_HAND, REFERENCE_LANDMARK_ID_HAND,
                   data["landmark_confidence_right"]),
        "body": (data["landmarks_body"], data["confidence_body"], LANDMARK_IDS_BODY, REFERENCE_LANDMARK_ID_BODY,
                  data["landmark_confidence_body"]),
    }

    exclude = None
    if exclude_flags:
        with open(exclude_flags) as f:
            raw_flags = json.load(f)
        exclude = {cam: {} for cam in cam_ids}
        for cam, per_lm in raw_flags.items():
            for lm, frames in per_lm.items():
                for fr in frames:
                    exclude.setdefault(cam, {}).setdefault(f"{int(fr):06d}.jpg", {})[str(lm)] = True
        print(f"  excluding {sum(len(v) for per_lm in raw_flags.values() for v in per_lm.values())} flagged camera observations "
              f"(body only, min ray angle {min_ray_angle_deg} deg)")

    for part_name, (landmarks, confidence, landmark_ids, reference_landmark_id, landmark_confidence) in parts.items():
        part_fn = triangulate_part_stable if camera_mode == "stable" else triangulate_part
        extra, stats = {}, {}
        if camera_mode == "stable" and part_name == "body" and (exclude is not None or hip_cams or gate_hips):
            extra = dict(stats=stats)
            if exclude is not None:
                extra.update(exclude=exclude, min_ray_angle_deg=min_ray_angle_deg)
            if hip_cams:
                extra.update(lm_fixed_cams={11: list(hip_cams), 12: list(hip_cams)},
                             lm_min_cams={11: hip_min_cams, 12: hip_min_cams}, lm_fixed_reproj_px=hip_reproj_px)
            if gate_hips:
                extra.update(gate=dict(vmax_mm=gate_vmax_mm, torso_tol_mm=gate_torso_tol_mm, width_tol_mm=gate_width_tol_mm))
        raw, smoothed, selection, reproj_errors, jitter_flags = part_fn(
            landmarks, confidence, projection_matrices, width, height, landmark_ids, reference_landmark_id,
            landmark_confidence=landmark_confidence, **extra,
        )
        metrics = summarize_part(part_name, total_frames, selection, reproj_errors, jitter_flags)
        if stats:
            metrics["flag_exclusion"] = stats
            print(f"    flag exclusion: {stats}")
        metrics["per_landmark_confidence_used"] = landmark_confidence is not None
        print(f"    per-landmark confidence weighting: {'yes' if landmark_confidence is not None else 'no (uniform)'}")

        recon_path, diag_path = paths[part_name]
        with open(recon_path, "w") as f:
            json.dump({"raw": raw, "smoothed": smoothed}, f)
        with open(diag_path, "w") as f:
            json.dump({
                "metrics": metrics,
                "per_frame_selection": selection,
                "jitter_flagged_frame_keys": sorted(jitter_flags),
            }, f, indent=2)
        print(f"    wrote: {recon_path}")
        print(f"    wrote: {diag_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("take_dirs", nargs="+")
    parser.add_argument("--calib", default=DEFAULT_CALIB_PATH)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--camera-mode", choices=["ransac", "stable"], default="ransac",
        help="'ransac' (default): per-frame RANSAC camera set. 'stable': slowly-changing camera set "
             "(windowed reprojection error + hysteresis) with per-landmark outlier rejection; writes to "
             "<take>/aligned/<pose2d-dir>/stable/ unless --out-dir is given.",
    )
    parser.add_argument(
        "--exclude-flags", default=None,
        help="Stable mode, body only: JSON {cam: {landmark_id: [frame numbers]}} of camera observations to leave "
             "out (made by sapiens2/make_sil_flags.py).",
    )
    parser.add_argument(
        "--min-ray-angle", type=float, default=20.0,
        help="With --exclude-flags: a landmark that lost a camera to a flag is only kept if the remaining "
             "cameras' rays span at least this many degrees; otherwise the Kalman smoother bridges the frame.",
    )
    parser.add_argument(
        "--hip-cams", default=None,
        help="Stable mode, body only: comma-separated camera ids (e.g. cam0,cam1,cam3,cam4) that alone triangulate "
             "the hips, ignoring the stable/RANSAC camera sets.",
    )
    parser.add_argument("--hip-min-cams", type=int, default=3, help="With --hip-cams: minimum cameras per hip point.")
    parser.add_argument("--hip-reproj-px", type=float, default=None,
                        help="With --hip-cams: reprojection-error threshold (px) for rejecting a camera / skipping a "
                             "hip point (default: the 20 px landmark threshold).")
    parser.add_argument(
        "--gate-hips", action="store_true",
        help="Stable mode, body only: reject implausible raw hip points before smoothing (speed gate + torso/hip-width "
             "bone gate; see gate_hip_outliers).",
    )
    parser.add_argument("--gate-vmax-mm", type=float, default=30.0, help="Hip speed gate, mm per frame.")
    parser.add_argument("--gate-torso-tol-mm", type=float, default=40.0, help="Hip bone gate: torso length tolerance.")
    parser.add_argument("--gate-width-tol-mm", type=float, default=50.0, help="Hip bone gate: hip width tolerance.")
    parser.add_argument(
        "--out-dir", default=None,
        help="Write reconstruction/diagnostics here instead of <take_dir>/aligned/<pose2d-dir>/ "
             "(a stand-in take_dir -- its aligned/<pose2d-dir>/ subfolder gets the output). "
             "Use this to try an alternate --calib without overwriting a take's real results.",
    )
    parser.add_argument(
        "--pose2d-dir", default="pose2d",
        help="Which detector's 2D output to triangulate: 'pose2d' (MediaPipe, default) or "
             "'pose2d_dwpose' (DWPose) -- both live under <take_dir>/aligned/.",
    )
    args = parser.parse_args()
    for take_dir in args.take_dirs:
        run_take(take_dir, args.calib, args.force, out_dir=args.out_dir, pose2d_dir=args.pose2d_dir,
                 camera_mode=args.camera_mode, exclude_flags=args.exclude_flags, min_ray_angle_deg=args.min_ray_angle,
                 hip_cams=args.hip_cams.split(",") if args.hip_cams else None, hip_min_cams=args.hip_min_cams, hip_reproj_px=args.hip_reproj_px,
                 gate_hips=args.gate_hips, gate_vmax_mm=args.gate_vmax_mm,
                 gate_torso_tol_mm=args.gate_torso_tol_mm, gate_width_tol_mm=args.gate_width_tol_mm)


if __name__ == "__main__":
    main()
