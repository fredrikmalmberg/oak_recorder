"""take_dir -> the (frame_keys, target_points, confidence) tensors the
optimizer fits against, built entirely from files pose2d.triangulation
already produces -- no new triangulation work.

Design choices worth documenting explicitly (per the approved plan):

1. Uses the "smoothed" triangulated positions (reconstruction_*.json's
   Kalman/RTS-smoothed trajectory, see hand_pose.hand_multiview.
   smooth_reconstruction_sequence) as the fitting TARGET VALUE, but
   "raw"'s own presence to decide PRESENCE/gating. An earlier version of
   this pipeline used "raw" for both, reasoning that fitting against an
   already-smoothed proxy would double-count temporal-smoothness
   regularization -- reversed after observing visibly jittery fitted
   motion in practice: per-frame triangulation noise (confirmed via
   optimization_log.json -- smooth_hand_pose/reg_hand both grew
   substantially as k3d_hand tightened, i.e. the fit was contorting hand
   pose frame-to-frame to chase noise) was flowing straight into k3d/
   k3d_hand, and this pipeline's own smooth_* weights (copied from
   bvh2smplx's clean-BVH-mocap tuning) turned out far too weak to counter
   it. Using "smoothed" removes that noise at the source. Gating still
   comes from "raw" specifically because "smoothed" can interpolate
   THROUGH frames where triangulation genuinely failed -- trusting
   "smoothed"'s presence there would treat a fabricated/interpolated value
   as if it were a real observation.

2. Tensors are built over the FULL video frame range (0..total_frames-1,
   via hand_pose.hand_multiview.discover_frames), not just the frames
   where some part's RANSAC succeeded. This keeps array index == real
   video frame number, so temporal-smoothness loss terms always compare
   adjacent TRUE frames one capture-interval apart. Building over only the
   sparser "something succeeded" union would silently miscompute time
   deltas across gaps.

3. Confidence combines four independent, already-computed signals (see
   build_confidence docstring) into one per-(frame, landmark) weight in
   [0, 1] -- a landmark simply absent from a frame's reconstruction gets
   confidence exactly 0 (a hard gate), everything else is a soft
   multiplicative penalty.

4. Wrist fusion (joint_mapping.WRIST_FUSION): COCO body-wrist and
   MediaPipe hand-wrist ids observe the same physical joint from two
   different detectors. Confidence-weighted average when both are
   present; whichever is present when only one is; confidence 0 (like
   any other missing landmark) when neither is.

5. Wrist keypoint offset (joint_mapping.WRIST_KEYPOINT_OFFSET_M), OFF by default: with wrist_offset=True (fit_take
   --wrist-offset) the fused target is shifted by a constant MANO-wrist offset in the hand frame (hand_frame). The
   constants describe the MANO wrist, which is ~20 mm further along the hand than the SMPL-X wrist, so on SMPL-X it
   overshoots and worsens elbows and wrists (see README "SMPL-X wrist keypoint offset").
   drop_wrist=True (fit_take --drop-wrist) removes the wrist target altogether, so the wrist is placed by the elbow
   and the finger/MCP keypoints (needs hands).
"""
import json
import os

import numpy as np

from hand_pose import hand_multiview as hmv
from smplx_fit import joint_mapping as jm

REPROJ_ERR_SCALE_PX = 15.0  # matches ransac_select_cameras_sequence's own reproj_thresh_px
JITTER_PENALTY = 0.3  # penalize, don't zero -- a jitter flag means "less trustworthy", not "unusable"


def _load_diagnostics(take_dir, part, pose2d_dir):
    path = os.path.join(take_dir, "aligned", pose2d_dir, f"triangulation_diagnostics_{part}.json")
    with open(path) as f:
        return json.load(f)


def build_confidence_for_part(diagnostics, frame_keys, n_cams_total):
    """Per-frame confidence multiplier (NOT per-landmark -- per_frame_
    selection is the finest granularity triangulation_diagnostics_*.json
    offers today, a documented limitation, not an oversight). Combines:
      - n_inliers / n_cams_total, clipped to [0.3, 1.0]
      - 1 - mean_reproj_err_px / 15.0, clipped to [0.1, 1.0]
      - 0.3 if jitter-flagged, else 1.0
    Frames absent from per_frame_selection (RANSAC found nothing usable)
    get 0.0 here -- distinct from a low-but-nonzero confidence, since the
    presence gate (per-landmark, applied separately in build_take_arrays)
    already handles "no reconstruction at all" for those landmarks.
    """
    selection = diagnostics["per_frame_selection"]
    jitter_flagged = set(diagnostics["jitter_flagged_frame_keys"])
    conf = np.zeros(len(frame_keys), dtype=np.float32)
    for i, fk in enumerate(frame_keys):
        sel = selection.get(fk)
        if sel is None:
            continue
        inlier_factor = np.clip(sel["n_inliers"] / n_cams_total, 0.3, 1.0)
        reproj_factor = np.clip(1.0 - sel["mean_reproj_err"] / REPROJ_ERR_SCALE_PX, 0.1, 1.0)
        jitter_factor = JITTER_PENALTY if fk in jitter_flagged else 1.0
        conf[i] = inlier_factor * reproj_factor * jitter_factor
    return conf


def hand_frame(hand_recon, frame_keys, max_gap=10):
    """(nf, 3, 3) rotation matrices (columns x, y, z in world axes) of the hand frame used for the wrist keypoint
    offset (see joint_mapping.WRIST_KEYPOINT_OFFSET_M). Built from the triangulated hand joints 0, 5, 9, 17. Frames
    without a frame borrow the nearest valid one within max_gap frames; the rest stay NaN."""
    nf = len(frame_keys)
    R = np.full((nf, 3, 3), np.nan)
    for i, fk in enumerate(frame_keys):
        j = hand_recon.get(fk, {})
        if not all(k in j for k in ("0", "5", "9", "17")):
            continue
        w, im, m, p = (np.array(j[k], float) for k in ("0", "5", "9", "17"))
        x = m - w
        n = np.cross(im - w, p - w)
        n = n - (n @ x) / (x @ x) * x
        if np.linalg.norm(x) < 1e-6 or np.linalg.norm(n) < 1e-9:
            continue
        x, n = x / np.linalg.norm(x), n / np.linalg.norm(n)
        R[i] = np.stack([x, np.cross(n, x), n], axis=1)
    valid = np.where(np.isfinite(R[:, 0, 0]))[0]
    if len(valid):
        for i in np.where(~np.isfinite(R[:, 0, 0]))[0]:
            k = valid[np.argmin(np.abs(valid - i))]
            if abs(k - i) <= max_gap:
                R[i] = R[k]
    return R


def build_take_arrays(take_dir, pose2d_dir="pose2d", use_hands=True, wrist_offset=False, drop_wrist=False, legs_npz=None,
                      legs_smooth_sigma=2.0):
    """Returns (frame_keys, layout, target_points, confidence):
      frame_keys: list[str], length nf, the take's full %06d.jpg range.
      layout: joint_mapping.build_keypoint_layout()'s list, in the same
        order as target_points/confidence's K axis, PLUS two more entries
        appended at the end for the fused left/right wrists (kind="joint",
        matching joint_mapping.WRIST_FUSION's targets).
      target_points: (nf, K, 3) float32, meters. Zero where confidence==0.
      confidence: (nf, K) float32, in [0, 1]. Zero means "no usable
        observation this frame for this landmark" -- the fitter must treat
        this as a hard gate (zero loss contribution), not a low weight.
    """
    cam_ids = hmv.discover_cameras(take_dir)
    n_cams_total = len(cam_ids)
    total_frames = len(hmv.discover_frames(take_dir, cam_ids[0]))
    frame_keys = [f"{i:06d}.jpg" for i in range(total_frames)]

    layout = jm.build_keypoint_layout(legs=legs_npz is not None)
    wrist_entries = [(f"wrist_{side}", None, fusion["target"]) for side, fusion in jm.WRIST_FUSION.items()]
    full_layout = layout + wrist_entries
    K = len(full_layout)
    nf = total_frames

    target_points = np.zeros((nf, K, 3), dtype=np.float32)
    confidence = np.zeros((nf, K), dtype=np.float32)

    part_frame_confidence = {}
    for part in ("body", "left", "right"):
        diag = _load_diagnostics(take_dir, part, pose2d_dir)
        part_frame_confidence[part] = build_confidence_for_part(diag, frame_keys, n_cams_total)
        if not use_hands and part != "body":
            part_frame_confidence[part][:] = 0.0  # no hand targets; fused wrists fall back to the body wrist

    # Triangulated 3D positions live in reconstruction_{part}.json, not in
    # landmarks_{part} (which is 2D, per-camera, pre-triangulation).
    # PRESENCE/gating still comes from "raw" (a landmark missing from raw
    # means RANSAC genuinely found nothing that frame -- "smoothed" can
    # interpolate through such gaps, which is exactly the fabricated-data
    # risk noted in point 1 above, so gating must not be fooled by it).
    # The target VALUE itself is read from "smoothed", not "raw" -- see
    # point 1's update: using raw values directly fed per-frame
    # triangulation noise straight into k3d/k3d_hand, and this pipeline's
    # own smooth_* terms turned out far too weakly weighted (copied from
    # bvh2smplx's clean-BVH-mocap tuning) to counteract it, producing
    # visibly jittery fitted motion. Using the already-Kalman/RTS-smoothed
    # trajectory as the target removes that noise at the source instead of
    # relying entirely on retuning loss weights.
    raw_reconstructions, smoothed_reconstructions = {}, {}
    for part in ("body", "left", "right"):
        with open(os.path.join(take_dir, "aligned", pose2d_dir, f"reconstruction_{part}.json")) as f:
            recon = json.load(f)
        raw_reconstructions[part] = recon["raw"]
        smoothed_reconstructions[part] = recon["smoothed"]

    def target_xyz(part, fk, landmark_id):
        """The value to fit against for one (part, frame, landmark) --
        smoothed if available, else raw's own value as a defensive
        fallback (smoothed should cover every frame raw does, but don't
        assume it always will).
        """
        smoothed_xyz = smoothed_reconstructions[part].get(fk, {}).get(landmark_id)
        if smoothed_xyz is not None:
            return smoothed_xyz
        return raw_reconstructions[part].get(fk, {}).get(landmark_id)

    leg_tracks = {}
    if legs_npz is not None:
        # triangulated leg points (pose2d.triangulate_legs): gaps interpolated, 5-frame median + Gaussian, weight by camera count
        from smplx_fit.add_legs import smooth_track
        L = np.load(legs_npz)
        for j, lid in enumerate(jm.LEG_FILE_ORDER):
            X = L["X"][:, j]
            if not np.isfinite(X[:, 0]).any():
                continue
            w_ = np.clip((L["ncam"][:, j] - 1) / 3.0, 0, 1) * np.isfinite(X[:, 0])
            leg_tracks[lid] = (L["frames"], smooth_track(X, legs_smooth_sigma), w_)

    for k, (part, landmark_id, target) in enumerate(full_layout):
        if part == "body" and landmark_id in jm.LEG_TARGETS:
            if landmark_id in leg_tracks:
                fr, X, w_ = leg_tracks[landmark_id]
                for fi, f in enumerate(fr):
                    if 0 <= int(f) < nf and w_[fi] > 0:
                        target_points[int(f), k] = X[fi]
                        confidence[int(f), k] = w_[fi]
        elif part in ("body", "left", "right"):
            raw_part = raw_reconstructions[part]
            frame_conf = part_frame_confidence[part]
            for i, fk in enumerate(frame_keys):
                if raw_part.get(fk, {}).get(landmark_id) is None or frame_conf[i] <= 0:
                    continue
                target_points[i, k] = target_xyz(part, fk, landmark_id)
                confidence[i, k] = frame_conf[i]
        else:
            # wrist_left / wrist_right: fuse COCO body-wrist + MediaPipe
            # hand-wrist observations of the same physical joint (see
            # module docstring point 4).
            side = part.split("_")[1]
            fusion = jm.WRIST_FUSION[side]
            body_raw, hand_raw = raw_reconstructions["body"], raw_reconstructions[side]
            body_conf, hand_conf = part_frame_confidence["body"], part_frame_confidence[side]
            for i, fk in enumerate(frame_keys):
                body_present = body_raw.get(fk, {}).get(fusion["body_landmark_id"]) is not None
                hand_present = hand_raw.get(fk, {}).get(fusion["hand_landmark_id"]) is not None
                bc = body_conf[i] if body_present else 0.0
                hc = hand_conf[i] if hand_present else 0.0
                if bc <= 0 and hc <= 0:
                    continue
                body_xyz = target_xyz("body", fk, fusion["body_landmark_id"]) if body_present else None
                hand_xyz = target_xyz(side, fk, fusion["hand_landmark_id"]) if hand_present else None
                if bc > 0 and hc > 0:
                    w = bc + hc
                    xyz = (np.array(body_xyz) * bc + np.array(hand_xyz) * hc) / w
                    target_points[i, k] = xyz
                    confidence[i, k] = (bc + hc) / 2.0
                elif bc > 0:
                    target_points[i, k] = body_xyz
                    confidence[i, k] = bc
                else:
                    target_points[i, k] = hand_xyz
                    confidence[i, k] = hc

            if drop_wrist:
                confidence[:, k] = 0.0
                target_points[:, k] = 0.0
                continue

            if wrist_offset:
                R = hand_frame(smoothed_reconstructions[side], frame_keys)
                off = np.array(jm.WRIST_KEYPOINT_OFFSET_M[side])
                have = confidence[:, k] > 0
                ok = have & np.isfinite(R[:, 0, 0])
                target_points[ok, k] += (R[ok] @ off).astype(np.float32)
                if (have & ~ok).any():
                    print(f"  wrist offset: {side} wrist has no hand frame in {(have & ~ok).sum()} frames, target left unshifted")

    return frame_keys, full_layout, target_points, confidence
