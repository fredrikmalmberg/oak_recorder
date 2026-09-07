"""Temporal/consistency checks layered on top of pose2d.extraction's
per-frame-independent output. The wrist-proximity assignment in
extraction.py is already a strong per-frame signal; this adds the
sequence-level checks the old ransac_keypoint_reconstruction.ipynb Stage 1
used (temporal tracking with a distance gate, occlusion flagging,
handedness cross-checks) -- reusing hand_pose/hand_multiview.py's already
detector-agnostic 2D-only functions where they exist, and a MediaPipe-only
adaptation of Stage 1's occlusion flagging (ported concept, not code --
that cell's box source was a YOLO+MediaPipe union; YOLO is deferred, so
this uses each hand's own 21-landmark bounding box instead of an
independent second detector's box -- a real but weaker signal, noted as a
known limitation of this pass).
"""
import copy
import os

import numpy as np

from hand_pose import hand_multiview as hmv

OCCLUSION_IOU_THRESH = 0.1
BODY_WRIST_BBOX_MARGIN_PX = 40.0
BODY_TRUST_MAX_WRIST_DIST_PX = 250.0


def _hand_bbox(lms):
    """21-landmark normalized-coordinate bounding box, [x0, y0, x1, y1].
    Ported concept from legacy/ransac_keypoint_reconstruction.ipynb's
    hand_bbox (there operating on pixel coords with 'x'/'y' keys; here on
    this project's [x, y, z] list schema)."""
    xs = [pt[0] for pt in lms.values()]
    ys = [pt[1] for pt in lms.values()]
    return np.array([min(xs), min(ys), max(xs), max(ys)], dtype=np.float64)


def _iou(a, b):
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def flag_hand_occlusions(landmarks_left, landmarks_right, cam_ids, frame_keys, iou_thresh=OCCLUSION_IOU_THRESH):
    """MediaPipe-only occlusion proxy: flags (cam_id, frame) where both hands
    are detected AND their own landmark bounding boxes overlap more than
    `iou_thresh` -- hands close together/crossing are exactly where
    wrist-proximity disambiguation is least trustworthy. Weaker than the
    original YOLO+MediaPipe cross-detector version (see module docstring).

    Returns dict[cam_id][frame_key] -> bool.
    """
    flags = {}
    for cam_id in cam_ids:
        for frame in frame_keys:
            l_lms = landmarks_left.get(cam_id, {}).get(frame)
            r_lms = landmarks_right.get(cam_id, {}).get(frame)
            if not l_lms or not r_lms:
                continue
            iou = _iou(_hand_bbox(l_lms), _hand_bbox(r_lms))
            # numpy.bool_ (from the numpy-array math in _iou/_hand_bbox) isn't
            # JSON-serializable -- cast to a native bool before storing.
            flags.setdefault(cam_id, {})[frame] = bool(iou > iou_thresh)
    return flags


def flag_single_hand_body_wrist_occlusion(
    landmarks_left, landmarks_right, landmarks_body_hand_left, landmarks_body_hand_right,
    cam_ids, frame_keys, width, height, bbox_margin_px=BODY_WRIST_BBOX_MARGIN_PX,
):
    """Catches the occlusion case flag_hand_occlusions structurally can't see:
    one hand is physically covering the other, so MediaPipe Hands only
    detects ONE hand that frame -- there's no second hand box to compare
    against (and detect_body_wrist_hand_swap also can't run: it requires
    BOTH hands present). But the body-pose model still reports sparse
    points near both hands every frame regardless of how many hands Hands
    detected -- lower-precision than the 21-point hand landmarks, but
    present.

    Compares the ONE detected hand's own (high-precision, 21-landmark)
    bounding box against pose2d.extraction's landmarks_body_hand_left/right
    -- BlazePose's 4 sparse points per hand (wrist, pinky, index, thumb),
    not just the wrist. Flags occlusion when ANY sparse point of EACH side
    falls inside the box (expanded by `bbox_margin_px`, since these points
    are themselves noisy, especially for whichever hand is occluded) --
    using all 4 points instead of only the wrist matters in practice: on
    152859/take_1 cam3, frames 000289.jpg/000293.jpg are real occlusion but
    the wrist alone landed 23-49px outside even a 40px-expanded box (the
    same-side point closest to the detected hand isn't always the wrist).

    Deliberately flag-only, no auto-correction (unlike
    detect_body_wrist_hand_swap): there's no independent signal here for
    which side is actually correct -- both hypotheses look almost equally
    plausible by construction of exactly this case, so "correcting" would
    just be guessing. Feed into confidence penalty and surface in the grid
    video; use with caution.

    Returns dict[cam_id][frame_key] -> bool.
    """
    mx, my = bbox_margin_px / width, bbox_margin_px / height
    flags = {}
    for cam_id in cam_ids:
        for frame in frame_keys:
            has_left = frame in landmarks_left.get(cam_id, {})
            has_right = frame in landmarks_right.get(cam_id, {})
            if has_left == has_right:
                continue  # both or neither detected -- not this case
            hand_l = landmarks_body_hand_left.get(cam_id, {}).get(frame)
            hand_r = landmarks_body_hand_right.get(cam_id, {}).get(frame)
            if not hand_l or not hand_r:
                continue

            detected = landmarks_left[cam_id][frame] if has_left else landmarks_right[cam_id][frame]
            bbox = _hand_bbox(detected)
            x0, y0, x1, y1 = bbox[0] - mx, bbox[1] - my, bbox[2] + mx, bbox[3] + my

            def any_inside(points):
                return any(x0 <= pt[0] <= x1 and y0 <= pt[1] <= y1 for pt in points.values())

            flags.setdefault(cam_id, {})[frame] = bool(any_inside(hand_l) and any_inside(hand_r))
    return flags


def flag_body_pose_untrusted(
    landmarks_left, landmarks_right, landmarks_body, cam_ids, frame_keys, width, height,
    max_wrist_dist_px=BODY_TRUST_MAX_WRIST_DIST_PX,
):
    """Flags (cam_id, frame) where at least one hand IS detected but the
    body-pose signal everything else in this module leans on (swap
    detection, single-hand occlusion risk) can't actually be trusted that
    frame -- either body pose wasn't detected at all, or it WAS detected
    but neither of its wrists is anywhere near any detected hand (body
    tracking failure/drift, not just a left/right mixup).

    Verified on 152859/take_1, frame 000233.jpg: cam3 has body pose
    present but BOTH its wrists are 748-789px from the one detected hand
    (a body-tracking failure); cam4 has no body pose detected at all. Both
    are real cases where a mislabeled/swapped hand on that frame would go
    uncaught -- detect_body_wrist_hand_swap needs both hands AND a
    trustworthy body pose to work; this flag says when that assumption
    itself doesn't hold, so a lack of a SWAP flag on these frames is NOT
    confirmation the labeling is correct.

    Returns dict[cam_id][frame_key] -> bool, only for frames with >=1 hand
    detected (nothing to distrust when nothing was detected at all).
    """
    body_l_key = str(hmv.BODY_WRIST_COCO['left'])
    body_r_key = str(hmv.BODY_WRIST_COCO['right'])
    flags = {}
    for cam_id in cam_ids:
        for frame in frame_keys:
            l_lm = landmarks_left.get(cam_id, {}).get(frame)
            r_lm = landmarks_right.get(cam_id, {}).get(frame)
            if not l_lm and not r_lm:
                continue  # nothing detected -- nothing to (dis)trust

            body = landmarks_body.get(cam_id, {}).get(frame)
            body_l = body.get(body_l_key) if body else None
            body_r = body.get(body_r_key) if body else None
            if body_l is None or body_r is None:
                flags.setdefault(cam_id, {})[frame] = True
                continue

            def min_dist_to_body(hand_lm):
                hw = hand_lm["0"]
                d_l = ((hw[0] - body_l[0]) * width) ** 2 + ((hw[1] - body_l[1]) * height) ** 2
                d_r = ((hw[0] - body_r[0]) * width) ** 2 + ((hw[1] - body_r[1]) * height) ** 2
                return min(d_l, d_r) ** 0.5

            dists = [min_dist_to_body(lm) for lm in (l_lm, r_lm) if lm is not None]
            flags.setdefault(cam_id, {})[frame] = bool(min(dists) > max_wrist_dist_px)
    return flags


def _swap_landmark_confidence(lc_left, lc_right, swap_flags):
    """Mirrors hand_multiview.apply_swap_corrections' per-(cam,frame) dict-
    swap logic, but for an OPTIONAL 5th piece of data (per-landmark
    confidence) that function's fixed signature doesn't carry -- keeps
    landmark_confidence describing the SAME physical hand as the
    (already swap-corrected) landmarks it accompanies. No-ops when either
    side is None (the detector has no per-landmark confidence at all, e.g.
    MediaPipe Hands).
    """
    if lc_left is None or lc_right is None:
        return lc_left, lc_right
    new_left = copy.deepcopy(lc_left)
    new_right = copy.deepcopy(lc_right)
    for cam_id, by_frame in swap_flags.items():
        for fk, flagged in by_frame.items():
            if not flagged:
                continue
            l_val = lc_left.get(cam_id, {}).get(fk)
            r_val = lc_right.get(cam_id, {}).get(fk)
            if r_val is not None:
                new_left.setdefault(cam_id, {})[fk] = r_val
            else:
                new_left.get(cam_id, {}).pop(fk, None)
            if l_val is not None:
                new_right.setdefault(cam_id, {})[fk] = l_val
            else:
                new_right.get(cam_id, {}).pop(fk, None)
    return new_left, new_right


def run_tracking(
    take_dir, extraction, width, height,
    velocity_thresh_px=200.0, swap_margin_px=20.0, occlusion_iou_thresh=OCCLUSION_IOU_THRESH,
    body_wrist_bbox_margin_px=BODY_WRIST_BBOX_MARGIN_PX,
    body_trust_max_wrist_dist_px=BODY_TRUST_MAX_WRIST_DIST_PX, penalty_factor=0.01,
    penalize_occlusion=True,
):
    """Applies the full check/correction/penalty chain to one take's
    extraction() output. Returns (landmarks_left, confidence_left,
    landmarks_right, confidence_right, landmarks_body, confidence_body,
    diagnostics, landmark_confidence_left, landmark_confidence_right,
    landmark_confidence_body) -- the first six are the corrected/penalized
    versions (deep-copied, extraction's own output is never mutated);
    diagnostics is a plain-JSON-serializable summary + per-frame flag dict
    for the grid video and for reporting. The last three are per-landmark
    confidence (e.g. DWPose's per-keypoint SimCC scores), passed through
    swap-corrected the same way landmarks/confidence are -- None for any
    part the detector doesn't provide per-landmark confidence for (e.g.
    MediaPipe Hands only ever has one whole-hand score).

    penalize_occlusion=False skips applying the confidence penalty for
    occlusion_flags/single_hand_occlusion_flags specifically (swap/velocity/
    body-untrusted penalties still apply) -- the flags are still computed
    and included in diagnostics (so OCC/OCC1H? tags and the gray-tile-when-
    unused logic in the grid video still work), just not used to gray out
    the skeleton color via confidence. Useful when a detector (e.g. DWPose)
    handles occlusion well enough that the penalty just hides the very
    thing you're trying to visually inspect -- whether it's actually mixing
    up hands during close contact.
    """
    cam_ids = extraction["cam_ids"]
    landmarks_left = extraction["landmarks_left"]
    confidence_left = extraction["confidence_left"]
    landmarks_right = extraction["landmarks_right"]
    confidence_right = extraction["confidence_right"]
    landmarks_body = extraction["landmarks_body"]
    confidence_body = extraction["confidence_body"]
    landmarks_body_hand_left = extraction["landmarks_body_hand_left"]
    landmarks_body_hand_right = extraction["landmarks_body_hand_right"]
    landmark_confidence_left = extraction.get("landmark_confidence_left")
    landmark_confidence_right = extraction.get("landmark_confidence_right")
    landmark_confidence_body = extraction.get("landmark_confidence_body")

    frame_keys = hmv.discover_frames(take_dir, cam_ids[0])

    swap_results = hmv.detect_body_wrist_hand_swaps_sequence(
        landmarks_left, landmarks_right, landmarks_body, cam_ids, frame_keys, width, height,
        swap_margin_px=swap_margin_px,
    )
    swap_flags = {
        cam_id: {frame: r["swap_detected"] for frame, r in by_frame.items()}
        for cam_id, by_frame in swap_results.items()
    }
    landmarks_left, confidence_left, landmarks_right, confidence_right, n_swaps_corrected = (
        hmv.apply_swap_corrections(landmarks_left, confidence_left, landmarks_right, confidence_right, swap_flags)
    )
    landmark_confidence_left, landmark_confidence_right = _swap_landmark_confidence(
        landmark_confidence_left, landmark_confidence_right, swap_flags,
    )

    velocity_left = hmv.detect_high_velocity_frames(
        landmarks_left, cam_ids, frame_keys, width, height, velocity_thresh_px=velocity_thresh_px,
    )
    velocity_right = hmv.detect_high_velocity_frames(
        landmarks_right, cam_ids, frame_keys, width, height, velocity_thresh_px=velocity_thresh_px,
    )
    occlusion_flags = flag_hand_occlusions(
        landmarks_left, landmarks_right, cam_ids, frame_keys, iou_thresh=occlusion_iou_thresh,
    )
    single_hand_occlusion_flags = flag_single_hand_body_wrist_occlusion(
        landmarks_left, landmarks_right, landmarks_body_hand_left, landmarks_body_hand_right,
        cam_ids, frame_keys, width, height, bbox_margin_px=body_wrist_bbox_margin_px,
    )
    body_untrusted_flags = flag_body_pose_untrusted(
        landmarks_left, landmarks_right, landmarks_body, cam_ids, frame_keys, width, height,
        max_wrist_dist_px=body_trust_max_wrist_dist_px,
    )

    confidence_left, n_pen_l_vel = hmv.apply_confidence_penalty(confidence_left, velocity_left, penalty_factor)
    confidence_right, n_pen_r_vel = hmv.apply_confidence_penalty(confidence_right, velocity_right, penalty_factor)
    if penalize_occlusion:
        confidence_left, n_pen_l_occ = hmv.apply_confidence_penalty(confidence_left, occlusion_flags, penalty_factor)
        confidence_left, n_pen_l_occ1h = hmv.apply_confidence_penalty(
            confidence_left, single_hand_occlusion_flags, penalty_factor
        )
        confidence_right, n_pen_r_occ = hmv.apply_confidence_penalty(
            confidence_right, occlusion_flags, penalty_factor
        )
        confidence_right, n_pen_r_occ1h = hmv.apply_confidence_penalty(
            confidence_right, single_hand_occlusion_flags, penalty_factor
        )
    else:
        n_pen_l_occ = n_pen_l_occ1h = n_pen_r_occ = n_pen_r_occ1h = 0
    confidence_left, n_pen_l_body = hmv.apply_confidence_penalty(
        confidence_left, body_untrusted_flags, penalty_factor
    )
    confidence_right, n_pen_r_body = hmv.apply_confidence_penalty(
        confidence_right, body_untrusted_flags, penalty_factor
    )

    def flag_counts(flag_dict, key="flagged"):
        n = 0
        for by_frame in flag_dict.values():
            for v in by_frame.values():
                if (v.get(key) if isinstance(v, dict) else bool(v)):
                    n += 1
        return n

    diagnostics = {
        "cam_ids": cam_ids,
        "n_frames": len(frame_keys),
        "n_swaps_detected": sum(1 for r in swap_flags.values() for v in r.values() if v),
        "n_swaps_corrected": n_swaps_corrected,
        "n_velocity_flagged_left": flag_counts(velocity_left),
        "n_velocity_flagged_right": flag_counts(velocity_right),
        "n_occlusion_flagged": flag_counts(occlusion_flags),
        "n_single_hand_occlusion_flagged": flag_counts(single_hand_occlusion_flags),
        "n_body_untrusted_flagged": flag_counts(body_untrusted_flags),
        "n_confidence_penalized_left": n_pen_l_vel + n_pen_l_occ + n_pen_l_occ1h + n_pen_l_body,
        "n_confidence_penalized_right": n_pen_r_vel + n_pen_r_occ + n_pen_r_occ1h + n_pen_r_body,
        "swap_flags": swap_flags,
        "velocity_flags_left": {
            cid: {fk: v["flagged"] for fk, v in by_frame.items()} for cid, by_frame in velocity_left.items()
        },
        "velocity_flags_right": {
            cid: {fk: v["flagged"] for fk, v in by_frame.items()} for cid, by_frame in velocity_right.items()
        },
        "occlusion_flags": occlusion_flags,
        "single_hand_occlusion_flags": single_hand_occlusion_flags,
        "body_untrusted_flags": body_untrusted_flags,
    }

    return (
        landmarks_left, confidence_left, landmarks_right, confidence_right,
        landmarks_body, confidence_body, diagnostics,
        landmark_confidence_left, landmark_confidence_right, landmark_confidence_body,
    )
