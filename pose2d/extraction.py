"""MediaPipe Pose+Hands extraction (both hands + body) for one recorded,
aligned take on THIS rig's own cameras -- a companion to hand_pose/
h5_hand_extraction.py's extract_mediapipe_hands_from_h5, adapted for this
rig's real per-camera calibration/undistortion (calibrate.load_
calibration_output) instead of h5's averaged intrinsics.

BLAZEPOSE_TO_COCO17, _assign_hands_by_wrist_proximity, and
_extract_body_landmarks are ported near-verbatim from hand_pose/
h5_hand_extraction.py (see that file for full rationale on each) rather
than imported -- this project's existing convention for a rig-specific
module built on a different data source than an h5-tied one (see
hand_pose/grid_video.py's own docstring for the same reasoning).
"""
import json
import os

import cv2
import mediapipe as mp

from hand_pose import hand_multiview as hmv

POSE_LEFT_WRIST = 15
POSE_RIGHT_WRIST = 16
MIN_WRIST_VISIBILITY = 0.3

# COCO-17 ids 13-16 (knees/ankles -- hips 11/12 are kept as part of the
# upper body/torso) -- excluded from confidence_body's mean below so a
# poorly-visible/occluded leg (a desk, out of frame -- not meaningful for
# sign-language content anyway) doesn't drag down the whole-frame confidence
# used to gate camera selection for the landmarks that actually matter (see
# pose2d.triangulation.LANDMARK_IDS_BODY, which excludes the same ids from
# triangulation itself).
BODY_LEG_COCO_IDS = {"13", "14", "15", "16"}

# COCO_id -> BlazePose_id. Ported from hand_pose/h5_hand_extraction.py:31-34.
BLAZEPOSE_TO_COCO17 = {
    0: 0, 1: 2, 2: 5, 3: 7, 4: 8, 5: 11, 6: 12, 7: 13, 8: 14,
    9: 15, 10: 16, 11: 23, 12: 24, 13: 25, 14: 26, 15: 27, 16: 28,
}

# BlazePose's 33-point set has 4 points near each hand (wrist + 3 fingertip
# proxies: pinky/index/thumb), not just the wrist -- COCO-17 only has room
# for the wrist (BLAZEPOSE_TO_COCO17 above), so the other 3 are discarded by
# that reduction. Kept here separately (not folded into the COCO-17 body
# schema, to keep that schema's DWPose-interop meaning intact) specifically
# for pose2d.tracking's single-hand occlusion check, which needs a denser
# sparse-point set than one wrist per side to reliably catch occlusion (see
# that module for why the wrist alone missed real cases).
LEFT_HAND_POSE_IDS = {"wrist": 15, "pinky": 17, "index": 19, "thumb": 21}
RIGHT_HAND_POSE_IDS = {"wrist": 16, "pinky": 18, "index": 20, "thumb": 22}


def _hand_wrist_xy(hand_landmarks):
    lm = hand_landmarks.landmark[0]
    return lm.x, lm.y


def _assign_hands_by_wrist_proximity(multi_hand_landmarks, multi_handedness, pose_landmarks):
    """Ported from hand_pose/h5_hand_extraction.py:42-81. Disambiguates
    MediaPipe Hands' up-to-2 unordered detections by proximity to the Pose
    model's own wrist landmarks, rather than trusting MediaPipe Hands' own
    (unreliable for this rig's un-mirrored framing) L/R label -- see
    hand_multiview.HAND_LABEL for the same distrust elsewhere in this
    project. Returns (left_hand_or_None, left_conf, right_hand_or_None, right_conf).
    """
    if not multi_hand_landmarks:
        return None, 0.0, None, 0.0
    hands_conf = [float(h.classification[0].score) for h in multi_handedness]

    if pose_landmarks is not None:
        lw = pose_landmarks.landmark[POSE_LEFT_WRIST]
        rw = pose_landmarks.landmark[POSE_RIGHT_WRIST]
        if lw.visibility >= MIN_WRIST_VISIBILITY and rw.visibility >= MIN_WRIST_VISIBILITY:
            if len(multi_hand_landmarks) == 1:
                hx, hy = _hand_wrist_xy(multi_hand_landmarks[0])
                dl = (hx - lw.x) ** 2 + (hy - lw.y) ** 2
                dr = (hx - rw.x) ** 2 + (hy - rw.y) ** 2
                if dl <= dr:
                    return multi_hand_landmarks[0], hands_conf[0], None, 0.0
                return None, 0.0, multi_hand_landmarks[0], hands_conf[0]

            h0, h1 = multi_hand_landmarks[0], multi_hand_landmarks[1]
            x0, y0 = _hand_wrist_xy(h0)
            x1, y1 = _hand_wrist_xy(h1)
            d0l = (x0 - lw.x) ** 2 + (y0 - lw.y) ** 2
            d0r = (x0 - rw.x) ** 2 + (y0 - rw.y) ** 2
            d1l = (x1 - lw.x) ** 2 + (y1 - lw.y) ** 2
            d1r = (x1 - rw.x) ** 2 + (y1 - rw.y) ** 2
            if d0l + d1r <= d0r + d1l:
                return h0, hands_conf[0], h1, hands_conf[1]
            return h1, hands_conf[1], h0, hands_conf[0]

    # No usable pose wrists this frame -- fall back to returning detections in order.
    if len(multi_hand_landmarks) == 1:
        return multi_hand_landmarks[0], hands_conf[0], None, 0.0
    return multi_hand_landmarks[0], hands_conf[0], multi_hand_landmarks[1], hands_conf[1]


def _extract_body_landmarks(pose_landmarks):
    """Ported from hand_pose/h5_hand_extraction.py:84-96."""
    body_landmarks, body_landmark_confidence = {}, {}
    for coco_id, blaze_id in BLAZEPOSE_TO_COCO17.items():
        lm = pose_landmarks.landmark[blaze_id]
        body_landmarks[str(coco_id)] = [lm.x, lm.y, lm.z]
        body_landmark_confidence[str(coco_id)] = float(lm.visibility)
    return body_landmarks, body_landmark_confidence


def _extract_sparse_hand_points(pose_landmarks, ids):
    return {
        name: [pose_landmarks.landmark[idx].x, pose_landmarks.landmark[idx].y, pose_landmarks.landmark[idx].z]
        for name, idx in ids.items()
    }


def extract_pose2d_for_take(
    take_dir, calib, cam_ids=None,
    hand_detect_max_width=1280, min_detection_confidence=0.5, min_tracking_confidence=0.5,
    force=False,
):
    """Per-camera MediaPipe Pose + Hands(max_num_hands=2) extraction over
    <take_dir>/aligned/<cam>/*.jpg, undistorted with each camera's OWN
    K/dist from `calib` (calibrate.load_calibration_output's schema) --
    unlike h5_hand_extraction.py, which only had averaged intrinsics
    available for this rig. Caches to <take_dir>/aligned/pose2d/extraction
    .json -- deliberately separate from session_triangulation.py's
    single-hand aligned/landmarks.json so both pipelines coexist.

    Loop order is camera-outer, frame-inner (each camera's own aligned
    frames processed in temporal order start to finish) -- unlike h5_hand_
    extraction.py's frame-outer loop (which needed ALL cameras' Pose/Hands
    instances alive simultaneously since it interleaves cameras within one
    frame_key), this lets each camera's instances be created and closed one
    camera at a time, since video-mode tracking only requires THAT camera's
    own calls to stay temporally ordered -- not what happens between
    cameras.

    Returns dict with keys 'landmarks_left', 'confidence_left',
    'landmarks_right', 'confidence_right', 'landmarks_body',
    'confidence_body' (mean per frame), 'landmark_confidence_body'
    (per-landmark), 'landmarks_body_hand_left'/'landmarks_body_hand_right'
    (each dict[cam_id][frame_key] -> {'wrist'/'pinky'/'index'/'thumb':
    [x,y,z]} -- BlazePose's 4 sparse points nearest that hand, for pose2d.
    tracking's occlusion check) -- each dict[cam_id][frame_key] -> ...,
    except confidence_left/right which are dict[cam_id][frame_key] -> float.
    """
    out_dir = os.path.join(take_dir, "aligned", "pose2d")
    os.makedirs(out_dir, exist_ok=True)
    cache_path = os.path.join(out_dir, "extraction.json")
    if not force and os.path.exists(cache_path):
        with open(cache_path) as f:
            payload = json.load(f)
        print(f"Loaded cached pose2d extraction from {cache_path}")
        return payload

    if cam_ids is None:
        cam_ids = hmv.discover_cameras(take_dir)
    cam_ids = [c for c in cam_ids if c in calib]

    landmarks_left = {cid: {} for cid in cam_ids}
    confidence_left = {cid: {} for cid in cam_ids}
    landmarks_right = {cid: {} for cid in cam_ids}
    confidence_right = {cid: {} for cid in cam_ids}
    landmarks_body = {cid: {} for cid in cam_ids}
    confidence_body = {cid: {} for cid in cam_ids}
    landmark_confidence_body = {cid: {} for cid in cam_ids}
    landmarks_body_hand_left = {cid: {} for cid in cam_ids}
    landmarks_body_hand_right = {cid: {} for cid in cam_ids}

    for cam_id in cam_ids:
        frame_files = hmv.discover_frames(take_dir, cam_id)
        if not frame_files:
            continue
        c = calib[cam_id]
        w, h = hmv.get_image_size(take_dir, cam_id, frame_files[0])
        map1, map2 = hmv.build_undistort_maps(c["K"], c["dist"], w, h)

        pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        hands = mp.solutions.hands.Hands(
            static_image_mode=False, max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        try:
            for frame_file in frame_files:
                frame_path = os.path.join(take_dir, "aligned", cam_id, frame_file)
                img_bgr = cv2.imread(frame_path)
                if img_bgr is None:
                    continue
                img_bgr = hmv.undistort_fast(img_bgr, map1, map2)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                pose_result = pose.process(img_rgb)
                pose_landmarks = pose_result.pose_landmarks
                if pose_landmarks is not None:
                    body_lms, body_lm_conf = _extract_body_landmarks(pose_landmarks)
                    landmarks_body[cam_id][frame_file] = body_lms
                    landmark_confidence_body[cam_id][frame_file] = body_lm_conf
                    upper_body_conf = [v for k, v in body_lm_conf.items() if k not in BODY_LEG_COCO_IDS]
                    confidence_body[cam_id][frame_file] = float(
                        sum(upper_body_conf) / len(upper_body_conf)
                    )
                    landmarks_body_hand_left[cam_id][frame_file] = _extract_sparse_hand_points(
                        pose_landmarks, LEFT_HAND_POSE_IDS
                    )
                    landmarks_body_hand_right[cam_id][frame_file] = _extract_sparse_hand_points(
                        pose_landmarks, RIGHT_HAND_POSE_IDS
                    )

                # Downscale just for hand detection speed -- MediaPipe's landmark
                # x/y are normalized to whatever image was passed in, so this
                # needs no rescaling back afterward.
                scale = min(1.0, hand_detect_max_width / float(w))
                small_rgb = (
                    cv2.resize(img_rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                    if scale < 1.0 else img_rgb
                )
                hands_result = hands.process(small_rgb)

                left_hand, left_conf, right_hand, right_conf = _assign_hands_by_wrist_proximity(
                    hands_result.multi_hand_landmarks, hands_result.multi_handedness, pose_landmarks,
                )
                if left_hand is not None:
                    landmarks_left[cam_id][frame_file] = {
                        str(i): [lm.x, lm.y, lm.z] for i, lm in enumerate(left_hand.landmark)
                    }
                    confidence_left[cam_id][frame_file] = left_conf
                if right_hand is not None:
                    landmarks_right[cam_id][frame_file] = {
                        str(i): [lm.x, lm.y, lm.z] for i, lm in enumerate(right_hand.landmark)
                    }
                    confidence_right[cam_id][frame_file] = right_conf
        finally:
            pose.close()
            hands.close()
        print(f"  {cam_id}: {len(frame_files)} frames processed "
              f"(left={len(landmarks_left[cam_id])}, right={len(landmarks_right[cam_id])}, "
              f"body={len(landmarks_body[cam_id])})")

    payload = {
        "cam_ids": cam_ids,
        "landmarks_left": landmarks_left, "confidence_left": confidence_left,
        "landmarks_right": landmarks_right, "confidence_right": confidence_right,
        "landmarks_body": landmarks_body, "confidence_body": confidence_body,
        "landmark_confidence_body": landmark_confidence_body,
        "landmarks_body_hand_left": landmarks_body_hand_left,
        "landmarks_body_hand_right": landmarks_body_hand_right,
    }
    with open(cache_path, "w") as f:
        json.dump(payload, f)
    print(f"Saved pose2d extraction to {cache_path}")
    return payload
