"""DWPose extraction (both hands + body in one forward pass, model-native
left/right split -- no wrist-proximity disambiguation needed, unlike
MediaPipe) for one recorded, aligned take, using this rig's own per-camera
calibration/undistortion -- same rig-specific adaptation pattern as
pose2d/extraction.py (which does the equivalent for MediaPipe), reusing
hand_pose/dwpose_onnx.py's existing ONNX wrapper as-is.

Output schema matches pose2d/extraction.py's exactly (cam_ids,
landmarks_left/right/body, confidence_left/right/body,
landmarks_body_hand_left/right) so pose2d.tracking.run_tracking and
pose2d.grid_video.render_pose2d_grid_video work completely UNCHANGED
regardless of which detector produced the data -- this project's
established "downstream code doesn't care which detector produced it"
convention (see hand_pose/h5_hand_extraction.py's own docstring).

One real, honest difference: DWPose's COCO-WholeBody body slice only has
the wrist (COCO ids 9/10), not MediaPipe Pose's extra pinky/index/thumb
proxies (a BlazePose-specific bonus MediaPipe happens to provide beyond
COCO-17) -- so landmarks_body_hand_left/right here has only a 'wrist'
entry, not 4. pose2d.tracking.flag_single_hand_body_wrist_occlusion already
iterates generically over however many points are given, so this degrades
gracefully to a weaker (but real, not fake) signal for DWPose.

Also, unlike MediaPipe Hands (which only ever reports a hand it's
confident enough to detect at all), RTMPose's whole-body model always
outputs coordinates for all 133 keypoints once a person is detected, with
confidence as the only signal of trustworthiness -- so a `min_confidence`
gate is applied here at extraction time (per hand/body, mean keypoint
score) before treating a part as "detected" at all, matching MediaPipe's
present/absent semantics so downstream flag rates are actually comparable
between detectors, not an artifact of DWPose always populating something.
"""
import json
import os

import cv2

from hand_pose import dwpose_onnx
from hand_pose import hand_multiview as hmv

DET_MODEL_PATH = os.path.join("models", "dwpose", "yolox_l.onnx")
POSE_MODEL_PATH = os.path.join("models", "dwpose", "dw-ll_ucoco_384.onnx")

BODY_SLICE = slice(0, 17)
LEFT_HAND_SLICE = dwpose_onnx.LEFT_HAND_SLICE
RIGHT_HAND_SLICE = dwpose_onnx.RIGHT_HAND_SLICE
BODY_LEFT_WRIST_COCO = 9
BODY_RIGHT_WRIST_COCO = 10

DEFAULT_MIN_CONFIDENCE = 0.3  # matches DWposeOnnx.detect's own det_score_thr default


def _kpts_to_landmarks(kpts, w, h):
    return {str(i): [float(x / w), float(y / h), 0.0] for i, (x, y) in enumerate(kpts)}


def extract_dwpose_for_take(
    take_dir, calib, cam_ids=None, det_score_thr=0.3, min_confidence=DEFAULT_MIN_CONFIDENCE, force=False,
):
    """Per-camera DWPose extraction over <take_dir>/aligned/<cam>/*.jpg,
    undistorted with each camera's own K/dist from `calib` -- same
    undistort-before-detect convention as pose2d.extraction, so the two
    detectors see identical input and any difference in results is
    attributable to the detector, not the preprocessing. Caches to
    <take_dir>/aligned/pose2d_dwpose/extraction.json -- a separate
    directory from MediaPipe's aligned/pose2d/, so both detectors' outputs
    coexist without collision.
    """
    out_dir = os.path.join(take_dir, "aligned", "pose2d_dwpose")
    os.makedirs(out_dir, exist_ok=True)
    cache_path = os.path.join(out_dir, "extraction.json")
    if not force and os.path.exists(cache_path):
        with open(cache_path) as f:
            payload = json.load(f)
        print(f"Loaded cached dwpose extraction from {cache_path}")
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
    landmarks_body_hand_left = {cid: {} for cid in cam_ids}
    landmarks_body_hand_right = {cid: {} for cid in cam_ids}
    landmark_confidence_left = {cid: {} for cid in cam_ids}
    landmark_confidence_right = {cid: {} for cid in cam_ids}
    landmark_confidence_body = {cid: {} for cid in cam_ids}

    detector = dwpose_onnx.DWposeOnnx(DET_MODEL_PATH, POSE_MODEL_PATH)

    for cam_id in cam_ids:
        frame_files = hmv.discover_frames(take_dir, cam_id)
        if not frame_files:
            continue
        c = calib[cam_id]
        w, h = hmv.get_image_size(take_dir, cam_id, frame_files[0])
        map1, map2 = hmv.build_undistort_maps(c["K"], c["dist"], w, h)

        for frame_file in frame_files:
            frame_path = os.path.join(take_dir, "aligned", cam_id, frame_file)
            img_bgr = cv2.imread(frame_path)
            if img_bgr is None:
                continue
            img_bgr = hmv.undistort_fast(img_bgr, map1, map2)

            people = detector.detect(img_bgr, det_score_thr=det_score_thr)
            if not people:
                continue
            keypoints, scores = max(people, key=lambda p: p[1][:17].mean())

            body_kpts, body_scores = keypoints[BODY_SLICE], scores[BODY_SLICE]
            body_conf = float(body_scores.mean())
            if body_conf >= min_confidence:
                landmarks_body[cam_id][frame_file] = _kpts_to_landmarks(body_kpts, w, h)
                confidence_body[cam_id][frame_file] = body_conf
                landmark_confidence_body[cam_id][frame_file] = {
                    str(i): float(s) for i, s in enumerate(body_scores)
                }
                body_lw, body_rw = body_kpts[BODY_LEFT_WRIST_COCO], body_kpts[BODY_RIGHT_WRIST_COCO]
                landmarks_body_hand_left[cam_id][frame_file] = {
                    "wrist": [float(body_lw[0] / w), float(body_lw[1] / h), 0.0]
                }
                landmarks_body_hand_right[cam_id][frame_file] = {
                    "wrist": [float(body_rw[0] / w), float(body_rw[1] / h), 0.0]
                }

            left_kpts, left_scores = keypoints[LEFT_HAND_SLICE], scores[LEFT_HAND_SLICE]
            left_conf = float(left_scores.mean())
            if left_conf >= min_confidence:
                landmarks_left[cam_id][frame_file] = _kpts_to_landmarks(left_kpts, w, h)
                confidence_left[cam_id][frame_file] = left_conf
                landmark_confidence_left[cam_id][frame_file] = {
                    str(i): float(s) for i, s in enumerate(left_scores)
                }

            right_kpts, right_scores = keypoints[RIGHT_HAND_SLICE], scores[RIGHT_HAND_SLICE]
            right_conf = float(right_scores.mean())
            if right_conf >= min_confidence:
                landmarks_right[cam_id][frame_file] = _kpts_to_landmarks(right_kpts, w, h)
                confidence_right[cam_id][frame_file] = right_conf
                landmark_confidence_right[cam_id][frame_file] = {
                    str(i): float(s) for i, s in enumerate(right_scores)
                }

        print(f"  {cam_id}: {len(frame_files)} frames processed "
              f"(left={len(landmarks_left[cam_id])}, right={len(landmarks_right[cam_id])}, "
              f"body={len(landmarks_body[cam_id])})")

    payload = {
        "cam_ids": cam_ids,
        "landmarks_left": landmarks_left, "confidence_left": confidence_left,
        "landmarks_right": landmarks_right, "confidence_right": confidence_right,
        "landmarks_body": landmarks_body, "confidence_body": confidence_body,
        "landmarks_body_hand_left": landmarks_body_hand_left,
        "landmarks_body_hand_right": landmarks_body_hand_right,
        "landmark_confidence_left": landmark_confidence_left,
        "landmark_confidence_right": landmark_confidence_right,
        "landmark_confidence_body": landmark_confidence_body,
    }
    with open(cache_path, "w") as f:
        json.dump(payload, f)
    print(f"Saved dwpose extraction to {cache_path}")
    return payload
