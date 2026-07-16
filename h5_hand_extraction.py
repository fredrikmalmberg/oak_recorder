"""Extracts hand landmarks (both sides) from tmp/Testdata.h5's video frames using
either MediaPipe (Pose + Hands, mirroring h5_explore.ipynb's wrist-proximity
left/right disambiguation) or DWPose (dwpose_onnx.py -- body+both hands from one
forward pass, already correctly split left/right by the model itself). Both
write to the same normalized-landmark schema hand_multiview.py's triangulation
helpers expect, so downstream code doesn't care which detector produced it.
"""
import json
import os

import cv2
import numpy as np
import mediapipe as mp

import dwpose_onnx
import h5_dataset

POSE_LEFT_WRIST = 15
POSE_RIGHT_WRIST = 16
MIN_WRIST_VISIBILITY = 0.3

DEFAULT_CACHE_DIR = 'tmp/h5_extraction_cache'

# MediaPipe Pose's 33-point BlazePose topology uses different indices/point sets
# than DWPose's COCO-WholeBody body slice (indices 0-16, standard COCO-17 order) --
# this maps COCO-17 id -> BlazePose id so both detectors' body output can be
# reduced to the same shared schema (mirroring how both already agree on the
# 21-point hand schema), letting hand_multiview.combine_landmark_sources /
# triangulate_sequence work on body data unchanged. COCO's single "eye"/"ear" per
# side maps to BlazePose's plain eye/ear landmark, not its eye_inner/eye_outer.
BLAZEPOSE_TO_COCO17 = {
    0: 0, 1: 2, 2: 5, 3: 7, 4: 8, 5: 11, 6: 12, 7: 13, 8: 14,
    9: 15, 10: 16, 11: 23, 12: 24, 13: 25, 14: 26, 15: 27, 16: 28,
}  # COCO_id -> BlazePose_id


def _hand_wrist_xy(hand_landmarks):
    lm = hand_landmarks.landmark[0]
    return lm.x, lm.y


def _assign_hands_by_wrist_proximity(multi_hand_landmarks, multi_handedness, pose_landmarks):
    """Returns (left_hand_or_None, left_conf, right_hand_or_None, right_conf).
    Disambiguates by proximity to the Pose model's wrist landmarks rather than
    trusting MediaPipe Hands' own handedness label, which this project treats as
    unreliable throughout (see hand_multiview.HAND_LABEL) -- mirrors
    h5_explore.ipynb's assign_hands_by_pose_wrists, adapted to normalized
    coordinates (scale-independent, so no pixel conversion needed) and this
    module's (hand, confidence) return shape.
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
    """Remaps MediaPipe Pose's 33-point BlazePose landmarks through
    BLAZEPOSE_TO_COCO17 into a COCO-17-keyed dict, with `visibility` doubling as a
    genuine per-landmark confidence (unlike MediaPipe Hands, which only ever
    exposes one whole-hand score) -- symmetric with what DWPose already provides
    for hands via its per-keypoint SimCC scores.
    """
    body_landmarks, body_landmark_confidence = {}, {}
    for coco_id, blaze_id in BLAZEPOSE_TO_COCO17.items():
        lm = pose_landmarks.landmark[blaze_id]
        body_landmarks[str(coco_id)] = [lm.x, lm.y, lm.z]
        body_landmark_confidence[str(coco_id)] = float(lm.visibility)
    return body_landmarks, body_landmark_confidence


def extract_mediapipe_hands_from_h5(
    h5_path, cam_ids, frame_keys, cache_dir=DEFAULT_CACHE_DIR,
    hand_detect_max_width=1280, min_detection_confidence=0.5, min_tracking_confidence=0.5,
    force=False,
):
    cache_path = os.path.join(cache_dir, 'mediapipe_hands.json')
    if not force and os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            payload = json.load(f)
        if 'body' in payload:
            print(f'Loaded cached MediaPipe hands+body from {cache_path}')
            return (
                payload['left']['landmarks'], payload['left']['confidence'],
                payload['right']['landmarks'], payload['right']['confidence'],
                payload['body']['landmarks'], payload['body']['confidence'],
                payload['body']['landmark_confidence'],
            )
        print(f'Cached {cache_path} has no body data -- re-extracting to upgrade the cache')

    # Video mode (static_image_mode=False), not static-image mode -- this is a
    # video sequence, not a batch of unrelated stills, so MediaPipe can track
    # each hand/pose between frames instead of re-detecting from scratch every
    # frame. This requires one Pose/Hands instance PER CAMERA: the extraction
    # loop below interleaves frames from all 7 cameras through frame_key's outer
    # loop, and video mode's tracker assumes every process() call it receives is
    # the next consecutive frame of ONE continuous video -- a single shared
    # instance would corrupt its tracking state by treating cam01's frame as
    # following cam00's, etc.
    pose_by_cam = {
        cid: mp.solutions.pose.Pose(
            static_image_mode=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        for cid in cam_ids
    }
    hands_by_cam = {
        cid: mp.solutions.hands.Hands(
            static_image_mode=False, max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        for cid in cam_ids
    }

    landmarks_left = {cid: {} for cid in cam_ids}
    confidence_left = {cid: {} for cid in cam_ids}
    landmarks_right = {cid: {} for cid in cam_ids}
    confidence_right = {cid: {} for cid in cam_ids}
    landmarks_body = {cid: {} for cid in cam_ids}
    confidence_body = {cid: {} for cid in cam_ids}
    landmark_confidence_body = {cid: {} for cid in cam_ids}

    try:
        for frame_key, imgs in h5_dataset.iter_h5_frames(h5_path, cam_ids, frame_keys):
            for cam_id, img_bgr in imgs.items():
                h, w = img_bgr.shape[:2]
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                pose_result = pose_by_cam[cam_id].process(img_rgb)
                pose_landmarks = pose_result.pose_landmarks
                if pose_landmarks is not None:
                    body_lms, body_lm_conf = _extract_body_landmarks(pose_landmarks)
                    landmarks_body[cam_id][frame_key] = body_lms
                    landmark_confidence_body[cam_id][frame_key] = body_lm_conf
                    confidence_body[cam_id][frame_key] = float(np.mean(list(body_lm_conf.values())))

                # Downscale just for hand detection speed -- MediaPipe's landmark
                # x/y are normalized to whatever image was passed in, so this needs
                # no rescaling back afterward.
                scale = min(1.0, hand_detect_max_width / float(w))
                small_rgb = (
                    cv2.resize(img_rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                    if scale < 1.0 else img_rgb
                )
                hands_result = hands_by_cam[cam_id].process(small_rgb)

                left_hand, left_conf, right_hand, right_conf = _assign_hands_by_wrist_proximity(
                    hands_result.multi_hand_landmarks, hands_result.multi_handedness, pose_landmarks,
                )
                if left_hand is not None:
                    landmarks_left[cam_id][frame_key] = {
                        str(i): [lm.x, lm.y, lm.z] for i, lm in enumerate(left_hand.landmark)
                    }
                    confidence_left[cam_id][frame_key] = left_conf
                if right_hand is not None:
                    landmarks_right[cam_id][frame_key] = {
                        str(i): [lm.x, lm.y, lm.z] for i, lm in enumerate(right_hand.landmark)
                    }
                    confidence_right[cam_id][frame_key] = right_conf
    finally:
        for p in pose_by_cam.values():
            p.close()
        for h in hands_by_cam.values():
            h.close()

    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump({
            'left': {'landmarks': landmarks_left, 'confidence': confidence_left},
            'right': {'landmarks': landmarks_right, 'confidence': confidence_right},
            'body': {
                'landmarks': landmarks_body, 'confidence': confidence_body,
                'landmark_confidence': landmark_confidence_body,
            },
        }, f)
    print(f'Saved MediaPipe hands+body to {cache_path}')
    return (
        landmarks_left, confidence_left, landmarks_right, confidence_right,
        landmarks_body, confidence_body, landmark_confidence_body,
    )


def extract_dwpose_hands_from_h5(
    h5_path, cam_ids, frame_keys, det_model_path, pose_model_path,
    cache_dir=DEFAULT_CACHE_DIR, det_score_thr=0.3, force=False,
):
    cache_path = os.path.join(cache_dir, 'dwpose_hands.json')
    if not force and os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            payload = json.load(f)
        if 'body' in payload:
            print(f'Loaded cached DWPose hands+body from {cache_path}')
            return (
                payload['left']['landmarks'], payload['left']['confidence'], payload['left']['landmark_confidence'],
                payload['right']['landmarks'], payload['right']['confidence'], payload['right']['landmark_confidence'],
                payload['body']['landmarks'], payload['body']['confidence'], payload['body']['landmark_confidence'],
            )
        print(f'Cached {cache_path} has no body data -- re-extracting to upgrade the cache')

    detector = dwpose_onnx.DWposeOnnx(det_model_path, pose_model_path)

    landmarks_left = {cid: {} for cid in cam_ids}
    confidence_left = {cid: {} for cid in cam_ids}
    landmark_confidence_left = {cid: {} for cid in cam_ids}
    landmarks_right = {cid: {} for cid in cam_ids}
    confidence_right = {cid: {} for cid in cam_ids}
    landmark_confidence_right = {cid: {} for cid in cam_ids}
    landmarks_body = {cid: {} for cid in cam_ids}
    confidence_body = {cid: {} for cid in cam_ids}
    landmark_confidence_body = {cid: {} for cid in cam_ids}

    for frame_key, imgs in h5_dataset.iter_h5_frames(h5_path, cam_ids, frame_keys):
        for cam_id, img_bgr in imgs.items():
            h, w = img_bgr.shape[:2]
            people = detector.detect(img_bgr, det_score_thr=det_score_thr)
            if not people:
                continue
            keypoints, scores = max(people, key=lambda p: p[1][:17].mean())

            # Body slice (COCO-WholeBody indices 0-16, standard COCO-17 order) is
            # already computed for free above -- the same call already selected
            # the highest-scoring person using it (p[1][:17].mean()).
            body_kpts = keypoints[:17]
            body_scores = scores[:17]
            landmarks_body[cam_id][frame_key] = {
                str(i): [float(x / w), float(y / h), 0.0] for i, (x, y) in enumerate(body_kpts)
            }
            confidence_body[cam_id][frame_key] = float(body_scores.mean())
            landmark_confidence_body[cam_id][frame_key] = {str(i): float(s) for i, s in enumerate(body_scores)}

            left_kpts = keypoints[dwpose_onnx.LEFT_HAND_SLICE]
            left_scores = scores[dwpose_onnx.LEFT_HAND_SLICE]
            right_kpts = keypoints[dwpose_onnx.RIGHT_HAND_SLICE]
            right_scores = scores[dwpose_onnx.RIGHT_HAND_SLICE]

            landmarks_left[cam_id][frame_key] = {
                str(i): [float(x / w), float(y / h), 0.0] for i, (x, y) in enumerate(left_kpts)
            }
            confidence_left[cam_id][frame_key] = float(left_scores.mean())
            landmark_confidence_left[cam_id][frame_key] = {str(i): float(s) for i, s in enumerate(left_scores)}

            landmarks_right[cam_id][frame_key] = {
                str(i): [float(x / w), float(y / h), 0.0] for i, (x, y) in enumerate(right_kpts)
            }
            confidence_right[cam_id][frame_key] = float(right_scores.mean())
            landmark_confidence_right[cam_id][frame_key] = {str(i): float(s) for i, s in enumerate(right_scores)}

    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump({
            'left': {
                'landmarks': landmarks_left, 'confidence': confidence_left,
                'landmark_confidence': landmark_confidence_left,
            },
            'right': {
                'landmarks': landmarks_right, 'confidence': confidence_right,
                'landmark_confidence': landmark_confidence_right,
            },
            'body': {
                'landmarks': landmarks_body, 'confidence': confidence_body,
                'landmark_confidence': landmark_confidence_body,
            },
        }, f)
    print(f'Saved DWPose hands+body to {cache_path}')
    return (
        landmarks_left, confidence_left, landmark_confidence_left,
        landmarks_right, confidence_right, landmark_confidence_right,
        landmarks_body, confidence_body, landmark_confidence_body,
    )
