"""MediaPipe 2D on one camera's raw video (fast3d step 2, CPU).

Reads the camera's MJPEG forward once (no seeking, no JPEGs written), takes
the frame align_session would put in each aligned slot (slot_source_idx from
align_session.plan_alignment) and runs:
  - Pose (legacy solutions API, video mode, model complexity 1): 33 BlazePose
    points with visibility;
  - Hands (max 2, on the frame scaled to --hand-width px wide), assigned to
    left/right by distance to Pose's wrists (pose2d.extraction's
    _assign_hands_by_wrist_proximity);
  - FaceMesh (478 points, refine_landmarks) on a square crop around the face:
    3.5x the extent of the previous frame's face landmarks, else of Pose's
    face points (0-10), resampled to FACE_CROP px. MediaPipe's face detector
    misses faces that are small in the frame or that fill the image; a crop
    with the face at about a third of it works (face/face_fast.py).
Points are kept in raw (distorted) normalised image coordinates here; the
caller undistorts them (fast3d.triangulate.undistort_norm).

One call per camera; fast3d.run runs the cameras in parallel threads (MediaPipe
releases the GIL, so this scales).
"""
import cv2
import mediapipe as mp
import numpy as np

from pose2d.extraction import _assign_hands_by_wrist_proximity

FACE_CROP = 384
FACE_CROP_MULT = 3.5
POSE_FACE_IDS = list(range(11))


def _crop_affine(centre, side, out):
    s = out / side
    return np.array([[s, 0, out / 2 - centre[0] * s], [0, s, out / 2 - centre[1] * s]])


def extract_camera(video_path, slot_source_idx, width, height, hand_width=1280, progress=None):
    """Returns dict of arrays over slots T:
    body (T, 33, 3) normalised x, y (raw image) and BlazePose z, body_vis (T, 33);
    left/right (T, 21, 3), left_score/right_score (T,); face (T, 478, 3) normalised x, y (raw image)
    and MediaPipe's relative z, face_found (T,). Missing detections are NaN / 0."""
    cv2.setNumThreads(1)
    T = len(slot_source_idx)
    out = dict(body=np.full((T, 33, 3), np.nan, np.float32), body_vis=np.zeros((T, 33), np.float32),
               left=np.full((T, 21, 3), np.nan, np.float32), left_score=np.zeros(T, np.float32),
               right=np.full((T, 21, 3), np.nan, np.float32), right_score=np.zeros(T, np.float32),
               face=np.full((T, 478, 3), np.nan, np.float32), face_found=np.zeros(T, bool))
    pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1,
                                  min_detection_confidence=0.5, min_tracking_confidence=0.5)
    hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2,
                                     min_detection_confidence=0.5, min_tracking_confidence=0.5)
    face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True,
                                                min_detection_confidence=0.3)
    cap = cv2.VideoCapture(video_path)
    cur_idx, frame = -1, None
    prev_face = None
    hscale = min(1.0, hand_width / float(width))
    try:
        for t, idx in enumerate(slot_source_idx):
            if idx is None:
                continue
            while cur_idx < idx:                       # forward only; repeated idx reuses the frame
                ok, f = cap.read()
                if not ok:
                    break
                cur_idx, frame = cur_idx + 1, f
            if cur_idx != idx or frame is None:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            pr = pose.process(rgb)
            pl = pr.pose_landmarks
            if pl is not None:
                out['body'][t] = [[p.x, p.y, p.z] for p in pl.landmark]
                out['body_vis'][t] = [p.visibility for p in pl.landmark]

            small = cv2.resize(rgb, None, fx=hscale, fy=hscale, interpolation=cv2.INTER_AREA) if hscale < 1 else rgb
            hr = hands.process(small)
            lh, lc, rh, rc = _assign_hands_by_wrist_proximity(hr.multi_hand_landmarks, hr.multi_handedness, pl)
            if lh is not None:
                out['left'][t], out['left_score'][t] = [[p.x, p.y, p.z] for p in lh.landmark], lc
            if rh is not None:
                out['right'][t], out['right_score'][t] = [[p.x, p.y, p.z] for p in rh.landmark], rc

            # face crop: previous face, else Pose's face points
            src = None
            if prev_face is not None:
                src = prev_face
            elif pl is not None and out['body_vis'][t, POSE_FACE_IDS].mean() > 0.5:
                src = out['body'][t, POSE_FACE_IDS, :2] * (width, height)
            if src is not None:
                lo, hi = src.min(0), src.max(0)
                side = FACE_CROP_MULT * max(hi - lo) * (1.0 if prev_face is not None else 1.2)
                M = _crop_affine((lo + hi) / 2, max(side, 32.0), FACE_CROP)
                crop = cv2.warpAffine(rgb, M, (FACE_CROP, FACE_CROP), flags=cv2.INTER_LINEAR)
                fr = face_mesh.process(crop)
                if fr.multi_face_landmarks:
                    lm = np.array([[p.x, p.y, p.z] for p in fr.multi_face_landmarks[0].landmark], np.float32)
                    px = (lm[:, :2] * FACE_CROP - M[:, 2]) / M[0, 0]            # crop -> frame pixels
                    out['face'][t, :, :2] = px / (width, height)
                    out['face'][t, :, 2] = lm[:, 2]
                    out['face_found'][t] = True
                    prev_face = px
                else:
                    prev_face = None
            else:
                prev_face = None
            if progress is not None and t % 50 == 0:
                progress(t, T)
    finally:
        cap.release()
        pose.close()
        hands.close()
        face_mesh.close()
    return out
