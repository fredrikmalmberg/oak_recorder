import sys
import os
import copy
import glob
import json
import time
import types
import pickle
import random
import itertools

import numpy as np
import cv2
import h5py
import mediapipe as mp
import matplotlib.pyplot as plt
import viser
from viser import transforms as tf


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

class ChameleonCalibrationV1:
    """Stub for unpickling when calib package is not installed."""
    pass


def load_chameleon_calibration(h5_path):
    calib_mod = types.ModuleType('calib.chameleon_calibration')
    calib_mod.ChameleonCalibrationV1 = ChameleonCalibrationV1
    sys.modules['calib.chameleon_calibration'] = calib_mod
    sys.modules['calib'] = types.ModuleType('calib')

    with h5py.File(h5_path, 'r') as f:
        cal = pickle.loads(bytes(f['calibration/cal_pickle'][()]))
        camera_ids = sorted(f['rec'].keys())
    return cal, camera_ids


def build_camera_intrinsics(intrin_vec):
    cx, cy, fx, fy = intrin_vec[2], intrin_vec[3], intrin_vec[4], intrin_vec[5]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    dist = intrin_vec[6:14].astype(np.float64)
    return K, dist


def average_intrinsics(cal, camera_ids, skip_ids=frozenset({'cam02'})):
    Ks, dists, widths, heights = [], [], [], []
    for i, cam_id in enumerate(camera_ids):
        if cam_id in skip_ids:
            continue
        K, dist = build_camera_intrinsics(cal.intrinsics[i])
        Ks.append(K)
        dists.append(dist)
        widths.append(cal.intrinsics[i][0])
        heights.append(cal.intrinsics[i][1])
    return {
        'K': np.mean(np.stack(Ks), axis=0),
        'dist': np.mean(np.stack(dists), axis=0),
        'width': float(np.mean(widths)),
        'height': float(np.mean(heights)),
    }


def scale_intrinsics(K, from_size, to_size):
    from_w, from_h = from_size
    to_w, to_h = to_size
    sx = to_w / from_w
    sy = to_h / from_h
    K_scaled = K.copy()
    K_scaled[0, 0] *= sx  # fx
    K_scaled[0, 2] *= sx  # cx
    K_scaled[1, 1] *= sy  # fy
    K_scaled[1, 2] *= sy  # cy
    return K_scaled


def load_all_camera_calibration(cal, camera_ids):
    """Per-camera intrinsics *and* ground-truth extrinsics, unlike
    average_intrinsics (which discards individual cameras' extrinsics entirely --
    that function exists for the cam0-cam3 recording rig, which has no matching
    calibration of its own). Only valid for a h5 file whose own rig this `cal`
    belongs to (e.g. tmp/Testdata.h5's cam00-cam06), where cal.cam_to_world[i] is
    real, trustworthy ground truth and no pose estimation is needed at all.
    """
    result = {}
    for i, cam_id in enumerate(camera_ids):
        K, dist = build_camera_intrinsics(cal.intrinsics[i])
        cam_to_world = np.asarray(cal.cam_to_world[i], dtype=np.float64)
        T_wc = np.linalg.inv(cam_to_world)
        width, height = cal.intrinsics[i][0], cal.intrinsics[i][1]
        result[cam_id] = {
            'K': K, 'dist': dist,
            'R': T_wc[:3, :3], 't': T_wc[:3, 3],
            'width': float(width), 'height': float(height),
        }
    return result


def undistort_landmarks(landmarks, calib):
    """Corrects each camera's 2D landmarks for lens distortion at the pixel
    level (cv2.undistortPoints(..., P=K)), then re-normalizes back to [0,1]
    image-fraction coordinates -- the same schema every function in this module
    expects, so nothing downstream needs to change. This matters because
    triangulate_dlt's projection matrices (P = K @ [R|t]) assume a pure pinhole
    model with no distortion term, but MediaPipe/DWPose detect landmarks in the
    ORIGINAL, distorted image's pixel space -- feeding those in directly is a
    real geometric inconsistency, not just an approximation, and this rig's
    distortion is non-trivial (rational model, k2/k4 up to ~70 on some cameras).

    Only use the *output* of this function for triangulation-facing numeric work
    (2D pixel-distance comparisons, DLT, etc). Skeleton overlays drawn onto the
    actual (distorted) video frames should keep using the raw, un-undistorted
    landmarks, or the drawing will no longer line up with the pixels on screen --
    undistorting the landmarks doesn't undistort the image they're drawn on.

    landmarks: dict[cam_id][frame_key] -> {lm_id: [x, y, z]}, normalized to that
    camera's own (distorted) image. calib: dict[cam_id] -> {'K', 'dist', 'width',
    'height', ...} (load_all_camera_calibration's schema). Cameras absent from
    calib are skipped (e.g. a landmarks dict spanning a superset of cameras).
    """
    result = {}
    for cam_id, by_frame in landmarks.items():
        if cam_id not in calib:
            continue
        K, dist = calib[cam_id]['K'], calib[cam_id]['dist']
        w, h = calib[cam_id]['width'], calib[cam_id]['height']
        result[cam_id] = {}
        for frame_key, lms in by_frame.items():
            if not lms:
                continue
            lm_ids = sorted(lms.keys(), key=int)
            pts = np.array(
                [[lms[lm_id][0] * w, lms[lm_id][1] * h] for lm_id in lm_ids], dtype=np.float64,
            ).reshape(-1, 1, 2)
            undist = cv2.undistortPoints(pts, K, dist, P=K).reshape(-1, 2)
            result[cam_id][frame_key] = {
                lm_id: [float(undist[i, 0] / w), float(undist[i, 1] / h), lms[lm_id][2]]
                for i, lm_id in enumerate(lm_ids)
            }
    return result


# ---------------------------------------------------------------------------
# Camera / frame discovery
# ---------------------------------------------------------------------------

def discover_cameras(session_dir):
    aligned_dir = os.path.join(session_dir, 'aligned')
    cam_dirs = [d for d in glob.glob(os.path.join(aligned_dir, 'cam*')) if os.path.isdir(d)]
    cam_ids = [os.path.basename(d) for d in cam_dirs]
    return sorted(cam_ids, key=lambda name: int(name.replace('cam', '')))


def discover_frames(session_dir, cam_id):
    cam_dir = os.path.join(session_dir, 'aligned', cam_id)
    return sorted(f for f in os.listdir(cam_dir) if f.endswith('.jpg'))


def discover_all_frames(session_dir, cam_ids):
    return {cam_id: discover_frames(session_dir, cam_id) for cam_id in cam_ids}


def get_image_size(session_dir, cam_id, frame_file):
    path = os.path.join(session_dir, 'aligned', cam_id, frame_file)
    img = cv2.imread(path)
    h, w = img.shape[:2]
    return w, h


# ---------------------------------------------------------------------------
# Undistortion (maps cached once, reused across all frames of a session)
# ---------------------------------------------------------------------------

def build_undistort_maps(K, dist, width, height):
    return cv2.initUndistortRectifyMap(K, dist, None, K, (int(width), int(height)), cv2.CV_16SC2)


def undistort_fast(image, map1, map2):
    return cv2.remap(image, map1, map2, cv2.INTER_LINEAR)


# ---------------------------------------------------------------------------
# MediaPipe hand extraction + JSON caching
# ---------------------------------------------------------------------------

# Only one hand ever appears in these recordings, and it is always the signer's left
# hand. MediaPipe's own Left/Right classification assumes a mirrored (selfie-view)
# input and is unreliable for this rig's un-mirrored camera framing, so its label is
# never read or trusted anywhere in this module -- only its confidence score is kept,
# as a coarse detection-quality signal. Any code that needs a hand identity should use
# this constant rather than `results.multi_handedness[...].classification[0].label`.
HAND_LABEL = 'Left'


def landmark_dict_to_pixel_xy(lms, width, height):
    return {int(lm_id): np.array([xyz[0] * width, xyz[1] * height], dtype=np.float64)
            for lm_id, xyz in lms.items()}


def extract_single_frame_landmarks(hands, image_bgr):
    """Runs one already-undistorted BGR frame through an existing `mp.solutions.hands.Hands`
    instance and returns (landmarks_dict, confidence) in the same schema
    `extract_hand_landmarks_for_session` writes to JSON (normalized [0,1] xyz per landmark
    id, plus a handedness confidence score -- never the label, see HAND_LABEL). Returns
    (None, 0.0) if no hand was detected. For live per-frame use (a long-lived `Hands`
    instance reused across frames/cameras), unlike the session extractor which owns a
    `Hands` instance for the duration of a whole batch job.
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if not results.multi_hand_landmarks:
        return None, 0.0
    hand_landmarks = results.multi_hand_landmarks[0]
    landmarks = {str(i): [lm.x, lm.y, lm.z] for i, lm in enumerate(hand_landmarks.landmark)}
    confidence = 0.0
    if results.multi_handedness:
        confidence = float(results.multi_handedness[0].classification[0].score)
    return landmarks, confidence


def extract_hand_landmarks_for_session(
    session_dir, K, dist, cam_ids=None,
    max_num_hands=1, min_detection_confidence=0.2, min_tracking_confidence=0.5,
    force=False,
):
    aligned_dir = os.path.join(session_dir, 'aligned')
    landmarks_path = os.path.join(aligned_dir, 'landmarks.json')
    confidence_path = os.path.join(aligned_dir, 'confidence.json')

    if not force and os.path.exists(landmarks_path) and os.path.exists(confidence_path):
        with open(landmarks_path, 'r') as f:
            landmarks = json.load(f)
        with open(confidence_path, 'r') as f:
            confidence = json.load(f)
        print(f'Loaded cached landmarks/confidence from {aligned_dir}')
        return landmarks, confidence

    if cam_ids is None:
        cam_ids = discover_cameras(session_dir)
    frames_by_cam = discover_all_frames(session_dir, cam_ids)

    sample_cam = cam_ids[0]
    width, height = get_image_size(session_dir, sample_cam, frames_by_cam[sample_cam][0])
    map1, map2 = build_undistort_maps(K, dist, width, height)

    hands = mp.solutions.hands.Hands(
        max_num_hands=max_num_hands,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    landmarks = {}
    confidence = {}
    try:
        for cam_id in cam_ids:
            landmarks[cam_id] = {}
            confidence[cam_id] = {}
            for frame_file in frames_by_cam[cam_id]:
                frame_path = os.path.join(session_dir, 'aligned', cam_id, frame_file)
                image = cv2.imread(frame_path)
                if image is None:
                    continue
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = undistort_fast(image, map1, map2)
                results = hands.process(image)
                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    landmarks[cam_id][frame_file] = {
                        str(i): [lm.x, lm.y, lm.z] for i, lm in enumerate(hand_landmarks.landmark)
                    }
                if results.multi_handedness:
                    # Score only -- see HAND_LABEL, we don't trust classification[0].label here.
                    confidence[cam_id][frame_file] = float(
                        results.multi_handedness[0].classification[0].score
                    )
    finally:
        hands.close()

    with open(landmarks_path, 'w') as f:
        json.dump(landmarks, f)
    with open(confidence_path, 'w') as f:
        json.dump(confidence, f)
    print(f'Saved landmarks/confidence to {aligned_dir}')
    return landmarks, confidence


# ---------------------------------------------------------------------------
# Detection sampling / overlay sanity-check plots
# ---------------------------------------------------------------------------

def sample_detected_frames(landmarks, cam_ids, n_samples=8, seed=None):
    pool = [(cam_id, frame) for cam_id in cam_ids for frame in landmarks.get(cam_id, {})]
    if not pool:
        return []
    rng = random.Random(seed)
    return rng.sample(pool, min(n_samples, len(pool)))


def draw_hand_skeleton(image, pixel_xy, color=(0, 255, 0), point_color=(255, 0, 0), thickness=2,
                        connections=None):
    # HAND_CONNECTIONS is defined later in this file (viser section) -- resolved
    # here at call time, not as a default-argument value, since default values
    # are evaluated at `def` time and HAND_CONNECTIONS wouldn't exist yet then.
    if connections is None:
        connections = HAND_CONNECTIONS
    for a, b in connections:
        if a in pixel_xy and b in pixel_xy:
            pa = tuple(np.round(pixel_xy[a]).astype(int))
            pb = tuple(np.round(pixel_xy[b]).astype(int))
            cv2.line(image, pa, pb, color, thickness)
    for pt in pixel_xy.values():
        p = tuple(np.round(pt).astype(int))
        cv2.circle(image, p, 4, point_color, -1)
    return image


def plot_sample_detections(
    session_dir, landmarks, confidence, K, dist, width, height, samples, cols=4,
):
    if not samples:
        print('No detected frames to sample.')
        return None

    map1, map2 = build_undistort_maps(K, dist, width, height)
    rows = int(np.ceil(len(samples) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
    axes = axes.ravel()

    for ax, (cam_id, frame) in zip(axes, samples):
        frame_path = os.path.join(session_dir, 'aligned', cam_id, frame)
        image = cv2.imread(frame_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = undistort_fast(image, map1, map2)
        img_h, img_w = image.shape[:2]

        lms = landmarks.get(cam_id, {}).get(frame)
        if lms:
            pixel_xy = landmark_dict_to_pixel_xy(lms, img_w, img_h)
            draw_hand_skeleton(image, pixel_xy)
        conf = confidence.get(cam_id, {}).get(frame)
        conf_str = f'{conf:.2f}' if conf is not None else 'n/a'

        frame_idx = os.path.splitext(frame)[0]
        ax.imshow(image)
        ax.set_title(f'{cam_id}, img idx {frame_idx} (conf {conf_str})')
        ax.axis('off')

    for ax in axes[len(samples):]:
        ax.axis('off')

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Multi-frame correspondence pooling
# ---------------------------------------------------------------------------

def simultaneous_detection_frames(landmarks, confidence, cam_a, cam_b, min_confidence=0.5):
    frames_a = landmarks.get(cam_a, {})
    frames_b = landmarks.get(cam_b, {})
    conf_a = confidence.get(cam_a, {})
    conf_b = confidence.get(cam_b, {})
    common = [
        frame for frame in frames_a
        if frame in frames_b
        and conf_a.get(frame, 0) >= min_confidence
        and conf_b.get(frame, 0) >= min_confidence
    ]
    return sorted(common)


# Pose estimation uses only the wrist landmark as a correspondence: fingertip/finger
# landmarks dominate the 21-point set but carry far more noise (fast motion, motion
# blur, self-occlusion during signing), which swamps the fundamental-matrix RANSAC fit.
# Restricting to the wrist -- a slower-moving, harder-to-occlude point -- roughly
# triples the inlier ratio in practice. The full 21-point set is still used for the
# final per-frame triangulation (triangulate_sequence); only pose estimation is affected.
WRIST_LANDMARK_ID = 0
POSE_ESTIMATION_LANDMARK_IDS = {WRIST_LANDMARK_ID}


def score_pair_conditioning(
    landmarks, confidence, cam_a, cam_b, K, width, height,
    min_confidence=0.5, max_pool_frames=200, ransac_threshold=3.0,
    landmark_ids=POSE_ESTIMATION_LANDMARK_IDS,
):
    """Fundamental-matrix RANSAC inlier count for a pair -- a direct measure of how
    well-conditioned its correspondences are for essential-matrix pose estimation,
    unlike raw simultaneous-detection-frame count (a pair can have many common frames
    whose matched points are clustered in a small region of one view, which is close
    to degenerate for epipolar geometry)."""
    common_frames = simultaneous_detection_frames(landmarks, confidence, cam_a, cam_b, min_confidence)
    if len(common_frames) < 8:
        return {'cam_a': cam_a, 'cam_b': cam_b, 'common_frames': len(common_frames), 'inlier_count': 0}
    pooled_frames = select_frames_for_pooling(common_frames, max_pool_frames)
    src_pts, dst_pts, _ = pool_correspondences(landmarks, cam_a, cam_b, pooled_frames, width, height, landmark_ids)
    if len(src_pts) < 8:
        return {'cam_a': cam_a, 'cam_b': cam_b, 'common_frames': len(common_frames), 'inlier_count': 0}
    _, mask_f = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC, ransac_threshold, 0.999)
    inlier_count = int(mask_f.sum()) if mask_f is not None else 0
    return {'cam_a': cam_a, 'cam_b': cam_b, 'common_frames': len(common_frames), 'inlier_count': inlier_count}


def rank_camera_pairs_by_conditioning(
    cam_ids, landmarks, confidence, K, width, height,
    min_confidence=0.5, max_pool_frames=200, ransac_threshold=3.0,
    landmark_ids=POSE_ESTIMATION_LANDMARK_IDS,
):
    scores = [
        score_pair_conditioning(
            landmarks, confidence, cam_a, cam_b, K, width, height,
            min_confidence, max_pool_frames, ransac_threshold, landmark_ids,
        )
        for cam_a, cam_b in itertools.combinations(cam_ids, 2)
    ]
    return sorted(scores, key=lambda s: s['inlier_count'], reverse=True)


def select_frames_for_pooling(frames, max_frames=200):
    if len(frames) <= max_frames:
        return list(frames)
    indices = sorted({int(round(i)) for i in np.linspace(0, len(frames) - 1, max_frames)})
    return [frames[i] for i in indices]


def pool_correspondences(landmarks, cam_a, cam_b, frames, width, height, landmark_ids=None):
    src_pts, dst_pts, provenance = [], [], []
    for frame in frames:
        lms_a = landmarks[cam_a].get(frame)
        lms_b = landmarks[cam_b].get(frame)
        if not lms_a or not lms_b:
            continue
        for lm_id_str, xyz_a in lms_a.items():
            if landmark_ids is not None and int(lm_id_str) not in landmark_ids:
                continue
            xyz_b = lms_b.get(lm_id_str)
            if xyz_b is None:
                continue
            src_pts.append([xyz_a[0] * width, xyz_a[1] * height])
            dst_pts.append([xyz_b[0] * width, xyz_b[1] * height])
            provenance.append((frame, int(lm_id_str)))
    return np.array(src_pts, dtype=np.float64), np.array(dst_pts, dtype=np.float64), provenance


# ---------------------------------------------------------------------------
# Reference-pair pose via essential matrix (generalizes triangulation.ipynb)
# ---------------------------------------------------------------------------

def estimate_relative_pose(src_pts, dst_pts, K, ransac_threshold=3.0):
    F, mask_f = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC, ransac_threshold, 0.999)
    E = K.T @ F @ K
    # Pass the fundamental-matrix RANSAC mask through so recoverPose's cheirality check
    # only considers already-verified epipolar inliers, instead of filtering nothing.
    _, R, t, mask_pose = cv2.recoverPose(E, src_pts, dst_pts, K, mask=mask_f.copy())
    inlier_mask = mask_pose.ravel() > 0
    return R, t, inlier_mask


def build_projection_matrix(K, R, t):
    return K @ np.hstack([R, np.asarray(t).reshape(3, 1)])


def triangulate_pair(P1, P2, pts1, pts2):
    points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    return cv2.convertPointsFromHomogeneous(points_4d.T).reshape(-1, 3)


def estimate_reference_pair_world(cam_a, cam_b, R, t, K, src_pts, dst_pts, inlier_mask, provenance):
    # recoverPose(E, src, dst, K) returns the src->dst transform (x_dst = R @ x_src + t).
    # cam_a (src) is the world origin, so P_a = K[I|0] and P_b = K[R|t].
    P_a = build_projection_matrix(K, np.eye(3), np.zeros(3))
    P_b = build_projection_matrix(K, R, t)
    src_in = src_pts[inlier_mask]
    dst_in = dst_pts[inlier_mask]
    provenance_in = [p for p, keep in zip(provenance, inlier_mask) if keep]
    points_3d = triangulate_pair(P_a, P_b, src_in, dst_in)
    return {
        'poses': {cam_a: (np.eye(3), np.zeros((3, 1))), cam_b: (R, np.asarray(t).reshape(3, 1))},
        'projection_matrices': {cam_a: P_a, cam_b: P_b},
        'points_3d': points_3d,
        'points_2d': {cam_a: src_in, cam_b: dst_in},
        'provenance': provenance_in,
    }


def triangulate_pair_all_landmarks(
    cam_a, cam_b, P_a, P_b, landmarks, frames, width, height, max_reproj_error=30.0,
    reference_points=None, max_distance_factor=3.0,
):
    """Once (R, t) is fixed from the robust wrist-only estimate, triangulating is just
    linear algebra -- no RANSAC degeneracy risk -- so re-pool with all 21 landmarks to
    get a much larger 3D point bank for PnP-chaining the remaining cameras, instead of
    being limited to the wrist-only reference set.

    Reprojection error alone doesn't catch near-parallel-ray degeneracies (a wide range
    of depths along a ray can all reproject with low pixel error), which blow up finger
    landmarks -- especially fast-moving fingertips -- to absurd distances. When
    `reference_points` (the trusted wrist-only inlier cloud) is given, also reject any
    point farther than `max_distance_factor` times that cloud's own spread from its
    centroid, since a hand can't be meters away from its own wrist in the same frame."""
    src_pts, dst_pts, provenance = pool_correspondences(landmarks, cam_a, cam_b, frames, width, height)
    if len(src_pts) == 0:
        return np.zeros((0, 3)), []
    points_3d = triangulate_pair(P_a, P_b, src_pts, dst_pts)
    pts_h = np.hstack([points_3d, np.ones((len(points_3d), 1))])

    def reproj_err(P, pts2d):
        proj = (P @ pts_h.T).T
        return np.linalg.norm(proj[:, :2] / proj[:, 2:3] - pts2d, axis=1)

    keep = (reproj_err(P_a, src_pts) < max_reproj_error) & (reproj_err(P_b, dst_pts) < max_reproj_error)

    if reference_points is not None and len(reference_points) > 0:
        # Median/90th-percentile rather than mean/max: the wrist-only cloud is
        # "trusted" relative to the rest, but can still contain a stray bad point
        # from a near-degenerate ray pair, which would otherwise blow up the cutoff.
        centroid = np.median(reference_points, axis=0)
        ref_scale = np.percentile(np.linalg.norm(reference_points - centroid, axis=1), 90)
        max_distance = max_distance_factor * max(ref_scale, 1e-6)
        keep &= np.linalg.norm(points_3d - centroid, axis=1) < max_distance

    provenance = [p for p, k in zip(provenance, keep) if k]
    return points_3d[keep], provenance


# ---------------------------------------------------------------------------
# PnP chaining for the remaining cameras (fixed scale, no per-pair drift)
# ---------------------------------------------------------------------------

def build_pnp_correspondences(
    cam_id, landmarks, confidence, reference_points_3d, provenance, width, height, min_confidence=0.5
):
    object_points, image_points = [], []
    cam_landmarks = landmarks.get(cam_id, {})
    cam_confidence = confidence.get(cam_id, {})
    for (frame, lm_id), xyz in zip(provenance, reference_points_3d):
        if cam_confidence.get(frame, 0) < min_confidence:
            continue
        lms = cam_landmarks.get(frame)
        if not lms:
            continue
        obs = lms.get(str(lm_id))
        if obs is None:
            continue
        object_points.append(xyz)
        image_points.append([obs[0] * width, obs[1] * height])
    return np.array(object_points, dtype=np.float64), np.array(image_points, dtype=np.float64)


def solve_camera_pose_pnp(object_points, image_points, K, reprojection_error=24.0):
    # Loosened from a typical 8px default: the reference 3D points come from a rough,
    # cross-rig-averaged K rather than real per-camera calibration, so absolute
    # reprojection error is inherently large -- this is a coarse localization, not a
    # precise one. SQPNP (rather than the default ITERATIVE) because with this poorly-
    # conditioned data ITERATIVE/EPNP were found empirically to converge to wildly
    # different, implausible poses depending on the threshold, while SQPNP's global
    # solve stayed consistent.
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        object_points, image_points, K, None,
        reprojectionError=reprojection_error, flags=cv2.SOLVEPNP_SQPNP,
    )
    if not success:
        raise RuntimeError('solvePnPRansac failed to find a pose')
    R, _ = cv2.Rodrigues(rvec)
    inlier_mask = np.zeros(len(object_points), dtype=bool)
    if inliers is not None:
        inlier_mask[inliers.ravel()] = True
    return R, tvec, inlier_mask


def estimate_all_camera_poses(
    cam_ids, landmarks, confidence, K, dist, width, height,
    min_confidence=0.5, max_pool_frames=200, ransac_threshold=3.0,
    min_pair_inliers=30, min_correspondences_pnp=6,
    landmark_ids=POSE_ESTIMATION_LANDMARK_IDS,
):
    pair_ranking = rank_camera_pairs_by_conditioning(
        cam_ids, landmarks, confidence, K, width, height,
        min_confidence, max_pool_frames, ransac_threshold, landmark_ids,
    )
    if not pair_ranking or pair_ranking[0]['inlier_count'] < min_pair_inliers:
        best = pair_ranking[0]['inlier_count'] if pair_ranking else 0
        raise RuntimeError(
            f'Insufficient epipolar conditioning to bootstrap a reference pair (best pair has '
            f'{best} fundamental-matrix inliers, need at least {min_pair_inliers})'
        )
    cam_a, cam_b = pair_ranking[0]['cam_a'], pair_ranking[0]['cam_b']
    common_frames = simultaneous_detection_frames(landmarks, confidence, cam_a, cam_b, min_confidence)
    pooled_frames = select_frames_for_pooling(common_frames, max_pool_frames)
    src_pts, dst_pts, provenance = pool_correspondences(
        landmarks, cam_a, cam_b, pooled_frames, width, height, landmark_ids,
    )

    R, t, inlier_mask = estimate_relative_pose(src_pts, dst_pts, K, ransac_threshold=ransac_threshold)
    ref = estimate_reference_pair_world(cam_a, cam_b, R, t, K, src_pts, dst_pts, inlier_mask, provenance)

    poses = dict(ref['poses'])
    projection_matrices = dict(ref['projection_matrices'])
    pnp_inlier_counts = {}

    # (R, t) is now fixed and robust (wrist-only); re-triangulate with all 21 landmarks
    # to get a large enough 3D point bank to actually chain PnP for the other cameras.
    pnp_points_3d, pnp_provenance = triangulate_pair_all_landmarks(
        cam_a, cam_b, projection_matrices[cam_a], projection_matrices[cam_b],
        landmarks, common_frames, width, height, reference_points=ref['points_3d'],
    )

    for cam_id in cam_ids:
        if cam_id in (cam_a, cam_b):
            continue
        object_points, image_points = build_pnp_correspondences(
            cam_id, landmarks, confidence, pnp_points_3d, pnp_provenance, width, height, min_confidence
        )
        if len(object_points) < min_correspondences_pnp:
            raise RuntimeError(
                f'Camera {cam_id} has only {len(object_points)} correspondences with the '
                f'reference pair ({cam_a}, {cam_b}); need at least {min_correspondences_pnp}'
            )
        R_c, t_c, inlier_mask_c = solve_camera_pose_pnp(object_points, image_points, K)
        poses[cam_id] = (R_c, t_c)
        projection_matrices[cam_id] = build_projection_matrix(K, R_c, t_c)
        pnp_inlier_counts[cam_id] = int(inlier_mask_c.sum())

    return {
        'reference_pair': (cam_a, cam_b),
        'pair_ranking': [(s['cam_a'], s['cam_b'], s['common_frames'], s['inlier_count']) for s in pair_ranking],
        'poses': poses,
        'projection_matrices': projection_matrices,
        'pnp_inlier_counts': pnp_inlier_counts,
        'reference_points_3d': ref['points_3d'],
        'reference_points_2d': ref['points_2d'],
        'reference_provenance': ref['provenance'],
        'pnp_points_3d': pnp_points_3d,
        'pnp_provenance': pnp_provenance,
    }


def save_camera_poses(path, pose_result, K, dist, width, height):
    payload = {
        'K': K.tolist(),
        'dist': dist.tolist(),
        'width': width,
        'height': height,
        'reference_pair': list(pose_result['reference_pair']),
        'pair_ranking': [list(entry) for entry in pose_result['pair_ranking']],
        'pnp_inlier_counts': pose_result['pnp_inlier_counts'],
        'poses': {
            cam_id: {'R': R.tolist(), 't': np.asarray(t).reshape(3).tolist()}
            for cam_id, (R, t) in pose_result['poses'].items()
        },
    }
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)


def load_camera_poses(path):
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        payload = json.load(f)
    K = np.array(payload['K'], dtype=np.float64)
    poses = {
        cam_id: (np.array(v['R'], dtype=np.float64), np.array(v['t'], dtype=np.float64).reshape(3, 1))
        for cam_id, v in payload['poses'].items()
    }
    projection_matrices = {cam_id: build_projection_matrix(K, R, t) for cam_id, (R, t) in poses.items()}
    return {
        'reference_pair': tuple(payload['reference_pair']),
        'pair_ranking': [tuple(p) for p in payload['pair_ranking']],
        'pnp_inlier_counts': payload['pnp_inlier_counts'],
        'poses': poses,
        'projection_matrices': projection_matrices,
    }


# ---------------------------------------------------------------------------
# Combining multiple 2D-landmark sources (e.g. MediaPipe + DWPose) before
# triangulation. Treating both sources' output for the same camera/frame as if
# they were two independent camera views would double-count a single ray and
# isn't a valid DLT input -- combination has to happen at the pixel level.
# ---------------------------------------------------------------------------

def combine_landmark_sources(sources, mode='max_confidence'):
    """`sources` is a list of (landmarks, confidence) pairs, same schema
    throughout this module (landmarks[cam_id][frame] = {"lm_id": [x, y, z]},
    confidence[cam_id][frame] = float). Returns a single (landmarks, confidence)
    pair in the same schema, so every existing triangulation helper works on the
    combined result unchanged.

    - "max_confidence": per camera/frame, keep whichever source has the higher
      frame-level confidence that frame (simple ensemble).
    - "weighted_average": per camera/frame/landmark, confidence-weight-average
      the sources' pixel positions. A source without a per-landmark confidence
      of its own (e.g. MediaPipe) has its frame-level score broadcast to every
      landmark for weighting purposes.
    """
    cam_ids = sorted({cam_id for landmarks, _confidence in sources for cam_id in landmarks})
    combined_landmarks, combined_confidence = {}, {}

    for cam_id in cam_ids:
        combined_landmarks[cam_id] = {}
        combined_confidence[cam_id] = {}
        frames = sorted({
            frame for landmarks, _confidence in sources for frame in landmarks.get(cam_id, {})
        })
        for frame in frames:
            entries = []
            for landmarks, confidence in sources:
                lms = landmarks.get(cam_id, {}).get(frame)
                if not lms:
                    continue
                conf = confidence.get(cam_id, {}).get(frame, 0.0)
                entries.append((lms, conf))
            if not entries:
                continue

            if mode == 'max_confidence':
                lms, conf = max(entries, key=lambda e: e[1])
                combined_landmarks[cam_id][frame] = lms
                combined_confidence[cam_id][frame] = conf
            elif mode == 'weighted_average':
                lm_ids = sorted({lm_id for lms, _conf in entries for lm_id in lms})
                merged = {}
                for lm_id in lm_ids:
                    xyzs, weights = [], []
                    for lms, conf in entries:
                        if lm_id not in lms:
                            continue
                        xyzs.append(lms[lm_id])
                        weights.append(max(conf, 1e-6))
                    if not xyzs:
                        continue
                    xyzs = np.array(xyzs, dtype=np.float64)
                    weights = np.array(weights, dtype=np.float64)
                    merged[lm_id] = np.average(xyzs, axis=0, weights=weights).tolist()
                combined_landmarks[cam_id][frame] = merged
                combined_confidence[cam_id][frame] = max(conf for _lms, conf in entries)
            else:
                raise ValueError(f"Unknown mode '{mode}'")

    return combined_landmarks, combined_confidence


# ---------------------------------------------------------------------------
# N-view DLT triangulation over the full sequence (cameras fixed for all frames)
# ---------------------------------------------------------------------------

def triangulate_dlt(points_2d, projection_matrices):
    if len(points_2d) < 2:
        return None
    A = []
    for cam_id, (x, y) in points_2d.items():
        P = projection_matrices[cam_id]
        A.append(x * P[2, :] - P[0, :])
        A.append(y * P[2, :] - P[1, :])
    _, _, Vt = np.linalg.svd(np.array(A))
    X = Vt[-1]
    if abs(X[3]) < 1e-12:
        return None
    return X[:3] / X[3]


def triangulate_sequence(landmarks, projection_matrices, width, height, min_confidence=0.5,
                          confidence=None, landmark_ids=range(21)):
    cam_ids = list(projection_matrices.keys())
    all_frames = sorted({frame for cam_id in cam_ids for frame in landmarks.get(cam_id, {})})
    reconstruction = {}
    for frame in all_frames:
        frame_key = os.path.splitext(frame)[0]
        frame_points = {}
        for lm_id in landmark_ids:
            points_2d = {}
            for cam_id in cam_ids:
                if confidence is not None and confidence.get(cam_id, {}).get(frame, 0) < min_confidence:
                    continue
                lms = landmarks.get(cam_id, {}).get(frame)
                if not lms:
                    continue
                obs = lms.get(str(lm_id))
                if obs is None:
                    continue
                points_2d[cam_id] = (obs[0] * width, obs[1] * height)
            xyz = triangulate_dlt(points_2d, projection_matrices)
            if xyz is not None:
                frame_points[str(lm_id)] = xyz.tolist()
        if frame_points:
            reconstruction[frame_key] = frame_points
    return reconstruction


def triangulate_dlt_weighted(points_2d, projection_matrices, weights):
    """Same SVD-based DLT as triangulate_dlt, but each point's two equation rows
    are scaled by its confidence weight first, so higher-confidence 2D
    observations dominate the least-squares solve. weights: dict[cam_id -> float],
    same keys as points_2d.
    """
    if len(points_2d) < 2:
        return None
    A = []
    for cam_id, (x, y) in points_2d.items():
        P = projection_matrices[cam_id]
        w = max(weights.get(cam_id, 1.0), 1e-6)
        A.append(w * (x * P[2, :] - P[0, :]))
        A.append(w * (y * P[2, :] - P[1, :]))
    _, _, Vt = np.linalg.svd(np.array(A))
    X = Vt[-1]
    if abs(X[3]) < 1e-12:
        return None
    return X[:3] / X[3]


def triangulate_sequence_weighted(
    landmarks, projection_matrices, width, height, landmark_confidence,
    min_confidence=0.5, confidence=None, landmark_ids=range(21),
):
    """Like triangulate_sequence, but weights each camera's 2D observation of a
    landmark by that landmark's own confidence (landmark_confidence[cam_id][frame]
    = {"lm_id": score}) rather than treating every view as equally trustworthy.
    A camera/frame missing from landmark_confidence falls back to 1.0 (e.g. a
    source that only has a frame-level score, no per-landmark one).
    """
    cam_ids = list(projection_matrices.keys())
    all_frames = sorted({frame for cam_id in cam_ids for frame in landmarks.get(cam_id, {})})
    reconstruction = {}
    for frame in all_frames:
        frame_key = os.path.splitext(frame)[0]
        frame_points = {}
        for lm_id in landmark_ids:
            points_2d, weights = {}, {}
            for cam_id in cam_ids:
                if confidence is not None and confidence.get(cam_id, {}).get(frame, 0) < min_confidence:
                    continue
                lms = landmarks.get(cam_id, {}).get(frame)
                if not lms:
                    continue
                obs = lms.get(str(lm_id))
                if obs is None:
                    continue
                points_2d[cam_id] = (obs[0] * width, obs[1] * height)
                weights[cam_id] = landmark_confidence.get(cam_id, {}).get(frame, {}).get(str(lm_id), 1.0)
            xyz = triangulate_dlt_weighted(points_2d, projection_matrices, weights)
            if xyz is not None:
                frame_points[str(lm_id)] = xyz.tolist()
        if frame_points:
            reconstruction[frame_key] = frame_points
    return reconstruction


# ---------------------------------------------------------------------------
# RANSAC camera-set selection, Kalman/RTS smoothing, jitter flagging -- ported
# and generalized from ransac_keypoint_reconstruction.ipynb (validated on a
# different, uncalibrated rig) to this module's dict[cam_id][frame_key] schema.
# Unlike that notebook's dense (N_FRAMES, N_HANDS) numpy arrays over integer
# camera indices, everything here keys directly on cam_id/frame strings so it
# composes with the rest of this module's functions unchanged.
# ---------------------------------------------------------------------------

def _reproj_error_px(X, P, pixel):
    proj = P @ np.append(X, 1.0)
    return float(np.linalg.norm(proj[:2] / proj[2] - np.asarray(pixel)))


def ransac_select_cameras_sequence(
    landmarks, confidence, projection_matrices, width, height,
    reference_landmark_id=0, min_cameras=2, reproj_thresh_px=15.0,
    hysteresis_inlier_margin=0, hysteresis_err_factor=1.5, min_confidence=0.5,
):
    """Per-frame RANSAC over camera pairs, using `reference_landmark_id` (wrist=0
    for hands; a stable body joint for body) as the representative point -- once a
    trusted camera SET is chosen this way, every other landmark for that frame can
    be triangulated from it directly (see triangulate_sequence_ransac), rather than
    re-running RANSAC per landmark.

    For each frame: gate candidate cameras by presence + confidence >=
    min_confidence (a low-confidence detection shouldn't be eligible to be
    selected as an inlier by chance); try every gated camera pair, triangulate the
    reference landmark, count reprojection-error inliers across ALL gated
    cameras; keep the pair whose inlier set is largest (tie-break: lowest mean
    inlier residual). Inlier counting is purely geometric (reprojection error
    only) -- confidence already did its job at the gating step, so a
    confident-but-wrong detection can't buy its way into the set by weight here.

    Hysteresis: if the previous frame's winning camera set is still available
    this frame and reprojects acceptably (inlier count within
    hysteresis_inlier_margin of this frame's best, and mean error within
    hysteresis_err_factor of it), keep it instead of switching -- damps
    frame-to-frame flicker in which cameras are trusted when several camera-set
    choices are similarly good.

    Returns dict[frame_key] -> {'cameras': [cam_id, ...], 'mean_reproj_err': float,
    'n_inliers': int}, omitting frames with fewer than min_cameras gated cameras or
    where no pair reaches min_cameras inliers.
    """
    cam_ids = list(projection_matrices.keys())
    all_frames = sorted({frame for cam_id in cam_ids for frame in landmarks.get(cam_id, {})})
    lm_key = str(reference_landmark_id)
    selection = {}
    prev_cameras = None

    for frame in all_frames:
        gated = [
            cam_id for cam_id in cam_ids
            if (confidence is None or confidence.get(cam_id, {}).get(frame, 0) >= min_confidence)
            and landmarks.get(cam_id, {}).get(frame, {}).get(lm_key) is not None
        ]
        if len(gated) < min_cameras:
            prev_cameras = None
            continue

        def px(cam_id):
            obs = landmarks[cam_id][frame][lm_key]
            return (obs[0] * width, obs[1] * height)

        pixel_by_cam = {cam_id: px(cam_id) for cam_id in gated}

        best_inliers, best_err = None, None
        for cam_a, cam_b in itertools.combinations(gated, 2):
            X = triangulate_dlt({cam_a: pixel_by_cam[cam_a], cam_b: pixel_by_cam[cam_b]}, projection_matrices)
            if X is None:
                continue
            errs = {c: _reproj_error_px(X, projection_matrices[c], pixel_by_cam[c]) for c in gated}
            inliers = [c for c in gated if errs[c] < reproj_thresh_px]
            if len(inliers) < min_cameras:
                continue
            mean_err = float(np.mean([errs[c] for c in inliers]))
            if (best_inliers is None or len(inliers) > len(best_inliers)
                    or (len(inliers) == len(best_inliers) and mean_err < best_err)):
                best_inliers, best_err = inliers, mean_err

        if best_inliers is None:
            prev_cameras = None
            continue

        chosen, chosen_err = best_inliers, best_err
        if prev_cameras is not None:
            prev_avail = [c for c in prev_cameras if c in gated]
            if len(prev_avail) >= min_cameras:
                X_prev = triangulate_dlt({c: pixel_by_cam[c] for c in prev_avail}, projection_matrices)
                if X_prev is not None:
                    errs_prev = {c: _reproj_error_px(X_prev, projection_matrices[c], pixel_by_cam[c]) for c in gated}
                    prev_inliers = [c for c in gated if errs_prev[c] < reproj_thresh_px]
                    if len(prev_inliers) >= min_cameras:
                        prev_mean_err = float(np.mean([errs_prev[c] for c in prev_inliers]))
                        if (len(prev_inliers) >= len(best_inliers) - hysteresis_inlier_margin
                                and prev_mean_err <= best_err * hysteresis_err_factor):
                            chosen, chosen_err = prev_inliers, prev_mean_err

        selection[frame] = {'cameras': chosen, 'mean_reproj_err': chosen_err, 'n_inliers': len(chosen)}
        prev_cameras = chosen

    return selection


def triangulate_sequence_ransac(
    landmarks, projection_matrices, width, height, landmark_ids, camera_selection,
    landmark_confidence=None,
):
    """Triangulates every id in landmark_ids per frame using ONLY the cameras
    ransac_select_cameras_sequence selected for that frame, via
    triangulate_dlt_weighted. Weighted by landmark_confidence within that
    already-chosen inlier set when given (this is where confidence-aware
    consensus actually happens -- the camera SET itself was chosen purely
    geometrically), else uniform weight among the inliers.
    """
    reconstruction = {}
    for frame, sel in camera_selection.items():
        cams = sel['cameras']
        frame_points = {}
        for lm_id in landmark_ids:
            lm_key = str(lm_id)
            points_2d, weights = {}, {}
            for cam_id in cams:
                obs = landmarks.get(cam_id, {}).get(frame, {}).get(lm_key)
                if obs is None:
                    continue
                points_2d[cam_id] = (obs[0] * width, obs[1] * height)
                weights[cam_id] = (
                    landmark_confidence.get(cam_id, {}).get(frame, {}).get(lm_key, 1.0)
                    if landmark_confidence is not None else 1.0
                )
            xyz = triangulate_dlt_weighted(points_2d, projection_matrices, weights)
            if xyz is not None:
                frame_points[lm_key] = xyz.tolist()
        if frame_points:
            reconstruction[frame] = frame_points
    return reconstruction


def ransac_per_camera_reproj_error(
    landmarks, confidence, reconstruction, projection_matrices, width, height,
    reference_landmark_id=0, min_confidence=0.5,
):
    """For each frame in `reconstruction` (e.g. triangulate_sequence_ransac's
    output), reprojects that frame's already-triangulated reference_landmark_id
    point into EVERY camera with a confident 2D observation of it that frame --
    not just the inlier subset RANSAC actually selected. A camera excluded from
    the winning inlier set on most frames will show up here as consistently
    high-error, which is exactly the signal worth plotting per camera (unlike
    ransac_select_cameras_sequence's own output, which only reports the
    aggregate mean_reproj_err of the cameras it chose, not a per-camera
    breakdown).

    Returns dict[cam_id][frame_key] -> float (px).
    """
    lm_key = str(reference_landmark_id)
    errors = {}
    for frame, frame_points in reconstruction.items():
        X = frame_points.get(lm_key)
        if X is None:
            continue
        X = np.asarray(X, dtype=np.float64)
        for cam_id, P in projection_matrices.items():
            if confidence is not None and confidence.get(cam_id, {}).get(frame, 0) < min_confidence:
                continue
            obs = landmarks.get(cam_id, {}).get(frame, {}).get(lm_key)
            if obs is None:
                continue
            pixel = (obs[0] * width, obs[1] * height)
            errors.setdefault(cam_id, {})[frame] = _reproj_error_px(X, P, pixel)
    return errors


def kalman_rts_smooth(positions, valid_mask, meas_std, process_std=0.03):
    """Constant-velocity Kalman filter + Rauch-Tung-Striebel (RTS) backward
    smoother over one landmark's 3D trajectory. positions: (T,3); valid_mask:
    (T,) bool; meas_std: (T,) per-frame position-measurement std (meters).
    Frames with no triangulation (valid_mask False) get pure predictions, filled
    in by the backward pass using later valid frames. Returns (T,3) smoothed
    positions, or all-NaN if valid_mask is all False. Ported near-verbatim from
    ransac_keypoint_reconstruction.ipynb's Stage 2c.
    """
    T = positions.shape[0]
    if not valid_mask.any():
        return np.full((T, 3), np.nan)

    F = np.eye(6)
    F[:3, 3:] = np.eye(3)
    q = process_std ** 2
    Q = np.block([
        [np.eye(3) * (q / 4.0), np.eye(3) * (q / 2.0)],
        [np.eye(3) * (q / 2.0), np.eye(3) * q],
    ])
    H = np.hstack([np.eye(3), np.zeros((3, 3))])

    first = int(np.argmax(valid_mask))
    x = np.zeros(6)
    x[:3] = positions[first]
    P = np.eye(6) * 1.0

    xs_pred = np.zeros((T, 6))
    Ps_pred = np.zeros((T, 6, 6))
    xs_filt = np.zeros((T, 6))
    Ps_filt = np.zeros((T, 6, 6))

    for t in range(T):
        if t > 0:
            x = F @ x
            P = F @ P @ F.T + Q
        xs_pred[t], Ps_pred[t] = x, P
        if valid_mask[t]:
            R = np.eye(3) * meas_std[t] ** 2
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            x = x + K @ (positions[t] - H @ x)
            P = (np.eye(6) - K @ H) @ P
        xs_filt[t], Ps_filt[t] = x, P

    xs_smooth = xs_filt.copy()
    for t in range(T - 2, -1, -1):
        C = Ps_filt[t] @ F.T @ np.linalg.inv(Ps_pred[t + 1])
        xs_smooth[t] = xs_filt[t] + C @ (xs_smooth[t + 1] - xs_pred[t + 1])

    return xs_smooth[:, :3]


def smooth_reconstruction_sequence(
    reconstruction, frame_keys, landmark_ids, n_inliers_by_frame,
    process_std=0.03, meas_std_base=0.01, meas_std_2cam=0.015,
):
    """Applies kalman_rts_smooth per landmark id across frame_keys (in sequence
    order), inflating measurement noise for frames triangulated from only 2
    inlier cameras (n_inliers_by_frame[frame_key] == 2 -- less depth precision
    than 3+ views). Returns a reconstruction dict in the same schema as the
    input, covering only the span from each landmark's first to last valid
    frame (matches ransac_keypoint_reconstruction.ipynb's Stage 2c convention).
    """
    T = len(frame_keys)
    smoothed = {}
    for lm_id in landmark_ids:
        lm_key = str(lm_id)
        positions = np.full((T, 3), np.nan)
        valid_mask = np.zeros(T, dtype=bool)
        for t, frame in enumerate(frame_keys):
            pt = reconstruction.get(frame, {}).get(lm_key)
            if pt is not None:
                positions[t] = pt
                valid_mask[t] = True
        if not valid_mask.any():
            continue
        meas_std = np.array([
            meas_std_2cam if n_inliers_by_frame.get(frame, 0) == 2 else meas_std_base
            for frame in frame_keys
        ])
        smoothed_positions = kalman_rts_smooth(positions, valid_mask, meas_std, process_std=process_std)
        valid_idx = np.where(valid_mask)[0]
        lo, hi = valid_idx[0], valid_idx[-1] + 1
        for t in range(lo, hi):
            smoothed.setdefault(frame_keys[t], {})[lm_key] = smoothed_positions[t].tolist()
    return smoothed


def flag_jitter_frames(raw_reconstruction, smoothed_reconstruction, frame_keys,
                        reference_landmark_id=0, mad_factor=4.0):
    """Flags frames where the raw (RANSAC-triangulated) reference-landmark
    position deviates sharply from its Kalman/RTS-smoothed trajectory -- evidence
    that frame's camera-set choice (or underlying 2D detection) was an outlier,
    now corrected by the smoother. This directly implements "a wrist can't move
    too much frame to frame" as a principled median+MAD statistical test rather
    than an ad hoc displacement threshold. Ported from
    ransac_keypoint_reconstruction.ipynb's Stage 2d. Returns (flags, residuals):
    flags is a set of flagged frame_keys; residuals is dict[frame_key] -> float
    (meters), covering only frames present in both reconstructions.
    """
    lm_key = str(reference_landmark_id)
    residuals = {}
    for frame in frame_keys:
        raw_pt = raw_reconstruction.get(frame, {}).get(lm_key)
        smooth_pt = smoothed_reconstruction.get(frame, {}).get(lm_key)
        if raw_pt is None or smooth_pt is None:
            continue
        residuals[frame] = float(np.linalg.norm(np.array(raw_pt) - np.array(smooth_pt)))
    if not residuals:
        return set(), residuals
    values = np.array(list(residuals.values()))
    med = np.median(values)
    mad = np.median(np.abs(values - med)) + 1e-9
    thresh = med + mad_factor * mad
    flags = {frame for frame, r in residuals.items() if r > thresh}
    return flags, residuals


# ---------------------------------------------------------------------------
# Hand-identity mixup detection. Motivated by an observation on the h5 rig
# (ground-truth calibration, never inferred from the hand keypoints themselves):
# when one hand's detection for a camera/frame is bad, the other hand for that
# same camera/frame often looks bad too -- evidence the camera's detector
# confused/swapped left vs right for that frame, not two independent unlucky
# detections.
#
# detect_body_wrist_hand_swap is deliberately listed, and should be run, first:
# it's the cheapest and most reliable of these tests (a same-detector, same-frame
# body-pose estimate needs no other camera or the other detector to be confident
# that frame, unlike the LOO/cross-detector tests below), so it belongs first in
# any correction pipeline -- fix what it can cheaply and unambiguously fix before
# spending the pricier multiview/cross-detector tests on what's left.
# ---------------------------------------------------------------------------

BODY_WRIST_COCO = {'left': 9, 'right': 10}  # COCO-17: left_wrist=9, right_wrist=10


def detect_body_wrist_hand_swap(
    landmarks_left, landmarks_right, landmarks_body, cam_id, frame, width, height,
    hand_wrist_id=0, body_wrist_left_id=BODY_WRIST_COCO['left'], body_wrist_right_id=BODY_WRIST_COCO['right'],
    swap_margin_px=20.0,
):
    """Cheapest, single-camera/frame, purely-2D swap test: a detector's own body
    pose already carries independent left/right wrist keypoints for that same
    frame, so they're a same-detector reference for which hand is which -- no
    other camera or the other detector needs to have a confident detection that
    frame, unlike detect_camera_hand_swap (needs >= min_loo_cameras other
    cameras) or cross_detector_hand_disagreement (needs both detectors). Meant to
    run FIRST in a correction pipeline, ahead of those pricier tests (see the
    module comment above).

    Flags a swap when swapping the hand-left/hand-right assignment reduces total
    wrist-to-wrist pixel distance to the body's own left/right wrist by more than
    `swap_margin_px` -- an explicit margin (not a bare comparison), since body-pose
    wrist estimates are noisier than hand-model wrists and a bare comparison would
    flip-flop on frames where the two assignments are nearly tied.

    Returns None when undecidable: the body or either hand lacks a detection for
    this camera/frame. Otherwise a dict with 'swap_detected' (bool),
    'as_labeled_err'/'swapped_err' (summed px distance to the body's wrists under
    each hypothesis), and 'margin' (as_labeled_err - swapped_err, positive favors
    a swap).
    """
    hand_key = str(hand_wrist_id)
    body_l_key = str(body_wrist_left_id)
    body_r_key = str(body_wrist_right_id)

    body = landmarks_body.get(cam_id, {}).get(frame)
    l_lm = landmarks_left.get(cam_id, {}).get(frame)
    r_lm = landmarks_right.get(cam_id, {}).get(frame)
    if not body or not l_lm or not r_lm:
        return None
    body_l = body.get(body_l_key)
    body_r = body.get(body_r_key)
    hand_l = l_lm.get(hand_key)
    hand_r = r_lm.get(hand_key)
    if body_l is None or body_r is None or hand_l is None or hand_r is None:
        return None

    def px(pt):
        return np.array([pt[0] * width, pt[1] * height], dtype=np.float64)

    body_l_px, body_r_px, hand_l_px, hand_r_px = px(body_l), px(body_r), px(hand_l), px(hand_r)

    as_labeled_err = float(np.linalg.norm(hand_l_px - body_l_px) + np.linalg.norm(hand_r_px - body_r_px))
    swapped_err = float(np.linalg.norm(hand_r_px - body_l_px) + np.linalg.norm(hand_l_px - body_r_px))

    return {
        'swap_detected': swapped_err < as_labeled_err - swap_margin_px,
        'as_labeled_err': as_labeled_err,
        'swapped_err': swapped_err,
        'margin': as_labeled_err - swapped_err,
    }


def detect_body_wrist_hand_swaps_sequence(
    landmarks_left, landmarks_right, landmarks_body, cam_ids, frame_keys, width, height,
    hand_wrist_id=0, swap_margin_px=20.0,
):
    """Loops detect_body_wrist_hand_swap over every (camera, frame) pair.
    Returns dict[cam_id][frame_key] -> result dict, omitting undecidable
    combinations (missing body or hand detection that frame)."""
    results = {}
    for cam_id in cam_ids:
        for frame in frame_keys:
            r = detect_body_wrist_hand_swap(
                landmarks_left, landmarks_right, landmarks_body, cam_id, frame, width, height,
                hand_wrist_id=hand_wrist_id, swap_margin_px=swap_margin_px,
            )
            if r is not None:
                results.setdefault(cam_id, {})[frame] = r
    return results


def detect_camera_hand_swap(
    landmarks_left, confidence_left, landmarks_right, confidence_right,
    projection_matrices, width, height, cam_id, frame,
    reference_landmark_id=0, min_loo_cameras=2, min_confidence=0.5, swap_margin_px=5.0,
):
    """Leave-one-out (LOO) swap-hypothesis test for one camera/frame: triangulate
    left and right hands independently using every OTHER camera with a confident
    detection that frame (the multiview consensus, deliberately excluding
    cam_id's own observation so the test isn't circular), then compare cam_id's
    reprojection error under the as-labeled assignment vs. the swapped one.
    Flags a swap only when the swapped assignment fits distinctly better (by more
    than swap_margin_px) -- an explicit margin, not a bare comparison, so
    near-tied/ambiguous frames aren't flip-flopped (same rationale as
    ransac_select_cameras_sequence's hysteresis). This directly tests the
    hand-mixup hypothesis using the dataset's one fully-trustworthy asset --
    ground-truth multiview geometry -- rather than inferring it indirectly.

    Uses only reference_landmark_id (wrist=0) for both the LOO consensus and the
    error comparison, matching this module's existing wrist-only convention for
    pose/consensus decisions (see POSE_ESTIMATION_LANDMARK_IDS above) -- cheap and
    robust, since a genuine hand-identity swap moves every landmark together, not
    just some.

    Returns None when undecidable: fewer than min_loo_cameras other cameras have
    a confident detection for a hand, or cam_id itself lacks a confident
    detection on either hand that frame. Otherwise a dict with 'swap_detected'
    (bool), 'as_labeled_err'/'swapped_err' (mean px reprojection error under each
    hypothesis), and 'margin' (as_labeled_err - swapped_err, positive favors a
    swap).
    """
    lm_key = str(reference_landmark_id)
    other_cams = [c for c in projection_matrices if c != cam_id]

    def loo_point(landmarks, side_confidence):
        points_2d = {}
        for c in other_cams:
            if side_confidence.get(c, {}).get(frame, 0) < min_confidence:
                continue
            obs = landmarks.get(c, {}).get(frame, {}).get(lm_key)
            if obs is None:
                continue
            points_2d[c] = (obs[0] * width, obs[1] * height)
        if len(points_2d) < min_loo_cameras:
            return None
        return triangulate_dlt(points_2d, projection_matrices)

    def cam_pixel(landmarks, side_confidence):
        if side_confidence.get(cam_id, {}).get(frame, 0) < min_confidence:
            return None
        obs = landmarks.get(cam_id, {}).get(frame, {}).get(lm_key)
        if obs is None:
            return None
        return (obs[0] * width, obs[1] * height)

    cam_left_px = cam_pixel(landmarks_left, confidence_left)
    cam_right_px = cam_pixel(landmarks_right, confidence_right)
    if cam_left_px is None or cam_right_px is None:
        return None

    X_left_loo = loo_point(landmarks_left, confidence_left)
    X_right_loo = loo_point(landmarks_right, confidence_right)
    if X_left_loo is None or X_right_loo is None:
        return None

    P = projection_matrices[cam_id]
    as_labeled_err = (
        _reproj_error_px(X_left_loo, P, cam_left_px) + _reproj_error_px(X_right_loo, P, cam_right_px)
    )
    swapped_err = (
        _reproj_error_px(X_right_loo, P, cam_left_px) + _reproj_error_px(X_left_loo, P, cam_right_px)
    )

    return {
        'swap_detected': swapped_err < as_labeled_err - swap_margin_px,
        'as_labeled_err': as_labeled_err,
        'swapped_err': swapped_err,
        'margin': as_labeled_err - swapped_err,
    }


def detect_camera_hand_swaps_sequence(
    landmarks_left, confidence_left, landmarks_right, confidence_right,
    projection_matrices, width, height, cam_ids, frame_keys,
    reference_landmark_id=0, min_loo_cameras=2, min_confidence=0.5, swap_margin_px=5.0,
):
    """Loops detect_camera_hand_swap over every (camera, frame) pair. Returns
    dict[cam_id][frame_key] -> result dict, omitting undecidable combinations."""
    results = {}
    for cam_id in cam_ids:
        for frame in frame_keys:
            r = detect_camera_hand_swap(
                landmarks_left, confidence_left, landmarks_right, confidence_right,
                projection_matrices, width, height, cam_id, frame,
                reference_landmark_id=reference_landmark_id, min_loo_cameras=min_loo_cameras,
                min_confidence=min_confidence, swap_margin_px=swap_margin_px,
            )
            if r is not None:
                results.setdefault(cam_id, {})[frame] = r
    return results


def cross_detector_hand_disagreement(
    landmarks_mp_left, landmarks_mp_right, landmarks_dw_left, landmarks_dw_right,
    cam_id, frame, width, height, landmark_ids=range(21), swap_ratio_thresh=0.5,
):
    """Cheap, single-camera/frame, purely-2D pre-filter (no triangulation): mean
    pixel distance between MediaPipe's and DWPose's landmarks under the
    as-labeled pairing (mp_left-vs-dw_left, mp_right-vs-dw_right) vs. the swapped
    pairing (mp_left-vs-dw_right, mp_right-vs-dw_left). Flags 'likely_swap' when
    the swapped pairing agrees distinctly better -- an order of magnitude cheaper
    than detect_camera_hand_swap, used to prioritize which (camera, frame) pairs
    are worth running that geometric test on, rather than sweeping it over every
    camera/frame. Returns None if any of the four (detector, side) combinations
    lacks a detection for this camera/frame.
    """
    def pixel_dist(lms_a, lms_b):
        dists = []
        for lm_id in landmark_ids:
            a = lms_a.get(str(lm_id))
            b = lms_b.get(str(lm_id))
            if a is None or b is None:
                continue
            dists.append(float(np.hypot((a[0] - b[0]) * width, (a[1] - b[1]) * height)))
        return float(np.mean(dists)) if dists else None

    mp_left = landmarks_mp_left.get(cam_id, {}).get(frame)
    mp_right = landmarks_mp_right.get(cam_id, {}).get(frame)
    dw_left = landmarks_dw_left.get(cam_id, {}).get(frame)
    dw_right = landmarks_dw_right.get(cam_id, {}).get(frame)
    if not (mp_left and mp_right and dw_left and dw_right):
        return None

    d_ll = pixel_dist(mp_left, dw_left)
    d_rr = pixel_dist(mp_right, dw_right)
    d_lr = pixel_dist(mp_left, dw_right)
    d_rl = pixel_dist(mp_right, dw_left)
    if None in (d_ll, d_rr, d_lr, d_rl):
        return None

    same_side = (d_ll + d_rr) / 2.0
    cross_side = (d_lr + d_rl) / 2.0
    return {
        'likely_swap': cross_side < swap_ratio_thresh * same_side,
        'same_side': same_side,
        'cross_side': cross_side,
    }


def detect_high_velocity_frames(
    landmarks, cam_ids, frame_keys, width, height, reference_landmark_id=0, velocity_thresh_px=200.0,
):
    """Per-camera, frame-to-frame 2D pixel velocity of `reference_landmark_id`
    (wrist=0), flagging frames where it exceeds `velocity_thresh_px` -- a hand
    physically can't teleport that far in one frame at this capture rate, so a
    spike almost always means the underlying 2D detection (or its L/R identity)
    was wrong that frame, not that the hand actually moved that fast. `frame_keys`
    must already be in temporal sequence order (velocity is undefined for the
    first frame any camera has a detection, and after any gap the "previous"
    position is reset rather than measured across the gap).

    Deliberately a plain per-camera 2D measurement (no triangulation) -- this is
    meant to run on whichever landmark space the caller is about to act on next
    (raw pixels if immediately re-rendering to a video, undistorted if feeding
    triangulation), so it takes `landmarks` directly rather than assuming one.

    Returns dict[cam_id][frame_key] -> {'velocity_px': float, 'flagged': bool}.
    """
    result = {}
    lm_key = str(reference_landmark_id)
    for cam_id in cam_ids:
        prev_px = None
        for fk in frame_keys:
            obs = landmarks.get(cam_id, {}).get(fk, {}).get(lm_key)
            if obs is None:
                prev_px = None
                continue
            px = np.array([obs[0] * width, obs[1] * height], dtype=np.float64)
            if prev_px is not None:
                vel = float(np.linalg.norm(px - prev_px))
                result.setdefault(cam_id, {})[fk] = {'velocity_px': vel, 'flagged': vel > velocity_thresh_px}
            prev_px = px
    return result


def apply_swap_corrections(landmarks_left, confidence_left, landmarks_right, confidence_right, swap_flags):
    """Corrects a detector's left/right hand assignment wherever a mixup detector
    flagged a (camera, frame) as swapped (e.g. detect_camera_hand_swaps_sequence's
    'swap_detected', or any dict[cam_id][frame_key] -> bool of the same shape):
    swaps that (camera, frame)'s 2D landmarks AND its confidence between the left
    and right dicts, since the underlying detection didn't change, only which
    physical hand it actually belongs to. Frames not flagged are passed through
    unchanged (deep-copied, so the inputs are never mutated).

    Returns (landmarks_left, confidence_left, landmarks_right, confidence_right,
    n_corrected) -- the same 4-dict shape as the inputs plus a correction count.
    """
    new_landmarks_left = copy.deepcopy(landmarks_left)
    new_confidence_left = copy.deepcopy(confidence_left)
    new_landmarks_right = copy.deepcopy(landmarks_right)
    new_confidence_right = copy.deepcopy(confidence_right)

    n_corrected = 0
    for cam_id, by_frame in swap_flags.items():
        for fk, flagged in by_frame.items():
            if not flagged:
                continue
            l_lm = landmarks_left.get(cam_id, {}).get(fk)
            r_lm = landmarks_right.get(cam_id, {}).get(fk)
            l_conf = confidence_left.get(cam_id, {}).get(fk)
            r_conf = confidence_right.get(cam_id, {}).get(fk)

            if r_lm is not None:
                new_landmarks_left.setdefault(cam_id, {})[fk] = r_lm
            else:
                new_landmarks_left.get(cam_id, {}).pop(fk, None)
            if l_lm is not None:
                new_landmarks_right.setdefault(cam_id, {})[fk] = l_lm
            else:
                new_landmarks_right.get(cam_id, {}).pop(fk, None)

            if r_conf is not None:
                new_confidence_left.setdefault(cam_id, {})[fk] = r_conf
            if l_conf is not None:
                new_confidence_right.setdefault(cam_id, {})[fk] = l_conf
            n_corrected += 1

    return new_landmarks_left, new_confidence_left, new_landmarks_right, new_confidence_right, n_corrected


def apply_confidence_penalty(confidence, flags, penalty_factor=0.01):
    """Multiplies confidence[cam_id][frame_key] by `penalty_factor` wherever
    flags[cam_id][frame_key] is truthy (e.g. detect_high_velocity_frames'
    'flagged') -- e.g. a physically-implausible velocity spike shouldn't
    necessarily exclude a frame outright (a later, better detector might still
    want it), but it should stop dominating a confidence-weighted combination.
    Returns (confidence, n_penalized); the input dict is not mutated.
    """
    new_confidence = copy.deepcopy(confidence)
    n_penalized = 0
    for cam_id, by_frame in flags.items():
        for fk, v in by_frame.items():
            flagged = v['flagged'] if isinstance(v, dict) else bool(v)
            if not flagged:
                continue
            orig = confidence.get(cam_id, {}).get(fk, 0.0)
            new_confidence.setdefault(cam_id, {})[fk] = orig * penalty_factor
            n_penalized += 1
    return new_confidence, n_penalized


def save_reconstruction(path, data):
    with open(path, 'w') as f:
        json.dump(data, f)


def load_reconstruction(path):
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return json.load(f)


def plot_camera_positions(poses, views=((20, -60), (20, 30)), figsize=(12, 6)):
    """Static two-view sanity check of the estimated camera layout (complements the
    interactive viser scene from add_camera_frustums)."""
    cam_ids = sorted(poses.keys())
    positions = np.array([(-R.T @ np.asarray(t).reshape(3, 1)).ravel() for R, t in (poses[c] for c in cam_ids)])

    fig = plt.figure(figsize=figsize)
    for i, (elev, azim) in enumerate(views):
        ax = fig.add_subplot(1, len(views), i + 1, projection='3d')
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='tab:blue', s=60, depthshade=False)
        for cam_id, p in zip(cam_ids, positions):
            ax.text(p[0], p[1], p[2], f'  {cam_id}', fontsize=9)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f'elev={elev}, azim={azim}')
    fig.suptitle('Estimated camera positions')
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Viser visualization
# ---------------------------------------------------------------------------

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

# Standard COCO-17 body topology (nose, l/r eye, l/r ear, l/r shoulder, l/r elbow,
# l/r wrist, l/r hip, l/r knee, l/r ankle -- ids 0-16 in that order), for skeletons
# triangulated with landmark_ids=range(17) (see load_all_camera_calibration's
# sibling body-extraction path in h5_hand_extraction.py).
BODY_CONNECTIONS_COCO = [
    (0, 1), (0, 2), (1, 3), (2, 4),          # face: nose-eyes-ears
    (5, 6),                                    # shoulders
    (5, 7), (7, 9), (6, 8), (8, 10),           # arms
    (5, 11), (6, 12), (11, 12),                # torso
    (11, 13), (13, 15), (12, 14), (14, 16),    # legs
]


def start_viser_server(port=None):
    return viser.ViserServer(port=port) if port is not None else viser.ViserServer()


def add_camera_frustums(server, poses, K, width, height, scale=0.15):
    fov = 2.0 * np.arctan(height / (2.0 * K[1, 1]))
    aspect = float(width / height)
    for cam_id, (R, t) in poses.items():
        position = (-R.T @ np.asarray(t).reshape(3, 1)).ravel()
        wxyz = tf.SO3.from_matrix(R.T).wxyz
        server.scene.add_camera_frustum(
            name=f'/cameras/{cam_id}',
            fov=float(fov),
            aspect=aspect,
            scale=scale,
            color=(80, 160, 255),
            wxyz=wxyz,
            position=position,
        )
        server.scene.add_frame(
            name=f'/cameras/{cam_id}/axis',
            axes_length=scale * 0.5,
            axes_radius=scale * 0.03,
            wxyz=wxyz,
            position=position,
        )


def add_skeleton_frame(
    server, frame_points, name_prefix='/skeleton',
    connections=HAND_CONNECTIONS, point_color=(220, 220, 220), bone_color=(0, 255, 120),
    flagged=False, flagged_color=(230, 40, 40),
):
    """connections/point_color/bone_color default to exactly the previous
    hardcoded behavior (backward compatible with run_skeleton_player and any
    other existing caller). Pass connections=BODY_CONNECTIONS_COCO for a
    body skeleton instead of a hand one. flagged=True overrides both colors to
    flagged_color -- a whole-skeleton override rather than per-joint, matching
    the granularity the jitter/swap mixup detectors actually operate at
    (frame/hand, not individual joint)."""
    if flagged:
        point_color = flagged_color
        bone_color = flagged_color
    if not frame_points:
        server.scene.add_point_cloud(
            name=f'{name_prefix}/joints',
            points=np.zeros((0, 3), dtype=np.float32),
            colors=np.zeros((0, 3), dtype=np.uint8),
            point_size=0.005,
        )
        return
    joint_coords = {int(k): np.array(v, dtype=np.float32) for k, v in frame_points.items()}
    ids = sorted(joint_coords.keys())
    pts = np.stack([joint_coords[i] for i in ids])
    colors = np.full((len(ids), 3), point_color, dtype=np.uint8)
    server.scene.add_point_cloud(name=f'{name_prefix}/joints', points=pts, colors=colors, point_size=0.005)

    segs = [
        [joint_coords[a], joint_coords[b]]
        for a, b in connections
        if a in joint_coords and b in joint_coords
    ]
    if segs:
        server.scene.add_line_segments(
            name=f'{name_prefix}/bones',
            points=np.array(segs, dtype=np.float32),
            colors=np.array(bone_color, dtype=np.uint8),
            line_width=2.0,
        )


def run_skeleton_player(server, reconstruction, fps=15.0):
    frame_keys = sorted(reconstruction.keys(), key=int)
    if not frame_keys:
        print('No reconstructed frames to play.')
        return

    play_button = server.gui.add_button('Play / Pause')
    stop_button = server.gui.add_button('Stop viewer')
    speed_slider = server.gui.add_slider('Playback Speed (FPS)', min=1, max=60, step=1, initial_value=fps)
    frame_slider = server.gui.add_slider(
        'Timeline Frame', min=0, max=len(frame_keys) - 1, step=1, initial_value=0
    )

    state = {'playing': False, 'running': True}

    @play_button.on_click
    def _(_):
        state['playing'] = not state['playing']

    @stop_button.on_click
    def _(_):
        state['running'] = False

    current_idx = 0
    try:
        while state['running']:
            if not state['playing']:
                current_idx = frame_slider.value
            frame_key = frame_keys[current_idx]
            add_skeleton_frame(server, reconstruction[frame_key])
            if state['playing']:
                current_idx = (current_idx + 1) % len(frame_keys)
                frame_slider.value = current_idx
                time.sleep(1.0 / speed_slider.value)
            else:
                time.sleep(0.05)
    except KeyboardInterrupt:
        print('Viewer interrupted.')
