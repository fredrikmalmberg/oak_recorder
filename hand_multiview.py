import sys
import os
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


def draw_hand_skeleton(image, pixel_xy, color=(0, 255, 0), point_color=(255, 0, 0)):
    for a, b in HAND_CONNECTIONS:
        if a in pixel_xy and b in pixel_xy:
            pa = tuple(np.round(pixel_xy[a]).astype(int))
            pb = tuple(np.round(pixel_xy[b]).astype(int))
            cv2.line(image, pa, pb, color, 2)
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


def triangulate_sequence(landmarks, projection_matrices, width, height, min_confidence=0.5, confidence=None):
    cam_ids = list(projection_matrices.keys())
    all_frames = sorted({frame for cam_id in cam_ids for frame in landmarks.get(cam_id, {})})
    reconstruction = {}
    for frame in all_frames:
        frame_key = os.path.splitext(frame)[0]
        frame_points = {}
        for lm_id in range(21):
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


def add_skeleton_frame(server, frame_points, name_prefix='/skeleton'):
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
    colors = np.full((len(ids), 3), 220, dtype=np.uint8)
    server.scene.add_point_cloud(name=f'{name_prefix}/joints', points=pts, colors=colors, point_size=0.005)

    segs = [
        [joint_coords[a], joint_coords[b]]
        for a, b in HAND_CONNECTIONS
        if a in joint_coords and b in joint_coords
    ]
    if segs:
        server.scene.add_line_segments(
            name=f'{name_prefix}/bones',
            points=np.array(segs, dtype=np.float32),
            colors=np.array((0, 255, 120), dtype=np.uint8),
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
