"""Triangulation of the per-camera MediaPipe 2D (fast3d step 4).

Points are undistorted per camera (undistort_norm) and put into pose2d's
dict schema (landmarks[cam][frame_key][lm_id] = [x, y, z], normalised), then
each part goes through pose2d.triangulation.triangulate_part: per-frame
RANSAC camera choice on a reference point, weighted DLT, Kalman/RTS
smoothing. Parts: body (33 BlazePose points, reference left shoulder 11,
weights = visibility), left/right hand (21, wrist 0, Hands' handedness
score), face (478 FaceMesh points, nose tip 1).
"""
import os

import cv2
import numpy as np

from pose2d.triangulation import triangulate_part

PARTS = {  # name: (n points, reference point, min whole-part confidence)
    'body': (33, 11, 0.5),
    'left': (21, 0, 0.5),
    'right': (21, 0, 0.5),
    'face': (478, 1, 0.5),
}
BODY_UPPER = list(range(25))          # BlazePose points above the knees: whole-body confidence


def undistort_norm(pts, K, dist, w, h):
    """pts (..., 2+) normalised raw-image coordinates (NaN allowed) -> same shape, undistorted (pinhole K)."""
    out = pts.copy()
    flat = pts[..., :2].reshape(-1, 2).astype(np.float64)
    ok = np.isfinite(flat).all(1)
    if ok.any():
        px = flat[ok] * (w, h)
        und = cv2.undistortPoints(px[:, None], K, dist, P=K)[:, 0]
        res = flat.copy()
        res[ok] = und / (w, h)
        out[..., :2] = res.reshape(pts[..., :2].shape)
    return out


def to_dicts(per_cam, keys, part):
    """per_cam[cam] = extract2d output -> (landmarks, confidence, landmark_confidence) dicts for one part."""
    lms, conf, lconf = {}, {}, {}
    for cam, d in per_cam.items():
        P = d[part]
        if part == 'body':
            c = d['body_vis'][:, BODY_UPPER].mean(1)
        elif part == 'face':
            c = d['face_found'].astype(np.float32)
        else:
            c = d[f'{part}_score']
        lms[cam], conf[cam], lconf[cam] = {}, {}, {}
        for t, k in enumerate(keys):
            if not np.isfinite(P[t, 0, 0]) or c[t] <= 0:
                continue
            lms[cam][k] = {str(i): P[t, i].tolist() for i in range(P.shape[1])}
            conf[cam][k] = float(c[t])
            if part == 'body':
                lconf[cam][k] = {str(i): float(d['body_vis'][t, i]) for i in range(33)}
    return lms, conf, (lconf if part == 'body' else None)


def triangulate(part, per_cam, keys, P, width, height):
    """Returns dict raw (T, N, 3), smoothed (T, N, 3) (NaN where missing), n_cams (T,), reproj_px {cam: median}."""
    n, ref, min_conf = PARTS[part]
    lms, conf, lconf = to_dicts(per_cam, keys, part)
    ids = [str(i) for i in range(n)]
    raw, smooth, sel, reproj, _jit = triangulate_part(lms, conf, P, width, height, ids, str(ref),
                                                     min_confidence=min_conf, landmark_confidence=lconf)
    T = len(keys)
    R, S = np.full((T, n, 3), np.nan, np.float32), np.full((T, n, 3), np.nan, np.float32)
    ncam = np.zeros(T, np.int8)
    for t, k in enumerate(keys):
        for src, dst in ((raw, R), (smooth, S)):
            fr = src.get(k)
            if fr:
                for i, xyz in fr.items():
                    dst[t, int(i)] = xyz
        if k in sel:
            ncam[t] = sel[k].get('n_inliers', 0)
    # the smoother predicts through gaps; keep smoothed points only where the frame was triangulated
    S[~np.isfinite(R).all(-1).any(-1)] = np.nan
    rp = {cam: float(np.median(list(v.values()))) for cam, v in reproj.items() if v}
    return dict(raw=R, smoothed=S, n_cams=ncam, reproj_px=rp)
