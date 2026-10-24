"""Calibration-guided flash search: fill in cameras where the blind detector is weak or found nothing.

The badge is one physical point seen by every camera. When enough cameras detected the white flash, its pixel
positions are triangulated (RANSAC over camera subsets, so one camera that locked onto clutter cannot spoil it); the
3D point is projected into every camera and the flash onset is read from a small box around that projection. A camera
that the blind detector missed (weak signal, clutter, small blob) usually shows a clean flash there.

Needs a calibration of the same resolution and the same devices as the take (find_calibration). Takes record
"uncalibrated" when no calibration was loaded in the app, so the closest-in-time file of the right resolution is used.
Calibration folders searched: CALIBRATION_DIRS (output/calibration and output/calibration_half in the repo), plus any
folders listed in the OAK_CALIBRATION_DIRS environment variable (os.pathsep separated).

Pure numpy/cv2; reads the calibration json itself instead of importing calibrate.py (viser, networkx, ...).
"""
import glob
import itertools
import json
import os
from datetime import datetime

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CALIBRATION_DIRS = [os.path.join(HERE, "output", "calibration"), os.path.join(HERE, "output", "calibration_half")]
INLIER_PX_FRAC = 0.01     # a camera agrees with the triangulated point if it projects within this fraction of the frame width
MIN_INLIERS = 3           # cameras that must agree on the 3D point
BOX_FRAC = 0.036          # half-size of the search box around the projected point, as a fraction of the frame width
WINDOW_S = 0.4            # onset is searched this far either side of the reference camera's flash frame
MIN_RISE = 15.0           # smallest brightness rise (score units) in the box that counts as a flash
WHITE_FLOOR = 200         # same white score as flash_offsets.py


def calibration_dirs():
    extra = [d for d in os.environ.get("OAK_CALIBRATION_DIRS", "").split(os.pathsep) if d]
    return list(CALIBRATION_DIRS) + extra


def load_calibration(path):
    """Calibration json -> {cam: {K, dist, R, t, width, height, device_id}} for the posed cameras (same content as
    calibrate.load_calibration_output)."""
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    out = {}
    for cam, entry in payload["cameras"].items():
        if entry.get("extrinsics") is None:
            continue
        intr = entry["intrinsics"]
        out[cam] = {"K": np.array(intr["camera_matrix"], np.float64), "dist": np.array(intr["dist_coeffs"], np.float64),
                    "R": np.array(entry["extrinsics"]["rotation"], np.float64),
                    "t": np.array(entry["extrinsics"]["translation"], np.float64).reshape(3),
                    "width": intr["image_width"], "height": intr["image_height"], "device_id": entry.get("device_id")}
    return out


def _calibration_time(path, payload=None):
    try:
        if payload is None:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        return datetime.fromisoformat(payload["metadata"]["timestamp"])
    except (KeyError, ValueError, OSError, json.JSONDecodeError):
        return None


def find_calibration(take_dir, cam_ids, width, dirs=None):
    """(path, cameras) of the calibration to use for this take, or (None, reason).
    Uses the take's own recorded calibration path if it exists; otherwise the closest in time (before or after) among the
    files in `dirs` that pose every camera of the take, at this frame width, on the same devices."""
    try:
        with open(os.path.join(take_dir, "take_meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
    except (OSError, json.JSONDecodeError):
        meta = {}
    devices = {c: v.get("device_id") for c, v in meta.get("cameras", {}).items()}
    try:
        start = datetime.fromisoformat(meta["start_time_iso"])
    except (KeyError, ValueError):
        start = None
    recorded = (meta.get("calibration") or {}).get("path")
    paths = []
    if recorded:
        paths += [recorded, os.path.join(HERE, recorded)]
        paths += [os.path.join(d, os.path.basename(recorded.replace("\\", "/"))) for d in (dirs or calibration_dirs())]
        for p in paths:
            if os.path.isfile(p):
                try:
                    cams = load_calibration(p)
                except (OSError, ValueError, KeyError, json.JSONDecodeError):
                    continue
                if all(c in cams for c in cam_ids):
                    return p, cams
    cands = []
    for d in (dirs or calibration_dirs()):
        for p in glob.glob(os.path.join(d, "*.json")):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                cams = load_calibration(p)
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                continue
            if "cameras" not in payload or not all(c in cams for c in cam_ids):
                continue
            if any(cams[c]["width"] != width for c in cam_ids):
                continue
            if any(devices.get(c) and cams[c]["device_id"] and devices[c] != cams[c]["device_id"] for c in cam_ids):
                continue
            ts = _calibration_time(p, payload)
            gap = abs((ts - start).total_seconds()) if (ts and start) else float("inf")
            cands.append((gap, p, cams))
    if not cands:
        return None, f"no calibration at {width}px width for these cameras in {', '.join(dirs or calibration_dirs())}"
    cands.sort(key=lambda t: t[0])
    return cands[0][1], cands[0][2]


def _proj_matrix(c):
    return c["K"] @ np.hstack([c["R"], c["t"].reshape(3, 1)])


def project(X, c):
    rvec, _ = cv2.Rodrigues(c["R"])
    p, _ = cv2.projectPoints(np.asarray(X, np.float64).reshape(1, 3), rvec, c["t"], c["K"], c["dist"])
    return p.reshape(2)


def _dlt(obs, Ps):
    A = []
    for (u, v), P in zip(obs, Ps):
        A += [u * P[2] - P[0], v * P[2] - P[1]]
    X = np.linalg.svd(np.asarray(A))[2][-1]
    return X[:3] / X[3]


def triangulate(points, calib, width):
    """points {cam: (x, y)} in distorted pixels -> {"X", "inliers", "errors"} or None.
    Tries every camera subset (<= 8 cameras: 247 subsets) and keeps the one with the most cameras within
    INLIER_PX_FRAC * width of the point, then refits on those. None if fewer than MIN_INLIERS agree."""
    ids = sorted(c for c in points if c in calib)
    if len(ids) < MIN_INLIERS:
        return None
    pts = {c: np.asarray(points[c], np.float64) for c in ids}
    P = {c: _proj_matrix(calib[c]) for c in ids}
    ud = {c: cv2.undistortPoints(pts[c].reshape(1, 1, 2), calib[c]["K"], calib[c]["dist"], P=calib[c]["K"]).reshape(2) for c in ids}
    thr = INLIER_PX_FRAC * width

    def errors(X):
        return {c: float(np.linalg.norm(project(X, calib[c]) - pts[c])) for c in ids}

    best = None
    for k in range(2, len(ids) + 1):
        for sub in itertools.combinations(ids, k):
            X = _dlt([ud[c] for c in sub], [P[c] for c in sub])
            if not np.all(np.isfinite(X)):
                continue
            err = errors(X)
            inl = [c for c in ids if err[c] < thr]
            key = (len(inl), -sum(err[c] for c in inl) / max(len(inl), 1))
            if best is None or key > best[0]:
                best = (key, inl)
    if best is None or len(best[1]) < MIN_INLIERS:
        return None
    inl = best[1]
    X = _dlt([ud[c] for c in inl], [P[c] for c in inl])
    err = errors(X)
    return {"X": X, "inliers": inl, "errors": err}


def guided_onsets(take_dir, cam_ids, X, calib, fps, ref_frame, width):
    """Flash onset frame per camera, read from a box around X projected into the camera.
    Returns {cam: {"onset": frame or None, "rise": float, "feature": "blue"|"white", "xy": (x, y)}}.
    Onset = first frame in the window where the brighter of the two scores (blue dominance / near-saturated white)
    has risen halfway to its peak."""
    half_box = max(20, int(BOX_FRAC * width))
    lo = max(int(ref_frame - WINDOW_S * fps), 0)
    hi = int(ref_frame + WINDOW_S * fps)
    out = {}
    for cam in cam_ids:
        xy = project(X, calib[cam])
        x, y = int(round(xy[0])), int(round(xy[1]))
        entry = {"onset": None, "rise": 0.0, "feature": None, "xy": (x, y)}
        out[cam] = entry
        cap = cv2.VideoCapture(os.path.join(take_dir, f"video_{cam}.mjpeg"))
        blue, white = [], []
        for i in range(hi + 1):
            ok, im = cap.read()
            if not ok:
                break
            if i < lo:
                continue
            crop = im[max(y - half_box, 0):y + half_box, max(x - half_box, 0):x + half_box].astype(np.int16)
            if crop.size == 0:
                break
            b = np.maximum(crop[..., 0] - crop[..., 1:].max(2), 0)
            w = np.maximum(crop.min(2) - WHITE_FLOOR, 0)
            blue.append(cv2.GaussianBlur(b.astype(np.float32), (0, 0), 2).max())
            white.append(cv2.GaussianBlur(w.astype(np.float32), (0, 0), 2).max())
        cap.release()
        if len(blue) < 4:
            continue
        blue, white = np.array(blue), np.array(white)
        s, feat = (white, "white") if np.ptp(white) >= np.ptp(blue) else (blue, "blue")
        rise = float(np.ptp(s))
        entry["rise"], entry["feature"] = round(rise, 1), feat
        if rise >= MIN_RISE:
            entry["onset"] = lo + int(np.argmax(s >= s.min() + 0.5 * rise))
    return out
