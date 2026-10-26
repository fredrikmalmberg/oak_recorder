"""Calibration check from the take's own MediaPipe body detections (fast3d).

For every camera c: triangulate the confident upper-body points (BlazePose
0-24 with visibility >= MIN_VIS) from the OTHER cameras (weighted DLT, at
least 2 of them) and reproject them into c. The median error is c's
leave-one-out reprojection error, in c's pixels, on every --step-th frame.
A wrong intrinsic or extrinsic for camera c, or swapped camera labels, shows
as a large error for c; a calibration that does not fit the take at all
(other rig setup, other resolution) shows as large errors everywhere. Errors
are also given as a fraction of the image diagonal so thresholds do not
depend on the resolution.

A camera is flagged when its error is above WARN_FRAC of the diagonal AND
above REL_FLAG x the median of the other cameras (a bad camera also raises
the others' errors somewhat, since it takes part in their triangulation).
Verdict from the median over cameras: ok up to WARN_FRAC, warning up to
BAD_FRAC, bad above; a flagged camera makes ok a warning. Thresholds from
take_2 of 20261022_171905 with its calibration (all cameras 6-13 px, median
10 px at 2016x1520) and damaged copies of it (README.md, "fast3d").

    python -m fast3d.calib_check <take_dir> --calib CAL [--calib CAL2 ...]
"""
import argparse
import json
import os

import numpy as np

MIN_VIS = 0.7
UPPER = list(range(25))
WARN_FRAC = 0.006       # of the image diagonal (2016x1520: 15 px)
BAD_FRAC = 0.012        # (30 px)
REL_FLAG = 1.8


def _undistort(px, K, dist):
    import cv2
    out = np.full_like(px, np.nan)
    ok = np.isfinite(px).all(-1)
    if ok.any():
        out[ok] = cv2.undistortPoints(px[ok][:, None].astype(np.float64), K, dist, P=K)[:, 0]
    return out


def _dlt(P, x, w):
    """P (C, 3, 4), x (C, N, 2), w (C, N) weights (0 = unused) -> X (N, 3), NaN with < 2 views."""
    C, N = w.shape
    A = np.zeros((N, 2 * C, 4))
    A[:, 0::2] = (x[..., 0:1] * P[:, None, 2] - P[:, None, 0]).transpose(1, 0, 2) * w.T[..., None]
    A[:, 1::2] = (x[..., 1:2] * P[:, None, 2] - P[:, None, 1]).transpose(1, 0, 2) * w.T[..., None]
    A = np.nan_to_num(A)
    _, _, vt = np.linalg.svd(A)
    X = vt[:, -1]
    X = X[:, :3] / X[:, 3:4]
    X[(w > 0).sum(0) < 2] = np.nan
    return X


def check(per_cam_body, per_cam_vis, cams, calib, width, height, step=5):
    """per_cam_body[c] (T, 33, 3) normalised raw-image coordinates, per_cam_vis[c] (T, 33).
    Returns dict per_camera_px, per_camera_frac, median_px, median_frac, verdict, flagged, n_points."""
    T = per_cam_body[cams[0]].shape[0]
    frames = np.arange(0, T, step)
    P = np.stack([calib[c]['K'] @ np.hstack([calib[c]['R'], np.reshape(calib[c]['t'], (3, 1))]) for c in cams])
    x = np.stack([_undistort(per_cam_body[c][frames][:, UPPER, :2] * (width, height), calib[c]['K'], calib[c]['dist'])
                  for c in cams])                                                     # (C, F, J, 2)
    v = np.stack([per_cam_vis[c][frames][:, UPPER] for c in cams])
    w = np.where((v >= MIN_VIS) & np.isfinite(x).all(-1), v, 0.0)                   # (C, F, J)
    C = len(cams)
    xf, wf = x.reshape(C, -1, 2), w.reshape(C, -1)
    errs = {}
    for ci, c in enumerate(cams):
        wo = wf.copy()
        wo[ci] = 0
        X = _dlt(P, xf, wo)
        q = np.c_[X, np.ones(len(X))] @ P[ci].T
        e = np.linalg.norm(q[:, :2] / q[:, 2:3] - xf[ci], axis=1)
        e = e[(wf[ci] > 0) & np.isfinite(e)]
        errs[c] = (float(np.median(e)) if len(e) else float('nan'), int(len(e)))
    diag = float(np.hypot(width, height))
    px = {c: e for c, (e, _) in errs.items()}
    frac = {c: e / diag for c, e in px.items()}
    med = float(np.nanmedian(list(px.values())))
    flagged = sorted(c for c, e in px.items()
                     if not (e <= WARN_FRAC * diag or e <= REL_FLAG * np.nanmedian([px[o] for o in px if o != c])))
    verdict = 'bad' if med > BAD_FRAC * diag else 'warning' if (med > WARN_FRAC * diag or flagged) else 'ok'
    return dict(per_camera_px=px, per_camera_frac=frac, median_px=med, median_frac=med / diag, verdict=verdict,
                flagged=flagged, n_points={c: n for c, (_, n) in errs.items()}, warn_px=WARN_FRAC * diag,
                bad_px=BAD_FRAC * diag)


def check_take(take_dir, calib_path, step=5):
    import calibrate
    d = np.load(os.path.join(take_dir, 'processed', 'fast3d', 'fast3d_2d.npz'))
    cams = [str(c) for c in d['cams']]
    with open(os.path.join(take_dir, 'take_meta.json'), encoding='utf-8') as f:
        width, height = json.load(f)['resolution']
    calib = calibrate.load_calibration_output(calib_path)
    missing = [c for c in cams if c not in calib]
    if missing:
        return dict(verdict='bad', error=f'calibration has no {", ".join(missing)}')
    return check({c: d[f'{c}_body'] for c in cams}, {c: d[f'{c}_body_vis'] for c in cams}, cams, calib, width, height, step)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('take_dir')
    ap.add_argument('--calib', action='append', required=True)
    ap.add_argument('--step', type=int, default=5)
    a = ap.parse_args()
    for p in a.calib:
        r = check_take(os.path.abspath(a.take_dir), p, a.step)
        cams = ' '.join(f'{c} {e:.1f}' for c, e in r.get('per_camera_px', {}).items())
        print(f'{os.path.basename(p)}: {r["verdict"]}  median {r.get("median_px", float("nan")):.1f} px  '
              f'flagged {r.get("flagged", r.get("error"))}  [{cams}]')


if __name__ == '__main__':
    main()
