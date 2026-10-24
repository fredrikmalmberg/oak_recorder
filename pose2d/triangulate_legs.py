"""Triangulate the Sapiens leg keypoints (goliath ids 11-20: knees, ankles, toes, heels) from the 308-point extraction.

Cameras where the point is clamped to the bottom of the frame (the foot is cut off) are skipped. Per frame and landmark:
RANSAC over camera pairs, inliers within --thresh-px, refit on the inliers.

  python -m pose2d.triangulate_legs --npz output/sapiens2_171905_t2/allframes_1b_smoothbox.npz --calib CAL --out legs.npz
"""
import argparse, itertools, os, sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import calibrate  # noqa: E402

LEG_IDS = list(range(11, 21))
NAMES = ["l_knee", "r_knee", "l_ankle", "r_ankle", "l_big_toe", "l_small_toe", "l_heel", "r_big_toe", "r_small_toe", "r_heel"]


def dlt(uv, Ps):
    A = []
    for (u, v), P in zip(uv, Ps):
        A += [u * P[2] - P[0], v * P[2] - P[1]]
    X = np.linalg.svd(np.asarray(A))[2][-1]
    return X[:3] / X[3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-score", type=float, default=0.5)
    ap.add_argument("--edge-px", type=float, default=25.0, help="skip a camera whose point is within this many px of the bottom edge")
    ap.add_argument("--thresh-px", type=float, default=15.0)
    a = ap.parse_args()
    calib = calibrate.load_calibration_output(a.calib)
    d = np.load(a.npz)
    frames, cams, kp, sc = d["frames"], [str(c) for c in d["cams"]], d["kp"], d["sc"]
    F = len(frames)
    P, K, ud = {}, {}, {}
    for ci, c in enumerate(cams):
        K[c] = calib[c]["K"]
        P[c] = K[c] @ np.hstack([calib[c]["R"], np.asarray(calib[c]["t"]).reshape(3, 1)])
        raw = kp[:, ci][:, LEG_IDS].reshape(-1, 1, 2).astype(np.float64)
        ok = np.isfinite(raw).all(axis=(1, 2))
        u = np.full((len(raw), 2), np.nan)
        u[ok] = cv2.undistortPoints(raw[ok], K[c], calib[c]["dist"], P=K[c]).reshape(-1, 2)
        ud[c] = u.reshape(F, len(LEG_IDS), 2)
    H = calib[cams[0]]["height"]
    X = np.full((F, len(LEG_IDS), 3), np.nan)
    ncam = np.zeros((F, len(LEG_IDS)), int)
    err = np.full((F, len(LEG_IDS)), np.nan)
    for f in range(F):
        for j, lid in enumerate(LEG_IDS):
            cand = []
            for ci, c in enumerate(cams):
                p = ud[c][f, j]
                raw_y = kp[f, ci, lid, 1]
                if np.isfinite(p).all() and sc[f, ci, lid] >= a.min_score and raw_y < H - a.edge_px:
                    cand.append((c, p))
            if len(cand) < 2:
                continue
            best = None
            for (c1, p1), (c2, p2) in itertools.combinations(cand, 2):
                Xc = dlt([p1, p2], [P[c1], P[c2]])
                inl, es = [], []
                for c, p in cand:
                    h = P[c] @ np.append(Xc, 1.0)
                    e = np.linalg.norm(h[:2] / h[2] - p)
                    if e < a.thresh_px:
                        inl.append((c, p)); es.append(e)
                key = (len(inl), -np.mean(es))
                if best is None or key > best[0]:
                    best = (key, inl)
            inl = best[1]
            if len(inl) < 2:
                continue
            Xf = dlt([p for _, p in inl], [P[c] for c, _ in inl])
            X[f, j], ncam[f, j] = Xf, len(inl)
            es = [np.linalg.norm((P[c] @ np.append(Xf, 1.0))[:2] / (P[c] @ np.append(Xf, 1.0))[2] - p) for c, p in inl]
            err[f, j] = np.mean(es)
    np.savez(a.out, frames=frames, ids=np.array(LEG_IDS), names=np.array(NAMES), X=X, ncam=ncam, err=err)
    for j, n in enumerate(NAMES):
        print(f"{n:12s} triangulated {np.isfinite(X[:, j, 0]).mean() * 100:5.1f}%  cams median {np.median(ncam[:, j][ncam[:, j] > 0]):.0f}  reproj median {np.nanmedian(err[:, j]):.1f} px")


if __name__ == "__main__":
    main()
