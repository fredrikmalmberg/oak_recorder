"""Triangulate Sapiens2 308 keypoints (conf > thr) with the calibration and report reprojection error.

  python -m sapiens2.triangulate --size 1b --frame 300 --min-conf 0.5
Keypoints are in distorted pixel space (as saved by run_pose.py); they are undistorted before DLT and
errors are measured in distorted space (cv2.projectPoints with dist coeffs).
"""
import argparse
import cv2, numpy as np
import calibrate


def proj(X, c):
    rvec, _ = cv2.Rodrigues(c["R"].astype(np.float64))
    p, _ = cv2.projectPoints(X.reshape(-1, 3).astype(np.float64), rvec, c["t"].astype(np.float64),
                             c["K"].astype(np.float64), c["dist"].astype(np.float64))
    return p.reshape(-1, 2)


def dlt(obs_ud, Ps):
    A = []
    for (u, v), P in zip(obs_ud, Ps):
        A += [u * P[2] - P[0], v * P[2] - P[1]]
    X = np.linalg.svd(np.asarray(A))[2][-1]
    return X[:3] / X[3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="1b")
    ap.add_argument("--frame", type=int, default=300)
    ap.add_argument("--cams", nargs="+", default=["cam0", "cam1", "cam2", "cam3", "cam4", "cam5"])
    ap.add_argument("--min-conf", type=float, default=0.5)
    ap.add_argument("--drop-px", type=float, default=15.0, help="robust mode: drop worst view while its error exceeds this")
    ap.add_argument("--calib", default="output/calibration/20260925_sp_lg_7cam_scaled.json")
    ap.add_argument("--dir", default="output/sapiens2")
    ap.add_argument("--save", default=None)
    a = ap.parse_args()

    cams = calibrate.load_calibration_output(a.calib)
    kp, sc = {}, {}
    for c in a.cams:
        d = np.load(f"{a.dir}/{a.size}_{c}_f{a.frame}.npz")
        kp[c], sc[c] = d["kp"], d["score"]
    K_ = len(kp[a.cams[0]])
    P = {c: cams[c]["K"] @ np.hstack([cams[c]["R"], cams[c]["t"].reshape(3, 1)]) for c in a.cams}
    ud = {c: cv2.undistortPoints(kp[c][:, None].astype(np.float64), cams[c]["K"], cams[c]["dist"], P=cams[c]["K"]).reshape(-1, 2)
          for c in a.cams}

    X_all = np.full((K_, 3), np.nan); X_rob = np.full((K_, 3), np.nan)
    err_all = {c: np.full(K_, np.nan) for c in a.cams}   # plain DLT, every view with conf>thr
    err_rob = {c: np.full(K_, np.nan) for c in a.cams}   # after dropping outlier views
    nviews = np.zeros(K_, int); nviews_rob = np.zeros(K_, int)
    for k in range(K_):
        vs = [c for c in a.cams if sc[c][k] > a.min_conf]
        nviews[k] = len(vs)
        if len(vs) < 2:
            continue
        X = dlt([ud[c][k] for c in vs], [P[c] for c in vs]); X_all[k] = X
        for c in vs:
            err_all[c][k] = np.linalg.norm(proj(X, cams[c])[0] - kp[c][k])
        act = list(vs)
        while True:
            X = dlt([ud[c][k] for c in act], [P[c] for c in act])
            e = {c: np.linalg.norm(proj(X, cams[c])[0] - kp[c][k]) for c in act}
            w = max(e, key=e.get)
            if e[w] > a.drop_px and len(act) > 2:
                act.remove(w)
            else:
                break
        X_rob[k] = X; nviews_rob[k] = len(act)
        for c in act:
            err_rob[c][k] = e[c]

    def stats(x):
        x = x[~np.isnan(x)]
        return f"n={len(x):5d} mean={x.mean():6.2f} med={np.median(x):6.2f} p90={np.percentile(x,90):6.2f} max={x.max():7.1f}" if len(x) else "n=0"

    print(f"{a.size} frame {a.frame}, conf>{a.min_conf}, cams {a.cams}")
    print(f"keypoints with >=2 views: {(nviews>=2).sum()}/{K_}; views/kp mean {nviews[nviews>=2].mean():.2f}")
    print("\n-- plain DLT over all confident views (reprojection error px, distorted space) --")
    for c in a.cams:
        print(f"  {c}: {stats(err_all[c])}")
    print("  ALL :", stats(np.concatenate([err_all[c] for c in a.cams])))
    print(f"\n-- robust: iteratively drop worst view while err > {a.drop_px}px (min 2 views) --")
    for c in a.cams:
        print(f"  {c}: {stats(err_rob[c])}")
    print("  ALL :", stats(np.concatenate([err_rob[c] for c in a.cams])))
    kept = sum(~np.isnan(err_rob[c]) for c in a.cams).sum(); tot = sum(~np.isnan(err_all[c]) for c in a.cams).sum()
    print(f"  views kept {kept}/{tot}; keypoints triangulated {(~np.isnan(X_rob[:,0])).sum()}; views/kp after {nviews_rob[nviews_rob>0].mean():.2f}")
    if a.save:
        np.savez(a.save, X_all=X_all, X_rob=X_rob, nviews=nviews, nviews_rob=nviews_rob)


if __name__ == "__main__":
    main()
