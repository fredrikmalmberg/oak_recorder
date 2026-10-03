"""Refine camera extrinsics with Sapiens2 keypoints (robust bundle adjustment), validate on the flash LED.

  PYTHONPATH=. python -m sapiens2_test.refine_extrinsics
Fixed: intrinsics/distortion, cam0 pose (defines the world frame) and the cam0<->anchor-cam baseline length
(real-world scale from the existing board-scaled calibration). Free: other cameras' R,t and all 3D points.
Residuals are in undistorted pixel space (observations are undistorted once with the fixed intrinsics);
all reported errors are in distorted pixel space (cv2.projectPoints).
"""
import argparse, json, copy
import cv2, numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
import calibrate


def rodrigues(rv):  # (3,) -> R
    return cv2.Rodrigues(rv.astype(np.float64))[0]


def dlt(obs, Ps):
    A = []
    for (u, v), P in zip(obs, Ps):
        A += [u * P[2] - P[0], v * P[2] - P[1]]
    X = np.linalg.svd(np.asarray(A))[2][-1]
    return X[:3] / X[3]


def proj_dist(X, c):
    rvec, _ = cv2.Rodrigues(c["R"].astype(np.float64))
    p, _ = cv2.projectPoints(np.asarray(X, np.float64).reshape(-1, 3), rvec, c["t"].astype(np.float64),
                             c["K"].astype(np.float64), c["dist"].astype(np.float64))
    return p.reshape(-1, 2)


def undistort(pts, c):
    return cv2.undistortPoints(pts[:, None].astype(np.float64), c["K"], c["dist"], P=c["K"]).reshape(-1, 2)


def select(data, cams, ids, a):
    """Return list of points: each = list of (cam_index, frame_index, kp_index, ud_xy, dist_xy)."""
    kp, sc = data["kp"], data["sc"]
    F = kp.shape[0]
    P = {c: cams[c]["K"] @ np.hstack([cams[c]["R"], cams[c]["t"].reshape(3, 1)]) for c in ids}
    pts = []
    rng = np.random.default_rng(0)
    for f in range(F):
        ud = {c: undistort(np.nan_to_num(kp[f, i]), cams[c]) for i, c in enumerate(ids)}
        cand = []
        for k in range(kp.shape[2]):
            vs = [i for i, c in enumerate(ids) if sc[f, i, k] > a.min_conf and not np.isnan(kp[f, i, k, 0])]
            if len(vs) < a.min_views:
                continue
            while len(vs) >= a.min_views:   # coarse gate with the *current* calibration, loose on purpose
                X = dlt([ud[ids[i]][k] for i in vs], [P[ids[i]] for i in vs])
                e = []
                for i in vs:
                    c = cams[ids[i]]
                    z = (c["R"] @ X + c["t"])[2]
                    e.append(np.linalg.norm(proj_dist(X, c)[0] - kp[f, i, k]) if z > 0 else 1e9)
                w = int(np.argmax(e))
                if e[w] > a.gate_px:
                    vs.pop(w)
                else:
                    break
            if len(vs) >= a.min_views:
                cand.append([(i, f, k, ud[ids[i]][k], kp[f, i, k].astype(np.float64)) for i in vs])
        if len(cand) > a.max_per_frame:
            cand = [cand[j] for j in rng.choice(len(cand), a.max_per_frame, replace=False)]
        pts += cand
    return pts


def build_cams(x, cams0, ids, anchor, d0):
    """x -> camera dict (R,t) for all ids. Layout: for each cam i>=1: rvec(3)+center(3); anchor cam: rvec(3)+(theta,phi)."""
    out = {}
    C0 = -cams0[ids[0]]["R"].T @ cams0[ids[0]]["t"]
    out[ids[0]] = (cams0[ids[0]]["R"], C0)
    o = 0
    for c in ids[1:]:
        R = rodrigues(x[o:o + 3])
        if c == anchor:
            th, ph = x[o + 3:o + 5]
            C = C0 + d0 * np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)]); o += 5
        else:
            C = x[o + 3:o + 6]; o += 6
        out[c] = (R, C)
    return out


def pack(cams0, ids, anchor, d0):
    C0 = -cams0[ids[0]]["R"].T @ cams0[ids[0]]["t"]
    x = []
    for c in ids[1:]:
        R, t = cams0[c]["R"], cams0[c]["t"]
        C = -R.T @ t
        x += list(cv2.Rodrigues(R)[0].ravel())
        if c == anchor:
            u = (C - C0) / np.linalg.norm(C - C0)
            x += [float(np.arccos(u[2])), float(np.arctan2(u[1], u[0]))]
        else:
            x += list(C)
    return np.array(x)


def with_ext(cams0, ext):
    new = copy.deepcopy(cams0)
    for c, (R, C) in ext.items():
        new[c]["R"] = R; new[c]["t"] = -R @ C
    return new


def refine(pts, cams0, ids, anchor, a):
    N = len(pts)
    P0 = {c: cams0[c]["K"] @ np.hstack([cams0[c]["R"], cams0[c]["t"].reshape(3, 1)]) for c in ids}
    X0 = np.array([dlt([o[3] for o in p], [P0[ids[o[0]]] for o in p]) for p in pts])
    d0 = np.linalg.norm(-cams0[anchor]["R"].T @ cams0[anchor]["t"] + cams0[ids[0]]["R"].T @ cams0[ids[0]]["t"])
    cam_i = np.array([o[0] for p in pts for o in p]); pt_i = np.array([j for j, p in enumerate(pts) for _ in p])
    uv = np.array([o[3] for p in pts for o in p])
    Ks = np.array([cams0[c]["K"] for c in ids])
    ncp = 5 * 1 + 6 * (len(ids) - 2)
    x0 = np.concatenate([pack(cams0, ids, anchor, d0), X0.ravel()])

    def resid(x):
        ext = build_cams(x[:ncp], cams0, ids, anchor, d0)
        Rs = np.array([ext[c][0] for c in ids]); Cs = np.array([ext[c][1] for c in ids])
        X = x[ncp:].reshape(-1, 3)[pt_i]
        Xc = np.einsum("nij,nj->ni", Rs[cam_i], X - Cs[cam_i])
        K = Ks[cam_i]
        u = K[:, 0, 0] * Xc[:, 0] / Xc[:, 2] + K[:, 0, 2]; v = K[:, 1, 1] * Xc[:, 1] / Xc[:, 2] + K[:, 1, 2]
        return np.stack([u - uv[:, 0], v - uv[:, 1]], 1).ravel()

    S = lil_matrix((2 * len(cam_i), len(x0)), dtype=int)
    cam_cols = {}
    o = 0
    for i, c in enumerate(ids[1:], 1):
        n = 5 if c == anchor else 6
        cam_cols[i] = range(o, o + n); o += n
    for r, (ci, pj) in enumerate(zip(cam_i, pt_i)):
        for rr in (2 * r, 2 * r + 1):
            for col in cam_cols.get(ci, []):
                S[rr, col] = 1
            for k in range(3):
                S[rr, ncp + 3 * pj + k] = 1
    sol = least_squares(resid, x0, jac_sparsity=S, loss=a.loss, f_scale=a.f_scale, x_scale="jac", method="trf",
                        max_nfev=a.max_nfev, ftol=1e-10, xtol=1e-10, gtol=1e-10, verbose=1)
    ext = build_cams(sol.x[:ncp], cams0, ids, anchor, d0)
    return with_ext(cams0, ext), sol.x[ncp:].reshape(-1, 3), X0


def refine_intr(pts, cams0, ids, anchor, a):
    """Like refine(), but also frees per-camera intrinsics. Residuals in *distorted* pixel space.
    Per camera: log focal scale (fx, fy together), cx, cy [, k1, k2 if --intr fck]."""
    n_i = 3 if a.intr == "fc" else 5
    N = len(pts)
    P0 = {c: cams0[c]["K"] @ np.hstack([cams0[c]["R"], cams0[c]["t"].reshape(3, 1)]) for c in ids}
    X0 = np.array([dlt([o[3] for o in p], [P0[ids[o[0]]] for o in p]) for p in pts])
    d0 = np.linalg.norm(-cams0[anchor]["R"].T @ cams0[anchor]["t"] + cams0[ids[0]]["R"].T @ cams0[ids[0]]["t"])
    cam_i = np.array([o[0] for p in pts for o in p]); pt_i = np.array([j for j, p in enumerate(pts) for _ in p])
    uv = np.array([o[4] for p in pts for o in p])
    K0 = np.array([cams0[c]["K"] for c in ids]); D0 = np.array([cams0[c]["dist"] for c in ids])
    ncp = 5 + 6 * (len(ids) - 2); nin = n_i * len(ids)
    x0 = np.concatenate([pack(cams0, ids, anchor, d0), np.zeros(nin), X0.ravel()])

    def intr(xi):
        v = xi.reshape(len(ids), n_i)
        K = K0.copy(); D = D0.copy()
        K[:, 0, 0] = K0[:, 0, 0] * np.exp(v[:, 0]); K[:, 1, 1] = K0[:, 1, 1] * np.exp(v[:, 0])
        K[:, 0, 2] = K0[:, 0, 2] + v[:, 1]; K[:, 1, 2] = K0[:, 1, 2] + v[:, 2]
        if n_i == 5:
            D[:, 0] = D0[:, 0] + v[:, 3]; D[:, 1] = D0[:, 1] + v[:, 4]
        return K, D

    def resid(x):
        ext = build_cams(x[:ncp], cams0, ids, anchor, d0)
        K, D = intr(x[ncp:ncp + nin])
        Rs = np.array([ext[c][0] for c in ids]); Cs = np.array([ext[c][1] for c in ids])
        X = x[ncp + nin:].reshape(-1, 3)[pt_i]
        Xc = np.einsum("nij,nj->ni", Rs[cam_i], X - Cs[cam_i])
        xn, yn = Xc[:, 0] / Xc[:, 2], Xc[:, 1] / Xc[:, 2]
        d = D[cam_i]; r2 = xn * xn + yn * yn
        rad = 1 + d[:, 0] * r2 + d[:, 1] * r2 ** 2 + d[:, 4] * r2 ** 3
        xd = xn * rad + 2 * d[:, 2] * xn * yn + d[:, 3] * (r2 + 2 * xn * xn)
        yd = yn * rad + d[:, 2] * (r2 + 2 * yn * yn) + 2 * d[:, 3] * xn * yn
        k = K[cam_i]
        return np.stack([k[:, 0, 0] * xd + k[:, 0, 2] - uv[:, 0], k[:, 1, 1] * yd + k[:, 1, 2] - uv[:, 1]], 1).ravel()

    S = lil_matrix((2 * len(cam_i), len(x0)), dtype=int)
    cam_cols = {}
    o = 0
    for i, c in enumerate(ids[1:], 1):
        n = 5 if c == anchor else 6
        cam_cols[i] = list(range(o, o + n)); o += n
    for r, (ci, pj) in enumerate(zip(cam_i, pt_i)):
        cols = cam_cols.get(ci, []) + list(range(ncp + ci * n_i, ncp + (ci + 1) * n_i))
        for rr in (2 * r, 2 * r + 1):
            for col in cols:
                S[rr, col] = 1
            for k in range(3):
                S[rr, ncp + nin + 3 * pj + k] = 1
    r0 = resid(x0)
    print(f"sanity: initial residual median {np.median(np.abs(r0)):.2f} px (should match old-calibration error scale)")
    sol = least_squares(resid, x0, jac_sparsity=S, loss=a.loss, f_scale=a.f_scale, x_scale="jac", method="trf",
                        max_nfev=a.max_nfev, ftol=1e-10, xtol=1e-10, gtol=1e-10, verbose=1)
    ext = build_cams(sol.x[:ncp], cams0, ids, anchor, d0)
    K, D = intr(sol.x[ncp:ncp + nin])
    new = with_ext(cams0, ext)
    for i, c in enumerate(ids):
        new[c]["K"] = K[i]; new[c]["dist"] = D[i]
    return new, sol.x[ncp + nin:].reshape(-1, 3), X0


def obs_errors(pts, X, cams, ids):
    e = {c: [] for c in ids}
    for p, x in zip(pts, X):
        for o in p:
            e[ids[o[0]]].append(np.linalg.norm(proj_dist(x, cams[ids[o[0]]])[0] - o[4]))
    return {c: np.array(v) for c, v in e.items()}


def retriangulate_error(pts, cams, ids):
    """Held-out check: triangulate each point from its views using `cams`, error of those views (dist. px)."""
    P = {c: cams[c]["K"] @ np.hstack([cams[c]["R"], cams[c]["t"].reshape(3, 1)]) for c in ids}
    X = np.array([dlt([cv2.undistortPoints(o[4].reshape(1, 1, 2), cams[ids[o[0]]]["K"], cams[ids[o[0]]]["dist"], P=cams[ids[o[0]]]["K"]).reshape(2)
                       for o in p], [P[ids[o[0]]] for o in p]) for p in pts])
    return obs_errors(pts, X, cams, ids)


def led_metrics(cams, ids, led_pts):
    P = {c: cams[c]["K"] @ np.hstack([cams[c]["R"], cams[c]["t"].reshape(3, 1)]) for c in ids}
    ud = {c: undistort(led_pts[c].reshape(1, 2), cams[c])[0] for c in ids}
    X = dlt([ud[c] for c in ids], [P[c] for c in ids])
    res = {}
    for c in ids:
        oth = [o for o in ids if o != c]
        Xl = dlt([ud[o] for o in oth], [P[o] for o in oth])
        res[c] = (np.linalg.norm(proj_dist(X, cams[c])[0] - led_pts[c]), np.linalg.norm(proj_dist(Xl, cams[c])[0] - led_pts[c]))
    return res


def med(x):
    return f"{np.median(x):6.2f}" if len(x) else "   n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="output/sapiens2/multiframe_1b.npz")
    ap.add_argument("--calib", default="output/calibration/20260925_sp_lg_7cam_scaled.json")
    ap.add_argument("--led", default="output/flash_reproj.json")
    ap.add_argument("--out", default="output/calibration/20260925_sp_lg_7cam_scaled_sapiens_refined.json")
    ap.add_argument("--anchor", default="cam4", help="camera whose baseline to cam0 stays at its calibrated length")
    ap.add_argument("--min-conf", type=float, default=0.6)
    ap.add_argument("--min-views", type=int, default=3)
    ap.add_argument("--gate-px", type=float, default=40.0)
    ap.add_argument("--max-per-frame", type=int, default=120)
    ap.add_argument("--loss", default="cauchy", choices=["cauchy", "huber", "soft_l1"])
    ap.add_argument("--f-scale", type=float, default=4.0)
    ap.add_argument("--max-nfev", type=int, default=60)
    ap.add_argument("--intr", default="none", choices=["none", "fc", "fck"],
                    help="also fit intrinsics: fc = focal+principal point, fck = + k1,k2")
    ap.add_argument("--holdout", action="store_true", help="fit on even frames, report on odd frames")
    a = ap.parse_args()

    cams0 = calibrate.load_calibration_output(a.calib)
    d = np.load(a.data)
    ids = [str(c) for c in d["cams"]]
    data = dict(kp=d["kp"], sc=d["sc"])
    print(f"data: {d['frames'].shape[0]} frames x {len(ids)} cams {ids}")
    pts = select(data, cams0, ids, a)
    nv = np.array([len(p) for p in pts])
    print(f"selected {len(pts)} points, {nv.sum()} observations (views/pt mean {nv.mean():.2f}); "
          f"per-cam obs {[int(sum(1 for p in pts for o in p if o[0] == i)) for i in range(len(ids))]}")
    train = [p for p in pts if (p[0][1] % 2 == 0)] if a.holdout else pts
    test = [p for p in pts if p[0][1] % 2 == 1] if a.holdout else []
    print(f"fit on {len(train)} points" + (f", hold out {len(test)}" if a.holdout else ""))

    cams1, X1, X0 = (refine if a.intr == 'none' else refine_intr)(train, cams0, ids, a.anchor, a)

    e0 = obs_errors(train, X0, cams0, ids)
    e1 = obs_errors(train, X1, cams1, ids)
    print("\nfit-set reprojection error, distorted px (median / p90)   [before: DLT with old cams | after: BA]")
    for c in ids:
        print(f"  {c}: before {med(e0[c])} / {np.percentile(e0[c],90):6.1f}    after {med(e1[c])} / {np.percentile(e1[c],90):6.1f}")
    print(f"  ALL: before {med(np.concatenate(list(e0.values())))}    after {med(np.concatenate(list(e1.values())))}")
    if a.holdout:
        h0 = retriangulate_error(test, cams0, ids); h1 = retriangulate_error(test, cams1, ids)
        print("\nHELD-OUT frames (re-triangulated from each camera set), median px")
        for c in ids:
            print(f"  {c}: old {med(h0[c])}   new {med(h1[c])}")
        print(f"  ALL: old {med(np.concatenate(list(h0.values())))}   new {med(np.concatenate(list(h1.values())))}")

    led = json.load(open(a.led))
    led_pts = {c: np.mean([np.array(led[f]["cams"][c]["pt"]) for f in led], 0) for c in ids}
    m0, m1 = led_metrics(cams0, ids, led_pts), led_metrics(cams1, ids, led_pts)
    print("\nFLASH LED (independent metric, not used in the fit), distorted px: all-cams err / leave-one-out err")
    for c in ids:
        print(f"  {c}: old {m0[c][0]:6.2f} / {m0[c][1]:6.2f}    new {m1[c][0]:6.2f} / {m1[c][1]:6.2f}")
    print(f"  mean: old {np.mean([v[0] for v in m0.values()]):.2f} / {np.mean([v[1] for v in m0.values()]):.2f}    "
          f"new {np.mean([v[0] for v in m1.values()]):.2f} / {np.mean([v[1] for v in m1.values()]):.2f}")

    print("\ncamera change (rotation deg, centre shift m):")
    for c in ids[1:]:
        dR = np.degrees(np.linalg.norm(cv2.Rodrigues(cams1[c]["R"] @ cams0[c]["R"].T)[0]))
        dC = np.linalg.norm((-cams1[c]["R"].T @ cams1[c]["t"]) - (-cams0[c]["R"].T @ cams0[c]["t"]))
        print(f"  {c}: {dR:.2f} deg, {dC*100:.1f} cm")

    if a.intr != "none":
        print("\nintrinsics change (focal %, cx px, cy px, k1, k2):")
        for c in ids:
            k0, k1 = cams0[c]["K"], cams1[c]["K"]
            print(f"  {c}: f {100*(k1[0,0]/k0[0,0]-1):+.2f}%  cx {k1[0,2]-k0[0,2]:+.1f}  cy {k1[1,2]-k0[1,2]:+.1f}  "
                  f"k1 {cams1[c]['dist'][0]-cams0[c]['dist'][0]:+.4f}  k2 {cams1[c]['dist'][1]-cams0[c]['dist'][1]:+.4f}")
    payload = json.load(open(a.calib))
    for c in ids[1:]:
        ex = payload["cameras"][c]["extrinsics"]
        ex["rotation"] = cams1[c]["R"].tolist(); ex["translation"] = cams1[c]["t"].reshape(-1).tolist()
    if a.intr != "none":
        for c in ids:
            it = payload["cameras"][c]["intrinsics"]
            it["camera_matrix"] = cams1[c]["K"].tolist(); it["dist_coeffs"] = cams1[c]["dist"].tolist()
            it["source"] = "Sep8-cache + Sapiens2 refinement"
    payload["metadata"]["method"] += f" + Sapiens2 keypoint refinement ({a.loss}, anchor {a.anchor})"
    json.dump(payload, open(a.out, "w"), indent=2)
    print("saved", a.out)


if __name__ == "__main__":
    main()
