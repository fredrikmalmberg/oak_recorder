"""Measure where Sapiens and WiLoR keypoints sit relative to the exact SMPL-X targets they are mapped to in the fit
(smplx_fit.joint_mapping), using the renders from synth_render.py. Triangulates each source over all views (DLT, worst
view dropped while reprojection > --max-reproj-px) and reports triangulated - SMPL-X in each hand's own frame
(data_loading.hand_frame convention: x wrist->middle MCP, z palm normal, y = z x x) and, for the body, in a person frame.

  PYTHONPATH=. python -m smplx_fit.synth_analyze --dir <synth dir> [--variant textured]
Writes <dir>/analysis_<variant>.npz with the per-pose offsets.
"""
import argparse
import os

import numpy as np

from hand_pose.wilor_wrist import triangulate_frame
from smplx_fit import joint_mapping as jm

HAND_RIGHT = [41, 24, 23, 22, 21, 28, 27, 26, 25, 32, 31, 30, 29, 36, 35, 34, 33, 40, 39, 38, 37]
HAND_LEFT = [62, 45, 44, 43, 42, 49, 48, 47, 46, 53, 52, 51, 50, 57, 56, 55, 54, 61, 60, 59, 58]
BODY_COCO = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
BODY_SAP = [0, 1, 2, 3, 4, 5, 6, 7, 8, 62, 41, 9, 10]
BODY_NAMES = ["nose", "L eye", "R eye", "L ear", "R ear", "L shoulder", "R shoulder", "L elbow", "R elbow", "L wrist", "R wrist",
              "L hip", "R hip"]
HAND_NAMES = ["wrist"] + [f"{f} {s}" for f in ("thumb", "index", "middle", "ring", "pinky") for s in ("MCP", "PIP", "DIP", "tip")]


def gt_point(verts, joints, tgt):
    return (joints if tgt.kind == "joint" else verts)[tgt.index]


def hand_targets(side):
    t = jm.LEFT_HAND_TARGETS if side == "left" else jm.RIGHT_HAND_TARGETS
    wr = jm.WRIST_FUSION[side]["target"]
    return [wr] + [t[str(i)] for i in range(1, 21)]


def body_targets():
    t = dict(jm.BODY_TARGETS)
    t["9"], t["10"] = jm.WRIST_FUSION["left"]["target"], jm.WRIST_FUSION["right"]["target"]
    return [t[str(i)] for i in BODY_COCO]


def frame_from(pts):
    """pts: GT hand landmarks [21,3] -> columns x, y, z (same as data_loading.hand_frame)."""
    w, im, m, p = pts[0], pts[5], pts[9], pts[17]
    x = m - w
    n = np.cross(im - w, p - w)
    n = n - (n @ x) / (x @ x) * x
    x, n = x / np.linalg.norm(x), n / np.linalg.norm(n)
    return np.stack([x, np.cross(n, x), n], axis=1)


def person_frame(joints):
    """x: right->left shoulder (lateral), y: pelvis->neck (up), z = x x y (forward-ish)."""
    lat = joints[16] - joints[17]
    up = joints[12] - joints[0]
    z = np.cross(lat, up); z /= np.linalg.norm(z)
    y = up - (up @ z) * z; y /= np.linalg.norm(y)
    x = np.cross(y, z)
    return np.stack([x, y, z], axis=1)


def project(K, R, t, X):
    c = X @ R.T + t
    return np.stack([K[0, 0] * c[:, 0] / c[:, 2] + K[0, 2], K[1, 1] * c[:, 1] / c[:, 2] + K[1, 2]], axis=1), c[:, 2]


def triangulate(pix, valid, Ps, max_reproj):
    """pix [V,2], valid [V] -> X or nan, n views used."""
    d = {v: pix[v] for v in range(len(pix)) if valid[v] and np.isfinite(pix[v]).all()}
    r = triangulate_frame(d, Ps, max_reproj)
    return (np.full(3, np.nan), 0) if r is None else (r[0], len(r[1]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--variant", default="textured")
    ap.add_argument("--min-score", type=float, default=0.3)
    ap.add_argument("--max-reproj-px", type=float, default=15.0)
    a = ap.parse_args()
    gt = np.load(f"{a.dir}/gt.npz")
    sap = np.load(f"{a.dir}/sapiens_{a.variant}.npz")
    wil = np.load(f"{a.dir}/wilor_{a.variant}.npz")
    P, V = len(gt["pose_names"]), len(gt["view_names"])
    K, Rs, ts = gt["K"], gt["R"], gt["t"]
    out = {}
    btg = body_targets()
    # body: offsets [P,13,3] in world and in person frame, per source (sapiens only)
    for src in ("sapiens", "wilor"):
        out[f"{src}_hand"] = np.full((P, 2, 21, 3), np.nan)       # hand frame
        out[f"{src}_hand_world"] = np.full((P, 2, 21, 3), np.nan)
        out[f"{src}_hand_nviews"] = np.zeros((P, 2, 21), int)
        out[f"{src}_hand_px"] = np.full((P, 2, 21, V), np.nan)    # per-view 2D error, px
        out[f"{src}_hand_mm"] = np.full((P, 2, 21, V), np.nan)    # same, scaled to mm at the keypoint depth
    out["sapiens_body"] = np.full((P, 13, 3), np.nan)
    out["sapiens_body_world"] = np.full((P, 13, 3), np.nan)
    out["sapiens_body_px"] = np.full((P, 13, V), np.nan)
    out["sapiens_body_mm"] = np.full((P, 13, V), np.nan)
    out["gt_hand_frames"] = np.zeros((P, 2, 3, 3))
    for p in range(P):
        Ps = {v: K @ np.hstack([Rs[p, v], ts[p, v][:, None]]) for v in range(V)}
        vt, jt = gt["verts"][p], gt["joints"][p]
        pf = person_frame(jt)
        # body (Sapiens)
        for k, tg in enumerate(btg):
            X0 = gt_point(vt, jt, tg)
            pix = sap["kp"][p][:, BODY_SAP[k]]
            ok = sap["sc"][p][:, BODY_SAP[k]] > a.min_score
            X, n = triangulate(pix, ok, Ps, a.max_reproj_px)
            out["sapiens_body_world"][p, k] = X - X0
            out["sapiens_body"][p, k] = pf.T @ (X - X0)
            for v in range(V):
                g, z = project(K, Rs[p, v], ts[p, v], X0[None])
                e = np.linalg.norm(pix[v] - g[0])
                out["sapiens_body_px"][p, k, v] = e
                out["sapiens_body_mm"][p, k, v] = e * z[0] / K[0, 0] * 1000 if ok[v] else np.nan
        # hands
        for s, side in enumerate(("left", "right")):
            tg = hand_targets(side)
            G = np.stack([gt_point(vt, jt, t_) for t_ in tg])
            Fr = frame_from(G)
            out["gt_hand_frames"][p, s] = Fr
            ids = HAND_LEFT if side == "left" else HAND_RIGHT
            for src in ("sapiens", "wilor"):
                if src == "sapiens":
                    pixs = sap["kp"][p][:, ids]                       # [V,21,2]
                    oks = sap["sc"][p][:, ids] > a.min_score          # [V,21]
                else:
                    pixs = wil["kp2d"][p][:, s]
                    oks = np.repeat(wil["valid"][p][:, s][:, None], 21, 1)
                for k in range(21):
                    X, n = triangulate(pixs[:, k], oks[:, k], Ps, a.max_reproj_px)
                    out[f"{src}_hand_world"][p, s, k] = X - G[k]
                    out[f"{src}_hand"][p, s, k] = Fr.T @ (X - G[k])
                    out[f"{src}_hand_nviews"][p, s, k] = n
                    for v in range(V):
                        g, z = project(K, Rs[p, v], ts[p, v], G[k][None])
                        e = np.linalg.norm(pixs[v, k] - g[0])
                        out[f"{src}_hand_px"][p, s, k, v] = e
                        out[f"{src}_hand_mm"][p, s, k, v] = e * z[0] / K[0, 0] * 1000 if oks[v, k] else np.nan
    np.savez(f"{a.dir}/analysis_{a.variant}.npz", **out, pose_names=gt["pose_names"])

    mm = lambda x: np.round(x * 1000, 1)  # noqa: E731
    names = [str(n) for n in gt["pose_names"]]
    fit = np.arange(P) > 0
    print(f"variant {a.variant}; offsets = triangulated - SMPL-X target, mm, hand frame (x along wrist->middle MCP, y across, z palm normal)")
    for s, side in enumerate(("left", "right")):
        print(f"\n== {side} wrist ==")
        for src in ("sapiens", "wilor"):
            o = out[f"{src}_hand"][:, s, 0]
            print(f"{src:8s} per pose:", {names[i]: mm(o[i]).tolist() for i in range(P)})
            print(f"{src:8s} median T-pose {mm(o[0]).tolist()}  median fits {mm(np.nanmedian(o[1:], 0)).tolist()}  std fits {mm(np.nanstd(o[1:], 0)).tolist()}")
        d = out["wilor_hand"][:, s, 0] - out["sapiens_hand"][:, s, 0]
        print(f"WiLoR - Sapiens (the real-data quantity: left +23.8/+2.0/-14.1, right +23.1/+1.8/+12.4) median fits {mm(np.nanmedian(d[1:], 0)).tolist()}  T-pose {mm(d[0]).tolist()}")
    print("\n== body, Sapiens - SMPL-X, person frame x lateral(R->L), y up, z forward; median over poses, mm; median 2D err mm ==")
    for k, nme in enumerate(BODY_NAMES):
        print(f"{nme:11s} {mm(np.nanmedian(out['sapiens_body'][:, k], 0)).tolist()}  |3D| {mm(np.nanmedian(np.linalg.norm(out['sapiens_body'][:, k], axis=-1))):5.1f}  2D {np.nanmedian(out['sapiens_body_mm'][:, k]):.1f}")
    print("\n== hand keypoints, hand frame, median over fit poses and both hands (right hand z mirrored), mm ==")
    for k, nme in enumerate(HAND_NAMES):
        row = [f"{nme:12s}"]
        for src in ("sapiens", "wilor"):
            o = out[f"{src}_hand"][1:, :, k].copy()
            o[:, 1, 2] *= -1
            row.append(f"{src[:3]} {mm(np.nanmedian(o.reshape(-1, 3), 0)).tolist()} |{mm(np.nanmedian(np.linalg.norm(o, axis=-1))):5.1f}|")
        print("  ".join(row))


if __name__ == "__main__":
    main()
