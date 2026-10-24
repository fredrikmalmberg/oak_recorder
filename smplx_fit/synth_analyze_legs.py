"""Sapiens leg keypoints (goliath 11-20) vs the SMPL-X targets they are mapped to (joint_mapping.LEG_TARGETS), on the renders of
synth_render / synth_sapiens. Same method as synth_analyze (DLT over all views, worst view dropped while reprojection > 15 px),
offset = triangulated - SMPL-X target, in the person frame (x right->left shoulder, y pelvis->neck, z = x cross y) and in world.
Also prints the 2D error of every keypoint against the projected target, which checks that the id -> target assignment is right.

  PYTHONPATH=. python -m smplx_fit.synth_analyze_legs --dir <synth dir> [--variant textured]
"""
import argparse
import numpy as np

from smplx_fit import joint_mapping as jm
from smplx_fit.synth_analyze import gt_point, person_frame, project, triangulate

SAP_IDS = list(range(11, 21))
NAMES = ["l_knee", "r_knee", "l_ankle", "r_ankle", "l_big_toe", "l_small_toe", "l_heel", "r_big_toe", "r_small_toe", "r_heel"]
TARGETS = [jm.LEG_TARGETS[k] for k in jm.LEG_FILE_ORDER]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--variant", default="textured")
    ap.add_argument("--min-score", type=float, default=0.3)
    ap.add_argument("--max-reproj-px", type=float, default=15.0)
    ap.add_argument("--by-direction", action="store_true", help="T-pose only: one row per viewing direction (front / sides / back)")
    a = ap.parse_args()
    if a.by_direction:
        return by_direction(a.dir, a.variant, a.min_score, a.max_reproj_px)
    gt = np.load(f"{a.dir}/gt.npz")
    sap = np.load(f"{a.dir}/sapiens_{a.variant}.npz")
    P, V = len(gt["pose_names"]), len(gt["view_names"])
    K, Rs, ts = gt["K"], gt["R"], gt["t"]
    off_p = np.full((P, 10, 3), np.nan); off_w = np.full((P, 10, 3), np.nan); px = np.full((P, 10, V), np.nan)
    for p in range(P):
        Ps = {v: K @ np.hstack([Rs[p, v], ts[p, v][:, None]]) for v in range(V)}
        vt, jt = gt["verts"][p], gt["joints"][p]
        pf = person_frame(jt)
        for k, tg in enumerate(TARGETS):
            X0 = gt_point(vt, jt, tg)
            pix = sap["kp"][p][:, SAP_IDS[k]]
            ok = sap["sc"][p][:, SAP_IDS[k]] > a.min_score
            X, n = triangulate(pix, ok, Ps, a.max_reproj_px)
            off_w[p, k] = X - X0
            off_p[p, k] = pf.T @ (X - X0)
            for v in range(V):
                g, _ = project(K, Rs[p, v], ts[p, v], X0[None])
                px[p, k, v] = np.linalg.norm(pix[v] - g[0]) if ok[v] else np.nan
    np.savez(f"{a.dir}/analysis_legs_{a.variant}.npz", names=NAMES, off_person=off_p, off_world=off_w, px=px)
    fit = slice(1, None)   # skip the T-pose
    print(f"{'point':12s} {'x lateral':>10s} {'y up':>8s} {'z fwd':>8s}   (mm, median over {P - 1} fitted poses; [std])   2D err px  |  T-pose x/y/z")
    for k, n in enumerate(NAMES):
        m = np.nanmedian(off_p[fit, k], 0) * 1000; s = np.nanstd(off_p[fit, k], 0) * 1000
        print(f"{n:12s} {m[0]:+8.1f} {m[1]:+8.1f} {m[2]:+8.1f}   [{s[0]:.1f} {s[1]:.1f} {s[2]:.1f}]   {np.nanmedian(px[fit, k]):6.1f}   |  "
              + " ".join(f"{v:+.1f}" for v in off_p[0, k] * 1000))
    print("\nlateral sign: x points from the person's right shoulder to the left. Left-leg and right-leg x are mirrored.")



def by_direction(d, variant, min_score=0.3, max_reproj_px=15.0, half_width=45.0):
    """T-pose only (pose 0): offsets using only the views within +-half_width deg of each direction (azimuth 0 = camera in front)."""
    gt = np.load(f"{d}/gt.npz"); sap = np.load(f"{d}/sapiens_{variant}.npz")
    V = len(gt["view_names"]); K, Rs, ts = gt["K"], gt["R"], gt["t"]
    az = np.array([float(n.split("_")[0][1:]) for n in gt["view_names"]])
    vt, jt = gt["verts"][0], gt["joints"][0]
    pf = person_frame(jt)
    Ps = {v: K @ np.hstack([Rs[0, v], ts[0, v][:, None]]) for v in range(V)}
    dirs = {"front": 0, "left side": 90, "back": 180, "right side": -90}
    print(f"{'point':12s} " + " | ".join(f"{n:^22s}" for n in dirs) + "    (x lateral / y up / z fwd, mm; views used per direction: "
          + ", ".join(str(int((np.abs((az - c + 180) % 360 - 180) <= half_width).sum())) for c in dirs.values()) + ")")
    res = np.full((len(dirs), 10, 3), np.nan)
    for k, tg in enumerate(TARGETS):
        X0 = gt_point(vt, jt, tg)
        row = []
        for di, c in enumerate(dirs.values()):
            sel = np.abs((az - c + 180) % 360 - 180) <= half_width
            pix = sap["kp"][0][:, SAP_IDS[k]]
            ok = (sap["sc"][0][:, SAP_IDS[k]] > min_score) & sel
            X, n = triangulate(pix, ok, Ps, max_reproj_px)
            res[di, k] = pf.T @ (X - X0) * 1000
            row.append("%+6.1f %+6.1f %+6.1f" % tuple(res[di, k]) + f" ({n})")
        print(f"{NAMES[k]:12s} " + " | ".join(row))
    np.savez(f"{d}/analysis_legs_bydir_{variant}.npz", names=NAMES, directions=list(dirs), off_person_mm=res)


if __name__ == "__main__":
    main()
