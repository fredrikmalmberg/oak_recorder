"""Numbers comparing MANO fits to the Sapiens-triangulated hands and to the WiLoR-triangulated hands (frames of interest only).

Four 3D sources per hand: Sapiens triangulation (Sap tri), MANO fit to it (MANO-Sap), WiLoR triangulation (Wil tri),
MANO fit to it (MANO-Wil), all from hand_pose.fit_mano npz files. Also the raw per-camera WiLoR 3D (joints_cam moved to world).
  1. fit residual per joint group (MANO fit vs its own target)
  2. reprojection error of every source into every camera vs the Sapiens 2D and the WiLoR 2D keypoints (undistorted px)
  3. disagreement between the two pipelines (3D mm per joint group, mesh distance, hand size)
  4. smoothness (second difference of joints, mm/frame^2) over consecutive frames
  5. per-camera WiLoR 3D vs the fits

  python -m hand_pose.compare_mano_fits --take T --calib C --sap output/manoSap/mano_fit.npz --wil output/manoWil/mano_fit.npz
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import calibrate  # noqa: E402
from pose2d import triangulation as T  # noqa: E402

GROUPS = {"wrist": [0], "mcp": [1, 5, 9, 13, 17], "pip": [2, 6, 10, 14, 18], "dip": [3, 7, 11, 15, 19], "tip": [4, 8, 12, 16, 20]}
SIDES = ("left", "right")


def stat(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return {"n": int(len(x)), "median": float(np.median(x)), "mean": float(x.mean()), "p95": float(np.percentile(x, 95))} if len(x) else None


def fmt(s):
    return "      -" if s is None else f"{s['median']:6.1f} / {s['mean']:6.1f} / {s['p95']:6.1f}"


def load_2d(path, cams, n_frames, w, h):
    """landmarks_*.json -> arr[T,C,21,2] undistorted px (NaN missing)."""
    out = np.full((n_frames, len(cams), 21, 2), np.nan)
    d = json.load(open(path))
    for ci, c in enumerate(cams):
        for fk, lm in d.get(c, {}).items():
            f = int(fk[:6])
            if f < n_frames:
                for k, v in lm.items():
                    if int(k) < 21:
                        out[f, ci, int(k)] = (v[0] * w, v[1] * h)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--take", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--sap", required=True)
    ap.add_argument("--wil", required=True)
    ap.add_argument("--sap-2d", default="pose2d_sapiens_smoothbox")
    ap.add_argument("--wil-2d", default="pose2d_wilor")
    ap.add_argument("--frames", type=int, nargs=2, default=[300, 650])
    ap.add_argument("--out", default=None, help="json output")
    a = ap.parse_args()

    S, W = np.load(a.sap), np.load(a.wil)
    wh = np.load(f"{a.take}/aligned/wilor/wilor_hands.npz")
    cams = [str(c) for c in wh["cams"]]
    calib = calibrate.load_calibration_output(a.calib)
    P = T.build_projection_matrices(calib, cams)
    n = len(S["frames"])
    sel = np.zeros(n, bool)
    sel[a.frames[0]:a.frames[1] + 1] = True
    w, h = calib[cams[0]]["width"], calib[cams[0]]["height"]

    src = {"Sap tri": S["target"], "MANO-Sap": S["joints"], "Wil tri": W["target"], "MANO-Wil": W["joints"]}
    res = {}
    print(f"frames {a.frames[0]}-{a.frames[1]}; all numbers: median / mean / p95\n")

    print("1. MANO fit residual vs its own 3D target (mm)")
    res["fit_residual_mm"] = {}
    for nm, fit, tgt in (("MANO-Sap vs Sap tri", S["joints"], S["target"]), ("MANO-Wil vs Wil tri", W["joints"], W["target"])):
        d = np.linalg.norm(fit - tgt, axis=-1) * 1000
        for s, side in enumerate(SIDES):
            row = {g: stat(d[sel, s][:, ids]) for g, ids in GROUPS.items()}
            row["all"] = stat(d[sel, s])
            res["fit_residual_mm"][f"{nm} {side}"] = row
            print(f"  {nm:20s} {side:5s} " + "  ".join(f"{g}: {fmt(v)}" for g, v in row.items()))

    print("\n2. Reprojection into all cameras (undistorted px), vs 2D keypoints of")
    det = {}
    for lab, d2 in (("Sapiens 2D", a.sap_2d), ("WiLoR 2D", a.wil_2d)):
        det[lab] = [load_2d(f"{a.take}/aligned/{d2}/landmarks_{side}.json", cams, n, w, h) for side in SIDES]
    res["reproj_px"] = {}
    for lab in det:
        print(f"  -- {lab}")
        for nm, J in src.items():
            row = {}
            for s, side in enumerate(SIDES):
                errs = []
                percam = {c: [] for c in cams}
                for ci, c in enumerate(cams):
                    X = J[sel, s]                                    # [F,21,3]
                    q = np.einsum("ij,fkj->fki", P[c], np.concatenate([X, np.ones((*X.shape[:2], 1))], -1))
                    uv = q[..., :2] / q[..., 2:3]
                    e = np.linalg.norm(uv - det[lab][s][sel][:, ci], axis=-1)
                    errs.append(e.ravel()); percam[c] = e.ravel()
                row[side] = stat(np.concatenate(errs))
                row[side + "_per_cam_median"] = {c: float(np.nanmedian(v)) if np.isfinite(v).any() else None for c, v in percam.items()}
                print(f"    {nm:9s} {side:5s} {fmt(row[side])}   per-cam median: " +
                      " ".join(f"{c}:{row[side + '_per_cam_median'][c]:.1f}" for c in cams if row[side + "_per_cam_median"][c] is not None))
            res["reproj_px"].setdefault(lab, {})[nm] = row

    print("\n3. Pipeline disagreement (3D mm, both sources present, no alignment)")
    res["disagreement_mm"] = {}
    for nm, A, B in (("Sap tri vs Wil tri", S["target"], W["target"]), ("MANO-Sap vs MANO-Wil", S["joints"], W["joints"])):
        d = np.linalg.norm(A - B, axis=-1) * 1000
        for s, side in enumerate(SIDES):
            row = {g: stat(d[sel, s][:, ids]) for g, ids in GROUPS.items()}
            res["disagreement_mm"][f"{nm} {side}"] = row
            print(f"  {nm:22s} {side:5s} " + "  ".join(f"{g}: {fmt(v)}" for g, v in row.items()))
    dv = np.linalg.norm(S["verts"] - W["verts"], axis=-1).mean(-1)
    res["mesh_mean_vertex_dist_mm"] = {side: stat(dv[sel, s] * 1000) for s, side in enumerate(SIDES)}
    print("  mean vertex distance MANO-Sap mesh vs MANO-Wil mesh (mm): " + "  ".join(f"{side}: {fmt(res['mesh_mean_vertex_dist_mm'][side])}" for side in SIDES))
    res["hand_size_mm"] = {}
    for s, side in enumerate(SIDES):
        row = {}
        for nm, J in src.items():
            L = np.linalg.norm(J[sel, s, 12] - J[sel, s, 9], axis=-1) + np.linalg.norm(J[sel, s, 9] - J[sel, s, 0], axis=-1)
            row[nm] = stat(L * 1000)
        res["hand_size_mm"][side] = row
        print(f"  wrist->middle MCP->middle tip chain length (mm) {side:5s}: " + "  ".join(f"{k}: {v['median']:.0f}" for k, v in row.items()))

    print("\n4. Smoothness: |second difference| of joints over consecutive-frame triples (mm/frame^2)")
    res["accel_mm"] = {}
    for nm, J in src.items():
        row = {}
        for s, side in enumerate(SIDES):
            ok = sel[1:-1] & sel[:-2] & sel[2:]
            acc = np.linalg.norm((J[2:, s] - 2 * J[1:-1, s] + J[:-2, s])[ok], axis=-1) * 1000
            row[side] = stat(acc)
        res["accel_mm"][nm] = row
        print(f"  {nm:9s} " + "  ".join(f"{side}: {fmt(row[side])}" for side in SIDES))

    print("\n5. Raw per-camera WiLoR 3D (joints_cam -> world) vs 3D sources (mm, median over frames+joints)")
    res["wilor_percam_mm"] = {}
    for nm, J in src.items():
        row = {}
        for s, side in enumerate(SIDES):
            per = {}
            for ci, c in enumerate(cams):
                R, t = calib[c]["R"], np.asarray(calib[c]["t"]).reshape(3)
                m = sel & wh["valid"][:, ci, s] & np.isfinite(J[:, s]).all((-1, -2))
                Xw = (wh["joints_cam"][m, ci, s].astype(np.float64) - t) @ R          # R^T (Xc - t)
                per[c] = float(np.median(np.linalg.norm(Xw - J[m, s], axis=-1))) * 1000 if m.any() else None
            row[side] = per
        res["wilor_percam_mm"][nm] = row
        for side in SIDES:
            print(f"  vs {nm:9s} {side:5s} " + " ".join(f"{c}:{v:.0f}" for c, v in row[side].items() if v is not None))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
        print("\nwrote", a.out)


if __name__ == "__main__":
    main()
