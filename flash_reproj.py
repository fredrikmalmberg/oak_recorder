"""Reprojection error of the sync-LED flash under the current calibration.

The LED is a static physical point seen by every camera. For each flash we take, per camera, the 3 brightest
frames in the flash window, find the brightest blob in a crop around the (known) LED location, take its
sub-pixel centroid (distorted pixel space), average over the 3 frames, undistort, triangulate (DLT) and
measure the error back in distorted pixel space.

  python flash_reproj.py --take /mnt/recordings/20260925_153809/take_3 --calib output/calibration/20260925_sp_lg_7cam_scaled.json
"""
import argparse, json
import cv2, numpy as np
import calibrate

LED = {"cam0": (1623, 1645), "cam1": (1871, 1599), "cam2": (1201, 1751),
       "cam3": (3189, 1775), "cam4": (3003, 1591), "cam5": (1323, 1529)}   # from detect_flash.py
WINDOWS = {"flash1": (124, 141), "flash2": (157, 173)}
H = 40


def blob(gray):
    """Sub-pixel centroid of the brightest blob. Returns (cx, cy, peak) in crop coords."""
    g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.0)
    y, x = np.unravel_index(np.argmax(g), g.shape)
    peak = float(g[y, x]); lo = float(np.median(g))
    r = 12
    y0, y1, x0, x1 = max(y - r, 0), min(y + r + 1, g.shape[0]), max(x - r, 0), min(x + r + 1, g.shape[1])
    w = np.clip(gray[y0:y1, x0:x1].astype(np.float32) - (lo + 0.5 * (peak - lo)), 0, None)
    if w.sum() == 0:
        return float(x), float(y), peak
    yy, xx = np.mgrid[y0:y1, x0:x1]
    return float((w * xx).sum() / w.sum()), float((w * yy).sum() / w.sum()), peak


def proj(X, c):
    rvec, _ = cv2.Rodrigues(c["R"].astype(np.float64))
    p, _ = cv2.projectPoints(np.asarray(X, np.float64).reshape(1, 3), rvec, c["t"].astype(np.float64),
                             c["K"].astype(np.float64), c["dist"].astype(np.float64))
    return p.reshape(2)


def dlt(obs, Ps):
    A = []
    for (u, v), P in zip(obs, Ps):
        A += [u * P[2] - P[0], v * P[2] - P[1]]
    X = np.linalg.svd(np.asarray(A))[2][-1]
    return X[:3] / X[3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--take", default="/mnt/recordings/20260925_153809/take_3")
    ap.add_argument("--calib", default="output/calibration/20260925_sp_lg_7cam_scaled.json")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--out", default="output/flash_reproj.json")
    a = ap.parse_args()
    cams = calibrate.load_calibration_output(a.calib)
    ids = list(LED)
    rec = {c: {} for c in ids}   # rec[cam][frame] = (cx, cy, peak) in full-image distorted pixels
    for c in ids:
        cap = cv2.VideoCapture(f"{a.take}/video_{c}.mjpeg"); i = 0
        lx, ly = LED[c]
        while i <= max(w[1] for w in WINDOWS.values()):
            ok, f = cap.read()
            if not ok:
                break
            if any(w[0] <= i <= w[1] for w in WINDOWS.values()):
                g = cv2.cvtColor(f[ly - H:ly + H, lx - H:lx + H], cv2.COLOR_BGR2GRAY)
                cx, cy, pk = blob(g)
                rec[c][i] = (lx - H + cx, ly - H + cy, pk)
            i += 1

    P = {c: cams[c]["K"] @ np.hstack([cams[c]["R"], cams[c]["t"].reshape(3, 1)]) for c in ids}
    out = {}
    for name, (f0, f1) in WINDOWS.items():
        pts, info = {}, {}
        for c in ids:
            fr = sorted((f for f in rec[c] if f0 <= f <= f1), key=lambda f: -rec[c][f][2])[:a.topk]
            xy = np.array([rec[c][f][:2] for f in fr])
            pts[c] = xy.mean(0); info[c] = (sorted(fr), xy.std(0), [round(rec[c][f][2]) for f in fr])
        ud = {c: cv2.undistortPoints(pts[c].reshape(1, 1, 2).astype(np.float64), cams[c]["K"], cams[c]["dist"], P=cams[c]["K"]).reshape(2) for c in ids}
        X = dlt([ud[c] for c in ids], [P[c] for c in ids])
        print(f"\n== {name}: LED triangulated from {len(ids)} cameras -> X = {X.round(3)} m")
        print("cam   frames(top3)        peak      centroid (distorted px)    sd(px)   all-cams err   leave-one-out err")
        res = {}
        for c in ids:
            e_all = np.linalg.norm(proj(X, cams[c]) - pts[c])
            oth = [o for o in ids if o != c]
            Xl = dlt([ud[o] for o in oth], [P[o] for o in oth])
            e_loo = np.linalg.norm(proj(Xl, cams[c]) - pts[c])
            fr, sd, pk = info[c]
            print(f"{c}  {str(fr):18s} {str(pk):16s} ({pts[c][0]:7.1f},{pts[c][1]:7.1f})  {np.linalg.norm(sd):5.2f}   {e_all:8.2f}      {e_loo:8.2f}")
            res[c] = dict(frames=fr, pt=pts[c].tolist(), err_all=float(e_all), err_loo=float(e_loo))
        ea = [r["err_all"] for r in res.values()]; el = [r["err_loo"] for r in res.values()]
        print(f"mean/max all-cams err {np.mean(ea):.2f}/{np.max(ea):.2f} px ; leave-one-out mean/max {np.mean(el):.2f}/{np.max(el):.2f} px")
        out[name] = dict(X=X.tolist(), cams=res)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
