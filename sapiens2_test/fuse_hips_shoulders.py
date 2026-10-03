"""Fuse two triangulations (e.g. MediaPipe + Sapiens, same world frame) for shoulders/hips with cross-check outlier rejection.

Per target landmark (body ids 5,6,11,12):
  1. Remove the constant detector offset: b = median(A - B) over frames where both exist (hip/shoulder definitions differ).
  2. Frames where |A - (B+b)| < --agree-mm: measurement = their mean (std --meas-agree).
  3. Disagreeing/single-source frames: pick the source closest to a temporal reference (median filter, +---ref-win frames,
     of the agreeing frames' trajectory); accepted only if within --gate-mm of it (std --meas-single), else dropped.
  4. Kalman/RTS (--q, defaults to the 'medium' level) over the accepted measurements.
Output dir gets reconstruction_{left,right,body}.json: body raw = fused measurements, body smoothed = Kalman for targets;
non-target body joints and hands are copied from --base (Sapiens). Both 'raw' and 'smoothed' of non-targets are the base's.

  python -m sapiens2_test.fuse_hips_shoulders --a <MP dir> --b <Sapiens dir> --out output/fused/aligned/pose2d_fused
"""
import argparse, json, os, shutil, sys
import numpy as np
from scipy.ndimage import median_filter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hand_pose import hand_multiview as hmv  # noqa: E402

TARGET = ("5", "6", "11", "12")
T = 768


def traj(rec, lm):
    a = np.full((T, 3), np.nan)
    for k, pts in rec.items():
        p = pts.get(lm)
        if p:
            a[int(os.path.splitext(k)[0])] = p
    return a


def fuse(A, B, agree_mm, gate_mm, ref_win, q, m_agree, m_single):
    ok = ~np.isnan(A[:, 0]) & ~np.isnan(B[:, 0])
    b = np.median(A[ok] - B[ok], axis=0)
    B = B + b
    agree = ok & (np.linalg.norm(A - B, axis=1) * 1000 < agree_mm)
    M = np.where(agree[:, None], (A + B) / 2, np.nan)
    # temporal reference from agreeing frames (interpolate gaps, then median filter)
    idx = np.where(agree)[0]
    ref = np.stack([np.interp(np.arange(T), idx, M[idx, d]) for d in range(3)], axis=1)
    ref = median_filter(ref, size=(2 * ref_win + 1, 1), mode="nearest")
    std = np.full(T, m_agree)
    status = np.where(agree, "agree", "").astype(object)
    for t in np.where(~agree)[0]:
        cands = [(np.linalg.norm(X[t] - ref[t]) * 1000, X[t]) for X in (A, B) if not np.isnan(X[t, 0])]
        if not cands:
            status[t] = "none"; continue
        d, X = min(cands, key=lambda c: c[0])
        if d < gate_mm:
            M[t], std[t], status[t] = X, m_single, "single"
        else:
            status[t] = "dropped"
    valid = ~np.isnan(M[:, 0])
    lo, hi = np.argmax(valid), T - np.argmax(valid[::-1])
    S = np.full((T, 3), np.nan)
    sm = hmv.kalman_rts_smooth(np.nan_to_num(M), valid, std, process_std=q)
    S[lo:hi] = sm[lo:hi]
    return M, S, status, b




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="dir with reconstruction_body.json (e.g. MediaPipe, ext8000)")
    ap.add_argument("--b", required=True, help="Sapiens dir (also --base source for non-target joints and hands)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--agree-mm", type=float, default=50.0)
    ap.add_argument("--gate-mm", type=float, default=60.0)
    ap.add_argument("--ref-win", type=int, default=15)
    ap.add_argument("--q", type=float, default=0.003)
    ap.add_argument("--meas-agree", type=float, default=0.015)
    ap.add_argument("--meas-single", type=float, default=0.03)
    a = ap.parse_args()

    ra = json.load(open(os.path.join(a.a, "reconstruction_body.json")))
    rb = json.load(open(os.path.join(a.b, "reconstruction_body.json")))
    out = {"raw": {k: dict(v) for k, v in rb["raw"].items()}, "smoothed": {k: dict(v) for k, v in rb["smoothed"].items()}}
    diag = {}
    for lm in TARGET:
        A, B = traj(ra["raw"], lm), traj(rb["raw"], lm)
        M, S, status, b = fuse(A, B, a.agree_mm, a.gate_mm, a.ref_win, a.q, a.meas_agree, a.meas_single)
        for t in range(T):
            k = f"{t:06d}.jpg"
            for kind, arr in (("raw", M), ("smoothed", S)):
                d = out[kind].setdefault(k, {})
                if np.isnan(arr[t, 0]):
                    d.pop(lm, None)
                else:
                    d[lm] = arr[t].tolist()
        s, n = status, len(status)
        diag[lm] = {"offset_mm": (b * 1000).round(1).tolist(), **{k: int((s == k).sum()) for k in ("agree", "single", "dropped", "none")}}
        print(lm, diag[lm])
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(os.path.join(a.out, "reconstruction_body.json"), "w"))
    for part in ("left", "right"):
        shutil.copy(os.path.join(a.b, f"reconstruction_{part}.json"), a.out)
    json.dump(diag, open(os.path.join(a.out, "fusion_diagnostics.json"), "w"), indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
