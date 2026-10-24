"""Triangulate the WiLoR wrist (MANO joint 0) over all cameras, per frame and hand.

Uses only the 2D wrist projection of each WiLoR detection (kp2d_calib joint 0, raw/distorted pixels -> undistorted with
the calibration), not WiLoR's own depth. Per frame: DLT over all valid cameras, then the worst camera is dropped while its
reprojection error exceeds --max-reproj-px and more than 2 cameras remain; frames with < 2 cameras (or a final error
above the threshold) get no wrist.

  ~/miniconda3/envs/oak_env/bin/python -m hand_pose.wilor_wrist --take <take> --calib <calib.json>
Output <take>/aligned/wilor/wilor_wrist_3d.npz: frames[T], sides[left,right], wrist[T,2,3] (NaN if none), n_cams[T,2],
reproj_px[T,2] (mean over kept cameras), cams_used[T,2,C] (bool).
"""
import argparse, os, sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import calibrate  # noqa: E402
from hand_pose import hand_multiview as hmv  # noqa: E402
from pose2d import triangulation as T  # noqa: E402


def triangulate_frame(pix, P, max_reproj):
    """pix: {cam: (x, y)} undistorted pixels. Returns (X, kept cams, mean error) or None."""
    pix = dict(pix)
    while len(pix) >= 2:
        X = hmv.triangulate_dlt(pix, P)
        if X is None:
            return None
        err = {c: hmv._reproj_error_px(X, P[c], p) for c, p in pix.items()}
        worst = max(err, key=err.get)
        if err[worst] <= max_reproj:
            return X, list(pix), float(np.mean(list(err.values())))
        if len(pix) == 2:
            return None
        del pix[worst]
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--take", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--wilor-dir", default=None, help="default <take>/aligned/wilor")
    ap.add_argument("--max-reproj-px", type=float, default=25.0)
    a = ap.parse_args()

    wdir = a.wilor_dir or f"{a.take}/aligned/wilor"
    d = np.load(f"{wdir}/wilor_hands.npz")
    frames, cams, valid, kp2d = d["frames"], [str(c) for c in d["cams"]], d["valid"], d["kp2d_calib"][..., 0, :]
    calib = calibrate.load_calibration_output(a.calib)
    P = T.build_projection_matrices(calib, cams)
    und = np.full_like(kp2d, np.nan, dtype=np.float64)
    for ci, c in enumerate(cams):
        for s in range(2):
            ok = valid[:, ci, s]
            if ok.any():
                und[ok, ci, s] = cv2.undistortPoints(kp2d[ok, ci, s].reshape(-1, 1, 2).astype(np.float64),
                                                     calib[c]["K"], calib[c]["dist"], P=calib[c]["K"]).reshape(-1, 2)

    Tn, C = len(frames), len(cams)
    wrist = np.full((Tn, 2, 3), np.nan, np.float32)
    n_cams = np.zeros((Tn, 2), np.int8)
    reproj = np.full((Tn, 2), np.nan, np.float32)
    used = np.zeros((Tn, 2, C), bool)
    for t in range(Tn):
        for s in range(2):
            pix = {c: tuple(und[t, ci, s]) for ci, c in enumerate(cams) if valid[t, ci, s] and np.isfinite(und[t, ci, s]).all()}
            r = triangulate_frame(pix, P, a.max_reproj_px)
            if r is None:
                continue
            X, kept, e = r
            wrist[t, s], n_cams[t, s], reproj[t, s] = X, len(kept), e
            for c in kept:
                used[t, s, cams.index(c)] = True
    out = f"{wdir}/wilor_wrist_3d.npz"
    np.savez_compressed(out, frames=frames, cams=np.array(cams), sides=np.array(["left", "right"]), wrist=wrist,
                        n_cams=n_cams, reproj_px=reproj, cams_used=used)
    lo, hi = 300, 650
    for s, nm in enumerate(("left", "right")):
        ok = np.isfinite(wrist[:, s, 0])
        r = ok & (frames >= lo) & (frames <= hi)
        print(f"{nm}: wrist in {ok.sum()}/{Tn} frames ({r.sum()}/{hi - lo + 1} in {lo}-{hi}); cams used median "
              f"{np.median(n_cams[ok, s]):.0f}, reproj median {np.nanmedian(reproj[ok, s]):.1f}px")
    print("saved", out)


if __name__ == "__main__":
    main()
