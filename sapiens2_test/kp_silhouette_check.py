"""Keypoint-vs-silhouette consistency check.

For a keypoint k (default shoulders 5,6 and hips 9,10) in camera c, compare its frame-to-frame motion
d_kp = kp(t) - kp(t-1) with the motion of the RVM silhouette in a window around it, estimated by
Lucas-Kanade on the soft mask (frames and masks are both in distorted pixel space, so no remapping):

    M_t(x) ~ M_{t-1}(x - v)  =>  v = -J^-1 b,   J = sum w grad(M) grad(M)^T,  b = sum w grad(M) (M_t - M_{t-1})

Aperture problem: a straight silhouette edge only constrains motion along its normal, so v is only
estimated along eigen-directions of J whose eigenvalue exceeds --lam-min (J is in low-res px units;
lam ~ 0.14 * edge length in px for a sigma-2 blurred edge). The reported discrepancy
    disc = d_kp - v   (projected on the observable directions only)
is "the keypoint moved but the silhouette around it didn't (or moved differently)". Direction with no
silhouette constraint are reported as unobservable (rank counts observable directions: 0, 1, 2).

  python -m sapiens2_test.kp_silhouette_check --take T --npz output/sapiens2/allframes_1b_smoothbox.npz \
      --out output/sapiens2/kp_sil_check.npz
"""
import argparse, os
from concurrent.futures import ThreadPoolExecutor
import cv2, numpy as np

KPS = {"Lsh": 5, "Rsh": 6, "Lhip": 9, "Rhip": 10}  # Sapiens ids


def load_mask(path, s, sigma):
    m = cv2.imread(path, 0)
    if m is None:
        return None
    m = cv2.resize(m, None, fx=s, fy=s, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    return cv2.GaussianBlur(m, (0, 0), sigma)


ARMS = {"L": (5, 7, 62, range(42, 62)), "R": (6, 8, 41, range(21, 41))}  # shoulder, elbow, wrist, hand kps (Sapiens ids)


def arm_mask(kp_f, sc_f, s, shape, thr=0.3, r_arm=85.0, r_hand=60.0, start=0.5):
    """Low-res uint8 mask of forearm/upper-arm (from `start` of the way shoulder->elbow) and hand pixels, from Sapiens keypoints
    (distorted full-res px, r_* in full-res px). Arms hide/merge with the torso silhouette, so they are excluded from the window."""
    E = np.zeros(shape, np.uint8)
    ok = lambda j: sc_f[j] > thr and np.isfinite(kp_f[j]).all()
    pt = lambda j: tuple(int(round(v * s)) for v in kp_f[j])
    for sh, el, wr, hand in ARMS.values():
        if ok(sh) and ok(el):
            a = kp_f[sh] + start * (kp_f[el] - kp_f[sh]); a = tuple(int(round(v * s)) for v in a)
            cv2.line(E, a, pt(el), 1, int(round(2 * r_arm * s)))
            cv2.circle(E, pt(el), int(round(r_arm * s)), 1, -1)
            if ok(wr):
                cv2.line(E, pt(el), pt(wr), 1, int(round(2 * r_arm * s)))
        for j in [wr, *hand]:
            if ok(j):
                cv2.circle(E, pt(j), int(round(r_hand * s)), 1, -1)
    return E


def window_flow(M0, M1, centre, R, lam_min, iters=4, excl=None):
    """LK translation of M0->M1 in a (2R+1)^2 window around centre (low-res px).
    Returns (v[2] with NaN on unobservable axes, evecs[2,2] columns, evals[2], rank)."""
    h, w = M0.shape
    cx, cy = int(round(centre[0])), int(round(centre[1]))
    x0, x1, y0, y1 = max(cx - R, 1), min(cx + R + 1, w - 1), max(cy - R, 1), min(cy + R + 1, h - 1)
    if x1 - x0 < R or y1 - y0 < R:
        return None
    yy, xx = np.mgrid[y0:y1, x0:x1] - np.array([cy, cx])[:, None, None]
    wgt = np.exp(-(xx ** 2 + yy ** 2) / (2 * (R / 2.0) ** 2)).astype(np.float32)
    if excl is not None:
        wgt = wgt * (1.0 - excl[y0:y1, x0:x1].astype(np.float32))
    P0 = M0[y0:y1, x0:x1]
    gy, gx = np.gradient(P0)
    gx, gy = gx * wgt, gy * wgt
    J = np.array([[(gx * gx).sum(), (gx * gy).sum()], [(gx * gy).sum(), (gy * gy).sum()]], np.float64)
    ev, U = np.linalg.eigh(J)  # ascending
    for i in range(2):  # canonical sign so signed components can be summed over frames
        k = int(np.argmax(np.abs(U[:, i])))
        if U[k, i] < 0:
            U[:, i] = -U[:, i]
    obs = ev > lam_min
    v = np.zeros(2)
    for _ in range(iters):  # iterative LK: warp M1 back by current v
        A = np.float32([[1, 0, v[0]], [0, 1, v[1]]])
        W1 = cv2.warpAffine(M1, A, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
        It = (W1[y0:y1, x0:x1] - P0) * wgt
        b = np.array([(gx * It).sum(), (gy * It).sum()], np.float64)
        step = np.zeros(2)
        for i in range(2):
            if obs[i]:
                step += U[:, i] * (-(U[:, i] @ b) / ev[i])
        # step is the correction to v
        v = v + step
        if np.abs(step).max() < 0.02:
            break
    return v, U, ev, obs


def run_cam(take, cam, frames, kp, thr_sc, sc, s, sigma, R, Rhip, lam_min, med_win, ids, exclude_arms=True):
    F = len(frames)
    masks = [None] * F
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(lambda f: load_mask(f"{take}/aligned/masks_rvm/{cam}/{int(f):06d}.jpg", s, sigma), frames))
    masks = res
    E = [arm_mask(kp[t], sc[t], s, masks[t].shape) if (exclude_arms and masks[t] is not None) else None for t in range(F)]
    out = {n: np.full((F, 14), np.nan, np.float32) for n in ids}
    # columns: dkp_x dkp_y vsil_x vsil_y disc_x disc_y rank e_max ev0 ev1 centre_x centre_y e0 e1
    # e0/e1: signed disagreement (full-res px) along the weak/strong eigen-direction (ev0<=ev1), NaN if that direction is below lam_min
    for name, j in ids.items():
        P = kp[:, j, :].astype(np.float64)
        ok = np.isfinite(P).all(axis=1) & (sc[:, j] > thr_sc)
        P = np.where(ok[:, None], P, np.nan)
        for t in range(1, F):
            if masks[t] is None or masks[t - 1] is None or not (ok[t] and ok[t - 1]):
                continue
            lo, hi = max(0, t - med_win), min(F, t + med_win + 1)
            seg = P[lo:hi]
            seg = seg[np.isfinite(seg).all(axis=1)]
            centre = np.median(seg, axis=0) * s
            ex = None if E[t] is None or E[t - 1] is None else np.maximum(E[t], E[t - 1])
            r = window_flow(masks[t - 1], masks[t], centre, Rhip if 'hip' in name else R, lam_min, excl=ex)
            if r is None:
                continue
            v, U, ev, obs = r
            dkp = (P[t] - P[t - 1])
            v_full = v / s
            rank = int(obs.sum())
            disc = np.zeros(2)
            emax = 0.0
            ee = [np.nan, np.nan]
            for i in range(2):
                if obs[i]:
                    e = U[:, i] @ (dkp - v_full)
                    disc += e * U[:, i]
                    emax = max(emax, abs(e))
                    ee[i] = e
            out[name][t] = [*dkp, *v_full, *disc, rank, emax, ev[0], ev[1], *(centre / s), *ee]
    return out


def windowed_z(a, K=4, before=1):
    """Centred sustained-disagreement score per frame from one [F,14] result array: for steps t-before .. t-before+K-1,
    z = max over eigen-directions of |sum of signed disagreement (px)| * sqrt(mean eigenvalue); NaN unless the direction is observable in all K steps."""
    n = len(a); z = np.full(n, np.nan)
    for t in range(before, n - (K - before - 1)):
        best = np.nan
        for col, evc in ((12, 8), (13, 9)):
            e = a[t - before:t - before + K, col]; ev = a[t - before:t - before + K, evc]
            if np.isfinite(e).all():
                v = abs(e.sum()) * np.sqrt(ev.mean()); best = v if not np.isfinite(best) else max(best, v)
        z[t] = best
    return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--take", required=True)
    ap.add_argument("--npz", default="output/sapiens2/allframes_1b_smoothbox.npz")
    ap.add_argument("--out", default="output/sapiens2/kp_sil_check.npz")
    ap.add_argument("--scale", type=float, default=0.25, help="mask downscale")
    ap.add_argument("--sigma", type=float, default=2.0, help="mask blur at low-res px")
    ap.add_argument("--radius", type=int, default=48, help="window half-size at low-res px (x1/scale = full-res)")
    ap.add_argument("--radius-hip", type=int, default=96, help="window half-size for hips (low-res px)")
    ap.add_argument("--lam-min", type=float, default=3.0)
    ap.add_argument("--med-win", type=int, default=5)
    ap.add_argument("--thr-sc", type=float, default=0.3)
    ap.add_argument("--keep-arms", action="store_true", help="do not exclude arm/hand pixels from the windows")
    ap.add_argument("--cams", nargs="+", default=None)
    a = ap.parse_args()
    d = np.load(a.npz)
    frames, cams, kp, sc = d["frames"], [str(c) for c in d["cams"]], d["kp"], d["sc"]
    use = a.cams or cams
    res = {}
    for c in use:
        ci = cams.index(c)
        o = run_cam(a.take, c, frames, kp[:, ci], a.thr_sc, sc[:, ci], a.scale, a.sigma, a.radius, a.radius_hip, a.lam_min, a.med_win, KPS, exclude_arms=not a.keep_arms)
        for n, arr in o.items():
            res[f"{c}/{n}"] = arr
        print(c, {n: int(np.isfinite(arr[:, 7]).sum()) for n, arr in o.items()}, flush=True)
    np.savez_compressed(a.out, frames=frames, cols=np.array("dkp_x dkp_y vsil_x vsil_y disc_x disc_y rank e_max ev0 ev1 cx cy e0 e1".split()), **res)
    print("saved", a.out)


if __name__ == "__main__":
    main()
