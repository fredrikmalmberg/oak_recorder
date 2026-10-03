"""Temporally stable person boxes for Sapiens2 from the RVM masks.

Per-frame mask extents jump when the hands go up, and Sapiens (top-down, crops to the box) then
shifts other keypoints such as the hips. Here each edge is taken as a sliding-window percentile
(outer edges: 10th/90th percentile over +-win frames), Gaussian-smoothed, padded, and forced to
the model aspect (3:4). The pipeline multiplies the box by a further 1.25, so the crop the
model actually sees is `effective_crop(box)`.

  python -m sapiens2_test.smooth_bbox --take TAKE --out output/sapiens2/boxes_smooth.npz
Output npz: frames[F], cams[C], boxes[F,C,4] (x0,y0,x1,y1; NaN if no mask), raw[F,C,4]
"""
import argparse
import cv2, numpy as np
from scipy.ndimage import gaussian_filter1d, percentile_filter

ASPECT = 768 / 1024  # w / h of the model input
CROP_PAD = 1.25      # PoseGetBBoxCenterScale default


def raw_box(mask_path):
    m = cv2.imread(mask_path, 0)
    if m is None:
        return np.full(4, np.nan)
    ys, xs = np.nonzero(m > 128)
    if len(xs) == 0:
        return np.full(4, np.nan)
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], np.float64)


def smooth_cam(raw, win=30, sigma=10.0, pad=0.02, bottom=None):
    """raw [F,4] -> smoothed [F,4] (aspect-fixed). NaN rows are interpolated.

    bottom: if given (frame height), the lower edge is pinned there and not padded: the RVM masks are
    cut off above the knees by their keypoint-guided ROI, while the legs run to the frame edge."""
    raw = raw.copy()
    idx = np.arange(len(raw))
    ok = np.isfinite(raw).all(axis=1)
    if not ok.any():
        return np.full_like(raw, np.nan)
    for k in range(4):
        raw[:, k] = np.interp(idx, idx[ok], raw[ok, k])
    size = 2 * win + 1
    e = np.empty_like(raw)
    for k, pct in zip(range(4), (10, 10, 90, 90)):
        e[:, k] = percentile_filter(raw[:, k], pct, size=size, mode="nearest")
        e[:, k] = gaussian_filter1d(e[:, k], sigma, mode="nearest")
    bw, bh = e[:, 2] - e[:, 0], e[:, 3] - e[:, 1]
    x0, x1, y0, y1 = e[:, 0] - pad * bw, e[:, 2] + pad * bw, e[:, 1] - pad * bh, e[:, 3] + pad * bh
    if bottom is not None:
        y1 = np.full_like(y1, float(bottom))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w, h = x1 - x0, y1 - y0
    w, h = np.maximum(w, h * ASPECT), np.maximum(h, w / ASPECT)
    out = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    out[~np.isfinite(raw).all(axis=1) & ~ok] = np.nan
    return out


def effective_crop(box):
    """The region the model actually sees: box scaled by CROP_PAD about its centre, then aspect-fixed."""
    box = np.asarray(box, np.float64)
    c = (box[..., :2] + box[..., 2:]) / 2
    s = (box[..., 2:] - box[..., :2]) * CROP_PAD
    w, h = s[..., 0], s[..., 1]
    w, h = np.where(w > h * ASPECT, w, h * ASPECT), np.where(w > h * ASPECT, w / ASPECT, h)
    s = np.stack([w, h], -1)
    return np.concatenate([c - s / 2, c + s / 2], -1)


def padded_raw(raw, pad=0.05):
    """run_pose.mask_bbox: raw mask extents with 5% padding (what the current run fed Sapiens)."""
    raw = np.asarray(raw, np.float64)
    w, h = raw[..., 2] - raw[..., 0], raw[..., 3] - raw[..., 1]
    return raw + np.stack([-pad * w, -pad * h, pad * w, pad * h], -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--take", required=True)
    ap.add_argument("--out", default="output/sapiens2/boxes_smooth.npz")
    ap.add_argument("--n-frames", type=int, default=768)
    ap.add_argument("--cams", nargs="+", default=["cam0", "cam1", "cam2", "cam3", "cam4", "cam5"])
    ap.add_argument("--win", type=int, default=30)
    ap.add_argument("--sigma", type=float, default=10.0)
    ap.add_argument("--pad", type=float, default=0.02)
    ap.add_argument("--frame-h", type=float, default=2160, help="pin box bottom here (0 = off)")
    a = ap.parse_args()
    F = np.arange(a.n_frames)
    raw = np.stack([np.stack([raw_box(f"{a.take}/aligned/masks_rvm/{c}/{f:06d}.jpg") for f in F]) for c in a.cams], 1)
    sm = np.stack([smooth_cam(raw[:, i], a.win, a.sigma, a.pad, a.frame_h or None) for i in range(len(a.cams))], 1)
    np.savez(a.out, frames=F, cams=np.array(a.cams), boxes=sm, raw=raw)
    for i, c in enumerate(a.cams):
        ok = np.isfinite(raw[:, i]).all(1)
        r = raw[ok, i]; s = sm[ok, i]
        # fraction of frames where a raw mask extent pokes out of the smoothed box by > 2% of box size
        tol = 0.02 * (s[:, 2:] - s[:, :2]).repeat(2, 1)[:, [0, 1, 2, 3]]
        out = ((r[:, :2] < s[:, :2] - tol[:, :2]).any(1)) | ((r[:, 2:] > s[:, 2:] + tol[:, 2:]).any(1))
        print(f"{c}: masks {ok.sum()}/{len(F)}  raw mask outside smoothed box in {out.mean()*100:.1f}% of frames; "
              f"box h median {np.median(s[:,3]-s[:,1]):.0f}")
    print("saved", a.out)


if __name__ == "__main__":
    main()
