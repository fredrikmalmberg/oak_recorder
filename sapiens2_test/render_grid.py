"""Render a 3x2 camera grid video with Sapiens2 keypoints overlaid.

  python -m sapiens2_test.render_grid --take ... --npz output/sapiens2/allframes_1b.npz --out output/sapiens2/grid_1b.mp4
Frames and keypoints are both in distorted pixel space, so the overlay needs no remapping.
"""
import argparse, subprocess, os
from concurrent.futures import ThreadPoolExecutor
import cv2, numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--take", required=True)
    ap.add_argument("--npz", default="output/sapiens2/allframes_1b.npz")
    ap.add_argument("--npz-old", default="", help="overlay: draw this run in light blue (larger dots) under --npz in yellow")
    ap.add_argument("--out", default="output/sapiens2/grid_1b.mp4")
    ap.add_argument("--cell-w", type=int, default=800)
    ap.add_argument("--thr", type=float, default=0.3)
    ap.add_argument("--radius", type=float, default=3, help="dot radius in cell pixels")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--crf", type=int, default=28)
    ap.add_argument("--masks", action="store_true", help="outline the RVM masks (aligned/masks_rvm) in red")
    ap.add_argument("--ring", type=int, nargs="*", default=[], help="Sapiens keypoint ids to circle with a large magenta ring")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=None)
    a = ap.parse_args()

    d = np.load(a.npz)
    frames, cams, kp, sc = d["frames"], list(d["cams"]), d["kp"], d["sc"]
    old = np.load(a.npz_old) if a.npz_old else None
    if old is not None:
        assert np.array_equal(old["frames"], d["frames"]) and [str(c) for c in old["cams"]] == [str(c) for c in cams]
    if old is not None:
        old_kp, old_sc = old["kp"], old["sc"]
    cw = a.cell_w; ch = cw * 9 // 16; s = cw / 3840.0
    W, H = 3 * cw, 2 * ch
    tmp = a.out + ".tmp.mp4"
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (W, H))
    stop = a.stop if a.stop is not None else len(frames)

    def cell(ci, fi):
        c = cams[ci]
        img = cv2.imread(f"{a.take}/aligned/{c}/{int(frames[fi]):06d}.jpg")
        img = cv2.resize(img, (cw, ch), interpolation=cv2.INTER_AREA)
        if a.masks:
            m = cv2.imread(f"{a.take}/aligned/masks_rvm/{c}/{int(frames[fi]):06d}.jpg", 0)
            if m is not None:
                m = cv2.resize(m, (cw, ch), interpolation=cv2.INTER_AREA) > 128
                cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(img, cs, -1, (60, 60, 255), 2, cv2.LINE_AA)
        layers = [(kp, sc, (0, 255, 0), a.radius)] if old is None else [
            (old_kp, old_sc, (255, 200, 120), a.radius * 1.7), (kp, sc, (0, 255, 255), a.radius)]
        for K, S, col, rad in layers:
            for (x, y), v in zip(K[fi, ci], S[fi, ci]):
                if v > a.thr and np.isfinite(x) and np.isfinite(y):
                    px, py = float(x) * s, float(y) * s
                    if -50 < px < cw + 50 and -50 < py < ch + 50:
                        cv2.circle(img, (int(round(px * 16)), int(round(py * 16))), int(round(rad * 16)), col, -1, cv2.LINE_AA, 4)
        for j in a.ring:
            x, y = kp[fi, ci, j]
            if sc[fi, ci, j] > a.thr and np.isfinite(x) and np.isfinite(y):
                cv2.circle(img, (int(round(x * s * 16)), int(round(y * s * 16))), 8 * 16, (255, 0, 255), 2, cv2.LINE_AA, 4)
        cv2.putText(img, f"{c}  frame {int(frames[fi])}", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        return img

    with ThreadPoolExecutor(6) as ex:
        for fi in range(a.start, stop):
            cells = list(ex.map(lambda ci: cell(ci, fi), range(len(cams))))
            vw.write(np.vstack([np.hstack(cells[:3]), np.hstack(cells[3:])]))
            if fi % 50 == 0:
                print(f"{fi}/{stop}", flush=True)
    vw.release()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264", "-crf", str(a.crf),
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", a.out], check=True)
    os.remove(tmp)
    print("saved", a.out, f"{os.path.getsize(a.out)/1e6:.1f} MB")


if __name__ == "__main__":
    main()
