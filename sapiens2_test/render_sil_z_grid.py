"""3x2 grid video of the windowed keypoint-vs-silhouette score (kp_silhouette_check.windowed_z).

Per shoulder/hip keypoint and frame t, over the 4 steps t-1..t+2:
  cyan arrow   keypoint displacement over the window       (x --gain)
  yellow arrow silhouette displacement over the window, observable directions only (x --gain)
  ring         red = z > --thr (keypoint moved but silhouette did not), green = ok, grey = not observable
Dim blue tint = arm/hand pixels excluded from the silhouette windows. Red line = RVM outline.
"""
import argparse, os, subprocess
from concurrent.futures import ThreadPoolExecutor
import cv2, numpy as np
from sapiens2_test.kp_silhouette_check import windowed_z, arm_mask

NAMES = {"Lsh": 5, "Rsh": 6, "Lhip": 9, "Rhip": 10}
GREEN, RED, GREY, CYAN, YEL = (0, 220, 0), (0, 0, 255), (150, 150, 150), (255, 255, 0), (0, 230, 255)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--take", required=True)
    ap.add_argument("--npz", default="output/sapiens2/allframes_1b_smoothbox.npz")
    ap.add_argument("--check", default="output/sapiens2/kp_sil_check_v2.npz")
    ap.add_argument("--out", default="output/sapiens2/grid_sil_z_300_650.mp4")
    ap.add_argument("--thr", type=float, default=15.0)
    ap.add_argument("--gain", type=float, default=3.0)
    ap.add_argument("--start", type=int, default=300)
    ap.add_argument("--stop", type=int, default=651, help="frame number, exclusive")
    ap.add_argument("--cell-w", type=int, default=800)
    ap.add_argument("--fps", type=int, default=5)
    ap.add_argument("--crf", type=int, default=30)
    a = ap.parse_args()

    d = np.load(a.npz); frames, cams, kp, sc = d["frames"], [str(c) for c in d["cams"]], d["kp"], d["sc"]
    chk = np.load(a.check); fl = list(frames)
    Z = {k: windowed_z(chk[k]) for k in chk.files if "/" in k}
    cw = a.cell_w; ch = cw * 9 // 16; s = cw / 3840.0
    vw = cv2.VideoWriter(a.out + ".tmp.mp4", cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (3 * cw, 2 * ch))
    P = lambda x, y: (int(round(x * s * 16)), int(round(y * s * 16)))

    def cell(ci, fi):
        c = cams[ci]; f = int(frames[fi])
        img = cv2.resize(cv2.imread(f"{a.take}/aligned/{c}/{f:06d}.jpg"), (cw, ch), interpolation=cv2.INTER_AREA)
        E = arm_mask(kp[fi, ci], sc[fi, ci], s, (ch, cw)).astype(bool)
        img[E] = (0.7 * img[E] + 0.3 * np.array([255, 120, 0])).astype(np.uint8)
        m = cv2.imread(f"{a.take}/aligned/masks_rvm/{c}/{f:06d}.jpg", 0)
        if m is not None:
            m = cv2.resize(m, (cw, ch), interpolation=cv2.INTER_AREA) > 128
            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(img, cs, -1, (60, 60, 255), 1, cv2.LINE_AA)
        nbad = 0
        for n, j in NAMES.items():
            arr = chk[f"{c}/{n}"]; z = Z[f"{c}/{n}"][fi]; x, y = kp[fi, ci, j]
            if not np.isfinite(x):
                continue
            if not np.isfinite(z):
                cv2.circle(img, P(x, y), 6 * 16, GREY, 2, cv2.LINE_AA, 4); continue
            w = arr[fi - 1:fi + 3]
            dk = np.nansum(w[:, 0:2], axis=0); vs = np.nansum(w[:, 2:4], axis=0)
            bad = z > a.thr; nbad += bad
            cv2.circle(img, P(x, y), 6 * 16, RED if bad else GREEN, 3 if bad else 2, cv2.LINE_AA, 4)
            cv2.arrowedLine(img, P(x, y), P(x + a.gain * dk[0], y + a.gain * dk[1]), CYAN, 1, cv2.LINE_AA, 4, 0.25)
            cv2.arrowedLine(img, P(x, y), P(x + a.gain * vs[0], y + a.gain * vs[1]), YEL, 2, cv2.LINE_AA, 4, 0.25)
            if bad:
                cv2.putText(img, f"{n} z{z:.0f}", (int(x * s) + 9, int(y * s) - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.45, RED, 1, cv2.LINE_AA)
        cv2.putText(img, f"{c}  frame {f}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        if nbad:
            cv2.putText(img, f"{nbad} flagged", (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2, cv2.LINE_AA)
        return img

    with ThreadPoolExecutor(6) as ex:
        for fi in range(fl.index(a.start), fl.index(a.stop - 1) + 1):
            cells = list(ex.map(lambda ci: cell(ci, fi), range(len(cams))))
            vw.write(np.vstack([np.hstack(cells[:3]), np.hstack(cells[3:])]))
    vw.release()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", a.out + ".tmp.mp4", "-c:v", "libx264", "-crf", str(a.crf),
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", a.out], check=True)
    os.remove(a.out + ".tmp.mp4")
    print("saved", a.out, f"{os.path.getsize(a.out)/1e6:.1f} MB")


if __name__ == "__main__":
    main()
