"""3x2 grid video of the keypoint-vs-silhouette check (output of kp_silhouette_check).

Per keypoint (Lsh, Rsh, Lhip, Rhip) and frame: dot colour = status, cyan arrow = keypoint step (t-1 -> t),
yellow arrow = silhouette flow in the window (only the observable component), both x --gain.
  green  rank>0 and e_max <= thr (keypoint agrees with the silhouette)
  red    rank>0 and e_max >  thr (keypoint moved differently from the silhouette: do not trust)
  grey   rank 0 (no silhouette edge in the window: cannot tell)
Frames and keypoints are both in distorted pixel space, so no remapping is needed.
"""
import argparse, os, subprocess
from concurrent.futures import ThreadPoolExecutor
import cv2, numpy as np

NAMES = {"Lsh": 5, "Rsh": 6, "Lhip": 9, "Rhip": 10}
GREEN, RED, GREY, CYAN, YEL = (0, 220, 0), (0, 0, 255), (150, 150, 150), (255, 255, 0), (0, 230, 255)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--take", required=True)
    ap.add_argument("--npz", default="output/sapiens2/allframes_1b_smoothbox.npz")
    ap.add_argument("--check", default="output/sapiens2/kp_sil_check.npz")
    ap.add_argument("--out", default="output/sapiens2/grid_sil_check.mp4")
    ap.add_argument("--thr", type=float, default=4.0, help="e_max (full-res px) above which a keypoint is flagged")
    ap.add_argument("--gain", type=float, default=8.0)
    ap.add_argument("--flow-only", action="store_true", help="only draw the silhouette-flow arrow (neutral keypoint dot, no step arrow, no status colours)")
    ap.add_argument("--cell-w", type=int, default=800)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--crf", type=int, default=28)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=None)
    a = ap.parse_args()

    d = np.load(a.npz); frames, cams, kp, sc = d["frames"], [str(c) for c in d["cams"]], d["kp"], d["sc"]
    chk = np.load(a.check)
    cw = a.cell_w; ch = cw * 9 // 16; s = cw / 3840.0
    vw = cv2.VideoWriter(a.out + ".tmp.mp4", cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (3 * cw, 2 * ch))
    stop = a.stop if a.stop is not None else len(frames)
    P = lambda x, y: (int(round(x * s * 16)), int(round(y * s * 16)))

    def cell(ci, fi):
        c = cams[ci]; f = int(frames[fi])
        img = cv2.resize(cv2.imread(f"{a.take}/aligned/{c}/{f:06d}.jpg"), (cw, ch), interpolation=cv2.INTER_AREA)
        m = cv2.imread(f"{a.take}/aligned/masks_rvm/{c}/{f:06d}.jpg", 0)
        if m is not None:
            m = cv2.resize(m, (cw, ch), interpolation=cv2.INTER_AREA) > 128
            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(img, cs, -1, (60, 60, 255), 1, cv2.LINE_AA)
        nbad = 0
        for n, j in NAMES.items():
            r = chk[f"{c}/{n}"][fi]
            x, y = kp[fi, ci, j]
            if not np.isfinite(r[7]) or not np.isfinite(x):
                continue
            rank, emax = int(r[6]), r[7]
            bad = rank > 0 and emax > a.thr
            col = GREY if rank == 0 else (RED if bad else GREEN)
            nbad += bad
            if a.flow_only:
                cv2.circle(img, P(x, y), 3 * 16, (255, 255, 255), -1, cv2.LINE_AA, 4)
                if rank > 0:
                    cv2.arrowedLine(img, P(x, y), P(x + a.gain * r[2], y + a.gain * r[3]), YEL, 2, cv2.LINE_AA, 4, 0.3)
                continue
            cv2.circle(img, P(x, y), 6 * 16, col, 2 if not bad else 3, cv2.LINE_AA, 4)
            cv2.arrowedLine(img, P(x, y), P(x + a.gain * r[0], y + a.gain * r[1]), CYAN, 1, cv2.LINE_AA, 4, 0.25)
            cv2.arrowedLine(img, P(x, y), P(x + a.gain * r[2], y + a.gain * r[3]), YEL, 1, cv2.LINE_AA, 4, 0.25)
            if bad:
                cv2.putText(img, f"{n} {emax:.0f}px", (int(x * s) + 9, int(y * s) - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.45, RED, 1, cv2.LINE_AA)
        cv2.putText(img, f"{c}  frame {f}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        if nbad and not a.flow_only:
            cv2.putText(img, f"{nbad} flagged", (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2, cv2.LINE_AA)
        return img

    with ThreadPoolExecutor(6) as ex:
        for fi in range(a.start, stop):
            cells = list(ex.map(lambda ci: cell(ci, fi), range(len(cams))))
            vw.write(np.vstack([np.hstack(cells[:3]), np.hstack(cells[3:])]))
            if fi % 100 == 0: print(f"{fi}/{stop}", flush=True)
    vw.release()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", a.out + ".tmp.mp4", "-c:v", "libx264", "-crf", str(a.crf),
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", a.out], check=True)
    os.remove(a.out + ".tmp.mp4")
    print("saved", a.out, f"{os.path.getsize(a.out)/1e6:.1f} MB")


if __name__ == "__main__":
    main()
