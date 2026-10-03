"""3x2 camera grid video showing the Sapiens crop boxes (no model needed).

Red    = raw per-frame RVM mask extents (tight)
Yellow = crop Sapiens actually saw in the current run (mask extents +5%, x1.25, aspect-fixed)
Cyan   = crop from the modified (smoothed, bottom-pinned) box, same x1.25 + aspect fix

  python -m sapiens2.render_boxes --take TAKE --boxes output/sapiens2/boxes_smooth.npz --out output/sapiens2/boxes_grid.mp4
"""
import argparse, os, subprocess
from concurrent.futures import ThreadPoolExecutor
import cv2, numpy as np
from sapiens2.smooth_bbox import effective_crop, padded_raw


def rect(img, b, s, col, th):
    if np.isfinite(b).all():
        cv2.rectangle(img, (int(b[0] * s), int(b[1] * s)), (int(b[2] * s), int(b[3] * s)), col, th, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--take", required=True)
    ap.add_argument("--boxes", default="output/sapiens2/boxes_smooth.npz")
    ap.add_argument("--out", default="output/sapiens2/boxes_grid.mp4")
    ap.add_argument("--cell-w", type=int, default=800)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--crf", type=int, default=28)
    a = ap.parse_args()
    d = np.load(a.boxes)
    frames, cams, sm, raw = d["frames"], [str(c) for c in d["cams"]], d["boxes"], d["raw"]
    crop = effective_crop(sm)
    old = effective_crop(padded_raw(raw))
    cw = a.cell_w; ch = cw * 9 // 16; s = cw / 3840.0
    tmp = a.out + ".tmp.mp4"
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (3 * cw, 2 * ch))

    def cell(ci, fi):
        c = cams[ci]
        img = cv2.resize(cv2.imread(f"{a.take}/aligned/{c}/{int(frames[fi]):06d}.jpg"), (cw, ch), interpolation=cv2.INTER_AREA)
        rect(img, raw[fi, ci], s, (60, 60, 255), 2)
        rect(img, old[fi, ci], s, (0, 215, 255), 2)
        rect(img, crop[fi, ci], s, (255, 220, 0), 2)
        cv2.putText(img, f"{c}  frame {int(frames[fi])}", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        return img

    with ThreadPoolExecutor(6) as ex:
        for fi in range(len(frames)):
            cells = list(ex.map(lambda ci: cell(ci, fi), range(len(cams))))
            vw.write(np.vstack([np.hstack(cells[:3]), np.hstack(cells[3:])]))
            if fi % 100 == 0:
                print(f"{fi}/{len(frames)}", flush=True)
    vw.release()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264", "-crf", str(a.crf),
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", a.out], check=True)
    os.remove(tmp)
    print("saved", a.out, f"{os.path.getsize(a.out)/1e6:.1f} MB")


if __name__ == "__main__":
    main()
