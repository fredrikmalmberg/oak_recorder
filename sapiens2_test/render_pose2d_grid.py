"""3x2 grid video of the 2D landmarks that actually feed triangulation (pose2d_<name>/landmarks_*.json,
after tracking), drawn on the raw distorted frames. Stored coordinates are normalised UNDISTORTED, so
they are re-distorted with cv2.projectPoints (rvec=t=0 on K^-1 rays) before drawing.
Only parts whose part confidence >= --min-conf (the triangulation gate) are drawn.

With --boxes (smooth_bbox npz): red = raw RVM mask extents, cyan = crop Sapiens actually saw
(smoothed box x1.25, aspect-fixed). Hips (body ids 11, 12) are drawn as large magenta rings.

  python -m sapiens2_test.render_pose2d_grid --take T --calib CAL --pose2d-dir pose2d_sapiens_smoothbox --out out.mp4
"""
import argparse, json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
import cv2, numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import calibrate  # noqa: E402
from sapiens2_test.smooth_bbox import effective_crop  # noqa: E402

COLORS = {"body": (0, 255, 255), "left": (255, 200, 120), "right": (60, 140, 255)}  # BGR
RADIUS = {"body": 1.0, "left": 0.6, "right": 0.6}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--take", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--pose2d-dir", default="pose2d_sapiens_smoothbox")
    ap.add_argument("--out", required=True)
    ap.add_argument("--boxes", default="", help="boxes_smooth.npz: also draw raw mask extents (red) and the used crop (cyan)")
    ap.add_argument("--cams", nargs="+", default=["cam0", "cam1", "cam2", "cam3", "cam4", "cam5"])
    ap.add_argument("--cell-w", type=int, default=800)
    ap.add_argument("--radius", type=float, default=3.5)
    ap.add_argument("--min-conf", type=float, default=0.5)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--crf", type=int, default=28)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=768)
    a = ap.parse_args()

    base = os.path.join(a.take, "aligned", a.pose2d_dir)
    calib = calibrate.load_calibration_output(a.calib)
    L = {p: json.load(open(f"{base}/landmarks_{p}.json")) for p in ("left", "right", "body")}
    C = {"left": json.load(open(f"{base}/confidence_left.json")), "right": json.load(open(f"{base}/confidence_right.json")),
         "body": json.load(open(f"{base}/extraction.json"))["confidence_body"]}
    bx = None
    if a.boxes:
        d = np.load(a.boxes)
        assert [str(c) for c in d["cams"]] == a.cams, (list(d["cams"]), a.cams)
        bx = {"frames": {int(f): i for i, f in enumerate(d["frames"])}, "raw": d["raw"], "crop": effective_crop(d["boxes"])}
    cw = a.cell_w; ch = cw * 9 // 16; s = cw / 3840.0
    W, H = 3 * cw, 2 * ch
    tmp = a.out + ".tmp.mp4"
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (W, H))

    def redistort(c, uv):
        K = np.asarray(calib[c]["K"], np.float64)
        rays = np.linalg.solve(K, np.c_[uv, np.ones(len(uv))].T).T
        p, _ = cv2.projectPoints(rays.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, np.asarray(calib[c]["dist"], np.float64))
        return p.reshape(-1, 2)

    def cell(c, f):
        k = f"{f:06d}.jpg"
        img = cv2.resize(cv2.imread(f"{a.take}/aligned/{c}/{k}"), (cw, ch), interpolation=cv2.INTER_AREA)
        w, h = calib[c]["width"], calib[c]["height"]
        if bx is not None and f in bx["frames"]:
            fi, ci = bx["frames"][f], a.cams.index(c)
            for arr, col in ((bx["raw"], (60, 60, 255)), (bx["crop"], (255, 220, 0))):
                b = arr[fi, ci]
                if np.isfinite(b).all():
                    cv2.rectangle(img, (int(b[0] * s), int(b[1] * s)), (int(b[2] * s), int(b[3] * s)), col, 2, cv2.LINE_AA)
        for part in ("left", "right", "body"):
            lm = L[part].get(c, {}).get(k)
            if not lm or C[part].get(c, {}).get(k, 0) < a.min_conf:
                continue
            ids = list(lm.keys())
            uv = np.array([[v[0] * w, v[1] * h] for v in lm.values()])
            for lid, (x, y) in zip(ids, redistort(c, uv)):
                px, py = x * s, y * s
                if -50 < px < cw + 50 and -50 < py < ch + 50:
                    if part == "body" and lid in ("11", "12"):
                        cv2.circle(img, (int(round(px * 16)), int(round(py * 16))), int(round(8 * 16)), (255, 0, 255), 2, cv2.LINE_AA, 4)
                    cv2.circle(img, (int(round(px * 16)), int(round(py * 16))), int(round(a.radius * RADIUS[part] * 16)),
                               COLORS[part], -1, cv2.LINE_AA, 4)
        cv2.putText(img, f"{c}  frame {f}", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        return img

    with ThreadPoolExecutor(6) as ex:
        for f in range(a.start, a.stop):
            cells = list(ex.map(lambda c: cell(c, f), a.cams))
            vw.write(np.vstack([np.hstack(cells[:3]), np.hstack(cells[3:])]))
            if f % 50 == 0:
                print(f"{f}/{a.stop}", flush=True)
    vw.release()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264", "-crf", str(a.crf),
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", a.out], check=True)
    os.remove(tmp)
    print("saved", a.out, f"{os.path.getsize(a.out)/1e6:.1f} MB")


if __name__ == "__main__":
    main()
