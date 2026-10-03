"""3x2 camera grid video with two 3D reconstructions reprojected onto the (distorted) frames.

  python -m sapiens2.render_recon_grid --take T --calib CAL \
      --recon-a <dir> --recon-b <dir> --out output/sapiens2/grid_recon_cmp.mp4
A = light blue (larger dots, underneath), B = yellow (on top). Projection uses cv2.projectPoints
with the calibration's dist coeffs, so points land in distorted pixel space like the raw frames.
"""
import argparse, json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
import cv2, numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import calibrate  # noqa: E402

COL_A, COL_B = (255, 200, 120), (0, 255, 255)


def load_points(recon_dir, kind):
    """-> {frame_idx: [(part, lm, xyz)]} flattened to arrays per frame."""
    out = {}
    for part in ("left", "right", "body"):
        rec = json.load(open(os.path.join(recon_dir, f"reconstruction_{part}.json")))[kind]
        for fk, pts in rec.items():
            f = int(os.path.splitext(fk)[0])
            for lm, xyz in pts.items():
                out.setdefault(f, []).append((part == "body", xyz))
    return {f: (np.array([b for b, _ in v]), np.array([x for _, x in v], np.float64)) for f, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--take", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--recon-a", required=True)
    ap.add_argument("--recon-b", required=True)
    ap.add_argument("--kind", default="raw", choices=["raw", "smoothed"])
    ap.add_argument("--cams", nargs="+", default=["cam0", "cam1", "cam2", "cam3", "cam4", "cam5"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--cell-w", type=int, default=800)
    ap.add_argument("--radius", type=float, default=3)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--crf", type=int, default=28)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=768)
    a = ap.parse_args()

    calib = calibrate.load_calibration_output(a.calib)
    cams = a.cams
    A, B = load_points(a.recon_a, a.kind), load_points(a.recon_b, a.kind)
    cw = a.cell_w; ch = cw * 9 // 16; s = cw / 3840.0
    W, H = 3 * cw, 2 * ch
    tmp = a.out + ".tmp.mp4"
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (W, H))

    def project(c, xyz):
        cal = calib[c]
        rvec, _ = cv2.Rodrigues(np.asarray(cal["R"], np.float64))
        p, _ = cv2.projectPoints(xyz, rvec, np.asarray(cal["t"], np.float64).reshape(3, 1),
                                 np.asarray(cal["K"], np.float64), np.asarray(cal["dist"], np.float64))
        return p.reshape(-1, 2)

    def cell(c, f):
        img = cv2.imread(f"{a.take}/aligned/{c}/{f:06d}.jpg")
        img = cv2.resize(img, (cw, ch), interpolation=cv2.INTER_AREA)
        for pts, col, rm in ((A, COL_A, 1.7), (B, COL_B, 1.0)):
            if f not in pts:
                continue
            is_body, xyz = pts[f]
            for (x, y), body in zip(project(c, xyz), is_body):
                px, py = x * s, y * s
                if -50 < px < cw + 50 and -50 < py < ch + 50:
                    r = a.radius * rm * (1.0 if body else 0.6)
                    cv2.circle(img, (int(round(px * 16)), int(round(py * 16))), int(round(r * 16)), col, -1, cv2.LINE_AA, 4)
        cv2.putText(img, f"{c}  frame {f}", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        return img

    with ThreadPoolExecutor(6) as ex:
        for f in range(a.start, a.stop):
            cells = list(ex.map(lambda c: cell(c, f), cams))
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
