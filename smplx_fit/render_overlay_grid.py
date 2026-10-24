"""Grid video (all cameras) with a fitted SMPL-X body mesh overlaid on the undistorted frames.

Per camera cell, a fixed-size crop following the smoothed mesh centre:
  - filled semi-transparent mesh (upper body, hands hidden; legs hidden unless --show-legs)
  - cyan contour: SAM3 person mask, remapped to undistorted space with the calibration's K/dist
  - white dots + bones: Sapiens 2D body keypoints (ids 0-12, undistorted with the same calibration)
Meshes are in world metres and are projected with K[R|t] straight into the undistorted image.

  python -m smplx_fit.render_overlay_grid --take T --calib C --npz smplx_params.npz --pose2d-dir pose2d_sapiens_smoothbox --out grid.mp4
"""
import argparse, json, os, subprocess, sys
import cv2
import numpy as np
from scipy.ndimage import uniform_filter1d

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import calibrate  # noqa: E402
from hand_pose import hand_multiview as hmv  # noqa: E402
from hand_pose.render_mano_grid import draw_filled, project  # noqa: E402
from pose2d import triangulation as T  # noqa: E402
from pose2d.visualize_triangulation import load_smplx_mesh_sequence  # noqa: E402

BONES = [(0, 1), (1, 3), (0, 2), (2, 4), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (5, 6), (11, 12)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--take", required=True, help="dir with aligned/<cam>/ frames, aligned/masks_sam3 and aligned/<pose2d-dir>/")
    ap.add_argument("--calib", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--pose2d-dir", default="pose2d_sapiens_smoothbox")
    ap.add_argument("--landmarks-dir", default=None, help="dir with landmarks_body.json (default: <take>/aligned/<pose2d-dir>)")
    ap.add_argument("--model-path", default="models/SMPLX")
    ap.add_argument("--mask-subdir", default="masks_sam3")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, nargs=2, default=[300, 650])
    ap.add_argument("--crop", type=int, nargs=2, default=[1920, 1080])
    ap.add_argument("--cell", type=int, nargs=2, default=[640, 360])
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--follow-window", type=int, default=31)
    ap.add_argument("--show-legs", action="store_true")
    a = ap.parse_args()

    calib = calibrate.load_calibration_output(a.calib)
    cams = [c for c in sorted(calib) if os.path.isdir(f"{a.take}/aligned/{c}") and c != "cam6"]
    P = T.build_projection_matrices(calib, cams)
    verts, faces = load_smplx_mesh_sequence(a.npz, a.model_path, hide_lower_body=not a.show_legs, hide_hands=True)
    faces = np.asarray(faces)
    ld = a.landmarks_dir or f"{a.take}/aligned/{a.pose2d_dir}"
    lm = json.load(open(f"{ld}/landmarks_body.json"))
    frames = [f for f in range(a.frames[0], a.frames[1] + 1) if f"{f:06d}.jpg" in verts]
    w, h = int(calib[cams[0]]["width"]), int(calib[cams[0]]["height"])
    CW, CH = a.crop
    maps = {c: hmv.build_undistort_maps(calib[c]["K"], calib[c]["dist"], w, h) for c in cams}

    centres = {}
    for c in cams:
        raw = np.array([project(P[c], verts[f"{f:06d}.jpg"].mean(0, keepdims=True))[0][0] for f in frames])
        centres[c] = uniform_filter1d(raw, a.follow_window, axis=0, mode="nearest")

    cw, ch = a.cell
    rows = -(-len(cams) // a.cols)
    Wd, Hd = cw * a.cols, ch * rows
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{Wd}x{Hd}",
                           "-r", str(a.fps), "-i", "-", "-c:v", "libx264", "-crf", "24", "-pix_fmt", "yuv420p", a.out], stdin=subprocess.PIPE)
    for i, f in enumerate(frames):
        fk = f"{f:06d}.jpg"
        grid = np.zeros((Hd, Wd, 3), np.uint8)
        for ci, c in enumerate(cams):
            img = cv2.remap(cv2.imread(f"{a.take}/aligned/{c}/{fk}"), *maps[c], cv2.INTER_LINEAR)
            x0 = int(np.clip(centres[c][i, 0] - CW / 2, 0, w - CW)); y0 = int(np.clip(centres[c][i, 1] - CH / 2, 0, h - CH))
            crop = img[y0:y0 + CH, x0:x0 + CW].copy()
            off = np.array([x0, y0], float)
            uv, z = project(P[c], verts[fk].astype(np.float64))
            draw_filled(crop, uv - off, z, faces, (60, 180, 255), a.alpha)
            m = cv2.imread(f"{a.take}/aligned/{a.mask_subdir}/{c}/{fk}", cv2.IMREAD_GRAYSCALE)
            if m is not None:
                m = cv2.remap(m, *maps[c], cv2.INTER_NEAREST)[y0:y0 + CH, x0:x0 + CW]
                cnt, _ = cv2.findContours((m > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(crop, cnt, -1, (255, 255, 0), 3, cv2.LINE_AA)
            pts = {int(k): (np.array(v[:2]) * [w, h] - off) for k, v in lm.get(c, {}).get(fk, {}).items() if int(k) < 13}
            for p, q in BONES:
                if p in pts and q in pts:
                    cv2.line(crop, tuple(np.round(pts[p]).astype(int)), tuple(np.round(pts[q]).astype(int)), (255, 255, 255), 2, cv2.LINE_AA)
            for pt in pts.values():
                cv2.circle(crop, tuple(np.round(pt).astype(int)), 7, (255, 255, 255), -1, cv2.LINE_AA)
            cell = cv2.resize(crop, (cw, ch), interpolation=cv2.INTER_AREA)
            for col, th in (((0, 0, 0), 4), ((255, 255, 255), 1)):
                cv2.putText(cell, f"{c}  frame {f}", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, th, cv2.LINE_AA)
            r, k = divmod(ci, a.cols)
            grid[r * ch:(r + 1) * ch, k * cw:(k + 1) * cw] = cell
        ff.stdin.write(grid.tobytes())
        if i % 50 == 0:
            print(f"frame {f} ({i + 1}/{len(frames)})", flush=True)
    ff.stdin.close(); ff.wait()
    print("wrote", a.out)


if __name__ == "__main__":
    main()
