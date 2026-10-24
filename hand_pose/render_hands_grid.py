"""Grid video (all cameras) of a triangulated hand reconstruction, reprojected onto the undistorted camera frames.

Per camera cell: a fixed-size crop of the undistorted frame that follows the (heavily smoothed) midpoint of the two
wrists, with
  - thick coloured bones: the smoothed 3D reconstruction reprojected into that camera (left hand orange, right blue)
  - small white dots: the raw per-camera 2D WiLoR detections (kp2d_calib, undistorted) that fed the triangulation
so 2D-detector jitter and 3D-reconstruction jitter can be told apart. With --mesh, each camera's own WiLoR MANO mesh
(wilor_meshes.npz, per-camera, not triangulated) is drawn semi-transparent underneath.

  python -m hand_pose.render_hands_grid --take <take> --calib <calib.json> \
      --recon output/wilorTri/aligned/pose2d_wilor --out output/wilorTri/wilor_grid.mp4 --frames 300 650
"""
import argparse, json, os, subprocess, sys
import cv2
import numpy as np
from scipy.ndimage import uniform_filter1d

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import calibrate  # noqa: E402
from pose2d import triangulation as T  # noqa: E402

BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (9, 10), (10, 11), (11, 12),
         (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20), (5, 9), (9, 13), (13, 17)]
COL = {"left": (40, 150, 255), "right": (255, 140, 40)}  # BGR: orange, blue


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--take", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--recon", required=True, help="dir with reconstruction_{left,right}.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--kind", default="smoothed", choices=["smoothed", "raw"])
    ap.add_argument("--frames", type=int, nargs=2, default=[300, 650])
    ap.add_argument("--crop", type=int, nargs=2, default=[1152, 648], help="crop W H in full-res px")
    ap.add_argument("--cell", type=int, nargs=2, default=[640, 360])
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--mesh", action="store_true", help="draw the per-camera WiLoR mesh (<take>/aligned/wilor/wilor_meshes.npz)")
    ap.add_argument("--mesh-alpha", type=float, default=0.55)
    ap.add_argument("--follow-window", type=int, default=31, help="frames of moving average for the crop centre")
    a = ap.parse_args()

    d = np.load(f"{a.take}/aligned/wilor/wilor_hands.npz")
    cams = [str(c) for c in d["cams"]]
    all_frames = d["frames"]
    calib = calibrate.load_calibration_output(a.calib)
    P = T.build_projection_matrices(calib, cams)
    rec = {p: json.load(open(f"{a.recon}/reconstruction_{p}.json"))[a.kind] for p in ("left", "right")}
    frames = [int(f) for f in all_frames if a.frames[0] <= f <= a.frames[1]]
    fidx = {int(f): i for i, f in enumerate(all_frames)}

    def proj(c, X):
        q = P[c] @ np.append(X, 1.0)
        return q[:2] / q[2]

    def pts3d(part, f):
        r = rec[part].get(f"{f:06d}.jpg")
        return None if not r else {int(k): np.asarray(v, float) for k, v in r.items()}

    if a.mesh:
        mz = np.load(f"{a.take}/aligned/wilor/wilor_meshes.npz")
        verts_cam, faces = mz["verts_cam"], mz["mano_faces"]

    def draw_mesh(crop, V, c, off, color):
        K = calib[c]["K"]
        uv = (V @ K.T)
        uv = uv[:, :2] / uv[:, 2:3]
        uv = cv2.undistortPoints(uv.reshape(-1, 1, 2).astype(np.float64), K, calib[c]["dist"], P=K).reshape(-1, 2) - off
        tri = V[faces]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        cen = tri.mean(1)
        shade = np.abs((n * cen).sum(1) / np.linalg.norm(cen, axis=1))
        pts = np.round(uv[faces]).astype(np.int32)
        ov = crop.copy()
        for k in np.argsort(-cen[:, 2]):
            cv2.fillConvexPoly(ov, pts[k], tuple(int(v) for v in np.array(color) * (0.35 + 0.65 * shade[k])), cv2.LINE_AA)
        cv2.addWeighted(ov, a.mesh_alpha, crop, 1 - a.mesh_alpha, 0, dst=crop)

    CW, CH = a.crop
    centres = {}
    for c in cams:
        raw = np.full((len(frames), 2), np.nan)
        for i, f in enumerate(frames):
            w = [p3[0] for p3 in (pts3d("left", f), pts3d("right", f)) if p3 and 0 in p3]
            if w:
                raw[i] = proj(c, np.mean(w, 0))
        for k in range(2):  # fill gaps, then smooth
            ok = np.isfinite(raw[:, k])
            raw[:, k] = np.interp(np.arange(len(frames)), np.where(ok)[0], raw[ok, k]) if ok.any() else calib[c]["width"] / 2
        centres[c] = uniform_filter1d(raw, a.follow_window, axis=0, mode="nearest")

    maps = {}
    for c in cams:
        w, h = calib[c]["width"], calib[c]["height"]
        maps[c] = cv2.initUndistortRectifyMap(calib[c]["K"], calib[c]["dist"], None, calib[c]["K"], (int(w), int(h)), cv2.CV_16SC2)

    cw, ch = a.cell
    rows = -(-len(cams) // a.cols)
    W, H = cw * a.cols, ch * rows
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                           "-r", str(a.fps), "-i", "-", "-c:v", "libx264", "-crf", "24", "-pix_fmt", "yuv420p", a.out],
                          stdin=subprocess.PIPE)
    for i, f in enumerate(frames):
        grid = np.zeros((H, W, 3), np.uint8)
        for ci, c in enumerate(cams):
            img = cv2.imread(f"{a.take}/aligned/{c}/{f:06d}.jpg")
            img = cv2.remap(img, *maps[c], cv2.INTER_LINEAR)
            ih, iw = img.shape[:2]
            x0 = int(np.clip(centres[c][i, 0] - CW / 2, 0, iw - CW)); y0 = int(np.clip(centres[c][i, 1] - CH / 2, 0, ih - CH))
            crop = img[y0:y0 + CH, x0:x0 + CW].copy()
            off = np.array([x0, y0], float)
            if a.mesh:
                for s, part in enumerate(("left", "right")):
                    if d["valid"][fidx[f], cams.index(c), s]:
                        draw_mesh(crop, verts_cam[fidx[f], cams.index(c), s].astype(np.float64), c, off, COL[part])
            for s, part in enumerate(("left", "right")):
                p3 = pts3d(part, f)
                if p3:
                    q = {j: proj(c, X) - off for j, X in p3.items()}
                    for u, v in BONES:
                        if u in q and v in q:
                            cv2.line(crop, tuple(np.round(q[u]).astype(int)), tuple(np.round(q[v]).astype(int)), COL[part], 4, cv2.LINE_AA)
                    for pt in q.values():
                        cv2.circle(crop, tuple(np.round(pt).astype(int)), 5, COL[part], -1, cv2.LINE_AA)
                t = fidx[f]
                ci_ = cams.index(c)
                if d["valid"][t, ci_, s]:
                    kp = cv2.undistortPoints(d["kp2d_calib"][t, ci_, s].reshape(-1, 1, 2).astype(np.float64),
                                             calib[c]["K"], calib[c]["dist"], P=calib[c]["K"]).reshape(-1, 2) - off
                    for pt in kp:
                        cv2.circle(crop, tuple(np.round(pt).astype(int)), 4, (255, 255, 255), -1, cv2.LINE_AA)
            cell = cv2.resize(crop, (cw, ch), interpolation=cv2.INTER_AREA)
            cv2.putText(cell, f"{c}  frame {f}", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(cell, f"{c}  frame {f}", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
            r, k = divmod(ci, a.cols)
            grid[r * ch:(r + 1) * ch, k * cw:(k + 1) * cw] = cell
        ff.stdin.write(grid.tobytes())
        if i % 50 == 0:
            print(f"frame {f} ({i + 1}/{len(frames)})", flush=True)
    ff.stdin.close(); ff.wait()
    print("wrote", a.out)


if __name__ == "__main__":
    main()
