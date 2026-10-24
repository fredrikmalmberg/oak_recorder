"""Grid video (all cameras) with the MANO fits overlaid on the undistorted frames.

Per camera cell, a fixed-size crop following the smoothed wrist midpoint of the MANO-Sapiens fit, with
  - filled semi-transparent mesh: MANO fit to the Sapiens-triangulated hand (left orange, right blue)
  - wireframe: MANO fit to the WiLoR-triangulated hand (left green, right magenta)
  - white dots: Sapiens 2D keypoints, yellow dots: WiLoR 2D keypoints (undistorted, after tracking)
Meshes are in world metres (hand_pose.fit_mano output) and are projected with K[R|t] straight into the undistorted image.

  python -m hand_pose.render_mano_grid --take T --calib C --sap output/manoSap/mano_fit.npz --wil output/manoWil/mano_fit.npz \
      --out output/manoCmp/mano_grid.mp4
"""
import argparse, json, os, subprocess, sys
import cv2
import numpy as np
from scipy.ndimage import uniform_filter1d

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import calibrate  # noqa: E402
from pose2d import triangulation as T  # noqa: E402
from hand_pose.compare_mano_fits import load_2d  # noqa: E402

SAP_COL = {0: (40, 150, 255), 1: (255, 140, 40)}   # BGR orange / blue (left, right)
WIL_COL = {0: (60, 220, 60), 1: (230, 60, 230)}    # green / magenta


def project(P, V):
    q = V @ P[:, :3].T + P[:, 3]
    return q[:, :2] / q[:, 2:3], q[:, 2]


def draw_filled(crop, uv, depth, faces, color, alpha):
    tri = uv[faces]
    cen = depth[faces].mean(1)
    e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    area = np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    shade = 0.55 + 0.45 * np.clip(area / (np.percentile(area, 95) + 1e-9), 0, 1)
    pts = np.round(tri).astype(np.int32)
    ov = crop.copy()
    for k in np.argsort(-cen):
        cv2.fillConvexPoly(ov, pts[k], tuple(int(v) for v in np.array(color) * shade[k]), cv2.LINE_AA)
    cv2.addWeighted(ov, alpha, crop, 1 - alpha, 0, dst=crop)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--take", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--sap", required=True)
    ap.add_argument("--wil", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sap-2d", default="pose2d_sapiens_smoothbox")
    ap.add_argument("--wil-2d", default="pose2d_wilor")
    ap.add_argument("--frames", type=int, nargs=2, default=[300, 650])
    ap.add_argument("--crop", type=int, nargs=2, default=[1152, 648])
    ap.add_argument("--cell", type=int, nargs=2, default=[640, 360])
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--alpha", type=float, default=0.55)
    ap.add_argument("--follow-window", type=int, default=31)
    a = ap.parse_args()

    S, W = np.load(a.sap), np.load(a.wil)
    cams = [str(c) for c in np.load(f"{a.take}/aligned/wilor/wilor_hands.npz")["cams"]]
    calib = calibrate.load_calibration_output(a.calib)
    P = T.build_projection_matrices(calib, cams)
    n = len(S["frames"])
    w, h = calib[cams[0]]["width"], calib[cams[0]]["height"]
    det = {lab: [load_2d(f"{a.take}/aligned/{d}/landmarks_{side}.json", cams, n, w, h) for side in ("left", "right")]
           for lab, d in (("sap", a.sap_2d), ("wil", a.wil_2d))}
    frames = list(range(a.frames[0], a.frames[1] + 1))
    CW, CH = a.crop

    centres = {}
    for c in cams:
        raw = np.full((len(frames), 2), np.nan)
        for i, f in enumerate(frames):
            wr = [J[f, s, 0] for J, pres in ((S["joints"], S["present"]), (W["joints"], W["present"])) for s in range(2) if pres[f, s]]
            if wr:
                raw[i] = project(P[c], np.mean(wr[:2] if len(wr) >= 2 else wr, 0)[None])[0][0]
        for k in range(2):
            ok = np.isfinite(raw[:, k])
            raw[:, k] = np.interp(np.arange(len(frames)), np.where(ok)[0], raw[ok, k]) if ok.any() else calib[c]["width"] / 2
        centres[c] = uniform_filter1d(raw, a.follow_window, axis=0, mode="nearest")
    maps = {c: cv2.initUndistortRectifyMap(calib[c]["K"], calib[c]["dist"], None, calib[c]["K"], (int(w), int(h)), cv2.CV_16SC2) for c in cams}

    cw, ch = a.cell
    rows = -(-len(cams) // a.cols)
    Wd, Hd = cw * a.cols, ch * rows
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{Wd}x{Hd}",
                           "-r", str(a.fps), "-i", "-", "-c:v", "libx264", "-crf", "24", "-pix_fmt", "yuv420p", a.out], stdin=subprocess.PIPE)
    for i, f in enumerate(frames):
        grid = np.zeros((Hd, Wd, 3), np.uint8)
        for ci, c in enumerate(cams):
            img = cv2.remap(cv2.imread(f"{a.take}/aligned/{c}/{f:06d}.jpg"), *maps[c], cv2.INTER_LINEAR)
            x0 = int(np.clip(centres[c][i, 0] - CW / 2, 0, w - CW)); y0 = int(np.clip(centres[c][i, 1] - CH / 2, 0, h - CH))
            crop = img[y0:y0 + CH, x0:x0 + CW].copy()
            off = np.array([x0, y0], float)
            for s in range(2):
                if S["present"][f, s]:
                    uv, z = project(P[c], S["verts"][f, s].astype(np.float64))
                    draw_filled(crop, uv - off, z, S["faces"][s], SAP_COL[s], a.alpha)
            for s in range(2):
                if W["present"][f, s]:
                    uv, _ = project(P[c], W["verts"][f, s].astype(np.float64))
                    tri = np.round(uv[W["faces"][s]] - off).astype(np.int32)
                    cv2.polylines(crop, list(tri), True, WIL_COL[s], 1, cv2.LINE_AA)
            for lab, col in (("sap", (255, 255, 255)), ("wil", (0, 255, 255))):
                for s in range(2):
                    for pt in det[lab][s][f, ci]:
                        if np.isfinite(pt).all():
                            cv2.circle(crop, tuple(np.round(pt - off).astype(int)), 4, col, -1, cv2.LINE_AA)
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
