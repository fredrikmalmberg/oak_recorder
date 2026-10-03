"""Viser playback of the triangulated Sapiens2 skeleton (pose2d_sapiens) with colour-coded bones.

Hands: one colour per finger (thumb..pinky), left/right hand tinted differently.
Body: left side blue, right side orange, midline white, plus neck (shoulder midpoint -> nose)
and spine (shoulder midpoint -> hip midpoint). Camera frustums from --calib, optionally a
second calibration with --ref-calib (orange frustums).

  python -m sapiens2_test.view_sapiens TAKE --calib output/calibration/..._sapiens_ext8000.json --port 8080
"""
import argparse, colorsys, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import calibrate  # noqa: E402
from hand_pose import hand_multiview as hmv  # noqa: E402
from pose2d import visualize_triangulation as vt  # noqa: E402

FINGERS = {"thumb": range(1, 5), "index": range(5, 9), "middle": range(9, 13), "ring": range(13, 17), "pinky": range(17, 21)}
FINGER_HUE = {"thumb": 0.0, "index": 0.12, "middle": 0.33, "ring": 0.55, "pinky": 0.80}


def rgb(h, s=0.85, v=1.0):
    return tuple(int(255 * c) for c in colorsys.hsv_to_rgb(h, s, v))


def hand_bones(sat):
    bones = []
    for name, ids in FINGERS.items():
        ids = list(ids)
        col = rgb(FINGER_HUE[name], sat)
        bones += [((0, ids[0]), col)] + [((a, b), col) for a, b in zip(ids[:-1], ids[1:])]
    bones += [((a, b), (150, 150, 150)) for a, b in ((5, 9), (9, 13), (13, 17))]
    return bones


BLUE, ORANGE, WHITE = (70, 150, 255), (255, 150, 40), (230, 230, 230)
BODY_BONES = [((0, 1), BLUE), ((1, 3), BLUE), ((0, 2), ORANGE), ((2, 4), ORANGE),
              ((5, 7), BLUE), ((7, 9), BLUE), ((6, 8), ORANGE), ((8, 10), ORANGE),
              ((5, 11), BLUE), ((6, 12), ORANGE), ((5, 6), WHITE), ((11, 12), WHITE)]


def draw(server, prefix, pts, bones, point_color, width=3.0, point_size=0.008, extra=()):
    pts = {int(k): np.asarray(v, np.float32) for k, v in pts.items()}
    if not pts:
        server.scene.add_point_cloud(f"{prefix}/joints", np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8), point_size=point_size)
        server.scene.add_line_segments(f"{prefix}/bones", np.zeros((0, 2, 3), np.float32), np.zeros((0, 2, 3), np.uint8), line_width=width)
        return
    ids = sorted(pts)
    server.scene.add_point_cloud(f"{prefix}/joints", np.stack([pts[i] for i in ids]),
                                 np.full((len(ids), 3), point_color, np.uint8), point_size=point_size)
    segs, cols = [], []
    for (a, b), c in list(bones) + list(extra):
        pa = pts[a] if isinstance(a, int) and a in pts else (a if isinstance(a, np.ndarray) else None)
        pb = pts[b] if isinstance(b, int) and b in pts else (b if isinstance(b, np.ndarray) else None)
        if pa is not None and pb is not None:
            segs.append([pa, pb]); cols.append([c, c])
    if segs:
        server.scene.add_line_segments(f"{prefix}/bones", np.array(segs, np.float32), np.array(cols, np.uint8), line_width=width)
    else:
        server.scene.add_line_segments(f"{prefix}/bones", np.zeros((0, 2, 3), np.float32), np.zeros((0, 2, 3), np.uint8), line_width=width)


def body_extra(p):
    """Neck and spine from midpoints (only when the needed joints exist)."""
    g = lambda i: np.asarray(p[str(i)], np.float32) if str(i) in p else None
    ls, rs, lh, rh, nose = g(5), g(6), g(11), g(12), g(0)
    if ls is None or rs is None:
        return []
    sm = (ls + rs) / 2
    out = []
    if nose is not None:
        out.append(((sm, nose), WHITE))
    if lh is not None and rh is not None:
        out.append(((sm, (lh + rh) / 2), WHITE))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("take")
    ap.add_argument("--calib", required=True)
    ap.add_argument("--ref-calib", default=None)
    ap.add_argument("--pose2d-dir", default="pose2d_sapiens")
    ap.add_argument("--raw", action="store_true")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--fps", type=float, default=30.0)
    a = ap.parse_args()

    calib = calibrate.load_calibration_output(a.calib)
    recon = vt.load_reconstructions(a.take, ["left", "right", "body"], not a.raw, pose2d_dir=a.pose2d_dir)
    frames = sorted({k for r in recon.values() for k in r}, key=lambda k: int(os.path.splitext(k)[0]))

    server = hmv.start_viser_server(port=a.port)
    vt.setup_app_style_scene(server)
    cams = list(calib)
    s = calib[cams[0]]
    hmv.add_camera_frustums(server, {c: (calib[c]["R"], calib[c]["t"]) for c in cams}, s["K"], s["width"], s["height"])
    if a.ref_calib:
        rc = calibrate.load_calibration_output(a.ref_calib)
        rs = rc[list(rc)[0]]
        hmv.add_camera_frustums(server, {f"ref/{c}": (rc[c]["R"], rc[c]["t"]) for c in rc}, rs["K"], rs["width"], rs["height"])

    play = server.gui.add_button("Play / Pause")
    speed = server.gui.add_slider("Playback Speed (FPS)", min=1, max=60, step=1, initial_value=a.fps)
    slider = server.gui.add_slider("Timeline Frame", min=0, max=len(frames) - 1, step=1, initial_value=0)
    label = server.gui.add_text("Frame", initial_value=frames[0], disabled=True)
    st = {"play": True}

    @play.on_click
    def _(_):
        st["play"] = not st["play"]

    left_b, right_b = hand_bones(0.9), hand_bones(0.55)
    i = 0
    print(f"viser running on port {a.port}; {len(frames)} frames", flush=True)
    try:
        while True:
            i = (i + 1) % len(frames) if st["play"] else slider.value
            if st["play"]:
                slider.value = i
            f = frames[i]; label.value = f
            draw(server, "/skeleton/left", recon["left"].get(f, {}), left_b, (120, 220, 255))
            draw(server, "/skeleton/right", recon["right"].get(f, {}), right_b, (255, 200, 120))
            bp = recon["body"].get(f, {})
            draw(server, "/skeleton/body", bp, BODY_BONES, (255, 255, 255), width=4.0, point_size=0.012,
                 extra=body_extra(bp) if bp else ())
            time.sleep(1.0 / speed.value if st["play"] else 0.05)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
