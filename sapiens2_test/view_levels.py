"""Viser playback of a triangulated take where extra Kalman/RTS smoothing is applied ONLY to
shoulders (body ids 5,6) and hips (11,12), at selectable levels. Everything else (other body
joints, hands) shows the pipeline's own smoothed output.

  python -m sapiens2_test.view_levels --calib CAL --recon-dir <dir with reconstruction_*.json> --port 8083
"""
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import calibrate  # noqa: E402
from hand_pose import hand_multiview as hmv  # noqa: E402
from pose2d import visualize_triangulation as vt  # noqa: E402
from sapiens2_test.view_sapiens import BODY_BONES, draw, hand_bones  # noqa: E402

TARGET = ("5", "6", "11", "12")
LEVELS = {  # name -> (process_std, meas_std); None = unfiltered raw triangulation
    "0 raw (no filter)": None,
    "1 pipeline default (q=0.03, m=0.01)": (0.03, 0.01),
    "2 light (q=0.01, m=0.02)": (0.01, 0.02),
    "3 medium (q=0.003, m=0.02)": (0.003, 0.02),
    "4 heavy (q=0.001, m=0.03)": (0.001, 0.03),
}
LIGHT_BLUE = (120, 200, 255)


def key(f):
    return f"{f:06d}.jpg"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", required=True)
    ap.add_argument("--recon-dir", required=True)
    ap.add_argument("--port", type=int, default=8083)
    ap.add_argument("--fps", type=float, default=30.0)
    a = ap.parse_args()

    calib = calibrate.load_calibration_output(a.calib)
    rec = {p: json.load(open(os.path.join(a.recon_dir, f"reconstruction_{p}.json"))) for p in ("left", "right", "body")}
    T = 1 + max(int(os.path.splitext(k)[0]) for p in rec.values() for k in p["raw"])

    # Per level, filtered shoulder/hip trajectories over all frames.
    traj = {}
    for lm in TARGET:
        raw = np.full((T, 3), np.nan)
        for f in range(T):
            p = rec["body"]["raw"].get(key(f), {}).get(lm)
            if p:
                raw[f] = p
        valid = ~np.isnan(raw[:, 0])
        for name, cfg in LEVELS.items():
            if cfg is None:
                out = raw
            else:
                out = np.full((T, 3), np.nan)
                lo, hi = np.argmax(valid), T - np.argmax(valid[::-1])
                sm = hmv.kalman_rts_smooth(np.nan_to_num(raw), valid, np.full(T, cfg[1]), process_std=cfg[0])
                out[lo:hi] = sm[lo:hi]
            traj[(name, lm)] = out

    def body_points(f, level):
        pts = dict(rec["body"]["smoothed"].get(key(f), {}))
        for lm in TARGET:
            v = traj[(level, lm)][f]
            if np.isnan(v[0]):
                pts.pop(lm, None)
            else:
                pts[lm] = v.tolist()
        return pts

    server = hmv.start_viser_server(port=a.port)
    vt.setup_app_style_scene(server)
    cams = list(calib)
    s = calib[cams[0]]
    hmv.add_camera_frustums(server, {c: (calib[c]["R"], calib[c]["t"]) for c in cams}, s["K"], s["width"], s["height"])

    play = server.gui.add_button("Play / Pause")
    speed = server.gui.add_slider("Playback Speed (FPS)", min=1, max=60, step=1, initial_value=a.fps)
    slider = server.gui.add_slider("Timeline Frame", min=0, max=T - 1, step=1, initial_value=300)
    label = server.gui.add_text("Frame", initial_value=key(300), disabled=True)
    level = server.gui.add_dropdown("Hip/shoulder smoothing", tuple(LEVELS), initial_value="3 medium (q=0.003, m=0.02)")
    compare = server.gui.add_dropdown("Compare against (light blue)", ("none",) + tuple(LEVELS), initial_value="0 raw (no filter)")
    trail = server.gui.add_slider("Hip/shoulder trail (frames)", min=0, max=120, step=5, initial_value=30)
    hands = server.gui.add_checkbox("Show hands", True)
    st = {"play": False}

    @play.on_click
    def _(_):
        st["play"] = not st["play"]

    white_hand = [(ab, (255, 255, 255)) for ab, _c in hand_bones(0.0)]
    blue_bones = [(ab, LIGHT_BLUE) for ab, _c in BODY_BONES]

    def render(i):
        f = i
        draw(server, "/main/body", body_points(f, level.value), BODY_BONES, (255, 255, 255), width=4.0, point_size=0.014)
        for part in ("left", "right"):
            pts = rec[part]["smoothed"].get(key(f), {}) if hands.value else {}
            draw(server, f"/main/{part}", pts, white_hand, (200, 200, 200), width=2.0, point_size=0.006)
        if compare.value != "none":
            draw(server, "/cmp/body", body_points(f, compare.value), blue_bones, LIGHT_BLUE, width=2.0, point_size=0.01)
        else:
            draw(server, "/cmp/body", {}, [], LIGHT_BLUE)
        lo = max(0, f - int(trail.value))
        for lm in TARGET:
            seg = traj[(level.value, lm)][lo:f + 1]
            seg = seg[~np.isnan(seg[:, 0])]
            name = f"/trail/{lm}"
            if trail.value > 0 and len(seg) >= 2:
                pairs = np.stack([seg[:-1], seg[1:]], axis=1).astype(np.float32)
                server.scene.add_line_segments(name, pairs, np.full(pairs.shape, (255, 90, 90), np.uint8), line_width=2.0)
            else:
                server.scene.add_line_segments(name, np.zeros((0, 2, 3), np.float32), np.zeros((0, 2, 3), np.uint8), line_width=2.0)

    print(f"viser running on port {a.port}; {T} frames", flush=True)
    i, last = 300, None
    try:
        while True:
            if st["play"]:
                i = (i + 1) % T; slider.value = i
            else:
                i = slider.value
            k = (i, level.value, compare.value, trail.value, hands.value)
            if st["play"] or k != last:
                label.value = key(i); render(i); last = k
            time.sleep(1.0 / speed.value if st["play"] else 0.05)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
