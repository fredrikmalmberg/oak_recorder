"""Viser playback comparing several triangulation reconstructions in one scene.

Each --set is LABEL=DIR where DIR holds reconstruction_{left,right,body}.json; sets get distinct colours
and a checkbox. Hip (body ids 11, 12) trails are drawn per set to make frame-to-frame hip motion visible.
  python -m sapiens2_test.view_compare --calib CAL --set "MediaPipe=<take>/aligned/pose2d" \
      --set "Sapiens=<take>/aligned/pose2d_sapiens_smoothbox" --port 8082
"""
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import calibrate  # noqa: E402
from hand_pose import hand_multiview as hmv  # noqa: E402
from pose2d import visualize_triangulation as vt  # noqa: E402
from sapiens2_test.view_sapiens import BODY_BONES, draw, hand_bones  # noqa: E402

PALETTE = [(255, 235, 0), (120, 200, 255), (255, 120, 220), (120, 255, 140)]


def load(d):
    return {p: json.load(open(os.path.join(d, f"reconstruction_{p}.json"))) for p in ("left", "right", "body")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", required=True)
    ap.add_argument("--set", action="append", required=True, help="LABEL=DIR (repeatable)")
    ap.add_argument("--port", type=int, default=8082)
    ap.add_argument("--fps", type=float, default=30.0)
    a = ap.parse_args()

    calib = calibrate.load_calibration_output(a.calib)
    sets = {}
    for i, spec in enumerate(a.set):
        label, d = spec.split("=", 1)
        sets[label] = {"rec": load(d), "col": PALETTE[i % len(PALETTE)]}
    T = 1 + max(int(os.path.splitext(k)[0]) for s in sets.values() for p in s["rec"].values() for k in p["raw"])
    key = lambda f: f"{f:06d}.jpg"

    server = hmv.start_viser_server(port=a.port)
    vt.setup_app_style_scene(server)
    cams = list(calib)
    s0 = calib[cams[0]]
    hmv.add_camera_frustums(server, {c: (calib[c]["R"], calib[c]["t"]) for c in cams}, s0["K"], s0["width"], s0["height"])

    play = server.gui.add_button("Play / Pause")
    speed = server.gui.add_slider("Playback Speed (FPS)", min=1, max=60, step=1, initial_value=a.fps)
    slider = server.gui.add_slider("Timeline Frame", min=0, max=T - 1, step=1, initial_value=300)
    label = server.gui.add_text("Frame", initial_value=key(300), disabled=True)
    kind = server.gui.add_dropdown("Reconstruction", ("raw", "smoothed"), initial_value="raw")
    trail = server.gui.add_slider("Hip trail (frames)", min=0, max=120, step=5, initial_value=30)
    hands = server.gui.add_checkbox("Show hands", True)
    show = {n: server.gui.add_checkbox(f"{n}", True) for n in sets}
    st = {"play": False}

    @play.on_click
    def _(_):
        st["play"] = not st["play"]

    hb = [(ab, (255, 255, 255)) for ab, _c in hand_bones(0.0)]
    empty = (np.zeros((0, 2, 3), np.float32), np.zeros((0, 2, 3), np.uint8))

    def hip_traj(rec, lm, lo, hi):
        pts = [rec["body"][kind.value].get(key(f), {}).get(lm) for f in range(lo, hi + 1)]
        return np.array([p for p in pts if p], np.float32)

    def render(f):
        for n, sd in sets.items():
            on, col = show[n].value, sd["col"]
            for part in ("left", "right", "body"):
                pts = sd["rec"][part][kind.value].get(key(f), {}) if on and (part == "body" or hands.value) else {}
                if part == "body":
                    draw(server, f"/{n}/body", pts, [(ab, col) for ab, _c in BODY_BONES], col, width=4.0, point_size=0.014)
                else:
                    draw(server, f"/{n}/{part}", pts, [(ab, col) for ab, _c in hb], col, width=2.0, point_size=0.006)
            for lm in ("11", "12"):
                tr = hip_traj(sd["rec"], lm, max(0, f - int(trail.value)), f) if on and trail.value > 0 else np.zeros((0, 3), np.float32)
                name = f"/{n}/trail{lm}"
                if len(tr) >= 2:
                    pairs = np.stack([tr[:-1], tr[1:]], axis=1)
                    server.scene.add_line_segments(name, pairs, np.full(pairs.shape, col, np.uint8), line_width=2.0)
                else:
                    server.scene.add_line_segments(name, *empty, line_width=2.0)

    print(f"viser running on port {a.port}; {T} frames; sets: {list(sets)}", flush=True)
    i, last = 300, None
    try:
        while True:
            if st["play"]:
                i = (i + 1) % T; slider.value = i
            else:
                i = slider.value
            k = (i, kind.value, trail.value, hands.value, tuple(c.value for c in show.values()))
            if st["play"] or k != last:
                label.value = key(i); render(i); last = k
            time.sleep(1.0 / speed.value if st["play"] else 0.05)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
