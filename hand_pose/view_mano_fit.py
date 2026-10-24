"""Viser playback of the MANO fits next to the triangulated hand skeletons they were fitted to.

Per source (Sapiens / WiLoR): triangulated 21-joint skeleton (the fit target) and the fitted MANO mesh, per hand, with checkboxes.
Also draws the camera frustums and optionally the body skeleton of a reconstruction dir for context.
  python -m hand_pose.view_mano_fit --calib C --sap output/manoSap/mano_fit.npz --wil output/manoWil/mano_fit.npz --port 8092
"""
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import calibrate  # noqa: E402
from hand_pose import hand_multiview as hmv  # noqa: E402
from pose2d import visualize_triangulation as vt  # noqa: E402
from sapiens2.view_sapiens import BODY_BONES, draw, hand_bones  # noqa: E402

COL = {"Sapiens": {0: (255, 150, 40), 1: (40, 140, 255)}, "WiLoR": {0: (80, 220, 80), 1: (230, 70, 230)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", required=True)
    ap.add_argument("--sap", required=True)
    ap.add_argument("--wil", required=True)
    ap.add_argument("--body", default="output/bonesA/aligned/pose2d_sapiens_smoothbox", help="dir with reconstruction_body.json for context")
    ap.add_argument("--port", type=int, default=8092)
    ap.add_argument("--fps", type=float, default=30.0)
    a = ap.parse_args()

    D = {"Sapiens": np.load(a.sap), "WiLoR": np.load(a.wil)}
    T = len(D["Sapiens"]["frames"])
    body = json.load(open(os.path.join(a.body, "reconstruction_body.json")))["smoothed"] if a.body else {}
    calib = calibrate.load_calibration_output(a.calib)
    cams = list(calib)
    server = hmv.start_viser_server(port=a.port)
    vt.setup_app_style_scene(server)
    s0 = calib[cams[0]]
    hmv.add_camera_frustums(server, {c: (calib[c]["R"], calib[c]["t"]) for c in cams}, s0["K"], s0["width"], s0["height"])

    play = server.gui.add_button("Play / Pause")
    speed = server.gui.add_slider("Playback Speed (FPS)", min=1, max=60, step=1, initial_value=a.fps)
    slider = server.gui.add_slider("Timeline Frame", min=0, max=T - 1, step=1, initial_value=300)
    label = server.gui.add_text("Frame", initial_value="000300", disabled=True)
    show = {}
    for n in D:
        show[(n, "mesh")] = server.gui.add_checkbox(f"{n}: MANO mesh", True)
        show[(n, "skel")] = server.gui.add_checkbox(f"{n}: triangulated skeleton (target)", True)
        show[(n, "fit")] = server.gui.add_checkbox(f"{n}: MANO joints", False)
    show_body = server.gui.add_checkbox("Body (bonesA)", True)
    st = {"play": False}

    @play.on_click
    def _(_):
        st["play"] = not st["play"]

    hb = [(ab, (255, 255, 255)) for ab, _c in hand_bones(0.0)]
    handles = {}

    def mesh(n, s, f, on):
        d = D[n]
        ok = on and d["present"][f, s]
        name = f"/{n}/mesh_{s}"
        if ok:
            col = COL[n][s]
            if (n, s) not in handles:
                handles[(n, s)] = server.scene.add_mesh_simple(name, d["verts"][f, s].astype(np.float32), d["faces"][s].astype(np.uint32),
                                                              color=col, flat_shading=False, opacity=0.85)
            h = handles[(n, s)]
            h.vertices = d["verts"][f, s].astype(np.float32)
            h.visible = True
        elif (n, s) in handles:
            handles[(n, s)].visible = False

    def pts(J):
        return {str(k): J[k].tolist() for k in range(21)} if np.isfinite(J).all() else {}

    def render(f):
        for n, d in D.items():
            for s in range(2):
                mesh(n, s, f, show[(n, "mesh")].value)
                c = COL[n][s]
                bones = [(ab, c) for ab, _c in hb]
                draw(server, f"/{n}/skel_{s}", pts(d["target"][f, s]) if show[(n, "skel")].value else {}, bones, c, width=2.0, point_size=0.005)
                bones2 = [(ab, (255, 255, 255)) for ab, _c in hb]
                draw(server, f"/{n}/fit_{s}", pts(d["joints"][f, s]) if show[(n, "fit")].value else {}, bones2, (255, 255, 255), width=1.5, point_size=0.004)
        bc = (200, 200, 200)
        draw(server, "/body", body.get(f"{f:06d}.jpg", {}) if show_body.value else {}, [(ab, bc) for ab, _c in BODY_BONES], bc, width=3.0, point_size=0.012)

    print(f"viser running on port {a.port}", flush=True)
    i, last = 300, None
    try:
        while True:
            if st["play"]:
                i = (i + 1) % T; slider.value = i
            else:
                i = slider.value
            k = (i, tuple(c.value for c in show.values()), show_body.value)
            if st["play"] or k != last:
                label.value = f"{i:06d}"; render(i); last = k
            time.sleep(1.0 / speed.value if st["play"] else 0.05)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
