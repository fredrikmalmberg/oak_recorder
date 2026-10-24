"""Viser playback of two SMPL-X fits of the same take (e.g. without / with --sapiens-offsets) plus the Sapiens targets.

Per fit: mesh and the model's own predicted target points (offset-corrected when the fit used --sapiens-offsets), with checkboxes.
Targets are the triangulated Sapiens points the fit was compared against (orange).
  PYTHONPATH=. python -m smplx_fit.view_fit_compare --calib C --a output/cam2hands/smplx_hw --b output/cam2hands/smplx_so --port 8093
"""
import argparse, json, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import calibrate  # noqa: E402
from hand_pose import hand_multiview as hmv  # noqa: E402
from pose2d import visualize_triangulation as vt  # noqa: E402
from smplx_fit import data_loading as dl, model as sm  # noqa: E402

SUB = "aligned/{}/smplx"
COL = {"A": (90, 150, 255), "B": (90, 220, 120)}
TARGET_COL = (255, 150, 40)


def load_fit(take, pose2d_dir, layer, layout, device, stage=None):
    """Mesh vertices and model joints for the final fit, or for the params saved after `stage` (stage_params.npz)."""
    base = os.path.join(take, SUB.format(pose2d_dir))
    d = dict(np.load(os.path.join(base, "smplx_params.npz"), allow_pickle=True))
    cfg = json.load(open(os.path.join(base, "fit_config.json")))
    keys = [int(k[:6]) for k in d["frame_keys"]]
    if stage:
        sp = np.load(os.path.join(base, "stage_params.npz"))
        d.update({k: sp[f"{stage}/{k}"] for k in ("betas", "global_orient", "transl", "body_pose", "lhand_pose", "rhand_pose")})
    t = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)  # noqa: E731
    with torch.no_grad():
        o = sm.forward(layer, t(d["betas"]).reshape(1, -1).expand(len(keys), -1), t(d["global_orient"]), t(d["body_pose"]),
                       t(d["lhand_pose"]), t(d["rhand_pose"]), t(d["transl"]))
        pts = sm.predict_all_points(o, layout)
        if cfg.get("sapiens_offsets"):
            pts = sm.apply_sapiens_offsets(pts, layout)
        if cfg.get("sapiens_leg_offsets"):
            pts = sm.apply_sapiens_leg_offsets(pts, layout)
    return keys, o.vertices.cpu().numpy(), pts.cpu().numpy(), bool(cfg.get("sapiens_offsets")), cfg.get("world", "calib")


STAGES = ["final", "2_global_rt", "3_body_pose", "4_hands_and_body", "5_hands_only"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--a", required=True, help="take dir of fit A")
    ap.add_argument("--b", required=True, help="take dir of fit B")
    ap.add_argument("--pose2d-dir", default="pose2d_sapiens_smoothbox")
    ap.add_argument("--model-path", default="models/SMPLX")
    ap.add_argument("--gender", default="male")
    ap.add_argument("--num-betas", type=int, default=20)
    ap.add_argument("--port", type=int, default=8093)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--floor-map", help="npz with Rg, tg: X_floor = Rg @ X_calib + tg (floor = Z 0). Everything is shown in the floor "
                                        "frame (pass the floor-aligned calibration as --calib) and a floor grid is drawn at Z = 0")
    ap.add_argument("--legs", help="legs.npz from pose2d.triangulate_legs (calibration frame); shown with the Sapiens targets")
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, layout, tp, conf = dl.build_take_arrays(a.a, pose2d_dir=a.pose2d_dir, legs_npz=a.legs)
    layer = sm.load_layer(a.model_path, gender=a.gender, num_betas=a.num_betas).to(device)
    faces = layer.faces.astype(np.uint32)
    # Drop everything below the waist: template vertices under the pelvis/spine1 midpoint height (rest pose, y up).
    tv = layer.v_template.detach().cpu().numpy()
    jt = (layer.J_regressor.detach().cpu().numpy() @ tv)
    y_cut = 0.5 * (jt[0, 1] + jt[3, 1])
    keep = tv[:, 1] >= y_cut
    new_idx = np.cumsum(keep) - 1
    faces_full, keep_full = faces, np.ones(len(tv), bool)
    faces_cut = new_idx[faces[keep[faces].all(1)]].astype(np.uint32)
    faces = faces_cut
    fm = np.load(a.floor_map) if a.floor_map else None

    def to_world(x, world):
        # fits stored in the calibration frame are mapped into the floor frame when --floor-map is given (use the floor calib then)
        return x if (fm is None or world == "floor") else x @ fm["Rg"].T + fm["tg"]
    fits = {}
    for name, take in (("A", a.a), ("B", a.b)):
        keys, verts, pts, off, world = load_fit(take, a.pose2d_dir, layer, layout, device)
        verts, pts = to_world(verts, world), to_world(pts, world)
        fits[name] = dict(keys=keys, verts=verts, pts=pts, off=off, label=f"{name}: {os.path.basename(take.rstrip('/'))}" + (" (+Sapiens offsets)" if off else ""))
    keys = fits["A"]["keys"]
    assert keys == fits["B"]["keys"], "fits cover different frames"
    T = len(keys)
    cache = {}

    def stage_data(n, stage):
        if stage == "final":
            return fits[n]
        if (n, stage) not in cache:
            take = a.a if n == "A" else a.b
            try:
                _, v, p, _, w_ = load_fit(take, a.pose2d_dir, layer, layout, device, stage)
                v, p = to_world(v, w_), to_world(p, w_)
            except (OSError, KeyError):
                return None
            cache[(n, stage)] = dict(verts=v, pts=p)
        return cache[(n, stage)]
    tgt, cf = tp[keys], conf[keys]
    if fm is not None:
        tgt = tgt @ fm["Rg"].T + fm["tg"]
    legs = None
    if False:  # leg targets are already in tgt (layout built with legs)
        L = np.load(a.legs)
        legs = L["X"] @ fm["Rg"].T + fm["tg"] if fm is not None else L["X"]
        legs = legs[[list(L["frames"]).index(k) for k in keys]]

    calib = calibrate.load_calibration_output(a.calib)
    cams = list(calib)
    server = hmv.start_viser_server(port=a.port)
    floor_grid = vt.setup_app_style_scene(server)
    if fm is not None:
        # setup_app_style_scene assumes an unaligned calibration (up = -Y, floor grid in the XZ plane); the floor frame is Z-up
        server.scene.set_up_direction((0.0, 0.0, 1.0))
        floor_grid.visible = False
    s0 = calib[cams[0]]
    hmv.add_camera_frustums(server, {c: (calib[c]["R"], calib[c]["t"]) for c in cams}, s0["K"], s0["width"], s0["height"])

    if fm is not None:
        hips = tgt[:, [11, 12]][cf[:, [11, 12]] > 0]
        c = np.median(hips, 0)
        z_min = min(float(f["verts"][..., 2].min()) for f in fits.values())   # lowest mesh point over all frames and fits
        print(f"floor grid at mesh z min = {z_min:.3f} m")
        server.scene.add_grid("/floor", width=4.0, height=4.0, plane="xy", cell_size=0.25, position=(float(c[0]), float(c[1]), z_min),
                              plane_color=(140, 140, 140), cell_color=(110, 110, 110))

    play = server.gui.add_button("Play / Pause")
    speed = server.gui.add_slider("Playback Speed (FPS)", min=1, max=60, step=1, initial_value=a.fps)
    start = max(0, int(np.argmin(np.abs(np.array(keys) - 300))))
    slider = server.gui.add_slider("Timeline Frame", min=0, max=T - 1, step=1, initial_value=start)
    label = server.gui.add_text("Frame", initial_value=f"{keys[start]:06d}", disabled=True)
    stage_dd = server.gui.add_dropdown("Optimization stage (final = after stage 5)", STAGES, initial_value="final")
    show = {}
    for n, f in fits.items():
        show[(n, "mesh")] = server.gui.add_checkbox(f"{f['label']}: mesh", True)
        show[(n, "pts")] = server.gui.add_checkbox(f"{f['label']}: predicted target points", False)
    show[("T", "pts")] = server.gui.add_checkbox("Sapiens targets", True)
    show[("T", "legs")] = server.gui.add_checkbox("Show lower body (legs)", True)
    st = {"play": False}

    @play.on_click
    def _(_):
        st["play"] = not st["play"]

    handles = {}

    def render(i):
        for n in fits:
            f = stage_data(n, stage_dd.value) or fits[n]
            on = show[(n, "mesh")].value
            low = show[("T", "legs")].value
            if handles.get((n, "low")) != low and (n, "mesh") in handles:
                handles.pop((n, "mesh")).remove()
            handles[(n, "low")] = low
            fcur, kcur = (faces_full, keep_full) if low else (faces_cut, keep)
            if on:
                if (n, "mesh") not in handles:
                    handles[(n, "mesh")] = server.scene.add_mesh_simple(f"/fit{n}/mesh", f["verts"][i][kcur], fcur, color=COL[n],
                                                                       flat_shading=False)
                h = handles[(n, "mesh")]
                h.vertices = f["verts"][i][kcur]
                h.visible = True
            elif (n, "mesh") in handles:
                handles[(n, "mesh")].visible = False
            server.scene.add_point_cloud(f"/fit{n}/pts", f["pts"][i][cf[i] > 0].astype(np.float32),
                                         np.tile(np.array(COL[n], np.uint8), (int((cf[i] > 0).sum()), 1)), point_size=0.012) \
                if show[(n, "pts")].value else server.scene.add_point_cloud(f"/fit{n}/pts", np.zeros((0, 3), np.float32),
                                                                             np.zeros((0, 3), np.uint8), point_size=0.012)
        m = cf[i] > 0
        pts = tgt[i][m].astype(np.float32) if show[("T", "pts")].value else np.zeros((0, 3), np.float32)
        if legs is not None and show[("T", "pts")].value:
            lg = legs[i][np.isfinite(legs[i]).all(1)].astype(np.float32)
            pts = np.concatenate([pts, lg])
        server.scene.add_point_cloud("/targets", pts, np.tile(np.array(TARGET_COL, np.uint8), (len(pts), 1)), point_size=0.014)

    print(f"viser running on port {a.port}", flush=True)
    i, last = start, None
    try:
        while True:
            if st["play"]:
                i = (i + 1) % T; slider.value = i
            else:
                i = slider.value
            k = (i, stage_dd.value, tuple(c.value for c in show.values()))
            if st["play"] or k != last:
                label.value = f"{keys[i]:06d}"; render(i); last = k
            time.sleep(1.0 / speed.value if st["play"] else 0.05)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
