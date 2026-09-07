"""Viser viewer with a dropdown to browse this project's saved calibration
files (output/calibration/*.json, calibrate.build_output's schema) by date
and inspect their camera frustums live. Same "app.py-style" scene setup
(floor grid, FOV slider) as pose2d/visualize_triangulation.py -- ported the
same way, as standalone code, rather than importing calibrate.ViserManager
(a much larger app-oriented class that builds its own ViserServer and a lot
of unrelated live-calibration GUI).

Usage:
    python view_calibration.py [--calib-dir output/calibration] [--port N]
"""
import argparse
import glob
import json
import math
import os
import time

import numpy as np
from viser import transforms as tf

import calibrate
from hand_pose import hand_multiview as hmv

VIEWER_FOV_MIN_DEG = 20
VIEWER_FOV_MAX_DEG = 120
VIEWER_FOV_DEFAULT_DEG = 50

FRUSTUM_COLOR = (80, 160, 255)
FRUSTUM_SCALE = 0.15


def discover_calibrations(calib_dir):
    """Returns [(label, path), ...] sorted oldest-first by metadata.timestamp.
    Skips intrinsics_cache.json (not a calibration-run snapshot -- a
    device-id-keyed intrinsics-only cache, different schema) and anything
    that doesn't parse as build_output's schema (metadata + cameras keys).
    label packs the info this session kept computing by hand: timestamp,
    how many cameras have real extrinsics / converged intrinsics, and the
    chain-consistency error -- so picking a calibration from the dropdown
    doesn't require already knowing which ones are any good.
    """
    paths = sorted(glob.glob(os.path.join(calib_dir, "*.json")))
    entries = []
    for path in paths:
        if os.path.basename(path) == "intrinsics_cache.json":
            continue
        try:
            with open(path) as f:
                d = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if "metadata" not in d or "cameras" not in d:
            continue

        cams = d["cameras"]
        timestamp = d["metadata"].get("timestamp", os.path.basename(path))
        n_ext = sum(1 for c in cams.values() if c.get("extrinsics") is not None)
        n_conv = sum(1 for c in cams.values() if c.get("intrinsics", {}).get("converged"))
        chain_errs = {
            round(c["extrinsics"]["chain_consistency_error_deg"], 2)
            for c in cams.values() if c.get("extrinsics")
        }
        chain_err_str = f"{max(chain_errs):.2f}deg" if chain_errs else "n/a"
        n_cams = len(cams)
        label = f"{timestamp}  ext={n_ext}/{n_cams} conv={n_conv}/{n_cams} chain_err={chain_err_str}"
        entries.append((label, path, d["metadata"].get("timestamp", "")))

    entries.sort(key=lambda e: e[2])
    return [(label, path) for label, path, _ts in entries]


def setup_app_style_scene(server):
    """Ports calibrate.ViserManager.__init__/add_view_controls/the floor-grid
    setup (calibrate.py:1176-1212, 1300-1319) as standalone code -- same
    port as pose2d/visualize_triangulation.setup_app_style_scene, kept
    separate rather than shared since these are two independent small
    scripts, not a growing shared viewer library.
    """
    server.gui.configure_theme(dark_mode=True)
    server.scene.set_background_image(np.zeros((2, 2, 3), dtype=np.uint8))
    server.scene.configure_default_lights(cast_shadow=True)

    with server.gui.add_folder("View"):
        fov_slider = server.gui.add_slider(
            "Viewer FOV (deg)", min=VIEWER_FOV_MIN_DEG, max=VIEWER_FOV_MAX_DEG, step=1,
            initial_value=VIEWER_FOV_DEFAULT_DEG,
        )

        def _apply_fov(fov_deg):
            fov_rad = math.radians(fov_deg)
            for client in server.get_clients().values():
                client.camera.fov = fov_rad

        fov_slider.on_update(lambda _: _apply_fov(fov_slider.value))
        server.on_client_connect(
            lambda client: setattr(client.camera, "fov", math.radians(fov_slider.value))
        )

    server.scene.add_grid(
        "/floor_grid", width=20.0, height=20.0, cell_size=0.5, plane="xy", visible=True,
        plane_color=(220, 220, 220), plane_opacity=0.6, shadow_opacity=0.4,
    )


def rebuild_frustums(server, calib, handles):
    """Removes every existing frustum/axis/label handle and rebuilds fresh
    from `calib` -- simplest correct way to handle switching between
    calibrations with different camera sets (a camera missing extrinsics in
    the newly-selected file must not leave a stale frustum from the
    previous one). Unlike hand_multiview.add_camera_frustums (one shared
    K/width/height for every camera, a known simplification), this uses
    each camera's OWN K/width/height -- calibrate.load_calibration_output
    already returns them per-camera, so there's no reason not to.

    Only the frustum handle is kept/removed -- its /axis and /label
    children (nested under it via viser's "/"-as-hierarchy scene-path
    convention) get cascade-removed along with it; explicitly removing them
    too just throws "already removed" warnings.
    """
    for h in handles.values():
        h.remove()
    handles.clear()

    for cam_id, c in calib.items():
        K, R, t = c["K"], c["R"], c["t"]
        width, height = c["width"], c["height"]
        fov = 2.0 * math.atan(height / (2.0 * K[1, 1]))
        aspect = float(width / height)
        position = (-R.T @ np.asarray(t).reshape(3, 1)).ravel()
        wxyz = tf.SO3.from_matrix(R.T).wxyz

        frustum_h = server.scene.add_camera_frustum(
            f"/cameras/{cam_id}", fov=float(fov), aspect=aspect, scale=FRUSTUM_SCALE,
            wxyz=wxyz, position=position, color=FRUSTUM_COLOR, variant="filled",
        )
        server.scene.add_frame(
            f"/cameras/{cam_id}/axis", axes_length=FRUSTUM_SCALE * 0.5, axes_radius=FRUSTUM_SCALE * 0.03,
            wxyz=wxyz, position=position,
        )
        server.scene.add_label(f"/cameras/{cam_id}/label", cam_id, position=position)
        handles[cam_id] = frustum_h


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--calib-dir", default=os.path.join("output", "calibration"))
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    entries = discover_calibrations(args.calib_dir)
    if not entries:
        print(f"No calibration files found in {args.calib_dir}")
        return

    server = hmv.start_viser_server(port=args.port)
    setup_app_style_scene(server)

    labels = [label for label, _path in entries]
    path_by_label = dict(entries)
    dropdown = server.gui.add_dropdown("Calibration", options=labels, initial_value=labels[-1])
    info_md = server.gui.add_markdown("")
    handles = {}

    def load_selected(label):
        path = path_by_label[label]
        calib = calibrate.load_calibration_output(path)
        rebuild_frustums(server, calib, handles)
        info_md.content = f"**{os.path.basename(path)}**\n\n{len(calib)} camera(s) with real extrinsics shown"
        print(f"Loaded {path}: {len(calib)} camera(s) with real extrinsics")

    dropdown.on_update(lambda _: load_selected(dropdown.value))
    load_selected(dropdown.value)

    print(
        f"Viser server running -- open the printed URL in a browser. "
        f"{len(entries)} calibration(s) available in the dropdown."
    )
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Viewer interrupted.")


if __name__ == "__main__":
    main()
