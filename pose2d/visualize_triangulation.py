"""Playback for pose2d.triangulation's output -- a viser 3D viewer showing
left hand, right hand, and body TOGETHER in one scene (unlike hand_pose/
visualize_triangulation.py's run_skeleton_player, which only ever plays one
skeleton at a time -- pose2d triangulates all three parts independently, so
this drives three simultaneous hand_multiview.add_skeleton_frame calls per
tick instead of reusing that single-skeleton player as-is).

Matches app.py's own viewer setup (calibrate.ViserManager, not imported --
that class is a large app-oriented GUI object with a live ViserServer of
its own; the relevant bits are ported here as standalone code, same
convention as h5_hand_extraction.py vs. pose2d/extraction.py elsewhere in
this project): dark theme, solid background, shadowed default lights, the
same "Viewer FOV (deg)" slider (20-120deg, default 50), and the same floor
grid (xy plane, since app.py's world-alignment convention puts +Z up;
20x20m, 0.5m cells). The floor is visible by default here (unlike app.py,
which starts it hidden until live world alignment succeeds) since a
reconstructed take's calibration is already whatever frame it was solved
in -- there's no "alignment pending" state to gate on post-hoc. Whether
that frame is actually Z-up (i.e. whether "Set down direction from board"
was run during the calibration session used) isn't recorded in the saved
calibration JSON -- if the floor looks tilted/wrong, that's what to check.

Usage:
    python -m pose2d.visualize_triangulation <take_dir> [--calib PATH] [--part left,right,body] [--raw] [--port N]
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import calibrate  # noqa: E402

from hand_pose import hand_multiview as hmv  # noqa: E402
from pose2d import triangulation  # noqa: E402
from smplx_fit import model as smplx_model  # noqa: E402

VIEWER_FOV_MIN_DEG = 20
VIEWER_FOV_MAX_DEG = 120
VIEWER_FOV_DEFAULT_DEG = 50

PART_STYLE = {
    "left": {"connections": hmv.HAND_CONNECTIONS, "point_color": (0, 200, 255), "bone_color": (0, 160, 220)},
    "right": {"connections": hmv.HAND_CONNECTIONS, "point_color": (255, 160, 0), "bone_color": (220, 120, 0)},
    "body": {"connections": hmv.BODY_CONNECTIONS_COCO_UPPER, "point_color": (255, 255, 255), "bone_color": (200, 200, 200)},
}


def load_reconstructions(take_dir, parts, use_smoothed, pose2d_dir="pose2d"):
    out = {}
    for part in parts:
        path = os.path.join(take_dir, "aligned", pose2d_dir, f"reconstruction_{part}.json")
        with open(path) as f:
            recon = json.load(f)
        out[part] = recon["smoothed"] if use_smoothed else recon["raw"]
    return out


def load_smplx_mesh_sequence(npz_path, model_path, hide_lower_body=False, hide_pelvis=False):
    """Loads a smplx_fit.fit_take output .npz and returns (vertices_by_frame,
    faces) -- vertices_by_frame maps EXACTLY the frame_key strings the fit
    covered (often a subset of the take, e.g. a --frames slice) to that
    frame's (V, 3) vertex array; faces is the constant (F, 3) face-index
    array shared by every frame.

    One batched SMPLXLayer forward pass over every fitted frame at load
    time, not per-tick during playback -- on CPU-only torch, one batched
    call is far cheaper than replaying hundreds of individual forward
    passes while scrubbing/playing.

    hide_lower_body: drop legs/feet faces (smplx_model.upper_body_faces) --
    useful since this pipeline supplies zero leg keypoints (see optimize.py's
    known_limitations), so a fit's legs are held near-neutral by the pose
    prior/silhouette term alone and can visually mislead if shown as if
    they were as trustworthy as the rest of the mesh.
    """
    data = np.load(npz_path, allow_pickle=True)
    frame_keys = [str(fk) for fk in data["frame_keys"]]
    gender = str(data["gender"])
    num_betas = int(data["num_betas"])
    model = smplx_model.load_layer(model_path, gender=gender, num_betas=num_betas)

    nf = len(frame_keys)
    betas = torch.as_tensor(data["betas"], dtype=torch.float32).expand(nf, -1)
    global_orient = torch.as_tensor(data["global_orient"], dtype=torch.float32)
    transl = torch.as_tensor(data["transl"], dtype=torch.float32)
    body_pose = torch.as_tensor(data["body_pose"], dtype=torch.float32)
    lhand_pose = torch.as_tensor(data["lhand_pose"], dtype=torch.float32)
    rhand_pose = torch.as_tensor(data["rhand_pose"], dtype=torch.float32)
    with torch.no_grad():
        output = smplx_model.forward(model, betas, global_orient, body_pose, lhand_pose, rhand_pose, transl)

    vertices = output.vertices.numpy()
    vertices_by_frame = {fk: vertices[i] for i, fk in enumerate(frame_keys)}
    if hide_lower_body:
        faces = smplx_model.upper_body_faces(model, hide_pelvis=hide_pelvis)
    else:
        faces = model.faces
    return vertices_by_frame, faces


def run_multi_skeleton_player(server, reconstructions, fps=15.0, smplx_mesh=None, frame_range=None):
    """reconstructions: dict[part_name] -> reconstruction dict (frame_key ->
    {lm_id: [x,y,z]}). All parts share one timeline -- the union of frame
    keys across parts, since each part's own RANSAC selection can succeed on
    a different subset of frames; a part with no data for the current frame
    just renders empty (add_skeleton_frame's own empty-point-cloud path).

    smplx_mesh: optional (vertices_by_frame, faces) from
    load_smplx_mesh_sequence -- both the skeleton and the mesh already live
    in the same calibration/world frame by construction (both come from the
    SAME triangulated keypoints), so no coordinate transform is needed
    between them.

    frame_range: optional (start, end) slice (Python slice semantics, same
    convention as smplx_fit.fit_take's --frames) applied to the FULL take's
    timeline -- e.g. to focus playback on exactly the slice a --smplx-npz
    fit covers, instead of looping the whole take with the mesh only
    visible for a brief window in the middle.
    """
    frame_keys = sorted(
        {fk for recon in reconstructions.values() for fk in recon},
        key=lambda k: int(os.path.splitext(k)[0]),
    )
    if frame_range is not None:
        start, end = frame_range
        frame_keys = frame_keys[start:end]
    if not frame_keys:
        print("No reconstructed frames to play.")
        return

    play_button = server.gui.add_button("Play / Pause")
    stop_button = server.gui.add_button("Stop viewer")
    speed_slider = server.gui.add_slider("Playback Speed (FPS)", min=1, max=60, step=1, initial_value=fps)
    frame_slider = server.gui.add_slider("Timeline Frame", min=0, max=len(frame_keys) - 1, step=1, initial_value=0)

    mesh_handle = None
    show_mesh_checkbox = None
    vertices_by_frame, faces = ({}, None) if smplx_mesh is None else smplx_mesh
    if smplx_mesh is not None:
        show_mesh_checkbox = server.gui.add_checkbox("Show SMPL-X mesh", initial_value=True)
        first_verts = next(iter(vertices_by_frame.values()))
        # Created ONCE here; mutated in place below (.vertices/.visible) --
        # the same persisted-handle-mutation pattern calibrate.py's
        # ViserManager._update_board_visual uses, meaningfully cheaper than
        # re-uploading a ~20k-face mesh's full vertex+face buffer every tick.
        mesh_handle = server.scene.add_mesh_simple(
            "/smplx_mesh", vertices=first_verts, faces=faces, color=(180, 180, 220),
        )

    state = {"playing": False, "running": True}

    @play_button.on_click
    def _(_):
        state["playing"] = not state["playing"]

    @stop_button.on_click
    def _(_):
        state["running"] = False

    current_idx = 0
    try:
        while state["running"]:
            if not state["playing"]:
                current_idx = frame_slider.value
            frame_key = frame_keys[current_idx]
            for part_name, recon in reconstructions.items():
                style = PART_STYLE[part_name]
                hmv.add_skeleton_frame(
                    server, recon.get(frame_key, {}), name_prefix=f"/skeleton/{part_name}",
                    connections=style["connections"], point_color=style["point_color"], bone_color=style["bone_color"],
                )
            if mesh_handle is not None:
                verts = vertices_by_frame.get(frame_key)
                if verts is not None and show_mesh_checkbox.value:
                    mesh_handle.vertices = verts
                    mesh_handle.visible = True
                else:
                    mesh_handle.visible = False
            if state["playing"]:
                current_idx = (current_idx + 1) % len(frame_keys)
                frame_slider.value = current_idx
                time.sleep(1.0 / speed_slider.value)
            else:
                time.sleep(0.05)
    except KeyboardInterrupt:
        print("Viewer interrupted.")


def setup_app_style_scene(server):
    """Ports calibrate.ViserManager.__init__/add_view_controls/the floor-grid
    setup (calibrate.py:1176-1212, 1300-1319) as standalone code, so this
    viewer looks and controls the same as app.py's live 3D view. The FOV
    slider applies to every currently-connected client and any client that
    connects later, same broadcast-plus-on_client_connect pattern as the
    original. Returns the floor grid handle (already visible=True here --
    see module docstring for why that differs from app.py's default).
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

    floor_grid = server.scene.add_grid(
        "/floor_grid", width=20.0, height=20.0, cell_size=0.5, plane="xy", visible=True,
        plane_color=(220, 220, 220), plane_opacity=0.6, shadow_opacity=0.4,
    )
    return floor_grid


def launch_3d_viewer(
    take_dir, calib_path, parts, port, fps, use_smoothed, pose2d_dir="pose2d",
    smplx_npz=None, smplx_model_path=None, frame_range=None, smplx_hide_lower_body=False,
    smplx_hide_hips=False,
):
    calib = calibrate.load_calibration_output(calib_path)
    reconstructions = load_reconstructions(take_dir, parts, use_smoothed, pose2d_dir=pose2d_dir)

    server = hmv.start_viser_server(port=port)
    setup_app_style_scene(server)
    cam_ids = list(calib.keys())
    if cam_ids:
        poses = {cam_id: (calib[cam_id]["R"], calib[cam_id]["t"]) for cam_id in cam_ids}
        sample = calib[cam_ids[0]]
        hmv.add_camera_frustums(server, poses, sample["K"], sample["width"], sample["height"])

    smplx_mesh = None
    if smplx_npz is not None:
        print(f"Loading SMPL-X fit from {smplx_npz}...")
        smplx_mesh = load_smplx_mesh_sequence(smplx_npz, smplx_model_path,
                                               hide_lower_body=smplx_hide_lower_body,
                                               hide_pelvis=smplx_hide_hips)
        print(f"SMPL-X mesh covers {len(smplx_mesh[0])} frame(s).")

    frame_counts = ", ".join(f"{p}={len(reconstructions[p])}" for p in parts)
    print(
        f"Viser server running -- open the printed URL in a browser to view. "
        f"parts: {frame_counts} ({'smoothed' if use_smoothed else 'raw'})."
    )
    run_multi_skeleton_player(server, reconstructions, fps=fps, smplx_mesh=smplx_mesh, frame_range=frame_range)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("take_dir")
    parser.add_argument("--calib", default=triangulation.DEFAULT_CALIB_PATH)
    parser.add_argument("--part", default="left,right,body", help="Comma-separated subset of left,right,body")
    parser.add_argument("--raw", action="store_true", help="Play the raw (unsmoothed) reconstruction")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument(
        "--pose2d-dir", default="pose2d",
        help="Which detector's triangulation to view: 'pose2d' (MediaPipe, default) or 'pose2d_dwpose' (DWPose).",
    )
    parser.add_argument(
        "--smplx-npz", default=None,
        help="Optional smplx_fit.fit_take output (smplx_params.npz) to render as a mesh alongside the skeleton.",
    )
    parser.add_argument(
        "--smplx-model-path", default="models/SMPLX",
        help="SMPL-X model directory (only used if --smplx-npz is given).",
    )
    parser.add_argument(
        "--frames", default=None,
        help="START:END (Python slice semantics, same convention as smplx_fit.fit_take's --frames) to "
             "restrict playback to a sub-range of the take -- e.g. to focus on exactly the slice a "
             "--smplx-npz fit covers instead of looping the whole take. Default: whole take.",
    )
    parser.add_argument(
        "--smplx-hide-lower-body", action="store_true",
        help="Drop the SMPL-X mesh's legs/feet faces -- this pipeline supplies zero leg keypoints, "
             "so a fit's legs are held near-neutral by the pose prior/silhouette term alone, not "
             "real data; hiding them avoids visually overclaiming their accuracy.",
    )
    parser.add_argument(
        "--smplx-hide-hips", action="store_true",
        help="Also drop the pelvis/hip region (glutes, lower abdomen). Requires --smplx-hide-lower-body.",
    )
    args = parser.parse_args()
    parts = [p.strip() for p in args.part.split(",") if p.strip()]
    frame_range = None
    if args.frames is not None:
        start_s, end_s = args.frames.split(":")
        frame_range = (int(start_s) if start_s else 0, int(end_s) if end_s else None)
    launch_3d_viewer(
        args.take_dir, args.calib, parts, args.port, args.fps, use_smoothed=not args.raw,
        pose2d_dir=args.pose2d_dir, smplx_npz=args.smplx_npz, smplx_model_path=args.smplx_model_path,
        frame_range=frame_range, smplx_hide_lower_body=args.smplx_hide_lower_body,
        smplx_hide_hips=args.smplx_hide_hips,
    )


if __name__ == "__main__":
    main()
