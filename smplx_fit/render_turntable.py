"""Render a textured SMPL-X mesh as a turntable video using nvdiffrast.

Camera sweeps from the leftmost to the rightmost of a set of bake cameras
(default: cam2 → cam0 → cam3) over the full take's pose sequence.
World 'up' is derived from the calibrated camera orientations.

Usage:
    python -m smplx_fit.render_turntable \\
        --smplx-npz  /path/to/smplx_params.npz \\
        --calib      output/calibration/20260908_174749_7cam.json \\
        --texture    /path/to/texture_baked.png \\
        --output     /path/to/turntable.mp4

Key design notes:
  - Per-loop vertex inflation: UV vertex i*3+j maps to 3D vertex faces[i,j],
    so rasterizing with the inflated mesh + UV coords gives correct texture
    sampling without needing per-vertex unique UVs.
  - World up derived from calibration: average of -R[1] across reference
    cameras (R[1] = world direction that maps to image 'down').
  - GPU memory: rast_db/uv_db gradient buffers are deleted each frame;
    torch.cuda.synchronize() + empty_cache() prevents VRAM accumulation.
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import calibrate  # noqa: E402

from smplx_fit import model as smplx_model  # noqa: E402
from smplx_fit.texture_bake import load_uv_data  # noqa: E402
from smplx_fit.silhouette import _get_nvdiffrast_ctx  # noqa: E402

UV_NPZ_DEFAULT   = "/home/fmalmb/CODE/sl_reconstruction/visualization/textures/smplx_uv_2023.npz"
CALIB_DEFAULT    = "output/calibration/20260908_174749_7cam.json"

# SMPL-X body joint indices
J_LEFT_SHOULDER  = 16
J_RIGHT_SHOULDER = 17


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------

def world_up_from_calib(calib, cam_ids):
    """Derive world 'up' direction from calibrated cameras.
    R[1] = world direction that maps to image 'down'; average -R[1] → world up.
    """
    downs = [np.array(calib[cid]["R"], dtype=np.float64)[1] for cid in cam_ids]
    world_up = -np.mean(downs, axis=0)
    return world_up / np.linalg.norm(world_up)


def look_at(cam_pos, target, world_up):
    """Build R, t (OpenCV: Z forward, Y down) from world-space position and target."""
    fwd = target - cam_pos
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, world_up)
    norm = np.linalg.norm(right)
    if norm < 1e-6:
        right = np.cross(fwd, np.array([1.0, 0.0, 0.0]))
        norm = np.linalg.norm(right)
    right /= norm
    down = np.cross(right, fwd)
    R = np.stack([right, down, fwd], axis=0).astype(np.float64)
    return R, -R @ cam_pos


def verts_to_clip(verts_t, K, R, t, W, H):
    """Project (V,3) world verts to nvdiffrast clip space (V,4)."""
    R_t = torch.as_tensor(R, dtype=torch.float32, device="cuda")
    t_t = torch.as_tensor(t, dtype=torch.float32, device="cuda")
    K_t = torch.as_tensor(K, dtype=torch.float32, device="cuda")
    p   = (R_t @ verts_t.T).T + t_t
    q   = (K_t @ p.T).T
    z   = q[:, 2:3].clamp(min=1e-4)
    px  = q[:, 0:1] / z
    py  = q[:, 1:2] / z
    x_n = (px / W) * 2.0 - 1.0
    y_n = 1.0 - (py / H) * 2.0      # flip Y: image top → NDC +1
    z_n = (z - 0.01) / (50.0 - 0.01)
    w   = torch.ones(len(verts_t), 1, device="cuda")
    return torch.cat([x_n, y_n, z_n, w], dim=1)


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

def render_turntable(
    smplx_npz,
    calib_path,
    texture_path,
    output_path,
    uv_npz=UV_NPZ_DEFAULT,
    model_path="models/SMPLX",
    sweep_cams=None,        # [left_cam, front_cam, right_cam] in sweep order
    render_w=1080,
    render_h=1080,
    fps=30,
    bg_color=(240, 240, 240),
):
    import nvdiffrast.torch as dr

    print("Loading calibration...")
    calib = calibrate.load_calibration_output(calib_path)

    if sweep_cams is None:
        sweep_cams = ["cam2", "cam0", "cam3"]

    world_up = world_up_from_calib(calib, sweep_cams)
    print(f"World up (from cameras): {world_up.round(3)}")

    # --- SMPL-X fit ---
    print("Loading SMPL-X fit...")
    data      = np.load(smplx_npz, allow_pickle=True)
    n_frames  = len(data["frame_keys"])
    model_layer = smplx_model.load_layer(
        model_path, gender=str(data["gender"]), num_betas=int(data["num_betas"]))

    betas = torch.as_tensor(data["betas"], dtype=torch.float32).reshape(1, -1).expand(n_frames, -1)
    go    = torch.as_tensor(data["global_orient"], dtype=torch.float32)
    bp    = torch.as_tensor(data["body_pose"],     dtype=torch.float32)
    lh    = torch.as_tensor(data["lhand_pose"],    dtype=torch.float32)
    rh    = torch.as_tensor(data["rhand_pose"],    dtype=torch.float32)
    tr    = torch.as_tensor(data["transl"],        dtype=torch.float32)

    print(f"Running SMPL-X forward on {n_frames} frames...")
    BATCH      = 64
    verts_all  = []
    joints_all = []
    with torch.no_grad():
        for i in range(0, n_frames, BATCH):
            out = smplx_model.forward(model_layer,
                                       betas[i:i+BATCH], go[i:i+BATCH], bp[i:i+BATCH],
                                       lh[i:i+BATCH], rh[i:i+BATCH], tr[i:i+BATCH])
            verts_all.append(out.vertices.cpu())
            joints_all.append(out.joints.cpu())
    verts_all  = torch.cat(verts_all,  dim=0).numpy()   # (N, V, 3)
    joints_all = torch.cat(joints_all, dim=0).numpy()   # (N, J, 3)
    faces_np   = model_layer.faces

    # --- UV data ---
    print("Loading UV data...")
    vt, _, _ = load_uv_data(uv_npz)
    ft_loop   = np.arange(len(vt), dtype=np.int32).reshape(-1, 3)
    vt_loop_t = torch.as_tensor(vt,     dtype=torch.float32, device="cuda")
    ft_loop_t = torch.as_tensor(ft_loop, dtype=torch.int32,  device="cuda")

    # --- Texture ---
    print(f"Loading texture: {texture_path}")
    tex_bgr  = cv2.imread(texture_path)
    tex_rgb  = tex_bgr[:, :, ::-1].astype(np.float32) / 255.0
    tex_t    = torch.as_tensor(tex_rgb[::-1].copy(),   # V=0 at bottom for dr.texture
                                dtype=torch.float32, device="cuda").unsqueeze(0)

    # --- Camera path: sweep_cams[0] → sweep_cams[1] → sweep_cams[2] ---
    print("Building camera path...")
    cam_pos = {}
    for cid in sweep_cams:
        R_c = np.array(calib[cid]["R"], dtype=np.float64)
        t_c = np.array(calib[cid]["t"], dtype=np.float64)
        cam_pos[cid] = -R_c.T @ t_c

    mid            = n_frames // 2
    path_positions = np.zeros((n_frames, 3))
    for fi in range(n_frames):
        if fi <= mid:
            a = fi / mid
            path_positions[fi] = (1 - a) * cam_pos[sweep_cams[0]] + a * cam_pos[sweep_cams[1]]
        else:
            a = (fi - mid) / (n_frames - 1 - mid)
            path_positions[fi] = (1 - a) * cam_pos[sweep_cams[1]] + a * cam_pos[sweep_cams[2]]

    # Fixed look-at: midpoint between shoulders, averaged over all frames
    target = joints_all[:, [J_LEFT_SHOULDER, J_RIGHT_SHOULDER], :].mean(axis=(0, 1))
    print(f"Look-at target (avg shoulder midpoint): {target.round(3)}")

    # Render K: use cam0 fy, uniform scale to render height, square pixels
    K_src     = np.array(calib[sweep_cams[1]]["K"], dtype=np.float64)
    fy_scaled = K_src[1, 1] * (render_h / float(calib[sweep_cams[1]]["height"]))
    K_render  = np.array([[fy_scaled, 0, render_w / 2.0],
                           [0, fy_scaled, render_h / 2.0],
                           [0, 0, 1.0]], dtype=np.float64)
    print(f"Render K: fx=fy={fy_scaled:.1f}")

    # --- Render loop ---
    glctx = _get_nvdiffrast_ctx()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (render_w, render_h))

    faces_flat     = faces_np.reshape(-1)
    verts_loop_buf = torch.empty(len(faces_flat), 3, dtype=torch.float32, device="cuda")
    bg_np          = np.array(bg_color, dtype=np.uint8)
    bg_buf         = np.full((render_h, render_w, 3), bg_np, dtype=np.uint8)

    print(f"Rendering {n_frames} frames → {output_path}")
    for fi in range(n_frames):
        verts_loop_buf.copy_(torch.as_tensor(verts_all[fi][faces_flat], dtype=torch.float32))
        R_v, t_v = look_at(path_positions[fi], target, world_up)

        with torch.no_grad():
            clips = verts_to_clip(verts_loop_buf, K_render, R_v, t_v,
                                  render_w, render_h).unsqueeze(0)
            rast, rast_db    = dr.rasterize(glctx, clips.contiguous(),
                                             ft_loop_t.contiguous(),
                                             resolution=[render_h, render_w])
            del rast_db
            uv_interp, uv_db = dr.interpolate(vt_loop_t[None], rast, ft_loop_t)
            del uv_db
            color_t = dr.texture(tex_t, uv_interp, filter_mode="linear")
            color_t = dr.antialias(color_t, rast, clips, ft_loop_t)

            color_np = (color_t[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            mask_np  = (rast[0, :, :, 3] > 0).cpu().numpy()
            del clips, rast, uv_interp, color_t

        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        bg_buf[:] = bg_np
        bg_buf[mask_np] = color_np[:, :, ::-1][mask_np]
        writer.write(bg_buf)
        del color_np, mask_np

        if (fi + 1) % 100 == 0 or fi == n_frames - 1:
            print(f"  {fi + 1}/{n_frames} frames rendered", flush=True)

    writer.release()
    print(f"Saved → {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smplx-npz",    required=True, help="smplx_params.npz from fit_take")
    parser.add_argument("--calib",        default=CALIB_DEFAULT)
    parser.add_argument("--texture",      required=True, help="Baked texture PNG")
    parser.add_argument("--output",       default=None,  help="Output .mp4 path")
    parser.add_argument("--uv-npz",       default=UV_NPZ_DEFAULT)
    parser.add_argument("--smplx-model-path", default="models/SMPLX")
    parser.add_argument("--sweep-cams",   default="cam2,cam0,cam3",
                        help="Comma-separated camera IDs: left,front,right sweep order")
    parser.add_argument("--render-size",  type=int, default=1080,
                        help="Square render resolution (default 1080)")
    parser.add_argument("--fps",          type=int, default=30)
    args = parser.parse_args()

    output = args.output or args.smplx_npz.replace("smplx_params.npz", "turntable.mp4")
    sweep_cams = [c.strip() for c in args.sweep_cams.split(",")]

    render_turntable(
        smplx_npz=args.smplx_npz,
        calib_path=args.calib,
        texture_path=args.texture,
        output_path=output,
        uv_npz=args.uv_npz,
        model_path=args.smplx_model_path,
        sweep_cams=sweep_cams,
        render_w=args.render_size,
        render_h=args.render_size,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
