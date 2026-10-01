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
from smplx_fit.texture.texture_bake import load_uv_data  # noqa: E402
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

# COCO-17 skeleton connections for body landmarks 0-12
BODY_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),          # face
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # arms
    (5, 11), (6, 12), (11, 12),               # torso
]


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
    flat_color=None,        # (R,G,B) 0-255 — if set, skip texture and use Lambertian shading
    mesh_alpha=1.0,         # mesh opacity (0=transparent, 1=opaque)
    skeleton_path=None,     # path to reconstruction_body.json; None = no skeleton
):
    import nvdiffrast.torch as dr

    print("Loading calibration...")
    calib = calibrate.load_calibration_output(calib_path)

    if sweep_cams is None:
        sweep_cams = ["cam2", "cam0", "cam3"]

    world_up = world_up_from_calib(calib, sweep_cams)
    print(f"World up (from cameras): {world_up.round(3)}")

    # --- Skeleton data (optional) ---
    skel_by_key = {}     # body: {frame_key: {landmark_id: (x,y,z)}}
    hand_by_key = {}     # hands: {frame_key: [(x,y,z), ...]}  flat list of all hand pts
    if skeleton_path is not None:
        import json
        print(f"Loading skeleton from {skeleton_path}...")
        with open(skeleton_path) as f:
            skel_data = json.load(f)
        # Use smoothed — matches what fit_take uses as target
        skel_src = skel_data.get("smoothed", skel_data.get("raw"))
        for fk, lms in skel_src.items():
            skel_by_key[fk] = {int(k): np.array(v, dtype=np.float32) for k, v in lms.items()}
        # Also load hand reconstructions (smoothed) if present alongside body
        skel_dir = os.path.dirname(skeleton_path)
        for hand_file in ("reconstruction_left.json", "reconstruction_right.json"):
            hp = os.path.join(skel_dir, hand_file)
            if os.path.exists(hp):
                with open(hp) as f:
                    hr = json.load(f)
                hand_src = hr.get("smoothed", hr.get("raw"))
                for fk, lms in hand_src.items():
                    pts = [np.array(v, dtype=np.float32) for v in lms.values()]
                    hand_by_key.setdefault(fk, []).extend(pts)
        print(f"  body frames: {len(skel_by_key)}, hand points loaded: {sum(len(v) for v in hand_by_key.values())}")

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

    # Load FLAME face parameters if present; fall back to zeros
    expr_np = data["expression"] if "expression" in data else np.zeros((n_frames, 10), np.float32)
    jaw_np  = data["jaw_pose"]   if "jaw_pose"   in data else np.zeros((n_frames, 3),  np.float32)
    expr = torch.as_tensor(expr_np, dtype=torch.float32)
    jaw  = torch.as_tensor(jaw_np,  dtype=torch.float32)
    has_expr = np.abs(expr_np).max() > 1e-6
    print(f"  expression/jaw_pose: {'fitted' if has_expr else 'zeros (not fitted)'}")

    print(f"Running SMPL-X forward on {n_frames} frames...")
    BATCH      = 64
    verts_all  = []
    joints_all = []
    with torch.no_grad():
        for i in range(0, n_frames, BATCH):
            out = smplx_model.forward_with_expr(
                model_layer,
                betas[i:i+BATCH], go[i:i+BATCH], bp[i:i+BATCH],
                lh[i:i+BATCH], rh[i:i+BATCH], tr[i:i+BATCH],
                expr[i:i+BATCH], jaw[i:i+BATCH])
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

    # --- Texture OR flat color ---
    if flat_color is not None:
        tex_t = None
        base_rgb = torch.tensor([flat_color[0] / 255.0,
                                  flat_color[1] / 255.0,
                                  flat_color[2] / 255.0], dtype=torch.float32, device="cuda")
        print(f"Flat color mode: RGB {flat_color}")
    else:
        print(f"Loading texture: {texture_path}")
        tex_bgr  = cv2.imread(texture_path)
        tex_rgb  = tex_bgr[:, :, ::-1].astype(np.float32) / 255.0
        tex_t    = torch.as_tensor(tex_rgb[::-1].copy(),   # V=0 at bottom for dr.texture
                                    dtype=torch.float32, device="cuda").unsqueeze(0)
        base_rgb = None

    faces_t = torch.as_tensor(faces_np.astype(np.int64), dtype=torch.long, device="cuda")

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
        verts_world = torch.as_tensor(verts_all[fi], dtype=torch.float32, device="cuda")
        verts_loop_buf.copy_(verts_world[faces_t.reshape(-1)])
        R_v, t_v = look_at(path_positions[fi], target, world_up)

        with torch.no_grad():
            clips = verts_to_clip(verts_loop_buf, K_render, R_v, t_v,
                                  render_w, render_h).unsqueeze(0)
            rast, rast_db    = dr.rasterize(glctx, clips.contiguous(),
                                             ft_loop_t.contiguous(),
                                             resolution=[render_h, render_w])
            del rast_db

            if flat_color is not None:
                # Lambertian shading with two lights: key (front-above-cam) + fill (ambient)
                # Per-vertex normals on the original mesh (not loop-inflated)
                nv = verts_world.shape[0]
                v0 = verts_world[faces_t[:, 0]]
                v1 = verts_world[faces_t[:, 1]]
                v2 = verts_world[faces_t[:, 2]]
                face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)  # (F, 3)
                area = face_normals.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                face_normals_unit = face_normals / area                 # unit per-face normal

                # Scatter: each face contributes its normal to its three vertices
                vn = torch.zeros(nv, 3, device="cuda")
                idx = faces_t.reshape(-1, 1).expand(-1, 3)             # (F*3, 3)
                fn3 = face_normals_unit.repeat_interleave(3, dim=0)    # (F*3, 3)
                vn.scatter_add_(0, idx, fn3)
                vn = torch.nn.functional.normalize(vn, dim=-1)         # (V, 3)

                # Transform normals to camera space
                R_t = torch.as_tensor(R_v, dtype=torch.float32, device="cuda")
                normals_cam = vn @ R_t.T  # (V, 3)

                # Loop-inflate normals to match ft_loop_t layout, then interpolate
                faces_flat_t = faces_t.reshape(-1)
                normals_loop = normals_cam[faces_flat_t]               # (F*3, 3)
                norm_interp, _ = dr.interpolate(normals_loop[None], rast, ft_loop_t)
                norm_interp = torch.nn.functional.normalize(norm_interp, dim=-1)  # (1,H,W,3)

                # Key light: slightly above and in front of the camera (camera-space direction)
                key_dir = torch.tensor([0.3, -0.5, -0.8], device="cuda")  # cam space: -z = forward
                key_dir = torch.nn.functional.normalize(key_dir, dim=0)
                fill_dir = torch.tensor([-0.5, 0.2, -0.6], device="cuda")
                fill_dir = torch.nn.functional.normalize(fill_dir, dim=0)

                ndotl_key  = (norm_interp[..., :3] * key_dir).sum(-1, keepdim=True).clamp(0, 1)
                ndotl_fill = (norm_interp[..., :3] * fill_dir).sum(-1, keepdim=True).clamp(0, 1)

                shading = 0.15 + 0.65 * ndotl_key + 0.20 * ndotl_fill  # ambient + key + fill
                color_t = (base_rgb * shading).clamp(0, 1)              # (1, H, W, 3)
                color_t = dr.antialias(color_t, rast, clips, ft_loop_t)

                del norm_interp, ndotl_key, ndotl_fill, shading
            else:
                uv_interp, uv_db = dr.interpolate(vt_loop_t[None], rast, ft_loop_t)
                del uv_db
                color_t = dr.texture(tex_t, uv_interp, filter_mode="linear")
                color_t = dr.antialias(color_t, rast, clips, ft_loop_t)
                del uv_interp

            color_np = (color_t[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            mask_np  = (rast[0, :, :, 3] > 0).cpu().numpy()
            del clips, rast, color_t

        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        bg_buf[:] = bg_np
        frame_bgr = color_np[:, :, ::-1]
        if mesh_alpha >= 1.0:
            bg_buf[mask_np] = frame_bgr[mask_np]
        else:
            alpha_f = np.float32(mesh_alpha)
            bg_buf[mask_np] = (alpha_f * frame_bgr[mask_np].astype(np.float32)
                               + (1 - alpha_f) * bg_np).clip(0, 255).astype(np.uint8)

        # --- Skeleton overlay ---
        if skel_by_key:
            fk = str(data["frame_keys"][fi])
            if fk in skel_by_key:
                lms_3d = skel_by_key[fk]
                R_t = np.asarray(R_v, dtype=np.float32)
                t_t = np.asarray(t_v, dtype=np.float32)
                K_t = np.asarray(K_render, dtype=np.float32)
                pts2d = {}
                for idx, p3d in lms_3d.items():
                    pc = R_t @ p3d + t_t
                    if pc[2] < 0.01:
                        continue
                    px = K_t[0, 0] * pc[0] / pc[2] + K_t[0, 2]
                    py = K_t[1, 1] * pc[1] / pc[2] + K_t[1, 2]
                    py_flip = render_h - 1 - py  # match verts_to_clip Y-flip
                    if 0 <= px < render_w and 0 <= py_flip < render_h:
                        pts2d[idx] = (int(px), int(py_flip))
                for a, b in BODY_SKELETON:
                    if a in pts2d and b in pts2d:
                        cv2.line(bg_buf, pts2d[a], pts2d[b], (50, 220, 50), 3, cv2.LINE_AA)
                for pt in pts2d.values():
                    cv2.circle(bg_buf, pt, 6, (255, 80, 0), -1, cv2.LINE_AA)
        # Hand keypoints
        if hand_by_key and fk in hand_by_key:
            for p3d in hand_by_key[fk]:
                pc = R_t @ p3d + t_t
                if pc[2] < 0.01:
                    continue
                px = K_t[0, 0] * pc[0] / pc[2] + K_t[0, 2]
                py = K_t[1, 1] * pc[1] / pc[2] + K_t[1, 2]
                py_flip = render_h - 1 - py
                if 0 <= px < render_w and 0 <= py_flip < render_h:
                    cv2.circle(bg_buf, (int(px), int(py_flip)), 4, (0, 120, 255), -1, cv2.LINE_AA)

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
    parser.add_argument("--texture",      default=None, help="Baked texture PNG")
    parser.add_argument("--flat-color",   default=None,
                        help="Flat shaded solid color as R,G,B (0-255), e.g. 70,130,200. "
                             "Overrides --texture.")
    parser.add_argument("--output",       default=None,  help="Output .mp4 path")
    parser.add_argument("--uv-npz",       default=UV_NPZ_DEFAULT)
    parser.add_argument("--smplx-model-path", default="models/SMPLX")
    parser.add_argument("--sweep-cams",   default="cam2,cam0,cam3",
                        help="Comma-separated camera IDs: left,front,right sweep order")
    parser.add_argument("--render-size",  type=int, default=1080,
                        help="Square render resolution (default 1080)")
    parser.add_argument("--fps",          type=int, default=30)
    parser.add_argument("--alpha",        type=float, default=1.0,
                        help="Mesh opacity 0–1 (default 1.0 = fully opaque).")
    parser.add_argument("--skeleton",     default=None,
                        help="Path to reconstruction_body.json for skeleton overlay. "
                             "Auto-detected from --smplx-npz location if 'auto'.")
    args = parser.parse_args()

    output = args.output or args.smplx_npz.replace("smplx_params.npz", "turntable.mp4")
    sweep_cams = [c.strip() for c in args.sweep_cams.split(",")]
    flat_color = None
    if args.flat_color:
        flat_color = tuple(int(x) for x in args.flat_color.split(","))
    skeleton_path = args.skeleton
    if skeleton_path == "auto":
        import os as _os
        skeleton_path = _os.path.join(
            _os.path.dirname(_os.path.dirname(args.smplx_npz)),
            "reconstruction_body.json")

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
        flat_color=flat_color,
        mesh_alpha=args.alpha,
        skeleton_path=skeleton_path,
    )


if __name__ == "__main__":
    main()
