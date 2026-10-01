"""Thin-plate-spline face texture correction via landmark alignment.

After the standard texture bake, any residual error between the fitted SMPL-X
face geometry and the person's actual face in the image causes the baked texture
to sample the wrong pixels. This module corrects that by:

  1. Projecting SMPL-X head vertices to the cam0 hero frame (undistorted)
  2. Detecting MediaPipe face landmarks in the same frame
  3. Building a TPS warp from projected positions → detected positions
  4. Warping the hero frame image so mesh-projected positions now sample the
     correct (detected) pixels
  5. Re-baking the face region from the warped image

The TPS-warped image is used as a drop-in replacement for the original frame in
the face bake pass — all existing baking infrastructure is reused unchanged.

DISTORTION NOTE: all operations use undistorted pixel space (cv2.remap applied
to raw frame before any projection). Consistent with fit_face_expr.py.

Usage:
    python -m smplx_fit.face_warp \\
        --smplx-npz  /tmp/smplx_full/aligned/pose2d_sil/smplx/smplx_params.npz \\
        --calib      output/calibration/20260908_174749_7cam.json \\
        --take-dir   /tmp/smplx_full \\
        --hero-frame 127 \\
        --input-texture  /tmp/.../texture_face_flame2.png \\
        --output-texture /tmp/.../texture_face_warped.png
"""
import argparse
import os
import sys
import tempfile

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import calibrate  # noqa: E402

from hand_pose.hand_multiview import build_undistort_maps, undistort_fast  # noqa: E402
from smplx_fit import model as smplx_model  # noqa: E402
from smplx_fit.texture.fit_face_expr import _load_head_vert_indices, _mediapipe_on_crop  # noqa: E402
from smplx_fit.silhouette import project_points  # noqa: E402

UV_NPZ_DEFAULT = "/home/fmalmb/CODE/sl_reconstruction/visualization/textures/smplx_uv_2023.npz"


def _build_tps_warp(pts_src, pts_dst, img_h, img_w,
                    face_bbox, smoothing=1.0, margin_px=50):
    """Build a dense displacement field over the face bbox using TPS.

    pts_src: (N, 2) float32 — projected SMPL-X vertex positions in image
    pts_dst: (N, 2) float32 — corresponding MediaPipe-detected positions
    face_bbox: (x1, y1, x2, y2) pixel bbox of the face region to warp

    Returns map_x, map_y: (img_h, img_w) float32 arrays for cv2.remap.
    Each pixel (y, x) will be sampled from map_x[y,x], map_y[y,x].
    Outside face_bbox, map = identity (no displacement).
    """
    from scipy.interpolate import RBFInterpolator

    x1, y1, x2, y2 = face_bbox
    # Expand bbox slightly to ensure smooth border blending
    bx1 = max(0,      x1 - margin_px)
    bx2 = min(img_w,  x2 + margin_px)
    by1 = max(0,      y1 - margin_px)
    by2 = min(img_h,  y2 + margin_px)

    # TPS displacement: delta = pts_dst - pts_src
    delta = pts_dst - pts_src  # (N, 2)

    interp = RBFInterpolator(
        pts_src, delta,
        kernel="thin_plate_spline",
        smoothing=smoothing,
    )

    # Dense grid over face bbox
    bh = by2 - by1
    bw = bx2 - bx1
    gy, gx = np.mgrid[by1:by2, bx1:bx2]
    grid_pts = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float64)
    disp = interp(grid_pts).reshape(bh, bw, 2)

    # Start from identity map
    full_y, full_x = np.mgrid[0:img_h, 0:img_w]
    map_x = full_x.astype(np.float32)
    map_y = full_y.astype(np.float32)

    # Apply TPS displacement in face region
    # Soft edge: blend displacement to zero at bbox border using a distance mask
    dist_x = np.minimum(gx - bx1, bx2 - gx - 1).astype(np.float32)
    dist_y = np.minimum(gy - by1, by2 - gy - 1).astype(np.float32)
    alpha  = np.minimum(dist_x, dist_y).clip(0, margin_px) / margin_px
    alpha  = alpha[..., np.newaxis]  # (bh, bw, 1)

    alpha2d = alpha[..., 0]  # (bh, bw)
    map_x[by1:by2, bx1:bx2] += (disp[..., 0] * alpha2d).astype(np.float32)
    map_y[by1:by2, bx1:bx2] += (disp[..., 1] * alpha2d).astype(np.float32)

    return map_x, map_y


def warp_face_frame(
    smplx_npz,
    calib_path,
    take_dir,
    hero_frame,
    cam_id="cam0",
    model_path="models/SMPLX",
    corr_dist_threshold=25.0,
    min_correspondences=20,
    tps_smoothing=1.0,
    max_control_pts=200,
    save_debug=True,
    return_interp=False,
):
    """Build TPS-warped version of the hero frame and return its path.

    Saves the warped image to a temp file. If return_interp=True, also returns
    the fitted RBFInterpolator for direct UV-texel displacement queries.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------------------------------------------------
    # Load fit
    # -----------------------------------------------------------------------
    data       = np.load(smplx_npz, allow_pickle=True)
    frame_keys = list(data["frame_keys"])
    frame_key  = frame_keys[hero_frame]

    gender    = str(data["gender"])
    num_betas = int(data["num_betas"])
    model_layer = smplx_model.load_layer(model_path, gender=gender, num_betas=num_betas)
    model_layer = model_layer.to(device)

    def _t(key, fi=None):
        v = data[key] if fi is None else data[key][fi:fi+1]
        return torch.as_tensor(np.asarray(v), dtype=torch.float32, device=device)

    betas_shared = _t("betas").expand(1, -1)

    # -----------------------------------------------------------------------
    # Camera setup
    # -----------------------------------------------------------------------
    calib = calibrate.load_calibration_output(calib_path)
    cam   = calib[cam_id]
    K     = np.array(cam["K"],    dtype=np.float64)
    R     = np.array(cam["R"],    dtype=np.float64)
    t_vec = np.array(cam["t"],    dtype=np.float64)
    img_w = int(cam["width"])
    img_h = int(cam["height"])
    map1, map2 = build_undistort_maps(K, cam["dist"], img_w, img_h)

    # -----------------------------------------------------------------------
    # Load and undistort hero frame
    # -----------------------------------------------------------------------
    img_path = os.path.join(take_dir, "aligned", cam_id, frame_key)
    img_bgr  = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Frame not found: {img_path}")
    img_u = undistort_fast(img_bgr, map1, map2)
    print(f"TPS warp: frame {hero_frame} ({frame_key})")

    # -----------------------------------------------------------------------
    # SMPL-X forward at hero frame params
    # -----------------------------------------------------------------------
    aa = smplx_model.axis_angle_to_rotmat
    with torch.no_grad():
        out = model_layer(
            betas=betas_shared,
            global_orient=aa(_t("global_orient", hero_frame)),
            body_pose=aa(_t("body_pose", hero_frame)),
            left_hand_pose=aa(_t("lhand_pose", hero_frame)),
            right_hand_pose=aa(_t("rhand_pose", hero_frame)),
            transl=_t("transl", hero_frame),
            expression=_t("expression", hero_frame),
            jaw_pose=aa(_t("jaw_pose", hero_frame)),
            return_verts=True,
        )
    verts_world = out.vertices[0]  # (10475, 3)

    # -----------------------------------------------------------------------
    # Project head vertices
    # -----------------------------------------------------------------------
    head_idx_np = _load_head_vert_indices()
    head_idx_t  = torch.as_tensor(head_idx_np, dtype=torch.long, device=device)

    head_verts = verts_world[head_idx_t]
    uv_proj, depth = project_points(head_verts, K, R, t_vec)
    uv_proj_np = uv_proj.cpu().numpy()  # (H, 2)
    in_front   = (depth > 0.01).cpu().numpy()
    in_bounds  = ((uv_proj[:, 0] >= 0) & (uv_proj[:, 0] < img_w) &
                  (uv_proj[:, 1] >= 0) & (uv_proj[:, 1] < img_h)).cpu().numpy()
    visible    = in_front & in_bounds

    if visible.sum() < 50:
        raise RuntimeError("Too few visible head vertices for TPS warp")

    vis_uv = uv_proj_np[visible]
    pad_px = 200
    face_bbox = (
        int(vis_uv[:, 0].min()) - pad_px,
        int(vis_uv[:, 1].min()) - pad_px,
        int(vis_uv[:, 0].max()) + pad_px,
        int(vis_uv[:, 1].max()) + pad_px,
    )
    cx1, cy1, cx2, cy2 = face_bbox

    # -----------------------------------------------------------------------
    # MediaPipe detection
    # -----------------------------------------------------------------------
    import mediapipe as mp
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1,
        refine_landmarks=True, min_detection_confidence=0.5,
    )
    mp_pts = _mediapipe_on_crop(img_u, cx1, cy1, cx2, cy2, face_mesh)
    face_mesh.close()

    if mp_pts is None:
        raise RuntimeError(f"MediaPipe failed on hero frame {frame_key}")

    print(f"  MediaPipe: {len(mp_pts)} landmarks")

    # -----------------------------------------------------------------------
    # Build TPS control points: project visible head verts → nearest MP
    # -----------------------------------------------------------------------
    from scipy.spatial import KDTree

    vis_indices = np.where(visible)[0]
    tree = KDTree(mp_pts)
    dist, nn = tree.query(vis_uv)
    good = dist < corr_dist_threshold

    if good.sum() < min_correspondences:
        raise RuntimeError(
            f"Only {good.sum()} TPS control points (need ≥{min_correspondences})")

    pts_src_all = vis_uv[good]                     # (M, 2) projected vertex positions
    pts_dst_all = mp_pts[nn[good]]                 # (M, 2) nearest MP landmarks

    # Thin out: keep at most max_control_pts, spread across the face
    if len(pts_src_all) > max_control_pts:
        step = len(pts_src_all) // max_control_pts
        pts_src = pts_src_all[::step][:max_control_pts]
        pts_dst = pts_dst_all[::step][:max_control_pts]
    else:
        pts_src = pts_src_all
        pts_dst = pts_dst_all

    print(f"  TPS control points: {len(pts_src)}  "
          f"(from {good.sum()} candidates, dist < {corr_dist_threshold}px)")
    mean_disp = np.linalg.norm(pts_dst - pts_src, axis=1).mean()
    print(f"  Mean displacement: {mean_disp:.1f}px")

    # -----------------------------------------------------------------------
    # Build TPS interpolator (shared with apply_tps_and_rebake if needed)
    # -----------------------------------------------------------------------
    from scipy.interpolate import RBFInterpolator
    delta  = pts_dst - pts_src
    interp = RBFInterpolator(pts_src, delta,
                             kernel="thin_plate_spline", smoothing=tps_smoothing)

    map_x, map_y = _build_tps_warp(
        pts_src, pts_dst, img_h, img_w,
        face_bbox=(max(0, cx1), max(0, cy1), min(img_w, cx2), min(img_h, cy2)),
        smoothing=tps_smoothing,
    )
    warped = cv2.remap(img_u, map_x, map_y,
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # -----------------------------------------------------------------------
    # Debug image
    # -----------------------------------------------------------------------
    if save_debug:
        debug = img_u.copy()
        for s, d in zip(pts_src, pts_dst):
            sx, sy = int(s[0]), int(s[1])
            dx, dy = int(d[0]), int(d[1])
            cv2.circle(debug, (sx, sy), 3, (0, 0, 255), -1)   # red = mesh
            cv2.circle(debug, (dx, dy), 3, (0, 255, 0), -1)   # green = MediaPipe
            cv2.arrowedLine(debug, (sx, sy), (dx, dy),
                            (255, 200, 0), 1, tipLength=0.3)   # arrow = displacement
        debug_dir = os.path.dirname(smplx_npz)
        debug_path = os.path.join(debug_dir, f"debug_tps_{hero_frame}.jpg")
        cv2.imwrite(debug_path, debug)

        # Side-by-side: original crop vs warped crop
        bx1 = max(0, cx1); by1 = max(0, cy1)
        bx2 = min(img_w, cx2); by2 = min(img_h, cy2)
        orig_crop   = img_u[by1:by2, bx1:bx2]
        warped_crop = warped[by1:by2, bx1:bx2]
        compare     = np.concatenate([orig_crop, warped_crop], axis=1)
        compare_path = os.path.join(debug_dir, f"debug_tps_compare_{hero_frame}.jpg")
        cv2.imwrite(compare_path, compare)
        print(f"  Debug: {os.path.basename(debug_path)}  |  {os.path.basename(compare_path)}")
        print("    Red=mesh projected, Green=MediaPipe detected, Arrow=TPS displacement")

    # -----------------------------------------------------------------------
    # Save warped frame to temp file
    # -----------------------------------------------------------------------
    tmp_dir = os.path.dirname(smplx_npz)
    warped_path = os.path.join(tmp_dir, f"_warped_hero_{hero_frame}.jpg")
    cv2.imwrite(warped_path, warped, [cv2.IMWRITE_JPEG_QUALITY, 97])
    print(f"  Warped frame → {os.path.basename(warped_path)}")
    if return_interp:
        return warped_path, interp
    return warped_path


def apply_tps_and_rebake(
    smplx_npz,
    calib_path,
    take_dir,
    hero_frame,
    input_texture,
    output_texture,
    cam_id="cam0",
    model_path="models/SMPLX",
    uv_npz=UV_NPZ_DEFAULT,
    corr_dist_threshold=25.0,
    tps_smoothing=1.0,
    max_control_pts=200,
    save_debug=True,
):
    """Full pipeline: TPS warp the hero frame, then patch the face texture.

    Rather than injecting the warped image into the (potentially read-only)
    take directory, this function directly patches the face-region UV texels
    by projecting each texel's 3D world position to cam0, applying the TPS
    displacement, and sampling the warped image at the corrected coordinates.
    """
    import torch.nn.functional as F
    from smplx_fit.silhouette import _get_nvdiffrast_ctx
    from smplx_fit.texture_bake import load_uv_data, build_region_uv_mask, REGION_GROUPS

    # -----------------------------------------------------------------------
    # Step 1: build warped image + TPS interpolator
    # -----------------------------------------------------------------------
    warped_path, tps_interp = warp_face_frame(
        smplx_npz=smplx_npz,
        calib_path=calib_path,
        take_dir=take_dir,
        hero_frame=hero_frame,
        cam_id=cam_id,
        model_path=model_path,
        corr_dist_threshold=corr_dist_threshold,
        tps_smoothing=tps_smoothing,
        max_control_pts=max_control_pts,
        save_debug=save_debug,
        return_interp=True,
    )

    warped_img = cv2.imread(warped_path)  # (H, W, 3) BGR, undistorted
    img_h_cam, img_w_cam = warped_img.shape[:2]

    # -----------------------------------------------------------------------
    # Step 2: rasterize UV canvas → per-texel 3D world positions
    # -----------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data       = np.load(smplx_npz, allow_pickle=True)
    frame_keys = list(data["frame_keys"])
    gender     = str(data["gender"])
    num_betas  = int(data["num_betas"])
    model_layer = smplx_model.load_layer(model_path, gender=gender, num_betas=num_betas)
    model_layer = model_layer.to(device)

    def _t(key, fi=None):
        v = data[key] if fi is None else data[key][fi:fi+1]
        return torch.as_tensor(np.asarray(v), dtype=torch.float32, device=device)

    betas_shared = _t("betas").expand(1, -1)
    aa = smplx_model.axis_angle_to_rotmat

    with torch.no_grad():
        out = model_layer(
            betas=betas_shared,
            global_orient=aa(_t("global_orient", hero_frame)),
            body_pose=aa(_t("body_pose", hero_frame)),
            left_hand_pose=aa(_t("lhand_pose", hero_frame)),
            right_hand_pose=aa(_t("rhand_pose", hero_frame)),
            transl=_t("transl", hero_frame),
            expression=_t("expression", hero_frame),
            jaw_pose=aa(_t("jaw_pose", hero_frame)),
            return_verts=True,
        )
    verts_world = out.vertices[0]  # (V, 3)

    vt, _ft_uv_loop, vt_clip = load_uv_data(uv_npz)
    # ft_uv_loop is sequential arange — we need actual 3D face connectivity separately
    ft_geom = model_layer.faces   # (F, 3) int32 — 3D vertex indices
    H_tex = W_tex = 2048
    ft_loop   = np.arange(len(vt), dtype=np.int32).reshape(-1, 3)
    ft_loop_t = torch.as_tensor(ft_loop,  dtype=torch.int32,  device=device)
    vt_clip_t = torch.as_tensor(vt_clip,  dtype=torch.float32, device=device)

    glctx = _get_nvdiffrast_ctx()
    import nvdiffrast.torch as dr

    rast_out, _ = dr.rasterize(glctx, vt_clip_t[None].contiguous(),
                                ft_loop_t.contiguous(),
                                resolution=[H_tex, W_tex])
    valid_mask = (rast_out[0, :, :, 3] > 0).cpu().numpy()  # (H, W)

    # Inflate vertices to loop order (one entry per UV loop position), then interpolate
    ft_geom_flat = ft_geom.reshape(-1).astype(np.int64)   # (F*3,) 3D vert indices
    v_loop = verts_world[torch.as_tensor(ft_geom_flat, dtype=torch.long, device=device)]
    pos3d_uv, _ = dr.interpolate(v_loop[None].contiguous(), rast_out, ft_loop_t)
    pos3d_uv = pos3d_uv[0].cpu().numpy()  # (H, W, 3)

    del rast_out, v_loop

    # -----------------------------------------------------------------------
    # Step 3: project UV texels to cam0, apply TPS, sample warped image
    # -----------------------------------------------------------------------
    calib = calibrate.load_calibration_output(calib_path)
    cam   = calib[cam_id]
    K     = np.array(cam["K"], dtype=np.float64)
    R     = np.array(cam["R"], dtype=np.float64)
    t_vec = np.array(cam["t"], dtype=np.float64)

    # Build face UV mask (head + eyes region) — second rasterize call is cheap
    rast_for_mask, _ = dr.rasterize(glctx, vt_clip_t[None].contiguous(),
                                    ft_loop_t.contiguous(),
                                    resolution=[H_tex, W_tex])
    face_uv_mask_t = build_region_uv_mask(
        glctx, rast_for_mask, model_layer.faces, REGION_GROUPS["head"],
        tex_H=H_tex, tex_W=W_tex)
    face_uv_mask = face_uv_mask_t.cpu().numpy()                     # (H, W) bool
    patch_mask = valid_mask & face_uv_mask                          # only patch face

    # Get 3D positions for face UV texels
    ys, xs = np.where(patch_mask)
    pts3d  = pos3d_uv[ys, xs]                                       # (N, 3) world
    # Project to cam0 (pinhole, undistorted)
    pts_cam = pts3d @ R.T + t_vec                                    # (N, 3)
    z = pts_cam[:, 2].clip(min=1e-4)
    uv_proj = (K[:2, :] @ pts_cam.T).T / z[:, np.newaxis]          # (N, 2)
    uv_proj = uv_proj[:, :2]

    # Apply TPS displacement
    uv_corrected = uv_proj + tps_interp(uv_proj)                    # (N, 2)

    # Sample warped image at corrected positions using bilinear interpolation
    # Normalise to [-1, 1] for F.grid_sample
    grid_x = (uv_corrected[:, 0] / (img_w_cam - 1)) * 2.0 - 1.0
    grid_y = (uv_corrected[:, 1] / (img_h_cam - 1)) * 2.0 - 1.0
    grid   = torch.tensor(
        np.stack([grid_x, grid_y], axis=1), dtype=torch.float32, device=device
    ).view(1, 1, -1, 2)

    warped_rgb = warped_img[:, :, ::-1].astype(np.float32) / 255.0  # RGB float
    warped_t   = torch.tensor(warped_rgb, dtype=torch.float32, device=device
                              ).permute(2, 0, 1).unsqueeze(0)        # (1,3,H,W)
    sampled = F.grid_sample(warped_t, grid,
                            mode="bilinear", padding_mode="border",
                            align_corners=True)                       # (1,3,1,N)
    sampled_np = sampled[0, :, 0, :].permute(1, 0).cpu().numpy()     # (N,3) RGB

    # -----------------------------------------------------------------------
    # Step 4: patch the input texture at face region
    # -----------------------------------------------------------------------
    tex_bgr = cv2.imread(input_texture)
    tex_rgb = tex_bgr[:, :, ::-1].astype(np.float32) / 255.0

    # UV origin: nvdiffrast row 0 = y_ndc=-1 = V=0 → texture bottom → flip
    tex_h, tex_w = tex_rgb.shape[:2]
    ys_flipped = tex_h - 1 - ys      # flip V axis to match texture convention
    tex_rgb[ys_flipped, xs] = sampled_np.clip(0, 1)

    out_bgr = (tex_rgb[:, :, ::-1] * 255).astype(np.uint8)
    cv2.imwrite(output_texture, out_bgr)

    if os.path.exists(warped_path):
        os.remove(warped_path)

    print(f"Patched {patch_mask.sum()} face UV texels → {output_texture}")
    return output_texture


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smplx-npz",       required=True)
    parser.add_argument("--calib",            required=True)
    parser.add_argument("--take-dir",         required=True)
    parser.add_argument("--hero-frame",       type=int, required=True,
                        help="Frame index used for the face texture bake")
    parser.add_argument("--input-texture",    required=True,
                        help="Baked face texture to correct (texture_face_*.png)")
    parser.add_argument("--output-texture",   required=True,
                        help="Output path for the TPS-corrected texture")
    parser.add_argument("--cam",              default="cam0")
    parser.add_argument("--smplx-model-path", default="models/SMPLX")
    parser.add_argument("--uv-npz",           default=UV_NPZ_DEFAULT)
    parser.add_argument("--corr-dist",        type=float, default=25.0,
                        help="Max pixel distance for mesh→MP correspondence")
    parser.add_argument("--tps-smoothing",    type=float, default=1.0,
                        help="RBFInterpolator smoothing (0=exact TPS, higher=smoother)")
    parser.add_argument("--max-control-pts",  type=int, default=200,
                        help="Max TPS control points (subsampled from correspondences)")
    parser.add_argument("--no-debug",         action="store_true")
    args = parser.parse_args()

    apply_tps_and_rebake(
        smplx_npz        = args.smplx_npz,
        calib_path       = args.calib,
        take_dir         = args.take_dir,
        hero_frame       = args.hero_frame,
        input_texture    = args.input_texture,
        output_texture   = args.output_texture,
        cam_id           = args.cam,
        model_path       = args.smplx_model_path,
        uv_npz           = args.uv_npz,
        corr_dist_threshold = args.corr_dist,
        tps_smoothing    = args.tps_smoothing,
        max_control_pts  = args.max_control_pts,
        save_debug       = not args.no_debug,
    )


if __name__ == "__main__":
    main()
