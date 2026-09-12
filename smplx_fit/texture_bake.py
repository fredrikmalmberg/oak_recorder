"""UV texture baking for SMPL-X avatars from multi-camera footage.

Inverts the normal rendering direction: treats the flat 2D UV canvas as the
render target, bakes 3D world positions into every UV texel via nvdiffrast
interpolation, projects those texel positions to each camera view, samples
colour + RVM mask, weights by surface-angle cosine, and accumulates into a
single texture PNG.

Key features:
  --hero-frames      : select still, pose-diverse keyframes automatically
  --winner-take-all  : per-texel best-quality overwrite (default) vs. avg
  --optical-flow     : warp non-anchor UV textures to anchor before accumulation
  --seam-blur-radius : blur only at camera-boundary seam pixels

DISTORTION CONSISTENCY — READ THIS BEFORE TOUCHING ANY PIXEL-SPACE CODE:
  project_points() and _verts_to_clip() both use the pinhole K with NO lens
  distortion — they produce pixel coordinates in *undistorted* space.
  The raw aligned frames (aligned/cam*/) and masks (aligned/masks_rvm/) are
  saved from the original distorted video — they live in *distorted* space.
  Any comparison or sampling that crosses this boundary is wrong.
  This has caused subtle bugs multiple times (IoU values deflated, colour
  samples misregistered near frame edges).
  Rule: undistort frames and masks (cv2.remap with pre-built maps) BEFORE
  doing any pixel-space operation. Maps are built once per camera from
  c["K"] + c["dist"] via hand_multiview.build_undistort_maps.
  silhouette.py does this correctly — see load_frame_masks_for_cameras().

Usage:
    python -m smplx_fit.texture_bake \\
        --smplx-npz /tmp/smplx_full/aligned/pose2d_sil/smplx/smplx_params.npz \\
        --calib output/calibration/20260908_174749_7cam.json \\
        --take-dir /tmp/smplx_full \\
        --hero-frames --winner-take-all --optical-flow
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import calibrate  # noqa: E402

from hand_pose.hand_multiview import build_undistort_maps, undistort_fast  # noqa: E402
from smplx_fit import model as smplx_model  # noqa: E402
from smplx_fit.silhouette import _get_nvdiffrast_ctx, project_points, _verts_to_clip  # noqa: E402

UV_NPZ_DEFAULT = "/home/fmalmb/CODE/sl_reconstruction/visualization/textures/smplx_uv_2023.npz"

# SMPL-X body joint indices (fixed across model versions)
J_LEFT_HIP    = 1
J_RIGHT_HIP   = 2
J_LEFT_WRIST  = 20
J_RIGHT_WRIST = 21

# Lazy RAFT singleton
_raft_model = None
_raft_model_name = None


# ---------------------------------------------------------------------------
# UV data helpers
# ---------------------------------------------------------------------------

def load_uv_data(path):
    """Load per-loop UV coordinates from smplx_uv_2023.npz.

    Returns:
        vt:      (62724, 2) float32 UV coords in [0, 1]
        ft:      (20908, 3) int32  UV face indices (per-loop: ft[i,j] = i*3+j)
        vt_clip: (62724, 4) float32 nvdiffrast clip-space positions
                 x ∈ [-1,1] left→right, y flipped: Blender V=0 is bottom but
                 nvdiffrast stores row 0 at y_ndc=-1 (OpenGL bottom), so
                 y_clip = (1 - v)*2 - 1 maps V=1 (top) → y_ndc=+1 (top).
                 z=0, w=1 (orthographic; no perspective divide needed).
    """
    data = np.load(path)
    vt = data["uv_coordinates"].astype(np.float32)  # (62724, 2)
    ft = np.arange(len(vt), dtype=np.int32).reshape(-1, 3)  # (20908, 3)
    x_clip = vt[:, 0] * 2.0 - 1.0
    y_clip = (1.0 - vt[:, 1]) * 2.0 - 1.0
    vt_clip = np.stack([x_clip, y_clip, np.zeros(len(vt), dtype=np.float32),
                        np.ones(len(vt), dtype=np.float32)], axis=-1)
    return vt, ft, vt_clip


# ---------------------------------------------------------------------------
# UV canvas rasterization
# ---------------------------------------------------------------------------

def rasterize_uv_canvas(glctx, vt_clip, ft, resolution):
    """Rasterize the UV parameter space; run once, shared across all frames.

    Returns:
        rast_out: (1, H, W, 4) — channels (u_bary, v_bary, depth, face_id+1);
                  face_id+1 > 0 marks valid texels.
    """
    import nvdiffrast.torch as dr
    vt_t = torch.as_tensor(vt_clip, dtype=torch.float32, device="cuda")
    ft_t = torch.as_tensor(ft, dtype=torch.int32, device="cuda")
    rast_out, _ = dr.rasterize(glctx, vt_t[None], ft_t, resolution=list(resolution))
    return rast_out


# ---------------------------------------------------------------------------
# Smooth vertex normals
# ---------------------------------------------------------------------------

def compute_smooth_normals(vertices, faces):
    """Angle-weighted per-vertex normals.

    Args:
        vertices: (V, 3) numpy or torch float32
        faces:    (F, 3) int array/tensor

    Returns:
        normals: (V, 3) float32 torch tensor (normalised)
    """
    if isinstance(vertices, np.ndarray):
        vertices = torch.as_tensor(vertices, dtype=torch.float32)
    if isinstance(faces, np.ndarray):
        faces = torch.as_tensor(faces.astype(np.int64), dtype=torch.int64)
    else:
        faces = faces.long()

    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)  # (F, 3)

    normals = torch.zeros_like(vertices)
    for i in range(3):
        normals.scatter_add_(0, faces[:, i:i+1].expand(-1, 3), face_normals)
    normals = F.normalize(normals, dim=-1)
    return normals


# ---------------------------------------------------------------------------
# Hero frame selection
# ---------------------------------------------------------------------------

def select_hero_frames(data, model_path, n_arms_down=2, n_arms_wide=2,
                        velocity_threshold=0.01, frame_pool=None):
    """Select pose-diverse still keyframes from the full sequence.

    Runs a batched SMPL-X forward over all frames, computes per-frame vertex
    velocity, filters to still frames, then selects frames by pose type:

      arms-down: wrists below hips (hip_Z - wrist_Z maximised)
      arms-wide: wrists far apart (|left_wrist - right_wrist| maximised)

    Returns:
        indices: sorted list of frame indices
        labels:  dict mapping index → pose-type label string
    """
    frame_keys = [str(fk) for fk in data["frame_keys"]]
    nf = len(frame_keys)
    gender = str(data["gender"])
    num_betas = int(data["num_betas"])

    print(f"Hero frame selection: running SMPL-X forward on all {nf} frames...")
    model = smplx_model.load_layer(model_path, gender=gender, num_betas=num_betas)

    betas = torch.as_tensor(data["betas"], dtype=torch.float32)
    if betas.shape[0] == 1:
        betas = betas.expand(nf, -1)

    # Run in CPU batches of 100 to avoid OOM
    all_verts = []
    all_joints = []
    batch_sz = 100
    for start in range(0, nf, batch_sz):
        end = min(start + batch_sz, nf)
        sl = slice(start, end)
        with torch.no_grad():
            out = smplx_model.forward(
                model,
                betas[sl],
                torch.as_tensor(data["global_orient"][sl], dtype=torch.float32),
                torch.as_tensor(data["body_pose"][sl], dtype=torch.float32),
                torch.as_tensor(data["lhand_pose"][sl], dtype=torch.float32),
                torch.as_tensor(data["rhand_pose"][sl], dtype=torch.float32),
                torch.as_tensor(data["transl"][sl], dtype=torch.float32),
            )
        all_verts.append(out.vertices.numpy())
        all_joints.append(out.joints.numpy())

    vertices = np.concatenate(all_verts, axis=0)   # (N, V, 3)
    joints   = np.concatenate(all_joints, axis=0)  # (N, 55, 3)

    # Per-frame velocity: mean per-vertex displacement vs previous frame
    dv = np.abs(vertices[1:] - vertices[:-1]).mean(axis=(1, 2))  # (N-1,)
    velocity = np.concatenate([[dv[0]], dv])  # (N,) — pad first frame

    allowed = np.array(frame_pool) if frame_pool is not None else np.arange(nf)
    still_pool = allowed[velocity[allowed] < velocity_threshold]
    if len(still_pool) == 0:
        print(f"  Warning: no frames below velocity threshold {velocity_threshold}; "
              f"relaxing to 3× threshold.")
        still_pool = allowed[velocity[allowed] < velocity_threshold * 3]
    pool_desc = f"{len(allowed)} allowed" if frame_pool is not None else f"all {nf}"
    print(f"  Still-frame pool: {len(still_pool)} / {pool_desc} frames "
          f"(threshold {velocity_threshold} m/frame)")

    # Extract joint positions for the still pool
    lw = joints[still_pool, J_LEFT_WRIST,  :]  # (S, 3)
    rw = joints[still_pool, J_RIGHT_WRIST, :]
    lh = joints[still_pool, J_LEFT_HIP,    :]
    rh = joints[still_pool, J_RIGHT_HIP,   :]

    mean_hip_z   = (lh[:, 2] + rh[:, 2]) / 2.0
    mean_wrist_z = (lw[:, 2] + rw[:, 2]) / 2.0
    arms_down_score = mean_hip_z - mean_wrist_z   # larger = wrists further below hips
    arms_wide_score = np.linalg.norm(lw - rw, axis=-1)  # 3D wrist separation

    selected = {}  # index → label

    top_down = still_pool[np.argsort(-arms_down_score)[:n_arms_down]]
    for idx in top_down:
        selected[int(idx)] = "arms-down"

    top_wide = still_pool[np.argsort(-arms_wide_score)[:n_arms_wide]]
    for idx in top_wide:
        if idx not in selected:
            selected[int(idx)] = "arms-wide"
        else:
            selected[int(idx)] += "+arms-wide"

    indices = sorted(selected.keys())
    print("  Selected hero frames:")
    for idx in indices:
        print(f"    frame {idx:4d}  ({frame_keys[idx]})  [{selected[idx]}]"
              f"  velocity={velocity[idx]:.4f}m/f")

    return indices, selected


# ---------------------------------------------------------------------------
# Per-camera hero frame selection
# ---------------------------------------------------------------------------

def select_hero_frames_per_camera(
    data, model_path, calib, take_dir,
    n_arms_down=2, n_arms_wide=2,
    velocity_threshold=0.01, frame_pool=None,
    min_iou=0.5, mask_subdir="masks_rvm",
    mask_threshold=50, downsample=4, glctx=None,
):
    """Select hero frames independently per camera.

    For each camera, IoU(mesh silhouette, RVM mask) is computed for every
    still-pool frame. Frames below min_iou are excluded, then the best
    arms-down and arms-wide frames are chosen from the remainder.

    Returns dict[cam_id → sorted list of frame indices].
    """
    import nvdiffrast.torch as dr

    frame_keys = [str(fk) for fk in data["frame_keys"]]
    nf = len(frame_keys)
    gender = str(data["gender"])
    num_betas = int(data["num_betas"])

    print(f"Per-camera hero selection: SMPL-X forward on all {nf} frames...")
    model = smplx_model.load_layer(model_path, gender=gender, num_betas=num_betas)

    betas = torch.as_tensor(data["betas"], dtype=torch.float32)
    if betas.shape[0] == 1:
        betas = betas.expand(nf, -1)

    all_verts, all_joints = [], []
    for start in range(0, nf, 100):
        end = min(start + 100, nf)
        sl = slice(start, end)
        with torch.no_grad():
            out = smplx_model.forward(
                model,
                betas[sl],
                torch.as_tensor(data["global_orient"][sl], dtype=torch.float32),
                torch.as_tensor(data["body_pose"][sl], dtype=torch.float32),
                torch.as_tensor(data["lhand_pose"][sl], dtype=torch.float32),
                torch.as_tensor(data["rhand_pose"][sl], dtype=torch.float32),
                torch.as_tensor(data["transl"][sl], dtype=torch.float32),
            )
        all_verts.append(out.vertices.numpy())
        all_joints.append(out.joints.numpy())

    vertices = np.concatenate(all_verts, axis=0)   # (N, V, 3)
    joints   = np.concatenate(all_joints, axis=0)  # (N, 55, 3)

    dv = np.abs(vertices[1:] - vertices[:-1]).mean(axis=(1, 2))
    velocity = np.concatenate([[dv[0]], dv])

    allowed = np.array(frame_pool) if frame_pool is not None else np.arange(nf)
    still_pool = allowed[velocity[allowed] < velocity_threshold]
    if len(still_pool) == 0:
        still_pool = allowed[velocity[allowed] < velocity_threshold * 3]
    print(f"  Still-frame pool: {len(still_pool)} frames")

    # Pose scores indexed by still_pool position
    lw = joints[still_pool, J_LEFT_WRIST,  :]
    rw = joints[still_pool, J_RIGHT_WRIST, :]
    lh = joints[still_pool, J_LEFT_HIP,    :]
    rh = joints[still_pool, J_RIGHT_HIP,   :]
    arms_down_score = (lh[:, 2] + rh[:, 2]) / 2.0 - (lw[:, 2] + rw[:, 2]) / 2.0
    arms_wide_score = np.linalg.norm(lw - rw, axis=-1)

    faces_i32 = torch.as_tensor(model.faces.astype(np.int32), dtype=torch.int32).cuda()
    aligned_dir = os.path.join(take_dir, "aligned")
    per_cam_frames = {}

    for cam_id in sorted(calib.keys()):
        cam = calib[cam_id]
        K, R, t = cam["K"], cam["R"], cam["t"]
        img_w, img_h = cam["width"], cam["height"]
        ds_w, ds_h = img_w // downsample, img_h // downsample
        # Undistort maps for this camera — masks are from distorted frames,
        # mesh silhouette uses pinhole K (undistorted space).
        m1, m2 = build_undistort_maps(K, cam["dist"], img_w, img_h)

        iou_scores = np.full(len(still_pool), -1.0)
        for si, frame_idx in enumerate(still_pool):
            fk = frame_keys[frame_idx]
            mask_path = os.path.join(aligned_dir, mask_subdir, cam_id, fk)
            if not os.path.exists(mask_path):
                continue
            mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_gray is None:
                continue
            mask_gray = cv2.remap(mask_gray, m1, m2, interpolation=cv2.INTER_NEAREST)
            mask_ds = cv2.resize(mask_gray, (ds_w, ds_h), interpolation=cv2.INTER_AREA)
            rvm_sil = mask_ds > mask_threshold

            verts_t = torch.as_tensor(vertices[frame_idx], dtype=torch.float32)
            clips = _verts_to_clip(verts_t.unsqueeze(0), K, R, t, img_w, img_h).cuda()
            with torch.no_grad():
                rast, _ = dr.rasterize(glctx, clips.contiguous(), faces_i32.contiguous(),
                                       resolution=[ds_h, ds_w])
            mesh_sil = (rast[0, :, :, 3] > 0).cpu().numpy()
            inter = (mesh_sil & rvm_sil).sum()
            union = (mesh_sil | rvm_sil).sum()
            iou_scores[si] = inter / max(union, 1)

        good = np.where(iou_scores >= min_iou)[0]  # indices into still_pool arrays
        if len(good) == 0:
            mx = iou_scores[iou_scores >= 0].max() if (iou_scores >= 0).any() else 0.0
            print(f"  {cam_id}: 0 frames pass IoU ≥ {min_iou:.2f} (max {mx:.3f}) — skipped")
            per_cam_frames[cam_id] = []
            continue

        selected = {}
        top_down = good[np.argsort(-arms_down_score[good])[:n_arms_down]]
        for i in top_down:
            selected[int(still_pool[i])] = "arms-down"
        top_wide = good[np.argsort(-arms_wide_score[good])[:n_arms_wide]]
        for i in top_wide:
            idx = int(still_pool[i])
            selected[idx] = selected.get(idx, "") + ("" if idx not in selected else "+") + "arms-wide"
            if idx not in selected:
                selected[idx] = "arms-wide"

        per_cam_frames[cam_id] = sorted(selected.keys())
        details = "  ".join(
            f"{frame_keys[idx]}(IoU={iou_scores[np.where(still_pool == idx)[0][0]]:.3f},{selected[idx]})"
            for idx in per_cam_frames[cam_id]
        )
        print(f"  {cam_id}: {details}")

    return per_cam_frames


# ---------------------------------------------------------------------------
# Optical flow registration (RAFT, UV space)
# ---------------------------------------------------------------------------

def _get_raft_model(model_name="raft_large"):
    global _raft_model, _raft_model_name
    if _raft_model is None or _raft_model_name != model_name:
        try:
            from torchvision.models.optical_flow import (
                raft_large, raft_small,
                Raft_Large_Weights, Raft_Small_Weights,
            )
            if model_name == "raft_small":
                m = raft_small(weights=Raft_Small_Weights.C_T_V2)
            else:
                m = raft_large(weights=Raft_Large_Weights.C_T_SKHT_K_V2)
            _raft_model = m.cuda().eval()
            _raft_model_name = model_name
            print(f"  RAFT model loaded: {model_name}")
        except Exception as e:
            print(f"  Warning: could not load RAFT ({e}); optical flow disabled.")
            _raft_model = None
    return _raft_model


def register_with_optical_flow(cam_colors, cam_qualities, model_name="raft_large",
                                flow_downsample=4, flow_max_pixels=64):
    """Warp all non-anchor UV textures onto the anchor camera's perspective.

    Flow is computed at (H/flow_downsample, W/flow_downsample) to fit in GPU
    memory, then upsampled back to full resolution before warping.

    Args:
        cam_colors:      dict cam_id → (3, H, W) float32 GPU tensor [0,1]
        cam_qualities:   dict cam_id → (1, H, W) float32 GPU tensor
        flow_downsample: factor to reduce UV texture resolution for RAFT

    Returns:
        warped_colors, warped_qualities: same structure, non-anchor tensors
        warped to align with the anchor.
    """
    raft = _get_raft_model(model_name)
    if raft is None:
        return cam_colors, cam_qualities

    cam_ids = list(cam_colors.keys())
    mean_q = {c: cam_qualities[c][cam_qualities[c] > 0].mean().item()
              if (cam_qualities[c] > 0).any() else 0.0
              for c in cam_ids}
    anchor_id = max(mean_q, key=mean_q.get)
    print(f"    Optical flow anchor: {anchor_id} (quality {mean_q[anchor_id]:.4f})")

    H, W = cam_colors[anchor_id].shape[1], cam_colors[anchor_id].shape[2]
    fH, fW = H // flow_downsample, W // flow_downsample

    # Downsampled anchor for RAFT [0,255]
    anchor_small = F.interpolate(cam_colors[anchor_id].unsqueeze(0),
                                  size=(fH, fW), mode="bilinear",
                                  align_corners=False) * 255.0  # (1,3,fH,fW)

    warped_colors    = {anchor_id: cam_colors[anchor_id]}
    warped_qualities = {anchor_id: cam_qualities[anchor_id]}

    # Full-res base grid for warping
    ys = torch.linspace(-1, 1, H, device="cuda")
    xs = torch.linspace(-1, 1, W, device="cuda")
    base_grid_y, base_grid_x = torch.meshgrid(ys, xs, indexing="ij")
    base_grid = torch.stack([base_grid_x, base_grid_y], dim=-1).unsqueeze(0)  # (1,H,W,2)

    for cam_id in cam_ids:
        if cam_id == anchor_id:
            continue
        src_small = F.interpolate(cam_colors[cam_id].unsqueeze(0),
                                   size=(fH, fW), mode="bilinear",
                                   align_corners=False) * 255.0

        with torch.no_grad():
            flow_list = raft(anchor_small, src_small)
        flow_small = flow_list[-1].clone()  # (1,2,fH,fW)
        del flow_list
        torch.cuda.empty_cache()

        # Upsample flow to full res and rescale pixel displacements
        flow_full = F.interpolate(flow_small, size=(H, W), mode="bilinear",
                                   align_corners=False) * flow_downsample  # (1,2,H,W)
        del flow_small

        # Clamp flow magnitude — large vectors near coverage boundaries are RAFT
        # noise, not real surface correspondence. Genuine camera-to-camera UV
        # misalignment is small; anything beyond flow_max_pixels is discarded.
        if flow_max_pixels > 0:
            mag = flow_full.norm(dim=1, keepdim=True)  # (1,1,H,W)
            scale = (mag.clamp(max=flow_max_pixels) / mag.clamp(min=1e-6))
            flow_full = flow_full * scale

        flow_norm = torch.stack([
            flow_full[:, 0] / (W / 2.0),
            flow_full[:, 1] / (H / 2.0),
        ], dim=-1)  # (1,H,W,2)
        del flow_full
        warp_grid = base_grid + flow_norm

        # Remember which texels this camera actually covered BEFORE warping.
        # Flow over black (uncovered) UV regions is meaningless — RAFT hallucinates
        # vectors there that warp black into covered areas. Mask them out.
        src_valid = (cam_qualities[cam_id] > 0)  # (1, H, W) bool

        w_color = F.grid_sample(cam_colors[cam_id].unsqueeze(0), warp_grid,
                                 mode="bilinear", align_corners=False,
                                 padding_mode="zeros").squeeze(0)
        # cam_qualities is (1,H,W) — needs (N,C,H,W) for grid_sample
        w_quality = F.grid_sample(cam_qualities[cam_id].unsqueeze(0), warp_grid,
                                   mode="bilinear", align_corners=False,
                                   padding_mode="zeros").squeeze(0)

        # Zero out warped results where the source had no coverage — warp into
        # black is always wrong; also zero where the warp pulled from outside [-1,1].
        w_src_valid = F.grid_sample(src_valid.float().unsqueeze(0), warp_grid,
                                     mode="nearest", align_corners=False,
                                     padding_mode="zeros").squeeze(0)  # (1,H,W)
        w_color   = w_color   * w_src_valid
        w_quality = w_quality * w_src_valid

        warped_colors[cam_id]    = w_color
        warped_qualities[cam_id] = w_quality
        torch.cuda.empty_cache()

    return warped_colors, warped_qualities


# ---------------------------------------------------------------------------
# Seam smoothing
# ---------------------------------------------------------------------------

SMPLX_SEG_JSON = "/home/fmalmb/CODE/sequence_rendering/smplx_models/smplx/smplx_vert_segmentation.json"

# Body part groups available via --protect-region
REGION_GROUPS = {
    "head": ["head", "eyeballs", "leftEye", "rightEye"],
    "neck": ["neck"],
}


def build_region_uv_mask(glctx, rast_out, smplx_faces, region_names, tex_H, tex_W,
                          seg_json=SMPLX_SEG_JSON):
    """Binary UV-space mask for the given SMPL-X body part region names.

    Maps UV vertices back to 3D vertices via smplx_faces, looks up each vertex
    in the segmentation JSON, then interpolates a per-UV-vertex 0/1 attribute
    into UV space with dr.interpolate.

    Args:
        glctx:         nvdiffrast context
        rast_out:      (1, H, W, 4) UV canvas rasterization output
        smplx_faces:   (F, 3) SMPL-X face connectivity (3D vertex indices, int32)
        region_names:  list of segmentation part names to mark as protected
        tex_H, tex_W:  texture resolution (must match rast_out)
        seg_json:      path to smplx_vert_segmentation.json

    Returns:
        (H, W) bool CPU tensor — True inside the protected region
    """
    import json
    import nvdiffrast.torch as dr

    with open(seg_json) as f:
        seg = json.load(f)

    protected_verts = set()
    for name in region_names:
        if name in seg:
            protected_verts.update(seg[name])
        else:
            print(f"  Warning: region '{name}' not in segmentation JSON — skipped")

    # UV vertex i*3+j corresponds to 3D vertex smplx_faces[i, j]
    n_faces = len(smplx_faces)
    attr = np.zeros(n_faces * 3, dtype=np.float32)
    faces_flat = smplx_faces.reshape(-1)  # (F*3,)
    for idx, v3d in enumerate(faces_flat):
        if int(v3d) in protected_verts:
            attr[idx] = 1.0

    attr_t  = torch.as_tensor(attr, dtype=torch.float32, device="cuda").reshape(-1, 1)
    ft_uv   = torch.arange(n_faces * 3, dtype=torch.int32, device="cuda").reshape(-1, 3)
    interp, _ = dr.interpolate(attr_t[None], rast_out, ft_uv)   # (1, H, W, 1)
    return (interp[0, :, :, 0] > 0.5).cpu()                     # (H, W) bool


def smooth_seams(color_bgr, source_map_np, blur_radius=3):
    """Gaussian-blur only the pixels where the winning camera changes.

    Args:
        color_bgr:     (H, W, 3) uint8 BGR texture
        source_map_np: (H, W) int32 numpy, winning source id per texel (-1=uncovered)
        blur_radius:   kernel half-width (0 = skip)

    Returns:
        (H, W, 3) uint8 smoothed BGR texture
    """
    if blur_radius <= 0:
        return color_bgr

    # Seam pixels: any texel whose 4-neighbourhood has a different source id
    seam = np.zeros(source_map_np.shape, dtype=np.uint8)
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        shifted = np.roll(source_map_np, (dy, dx), axis=(0, 1))
        seam |= (source_map_np != shifted).astype(np.uint8)
    # Don't blur at uncovered texels
    seam[source_map_np < 0] = 0

    k = blur_radius * 2 + 1
    seam_dilated = cv2.dilate(seam, np.ones((k, k), np.uint8))
    blurred = cv2.GaussianBlur(color_bgr, (k, k), 0)
    result = color_bgr.copy()
    result[seam_dilated > 0] = blurred[seam_dilated > 0]
    return result


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def bake_texture(
    smplx_npz,
    calib_path,
    take_dir,
    uv_npz=UV_NPZ_DEFAULT,
    model_path="models/SMPLX",
    tex_resolution=2048,
    n_frames=50,
    downsample=4,
    mask_subdir="masks_rvm",
    mask_threshold=50,
    output_path=None,
    debug_phase1=False,
    cam_ids=None,
    frame_range=None,
    normal_threshold=0.0,
    depth_tolerance=0.02,
    hero_frames=False,
    per_camera_hero=False,
    n_arms_down=2,
    n_arms_wide=2,
    velocity_threshold=0.01,
    winner_take_all=True,
    optical_flow=False,
    flow_model="raft_large",
    flow_max_pixels=64,
    seam_blur_radius=3,
    min_iou=0.5,
    front_cam=None,
    front_cam_bias=2.0,
    base_texture=None,
    protect_region=None,
):
    """Bake a texture from multi-camera footage onto the SMPL-X UV layout.

    Returns output_path (str) where the PNG was saved.
    """
    if output_path is None:
        output_path = os.path.join(os.path.dirname(smplx_npz), "texture_baked.png")

    import nvdiffrast.torch as dr

    print("Loading calibration...")
    calib = calibrate.load_calibration_output(calib_path)
    if cam_ids is None:
        cam_ids = sorted(calib.keys())

    print("Loading UV data...")
    _vt, ft, vt_clip = load_uv_data(uv_npz)

    print("Initialising nvdiffrast context...")
    glctx = _get_nvdiffrast_ctx()

    tex_H = tex_W = tex_resolution
    print(f"Rasterizing UV canvas ({tex_H}×{tex_W})...")
    rast_out = rasterize_uv_canvas(glctx, vt_clip, ft, (tex_H, tex_W))
    valid_mask = rast_out[0, :, :, 3] > 0  # (H, W) bool

    coverage_pct = valid_mask.float().mean().item() * 100
    debug_path = output_path.replace(".png", "_uv_coverage.png")
    coverage_img = (valid_mask.cpu().numpy() * 255).astype(np.uint8)
    cv2.imwrite(debug_path, coverage_img)
    print(f"UV coverage map → {debug_path}")
    print(f"  Valid texels: {valid_mask.sum().item()} / {valid_mask.numel()} ({coverage_pct:.1f}%)")

    if debug_phase1:
        print("--debug-phase1: stopping. Inspect the coverage map before running the full bake.")
        return debug_path

    # --- Load SMPL-X fit and select frames ---
    print("Loading SMPL-X fit...")
    data = np.load(smplx_npz, allow_pickle=True)
    frame_keys = [str(fk) for fk in data["frame_keys"]]
    gender = str(data["gender"])
    num_betas = int(data["num_betas"])

    nf = len(frame_keys)
    frame_pool_range = list(range(*frame_range)) if frame_range is not None else None

    per_cam_frames = None  # None means "all cameras use all selected frames"

    if per_camera_hero:
        per_cam_frames = select_hero_frames_per_camera(
            data, model_path, calib, take_dir,
            n_arms_down=n_arms_down, n_arms_wide=n_arms_wide,
            velocity_threshold=velocity_threshold,
            frame_pool=frame_pool_range,
            min_iou=min_iou,
            mask_subdir=mask_subdir, mask_threshold=mask_threshold,
            downsample=downsample, glctx=glctx,
        )
        all_per_cam = set().union(*per_cam_frames.values())
        if not all_per_cam:
            print("No frames selected for any camera — check min_iou threshold.")
            return output_path
        indices = sorted(all_per_cam)
        hero_labels = {}
        print(f"  Union of per-camera hero frames: {len(indices)} frames")
    elif hero_frames:
        indices, hero_labels = select_hero_frames(
            data, model_path,
            n_arms_down=n_arms_down, n_arms_wide=n_arms_wide,
            velocity_threshold=velocity_threshold,
            frame_pool=frame_pool_range,
        )
    else:
        pool = frame_pool_range if frame_pool_range is not None else list(range(nf))
        indices = [pool[i] for i in np.linspace(0, len(pool) - 1,
                                                  min(n_frames, len(pool)), dtype=int)]
        hero_labels = {}
        print(f"Selected {len(indices)} frames (pool {pool[0]}–{pool[-1]}) for baking.")

    sel_keys = [frame_keys[i] for i in indices]

    # Batched SMPL-X forward for the selected frames only
    model = smplx_model.load_layer(model_path, gender=gender, num_betas=num_betas)
    betas = torch.as_tensor(data["betas"], dtype=torch.float32)
    if betas.shape[0] == 1:
        betas = betas.expand(len(indices), -1)
    else:
        betas = betas[indices]

    with torch.no_grad():
        output = smplx_model.forward(
            model,
            betas,
            torch.as_tensor(data["global_orient"][indices], dtype=torch.float32),
            torch.as_tensor(data["body_pose"][indices], dtype=torch.float32),
            torch.as_tensor(data["lhand_pose"][indices], dtype=torch.float32),
            torch.as_tensor(data["rhand_pose"][indices], dtype=torch.float32),
            torch.as_tensor(data["transl"][indices], dtype=torch.float32),
        )
    all_vertices = output.vertices.numpy()  # (N, V, 3)
    faces = model.faces  # (F, 3) int32
    ft_geom = torch.as_tensor(faces.astype(np.int64), dtype=torch.int64)
    ft_i32  = torch.as_tensor(ft, dtype=torch.int32, device="cuda")

    # Accumulation buffers — winner-take-all or weighted average
    if winner_take_all:
        quality_map = torch.full((1, tex_H, tex_W), -1.0, device="cuda")
        color_map   = torch.zeros(3, tex_H, tex_W, device="cuda")
        source_map  = torch.full((tex_H, tex_W), -1, dtype=torch.int32, device="cuda")
    else:
        color_sum   = torch.zeros(3, tex_H, tex_W, device="cuda")
        weight_sum  = torch.zeros(1, tex_H, tex_W, device="cuda")

    # Base texture + region protection — initialise from an existing texture and
    # lock specified body-part regions so they are never overwritten.
    if base_texture is not None and winner_take_all:
        base_bgr = cv2.imread(base_texture)
        if base_bgr is None:
            raise FileNotFoundError(f"--base-texture not found: {base_texture}")
        base_rgb = base_bgr[:, :, ::-1].astype(np.float32) / 255.0
        if base_rgb.shape[:2] != (tex_H, tex_W):
            base_rgb = cv2.resize(base_rgb, (tex_W, tex_H), interpolation=cv2.INTER_LINEAR)
        color_map[:] = torch.as_tensor(base_rgb, dtype=torch.float32,
                                        device="cuda").permute(2, 0, 1)
        print(f"Initialised color_map from base texture: {base_texture}")

        if protect_region:
            parts = REGION_GROUPS.get(protect_region, [protect_region])
            print(f"Building '{protect_region}' UV mask ({', '.join(parts)})...")
            head_mask = build_region_uv_mask(glctx, rast_out, faces, parts, tex_H, tex_W)
            # +inf quality at protected texels → winner-take-all can never beat it
            quality_map[0][head_mask.cuda()] = float("inf")
            source_map[head_mask.cuda()] = -2   # sentinel: protected
            n_protected = head_mask.sum().item()
            print(f"  Protected {n_protected} texels ({n_protected/valid_mask.sum().item()*100:.1f}% of UV coverage)")

    aligned_dir = os.path.join(take_dir, "aligned")

    # Build undistort maps once per camera (full resolution).
    # project_points / _verts_to_clip use pinhole K — undistorted pixel space.
    # Frames and masks on disk are from the raw distorted video.
    # Remap both to undistorted space before any pixel-space operation.
    undistort_maps = {}
    for cid in cam_ids:
        c = calib[cid]
        m1, m2 = build_undistort_maps(c["K"], c["dist"], c["width"], c["height"])
        undistort_maps[cid] = (m1, m2)

    for fi, (frame_key, verts_np) in enumerate(zip(sel_keys, all_vertices)):
        label = hero_labels.get(indices[fi], "")
        print(f"  Frame {fi+1}/{len(sel_keys)}: {frame_key}  {label}")
        v_world = torch.as_tensor(verts_np, dtype=torch.float32, device="cuda")  # (V, 3)

        # UV-space 3D positions
        v_loop = v_world[ft_geom.flatten().to("cuda")].unsqueeze(0)  # (1, 62724, 3)
        with torch.no_grad():
            pos3d_uv, _ = dr.interpolate(v_loop, rast_out, ft_i32)
        # pos3d_uv: (1, H, W, 3)

        # UV-space normals
        normals = compute_smooth_normals(verts_np, faces).to("cuda")  # (V, 3)
        n_loop = normals[ft_geom.flatten().to("cuda")].unsqueeze(0)   # (1, 62724, 3)
        with torch.no_grad():
            norm_uv, _ = dr.interpolate(n_loop, rast_out, ft_i32)
        norm_uv = F.normalize(norm_uv, dim=-1)  # (1, H, W, 3)

        # Precompute camera-space clip vertices and depth for each camera
        # (reused below for depth buffer)
        faces_i32 = torch.as_tensor(faces.astype(np.int32), dtype=torch.int32, device="cuda")

        # Collect per-camera UV projections for this frame
        cam_colors    = {}
        cam_qualities = {}

        for cam_id in cam_ids:
            # Per-camera hero mode: skip if this frame isn't in this camera's hero set
            if per_cam_frames is not None and indices[fi] not in per_cam_frames.get(cam_id, []):
                continue

            cam = calib[cam_id]
            K, R, t = cam["K"], cam["R"], cam["t"]
            img_w, img_h = cam["width"], cam["height"]

            # Load frame and undistort before any pixel-space operation.
            # project_points / _verts_to_clip use pinhole K (undistorted space);
            # raw frames are from the distorted video — remap to match.
            frame_path = os.path.join(aligned_dir, cam_id, frame_key)
            if not os.path.exists(frame_path):
                continue
            frame_bgr = cv2.imread(frame_path)
            if frame_bgr is None:
                continue
            m1, m2 = undistort_maps[cam_id]
            frame_bgr = undistort_fast(frame_bgr, m1, m2)
            ds_w, ds_h = img_w // downsample, img_h // downsample
            frame_small = cv2.resize(frame_bgr, (ds_w, ds_h), interpolation=cv2.INTER_AREA)
            frame_t = torch.as_tensor(frame_small[:, :, ::-1].copy(),
                                       dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to("cuda") / 255.0

            # RVM mask — also undistorted before use (same pixel space as projection).
            mask_path = os.path.join(aligned_dir, mask_subdir, cam_id, frame_key)
            if os.path.exists(mask_path):
                mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                mask_gray = cv2.remap(mask_gray, m1, m2, interpolation=cv2.INTER_NEAREST)
                mask_small = cv2.resize(mask_gray, (ds_w, ds_h), interpolation=cv2.INTER_AREA)
                mask_t = torch.as_tensor(mask_small, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to("cuda")
            else:
                mask_t = torch.ones(1, 1, ds_h, ds_w, device="cuda") * 255.0

            # Project UV texel positions to this camera
            pts = pos3d_uv[0].reshape(-1, 3)
            px, depth = project_points(pts, K, R, t)
            px    = px.reshape(1, tex_H, tex_W, 2)
            depth = depth.reshape(1, tex_H, tex_W)

            grid_x = px[..., 0] / img_w * 2.0 - 1.0
            grid_y = px[..., 1] / img_h * 2.0 - 1.0
            grid = torch.stack([grid_x, grid_y], dim=-1)  # (1, H, W, 2)

            # Bounds and in-front check
            in_front  = depth > 0
            in_bounds = (grid[..., 0].abs() <= 1.0) & (grid[..., 1].abs() <= 1.0)

            # Self-occlusion depth test
            v_clip = _verts_to_clip(v_world.unsqueeze(0), K, R, t, img_w, img_h)
            with torch.no_grad():
                rast_cam, _ = dr.rasterize(glctx, v_clip, faces_i32, resolution=[ds_h, ds_w])
            R_t = torch.as_tensor(np.asarray(R), dtype=torch.float32, device="cuda")
            t_v = torch.as_tensor(np.asarray(t), dtype=torch.float32, device="cuda")
            v_cam_z = (v_world @ R_t.T + t_v)[:, 2:3]  # (V, 1)
            with torch.no_grad():
                depth_buf, _ = dr.interpolate(v_cam_z.unsqueeze(0).contiguous(),
                                               rast_cam.contiguous(), faces_i32.contiguous())
            depth_buf = depth_buf[..., 0].unsqueeze(1)  # (1, 1, ds_h, ds_w)
            depth_at_px = F.grid_sample(depth_buf, grid, mode="bilinear",
                                         align_corners=False, padding_mode="zeros").squeeze(1)
            not_occluded = depth <= depth_at_px + depth_tolerance

            # Segmentation IoU check (skipped in per_camera_hero mode — already filtered).
            if min_iou > 0 and per_cam_frames is None:
                mesh_sil = rast_cam[0, :, :, 3] > 0                         # (ds_h, ds_w) bool
                rvm_sil  = mask_t.squeeze() > (mask_threshold / 255.0)      # (ds_h, ds_w) bool
                intersection = (mesh_sil & rvm_sil).sum().float()
                union        = (mesh_sil | rvm_sil).sum().float()
                iou = (intersection / union.clamp(min=1)).item()
                if iou < min_iou:
                    print(f"      Skipping {cam_id} frame {frame_key}: IoU={iou:.3f} < {min_iou}")
                    continue

            valid = (in_front & in_bounds & not_occluded & valid_mask.unsqueeze(0)).float()

            # Cosine quality score
            cam_pos = torch.as_tensor(-np.asarray(R).T @ np.asarray(t),
                                       dtype=torch.float32, device="cuda")
            cam_dir = F.normalize(cam_pos.reshape(1, 1, 1, 3) - pos3d_uv, dim=-1)
            cos_raw = (norm_uv * cam_dir).sum(-1)
            facing  = (cos_raw > normal_threshold).float()
            cos_w   = cos_raw.clamp(min=0) * facing * valid  # (1, H, W)

            # Sample colour
            color = F.grid_sample(frame_t, grid, mode="bilinear",
                                   align_corners=False, padding_mode="zeros")  # (1, 3, H, W)

            # Mask gate
            mask_val = F.grid_sample(mask_t, grid, mode="bilinear",
                                      align_corners=False, padding_mode="zeros")
            mask_ok  = (mask_val.squeeze(1) > (mask_threshold / 255.0)).float()

            quality = cos_w * mask_ok  # (1, H, W)

            # Bias the front camera so it wins winner-take-all on front-facing texels
            if front_cam is not None and cam_id == front_cam:
                quality = quality * front_cam_bias

            cam_colors[cam_id]    = color.squeeze(0)  # (3, H, W)
            cam_qualities[cam_id] = quality           # (1, H, W)

        if not cam_colors:
            continue

        # Step 3.5 — optical flow UV-space registration
        if optical_flow and len(cam_colors) > 1:
            cam_colors, cam_qualities = register_with_optical_flow(
                cam_colors, cam_qualities, model_name=flow_model,
                flow_max_pixels=flow_max_pixels)

        # Step 4 — accumulate
        for ci, cam_id in enumerate(cam_colors):
            color   = cam_colors[cam_id]    # (3, H, W)
            quality = cam_qualities[cam_id]  # (1, H, W)
            source_id = fi * len(cam_ids) + ci

            if winner_take_all:
                update = (quality > 0) & (quality > quality_map)  # (1, H, W)
                color_map   = torch.where(update, color, color_map)
                quality_map = torch.where(update, quality, quality_map)
                source_map  = torch.where(update.squeeze(0),
                                           torch.full_like(source_map, source_id),
                                           source_map)
            else:
                color_sum  += quality * color
                weight_sum += quality

    # --- Finalise ---
    print("Normalising and saving texture...")
    if winner_take_all:
        texture_t = color_map  # (3, H, W) — already the winning colour
        no_coverage = (quality_map[0] < 0).cpu().numpy()
        source_np = source_map.cpu().numpy()
    else:
        texture_t = color_sum / weight_sum.clamp(min=1e-8)
        no_coverage = (weight_sum[0] == 0).cpu().numpy()
        source_np = np.full((tex_H, tex_W), -1, dtype=np.int32)

    texture_np  = (texture_t.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    texture_bgr = texture_np[:, :, ::-1].copy()
    texture_bgr[no_coverage] = 0

    # Step 5 — seam smoothing
    if winner_take_all and seam_blur_radius > 0:
        texture_bgr = smooth_seams(texture_bgr, source_np, blur_radius=seam_blur_radius)

    cv2.imwrite(output_path, texture_bgr)
    covered = (~no_coverage).sum()
    total   = no_coverage.size
    print(f"Texture saved → {output_path}  ({covered}/{total} texels, {covered/total*100:.1f}% coverage)")
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smplx-npz", required=True)
    parser.add_argument("--calib", required=True)
    parser.add_argument("--take-dir", required=True)
    parser.add_argument("--uv-npz", default=UV_NPZ_DEFAULT)
    parser.add_argument("--smplx-model-path", default="models/SMPLX")
    parser.add_argument("--tex-resolution", type=int, default=2048)
    parser.add_argument("--n-frames", type=int, default=50)
    parser.add_argument("--downsample", type=int, default=4)
    parser.add_argument("--mask-subdir", default="masks_rvm")
    parser.add_argument("--mask-threshold", type=int, default=50)
    parser.add_argument("--output", default=None)
    parser.add_argument("--debug-phase1", action="store_true")
    parser.add_argument("--cam", default=None,
                        help="Comma-separated camera IDs (default: all from calib)")
    parser.add_argument("--depth-tolerance", type=float, default=0.02,
                        help="Occlusion depth tolerance in metres (default 2 cm)")
    parser.add_argument("--normal-threshold", type=float, default=0.0,
                        help="Min dot(normal, cam_dir) to accept a texel")
    parser.add_argument("--frame-range", default=None,
                        help="START:END slice for frame pool, e.g. 373:413")
    # Hero frame selection
    parser.add_argument("--hero-frames", action="store_true",
                        help="Auto-select pose-diverse still keyframes (global)")
    parser.add_argument("--per-camera-hero", action="store_true",
                        help="Select hero frames independently per camera based on IoU")
    parser.add_argument("--n-arms-down", type=int, default=2)
    parser.add_argument("--n-arms-wide", type=int, default=2)
    parser.add_argument("--velocity-threshold", type=float, default=0.01,
                        help="Max mean vertex velocity (m/frame) for still-frame pool")
    # Accumulation
    parser.add_argument("--no-winner-take-all", dest="winner_take_all",
                        action="store_false", default=True,
                        help="Use weighted average instead of winner-take-all")
    parser.add_argument("--seam-blur-radius", type=int, default=3,
                        help="Gaussian blur half-width at camera seam pixels (0=off)")
    # Optical flow
    parser.add_argument("--optical-flow", action="store_true",
                        help="RAFT UV-space registration before accumulation")
    parser.add_argument("--flow-model", default="raft_large",
                        choices=["raft_large", "raft_small"])
    parser.add_argument("--flow-max-pixels", type=float, default=64,
                        help="Clamp flow vectors larger than this many pixels (0=off). "
                             "Prevents RAFT noise at coverage boundaries warping in garbage.")
    parser.add_argument("--min-iou", type=float, default=0.5,
                        help="Minimum IoU between projected mesh silhouette and RVM mask "
                             "to accept a camera view (0 = disabled, default 0.5)")
    # Front-camera priority
    parser.add_argument("--front-cam", default=None,
                        help="Camera ID of the front-facing camera; its quality scores are "
                             "multiplied by --front-cam-bias so it wins winner-take-all on "
                             "front-facing texels")
    parser.add_argument("--front-cam-bias", type=float, default=2.0,
                        help="Multiplicative bias applied to the front camera's quality "
                             "scores (default 2.0)")
    # Region protection
    parser.add_argument("--base-texture", default=None,
                        help="Existing texture PNG to initialise the UV canvas from; "
                             "use with --protect-region to keep specific areas unchanged")
    parser.add_argument("--protect-region", default=None,
                        choices=list(REGION_GROUPS.keys()),
                        help="Body region to protect from overwriting (requires --base-texture). "
                             f"Available: {', '.join(REGION_GROUPS.keys())}")
    args = parser.parse_args()

    cam_ids = [c.strip() for c in args.cam.split(",")] if args.cam else None
    frame_range = None
    if args.frame_range:
        s, e = args.frame_range.split(":")
        frame_range = (int(s), int(e))

    bake_texture(
        smplx_npz=args.smplx_npz,
        calib_path=args.calib,
        take_dir=args.take_dir,
        uv_npz=args.uv_npz,
        model_path=args.smplx_model_path,
        tex_resolution=args.tex_resolution,
        n_frames=args.n_frames,
        downsample=args.downsample,
        mask_subdir=args.mask_subdir,
        mask_threshold=args.mask_threshold,
        output_path=args.output,
        debug_phase1=args.debug_phase1,
        cam_ids=cam_ids,
        frame_range=frame_range,
        normal_threshold=args.normal_threshold,
        depth_tolerance=args.depth_tolerance,
        hero_frames=args.hero_frames,
        per_camera_hero=args.per_camera_hero,
        n_arms_down=args.n_arms_down,
        n_arms_wide=args.n_arms_wide,
        velocity_threshold=args.velocity_threshold,
        winner_take_all=args.winner_take_all,
        optical_flow=args.optical_flow,
        flow_model=args.flow_model,
        flow_max_pixels=args.flow_max_pixels,
        seam_blur_radius=args.seam_blur_radius,
        min_iou=args.min_iou,
        front_cam=args.front_cam,
        front_cam_bias=args.front_cam_bias,
        base_texture=args.base_texture,
        protect_region=args.protect_region,
    )


if __name__ == "__main__":
    main()
