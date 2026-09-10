"""Dependency-free silhouette rendering for Phase 3 of the EasyMoCap-
inspired plan (see the approved plan / conversation history).

PyTorch3D -- the standard differentiable mesh rasterizer, and what
EasyMoCap itself uses for this -- was evaluated and rejected for this
project: it ships no PyPI wheel and no win-64 conda package (confirmed by
direct query against both index/channel this session), so the only path in
is a from-source build via VS Build Tools with no CUDA toolkit present --
exactly the fragile "historically common outcome on Windows" the plan
flagged as the trigger to fall back rather than sink more effort into the
install itself.

This module implements the plan's documented fallback instead: a crude but
fully-differentiable, dependency-free "soft splat" renderer --
  1. Sample points across the mesh surface (fixed face indices + barycentric
     weights, chosen ONCE at import time and reused every call -- only the
     vertex positions vary per forward pass, so this stays a plain linear
     combination of vertex positions, differentiable w.r.t. them).
  2. Project each sample point into a camera view with the SAME K/R/t
     projection convention this project already uses everywhere else for
     reprojection (hand_pose.hand_multiview.build_projection_matrix; world
     points project via x_cam = R @ X_world + t, matching calibrate.
     load_calibration_output's schema).
  3. Splat each projected point as a small soft 2D Gaussian into a
     downsampled occupancy image, compared against a Phase 2 segmentation
     mask via a pixel-wise loss.

This is cruder than true triangle rasterization (a point cloud of surface
samples, not a filled/occluded-aware silhouette -- notably, it does NOT
handle self-occlusion: a sample point on the far side of a limb splats onto
the image exactly as if it were visible, silently double-counting depth
layers the mask can only see once). Acceptable for this project's
documented purpose -- pulling body_pose's normally-unconstrained leg slots
toward the signer's real stance via silhouette overlap, not a
photorealistic renderer -- but NOT a drop-in substitute for PyTorch3D
elsewhere.
"""
import os

import cv2
import numpy as np
import torch

from hand_pose import hand_multiview as hmv
from smplx_fit import losses

N_SURFACE_SAMPLES = 8000
SAMPLE_SEED = 0


def _face_areas(vertices_np, faces_np):
    v0, v1, v2 = vertices_np[faces_np[:, 0]], vertices_np[faces_np[:, 1]], vertices_np[faces_np[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)


def build_surface_samples(faces, rest_vertices_np, n_samples=N_SURFACE_SAMPLES, seed=SAMPLE_SEED):
    """Precomputes WHICH faces to sample and their barycentric weights, once,
    using face-area-weighted sampling over the rest-pose template so sample
    density is roughly uniform across the mesh surface (SMPL-X's faces vary
    hugely in size -- fingers vs. torso -- so uniform-over-faces sampling
    would hugely oversample small parts). These indices/weights are fixed
    constants reused for every subsequent call to sample_surface_points --
    only actual vertex positions change per frame/hypothesis, not which
    mesh location each of the N_SURFACE_SAMPLES samples represents.

    Returns (face_idx (n_samples,) int64 tensor, bary (n_samples, 3) float32
    tensor, each row summing to 1).
    """
    faces_np = np.asarray(faces)
    areas = _face_areas(rest_vertices_np, faces_np)
    probs = areas / areas.sum()
    rng = np.random.default_rng(seed)
    face_idx = rng.choice(len(faces_np), size=n_samples, p=probs)

    # Uniform sampling over a triangle via two uniforms (Turk 1990, the
    # standard sqrt-trick): u,v ~ U(0,1); if u+v>1, fold back into the
    # triangle. Barycentric weights are (1-u-v, u, v) w.r.t. (v0, v1, v2).
    u = rng.random(n_samples)
    v = rng.random(n_samples)
    fold = u + v > 1.0
    u[fold], v[fold] = 1.0 - u[fold], 1.0 - v[fold]
    bary = np.stack([1.0 - u - v, u, v], axis=1)

    return torch.as_tensor(face_idx, dtype=torch.int64), torch.as_tensor(bary, dtype=torch.float32)


def sample_surface_points(vertices, faces, face_idx, bary):
    """vertices: (B, V, 3) or (V, 3) torch tensor (whatever the current
    optimization hypothesis's mesh vertices are -- gradients flow through
    this). faces: (F, 3) int array/tensor (SMPL-X's fixed topology).
    face_idx/bary: from build_surface_samples. Returns (B, n_samples, 3) or
    (n_samples, 3) sampled surface points, same batch-ness as `vertices`.
    """
    faces_t = torch.as_tensor(np.asarray(faces), dtype=torch.int64, device=vertices.device)
    tri = faces_t[face_idx]  # (n_samples, 3)
    batched = vertices.dim() == 3
    v = vertices if batched else vertices.unsqueeze(0)
    v0, v1, v2 = v[:, tri[:, 0]], v[:, tri[:, 1]], v[:, tri[:, 2]]  # each (B, n_samples, 3)
    b = bary.to(vertices.device)
    pts = v0 * b[:, 0:1] + v1 * b[:, 1:2] + v2 * b[:, 2:3]
    return pts if batched else pts.squeeze(0)


def project_points(points_world, K, R, t):
    """points_world: (..., 3) torch tensor. K, R: (3,3), t: (3,) numpy or
    torch, this project's standard extrinsics convention (x_cam = R @
    X_world + t; see calibrate.load_calibration_output's schema and
    hand_pose.hand_multiview.build_projection_matrix -- reused here rather
    than re-derived, for consistency with every other reprojection in this
    codebase). Returns (..., 2) pixel xy and (...,) camera-space depth (for
    an optional in-front-of-camera validity mask).
    """
    device, dtype = points_world.device, points_world.dtype
    K_t = torch.as_tensor(np.asarray(K), dtype=dtype, device=device)
    R_t = torch.as_tensor(np.asarray(R), dtype=dtype, device=device)
    t_t = torch.as_tensor(np.asarray(t), dtype=dtype, device=device).reshape(3)

    cam_pts = points_world @ R_t.T + t_t
    proj = cam_pts @ K_t.T
    z = proj[..., 2].clamp(min=1e-6)
    uv = proj[..., :2] / z.unsqueeze(-1)
    return uv, cam_pts[..., 2]


def render_silhouette(uv, image_size, out_size=(128, 128), sigma_px=1.0, valid_mask=None):
    """Soft-splats projected 2D points into a downsampled occupancy image,
    fully differentiable w.r.t. `uv`. uv: (N, 2) pixel coords in the FULL
    image_size=(W,H) frame (as project_points returns) -- rescaled here to
    out_size before splatting, since splatting is O(N * out_H*out_W) and
    full-resolution (e.g. 3840x2160) would be intractable; out_size=(128,128)
    keeps a single-frame/single-camera call fast enough for interactive
    validation (see this module's docstring on scope: Phase 3 is validated
    on single frames before ever running across a full optimization stage,
    per the approved plan -- full-stage performance is a separate, later
    concern if this approach proves worthwhile at all).

    Returns an (out_H, out_W) tensor in [0, 1] -- soft-OR of each point's
    Gaussian contribution (1 - product(1 - gaussian_i)), so overlapping
    points saturate toward 1 rather than summing past it.

    `sigma_px` is in OUTPUT-GRID pixel units, not full-resolution pixels --
    confirmed empirically this session that reusing the same sigma_px
    across different out_size values is wrong: at a fixed sigma_px, a
    smaller out_size makes each splat cover a much larger FRACTION of the
    image (e.g. sigma_px=2.0 looked reasonable at out_size=(64,36) but
    inflated the rendered silhouette to ~3x the image at out_size=(32,18),
    tanking IoU against a real mask from 0.5+ to ~0.2). Scale sigma_px down
    proportionally with out_size if you shrink it for speed.
    """
    W, H = image_size
    out_w, out_h = out_size
    sx, sy = out_w / W, out_h / H
    device, dtype = uv.device, uv.dtype

    u = uv[:, 0] * sx
    v = uv[:, 1] * sy
    if valid_mask is not None:
        u, v = u[valid_mask], v[valid_mask]
    if u.numel() == 0:
        return torch.zeros(out_h, out_w, dtype=dtype, device=device)

    gy = torch.arange(out_h, dtype=dtype, device=device).view(out_h, 1, 1)
    gx = torch.arange(out_w, dtype=dtype, device=device).view(1, out_w, 1)
    du = gx - u.view(1, 1, -1)  # (out_h, out_w, N) -- broadcast over rows
    dv = gy - v.view(1, 1, -1)
    sq_dist = du * du + dv * dv
    gauss = torch.exp(-0.5 * sq_dist / (sigma_px * sigma_px))  # (out_h, out_w, N)
    occupancy = 1.0 - torch.prod(1.0 - gauss.clamp(max=1.0 - 1e-6), dim=-1)
    return occupancy


def load_silhouette_data(take_dir, calib, cam_ids, frame_keys, out_size=(64, 36), device=None):
    """Precomputes, ONCE before optimization starts, everything the
    per-iteration silhouette term needs: each camera's (K, R, t, image
    size) and a (n_frames, out_h, out_w) tensor of that camera's cached
    Phase 2 masks -- downsampled to `out_size` up front so every subsequent
    LBFGS closure call is pure tensor math, no per-iteration disk I/O or
    cv2.resize calls.

    frame_keys MUST be in the exact order (and already sliced to the exact
    range) the caller will later pass as the optimized batch's vertices --
    a mismatch here would silently compare each frame's rendered silhouette
    against the WRONG real mask.

    Returns (cams, masks, valid): cams[cam_id] = (K, R, t, (w, h));
    masks[cam_id] = (n_frames, out_h, out_w) float32 tensor in [0, 1] (0
    where no cached mask was found); valid[cam_id] = (n_frames,) bool numpy
    array marking which frames actually had a mask, so a missing mask reads
    as "skip this (frame, camera) term", not as "empty person" (real
    negative signal).
    """
    cams, masks, valid = {}, {}, {}
    out_w, out_h = out_size
    for cam_id in cam_ids:
        if cam_id not in calib:
            continue
        c = calib[cam_id]
        w, h = hmv.get_image_size(take_dir, cam_id, frame_keys[0])
        cam_masks = np.zeros((len(frame_keys), out_h, out_w), dtype=np.float32)
        cam_valid = np.zeros(len(frame_keys), dtype=bool)
        for i, frame_key in enumerate(frame_keys):
            mask_path = os.path.join(take_dir, "aligned", "masks", cam_id, frame_key)
            m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            cam_masks[i] = cv2.resize(m, (out_w, out_h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
            cam_valid[i] = True
        if not cam_valid.any():
            continue  # no masks cached for this camera at all -- skip it entirely
        cams[cam_id] = (c["K"], c["R"], c["t"], (w, h))
        masks[cam_id] = torch.as_tensor(cam_masks, device=device)
        valid[cam_id] = cam_valid
    return cams, masks, valid


def compute_silhouette_term(vertices_batch, faces, face_idx, bary, cams, masks, valid, sigma_px=2.0):
    """Vectorized (camera × frame) silhouette loss with automatic frame
    chunking to stay within VRAM bounds.

    The dominant memory cost is the 5-D distance grid (nc, chunk, out_h,
    out_w, N). At default settings (nc=7, out_h=18, out_w=32, N=1500) this
    is ~23 MB per frame, so the chunk size is auto-capped at ~86 frames to
    stay under a 2 GB budget. For the 40-frame test slice this means no
    chunking; for the full 786-frame take, ~9 chunks of ~87 frames.

    When chunking, each chunk is wrapped in torch.utils.checkpoint so only
    one chunk's activations are live at a time during backward (~2x the
    chunk's forward memory, not all-chunks combined). This trades ~2x
    compute for bounded VRAM regardless of take length.

    vertices_batch: (n_frames, V, 3). cams/masks/valid: from
    load_silhouette_data, frame axis must match vertices_batch. Returns
    (mean_loss_over_valid_pairs, n_pairs).
    """
    from torch.utils.checkpoint import checkpoint as torch_checkpoint

    device = vertices_batch.device
    dtype = vertices_batch.dtype
    pts = sample_surface_points(vertices_batch, faces, face_idx, bary)  # (nf, N, 3)
    nf, N = pts.shape[0], pts.shape[1]

    cam_ids_list = list(cams.keys())
    nc = len(cam_ids_list)
    if nc == 0:
        return torch.zeros((), dtype=dtype, device=device), 0

    K_t = torch.stack([
        torch.as_tensor(np.asarray(cams[c][0]), dtype=dtype, device=device)
        for c in cam_ids_list
    ])  # (nc, 3, 3)
    R_t = torch.stack([
        torch.as_tensor(np.asarray(cams[c][1]), dtype=dtype, device=device)
        for c in cam_ids_list
    ])  # (nc, 3, 3)
    t_t = torch.stack([
        torch.as_tensor(np.asarray(cams[c][2]), dtype=dtype, device=device).reshape(3)
        for c in cam_ids_list
    ])  # (nc, 3)
    image_sizes = [cams[c][3] for c in cam_ids_list]
    out_h, out_w = masks[cam_ids_list[0]].shape[1:]
    masks_stack = torch.stack([masks[c].to(device) for c in cam_ids_list])  # (nc, nf, out_h, out_w)
    valid_np = np.stack([valid[c] for c in cam_ids_list])  # (nc, nf) bool

    scale = torch.tensor(
        [[out_w / W, out_h / H] for W, H in image_sizes],
        dtype=dtype, device=device,
    )  # (nc, 2)

    # Auto chunk size: cap 5-D distance grid at ~2 GB
    elems_per_frame = nc * out_h * out_w * N
    chunk_size = max(1, min(nf, int(2e9 / 4 / elems_per_frame)))
    use_ckpt = nf > chunk_size  # only checkpoint when actually splitting

    # _occ captures R_t/K_t/t_t/scale/N/nc/out_h/out_w/sigma_px via closure;
    # pts_chunk is the only gradient-tracking input.
    def _occ(pts_chunk):
        ch = pts_chunk.shape[0]
        pts_exp = pts_chunk.unsqueeze(0).expand(nc, -1, -1, -1)
        cam_pts = (torch.matmul(pts_exp, R_t.unsqueeze(1).permute(0, 1, 3, 2))
                   + t_t[:, None, None, :])                              # (nc, ch, N, 3)
        proj = torch.matmul(cam_pts, K_t.unsqueeze(1).permute(0, 1, 3, 2))
        z_pos = cam_pts[..., 2] > 0
        z = proj[..., 2].clamp(min=1e-6)
        uv_s = (proj[..., :2] / z.unsqueeze(-1)) * scale[:, None, None, :]  # (nc, ch, N, 2)
        gx = torch.arange(out_w, dtype=pts_chunk.dtype, device=pts_chunk.device)
        gy = torch.arange(out_h, dtype=pts_chunk.dtype, device=pts_chunk.device)
        u = uv_s[..., 0].reshape(nc, ch, 1, 1, N)
        v = uv_s[..., 1].reshape(nc, ch, 1, 1, N)
        sq_dist = ((gx.reshape(1, 1, 1, out_w, 1) - u) ** 2
                   + (gy.reshape(1, 1, out_h, 1, 1) - v) ** 2)         # (nc, ch, oh, ow, N)
        gauss = torch.exp(-0.5 * sq_dist / (sigma_px ** 2))
        gauss = gauss * z_pos.reshape(nc, ch, 1, 1, N).to(pts_chunk.dtype)
        return 1.0 - torch.prod(1.0 - gauss.clamp(max=1.0 - 1e-6), dim=-1)  # (nc, ch, oh, ow)

    total = torch.zeros((), dtype=dtype, device=device)
    count = 0
    for f0 in range(0, nf, chunk_size):
        pts_ch = pts[f0:f0 + chunk_size]
        masks_ch = masks_stack[:, f0:f0 + chunk_size]
        valid_ch = valid_np[:, f0:f0 + chunk_size]
        occ = (torch_checkpoint(_occ, pts_ch, use_reentrant=False)
               if use_ckpt else _occ(pts_ch))
        per_pair = ((occ - masks_ch) ** 2).mean(dim=(-2, -1))
        total = total + (per_pair * torch.as_tensor(valid_ch, dtype=dtype, device=device)).sum()
        count += int(valid_ch.sum())

    if count == 0:
        return torch.zeros((), dtype=dtype, device=device), 0
    return total / count, count
