"""High-quality single-frame face refit for the texture bake hero frame.

Extends fit_face_expr.py with three improvements:
  1. Targets a single frame only — the hero frame used for face texture baking
  2. Also optimises neck_pose (body_pose[11], the neck joint) in addition to
     expression and jaw_pose, allowing the head orientation to shift slightly
     without changing the global body alignment
  3. 100 L-BFGS steps and tighter regularisation for a closer fit

All other body parameters (betas, global_orient, transl, hands, remaining
body_pose joints) remain frozen. The result patches smplx_params.npz for the
single target frame only.

DISTORTION NOTE: MediaPipe runs on undistorted cam0 frames. project_points()
uses pinhole K — undistorted pixel space. Consistent with fit_face_expr.py.

Usage:
    python -m smplx_fit.fit_face_hero \\
        --smplx-npz /tmp/smplx_full/aligned/pose2d_sil/smplx/smplx_params.npz \\
        --calib     output/calibration/20260908_174749_7cam.json \\
        --take-dir  /tmp/smplx_full \\
        --frame     127
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import calibrate  # noqa: E402

from hand_pose.hand_multiview import build_undistort_maps, undistort_fast  # noqa: E402
from smplx_fit import model as smplx_model  # noqa: E402
from smplx_fit.texture.fit_face_expr import _load_head_vert_indices, _mediapipe_on_crop  # noqa: E402
from smplx_fit.silhouette import project_points  # noqa: E402

# neck is body_pose joint 11 (0-indexed; joint 12 in SMPL-X full ordering)
_NECK_IDX = 11


def _forward_with_face_neck(model_layer, betas, go, bp, lh, rh, tr, expr, jaw, neck):
    """SMPL-X forward including expression, jaw_pose, and neck rotation.

    neck: (1, 3) axis-angle, replaces body_pose[:, _NECK_IDX, :].
    """
    aa = smplx_model.axis_angle_to_rotmat
    bp_clone = bp.clone()
    bp_clone[:, _NECK_IDX, :] = neck
    return model_layer(
        betas=betas,
        global_orient=aa(go),
        body_pose=aa(bp_clone),
        left_hand_pose=aa(lh),
        right_hand_pose=aa(rh),
        transl=tr,
        expression=expr,
        jaw_pose=aa(jaw),
        return_verts=True,
    )


def _lmk_positions(model_layer, verts):
    """Compute 51 FLAME landmark positions from mesh vertices via barycentric coords.

    Returns (51, 3) world-space positions.
    """
    lmk_fi   = model_layer.lmk_faces_idx    # (51,) face indices
    lmk_bary = model_layer.lmk_bary_coords  # (51, 3) barycentric weights
    faces    = torch.as_tensor(model_layer.faces.astype(np.int64),
                               dtype=torch.long, device=verts.device)  # (F, 3)
    face_verts = verts[faces[lmk_fi]]       # (51, 3, 3)
    return (lmk_bary.unsqueeze(-1) * face_verts).sum(dim=1)  # (51, 3)


def fit_face_hero(
    smplx_npz,
    calib_path,
    take_dir,
    frame_idx,
    cam_id="cam0",
    model_path="models/SMPLX",
    n_lbfgs_steps=100,
    corr_dist_threshold=30.0,
    min_correspondences=30,
    expr_reg=0.01,
    jaw_reg=0.005,
    neck_reg=0.01,
    save_debug_image=True,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hero-frame face refit  frame={frame_idx}  device={device}")

    # -----------------------------------------------------------------------
    # Load fit
    # -----------------------------------------------------------------------
    data       = np.load(smplx_npz, allow_pickle=True)
    frame_keys = list(data["frame_keys"])
    n_frames   = len(frame_keys)

    if frame_idx < 0 or frame_idx >= n_frames:
        raise ValueError(f"frame_idx {frame_idx} out of range [0, {n_frames})")

    frame_key = frame_keys[frame_idx]

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

    K_t = torch.as_tensor(K,     dtype=torch.float32, device=device)
    R_t = torch.as_tensor(R,     dtype=torch.float32, device=device)
    t_t = torch.as_tensor(t_vec, dtype=torch.float32, device=device)

    # -----------------------------------------------------------------------
    # Head vertex indices
    # -----------------------------------------------------------------------
    head_idx_np = _load_head_vert_indices()
    head_idx_t  = torch.as_tensor(head_idx_np, dtype=torch.long, device=device)

    # -----------------------------------------------------------------------
    # Load and undistort hero frame
    # -----------------------------------------------------------------------
    img_path = os.path.join(take_dir, "aligned", cam_id, frame_key)
    img_bgr  = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Frame not found: {img_path}")
    img_u = undistort_fast(img_bgr, map1, map2)
    print(f"Loaded frame: {frame_key}")

    # -----------------------------------------------------------------------
    # SMPL-X forward at current params (zero expr/jaw) for face crop bbox
    # -----------------------------------------------------------------------
    go_fi = _t("global_orient", frame_idx).detach()
    bp_fi = _t("body_pose",     frame_idx).detach()
    lh_fi = _t("lhand_pose",    frame_idx).detach()
    rh_fi = _t("rhand_pose",    frame_idx).detach()
    tr_fi = _t("transl",        frame_idx).detach()

    # Init expr/jaw from existing fit (not zeros — warm start)
    expr_init = _t("expression", frame_idx).detach()
    jaw_init  = _t("jaw_pose",   frame_idx).detach()
    neck_init = bp_fi[:, _NECK_IDX, :].clone().detach()

    with torch.no_grad():
        out0 = _forward_with_face_neck(
            model_layer, betas_shared,
            go_fi, bp_fi, lh_fi, rh_fi, tr_fi,
            expr_init, jaw_init, neck_init,
        )

    head_verts_3d = out0.vertices[0, head_idx_t]
    uv0, depth0 = project_points(head_verts_3d, K, R, t_vec)
    in_front = (depth0 > 0.01).cpu().numpy()

    if in_front.sum() < 10:
        raise RuntimeError("Too few head vertices in front of camera")

    vis_uv = uv0.cpu().numpy()[in_front]
    pad_px = 200
    cx1 = int(vis_uv[:, 0].min()) - pad_px
    cx2 = int(vis_uv[:, 0].max()) + pad_px
    cy1 = int(vis_uv[:, 1].min()) - pad_px
    cy2 = int(vis_uv[:, 1].max()) + pad_px

    # -----------------------------------------------------------------------
    # MediaPipe with iris landmarks
    # -----------------------------------------------------------------------
    import mediapipe as mp
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,   # adds 10 iris landmarks (indices 468-477)
        min_detection_confidence=0.5,
    )
    mp_pts = _mediapipe_on_crop(img_u, cx1, cy1, cx2, cy2, face_mesh)
    face_mesh.close()

    if mp_pts is None:
        raise RuntimeError(f"MediaPipe failed to detect face in frame {frame_key}")

    n_mp = len(mp_pts)
    print(f"MediaPipe detected {n_mp} landmarks  ({n_mp - 468} iris)")

    # -----------------------------------------------------------------------
    # Nearest-neighbour: MediaPipe pts → visible SMPL-X head vertices
    # -----------------------------------------------------------------------
    in_bounds = ((uv0[:, 0] >= 0) & (uv0[:, 0] < img_w) &
                 (uv0[:, 1] >= 0) & (uv0[:, 1] < img_h))
    visible    = (torch.as_tensor(in_front, device=device) & in_bounds).cpu().numpy()

    if visible.sum() < 50:
        raise RuntimeError("Too few head vertices visible in hero frame")

    from scipy.spatial import KDTree
    uv_np      = uv0.cpu().numpy()
    uv_visible = uv_np[visible]
    vis_indices = np.where(visible)[0]

    tree = KDTree(uv_visible)
    dist, nn = tree.query(mp_pts)
    good_mask = dist < corr_dist_threshold

    if good_mask.sum() < min_correspondences:
        raise RuntimeError(
            f"Only {good_mask.sum()} correspondences (need ≥ {min_correspondences})")

    tgt_pts  = torch.as_tensor(mp_pts[good_mask], dtype=torch.float32, device=device)
    vert_idx = torch.as_tensor(
        head_idx_np[vis_indices[nn[good_mask]]], dtype=torch.long, device=device)

    print(f"Correspondences: {good_mask.sum()}/{n_mp}  "
          f"(within {corr_dist_threshold}px)")

    # -----------------------------------------------------------------------
    # Before-fit reprojection error on 68 FLAME landmarks
    # -----------------------------------------------------------------------
    with torch.no_grad():
        out_before = _forward_with_face_neck(
            model_layer, betas_shared,
            go_fi, bp_fi, lh_fi, rh_fi, tr_fi,
            expr_init, jaw_init, neck_init,
        )
        lmk3d_before = _lmk_positions(model_layer, out_before.vertices[0])
        cam_before   = lmk3d_before @ R_t.T + t_t
        px_before    = cam_before @ K_t.T
        z_b          = px_before[:, 2:].clamp(min=1e-4)
        uv_before    = px_before[:, :2] / z_b  # (68, 2)

    # -----------------------------------------------------------------------
    # L-BFGS: expression + jaw + neck
    # -----------------------------------------------------------------------
    expr_fi = expr_init.clone().requires_grad_(True)
    jaw_fi  = jaw_init.clone().requires_grad_(True)
    neck_fi = neck_init.clone().requires_grad_(True)

    optimizer = torch.optim.LBFGS(
        [expr_fi, jaw_fi, neck_fi], lr=0.3, max_iter=n_lbfgs_steps,
        history_size=15, line_search_fn="strong_wolfe",
    )

    final_loss = [float("inf")]

    def closure():
        optimizer.zero_grad()
        out = _forward_with_face_neck(
            model_layer, betas_shared,
            go_fi, bp_fi, lh_fi, rh_fi, tr_fi,
            expr_fi, jaw_fi, neck_fi,
        )
        verts = out.vertices[0]
        v3    = verts[vert_idx]
        cam_v = v3 @ R_t.T + t_t
        px_v  = cam_v @ K_t.T
        z_v   = px_v[:, 2:].clamp(min=1e-4)
        uv2   = px_v[:, :2] / z_v
        reproj = (uv2 - tgt_pts).pow(2).sum(dim=1).mean()
        reg    = (expr_reg * expr_fi.pow(2).mean()
                  + jaw_reg  * jaw_fi.pow(2).mean()
                  + neck_reg * neck_fi.pow(2).mean())
        loss = reproj + reg
        loss.backward()
        final_loss[0] = float(loss.item())
        return loss

    optimizer.step(closure)

    # -----------------------------------------------------------------------
    # After-fit reprojection error on 68 FLAME landmarks
    # -----------------------------------------------------------------------
    with torch.no_grad():
        bp_after       = bp_fi.clone()
        bp_after[:, _NECK_IDX, :] = neck_fi
        out_after = _forward_with_face_neck(
            model_layer, betas_shared,
            go_fi, bp_fi, lh_fi, rh_fi, tr_fi,
            expr_fi, jaw_fi, neck_fi,
        )
        lmk3d_after = _lmk_positions(model_layer, out_after.vertices[0])
        cam_after   = lmk3d_after @ R_t.T + t_t
        px_after    = cam_after @ K_t.T
        z_a         = px_after[:, 2:].clamp(min=1e-4)
        uv_after    = px_after[:, :2] / z_a  # (68, 2)

    # Measure against their nearest MediaPipe counterparts
    tree2 = KDTree(mp_pts)
    _, nn_b = tree2.query(uv_before.cpu().numpy())
    _, nn_a = tree2.query(uv_after.cpu().numpy())
    err_before = np.linalg.norm(
        uv_before.cpu().numpy() - mp_pts[nn_b], axis=1).mean()
    err_after  = np.linalg.norm(
        uv_after.cpu().numpy()  - mp_pts[nn_a], axis=1).mean()

    print(f"\nFinal loss:  {final_loss[0]:.4f}")
    print(f"FLAME 68-lmk reprojection error:  "
          f"before={err_before:.1f}px  after={err_after:.1f}px  "
          f"(Δ={err_before - err_after:.1f}px)")
    print(f"expr  range: [{expr_fi.min().item():.3f}, {expr_fi.max().item():.3f}]")
    print(f"jaw   norm:  {jaw_fi.norm().item():.4f}")
    print(f"neck  norm:  {neck_fi.norm().item():.4f}")

    # -----------------------------------------------------------------------
    # Debug image: annotate hero frame with landmarks before/after
    # -----------------------------------------------------------------------
    if save_debug_image:
        debug = img_u.copy()
        for pt in uv_before.cpu().numpy():
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < img_w and 0 <= y < img_h:
                cv2.circle(debug, (x, y), 4, (0, 0, 255), -1)  # red = before
        for pt in uv_after.cpu().numpy():
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < img_w and 0 <= y < img_h:
                cv2.circle(debug, (x, y), 4, (0, 255, 0), -1)  # green = after
        for pt in mp_pts:
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < img_w and 0 <= y < img_h:
                cv2.circle(debug, (x, y), 2, (255, 200, 0), -1)  # cyan = MediaPipe
        debug_path = smplx_npz.replace("smplx_params.npz", f"debug_hero_{frame_idx}.jpg")
        cv2.imwrite(debug_path, debug)
        print(f"Debug image → {os.path.basename(debug_path)}")
        print("  Red=before, Green=after, Cyan=MediaPipe")

    # -----------------------------------------------------------------------
    # Save: back up, then patch this frame's params in-place
    # -----------------------------------------------------------------------
    backup = smplx_npz.replace(".npz", "_pre_hero.npz")
    if not os.path.exists(backup):
        import shutil
        shutil.copy2(smplx_npz, backup)
        print(f"Backed up → {os.path.basename(backup)}")

    params = dict(data)
    expr_arr = data["expression"].copy()
    jaw_arr  = data["jaw_pose"].copy()
    bp_arr   = data["body_pose"].copy()

    expr_arr[frame_idx] = expr_fi.detach().cpu().numpy()
    jaw_arr[frame_idx]  = jaw_fi.detach().cpu().numpy()
    bp_arr[frame_idx, _NECK_IDX, :] = neck_fi.detach().cpu().numpy()

    params["expression"] = expr_arr
    params["jaw_pose"]   = jaw_arr
    params["body_pose"]  = bp_arr

    np.savez(smplx_npz.replace(".npz", ""), **params)
    print(f"Saved patched params → {smplx_npz}  (frame {frame_idx} only)")

    return expr_fi.detach(), jaw_fi.detach(), neck_fi.detach()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smplx-npz",    required=True)
    parser.add_argument("--calib",        required=True)
    parser.add_argument("--take-dir",     required=True)
    parser.add_argument("--frame",        type=int, required=True,
                        help="Frame index (0-based) — typically the face bake hero frame")
    parser.add_argument("--cam",          default="cam0")
    parser.add_argument("--smplx-model-path", default="models/SMPLX")
    parser.add_argument("--lbfgs-steps",  type=int,   default=100)
    parser.add_argument("--corr-dist",    type=float, default=30.0)
    parser.add_argument("--expr-reg",     type=float, default=0.01)
    parser.add_argument("--jaw-reg",      type=float, default=0.005)
    parser.add_argument("--neck-reg",     type=float, default=0.01)
    parser.add_argument("--no-debug",     action="store_true",
                        help="Skip saving the annotated debug image")
    args = parser.parse_args()

    fit_face_hero(
        smplx_npz   = args.smplx_npz,
        calib_path  = args.calib,
        take_dir    = args.take_dir,
        frame_idx   = args.frame,
        cam_id      = args.cam,
        model_path  = args.smplx_model_path,
        n_lbfgs_steps       = args.lbfgs_steps,
        corr_dist_threshold = args.corr_dist,
        expr_reg    = args.expr_reg,
        jaw_reg     = args.jaw_reg,
        neck_reg    = args.neck_reg,
        save_debug_image = not args.no_debug,
    )


if __name__ == "__main__":
    main()
