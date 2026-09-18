"""Fit SMPL-X expression and jaw pose to MediaPipe face landmarks on cam0.

SMPL-X incorporates FLAME for the face region. The main fitting pipeline
(fit_take.py) never optimises expression or jaw_pose — they default to zero,
leaving the face in a rigid neutral template. This script fits those 13
face parameters (expression: 10, jaw_pose: 3) per frame to 2D face landmarks
detected by MediaPipe FaceMesh, giving the face geometry that actually matches
the person rather than the FLAME template mean.

The fitting is post-hoc: body pose, shape, hands, and global RT are all frozen
at whatever fit_take.py produced. Only expression and jaw_pose move.

DISTORTION NOTE: MediaPipe runs on undistorted cam0 frames. project_points()
uses pinhole K — undistorted pixel space. The two are consistent.

Usage:
    python -m smplx_fit.fit_face_expr \\
        --smplx-npz /tmp/smplx_full/aligned/pose2d_sil/smplx/smplx_params.npz \\
        --calib     output/calibration/20260908_174749_7cam.json \\
        --take-dir  /tmp/smplx_full \\
        --cam       cam0

Output: updates expression and jaw_pose in the npz in-place (original backed up
as smplx_params_pre_face.npz).
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import calibrate  # noqa: E402

from hand_pose.hand_multiview import build_undistort_maps, undistort_fast  # noqa: E402
from smplx_fit import model as smplx_model  # noqa: E402
from smplx_fit.silhouette import project_points  # noqa: E402

# smplx_vert_segmentation.json — found in several sibling repos
_SEG_CANDIDATES = [
    "/home/fmalmb/CODE/sequence_rendering/smplx_models/smplx/smplx_vert_segmentation.json",
    "/home/fmalmb/CODE/temporal_denoising/preprocessing/representations/smplx_models/smplx/smplx_vert_segmentation.json",
    "/home/fmalmb/CODE/sl_reconstruction/data_preprocessing/smplx/smplx_models/smplx/smplx_vert_segmentation.json",
]


def _load_head_vert_indices():
    for p in _SEG_CANDIDATES:
        if os.path.exists(p):
            with open(p) as f:
                seg = json.load(f)
            idx = np.array(seg["head"], dtype=np.int64)
            print(f"Head segment: {len(idx)} vertices  (from {os.path.basename(os.path.dirname(p))})")
            return idx
    raise FileNotFoundError("smplx_vert_segmentation.json not found in any known location")


def _forward_with_face(model_layer, betas, go, bp, lh, rh, tr, expr, jaw):
    """SMPL-X forward including expression + jaw_pose.

    All pose inputs are axis-angle (B,*,3); expression is (B,10).
    """
    aa = smplx_model.axis_angle_to_rotmat
    return model_layer(
        betas=betas,
        global_orient=aa(go),              # (B,3,3)
        body_pose=aa(bp),                  # (B,21,3,3)
        left_hand_pose=aa(lh),             # (B,15,3,3)
        right_hand_pose=aa(rh),            # (B,15,3,3)
        transl=tr,
        expression=expr,                   # (B,10)
        jaw_pose=aa(jaw),                  # (B,3,3) — SMPLXLayer expects 3x3 for jaw
        return_verts=True,
    )


def _mediapipe_on_crop(img_u, cx1, cy1, cx2, cy2, face_mesh):
    """Run MediaPipe on a crop, return (468, 2) landmarks in full-image px coords.

    MediaPipe fails on full 4K frames (face too small relative to image).
    Crop the head region (from SMPL-X head vertex projection bbox + padding),
    run MediaPipe on the crop, then offset results back to full-image coords.

    Returns None if no face is detected.
    """
    h_img, w_img = img_u.shape[:2]
    x1 = max(0, cx1)
    y1 = max(0, cy1)
    x2 = min(w_img, cx2)
    y2 = min(h_img, cy2)
    crop = img_u[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    crop_h, crop_w = crop.shape[:2]
    result = face_mesh.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    if not result.multi_face_landmarks:
        return None
    lms = result.multi_face_landmarks[0].landmark
    # Coordinates are normalised to crop size — convert to full-image pixels
    pts = np.array([[lm.x * crop_w + x1,
                     lm.y * crop_h + y1] for lm in lms], dtype=np.float32)
    return pts


def fit_face_expr(
    smplx_npz,
    calib_path,
    take_dir,
    cam_id="cam0",
    model_path="models/SMPLX",
    frames=None,              # None = all; (start, end) = slice
    n_lbfgs_steps=40,
    corr_dist_threshold=25.0, # max pixel distance for MP→vertex correspondence
    min_correspondences=30,
    expr_reg=0.05,            # L2 weight on expression
    jaw_reg=0.02,             # L2 weight on jaw_pose
    verbose=True,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------------------------------------------------
    # Load fit
    # -----------------------------------------------------------------------
    data     = np.load(smplx_npz, allow_pickle=True)
    n_frames = len(data["frame_keys"])
    frame_keys = list(data["frame_keys"])

    frame_indices = list(range(n_frames))
    if frames is not None:
        s, e = frames
        frame_indices = frame_indices[s:e]

    print(f"Fitting face expression on {len(frame_indices)} frames  (device={device})")

    # -----------------------------------------------------------------------
    # Load SMPL-X body params (frozen)
    # -----------------------------------------------------------------------
    gender    = str(data["gender"])
    num_betas = int(data["num_betas"])
    model_layer = smplx_model.load_layer(model_path, gender=gender, num_betas=num_betas)
    model_layer = model_layer.to(device)

    def _t(key, fi=None):
        v = data[key] if fi is None else data[key][fi:fi+1]
        return torch.as_tensor(np.asarray(v), dtype=torch.float32, device=device)

    betas_shared = _t("betas").expand(1, -1)  # (1, num_betas)

    # -----------------------------------------------------------------------
    # Camera setup (cam0 only)
    # -----------------------------------------------------------------------
    calib = calibrate.load_calibration_output(calib_path)
    cam   = calib[cam_id]
    K     = np.array(cam["K"], dtype=np.float64)
    R     = np.array(cam["R"], dtype=np.float64)
    t_vec = np.array(cam["t"], dtype=np.float64)
    img_w = int(cam["width"])
    img_h = int(cam["height"])

    map1, map2 = build_undistort_maps(K, cam["dist"], img_w, img_h)

    # -----------------------------------------------------------------------
    # Head vertex indices
    # -----------------------------------------------------------------------
    head_idx_np = _load_head_vert_indices()  # (H,) indices into 10475 verts
    head_idx_t  = torch.as_tensor(head_idx_np, dtype=torch.long, device=device)

    # -----------------------------------------------------------------------
    # MediaPipe FaceMesh
    # -----------------------------------------------------------------------
    import mediapipe as mp
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    )

    # -----------------------------------------------------------------------
    # Results buffers
    # -----------------------------------------------------------------------
    expr_out = np.zeros((n_frames, 10), dtype=np.float32)
    jaw_out  = np.zeros((n_frames,  3), dtype=np.float32)

    # -----------------------------------------------------------------------
    # Per-frame fitting loop
    # -----------------------------------------------------------------------
    n_fitted = 0
    for fi in frame_indices:
        frame_key = frame_keys[fi]
        img_path  = os.path.join(take_dir, "aligned", cam_id, frame_key)
        img_bgr   = cv2.imread(img_path)
        if img_bgr is None:
            continue

        img_u = undistort_fast(img_bgr, map1, map2)  # undistorted, distortion-free

        # ---- SMPL-X head vertices at current body pose (zero expr/jaw) ----
        # Run SMPL-X first so we can build the face crop bbox for MediaPipe.
        # MediaPipe fails on full 4K images (face too small) — we crop the
        # projected head region first and offset results back to full-image px.
        with torch.no_grad():
            out0 = _forward_with_face(
                model_layer,
                betas_shared,
                _t("global_orient", fi), _t("body_pose", fi),
                _t("lhand_pose",    fi), _t("rhand_pose", fi),
                _t("transl",        fi),
                torch.zeros(1, 10, device=device),
                torch.zeros(1,  3, device=device),
            )

        head_verts_3d = out0.vertices[0, head_idx_t]  # (H, 3)
        uv, depth     = project_points(head_verts_3d, K, R, t_vec)  # (H,2), (H,)

        uv_np_all = uv.cpu().numpy()
        in_front  = (depth > 0.01).cpu().numpy()
        # Crop bbox from projected head vertices (with generous padding)
        if in_front.sum() < 10:
            continue
        vis_uv  = uv_np_all[in_front]
        pad_px  = 200
        cx1 = int(vis_uv[:, 0].min()) - pad_px
        cx2 = int(vis_uv[:, 0].max()) + pad_px
        cy1 = int(vis_uv[:, 1].min()) - pad_px
        cy2 = int(vis_uv[:, 1].max()) + pad_px

        # ---- MediaPipe detection on face crop ----------------------------
        mp_pts = _mediapipe_on_crop(img_u, cx1, cy1, cx2, cy2, face_mesh)
        if mp_pts is None:
            if verbose:
                print(f"  frame {fi:4d} ({frame_key}): no face detected")
            continue

        # Keep only vertices in front of camera and inside image frame
        in_bounds = ((uv[:, 0] >= 0) & (uv[:, 0] < img_w) &
                     (uv[:, 1] >= 0) & (uv[:, 1] < img_h))
        visible = (torch.as_tensor(in_front, device=device) & in_bounds).cpu().numpy()

        if visible.sum() < 50:
            continue

        # ---- Nearest-neighbour correspondences MP → SMPL-X ---------------
        from scipy.spatial import KDTree
        uv_np       = uv.cpu().numpy()          # (H, 2)
        uv_visible  = uv_np[visible]            # (Hv, 2)
        vis_indices = np.where(visible)[0]      # → head_idx_np[vis_indices]

        tree = KDTree(uv_visible)
        dist, nn = tree.query(mp_pts)           # (468,)

        good_mask = dist < corr_dist_threshold
        if good_mask.sum() < min_correspondences:
            if verbose:
                print(f"  frame {fi:4d} ({frame_key}): only {good_mask.sum()} corr — skipping")
            continue

        # Target 2D points and their corresponding SMPL-X vertex indices
        tgt_pts  = torch.as_tensor(mp_pts[good_mask], dtype=torch.float32, device=device)  # (M,2)
        vert_idx = torch.as_tensor(
            head_idx_np[vis_indices[nn[good_mask]]], dtype=torch.long, device=device)       # (M,)

        # ---- L-BFGS optimisation: expression + jaw_pose ------------------
        expr_fi = torch.zeros(1, 10, device=device).requires_grad_(True)
        jaw_fi  = torch.zeros(1,  3, device=device).requires_grad_(True)

        go_fi = _t("global_orient", fi).detach()
        bp_fi = _t("body_pose",     fi).detach()
        lh_fi = _t("lhand_pose",    fi).detach()
        rh_fi = _t("rhand_pose",    fi).detach()
        tr_fi = _t("transl",        fi).detach()
        K_t   = torch.as_tensor(K,     dtype=torch.float32, device=device)
        R_t   = torch.as_tensor(R,     dtype=torch.float32, device=device)
        t_t   = torch.as_tensor(t_vec, dtype=torch.float32, device=device)

        optimizer = torch.optim.LBFGS(
            [expr_fi, jaw_fi], lr=0.3, max_iter=n_lbfgs_steps,
            history_size=10, line_search_fn="strong_wolfe",
        )

        final_loss = [float("inf")]

        def closure():
            optimizer.zero_grad()
            out = _forward_with_face(
                model_layer, betas_shared,
                go_fi, bp_fi, lh_fi, rh_fi, tr_fi,
                expr_fi, jaw_fi,
            )
            verts = out.vertices[0]  # (10475, 3)

            # Reproject corresponding vertices
            v3  = verts[vert_idx]                        # (M, 3)
            cam = v3 @ R_t.T + t_t                       # (M, 3)
            px  = cam @ K_t.T                            # (M, 3)
            z   = px[:, 2:].clamp(min=1e-4)
            uv2 = px[:, :2] / z                          # (M, 2)

            reproj = (uv2 - tgt_pts).pow(2).sum(dim=1).mean()
            reg    = expr_reg * expr_fi.pow(2).mean() + jaw_reg * jaw_fi.pow(2).mean()
            loss   = reproj + reg

            loss.backward()
            final_loss[0] = float(loss.item())
            return loss

        optimizer.step(closure)

        expr_out[fi] = expr_fi.detach().cpu().numpy()
        jaw_out[fi]  = jaw_fi.detach().cpu().numpy()
        n_fitted += 1

        if verbose and (n_fitted % 50 == 0 or n_fitted == 1):
            print(f"  frame {fi:4d} ({frame_key}):  loss={final_loss[0]:.4f}  "
                  f"corr={good_mask.sum()}  jaw={jaw_fi.detach().cpu().numpy().round(3)}")

    face_mesh.close()
    print(f"\nFitted {n_fitted}/{len(frame_indices)} frames successfully")

    # -----------------------------------------------------------------------
    # Save — back up original first, then overwrite expression + jaw_pose
    # -----------------------------------------------------------------------
    backup = smplx_npz.replace(".npz", "_pre_face.npz")
    if not os.path.exists(backup):
        import shutil
        shutil.copy2(smplx_npz, backup)
        print(f"Backed up original → {os.path.basename(backup)}")

    params = dict(data)
    params["expression"] = expr_out
    params["jaw_pose"]   = jaw_out
    np.savez(smplx_npz.replace(".npz", ""), **params)
    print(f"Saved updated params → {smplx_npz}")

    return expr_out, jaw_out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smplx-npz",    required=True)
    parser.add_argument("--calib",        required=True)
    parser.add_argument("--take-dir",     required=True)
    parser.add_argument("--cam",          default="cam0")
    parser.add_argument("--smplx-model-path", default="models/SMPLX")
    parser.add_argument("--frames",       default=None,
                        help="START:END slice (Python semantics), default: all frames")
    parser.add_argument("--lbfgs-steps",  type=int, default=40)
    parser.add_argument("--corr-dist",    type=float, default=25.0,
                        help="Max pixel distance (undistorted) for MP→vertex correspondence")
    parser.add_argument("--expr-reg",     type=float, default=0.05)
    parser.add_argument("--jaw-reg",      type=float, default=0.02)
    args = parser.parse_args()

    frames = None
    if args.frames:
        parts = args.frames.split(":")
        frames = (int(parts[0]) if parts[0] else 0,
                  int(parts[1]) if len(parts) > 1 and parts[1] else None)

    fit_face_expr(
        smplx_npz   = args.smplx_npz,
        calib_path  = args.calib,
        take_dir    = args.take_dir,
        cam_id      = args.cam,
        model_path  = args.smplx_model_path,
        frames      = frames,
        n_lbfgs_steps = args.lbfgs_steps,
        corr_dist_threshold = args.corr_dist,
        expr_reg    = args.expr_reg,
        jaw_reg     = args.jaw_reg,
    )


if __name__ == "__main__":
    main()
