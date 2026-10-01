"""Multi-camera extrinsic calibration from synchronized video.

Recovers R, t per camera using SuperPoint+LightGlue background matching
pooled across many frames, followed by essential-matrix initialization,
incremental PnP registration, and sparse bundle adjustment.

Usage:
    python extrinsic_calib.py <take_dir> \\
        --calib output/calibration/20260908_174749_7cam.json \\
        --out   output/calibration/20260925_sp_lg_7cam.json \\
        [--sample-every 30] [--min-conf 0.3] [--cam cam0,cam1,...] [--verbose]

Pre-extracted JPEGs at <take_dir>/aligned/<cam_id>/ are in distorted space;
SAM3 masks at <take_dir>/aligned/masks_sam3/<cam_id>/ are optional.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from itertools import combinations

import cv2
import numpy as np
import torch
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

sys.path.insert(0, os.path.dirname(__file__))
import calibrate


# ---------------------------------------------------------------------------
# Union-Find
# ---------------------------------------------------------------------------

class UnionFind:
    def __init__(self):
        self._parent = {}

    def find(self, x):
        if x not in self._parent:
            self._parent[x] = x
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[rb] = ra

    def groups(self):
        groups = {}
        for x in self._parent:
            r = self.find(x)
            groups.setdefault(r, []).append(x)
        return list(groups.values())


# ---------------------------------------------------------------------------
# Frame sampling
# ---------------------------------------------------------------------------

def _lap_var(img_gray):
    return cv2.Laplacian(img_gray, cv2.CV_64F).var()


def sample_frames(take_dir, cam_ids, calib, sample_every=30, blur_thresh=50.0, mask_subdir="masks_sam3"):
    """Return {cam_id: [(frame_idx, undistorted_masked_bgr), ...]}."""
    result = {}
    for cam_id in cam_ids:
        cam_dir = os.path.join(take_dir, "aligned", cam_id)
        mask_dir = os.path.join(take_dir, "aligned", mask_subdir, cam_id)
        has_masks = os.path.isdir(mask_dir)

        K = calib[cam_id]["K"]
        dist = calib[cam_id]["dist"]

        all_frames = sorted(f for f in os.listdir(cam_dir) if f.endswith(".jpg"))
        sampled = []
        for i, fname in enumerate(all_frames):
            if i % sample_every != 0:
                continue
            img = cv2.imread(os.path.join(cam_dir, fname))
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if _lap_var(gray) < blur_thresh:
                continue
            undist = cv2.undistort(img, K, dist)
            if has_masks:
                mask_path = os.path.join(mask_dir, fname.replace(".jpg", ".png"))
                if not os.path.exists(mask_path):
                    mask_path = os.path.join(mask_dir, fname)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    # zero out the person; keep background
                    person = mask > 128
                    undist[person] = 0
            sampled.append((i, undist))
        result[cam_id] = sampled
    return result


# ---------------------------------------------------------------------------
# SuperPoint + LightGlue
# ---------------------------------------------------------------------------

def _to_tensor(img_bgr, device, max_side=1280):
    """Returns (tensor, scale) where scale = resized/original. kpts must be divided by scale."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    h, w = gray.shape
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    elif w < 1000:
        scale = 2.0
        gray = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_LINEAR)
    return torch.from_numpy(gray)[None, None].to(device), scale


def extract_features(frames_by_cam, extractor, device, verbose=False):
    """Return {cam_id: [(frame_idx, kpts_np, scores_np, descriptors_tensor), ...]}."""
    feats_by_cam = {}
    for cam_id, frames in frames_by_cam.items():
        feats = []
        for idx, (frame_idx, img) in enumerate(frames):
            t, scale = _to_tensor(img, device)
            with torch.no_grad():
                f = extractor.extract(t)
            kpts = f["keypoints"][0].cpu().numpy() / scale  # back to full-res pixels
            scores = f["keypoint_scores"][0].cpu().numpy()
            descs = f["descriptors"][0]                      # (N, D) on device
            feats.append((frame_idx, kpts, scores, descs, f))
            if verbose:
                print(f"  {cam_id} frame {frame_idx}: {len(kpts)} keypoints", end="\r")
        feats_by_cam[cam_id] = feats
        if verbose:
            print(f"  {cam_id}: {len(feats)} frames, avg {np.mean([len(f[1]) for f in feats]):.0f} kpts")
    return feats_by_cam


def match_pairs(feats_by_cam, cam_ids, matcher, min_conf=0.3, max_pairs_per_cam=None, verbose=False):
    """For each camera pair, match across frame pairs (capped at max_pairs_per_cam).
    Return pooled (kpts_a, kpts_b) per pair."""
    pair_matches = {}
    pairs = list(combinations(cam_ids, 2))
    rng = np.random.default_rng(0)
    for ca, cb in pairs:
        fa_list = feats_by_cam[ca]
        fb_list = feats_by_cam[cb]
        # Build all (i,j) frame-pair indices, then sub-sample if needed
        all_combos = [(i, j) for i in range(len(fa_list)) for j in range(len(fb_list))]
        if max_pairs_per_cam and len(all_combos) > max_pairs_per_cam:
            idxs = rng.choice(len(all_combos), max_pairs_per_cam, replace=False)
            all_combos = [all_combos[k] for k in idxs]
        all_ka, all_kb = [], []
        for i, j in all_combos:
            _, kpa, _, _, fa = fa_list[i]
            _, kpb, _, _, fb = fb_list[j]
            with torch.no_grad():
                m = matcher({"image0": fa, "image1": fb})
            matches = m["matches"][0].cpu().numpy()   # (M, 2)
            scores  = m["scores"][0].cpu().numpy()    # (M,)
            keep = scores >= min_conf
            if keep.sum() == 0:
                continue
            all_ka.append(kpa[matches[keep, 0]])
            all_kb.append(kpb[matches[keep, 1]])
        if all_ka:
            pair_matches[(ca, cb)] = (
                np.concatenate(all_ka, axis=0),
                np.concatenate(all_kb, axis=0),
            )
            if verbose:
                print(f"  ({ca},{cb}): {len(pair_matches[(ca,cb)][0])} pooled matches "
                      f"from {len(all_combos)} frame pairs")
        else:
            if verbose:
                print(f"  ({ca},{cb}): 0 matches")
    return pair_matches


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _triangulate(R0, t0, R1, t1, K0, K1, pts0, pts1):
    # pts0/pts1 are already in undistorted pixel space — use K@[R|t] directly
    P0 = K0 @ np.hstack([R0, t0.reshape(3, 1)])
    P1 = K1 @ np.hstack([R1, t1.reshape(3, 1)])
    pts4d = cv2.triangulatePoints(P0, P1,
                                  pts0.T.astype(np.float64),
                                  pts1.T.astype(np.float64))
    pts3d = (pts4d[:3] / pts4d[3]).T
    return pts3d


def _reproj_error(pts3d, pts2d, R, t, K):
    rvec, _ = cv2.Rodrigues(R)
    proj, _ = cv2.projectPoints(pts3d.astype(np.float64), rvec, t.astype(np.float64), K, None)
    err = np.linalg.norm(proj.reshape(-1, 2) - pts2d, axis=1)
    return err


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------

def initialize_two_view(pair_matches, calib, cam_ids, verbose=False):
    """Pick best pair, solve essential matrix, return (cam_a, cam_b, R_a, t_a, R_b, t_b, pts3d, obs)."""
    best_pair, best_inliers, best_R, best_t, best_pts3d = None, -1, None, None, None
    best_pts_a, best_pts_b = None, None

    for (ca, cb), (kpa, kpb) in pair_matches.items():
        if len(kpa) < 20:
            continue
        Ka, Kb = calib[ca]["K"], calib[cb]["K"]
        # Normalize by each camera's own K so essential matrix is well-defined
        # even when Ka ≠ Kb (our cameras have different fx/cx/cy).
        kpa_n = cv2.undistortPoints(kpa.reshape(-1, 1, 2), Ka, None).reshape(-1, 2)
        kpb_n = cv2.undistortPoints(kpb.reshape(-1, 1, 2), Kb, None).reshape(-1, 2)
        # threshold in normalized coords: 1.5px / fx ≈ 7e-4
        thresh_n = 1.5 / float(Ka[0, 0])
        E, mask = cv2.findEssentialMat(kpa_n, kpb_n, np.eye(3),
                                       method=cv2.RANSAC, prob=0.999, threshold=thresh_n)
        if E is None or mask is None:
            continue
        mask = mask.ravel().astype(bool)
        if mask.sum() < 20:
            continue
        _, R, t, pose_mask = cv2.recoverPose(E, kpa_n[mask], kpb_n[mask])
        pose_mask = pose_mask.ravel().astype(bool)
        inliers = pose_mask.sum()
        if inliers > best_inliers:
            best_inliers = inliers
            best_pair = (ca, cb)
            best_R, best_t = R, t
            best_pts_a = kpa[mask][pose_mask]
            best_pts_b = kpb[mask][pose_mask]

    if best_pair is None:
        raise RuntimeError("No valid two-view initialization found.")

    ca, cb = best_pair
    Ka, Kb = calib[ca]["K"], calib[cb]["K"]
    # Reference cam_a: R=I, t=0
    Ra = np.eye(3); ta = np.zeros(3)
    Rb = best_R; tb = best_t.ravel()
    pts3d = _triangulate(Ra, ta, Rb, tb, Ka, Kb, best_pts_a, best_pts_b)

    # Keep only points in front of both cameras
    front = (pts3d[:, 2] > 0) & ((Rb @ pts3d.T + tb[:, None])[2] > 0)
    pts3d = pts3d[front]
    best_pts_a = best_pts_a[front]
    best_pts_b = best_pts_b[front]

    # observations: {point_idx: {cam_id: pt2d}}
    obs = {i: {ca: best_pts_a[i], cb: best_pts_b[i]} for i in range(len(pts3d))}

    extrinsics = {ca: (Ra, ta), cb: (Rb, tb)}
    if verbose:
        err_a = _reproj_error(pts3d, best_pts_a, Ra, ta, Ka)
        err_b = _reproj_error(pts3d, best_pts_b, Rb, tb, Kb)
        print(f"  Init ({ca},{cb}): {len(pts3d)} pts, "
              f"reproj {err_a.median() if hasattr(err_a,'median') else np.median(err_a):.1f}/"
              f"{np.median(err_b):.1f} px (a/b), {best_inliers} inliers")

    return extrinsics, pts3d, obs


def register_camera(cam_id, pair_matches, extrinsics, pts3d, obs, calib, verbose=False):
    """PnP: register cam_id against the existing 3D point cloud. Returns (R, t) or None."""
    K = calib[cam_id]["K"]
    pts3d_matched, pts2d_matched = [], []

    for registered_cam in extrinsics:
        key = (min(cam_id, registered_cam), max(cam_id, registered_cam))
        if key not in pair_matches:
            continue
        kp_a, kp_b = pair_matches[key]
        if key[0] == cam_id:
            kp_new, kp_reg = kp_a, kp_b
        else:
            kp_new, kp_reg = kp_b, kp_a

        R_reg, t_reg = extrinsics[registered_cam]
        K_reg = calib[registered_cam]["K"]

        # Match pooled 2D observations in registered cam to 3D points
        for pi, point_obs in obs.items():
            if registered_cam not in point_obs:
                continue
            obs2d_reg = point_obs[registered_cam]
            # find closest match in kp_reg
            dists = np.linalg.norm(kp_reg - obs2d_reg, axis=1)
            idx = np.argmin(dists)
            if dists[idx] < 2.0:
                pts3d_matched.append(pts3d[pi])
                pts2d_matched.append(kp_new[idx])

    if len(pts3d_matched) < 10:
        return None

    pts3d_arr = np.array(pts3d_matched, dtype=np.float64)
    pts2d_arr = np.array(pts2d_matched, dtype=np.float64)
    ok, rvec, t, inliers = cv2.solvePnPRansac(
        pts3d_arr, pts2d_arr, K, None,
        iterationsCount=1000, reprojectionError=5.0, confidence=0.999,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not ok or inliers is None or len(inliers) < 10:
        return None

    R, _ = cv2.Rodrigues(rvec)
    t = t.ravel()
    if verbose:
        err = _reproj_error(pts3d_arr[inliers.ravel()], pts2d_arr[inliers.ravel()], R, t, K)
        print(f"  Registered {cam_id}: {len(inliers)} inliers, median reproj {np.median(err):.1f} px")
    return R, t


def add_new_points(cam_id, pair_matches, extrinsics, pts3d, obs, calib, verbose=False):
    """Triangulate new 3D points visible in cam_id and at least one registered camera."""
    K_new = calib[cam_id]["K"]
    R_new, t_new = extrinsics[cam_id]
    n_added = 0

    for reg_cam in extrinsics:
        if reg_cam == cam_id:
            continue
        key = (min(cam_id, reg_cam), max(cam_id, reg_cam))
        if key not in pair_matches:
            continue
        kp_a, kp_b = pair_matches[key]
        if key[0] == cam_id:
            kp_new, kp_reg = kp_a, kp_b
        else:
            kp_new, kp_reg = kp_b, kp_a

        R_reg, t_reg = extrinsics[reg_cam]
        K_reg = calib[reg_cam]["K"]

        pts3d_new = _triangulate(R_new, t_new, R_reg, t_reg, K_new, K_reg, kp_new, kp_reg)
        front = (pts3d_new[:, 2] > 0) & ((R_reg @ pts3d_new.T + t_reg[:, None])[2] > 0)
        err_new = _reproj_error(pts3d_new[front], kp_new[front], R_new, t_new, K_new)
        err_reg = _reproj_error(pts3d_new[front], kp_reg[front], R_reg, t_reg, K_reg)
        good = front.copy()
        good[front] &= (err_new < 8.0) & (err_reg < 8.0)

        for i in np.where(good)[0]:
            pi = len(pts3d)
            pts3d = np.vstack([pts3d, pts3d_new[i]])
            obs[pi] = {cam_id: kp_new[i], reg_cam: kp_reg[i]}
            n_added += 1

    if verbose and n_added:
        print(f"  {cam_id}: added {n_added} new 3D points")
    return pts3d, obs


# ---------------------------------------------------------------------------
# Bundle adjustment
# ---------------------------------------------------------------------------

def _project(pts3d, rvec, tvec, K):
    R, _ = cv2.Rodrigues(rvec)
    Xc = (R @ pts3d.T).T + tvec
    px = Xc[:, 0] / Xc[:, 2] * K[0, 0] + K[0, 2]
    py = Xc[:, 1] / Xc[:, 2] * K[1, 1] + K[1, 2]
    return np.column_stack([px, py])


def bundle_adjust(extrinsics, pts3d, obs, calib, ref_cam, verbose=False):
    cam_ids = list(extrinsics.keys())
    free_cams = [c for c in cam_ids if c != ref_cam]
    cam_idx = {c: i for i, c in enumerate(free_cams)}

    def pack(extr, pts):
        rvecs, tvecs = [], []
        for c in free_cams:
            R, t = extr[c]
            rv, _ = cv2.Rodrigues(R)
            rvecs.append(rv.ravel())
            tvecs.append(t.ravel())
        return np.concatenate(rvecs + tvecs + [pts.ravel()])

    def unpack(x, n_free, n_pts):
        rvecs = x[:n_free * 3].reshape(n_free, 3)
        tvecs = x[n_free * 3:n_free * 6].reshape(n_free, 3)
        pts = x[n_free * 6:].reshape(n_pts, 3)
        return rvecs, tvecs, pts

    def residuals(x, obs_list, cam_ids_obs, pt_indices, ref_R, ref_t, ref_K, n_free, n_pts, calib_local, free_cams_local, cam_idx_local):
        rvecs, tvecs, pts = unpack(x, n_free, n_pts)
        res = []
        R_ref, _ = cv2.Rodrigues(ref_R)
        for obs2d, cam_id, pi in zip(obs_list, cam_ids_obs, pt_indices):
            pt = pts[pi:pi+1]
            if cam_id == ref_cam:
                proj = _project(pt, ref_R, ref_t, calib_local[cam_id]["K"])
            else:
                ci = cam_idx_local[cam_id]
                proj = _project(pt, rvecs[ci], tvecs[ci], calib_local[cam_id]["K"])
            res.extend((proj[0] - obs2d).tolist())
        return np.array(res)

    # Flatten observations
    obs_list, cam_ids_obs, pt_indices = [], [], []
    valid_pts = set()
    for pi, point_obs in obs.items():
        if pi >= len(pts3d):
            continue
        for cam_id, pt2d in point_obs.items():
            if cam_id not in extrinsics:
                continue
            obs_list.append(pt2d)
            cam_ids_obs.append(cam_id)
            pt_indices.append(pi)
            valid_pts.add(pi)

    # Re-index points to contiguous array
    sorted_pts = sorted(valid_pts)
    pt_remap = {old: new for new, old in enumerate(sorted_pts)}
    pts3d_ba = pts3d[sorted_pts]
    pt_indices_remapped = [pt_remap[pi] for pi in pt_indices]

    n_free = len(free_cams)
    n_pts = len(sorted_pts)
    n_obs = len(obs_list)

    ref_R_vec, _ = cv2.Rodrigues(extrinsics[ref_cam][0])
    ref_t = extrinsics[ref_cam][1]

    x0 = pack(extrinsics, pts3d_ba)

    # Sparse Jacobian sparsity
    jac_sparsity = lil_matrix((n_obs * 2, len(x0)), dtype=int)
    for i, (cam_id, pi) in enumerate(zip(cam_ids_obs, pt_indices_remapped)):
        if cam_id != ref_cam:
            ci = cam_idx[cam_id]
            jac_sparsity[2*i:2*i+2, ci*3:(ci+1)*3] = 1          # rvec
            jac_sparsity[2*i:2*i+2, n_free*3+ci*3:n_free*3+(ci+1)*3] = 1  # tvec
        jac_sparsity[2*i:2*i+2, n_free*6+pi*3:n_free*6+(pi+1)*3] = 1  # 3D point

    def prune_and_run(x_init, thresh, label):
        res_init = residuals(x_init, obs_list, cam_ids_obs, pt_indices_remapped,
                             ref_R_vec, ref_t, ref_K=None, n_free=n_free, n_pts=n_pts,
                             calib_local=calib, free_cams_local=free_cams, cam_idx_local=cam_idx)
        errs = np.sqrt(res_init[0::2]**2 + res_init[1::2]**2)
        keep = errs < thresh
        if verbose:
            print(f"  BA {label}: {keep.sum()}/{len(keep)} obs kept (<{thresh}px)")
        obs_f = [o for o, k in zip(obs_list, keep) if k]
        cam_f = [c for c, k in zip(cam_ids_obs, keep) if k]
        pt_f  = [p for p, k in zip(pt_indices_remapped, keep) if k]
        jac_f = lil_matrix((len(obs_f)*2, len(x_init)), dtype=int)
        for i, (cam_id, pi) in enumerate(zip(cam_f, pt_f)):
            if cam_id != ref_cam:
                ci = cam_idx[cam_id]
                jac_f[2*i:2*i+2, ci*3:(ci+1)*3] = 1
                jac_f[2*i:2*i+2, n_free*3+ci*3:n_free*3+(ci+1)*3] = 1
            jac_f[2*i:2*i+2, n_free*6+pi*3:n_free*6+(pi+1)*3] = 1
        result = least_squares(
            residuals, x_init, jac_sparsity=jac_f,
            method='trf', loss='soft_l1', max_nfev=200, verbose=0,
            args=(obs_f, cam_f, pt_f, ref_R_vec, ref_t, None, n_free, n_pts, calib, free_cams, cam_idx),
        )
        return result.x, obs_f, cam_f, pt_f

    x, obs_list, cam_ids_obs, pt_indices_remapped = prune_and_run(x0, 8.0, "pass1")
    x, obs_list, cam_ids_obs, pt_indices_remapped = prune_and_run(x, 5.0, "pass2")
    x, _, _, _ = prune_and_run(x, 5.0, "pass3")

    rvecs_opt, tvecs_opt, pts3d_opt = unpack(x, n_free, n_pts)

    extrinsics_out = {ref_cam: extrinsics[ref_cam]}
    for i, c in enumerate(free_cams):
        R, _ = cv2.Rodrigues(rvecs_opt[i])
        extrinsics_out[c] = (R, tvecs_opt[i])

    return extrinsics_out, pts3d_opt, sorted_pts


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(extrinsics, pts3d, obs, calib, verbose=True):
    print("\n=== Reprojection errors ===")
    for cam_id, (R, t) in extrinsics.items():
        K = calib[cam_id]["K"]
        errs = []
        for pi, point_obs in obs.items():
            if cam_id not in point_obs or pi >= len(pts3d):
                continue
            err = _reproj_error(pts3d[pi:pi+1], point_obs[cam_id].reshape(1, 2), R, t, K)
            errs.append(err[0])
        if errs:
            print(f"  {cam_id}: n={len(errs):4d}  median={np.median(errs):.2f}px  p95={np.percentile(errs,95):.1f}px")
        else:
            print(f"  {cam_id}: no observations")


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export(extrinsics, calib, ref_cam, out_path, cam_ids):
    timestamp = datetime.now().isoformat()
    cameras = {}
    for cam_id in cam_ids:
        if cam_id not in extrinsics:
            continue
        R, t = extrinsics[cam_id]
        c = calib[cam_id]
        cameras[cam_id] = {
            "device_id": c.get("device_id"),
            "intrinsics": {
                "camera_matrix": c["K"].tolist(),
                "dist_coeffs": c["dist"].ravel().tolist(),
                "image_width": c["width"],
                "image_height": c["height"],
                "reprojection_error_px": None,
                "converged": True,
                "source": "Sep8-cache",
            },
            "extrinsics": {
                "rotation": R.tolist(),
                "translation": t.tolist(),
                "reference_camera": ref_cam,
                "is_reference": (cam_id == ref_cam),
                "method": "SuperPoint+LightGlue+BundleAdjust",
            },
        }
    payload = {
        "metadata": {
            "timestamp": timestamp,
            "method": "SuperPoint+LightGlue bundle adjustment",
            "reference_cameras": {ref_cam: list(extrinsics.keys())},
            "scale": "arbitrary",
        },
        "cameras": cameras,
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("take_dir")
    ap.add_argument("--calib", default="output/calibration/20260908_174749_7cam.json")
    ap.add_argument("--out", default="output/calibration/sp_lg_extrinsics.json")
    ap.add_argument("--sample-every", type=int, default=30)
    ap.add_argument("--min-conf", type=float, default=0.3)
    ap.add_argument("--max-pairs-per-cam", type=int, default=15,
                    help="Max frame-pair combinations per camera pair (caps N² growth). Default 15.")
    ap.add_argument("--max-kpts", type=int, default=1024,
                    help="SuperPoint max keypoints per frame. Default 1024.")
    ap.add_argument("--cam", default=None, help="Comma-separated camera subset, e.g. cam0,cam1,cam2")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    calib = calibrate.load_calibration_output(args.calib)
    cam_ids = args.cam.split(",") if args.cam else sorted(calib.keys())
    cam_ids = [c for c in cam_ids if c in calib]
    print(f"Cameras: {cam_ids}")
    print(f"Take:    {args.take_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device:  {device}")

    from lightglue import LightGlue, SuperPoint
    extractor = SuperPoint(max_num_keypoints=args.max_kpts).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)

    print("\n[1] Sampling frames...")
    frames_by_cam = sample_frames(args.take_dir, cam_ids, calib, sample_every=args.sample_every)
    for cam_id, frames in frames_by_cam.items():
        print(f"  {cam_id}: {len(frames)} frames")

    print("\n[2] Extracting SuperPoint features...")
    feats_by_cam = extract_features(frames_by_cam, extractor, device, verbose=args.verbose)

    print("\n[3] LightGlue matching (all pairs × all frame combos)...")
    pair_matches = match_pairs(feats_by_cam, cam_ids, matcher, min_conf=args.min_conf,
                               max_pairs_per_cam=args.max_pairs_per_cam, verbose=True)

    print("\n[4] Two-view initialization...")
    extrinsics, pts3d, obs = initialize_two_view(pair_matches, calib, cam_ids, verbose=True)
    print(f"  Initialized: {list(extrinsics.keys())}, {len(pts3d)} 3D points")

    print("\n[5] Registering remaining cameras...")
    remaining = [c for c in cam_ids if c not in extrinsics]
    max_iters = len(remaining) * 2
    itr = 0
    while remaining and itr < max_iters:
        itr += 1
        registered_any = False
        for cam_id in list(remaining):
            result = register_camera(cam_id, pair_matches, extrinsics, pts3d, obs, calib, verbose=args.verbose)
            if result is not None:
                extrinsics[cam_id] = result
                pts3d, obs = add_new_points(cam_id, pair_matches, extrinsics, pts3d, obs, calib, verbose=args.verbose)
                remaining.remove(cam_id)
                registered_any = True
        if not registered_any:
            break

    if remaining:
        print(f"  WARNING: could not register: {remaining}")

    # Choose reference camera as the one with the most observations
    obs_count = {c: sum(1 for po in obs.values() if c in po) for c in extrinsics}
    ref_cam = max(obs_count, key=obs_count.get)
    print(f"  Reference camera: {ref_cam} ({obs_count[ref_cam]} obs)")

    print("\n[6] Bundle adjustment...")
    extrinsics, pts3d_ba, sorted_pts = bundle_adjust(extrinsics, pts3d, obs, calib, ref_cam, verbose=True)

    # Remap obs to bundle-adjusted point indices
    pt_remap = {old: new for new, old in enumerate(sorted_pts)}
    obs_ba = {}
    for old_pi, new_pi in pt_remap.items():
        if old_pi in obs:
            obs_ba[new_pi] = obs[old_pi]

    print("\n[7] Validation...")
    validate(extrinsics, pts3d_ba, obs_ba, calib)

    export(extrinsics, calib, ref_cam, args.out, cam_ids)


if __name__ == "__main__":
    main()
