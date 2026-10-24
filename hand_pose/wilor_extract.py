"""Run WiLoR on every frame/camera of a take and store MANO parameters, joints and meshes on disk.

Hand boxes and handedness come from the Sapiens2 hand keypoints (raw/distorted pixel space, same images Sapiens ran on).
Run in the `wilor` conda env with cwd irrelevant (WiLoR is imported from /home/fmalmb/CODE/WiLoR, read-only):

  ~/miniconda3/envs/wilor/bin/python -m hand_pose.wilor_extract --take /data/oak_recorder_sessions/20260925_153809_take3 \
      --npz output/sapiens2/allframes_1b_smoothbox.npz --calib output/calibration/20260925_sp_lg_7cam_scaled_sapiens_ext8000.json

Outputs in <take>/aligned/wilor/ (side index 0 = left, 1 = right; NaN / valid=False where no hand):
  wilor_hands.npz  per-camera, per-frame parameters and joints (see README keys below)
  wilor_meshes.npz verts_cam [T,C,2,778,3] float32 + mano_faces

Conventions (same as wilor_hand_reconstruction_v2.ipynb):
  - cam_t / joints_cam / verts_cam use the calibrated fx, cx, cy (metric camera frame), so projecting with K and no
    distortion lands on the raw-image pixel of the hand; left hands are mirrored x -> -x.
  - *_rotmat_raw are the network outputs; for left hands they live in the mirrored "as if right hand" frame.
  - global_orient_aa / hand_pose_aa are side-corrected (left: R -> M R M, M = diag(-1,1,1)) so they can be fed to a
    left-hand MANO model; right hands are unchanged. hand_pose is the 15 MANO finger joints (no wrist) in axis-angle.
"""
import argparse, json, os, sys, time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation as Rot
from torch.utils.data import default_collate

WILOR_DIR = "/home/fmalmb/CODE/WiLoR"
sys.path.insert(0, WILOR_DIR)
from wilor.models import load_wilor  # noqa: E402
from wilor.datasets.vitdet_dataset import ViTDetDataset  # noqa: E402
from wilor.utils import recursive_to  # noqa: E402

# Sapiens (Goliath) 308-keypoint ids in MediaPipe/OpenPose hand order (wrist, thumb..pinky base -> tip).
HAND_RIGHT = [41, 24, 23, 22, 21, 28, 27, 26, 25, 32, 31, 30, 29, 36, 35, 34, 33, 40, 39, 38, 37]
HAND_LEFT = [62, 45, 44, 43, 42, 49, 48, 47, 46, 53, 52, 51, 50, 57, 56, 55, 54, 61, 60, 59, 58]
SIDES = (HAND_LEFT, HAND_RIGHT)
MIRROR = np.diag([-1.0, 1.0, 1.0])


def load_model(device):
    cwd = os.getcwd()
    os.chdir(WILOR_DIR)  # load_wilor uses relative ./mano_data paths
    try:
        model, cfg = load_wilor(checkpoint_path=f"{WILOR_DIR}/pretrained_models/wilor_final.ckpt",
                                cfg_path=f"{WILOR_DIR}/pretrained_models/model_config.yaml")
    finally:
        os.chdir(cwd)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, cfg


def run_wilor(model, cfg, device, img, boxes, rights, K, rescale):
    ds = ViTDetDataset(cfg, img, boxes, rights, rescale_factor=rescale)
    batch = recursive_to(default_collate([ds[i] for i in range(len(ds))]), device)
    with torch.no_grad():
        out = model(batch)
    pc = out["pred_cam"].clone()
    pc[:, 1] = (2 * batch["right"] - 1) * pc[:, 1]
    bc, bs_ = batch["box_center"].float(), batch["box_size"].float()
    s = bs_ * pc[:, 0] + 1e-9
    cam_t = torch.stack([2 * (bc[:, 0] - K[0, 2]) / s + pc[:, 1], 2 * (bc[:, 1] - K[1, 2]) / s + pc[:, 2],
                         2 * K[0, 0] / s], dim=-1).cpu().numpy()
    f = (2 * batch["right"] - 1).cpu().numpy()[:, None]
    verts = out["pred_vertices"].cpu().numpy().copy()
    joints = out["pred_keypoints_3d"].cpu().numpy().copy()
    verts[:, :, 0] *= f
    joints[:, :, 0] *= f
    mp = out["pred_mano_params"]
    return dict(cam_t=cam_t, pred_cam=pc.cpu().numpy(), joints_local=joints, verts_local=verts,
                betas=mp["betas"].cpu().numpy(), go=mp["global_orient"][:, 0].cpu().numpy(),
                hp=mp["hand_pose"].cpu().numpy())


def side_corrected_aa(R, is_left):
    R = R.astype(np.float64)
    if is_left:
        R = MIRROR @ R @ MIRROR
    return Rot.from_matrix(R.reshape(-1, 3, 3)).as_rotvec().reshape(R.shape[:-2] + (3,))


def project(K, pts):
    return np.stack([K[0, 0] * pts[:, 0] / pts[:, 2] + K[0, 2], K[1, 1] * pts[:, 1] / pts[:, 2] + K[1, 2]], axis=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--take", required=True)
    ap.add_argument("--npz", required=True, help="Sapiens all-frames npz (raw pixel keypoints)")
    ap.add_argument("--calib", required=True)
    ap.add_argument("--out-dir", default=None, help="default <take>/aligned/wilor")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=None, help="inclusive")
    ap.add_argument("--cams", nargs="+", default=None)
    ap.add_argument("--rescale-factor", type=float, default=2.0)
    ap.add_argument("--min-score", type=float, default=0.3, help="mean Sapiens score of the 21 hand keypoints")
    ap.add_argument("--max-box-px", type=float, default=600.0, help="reject boxes whose longer side exceeds this")
    ap.add_argument("--max-aspect", type=float, default=3.0, help="reject boxes whose long/short side ratio exceeds this")
    ap.add_argument("--swap-handedness", action="store_true", help="debug: feed WiLoR the opposite handedness")
    ap.add_argument("--no-meshes", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    d = np.load(a.npz)
    sap_frames, sap_cams, kp, sc = d["frames"], [str(c) for c in d["cams"]], d["kp"], d["sc"]
    cams = a.cams or sap_cams
    cidx = [sap_cams.index(c) for c in cams]
    stop = int(sap_frames.max()) if a.stop is None else a.stop
    frames = [int(f) for f in sap_frames if a.start <= f <= stop]
    fidx = {int(f): i for i, f in enumerate(sap_frames)}
    with open(a.calib) as f:
        cal = json.load(f)["cameras"]
    Ks = {c: np.array(cal[c]["intrinsics"]["camera_matrix"], dtype=np.float64) for c in cams}

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(dev)
    faces = np.asarray(model.mano.faces, dtype=np.int64)
    T, C = len(frames), len(cams)
    nan = lambda *s: np.full((T, C, 2) + s, np.nan, np.float32)  # noqa: E731
    R = dict(valid=np.zeros((T, C, 2), bool), score=nan(), box=nan(4), betas=nan(10), pred_cam=nan(3), cam_t=nan(3),
             global_orient_rotmat_raw=nan(3, 3), hand_pose_rotmat_raw=nan(15, 3, 3), global_orient_aa=nan(3),
             hand_pose_aa=nan(15, 3), joints_local=nan(21, 3), joints_cam=nan(21, 3), kp2d_calib=nan(21, 2),
             kp2d_sapiens=nan(21, 2))
    V = None if a.no_meshes else np.full((T, C, 2, 778, 3), np.nan, np.float32)

    def load(c, f):
        return cv2.imread(f"{a.take}/aligned/{c}/{f:06d}.jpg")

    n_rejected = [0, 0]
    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        for ci, (c, si_cam) in enumerate(zip(cams, cidx)):
            K = Ks[c]
            q = deque()
            todo = iter(range(T))
            for _ in range(a.workers * 2):
                t = next(todo, None)
                if t is not None:
                    q.append((t, ex.submit(load, c, frames[t])))
            while q:
                t, fut = q.popleft()
                nt = next(todo, None)
                if nt is not None:
                    q.append((nt, ex.submit(load, c, frames[nt])))
                fi = fidx[frames[t]]
                boxes, rights, sides = [], [], []
                for s_, ids in enumerate(SIDES):
                    xy, s = kp[fi, si_cam, ids], sc[fi, si_cam, ids]
                    if not np.isfinite(xy).all() or s.mean() < a.min_score:
                        continue
                    wh = xy.max(0) - xy.min(0)
                    if wh.max() > a.max_box_px or wh.max() > a.max_aspect * max(wh.min(), 1.0):
                        n_rejected[s_] += 1
                        continue
                    boxes.append([*xy.min(0), *xy.max(0)])
                    rights.append(float(s_ == 1) if not a.swap_handedness else float(s_ == 0))
                    sides.append(s_)
                    R["score"][t, ci, s_] = s.mean()
                    R["kp2d_sapiens"][t, ci, s_] = xy
                if not boxes:
                    continue
                img = fut.result()
                if img is None:
                    continue
                o = run_wilor(model, cfg, dev, img, np.array(boxes, np.float32), np.array(rights, np.float32), K,
                              a.rescale_factor)
                for n, s_ in enumerate(sides):
                    R["valid"][t, ci, s_] = True
                    R["box"][t, ci, s_] = boxes[n]
                    R["betas"][t, ci, s_] = o["betas"][n]
                    R["pred_cam"][t, ci, s_] = o["pred_cam"][n]
                    R["cam_t"][t, ci, s_] = o["cam_t"][n]
                    R["global_orient_rotmat_raw"][t, ci, s_] = o["go"][n]
                    R["hand_pose_rotmat_raw"][t, ci, s_] = o["hp"][n]
                    R["global_orient_aa"][t, ci, s_] = side_corrected_aa(o["go"][n], s_ == 0)
                    R["hand_pose_aa"][t, ci, s_] = side_corrected_aa(o["hp"][n], s_ == 0)
                    R["joints_local"][t, ci, s_] = o["joints_local"][n]
                    jc = o["joints_local"][n] + o["cam_t"][n]
                    R["joints_cam"][t, ci, s_] = jc
                    R["kp2d_calib"][t, ci, s_] = project(K, jc)
                    if V is not None:
                        V[t, ci, s_] = o["verts_local"][n] + o["cam_t"][n]
                if t % 64 == 0:
                    print(f"{c} frame {frames[t]} ({t + 1}/{T}) {time.time() - t0:.0f}s", flush=True)

    out = a.out_dir or f"{a.take}/aligned/wilor"
    os.makedirs(out, exist_ok=True)
    meta = dict(frames=np.array(frames), cams=np.array(cams), sides=np.array(["left", "right"]),
                rescale_factor=np.float32(a.rescale_factor), swap_handedness=np.bool_(a.swap_handedness))
    np.savez_compressed(f"{out}/wilor_hands.npz", **R, **meta)
    if V is not None:
        np.savez_compressed(f"{out}/wilor_meshes.npz", verts_cam=V, mano_faces=faces, **meta)
    print("saved", out, f"in {time.time() - t0:.0f}s; boxes rejected by size/aspect gate: left {n_rejected[0]}, right {n_rejected[1]}")
    err = np.linalg.norm(R["kp2d_calib"] - R["kp2d_sapiens"], axis=-1)
    for ci, c in enumerate(cams):
        row = [f"{c}:"]
        for s_, nm in enumerate(("L", "R")):
            v = R["valid"][:, ci, s_]
            row.append(f"{nm} {v.sum()}/{T} valid, median reproj-vs-Sapiens {np.nanmedian(err[:, ci, s_][v]):.1f}px"
                       if v.any() else f"{nm} none")
        print("  ".join(row))


if __name__ == "__main__":
    main()
