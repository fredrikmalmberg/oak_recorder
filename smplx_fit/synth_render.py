"""Render SMPL-X (T-pose + poses picked from a fit) from a ring of cameras and save exact 3D ground truth.

Used to measure how Sapiens and WiLoR keypoints sit relative to SMPL-X's own joints (see synth_sapiens.py,
synth_wilor.py, synth_analyze.py). Images are 3840x2160 like the take, pinhole (no distortion), K from a real camera.

  python -m smplx_fit.synth_render --smplx-npz output/cam2hands/smplx_hw/aligned/pose2d_sapiens_smoothbox/smplx/smplx_params.npz \
      --calib output/calibration/20260925_sp_lg_7cam_scaled_sapiens_cam2hands.json --out-dir <dir> [--flat]

Output <out-dir>/<variant>/: p{pose}_a{azimuth}_e{elevation}.jpg + masks/*.png, and <out-dir>/gt.npz with
  verts [P,10475,3], joints [P,J,3] (world), pose_names, frames, view names, K, R[P,V,3,3], t[P,V,3], width, height.
Azimuth 0 = camera in front of the person, positive = towards the person's left. Elevation in degrees above the target.
"""
import argparse, json, os, sys

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import calibrate  # noqa: E402
from smplx_fit import model as smplx_model  # noqa: E402
from smplx_fit.render_turntable import world_up_from_calib, UV_NPZ_DEFAULT  # noqa: E402
from smplx_fit.silhouette import _get_nvdiffrast_ctx, _verts_to_clip  # noqa: E402
from smplx_fit.texture.texture_bake import load_uv_data  # noqa: E402

TEXTURE_DEFAULT = "/home/fmalmb/CODE/sl_reconstruction/visualization/textures/smplx_texture_m_long_sleeve.png"
J_PELVIS, J_LSHO, J_RSHO = 0, 16, 17


def look_at(cam_pos, target, up):
    """OpenCV camera (x right, y down, z forward, x_cam = R X + t). render_turntable.look_at is not used: its y axis points up."""
    fwd = target - cam_pos
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    R = np.stack([right, np.cross(fwd, right), fwd])
    return R, -R @ cam_pos


def pick_poses(joints, n):
    """Farthest-point sampling of fit frames on pelvis-relative joint positions (body + hands), starting at the median."""
    x = (joints[:, :55] - joints[:, :1]).reshape(len(joints), -1)
    first = int(np.argmin(np.linalg.norm(x - np.median(x, 0), axis=1)))
    chosen = [first]
    d = np.linalg.norm(x - x[first], axis=1)
    while len(chosen) < n:
        nxt = int(np.argmax(d))
        chosen.append(nxt)
        d = np.minimum(d, np.linalg.norm(x - x[nxt], axis=1))
    return chosen


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smplx-npz", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--frames", default="300:650", help="fit frames (file numbers) to pick poses from")
    ap.add_argument("--n-poses", type=int, default=5)
    ap.add_argument("--azimuths", type=float, nargs="+", default=[-60, -40, -20, 0, 20, 40, 60])
    ap.add_argument("--elevations", type=float, nargs="+", default=[0, 20])
    ap.add_argument("--radius", type=float, default=1.55)
    ap.add_argument("--ref-cam", default="cam0", help="camera whose K / image size is used")
    ap.add_argument("--texture", default=TEXTURE_DEFAULT)
    ap.add_argument("--flat", action="store_true", help="grey Lambertian mesh instead of the texture")
    ap.add_argument("--variant", default=None, help="subdir name (default: textured / flat)")
    ap.add_argument("--model-path", default="models/SMPLX")
    ap.add_argument("--beta-seed", type=int, default=None, help="replace the fit betas with N(0, beta-std^2) random shape betas")
    ap.add_argument("--beta-std", type=float, default=1.0)
    a = ap.parse_args()
    variant = a.variant or ("flat" if a.flat else "textured")
    import nvdiffrast.torch as dr

    calib = calibrate.load_calibration_output(a.calib)
    up = world_up_from_calib(calib, sorted(calib))
    K = np.array(calib[a.ref_cam]["K"], dtype=np.float64)
    W, H = int(calib[a.ref_cam]["width"]), int(calib[a.ref_cam]["height"])

    d = np.load(a.smplx_npz, allow_pickle=True)
    keys = [int(k[:6]) for k in d["frame_keys"]]
    f0, f1 = [int(v) for v in a.frames.split(":")]
    sel = np.array([i for i, k in enumerate(keys) if f0 <= k <= f1])
    layer = smplx_model.load_layer(a.model_path, gender=str(d["gender"]), num_betas=int(d["num_betas"]))
    betas = torch.as_tensor(d["betas"], dtype=torch.float32).reshape(1, -1)
    if a.beta_seed is not None:
        betas = torch.as_tensor(np.random.default_rng(a.beta_seed).normal(0, a.beta_std, betas.shape[1]), dtype=torch.float32).reshape(1, -1)
        print("random betas:", betas.numpy().round(2))

    def fwd(go, bp, lh, rh, tr):
        t = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32)  # noqa: E731
        with torch.no_grad():
            o = smplx_model.forward_with_expr(layer, betas.expand(len(go), -1), t(go), t(bp), t(lh), t(rh), t(tr),
                                              torch.zeros(len(go), 10), torch.zeros(len(go), 3))
        return o.vertices.numpy(), o.joints.numpy()

    _, jts_all = fwd(d["global_orient"][sel], d["body_pose"][sel], d["lhand_pose"][sel], d["rhand_pose"][sel], d["transl"][sel])
    pick = pick_poses(jts_all, a.n_poses)
    idx = sel[pick]
    go = d["global_orient"][idx]
    # T-pose keeps the mean orientation and position of the picked frames so it faces the same way
    go_t = Rot.from_rotvec(go).mean().as_rotvec()
    tr_t = d["transl"][idx].mean(0)
    GO = np.concatenate([go_t[None], go]); TR = np.concatenate([tr_t[None], d["transl"][idx]])
    BP = np.concatenate([np.zeros((1, 21, 3)), d["body_pose"][idx]]).astype(np.float32)
    LH = np.concatenate([np.zeros((1, 15, 3)), d["lhand_pose"][idx]]).astype(np.float32)
    RH = np.concatenate([np.zeros((1, 15, 3)), d["rhand_pose"][idx]]).astype(np.float32)
    names = ["tpose"] + [f"fit{keys[i]}" for i in idx]
    verts, joints = fwd(GO, BP, LH, RH, TR)
    P = len(names)
    print("poses:", names)

    views = [(az, el) for el in a.elevations for az in a.azimuths]
    V = len(views)
    Rs = np.zeros((P, V, 3, 3)); ts = np.zeros((P, V, 3))
    for p in range(P):
        face = Rot.from_rotvec(GO[p]).as_matrix()[:, 2]
        face = face - up * (face @ up); face /= np.linalg.norm(face)
        target = 0.5 * joints[p, J_PELVIS] + 0.25 * (joints[p, J_LSHO] + joints[p, J_RSHO])
        for v, (az, el) in enumerate(views):
            dh = Rot.from_rotvec(np.deg2rad(az) * up).apply(face)
            dir_ = np.cos(np.deg2rad(el)) * dh + np.sin(np.deg2rad(el)) * up
            Rs[p, v], ts[p, v] = look_at(target + a.radius * dir_, target, up)

    vt, _, _ = load_uv_data(UV_NPZ_DEFAULT)
    ft = torch.as_tensor(np.arange(len(vt), dtype=np.int32).reshape(-1, 3), device="cuda")
    vt_t = torch.as_tensor(vt, dtype=torch.float32, device="cuda")
    faces = layer.faces.astype(np.int64)
    faces_t = torch.as_tensor(faces, device="cuda")
    tex = None
    if not a.flat:
        rgb = cv2.imread(a.texture)[:, :, ::-1].astype(np.float32) / 255.0
        tex = torch.as_tensor(rgb[::-1].copy(), device="cuda").unsqueeze(0)
    base = torch.tensor([0.62, 0.62, 0.66], device="cuda")
    glctx = _get_nvdiffrast_ctx()
    out_img = os.path.join(a.out_dir, variant)
    os.makedirs(os.path.join(out_img, "masks"), exist_ok=True)
    bg = np.full((H, W, 3), 205, np.uint8)
    names_v = [f"a{int(az):+03d}_e{int(el):02d}" for az, el in views]

    for p in range(P):
        vw = torch.as_tensor(verts[p], dtype=torch.float32, device="cuda")
        loop = vw[faces_t.reshape(-1)]
        v0, v1, v2 = vw[faces_t[:, 0]], vw[faces_t[:, 1]], vw[faces_t[:, 2]]
        fn = torch.nn.functional.normalize(torch.cross(v1 - v0, v2 - v0, dim=-1), dim=-1)
        vn = torch.zeros_like(vw).scatter_add_(0, faces_t.reshape(-1, 1).expand(-1, 3), fn.repeat_interleave(3, dim=0))
        vn = torch.nn.functional.normalize(vn, dim=-1)
        for v in range(V):
            R_t = torch.as_tensor(Rs[p, v], dtype=torch.float32, device="cuda")
            with torch.no_grad():
                clips = _verts_to_clip(loop[None], K, Rs[p, v], ts[p, v], W, H)
                rast, _ = dr.rasterize(glctx, clips.contiguous(), ft, resolution=[H, W])
                n_loop = (vn @ R_t.T)[faces_t.reshape(-1)]
                n_i, _ = dr.interpolate(n_loop[None], rast, ft)
                n_i = torch.nn.functional.normalize(n_i, dim=-1)
                key = torch.nn.functional.normalize(torch.tensor([0.25, -0.45, -0.85], device="cuda"), dim=0)
                fill = torch.nn.functional.normalize(torch.tensor([-0.6, 0.1, -0.5], device="cuda"), dim=0)
                shade = 0.45 + 0.45 * (n_i * key).sum(-1, keepdim=True).clamp(0, 1) + 0.20 * (n_i * fill).sum(-1, keepdim=True).clamp(0, 1)
                if tex is None:
                    col = base * shade
                else:
                    uv, _ = dr.interpolate(vt_t[None], rast, ft)
                    col = dr.texture(tex, uv, filter_mode="linear") * shade
                col = dr.antialias(col.clamp(0, 1), rast, clips, ft)
                img = (col[0].cpu().numpy() * 255).astype(np.uint8)[:, :, ::-1]
                m = (rast[0, :, :, 3] > 0).cpu().numpy()
            a_ = m[:, :, None]
            out = np.where(a_, img, bg)
            nm = f"p{p:02d}_{names_v[v]}"
            cv2.imwrite(f"{out_img}/{nm}.jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(f"{out_img}/masks/{nm}.png", m.astype(np.uint8) * 255)
            torch.cuda.empty_cache()
        print("rendered pose", names[p], flush=True)

    np.savez(os.path.join(a.out_dir, "gt.npz"), verts=verts, joints=joints, betas=betas.numpy(), pose_names=np.array(names),
             frames=np.array([-1] + [keys[i] for i in idx]), view_names=np.array(names_v), azimuth=np.array([v[0] for v in views]),
             elevation=np.array([v[1] for v in views]), K=K, R=Rs, t=ts, width=W, height=H, faces=faces)
    print("saved", a.out_dir)


if __name__ == "__main__":
    main()
