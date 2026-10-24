"""Fit a MANO hand to triangulated 3D hand keypoints (21 joints, MediaPipe order), staged.

Targets: reconstruction_{left,right}.json ("smoothed") of any pose2d.triangulation output, e.g. the Sapiens hands
(output/silhipB/aligned/pose2d_sapiens_smoothbox) or the WiLoR hands (output/wilorTri/aligned/pose2d_wilor).
Left hands are mirrored (x -> -x), fitted with the right-hand model and mirrored back. One shape (betas) per hand,
shared over all frames; everything else is per frame.

Stages (each continues from the previous one):
  1 translation          (Kabsch-free: palm centroid match, then Adam on translation only; flat/mean hand, identity rotation)
  2 translation + wrist rotation (global orient initialised by Kabsch on the palm joints, then Adam on both)
  3 finger joints        (15x3 axis-angle offsets from the MANO mean pose; translation and rotation frozen)
  4 everything           (translation, rotation, fingers, betas)
Loss: Geman-McClure on per-joint 3D distance (sigma mm) + pose prior + betas prior + joint acceleration smoothness
(only between consecutive frames that all have a target).

  ~/miniconda3/envs/oak_env/bin/python -m hand_pose.fit_mano --recon output/silhipB/aligned/pose2d_sapiens_smoothbox \
      --out output/manoSap/mano_fit.npz
Output npz: frames[T], present[T,2] (target available), target/joints[T,2,21,3], verts[T,2,778,3] (world metres,
side 0=left, 1=right, unmirrored), faces[F,3] (winding flipped for left), params (mirrored space), stage log in <out>.json.
"""
import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hand_pose.mano_torch import Mano, rodrigues  # noqa: E402

PALM = [0, 5, 9, 13, 17]
MIRROR = np.array([-1.0, 1.0, 1.0], np.float32)


def load_targets(recon_dir, n_frames):
    tgt = np.full((n_frames, 2, 21, 3), np.nan, np.float32)
    for s, part in enumerate(("left", "right")):
        rec = json.load(open(os.path.join(recon_dir, f"reconstruction_{part}.json")))["smoothed"]
        for fk, lm in rec.items():
            f = int(fk[:6])
            if f < n_frames:
                for k, v in lm.items():
                    if int(k) < 21:
                        tgt[f, s, int(k)] = v
    return tgt


def kabsch(src, dst):
    """Rotation R (and t) such that dst ~ R src + t, batched [B,N,3]."""
    sc, dc = src.mean(1, keepdim=True), dst.mean(1, keepdim=True)
    H = (src - sc).transpose(1, 2) @ (dst - dc)
    U, _, Vh = torch.linalg.svd(H)
    d = torch.sign(torch.linalg.det(Vh.transpose(1, 2) @ U.transpose(1, 2)))
    D = torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], -1))
    R = Vh.transpose(1, 2) @ D @ U.transpose(1, 2)
    return R


def rotmat_to_aa(R):
    from scipy.spatial.transform import Rotation
    return torch.tensor(Rotation.from_matrix(R.detach().cpu().numpy()).as_rotvec(), dtype=torch.float32, device=R.device)


def fit_hand(mano, tgt_m, a, log):
    """tgt_m: [F,21,3] metres (already mirrored for left), contiguous frame ids in a['fid']."""
    dev = mano.device
    F = tgt_m.shape[0]
    tgt = torch.tensor(tgt_m, device=dev)
    msk = torch.isfinite(tgt).all(-1)
    palm = PALM
    if a.get("no_wrist_target"):
        msk[:, 0] = False  # wrist placed by the MCPs and fingers only
        palm = [k for k in PALM if k != 0]
    tgt = torch.nan_to_num(tgt)
    fid = a["fid"]
    cons = torch.tensor((fid[2:] - fid[1:-1] == 1) & (fid[1:-1] - fid[:-2] == 1), device=dev)  # centre frame t has t-1,t+1

    betas = torch.zeros(1, 10, device=dev, requires_grad=True)
    transl = torch.zeros(F, 3, device=dev, requires_grad=True)
    rot = torch.zeros(F, 3, device=dev, requires_grad=True)
    off = torch.zeros(F, 15, 3, device=dev, requires_grad=True)

    def model(b=None):
        R_root = rodrigues(rot)
        R_f = rodrigues(mano.hands_mean[None] + off)
        R = torch.cat([R_root[:, None], R_f], 1)
        return mano.forward((betas if b is None else b).expand(F, -1), R, transl)

    sig = a["sigma_mm"]
    def data_loss(j):
        r = ((j - tgt) * 1000).norm(dim=-1)
        rho = r ** 2 / (r ** 2 + sig ** 2) * sig ** 2
        return (rho * msk).sum() / F

    def smooth_loss(j):
        acc = (j[2:] - 2 * j[1:-1] + j[:-2]) * 1000
        m = cons & msk[1:-1].all(-1) & msk[:-2].all(-1) & msk[2:].all(-1)
        return ((acc ** 2).sum((-1, -2)) * m).sum() / F

    def report(name):
        with torch.no_grad():
            _, j = model()
            r = ((j - tgt) * 1000).norm(dim=-1)[msk]
            e = {"stage": name, "median_mm": float(r.median()), "mean_mm": float(r.mean()), "p95_mm": float(r.quantile(0.95))}
        log.append(e)
        print(f"   [{name}] 3D residual over all joints: median {e['median_mm']:.1f}  mean {e['mean_mm']:.1f}  p95 {e['p95_mm']:.1f} mm", flush=True)

    def run(name, params, steps, lr, with_reg):
        opt = torch.optim.Adam(params, lr=lr)
        for it in range(steps):
            opt.zero_grad()
            _, j = model()
            loss = data_loss(j)
            if with_reg:
                loss = loss + a["pose_prior"] * (off ** 2).sum() / F + a["beta_prior"] * (betas ** 2).sum() + a["smooth"] * smooth_loss(j)
            loss.backward()
            opt.step()
        report(name)

    with torch.no_grad():  # stage 1 init: palm centroid match with flat/mean hand, identity rotation
        _, j0 = mano.forward(torch.zeros(1, 10, device=dev), torch.cat([torch.eye(3, device=dev)[None], rodrigues(mano.hands_mean)])[None], torch.zeros(1, 3, device=dev))
        pm = msk[:, palm].float()[..., None]
        c_t = (tgt[:, palm] * pm).sum(1) / pm.sum(1).clamp_min(1)
        transl.copy_(c_t - j0[0, palm].mean(0))
    run("1 translation", [transl], a["steps"], 3e-3, False)

    with torch.no_grad():  # stage 2 init: Kabsch of palm joints (model palm at rot=0, then rotate about the model wrist)
        _, jm = model()
        R = kabsch((jm[:, palm] - transl[:, None]), tgt[:, palm])  # maps model palm (no transl) onto target palm
        rot.copy_(rotmat_to_aa(R))
        _, jm2 = model()
        transl.add_((tgt[:, palm] - jm2[:, palm]).mean(1))
    run("2 translation + wrist rotation", [transl, rot], a["steps"], 3e-3, False)
    run("3 finger joints", [off], a["steps"], 1e-2, True)
    run("4 everything", [transl, rot, off, betas], 2 * a["steps"], 3e-3, True)

    with torch.no_grad():
        v, j = model()
    return {"verts": v.cpu().numpy(), "joints": j.cpu().numpy(), "betas": betas[0].detach().cpu().numpy(),
            "transl": transl.detach().cpu().numpy(), "rot": rot.detach().cpu().numpy(), "off": off.detach().cpu().numpy()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recon", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-frames", type=int, default=768)
    ap.add_argument("--min-joints", type=int, default=15, help="min joints present for a frame to be fitted")
    ap.add_argument("--steps", type=int, default=300, help="Adam steps per stage (stage 4 uses 2x)")
    ap.add_argument("--sigma-mm", type=float, default=20.0)
    ap.add_argument("--pose-prior", type=float, default=2.0)
    ap.add_argument("--beta-prior", type=float, default=5.0)
    ap.add_argument("--smooth", type=float, default=0.3)
    ap.add_argument("--no-wrist-target", action="store_true", help="drop joint 0 (wrist keypoint) from the data loss")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    mano = Mano(a.device)
    tgt = load_targets(a.recon, a.n_frames)
    T = a.n_frames
    out = {k: None for k in ("verts", "joints")}
    verts = np.full((T, 2, 778, 3), np.nan, np.float32)
    joints = np.full((T, 2, 21, 3), np.nan, np.float32)
    present = np.zeros((T, 2), bool)
    params, logs = {}, {}
    for s, nm in enumerate(("left", "right")):
        ok = np.isfinite(tgt[:, s]).all(-1).sum(-1) >= a.min_joints
        fid = np.flatnonzero(ok)
        present[ok, s] = True
        t = tgt[fid, s] * (MIRROR if s == 0 else 1.0)
        print(f"== {nm} hand: {len(fid)} frames fitted", flush=True)
        log = []
        r = fit_hand(mano, t, dict(a.__dict__, fid=fid), log)
        sign = MIRROR if s == 0 else np.ones(3, np.float32)
        verts[fid, s] = r["verts"] * sign
        joints[fid, s] = r["joints"] * sign
        params[nm] = {k: r[k] for k in ("betas", "transl", "rot", "off")}
        logs[nm] = log
    faces = np.stack([mano.faces[:, ::-1], mano.faces])  # [2,F,3]: left mirrored -> flipped winding
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out, frames=np.arange(T), present=present, target=tgt, joints=joints, verts=verts, faces=faces,
                        **{f"{nm}_{k}": v for nm, p in params.items() for k, v in p.items()})
    json.dump({"args": {k: v for k, v in a.__dict__.items()}, "stages": logs}, open(os.path.splitext(a.out)[0] + ".json", "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
