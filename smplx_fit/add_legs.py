"""Re-express a finished SMPL-X fit in a floor-aligned world and add the legs, leaving the upper body and hands untouched.

1. World alignment: X_floor = Rg X_calib + tg (the rigid map in calib_*_floor_map.npz). SMPL-X rotates about the pelvis joint J0
   (a function of the shape only) and then adds transl, so the exact equivalent parameters are
   global_orient' = Rg R_go,  transl' = Rg (J0 + transl) + tg - J0.   Verified by comparing vertices.
2. Legs: only body_pose joints 0, 1, 3, 4, 6, 7, 9, 10 (hips, knees, ankles, feet) are optimised, against the triangulated
   Sapiens leg points (pose2d.triangulate_legs, smoothed in time) plus a second-difference smoothness term. Every other parameter
   is frozen, so the body and hands stay exactly as fitted.

  python -m smplx_fit.add_legs --fit output/t2_171905/smplx_in_freeze --pose2d-dir pose2d_sapiens_smoothbox \
      --legs output/t2_171905/legs.npz --floor-map output/t2_171905/calib_half_floor_map.npz --out output/t2_171905/smplx_floor
"""
import argparse, json, os, shutil, sys
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.spatial.transform import Rotation
from smplx.vertex_ids import vertex_ids as _VIDS

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from smplx_fit import model as sm  # noqa: E402

SUB = "aligned/{}/smplx"
LEG_JOINTS = [0, 1, 3, 4, 6, 7, 9, 10]            # body_pose indices: l/r hip, l/r knee, l/r ankle, l/r foot
V = _VIDS["smplx"]
# order of pose2d.triangulate_legs.NAMES -> ("joint"|"vertex", index into SMPL-X joints / vertices)
TARGETS = [("joint", 4), ("joint", 5), ("joint", 7), ("joint", 8),
           ("vertex", V["LBigToe"]), ("vertex", V["LSmallToe"]), ("vertex", V["LHeel"]),
           ("vertex", V["RBigToe"]), ("vertex", V["RSmallToe"]), ("vertex", V["RHeel"])]


def smooth_track(X, sigma):
    """X (F, 3) with NaN gaps -> interpolated, 5-frame median then Gaussian."""
    X = X.copy()
    t = np.arange(len(X))
    for k in range(3):
        ok = np.isfinite(X[:, k])
        X[:, k] = np.interp(t, t[ok], X[ok, k])
    X = median_filter(X, size=(5, 1), mode="nearest")
    return gaussian_filter1d(X, sigma, axis=0, mode="nearest")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", required=True, help="take dir of the finished fit")
    ap.add_argument("--pose2d-dir", default="pose2d_sapiens_smoothbox")
    ap.add_argument("--legs", default=None, help="not needed with --align-only")
    ap.add_argument("--floor-map", required=True)
    ap.add_argument("--out", required=True, help="stand-in take dir for the result (the nolegs variant gets '_nolegs' appended)")
    ap.add_argument("--model-path", default="models/SMPLX")
    ap.add_argument("--smooth-sigma", type=float, default=2.0, help="Gaussian sigma (frames) applied to the leg targets")
    ap.add_argument("--smooth-weight", type=float, default=1.0, help="weight of the leg pose second difference (rad^2 per frame^2)")
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--align-only", action="store_true", help="only re-express the fit in the floor frame (no leg stage)")
    ap.add_argument("--leg-length", action="store_true",
                    help="also optimise the shape (betas) so the legs get the right length; the pelvis joint stays where it is and "
                         "a penalty keeps the above-waist mesh at its current position")
    ap.add_argument("--hold-weight", type=float, default=20.0, help="weight of the above-waist mesh hold (mean squared distance, m^2)")
    ap.add_argument("--beta-lr", type=float, default=0.1)
    ap.add_argument("--beta-weight", type=float, default=0.05, help="L2 on the change of betas")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    src = os.path.join(a.fit, SUB.format(a.pose2d_dir))
    d = dict(np.load(os.path.join(src, "smplx_params.npz"), allow_pickle=True))
    cfg = json.load(open(os.path.join(src, "fit_config.json")))
    layer = sm.load_layer(a.model_path, gender=cfg["gender"], num_betas=cfg["num_betas"]).to(dev)
    t = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=dev)  # noqa: E731
    F = len(d["frame_keys"])
    betas = t(d["betas"]).reshape(1, -1)

    def run(go, bp, tr, b=None):
        return sm.forward(layer, (betas if b is None else b).expand(F, -1), go, bp, t(d["lhand_pose"]), t(d["rhand_pose"]), tr)

    def pelvis(b):
        z_ = torch.zeros(1, 3, device=dev)
        return sm.forward(layer, b, z_, torch.zeros(1, 21, 3, device=dev), torch.zeros(1, 15, 3, device=dev),
                          torch.zeros(1, 15, 3, device=dev), z_).joints[0, 0]

    with torch.no_grad():
        old = run(t(d["global_orient"]), t(d["body_pose"]), t(d["transl"]))
        z = torch.zeros(1, 3, device=dev)
        J0 = sm.forward(layer, betas, z, torch.zeros(1, 21, 3, device=dev), torch.zeros(1, 15, 3, device=dev),
                        torch.zeros(1, 15, 3, device=dev), z).joints[0, 0].cpu().numpy()
    fm = np.load(a.floor_map)
    Rg, tg = fm["Rg"].astype(np.float64), fm["tg"].astype(np.float64)
    R_go = Rotation.from_rotvec(np.asarray(d["global_orient"], np.float64))
    go_new = (Rotation.from_matrix(Rg) * R_go).as_rotvec()
    tr_new = (np.asarray(d["transl"], np.float64) + J0) @ Rg.T + tg - J0
    with torch.no_grad():
        new = run(t(go_new), t(d["body_pose"]), t(tr_new))
    expect = old.vertices.cpu().numpy() @ Rg.T + tg
    print("world alignment check: max vertex error %.4f mm" % (np.abs(new.vertices.cpu().numpy() - expect).max() * 1000))

    if a.align_only:
        dst = os.path.join(a.out, SUB.format(a.pose2d_dir))
        os.makedirs(dst, exist_ok=True)
        nd = dict(d); nd.update(global_orient=go_new.astype(np.float32), transl=tr_new.astype(np.float32))
        np.savez(os.path.join(dst, "smplx_params.npz"), **nd)
        c2 = dict(cfg); c2.update(world="floor", source_fit=a.fit, floor_map=a.floor_map)
        json.dump(c2, open(os.path.join(dst, "fit_config.json"), "w"), indent=2)
        print("wrote", dst)
        return

    L = np.load(a.legs)
    ks = [int(str(k)[:6]) if not isinstance(k, (int, np.integer)) else int(k) for k in d["frame_keys"]]
    assert list(L["frames"]) == ks, "leg frames differ from the fit's frames"
    Xl = np.stack([smooth_track(L["X"][:, j] @ Rg.T + tg, a.smooth_sigma) for j in range(L["X"].shape[1])], 1)   # (F, 10, 3), floor frame
    w = np.clip((L["ncam"] - 1) / 3.0, 0, 1)
    w[~np.isfinite(L["X"][..., 0])] = 0.0
    tgt, wt = t(Xl), t(w)

    def pred_legs(out):
        pts = [out.joints[:, i] if kind == "joint" else out.vertices[:, i] for kind, i in TARGETS]
        return torch.stack(pts, 1)

    mask = torch.zeros(1, 21, 3, device=dev)
    mask[:, LEG_JOINTS] = 1.0
    bp0 = t(d["body_pose"])
    delta = torch.zeros_like(bp0, requires_grad=True)
    go_t, tr_t = t(go_new), t(tr_new)
    dbeta = torch.zeros_like(betas, requires_grad=True)
    opt = torch.optim.Adam([{"params": [delta], "lr": 0.02}] + ([{"params": [dbeta], "lr": a.beta_lr}] if a.leg_length else []))
    tv_ = layer.v_template.detach().cpu().numpy(); jt_ = layer.J_regressor.detach().cpu().numpy() @ tv_
    upper_idx = torch.as_tensor(np.nonzero(tv_[:, 1] >= 0.5 * (jt_[0, 1] + jt_[3, 1]))[0][::8], device=dev)
    J0_t = t(J0)
    hold_ref = new.vertices[:, upper_idx].detach()
    with torch.no_grad():
        e0 = (pred_legs(new) - tgt).norm(dim=-1)
    for it in range(a.iters):
        opt.zero_grad()
        bp = bp0 + delta * mask
        if a.leg_length:
            b_ = betas + dbeta
            out = run(go_t, bp, tr_t + J0_t - pelvis(b_), b_)      # pelvis joint keeps its position when the shape changes
        else:
            out = run(go_t, bp, tr_t)
        err = ((pred_legs(out) - tgt) ** 2).sum(-1)
        k3d = (wt * err).sum() / wt.sum().clamp(min=1)
        lp = bp[:, LEG_JOINTS]
        smooth = ((lp[2:] - 2 * lp[1:-1] + lp[:-2]) ** 2).sum(-1).mean()
        loss = k3d + a.smooth_weight * smooth + 1e-4 * (delta * mask).pow(2).mean()
        if a.leg_length:
            hold = ((out.vertices[:, upper_idx] - hold_ref) ** 2).sum(-1).mean()
            loss = loss + a.hold_weight * hold + a.beta_weight * dbeta.pow(2).sum()
        loss.backward(); opt.step()
        if it % 100 == 0 or it == a.iters - 1:
            print(f"iter {it}: k3d {k3d.item():.5f} smooth {smooth.item():.6f}", flush=True)
    with torch.no_grad():
        bp = bp0 + delta * mask
        if a.leg_length:
            betas_f = betas + dbeta
            tr_f = tr_t + J0_t - pelvis(betas_f)
            out = run(go_t, bp, tr_f, betas_f)
            print("betas change (L2) %.3f" % dbeta.norm().item())
        else:
            betas_f, tr_f = betas, tr_t
            out = run(go_t, bp, tr_t)
        e1 = (pred_legs(out) - tgt).norm(dim=-1)
        upper = np.ones(out.vertices.shape[1], bool)
        tv = layer.v_template.cpu().numpy(); jt = layer.J_regressor.cpu().numpy() @ tv
        upper = tv[:, 1] >= 0.5 * (jt[0, 1] + jt[3, 1])
        dv = (out.vertices - new.vertices).norm(dim=-1).cpu().numpy()
        print("above-waist vertex shift: mean %.1f mm, p95 %.1f mm" % (dv[:, upper].mean() * 1000, np.percentile(dv[:, upper], 95) * 1000))
    names = list(L["names"])
    print("leg target error mm (before -> after), mean over frames:")
    for j, n in enumerate(names):
        print(f"  {n:12s} {e0[:, j].mean().item() * 1000:6.1f} -> {e1[:, j].mean().item() * 1000:6.1f}")
    print("above-waist vertices moved by the leg stage: max %.3f mm" % (dv[:, upper].max() * 1000))
    v = out.vertices.cpu().numpy()
    print("lowest mesh point (m, floor = 0): median %.3f  min %.3f" % (np.median(v[..., 2].min(1)), v[..., 2].min()))

    tr_out = tr_f.cpu().numpy()
    for tag, go_, bp_, tr_, legs in (("", go_new, bp.cpu().numpy(), tr_out, True), ("_nolegs", go_new, d["body_pose"], tr_new, False)):
        dst_take = a.out + tag
        dst = os.path.join(dst_take, SUB.format(a.pose2d_dir))
        os.makedirs(dst, exist_ok=True)
        nd = dict(d); nd.update(betas=(betas_f.cpu().numpy() if legs else np.asarray(d['betas'])).astype(np.float32), global_orient=go_.astype(np.float32), body_pose=np.asarray(bp_, np.float32), transl=tr_.astype(np.float32))
        np.savez(os.path.join(dst, "smplx_params.npz"), **nd)
        c2 = dict(cfg); c2.update(world="floor", legs=legs, source_fit=a.fit, floor_map=a.floor_map)
        json.dump(c2, open(os.path.join(dst, "fit_config.json"), "w"), indent=2)
        print("wrote", dst)


if __name__ == "__main__":
    main()
