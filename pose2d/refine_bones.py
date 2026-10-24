"""Sequence-level body refinement with bone-length consistency.

Fits all body joints (ids 0-12) of a take at once: robust reprojection error in every camera that sees the joint, constant
bone lengths shared by all frames (free parameters, left/right tied by default) and a second-difference smoothness term.
Initialised from an existing triangulation (e.g. the hip-cams + gate output); the result is written as that
reconstruction's "smoothed" entry so it can be compared with the Kalman result in sapiens2.view_compare.

  python -m pose2d.refine_bones --take T --calib CAL --pose2d-dir pose2d_sapiens_smoothbox \
      --init-dir output/silhipB/aligned/pose2d_sapiens_smoothbox --out-dir output/bones
"""
import argparse, json, os, shutil, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import calibrate  # noqa: E402
from hand_pose import hand_multiview as hmv  # noqa: E402
from pose2d import triangulation as T  # noqa: E402

JOINTS = list(range(13))
HIPS = (11, 12)
# (name, [(a, b), ...]); pairs in one group share a length when --tie is on, otherwise each pair is its own parameter.
BONES = [
    ("shoulder_width", [(5, 6)]),
    ("hip_width", [(11, 12)]),
    ("torso", [(5, 11), (6, 12)]),
    ("upper_arm", [(5, 7), (6, 8)]),
    ("forearm", [(7, 9), (8, 10)]),
    ("nose_eye", [(0, 1), (0, 2)]),
    ("eye_ear", [(1, 3), (2, 4)]),
    ("nose_ear", [(0, 3), (0, 4)]),
    ("eye_dist", [(1, 2)]),
    ("ear_dist", [(3, 4)]),
]


def bone_groups(tie):
    groups = []
    for name, pairs in BONES:
        if tie or len(pairs) == 1:
            groups.append((name, pairs))
        else:
            groups.extend((f"{name}_{'LR'[i]}", [p]) for i, p in enumerate(pairs))
    return groups


def load_init(init_dir):
    with open(os.path.join(init_dir, "reconstruction_body.json")) as f:
        rec = json.load(f)
    frames = sorted(set(rec["raw"]) | set(rec["smoothed"]))
    X = np.full((len(frames), len(JOINTS), 3), np.nan)
    for t, fk in enumerate(frames):
        for j in JOINTS:
            p = rec["smoothed"].get(fk, {}).get(str(j))
            if p is None:
                p = rec["raw"].get(fk, {}).get(str(j))
            if p is not None:
                X[t, j] = p
    return rec, frames, X


def fill_missing(X):
    """Linear interpolation (edge hold) of NaN runs per joint/axis; a joint that is never valid falls back to the joint mean."""
    X = X.copy()
    t = np.arange(len(X))
    for j in range(X.shape[1]):
        ok = np.isfinite(X[:, j, 0])
        if not ok.any():
            X[:, j] = np.nanmean(X[np.isfinite(X[:, :, 0]).any(1)][:, :, :].reshape(-1, 3), axis=0)
            continue
        for a in range(3):
            X[:, j, a] = np.interp(t, t[ok], X[ok, j, a])
    return X


def collect_observations(data, P, W, H, frames, min_conf, hip_cams):
    L, C, LC = data["landmarks_body"], data["confidence_body"], data["landmark_confidence_body"]
    cams = list(P)
    rows = []
    for t, fk in enumerate(frames):
        for ci, c in enumerate(cams):
            if C.get(c, {}).get(fk, 0.0) < min_conf:
                continue
            for j in JOINTS:
                if j in HIPS and hip_cams is not None and c not in hip_cams:
                    continue
                p = L.get(c, {}).get(fk, {}).get(str(j))
                if p is None:
                    continue
                w = 1.0 if LC is None else float(np.clip(LC.get(c, {}).get(fk, {}).get(str(j), 1.0), 0.0, 1.0))
                rows.append((t, j, ci, p[0] * W, p[1] * H, w))
    a = np.array(rows, dtype=np.float64)
    return a, cams


def pixel_residuals(X, obs, Pm):
    t, j, c = obs[:, 0].long(), obs[:, 1].long(), obs[:, 2].long()
    Xh = torch.cat([X[t, j], torch.ones(len(t), 1, dtype=X.dtype, device=X.device)], dim=1)
    p = torch.einsum("nij,nj->ni", Pm[c], Xh)
    return p[:, :2] / p[:, 2:3] - obs[:, 3:5]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--take", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--pose2d-dir", default="pose2d_sapiens_smoothbox")
    ap.add_argument("--init-dir", required=True, help="dir with reconstruction_{left,right,body}.json to start from")
    ap.add_argument("--out-dir", required=True, help="stand-in take dir; output goes to <out-dir>/aligned/<pose2d-dir>/")
    ap.add_argument("--hip-cams", default="cam0,cam1,cam3,cam4", help="cameras allowed for the hips ('all' for every camera)")
    ap.add_argument("--min-confidence", type=float, default=0.5)
    ap.add_argument("--sigma-px", type=float, default=8.0, help="reprojection scale of the robust (Cauchy) loss")
    ap.add_argument("--hip-sigma-px", type=float, default=20.0, help="same, for the hips (their 2D points disagree by 15-50 px)")
    ap.add_argument("--bone-sigma-mm", type=float, default=5.0)
    ap.add_argument("--accel-sigma-mm", type=float, default=4.0)
    ap.add_argument("--no-tie", action="store_true", help="free left/right bone lengths instead of one shared length")
    ap.add_argument("--len-range", type=float, default=None,
                    help="bound each bone length to +-this fraction of its initial median (sigmoid); default unbounded")
    ap.add_argument("--no-bones", action="store_true", help="ablation: drop the bone-length term")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    calib = calibrate.load_calibration_output(a.calib)
    data = T.load_pose2d_take_data(a.take, pose2d_dir=a.pose2d_dir)
    cams = [c for c in data["cam_ids"] if c in calib]
    P = T.build_projection_matrices(calib, cams)
    W, H = hmv.get_image_size(a.take, cams[0], next(iter(data["landmarks_body"][cams[0]])))
    rec, frames, X0 = load_init(a.init_dir)
    X0 = fill_missing(X0)
    hip_cams = None if a.hip_cams == "all" else set(a.hip_cams.split(","))
    obs_np, cam_order = collect_observations(data, P, W, H, frames, a.min_confidence, hip_cams)
    print(f"{len(frames)} frames, {len(obs_np)} observations, cameras {cam_order}")

    dev, dt = torch.device(a.device), torch.float64
    obs = torch.tensor(obs_np, dtype=dt, device=dev)
    Pm = torch.tensor(np.stack([P[c] for c in cam_order]), dtype=dt, device=dev)
    is_hip = torch.isin(obs[:, 1].long(), torch.tensor(HIPS, device=dev))
    sig = torch.where(is_hip, a.hip_sigma_px, a.sigma_px).to(dt)
    w = obs[:, 5]

    groups = bone_groups(not a.no_tie)
    pair_a = torch.tensor([p[0] for _, ps in groups for p in ps], device=dev)
    pair_b = torch.tensor([p[1] for _, ps in groups for p in ps], device=dev)
    pair_g = torch.tensor([g for g, (_, ps) in enumerate(groups) for _ in ps], device=dev)
    X0t = torch.tensor(X0, dtype=dt, device=dev)
    d0 = (X0t[:, pair_a] - X0t[:, pair_b]).norm(dim=-1)
    L0 = torch.stack([d0[:, pair_g == g].median() for g in range(len(groups))])
    print("initial bone lengths (mm):", {n: round(float(l) * 1000, 1) for (n, _), l in zip(groups, L0)})

    X = X0t.clone().requires_grad_(True)
    logL = torch.log(L0).clone().requires_grad_(True)
    bone_s, acc_s = a.bone_sigma_mm / 1000.0, a.accel_sigma_mm / 1000.0

    def lengths():
        if a.len_range is None:
            return torch.exp(logL)
        # sigmoid-bounded to +-len_range of the initial median; logL=log(L0) maps to the initial value
        return L0 * (1 - a.len_range + 2 * a.len_range * torch.sigmoid(logL - torch.log(L0)))

    def terms():
        r = pixel_residuals(X, obs, Pm).norm(dim=1)
        reproj = (w * sig ** 2 * torch.log1p((r / sig) ** 2)).sum() / a.sigma_px ** 2
        d = (X[:, pair_a] - X[:, pair_b]).norm(dim=-1)
        bone = (((d - lengths()[pair_g]) / bone_s) ** 2).sum()
        acc = ((X[2:] - 2 * X[1:-1] + X[:-2]) / acc_s).pow(2).sum()
        return reproj, bone, acc

    opt = torch.optim.LBFGS([X, logL], lr=1.0, max_iter=a.iters, history_size=50, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-9, tolerance_change=1e-12)

    def closure():
        opt.zero_grad()
        rp, bn, ac = terms()
        loss = rp + (0.0 if a.no_bones else bn) + ac
        loss.backward()
        return loss

    for rnd in range(4):
        opt.step(closure)
        with torch.no_grad():
            rp, bn, ac = (float(v) for v in terms())
        print(f"round {rnd}: reproj {rp:.0f}  bone {bn:.0f}  accel {ac:.0f}")

    Xo = X.detach().cpu().numpy()
    Lo = lengths().detach().cpu().numpy()
    print("fitted bone lengths (mm):", {n: round(float(l) * 1000, 1) for (n, _), l in zip(groups, Lo)})
    with torch.no_grad():
        r = pixel_residuals(X, obs, Pm).norm(dim=1).cpu().numpy()
    print(f"reprojection residual px: median {np.median(r):.1f}  p90 {np.percentile(r, 90):.1f}")

    out = os.path.join(a.out_dir, "aligned", a.pose2d_dir)
    os.makedirs(out, exist_ok=True)
    for f in os.listdir(a.init_dir):
        if f != "reconstruction_body.json":
            shutil.copy(os.path.join(a.init_dir, f), out)
    smoothed = {fk: {str(j): Xo[t, j].tolist() for j in JOINTS} for t, fk in enumerate(frames)}
    with open(os.path.join(out, "reconstruction_body.json"), "w") as f:
        json.dump({"raw": rec["raw"], "smoothed": smoothed}, f)
    print("wrote", out)


if __name__ == "__main__":
    main()
