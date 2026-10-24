"""Golden-master check for the Sapiens -> SMPL-X pipeline on 20261022_171905/take_2 (60 fps, 8 cameras, half-res calibration).

The reference outputs are frozen in a directory outside git (default output/golden_t2_171905: fits, leg triangulation, bone-refined body,
floor map, plus the stand-in take dirs they were made from under output/t2_171905). `replay` re-runs a fit with the code of a given
checkout (a `git worktree` of a commit state; the working tree is not touched) and `compare` checks the result against the reference.

The GPU fit is not bit-reproducible (two identical runs differ by about 1% in the stage-3 loss), so the fit is compared with tolerances;
`baseline` measures that spread by comparing two replays of the same code. The deterministic steps (leg triangulation, floor alignment)
are compared tightly.

  PYTHONPATH=. python -m smplx_fit.golden_compare replay  --repo <checkout> --case freeze --out <dir>
  PYTHONPATH=. python -m smplx_fit.golden_compare compare --case freeze --new <dir>
  PYTHONPATH=. python -m smplx_fit.golden_compare compare --case freeze --new <dirA> --ref <dirB>     # run-to-run spread
  PYTHONPATH=. python -m smplx_fit.golden_compare check   --repo <checkout> --case legsoff --out <dir> [--attempts 2]
  PYTHONPATH=. python -m smplx_fit.golden_compare legs    --repo <checkout> --out <npz>             # leg triangulation vs reference
  PYTHONPATH=. python -m smplx_fit.golden_compare floor   --repo <checkout> --case legs --out <dir>  # floor alignment vs reference

Cases: freeze (torso frozen in stage 4, no legs), legs (+ leg targets), legsoff (+ leg offsets).

`check` = replay + compare, retried up to --attempts times, and passes if any attempt passes. Reason: in the baseline runs of the unchanged code
the freeze and legs cases reproduced the reference in 2 of 2 runs, but one of two legsoff runs landed in a different solution (upper-body mesh
5 mm and right shoulder 10 mm away from the reference, and 5 mm from the run that matched), so a single failing run is not proof of a regression.
"""
import argparse, json, os, shutil, subprocess, sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN = os.path.join(HERE, "output", "golden_t2_171905")
STANDIN = os.path.join(HERE, "output", "t2_171905", "smplx_in_freeze")        # take dir with the data links the fits read
CALIB = "/mnt/mmocap/temp_calib/20261022_173214_8cam_half.json"
POSE2D = "pose2d_sapiens_smoothbox"
SUB = os.path.join("aligned", POSE2D, "smplx")

BASE_ARGS = ["--pose2d-dir", POSE2D, "--gender", "male", "--num-betas", "20", "--calib", CALIB, "--use-silhouette",
             "--silhouette-weight", "0.1", "--mask-subdir", "masks_rvm", "--silhouette-out-size", "32x24",
             "--silhouette-n-samples", "1500", "--silhouette-sigma-px", "1.0", "--hand-k3d-weight", "10",
             "--body-smooth-weight", "5", "--shape-reg-weight", "1.0", "--n-shape-betas", "0", "--hand-reg-weight", "0.0001",
             "--pose-prior-weight", "0.01", "--max-outer-iters", "10", "--force"]
CASES = {
    "freeze": ["--stage4-freeze-torso"],
    "legs": ["--stage4-freeze-torso", "--legs-npz", os.path.join(GOLDEN, "legs.npz")],
    "legsoff": ["--stage4-freeze-torso", "--legs-npz", os.path.join(GOLDEN, "legs.npz"), "--sapiens-leg-offsets"],
}
REF_DIR = {"freeze": "smplx_in_freeze", "legs": "smplx_in_legs", "legsoff": "smplx_in_legsoff"}
FLOOR_REF = {"legs": "smplx_floor_full", "legsoff": "smplx_floor_legsoff"}

# Default tolerances (the baseline run-to-run spread, times two, is the lower bound; see `baseline` output).
# Vertex differences are checked per region: the legs of the `freeze` case have no targets (only the pose prior holds them), so they
# differ by ~13-21 mm between identical runs and are only reported there; in the leg cases they are checked.
TOL = {"stage_loss_rel": 0.03, "body_rms_mm": 1.5, "hand_rms_mm": 1.5, "landmark_mm": 5.0,
       "upper_mean_mm": 3.0, "upper_p95_mm": 8.0, "hands_mean_mm": 2.0, "hands_p95_mm": 5.0,
       "lower_mean_mm": 6.0, "lower_p95_mm": 20.0, "accel_median_rel": 0.15, "det_mm": 1e-3, "bones_mm": 0.5}
# With leg targets the whole body is constrained and two identical runs agree much more closely (upper body <= 1 mm mean / 2.3 mm p95,
# hands <= 0.8 / 1.9, legs <= 0.7 / 1.6, worst landmark 1.2 mm in the baseline runs); the limits are about twice that spread.
TOL_LEGS = dict(TOL, stage_loss_rel=0.02, landmark_mm=2.5, upper_mean_mm=2.0, upper_p95_mm=5.0, hands_mean_mm=1.5, hands_p95_mm=4.0,
                lower_mean_mm=1.5, lower_p95_mm=4.0, accel_median_rel=0.10)


def standin_copy(dst):
    """Copy of the stand-in take dir (symlinks to the aligned frames / masks / reconstructions) without any fit output."""
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(STANDIN, dst, symlinks=True, ignore=shutil.ignore_patterns("smplx"))


def replay(repo, case, out):
    standin_copy(out)
    cmd = [sys.executable, "-m", "smplx_fit.fit_take", out, "--model-path", os.path.join(HERE, "models", "SMPLX")] + BASE_ARGS + CASES[case]
    env = dict(os.environ, PYTHONPATH=repo)
    print("running in", repo, ":", " ".join(cmd[2:6]), "...", flush=True)
    subprocess.run(cmd, cwd=repo, env=env, check=True)


def load_fit_arrays(take_dir, repo_import=True):
    base = os.path.join(take_dir, SUB)
    d = dict(np.load(os.path.join(base, "smplx_params.npz"), allow_pickle=True))
    log = json.load(open(os.path.join(base, "optimization_log.json")))
    cfg = json.load(open(os.path.join(base, "fit_config.json")))
    return d, log, cfg


def final_stage_losses(log):
    out = {}
    for name, rec in log["stages"].items():
        last = rec[-1] if isinstance(rec, list) and rec else rec
        if isinstance(last, dict) and "total" in last:
            out[name] = last["total"]
    return out


def points_and_vertices(d, cfg, layout, layer, device):
    import torch
    from smplx_fit import model as sm
    t = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)  # noqa: E731
    F = len(d["frame_keys"])
    with torch.no_grad():
        o = sm.forward(layer, t(d["betas"]).reshape(1, -1).expand(F, -1), t(d["global_orient"]), t(d["body_pose"]),
                       t(d["lhand_pose"]), t(d["rhand_pose"]), t(d["transl"]))
        pts = sm.predict_all_points(o, layout)
    return pts.cpu().numpy(), o.vertices.cpu().numpy()


def compare_fits(ref_take, new_take, tol):
    import torch
    sys.path.insert(0, HERE)
    from smplx_fit import data_loading as dl, model as sm
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dr, lr, cr = load_fit_arrays(ref_take)
    dn, ln, cn = load_fit_arrays(new_take)
    has_legs = cr.get("legs_npz") is not None
    if has_legs:
        tol = TOL_LEGS
    layer = sm.load_layer(os.path.join(HERE, "models", "SMPLX"), gender=cr["gender"], num_betas=cr["num_betas"]).to(dev)
    _, layout, _, _ = dl.build_take_arrays(STANDIN, pose2d_dir=POSE2D, legs_npz=cr.get("legs_npz"))
    pr, vr = points_and_vertices(dr, cr, layout, layer, dev)
    pn, vn = points_and_vertices(dn, cn, layout, layer, dev)
    names = [l[2].name for l in layout]
    rows, ok = [], True

    def check(label, value, limit, unit=""):
        nonlocal ok
        good = value <= limit
        ok &= good
        rows.append((label, value, limit, unit, good))

    lo, ln_ = final_stage_losses(lr), final_stage_losses(ln)
    for st in lo:
        if st in ln_ and lo[st] > 0:
            check(f"final loss {st}", abs(ln_[st] - lo[st]) / lo[st], tol["stage_loss_rel"], "(rel)")
    for k in ("body_rms_mm", "hand_rms_mm"):
        if lr["fit_quality"].get(k) is not None:
            check(k + " diff", abs(ln["fit_quality"][k] - lr["fit_quality"][k]), tol[k], "mm")
    dp = np.linalg.norm(pn - pr, axis=-1).mean(0) * 1000
    check("worst per-landmark mean distance", float(dp.max()), tol["landmark_mm"], f"mm ({names[int(dp.argmax())]})")
    dv = np.linalg.norm(vn - vr, axis=-1) * 1000                       # (frames, vertices), mm
    tv = layer.v_template.detach().cpu().numpy(); jt = layer.J_regressor.detach().cpu().numpy() @ tv
    upper = tv[:, 1] >= 0.5 * (jt[0, 1] + jt[3, 1])                      # above the waist (rest pose, y up)
    hands = np.isin(layer.lbs_weights.detach().cpu().numpy().argmax(1), list(range(20, 55)))
    regions = {"upper body": upper & ~hands, "hands": hands, "lower body": ~upper}
    for name, key in (("upper body", "upper"), ("hands", "hands"), ("lower body", "lower")):
        m = regions[name]
        mean, p95 = float(dv[:, m].mean()), float(np.percentile(dv[:, m], 95))
        if name == "lower body" and not has_legs:
            print(f"  info {'mesh vertex distance, lower body (no leg targets, not checked)':42s} mean {mean:.1f} / p95 {p95:.1f} mm")
            continue
        check(f"mesh distance {name}, mean", mean, tol[key + "_mean_mm"], "mm")
        check(f"mesh distance {name}, p95", p95, tol[key + "_p95_mm"], "mm")
    ar = np.sqrt((np.diff(pr, 2, axis=0) ** 2).sum(-1).mean(0)); an = np.sqrt((np.diff(pn, 2, axis=0) ** 2).sum(-1).mean(0))
    check("median joint acceleration (rel diff)", abs(float(np.median(an) / np.median(ar)) - 1.0), tol["accel_median_rel"], "")
    for label, v, lim, unit, good in rows:
        print(f"  {'ok  ' if good else 'FAIL'} {label:42s} {v:9.4f}  (limit {lim:g}) {unit}")
    return ok


def compare_arrays(ref, new, tol_mm, label):
    a, b = np.load(ref), np.load(new)
    worst = 0.0
    for k in a.files:
        if a[k].dtype.kind in "fc" and k in b.files:
            worst = max(worst, float(np.nanmax(np.abs(a[k] - b[k]))) * (1000 if k in ("X",) else 1))
    good = worst <= tol_mm
    print(f"  {'ok  ' if good else 'FAIL'} {label}: max abs difference {worst:.6f} (limit {tol_mm:g})")
    return good


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["replay", "compare", "check", "legs", "floor"])
    ap.add_argument("--attempts", type=int, default=2)
    ap.add_argument("--repo", default=HERE, help="checkout whose code is replayed")
    ap.add_argument("--case", choices=sorted(CASES), default="freeze")
    ap.add_argument("--out")
    ap.add_argument("--new", help="take dir of the fit to compare")
    ap.add_argument("--ref", help="reference take dir (default: the frozen golden fit for the case)")
    ap.add_argument("--golden", default=GOLDEN)
    a = ap.parse_args()
    if a.action == "replay":
        replay(os.path.abspath(a.repo), a.case, a.out)
    elif a.action == "check":
        passed = False
        for i in range(a.attempts):
            out = f"{a.out}_{i}"
            replay(os.path.abspath(a.repo), a.case, out)
            ref = os.path.join(a.golden, "_ref_" + a.case)
            os.makedirs(os.path.join(ref, SUB), exist_ok=True)
            for f in os.listdir(os.path.join(a.golden, REF_DIR[a.case])):
                shutil.copy(os.path.join(a.golden, REF_DIR[a.case], f), os.path.join(ref, SUB, f))
            print(f"attempt {i + 1}/{a.attempts}: compare {a.case}")
            if compare_fits(ref, out, TOL):
                passed = True
                break
        print("PASS" if passed else "FAIL")
        sys.exit(0 if passed else 1)
    elif a.action == "compare":
        ref = a.ref or None
        if ref is None:   # the frozen reference holds only the smplx folder; wrap it in a take-dir layout
            ref = os.path.join(a.golden, "_ref_" + a.case)
            os.makedirs(os.path.join(ref, SUB), exist_ok=True)
            for f in os.listdir(os.path.join(a.golden, REF_DIR[a.case])):
                shutil.copy(os.path.join(a.golden, REF_DIR[a.case], f), os.path.join(ref, SUB, f))
        print(f"compare {a.case}: {a.new} vs {ref}")
        sys.exit(0 if compare_fits(ref, a.new, TOL) else 1)
    elif a.action == "legs":
        cmd = [sys.executable, "-m", "pose2d.triangulate_legs", "--npz", os.path.join(HERE, "output", "sapiens2_171905_t2", "allframes_1b_smoothbox.npz"),
               "--calib", CALIB, "--out", a.out]
        subprocess.run(cmd, cwd=os.path.abspath(a.repo), env=dict(os.environ, PYTHONPATH=os.path.abspath(a.repo)), check=True)
        sys.exit(0 if compare_arrays(os.path.join(a.golden, "legs.npz"), a.out, TOL["det_mm"], "leg triangulation (mm)") else 1)
    elif a.action == "floor":
        fit = a.new
        cmd = [sys.executable, "-m", "smplx_fit.add_legs", "--fit", fit, "--floor-map", os.path.join(a.golden, "calib_half_floor_map.npz"),
               "--out", a.out, "--align-only", "--model-path", os.path.join(HERE, "models", "SMPLX")]
        subprocess.run(cmd, cwd=os.path.abspath(a.repo), env=dict(os.environ, PYTHONPATH=os.path.abspath(a.repo)), check=True)
        new = np.load(os.path.join(a.out, SUB, "smplx_params.npz"))
        ref = np.load(os.path.join(a.golden, FLOOR_REF[a.case], "smplx_params.npz"))
        worst = max(float(np.abs(new[k] - ref[k]).max()) for k in ("global_orient", "transl", "body_pose", "betas"))
        good = worst <= 1e-5
        print(f"  {'ok  ' if good else 'FAIL'} floor-aligned {a.case}: max abs parameter difference {worst:.2e} (limit 1e-05)")
        sys.exit(0 if good else 1)


if __name__ == "__main__":
    main()
