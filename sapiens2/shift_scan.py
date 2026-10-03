"""Scan per-camera frame shifts around a frame; pick shifts minimising multi-view reprojection error.

  PYTHONPATH=/home/fmalmb/CODE/sapiens2:. python -m sapiens2.shift_scan --take ... --frame 300 --size 1b
Objective = median plain-DLT reprojection error (distorted px) over all keypoint views with conf > thr.
"""
import argparse, numpy as np, cv2, torch
import calibrate
from sapiens2 import run_pose as rp
from sapiens2.triangulate import proj, dlt


def objective(kp, sc, cams, ids, thr, per_cam=False):
    P = {c: cams[c]["K"] @ np.hstack([cams[c]["R"], cams[c]["t"].reshape(3, 1)]) for c in ids}
    ud = {c: cv2.undistortPoints(kp[c][:, None].astype(np.float64), cams[c]["K"], cams[c]["dist"], P=cams[c]["K"]).reshape(-1, 2) for c in ids}
    errs = {c: [] for c in ids}
    for k in range(len(kp[ids[0]])):
        vs = [c for c in ids if sc[c][k] > thr]
        if len(vs) < 3:
            continue
        X = dlt([ud[c][k] for c in vs], [P[c] for c in vs])
        for c in vs:
            errs[c].append(np.linalg.norm(proj(X, cams[c])[0] - kp[c][k]))
    allv = np.concatenate([errs[c] for c in ids])
    return (np.median(allv), {c: float(np.median(errs[c])) for c in ids}) if per_cam else np.median(allv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--take", required=True); ap.add_argument("--frame", type=int, default=300)
    ap.add_argument("--size", default="1b"); ap.add_argument("--range", type=int, default=6)
    ap.add_argument("--cams", nargs="+", default=["cam0", "cam1", "cam2", "cam3", "cam4", "cam5"])
    ap.add_argument("--min-conf", type=float, default=0.5)
    ap.add_argument("--calib", default="output/calibration/20260925_sp_lg_7cam_scaled.json")
    a = ap.parse_args()
    from sapiens.pose.models import init_model
    from sapiens.pose.datasets import parse_pose_metainfo, UDPHeatmap
    cams = calibrate.load_calibration_output(a.calib)
    model = init_model(rp.cfg_path(a.size), f"{rp.CKPT_ROOT}/pose/sapiens2_{a.size}_pose.safetensors", "cuda:0").eval()
    model.pose_metainfo = parse_pose_metainfo(dict(from_file=f"{rp.SAP}/sapiens/pose/configs/_base_/keypoints308.py"))
    cc = dict(model.cfg.codec); cc.pop("type"); model.codec = UDPHeatmap(**cc)

    R = range(-a.range, a.range + 1)
    cache = {}
    for c in a.cams:
        for s in R:
            key = f"{a.frame + s:06d}.jpg"
            img = cv2.imread(f"{a.take}/aligned/{c}/{key}")
            bb = rp.mask_bbox(cv2.imread(f"{a.take}/aligned/masks_rvm/{c}/{key}", 0) > 128)
            x, smp = rp.prep(model, img, bb)
            kp, sc = rp.decode(model, rp.forward(model, x, torch.bfloat16), [smp])[0]
            cache[(c, s)] = (kp, sc)
        print("cached", c, flush=True)

    def ev(sh, pc=False):
        return objective({c: cache[(c, sh[c])][0] for c in a.cams}, {c: cache[(c, sh[c])][1] for c in a.cams}, cams, a.cams, a.min_conf, pc)

    sh = {c: 0 for c in a.cams}
    print("zero shifts:", ev(sh, True))
    for sweep in range(3):
        changed = False
        for c in a.cams[1:]:   # cam0 is the time reference
            best = min(R, key=lambda s: ev({**sh, c: s}))
            scores = {s: round(float(ev({**sh, c: s})), 2) for s in R}
            if best != sh[c]:
                changed = True
            sh[c] = best
            print(f"sweep {sweep} {c}: best {best:+d}  curve {scores}", flush=True)
        if not changed:
            break
    print("final shifts:", sh)
    print("final:", ev(sh, True))


if __name__ == "__main__":
    main()
