"""Run WiLoR on the renders from synth_render.py with the same box / handedness logic as hand_pose.wilor_extract
(hand box = extent of the Sapiens hand keypoints, rescale 2.0). Needs synth_sapiens.py output first. Run in the wilor env:

  PYTHONPATH=. ~/miniconda3/envs/wilor/bin/python -m smplx_fit.synth_wilor --dir <synth dir> [--variant textured]
Writes <dir>/wilor_<variant>.npz: valid[P,V,2], kp2d[P,V,2,21,2] (calibrated-K projection, pixels), joints_cam[P,V,2,21,3],
verts_cam[P,V,2,778,3]; side 0 = left, 1 = right.
"""
import argparse

import cv2
import numpy as np
import torch

from hand_pose import wilor_extract as we


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--variant", default="textured")
    ap.add_argument("--rescale-factor", type=float, default=2.0)
    ap.add_argument("--min-score", type=float, default=0.3)
    a = ap.parse_args()
    gt = np.load(f"{a.dir}/gt.npz")
    sap = np.load(f"{a.dir}/sapiens_{a.variant}.npz")
    P, V = len(gt["pose_names"]), len(gt["view_names"])
    K = gt["K"]
    dev = torch.device("cuda")
    model, cfg = we.load_model(dev)
    valid = np.zeros((P, V, 2), bool)
    kp2d = np.full((P, V, 2, 21, 2), np.nan, np.float32)
    jc = np.full((P, V, 2, 21, 3), np.nan, np.float32)
    vc = np.full((P, V, 2, 778, 3), np.nan, np.float32)
    for p in range(P):
        for v in range(V):
            xy_all, sc_all = sap["kp"][p, v], sap["sc"][p, v]
            boxes, rights, sides = [], [], []
            for s_, ids in enumerate(we.SIDES):
                xy, s = xy_all[ids], sc_all[ids]
                if s.mean() < a.min_score:
                    continue
                boxes.append([*xy.min(0), *xy.max(0)]); rights.append(float(s_ == 1)); sides.append(s_)
            if not boxes:
                continue
            img = cv2.imread(f"{a.dir}/{a.variant}/p{p:02d}_{gt['view_names'][v]}.jpg")
            o = we.run_wilor(model, cfg, dev, img, np.array(boxes, np.float32), np.array(rights, np.float32), K, a.rescale_factor)
            for n, s_ in enumerate(sides):
                j = o["joints_local"][n] + o["cam_t"][n]
                valid[p, v, s_] = True
                jc[p, v, s_] = j
                vc[p, v, s_] = o["verts_local"][n] + o["cam_t"][n]
                kp2d[p, v, s_] = we.project(K, j)
        print("pose", p, flush=True)
    np.savez(f"{a.dir}/wilor_{a.variant}.npz", valid=valid, kp2d=kp2d, joints_cam=jc, verts_cam=vc)
    print("saved; valid per side:", valid.sum((0, 1)), "of", P * V)


if __name__ == "__main__":
    main()
