"""Run Sapiens2 (308 keypoints) on the renders from synth_render.py; person box = render mask extent + 5% (as run_pose).

  PYTHONPATH=/home/fmalmb/CODE/sapiens2:. python -m smplx_fit.synth_sapiens --dir <synth dir> [--variant textured]
Writes <dir>/sapiens_<variant>.npz: kp [P,V,308,2] (pixels), sc [P,V,308], boxes [P,V,4].
"""
import argparse

import cv2
import numpy as np
import torch

from sapiens2 import run_pose as rp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--variant", default="textured")
    ap.add_argument("--size", default="1b")
    a = ap.parse_args()
    from sapiens.pose.models import init_model
    from sapiens.pose.datasets import parse_pose_metainfo, UDPHeatmap

    gt = np.load(f"{a.dir}/gt.npz")
    P, V = len(gt["pose_names"]), len(gt["view_names"])
    model = init_model(rp.cfg_path(a.size), f"{rp.CKPT_ROOT}/pose/sapiens2_{a.size}_pose.safetensors", "cuda:0").eval()
    model.pose_metainfo = parse_pose_metainfo(dict(from_file=f"{rp.SAP}/sapiens/pose/configs/_base_/keypoints308.py"))
    cc = dict(model.cfg.codec); cc.pop("type"); model.codec = UDPHeatmap(**cc)
    kp = np.full((P, V, 308, 2), np.nan, np.float32); sc = np.zeros((P, V, 308), np.float32)
    boxes = np.full((P, V, 4), np.nan, np.float32)
    for p in range(P):
        for v in range(V):
            nm = f"p{p:02d}_{gt['view_names'][v]}"
            img = cv2.imread(f"{a.dir}/{a.variant}/{nm}.jpg")
            m = cv2.imread(f"{a.dir}/{a.variant}/masks/{nm}.png", 0)
            bb = rp.mask_bbox(m > 128)
            x, smp = rp.prep(model, img, bb)
            kp[p, v], sc[p, v] = rp.decode(model, rp.forward(model, x, torch.bfloat16), [smp])[0]
            boxes[p, v] = bb
        print("pose", p, flush=True)
    np.savez(f"{a.dir}/sapiens_{a.variant}.npz", kp=kp, sc=sc, boxes=boxes)
    print("saved", f"{a.dir}/sapiens_{a.variant}.npz")


if __name__ == "__main__":
    main()
