"""Run Sapiens2 pose on many frames x cameras and cache keypoints (distorted pixel space).

  PYTHONPATH=/home/fmalmb/CODE/sapiens2:. python -m sapiens2_test.extract_multiframe --take ... --n 48
Output npz: frames[F], cams[C], kp[F,C,308,2] (NaN if no person mask), sc[F,C,308]
"""
import argparse, time
import cv2, numpy as np, torch
from sapiens2_test import run_pose as rp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--take", required=True)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--start", type=int, default=30)
    ap.add_argument("--stop", type=int, default=740)
    ap.add_argument("--size", default="1b")
    ap.add_argument("--cams", nargs="+", default=["cam0", "cam1", "cam2", "cam3", "cam4", "cam5"])
    ap.add_argument("--offsets", default="", help="per-camera frame shifts, e.g. cam4=-1")
    ap.add_argument("--boxes", default="", help="smooth_bbox npz: use its boxes (only where the RVM mask exists) instead of per-frame mask extents")
    ap.add_argument("--out", default="output/sapiens2/multiframe_1b.npz")
    a = ap.parse_args()
    from sapiens.pose.models import init_model
    from sapiens.pose.datasets import parse_pose_metainfo, UDPHeatmap
    off = {kv.split("=")[0]: int(kv.split("=")[1]) for kv in a.offsets.split(",") if kv}
    model = init_model(rp.cfg_path(a.size), f"{rp.CKPT_ROOT}/pose/sapiens2_{a.size}_pose.safetensors", "cuda:0").eval()
    model.pose_metainfo = parse_pose_metainfo(dict(from_file=f"{rp.SAP}/sapiens/pose/configs/_base_/keypoints308.py"))
    cc = dict(model.cfg.codec); cc.pop("type"); model.codec = UDPHeatmap(**cc)

    bx = np.load(a.boxes) if a.boxes else None
    if bx is not None:
        assert [str(c) for c in bx["cams"]] == list(a.cams), "box cams must match --cams"
    frames = np.linspace(a.start, a.stop, a.n).round().astype(int)
    F, C = len(frames), len(a.cams)
    kp = np.full((F, C, 308, 2), np.nan, np.float32); sc = np.zeros((F, C, 308), np.float32)
    t0 = time.time()
    for fi, f in enumerate(frames):
        for ci, c in enumerate(a.cams):
            key = f"{f + off.get(c, 0):06d}.jpg"
            img = cv2.imread(f"{a.take}/aligned/{c}/{key}")
            m = cv2.imread(f"{a.take}/aligned/masks_rvm/{c}/{key}", 0)
            bb = rp.mask_bbox(m > 128) if m is not None else None
            if bx is not None and bb is not None:
                bb = bx["boxes"][f, ci].astype(np.float32)
            if img is None or bb is None:
                continue
            x, smp = rp.prep(model, img, bb)
            k, s = rp.decode(model, rp.forward(model, x, torch.bfloat16), [smp])[0]
            kp[fi, ci], sc[fi, ci] = k, s
        if fi % 6 == 0:
            print(f"frame {fi+1}/{F} ({time.time()-t0:.0f}s)", flush=True)
    np.savez(a.out, frames=frames, cams=np.array(a.cams), kp=kp, sc=sc)
    print("saved", a.out, kp.shape)


if __name__ == "__main__":
    main()
