"""Sapiens2 308-keypoint pose on our camera frames + speed benchmark.

Run from repo root:
  PYTHONPATH=/home/fmalmb/CODE/sapiens2 python -m sapiens2_test.run_pose \
      --take /data/oak_recorder_sessions/20260925_153809_take3 --frame 300 --sizes 0.4b 0.8b 1b

Frames and RVM masks are both in raw (distorted) pixel space; person bboxes come from the
RVM masks and keypoints are returned in the same distorted pixel space.
"""
import argparse, json, os, time
import cv2, numpy as np, torch

CKPT_ROOT = os.path.expanduser(os.environ.get("SAPIENS_CHECKPOINT_ROOT", "~/sapiens2_host"))
SAP = os.environ.get("SAPIENS2_REPO", "/home/fmalmb/CODE/sapiens2")


def cfg_path(size):
    return (f"{SAP}/sapiens/pose/configs/keypoints308/shutterstock_goliath_3po/"
            f"sapiens2_{size}_keypoints308_shutterstock_goliath_3po-1024x768.py")


def mask_bbox(mask, pad=0.05):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    w, h = x1 - x0, y1 - y0
    return np.array([x0 - pad * w, y0 - pad * h, x1 + pad * w, y1 + pad * h], np.float32)


def sync():
    torch.cuda.synchronize()


def prep(model, img, bbox):
    d = model.pipeline(dict(img=img, bbox=bbox[None], bbox_score=np.ones(1, np.float32)))
    d = model.data_preprocessor(d)
    return d["inputs"], d["data_samples"]


@torch.no_grad()
def forward(model, inputs, dtype):
    with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
        pred = model(inputs)
        if model.cfg.val_cfg is not None and model.cfg.val_cfg.get("flip_test", False):
            pf = model(inputs.flip(-1)).flip(-1)[:, model.pose_metainfo["flip_indices"]]
            pred = (pred + pf) / 2.0
    return pred


def decode(model, pred, samples):
    out = []
    for i, ds in enumerate(samples):
        kp, sc = model.codec.decode(pred[i].float().cpu().numpy())
        m = ds["meta"]
        kp = kp / m["input_size"] * m["bbox_scale"] + m["bbox_center"] - 0.5 * m["bbox_scale"]
        out.append((kp[0], sc[0]))
    return out


def draw(img, kp, sc, thr=0.3):
    img = img.copy()
    for (x, y), s in zip(kp, sc):
        if s > thr:
            cv2.circle(img, (int(x), int(y)), 4, (0, 255, 0), -1)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--take", required=True)
    ap.add_argument("--frame", type=int, default=300)
    ap.add_argument("--cams", nargs="+", default=["cam0", "cam1", "cam2", "cam3", "cam4", "cam5"])
    ap.add_argument("--sizes", nargs="+", default=["0.4b", "0.8b", "1b"])
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--no-flip", action="store_true", help="disable flip-test TTA (halves forward cost)")
    ap.add_argument("--offsets", default="", help="per-camera frame shifts, e.g. cam4=-2,cam5=-4")
    ap.add_argument("--out", default="output/sapiens2")
    a = ap.parse_args()

    from sapiens.pose.models import init_model
    from sapiens.pose.datasets import parse_pose_metainfo, UDPHeatmap
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[a.dtype]
    key = f"{a.frame:06d}.jpg"
    off = {kv.split("=")[0]: int(kv.split("=")[1]) for kv in a.offsets.split(",") if kv}
    imgs, bboxes = {}, {}
    for c in a.cams:
        ck_ = f"{a.frame + off.get(c, 0):06d}.jpg"
        imgs[c] = cv2.imread(f"{a.take}/aligned/{c}/{ck_}")
        bboxes[c] = mask_bbox(cv2.imread(f"{a.take}/aligned/masks_rvm/{c}/{ck_}", 0) > 128)
        print(c, ck_, imgs[c].shape[:2], "bbox", bboxes[c])
    cams = [c for c in a.cams if bboxes[c] is not None]
    os.makedirs(a.out, exist_ok=True)

    results = {}
    for size in a.sizes:
        ck = f"{CKPT_ROOT}/pose/sapiens2_{size}_pose.safetensors"
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        model = init_model(cfg_path(size), ck, "cuda:0").eval()
        sync(); load_s = time.time() - t0
        model.pose_metainfo = parse_pose_metainfo(dict(from_file=f"{SAP}/sapiens/pose/configs/_base_/keypoints308.py"))
        codec_cfg = dict(model.cfg.codec); assert codec_cfg.pop("type") == "UDPHeatmap"
        model.codec = UDPHeatmap(**codec_cfg)
        if a.no_flip:
            model.cfg.val_cfg["flip_test"] = False
        n_par = sum(p.numel() for p in model.parameters()) / 1e9
        print(f"\n=== {size}: {n_par:.2f}B params, load {load_s:.1f}s, flip_test="
              f"{bool(model.cfg.val_cfg and model.cfg.val_cfg.get('flip_test', False))}, {a.dtype} ===")

        # CPU preprocess (4K crop+affine), single camera
        pp = []
        for _ in range(a.warmup + a.iters):
            t = time.perf_counter(); x, s = prep(model, imgs[cams[0]], bboxes[cams[0]]); pp.append(time.perf_counter() - t)
        pp_ms = np.mean(pp[a.warmup:]) * 1e3

        batch = [prep(model, imgs[c], bboxes[c]) for c in cams]
        x1 = batch[0][0]
        xb = torch.cat([b[0] for b in batch], 0)
        samples = [b[1] for b in batch]

        def bench(x):
            for _ in range(a.warmup):
                forward(model, x, dtype)
            sync(); ts = []
            for _ in range(a.iters):
                t = time.perf_counter(); forward(model, x, dtype); sync(); ts.append(time.perf_counter() - t)
            return np.array(ts) * 1e3

        t1 = bench(x1)
        tb = bench(xb)
        pred = forward(model, xb, dtype)
        decode(model, pred, samples)
        t = time.perf_counter()
        for _ in range(5):
            res = decode(model, pred, samples)
        dec_ms = (time.perf_counter() - t) * 1e3 / 5 / len(cams)
        peak = torch.cuda.max_memory_allocated() / 2**30

        r = dict(params_B=n_par, load_s=load_s, preprocess_ms=pp_ms, fwd_b1_ms=float(t1.mean()), fwd_b1_std=float(t1.std()),
                 batch=len(cams), fwd_batch_ms=float(tb.mean()), fwd_per_img_ms=float(tb.mean() / len(cams)),
                 decode_ms_per_img=dec_ms, peak_vram_GiB=peak, dtype=a.dtype, flip=not a.no_flip)
        results[size] = r
        print(json.dumps({k: round(v, 2) if isinstance(v, float) else v for k, v in r.items()}))
        for c, (kp, sc) in zip(cams, res):
            cv2.imwrite(f"{a.out}/{size}_{c}.jpg", cv2.resize(draw(imgs[c], kp, sc), None, fx=0.4, fy=0.4))
            np.savez(f"{a.out}/{size}_{c}_f{a.frame}.npz", kp=kp, score=sc)
        del model, x1, xb, pred
        torch.cuda.empty_cache()

    json.dump(results, open(f"{a.out}/bench_{a.dtype}{'_noflip' if a.no_flip else ''}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
