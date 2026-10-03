"""Convert cached Sapiens2 keypoints into pose2d's detector-agnostic schema.

Same role as pose2d/dwpose_extraction.py + run_pipeline_dwpose.py: writes
<take>/aligned/pose2d_sapiens/{landmarks,confidence,landmark_confidence}_*.json,
tracking_diagnostics.json and extraction.json, so `python -m pose2d.triangulation
<take> --pose2d-dir pose2d_sapiens` and pose2d.visualize_triangulation work unchanged.

Sapiens keypoints are in DISTORTED pixel space; pose2d's schema is undistorted pixel
coords / (w, h) (the projection matrices have no distortion), so they are undistorted
here with each camera's own K/dist before normalising.

  python -m sapiens2_test.to_pose2d --take /data/oak_recorder_sessions/20260925_153809_take3
"""
import argparse, json, os, sys
import cv2, numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import calibrate  # noqa: E402
from pose2d import tracking  # noqa: E402

# MediaPipe hand order: wrist, then thumb/index/middle/ring/pinky each base -> tip.
# Sapiens (Goliath) order per finger is tip -> base ("finger4" tip ... "third_joint" base).
HAND_RIGHT = [41, 24, 23, 22, 21, 28, 27, 26, 25, 32, 31, 30, 29, 36, 35, 34, 33, 40, 39, 38, 37]
HAND_LEFT = [62, 45, 44, 43, 42, 49, 48, 47, 46, 53, 52, 51, 50, 57, 56, 55, 54, 61, 60, 59, 58]
# COCO-17 upper body (ids 0-12), Sapiens has wrists at 62 (left) / 41 (right) and hips at 9 / 10.
BODY = [0, 1, 2, 3, 4, 5, 6, 7, 8, 62, 41, 9, 10]


def lm(xy, w, h):
    return {str(i): [float(x / w), float(y / h), 0.0] for i, (x, y) in enumerate(xy)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--take", required=True)
    ap.add_argument("--npz", default="output/sapiens2/allframes_1b.npz")
    ap.add_argument("--calib", default="output/calibration/20260925_sp_lg_7cam_scaled.json")
    ap.add_argument("--name", default="pose2d_sapiens")
    ap.add_argument("--min-conf", type=float, default=0.3, help="part counts as detected if mean keypoint score >= this")
    a = ap.parse_args()

    calib = calibrate.load_calibration_output(a.calib)
    d = np.load(a.npz)
    frames, cams, kp, sc = d["frames"], [str(c) for c in d["cams"]], d["kp"], d["sc"]
    out_dir = os.path.join(a.take, "aligned", a.name)
    os.makedirs(out_dir, exist_ok=True)

    names = ["landmarks_left", "confidence_left", "landmarks_right", "confidence_right", "landmarks_body",
             "confidence_body", "landmarks_body_hand_left", "landmarks_body_hand_right",
             "landmark_confidence_left", "landmark_confidence_right", "landmark_confidence_body"]
    ext = {"cam_ids": cams, **{n: {c: {} for c in cams} for n in names}}
    parts = {"left": HAND_LEFT, "right": HAND_RIGHT, "body": BODY}

    w = h = None
    for ci, c in enumerate(cams):
        K, dist = calib[c]["K"], calib[c]["dist"]
        w, h = calib[c]["width"], calib[c]["height"]
        raw = kp[:, ci].reshape(-1, 1, 2).astype(np.float64)
        ok_pt = np.isfinite(raw).all(axis=(1, 2))
        ud = np.full((len(raw), 2), np.nan)
        ud[ok_pt] = cv2.undistortPoints(raw[ok_pt], K, dist, P=K).reshape(-1, 2)
        ud = ud.reshape(len(frames), 308, 2)
        for fi, f in enumerate(frames):
            key = f"{int(f):06d}.jpg"
            if not np.isfinite(ud[fi]).all(axis=1).any():
                continue
            for part, idx in parts.items():
                xy, s = ud[fi, idx], sc[fi, ci, idx]
                if not np.isfinite(xy).all() or s.mean() < a.min_conf:
                    continue
                ext[f"landmarks_{part}"][c][key] = lm(xy, w, h)
                ext[f"confidence_{part}"][c][key] = float(s.mean())
                ext[f"landmark_confidence_{part}"][c][key] = {str(i): float(v) for i, v in enumerate(s)}
            if key in ext["landmarks_body"][c]:
                for side, j in (("left", 62), ("right", 41)):
                    ext[f"landmarks_body_hand_{side}"][c][key] = {"wrist": [float(ud[fi, j, 0] / w), float(ud[fi, j, 1] / h), 0.0]}
        print(c, {p: len(ext[f"landmarks_{p}"][c]) for p in parts}, flush=True)

    (ll, cl, lr, cr, lb, cb, diag, lcl, lcr, lcb) = tracking.run_tracking(a.take, ext, w, h)
    out = {"landmarks_left": ll, "confidence_left": cl, "landmarks_right": lr, "confidence_right": cr,
           "landmarks_body": lb, "confidence_body": cb, "tracking_diagnostics": diag,
           "landmark_confidence_left": lcl, "landmark_confidence_right": lcr, "landmark_confidence_body": lcb}
    for k, v in out.items():
        with open(os.path.join(out_dir, k + ".json"), "w") as f:
            json.dump(v, f)
    with open(os.path.join(out_dir, "extraction.json"), "w") as f:
        json.dump(ext, f)
    print("wrote", out_dir, "| diagnostics keys:", list(diag)[:8] if isinstance(diag, dict) else type(diag))


if __name__ == "__main__":
    main()
