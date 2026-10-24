"""Convert the per-camera WiLoR hand keypoints into pose2d's detector-agnostic schema, so they go through the same
tracking + pose2d.triangulation chain as the Sapiens body/hands.

The two hands come from <take>/aligned/wilor/wilor_hands.npz (kp2d_calib: the 21 MANO joints projected with the
calibrated camera, in raw/distorted pixels -> undistorted here, then normalised by width/height). Joint order is
already MediaPipe/OpenPose order. The whole-hand confidence is the mean Sapiens hand score the crop was made from
(wilor_hands.npz 'score'); WiLoR has no per-joint confidence, so none is written (uniform weighting in the DLT).
The body, and the body-wrist reference the tracker uses for swap/trust checks, are taken unchanged from an existing
Sapiens pose2d dir (its extraction.json).

  python -m hand_pose.wilor_to_pose2d --take <take> --calib <calib.json> [--sapiens-dir pose2d_sapiens_smoothbox]
Writes <take>/aligned/pose2d_wilor/. Then:
  python -m pose2d.triangulation <take> --calib <calib.json> --pose2d-dir pose2d_wilor --camera-mode stable --out-dir <out> --force
"""
import argparse, json, os, sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import calibrate  # noqa: E402
from pose2d import tracking  # noqa: E402


def lm(xy, w, h):
    return {str(i): [float(x / w), float(y / h), 0.0] for i, (x, y) in enumerate(xy)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--take", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--sapiens-dir", default="pose2d_sapiens_smoothbox")
    ap.add_argument("--name", default="pose2d_wilor")
    a = ap.parse_args()

    calib = calibrate.load_calibration_output(a.calib)
    h5 = np.load(f"{a.take}/aligned/wilor/wilor_hands.npz")
    frames, cams, valid, score, kp = h5["frames"], [str(c) for c in h5["cams"]], h5["valid"], h5["score"], h5["kp2d_calib"]
    with open(f"{a.take}/aligned/{a.sapiens_dir}/extraction.json") as f:
        ext = json.load(f)
    assert ext["cam_ids"] == cams, (ext["cam_ids"], cams)
    for part in ("left", "right"):
        ext[f"landmarks_{part}"] = {c: {} for c in cams}
        ext[f"confidence_{part}"] = {c: {} for c in cams}
        ext.pop(f"landmark_confidence_{part}", None)

    w = h = None
    for ci, c in enumerate(cams):
        K, dist = calib[c]["K"], calib[c]["dist"]
        w, h = calib[c]["width"], calib[c]["height"]
        for s, part in enumerate(("left", "right")):
            for t, f in enumerate(frames):
                if not valid[t, ci, s]:
                    continue
                ud = cv2.undistortPoints(kp[t, ci, s].reshape(-1, 1, 2).astype(np.float64), K, dist, P=K).reshape(-1, 2)
                key = f"{int(f):06d}.jpg"
                ext[f"landmarks_{part}"][c][key] = lm(ud, w, h)
                ext[f"confidence_{part}"][c][key] = float(score[t, ci, s])
        print(c, {p: len(ext[f"landmarks_{p}"][c]) for p in ("left", "right")}, flush=True)

    (ll, cl, lr, cr, lb, cb, diag, lcl, lcr, lcb) = tracking.run_tracking(a.take, ext, w, h)
    out = {"landmarks_left": ll, "confidence_left": cl, "landmarks_right": lr, "confidence_right": cr,
           "landmarks_body": lb, "confidence_body": cb, "tracking_diagnostics": diag,
           "landmark_confidence_left": lcl, "landmark_confidence_right": lcr, "landmark_confidence_body": lcb}
    out_dir = os.path.join(a.take, "aligned", a.name)
    os.makedirs(out_dir, exist_ok=True)
    for k, v in out.items():
        with open(os.path.join(out_dir, k + ".json"), "w") as f:
            json.dump(v, f)
    with open(os.path.join(out_dir, "extraction.json"), "w") as f:
        json.dump(ext, f)
    print("wrote", out_dir, "| landmark_confidence_left/right:", type(lcl).__name__, type(lcr).__name__)


if __name__ == "__main__":
    main()
