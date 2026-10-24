"""Re-express stored pose2d landmarks (undistorted pixel coords / (w, h)) in a new calibration's intrinsics.

Distortion acts on K-normalised coordinates, so changing K (even with the same dist coefficients) is NOT an affine map of the
undistorted pixels. Each stored point is therefore taken back to the raw distorted pixel (old K + old dist, projectPoints) and
undistorted again with the new K + dist (undistortPoints, P=K_new). Copies every other file of the pose2d dir unchanged.

  python -m pose2d.remap_intrinsics --src <take>/aligned/pose2d --dst <stand-in>/aligned/pose2d --old-calib A.json --new-calib B.json
"""
import argparse, json, os, shutil, sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import calibrate  # noqa: E402

LM_FILES = ["landmarks_left", "landmarks_right", "landmarks_body"]
EXT_KEYS = LM_FILES + ["landmarks_body_hand_left", "landmarks_body_hand_right"]


def remap_block(block, old, new, w, h):
    out = {}
    for cam, frames in block.items():
        if cam not in new:
            out[cam] = frames
            continue
        keys = [(fk, k) for fk, lm in frames.items() for k in lm]
        if not keys:
            out[cam] = frames
            continue
        uv = np.array([[frames[fk][k][0] * w, frames[fk][k][1] * h] for fk, k in keys], np.float64)
        Ko, do, Kn, dn = old[cam]["K"], old[cam]["dist"], new[cam]["K"], new[cam]["dist"]
        ray = np.stack([(uv[:, 0] - Ko[0, 2]) / Ko[0, 0], (uv[:, 1] - Ko[1, 2]) / Ko[1, 1], np.ones(len(uv))], 1)
        raw = cv2.projectPoints(ray, np.zeros(3), np.zeros(3), Ko, do)[0].reshape(-1, 1, 2)
        ud = cv2.undistortPoints(raw, Kn, dn, P=Kn).reshape(-1, 2)
        res = {fk: {} for fk in frames}
        for (fk, k), (x, y) in zip(keys, ud):
            res[fk][k] = [float(x / w), float(y / h), frames[fk][k][2]]
        out[cam] = res
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--old-calib", required=True)
    ap.add_argument("--new-calib", required=True)
    a = ap.parse_args()
    old, new = calibrate.load_calibration_output(a.old_calib), calibrate.load_calibration_output(a.new_calib)
    c0 = next(iter(new))
    w, h = new[c0]["width"], new[c0]["height"]
    os.makedirs(a.dst, exist_ok=True)
    for fn in os.listdir(a.src):
        s, d = os.path.join(a.src, fn), os.path.join(a.dst, fn)
        if fn == "smplx" or fn.startswith("reconstruction_") or fn.startswith("triangulation_diagnostics_") or fn == "grid_video.mp4":
            continue
        if os.path.isdir(s):
            continue
        if fn[:-5] in LM_FILES and fn.endswith(".json"):
            json.dump(remap_block(json.load(open(s)), old, new, w, h), open(d, "w"))
        elif fn == "extraction.json":
            e = json.load(open(s))
            for k in EXT_KEYS:
                if k in e:
                    e[k] = remap_block(e[k], old, new, w, h)
            json.dump(e, open(d, "w"))
        else:
            shutil.copy(s, d)
    print("wrote", a.dst)


if __name__ == "__main__":
    main()
