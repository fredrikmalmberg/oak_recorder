"""Reprojection error of pose2d triangulation expressed in mm as well as px.

Mirrors hand_multiview.ransac_per_camera_reproj_error (what pose2d.triangulation
reports): the REFERENCE landmark (hand wrist / body left shoulder) of the raw
reconstruction is reprojected into every camera with a confident observation.
Each pixel error is converted to a lateral distance at the point's depth in that
camera: mm = px * Z_cam / fx * 1000.

The calibration scale is "arbitrary"; mm assumes 1 calibration unit = 1 m, which is
only a sanity-checked assumption (triangulated shoulder width median 0.36 units).

  python -m sapiens2_test.reproj_mm --take TAKE --pose2d-dir pose2d_sapiens \
      --calib output/calibration/....json --recon-dir <dir with reconstruction_*.json>
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import calibrate  # noqa: E402
from hand_pose import hand_multiview as hmv  # noqa: E402
from pose2d import triangulation as tri  # noqa: E402

REF = {"left": tri.REFERENCE_LANDMARK_ID_HAND, "right": tri.REFERENCE_LANDMARK_ID_HAND,
       "body": tri.REFERENCE_LANDMARK_ID_BODY}


def run(take, pose2d_dir, calib_path, recon_dir, min_conf=0.5):
    calib = calibrate.load_calibration_output(calib_path)
    data = tri.load_pose2d_take_data(take, pose2d_dir=pose2d_dir)
    cams = [c for c in data["cam_ids"] if c in calib]
    P = tri.build_projection_matrices(calib, cams)
    w, h = hmv.get_image_size(take, cams[0], next(iter(data["landmarks_body"][cams[0]])))
    out = {}
    for part in ("left", "right", "body"):
        recon = json.load(open(os.path.join(recon_dir, f"reconstruction_{part}.json")))["raw"]
        lms, conf = data[f"landmarks_{part}"], data[f"confidence_{part}"]
        px, mm = [], []
        for frame, pts in recon.items():
            X = pts.get(str(REF[part]))
            if X is None:
                continue
            X = np.asarray(X, float)
            for c in cams:
                if conf.get(c, {}).get(frame, 0) < min_conf:
                    continue
                obs = lms.get(c, {}).get(frame, {}).get(str(REF[part]))
                if obs is None:
                    continue
                e = hmv._reproj_error_px(X, P[c], (obs[0] * w, obs[1] * h))
                z = (calib[c]["R"] @ X + np.asarray(calib[c]["t"]).reshape(3))[2]
                px.append(e)
                mm.append(e * z / calib[c]["K"][0][0] * 1000.0)
        px, mm = np.array(px), np.array(mm)
        out[part] = dict(n=len(px), px=np.percentile(px, [50, 95]).tolist() + [px.mean()],
                         mm=np.percentile(mm, [50, 95]).tolist() + [mm.mean()])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--take", required=True)
    ap.add_argument("--pose2d-dir", default="pose2d_sapiens")
    ap.add_argument("--calib", required=True)
    ap.add_argument("--recon-dir", required=True)
    a = ap.parse_args()
    for part, r in run(a.take, a.pose2d_dir, a.calib, a.recon_dir).items():
        print(f"{part:6s} n={r['n']:5d}  px median/p95/mean = {r['px'][0]:.2f}/{r['px'][1]:.1f}/{r['px'][2]:.1f}"
              f"   mm median/p95/mean = {r['mm'][0]:.1f}/{r['mm'][1]:.0f}/{r['mm'][2]:.0f}")


if __name__ == "__main__":
    main()
