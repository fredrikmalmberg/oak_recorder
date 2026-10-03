"""Turn the windowed silhouette score into a flags JSON for `pose2d.triangulation --exclude-flags`.

A frame t with score z > --thr flags the steps t-1..t+2 that the score window covers: the keypoint
observations at frames t-1 .. t+2 (--before/--after) are marked for that camera and landmark.
Sapiens ids -> pose2d body ids: 5->5, 6->6 (shoulders), 9->11, 10->12 (hips).

  python -m sapiens2_test.make_sil_flags --check output/sapiens2/kp_sil_check_v2.npz --out output/sapiens2/sil_flags.json
Output: {"cam1": {"11": [516, 517, ...], ...}, ...}
"""
import argparse, json
import numpy as np
from sapiens2_test.kp_silhouette_check import windowed_z

NAME_TO_POSE2D = {"Lsh": 5, "Rsh": 6, "Lhip": 11, "Rhip": 12}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", default="output/sapiens2/kp_sil_check_v2.npz")
    ap.add_argument("--out", default="output/sapiens2/sil_flags.json")
    ap.add_argument("--thr", type=float, default=15.0)
    ap.add_argument("--before", type=int, default=1)
    ap.add_argument("--after", type=int, default=2)
    ap.add_argument("--min-frame", type=int, default=None)
    ap.add_argument("--max-frame", type=int, default=None)
    a = ap.parse_args()
    d = np.load(a.check); frames = d["frames"]
    lo = a.min_frame if a.min_frame is not None else int(frames.min())
    hi = a.max_frame if a.max_frame is not None else int(frames.max())
    out, total = {}, 0
    for k in d.files:
        if "/" not in k:
            continue
        cam, name = k.split("/")
        z = windowed_z(d[k])
        hit = np.where(np.isfinite(z) & (z > a.thr))[0]
        fl = set()
        for i in hit:
            for j in range(i - a.before, i + a.after + 1):
                if 0 <= j < len(frames) and lo <= frames[j] <= hi:
                    fl.add(int(frames[j]))
        if fl:
            out.setdefault(cam, {})[str(NAME_TO_POSE2D[name])] = sorted(fl); total += len(fl)
    json.dump(out, open(a.out, "w"))
    print(f"{total} flagged camera observations -> {a.out}")
    for cam in sorted(out):
        print(" ", cam, {lm: len(v) for lm, v in sorted(out[cam].items())})


if __name__ == "__main__":
    main()
