"""Replace flagged 2D keypoints by linear interpolation between the nearest unflagged frames.

Alternative to dropping flagged camera observations in triangulation (pose2d.triangulation
--exclude-flags): the camera stays in the set, only the flagged 2D positions are repaired. Copies
<take>/aligned/<src> to <take>/aligned/<dst> and patches landmarks_body.json (pose2d schema,
undistorted pixel coords / (w, h)).

  python -m sapiens2.interp_flagged_kps --take T --flags output/sapiens2/sil_flags.json \
      --src pose2d_sapiens_smoothbox --dst pose2d_sapiens_smoothbox_interp
"""
import argparse, json, os, shutil


def interpolate_runs(frames, flagged, pts):
    """frames: sorted ints; flagged: set of ints; pts: {frame: [x, y, z]} (unflagged valid anchors)."""
    anchors = [f for f in frames if f not in flagged and pts.get(f) is not None]
    out = {}
    for f in frames:
        if f not in flagged:
            continue
        prev = max((a for a in anchors if a < f), default=None)
        nxt = min((a for a in anchors if a > f), default=None)
        if prev is None and nxt is None:
            continue
        if prev is None or nxt is None:
            a = prev if nxt is None else nxt
            out[f] = [pts[a][0], pts[a][1], pts[a][2]]
            continue
        w = (f - prev) / (nxt - prev)
        out[f] = [pts[prev][i] * (1 - w) + pts[nxt][i] * w for i in range(3)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--take", required=True)
    ap.add_argument("--flags", default="output/sapiens2/sil_flags.json")
    ap.add_argument("--src", default="pose2d_sapiens_smoothbox")
    ap.add_argument("--dst", default="pose2d_sapiens_smoothbox_interp")
    a = ap.parse_args()

    src = os.path.join(a.take, "aligned", a.src)
    dst = os.path.join(a.take, "aligned", a.dst)
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    with open(os.path.join(dst, "landmarks_body.json")) as f:
        lmb = json.load(f)
    with open(a.flags) as f:
        flags = json.load(f)

    n = 0
    for cam, per_lm in flags.items():
        by_frame = lmb[cam]
        frames = sorted(int(k[:6]) for k in by_frame)
        for lm, fl in per_lm.items():
            fl = set(int(x) for x in fl)
            pts = {f: by_frame[f"{f:06d}.jpg"].get(lm) for f in frames}
            for f, p in interpolate_runs(frames, fl, pts).items():
                if by_frame[f"{f:06d}.jpg"].get(lm) is None:
                    continue
                by_frame[f"{f:06d}.jpg"][lm] = p
                n += 1
    with open(os.path.join(dst, "landmarks_body.json"), "w") as f:
        json.dump(lmb, f)
    print(f"replaced {n} flagged keypoints by interpolation -> {dst}")


if __name__ == "__main__":
    main()
