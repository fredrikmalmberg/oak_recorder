"""Build a body-only reconstruction whose wrists (body ids 9, 10) come from the triangulated WiLoR wrist.

Starts from a reconstruction dir (e.g. the bone-refined output/bonesA), replaces ids 9/10 in "raw" with the WiLoR wrist
(hand_pose.wilor_wrist output) and in "smoothed" with its Kalman/RTS-smoothed version (hand_multiview.smooth_reconstruction_sequence,
the pipeline's own smoother). Frames without a WiLoR wrist get no wrist (no fallback to the Sapiens wrist). The hand
reconstructions are emptied so a downstream SMPL-X fit cannot use them. Also creates a stand-in take dir with symlinks to the
camera frames and masks so smplx_fit.fit_take can run on it.

  python -m pose2d.swap_wrist --take T --init-dir output/bonesA/aligned/pose2d_sapiens_smoothbox --out-dir output/bodyW
"""
import argparse, json, os, shutil, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hand_pose import hand_multiview as hmv  # noqa: E402

WRIST_IDS = ("9", "10")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--take", required=True)
    ap.add_argument("--init-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pose2d-dir", default="pose2d_sapiens_smoothbox")
    ap.add_argument("--wrist-npz", default=None, help="default <take>/aligned/wilor/wilor_wrist_3d.npz")
    a = ap.parse_args()

    w = np.load(a.wrist_npz or f"{a.take}/aligned/wilor/wilor_wrist_3d.npz")
    frames = [f"{int(f):06d}.jpg" for f in w["frames"]]
    with open(os.path.join(a.init_dir, "reconstruction_body.json")) as f:
        rec = json.load(f)

    out = os.path.join(a.out_dir, "aligned", a.pose2d_dir)
    os.makedirs(out, exist_ok=True)
    for f in os.listdir(a.init_dir):
        if not f.startswith("reconstruction_"):
            shutil.copy(os.path.join(a.init_dir, f), out)

    raw, smoothed = rec["raw"], rec["smoothed"]
    for d in (raw, smoothed):
        for fk in d:
            for j in WRIST_IDS:
                d[fk].pop(j, None)
    for s, j in enumerate(WRIST_IDS):
        single = {fk: {j: w["wrist"][t, s].tolist()} for t, fk in enumerate(frames) if np.isfinite(w["wrist"][t, s, 0])}
        n_in = {fk: int(w["n_cams"][t, s]) for t, fk in enumerate(frames)}
        sm = hmv.smooth_reconstruction_sequence(single, frames, [j], n_in)
        for fk, v in single.items():
            raw.setdefault(fk, {}).update(v)
        for fk, v in sm.items():
            smoothed.setdefault(fk, {}).update(v)
        print(f"wrist {j}: {len(single)} raw frames, {len(sm)} smoothed frames")
    with open(os.path.join(out, "reconstruction_body.json"), "w") as f:
        json.dump({"raw": raw, "smoothed": smoothed}, f)
    for part in ("left", "right"):
        with open(os.path.join(out, f"reconstruction_{part}.json"), "w") as f:
            json.dump({"raw": {}, "smoothed": {}}, f)

    src = os.path.join(a.take, "aligned")
    for name in sorted(os.listdir(src)):
        if name.startswith("cam") or name.startswith("masks"):
            dst = os.path.join(a.out_dir, "aligned", name)
            if not os.path.lexists(dst):
                os.symlink(os.path.join(src, name), dst)
    print("wrote", out)


if __name__ == "__main__":
    main()
