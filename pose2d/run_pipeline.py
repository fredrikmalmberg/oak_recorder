"""CLI: extraction -> tracking/flagging -> grid-video render for one take.

Usage:
    python -m pose2d.run_pipeline <take_dir> [--calib PATH] [--force]

Everything is written under <take_dir>/aligned/pose2d/.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import calibrate  # noqa: E402

from hand_pose import hand_multiview as hmv  # noqa: E402
from pose2d import extraction  # noqa: E402
from pose2d import tracking  # noqa: E402
from pose2d import grid_video  # noqa: E402

DEFAULT_CALIB_PATH = os.path.join("output", "calibration", "20260903_153052_7cam.json")


def run_take(take_dir, calib_path, force):
    calib = calibrate.load_calibration_output(calib_path)
    ext = extraction.extract_pose2d_for_take(take_dir, calib, force=force)
    cam_ids = ext["cam_ids"]

    sample_frame = hmv.discover_frames(take_dir, cam_ids[0])[0]
    width, height = hmv.get_image_size(take_dir, cam_ids[0], sample_frame)

    (landmarks_left, confidence_left, landmarks_right, confidence_right,
     landmarks_body, confidence_body, diagnostics) = tracking.run_tracking(take_dir, ext, width, height)

    out_dir = os.path.join(take_dir, "aligned", "pose2d")
    with open(os.path.join(out_dir, "landmarks_left.json"), "w") as f:
        json.dump(landmarks_left, f)
    with open(os.path.join(out_dir, "landmarks_right.json"), "w") as f:
        json.dump(landmarks_right, f)
    with open(os.path.join(out_dir, "landmarks_body.json"), "w") as f:
        json.dump(landmarks_body, f)
    with open(os.path.join(out_dir, "confidence_left.json"), "w") as f:
        json.dump(confidence_left, f)
    with open(os.path.join(out_dir, "confidence_right.json"), "w") as f:
        json.dump(confidence_right, f)
    with open(os.path.join(out_dir, "tracking_diagnostics.json"), "w") as f:
        json.dump(diagnostics, f)

    total = diagnostics["n_frames"]
    n_cams = len(cam_ids)
    n_left = sum(len(by_frame) for by_frame in landmarks_left.values())
    n_right = sum(len(by_frame) for by_frame in landmarks_right.values())
    n_body = sum(len(by_frame) for by_frame in landmarks_body.values())
    denom = n_cams * total
    print(f"\n=== {take_dir} ===")
    print(f"  cameras: {n_cams}, frames per camera: {total}")
    if denom:
        print(f"  left-hand detections: {n_left}/{denom} ({n_left / denom:.1%})")
        print(f"  right-hand detections: {n_right}/{denom} ({n_right / denom:.1%})")
        print(f"  body detections: {n_body}/{denom} ({n_body / denom:.1%})")
    print(f"  swaps detected/corrected: {diagnostics['n_swaps_detected']}/{diagnostics['n_swaps_corrected']}")
    print(
        f"  velocity flags L/R: {diagnostics['n_velocity_flagged_left']}/"
        f"{diagnostics['n_velocity_flagged_right']}"
    )
    print(f"  occlusion flags: {diagnostics['n_occlusion_flagged']}")
    print(f"  single-hand occlusion risk flags: {diagnostics['n_single_hand_occlusion_flagged']}")
    print(f"  body pose untrusted (BODYRISK) flags: {diagnostics['n_body_untrusted_flagged']}")

    grid_path = os.path.join(out_dir, "grid_video.mp4")
    grid_video.render_pose2d_grid_video(
        take_dir, calib, cam_ids, landmarks_left, confidence_left, landmarks_right, confidence_right,
        landmarks_body, diagnostics, grid_path,
    )
    print(f"  wrote: {grid_path}")
    return grid_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("take_dir")
    parser.add_argument("--calib", default=DEFAULT_CALIB_PATH)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run_take(args.take_dir, args.calib, args.force)


if __name__ == "__main__":
    main()
