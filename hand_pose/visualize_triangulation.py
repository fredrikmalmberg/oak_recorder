"""Visualization for hand_pose.session_triangulation's outputs. Built on
already-existing viser/grid-video primitives (hand_multiview.
start_viser_server/add_camera_frustums/run_skeleton_player; grid_video.py's
grid-canvas pattern), not new machinery.

Two modes:

  3D player (default) -- loads <take_dir>/aligned/reconstruction[_multioffset]
  .json and plays it back in viser via hand_multiview.run_skeleton_player,
  with camera frustums from the calibration file.

  Usage grid video (--grid-video) -- composites a per-camera grid video of
  the aligned frames with each camera's raw MediaPipe hand skeleton
  overlaid, drawn on the UNDISTORTED frame (matching how
  extract_hand_landmarks_for_session_multicam actually detected those
  landmarks -- overlaying them on the raw distorted JPEGs on disk would not
  line up, especially given this rig's non-trivial lens distortion). Each
  camera's tile is shown in full color for a frame where session_
  triangulation's RANSAC selection actually used that camera
  (reconstruction_diagnostics.json's per_frame_selection), and desaturated
  to grayscale otherwise -- so which detections the reconstruction is
  actually built from is visible at a glance, camera by camera and frame by
  frame.

Usage:
    python -m hand_pose.visualize_triangulation <take_dir> [--variant baseline|multioffset] [--raw]
    python -m hand_pose.visualize_triangulation <take_dir> --grid-video [--output PATH] [--fps N]
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import calibrate  # noqa: E402

from hand_pose import hand_multiview as hmv  # noqa: E402
from hand_pose import grid_video  # noqa: E402
from hand_pose import session_triangulation  # noqa: E402


USED_COLOR = (0, 220, 255)  # bright cyan/yellow -- reads clearly over both color and grayscale tiles


def _suffix(variant):
    return "_multioffset" if variant == "multioffset" else ""


def load_take_outputs(take_dir, variant):
    suffix = _suffix(variant)
    aligned_dir = os.path.join(take_dir, "aligned")
    with open(os.path.join(aligned_dir, f"reconstruction{suffix}.json")) as f:
        recon = json.load(f)
    with open(os.path.join(aligned_dir, f"reconstruction_diagnostics{suffix}.json")) as f:
        diagnostics = json.load(f)
    with open(os.path.join(aligned_dir, "landmarks.json")) as f:
        landmarks = json.load(f)
    with open(os.path.join(aligned_dir, "confidence.json")) as f:
        confidence = json.load(f)
    return recon, diagnostics, landmarks, confidence


def launch_3d_viewer(take_dir, variant, calib, port, fps, use_smoothed):
    recon, diagnostics, _, _ = load_take_outputs(take_dir, variant)
    reconstruction = recon["smoothed"] if use_smoothed else recon["raw"]
    cam_ids = diagnostics["metrics"]["cam_ids"]

    server = hmv.start_viser_server(port=port)
    poses = {cam_id: (calib[cam_id]["R"], calib[cam_id]["t"]) for cam_id in cam_ids if cam_id in calib}
    if poses:
        sample = calib[cam_ids[0]]
        hmv.add_camera_frustums(server, poses, sample["K"], sample["width"], sample["height"])

    print(
        f"Viser server running -- open the printed URL in a browser to view. "
        f"{len(reconstruction)} frames ({'smoothed' if use_smoothed else 'raw'}, {variant})."
    )
    hmv.run_skeleton_player(server, reconstruction, fps=fps)


def render_usage_grid_video(
    take_dir, calib, cam_ids, diagnostics, landmarks, confidence,
    output_path, fps=15.0, cols=3, canvas_size=(1920, 1080),
):
    selection = diagnostics["per_frame_selection"]
    frame_keys = hmv.discover_frames(take_dir, cam_ids[0])

    undistort_maps = {}
    for cam_id in cam_ids:
        c = calib[cam_id]
        w, h = hmv.get_image_size(take_dir, cam_id, frame_keys[0])
        undistort_maps[cam_id] = (w, h, hmv.build_undistort_maps(c["K"], c["dist"], w, h))

    canvas_w, canvas_h = canvas_size
    rows = int(np.ceil(len(cam_ids) / cols))
    slot_w, slot_h = canvas_w // cols, canvas_h // rows

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (canvas_w, canvas_h))
    try:
        for idx, frame_key in enumerate(frame_keys):
            canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
            sel = selection.get(frame_key, {})
            used_cams = set(sel.get("cameras", []))
            offsets = sel.get("offsets", {})

            for cam_idx, cam_id in enumerate(cam_ids):
                frame_path = os.path.join(take_dir, "aligned", cam_id, frame_key)
                img = cv2.imread(frame_path)
                if img is None:
                    continue
                w, h, (map1, map2) = undistort_maps[cam_id]
                img = hmv.undistort_fast(img, map1, map2)

                is_used = cam_id in used_cams
                if not is_used:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

                lms = landmarks.get(cam_id, {}).get(frame_key)
                if lms:
                    pixel_xy = hmv.landmark_dict_to_pixel_xy(lms, w, h)
                    hmv.draw_hand_skeleton(
                        img, pixel_xy, color=USED_COLOR, point_color=USED_COLOR,
                        thickness=grid_video.BONE_THICKNESS,
                    )

                resized = cv2.resize(img, (slot_w, slot_h), interpolation=cv2.INTER_AREA)
                label = f"{cam_id}  {'USED' if is_used else 'unused'}"
                offset = offsets.get(cam_id)
                if offset:
                    label += f"  (offset {offset:+d})"
                cv2.putText(
                    resized, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    USED_COLOR if is_used else (255, 255, 255), 2, cv2.LINE_AA,
                )
                conf = confidence.get(cam_id, {}).get(frame_key)
                if conf is not None:
                    cv2.putText(
                        resized, f"conf {conf:.2f}", (10, slot_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2, cv2.LINE_AA,
                    )

                row, col = cam_idx // cols, cam_idx % cols
                y1, x1 = row * slot_h, col * slot_w
                canvas[y1:y1 + slot_h, x1:x1 + slot_w] = resized

            header = f"frame {idx} ({frame_key})  n_inliers={sel.get('n_inliers', 0)}"
            cv2.putText(canvas, header, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)

            writer.write(canvas)
            if (idx + 1) % 50 == 0 or (idx + 1) == len(frame_keys):
                print(f"  Progress: {idx + 1}/{len(frame_keys)} grid frames compiled.")
    finally:
        writer.release()
        print(f"Saved usage grid video to {os.path.abspath(output_path)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("take_dir")
    parser.add_argument("--variant", choices=["baseline", "multioffset"], default="baseline")
    parser.add_argument("--calib", default=session_triangulation.DEFAULT_CALIB_PATH)
    parser.add_argument(
        "--grid-video", action="store_true",
        help="Render the usage grid video instead of launching the 3D viewer",
    )
    parser.add_argument(
        "--output", default=None,
        help="Grid video output path (default: <take_dir>/aligned/usage_grid[_multioffset].mp4)",
    )
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--cols", type=int, default=3)
    parser.add_argument("--canvas-width", type=int, default=1920)
    parser.add_argument("--canvas-height", type=int, default=1080)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument(
        "--raw", action="store_true",
        help="3D mode only: play the raw (unsmoothed) reconstruction instead of the Kalman-smoothed one",
    )
    args = parser.parse_args()

    calib = calibrate.load_calibration_output(args.calib)

    if args.grid_video:
        _, diagnostics, landmarks, confidence = load_take_outputs(args.take_dir, args.variant)
        cam_ids = diagnostics["metrics"]["cam_ids"]
        output_path = args.output or os.path.join(
            args.take_dir, "aligned", f"usage_grid{_suffix(args.variant)}.mp4"
        )
        render_usage_grid_video(
            args.take_dir, calib, cam_ids, diagnostics, landmarks, confidence,
            output_path, fps=args.fps, cols=args.cols, canvas_size=(args.canvas_width, args.canvas_height),
        )
    else:
        launch_3d_viewer(args.take_dir, args.variant, calib, args.port, args.fps, use_smoothed=not args.raw)


if __name__ == "__main__":
    main()
