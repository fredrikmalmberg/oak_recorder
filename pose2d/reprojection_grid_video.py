"""Reprojection-diagnostic grid video: for every camera, shows the raw
MediaPipe 2D keypoints (light gray, de-emphasized) alongside the
triangulated 3D points reprojected BACK into that camera's own view (in
color), using the same calibration the triangulation was built from. Where
the two overlap closely, that camera's geometry is consistent with the 3D
result; where they diverge, that's exactly the camera/frame worth
investigating (bad calibration for that camera, a bad RANSAC inlier
choice, occlusion/mislabel, etc.) -- a per-camera reprojection-error
readout as a picture, not just an aggregate number.

Reprojects into EVERY camera the calibration covers, not just the ones
pose2d.triangulation's RANSAC actually selected that frame -- comparing a
frame's raw detection against what a non-selected camera's own geometry
would have predicted is exactly the point. Each tile is tagged USED/unused
per part (from pose2d.triangulation's per_frame_selection) so a big
mismatch on a correctly-excluded camera reads as expected, not alarming.

Usage:
    python -m pose2d.reprojection_grid_video <take_dir> [--calib PATH] [--smoothed] [--output PATH]
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
from pose2d import triangulation  # noqa: E402

GRAY_COLOR = (170, 170, 170)
REPROJ_COLOR = {"left": (0, 200, 255), "right": (255, 160, 0), "body": (0, 255, 0)}
CONNECTIONS = {"left": hmv.HAND_CONNECTIONS, "right": hmv.HAND_CONNECTIONS, "body": hmv.BODY_CONNECTIONS_COCO}
FLAG_COLOR = (0, 0, 255)


def load_data(take_dir):
    base = os.path.join(take_dir, "aligned", "pose2d")
    data = {}
    for part in ("left", "right", "body"):
        with open(os.path.join(base, f"landmarks_{part}.json")) as f:
            data[f"landmarks_{part}"] = json.load(f)
        with open(os.path.join(base, f"reconstruction_{part}.json")) as f:
            data[f"reconstruction_{part}"] = json.load(f)
        with open(os.path.join(base, f"triangulation_diagnostics_{part}.json")) as f:
            data[f"diagnostics_{part}"] = json.load(f)
    return data


def reproject_points(frame_points, P):
    """frame_points: dict[lm_id_str] -> [x, y, z]. Returns dict[int] -> np.array([u, v])
    in this camera's own (undistorted) pixel space, using the same
    projection matrix P = K[R|t] the triangulation itself used.
    """
    pixel_xy = {}
    for lm_id, xyz in frame_points.items():
        X = np.array([xyz[0], xyz[1], xyz[2], 1.0])
        uvw = P @ X
        if abs(uvw[2]) < 1e-9:
            continue
        pixel_xy[int(lm_id)] = np.array([uvw[0] / uvw[2], uvw[1] / uvw[2]])
    return pixel_xy


def render_reprojection_grid_video(
    take_dir, calib, cam_ids, data, output_path,
    fps=15.0, cols=3, canvas_size=(1920, 1080), use_smoothed=False,
):
    frame_keys = hmv.discover_frames(take_dir, cam_ids[0])
    recon_key = "smoothed" if use_smoothed else "raw"

    undistort_maps = {}
    projection_matrices = {}
    for cam_id in cam_ids:
        c = calib[cam_id]
        w, h = hmv.get_image_size(take_dir, cam_id, frame_keys[0])
        undistort_maps[cam_id] = (w, h, hmv.build_undistort_maps(c["K"], c["dist"], w, h))
        projection_matrices[cam_id] = hmv.build_projection_matrix(c["K"], c["R"], c["t"])

    canvas_w, canvas_h = canvas_size
    rows = int(np.ceil(len(cam_ids) / cols))
    slot_w, slot_h = canvas_w // cols, canvas_h // rows
    empty_slots = list(range(len(cam_ids), rows * cols))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (canvas_w, canvas_h))
    try:
        for idx, frame_key in enumerate(frame_keys):
            canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

            for cam_idx, cam_id in enumerate(cam_ids):
                frame_path = os.path.join(take_dir, "aligned", cam_id, frame_key)
                img = cv2.imread(frame_path)
                if img is None:
                    continue
                w, h, (map1, map2) = undistort_maps[cam_id]
                img = hmv.undistort_fast(img, map1, map2)
                P = projection_matrices[cam_id]

                # Decide per-part "used this frame" first, so the background
                # can be desaturated BEFORE any keypoints are drawn on top --
                # keypoint coloring (gray raw / colored reprojection) always
                # stays as-is regardless of this.
                used_by_part = {}
                for part in ("left", "right", "body"):
                    sel = data[f"diagnostics_{part}"]["per_frame_selection"].get(frame_key, {})
                    used_by_part[part] = cam_id in sel.get("cameras", [])
                used_any = any(used_by_part.values())
                if not used_any:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

                tags = []
                for part in ("left", "right", "body"):
                    raw_lms = data[f"landmarks_{part}"].get(cam_id, {}).get(frame_key)
                    if raw_lms:
                        pixel_xy = hmv.landmark_dict_to_pixel_xy(raw_lms, w, h)
                        hmv.draw_hand_skeleton(
                            img, pixel_xy, color=GRAY_COLOR, point_color=GRAY_COLOR,
                            thickness=2, connections=CONNECTIONS[part],
                        )

                    frame_points = data[f"reconstruction_{part}"][recon_key].get(frame_key)
                    if frame_points:
                        reproj_xy = reproject_points(frame_points, P)
                        hmv.draw_hand_skeleton(
                            img, reproj_xy, color=REPROJ_COLOR[part], point_color=REPROJ_COLOR[part],
                            thickness=2, connections=CONNECTIONS[part],
                        )
                        tags.append(f"{part[0].upper()}:{'Y' if used_by_part[part] else 'n'}")

                resized = cv2.resize(img, (slot_w, slot_h), interpolation=cv2.INTER_AREA)
                cv2.putText(
                    resized, cam_id, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA,
                )
                if tags:
                    cv2.putText(
                        resized, " ".join(tags), (10, slot_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        FLAG_COLOR, 2, cv2.LINE_AA,
                    )

                row, col = cam_idx // cols, cam_idx % cols
                y1, x1 = row * slot_h, col * slot_w
                canvas[y1:y1 + slot_h, x1:x1 + slot_w] = resized

            if empty_slots:
                panel = np.zeros((slot_h, slot_w, 3), dtype=np.uint8)
                lines = [
                    "LEGEND",
                    "grayscale TILE = not used for ANY part", "this frame (keypoint colors unchanged)",
                    "gray KEYPOINTS = raw MediaPipe detection", "",
                    "colored = triangulated point reprojected", "into THIS camera's own view:",
                    "  cyan/orange = left/right hand", "  green = body", "",
                    "tag L/R/B:Y = camera WAS in that", "part's RANSAC inlier set this frame",
                    "tag ...:n = reprojected for comparison", "only -- not used to triangulate",
                    "", f"reconstruction: {'smoothed' if use_smoothed else 'raw'}",
                ]
                y = 24
                for line in lines:
                    cv2.putText(panel, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    y += 24
                row, col = empty_slots[0] // cols, empty_slots[0] % cols
                canvas[row * slot_h:(row + 1) * slot_h, col * slot_w:(col + 1) * slot_w] = panel

            header = f"frame {idx} ({frame_key})"
            cv2.putText(canvas, header, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)

            writer.write(canvas)
            if (idx + 1) % 50 == 0 or (idx + 1) == len(frame_keys):
                print(f"  Progress: {idx + 1}/{len(frame_keys)} grid frames compiled.")
    finally:
        writer.release()
        print(f"Saved reprojection grid video to {os.path.abspath(output_path)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("take_dir")
    parser.add_argument("--calib", default=triangulation.DEFAULT_CALIB_PATH)
    parser.add_argument("--smoothed", action="store_true", help="Reproject the smoothed reconstruction instead of raw")
    parser.add_argument("--output", default=None)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--cols", type=int, default=3)
    args = parser.parse_args()

    calib = calibrate.load_calibration_output(args.calib)
    data = load_data(args.take_dir)
    cam_ids = [c for c in hmv.discover_cameras(args.take_dir) if c in calib]

    output_path = args.output or os.path.join(
        args.take_dir, "aligned", "pose2d",
        f"reprojection_grid{'_smoothed' if args.smoothed else ''}.mp4",
    )
    render_reprojection_grid_video(
        args.take_dir, calib, cam_ids, data, output_path,
        fps=args.fps, cols=args.cols, use_smoothed=args.smoothed,
    )


if __name__ == "__main__":
    main()
