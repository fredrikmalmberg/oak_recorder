"""Grid-canvas video compositor for pose2d's extraction+tracking output --
this IS the deliverable used to judge 2D keypoint quality (per the user's
own acceptance criterion). Structurally mirrors hand_pose/grid_video.py's
render_hand_grid_video (grid math, skeleton drawing via hand_multiview.
draw_hand_skeleton/landmark_dict_to_pixel_xy, confidence-gray fallback,
leftover-slot legend panel) but reads this rig's own per-camera-undistorted
aligned JPEGs instead of h5 frames, and overlays pose2d.tracking's flags.
"""
import os

import cv2
import numpy as np

from hand_pose import hand_multiview as hmv

LEFT_COLOR = (0, 200, 255)
RIGHT_COLOR = (255, 160, 0)
BODY_COLOR = (255, 255, 255)
GRAY_COLOR = (140, 140, 140)
FLAG_COLOR = (0, 0, 255)


def render_pose2d_grid_video(
    take_dir, calib, cam_ids,
    landmarks_left, confidence_left, landmarks_right, confidence_right,
    landmarks_body, diagnostics, output_path,
    fps=15.0, cols=3, canvas_size=(1920, 1080), min_confidence=0.5,
):
    frame_keys = hmv.discover_frames(take_dir, cam_ids[0])

    undistort_maps = {}
    for cam_id in cam_ids:
        c = calib[cam_id]
        w, h = hmv.get_image_size(take_dir, cam_id, frame_keys[0])
        undistort_maps[cam_id] = (w, h, hmv.build_undistort_maps(c["K"], c["dist"], w, h))

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

                body_lms = landmarks_body.get(cam_id, {}).get(frame_key)
                if body_lms:
                    pixel_xy = hmv.landmark_dict_to_pixel_xy(body_lms, w, h)
                    hmv.draw_hand_skeleton(
                        img, pixel_xy, color=BODY_COLOR, point_color=BODY_COLOR,
                        thickness=3, connections=hmv.BODY_CONNECTIONS_COCO,
                    )

                left_lms = landmarks_left.get(cam_id, {}).get(frame_key)
                if left_lms:
                    left_conf = confidence_left.get(cam_id, {}).get(frame_key, 0)
                    color = LEFT_COLOR if left_conf >= min_confidence else GRAY_COLOR
                    pixel_xy = hmv.landmark_dict_to_pixel_xy(left_lms, w, h)
                    hmv.draw_hand_skeleton(img, pixel_xy, color=color, point_color=color, thickness=4)

                right_lms = landmarks_right.get(cam_id, {}).get(frame_key)
                if right_lms:
                    right_conf = confidence_right.get(cam_id, {}).get(frame_key, 0)
                    color = RIGHT_COLOR if right_conf >= min_confidence else GRAY_COLOR
                    pixel_xy = hmv.landmark_dict_to_pixel_xy(right_lms, w, h)
                    hmv.draw_hand_skeleton(img, pixel_xy, color=color, point_color=color, thickness=4)

                resized = cv2.resize(img, (slot_w, slot_h), interpolation=cv2.INTER_AREA)
                cv2.putText(
                    resized, cam_id, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA,
                )

                flags = []
                if diagnostics["swap_flags"].get(cam_id, {}).get(frame_key):
                    flags.append("SWAP")
                if diagnostics["velocity_flags_left"].get(cam_id, {}).get(frame_key):
                    flags.append("VEL-L")
                if diagnostics["velocity_flags_right"].get(cam_id, {}).get(frame_key):
                    flags.append("VEL-R")
                if diagnostics["occlusion_flags"].get(cam_id, {}).get(frame_key):
                    flags.append("OCC")
                if diagnostics.get("single_hand_occlusion_flags", {}).get(cam_id, {}).get(frame_key):
                    flags.append("OCC1H?")
                if diagnostics.get("body_untrusted_flags", {}).get(cam_id, {}).get(frame_key):
                    flags.append("BODYRISK")
                if flags:
                    cv2.putText(
                        resized, ",".join(flags), (10, slot_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        FLAG_COLOR, 2, cv2.LINE_AA,
                    )

                row, col = cam_idx // cols, cam_idx % cols
                y1, x1 = row * slot_h, col * slot_w
                canvas[y1:y1 + slot_h, x1:x1 + slot_w] = resized

            if empty_slots:
                panel = np.zeros((slot_h, slot_w, 3), dtype=np.uint8)
                lines = [
                    "LEGEND", "white = body", "orange = left hand", "blue = right hand",
                    "gray = low confidence", "red text = flagged", "",
                    f"swaps corrected: {diagnostics['n_swaps_corrected']}",
                    f"occlusion flags: {diagnostics['n_occlusion_flagged']}",
                    f"single-hand occlusion risk: {diagnostics.get('n_single_hand_occlusion_flagged', 0)}",
                    f"body pose untrusted (BODYRISK): {diagnostics.get('n_body_untrusted_flagged', 0)}",
                    f"velocity flags L/R: {diagnostics['n_velocity_flagged_left']}/"
                    f"{diagnostics['n_velocity_flagged_right']}",
                ]
                y = 28
                for line in lines:
                    cv2.putText(panel, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
                    y += 27
                row, col = empty_slots[0] // cols, empty_slots[0] % cols
                canvas[row * slot_h:(row + 1) * slot_h, col * slot_w:(col + 1) * slot_w] = panel

            header = f"frame {idx} ({frame_key})"
            cv2.putText(canvas, header, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)

            writer.write(canvas)
            if (idx + 1) % 50 == 0 or (idx + 1) == len(frame_keys):
                print(f"  Progress: {idx + 1}/{len(frame_keys)} grid frames compiled.")
    finally:
        writer.release()
        print(f"Saved pose2d grid video to {os.path.abspath(output_path)}")
