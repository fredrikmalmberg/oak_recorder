"""Composites a grid-layout, skeleton-overlay video from tmp/Testdata.h5's
per-camera frames for a single hand-landmark detector at a time (mediapipe or
dwpose). Reuses h5_explore.ipynb's validated grid-canvas layout (fixed-size
per-camera slots blitted into one canvas, written via cv2.VideoWriter) as fresh,
standalone code rather than editing that notebook, so it stays independently
reusable/cleanable once a detector setup is chosen for a real pipeline. This is
also what actually surfaces per-frame detection failures (a bad frame is visible
at a glance across the whole camera rig, not just in a handful of sampled stills).

Also overlays mixup-detector results (see hand_multiview.py's
detect_camera_hand_swap/cross_detector_hand_disagreement/flag_jitter_frames) so a
detector's output can be sanity-checked directly against the footage that
triggered it, rather than only as printed tables. The grid's leftover slots
(cam count < rows*cols, e.g. 7 cameras in a 3x3 grid leaves 2 empty) show the
thresholds/config used to generate the video and any frame-level (not
single-camera) detector flags, so a rendered video is self-describing.
"""
import os

import cv2
import numpy as np

import h5_dataset
import hand_multiview

BONE_THICKNESS = 4  # 2x hand_multiview.draw_hand_skeleton's own default (2)
FLAG_COLOR = (0, 0, 255)
GRAY_COLOR = (140, 140, 140)


def _wrist_px(lms, w, h):
    if lms is None:
        return None
    obs = lms.get('0')
    if obs is None:
        return None
    return np.array([obs[0] * w, obs[1] * h], dtype=np.float64)


def _fmt_vel(vel):
    return f'{vel:.1f}' if vel is not None else 'n/a'


def _fmt_conf(conf):
    return f'{conf:.2f}' if conf is not None else 'n/a'


def _draw_text_block(panel, lines, origin=(20, 40), line_height=32, font_scale=0.7, color=(255, 255, 255)):
    x, y = origin
    for line in lines:
        cv2.putText(panel, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2, cv2.LINE_AA)
        y += line_height


def render_hand_grid_video(
    h5_path, cam_ids, frame_keys, output_path,
    landmarks_left, confidence_left, landmarks_right, confidence_right,
    method_label='', left_color=(0, 200, 255), right_color=(255, 160, 0),
    fps=15.0, cols=3, canvas_size=(3840, 2160), min_confidence=0.5,
    per_camera_flags=None, global_flags=None, thresholds_text=None, camera_stats_text=None,
    landmark_confidence_left=None, landmark_confidence_right=None, wrist_landmark_id=0,
    draw_velocity_circles=True, velocity_thresh_px=200.0,
    landmarks_body=None, body_color=(255, 255, 255),
):
    """One grid-layout video per call -- pass a single detector's
    (landmarks_left/right, confidence_left/right) pair (e.g. MediaPipe's or
    DWPose's) and a `method_label` naming it (e.g. 'MediaPipe', 'DWPose'),
    matching hand_multiview.py's schema throughout this project. Left/right hands
    are drawn in distinct colors (left_color/right_color) via
    hand_multiview.draw_hand_skeleton whenever landmarks exist at all -- but in
    GRAY_COLOR instead when that hand's confidence is below `min_confidence`, so
    it's visually obvious which detections would actually be excluded downstream
    (a low-confidence detection isn't invisible, it's just marked as unusable).
    Each frame overlays, in the video's top-left corner, the method label and
    frame number/key; per camera slot, that camera's own frame-to-frame wrist
    pixel-velocity (px/frame, 2D image space, bottom-left) and, above it, which
    per-camera mixup detectors triggered this frame (a swap-type flag inherently
    concerns the left/right PAIR, not a single hand, so no hand suffix is shown
    there).

    draw_velocity_circles: when True (the default -- intended for the original,
    uncorrected videos; pass False for "corrected" renders, which have already
    had implausible jumps flip-corrected and don't need the same visual aid),
    draws a circle in each hand's color centered on that hand's last-known wrist
    position, radius = velocity_thresh_px * (number of frames since that
    position was last detected) -- 1 frame in the normal case, larger after a
    detection gap. The current detection landing inside its own circle is the
    visual equivalent of hand_multiview.detect_high_velocity_frames NOT flagging
    it. The anchor itself only advances to a new detection when that detection
    lands inside the circle; a detection outside it is treated like no detection
    at all (anchor kept, circle keeps growing) so the tracker can never lock onto
    a wrong-hand detection and start following it.

    per_camera_flags: dict[label] -> dict[cam_id][frame_key] -> bool. Rendered
        per camera slot (e.g. {'XDET': cross_detector_hand_disagreement's
        likely_swap, 'LOO': detect_camera_hand_swap's swap_detected}).
    global_flags: dict[label] -> {'left': set(frame_key,...), 'right': set(...)}.
        These are frame-level, not tied to one camera (e.g. jitter/body-wrist
        consistency flags from a RANSAC-selected camera SET) -- rendered once,
        per hand, in the first leftover grid slot beyond len(cam_ids), labeled
        with which hand triggered since that IS meaningful here.
    thresholds_text / camera_stats_text: list[str] rendered verbatim in the
        second leftover slot, so the rendered video is self-describing about
        what config produced it.
    landmarks_body: optional, same detector's COCO-17 body landmarks (see
        h5_hand_extraction.py's BLAZEPOSE_TO_COCO17 / DWPose's free body slice)
        -- drawn underneath the hands in `body_color` (plain white by default,
        not confidence-graded -- unlike the hands, there's only ever one body
        per camera so there's no identity-mixup concern to highlight) via
        hand_multiview.BODY_CONNECTIONS_COCO, whenever present for that
        camera/frame.
    landmark_confidence_left/right: optional dict[cam_id][frame_key] ->
        {lm_id: score} (e.g. DWPose's per-landmark confidence). When given, the
        per-slot "conf" readout shows this source's actual wrist-landmark
        (`wrist_landmark_id`) confidence rather than the whole-hand
        confidence_left/right score -- MediaPipe has no per-landmark confidence
        of its own, so its calls should leave this at the default (None), which
        falls back to confidence_left/right (its only, whole-hand score, which
        does still cover the wrist).
    """
    canvas_w, canvas_h = canvas_size
    rows = int(np.ceil(len(cam_ids) / cols))
    slot_w = canvas_w // cols
    slot_h = canvas_h // rows
    total = len(frame_keys)
    empty_slots = list(range(len(cam_ids), rows * cols))

    prev_wrist_px = {cam_id: {'left': None, 'right': None} for cam_id in cam_ids}
    frames_since_detection = {cam_id: {'left': 0, 'right': 0} for cam_id in cam_ids}

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (canvas_w, canvas_h))
    try:
        for idx, (frame_key, imgs) in enumerate(h5_dataset.iter_h5_frames(h5_path, cam_ids, frame_keys)):
            canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
            for cam_idx, cam_id in enumerate(cam_ids):
                img = imgs.get(cam_id)
                if img is None:
                    continue
                img = img.copy()
                h, w = img.shape[:2]

                if landmarks_body is not None:
                    body_lms = landmarks_body.get(cam_id, {}).get(frame_key)
                    if body_lms:
                        pixel_xy = hand_multiview.landmark_dict_to_pixel_xy(body_lms, w, h)
                        hand_multiview.draw_hand_skeleton(
                            img, pixel_xy, color=body_color, point_color=body_color,
                            thickness=BONE_THICKNESS, connections=hand_multiview.BODY_CONNECTIONS_COCO,
                        )

                left_lms = landmarks_left.get(cam_id, {}).get(frame_key)
                left_present = left_lms is not None
                left_confident = left_present and confidence_left.get(cam_id, {}).get(frame_key, 0) >= min_confidence
                if left_present:
                    pixel_xy = hand_multiview.landmark_dict_to_pixel_xy(left_lms, w, h)
                    hand_multiview.draw_hand_skeleton(
                        img, pixel_xy, color=left_color if left_confident else GRAY_COLOR,
                        point_color=left_color if left_confident else GRAY_COLOR, thickness=BONE_THICKNESS,
                    )

                right_lms = landmarks_right.get(cam_id, {}).get(frame_key)
                right_present = right_lms is not None
                right_confident = (
                    right_present and confidence_right.get(cam_id, {}).get(frame_key, 0) >= min_confidence
                )
                if right_present:
                    pixel_xy = hand_multiview.landmark_dict_to_pixel_xy(right_lms, w, h)
                    hand_multiview.draw_hand_skeleton(
                        img, pixel_xy, color=right_color if right_confident else GRAY_COLOR,
                        point_color=right_color if right_confident else GRAY_COLOR, thickness=BONE_THICKNESS,
                    )

                # Frame-to-frame wrist pixel displacement in this camera's own 2D
                # image space (not the triangulated 3D point) -- a cheap, camera-
                # local complement to the 3D jitter flagging in hand_multiview.py.
                # Based on presence, not confidence -- a low-confidence detection
                # is still a detection for tracking/velocity purposes (matching
                # hand_multiview.detect_high_velocity_frames, which doesn't gate
                # on confidence either); confidence only controls skeleton color.
                left_wrist_px = _wrist_px(left_lms, w, h) if left_present else None
                right_wrist_px = _wrist_px(right_lms, w, h) if right_present else None
                left_vel = (
                    float(np.linalg.norm(left_wrist_px - prev_wrist_px[cam_id]['left']))
                    if left_wrist_px is not None and prev_wrist_px[cam_id]['left'] is not None else None
                )
                right_vel = (
                    float(np.linalg.norm(right_wrist_px - prev_wrist_px[cam_id]['right']))
                    if right_wrist_px is not None and prev_wrist_px[cam_id]['right'] is not None else None
                )

                if draw_velocity_circles:
                    for side, wrist_px, color in (
                        ('left', left_wrist_px, left_color), ('right', right_wrist_px, right_color),
                    ):
                        center = prev_wrist_px[cam_id][side]
                        if center is not None:
                            radius = velocity_thresh_px * max(frames_since_detection[cam_id][side], 1)
                            cv2.circle(img, tuple(np.round(center).astype(int)), int(round(radius)), color, 2)

                # Anchor update: a new detection only becomes the tracked anchor
                # if it falls within its own confidence circle (radius grows with
                # frames_since_detection, same threshold as the circle just
                # drawn above). A detection landing OUTSIDE that circle is
                # treated the same as no detection at all -- the anchor is kept
                # and the circle keeps growing -- rather than snapping the
                # tracker onto what's almost certainly the wrong hand. This is
                # what stops a single wrong-hand detection from then anchoring
                # every velocity/circle computation on subsequent frames too.
                for side, wrist_px in (('left', left_wrist_px), ('right', right_wrist_px)):
                    prev = prev_wrist_px[cam_id][side]
                    if wrist_px is None:
                        frames_since_detection[cam_id][side] += 1
                        continue
                    if prev is not None:
                        radius = velocity_thresh_px * max(frames_since_detection[cam_id][side], 1)
                        if float(np.linalg.norm(wrist_px - prev)) > radius:
                            frames_since_detection[cam_id][side] += 1
                            continue
                    prev_wrist_px[cam_id][side] = wrist_px
                    frames_since_detection[cam_id][side] = 1

                wrist_key = str(wrist_landmark_id)
                left_wrist_conf = (
                    landmark_confidence_left.get(cam_id, {}).get(frame_key, {}).get(wrist_key)
                    if landmark_confidence_left is not None
                    else confidence_left.get(cam_id, {}).get(frame_key)
                )
                right_wrist_conf = (
                    landmark_confidence_right.get(cam_id, {}).get(frame_key, {}).get(wrist_key)
                    if landmark_confidence_right is not None
                    else confidence_right.get(cam_id, {}).get(frame_key)
                )

                resized = cv2.resize(img, (slot_w, slot_h), interpolation=cv2.INTER_AREA)
                cv2.putText(
                    resized, cam_id, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA,
                )

                triggered = [
                    label for label, per_cam in (per_camera_flags or {}).items()
                    if per_cam.get(cam_id, {}).get(frame_key)
                ]
                if triggered:
                    cv2.putText(
                        resized, 'MIXUP: ' + ','.join(triggered), (10, slot_h - 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, FLAG_COLOR, 2, cv2.LINE_AA,
                    )
                conf_text = f'conf L:{_fmt_conf(left_wrist_conf)}  R:{_fmt_conf(right_wrist_conf)}'
                cv2.putText(
                    resized, conf_text, (10, slot_h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2, cv2.LINE_AA,
                )
                vel_text = f'L:{_fmt_vel(left_vel)}  R:{_fmt_vel(right_vel)} px/f'
                cv2.putText(
                    resized, vel_text, (10, slot_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2, cv2.LINE_AA,
                )

                row, col = cam_idx // cols, cam_idx % cols
                y1, x1 = row * slot_h, col * slot_w
                canvas[y1:y1 + slot_h, x1:x1 + slot_w] = resized

            if len(empty_slots) >= 1:
                panel = np.zeros((slot_h, slot_w, 3), dtype=np.uint8)
                lines = ['CONFIG'] + list(thresholds_text or []) + ['', 'CAMERA STATS'] + list(camera_stats_text or [])
                _draw_text_block(panel, lines)
                row, col = empty_slots[0] // cols, empty_slots[0] % cols
                y1, x1 = row * slot_h, col * slot_w
                canvas[y1:y1 + slot_h, x1:x1 + slot_w] = panel

            if len(empty_slots) >= 2:
                panel = np.zeros((slot_h, slot_w, 3), dtype=np.uint8)
                lines = ['GLOBAL FLAGS (this frame)']
                for label, sides in (global_flags or {}).items():
                    marks = [s.upper()[0] for s in ('left', 'right') if frame_key in sides.get(s, set())]
                    lines.append(f'{label}: {" ".join(marks) if marks else "-"}')
                _draw_text_block(panel, lines, font_scale=0.9, line_height=42)
                row, col = empty_slots[1] // cols, empty_slots[1] % cols
                y1, x1 = row * slot_h, col * slot_w
                canvas[y1:y1 + slot_h, x1:x1 + slot_w] = panel

            header = f'{method_label}  frame {idx} ({frame_key})' if method_label else f'frame {idx} ({frame_key})'
            cv2.putText(canvas, header, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3, cv2.LINE_AA)

            writer.write(canvas)
            if (idx + 1) % 20 == 0 or (idx + 1) == total:
                print(f'  Progress: {idx + 1}/{total} grid frames compiled.')
    finally:
        writer.release()
        print(f'Saved grid video to {os.path.abspath(output_path)}')
