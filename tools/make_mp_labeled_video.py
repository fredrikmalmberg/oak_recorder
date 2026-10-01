"""Grid video with MediaPipe Pose joint indices labeled on each camera frame.

Diagnostic video for verifying that the 33-joint labeling is consistent across
cameras (helpful for spotting left/right swaps or indexing bugs).

The empty 8th grid cell shows the full joint-name legend.

Usage:
    python make_mp_labeled_video.py
    python make_mp_labeled_video.py --out mp_labeled.mp4 --scale 0.25 --fps 10
"""

import argparse
import json

import cv2
import h5py
import numpy as np

from v3.skeletons import BODY_CONNECTIONS

# ---- Config ------------------------------------------------------------------

H5_PATH = '/data/fmalmb/DATA/tmp/Testdata.h5'
KEYPOINTS_JSON = 'extracted_2d_keypoints.json'
CAMERA_IDS = ['cam00', 'cam01', 'cam02', 'cam03', 'cam04', 'cam05', 'cam06']
N_FRAMES = 128
FRAME_OFFSET = 0
DEFAULT_FPS = 15.0
DEFAULT_SCALE = 1 / 6
DEFAULT_OUT = 'mp_labeled.mp4'

GRID_COLS = 4
GRID_ROWS = 2

# MediaPipe Pose 33 joint names (in index order).
MP_JOINT_NAMES = [
    'nose', 'l_eye_in', 'l_eye', 'l_eye_out', 'r_eye_in', 'r_eye', 'r_eye_out',
    'l_ear', 'r_ear', 'mouth_l', 'mouth_r',
    'l_sho', 'r_sho', 'l_elb', 'r_elb', 'l_wri', 'r_wri',
    'l_pin', 'r_pin', 'l_idx', 'r_idx', 'l_thu', 'r_thu',
    'l_hip', 'r_hip', 'l_kne', 'r_kne', 'l_ank', 'r_ank',
    'l_hel', 'r_hel', 'l_ft', 'r_ft',
]

_LEFT_IDX = {1, 2, 3, 7, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31}
_RIGHT_IDX = {4, 5, 6, 8, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32}

_COLOR_LEFT = (50, 200, 50)
_COLOR_RIGHT = (50, 50, 220)
_COLOR_CENTER = (50, 210, 210)
_COLOR_LINE = (180, 180, 180)
_COLOR_LABEL = (240, 240, 240)
_COLOR_LEGEND_BG = (15, 15, 15)

FONT = cv2.FONT_HERSHEY_SIMPLEX
LABEL_FONT_SCALE = 0.38
LABEL_THICKNESS = 1
DOT_RADIUS = 5


def _joint_color(idx):
    if idx in _LEFT_IDX:
        return _COLOR_LEFT
    if idx in _RIGHT_IDX:
        return _COLOR_RIGHT
    return _COLOR_CENTER


def draw_joints(cell_img, pts_cell):
    """pts_cell: dict {joint_idx: (x_cell, y_cell)}"""
    cell_h, cell_w = cell_img.shape[:2]

    # Skeleton lines first.
    for a, b in BODY_CONNECTIONS:
        if a in pts_cell and b in pts_cell:
            cv2.line(cell_img, pts_cell[a], pts_cell[b], _COLOR_LINE, 1, cv2.LINE_AA)

    for i, (cx, cy) in pts_cell.items():
        color = _joint_color(i)
        cv2.circle(cell_img, (cx, cy), DOT_RADIUS, color, -1, cv2.LINE_AA)
        cv2.circle(cell_img, (cx, cy), DOT_RADIUS, (255, 255, 255), 1, cv2.LINE_AA)

        # Index label slightly to the right and up from the joint center.
        label = str(i)
        ox, oy = cx + DOT_RADIUS + 2, cy - 2
        (tw, th), _ = cv2.getTextSize(label, FONT, LABEL_FONT_SCALE, LABEL_THICKNESS)
        cv2.rectangle(cell_img, (ox - 1, oy - th - 1), (ox + tw + 1, oy + 2),
                      (10, 10, 10), -1)
        cv2.putText(cell_img, label, (ox, oy), FONT, LABEL_FONT_SCALE,
                    color, LABEL_THICKNESS, cv2.LINE_AA)


def add_cam_label(cell_img, text):
    fs = 0.55
    (tw, th), bl = cv2.getTextSize(text, FONT, fs, 1)
    pad = 4
    cv2.rectangle(cell_img, (pad, pad), (pad + tw + 3, pad + th + bl + 3), (20, 20, 20), -1)
    cv2.putText(cell_img, text, (pad + 2, pad + th + 1), FONT, fs, _COLOR_LABEL, 1, cv2.LINE_AA)


def draw_legend(cell_img):
    """Fill the empty 8th cell with the joint-name legend."""
    cell_img[:] = _COLOR_LEGEND_BG
    cell_h, cell_w = cell_img.shape[:2]

    title = 'MediaPipe Pose — joint indices'
    (tw, th), _ = cv2.getTextSize(title, FONT, 0.45, 1)
    cv2.putText(cell_img, title, ((cell_w - tw) // 2, th + 8),
                FONT, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    # Two-column layout (joints split ~half and half).
    col_w = cell_w // 2
    fs = 0.36
    line_h = 16
    y_start = th + 22

    for i, name in enumerate(MP_JOINT_NAMES):
        col = 0 if i < 17 else 1
        row = i if i < 17 else i - 17
        x = 8 + col * col_w
        y = y_start + row * line_h
        if y + line_h > cell_h:
            break
        color = _joint_color(i)
        label = f'{i:2d}: {name}'
        cv2.putText(cell_img, label, (x, y), FONT, fs, color, 1, cv2.LINE_AA)


# ---- Main --------------------------------------------------------------------

def make_grid_video(out_path, scale, fps, verbose=True):
    with open(KEYPOINTS_JSON) as f:
        keypoints_2d = json.load(f)

    with h5py.File(H5_PATH, 'r') as hf:
        ref = cv2.imdecode(hf['rec/cam00/frames/000000/color'][:], cv2.IMREAD_COLOR)
    orig_h, orig_w = ref.shape[:2]

    cell_w = int(round(orig_w * scale))
    cell_h = int(round(orig_h * scale))
    cell_w += cell_w % 2
    cell_h += cell_h % 2

    grid_w = GRID_COLS * cell_w
    grid_h = GRID_ROWS * cell_h

    sx = cell_w / orig_w
    sy = cell_h / orig_h

    if verbose:
        print(f'grid {grid_w}×{grid_h}  cell {cell_w}×{cell_h}  fps={fps}')
        print(f'writing {N_FRAMES} frames to {out_path!r} ...')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (grid_w, grid_h))
    if not writer.isOpened():
        raise RuntimeError(f'Cannot open VideoWriter for {out_path!r}')

    with h5py.File(H5_PATH, 'r') as hf:
        for fi in range(N_FRAMES):
            frame_key = f'{FRAME_OFFSET + fi:06d}'
            grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

            for ci, cam_id in enumerate(CAMERA_IDS):
                raw = hf[f'rec/{cam_id}/frames/{frame_key}/color'][:]
                img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
                cell_img = cv2.resize(img, (cell_w, cell_h), interpolation=cv2.INTER_AREA)

                landmarks = keypoints_2d[frame_key][cam_id]['body']
                if len(landmarks) == 33:
                    pts_cell = {
                        lm['id']: (
                            int(round(np.clip(lm['x'] * sx, 0, cell_w - 1))),
                            int(round(np.clip(lm['y'] * sy, 0, cell_h - 1))),
                        )
                        for lm in landmarks
                    }
                    draw_joints(cell_img, pts_cell)

                add_cam_label(cell_img, f'{cam_id}  f{frame_key}')

                row, col = divmod(ci, GRID_COLS)
                y0, x0 = row * cell_h, col * cell_w
                grid[y0:y0 + cell_h, x0:x0 + cell_w] = cell_img

            # Legend in the 8th cell (row=1, col=3).
            legend = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
            draw_legend(legend)
            grid[cell_h:2 * cell_h, 3 * cell_w:4 * cell_w] = legend

            writer.write(grid)

            if verbose and (fi % 16 == 0 or fi == N_FRAMES - 1):
                print(f'  frame {fi + 1}/{N_FRAMES}')

    writer.release()
    if verbose:
        print('done.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default=DEFAULT_OUT)
    parser.add_argument('--scale', type=float, default=DEFAULT_SCALE)
    parser.add_argument('--fps', type=float, default=DEFAULT_FPS)
    args = parser.parse_args()
    make_grid_video(args.out, args.scale, args.fps)


if __name__ == '__main__':
    main()
