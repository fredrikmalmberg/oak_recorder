"""Generate a grid video with MediaPipe body landmarks overlaid on each camera frame.

Reads raw frames from the H5 recording and body keypoints from
extracted_2d_keypoints.json, then writes a 4x2 grid video (7 cameras,
one black placeholder cell) with annotated landmarks and skeleton connections.

Usage:
    python make_body_landmark_video.py
    python make_body_landmark_video.py --out my_video.mp4 --scale 0.2 --fps 15
"""

import argparse
import json
import sys

import cv2
import h5py
import numpy as np

from v3.skeletons import BODY_CONNECTIONS

# ---- Configuration -----------------------------------------------------------

H5_PATH = '/data/fmalmb/DATA/tmp/Testdata.h5'
KEYPOINTS_JSON = 'extracted_2d_keypoints.json'
CAMERA_IDS = ['cam00', 'cam01', 'cam02', 'cam03', 'cam04', 'cam05', 'cam06']
N_FRAMES = 128
FRAME_OFFSET = 0
DEFAULT_FPS = 30.0
DEFAULT_SCALE = 1 / 6  # each cell = 560x315 from 3360x1890; 4x2 grid = 2240x630
DEFAULT_OUT = 'body_landmarks.mp4'

# Grid layout: 4 columns x 2 rows (8 cells, 7 cameras, 1 black placeholder).
GRID_COLS = 4
GRID_ROWS = 2

# MediaPipe Pose landmark index sets for color-coding by body side.
_LEFT_INDICES = {1, 2, 3, 7, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31}
_RIGHT_INDICES = {4, 5, 6, 8, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32}

# BGR colors
_COLOR_LEFT = (50, 200, 50)     # green
_COLOR_RIGHT = (50, 50, 220)    # red
_COLOR_CENTER = (50, 210, 210)  # yellow
_COLOR_LINE = (200, 200, 200)   # light gray skeleton lines
_COLOR_LABEL = (240, 240, 240)  # camera label text


# ---- Drawing -----------------------------------------------------------------

def _landmark_color(idx):
    if idx in _LEFT_INDICES:
        return _COLOR_LEFT
    if idx in _RIGHT_INDICES:
        return _COLOR_RIGHT
    return _COLOR_CENTER


def draw_body_landmarks(img, landmarks, dot_radius=8, line_thickness=3):
    """Overlay body landmarks and skeleton connections on `img` in-place.

    `landmarks` is the per-camera, per-frame list from extracted_2d_keypoints.json.
    Does nothing (returns early) if landmarks is empty / not 33 points.
    """
    if len(landmarks) != 33:
        return

    h, w = img.shape[:2]
    pts = {}
    for lm in landmarks:
        i = lm['id']
        x = int(round(np.clip(lm['x'], 0, w - 1)))
        y = int(round(np.clip(lm['y'], 0, h - 1)))
        pts[i] = (x, y)

    # lines first so dots render on top
    for a, b in BODY_CONNECTIONS:
        if a in pts and b in pts:
            cv2.line(img, pts[a], pts[b], _COLOR_LINE, line_thickness, cv2.LINE_AA)

    for i, (x, y) in pts.items():
        color = _landmark_color(i)
        cv2.circle(img, (x, y), dot_radius, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x, y), dot_radius, (255, 255, 255), 1, cv2.LINE_AA)


def add_label(img, text, scale=1.0):
    """Put `text` in the top-left corner of `img` in-place."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.7 * scale
    thickness = max(1, int(2 * scale))
    (tw, th), baseline = cv2.getTextSize(text, font, fs, thickness)
    pad = max(4, int(6 * scale))
    # dark background rectangle
    cv2.rectangle(img, (pad, pad), (pad + tw + 4, pad + th + baseline + 4),
                  (20, 20, 20), -1)
    cv2.putText(img, text, (pad + 2, pad + th + 2), font, fs, _COLOR_LABEL,
                thickness, cv2.LINE_AA)


# ---- Main --------------------------------------------------------------------

def make_grid_video(out_path, scale, fps, verbose=True):
    with open(KEYPOINTS_JSON) as f:
        keypoints_2d = json.load(f)

    # Decode one reference frame to get original dimensions.
    with h5py.File(H5_PATH, 'r') as hf:
        ref = cv2.imdecode(hf['rec/cam00/frames/000000/color'][:], cv2.IMREAD_COLOR)
    orig_h, orig_w = ref.shape[:2]

    cell_w = int(round(orig_w * scale))
    cell_h = int(round(orig_h * scale))
    # keep even dimensions (required by most video codecs)
    cell_w += cell_w % 2
    cell_h += cell_h % 2

    grid_w = GRID_COLS * cell_w
    grid_h = GRID_ROWS * cell_h

    # dot radius and line thickness scaled with cell size
    dot_radius = max(2, int(8 * scale))
    line_thickness = max(1, int(3 * scale))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (grid_w, grid_h))
    if not writer.isOpened():
        raise RuntimeError(f'Could not open VideoWriter for {out_path!r}')

    if verbose:
        print(f'grid size: {grid_w}x{grid_h}  cell: {cell_w}x{cell_h}  fps={fps}')
        print(f'writing {N_FRAMES} frames to {out_path!r} ...')

    with h5py.File(H5_PATH, 'r') as hf:
        for fi in range(N_FRAMES):
            frame_key = f'{FRAME_OFFSET + fi:06d}'
            grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

            for ci, cam_id in enumerate(CAMERA_IDS):
                raw = hf[f'rec/{cam_id}/frames/{frame_key}/color'][:]
                img = cv2.imdecode(raw, cv2.IMREAD_COLOR)

                landmarks = keypoints_2d[frame_key][cam_id]['body']
                draw_body_landmarks(img, landmarks,
                                     dot_radius=dot_radius,
                                     line_thickness=line_thickness)
                add_label(img, f'{cam_id}  f{frame_key}', scale=scale * 6)

                cell_img = cv2.resize(img, (cell_w, cell_h), interpolation=cv2.INTER_AREA)

                row, col = divmod(ci, GRID_COLS)
                y0, x0 = row * cell_h, col * cell_w
                grid[y0:y0 + cell_h, x0:x0 + cell_w] = cell_img

            writer.write(grid)

            if verbose and (fi % 16 == 0 or fi == N_FRAMES - 1):
                print(f'  frame {fi + 1}/{N_FRAMES}')

    writer.release()
    if verbose:
        print('done.')


def main():
    parser = argparse.ArgumentParser(description='Body landmark grid video')
    parser.add_argument('--out', default=DEFAULT_OUT)
    parser.add_argument('--scale', type=float, default=DEFAULT_SCALE,
                        help='Scale factor for each camera cell (default: 1/6)')
    parser.add_argument('--fps', type=float, default=DEFAULT_FPS)
    args = parser.parse_args()
    make_grid_video(args.out, args.scale, args.fps)


if __name__ == '__main__':
    main()
