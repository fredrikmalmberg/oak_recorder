"""Generate a grid video with DWPose body landmarks overlaid on each camera frame.

Uses the ONNX-based DWPose inference (YOLOX detector + RTMPose/DWPose pose model)
from /home/fmalmb/CODE/vq-sign/pose_extract/DWPose/ctrlnet. No CWD dependency.

Output format is identical to make_body_landmark_video.py (4x2 grid, same
color-coding) so the two videos can be compared side-by-side.

Usage:
    python make_dwpose_video.py
    python make_dwpose_video.py --out my_video.mp4 --scale 0.2 --fps 15
    python make_dwpose_video.py --cpu   # force CPU (slower, but no CUDA needed)
"""

import argparse
import sys

import cv2
import h5py
import numpy as np
import onnxruntime as ort

# DWPose inference utilities from the existing vq-sign installation.
_DWPOSE_ROOT = '/home/fmalmb/CODE/vq-sign/pose_extract/DWPose/ctrlnet'
sys.path.insert(0, _DWPOSE_ROOT)
from annotator.dwpose.onnxdet import inference_detector
from annotator.dwpose.onnxpose import inference_pose

# ---- Configuration -----------------------------------------------------------

DET_ONNX = f'{_DWPOSE_ROOT}/annotator/ckpts/yolox_l.onnx'
POSE_ONNX = f'{_DWPOSE_ROOT}/annotator/ckpts/dw-ll_ucoco_384.onnx'

H5_PATH = '/data/fmalmb/DATA/tmp/Testdata.h5'
CAMERA_IDS = ['cam00', 'cam01', 'cam02', 'cam03', 'cam04', 'cam05', 'cam06']
N_FRAMES = 128
FRAME_OFFSET = 0
DEFAULT_FPS = 30.0
DEFAULT_SCALE = 1 / 6
DEFAULT_OUT = 'dwpose_body.mp4'

GRID_COLS = 4
GRID_ROWS = 2

SCORE_THRESH = 0.3  # minimum joint confidence to draw

# OpenPose 18-joint body order (output of Wholebody after mmpose→openpose remap):
#   0:nose  1:neck  2:r_sho  3:r_elb  4:r_wri  5:l_sho  6:l_elb  7:l_wri
#   8:r_hip  9:r_kne  10:r_ank  11:l_hip  12:l_kne  13:l_ank
#   14:r_eye  15:l_eye  16:r_ear  17:l_ear
N_BODY = 18

BODY_CONNECTIONS = [
    (0, 1), (1, 2), (1, 5),
    (2, 3), (3, 4),
    (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10),
    (1, 11), (11, 12), (12, 13),
    (0, 14), (14, 16), (0, 15), (15, 17),
    (2, 16), (5, 17),
]

# BGR colors: right=red, left=green, center/face=yellow
_RIGHT_IDX = {2, 3, 4, 8, 9, 10, 16}
_LEFT_IDX = {5, 6, 7, 11, 12, 13, 17}
_COLOR_RIGHT = (50, 50, 220)
_COLOR_LEFT = (50, 200, 50)
_COLOR_CENTER = (50, 210, 210)
_COLOR_LINE = (200, 200, 200)
_COLOR_LABEL = (240, 240, 240)


# ---- Model -------------------------------------------------------------------

def load_model(use_gpu=True):
    providers = (['CUDAExecutionProvider', 'CPUExecutionProvider']
                 if use_gpu else ['CPUExecutionProvider'])
    session_det = ort.InferenceSession(DET_ONNX, providers=providers)
    session_pose = ort.InferenceSession(POSE_ONNX, providers=providers)
    return session_det, session_pose


def run_dwpose(session_det, session_pose, img):
    """Run DWPose on a BGR image. Returns body keypoints (N_BODY, 2) and
    scores (N_BODY,) for the most prominent detected person, or (None, None)
    if no person is detected.

    Keypoints are in pixel coordinates matching the input image.
    """
    bboxes = inference_detector(session_det, img)
    if bboxes is None or len(bboxes) == 0:
        return None, None

    keypoints, scores = inference_pose(session_pose, bboxes, img)
    # keypoints: (n_persons, 133, 2), scores: (n_persons, 133)

    # Synthetic neck = mean of shoulders (joints 5 and 6 in COCO body order,
    # indices 5,6 in the 133-point output before reindexing).
    # The Wholebody.__call__ also does the mmpose→openpose remap; replicate it.
    kps_info = np.concatenate((keypoints, scores[..., None]), axis=-1)

    neck = np.mean(kps_info[:, [5, 6]], axis=1)
    neck[:, 2:4] = np.logical_and(
        kps_info[:, 5, 2:4] > 0.3, kps_info[:, 6, 2:4] > 0.3).astype(int)
    kps_info = np.insert(kps_info, 17, neck, axis=1)  # now 134 joints

    mmpose_idx = [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]
    openpose_idx = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
    kps_info[:, openpose_idx] = kps_info[:, mmpose_idx]

    # Pick the person with highest mean body-joint confidence.
    body_scores = kps_info[:, :N_BODY, 2]
    best = int(np.argmax(body_scores.mean(axis=1)))

    return kps_info[best, :N_BODY, :2], kps_info[best, :N_BODY, 2]


# ---- Drawing -----------------------------------------------------------------

def _joint_color(idx):
    if idx in _RIGHT_IDX:
        return _COLOR_RIGHT
    if idx in _LEFT_IDX:
        return _COLOR_LEFT
    return _COLOR_CENTER


def draw_dwpose(img, keypoints, scores, dot_radius=8, line_thickness=3):
    """Overlay DWPose body landmarks on `img` in-place.

    `keypoints` (N_BODY, 2) and `scores` (N_BODY,) as returned by run_dwpose.
    Joints below SCORE_THRESH are skipped.
    """
    if keypoints is None:
        return

    h, w = img.shape[:2]
    pts = {}
    for i, ((x, y), s) in enumerate(zip(keypoints, scores)):
        if s < SCORE_THRESH:
            continue
        xi = int(round(np.clip(x, 0, w - 1)))
        yi = int(round(np.clip(y, 0, h - 1)))
        pts[i] = (xi, yi)

    for a, b in BODY_CONNECTIONS:
        if a in pts and b in pts:
            cv2.line(img, pts[a], pts[b], _COLOR_LINE, line_thickness, cv2.LINE_AA)

    for i, p in pts.items():
        color = _joint_color(i)
        cv2.circle(img, p, dot_radius, color, -1, cv2.LINE_AA)
        cv2.circle(img, p, dot_radius, (255, 255, 255), 1, cv2.LINE_AA)


def add_label(img, text, scale=1.0):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.7 * scale
    thickness = max(1, int(2 * scale))
    (tw, th), baseline = cv2.getTextSize(text, font, fs, thickness)
    pad = max(4, int(6 * scale))
    cv2.rectangle(img, (pad, pad), (pad + tw + 4, pad + th + baseline + 4),
                  (20, 20, 20), -1)
    cv2.putText(img, text, (pad + 2, pad + th + 2), font, fs, _COLOR_LABEL,
                thickness, cv2.LINE_AA)


# ---- Main --------------------------------------------------------------------

def make_grid_video(out_path, scale, fps, use_gpu=True, verbose=True):
    session_det, session_pose = load_model(use_gpu)
    if verbose:
        ep = session_det.get_providers()[0]
        print(f'DWPose loaded  det+pose  ({ep})')

    with h5py.File(H5_PATH, 'r') as hf:
        ref = cv2.imdecode(hf['rec/cam00/frames/000000/color'][:], cv2.IMREAD_COLOR)
    orig_h, orig_w = ref.shape[:2]

    cell_w = int(round(orig_w * scale))
    cell_h = int(round(orig_h * scale))
    cell_w += cell_w % 2
    cell_h += cell_h % 2

    grid_w = GRID_COLS * cell_w
    grid_h = GRID_ROWS * cell_h

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

                kps, scores = run_dwpose(session_det, session_pose, img)
                draw_dwpose(img, kps, scores,
                             dot_radius=dot_radius, line_thickness=line_thickness)
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
    parser = argparse.ArgumentParser(description='DWPose body grid video')
    parser.add_argument('--out', default=DEFAULT_OUT)
    parser.add_argument('--scale', type=float, default=DEFAULT_SCALE)
    parser.add_argument('--fps', type=float, default=DEFAULT_FPS)
    parser.add_argument('--cpu', action='store_true', help='Force CPU inference')
    args = parser.parse_args()
    make_grid_video(args.out, args.scale, args.fps, use_gpu=not args.cpu)


if __name__ == '__main__':
    main()
