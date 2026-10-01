"""Generate a grid video with DWPose keypoints overlaid, highlighting RANSAC inlier
cameras per frame and scaling keypoint radius by detection confidence.

Reads pre-computed 2D detections from dwpose_keypoints.npz and 3D triangulations
from dwpose_keypoint_3d.npz (produced by dwpose_body_reconstruction.ipynb).

Visual encoding
---------------
  Keypoint RADIUS: scales linearly with joint confidence (small = low, large = high).
  Inlier camera  : keypoints in the normal red / green / yellow color scheme.
  Non-inlier cam : keypoints in grey; skeleton in dark grey.
  Cell border    : bright-green  = RANSAC inlier for this frame.
                   grey          = DWPose detected someone but camera not in consensus.
                   (no border)   = no person detected.

Usage
-----
    python make_dwpose_ransac_video.py
    python make_dwpose_ransac_video.py --out my.mp4 --scale 0.25 --fps 15
"""

import argparse

import cv2
import h5py
import numpy as np

from v3.calibration import load_calibration

# ---- Configuration -----------------------------------------------------------

H5_PATH = '/data/fmalmb/DATA/tmp/Testdata.h5'
DWPOSE_NPZ = 'dwpose_keypoints.npz'
KEYPOINTS_3D_NPZ = 'dwpose_keypoint_3d.npz'

CAMERA_IDS = ['cam00', 'cam01', 'cam02', 'cam03', 'cam04', 'cam05', 'cam06']
N_FRAMES = 128
FRAME_OFFSET = 0

DEFAULT_FPS = 30.0
DEFAULT_SCALE = 1 / 6
DEFAULT_OUT = 'dwpose_ransac.mp4'

GRID_COLS = 4
GRID_ROWS = 2

RANSAC_REPROJ_THRESH_PX = 50.0  # must match what ransac_triangulate_sequence used

# DWPose OpenPose-18 body skeleton edges.
BODY_CONNECTIONS = [
    (0, 1), (1, 2), (1, 5),
    (2, 3), (3, 4),
    (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10),
    (1, 11), (11, 12), (12, 13),
    (0, 14), (14, 16), (0, 15), (15, 17),
    (2, 16), (5, 17),
]

N_BODY = 18

# Right-side and left-side joints for color-coding.
_RIGHT_IDX = {2, 3, 4, 8, 9, 10, 16}
_LEFT_IDX = {5, 6, 7, 11, 12, 13, 17}

# BGR colors for inlier cameras.
_COLOR_RIGHT = (50, 50, 220)
_COLOR_LEFT = (50, 200, 50)
_COLOR_CENTER = (50, 210, 210)
_COLOR_LINE_INLIER = (200, 200, 200)

# Grey tones for non-inlier cameras.
_COLOR_NON_INLIER = (110, 110, 110)
_COLOR_LINE_NON_INLIER = (55, 55, 55)

_COLOR_LABEL = (240, 240, 240)

# Cell border colors and thickness (drawn around each camera cell in the grid).
_BORDER_INLIER = (50, 220, 50)   # green  – camera is in RANSAC consensus
_BORDER_OUTLIER = (80, 80, 80)   # grey   – detected but not inlier
_BORDER_THICKNESS = 4


# ---- Inlier computation ------------------------------------------------------

def _reproject(X3d, P):
    """Project 3-D point X3d via projection matrix P → 2-D pixel (no distortion)."""
    h = P @ np.append(X3d, 1.0)
    return h[:2] / h[2]


def compute_inlier_mask(joints3d, valid, keypoints_px, camera_ids,
                         camera_calib, projection_matrices):
    """Determine per-frame per-camera RANSAC inlier status by reprojection.

    A camera is an inlier for frame fi if:
      - valid[fi] is True, AND
      - the undistorted reprojection error of the hip-center seed < RANSAC threshold.

    DWPose seed: midpoint of joints 8 (right_hip) and 11 (left_hip).

    Returns inlier_mask (n_cams, n_frames) bool.
    """
    n_cams = len(camera_ids)
    n_frames = joints3d.shape[0]
    inlier_mask = np.zeros((n_cams, n_frames), dtype=bool)

    for ci, cam_id in enumerate(camera_ids):
        K = camera_calib[cam_id]['K']
        dist = camera_calib[cam_id]['dist']
        P = projection_matrices[cam_id]

        # Batch-undistort the seed hip pixels for this camera.
        hip_r = keypoints_px[ci, :, 8, :]   # (n_frames, 2)
        hip_l = keypoints_px[ci, :, 11, :]  # (n_frames, 2)
        stacked = np.stack([hip_r, hip_l], axis=1).reshape(-1, 1, 2).astype(np.float64)
        undist = cv2.undistortPoints(stacked, K, dist, P=K).reshape(n_frames, 2, 2)
        seed_undist = undist.mean(axis=1)  # (n_frames, 2) undistorted midpoint

        for fi in range(n_frames):
            if not valid[fi]:
                continue
            hip3d = (joints3d[fi, 8] + joints3d[fi, 11]) / 2.0
            if np.any(np.isnan(hip3d)):
                continue
            proj = _reproject(hip3d, P)
            err = float(np.linalg.norm(proj - seed_undist[fi]))
            if err < RANSAC_REPROJ_THRESH_PX:
                inlier_mask[ci, fi] = True

    return inlier_mask


# ---- Drawing -----------------------------------------------------------------

def _joint_color(idx):
    if idx in _RIGHT_IDX:
        return _COLOR_RIGHT
    if idx in _LEFT_IDX:
        return _COLOR_LEFT
    return _COLOR_CENTER


def draw_dwpose(img, keypoints, scores, is_inlier, r_min, r_max, line_thickness=2):
    """Overlay DWPose body keypoints on `img` in-place.

    keypoints: (N_BODY, 2) pixel coords.  scores: (N_BODY,) confidences [0, 1].
    is_inlier: bool – True  → normal right/left/center colors.
                      False → grey.
    r_min, r_max: radius range driven by confidence.
    """
    h, w = img.shape[:2]
    pts = {}
    radii = {}
    for i, ((x, y), s) in enumerate(zip(keypoints, scores)):
        xi = int(round(float(np.clip(x, 0, w - 1))))
        yi = int(round(float(np.clip(y, 0, h - 1))))
        pts[i] = (xi, yi)
        radii[i] = max(1, int(round(r_min + (r_max - r_min) * float(np.clip(s, 0, 1)))))

    line_color = _COLOR_LINE_INLIER if is_inlier else _COLOR_LINE_NON_INLIER
    for a, b in BODY_CONNECTIONS:
        if a in pts and b in pts:
            cv2.line(img, pts[a], pts[b], line_color, line_thickness, cv2.LINE_AA)

    for i, p in pts.items():
        color = _joint_color(i) if is_inlier else _COLOR_NON_INLIER
        r = radii[i]
        cv2.circle(img, p, r, color, -1, cv2.LINE_AA)
        cv2.circle(img, p, r, (255, 255, 255), 1, cv2.LINE_AA)


def add_label(img, text, scale=1.0):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.7 * scale
    thickness = max(1, int(2 * scale))
    (tw, th), baseline = cv2.getTextSize(text, font, fs, thickness)
    pad = max(4, int(6 * scale))
    cv2.rectangle(img, (pad, pad), (pad + tw + 4, pad + th + baseline + 4), (20, 20, 20), -1)
    cv2.putText(img, text, (pad + 2, pad + th + 2), font, fs, _COLOR_LABEL, thickness, cv2.LINE_AA)


def draw_border(grid, row, col, cell_h, cell_w, color, thickness):
    y0 = row * cell_h
    x0 = col * cell_w
    cv2.rectangle(grid, (x0, y0), (x0 + cell_w - 1, y0 + cell_h - 1),
                  color, thickness)


# ---- Main --------------------------------------------------------------------

def make_grid_video(out_path, scale, fps, verbose=True):
    # Load pre-computed DWPose keypoints.
    d2 = np.load(DWPOSE_NPZ)
    keypoints_px = d2['keypoints_px']    # (n_cams, n_frames, 18, 2)
    keypoints_conf = d2['keypoints_conf'] # (n_cams, n_frames, 18)
    presence = d2['presence']             # (n_cams, n_frames)

    # Load triangulation outputs.
    d3 = np.load(KEYPOINTS_3D_NPZ)
    joints3d = d3['joints3d']   # (n_frames, 18, 3)
    valid = d3['valid']          # (n_frames,)

    # Load calibration.
    camera_calib, camera_world_transforms, projection_matrices, camera_ids_cal = \
        load_calibration(H5_PATH)

    if verbose:
        print('Computing RANSAC inlier masks ...')
    inlier_mask = compute_inlier_mask(
        joints3d, valid, keypoints_px, CAMERA_IDS,
        camera_calib, projection_matrices)

    n_inlier_frames = int(inlier_mask.any(axis=0).sum())
    if verbose:
        print(f'  {n_inlier_frames}/{N_FRAMES} frames have at least one inlier camera')
        for ci, cam_id in enumerate(CAMERA_IDS):
            print(f'  {cam_id}: inlier in {int(inlier_mask[ci].sum())}/{N_FRAMES} frames')

    # Reference frame for dimensions.
    with h5py.File(H5_PATH, 'r') as hf:
        ref = cv2.imdecode(hf['rec/cam00/frames/000000/color'][:], cv2.IMREAD_COLOR)
    orig_h, orig_w = ref.shape[:2]

    cell_w = int(round(orig_w * scale))
    cell_h = int(round(orig_h * scale))
    cell_w += cell_w % 2
    cell_h += cell_h % 2

    grid_w = GRID_COLS * cell_w
    grid_h = GRID_ROWS * cell_h

    # Radius bounds for confidence-based scaling (at the CELL resolution).
    # label_scale is the scale factor used for the text; ~1.0 at scale=1/6.
    label_scale = scale * 6
    r_min = max(1, int(round(3 * label_scale)))
    r_max = max(r_min + 2, int(round(10 * label_scale)))
    line_thickness = max(1, int(round(2 * label_scale)))

    if verbose:
        print(f'grid {grid_w}x{grid_h}  cell {cell_w}x{cell_h}  '
              f'fps={fps}  radius={r_min}..{r_max}px')
        print(f'writing {N_FRAMES} frames to {out_path!r} ...')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (grid_w, grid_h))
    if not writer.isOpened():
        raise RuntimeError(f'Could not open VideoWriter for {out_path!r}')

    with h5py.File(H5_PATH, 'r') as hf:
        for fi in range(N_FRAMES):
            frame_key = f'{FRAME_OFFSET + fi:06d}'
            grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

            for ci, cam_id in enumerate(CAMERA_IDS):
                raw = hf[f'rec/{cam_id}/frames/{frame_key}/color'][:]
                img = cv2.imdecode(raw, cv2.IMREAD_COLOR)

                has_detection = bool(presence[ci, fi])
                is_inlier = bool(inlier_mask[ci, fi])

                if has_detection:
                    kps = keypoints_px[ci, fi]    # (18, 2)
                    scores = keypoints_conf[ci, fi] # (18,)
                    draw_dwpose(img, kps, scores, is_inlier,
                                r_min=r_min, r_max=r_max,
                                line_thickness=line_thickness)

                inlier_tag = ' [IN]' if is_inlier else ''
                add_label(img, f'{cam_id}  f{frame_key}{inlier_tag}', scale=label_scale)

                cell_img = cv2.resize(img, (cell_w, cell_h), interpolation=cv2.INTER_AREA)

                row, col = divmod(ci, GRID_COLS)
                y0, x0 = row * cell_h, col * cell_w
                grid[y0:y0 + cell_h, x0:x0 + cell_w] = cell_img

                # Cell border: green = inlier, grey = detected-but-not-inlier.
                if is_inlier:
                    draw_border(grid, row, col, cell_h, cell_w, _BORDER_INLIER, _BORDER_THICKNESS)
                elif has_detection:
                    draw_border(grid, row, col, cell_h, cell_w, _BORDER_OUTLIER, _BORDER_THICKNESS)

            writer.write(grid)

            if verbose and (fi % 16 == 0 or fi == N_FRAMES - 1):
                print(f'  frame {fi + 1}/{N_FRAMES}')

    writer.release()
    if verbose:
        print('done.')


def main():
    parser = argparse.ArgumentParser(description='DWPose RANSAC grid video')
    parser.add_argument('--out', default=DEFAULT_OUT)
    parser.add_argument('--scale', type=float, default=DEFAULT_SCALE)
    parser.add_argument('--fps', type=float, default=DEFAULT_FPS)
    args = parser.parse_args()
    make_grid_video(args.out, args.scale, args.fps)


if __name__ == '__main__':
    main()
