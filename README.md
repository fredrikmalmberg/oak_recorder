# oak_recorder

Multi-camera OAK rig for capturing hand/sign-language footage, plus the
tooling to calibrate that rig and extract/reconstruct hand and body pose from
its recordings.

The pipeline has two independent branches after capture:

- **Calibration**: live (`calibrate.py`, with a viser viewer) or offline from
  already-captured images (`calibrate_charuco.py` + `compute_extrinsics.py`).
- **Hand/pose reconstruction**: live (`hand_capture_live.py`) or
  exploratory/offline, either on the ground-truth-calibrated h5 rig
  (`dwpose_vs_mediapipe_h5.ipynb`) or the separate, uncalibrated cam0-cam3 rig
  (`multiview_hand_triangulation.ipynb`).

A few notebooks on disk are earlier prototypes superseded by the ones above
(`h5_explore.ipynb`; `hand_triangulation.ipynb` and `triangulation.ipynb`,
merged into `multiview_hand_triangulation.ipynb`) and are not documented as
their own entry points here. A separate MANO/WiLoR dense-mesh exploration
(`ransac_keypoint_reconstruction.ipynb`, `wilor_hand_reconstruction.ipynb`,
`notes_on_wilor_implementation.md`) is also on disk but isn't runnable as
committed — see [Setup / requirements](#setup--requirements).

## Capture — `capture.py`

```
python capture.py [flags]
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `-i, --iso` | 200 | ISO |
| `-s, --shutter` | 10000 | Shutter speed, microseconds |
| `-f, --fps` | 30 | Frames per second |
| `--no-preview` | off | Disable the live preview window |
| `--no-record` | off | Disable saving MJPEG/MP4 |
| `--no-warmup` | off | Skip the pre-recording warmup frame discard |
| `--warmup-frames` | `DEFAULT_WARMUP_FRAMES` | Frames to discard per camera during warmup |
| `--no-align` | off | Skip post-capture timestamp alignment |
| `--align-threshold-ms` | half a frame period | Max timestamp offset accepted as a match during alignment |
| `--align-host` | off | Align on raw host receive timestamps instead of the device-synced clock |
| `--align-raw` | off | Skip alignment entirely; build the grid MP4 from raw MJPEG. Mutually exclusive with `--align-host` |
| `--output-mp4` | none | `small` (one downscaled grid MP4) or `actual` (full-res MP4 per camera) |

**Output**: a session folder `recordings/sign_capture_<timestamp>_iso<ISO>.../`
containing `video_cam*.mjpeg` + `frame_timestamps_cam*.log` per camera. If
alignment runs (on by default), also `aligned/camN/*.jpg` +
`alignment_report.json`. If `--output-mp4` is set, also
`compressed_video_grid.mp4` (small) or one full-resolution MP4 per camera
(actual). No viser viewer — this is a recording tool.

### Re-running alignment standalone — `align_session.py`

`capture.py` calls this automatically after recording, but it also has its own
CLI for re-aligning an existing session (e.g. with a different threshold)
without re-recording:

```
python align_session.py <session_dir> [--align-threshold-ms MS] [--fps FPS]
                         [--output-mp4 small] [--align-host] [--align-raw]
```

Writes the same `aligned/camN/*.jpg` + `alignment_report.json` output as
above (the report covers per-camera matched/discarded/missing frame counts,
cross-camera timestamp spread, and warnings such as drift trend or a
log/video frame-count mismatch). `--output-mp4 small` additionally writes
`compressed_video_grid.mp4`. No viser viewer.

## Live camera preview/monitor — `oak_camera.py`

```
python oak_camera.py
```

No CLI arguments. Discovers every connected OAK device and opens a dearpygui
window with one toggleable RGB panel per camera (plus a depth panel for
OAK-D stereo units); press `1`-`9` to turn a camera on/off. Unrelated to
`capture.py`'s pipeline — no recording, no shared code, just a live visual
sanity check. Writes no files; opens a dearpygui window, not a viser viewer.

## Frame extraction utility — `extract_frames.py`

```
python extract_frames.py <video_path> [-s START_IDX] [-m MAX_FRAMES]
                          [-f {png,jpg,jpeg}] [-q QUALITY] [-b BRIGHTNESS]
                          [-c CONTRAST]
```

Takes one `.mjpeg` file (e.g. from a `capture.py` session) and writes its
frames as individual images to `<video_dir>/<video_basename>/`. No viser
viewer.

## Live multi-camera calibration — `calibrate.py`

```
python calibrate.py [--config calibrate_config.yaml] [--duration N]
```

Needs OAK hardware attached (`depthai`). `calibrate_config.yaml` isn't
tracked in git — if it's missing, `calibrate.py` writes it with built-in
defaults on first run. Board spec, quality-gate thresholds, convergence
criteria, etc. all live in that file.

While running, type commands at the terminal: `save`, `reset <camN|all>`,
`quit`. It also auto-stops once every camera has converged, or after
`--duration` seconds if given.

**Opens a live viser viewer** at `http://localhost:<port>` (printed on
startup): camera frustums (greyed out until a pose is solved), live detected
board pose, a per-camera status panel (reprojection error, sample count,
coverage %, converged yes/no), and a camera settings panel.

**Output**: JSON written to `<output.dir>/<timestamp>_<N>cam.json`
(intrinsics + extrinsics + convergence report), written on `save`, `quit`,
or auto-stop.

> Recent fix: `calibrate.py`'s board-to-world "up" alignment was inverted —
> solvePnP's local Z axis for a planar target points *into* the board, not
> out toward the camera, so a correction (`_BOARD_UP_CORRECTION`) now flips
> local Z/Y before aligning to world +Z.

## Live hand tracking — `hand_capture_live.py`

```
python hand_capture_live.py [--config calibrate_config.yaml] [--duration N]
```

Needs OAK hardware. Reuses `calibrate.py`'s camera-session and calibration
code and `hand_multiview.py`'s triangulation helpers directly rather than
reimplementing them.

**Opens a live viser viewer.** On startup you choose to either load a
previously-saved `calibrate.py` calibration JSON, or run calibration live in
the same app. Once at least 2 cameras have a resolved pose, a "Start hand
tracking" button freezes every camera's pose (no per-frame re-estimation for
the rest of the session) and switches to live MediaPipe hand-landmark
triangulation, rendered as a 3D skeleton in the same viewer. No file output —
this is a live-viewing tool, not an export tool.

## Offline per-camera intrinsics — `calibrate_charuco.py`

```
python calibrate_charuco.py --images <folder> --squares-x N --squares-y N \
    --square-size S --marker-size S [--out DIR] [--aruco_dict NAME] \
    [--min-corners N] [--min-views N] [--use-intrinsic-guess]
```

Runs on a folder of already-captured images from **one** camera — no live
camera or OAK hardware needed. Writes `<out>/calibration_result.json` (camera
matrix, distortion coefficients, RMSE, board spec, list of images used). No
viser viewer.

## Offline extrinsics + rig viewer — `compute_extrinsics.py`

```
python compute_extrinsics.py --session <dir> \
    --calib cam0:calib0/calibration_result.json cam1:calib1/calibration_result.json ... \
    [--ref-cam cam0] [--min-corners N] [--out extrinsics.json] \
    [--no-viewer] [--units-per-meter 1000]
```

Reads `<session>/aligned/camN/*.jpg` — the layout `align_session.py` writes.
Solves each camera's board pose per frame from the intrinsics passed via
`--calib`, then averages relative transforms back to `--ref-cam` across every
frame the cameras share.

**Output**: `extrinsics.json` (R/t per camera relative to `--ref-cam`) is
always written. **Also opens a viser viewer** — camera frustums plus one
sampled frame's board corners as a sanity check — unless `--no-viewer` is
passed. Unlike `calibrate.py`'s viewer this one is a one-shot display of an
already-computed result, not a live-updating view.

`camera_positions.png` is a reference screenshot of this viewer's output.
`multicam_charuco_calibration_spec.md` is the design spec this pair of
scripts (`calibrate_charuco.py` + `compute_extrinsics.py`) implements a first
offline pass of — the fuller live/pose-graph version of that same spec is
`calibrate.py`.

## MediaPipe vs. DWPose comparison — `dwpose_vs_mediapipe_h5.ipynb`

Open in Jupyter and run top-to-bottom. Requires `tmp/Testdata.h5` and
DWPose's ONNX weights under `models/` (see [Setup](#setup--requirements)).

This is the entry point for a small library of modules with no CLI of their
own, used entirely from here:

- **`h5_dataset.py`** — reads video frames from `tmp/Testdata.h5`.
- **`dwpose_onnx.py`** — `DWposeOnnx(det_model_path, pose_model_path)` wraps
  the DWPose ONNX models. Needs `yolox_l.onnx` + `dw-ll_ucoco_384.onnx`, which
  aren't included in the repo.
- **`h5_hand_extraction.py`** — `extract_mediapipe_hands_from_h5(...)` /
  `extract_dwpose_hands_from_h5(...)` run each detector over the h5 file's
  frames, caching results to
  `tmp/h5_extraction_cache/{mediapipe,dwpose}_hands.json`
  (pass `force=True` to bypass the cache).
- **`hand_multiview.py`** — the reconstruction/correction toolkit: DLT/RANSAC
  triangulation, Kalman/RTS smoothing, jitter flagging, the three mixup/swap
  detectors (`detect_body_wrist_hand_swap`, `detect_camera_hand_swap`,
  `cross_detector_hand_disagreement`), `detect_high_velocity_frames`, and the
  correction functions `apply_swap_corrections`/`apply_confidence_penalty`.
  No file output of its own — pure functions called from the notebook.
- **`grid_video.py`** — `render_hand_grid_video(...)` composites the grid
  videos below, with anchor-rejection wrist tracking (a new detection only
  becomes the tracked anchor if it falls inside the growing confidence
  circle around the last known position; otherwise the old anchor is kept).

The notebook runs a three-stage flip-correction pipeline on top of the above:
body-wrist proximity, then a leave-one-out cross-camera swap test re-run on
that output, then a velocity-based confidence penalty.

**Outputs**:
- Grid videos: `mediapipe_hand_grid.mp4`, `dwpose_hand_grid.mp4` (raw, with
  detector-flag overlays) and `mediapipe_hand_grid_corrected.mp4`,
  `dwpose_hand_grid_corrected.mp4` (after the 3-stage correction pipeline).
- JSON extraction caches under `tmp/h5_extraction_cache/`.
- **A viser skeleton viewer** (`http://localhost:<port>`) for visually
  comparing detector variants in 3D.

This notebook supersedes an earlier, ad hoc prototype in `h5_explore.ipynb`
(kept only as informal reference — `h5_dataset.py`'s access pattern and
`grid_video.py`'s grid layout were both lifted from it).

## Multi-view hand triangulation (cam0-cam3 rig) — `multiview_hand_triangulation.ipynb`

Open in Jupyter and run top-to-bottom; imports `hand_multiview.py`. Targets
the separate cam0-cam3 recording rig, **not** the h5 ground-truth rig above.

**Known limitation**: this rig has no real calibration — its intrinsics are
averaged from the h5 rig's calibration, so reprojection error is large. This
is a limitation of the rig's data, not something this README resolves.

It merges and supersedes two earlier notebooks, `hand_triangulation.ipynb`
and `triangulation.ipynb` (per its own intro cell) — those are left on disk
but aren't documented separately here.

## Setup / requirements

- `conda env create -f environment.yml` (environment name `oak_env`).
- OAK cameras + `depthai` are required for `capture.py`, `oak_camera.py`,
  `calibrate.py`, and `hand_capture_live.py`.
- `tmp/Testdata.h5` and DWPose's ONNX weights under `models/` are required
  for `dwpose_vs_mediapipe_h5.ipynb`.
- Gitignored, not shipped in this repo: `recordings/`, `models/`, `*.h5`,
  `calibrate_config.yaml` (self-generated on first `calibrate.py` run), and
  all generated JSON/MP4 outputs described above.
- `ransac_keypoint_reconstruction.ipynb`, `wilor_hand_reconstruction.ipynb`,
  and `notes_on_wilor_implementation.md` explored a MANO/WiLoR dense
  hand-mesh reconstruction as an alternative to the keypoint-triangulation
  notebooks above. They hardcode paths to an external `WiLoR` checkout and
  `/tmp/*` scripts that aren't part of this repo, so they aren't runnable as
  committed — kept for reference only.
