# Notes on WiLoR implementation for multi-view hand reconstruction

High-level notes from validating the per-camera WiLoR pipeline before building the
v2 multi-view fusion (`wilor_hand_reconstruction_v2.ipynb`, source of truth
`/tmp/build_v2_notebook.py`). Goal of this doc: record *why* the rendering looks the
way it does, so future work doesn't re-derive this from scratch.

## Two unrelated jobs, two different camera conventions

WiLoR's network output (`pred_cam`, `pred_vertices`, `pred_keypoints_3d`) is a
weak-perspective crop-relative prediction. Turning it into a 3D point in *some*
camera's frame requires picking a focal length and principal point. We need this for
two very different purposes, and they pull in opposite directions.

### A. WiLoR demo convention -- for visualization

`demo.py` / `cam_crop_to_full` uses:
- a "virtual" focal length `scaled_focal_length = FOCAL_LENGTH / MODEL.IMAGE_SIZE *
  img_size.max()` (~65,625 for our 3360px-wide frames -- about 32x our real
  calibrated `fx` ~2000),
- principal point = image center (not the calibrated `(cx, cy)`),
- rendered with `Renderer.render_rgba_multiple` onto the **raw (distorted)** image.

Because `tz` ends up ~32x larger than the real metric depth, the hand's own depth
extent (~5-10cm) is negligible relative to `tz` -- the projection is effectively
**near-orthographic**. This hides any noise in WiLoR's per-vertex depth predictions.
Running `demo.py` unmodified on our footage (cam00/cam03/cam04) looked excellent: both
hands tracked, no jitter or floating.

**This convention is not metric** -- `cam_t` does not correspond to the hand's real
position relative to the camera, so it cannot be used for multi-view triangulation.

### B. Calibrated-K convention -- for geometry / triangulation

`run_wilor_on_image` (in `build_v2_notebook.py`) uses the real calibrated `fx, cx, cy`
for `cam_t`, giving a true metric depth (~1.5-2m). This is **required** for the v2
pipeline's Step 5 (triangulating each MANO joint across all 7 cameras via
`CAMERA_WORLD_TRANSFORMS`) -- the camera-frame 3D position of each joint must be a
real metric quantity for that to work at all.

The cost: at a real ~1.5-2m depth, the hand's own ~5-10cm depth extent (3-7% of `tz`)
causes real perspective foreshortening/magnification. Any z-prediction noise from
WiLoR -- invisible under convention A's near-orthographic projection -- becomes
visible as size/shape distortion (rendered hand looks larger / more splayed than the
real hand).

### C. Raw vs. undistorted background (position offset)

`run_wilor_on_image`'s `cam_t` is built so that `cv2.projectPoints(verts_cam, K,
dist=None)` (pinhole) lands at `box_center` -- a pixel coordinate measured on the
**raw/distorted** image (from MediaPipe/YOLO detection). Rendering that mesh onto an
**undistorted** (`cv2.undistort`'d) background introduces a position offset equal to
the local lens-distortion shift at that pixel: the real hand moves to its
pinhole-projected location in the undistorted image, but the mesh stays anchored at
the raw-pixel `box_center` value.

Measured for cam03 (`cv2.undistortPoints` at the actual detection points):
- hand near the principal point (~550-730px away): offset ~3-9px at half-res render
  (1-2% of crop width) -- small, visually indistinguishable in side-by-side crops.
- generic grid samples further out (~1300-1900px from principal point, near image
  edges): offset estimated up to ~150px.

Conclusion: this offset is real and grows with distance from the principal point, but
for these two examples it was *not* the dominant source of "mesh looks misplaced" --
that's still mostly convention B's size/perspective effect (section B above).
**Always render calibK-based overlays onto the raw/distorted background** -- it's
free, strictly more correct, and matches both `demo.py`'s convention and
`run_wilor_on_image`'s documented convention.

## Chosen direction

Keep convention A and B as two **separate** computations for two separate purposes,
rather than trying to make one convention serve both jobs:

1. **Per-camera visual sanity check** ("does WiLoR's raw per-camera prediction look
   right?"): use convention A as-is (demo's scaled focal length, image-center
   principal point, raw background, `Renderer.render_rgba_multiple`). This is a
   diagnostic only -- its output is not consumed by the fusion pipeline.
2. **Triangulation input** (Step 5 of the v2 plan): use convention B
   (`run_wilor_on_image`'s calibrated-K `cam_t` / `joints_cam`). The size/shape
   distortion visible when *rendering* convention B does not necessarily mean the
   *joint positions* fed to triangulation are bad -- triangulating across 7 cameras
   plus a single MANO refit (Step 6) should average out single-camera z-noise. To be
   checked empirically once Steps 5-6 are implemented.

## Per-camera detection union, tracking, and handedness resolution

`/tmp/diag_percam7_robust_render.py` builds a robust per-camera pipeline on top of
the convention-A render (`diag_ablation_detector.py`'s `render_overlay`, split here
into `run_wilor_batch` + `composite_meshes` so different hands can be rendered with
different mesh colors).

- **Detection union**: per frame, run both YOLO (capped to top-2 boxes by area) and
  MediaPipe (`hand_boxes_for`). Match boxes by IoU (`IOU_THRESH=0.2`, permissive since
  YOLO boxes run larger). For a matched pair, use YOLO's box geometry with both
  detectors' handedness as votes (`sources='YM'`); unmatched boxes keep their own
  detector's box and single vote (`sources='Y'` or `'M'`).
- **Tracking**: greedy nearest-neighbor matching of detection box centers across
  frames (`DIST_THRESH=500px`), allowing gaps up to `MAX_GAP=20` frames so a track
  can bridge missed detections. Tracks are sorted by number of present frames and the
  top 2 are kept (single-subject assumption: one left + one right hand).
- **Handedness resolution**: each kept track's final L/R label is a majority vote
  over all `(value, source)` votes collected from its frames, tie-broken by
  MediaPipe's votes.

### Occlusion handling: hands passing in front of each other

Found in cam00 frame 89: the right hand passes in front of the left hand, and for
that frame only **one** detection exists (covering both hands' 2D region) — the
forward/backward gap-bridging in tracking doesn't catch this because the *present*
track's per-frame vote can simply be wrong for that one frame (no track-assignment
flip is needed for the bug to occur).

Fix (`flag_occlusions`, run after `build_tracks`/`kept` selection, before
`resolve_handedness`): for each frame where exactly one of the 2 kept tracks has a
detection and the other is absent (gap), the present detection is flagged
occlusion-ambiguous if **either**:

1. its box overlaps the absent track's temporally-nearest box (`nearest_box`, looks
   forward/backward), `IoU > OCCLUSION_IOU_THRESH = 0.1`; or
2. it's a single merged YOLO+MediaPipe detection (`sources=='YM'`) where the two
   detectors disagree on handedness - itself evidence the box covers both hands.

Signal 2 was added after checking why cam00 f89 (idx=89, fi=49) wasn't initially
flagged by signal 1 alone: at fi=48 the merged box overlaps track1's last box (fi=47)
with IoU=0.322 (flagged), but at fi=49 the same comparison gives IoU=0.091 - just
under the 0.1 threshold, because track1's reference box is now 2 frames stale and the
merged box has moved on. In both fi=48 and fi=49, however, the single merged
detection has YOLO voting right (1.0) and MediaPipe voting left (0.0) for the *same*
box - signal 2 catches this directly without relying on the absent track's
increasingly-stale last position.

Once flagged:
- The vote(s) are downweighted by `OCCLUSION_CONF = 0.5` (a `conf` factor, tunable)
  when accumulated into the track's majority vote, so ambiguous frames don't skew the
  track's overall L/R label.
- The rendered hand mesh for that frame is colored **red** (instead of the usual
  light purple) and annotated `OCC`, so the ambiguity is visible in the video.

Result on the 7-camera test set (frames 40-89), with both signals active:
- cam00 track 0 (L, present 50/50): 10/50 frames flagged - exactly the 10 frames
  where track1 (R) has a gap.
- cam01 track 0 (L, present 50/50): 7/50 flagged, out of track1's (R) 13 gap frames.
- cam02-cam06: 0 flagged (these tracks have no gaps at all).

**Open items**:
- This only covers the "1 present + 1 absent" case for a given frame. It does not yet
  handle a frame where *both* kept tracks have a detection but the two detections are
  actually swapped (each hand's detection assigned to the other's track) during an
  occlusion event.
- cam01 has 6 of track1's 13 gap frames *not* flagged by either signal - i.e. frames
  where track1 has no detection and track0's detection neither overlaps track1's last
  box nor shows a Y/M handedness disagreement.

## Open items

- "Target the missed detections" (MediaPipe vs. YOLO hand detection coverage) -- not
  yet started.

## v3: pure-geometric RANSAC triangulation of MediaPipe keypoints (no WiLoR/MANO)

`ransac_keypoint_reconstruction.ipynb` (source `/tmp/build_ransac_keypoint_notebook.py`,
runs entirely in `oak_env`) sidesteps WiLoR/MANO altogether: it reuses the per-camera
detection/tracking/occlusion pipeline above (now also carrying each detection's
MediaPipe 21-landmark array, `det['mp_landmarks']`), then triangulates each of the 21
hand landmarks directly via weighted multi-view DLT, with RANSAC over camera pairs
(on the wrist joint) to pick the inlier camera set per frame/hand. This answers the
"Open items" question above about per-camera z-noise -- now reframed in terms of 2D
reprojection error, which can be measured directly without a render.

Results on frames 40-89 (50 frames), all 7 cameras, `RANSAC_REPROJ_THRESH_PX=50.0`:

- **left hand**: 46/50 frames valid; inlier cameras min=2, mean=2.76, max=3; mean
  reprojection error 15.6px.
- **right hand**: 42/50 frames valid; inlier cameras min=2, mean=2.33, max=3; mean
  reprojection error 23.6px.
- **wrist RANSAC residual** (the value being thresholded): mean=19.7px,
  median=17.8px, max=48.4px.

Takeaways:
- The wrist residual distribution (median ~18px, max ~48px) sits comfortably under
  the 50px threshold -- no tuning needed for this clip, but the max is close enough
  to the threshold that a noisier clip could start losing inlier pairs at 50px.
- Inlier counts maxing out at 3 (out of up to 7 cameras with `presence=True`) shows
  the RANSAC step is doing real filtering work -- most cameras' keypoints for a given
  frame/hand don't agree to within 50px, consistent with the per-camera z-noise
  suspected in convention B above, except here it shows up as 2D pixel disagreement
  rather than mesh size/shape distortion.
- The 4/50 (left) and 8/50 (right) invalid frames are frames where `<2` cameras had
  MediaPipe landmarks for that hand at all (see per-camera `mp_landmarks` coverage,
  e.g. cam01 hand=R only has landmarks on 10/50 frames) -- a detection-coverage gap,
  not a triangulation failure.

### v3 package refactor (source of truth)

The hand pipeline above has been extracted into a reusable `v3/` package
(`v3/calibration.py`, `v3/keypoints_io.py`, `v3/hand_detection.py`,
`v3/triangulation.py`, `v3/smoothing.py`, `v3/skeletons.py`,
`v3/visualization.py`), generalized over a leading "instance" axis (empty for
single-instance body keypoints, `(2,)` for left/right hands) and over
`n_joints` (21 for hands, 33 for MediaPipe Pose body landmarks).
`ransac_keypoint_reconstruction.ipynb` is now regenerated from
`v3/notebooks/build_hand_notebook.py` (`python -m v3.notebooks.build_hand_notebook`);
the old `/tmp/build_ransac_keypoint_notebook.py` is no longer used. The v3
port was verified bit-for-bit equivalent to the original source-string code
on the cached `ransac_keypoint_detections.npz` (detection arrays and all
triangulation/smoothing/jitter/SG outputs).

One pre-existing discrepancy noted during verification: re-running the
*original* (pre-v3) code on the current `ransac_keypoint_detections.npz`
gives right-hand inlier-camera mean=2.36 (vs. 2.33 above) and mean
reprojection error 24.3px (vs. 23.6px above), with wrist-residual
median≈17.7px (vs. ≈17.8px). Since the original code reproduces the same
numbers as v3 on this npz, the discrepancy is a stale-cache artifact (the npz
was regenerated at some point after the numbers above were written), not a
v3 porting bug.

## v3: body keypoint triangulation (frames 0-127)

`body_keypoint_reconstruction.ipynb` (source `v3/notebooks/build_body_notebook.py`)
applies the same RANSAC-triangulation + Kalman/RTS-smoothing + Savitzky-Golay/2x
pipeline to the 33-landmark MediaPipe Pose body keypoints from
`extracted_2d_keypoints.json`'s `'body'` field, for the full 128-frame range
(0-127), single instance (no left/right split). The RANSAC seed point is the
hip-center (pixel-space midpoint of `left_hip`=23 and `right_hip`=24),
analogous to SMPL-X's pelvis joint.

**Presence**: cam00 is missing body landmarks on 4/128 frames (49, 85, 87, 88);
all other cameras have 128/128. This matches the plan's expectation and never
drops a frame below 6 cameras, so RANSAC is unaffected.

**Threshold choice**: kept `RANSAC_REPROJ_THRESH_PX=50.0`, same as the hand
pipeline -- the empirical-check criteria from the plan (frames with `<2`
inliers, or residuals close to/over 50px) don't trigger: all 128/128 frames
are valid, and the hip-center residual distribution (mean=17.6px,
median=17.0px, max=47.2px) is essentially identical to the hands' wrist
residuals (mean=19.7px, median=17.8px, max=48.4px) and sits comfortably under
50px. (For reference, 75px and 100px thresholds were also tried: both still
give 128/128 valid with more inlier cameras (mean 2.44 and 3.09 respectively)
but push the hip-center residual mean to 28.0px / 48.4px -- the 100px residual
mean is right at the hands' threshold, so 50px was kept.)

**Results** (`RANSAC_REPROJ_THRESH_PX=50.0`):
- 128/128 frames valid; inlier cameras min=2, mean=2.12, max=3 (histogram:
  112 frames with 2 inliers, 16 with 3).
- mean reprojection error across all 33 joints: 52.5px (higher than the
  hip-center's own 17.6px mean -- limb/extremity landmarks disagree across
  cameras more than the central hip does; this is informational, not a
  validity criterion).
- hip-center RANSAC residual (px): mean=17.6, median=17.0, max=47.2.
- hysteresis kept the previous frame's camera set on 27/128 valid frames.

**Smoothing** (`min_inliers_base=3`, same as hands -- n_inliers is in {2,3}
here too, so 112/128 frames get the larger 2-camera measurement uncertainty):
- Kalman/RTS correction on hip-center: mean=2.10cm, median=0.76cm,
  max=9.98cm.
- jitter flagging (raw vs. smoothed hip-center, MAD factor 4.0): 36/128
  frames flagged (median residual 0.76cm, threshold 2.81cm, max 9.98cm).
- Savitzky-Golay correction on hip-center: mean=0.865cm, max=4.914cm;
  255/255 frames valid at 2x rate.

These corrections/jitter rates are noticeably larger than the hands' (Kalman
mean 0.15-1.62cm, 0/46-2/42 jitter-flagged), which is expected: the hip-center
is a lower-precision MediaPipe landmark (low-texture, often clothing-occluded)
than the wrist, and full-body motion over 128 frames covers much larger
excursions than the 50-frame hand clip. No retuning was done in response to
this -- it reflects real data characteristics, not a pipeline issue.

Saves `body_keypoint_3d.npz` (gitignored via the repo's blanket `*.npz` rule)
with `joints3d`/`valid`/`n_inliers`/`reproj_err` (128-frame rate) and their
smoothed/SG/2x counterparts, for Phase 3's SMPL-X fit.

## v3: SMPL-X body-only fit (smplx_body_fit.ipynb)

`smplx_body_fit.ipynb` (source `v3/notebooks/build_smplx_notebook.py`) fits
an SMPL-X body model to the triangulated body joints from `body_keypoint_3d.npz`.

**Setup**: `smplx.create(model_path='/home/fmalmb/CODE/smplx_debug',
model_type='smplx', gender='neutral', use_pca=False, flat_hand_mean=True,
num_betas=10, batch_size=1)` — neutral gender, body-only (no hand/face pose
optimized), `output.joints[0:22]` used for correspondence.

**Joint correspondences** (`v3.smplx_body_fit.JOINT_CORRESPONDENCES`, 15 entries):
- Direct pairs, weight 1.0: shoulders (MP 11→SMPLX 16, 12→17), elbows
  (13→18, 14→19), wrists (15→20, 16→21), hips (23→1, 24→2), knees (25→4,
  26→5), ankles (27→7, 28→8).
- Synthetic midpoint, weight 1.0: pelvis from (hip_L, hip_R) midpoint → SMPLX 0.
- Synthetic midpoint, weight 0.5: neck from (shoulder_L, shoulder_R) midpoint → SMPLX 12.
- Direct, weight 0.3: head from nose (0) → SMPLX 15.

**Optimizer**: Adam, lr=0.05, N_ITERS=300, w_betas=1e-2, w_pose=1e-3,
init transl = pelvis target (calibration-world coordinates). Independent fit per
frame (no warm-starting).

**Representative frames**: 5 frames drawn from those with max n_inliers (==3)
and no jitter-flag: [46, 61, 64, 74, 86].

**Results** (per-joint residuals averaged over 5 frames):

| Joint          | Residual (mean mm, 5 frames)     |
|----------------|----------------------------------|
| wrists         | ~5-19mm (smallest)               |
| elbows/knees   | ~9-37mm                          |
| shoulders      | ~31-56mm                         |
| ankles         | ~10-93mm (variable)              |
| hips           | ~67-129mm (systematically large) |
| pelvis         | ~71-102mm                        |
| neck/head      | ~14-58mm                         |

Mean residual across frames: ~43-46mm.

**Interpretation**: Wrist/elbow/knee residuals are small-moderate (5-37mm),
consistent with the optimizer having degrees of freedom (body_pose) to match
those joints closely. Hip and pelvis residuals are large (67-129mm) by design
— MediaPipe hip landmarks are surface points (greater trochanter area) while
SMPL-X hip/pelvis joints are internal skeletal joints; this geometric offset
(~10cm) is expected and is NOT a correspondence-table bug (it's symmetric
between left/right and consistent across frames). Ankle variability reflects
foot/ankle pose being harder to constrain from the limited number of landmarks
and ankle-joint geometry differences (MediaPipe ankle ≠ SMPL-X ankle joint
center).

The optimizer runs forward passes through the smplx package installed at
`~/.local/lib/python3.10/site-packages/smplx/`, which has a leftover debug
`print(betas.device, shapedirs.device, full_pose.device)` call on every
forward pass (body_models.py:1230). `v3.smplx_body_fit.fit_frame` suppresses
this via `contextlib.redirect_stdout` to keep the notebook output clean.
