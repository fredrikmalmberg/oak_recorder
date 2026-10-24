# From RGB frames to a fitted SMPL-X sequence (Sapiens triangulation + SMPL-X fit)

This document describes the pipeline as it stands, with the default settings, every optimisation stage with its weights, and every special solution we use and why. The run on **`20261022_171905/take_2`** (60 fps, 8 cameras, 2016x1520, half-resolution calibration, 946 frames) gave a good result and is the **proposed pipeline for now**; section 8 lists the exact commands.

Related documents: `README.md` ("Full offline pipeline"), `pose2d/HIP_STABILIZATION.md` (the hip investigation on take3 at 30 fps), `smplx_opt_handoff.md` (earlier SMPL-X handoff). Reproducibility: `smplx_fit/golden_compare.py` replays the fits of this run and compares them with the frozen reference (section 10).

Conventions: lengths in metres in the code and millimetres in text; "calibration frame" is the world frame of the calibration file (camera 1 of the rig for today's files, not gravity aligned); "floor frame" is Z up with the floor at Z = 0 (section 5).

---

## 1. Overview

```
take (mjpeg + timestamp logs)
  1 align_session.py ........... aligned/camN/NNNNNN.jpg   (timestamp alignment; flash offsets only verify)
  2 pose2d.run_pipeline ........ MediaPipe 2D body/hand    (scaffold only, see 3.1)
    pose2d.triangulation ....... MediaPipe 3D               (scaffold for the mask crops)
    smplx_fit.segmentation ..... aligned/masks_rvm          (RVM person masks, keypoint-guided crops)
  3 sapiens2.smooth_bbox ....... boxes_smooth.npz           (temporally stable person boxes)
    sapiens2.extract_multiframe  allframes_1b_smoothbox.npz (Sapiens2 1b, 308 keypoints, every frame x camera)
    sapiens2.to_pose2d ......... aligned/pose2d_sapiens_*   (body ids 0-12 and both hands, tracking, occlusion handling)
  4 pose2d.triangulation ....... Sapiens 3D, hips: fixed cameras + gate     (output/<run>/silhip)
    pose2d.refine_bones ........ bone-length-consistent body                  (output/<run>/bones)
    pose2d.triangulate_legs .... knees, ankles, feet                           (legs.npz)
  5 smplx_fit.fit_take ......... 5-stage SMPL-X fit (silhouette, hands, legs)  (smplx_params.npz, stage_params.npz)
    smplx_fit.add_legs --align-only  re-express the fit in the floor frame
```

Inputs per take: calibration json (same cameras, same resolution), the take's mjpeg files and timestamp logs. Everything after step 1 works on the aligned frames; the scratch outputs of the 171905 run live under `output/t2_171905/` (not in git).

## 2. Inputs and checks

- **Calibration.** Today's takes recorded "uncalibrated". The half-resolution calibration `20261022_173214_8cam_half.json` (made after the takes) is used **unchanged**. `flash_guided.find_calibration` picks the closest-in-time calibration with the right width and devices for the flash check; the fit uses the file named on the command line.
- **Alignment.** `align_session.py` aligns frames by device timestamps (residuals about 17 ms or less, flash offsets +-1 frame). The flash offsets from `flash_offsets.py` (with the calibration-guided pass for weak cameras) are a verification, not an input. 946 frames, no drops on take_2.
- **Frame rate.** All time-based settings were chosen at 30 fps and **scaled by reasoning** to 60 fps (listed per step). The 60 fps values have not been validated against a 30 fps take.

## 3. 2D stage

### 3.1 Scaffold: MediaPipe + RVM masks
`pose2d.run_pipeline` and `pose2d.triangulation` (plain) produce a MediaPipe 3D reconstruction. It is only a scaffold: the RVM masks are cut with **keypoint-guided crops** (3D keypoints projected into each camera, padded 30%). Defaults of `smplx_fit.segmentation --backend rvm`: backbone mobilenetv3, downsample ratio 0.25, pad 0.30, smoothing window 5. RVM masks are used for the boxes and for the silhouette term of the fit.

### 3.2 Stable person boxes (`sapiens2/smooth_bbox.py`)
Sapiens is top-down: it crops to a box. Per-frame mask extents jump when the hands go up and that shifts other keypoints such as the hips. Each box edge is a sliding-window percentile (outer edges 10th/90th percentile over +-win frames), Gaussian smoothed, padded 0.02, forced to the model aspect 3:4 and pinned at the frame bottom (`--frame-h`). Defaults at 30 fps: `--win 30 --sigma 10`. **At 60 fps: `--win 60 --sigma 20`, `--frame-h 1520`.**

### 3.3 Sapiens2 keypoints (`sapiens2/extract_multiframe.py`, `to_pose2d.py`)
Sapiens2 1b, 308 keypoints, 1024x768 input, run on every frame and camera with the smoothed boxes (about 43 minutes for 946 frames x 8 cameras). `to_pose2d` keeps goliath ids 0-8 (nose, eyes, ears, shoulders, elbows), the wrists (62 left, 41 right) and hips (9, 10) as body ids 0-12 (COCO order), plus 21 points per hand; a part counts as detected if its mean keypoint score is at least 0.3. Leg points (goliath 11-20) stay in the npz and are used by section 5. Hand tracking, swap rules and **hand-occlusion handling** are in `pose2d/tracking.py` (see 7.1).

## 4. Triangulation

### 4.1 Plain Sapiens triangulation
`pose2d.triangulation <take> --calib CAL --pose2d-dir pose2d_sapiens_smoothbox`: per-frame RANSAC over camera pairs on a reference joint, inlier threshold 15 px, min confidence 0.5, then DLT on the chosen set for every landmark, then Kalman/RTS smoothing (constant velocity, process std 0.03). Take_2: body 946/946 frames, 7.8 cameras on average, median reprojection 5.3 px, hands 857/946 (with the occlusion penalty). In practice the 15 px threshold accepts almost every camera: 891 of 946 frames use all 8, the set changes on 20 frames, so RANSAC selects very little. The shake is not camera switching; it comes from per-camera, per-joint 2D differences inside a nearly fixed set (the median reprojection residual is 3.7 px in cam0 and 25 px in cam3).

### 4.2 Hips: fixed cameras and a gate (`--camera-mode stable ...`)
The Sapiens hip keypoint has a view-dependent definition: the cameras disagree by near-constant offsets (15-50 px), so the hip depends on which cameras are used. Fix: triangulate the hips only from a fixed camera subset, and gate outliers before smoothing (details and numbers in `pose2d/HIP_STABILIZATION.md`). For take_2 the subset was found by searching camera subsets for the smallest hip-width variation: **`--hip-cams cam1,cam2,cam4,cam5,cam7`** gave hip-width std 2.1 mm against 5.5 mm with all eight cameras. Command (60 fps):

```
python -m pose2d.triangulation $D --calib $CAL --pose2d-dir pose2d_sapiens_smoothbox --camera-mode stable \
  --hip-cams cam1,cam2,cam4,cam5,cam7 --hip-min-cams 3 --hip-reproj-px 60 --gate-hips --gate-vmax-mm 15 \
  --out-dir output/t2_171905/silhip --force
```

`--gate-vmax-mm` is a speed gate in mm per frame: 30 at 30 fps, **15 at 60 fps**. The torso and hip-width gates keep their defaults (40 and 50 mm). 88 frames were torso-gated on take_2. Camera subsets are specific to a rig and take; redo the subset search for another rig.

### 4.3 Bone-length refinement (`pose2d/refine_bones.py`)
"Bone refined" means: fit all body joints (ids 0-12) of the whole take at once, starting from the hip-stable triangulation. The loss has a robust (Cauchy) reprojection term in every camera that sees the joint (scale 8 px, hips 20 px, min confidence 0.5), constant bone lengths shared by all frames (free parameters, left and right tied; bone sigma 5 mm) and a second-difference smoothness term (`--accel-sigma-mm`, default 4, **1.0 at 60 fps**), 300 iterations on the GPU.

```
python -m pose2d.refine_bones --take $D --calib $CAL --pose2d-dir pose2d_sapiens_smoothbox \
  --init-dir output/t2_171905/silhip/aligned/pose2d_sapiens_smoothbox --out-dir output/t2_171905/bones \
  --hip-cams cam1,cam2,cam4,cam5,cam7 --accel-sigma-mm 1.0
```

Take_2 refined bone lengths: shoulder 372, hip 200, torso 577, upper arm 311, forearm 276 mm; body reprojection median 4.5 px after refinement. The refined joints are very smooth: shoulder and hip RMS acceleration 0.35-0.62 m/s2 against 1.5-1.8 for the triangulation, power below 1 Hz 99%+.

### 4.4 Hands without the occlusion penalty
Where the two hands overlap (signer claps) `tracking.flag_hand_occlusions` (IoU of the hand boxes above 0.1) multiplies both hands' confidence by 0.01 for that camera and frame, and the triangulation (min confidence 0.5) drops them. That removed the hands in 89 frames (632-682, 736-773 and short runs). Use `sapiens2.to_pose2d ... --name pose2d_sapiens_noocc --no-occlusion-penalty` and triangulate that set: hands 946/946. The fit uses **body from the bone-refined set and hands from the no-penalty set**. Hand quality during the clap itself has not been checked visually.

## 5. Legs, floor, and offsets of the keypoints

- **Leg triangulation** (`pose2d/triangulate_legs.py`): goliath 11-20 (knees, ankles, big/small toes, heels), min score 0.5, a camera is skipped if the point is within 25 px of the frame bottom (feet cut off), RANSAC over pairs with a 15 px threshold, refit on the inliers. Take_2: 94-100% of frames per point, knees and ankles from a median of 8 cameras, ankle reprojection 3-5 px. The fit smooths the tracks (5-frame median, Gaussian sigma 2 frames) and weights them by camera count.
- **Floor frame.** Today's calibration has no world alignment. The floor-aligned `20261022_165256_8cam.json` (4K, same rig, camera heights 1-2 m above Z = 0) was used: a rigid fit of the camera centres of the two calibrations (residuals 6-44 mm, rotation differences 0.5-7.8 degrees, so the floor tilt is uncertain by a degree or two) gives the map `X_floor = Rg X_calib + tg`. `smplx_fit.add_legs --align-only` re-expresses a finished fit in that frame exactly (vertex error 0.001 mm). In the floor frame the lowest mesh point of the current fit is 6 cm below the floor on median; the map, not the fit, is the likely cause and nothing in the fit uses the floor.
- **Sapiens-vs-SMPL-X offsets.** Sapiens places keypoints at skin-surface landmarks, not at SMPL-X joint centres. Offsets were measured on synthetic renders (`smplx_fit/synth_*.py`, README section "Sapiens keypoints vs SMPL-X joints"). Hands: wrist about 15 mm proximal, MCPs 10-21 mm proximal. Legs (measured on one texture/shape with `synth_analyze_legs.py`): ankles 10-17 mm outward and 7-16 mm up, heels about 20 mm forward, big toes 4-10 mm behind. Values: `smplx_fit/sapiens_offsets.py`. Both offset corrections are **options, off in the main fit** (7.4, 7.7).

## 6. SMPL-X fit (`smplx_fit/fit_take.py`, `optimize.py`)

Male SMPL-X, 20 betas, all frames fitted at once (shared betas, per-frame global orientation, translation, body pose, hand poses). L-BFGS (strong Wolfe line search, 30 inner iterations per outer step), at most 10 outer iterations per stage, stop when the relative loss change is below 1e-9. Targets are the smoothed triangulations; a landmark with zero confidence in a frame has zero weight. k3d is the confidence-weighted mean squared 3D distance of the model's mapped joints/vertices to the targets (metres squared).

| Stage | Free parameters | Loss (weights) |
|---|---|---|
| 1 shape | betas | bone-length consistency x 10000 (arm/hand bones, plus leg bones when legs are in), reg_shape (L2 on betas) x 1.0 |
| 2 global R,t | global_orient, transl | body k3d x 1, smooth transl x 0.5, smooth global orient x 0.1, pose prior x 0.01 |
| 3 body pose | global_orient, transl, body_pose, betas | body k3d x 1, smooth transl 0.5, smooth global orient 0.1, **smooth body pose x 5** (`--body-smooth-weight`), pose prior x 0.01, reg_shape x 0.1, **silhouette x 0.1** |
| 4 hands + body | body_pose, hand poses, transl (with `--stage4-freeze-torso`: only shoulder/elbow/wrist rotations and hands) | body k3d x 1, **hand k3d x 10**, smooth body 5, smooth hand x 0.1, smooth transl x 5, pose prior 0.01, hand reg x 0.0001 |
| 5 hands only | hand poses | same loss as 4, body frozen |

- **Silhouette term.** RVM masks, mesh rasterised with nvdiffrast at 32x24 (keeps the 4:3 frame shape), 1500 sampled mask pixels, sigma 1 px, weight 0.1, stage 3 only. Rasterising all 7568 camera-frame pairs at once ran out of GPU memory, so it runs in chunks of 2048 images (result identical to one batch).
- **Pose prior.** `l2` backend, weight 0.01 (the gmm backend uses 0.0001 and is not used).
- **Cost.** About 3 minutes for 946 frames. Stages 3 and 4 usually stop at the 10-iteration cap with the loss still falling slowly (not converged); stage 5 converges by the relative-change test.
- **Up axis.** The fit has no gravity information. In the calibration frame the body's up vector is about 95 degrees from +Z; this only affects `global_orient`.

## 7. Special solutions (why, what, effect, how to switch off)

1. **Hand-occlusion penalty and the clap case** (4.4). Why: hands vanished during claps. What: `--no-occlusion-penalty` in `to_pose2d`. Effect: hands 857 -> 946 of 946 frames. Off: default behaviour keeps the penalty.
2. **Hips from fixed cameras plus a gate** (4.2). Why: view-dependent hip keypoint definition and camera-set switching. Effect: hip width std 5.5 -> 2.1 mm on take_2. Off: leave out `--camera-mode stable ...`.
3. **Bone-length refinement** (4.3). Why: shoulder, hip and arm lengths should be constant; triangulation noise made them vary. Effect: body accel 1.5-1.8 -> 0.4-0.6 m/s2. Off: use the `silhip` set directly.
4. **Torso freeze in stage 4** (`--stage4-freeze-torso`). Why: when the fingers were fitted with the body free, the arms and the torso moved to satisfy the hand loss and the body joints became about 3 times rougher (head and shoulder points 6-8 m/s2 against 2 after stage 3). What: in stage 4 only the shoulder, elbow and wrist rotations and the hands move; collars, spine, neck, head, hips, legs and translation keep their stage-3 values. Effect on take_2: head/shoulder/elbow acceleration down to 2-3 m/s2, mean body target error 22.4 -> 20.4 mm, hands unchanged (9.6 mm); the elbows are 3-5 mm worse. Off: omit the flag.
5. **Hand offsets** (`--sapiens-offsets`). Adds the measured wrist/MCP offsets on the SMPL-X side before comparing with the Sapiens targets, in the model's own hand frame. Measured on synthetic renders only, not validated on real photos; on take_3 it improved elbows and index/middle MCPs and worsened thumb and pinky MCPs. **Off in the main fit** (a fit with it was kept for comparison).
6. **Legs** (`--legs-npz legs.npz`). Adds knee, ankle and foot targets and hip-knee-ankle bone lengths to the shape stage, so the shape sees leg lengths. Effect: knee errors 104-114 -> 16-23 mm, ankles 156-181 -> 22-32 mm, but the leg targets pull on the hips and torso: shoulder error 20 -> 40 mm, hips 46 -> 68 mm, body RMS 25.3 -> 37.3 mm. Adding the legs afterwards (`smplx_fit.add_legs`, legs-only stage, optionally with the shape) keeps the upper body but fits the legs less well (knees 64-71 mm).
7. **Leg offsets** (`--sapiens-leg-offsets`). Same idea as 5 for legs; effect: toes and left heel better (toe error about 20 -> 11 mm), knees and right ankle slightly worse, mesh floor contact unchanged, body RMS 37.3 -> 38.1 mm. One texture and one shape; off by default.
8. **Wrist offset experiment** (`--wrist-offset`, not used). Shifting the wrist by the MANO-wrist offset overshot by 10-15 mm and worsened elbows and wrists (README "SMPL-X wrist keypoint offset"). The SMPL-X wrist is already close to the Sapiens wrist.
9. **Chunked silhouette rasterisation** (`NVDIFFRAST_CHUNK = 2048`). Why: GPU out-of-memory on long takes. No change in results.
10. **Floor alignment from another calibration** (5). Why: this calibration has no floor. Caveats: rotation error up to a few degrees; used for viewing and for the floor-frame outputs only.

## 8. Recommended recipe for now (take_2 of session 20261022_171905)

Paths: `D=/data/oak_recorder_sessions/20261022_171905_take2` (links to the take's mjpeg and logs), `CAL=/mnt/mmocap/temp_calib/20261022_173214_8cam_half.json`.

```
python align_session.py $D
python -m pose2d.run_pipeline $D --calib $CAL                      # MediaPipe scaffold
python -m pose2d.triangulation $D --calib $CAL
python -m smplx_fit.segmentation $D --calib $CAL --backend rvm --mask-subdir masks_rvm
python -m sapiens2.smooth_bbox --take $D --out output/sapiens2_171905_t2/boxes_smooth.npz --n-frames 946 \
   --cams cam0 cam1 cam2 cam3 cam4 cam5 cam6 cam7 --win 60 --sigma 20 --frame-h 1520
python -m sapiens2.extract_multiframe --take $D --n 946 --start 0 --stop 945 --size 1b --cams cam0 ... cam7 \
   --boxes output/sapiens2_171905_t2/boxes_smooth.npz --out output/sapiens2_171905_t2/allframes_1b_smoothbox.npz
python -m sapiens2.to_pose2d --take $D --npz output/sapiens2_171905_t2/allframes_1b_smoothbox.npz --calib $CAL --name pose2d_sapiens_smoothbox
python -m sapiens2.to_pose2d --take $D --npz ... --calib $CAL --name pose2d_sapiens_noocc --no-occlusion-penalty
python -m pose2d.triangulation $D --calib $CAL --pose2d-dir pose2d_sapiens_smoothbox --camera-mode stable \
   --hip-cams cam1,cam2,cam4,cam5,cam7 --hip-min-cams 3 --hip-reproj-px 60 --gate-hips --gate-vmax-mm 15 --out-dir output/t2_171905/silhip --force
python -m pose2d.triangulation $D --calib $CAL --pose2d-dir pose2d_sapiens_noocc \
   --out-dir output/t2_171905/noocc --force                                                  # hands without the penalty
python -m pose2d.refine_bones --take $D --calib $CAL --pose2d-dir pose2d_sapiens_smoothbox \
   --init-dir output/t2_171905/silhip/aligned/pose2d_sapiens_smoothbox --out-dir output/t2_171905/bones \
   --hip-cams cam1,cam2,cam4,cam5,cam7 --accel-sigma-mm 1.0
python -m pose2d.triangulate_legs --npz output/sapiens2_171905_t2/allframes_1b_smoothbox.npz --calib $CAL --out output/t2_171905/legs.npz
# stand-in take dir smplx_in_legs: aligned/cam*, aligned/masks_rvm, and links to the bone-refined body (reconstruction_body.json) and the
# no-penalty hands (reconstruction_left/right.json) under aligned/pose2d_sapiens_smoothbox/
python -m smplx_fit.fit_take output/t2_171905/smplx_in_legs --model-path models/SMPLX --pose2d-dir pose2d_sapiens_smoothbox \
   --gender male --num-betas 20 --calib $CAL --use-silhouette --silhouette-weight 0.1 --mask-subdir masks_rvm \
   --silhouette-out-size 32x24 --silhouette-n-samples 1500 --silhouette-sigma-px 1.0 --hand-k3d-weight 10 \
   --stage4-freeze-torso --legs-npz output/t2_171905/legs.npz --body-smooth-weight 5 --shape-reg-weight 1.0 --n-shape-betas 0 \
   --hand-reg-weight 0.0001 --pose-prior-weight 0.01 --max-outer-iters 10 --force
python -m smplx_fit.add_legs --fit output/t2_171905/smplx_in_legs --floor-map output/t2_171905/calib_half_floor_map.npz \
   --out output/t2_171905/smplx_floor_full --align-only
```

Variants kept for comparison (the same command with one change): without legs (drop `--legs-npz`), and with leg offsets (`--sapiens-leg-offsets`). The user judged the legs-with-torso-freeze fit and the leg-offsets fit as good; the numbers below are for those and the legless baseline.

| Fit | Body RMS | Hand RMS | Leg errors (knee / ankle / heel, mm) | Body-point accel (m/s2) |
|---|---|---|---|---|
| `smplx_in_freeze` (no legs) | 25.3 | 11.0 | no leg targets | head 2.3-2.4, shoulders 2.0, elbows 3.0 |
| `smplx_in_legs` | 37.3 | 10.0 | 16-23 / 22-32 / 41-52 | not measured for this fit |
| `smplx_in_legsoff` | 38.1 | 10.0 | 22-23 / 34 / 26-55 | not measured for this fit |

## 9. Known open points

- 28 non-clap frames where the left/right order along the shoulder axis is flipped (crossed arms or a swap): not investigated.
- Stages 3 and 4 stop at the iteration cap; more iterations were not tried.
- 60 fps scalings (box window, hip speed gate, refinement smoothness) are reasoned, not validated against a 30 fps take.
- With legs, shoulder, hip and nose errors grow (section 7.6); a heavier weight on the body targets or a lighter leg weight is the obvious next experiment.
- The lowest mesh point is 6 cm below the floor in the floor frame; cause not isolated (floor map, leg shape or leg offsets).
- Offsets are measured on synthetic renders with one or a few textures; not validated on real photos. The knee offsets are asymmetric (left 30 mm medial, right 12 mm) and unexplained.
- Hand quality during the clap is not checked visually; the camera subset for the hips is rig- and take-specific.

## 10. Reproducibility check

`smplx_fit/golden_compare.py` replays a fit with the code of any checkout and compares with the frozen reference of this run (`output/golden_t2_171905`, outside git). The GPU fit is not bit-reproducible: two identical runs differ by about 1-3% in the stage-3 loss, 1 mm in the upper-body mesh and up to 15-20 mm in the legs when they have no targets, so comparisons use tolerances (printed with every comparison); the deterministic steps (leg triangulation, floor alignment) are compared at 0.001 mm.
