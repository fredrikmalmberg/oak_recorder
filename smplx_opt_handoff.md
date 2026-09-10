# SMPL-X fitting pipeline: GPU speedup handoff

Handoff for an agent with GPU access to speed up one specific stage of an
already-working CPU-only SMPL-X fitting pipeline. Everything below is
current as of this handoff; treat file line numbers as approximate.

## Project context

`oak_recorder` (this repo) captures multi-camera (up to 7 OAK cameras)
video of a person doing sign language, triangulates 2D hand/body keypoints
(MediaPipe) into 3D, then fits an SMPL-X body mesh to those 3D keypoints
per frame. The fitting pipeline lives entirely in `smplx_fit/` and is a
from-scratch reimplementation (inspired by a sibling project `bvh2smplx`
and by EasyMoCap's architecture), not a fork of either.

Current test data: `recordings/20260908_174819/take_1`, a 786-frame take,
calibrated via `output/calibration/20260908_174749_7cam.json` (7 cameras).
A 40-frame slice (`--frames 373:413`) has been used throughout development
as the fast iteration/validation set.

## Pipeline overview

Entry point: `python -m smplx_fit.fit_take <take_dir> --model-path
models/SMPLX --calib <calib.json> [flags...]`. Reads already-triangulated
3D keypoints (`smplx_fit/data_loading.py`, from `pose2d.triangulation`'s
output), runs a 5(+1)-stage L-BFGS/Adam optimizer
(`smplx_fit/optimize.py`), writes `smplx_params.npz` +
`optimization_log.json` (full per-iteration loss breakdown, every term
logged raw, not just the weighted sum) + `fit_config.json` under
`<take_dir>/aligned/pose2d/smplx/`.

SMPL-X model files: `models/SMPLX/SMPLX_NEUTRAL.npz` (already present).
Uses the `smplx` pip package's `SMPLXLayer` (pure functional, no internal
state) -- see `smplx_fit/model.py`'s docstring for the one non-obvious API
detail (it wants rotation matrices, not axis-angle; this project stores/
optimizes everything as axis-angle and converts once, right before each
forward pass).

### The 6 stages (`smplx_fit/optimize.py::multi_stage_optimize`)

1. **1_shape**: fit `betas` (shared across the whole take -- ONE shape for
   all frames) against bone-length consistency (`shape3d_loss`, comparing
   triangulated keypoint distances to the model's own rest-pose bone
   lengths). Rest pose only (global_orient/transl/body_pose all zero) --
   no camera placement exists yet at this point.
2. **2_global_rt**: fit `global_orient`/`transl` per frame against 3D
   keypoint distance (`k3d`).
3. **3_body_pose**: fit `body_pose` (+ refine global_orient/transl) against
   `k3d` + a pose prior (`reg_pose`, pluggable: L2-to-zero or a GMM
   pose prior, see `pose_prior.py`) + **optionally a silhouette-overlap
   term** (`--use-silhouette`, see below).
3b. **3b_shape_refine** (NEW, opt-in via `--use-silhouette-shape`,
   separate from `--use-silhouette`): re-fits `betas` using the silhouette
   term now that a real camera placement exists (impossible in stage 1's
   rest pose). **This is the stage that needs GPU speedup -- see "The
   task" below.**
4. **4_hands_and_body**: fit `body_pose` + both hand poses against `k3d` +
   `k3d_hand` (10x weight) + temporal smoothness + regularization.
5. **5_hands_only**: final hand-only polish pass.

Every stage uses `torch.optim.LBFGS(max_iter=30, line_search_fn=
"strong_wolfe")` in an outer convergence loop (`_run_lbfgs_stage`,
stops at `rel_tol=1e-9` relative loss change or `max_outer_iters`
outer iterations) -- **except stage 3b, which uses a new
`_run_adam_stage`** (plain Adam, 100 iterations, `lr=0.01`) because LBFGS's
line search was confirmed empirically to leave `betas` completely frozen
in stage 3b (see "Known LBFGS failure" below).

## Phase 3 (silhouette fitting) -- what it is and why it's slow

`smplx_fit/segmentation.py` extracts a person segmentation mask per
camera/frame (MediaPipe `SelfieSegmentation`), cached to
`<take_dir>/aligned/masks/<cam_id>/<frame>.jpg`. Already run for the full
take -- masks exist on disk, mean coverage ~12.5% of frame, no dependency
on GPU work.

`smplx_fit/silhouette.py` is a **from-scratch, dependency-free**
differentiable "soft splat" renderer, built specifically because
**PyTorch3D (the standard choice, and what EasyMoCap itself uses) has no
PyPI wheel and no win-64 conda package** -- confirmed by direct query
against both indexes this session. Building it from source on Windows
needs a full MSVC toolchain and, per PyTorch3D's own history, is fragile
even then. **On a GPU Linux box this constraint likely does not apply --
PyTorch3D ships real CUDA wheels/conda packages for Linux.** Switching to
true PyTorch3D rasterization (instead of this project's soft-splat
approximation) is probably the single highest-value change a GPU agent
could make -- see "Suggested approaches" below.

The soft-splat approach, as currently implemented:
1. `build_surface_samples`: pick `N_SURFACE_SAMPLES` (default 8000, but
   see below) points on the mesh surface once, via face-area-weighted
   sampling + barycentric interpolation -- fixed face indices/weights,
   reused every call; only vertex positions vary.
2. `sample_surface_points`: linear combination of the 3 vertices of each
   sampled face -> (B, N, 3) points, differentiable w.r.t. vertices.
3. `project_points`: standard `x_cam = R @ X_world + t` then perspective
   divide by `K` -- same convention as
   `hand_pose.hand_multiview.build_projection_matrix` uses everywhere
   else in this project.
4. `render_silhouette`: for EVERY output pixel and EVERY sample point,
   compute a Gaussian falloff by squared distance, then soft-OR
   (`1 - prod(1 - gaussian_i)`) across points. This is `O(out_h * out_w *
   N)` per (frame, camera) pair, done in a **Python for-loop over frames x
   cameras** (`compute_silhouette_term` in the same file) -- NOT batched
   across frames or cameras at all currently.
5. `losses.silhouette_loss`: plain per-pixel MSE against the cached mask
   (resized to the same `out_size`).

## Measured performance (all CPU-only, this session, on a 40-frame /
7-camera test slice = 280 (frame, camera) pairs)

- At "full fidelity" (`n_samples=8000`, render `out_size=(64,36)`): **~20s
  per single forward+backward pass** over the whole 280-pair batch.
  Confirmed intractable inside LBFGS's inner loop (up to 30 line-search
  steps per outer iteration).
- Reduced to `n_samples=1500`, `out_size=(32,18)`, `sigma_px=1.0`
  (`sigma_px` MUST scale down proportionally with `out_size`, confirmed
  empirically -- see `render_silhouette`'s docstring for the exact bug
  this caused): **~1-5s per forward+backward** over the same 280 pairs
  (timing varied noticeably run-to-run on this shared/loaded machine).
  This is the current default (`optimize.py`'s `silhouette_n_samples`/
  `silhouette_sigma_px` params, `fit_take.py`'s
  `--silhouette-n-samples`/`--silhouette-sigma-px` flags).
- **Stage 3 alone** (silhouette added to body_pose fitting, no stage 3b):
  40-frame slice full 5-stage run = ~293-480s total (varied run-to-run).
  IoU against the real mask on a validation frame improved from ~0.51-0.53
  (keypoint-only baseline) to ~0.84-0.87 after this stage -- confirmed
  genuine (not the optimizer gaming the metric): rendered/real mask area
  ratios converged closely, and fitted joint angles stayed anatomically
  plausible (max ~117 deg at the elbow).
- **Stage 3b added** (betas refinement, same 40-frame slice): total
  runtime jumped to **~1031s** (100 Adam iterations, each paying the same
  ~1-5s silhouette cost). Extrapolating LINEARLY to the full 786-frame
  take (19.65x more frames, and the per-iteration cost scales with frame
  count since it's a Python loop over frames): **~5.6 HOURS estimated**,
  vs. ~95 minutes without stage 3b. This is why stage 3b is currently
  **opt-in and NOT run on the full take** (see `--use-silhouette-shape`,
  independent from `--use-silhouette`) -- it's kept in the codebase,
  validated as functionally correct on the 40-frame slice, but not
  practical to run at full scale on CPU.

## Known LBFGS failure (already fixed, for context)

Stage 3b originally reused `_run_lbfgs_stage` like every other stage.
LBFGS's line search left `betas` **completely frozen** -- confirmed via a
standalone gradient check (`betas.grad` was nonzero, norm ~0.012, so the
computational graph was fine) that the failure was LBFGS's line search
specifically: `shape3d` is weighted 10000x and already sitting exactly at
its stage-1 minimum (an extremely steep local bowl), while silhouette's
gradient is comparatively tiny, so every LBFGS trial step overshot
`shape3d`'s sharp curvature and failed the Wolfe sufficient-decrease
condition, leaving betas byte-for-byte unmoved across "converged" outer
iterations. Fixed by adding `_run_adam_stage` (plain Adam, handles
per-parameter scale mismatches far more gracefully than a single global
line search) -- this fix is real and should be KEPT regardless of what
else changes; it's not itself the performance bottleneck.

## The task: make Stage 3b (and/or Stage 3's silhouette term) fast enough
to run on the full 786-frame take

Two independent angles, either alone would probably help enormously;
doing both would help most:

### 1. Real PyTorch3D rasterization (likely the bigger win)

On a CUDA-capable Linux box, `pip install pytorch3d` (or the conda
package) should actually work, unlike on this Windows CPU-only machine.
If so, replace `silhouette.py`'s soft-splat point-cloud approximation with
real differentiable mesh rasterization (`pytorch3d.renderer`'s
`MeshRasterizer` + a silhouette shader, e.g. `SoftSilhouetteShader`) using
this project's existing `K`/`R`/`t` camera convention (will need
conversion to PyTorch3D's camera parameterization -- `PerspectiveCameras`
with `R`/`T`/`K` or focal_length/principal_point, check PyTorch3D's docs
for the exact convention, which differs in some axis-sign conventions
from OpenCV's). This would (a) be much faster on GPU than the current
`O(out_h * out_w * N)` per-pixel-per-point loop, (b) correctly handle
self-occlusion, which the current point-splat approach does NOT (a sample
point on the far side of a limb splats onto the image as if visible,
silently double-counting depth layers), and (c) allow much higher render
resolution/fidelity than the current 32x18 compromise.

If PyTorch3D still doesn't work cleanly even with CUDA (possible -- it has
a history of being finicky about exact torch/CUDA version matches), the
fallback is optimizing the existing soft-splat approach in place (next
section) rather than losing more time fighting the install -- same
"spike first, don't over-invest" principle this project applied on
Windows.

### 2. Vectorize/GPU-ify the existing soft-splat approach

Even without PyTorch3D, straightforward wins on a GPU:
- **Move everything to CUDA**: currently nothing in `silhouette.py`,
  `optimize.py`, or `model.py` calls `.to(device)` or accepts a `device`
  argument at all -- it's all implicitly CPU (default tensor device).
  Needs threading a `device` parameter through `load_layer`, `FitParams`,
  `load_silhouette_data`, `build_surface_samples`, and the forward/loss
  functions.
- **Batch across frames**: `compute_silhouette_term` currently loops
  `for i in range(nf)` in Python, calling `project_points` +
  `render_silhouette` once per frame. `project_points` already supports
  batched input (leading dims broadcast fine); `render_silhouette` would
  need a batch dimension added to its distance-grid computation
  (currently `(out_h, out_w, N)` per call -- could become `(B, out_h,
  out_w, N)`, mind the memory cost: on GPU this is likely fine at
  `out_size=(32,18)`, N=1500, even for the full 786-frame batch, but watch
  for OOM if bumping resolution/sample count up too.
- **Batch across cameras**: same idea -- currently a second Python loop
  `for cam_id, (K, R, t, image_size) in cams.items()`. Since different
  cameras can share the same `out_size`, this could also be vectorized
  (stack K/R/t across cameras, add a camera batch dimension throughout).
- Once batched, it's very likely worth bumping `n_samples`/`out_size`
  back up toward the "full fidelity" settings validated in the initial
  single-frame spike (`n_samples=8000`, `out_size=(64,36)`, `sigma_px`
  scaled proportionally) -- GPU should make that cost trivial where it
  cost ~20s/eval on CPU.

## Relevant files (all in `smplx_fit/` unless noted)

- `silhouette.py` -- the renderer itself (surface sampling, projection,
  soft-splat rendering, `load_silhouette_data`, `compute_silhouette_term`).
  This is almost certainly the file to rewrite/extend.
- `optimize.py` -- `multi_stage_optimize` (stage orchestration, all loss
  closures), `_run_lbfgs_stage`/`_run_adam_stage` (per-stage optimizer
  loops with logging/NaN-guard/convergence). Stage 3's silhouette term and
  stage 3b are both here.
- `losses.py` -- `silhouette_loss` (the pixel-wise MSE comparison).
- `model.py` -- `load_layer`/`forward` (SMPLXLayer wrapper),
  `upper_body_faces` (excludes legs/feet faces via skinning-weight
  classification, used by the viewer, not the fitter).
- `fit_take.py` -- CLI entry point; `--use-silhouette`,
  `--use-silhouette-shape`, `--silhouette-weight`,
  `--silhouette-out-size`, `--silhouette-n-samples`,
  `--silhouette-sigma-px` are the relevant flags.
- `segmentation.py` -- mask extraction (already run, not part of the
  speedup task, but the masks it produced are what silhouette fitting
  reads).
- `pose2d/visualize_triangulation.py` -- the viser 3D viewer used to
  visually inspect fits (`--smplx-npz`, `--frames`,
  `--smplx-hide-lower-body` flags), useful for validating any changes.

## How to reproduce/validate on the 40-frame slice

```
python -m smplx_fit.fit_take recordings/20260908_174819/take_1 \
    --model-path models/SMPLX --frames 373:413 --max-outer-iters 10 \
    --pose-prior l2 --use-silhouette --use-silhouette-shape \
    --calib output/calibration/20260908_174749_7cam.json --force
```

Check `optimization_log.json`'s `"3b_shape_refine"` records: `shape3d`/
`reg_shape`/`silhouette` should visibly change iteration-to-iteration (not
be bit-identical -- that was the LBFGS bug's signature), `silhouette`
should trend down, and the final `betas` (in `smplx_params.npz`) should
stay in a plausible human range (max component roughly -3 to 3; ~1.5 was
typical in this session's runs). Compare wall-clock time before/after your
speedup change on this same slice before attempting the full 786-frame
take.

## Current full-take run status (at handoff time)

A full-take fit WITHOUT `--use-silhouette-shape` (i.e. stage 3 silhouette
only, betas from stage 1's keypoint/bone-length fit) is running/queued in
the background on the original (CPU) machine, expected ~95 minutes. Once
GPU-accelerated stage 3b is working, the natural follow-up is re-running
the full take WITH `--use-silhouette-shape` for silhouette-refined betas
too.
