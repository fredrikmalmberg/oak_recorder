# Multi-Camera Live ChArUco Calibration Tool — Agent Instructions

## 1. Objective
Build a Python tool that live-calibrates an arbitrary number of cameras (intrinsics + extrinsics) using a ChArUco board, with a real-time 3D viewer built in `viser`. The toolcan replicate the existing camera capture layer in `capture.py`.

## 2. Starting point — read before writing anything
- Read `capture.py` first. It already handles camera enumeration/connection, warmup, and per-camera capture settings (resolution, exposure, etc.).
- Reuse its camera settings objects/interfaces as-is. 
- Number of cameras is unknown ahead of time — discover it dynamically from whatever `capture.py` exposes (e.g. a list/dict of camera objects). Nothing in the pipeline should hardcode a camera count.

## 3. Board specification (fixed, put in config)
- Type: ChArUco
- Squares: 6 (X) × 4 (Y)
- Square size: 40 mm
- Marker size: 30 mm
- ArUco dictionary: `DICT_4X4_50` (confirm exact enum name/size against installed OpenCV version — if ambiguous between `DICT_4X4_50/100/250/1000`, default to `DICT_4X4_50` and expose it as a config value, not a hardcoded constant)
- Board parameters must live in one config location (see §9), not scattered through the code.

## 4. Camera assumptions
- Cameras: unknown/variable count, discovered at runtime.
- Not hardware-synced (free-running). Do not assume frame-index alignment between cameras — each camera's detection is independent per its own latest frame.
- Resolution: 4K per camera. Be mindful of performance — detection, undistortion, and viser image streaming at 4K on N cameras can bottleneck; downsample for display in viser if needed while keeping full-res for detection/calibration math (make this configurable).
- Cameras are not guaranteed to mutually overlap in field of view. Some camera pairs may never see the board at the same time.

## 5. Calibration scope
Both intrinsics and extrinsics, computed in the same run/pipeline:

### 5a. Intrinsics (per camera, independent)
- Accumulate ChArUco detections per camera over time.
- Standard OpenCV ChArUco intrinsic calibration (camera matrix + distortion coefficients).
- Run until a reprojection-error threshold is met (see §7) rather than a fixed frame count.

### 5b. Extrinsics (relative poses between cameras)
- Because not all cameras overlap, extrinsics cannot be solved by a single pairwise stereo step across all cameras. Use a **pose graph** approach:
  - Whenever two or more cameras detect the board simultaneously (or near-simultaneously, given no hardware sync — define a max timestamp/frame-age tolerance for "simultaneous"), compute the relative transform between them via their individual board-to-camera poses (using the already-solved intrinsics).
  - Build a graph where nodes = cameras, edges = observed relative transforms (with an associated confidence/error weight).
  - **Reference/world camera selection is an open assumption**: choose automatically as the camera with the most edges (best-connected node) once the graph has enough data. Document this choice explicitly in code comments and in the output metadata (§10), since it was not specified by the user and may need revisiting.
  - Solve for all camera poses relative to the reference by traversing/optimizing over the graph (shortest reliable path, or global pose graph optimization if precision requires it — pose graph optimization is preferred over naive chaining if reprojection error from naive chaining is poor).
  - Cameras that never share a view with any connected camera cannot be extrinsically calibrated — detect this case, warn clearly (console + viser), and exclude them from the extrinsic solve rather than failing the whole run.

## 6. Auto-capture (frame selection) logic
Fully automatic — no manual keypress needed. For each camera independently, on every incoming frame:
1. Run ChArUco detection.
2. Apply quality gates before accepting the frame as a calibration sample:
   - Minimum number of detected corners (configurable threshold).
   - Minimum board coverage/spread in the image (avoid clustering all samples in one region — e.g. divide image into a grid and track which cells have been covered).
   - Minimum angle diversity (avoid capturing the board fronto-parallel repeatedly — estimate board tilt from the pose and require a spread of angles).
   - Sharpness/blur check (e.g. variance of Laplacian) to reject motion-blurred frames.
   - Minimum spatial/pose distance from previously accepted samples for that camera (avoid near-duplicate captures).
3. Only frames passing all gates are added to that camera's calibration sample set.
4. All thresholds in step 2 must be config values, not magic numbers in code.

## 7. Termination condition
- Run continuously until a reprojection-error threshold is satisfied, not a fixed capture count.
- Suggested criterion (make configurable): per-camera intrinsic calibration is "done" when mean reprojection error drops below `X` px AND coverage grid (from §6) is sufficiently filled (e.g. ≥80% of grid cells hit). Extrinsic edges are "done" similarly once their relative-pose reprojection error is below threshold.
- Overall run can either stop automatically once all reachable cameras/edges meet threshold, or keep running indefinitely and simply mark things as "converged" in the viewer/CLI (recommended: keep running, since operator watches the viewer and stops manually — but auto-stop should be available as a config flag).

## 8. Viser live viewer — requirements
Must update every frame in which a board is discovered (not throttled to a fixed low rate, but avoid flooding — reasonable to coalesce updates if multiple cameras report in the same tick).

Show, at minimum:
- **3D scene**: camera frustums placed at their current best-estimate extrinsic pose (once solved; before solving, cameras with unknown pose should be visually distinguished, e.g. shown at origin/greyed out or omitted with a text label "pose unknown").
- **Live board pose**: the detected ChArUco board's 3D pose, updated live per camera that currently sees it (if multiple cameras see it at once, show consistency/disagreement if relevant).
- **Per-camera status panel**: reprojection error, number of accepted samples, coverage percentage, "converged" yes/no.
- **Coverage guidance**: for each camera not yet converged, indicate where the board should move next to improve coverage — e.g. highlight unfilled cells of that camera's coverage grid (either as an overlay in a 2D per-camera thumbnail, or as suggested 3D regions in the scene). This is best-effort guidance, not a hard requirement to be perfectly optimal — clearly comment this as a heuristic.
- **Live camera thumbnails**: downsampled live feed per camera (see §4 performance note), so the operator can see what each camera currently sees and where the board is in-frame. The live feed could potentially show a heatmap to indicate where coverage is bad

Interaction:
- The run is CLI-driven (start, stop, save, reset triggered from terminal/keyboard), but viser should also surface status and prompts (e.g. "camera 3 needs more coverage in top-left", "camera 5: pose unknown — no shared view yet") so the operator mainly watches viser and only touches the terminal to start/stop/save.
- viser panel(s) may include read-only or convenience buttons (e.g. "save now") but the source of truth for control flow is the CLI process, not viser callbacks driving core logic.

## 9. Configuration
Centralize all tunable parameters (board spec, dictionary, quality-gate thresholds, reprojection thresholds, coverage grid size, simultaneity tolerance, display downsample factor, output path) in a single config file (YAML or JSON — pick one and be consistent) loaded at startup. No magic numbers scattered in the code.

## 10. Output format
Save results as JSON, containing:
- Per-camera intrinsics: camera matrix, distortion coefficients, resolution, final reprojection error, number of samples used.
- Per-camera extrinsics: rotation + translation relative to the auto-selected reference camera, which edges of the pose graph were used to derive it (chain path or optimization residual), and reprojection error for that estimate.
- Metadata: which camera was auto-selected as reference and why (e.g. "most graph connections"), timestamp of calibration run, board spec used, list of any cameras excluded from extrinsic solve (never shared a view) with reason.
- Do not need to persist raw calibration images by default; keep this feature scoped out unless trivial to add — if included, make it an opt-in config flag, not default behavior, given 4K storage cost.

## 11. Error handling / edge cases to handle explicitly
- Camera disconnects/reconnects mid-run.
- Zero or one camera detected (extrinsics meaningless with one camera — should still run intrinsics and say so clearly).
- Board never detected by some camera for a long time — surface as a persistent warning, not a silent hang.
- Disconnected pose graph (islands of cameras that never share views with each other) — calibrate each connected component against its own local reference, and clearly report in output that there are multiple independent reference frames rather than one global one.

## 12. Open assumptions to flag back to the user for review
Since these weren't fully specified, implement with a clearly documented default and flag them in code comments/README so they're easy to revisit:
- Reference camera auto-selection strategy (most-connected node).
- Exact quality-gate threshold values (corner count, blur variance, coverage grid resolution, angle diversity).
- Reprojection error threshold(s) for "converged."
- Simultaneity tolerance window for treating two free-running cameras' frames as "the same" observation for extrinsic edges.
- Live display downsample factor for 4K feeds in viser.
