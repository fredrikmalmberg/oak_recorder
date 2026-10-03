# Hip stabilisation for Sapiens2 triangulation (take3, 2026-10-03)

Test case: `20260925_153809_take3` (768 frames, 6 cameras), 2D keypoints from Sapiens2 1B (308-keypoint set, body ids 0-12
used here), calibration `output/calibration/20260925_sp_lg_7cam_scaled_sapiens_ext8000.json`. All numbers are over
frames 300-650. Velocity/acceleration assume 30 fps.

## Adopted: fixed hip cameras + hip gate + Kalman/RTS smoothing

```
python -m pose2d.triangulation <take> --calib <calib.json> --pose2d-dir pose2d_sapiens_smoothbox \
    --camera-mode stable --hip-cams cam0,cam1,cam3,cam4 --hip-min-cams 3 --hip-reproj-px 60 --gate-hips \
    --out-dir <out> --force
```

- `--hip-cams`, `--hip-min-cams`, `--hip-reproj-px`: hips (landmarks 11, 12) are triangulated only from the given cameras
  (at least N of them, looser residual threshold). A hip point with too few cameras or a residual still above the
  threshold gets no measurement and the Kalman smoother bridges it.
- `--gate-hips` (`gate_hip_outliers`): before smoothing, a speed gate (`--gate-vmax-mm`, 30 mm/frame; 5+ consistent rejected
  points are accepted as a level shift) and a bone gate on torso length / hip width (`--gate-torso-tol-mm`,
  `--gate-width-tol-mm`). Shoulders are never rejected.
- Defaults are unchanged; only landmarks 11 and 12 differ from the plain stable mode. Shoulders, wrists and all other
  body landmarks are bit-identical to the baseline.

| | baseline (stable) | hip cams | hip cams + gate (adopted) |
|---|---|---|---|
| Rhip raw step max (mm) | 105 | 19 | 19 |
| Lhip raw step max (mm) | 83 | 52 | 23 |
| raw hip steps > 30 mm (R / L) | 10 / 5 | 0 / 7 | 0 / 0 |
| Rhip smoothed accel p95 (mm/frame^2) | 13.5 | 2.0 | 2.0 |
| Lhip smoothed accel p95 (mm/frame^2) | 10.9 | 8.7 | 5.8 |
| hip width std, raw (mm) | 12.9 | 3.3 | 3.3 |

Cause of the original jumps: the stable camera sets switched, and two camera groups disagree by about 80 mm in depth
(cams 2+5 versus cams 0/1/3/4; any two views agree trivially, the larger group is over-determined). Every raw hip step
over 30 mm coincided with a set change.

## Known limitation (accepted)

The hip 2D keypoints disagree between cameras by near-constant offsets (left hip, signed residual obs - proj under the
4-camera solution: cam0 +13/+52 px, cam1 -3/+16, cam3 +33/-36, cam4 -1/-29; right hip smaller). Std is only 2-12 px, so
this is a view-dependent keypoint definition, not noise. The hip is stable but its absolute position (depth especially)
is only good to a few cm: leaving out cam3 moves the left hip about 30 mm in depth. The left torso reads about 10 mm
longer than the right. Shoulders do not have this problem (residuals 1-11 px).

## Tried and not adopted

- **Silhouette flags** (`output/sapiens2/sil_flags.json`, from `sapiens2_test/make_sil_flags.py` /
  `kp_silhouette_check.py`): flag camera observations whose keypoint falls outside the SAM3 silhouette.
  `--exclude-flags` drops flagged camera observations in triangulation (kept in `triangulation.py`). Modest gain on the
  baseline, no gain once the hip cameras are fixed, and combined with the fixed cameras it was slightly worse (more gaps,
  Rhip accel p95 6.9 vs 2.0) and moved the shoulders by up to 32 mm.
- **2D interpolation of flagged keypoints** (`sapiens2_test/interp_flagged_kps.py`): replace flagged 2D points by linear
  interpolation, keep the camera in the set. Worse: it destabilised the stable-set selection.
- **Looser/tighter hip residual threshold**: 20 px skipped about 68% of hip landmark-frames, 40 px gave smoothed hip width
  std 51 mm from gap bridging, 60 px gives full coverage in 300-650.

## Not done (ideas)

- Silhouette check of hip depth: project the candidate hips into the masks and see which camera set keeps them inside the
  body at plausible depth. Cheapest independent reference for the bias above.
- Per-camera constant 2D hip offset estimated and subtracted (circular without an outside reference).
- Covariance-weighted Kalman with innovation gating (per-measurement geometric covariance, `meas_std_2cam` style
  inflation made continuous).
- Down-weight flagged observations instead of dropping them.

## Where things are

`sapiens2_test/` (helper scripts, still untracked) holds the flag and interpolation scripts and `view_compare.py`, which
shows several reconstructions in viser: `python -m sapiens2_test.view_compare --calib <calib> --set LABEL=<dir> ...`.
Variant outputs: `output/silbase` (baseline), `silflags`, `silinterp`, `silhipA` (hip cams), `silhipB` (adopted),
`silhipC` (B + flags).
