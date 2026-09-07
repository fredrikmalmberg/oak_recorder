"""Batch multi-view MediaPipe hand triangulation over one or more recorded,
aligned takes -- Step 2 of the plan at
C:\\Users\\signbot\\.claude\\plans\\calm-honking-sphinx.md.

For each `<take_dir>` (must already have `aligned/<cam>/%06d.jpg` from
align_session.py):
  1. Loads this rig's own calibration via calibrate.load_calibration_output
     (per-camera K/dist/R/t -- unlike the h5 rig's averaged-intrinsics path).
  2. Runs MediaPipe hand-landmark extraction per camera, undistorted with
     that camera's own K/dist (hand_multiview.
     extract_hand_landmarks_for_session_multicam), cached to aligned/
     landmarks.json + confidence.json.
  3. RANSAC-selects a per-frame camera set and triangulates every landmark
     from it, either at each camera's nominal aligned slot (default) or,
     with --frame-offsets, allowing each camera to be considered at
     neighboring slots too (hand_multiview.
     ransac_select_cameras_sequence[_multi_offset]).
  4. Applies Kalman/RTS smoothing + jitter flagging.
  5. Writes reconstruction + diagnostics JSON into `<take_dir>/aligned/`,
     and prints a quantitative summary (no images/video are ever opened by
     this script or by the model running it).

Usage:
    python -m hand_pose.session_triangulation <take_dir> [<take_dir> ...] \\
        [--calib PATH] [--frame-offsets 0 | -1,0,1] [--force-landmarks]
"""
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import calibrate  # noqa: E402

from hand_pose import hand_multiview as hmv  # noqa: E402


LANDMARK_IDS = list(range(21))
REFERENCE_LANDMARK_ID = 0  # wrist
DEFAULT_CALIB_PATH = os.path.join(
    "output", "calibration", "20260903_153052_7cam.json"
)


def build_projection_matrices(calib):
    return {cam_id: hmv.build_projection_matrix(c["K"], c["R"], c["t"]) for cam_id, c in calib.items()}


def broadcast_confidence(confidence, landmark_ids):
    """MediaPipe Hands gives one detection-level confidence per (cam, frame)
    -- extract_hand_landmarks_for_session_multicam's `confidence` schema --
    not per-landmark. triangulate_sequence_ransac's `landmark_confidence`
    weighting expects dict[cam][frame][lm_id], so broadcast that single
    score across every landmark id for the weighted DLT solve.
    """
    lm_keys = [str(lm_id) for lm_id in landmark_ids]
    return {
        cam_id: {frame: {lm_key: score for lm_key in lm_keys} for frame, score in by_frame.items()}
        for cam_id, by_frame in confidence.items()
    }


def run_take(take_dir, calib, frame_offsets, force_landmarks, reproj_thresh_px, min_confidence):
    all_aligned_cams = hmv.discover_cameras(take_dir)
    cam_ids = [c for c in all_aligned_cams if c in calib]
    if len(cam_ids) < 2:
        raise RuntimeError(
            f"{take_dir}: only {len(cam_ids)}/{len(all_aligned_cams)} aligned cameras have real "
            f"extrinsics in the calibration file -- need >=2 to triangulate"
        )

    landmarks, confidence = hmv.extract_hand_landmarks_for_session_multicam(
        take_dir, calib, cam_ids=cam_ids, force=force_landmarks,
    )

    projection_matrices = build_projection_matrices({c: calib[c] for c in cam_ids})
    sample_frame = hmv.discover_frames(take_dir, cam_ids[0])[0]
    width, height = hmv.get_image_size(take_dir, cam_ids[0], sample_frame)

    landmark_confidence = broadcast_confidence(confidence, LANDMARK_IDS)

    use_multi_offset = tuple(frame_offsets) != (0,)
    if use_multi_offset:
        selection = hmv.ransac_select_cameras_sequence_multi_offset(
            landmarks, confidence, projection_matrices, width, height,
            frame_offsets=tuple(frame_offsets), reference_landmark_id=REFERENCE_LANDMARK_ID,
            reproj_thresh_px=reproj_thresh_px, min_confidence=min_confidence,
        )
        raw_reconstruction = hmv.triangulate_sequence_ransac_multi_offset(
            landmarks, projection_matrices, width, height, LANDMARK_IDS, selection,
            landmark_confidence=landmark_confidence,
        )
    else:
        selection = hmv.ransac_select_cameras_sequence(
            landmarks, confidence, projection_matrices, width, height,
            reference_landmark_id=REFERENCE_LANDMARK_ID,
            reproj_thresh_px=reproj_thresh_px, min_confidence=min_confidence,
        )
        raw_reconstruction = hmv.triangulate_sequence_ransac(
            landmarks, projection_matrices, width, height, LANDMARK_IDS, selection,
            landmark_confidence=landmark_confidence,
        )

    frame_keys = sorted(selection.keys())
    n_inliers_by_frame = {f: sel["n_inliers"] for f, sel in selection.items()}
    smoothed_reconstruction = hmv.smooth_reconstruction_sequence(
        raw_reconstruction, frame_keys, LANDMARK_IDS, n_inliers_by_frame,
    )
    jitter_flags, jitter_residuals = hmv.flag_jitter_frames(
        raw_reconstruction, smoothed_reconstruction, frame_keys,
        reference_landmark_id=REFERENCE_LANDMARK_ID,
    )
    reproj_errors = hmv.ransac_per_camera_reproj_error(
        landmarks, confidence, raw_reconstruction, projection_matrices, width, height,
        reference_landmark_id=REFERENCE_LANDMARK_ID, min_confidence=min_confidence,
    )

    total_frames = len(hmv.discover_frames(take_dir, cam_ids[0]))
    all_frame_keys = sorted({f for cam in landmarks for f in landmarks[cam]})
    ge2_detection_frames = 0
    for frame in {f for cam in landmarks.values() for f in cam}:
        n_confident = sum(
            1 for cam_id in cam_ids
            if landmarks.get(cam_id, {}).get(frame, {}).get(str(REFERENCE_LANDMARK_ID)) is not None
            and confidence.get(cam_id, {}).get(frame, 0) >= min_confidence
        )
        if n_confident >= 2:
            ge2_detection_frames += 1

    flat_reproj = [err for by_frame in reproj_errors.values() for err in by_frame.values()]
    inlier_sizes = [sel["n_inliers"] for sel in selection.values()]

    metrics = {
        "cam_ids": cam_ids,
        "total_frames": total_frames,
        "frames_with_ge2_confident_detections": ge2_detection_frames,
        "detection_rate": ge2_detection_frames / total_frames if total_frames else 0.0,
        "ransac_success_frames": len(selection),
        "ransac_success_rate": len(selection) / total_frames if total_frames else 0.0,
        "mean_inlier_set_size": float(np.mean(inlier_sizes)) if inlier_sizes else None,
        "reproj_error_px": {
            "mean": float(np.mean(flat_reproj)) if flat_reproj else None,
            "median": float(np.median(flat_reproj)) if flat_reproj else None,
            "p95": float(np.percentile(flat_reproj, 95)) if flat_reproj else None,
        },
        "jitter_flagged_frames": len(jitter_flags),
        "jitter_fraction": len(jitter_flags) / len(frame_keys) if frame_keys else 0.0,
    }

    offset_tally = None
    if use_multi_offset:
        offset_tally = {cam_id: Counter() for cam_id in cam_ids}
        for sel in selection.values():
            for cam_id, offset in sel.get("offsets", {}).items():
                offset_tally[cam_id][offset] += 1
        metrics["offset_tally"] = {
            cam_id: dict(counter) for cam_id, counter in offset_tally.items()
        }

    diagnostics = {
        "take_dir": take_dir,
        "frame_offsets": list(frame_offsets),
        "reproj_thresh_px": reproj_thresh_px,
        "min_confidence": min_confidence,
        "metrics": metrics,
        "per_frame_selection": selection,
        "jitter_flagged_frame_keys": sorted(jitter_flags),
    }

    suffix = "_multioffset" if use_multi_offset else ""
    aligned_dir = os.path.join(take_dir, "aligned")
    reconstruction_path = os.path.join(aligned_dir, f"reconstruction{suffix}.json")
    diagnostics_path = os.path.join(aligned_dir, f"reconstruction_diagnostics{suffix}.json")

    with open(reconstruction_path, "w") as f:
        json.dump({"raw": raw_reconstruction, "smoothed": smoothed_reconstruction}, f)
    with open(diagnostics_path, "w") as f:
        json.dump(diagnostics, f, indent=2)

    return metrics, reconstruction_path, diagnostics_path


def print_metrics(take_dir, label, metrics):
    print(f"\n=== {take_dir} [{label}] ===")
    print(f"  cameras with extrinsics: {len(metrics['cam_ids'])} ({', '.join(metrics['cam_ids'])})")
    print(f"  total frames: {metrics['total_frames']}")
    print(
        f"  frames with >=2 confident detections: {metrics['frames_with_ge2_confident_detections']} "
        f"({metrics['detection_rate']:.1%})"
    )
    print(
        f"  RANSAC camera-set found: {metrics['ransac_success_frames']} frames "
        f"({metrics['ransac_success_rate']:.1%})"
    )
    if metrics["mean_inlier_set_size"] is not None:
        print(f"  mean inlier-set size: {metrics['mean_inlier_set_size']:.2f} cameras")
    rp = metrics["reproj_error_px"]
    if rp["mean"] is not None:
        print(f"  reprojection error (px): mean={rp['mean']:.2f} median={rp['median']:.2f} p95={rp['p95']:.2f}")
    print(f"  jitter-flagged frames: {metrics['jitter_flagged_frames']} ({metrics['jitter_fraction']:.1%})")
    if "offset_tally" in metrics:
        print("  per-camera offset preference (frame count by offset):")
        for cam_id, tally in metrics["offset_tally"].items():
            tally_str = ", ".join(f"{k:+d}:{v}" for k, v in sorted(tally.items()))
            print(f"    {cam_id}: {tally_str}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("take_dirs", nargs="+", help="One or more take directories (must have aligned/)")
    parser.add_argument("--calib", default=DEFAULT_CALIB_PATH, help="Calibration JSON path")
    parser.add_argument(
        "--frame-offsets", default="0",
        help="Comma-separated frame offsets to consider per camera, e.g. '0' (default, baseline) "
             "or '-1,0,1' (the multi-offset RANSAC variant)",
    )
    parser.add_argument("--force-landmarks", action="store_true", help="Re-run MediaPipe even if cached")
    parser.add_argument("--reproj-thresh-px", type=float, default=15.0)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    args = parser.parse_args()

    frame_offsets = tuple(int(x) for x in args.frame_offsets.split(","))
    label = "baseline" if frame_offsets == (0,) else f"offsets={frame_offsets}"

    calib = calibrate.load_calibration_output(args.calib)
    print(f"Loaded calibration: {len(calib)} camera(s) with real extrinsics from {args.calib}")

    for take_dir in args.take_dirs:
        metrics, recon_path, diag_path = run_take(
            take_dir, calib, frame_offsets, args.force_landmarks,
            args.reproj_thresh_px, args.min_confidence,
        )
        print_metrics(take_dir, label, metrics)
        print(f"  wrote: {recon_path}")
        print(f"  wrote: {diag_path}")


if __name__ == "__main__":
    main()
