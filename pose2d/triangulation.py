"""3D triangulation from pose2d's extraction+tracking output (both hands +
body), reusing hand_pose.hand_multiview's RANSAC/DLT triangulation and
Kalman/RTS smoothing machinery -- the same primitives hand_pose/session_
triangulation.py already validated against this rig, just fed by pose2d's
richer (two-hand + body + occlusion/swap/body-trust-flagged) landmarks
instead of the older single-hand pipeline's.

Triangulates left hand, right hand, and body each as an independent
RANSAC-selected sequence -- triangulate_sequence_ransac doesn't care about
semantic meaning, only landmark_ids + landmarks/confidence/projection
matrices -- rather than one combined skeleton: a signer's left hand, right
hand, and torso don't need to share one per-frame camera-set decision, and
letting each part choose its own tends to use more of the rig's coverage
than forcing one shared set.

Usage:
    python -m pose2d.triangulation <take_dir> [--calib PATH] [--force]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import calibrate  # noqa: E402

from hand_pose import hand_multiview as hmv  # noqa: E402

LANDMARK_IDS_HAND = list(range(21))
REFERENCE_LANDMARK_ID_HAND = 0  # wrist

LANDMARK_IDS_BODY = list(range(13))  # COCO-17's upper body: nose/eyes/ears/shoulders/elbows/wrists/hips.
# Only knees/ankles (ids 13-16) are excluded -- hips (11-12) are kept as
# part of the upper body/torso. Legs are irrelevant for sign-language
# content and a source of unreliable detections (occluded by a desk,
# cropped out of frame) that add nothing but noise here. Confirmed via
# experiment that this doesn't change RANSAC camera selection for a take
# already dominated by good shoulder visibility (see extraction.py's
# confidence_body, which the same experiment showed excludes legs from its
# gating mean too), but there's no upside to triangulating/reporting them
# either, and it removes a class of bad data by construction rather than
# relying on that experiment's result holding for every future take.
REFERENCE_LANDMARK_ID_BODY = 5  # left_shoulder -- more consistently in-frame than hips for a torso-framed signer

DEFAULT_CALIB_PATH = os.path.join("output", "calibration", "20260903_153052_7cam.json")


def _try_load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_pose2d_take_data(take_dir, pose2d_dir="pose2d"):
    """pose2d_dir selects which detector's output to read: "pose2d"
    (MediaPipe, pose2d.run_pipeline's default) or "pose2d_dwpose" (DWPose,
    pose2d.run_pipeline_dwpose) -- same landmark/confidence schema either
    way, so nothing else here needs to know which detector produced it.
    """
    base = os.path.join(take_dir, "aligned", pose2d_dir)
    with open(os.path.join(base, "landmarks_left.json")) as f:
        landmarks_left = json.load(f)
    with open(os.path.join(base, "landmarks_right.json")) as f:
        landmarks_right = json.load(f)
    with open(os.path.join(base, "landmarks_body.json")) as f:
        landmarks_body = json.load(f)
    with open(os.path.join(base, "confidence_left.json")) as f:
        confidence_left = json.load(f)
    with open(os.path.join(base, "confidence_right.json")) as f:
        confidence_right = json.load(f)
    with open(os.path.join(base, "tracking_diagnostics.json")) as f:
        diagnostics = json.load(f)
    # confidence_body was computed by pose2d.tracking.run_tracking but
    # pose2d.run_pipeline never wrote it out on its own -- read it back from
    # the extraction cache instead of re-running the pipeline just for this.
    with open(os.path.join(base, "extraction.json")) as f:
        confidence_body = json.load(f)["confidence_body"]
    # Per-landmark confidence (e.g. DWPose's per-keypoint SimCC scores) --
    # optional, None when the detector/part doesn't have it (MediaPipe
    # Hands only ever gives one whole-hand score).
    landmark_confidence_left = _try_load(os.path.join(base, "landmark_confidence_left.json"))
    landmark_confidence_right = _try_load(os.path.join(base, "landmark_confidence_right.json"))
    landmark_confidence_body = _try_load(os.path.join(base, "landmark_confidence_body.json"))
    return {
        "landmarks_left": landmarks_left, "confidence_left": confidence_left,
        "landmarks_right": landmarks_right, "confidence_right": confidence_right,
        "landmarks_body": landmarks_body, "confidence_body": confidence_body,
        "landmark_confidence_left": landmark_confidence_left,
        "landmark_confidence_right": landmark_confidence_right,
        "landmark_confidence_body": landmark_confidence_body,
        "cam_ids": diagnostics["cam_ids"],
    }


def build_projection_matrices(calib, cam_ids):
    return {cam_id: hmv.build_projection_matrix(calib[cam_id]["K"], calib[cam_id]["R"], calib[cam_id]["t"])
            for cam_id in cam_ids}


def triangulate_part(
    landmarks, confidence, projection_matrices, width, height,
    landmark_ids, reference_landmark_id, reproj_thresh_px=15.0, min_confidence=0.5,
    landmark_confidence=None,
):
    """Runs the RANSAC-select -> triangulate -> smooth -> jitter-flag chain
    for one part (a hand or the body), independent of the other parts.

    landmark_confidence (optional): dict[cam_id][frame_key][lm_id] -> float,
    e.g. DWPose's per-keypoint SimCC scores. When given, triangulate_
    sequence_ransac's weighted DLT solve favors higher-confidence cameras'
    observations of each INDIVIDUAL landmark within the already-
    (geometrically) chosen inlier set -- the camera SET itself is still
    chosen purely by ransac_select_cameras_sequence's reprojection-error
    test, using `confidence` (the whole-part score) only for gating which
    cameras are even eligible. None (the default -- what MediaPipe Hands'
    single whole-hand score leaves you with) falls back to uniform
    weighting among inliers, same as before this parameter existed.

    Returns (raw_reconstruction, smoothed_reconstruction, selection,
    reproj_errors, jitter_flagged_frame_keys).
    """
    selection = hmv.ransac_select_cameras_sequence(
        landmarks, confidence, projection_matrices, width, height,
        reference_landmark_id=reference_landmark_id, reproj_thresh_px=reproj_thresh_px,
        min_confidence=min_confidence,
    )
    raw_reconstruction = hmv.triangulate_sequence_ransac(
        landmarks, projection_matrices, width, height, landmark_ids, selection,
        landmark_confidence=landmark_confidence,
    )
    frame_keys = sorted(selection.keys(), key=lambda k: int(os.path.splitext(k)[0]))
    n_inliers_by_frame = {f: sel["n_inliers"] for f, sel in selection.items()}
    smoothed_reconstruction = hmv.smooth_reconstruction_sequence(
        raw_reconstruction, frame_keys, landmark_ids, n_inliers_by_frame,
    )
    jitter_flags, _jitter_residuals = hmv.flag_jitter_frames(
        raw_reconstruction, smoothed_reconstruction, frame_keys, reference_landmark_id=reference_landmark_id,
    )
    reproj_errors = hmv.ransac_per_camera_reproj_error(
        landmarks, confidence, raw_reconstruction, projection_matrices, width, height,
        reference_landmark_id=reference_landmark_id, min_confidence=min_confidence,
    )
    return raw_reconstruction, smoothed_reconstruction, selection, reproj_errors, jitter_flags


def summarize_part(name, total_frames, selection, reproj_errors, jitter_flags):
    inlier_sizes = [sel["n_inliers"] for sel in selection.values()]
    flat_reproj = [err for by_frame in reproj_errors.values() for err in by_frame.values()]
    metrics = {
        "total_frames": total_frames,
        "ransac_success_frames": len(selection),
        "ransac_success_rate": len(selection) / total_frames if total_frames else 0.0,
        "mean_inlier_set_size": float(np.mean(inlier_sizes)) if inlier_sizes else None,
        "reproj_error_px": {
            "mean": float(np.mean(flat_reproj)) if flat_reproj else None,
            "median": float(np.median(flat_reproj)) if flat_reproj else None,
            "p95": float(np.percentile(flat_reproj, 95)) if flat_reproj else None,
        },
        "jitter_flagged_frames": len(jitter_flags),
        "jitter_fraction": len(jitter_flags) / len(selection) if selection else 0.0,
    }
    print(f"\n  [{name}]")
    print(f"    RANSAC camera-set found: {metrics['ransac_success_frames']}/{total_frames} "
          f"({metrics['ransac_success_rate']:.1%})")
    if metrics["mean_inlier_set_size"] is not None:
        print(f"    mean inlier-set size: {metrics['mean_inlier_set_size']:.2f} cameras")
    rp = metrics["reproj_error_px"]
    if rp["mean"] is not None:
        print(f"    reprojection error (px): mean={rp['mean']:.2f} median={rp['median']:.2f} p95={rp['p95']:.2f}")
    print(f"    jitter-flagged frames: {metrics['jitter_flagged_frames']} ({metrics['jitter_fraction']:.1%})")
    return metrics


def run_take(take_dir, calib_path, force, out_dir=None, pose2d_dir="pose2d"):
    """Reads pose2d's already-extracted 2D landmarks from `take_dir`'s
    `pose2d_dir` subfolder (never re-extracts -- calibration only changes
    the triangulation projection matrices, not the 2D detection). Pass
    pose2d_dir="pose2d_dwpose" to triangulate DWPose's output instead of
    MediaPipe's -- same schema either way (see load_pose2d_take_data).

    Writes reconstruction/diagnostics files under `out_dir` if given
    (default: take_dir's own aligned/<pose2d_dir>/, the normal in-place
    behavior, so DWPose and MediaPipe results coexist without collision) --
    so trying an alternate calibration doesn't clobber a take's real
    results either. `out_dir`, if given, is treated as a stand-in take_dir:
    its aligned/<pose2d_dir>/ subfolder gets the output, in the same
    reconstruction_<part>.json layout pose2d.visualize_triangulation
    already expects, so the viewer can point at it directly with no
    changes.
    """
    out_dir = os.path.join(out_dir if out_dir else take_dir, "aligned", pose2d_dir)
    os.makedirs(out_dir, exist_ok=True)
    paths = {
        part: (os.path.join(out_dir, f"reconstruction_{part}.json"), os.path.join(out_dir, f"triangulation_diagnostics_{part}.json"))
        for part in ("left", "right", "body")
    }
    if not force and all(os.path.exists(p[0]) for p in paths.values()):
        print(f"Reconstructions already exist for {take_dir} -- pass --force to regenerate")
        return

    calib = calibrate.load_calibration_output(calib_path)
    data = load_pose2d_take_data(take_dir, pose2d_dir=pose2d_dir)
    cam_ids = [c for c in data["cam_ids"] if c in calib]
    if len(cam_ids) < 2:
        raise RuntimeError(f"{take_dir}: only {len(cam_ids)} cameras have real extrinsics in {calib_path}")

    projection_matrices = build_projection_matrices(calib, cam_ids)
    sample_frame = next(iter(data["landmarks_body"].get(cam_ids[0], {})), None)
    if sample_frame is None:
        sample_frame = next(iter(data["landmarks_left"].get(cam_ids[0], {})), None)
    width, height = hmv.get_image_size(take_dir, cam_ids[0], sample_frame)
    total_frames = len(hmv.discover_frames(take_dir, cam_ids[0]))

    print(f"\n=== {take_dir} (aligned/{pose2d_dir}) ===")
    print(f"  cameras: {len(cam_ids)}, frames per camera: {total_frames}")

    parts = {
        "left": (data["landmarks_left"], data["confidence_left"], LANDMARK_IDS_HAND, REFERENCE_LANDMARK_ID_HAND,
                  data["landmark_confidence_left"]),
        "right": (data["landmarks_right"], data["confidence_right"], LANDMARK_IDS_HAND, REFERENCE_LANDMARK_ID_HAND,
                   data["landmark_confidence_right"]),
        "body": (data["landmarks_body"], data["confidence_body"], LANDMARK_IDS_BODY, REFERENCE_LANDMARK_ID_BODY,
                  data["landmark_confidence_body"]),
    }

    for part_name, (landmarks, confidence, landmark_ids, reference_landmark_id, landmark_confidence) in parts.items():
        raw, smoothed, selection, reproj_errors, jitter_flags = triangulate_part(
            landmarks, confidence, projection_matrices, width, height, landmark_ids, reference_landmark_id,
            landmark_confidence=landmark_confidence,
        )
        metrics = summarize_part(part_name, total_frames, selection, reproj_errors, jitter_flags)
        metrics["per_landmark_confidence_used"] = landmark_confidence is not None
        print(f"    per-landmark confidence weighting: {'yes' if landmark_confidence is not None else 'no (uniform)'}")

        recon_path, diag_path = paths[part_name]
        with open(recon_path, "w") as f:
            json.dump({"raw": raw, "smoothed": smoothed}, f)
        with open(diag_path, "w") as f:
            json.dump({
                "metrics": metrics,
                "per_frame_selection": selection,
                "jitter_flagged_frame_keys": sorted(jitter_flags),
            }, f, indent=2)
        print(f"    wrote: {recon_path}")
        print(f"    wrote: {diag_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("take_dirs", nargs="+")
    parser.add_argument("--calib", default=DEFAULT_CALIB_PATH)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--out-dir", default=None,
        help="Write reconstruction/diagnostics here instead of <take_dir>/aligned/<pose2d-dir>/ "
             "(a stand-in take_dir -- its aligned/<pose2d-dir>/ subfolder gets the output). "
             "Use this to try an alternate --calib without overwriting a take's real results.",
    )
    parser.add_argument(
        "--pose2d-dir", default="pose2d",
        help="Which detector's 2D output to triangulate: 'pose2d' (MediaPipe, default) or "
             "'pose2d_dwpose' (DWPose) -- both live under <take_dir>/aligned/.",
    )
    args = parser.parse_args()
    for take_dir in args.take_dirs:
        run_take(take_dir, args.calib, args.force, out_dir=args.out_dir, pose2d_dir=args.pose2d_dir)


if __name__ == "__main__":
    main()
