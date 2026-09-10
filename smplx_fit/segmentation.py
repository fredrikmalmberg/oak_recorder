"""Person segmentation masks for one take's already-aligned camera frames --
Phase 2 of the EasyMoCap-inspired plan (see the approved plan / conversation
history), a standalone prerequisite for Phase 3's silhouette-based fitting
refinement. Deliberately decoupled from fitting itself, mirroring this
project's established extraction-is-separate-from-fitting convention
(pose2d/extraction.py vs pose2d/triangulation.py): mask quality can be
inspected and trusted on its own before anything downstream ever depends on
it.

Uses mediapipe.solutions.selfie_segmentation.SelfieSegmentation -- a real,
dedicated person-segmentation model already available via the mediapipe
dependency this project uses throughout (no new dependency), per the
explicit "use a segmentation model, not background subtraction" decision
(background subtraction would be fragile across this project's varying
per-take backgrounds/lighting).

Usage:
    python -m smplx_fit.segmentation <take_dir> --calib PATH [--force]

Writes <take_dir>/aligned/masks/<cam_id>/<frame_file> (soft probability
mask, 0-255 grayscale PNG, same filename as the source frame) -- following
the same per-camera/per-frame file layout convention as aligned/<cam_id>/
frames themselves.
"""
import argparse
import os

import cv2
import mediapipe as mp
import numpy as np

from hand_pose import hand_multiview as hmv

# SelfieSegmentation's own two models: 0 ("general", works at any distance)
# vs 1 ("landscape", optimized for a person filling most of the frame, faster).
# This rig's frames are close-range upper-body/hands (sign-language content),
# closer to "landscape" model's intended use case than a generic full-scene
# selfie -- confirmed reasonable via the plausible-coverage-fraction check
# this module's own verification step performs, not assumed blindly.
MODEL_SELECTION = 1


def extract_masks_for_take(take_dir, calib, cam_ids=None, force=False):
    """Runs SelfieSegmentation over each camera's already-undistorted
    aligned/<cam_id>/*.jpg frames -- the SAME undistorted images pose2d.
    extraction already uses, for consistency with how keypoints were
    triangulated (a mask computed on a differently-distorted image would
    misalign with the keypoint-based camera projections used elsewhere in
    this pipeline, e.g. hand_pose.hand_multiview.build_projection_matrix in
    Phase 3).

    Returns dict {cam_id: {frame_file: mask_path}} of what was written (or
    already cached).
    """
    out_root = os.path.join(take_dir, "aligned", "masks")
    if cam_ids is None:
        cam_ids = hmv.discover_cameras(take_dir)
    cam_ids = [c for c in cam_ids if c in calib]

    written = {cid: {} for cid in cam_ids}
    for cam_id in cam_ids:
        frame_files = hmv.discover_frames(take_dir, cam_id)
        if not frame_files:
            continue
        cam_out_dir = os.path.join(out_root, cam_id)
        os.makedirs(cam_out_dir, exist_ok=True)

        c = calib[cam_id]
        w, h = hmv.get_image_size(take_dir, cam_id, frame_files[0])
        map1, map2 = hmv.build_undistort_maps(c["K"], c["dist"], w, h)

        n_to_process = [
            ff for ff in frame_files
            if force or not os.path.exists(os.path.join(cam_out_dir, ff))
        ]
        if not n_to_process:
            for ff in frame_files:
                written[cam_id][ff] = os.path.join(cam_out_dir, ff)
            print(f"  {cam_id}: {len(frame_files)} masks already cached, skipping.")
            continue

        segmenter = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=MODEL_SELECTION)
        try:
            for frame_file in n_to_process:
                frame_path = os.path.join(take_dir, "aligned", cam_id, frame_file)
                img_bgr = cv2.imread(frame_path)
                if img_bgr is None:
                    continue
                img_bgr = hmv.undistort_fast(img_bgr, map1, map2)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                result = segmenter.process(img_rgb)
                mask_u8 = np.clip(result.segmentation_mask * 255.0, 0, 255).astype(np.uint8)

                mask_path = os.path.join(cam_out_dir, frame_file)
                cv2.imwrite(mask_path, mask_u8)
                written[cam_id][frame_file] = mask_path
        finally:
            segmenter.close()
        print(f"  {cam_id}: {len(n_to_process)} masks written to {cam_out_dir}")

    return written


SAM2_CKPT = "/home/fmalmb/CODE/sam2/checkpoints/sam2.1_hiera_large.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l"
SAM2_SYSPATH = "/home/fmalmb/CODE/sam2"


def _sam2_prompt_from_mask(mask_dir, frame_files, min_coverage=0.02):
    """Find the first frame with an existing mask and return (frame_idx, click_xy).

    The click is the centroid of the mask — the person's position varies per camera
    so a fixed center-frame click doesn't work. Using the existing MediaPipe mask
    centroid as the SAM2 starting point is fine: SAM2 produces its own higher-quality,
    temporally-consistent mask; MediaPipe only tells us *where* to click.

    Falls back to the midpoint of the first frame if no mask has sufficient coverage.
    """
    for i, ff in enumerate(frame_files):
        mask_path = os.path.join(mask_dir, ff)
        m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        binary = m > 127
        if binary.mean() < min_coverage:
            continue
        # Centroid of the mask
        ys, xs = np.where(binary)
        cx, cy = float(xs.mean()), float(ys.mean())
        return i, np.array([[cx, cy]], dtype=np.float32)
    return 0, None  # fallback: caller will use frame-center


def extract_masks_sam2(take_dir, calib, cam_ids=None, force=False, device="cuda"):
    """Like extract_masks_for_take but uses SAM2 video predictor for higher-quality,
    temporally-consistent person masks.

    For the initial click prompt, uses the centroid of the first existing MediaPipe
    mask in aligned/masks/<cam_id>/ with >2% coverage — the person's position varies
    across cameras, so a fixed center-click doesn't work. If no prior mask exists,
    falls back to frame-center (adequate when the person is roughly centered).

    Propagates bidirectionally from the prompt frame to cover all frames.

    Writes to the same aligned/masks/<cam_id>/<frame_file> schema as
    extract_masks_for_take -- the rest of the pipeline reads from the same path.

    Requires SAM2 installed at SAM2_SYSPATH and checkpoint at SAM2_CKPT.
    """
    import sys
    if SAM2_SYSPATH not in sys.path:
        sys.path.insert(0, SAM2_SYSPATH)
    import torch
    from sam2.build_sam import build_sam2_video_predictor

    out_root = os.path.join(take_dir, "aligned", "masks")
    if cam_ids is None:
        cam_ids = hmv.discover_cameras(take_dir)
    cam_ids = [c for c in cam_ids if c in calib]

    predictor = build_sam2_video_predictor(SAM2_CONFIG, ckpt_path=SAM2_CKPT, device=device)

    written = {cid: {} for cid in cam_ids}
    for cam_id in cam_ids:
        frame_files = hmv.discover_frames(take_dir, cam_id)
        if not frame_files:
            continue
        cam_out_dir = os.path.join(out_root, cam_id)
        os.makedirs(cam_out_dir, exist_ok=True)

        if not force and all(os.path.exists(os.path.join(cam_out_dir, ff)) for ff in frame_files):
            for ff in frame_files:
                written[cam_id][ff] = os.path.join(cam_out_dir, ff)
            print(f"  {cam_id}: {len(frame_files)} SAM2 masks already cached, skipping.")
            continue

        frame_dir = os.path.join(take_dir, "aligned", cam_id)
        first_img = cv2.imread(os.path.join(frame_dir, frame_files[0]))
        if first_img is None:
            print(f"  {cam_id}: could not read first frame, skipping.")
            continue
        h, w = first_img.shape[:2]

        prompt_frame_idx, prompt_pt = _sam2_prompt_from_mask(cam_out_dir, frame_files)
        if prompt_pt is None:
            prompt_pt = np.array([[w / 2.0, h / 2.0]], dtype=np.float32)
        print(f"  {cam_id}: prompt click at ({prompt_pt[0,0]:.0f},{prompt_pt[0,1]:.0f}) "
              f"on frame {prompt_frame_idx} ({frame_files[prompt_frame_idx]})")

        cam_masks = {}
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            inference_state = predictor.init_state(video_path=frame_dir)
            try:
                predictor.add_new_points_or_box(
                    inference_state, frame_idx=prompt_frame_idx, obj_id=1,
                    points=prompt_pt, labels=np.array([1], dtype=np.int32),
                )
                # Propagate forward from prompt frame
                for frame_idx, _obj_ids, mask_logits in predictor.propagate_in_video(
                    inference_state, start_frame_idx=prompt_frame_idx, reverse=False
                ):
                    cam_masks[frame_idx] = (mask_logits[0, 0] > 0).cpu().numpy().astype(np.uint8) * 255
                # Propagate backward to cover frames before the prompt
                if prompt_frame_idx > 0:
                    for frame_idx, _obj_ids, mask_logits in predictor.propagate_in_video(
                        inference_state, start_frame_idx=prompt_frame_idx, reverse=True
                    ):
                        cam_masks[frame_idx] = (mask_logits[0, 0] > 0).cpu().numpy().astype(np.uint8) * 255
            finally:
                predictor.reset_state(inference_state)

        for frame_idx, mask_u8 in cam_masks.items():
            ff = frame_files[frame_idx]
            mask_path = os.path.join(cam_out_dir, ff)
            cv2.imwrite(mask_path, mask_u8)
            written[cam_id][ff] = mask_path

        print(f"  {cam_id}: {len(cam_masks)} SAM2 masks written to {cam_out_dir}")

    return written


def summarize_mask_coverage(take_dir, written, threshold=127):
    """Sanity check per the approved plan's Phase 2 verification: spot-check
    that mask pixel-coverage fraction per frame is plausible (a person
    filling some meaningful but not all/none of the frame), not a
    degenerate all-black/all-white output that would silently corrupt
    Phase 3 without this check catching it first.
    """
    fractions = []
    for cam_id, by_frame in written.items():
        for frame_file, mask_path in by_frame.items():
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            fractions.append(float((mask > threshold).mean()))
    if not fractions:
        print("No masks found to summarize.")
        return None
    fractions = np.array(fractions)
    print(f"\n=== Mask coverage summary ({len(fractions)} frames) ===")
    print(f"  mean: {fractions.mean():.1%}  min: {fractions.min():.1%}  max: {fractions.max():.1%}")
    n_degenerate = int(((fractions < 0.01) | (fractions > 0.95)).sum())
    if n_degenerate:
        print(f"  WARNING: {n_degenerate}/{len(fractions)} frames have degenerate "
              f"(near-0% or near-100%) coverage -- inspect these before trusting "
              f"this take's masks for Phase 3.")
    else:
        print("  no degenerate (near-0%/near-100%) frames found.")
    return {
        "mean_coverage": float(fractions.mean()),
        "min_coverage": float(fractions.min()),
        "max_coverage": float(fractions.max()),
        "n_degenerate": n_degenerate,
        "n_frames": len(fractions),
    }


MASK_TINT_COLOR = (0, 200, 0)  # BGR green -- distinct from grid_video.py's body/hand palette


def render_mask_grid_video(
    take_dir, calib, cam_ids=None, output_path=None,
    fps=15.0, cols=3, canvas_size=(1920, 1080), alpha=0.45, mask_threshold=127,
):
    """Structurally mirrors pose2d.grid_video.render_pose2d_grid_video (grid
    math, per-camera undistortion, leftover-slot legend panel) -- this IS
    the deliverable for judging segmentation-mask quality by eye, same
    inspection-first convention as that module's own docstring describes
    for 2D keypoints. Tints each camera's undistorted frame green wherever
    that camera's cached mask exceeds `mask_threshold`, with the per-frame
    coverage fraction printed in-frame so degenerate frames (see
    summarize_mask_coverage) are easy to spot by eye too, not just by the
    aggregate stats.
    """
    if cam_ids is None:
        cam_ids = hmv.discover_cameras(take_dir)
    cam_ids = [c for c in cam_ids if c in calib]
    if output_path is None:
        output_path = os.path.join(take_dir, "aligned", "masks", "mask_grid_video.mp4")

    frame_keys = hmv.discover_frames(take_dir, cam_ids[0])
    undistort_maps = {}
    for cam_id in cam_ids:
        c = calib[cam_id]
        w, h = hmv.get_image_size(take_dir, cam_id, frame_keys[0])
        undistort_maps[cam_id] = (w, h, hmv.build_undistort_maps(c["K"], c["dist"], w, h))

    canvas_w, canvas_h = canvas_size
    rows = int(np.ceil(len(cam_ids) / cols))
    slot_w, slot_h = canvas_w // cols, canvas_h // rows
    empty_slots = list(range(len(cam_ids), rows * cols))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (canvas_w, canvas_h))
    try:
        for idx, frame_key in enumerate(frame_keys):
            canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

            for cam_idx, cam_id in enumerate(cam_ids):
                frame_path = os.path.join(take_dir, "aligned", cam_id, frame_key)
                img = cv2.imread(frame_path)
                if img is None:
                    continue
                w, h, (map1, map2) = undistort_maps[cam_id]
                img = hmv.undistort_fast(img, map1, map2)

                mask_path = os.path.join(take_dir, "aligned", "masks", cam_id, frame_key)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                coverage_pct = None
                if mask is not None:
                    binary = mask > mask_threshold
                    coverage_pct = float(binary.mean()) * 100.0
                    tinted = img.copy()
                    tinted[binary] = MASK_TINT_COLOR
                    img = cv2.addWeighted(tinted, alpha, img, 1.0 - alpha, 0)

                resized = cv2.resize(img, (slot_w, slot_h), interpolation=cv2.INTER_AREA)
                cv2.putText(
                    resized, cam_id, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA,
                )
                if coverage_pct is not None:
                    cv2.putText(
                        resized, f"{coverage_pct:.1f}%", (10, slot_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2, cv2.LINE_AA,
                    )
                else:
                    cv2.putText(
                        resized, "no mask", (10, slot_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255), 2, cv2.LINE_AA,
                    )

                row, col = cam_idx // cols, cam_idx % cols
                y1, x1 = row * slot_h, col * slot_w
                canvas[y1:y1 + slot_h, x1:x1 + slot_w] = resized

            if empty_slots:
                panel = np.zeros((slot_h, slot_w, 3), dtype=np.uint8)
                lines = [
                    "LEGEND", "green tint = segmented person",
                    f"tint alpha: {alpha}", f"mask threshold: {mask_threshold}/255",
                    "red 'no mask' = mask not found for this frame",
                ]
                y = 28
                for line in lines:
                    cv2.putText(panel, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
                    y += 27
                row, col = empty_slots[0] // cols, empty_slots[0] % cols
                canvas[row * slot_h:(row + 1) * slot_h, col * slot_w:(col + 1) * slot_w] = panel

            header = f"frame {idx} ({frame_key})"
            cv2.putText(canvas, header, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)

            writer.write(canvas)
            if (idx + 1) % 50 == 0 or (idx + 1) == len(frame_keys):
                print(f"  Progress: {idx + 1}/{len(frame_keys)} grid frames compiled.")
    finally:
        writer.release()
        print(f"Saved mask grid video to {os.path.abspath(output_path)}")
    return output_path


def main():
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import calibrate  # noqa: E402

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("take_dir")
    parser.add_argument("--calib", default=os.path.join("output", "calibration", "20260903_153052_7cam.json"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--sam2", action="store_true",
                         help="Use SAM2 video predictor instead of MediaPipe SelfieSegmentation. "
                              "Requires SAM2 installed at %(default)s.",
                         default=False)
    parser.add_argument("--grid-video", action="store_true",
                         help="Also render aligned/masks/mask_grid_video.mp4 for visual inspection.")
    parser.add_argument("--fps", type=float, default=15.0)
    args = parser.parse_args()

    calib = calibrate.load_calibration_output(args.calib)
    if args.sam2:
        written = extract_masks_sam2(args.take_dir, calib, force=args.force)
    else:
        written = extract_masks_for_take(args.take_dir, calib, force=args.force)
    summarize_mask_coverage(args.take_dir, written)
    if args.grid_video:
        render_mask_grid_video(args.take_dir, calib, fps=args.fps)


if __name__ == "__main__":
    main()
