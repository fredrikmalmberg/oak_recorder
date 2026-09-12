"""Person segmentation masks for one take's already-aligned camera frames.
Phase 2 of the pipeline: a standalone prerequisite for Phase 3's
silhouette-based fitting refinement. Deliberately decoupled from fitting
itself (extraction-is-separate-from-fitting convention).

Two backends:
  SAM3  — text prompt "person" on frame 0, propagated via video predictor.
           No clicking, no MediaPipe needed. Slow (~7 min/cam) but requires
           no prior 3D reconstruction. Use for first-pass mask generation.
  RVM   — Robust Video Matting with keypoint-guided ROI crop. Fast (<10s/cam
           on 4090 fp16). Requires triangulated 3D keypoints in pose2d_dir.

Usage:
    python -m smplx_fit.segmentation <take_dir> --calib PATH [--force]
        [--mask-subdir masks_sam3|masks_rvm] [--backend sam3|rvm] [--grid-video]

Writes <take_dir>/aligned/<mask_subdir>/<cam_id>/<frame_file>
(grayscale JPEG 0–255, same filename as the source frame).
"""
import argparse
import json
import os

import cv2
import numpy as np

from hand_pose import hand_multiview as hmv

SAM3_SYSPATH = "/home/fmalmb/CODE/sam3"
# Must use "sam3" not "sam3.1":
# sam3.1 uses Sam3MultiplexTrackingWithInteractivity whose init_state lacks
# offload_state_to_cpu AND whose add_prompt triggers sam3/perflib/fa3.py which
# requires flash_attn_interface (FlashAttention 3) — not installed in oak_env.
# sam3 uses Sam3VideoInferenceWithInstanceInteractivity which has neither issue
# and runs fine with standard SDPA. Do not change this to sam3.1 without
# first installing flash_attn_interface in oak_env.
SAM3_VERSION = "sam3"
SAM3_TEXT_PROMPT = "person"

RVM_SYSPATH = "/home/fmalmb/CODE/rvm"
RVM_CHECKPOINT_MOBILENET = "/home/fmalmb/CODE/rvm/checkpoints/rvm_mobilenetv3.pth"
RVM_CHECKPOINT_RESNET50  = "/home/fmalmb/CODE/rvm/checkpoints/rvm_resnet50.pth"


def extract_masks_sam3(take_dir, calib, cam_ids=None, force=False, mask_subdir="masks_sam3"):
    """Run SAM3 video predictor over each camera's aligned frames.

    Prompts with text "person" on frame 0, propagates through all frames.
    Per-camera: loads the model once, processes each camera sequentially.

    Returns dict {cam_id: {frame_file: mask_path}} of what was written.
    """
    import sys
    import torch
    if SAM3_SYSPATH not in sys.path:
        sys.path.insert(0, SAM3_SYSPATH)
    from sam3 import build_sam3_predictor

    if cam_ids is None:
        cam_ids = hmv.discover_cameras(take_dir)
    cam_ids = [c for c in cam_ids if c in calib]

    out_root = os.path.join(take_dir, "aligned", mask_subdir)

    print(f"Building SAM3 model ({SAM3_VERSION})...")
    model = build_sam3_predictor(version=SAM3_VERSION, compile=False, async_loading_frames=False)

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
            print(f"  {cam_id}: {len(frame_files)} SAM3 masks already cached, skipping.")
            continue

        frame_dir = os.path.join(take_dir, "aligned", cam_id)
        print(f"  {cam_id}: processing {len(frame_files)} frames...")

        response = model.handle_request({"type": "start_session", "resource_path": frame_dir})
        session_id = response["session_id"]

        model.handle_request({
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": 0,
            "text": SAM3_TEXT_PROMPT,
        })

        n_written = 0
        for response in model.handle_stream_request(
            {"type": "propagate_in_video", "session_id": session_id}
        ):
            frame_idx = response.get("frame_index")
            if frame_idx is None:
                continue
            outputs = response.get("outputs", {})
            binary_masks = outputs.get("out_binary_masks")
            if binary_masks is None:
                continue
            if isinstance(binary_masks, torch.Tensor):
                binary_masks = binary_masks.cpu().numpy()

            # Combine all detected objects into one binary mask
            combined = np.zeros(binary_masks.shape[-2:], dtype=np.uint8)
            for m in binary_masks:
                if m.ndim == 3:
                    m = m[0]
                combined |= (m > 0).astype(np.uint8)

            ff = frame_files[frame_idx]
            mask_path = os.path.join(cam_out_dir, ff)
            cv2.imwrite(mask_path, combined * 255)
            written[cam_id][ff] = mask_path
            n_written += 1

        print(f"  {cam_id}: {n_written} masks written to {cam_out_dir}")

    return written


# ---------------------------------------------------------------------------
# RVM backend
# ---------------------------------------------------------------------------

def _load_reconstruction_kps(take_dir, pose2d_dir="pose2d"):
    """Load smoothed 3D body + hand keypoints from reconstruction JSONs.

    Returns:
        body_kps  dict {frame_key: np.ndarray (N_body, 3)}
        left_kps  dict {frame_key: np.ndarray (N_left, 3)} — may be sparse
        right_kps dict {frame_key: np.ndarray (N_right, 3)} — may be sparse
    """
    pose2d_path = os.path.join(take_dir, "aligned", pose2d_dir)

    def _load(name):
        path = os.path.join(pose2d_path, f"reconstruction_{name}.json")
        if not os.path.exists(path):
            return {}
        with open(path) as f:
            d = json.load(f)
        smoothed = d.get("smoothed", {})
        result = {}
        for frame_key, kp_dict in smoothed.items():
            if not kp_dict:
                continue
            pts = np.array([kp_dict[str(i)] for i in range(len(kp_dict))], dtype=np.float32)
            result[frame_key] = pts
        return result

    return _load("body"), _load("left"), _load("right")


def compute_projected_bboxes(take_dir, calib, cam_id, frame_keys,
                             pad=0.30, smooth_window=5, pose2d_dir="pose2d"):
    """Project 3D body+hand keypoints into cam_id and compute padded bboxes.

    Returns np.ndarray of shape (N, 4) with integer [x1, y1, x2, y2] per
    frame, clamped to frame bounds. Frames with no visible keypoints reuse
    the last known box; if no box has been computed yet, uses the full frame.
    """
    body_kps, left_kps, right_kps = _load_reconstruction_kps(take_dir, pose2d_dir)

    cam = calib[cam_id]
    K  = np.array(cam["K"],  dtype=np.float64)
    R  = np.array(cam["R"],  dtype=np.float64)
    t  = np.array(cam["t"],  dtype=np.float64)
    W  = int(cam["width"])
    H  = int(cam["height"])

    def _project(pts3d):
        """pts3d: (N, 3) world → pixel (u, v). Returns (N, 2) float."""
        p_cam = (R @ pts3d.T).T + t        # (N, 3)
        p_img = (K @ p_cam.T).T            # (N, 3)
        z = p_img[:, 2:3]
        valid = z[:, 0] > 0                # behind-camera guard
        uv = p_img[:, :2] / np.where(z > 0, z, 1.0)
        return uv, valid

    raw_boxes = np.full((len(frame_keys), 4), -1, dtype=np.float32)
    for fi, fk in enumerate(frame_keys):
        pts_list = []
        if fk in body_kps:
            pts_list.append(body_kps[fk])
        if fk in left_kps:
            pts_list.append(left_kps[fk])
        if fk in right_kps:
            pts_list.append(right_kps[fk])
        if not pts_list:
            continue

        pts3d = np.concatenate(pts_list, axis=0)
        uv, valid = _project(pts3d)
        uv = uv[valid]
        if len(uv) < 2:
            continue

        u_min, v_min = uv.min(axis=0)
        u_max, v_max = uv.max(axis=0)

        # Pad by `pad` fraction of the larger dimension
        bw = u_max - u_min
        bh = v_max - v_min
        margin = max(bw, bh) * pad
        raw_boxes[fi] = [u_min - margin, v_min - margin,
                         u_max + margin, v_max + margin]

    # Temporal smoothing: fill gaps with last known box, then smooth
    last_box = np.array([0, 0, W, H], dtype=np.float32)
    filled = raw_boxes.copy()
    for fi in range(len(frame_keys)):
        if filled[fi, 0] < 0:
            filled[fi] = last_box
        else:
            last_box = filled[fi]

    # Simple causal moving average
    hw = smooth_window // 2
    smoothed = np.zeros_like(filled)
    for fi in range(len(frame_keys)):
        lo = max(0, fi - hw)
        hi = min(len(frame_keys), fi + hw + 1)
        smoothed[fi] = filled[lo:hi].mean(axis=0)

    # Clamp and convert to int
    smoothed[:, 0::2] = np.clip(smoothed[:, 0::2], 0, W)
    smoothed[:, 1::2] = np.clip(smoothed[:, 1::2], 0, H)
    return smoothed.astype(np.int32)


def extract_masks_rvm(take_dir, calib, cam_ids=None, force=False,
                      mask_subdir="masks_rvm", backbone="mobilenetv3",
                      downsample_ratio=0.25, pad=0.30, smooth_window=5,
                      input_size=(1080, 1920), pose2d_dir="pose2d"):
    """Run Robust Video Matting over each camera using keypoint-guided crops.

    Loads one RVM model and processes all cameras sequentially. Each camera
    gets a fresh recurrent state; the model stays in VRAM across cameras
    (it's only 32MB for MobileNetV3).

    Returns dict {cam_id: {frame_file: mask_path}} of what was written.
    """
    import sys
    import torch
    if RVM_SYSPATH not in sys.path:
        sys.path.insert(0, RVM_SYSPATH)
    from model.model import MattingNetwork  # noqa: E402

    if cam_ids is None:
        cam_ids = hmv.discover_cameras(take_dir)
    cam_ids = [c for c in cam_ids if c in calib]

    ckpt = RVM_CHECKPOINT_RESNET50 if backbone == "resnet50" else RVM_CHECKPOINT_MOBILENET
    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            f"RVM checkpoint not found: {ckpt}\n"
            f"Download with:\n"
            f"  wget -P /home/fmalmb/CODE/rvm/checkpoints \\\n"
            f"    https://github.com/PeterL1n/RobustVideoMatting/releases/"
            f"download/v1.0.0/rvm_{backbone}.pth"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Building RVM model ({backbone}) on {device}...")
    rvm_model = MattingNetwork(backbone).eval().to(device)
    if device.type == "cuda":
        rvm_model = rvm_model.half()
    rvm_model.load_state_dict(torch.load(ckpt, map_location=device))

    input_h, input_w = input_size
    out_root = os.path.join(take_dir, "aligned", mask_subdir)
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
            print(f"  {cam_id}: {len(frame_files)} RVM masks already cached, skipping.")
            continue

        bboxes = compute_projected_bboxes(
            take_dir, calib, cam_id, frame_files,
            pad=pad, smooth_window=smooth_window, pose2d_dir=pose2d_dir,
        )

        cam_h = int(calib[cam_id]["height"])
        cam_w = int(calib[cam_id]["width"])

        print(f"  {cam_id}: running RVM on {len(frame_files)} frames...")
        r1 = r2 = r3 = r4 = None  # recurrent state — reset per camera
        t0 = __import__("time").time()

        for fi, ff in enumerate(frame_files):
            frame_path = os.path.join(take_dir, "aligned", cam_id, ff)
            img = cv2.imread(frame_path)
            if img is None:
                continue

            x1, y1, x2, y2 = bboxes[fi]
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                crop = img

            crop_rs = cv2.resize(crop, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
            # BGR → RGB, HWC → CHW, [0,1] float
            src_np = crop_rs[:, :, ::-1].astype(np.float32) / 255.0
            src = torch.from_numpy(src_np).permute(2, 0, 1).unsqueeze(0).to(device)
            if device.type == "cuda":
                src = src.half()

            with torch.no_grad():
                fgr, pha, r1, r2, r3, r4 = rvm_model(
                    src, r1, r2, r3, r4, downsample_ratio=downsample_ratio
                )

            # pha: (1, 1, input_h, input_w) — resize back to crop size then un-crop
            alpha = pha[0, 0].float().cpu().numpy()  # (input_h, input_w) 0–1
            crop_h = y2 - y1
            crop_w = x2 - x1
            alpha_crop = cv2.resize(alpha, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)

            full_alpha = np.zeros((cam_h, cam_w), dtype=np.float32)
            full_alpha[y1:y2, x1:x2] = alpha_crop
            mask_img = (full_alpha * 255).clip(0, 255).astype(np.uint8)

            mask_path = os.path.join(cam_out_dir, ff)
            cv2.imwrite(mask_path, mask_img)
            written[cam_id][ff] = mask_path

        elapsed = __import__("time").time() - t0
        fps = len(frame_files) / elapsed
        print(f"  {cam_id}: {len(frame_files)} masks written in {elapsed:.1f}s ({fps:.0f} fps)")

    return written


def summarize_mask_coverage(take_dir, written, threshold=127):
    """Spot-check that mask pixel-coverage fraction per frame is plausible."""
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
              f"(near-0% or near-100%) coverage.")
    else:
        print("  no degenerate frames found.")
    return {
        "mean_coverage": float(fractions.mean()),
        "min_coverage": float(fractions.min()),
        "max_coverage": float(fractions.max()),
        "n_degenerate": n_degenerate,
        "n_frames": len(fractions),
    }


MASK_TINT_COLOR = (0, 200, 0)  # BGR green


def render_mask_grid_video(
    take_dir, calib, cam_ids=None, mask_subdir="masks_sam3", output_path=None,
    fps=15.0, cols=3, canvas_size=(1920, 1080), alpha=0.45, mask_threshold=127,
):
    """Render a grid video of masks overlaid on undistorted frames."""
    if cam_ids is None:
        cam_ids = hmv.discover_cameras(take_dir)
    cam_ids = [c for c in cam_ids if c in calib]
    if output_path is None:
        output_path = os.path.join(take_dir, "aligned", mask_subdir, "mask_grid_video.mp4")

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
                img = cv2.imread(os.path.join(take_dir, "aligned", cam_id, frame_key))
                if img is None:
                    continue
                w, h, (map1, map2) = undistort_maps[cam_id]
                img = hmv.undistort_fast(img, map1, map2)
                mask = cv2.imread(
                    os.path.join(take_dir, "aligned", mask_subdir, cam_id, frame_key),
                    cv2.IMREAD_GRAYSCALE,
                )
                coverage_pct = None
                if mask is not None:
                    binary = mask > mask_threshold
                    coverage_pct = float(binary.mean()) * 100.0
                    tinted = img.copy()
                    tinted[binary] = MASK_TINT_COLOR
                    img = cv2.addWeighted(tinted, alpha, img, 1.0 - alpha, 0)
                resized = cv2.resize(img, (slot_w, slot_h), interpolation=cv2.INTER_AREA)
                cv2.putText(resized, cam_id, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                label = f"{coverage_pct:.1f}%" if coverage_pct is not None else "no mask"
                color = (255, 255, 255) if coverage_pct is not None else (0, 0, 255)
                cv2.putText(resized, label, (10, slot_h - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
                row, col = cam_idx // cols, cam_idx % cols
                canvas[row * slot_h:(row + 1) * slot_h, col * slot_w:(col + 1) * slot_w] = resized

            if empty_slots:
                panel = np.zeros((slot_h, slot_w, 3), dtype=np.uint8)
                for i, line in enumerate(["LEGEND", "green = person mask",
                                          f"alpha: {alpha}", f"threshold: {mask_threshold}/255"]):
                    cv2.putText(panel, line, (20, 28 + i * 27),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
                row, col = empty_slots[0] // cols, empty_slots[0] % cols
                canvas[row * slot_h:(row + 1) * slot_h, col * slot_w:(col + 1) * slot_w] = panel

            cv2.putText(canvas, f"frame {idx} ({frame_key})", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(canvas)
            if (idx + 1) % 50 == 0 or (idx + 1) == len(frame_keys):
                print(f"  {idx + 1}/{len(frame_keys)} frames done.")
    finally:
        writer.release()
        print(f"Saved: {os.path.abspath(output_path)}")
    return output_path


def main():
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import calibrate  # noqa: E402

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("take_dir")
    parser.add_argument("--calib", default=os.path.join("output", "calibration", "20260903_153052_7cam.json"))
    parser.add_argument("--backend", choices=["sam3", "rvm"], default="sam3",
                        help="Segmentation backend: sam3 (video propagation) or rvm (keypoint-guided, faster).")
    parser.add_argument("--pose2d-dir", default="pose2d",
                        help="Subdirectory under aligned/ with reconstruction JSONs (RVM only).")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--mask-subdir", default=None,
                        help="Subdirectory under aligned/ to write masks to. "
                             "Defaults to masks_sam3 or masks_rvm based on --backend.")
    parser.add_argument("--grid-video", action="store_true",
                        help="Also render a mask grid video for visual inspection.")
    parser.add_argument("--fps", type=float, default=15.0)
    args = parser.parse_args()

    mask_subdir = args.mask_subdir or (f"masks_{args.backend}")
    calib = calibrate.load_calibration_output(args.calib)

    if args.backend == "rvm":
        written = extract_masks_rvm(args.take_dir, calib, force=args.force,
                                    mask_subdir=mask_subdir, pose2d_dir=args.pose2d_dir)
    else:
        written = extract_masks_sam3(args.take_dir, calib, force=args.force, mask_subdir=mask_subdir)

    summarize_mask_coverage(args.take_dir, written)
    if args.grid_video:
        render_mask_grid_video(args.take_dir, calib, mask_subdir=mask_subdir, fps=args.fps)


if __name__ == "__main__":
    main()
