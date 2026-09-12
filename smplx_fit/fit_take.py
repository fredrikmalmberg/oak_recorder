"""CLI entry point: fit SMPL-X to one take's already-triangulated
keypoints. See the module docstrings in data_loading.py/model.py/
losses.py/optimize.py for the individual pieces of this pipeline, and
alignment_improvements.md / the approved plan for the broader project
context this fits into.

Usage:
    python -m smplx_fit.fit_take <take_dir> --model-path models/SMPLX
        [--pose2d-dir pose2d] [--gender neutral] [--num-betas 20]
        [--frames START:END] [--max-outer-iters 10] [--force]

Writes <take_dir>/aligned/<pose2d_dir>/smplx/{smplx_params.npz,
optimization_log.json, fit_config.json}.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from smplx_fit import data_loading as dl
from smplx_fit import losses as L
from smplx_fit import model as smplx_model
from smplx_fit import optimize as opt
from smplx_fit import pose_prior
from smplx_fit import silhouette as sil
from hand_pose import hand_multiview as hmv


def parse_frame_range(spec, nf):
    """'START:END' (Python slice semantics, END exclusive) or None for the
    full take. A CLI option from the start (not a one-off flag added
    later), per the approved plan's rollout guidance -- short-slice runs
    during development and short clips later both use the same mechanism.
    """
    if spec is None:
        return 0, nf
    start_s, end_s = spec.split(":")
    start = int(start_s) if start_s else 0
    end = int(end_s) if end_s else nf
    return start, end


def summarize_fit_quality(params, model, full_layout, target_points, confidence):
    """Mirrors pose2d.triangulation.summarize_part's reporting style --
    the fit's own unweighted k3d/k3d_hand residual, for comparing against
    the triangulation's own inlier-only reprojection error (already
    measured for this project's takes: ~3-5mm typical) as a sanity check
    that the fit isn't grossly worse than the input data's own quality.
    """
    with torch.no_grad():
        output = params.forward_batch(model)
        pred = smplx_model.predict_all_points(output, full_layout)
    device = params.betas.device
    tp = torch.as_tensor(target_points, dtype=torch.float32, device=device)
    conf = torch.as_tensor(confidence, dtype=torch.float32, device=device)
    body_mask, hand_mask = opt.build_masks(full_layout)

    def unweighted_rms(mask):
        sub_pred, sub_target, sub_conf = pred[:, mask], tp[:, mask], conf[:, mask]
        valid = sub_conf > 0
        if valid.sum() == 0:
            return None
        sq_err = ((sub_pred - sub_target) ** 2).sum(dim=-1)
        return float(torch.sqrt(sq_err[valid].mean()).item())

    body_rms = unweighted_rms(body_mask)
    hand_rms = unweighted_rms(hand_mask)
    print("\n=== Fit quality summary (unweighted RMS distance, confident observations only) ===")
    print(f"  body: {body_rms * 1000:.2f} mm" if body_rms is not None else "  body: n/a")
    print(f"  hand: {hand_rms * 1000:.2f} mm" if hand_rms is not None else "  hand: n/a")
    print("  (compare against triangulation's own inlier-only reprojection error, "
          "typically ~3-5mm for this project's takes -- a fit dramatically worse "
          "than that suggests a joint-mapping or convergence problem worth investigating.)")
    return {"body_rms_mm": body_rms * 1000 if body_rms is not None else None,
            "hand_rms_mm": hand_rms * 1000 if hand_rms is not None else None}


def run_fit(take_dir, model_path, pose2d_dir, gender, num_betas, frame_range, max_outer_iters, force,
            pose_prior_backend="l2", pose_prior_weight=None,
            gmm_prior_path=pose_prior.DEFAULT_GMM_PATH, hand_reg_weight=0.0001,
            use_silhouette=False, silhouette_weight=2.0, calib_path=None,
            silhouette_out_size=(256, 144), silhouette_n_samples=1500,
            silhouette_sigma_px=1.0, use_silhouette_shape=False,
            use_sam3=False, use_rvm=False, mask_subdir="masks_sam3"):
    if pose_prior_weight is None:
        # See pose_prior.DEFAULT_POSE_PRIOR_WEIGHTS's comment -- "l2" and
        # "gmm" live on very different absolute scales, so there is no
        # single sane default weight shared across backends.
        pose_prior_weight = pose_prior.DEFAULT_POSE_PRIOR_WEIGHTS[pose_prior_backend]
    if (use_silhouette or use_silhouette_shape) and calib_path is None:
        raise ValueError("--use-silhouette/--use-silhouette-shape requires --calib (needed to project "
                          "the mesh into each camera's view for silhouette comparison).")
    out_dir = os.path.join(take_dir, "aligned", pose2d_dir, "smplx")
    npz_path = os.path.join(out_dir, "smplx_params.npz")
    if not force and os.path.exists(npz_path):
        print(f"{npz_path} already exists -- pass --force to refit.")
        return

    print(f"Loading triangulated keypoints from {take_dir} (pose2d_dir={pose2d_dir})...")
    frame_keys, full_layout, target_points, confidence = dl.build_take_arrays(take_dir, pose2d_dir=pose2d_dir)
    nf_total = len(frame_keys)
    start, end = parse_frame_range(frame_range, nf_total)
    frame_keys = frame_keys[start:end]
    target_points = target_points[start:end]
    confidence = confidence[start:end]
    print(f"Fitting frames [{start}:{end}) of {nf_total} total "
          f"({(confidence > 0).mean():.1%} confident observations in this range).")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading SMPL-X model from {model_path} (gender={gender}, num_betas={num_betas})...")
    model = smplx_model.load_layer(model_path, gender=gender, num_betas=num_betas)
    model = model.to(device)

    silhouette_kwargs = {}
    if use_silhouette or use_silhouette_shape:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        import calibrate  # noqa: E402
        from smplx_fit import segmentation as seg

        print(f"Loading calibration from {calib_path} and cached Phase 2 masks for silhouette fitting...")
        calib = calibrate.load_calibration_output(calib_path)
        cam_ids = [c for c in hmv.discover_cameras(take_dir) if c in calib]

        if use_sam3:
            print(f"Generating SAM3 masks into {mask_subdir}/ (will skip cameras that already have masks)...")
            seg.extract_masks_sam3(take_dir, calib, cam_ids=cam_ids, force=False, mask_subdir=mask_subdir)
        if use_rvm:
            print(f"Generating RVM masks into {mask_subdir}/ (will skip cameras that already have masks)...")
            seg.extract_masks_rvm(take_dir, calib, cam_ids=cam_ids, force=False,
                                   mask_subdir=mask_subdir, pose2d_dir=pose2d_dir)

        sil_cams, sil_masks, sil_valid = sil.load_silhouette_data(
            take_dir, calib, cam_ids, frame_keys, out_size=silhouette_out_size,
            device=device, mask_subdir=mask_subdir,
        )
        n_pairs = sum(int(v.sum()) for v in sil_valid.values())
        print(f"  silhouette: {len(sil_cams)}/{len(cam_ids)} cameras have cached masks "
              f"({n_pairs} total (frame, camera) pairs available in this frame range).")
        silhouette_kwargs = dict(
            use_silhouette=use_silhouette, use_silhouette_shape=use_silhouette_shape,
            silhouette_weight=silhouette_weight,
            silhouette_cams=sil_cams, silhouette_masks=sil_masks, silhouette_valid=sil_valid,
            silhouette_n_samples=silhouette_n_samples, silhouette_sigma_px=silhouette_sigma_px,
        )

    t0 = time.time()
    params, log = opt.multi_stage_optimize(
        model, target_points, confidence, full_layout,
        max_outer_iters=max_outer_iters, num_betas=num_betas,
        pose_prior_backend=pose_prior_backend, gmm_prior_path=gmm_prior_path,
        pose_prior_weight=pose_prior_weight, hand_reg_weight=hand_reg_weight,
        device=device,
        **silhouette_kwargs,
    )
    elapsed = time.time() - t0
    print(f"\nOptimization finished in {elapsed:.1f}s.")

    quality = summarize_fit_quality(params, model, full_layout, target_points, confidence)
    log["fit_quality"] = quality
    log["elapsed_seconds"] = elapsed

    os.makedirs(out_dir, exist_ok=True)
    param_dict = params.as_dict()
    np.savez(
        npz_path,
        frame_keys=np.array(frame_keys),
        gender=gender, num_betas=num_betas, model_type="smplx",
        jaw_pose=np.zeros((len(frame_keys), 3), dtype=np.float32),
        leye_pose=np.zeros((len(frame_keys), 3), dtype=np.float32),
        reye_pose=np.zeros((len(frame_keys), 3), dtype=np.float32),
        expression=np.zeros((len(frame_keys), 10), dtype=np.float32),
        **param_dict,
    )
    print(f"Wrote {npz_path}")

    with open(os.path.join(out_dir, "optimization_log.json"), "w") as f:
        json.dump(log, f, indent=2)
    print(f"Wrote {os.path.join(out_dir, 'optimization_log.json')}")

    fit_config = {
        "take_dir": take_dir, "model_path": model_path, "pose2d_dir": pose2d_dir,
        "gender": gender, "num_betas": num_betas, "frame_range": [start, end],
        "max_outer_iters": max_outer_iters,
        "pose_prior_backend": pose_prior_backend, "pose_prior_weight": pose_prior_weight,
        "gmm_prior_path": gmm_prior_path, "hand_reg_weight": hand_reg_weight,
        "use_silhouette": use_silhouette, "use_silhouette_shape": use_silhouette_shape,
        "silhouette_weight": silhouette_weight if (use_silhouette or use_silhouette_shape) else None,
        "calib_path": calib_path, "silhouette_out_size": list(silhouette_out_size),
        "silhouette_n_samples": silhouette_n_samples, "silhouette_sigma_px": silhouette_sigma_px,
    }
    with open(os.path.join(out_dir, "fit_config.json"), "w") as f:
        json.dump(fit_config, f, indent=2)
    print(f"Wrote {os.path.join(out_dir, 'fit_config.json')}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("take_dir")
    parser.add_argument("--model-path", default="models/SMPLX")
    parser.add_argument("--pose2d-dir", default="pose2d")
    parser.add_argument("--gender", default="neutral")
    parser.add_argument("--num-betas", type=int, default=20)
    parser.add_argument("--frames", default=None, help="START:END (Python slice semantics), default: whole take")
    parser.add_argument("--max-outer-iters", type=int, default=10)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--pose-prior", dest="pose_prior_backend", choices=["l2", "gmm", "none"], default="l2",
                         help="Body pose prior backend for stage 2/3/4/5's reg_pose term. "
                              "'l2' (default) matches today's behavior unchanged. Never affects hands.")
    parser.add_argument("--pose-prior-weight", type=float, default=None,
                         help="Weight applied to reg_pose. Default depends on --pose-prior: 0.01 for "
                              "l2 (today's hardcoded value) or none, 0.0001 for gmm (its NLL lives on "
                              "a much larger absolute scale -- see pose_prior.DEFAULT_POSE_PRIOR_WEIGHTS).")
    parser.add_argument("--gmm-prior-path", default=pose_prior.DEFAULT_GMM_PATH,
                         help=f"Path to the GMM pose prior pickle (default: {pose_prior.DEFAULT_GMM_PATH}). "
                              "Only used when --pose-prior gmm.")
    parser.add_argument("--hand-reg-weight", type=float, default=0.0001,
                         help="Independent L2 weight for hand pose regularization (default: 0.0001, "
                              "today's value). Never touched by --pose-prior.")
    parser.add_argument("--use-silhouette", action="store_true",
                         help="Phase 3: add a silhouette-overlap term to Stage 3 (body pose) only, "
                              "using cached Phase 2 masks (see smplx_fit.segmentation). Off by default "
                              "-- this is the least-verified piece of the pipeline so far. Requires --calib.")
    parser.add_argument("--use-silhouette-shape", action="store_true",
                         help="Add Stage 6: joint refinement of betas + body_pose + global_orient + "
                              "transl via silhouette + keypoints after all 5 keypoint stages. Shape and "
                              "pose co-adapt (EasyMoCap 'refine_poses' philosophy). Off by default. "
                              "Requires --calib.")
    parser.add_argument("--calib", dest="calib_path", default=None,
                         help="Calibration output JSON (calibrate.load_calibration_output's schema). "
                              "Required when --use-silhouette or --use-silhouette-shape is passed.")
    parser.add_argument("--use-sam3", action="store_true",
                         help="Before silhouette fitting, generate SAM3 masks (text prompt 'person', "
                              "no clicking needed). Only runs on cameras that don't already have masks "
                              "unless --force is also passed.")
    parser.add_argument("--use-rvm", action="store_true",
                         help="Before silhouette fitting, generate RVM masks using keypoint-guided "
                              "ROI crops (requires triangulated 3D keypoints in pose2d_dir). "
                              "Much faster than SAM3 (~10s/cam vs ~7min/cam on 4090). "
                              "Only runs on cameras that don't already have masks unless --force is passed.")
    parser.add_argument("--silhouette-weight", type=float, default=0.1)
    parser.add_argument("--silhouette-out-size", default="32x18",
                         help="WxH of the downsampled render used for the silhouette comparison.")
    parser.add_argument("--silhouette-n-samples", type=int, default=1500)
    parser.add_argument("--silhouette-sigma-px", type=float, default=1.0)
    parser.add_argument("--mask-subdir", default="masks_sam3",
                         help="Subdirectory under aligned/ containing per-camera mask images "
                              "(default: masks_sam3).")
    args = parser.parse_args()
    sil_w, sil_h = (int(x) for x in args.silhouette_out_size.lower().split("x"))
    run_fit(
        args.take_dir, args.model_path, args.pose2d_dir, args.gender, args.num_betas,
        args.frames, args.max_outer_iters, args.force,
        pose_prior_backend=args.pose_prior_backend, pose_prior_weight=args.pose_prior_weight,
        gmm_prior_path=args.gmm_prior_path, hand_reg_weight=args.hand_reg_weight,
        use_silhouette=args.use_silhouette, use_silhouette_shape=args.use_silhouette_shape,
        silhouette_weight=args.silhouette_weight,
        calib_path=args.calib_path, silhouette_out_size=(sil_w, sil_h),
        silhouette_n_samples=args.silhouette_n_samples, silhouette_sigma_px=args.silhouette_sigma_px,
        use_sam3=args.use_sam3, use_rvm=args.use_rvm, mask_subdir=args.mask_subdir,
    )


if __name__ == "__main__":
    main()
