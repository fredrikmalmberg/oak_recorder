"""The 5-stage L-BFGS optimizer, reimplementing bvh2smplx's documented
design (shape -> global pose -> body pose -> hands+body -> hands-only
polish) against our own triangulated keypoints. See the module docstrings
in losses.py/model.py for the individual pieces; this module is the
orchestration and the "document each step" logging infrastructure the
approved plan calls for explicitly.
"""
import torch
from smplx.joint_names import JOINT_NAMES

from smplx_fit import losses
from smplx_fit import model as smplx_model
from smplx_fit import pose_prior
from smplx_fit import silhouette as sil

BODY_MASK_PARTS = ("body", "wrist_left", "wrist_right")
HAND_MASK_PARTS = ("left", "right")


class FitParams:
    """The 6 SMPL-X parameter tensors this pipeline optimizes, all stored
    as axis-angle (see model.py's module docstring for why) and as plain
    torch.nn.Parameter so any subset can be handed to an LBFGS optimizer
    for a given stage while the rest stay fixed (requires_grad toggled per
    stage, not re-created).

    betas is (1, num_betas) -- ONE shared shape for the whole take
    (broadcast to nf when forming a batch), not per-frame like bvh2smplx's
    own design: a deliberate deviation, since it's the same physical signer
    throughout one take and per-frame shape would be meaningless extra DOF
    that the sparse (no-leg, no-BVH-ground-truth) shape signal here can't
    usefully support anyway.
    """

    def __init__(self, nf, num_betas=20, device=None):
        self.nf = nf
        device = device or torch.device("cpu")
        self.betas = torch.nn.Parameter(torch.zeros(1, num_betas, device=device))
        self.global_orient = torch.nn.Parameter(torch.zeros(nf, 3, device=device))
        self.transl = torch.nn.Parameter(torch.zeros(nf, 3, device=device))
        self.body_pose = torch.nn.Parameter(torch.zeros(nf, 21, 3, device=device))
        self.lhand_pose = torch.nn.Parameter(torch.zeros(nf, 15, 3, device=device))
        self.rhand_pose = torch.nn.Parameter(torch.zeros(nf, 15, 3, device=device))

    def forward_batch(self, model):
        betas = self.betas.expand(self.nf, -1)
        return smplx_model.forward(
            model, betas, self.global_orient, self.body_pose,
            self.lhand_pose, self.rhand_pose, self.transl,
        )

    def forward_rest_pose(self, model):
        """betas only, everything else zero -- used by shape3d_loss, which
        needs each bone's length under the CURRENT shape hypothesis with no
        pose distortion (see losses.shape3d_loss's docstring).
        """
        device = self.betas.device
        zero1 = torch.zeros(1, 3, device=device)
        return smplx_model.forward(
            model, self.betas, zero1, torch.zeros(1, 21, 3, device=device),
            torch.zeros(1, 15, 3, device=device), torch.zeros(1, 15, 3, device=device), zero1,
        )

    def as_dict(self):
        """axis-angle numpy arrays, ready for np.savez -- see fit_take.py."""
        return {
            "betas": self.betas.detach().cpu().numpy(),
            "global_orient": self.global_orient.detach().cpu().numpy(),
            "transl": self.transl.detach().cpu().numpy(),
            "body_pose": self.body_pose.detach().cpu().numpy(),
            "lhand_pose": self.lhand_pose.detach().cpu().numpy(),
            "rhand_pose": self.rhand_pose.detach().cpu().numpy(),
        }


def build_layout_index_by_local_id(full_layout):
    """part -> {local_landmark_id: k} for shape3d_loss's bone-length
    lookups. The two fused-wrist rows (part="wrist_left"/"wrist_right",
    landmark_id=None in full_layout) are registered under BOTH the COCO
    body wrist id ("9"/"10", needed by BODY_BONES' forearm segments (7,9)/
    (8,10)) and that hand's own local wrist id ("0", needed by every
    HAND_BONES finger chain's first segment) -- both bone lists reference a
    physical wrist that only exists as this one fused row, not as a
    separate "body" or "left"/"right" entry of its own.
    """
    idx = {"body": {}, "left": {}, "right": {}}
    for k, (part, landmark_id, _target) in enumerate(full_layout):
        if part in idx:
            idx[part][landmark_id] = k
    for k, (part, _landmark_id, _target) in enumerate(full_layout):
        if part == "wrist_left":
            idx["body"]["9"] = k
            idx["left"]["0"] = k
        elif part == "wrist_right":
            idx["body"]["10"] = k
            idx["right"]["0"] = k
    return idx


def build_masks(full_layout):
    body_mask = torch.tensor([part in BODY_MASK_PARTS for part, _, _ in full_layout])
    hand_mask = torch.tensor([part in HAND_MASK_PARTS for part, _, _ in full_layout])
    return body_mask, hand_mask


def _check_up_axis(params, model):
    """No fix needed mechanically -- every loss term here is a Euclidean
    distance, rotation-invariant, so global_orient/transl naturally land in
    whatever world frame the input keypoints already occupy (see
    pose2d.visualize_triangulation's own documented caveat: that frame may
    or may not be Z-up, and isn't recorded in the calibration JSON). This
    is a non-blocking, purely informational diagnostic: after Stage 2
    converges (global rotation/translation fit, before any body/hand pose
    fitting can bias it further), compute the fitted pelvis->head vector's
    angle from +Z and log a warning if it's large -- visibility into
    whether this take's calibration happened to be Z-up, without inventing
    a hard requirement the codebase can't actually verify from data alone.
    """
    with torch.no_grad():
        output = params.forward_batch(model)
    pelvis = output.joints[:, JOINT_NAMES.index("pelvis"), :].mean(dim=0)
    head = output.joints[:, JOINT_NAMES.index("head"), :].mean(dim=0)
    up_vec = head - pelvis
    up_vec = up_vec / up_vec.norm()
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=up_vec.device)
    cos_angle = torch.clamp((up_vec * z_axis).sum(), -1.0, 1.0)
    angle_deg = float(torch.rad2deg(torch.acos(cos_angle)).item())
    if angle_deg > 30.0:
        print(f"WARNING: fitted body's up-vector is {angle_deg:.1f} deg off +Z -- "
              f"expected if this take's calibration didn't run world-alignment; "
              f"verify visually before trusting global_orient/transl beyond internal consistency.")
    else:
        print(f"Up-axis check: fitted body's up-vector is {angle_deg:.1f} deg off +Z (looks Z-up).")
    return {"angle_from_z_deg": angle_deg}


def _run_lbfgs_stage(name, param_list, loss_fn, max_outer_iters, rel_tol):
    """One stage's outer convergence loop. LBFGS's own `max_iter` (set at
    construction) only bounds a SINGLE `.step(closure)` call's internal
    line-search budget -- this outer loop is what implements bvh2smplx's
    documented "stop at <1e-9 relative loss change" criterion, which needs
    to compare loss ACROSS `.step()` calls.

    loss_fn() -> (total_loss, {term_name: raw_value, ...}) -- called from
    inside the LBFGS closure (so once or more per outer iteration,
    depending on the line search), but only the LAST call's breakdown each
    outer iteration is logged, since that's the one whose gradient the
    optimizer actually used to step.

    Returns the per-iteration log records for optimization_log.json (see
    fit_take.py) -- a persisted, inspectable artifact, not just console
    output, per the "document each step" requirement.
    """
    optimizer = torch.optim.LBFGS(param_list, max_iter=30, line_search_fn="strong_wolfe")
    records = []
    prev_loss = None
    convergence_reason = "max_outer_iters"

    for outer_i in range(max_outer_iters):
        last_breakdown = {}

        def closure():
            optimizer.zero_grad()
            total, breakdown = loss_fn()
            if not torch.isfinite(total):
                # Caller checks records[-1]["nan_guard_tripped"] and stops
                # this stage early, keeping the last good params rather
                # than letting a bad step corrupt everything downstream.
                last_breakdown["nan_guard_tripped"] = True
                return total
            total.backward()
            last_breakdown.clear()
            last_breakdown.update(breakdown)
            last_breakdown["total"] = float(total.item())
            return total

        loss_value = optimizer.step(closure)
        total_loss = float(loss_value.item()) if torch.is_tensor(loss_value) else float(loss_value)

        if last_breakdown.get("nan_guard_tripped"):
            convergence_reason = "nan_guard"
            print(f"[{name}] outer {outer_i}: NaN/Inf loss detected -- stopping stage, keeping last good params.")
            break

        record = {"outer_iter": outer_i, **last_breakdown}
        records.append(record)
        terms_str = ", ".join(f"{k}={v:.6g}" for k, v in last_breakdown.items() if k != "total")
        print(f"[{name}] outer {outer_i}: total={total_loss:.6g} ({terms_str})")

        if prev_loss is not None:
            rel_change = abs(prev_loss - total_loss) / max(abs(prev_loss), 1e-12)
            if rel_change < rel_tol:
                convergence_reason = "rel_tol"
                prev_loss = total_loss
                break
        prev_loss = total_loss

    print(f"[{name}] stopped: {convergence_reason} after {len(records)} outer iteration(s).")
    return records, convergence_reason


def _run_adam_stage(name, param_list, loss_fn, max_iters, rel_tol, lr=0.01):
    """Same per-iteration logging/NaN-guard/rel_tol convergence contract as
    _run_lbfgs_stage, but plain Adam instead of LBFGS -- used specifically
    for stage 3b (betas via silhouette), where LBFGS's line search was
    confirmed empirically this session to fail on the very first step: the
    shape3d term is weighted 10000x and already sitting exactly at its
    stage-1 minimum (an extremely steep local bowl), while silhouette's
    gradient is comparatively tiny, so ANY LBFGS trial step overshoots
    shape3d's sharp curvature and fails the Wolfe sufficient-decrease
    condition, leaving betas completely unmoved (confirmed: shape3d/
    reg_shape/silhouette were bit-identical across "converged" outer
    iterations, despite betas.grad being verifiably nonzero). Adam's
    per-parameter adaptive step size handles wildly different term scales
    far more robustly than a single global line search does, at the cost
    of needing a fixed learning rate/iteration budget instead of LBFGS's
    own automatic step sizing.
    """
    optimizer = torch.optim.Adam(param_list, lr=lr)
    records = []
    prev_loss = None
    convergence_reason = "max_iters"

    for it in range(max_iters):
        optimizer.zero_grad()
        total, breakdown = loss_fn()
        if not torch.isfinite(total):
            convergence_reason = "nan_guard"
            print(f"[{name}] iter {it}: NaN/Inf loss detected -- stopping stage, keeping last good params.")
            break
        total.backward()
        optimizer.step()
        total_loss = float(total.item())

        record = {"iter": it, **breakdown, "total": total_loss}
        records.append(record)
        terms_str = ", ".join(f"{k}={v:.6g}" for k, v in breakdown.items())
        print(f"[{name}] iter {it}: total={total_loss:.6g} ({terms_str})")

        if prev_loss is not None:
            rel_change = abs(prev_loss - total_loss) / max(abs(prev_loss), 1e-12)
            if rel_change < rel_tol:
                convergence_reason = "rel_tol"
                break
        prev_loss = total_loss

    print(f"[{name}] stopped: {convergence_reason} after {len(records)} iteration(s).")
    return records, convergence_reason


def multi_stage_optimize(
    model, target_points_np, confidence_np, full_layout,
    max_outer_iters=10, rel_tol=1e-9, num_betas=20,
    pose_prior_backend="l2", gmm_prior_path=pose_prior.DEFAULT_GMM_PATH,
    pose_prior_weight=0.01, hand_reg_weight=0.0001,
    use_silhouette=False, silhouette_weight=2.0, silhouette_cams=None,
    silhouette_masks=None, silhouette_valid=None, silhouette_n_samples=1500,
    silhouette_sigma_px=1.0, use_silhouette_shape=False,
    device=None,
):
    """Runs all 5 stages in sequence, returns (FitParams, full_log_dict).
    full_log_dict is written verbatim to optimization_log.json by
    fit_take.py.

    pose_prior_backend: "l2" (default, today's behavior -- mean(body_pose**2)),
    "gmm" (the classic SMPLify/SMPLify-X MaxMixturePrior, see pose_prior.py),
    or "none" (disabled). Applies to body_pose ONLY, in every stage that
    regularizes it (stage 2/3's shared reg_pose term, and stage 4's) --
    deliberately never to hand_pose, which stays purely data-driven per an
    explicit project decision (see pose_prior.py's module docstring).
    hand_reg_weight independently controls reg_hand's weight (default
    0.0001, today's value) so hands can be tuned/zeroed without touching
    the body prior setting at all.

    use_silhouette (Phase 3 of the EasyMoCap-inspired plan, see silhouette.
    py's module docstring for the rendering approach): adds a silhouette
    overlap term to Stage 3 (body pose) -- deliberately NOT the original
    Stage 1 (shape), despite the plan's original wording, because Stage 1
    fits shape against the REST pose (body_pose/global_orient/transl all
    zero, a canonical T-pose-like frame -- see stage1_loss below), which
    has no meaningful placement in any camera's view to render against.
    Stage 3 is the first point in this pipeline where global_orient/transl
    are already fit (from Stage 2) and body_pose is what's being
    optimized, so it's where silhouette overlap first has an actual,
    correctly-placed mesh to compare. Never applied to hand_pose (stage
    4/5), per the same data-over-prior reasoning as pose_prior_backend.
    silhouette_cams/_masks/_valid come from silhouette.load_silhouette_data,
    already sliced to the SAME frame range/order as target_points_np --
    required (raises) if use_silhouette or use_silhouette_shape is True and
    left unset.

    use_silhouette_shape gates Stage 6 (default off, even when
    use_silhouette is on): a joint refinement of betas + body_pose +
    global_orient + transl TOGETHER with silhouette + keypoints, running
    after all 5 keypoint stages. Shape and pose co-adapt so neither hits a
    local optimum from being fit in isolation (the EasyMoCap "refine_poses"
    philosophy). hand_pose stays frozen. 200 Adam iterations at lr=0.001.

    silhouette_n_samples/silhouette_sigma_px default to 1500/1.0, NOT
    silhouette.py's own higher-fidelity single-frame-validation defaults
    (8000 samples, sigma_px scaled for a 64x36 render) -- confirmed
    empirically this session that the higher-fidelity settings cost ~20s
    per forward+backward over a 40-frame/7-camera batch, intractable
    inside LBFGS's inner loop (up to 30 steps x several line-search evals
    EACH, per outer iteration); these lighter defaults cost ~1s for the
    same batch (a validated real IoU signal, just lower-fidelity -- ~0.35-
    0.4 vs. ~0.5+ on the same test frame) and keep a full stage tractable.
    """
    device = device or torch.device("cpu")
    nf = target_points_np.shape[0]
    target_points = torch.as_tensor(target_points_np, dtype=torch.float32, device=device)
    confidence = torch.as_tensor(confidence_np, dtype=torch.float32, device=device)
    body_mask, hand_mask = build_masks(full_layout)
    body_mask = body_mask.to(device)
    hand_mask = hand_mask.to(device)
    layout_index_by_local_id = build_layout_index_by_local_id(full_layout)

    gmm = pose_prior.load_gmm_prior(gmm_prior_path) if pose_prior_backend == "gmm" else None

    def body_pose_prior_loss(body_pose):
        return pose_prior.compute_body_pose_prior(body_pose, pose_prior_backend, gmm=gmm)

    if use_silhouette or use_silhouette_shape:
        if silhouette_cams is None or silhouette_masks is None or silhouette_valid is None:
            raise ValueError("use_silhouette/use_silhouette_shape=True requires silhouette_cams/_masks/_valid "
                              "(see silhouette.load_silhouette_data).")
        face_idx, bary = sil.build_surface_samples(
            model.faces, model.v_template.detach().cpu().numpy(), n_samples=silhouette_n_samples,
        )
        face_idx = face_idx.to(device)
        bary = bary.to(device)

    params = FitParams(nf, num_betas=num_betas, device=device)
    log = {"stages": {}, "pose_prior_backend": pose_prior_backend, "use_silhouette": use_silhouette,
           "use_silhouette_shape": use_silhouette_shape,
           "silhouette_weight": silhouette_weight if (use_silhouette or use_silhouette_shape) else None,
           "known_limitations": [
        "No leg keypoints were fit -- body_pose's knee/ankle/foot slots are held "
        "near neutral pose almost entirely by the body pose prior (reg_pose), not by "
        "any observed data.",
        "betas (shape) is informed only by arm/hand bone lengths (no hip/leg bone "
        "observations at all) -- a sparse signal that can't meaningfully constrain "
        "many shape directions; num_betas defaults to 20 (not SMPL-X's/bvh2smplx's "
        "larger defaults) and reg_shape is weighted more heavily than bvh2smplx's "
        "own value for exactly this reason (see optimize.py's stage1_loss comment).",
        "World-frame up-axis is whatever the source calibration used (may not be "
        "Z-up) -- global_orient/transl are internally consistent with the input "
        "keypoints but not verified against gravity.",
    ]}

    # ---- Stage 1: Shape -----------------------------------------------
    # Unfrozen: betas. Frozen: everything else (still zero). Loss:
    # shape3d (limb-length consistency, our replacement for bvh2smplx's
    # ground-truth-BVH-limb-length term) + reg_shape. Legs contribute
    # nothing here (no leg keypoints), so shape is only informed by
    # arm/hand proportions -- see known_limitations above.
    #
    # reg_shape's weight is 1.0 here, NOT bvh2smplx's documented 0.001 --
    # verified empirically (see conversation/commit history) that copying
    # their literal weight let betas explode to +-20+ (real human bodies
    # stay within roughly +-2 to 3 per component) while shape3d itself sat
    # at ~1e-5, i.e. already-satisfied with betas near zero. bvh2smplx's
    # richer full-body BVH bone coverage (61 joints incl. legs/spine)
    # presumably keeps their shape3d landscape well-conditioned enough that
    # a tiny regularizer suffices; our sparse arm/hand-only bone set leaves
    # most of the 300 PCA directions essentially unconstrained, so L-BFGS
    # wanders arbitrarily far along those flat directions unless reg_shape
    # actually pushes back. 1.0 was confirmed (same betas-magnitude check)
    # to keep every fitted beta within a plausible human range while still
    # letting shape3d converge to the same near-zero residual.
    def stage1_loss():
        rest_output = params.forward_rest_pose(model)
        rest_points = smplx_model.predict_all_points(rest_output, full_layout)[0]
        shape3d, _ = losses.shape3d_loss(target_points, confidence, rest_points, layout_index_by_local_id)
        reg_shape = losses.l2_regularization(params.betas)
        total = shape3d * 10000.0 + reg_shape * 1.0
        return total, {"shape3d": float(shape3d.item()), "reg_shape": float(reg_shape.item())}

    log["stages"]["1_shape"], _ = _run_lbfgs_stage("1_shape", [params.betas], stage1_loss, max_outer_iters, rel_tol)

    # ---- Stage 2: Global rotation + translation ------------------------
    # Unfrozen: global_orient, transl. Frozen: betas (from stage 1),
    # body_pose/hand_pose (still zero). Loss: k3d (body keypoints only --
    # hands aren't posed yet, so their prediction is a poor target match
    # regardless) + temporal smoothness on transl/global_orient + reg_pose
    # (inert here since body_pose isn't optimized yet, kept for fidelity
    # to the stage's documented loss set).
    def stage2_loss():
        output = params.forward_batch(model)
        pred = smplx_model.predict_all_points(output, full_layout)
        k3d, _ = losses.weighted_keypoint_loss(pred, target_points, confidence, mask=body_mask)
        smooth_transl = losses.temporal_smoothness_loss(params.transl)
        smooth_go = losses.temporal_smoothness_loss(params.global_orient)
        reg_pose = body_pose_prior_loss(params.body_pose)
        total = k3d * 1.0 + smooth_transl * 0.5 + smooth_go * 0.1 + reg_pose * pose_prior_weight
        return total, {
            "k3d": float(k3d.item()), "smooth_transl": float(smooth_transl.item()),
            "smooth_global_orient": float(smooth_go.item()), "reg_pose": float(reg_pose.item()),
        }

    log["stages"]["2_global_rt"], _ = _run_lbfgs_stage(
        "2_global_rt", [params.global_orient, params.transl], stage2_loss, max_outer_iters, rel_tol,
    )
    log["up_axis_check"] = _check_up_axis(params, model)

    # ---- Stage 3: Body pose --------------------------------------------
    # Unfrozen: + body_pose. Frozen: betas, hand_pose. Same base loss set
    # as stage 2 -- reg_pose is doing real work here (including holding
    # legs at neutral, since no leg keypoints ever contribute to k3d) --
    # PLUS, when use_silhouette is on, a silhouette overlap term that gives
    # legs (and the rest of body_pose) a REAL non-prior signal to fit
    # against for the first time (see multi_stage_optimize's docstring for
    # why this is the first stage silhouette can apply to, not stage 1).
    # Needs its own forward_batch call (not stage2_loss's) since it also
    # needs `output.vertices`, which stage2_loss's k3d-only path discards.
    def stage3_loss():
        output = params.forward_batch(model)
        pred = smplx_model.predict_all_points(output, full_layout)
        k3d, _ = losses.weighted_keypoint_loss(pred, target_points, confidence, mask=body_mask)
        smooth_transl = losses.temporal_smoothness_loss(params.transl)
        smooth_go = losses.temporal_smoothness_loss(params.global_orient)
        reg_pose = body_pose_prior_loss(params.body_pose)
        total = k3d * 1.0 + smooth_transl * 0.5 + smooth_go * 0.1 + reg_pose * pose_prior_weight
        breakdown = {
            "k3d": float(k3d.item()), "smooth_transl": float(smooth_transl.item()),
            "smooth_global_orient": float(smooth_go.item()), "reg_pose": float(reg_pose.item()),
        }
        if use_silhouette:
            sil_loss, n_pairs = sil.compute_silhouette_term(
                output.vertices, model.faces, face_idx, bary,
                silhouette_cams, silhouette_masks, silhouette_valid, sigma_px=silhouette_sigma_px,
            )
            total = total + sil_loss * silhouette_weight
            breakdown["silhouette"] = float(sil_loss.item())
            breakdown["silhouette_n_pairs"] = n_pairs
        return total, breakdown

    log["stages"]["3_body_pose"], _ = _run_lbfgs_stage(
        "3_body_pose", [params.global_orient, params.transl, params.body_pose],
        stage3_loss, max_outer_iters, rel_tol,
    )

    # (stage 3b betas-only removed -- replaced by stage 6 joint refinement below)

    # ---- Stage 4: Hands + body ------------------------------------------
    # Unfrozen: body_pose, lhand_pose, rhand_pose. Frozen: betas,
    # global_orient, transl (from stage 3). Loss: k3d (body) + k3d_hand
    # (hands, weighted 10x higher -- hand shape matters most for sign
    # language) + smoothness on body/hand pose + reg_pose/reg_hand.
    #
    # smooth_hand's weight is 0.1 here, NOT bvh2smplx's documented 0.001 --
    # verified empirically (optimization_log.json) that copying their
    # literal weight let this fit visibly contort hand pose frame-to-frame
    # chasing per-frame triangulation noise (smooth_hand_pose and reg_hand
    # both grew substantially -- 0.0007->0.05 and 0.004->0.45 respectively
    # across one stage -- as k3d_hand tightened, the classic overfitting-
    # to-noise signature). bvh2smplx's clean BVH mocap presumably never hit
    # this because it has no comparable per-frame noise to overfit; our
    # RANSAC-triangulated-from-video keypoints do (confirmed independently:
    # ~20% of frames get jitter-flagged by the triangulation step itself).
    # Combined with data_loading.py's switch to "smoothed" (not "raw")
    # target values, this is belt-and-suspenders against the same root
    # cause rather than relying on either fix alone.
    def stage4_loss():
        output = params.forward_batch(model)
        pred = smplx_model.predict_all_points(output, full_layout)
        k3d, _ = losses.weighted_keypoint_loss(pred, target_points, confidence, mask=body_mask)
        k3d_hand, _ = losses.weighted_keypoint_loss(pred, target_points, confidence, mask=hand_mask)
        smooth_body = losses.temporal_smoothness_loss(params.body_pose)
        smooth_hand = losses.temporal_smoothness_loss(params.lhand_pose) + losses.temporal_smoothness_loss(params.rhand_pose)
        reg_pose = body_pose_prior_loss(params.body_pose)
        reg_hand = losses.l2_regularization(params.lhand_pose) + losses.l2_regularization(params.rhand_pose)
        total = (k3d * 1.0 + k3d_hand * 10.0 + smooth_body * 5.0 + smooth_hand * 0.1
                 + reg_pose * pose_prior_weight + reg_hand * hand_reg_weight)
        return total, {
            "k3d": float(k3d.item()), "k3d_hand": float(k3d_hand.item()),
            "smooth_body_pose": float(smooth_body.item()), "smooth_hand_pose": float(smooth_hand.item()),
            "reg_pose": float(reg_pose.item()), "reg_hand": float(reg_hand.item()),
        }

    log["stages"]["4_hands_and_body"], _ = _run_lbfgs_stage(
        "4_hands_and_body", [params.body_pose, params.lhand_pose, params.rhand_pose],
        stage4_loss, max_outer_iters, rel_tol,
    )

    # ---- Stage 5: Hands-only polish -------------------------------------
    # Unfrozen: lhand_pose, rhand_pose only. Frozen: everything else (from
    # stage 4). Same loss set as stage 4. This stage is bvh2smplx's own
    # addition on top of the base 4-stage design, "to improve finger
    # quality" -- a final pass that can't be undone by body_pose drifting
    # again, since body_pose is frozen here.
    def stage5_loss():
        return stage4_loss()

    log["stages"]["5_hands_only"], _ = _run_lbfgs_stage(
        "5_hands_only", [params.lhand_pose, params.rhand_pose], stage5_loss, max_outer_iters, rel_tol,
    )

    # ---- Stage 6: Joint shape + pose refinement via silhouette (opt-in) --
    # Unfrozen: betas + body_pose + global_orient + transl. hand_pose stays
    # frozen (stages 4/5 just converged it; reopening hands for marginal
    # silhouette benefit on body parts isn't worth undoing that work).
    #
    # Runs AFTER all keypoint stages so the pose is already well-initialised.
    # Jointly optimising shape AND pose lets them co-adapt: the optimal betas
    # depend on the current body_pose (and vice versa), so the earlier
    # betas-only approach hit a local optimum. This is the EasyMoCap
    # "refine_poses" philosophy: a final joint pass over everything that
    # matters for the silhouette.
    #
    # Loss:
    #   k3d          -- anchors body_pose/global_orient/transl to 3D keypoints
    #                   so pose can't drift just to satisfy silhouette
    #   silhouette   -- the new signal; pushes shape+pose toward mask boundary
    #   reg_shape    -- keeps betas near plausible human range (shared across
    #                   all frames AND cameras, so silhouette has strong shape
    #                   constraints already -- reg is a soft backstop, not
    #                   a dominant term)
    #   reg_pose     -- body prior keeps legs/spine from wandering
    #   smooth_*     -- temporal consistency on global motion
    #
    # Uses Adam (not LBFGS): silhouette gradient magnitude << k3d gradient
    # magnitude, so LBFGS's single global step size would be dominated by k3d
    # and effectively freeze betas. lr=0.001 (lower than the earlier 3b's
    # 0.01) to avoid destabilising the converged body_pose from stage 4.
    if use_silhouette_shape:
        def stage6_loss():
            output = params.forward_batch(model)
            pred = smplx_model.predict_all_points(output, full_layout)
            k3d, _ = losses.weighted_keypoint_loss(pred, target_points, confidence, mask=body_mask)
            smooth_transl = losses.temporal_smoothness_loss(params.transl)
            smooth_go = losses.temporal_smoothness_loss(params.global_orient)
            reg_shape = losses.l2_regularization(params.betas)
            reg_pose = body_pose_prior_loss(params.body_pose)
            sil_loss, n_pairs = sil.compute_silhouette_term(
                output.vertices, model.faces, face_idx, bary,
                silhouette_cams, silhouette_masks, silhouette_valid, sigma_px=silhouette_sigma_px,
            )
            total = (k3d * 1.0 + smooth_transl * 0.5 + smooth_go * 0.1
                     + reg_shape * 0.1 + reg_pose * pose_prior_weight
                     + sil_loss * silhouette_weight)
            return total, {
                "k3d": float(k3d.item()),
                "smooth_transl": float(smooth_transl.item()),
                "smooth_global_orient": float(smooth_go.item()),
                "reg_shape": float(reg_shape.item()),
                "reg_pose": float(reg_pose.item()),
                "silhouette": float(sil_loss.item()),
                "silhouette_n_pairs": n_pairs,
            }

        log["stages"]["6_joint_sil_refine"], _ = _run_adam_stage(
            "6_joint_sil_refine",
            [params.betas, params.body_pose, params.global_orient, params.transl],
            stage6_loss, max_iters=200, rel_tol=rel_tol, lr=0.001,
        )

    return params, log
