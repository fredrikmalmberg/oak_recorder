"""SMPLXLayer construction and the forward-pass plumbing shared by
verification, single-frame checks, and the real optimizer.

We use the "Layer" variant (matching bvh2smplx's own choice, per its
README) rather than the plain SMPLX/`smplx.create()` model class -- it's a
pure function of whatever parameters are passed in, with no internal
learnable state of its own, so every parameter is something OUR optimizer
explicitly owns and can save/load, rather than a default the model quietly
carries.

Important API detail (discovered by running this against the real model
weights, not documented anywhere obvious): SMPLXLayer.forward expects pose
parameters as ROTATION MATRICES (global_orient: Bx3x3, body_pose: BxJx3x3,
hand_pose: Bx15x3x3) -- NOT axis-angle vectors like the plain SMPLX class
accepts. We still store/optimize/regularize/save parameters as axis-angle
(3-vectors) throughout this package -- it's what bvh2smplx's own .npz
output format uses, what temporal-smoothness and L2-toward-zero priors are
naturally defined on, and there's exactly one place (axis_angle_to_rotmat,
called right before each forward pass) that needs to know about the
matrix requirement at all.
"""
import torch
from smplx import SMPLXLayer
from smplx.lbs import batch_rodrigues


def load_layer(model_path, gender="neutral", num_betas=300):
    return SMPLXLayer(
        model_path=model_path,
        gender=gender,
        num_betas=num_betas,
        use_pca=False,  # full 3-DOF-per-joint hand pose, not PCA-compressed -- matches bvh2smplx
        flat_hand_mean=False,
    )


def axis_angle_to_rotmat(aa):
    """aa: (..., 3) axis-angle -> (..., 3, 3) rotation matrices, via
    smplx.lbs.batch_rodrigues (the same conversion the plain SMPLX class
    uses internally -- SMPLXLayer just doesn't do it for you, since it's
    meant to accept either representation from an external predictor).
    """
    *lead, three = aa.shape
    assert three == 3
    flat = aa.reshape(-1, 3)
    rotmat = batch_rodrigues(flat)
    return rotmat.reshape(*lead, 3, 3)


def forward(model, betas, global_orient, body_pose, left_hand_pose, right_hand_pose, transl):
    """One SMPLXLayer forward pass. All pose args are axis-angle (see
    module docstring); converted to rotation matrices here, the one place
    that needs to know about SMPLXLayer's matrix requirement.

    betas: (1, num_betas) or (B, num_betas) -- one shared shape for the
    whole take is the normal case here (see optimize.py), broadcast by the
    caller to match B before calling this.
    global_orient, transl: (B, 3)
    body_pose: (B, 21, 3)
    left_hand_pose, right_hand_pose: (B, 15, 3)
    """
    return model(
        betas=betas,
        global_orient=axis_angle_to_rotmat(global_orient),
        body_pose=axis_angle_to_rotmat(body_pose),
        left_hand_pose=axis_angle_to_rotmat(left_hand_pose),
        right_hand_pose=axis_angle_to_rotmat(right_hand_pose),
        transl=transl,
        return_verts=True,
    )


def forward_with_expr(model, betas, global_orient, body_pose, left_hand_pose,
                      right_hand_pose, transl, expression, jaw_pose):
    """Like forward() but also passes FLAME expression + jaw_pose parameters.

    expression: (B, 10)  -- FLAME expression coefficients
    jaw_pose:   (B, 3)   -- jaw rotation in axis-angle; converted to (B,3,3) here
    """
    return model(
        betas=betas,
        global_orient=axis_angle_to_rotmat(global_orient),
        body_pose=axis_angle_to_rotmat(body_pose),
        left_hand_pose=axis_angle_to_rotmat(left_hand_pose),
        right_hand_pose=axis_angle_to_rotmat(right_hand_pose),
        transl=transl,
        expression=expression,
        jaw_pose=axis_angle_to_rotmat(jaw_pose),
        return_verts=True,
    )


# Lower-body joints in smplx.joint_names.JOINT_NAMES's fixed ordering:
# hips(1,2), knees(4,5), ankles(7,8), feet(10,11). Pelvis(0) is deliberately
# NOT included -- its dominant-weight vertices are the glutes/crotch region,
# which reads visually as part of the torso, not "legs"; cutting at the hip
# joint instead gives a cleaner torso base with no gap.
LOWER_BODY_JOINTS = {1, 2, 4, 5, 7, 8, 10, 11}
# Pelvis (joint 0) is kept by default because its dominant-weight vertices
# are the glutes/crotch area — it reads as a torso base. Add it to the
# exclusion set via hide_pelvis=True when a tighter upper-body cut is wanted.
PELVIS_JOINT = 0
HAND_JOINTS = (20, 21) + tuple(range(25, 55))  # wrists + all finger joints


def upper_body_faces(model, hide_pelvis=False, hide_hands=False):
    """Returns a (F, 3) face-index array containing only faces with NO
    vertex dominantly skinned (argmax of that vertex's lbs_weights row) to
    a lower-body joint -- i.e. legs/feet excluded, everything else kept.
    Classification via lbs_weights (always available on the model, no
    extra vertex-segmentation file needed) rather than a fixed vertex-index
    list, so it stays correct regardless of SMPL-X mesh topology version.

    hide_pelvis=True also excludes the pelvis joint (glutes/lower abdomen),
    giving a tighter cut just above the hip line.
    """
    exclude = set(LOWER_BODY_JOINTS)
    if hide_pelvis:
        exclude.add(PELVIS_JOINT)
    if hide_hands:
        exclude.update(HAND_JOINTS)
    dominant_joint = model.lbs_weights.argmax(dim=1)
    lower_mask = torch.tensor(
        [int(j) in exclude for j in dominant_joint], dtype=torch.bool,
    )
    faces_t = torch.as_tensor(model.faces.astype("int64"))
    keep = ~lower_mask[faces_t].any(dim=1)
    return model.faces[keep.numpy()]


def predict_all_points(output, full_layout):
    """output: an SMPLXOutput from forward() above. full_layout: the SAME
    (part, landmark_id, KeypointTarget) list data_loading.build_take_arrays
    returns (build_keypoint_layout()'s entries plus the two fused-wrist
    entries appended at the end) -- the fused-wrist entries already carry a
    valid KeypointTarget pointing at the ordinary wrist joint, so no
    special-casing is needed here: fusion is a TARGET-side concept only
    (combining two different observed sources into one number), the model
    itself only ever has one wrist joint to predict either way.

    Returns a (B, K, 3) tensor in exactly `full_layout`'s order, so it can
    be compared directly against a target tensor built the same way.
    """
    points = []
    for _part, _landmark_id, target in full_layout:
        source = output.joints if target.kind == "joint" else output.vertices
        points.append(source[:, target.index, :])
    return torch.stack(points, dim=1)


def apply_sapiens_leg_offsets(pred, full_layout):
    """Add sapiens_offsets.SAPIENS_LEG_OFFSET_MM to the leg rows, in the person frame built from the predicted shoulders and hips
    (x right -> left shoulder, y towards the shoulders, z = x cross y); frame detached. pred: (B, K, 3)."""
    from smplx_fit import sapiens_offsets as so

    rows = {lid: k for k, (part, lid, _t) in enumerate(full_layout) if part == "body" and lid is not None}
    if not all(i in rows for i in ("5", "6", "11", "12")):
        return pred
    pred = pred.clone()
    ls, rs, lh, rh = (pred[:, rows[i]].detach() for i in ("5", "6", "11", "12"))
    x = torch.nn.functional.normalize(ls - rs, dim=-1)
    z = torch.nn.functional.normalize(torch.cross(x, 0.5 * (ls + rs) - 0.5 * (lh + rh), dim=-1), dim=-1)
    y = torch.cross(z, x, dim=-1)
    R = torch.stack([x, y, z], dim=-1)
    for lid, off in so.SAPIENS_LEG_OFFSET_MM.items():
        if lid in rows:
            pred[:, rows[lid]] = pred[:, rows[lid]] + R @ (torch.tensor(off, dtype=pred.dtype, device=pred.device) * so.MM_TO_M)
    return pred


def apply_sapiens_offsets(pred, full_layout):
    """Turn SMPL-X predictions into what Sapiens would report for the same pose: add the per-landmark offsets of
    sapiens_offsets.SAPIENS_HAND_OFFSET_MM (wrist + the five MCPs) in the model's own hand frame (x wrist -> middle MCP,
    z palm normal, built from the predicted wrist and index/middle/pinky MCP joints, same recipe as synth_analyze.frame_from).
    The frame is detached so the offset acts as a fixed shift, not a pull on the hand orientation. pred: (B, K, 3)."""
    from smplx_fit import sapiens_offsets as so

    pred = pred.clone()
    for side in ("left", "right"):
        rows = {}
        for k, (part, lid, _t) in enumerate(full_layout):
            if part == f"wrist_{side}":
                rows[0] = k
            elif part == side and lid is not None and int(lid) in so.FIT_CORRECTED_LANDMARKS:
                rows[int(lid)] = k
        if not all(i in rows for i in (0, 5, 9, 17)):
            continue
        w, im, m, pk = (pred[:, rows[i]].detach() for i in (0, 5, 9, 17))
        x = m - w
        n = torch.cross(im - w, pk - w, dim=-1)
        n = n - (n * x).sum(-1, keepdim=True) / (x * x).sum(-1, keepdim=True) * x
        x, n = torch.nn.functional.normalize(x, dim=-1), torch.nn.functional.normalize(n, dim=-1)
        R = torch.stack([x, torch.cross(n, x, dim=-1), n], dim=-1)  # (B, 3, 3)
        for lid, key in so.FIT_CORRECTED_LANDMARKS.items():
            if lid not in rows:
                continue
            off = torch.tensor(so.SAPIENS_HAND_OFFSET_MM[key], dtype=pred.dtype, device=pred.device) * so.MM_TO_M
            if side == "right":
                off = off * torch.tensor([1.0, 1.0, -1.0], dtype=pred.dtype, device=pred.device)
            pred[:, rows[lid]] = pred[:, rows[lid]] + (R @ off)
    return pred
