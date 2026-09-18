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


def upper_body_faces(model, hide_pelvis=False):
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
