"""Correspondence between our triangulated keypoints (MediaPipe hand
21-point scheme, COCO upper-body ids 0-12) and SMPL-X's own joints/mesh
vertices -- the same kind of table the SMPLify-X ecosystem builds for
fitting OpenPose-style keypoints (nose/eyes/fingertips have no matching
kinematic joint, so they're pinned to specific mesh vertices instead; see
smplx.vertex_ids).

Verified directly against the installed `smplx` 0.1.28 package -- no SMPL-X
model weight files needed for this, joint names/vertex ids are fixed
package metadata, not derived from any specific body's .npz weights:
  - smplx.joint_names.JOINT_NAMES: the 55-joint body+hand order used by
    every SMPLXLayer's `.joints` output (body 0-21, jaw/eyes 22-24, left
    hand 25-39, right hand 40-54).
  - smplx.vertex_ids.vertex_ids['smplx']: mesh-vertex targets for
    nose/eyes/ears/fingertips.
  - Hand joint order per finger is index/middle/pinky/ring/thumb (NOT
    anatomical left-to-right) -- confirmed directly from JOINT_NAMES; this
    was the one thing flagged as needing empirical verification before
    trusting it, and it does NOT match a naive left-to-right assumption.
"""
from collections import namedtuple

from smplx.joint_names import JOINT_NAMES
from smplx.vertex_ids import vertex_ids as _SMPLX_VERTEX_IDS

VERTEX_IDS = _SMPLX_VERTEX_IDS["smplx"]


def _joint_index(name):
    return JOINT_NAMES.index(name)


# One entry per keypoint we can supply from triangulation. `kind` is
# "joint" (read from a SMPLXLayer output's `.joints[index]`) or "vertex"
# (read from `.vertices[index]`) -- both are just 3D points in the same
# output space, so the k3d loss treats them identically once gathered.
KeypointTarget = namedtuple("KeypointTarget", ["kind", "index", "name"])

# COCO-17 upper-body ids (0-12, per pose2d.triangulation.LANDMARK_IDS_BODY).
# Wrists (9, 10) are deliberately excluded here -- see WRIST_FUSION below.
BODY_TARGETS = {
    "0": KeypointTarget("vertex", VERTEX_IDS["nose"], "nose"),
    "1": KeypointTarget("vertex", VERTEX_IDS["leye"], "left_eye"),
    "2": KeypointTarget("vertex", VERTEX_IDS["reye"], "right_eye"),
    "3": KeypointTarget("vertex", VERTEX_IDS["lear"], "left_ear"),
    "4": KeypointTarget("vertex", VERTEX_IDS["rear"], "right_ear"),
    "5": KeypointTarget("joint", _joint_index("left_shoulder"), "left_shoulder"),
    "6": KeypointTarget("joint", _joint_index("right_shoulder"), "right_shoulder"),
    "7": KeypointTarget("joint", _joint_index("left_elbow"), "left_elbow"),
    "8": KeypointTarget("joint", _joint_index("right_elbow"), "right_elbow"),
    "11": KeypointTarget("joint", _joint_index("left_hip"), "left_hip"),
    "12": KeypointTarget("joint", _joint_index("right_hip"), "right_hip"),
}


def _hand_targets(side):
    """side: 'left' or 'right'. Returns dict[str(mediapipe_id)] -> KeypointTarget
    for the 20 non-wrist landmarks of one hand (ids 1-20; id 0 is the
    wrist, handled via WRIST_FUSION, not included here).

    Each non-tip landmark maps to the SMPL-X finger joint at the SAME
    position in the kinematic chain (MCP->finger1, PIP->finger2,
    DIP->finger3). Tips (ids 4/8/12/16/20) have no matching joint --
    SMPL-X's kinematic tree stops at the distal joint, not the fingertip
    surface -- so they're pinned to the corresponding mesh vertex instead,
    the same choice the SMPLify-X ecosystem makes for OpenPose fingertip
    keypoints.
    """
    prefix = side  # JOINT_NAMES uses e.g. "left_index1"
    v = "l" if side == "left" else "r"  # vertex_ids uses e.g. "lindex"
    fingers = [
        ("thumb", [1, 2, 3], 4),
        ("index", [5, 6, 7], 8),
        ("middle", [9, 10, 11], 12),
        ("ring", [13, 14, 15], 16),
        ("pinky", [17, 18, 19], 20),
    ]
    targets = {}
    for finger_name, joint_landmark_ids, tip_id in fingers:
        for slot, landmark_id in enumerate(joint_landmark_ids, start=1):
            joint_name = f"{prefix}_{finger_name}{slot}"
            targets[str(landmark_id)] = KeypointTarget("joint", _joint_index(joint_name), joint_name)
        targets[str(tip_id)] = KeypointTarget(
            "vertex", VERTEX_IDS[f"{v}{finger_name}"], f"{prefix}_{finger_name}_tip"
        )
    return targets


LEFT_HAND_TARGETS = _hand_targets("left")
RIGHT_HAND_TARGETS = _hand_targets("right")

# COCO wrist id <-> MediaPipe hand-wrist id (landmark "0" of that hand) --
# both observe the exact same physical joint, from two different
# detectors/crops (body-frame vs. hand-zoomed-crop). data_loading.py fuses
# them (confidence-weighted average when both present, whichever is
# present when only one is) into a single target rather than fitting one
# joint against two independent, possibly-conflicting observations. This
# is a judgment call specific to our data (BVH mocap has one skeleton, so
# bvh2smplx never faced this).
WRIST_FUSION = {
    "left": {
        "body_landmark_id": "9",
        "hand_landmark_id": "0",
        "target": KeypointTarget("joint", _joint_index("left_wrist"), "left_wrist"),
    },
    "right": {
        "body_landmark_id": "10",
        "hand_landmark_id": "0",
        "target": KeypointTarget("joint", _joint_index("right_wrist"), "right_wrist"),
    },
}


def build_keypoint_layout():
    """Single fixed ordered list of (part, landmark_id, KeypointTarget)
    covering every DIRECTLY-mapped keypoint (i.e. excluding the fused
    wrists, which have two source landmark ids -- see WRIST_FUSION and
    data_loading.py's fuse_wrists). Both data_loading.py (building the
    target 3D position tensor from our JSON files) and model.py (gathering
    the matching predicted position from a SMPLXLayer forward pass) index
    against this SAME list, so the mapping is derived once and never
    re-guessed in two places.

    `part` is "body", "left", or "right" -- matches
    pose2d.triangulation.load_pose2d_take_data's landmarks_{part} dict keys.
    """
    layout = []
    for landmark_id, target in BODY_TARGETS.items():
        layout.append(("body", landmark_id, target))
    for landmark_id, target in LEFT_HAND_TARGETS.items():
        layout.append(("left", landmark_id, target))
    for landmark_id, target in RIGHT_HAND_TARGETS.items():
        layout.append(("right", landmark_id, target))
    return layout


def is_hand_target(part):
    """True for left/right hand entries -- used to apply k3d_hand's higher
    weight (hands matter more for sign language than body pose) separately
    from k3d's body weight.
    """
    return part in ("left", "right")
