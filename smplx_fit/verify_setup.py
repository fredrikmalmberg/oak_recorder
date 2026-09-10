"""Stage-0/1 standalone verification (see the approved plan's Rollout
section) -- run this BEFORE trusting anything in optimize.py/fit_take.py.

Loads the real SMPL-X model weights, runs a zero-pose ("T-pose") forward
pass, and checks that everything joint_mapping.py assumes (output array
shapes, vertex index validity, joint count) actually holds against the real
model -- not just against the package's fixed metadata (joint_mapping.py's
own docstring already verified names/indices against smplx's *shipped*
constants, which needs no model weights; this script is the second half:
confirming those constants are consistent with the specific model file we
actually have).

Usage:
    python -m smplx_fit.verify_setup --model-path models/SMPLX
"""
import argparse
import os

import torch
import trimesh

from smplx_fit import joint_mapping as jm
from smplx_fit import model as smplx_model


def zero_pose_forward(model, batch_size=1):
    betas = torch.zeros(batch_size, model.num_betas)
    global_orient = torch.zeros(batch_size, 3)
    body_pose = torch.zeros(batch_size, 21, 3)
    left_hand_pose = torch.zeros(batch_size, 15, 3)
    right_hand_pose = torch.zeros(batch_size, 15, 3)
    transl = torch.zeros(batch_size, 3)
    return smplx_model.forward(
        model, betas, global_orient, body_pose, left_hand_pose, right_hand_pose, transl,
    )


def check_layout_against_model(layout, wrist_fusion, output, model):
    """joint_mapping.py's indices were built from smplx's package-level
    constants (JOINT_NAMES, vertex_ids) -- this cross-checks them against
    the ACTUAL loaded model's output array sizes, so a version mismatch or
    a wrong assumption fails loudly here rather than silently in the
    optimizer later.
    """
    n_joints = output.joints.shape[1]
    n_verts = output.vertices.shape[1]
    print(f"model output: joints={n_joints}, vertices={n_verts}")

    problems = []
    all_targets = [t for _, _, t in layout] + [f["target"] for f in wrist_fusion.values()]
    for target in all_targets:
        if target.kind == "joint" and target.index >= n_joints:
            problems.append(f"{target.name}: joint index {target.index} >= n_joints {n_joints}")
        if target.kind == "vertex" and target.index >= n_verts:
            problems.append(f"{target.name}: vertex index {target.index} >= n_vertices {n_verts}")

    if problems:
        print("PROBLEMS FOUND:")
        for p in problems:
            print("  -", p)
        raise RuntimeError(f"{len(problems)} joint_mapping.py entries are invalid against this model")
    print(f"OK: all {len(all_targets)} keypoint-layout targets (joint+vertex) are valid indices for this model.")


FINGER_TIP_VERTEX_NAMES = ["thumb", "index", "middle", "ring", "pinky"]


def check_hand_finger_ordering(model, side="left"):
    """Step 2 of the plan's rollout: perturb each of the 15 hand_pose
    slots individually (a moderate bend, not a full rotation) and confirm
    which fingertip moves the most -- the ONLY reliable way to confirm the
    slot order joint_mapping.py assumes (index/middle/pinky/ring/thumb,
    derived from smplx.joint_names.JOINT_NAMES) actually matches what this
    model does when posed, not just what the names suggest.
    """
    layout = jm.build_keypoint_layout()
    tip_names = {"left": {}, "right": {}}
    for part, lid, target in layout:
        if part == side and target.name.endswith("_tip"):
            finger = target.name.split("_")[1]  # "left_index_tip" -> "index"
            tip_names[side][finger] = target.index

    baseline = zero_pose_forward(model)
    baseline_tips = {f: baseline.vertices[0, idx].detach().numpy() for f, idx in tip_names[side].items()}

    print(f"\n--- Hand finger-ordering check ({side} hand) ---")
    print("Expected slot groups (0-indexed): 0-2=index, 3-5=middle, 6-8=pinky, 9-11=ring, 12-14=thumb")
    slot_to_finger = {}
    for i, finger in enumerate(["index", "middle", "pinky", "ring", "thumb"]):
        for j in range(3):
            slot_to_finger[i * 3 + j] = finger

    for slot in range(15):
        hand_pose = torch.zeros(1, 15, 3)
        hand_pose[0, slot, 1] = 0.8  # bend around one axis, moderate angle
        kwargs = dict(
            betas=torch.zeros(1, model.num_betas), global_orient=torch.zeros(1, 3),
            body_pose=torch.zeros(1, 21, 3), transl=torch.zeros(1, 3),
        )
        if side == "left":
            out = smplx_model.forward(model, left_hand_pose=hand_pose, right_hand_pose=torch.zeros(1, 15, 3), **kwargs)
        else:
            out = smplx_model.forward(model, left_hand_pose=torch.zeros(1, 15, 3), right_hand_pose=hand_pose, **kwargs)

        displacements = {}
        for finger, idx in tip_names[side].items():
            new_pos = out.vertices[0, idx].detach().numpy()
            displacements[finger] = float(((new_pos - baseline_tips[finger]) ** 2).sum() ** 0.5)
        moved_most = max(displacements, key=displacements.get)
        expected = slot_to_finger[slot]
        status = "OK" if moved_most == expected else "MISMATCH"
        print(f"  slot {slot:2d} (expected={expected:6s}): moved-most={moved_most:6s} "
              f"disp={displacements} [{status}]")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", default="models/SMPLX")
    parser.add_argument("--gender", default="neutral")
    parser.add_argument("--out-obj", default=os.path.join("smplx_fit", "verify_tpose.obj"))
    parser.add_argument("--perturb", action="store_true", help="Also run Step 2's hand finger-ordering check")
    args = parser.parse_args()

    print(f"Loading SMPLXLayer from {args.model_path} (gender={args.gender})...")
    model = smplx_model.load_layer(args.model_path, gender=args.gender)
    print("Loaded OK.")

    output = zero_pose_forward(model)
    print(f"Zero-pose forward pass OK. vertices shape={tuple(output.vertices.shape)}, "
          f"joints shape={tuple(output.joints.shape)}")

    layout = jm.build_keypoint_layout()
    check_layout_against_model(layout, jm.WRIST_FUSION, output, model)

    verts = output.vertices[0].detach().numpy()
    faces = model.faces
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh.export(args.out_obj)
    print(f"Saved T-pose mesh to {args.out_obj} -- open it to eyeball-confirm it looks like a plausible body.")

    # Print a few key target positions directly so they can be sanity-
    # checked by eye against the T-pose (e.g. left_shoulder should be a
    # small positive x, positive z; nose should be near the head's front).
    print("\nSample target positions (T-pose, meters):")
    for part, lid, target in layout:
        if target.name in ("left_shoulder", "right_shoulder", "nose", "left_index_tip", "right_thumb_tip"):
            pos = (output.joints[0, target.index] if target.kind == "joint" else output.vertices[0, target.index])
            print(f"  {target.name:20s} ({part} id={lid}, {target.kind} {target.index}): {pos.detach().numpy()}")

    if args.perturb:
        check_hand_finger_ordering(model, side="left")
        check_hand_finger_ordering(model, side="right")


if __name__ == "__main__":
    main()
