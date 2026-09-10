"""Loss terms for the 5-stage optimizer (see optimize.py). One function per
term, matching the loss categories bvh2smplx's README documents (3D
keypoint distance, limb-length/shape consistency, temporal smoothness, L2
pose regularization) -- reimplemented against our own data, since
bvh2smplx's actual formulas aren't available (its core optimizer is an
external repo not vendored in that checkout).

All functions take (and return) plain torch tensors -- no state, so they
can be unit-tested and logged individually (see optimize.py's
_run_lbfgs_stage, which logs every term's raw value every iteration, not
just their weighted sum, per the project's "document each step" requirement).
"""
import torch

from hand_pose import hand_multiview as hmv

# Bone pairs are the SAME anatomically-adjacent-link topology already used
# to draw skeletons elsewhere in this project (hand_pose.hand_multiview's
# HAND_CONNECTIONS / BODY_CONNECTIONS_COCO_UPPER), split here into two
# reliability tiers -- a classification those lists don't carry themselves,
# since drawing a line doesn't need to know whether the two endpoints form
# a rigid bone.
#
# EXACT: genuinely single rigid bones (a real skeletal segment whose
# length a pose change cannot alter) -- full weight in shape3d.
# APPROXIMATE: spans that cross a flexible joint (spine, palm splay) where
# bone length is only roughly pose-invariant -- included but down-weighted,
# rather than either trusted fully or dropped outright.
BODY_BONES_EXACT = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 7), (7, 9), (6, 8), (8, 10), (11, 12)]
BODY_BONES_APPROXIMATE = [(5, 6), (5, 11), (6, 12)]
HAND_BONES_EXACT = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]
HAND_BONES_APPROXIMATE = [(5, 9), (9, 13), (13, 17)]

APPROXIMATE_BONE_WEIGHT = 0.2


def _assert_partition_matches(exact, approximate, source, label):
    """Fails loudly at import time if this module's EXACT/APPROXIMATE
    classification ever drifts out of sync with the source connection list
    it's supposed to fully partition (e.g. if hand_multiview.py's topology
    is edited later without updating this file too).
    """
    combined = {frozenset(pair) for pair in exact + approximate}
    expected = {frozenset(pair) for pair in source}
    if combined != expected:
        raise AssertionError(
            f"{label} bone classification doesn't match {label} connections: "
            f"missing={expected - combined}, extra={combined - expected}"
        )


_assert_partition_matches(BODY_BONES_EXACT, BODY_BONES_APPROXIMATE, hmv.BODY_CONNECTIONS_COCO_UPPER, "body")
_assert_partition_matches(HAND_BONES_EXACT, HAND_BONES_APPROXIMATE, hmv.HAND_CONNECTIONS, "hand")


def weighted_keypoint_loss(pred, target, confidence, mask=None):
    """k3d / k3d_hand: Sum_{n,k} conf[n,k] * ||pred[n,k] - target[n,k]||^2
    / Sum conf[n,k], restricted to `mask` (a boolean index over the K axis)
    if given. Guards the all-zero-weight case explicitly -- a silent 0/0
    would produce a NaN that then poisons every downstream gradient with
    no indication of where it came from.
    """
    if mask is not None:
        pred, target, confidence = pred[:, mask], target[:, mask], confidence[:, mask]
    total_weight = confidence.sum()
    if total_weight <= 0:
        return torch.zeros((), dtype=pred.dtype, device=pred.device), True  # (loss, was_empty)
    sq_err = ((pred - target) ** 2).sum(dim=-1)
    return (confidence * sq_err).sum() / total_weight, False


def shape3d_loss(target_points, confidence, rest_pose_points, layout_index_by_local_id):
    """Limb-length-consistency shape loss -- our replacement for bvh2smplx's
    ground-truth-BVH-limb-length term, since we have no ground truth, only
    the triangulated keypoints themselves. For each bone (a, b):
      target_len = median over confident frames of ||X[n,a] - X[n,b]||
      pred_len   = ||J_a(betas) - J_b(betas)|| in the model's REST pose
                   (body_pose=0), which is where a bone's two endpoints'
                   distance under the current shape hypothesis is exactly
                   that bone's length -- true for ANY pose only for the
                   EXACT bones (see module docstring); APPROXIMATE bones
                   are included with a lower weight since their rest-pose
                   length is only a rough proxy for their true length.

    layout_index_by_local_id: dict mapping a bone-pair's local landmark id
    (e.g. hand landmark "0", "5", ...) to that landmark's row index `k` in
    target_points/rest_pose_points -- built once by the caller (optimize.py)
    since bone pairs are expressed in each detector's own local numbering
    (0-20 for a hand, COCO ids for body) while target_points is indexed by
    the flat layout order.
    """
    parts_bones = {
        "body": ((BODY_BONES_EXACT, 1.0), (BODY_BONES_APPROXIMATE, APPROXIMATE_BONE_WEIGHT)),
        "left": ((HAND_BONES_EXACT, 1.0), (HAND_BONES_APPROXIMATE, APPROXIMATE_BONE_WEIGHT)),
        "right": ((HAND_BONES_EXACT, 1.0), (HAND_BONES_APPROXIMATE, APPROXIMATE_BONE_WEIGHT)),
    }
    total_loss = torch.zeros((), dtype=rest_pose_points.dtype)
    total_weight = 0.0
    for part, bones_by_weight in parts_bones.items():
        idx = layout_index_by_local_id[part]
        for bones, weight in bones_by_weight:
            for a, b in bones:
                ka, kb = idx.get(str(a)), idx.get(str(b))
                if ka is None or kb is None:
                    continue
                conf_ab = confidence[:, ka] * confidence[:, kb]
                if conf_ab.sum() <= 0:
                    continue
                observed_len = torch.norm(target_points[:, ka] - target_points[:, kb], dim=-1)
                target_len = torch.median(observed_len[conf_ab > 0])
                pred_len = torch.norm(rest_pose_points[ka] - rest_pose_points[kb])
                total_loss = total_loss + weight * (target_len - pred_len) ** 2
                total_weight += weight
    if total_weight == 0:
        return torch.zeros((), dtype=rest_pose_points.dtype), True
    return total_loss / total_weight, False


def temporal_smoothness_loss(params):
    """smooth_*: mean_n ||P[n+1] - P[n]||^2 over the FULL-frame-indexed
    array (see data_loading.py point 2) -- a true gap in the middle of a
    take is exactly where this term (plus reg_*) is doing the real work of
    filling in a plausible pose, not an artifact to special-case around.
    """
    diffs = params[1:] - params[:-1]
    return (diffs ** 2).sum(dim=-1).mean()


def l2_regularization(params):
    """reg_*: mean(params^2), pulling toward zero/mean pose. This is also
    the ENTIRE mechanism holding SMPL-X's leg-related body_pose slots
    (knees/ankles/feet) near neutral, since we supply no leg keypoints at
    all (see joint_mapping.py / data_loading.py) -- those slots never
    receive any k3d gradient, only this prior (or, once --use-silhouette is
    on, the silhouette term below too). Without silhouette, a fit's legs
    sit in a static, roughly-neutral stance regardless of what the real
    signer's legs were doing.
    """
    return (params ** 2).mean()


def silhouette_loss(rendered, mask_target):
    """Phase 3 (EasyMoCap-inspired plan) -- the pixel-wise term comparing
    silhouette.render_silhouette's soft-splatted occupancy image against a
    real Phase 2 segmentation mask (see smplx_fit/silhouette.py's module
    docstring for the rendering side of this; this function is just the
    comparison, kept here with every other loss term per this module's own
    one-function-per-term convention). Plain MSE -- simple and sufficient
    for this feature's current validated scope (pulling body_pose's
    otherwise-unconstrained leg slots toward the real silhouette), not
    tuned further until proven worth refining past that.
    """
    return ((rendered - mask_target) ** 2).mean()
