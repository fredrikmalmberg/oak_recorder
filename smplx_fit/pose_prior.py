"""Pluggable body pose prior for optimize.py's stage 2/3 `reg_pose` term.

Three backends, selected via fit_take.py's `--pose-prior` flag:
  - "l2" (default): today's behavior, unchanged -- mean(body_pose**2).
  - "gmm": the classic SMPLify/SMPLify-X `MaxMixturePrior` -- an 8-component
    Gaussian-mixture negative-log-likelihood over pose, fit to real human
    mocap data. Penalizes anatomically implausible joint-angle
    *combinations*, not just "far from zero" -- more meaningful than L2 for
    body_pose slots we have almost no real keypoint constraint for
    (legs: zero keypoints at all; torso: sparse).
  - "none": prior fully disabled (equivalent to weight 0).

Deliberately BODY-ONLY. Hands are never touched by this module -- per an
explicit project decision, hand pose must stay purely data-driven (sign
language relies on precise, sometimes-unusual hand shapes that a generic
prior would incorrectly penalize), so hand regularization stays the
simple, independently-controlled `reg_hand` L2 term in optimize.py.
"""
import pickle

import numpy as np
import torch

DEFAULT_GMM_PATH = "models/Pose_prior/gmm_08.pkl"

# `--pose-prior-weight` needs a different default per backend -- "l2"'s
# mean(body_pose**2) is naturally tiny (body_pose is small angles in
# radians, so this sits around O(0.01-0.1)), while the GMM's negative
# log-likelihood lives on a completely different, much larger absolute
# scale (O(10-100+), even at a fully plausible pose -- see
# MaxMixturePrior's docstring). Reusing "l2"'s weight (0.01) for "gmm"
# confirmed empirically this session to badly over-regularize body_pose
# (reg_pose*weight dominated the total loss), which distorted the
# shoulder/elbow chain enough to measurably degrade HAND fit quality too
# (5mm -> 16mm RMS) via forward kinematics, even though hand_pose itself
# is never touched by this prior. 0.0001 was confirmed to bring GMM's
# effective regularization pull back in line with "l2"'s (comparable
# body/hand RMS numbers) while still meaningfully constraining body_pose.
DEFAULT_POSE_PRIOR_WEIGHTS = {"l2": 0.01, "gmm": 0.0001, "none": 0.01}

# gmm_08.pkl was trained on SMPL's 69-dim body pose (23 joints). SMPL-X's
# body_pose is only 63-dim (21 joints -- hands are separate parameters in
# SMPL-X). SMPL-X's body joints are a prefix-compatible subset of SMPL's in
# the same kinematic order (both share the same joint tree up through the
# wrists), so truncating the GMM's per-component mean/covariance to the
# first 63 dimensions is the standard SMPLify-X approach for reusing this
# exact prior file with SMPL-X.
SMPLX_BODY_POSE_DIM = 63


class MaxMixturePrior:
    """Loads gmm_08.pkl and exposes __call__(body_pose) -> scalar NLL.

    Uses the "max mixture" approximation (the class's namesake, and the
    same one SMPLify/SMPLify-X use): rather than a full log-sum-exp over
    all 8 components (which behaves poorly in this many dimensions -- each
    individual Gaussian's density is astronomically small in 63-dim space,
    so summing them gives a nearly flat, uninformative gradient), take the
    MINIMUM per-component penalized Mahalanobis distance -- i.e. score
    against whichever of the 8 pose "prototypes" fits best. This gives a
    much better-scaled gradient, pulling toward the nearest plausible pose
    cluster instead of an uninformative global average.
    """

    def __init__(self, gmm_path=DEFAULT_GMM_PATH, target_dim=SMPLX_BODY_POSE_DIM, epsilon=1e-12):
        try:
            with open(gmm_path, "rb") as f:
                data = pickle.load(f, encoding="latin1")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"GMM pose prior file not found at {gmm_path!r}. Pass --gmm-prior-path "
                f"to point at gmm_08.pkl (the classic SMPLify/SMPLify-X pose prior), or "
                f"use --pose-prior l2/none instead."
            )
        except (pickle.UnpicklingError, UnicodeDecodeError, KeyError) as e:
            raise RuntimeError(
                f"Failed to load GMM pose prior from {gmm_path!r} ({e!r}). This is often a "
                f"Python 2/3 pickle encoding mismatch -- this loader already uses "
                f"encoding='latin1', which resolves the common case; if it still fails, "
                f"the file may not be the expected SMPLify-style dict of "
                f"weights/means/covars."
            )

        means = np.asarray(data["means"], dtype=np.float64)  # (n_components, 69)
        covars = np.asarray(data["covars"], dtype=np.float64)  # (n_components, 69, 69)
        weights = np.asarray(data["weights"], dtype=np.float64)  # (n_components,)

        means = means[:, :target_dim]
        covars = covars[:, :target_dim, :target_dim]
        n_components, dim = means.shape

        precisions, log_norm_consts = [], []
        for k in range(n_components):
            cov = covars[k] + np.eye(dim) * epsilon  # numeric stability against near-singular covariances
            precisions.append(np.linalg.inv(cov))
            _sign, logdet = np.linalg.slogdet(cov)
            log_norm_consts.append(0.5 * (dim * np.log(2.0 * np.pi) + logdet))
        log_norm_consts = np.stack(log_norm_consts)

        # Shift all components' normalization constants by the smallest one
        # (the reference SMPLify-X `MaxMixturePrior` does the equivalent via
        # its `sqrdets / sqrdets.min()` ratio). An ABSOLUTE Gaussian
        # normalization constant can be arbitrarily negative for a
        # tightly-peaked (near-singular-covariance) component -- confirmed
        # empirically this session: without this shift, the optimizer could
        # slash `reg_pose` to -96+ by driving body_pose toward that
        # component's mean, which dominated the total loss and measurably
        # degraded HAND fit quality (5mm -> 16mm RMS) even though hands are
        # never supposed to be touched by this prior. Shifting by the
        # minimum makes every component's normalization term >= 0 (the
        # best-conditioned component contributes 0), so the whole per-
        # component NLL -- and thus the mixture's min over components -- is
        # bounded below by 0, while leaving the RELATIVE ranking between
        # components (which one is "closest"/best) exactly unchanged.
        log_norm_consts = log_norm_consts - log_norm_consts.min()

        self.means = torch.tensor(means, dtype=torch.float32)
        self.precisions = torch.tensor(np.stack(precisions), dtype=torch.float32)
        self.log_norm_consts = torch.tensor(log_norm_consts, dtype=torch.float32)
        self.log_weights = torch.log(torch.tensor(weights, dtype=torch.float32) + 1e-16)
        self.dim = dim

        # Sanity check (per the approved plan): the NLL at body_pose=0 should
        # land near the mixture's peak (low/typical), not an outlier -- if
        # truncating 69->63 dims corrupted the covariance structure, this
        # would likely show up as an implausibly large value here.
        with torch.no_grad():
            zero_nll = float(self(torch.zeros(1, dim)).item())
        if zero_nll > 500.0:  # a generous threshold -- typical values are O(1-50) for this prior
            raise RuntimeError(
                f"GMM prior sanity check failed: NLL at body_pose=0 is {zero_nll:.1f}, "
                f"implausibly high for a should-be-common pose. The 69->{target_dim} "
                f"truncation may not be valid for this specific gmm file."
            )

    def __call__(self, body_pose):
        """body_pose: (B, dim) or (B, n_joints, 3) axis-angle. Returns a
        scalar (mean over batch) negative log-likelihood.
        """
        x = body_pose.reshape(body_pose.shape[0], -1)
        diff = x.unsqueeze(1) - self.means.unsqueeze(0)  # (B, K, dim)
        mahalanobis = torch.einsum("bkd,kde,bke->bk", diff, self.precisions, diff)  # (B, K)
        per_component_nll = 0.5 * mahalanobis + self.log_norm_consts.unsqueeze(0) - self.log_weights.unsqueeze(0)
        best_component_nll, _ = torch.min(per_component_nll, dim=1)  # (B,)
        return best_component_nll.mean()


def compute_body_pose_prior(body_pose, backend, gmm=None):
    """backend: "l2" | "gmm" | "none". `gmm` is a pre-loaded MaxMixturePrior
    instance (loading involves disk I/O + a sanity-check forward pass, so
    callers load it once and pass it in, not re-load per call).
    """
    if backend == "none":
        return torch.zeros((), dtype=body_pose.dtype, device=body_pose.device)
    if backend == "l2":
        return (body_pose ** 2).mean()
    if backend == "gmm":
        if gmm is None:
            raise ValueError("backend='gmm' requires a loaded MaxMixturePrior instance (see load_gmm_prior).")
        return gmm(body_pose)
    raise ValueError(f"Unknown pose-prior backend: {backend!r} (expected 'l2', 'gmm', or 'none')")


def load_gmm_prior(gmm_path=DEFAULT_GMM_PATH):
    return MaxMixturePrior(gmm_path=gmm_path)
