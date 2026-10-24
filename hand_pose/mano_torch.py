"""Minimal differentiable MANO (right hand) in torch, loaded from the original MANO_RIGHT.pkl without chumpy.

forward(betas[B,10], rotmats[B,16,3,3] (root first), transl[B,3]) -> verts[B,778,3], joints21[B,21,3] in metres.
joints21 = the 16 MANO joints + 5 fingertip vertices, reordered to MediaPipe/OpenPose order (wrist, thumb..pinky, 4 each).
Left hands are handled by mirroring (x -> -x) targets and results, not by a separate model.
"""
import pickle
import numpy as np
import torch

MANO_PKL = "/home/fmalmb/CODE/WiLoR/mano_data/MANO_RIGHT.pkl"
TIP_VERTS = [744, 320, 443, 554, 671]  # thumb, index, middle, ring, pinky
OPENPOSE = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]


class _Dummy:
    def __init__(self, *a, **k):
        pass

    def __setstate__(self, st):
        self.__dict__.update(st if isinstance(st, dict) else {"state": st})


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        return _Dummy if module.startswith("chumpy") else super().find_class(module, name)


def rodrigues(aa):
    """aa[..., 3] -> rotation matrices [..., 3, 3]."""
    th = aa.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    k = aa / th
    K = torch.zeros(*aa.shape[:-1], 3, 3, dtype=aa.dtype, device=aa.device)
    K[..., 0, 1], K[..., 0, 2] = -k[..., 2], k[..., 1]
    K[..., 1, 0], K[..., 1, 2] = k[..., 2], -k[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -k[..., 1], k[..., 0]
    s, c = torch.sin(th)[..., None], torch.cos(th)[..., None]
    I = torch.eye(3, dtype=aa.dtype, device=aa.device)
    return I + s * K + (1 - c) * (K @ K)


class Mano:
    def __init__(self, device="cpu", pkl=MANO_PKL):
        d = _Unpickler(open(pkl, "rb"), encoding="latin1").load()
        t = lambda x: torch.tensor(np.asarray(x, dtype=np.float64), dtype=torch.float32, device=device)
        self.v_template, self.shapedirs, self.posedirs = t(d["v_template"]), t(d["shapedirs"]), t(d["posedirs"])
        self.J_reg, self.weights = t(d["J_regressor"].toarray()), t(d["weights"])
        self.hands_mean = t(d["hands_mean"]).reshape(15, 3)
        self.faces = np.asarray(d["f"], dtype=np.int64)
        self.parents = [int(p) for p in d["kintree_table"][0]]
        self.parents[0] = -1
        self.device = device

    def forward(self, betas, rotmats, transl):
        B = betas.shape[0]
        v_shaped = self.v_template + torch.einsum("vcs,bs->bvc", self.shapedirs, betas)
        J = torch.einsum("jv,bvc->bjc", self.J_reg, v_shaped)
        I = torch.eye(3, device=rotmats.device)
        pose_feat = (rotmats[:, 1:] - I).reshape(B, -1)
        v_posed = v_shaped + torch.einsum("vcp,bp->bvc", self.posedirs, pose_feat)
        rel = J.clone()
        rel[:, 1:] = J[:, 1:] - J[:, self.parents[1:]]
        T = torch.zeros(B, 16, 4, 4, device=rotmats.device)
        T[..., :3, :3], T[..., :3, 3], T[..., 3, 3] = rotmats, rel, 1.0
        G = [T[:, 0]]
        for j in range(1, 16):
            G.append(G[self.parents[j]] @ T[:, j])
        G = torch.stack(G, 1)
        joints = G[..., :3, 3] + transl[:, None]
        Jh = torch.cat([J, torch.zeros(B, 16, 1, device=J.device)], -1)[..., None]
        G = G - torch.cat([torch.zeros(B, 16, 4, 3, device=J.device), G @ Jh], -1)
        Tv = torch.einsum("vj,bjrc->bvrc", self.weights, G)
        verts = (Tv[..., :3, :3] @ v_posed[..., None]).squeeze(-1) + Tv[..., :3, 3] + transl[:, None]
        j16 = torch.cat([joints, verts[:, TIP_VERTS]], 1)
        return verts, j16[:, OPENPOSE]
