# ============================================================================
# OPTIMIZER  ·  NorMuon (transformer matrices) + AdamW (embed/head/bias/norm),
# mirroring the Toto-2 split. No u-muP here, so LRs are plain-parametrization
# values calibrated by the LR sweep and shared across arms — widths are
# identical in every arm, so approximate transfer holds.
# WSD schedule: warmup -> stable -> linear decay tail. Note: the 15k
# rank-stability checkpoints are UNdecayed for every arm (paired comparison
# stays valid); only the 30k finals see the decay tail.
# ============================================================================
"""NorMuon + AdamW optimizer split and the WSD learning-rate schedule."""
from __future__ import annotations

import torch


def newton_schulz(G, steps=5, eps=1e-7):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float(); X = X / (X.norm() + eps)
    transposed = X.shape[0] > X.shape[1]
    if transposed: X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return X.T if transposed else X


class NorMuon(torch.optim.Optimizer):
    def __init__(self, params, lr=8e-4, momentum=0.95, beta2=0.999, eps=1e-8):
        super().__init__(list(params), dict(lr=lr, momentum=momentum, beta2=beta2, eps=eps))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None: continue
                st = self.state[p]
                if "buf" not in st:
                    st["buf"] = torch.zeros_like(p.grad)
                    st["v"] = torch.zeros(p.grad.shape[0], device=p.device)
                buf = st["buf"]; buf.mul_(g["momentum"]).add_(p.grad)
                O = newton_schulz(buf)
                st["v"].mul_(g["beta2"]).add_((O * O).mean(1), alpha=1 - g["beta2"])
                p.add_(O / (st["v"].sqrt()[:, None] + g["eps"]), alpha=-g["lr"])


def build_optimizers(model, cfg):
    mat, other = [], []
    for n, p in model.named_parameters():
        in_backbone = n.startswith("blocks") and p.ndim == 2
        (mat if in_backbone else other).append(p)
    return [NorMuon(mat, lr=cfg.normuon_lr),
            torch.optim.AdamW(other, lr=cfg.lr, weight_decay=cfg.weight_decay,
                              betas=(0.9, 0.98))]


def lr_scale(step, total, warmup_frac, decay_frac):
    w = max(1, int(total * warmup_frac)); d = max(1, int(total * decay_frac))
    if step < w:            return step / w
    if step > total - d:    return max(0.0, (total - step) / d)
    return 1.0


def set_lr(opts, base_lrs, scale):
    for opt, base in zip(opts, base_lrs):
        for g in opt.param_groups: g["lr"] = base * scale
