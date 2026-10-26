"""The ladder's five-step bf16 NorMuon and width-corrected AdamW groups."""
from __future__ import annotations

import torch

from .moe_trainer import BatchedNorMuonAdamW


def orthogonalise(gradient):
    x = gradient.bfloat16()
    transpose = x.shape[-2] > x.shape[-1]
    if transpose:
        x = x.mT
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(5):
        a = x @ x.mT
        x = 3.4445 * x + (-4.775 * a + 2.0315 * (a @ a)) @ x
    return x.mT if transpose else x


class PyramidOptimizer(BatchedNorMuonAdamW):
    """Retain the named, pickle-free optimizer persistence interface."""

    @torch.no_grad()
    def step(self):
        lr = self.param_groups[0]["lr"]
        for p in self.muon_params:
            if p.grad is None:
                continue
            grad = p.grad.mT if p.ndim == 3 else p.grad
            view = p.mT if p.ndim == 3 else p
            momentum = self._bufs.get(p)
            if momentum is None:
                momentum = self._bufs[p] = torch.zeros_like(grad)
            momentum.lerp_(grad, 0.04)
            update = orthogonalise(grad * 0.04 + momentum * 0.96).to(grad.dtype)
            norm = update.norm(dim=(-2, -1), keepdim=True)
            variance = self._row_v.get(p)
            if variance is None:
                variance = self._row_v[p] = torch.zeros_like(update[..., :1])
            variance.lerp_(update.square().mean(-1, keepdim=True), 0.001)
            update = update / (variance.sqrt() + 1e-10)
            update = update * (norm / (update.norm(dim=(-2, -1), keepdim=True) + 1e-10))
            update = update * max(1, grad.shape[-2] / grad.shape[-1]) ** 0.5
            view.mul_(1 - lr * 2e-8).add_(update, alpha=-lr)
        if self.adamw is not None:
            self.adamw.step()


def build_optimizer(model, width):
    muon, groups = [], {"head": [], "router": [], "other": []}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "router" in name:
            groups["router"].append(p)
        elif name.startswith("head."):
            groups["head"].append(p)
        elif name.startswith("patch_embed.") or p.ndim != 2 and not name.endswith(("w_in", "w_out")):
            groups["other"].append(p)
        else:
            muon.append(p)
    adam = [{"params": params, "lr_scale": 512 / width if key == "head" else 1.0,
             "weight_decay": 0.0 if key == "router" else 0.05, "betas": (0.91, 0.972)}
            for key, params in groups.items() if params]
    opt = PyramidOptimizer(muon, adam, lr=0.002125, weight_decay=2e-8,
                           momentum=0.96, beta2=0.999)
    opt.param_groups[0]["base_lr"] = 0.002125 * (width / 512) ** 0.42
    if opt.adamw:
        for group in opt.adamw.param_groups:
            group["base_lr"] = 0.000425 * group["lr_scale"]
    return opt


def schedule_factor(step, total_steps=27500):
    """Exact reference endpoints; shortened local legs keep an explicit cooldown."""
    cooldown = max(1, round(total_steps / 11))
    start = total_steps - cooldown
    warmup = min(80, max(1, start))
    if step < warmup:
        return step / warmup
    if step >= start:
        return 0.0 if cooldown == 1 else max(0, 1 - (step - start) / (cooldown - 1))
    return 1.0
