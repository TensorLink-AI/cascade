"""Compact lane-B training state, without changing the checkpoint architecture.

The king's full tensors stay frozen and retain their canonical state-dict
names. Only selected rows have trainable leaves, gradients and optimizer/EMA
state. Frozen experts still propagate input gradients to earlier selected
layers. This module is training-only; checkpoints ship the original model.py.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .moe_model import MoEFFN


class _ExpertMatmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, frozen, selected, rows):
        out = torch.bmm(x, frozen)
        xs = x.index_select(0, rows)
        # Use the leaves themselves, including between optimizer.step() and
        # sync(). No dense differentiable copy or dense weight gradient.
        out.index_copy_(0, rows, torch.bmm(xs, selected))
        ctx.save_for_backward(frozen, selected, rows, xs)
        return out

    @staticmethod
    def backward(ctx, grad):
        frozen, selected, rows, xs = ctx.saved_tensors
        gs = grad.index_select(0, rows)
        dx = None
        if ctx.needs_input_grad[0]:
            dx = torch.bmm(grad, frozen.mT)
            dx.index_copy_(0, rows, torch.bmm(gs, selected.mT))
        ds = torch.bmm(xs.mT, gs) if ctx.needs_input_grad[2] else None
        return dx, None, ds, None


class _SelectedMoEFFN(MoEFFN):
    def __init__(self, base, rows):
        # Reuse the canonical parameters/modules/buffers, with identical
        # names. The tuple intentionally keeps the compact leaves outside
        # the checkpoint's parameter registry (ExpertSlices owns them).
        nn.Module.__init__(self)
        self.cfg = base.cfg
        self.router, self.shared = base.router, base.shared
        self.w_in, self.w_out = base.w_in, base.w_out
        self.register_buffer("route_bias", base.route_bias)
        self.last_load, self.last_aux, self.last_topk = None, None, None
        self.training = base.training
        self.slices = (rows, nn.Parameter(base.w_in.index_select(0, rows)),
                       nn.Parameter(base.w_out.index_select(0, rows)))

    def forward(self, x, seq_len):
        # Same stable dispatch and combine as MoEFFN.forward; only the two
        # expert matmuls have a selected-row weight backward.
        N, d = x.shape
        E, K = self.cfg.num_experts, self.cfg.top_k
        idx, w, s, logits = self.route(x)
        flat_e = idx.reshape(-1)
        order = torch.argsort(flat_e, stable=True)
        counts = torch.bincount(flat_e, minlength=E)
        cap = int(counts.max().item()) if N else 0
        starts = torch.cumsum(counts, 0) - counts
        sorted_e = flat_e[order]
        pos = torch.arange(order.numel(), device=x.device) - starts[sorted_e]
        tok = order // K
        buf = x.new_zeros(E, max(cap, 1), d)
        buf[sorted_e, pos] = x[tok]
        rows, wi, wo = self.slices
        hid = F.gelu(_ExpertMatmul.apply(buf, self.w_in.to(buf.dtype), wi.to(buf.dtype), rows))
        out = _ExpertMatmul.apply(hid, self.w_out.to(buf.dtype), wo.to(buf.dtype), rows)
        y_sorted = out[sorted_e, pos]
        inv = torch.empty_like(order)
        inv[order] = torch.arange(order.numel(), device=x.device)
        y = y_sorted[inv].view(N, K, d)
        y = (y * w.to(y.dtype).unsqueeze(-1)).sum(dim=1) + self.shared(x)
        self.last_load, self.last_topk = counts.detach(), idx.detach()
        self.last_aux = self._aux_loss(idx, s, logits, seq_len) if self.training else None
        return y


class ExpertSlices:
    """Parameter source for the optimizer, clipping and power EMAs.

    Construct after loading and moving the king to its training device/dtype.
    Call sync after each optimizer step so canonical checkpoints contain the
    selected endpoint too. All other model parameters remain frozen.
    """

    def __init__(self, model, selection):
        self.weights, self.rows, self.leaves = {}, {}, {}
        for p in model.parameters():
            p.requires_grad_(False)
        layers = [(n, m) for n, m in model.named_modules() if isinstance(m, MoEFFN)]
        for li, (name, base) in enumerate(layers):
            chosen = selection.get(li, selection.get(str(li), []))
            if not chosen:
                continue
            chosen = sorted(int(r) for r in chosen)
            if len(set(chosen)) != len(chosen) or chosen[0] < 0 or chosen[-1] >= base.cfg.num_experts:
                raise ValueError(f"invalid expert rows for MoE layer {li}")
            rows = torch.tensor(chosen, device=base.w_in.device, dtype=torch.long)
            layer = _SelectedMoEFFN(base, rows)
            parent, _, attr = name.rpartition(".")
            setattr(model.get_submodule(parent), attr, layer)
            for attr, leaf in zip(("w_in", "w_out"), layer.slices[1:], strict=True):
                key = f"{name}.{attr}"
                self.weights[key] = getattr(layer, attr)
                self.rows[key], self.leaves[key] = rows, leaf
        if not self.leaves:
            raise ValueError("an expert leg must select at least one existing expert row")

    def named_parameters(self):
        return iter(self.leaves.items())

    def parameters(self):
        return iter(self.leaves.values())

    @torch.no_grad()
    def sync(self):
        for name, leaf in self.leaves.items():
            self.weights[name].index_copy_(0, self.rows[name], leaf)
