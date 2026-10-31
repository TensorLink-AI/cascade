"""Training-time kernels for the pyramid model, bound onto a model instance.

``pyramid_model.py`` is the architecture: its source bytes are folded into the
LOCKED ``base_arch_digest`` that validators check, and validators evaluate
checkpoints with it. It stays byte-identical. This module re-implements two of
its forwards as the SAME operators with faster kernels and binds them onto a
lane-A training model with :func:`accelerate`; parameters, buffers and the
checkpoint are untouched.

* :func:`moe_forward` — ``MoEFFN.forward`` with load-binned expert dispatch
  (the reference takes a per-expert Python loop whenever routing is skewed:
  every production call on 2026-10-31) and an optional deferred balance term.
* :func:`conv_embed_forward` — ``DilatedConvEmbed.forward`` with each causal
  dilated conv as one im2col matmul (cuDNN picks a ~10x slower kernel).
"""
from __future__ import annotations

import types

import numpy as np
import torch
import torch.nn.functional as F

from .pyramid_model import CascadeMoEModel, DilatedConvEmbed, MoEFFN


def accelerate(model: CascadeMoEModel) -> CascadeMoEModel:
    """Bind the fast forwards onto ``model``'s own MoE and conv-embed modules
    (exact class matches only: a lane-B compact layer keeps its forward)."""
    for module in model.modules():
        if type(module) is MoEFFN:
            module.forward = types.MethodType(moe_forward, module)
        elif type(module) is DilatedConvEmbed:
            module.forward = types.MethodType(conv_embed_forward, module)
    model.fast_training = True
    return model


# ── conv embed ────────────────────────────────────────────────────────────────


def causal_conv(h: torch.Tensor, conv: torch.nn.Conv1d, dilation: int) -> torch.Tensor:
    """``conv(F.pad(h, ((k - 1) * dilation, 0)))`` as one matmul: the k dilated
    taps stacked (im2col) against the flattened ``(O, C*k)`` weight."""
    k, length = conv.kernel_size[0], h.shape[-1]
    padded = F.pad(h, ((k - 1) * dilation, 0))
    cols = torch.stack([padded[..., j * dilation:j * dilation + length] for j in range(k)], 2)
    weight = conv.weight.reshape(conv.weight.shape[0], -1)
    if torch.is_autocast_enabled():
        dtype = torch.get_autocast_gpu_dtype() if h.is_cuda else torch.get_autocast_cpu_dtype()
        weight, cols = weight.to(dtype), cols.to(dtype)
    out = torch.matmul(weight, cols.flatten(1, 2).to(weight.dtype))
    return out + conv.bias.to(out.dtype)[:, None] if conv.bias is not None else out


def conv_embed_forward(self: DilatedConvEmbed, x2: torch.Tensor) -> torch.Tensor:
    n, p, _ = x2.shape
    ps = self.patch_size
    h = torch.stack([x2[..., :ps], 1 - x2[..., ps:]], dim=1).reshape(n, 2, p * ps)
    for conv, dilation in zip(self.convs, self.dilations, strict=True):
        h = F.gelu(causal_conv(h, conv, dilation))
    return self.proj(h.reshape(n, h.shape[1], p, ps).mean(-1).transpose(1, 2))


# ── MoE ───────────────────────────────────────────────────────────────────────


def binned_experts(self: MoEFFN, x, tok, starts, counts):
    """Every routed assignment through its expert, in expert-sorted order.

    Experts are ranked by load and cut into bins whose members carry at least
    half the bin's largest load; each bin is one padded ``bmm``. A bin pads
    under 2x its real rows (a collapsed router included), with one host sync
    and one index upload per layer, and one token / weight gather per layer
    (a per-bin weight gather would back-propagate into a full-size zero
    gradient for every bin; split backward is one cat). Each row's result
    depends only on its own expert's weights."""
    sizes = counts.tolist()                       # the layer's one host sync
    total = sum(sizes)
    if not total:
        return x.new_zeros(0, x.shape[1])
    ranked = sorted((e for e, n in enumerate(sizes) if n), key=lambda e: (-sizes[e], e))
    bins, current = [], []
    for e in ranked:
        if current and 2 * sizes[e] < sizes[current[0]]:
            bins.append(current)
            current = []
        current.append(e)
    bins.append(current)
    # host-side layout: slot = row of the bin's flattened (members x cap)
    # buffer, src = the assignment's index in expert-sorted order
    n_of = np.asarray(sizes, dtype=np.int64)
    first = np.cumsum(n_of) - n_of
    slot, src, ids, shape = [], [], [], []
    for members in bins:
        m = np.asarray(members, dtype=np.int64)
        n = n_of[m]
        within = np.arange(int(n.sum())) - np.repeat(np.cumsum(n) - n, n)
        slot.append(np.repeat(np.arange(len(m)) * int(n[0]), n) + within)
        src.append(np.repeat(first[m], n) + within)
        ids.append(m)
        shape.append((len(m), int(n[0]), int(n.sum())))
    index = torch.from_numpy(np.concatenate([*slot, *src, *ids])).to(x.device)
    slot_t, src_t, ids_t = index[:total], index[total:2 * total], index[2 * total:]
    x_bins = x[tok[src_t]].split([n for _, _, n in shape])
    w_in = self.w_in.index_select(0, ids_t).split([k for k, _, _ in shape])
    w_out = self.w_out.index_select(0, ids_t).split([k for k, _, _ in shape])
    parts, at = [], 0
    for (k, cap, n), xb, wi, wo in zip(shape, x_bins, w_in, w_out, strict=True):
        rows = slot_t[at:at + n]
        buf = xb.new_zeros(k * cap, xb.shape[1]).index_put((rows,), xb)
        hid = F.gelu(self.expert_mm(buf.view(k, cap, -1), wi.to(buf.dtype), "w_in"))
        out = self.expert_mm(hid, wo.to(buf.dtype), "w_out")
        parts.append(out.reshape(k * cap, -1)[rows])
        at += n
    out = torch.cat(parts)
    return out.new_zeros(total, out.shape[1]).index_put((src_t,), out)


def moe_forward(self: MoEFFN, x: torch.Tensor, seq_len: int) -> torch.Tensor:
    """``MoEFFN.forward`` with :func:`binned_experts`. With ``defer_balance``
    set, the balance term's differentiable inputs are kept for
    :func:`finish_aux` instead of being finished against a pre-pass count."""
    N, d = x.shape
    K = self.cfg.top_k
    idx, w, s, logits = self.route(x)
    flat_e = idx.reshape(-1)                                  # (N·K,)
    order = torch.argsort(flat_e, stable=True)
    counts = torch.bincount(flat_e, minlength=self.num_experts)
    starts = torch.cumsum(counts, 0) - counts
    tok = order // K
    y_sorted = binned_experts(self, x, tok, starts, counts)
    inv = torch.empty_like(order)
    inv[order] = torch.arange(order.numel(), device=x.device)
    y = y_sorted[inv].view(N, K, d)
    y = (y * w.to(y.dtype).unsqueeze(-1)).sum(dim=1)
    y = y + self.shared(x)

    self.last_load = counts.detach()
    self.last_topk = idx.detach()
    if self.training and getattr(self, "defer_balance", False):
        # same expressions as MoEFFN._aux_loss, minus the global load
        self.pending_balance = ((s / s.sum(-1, keepdim=True).clamp_min(1e-20)).mean(0),
                                torch.logsumexp(logits, dim=-1).pow(2).mean())
        self.last_aux = None
    elif self.training:
        self.last_aux = self._aux_loss(idx, s, logits, seq_len)
    else:
        self.last_aux = None
    return y


def finish_balance(layer: MoEFFN, pending, global_load: torch.Tensor) -> torch.Tensor:
    """``MoEFFN._aux_loss`` from a deferred forward's kept terms: the value
    that forward computes with ``balance_load_override = global_load``."""
    p, z = pending
    f = global_load.float() / global_load.sum().clamp_min(1)
    bal = layer.num_experts * (f * p).sum()
    return layer.cfg.balance_loss_coef * bal + layer.cfg.router_z_loss_coef * z


def finish_aux(model: CascadeMoEModel, pending, loads) -> torch.Tensor:
    """The forward's ``aux_loss`` for a deferred-balance forward, summed in
    block order from a float32 zero exactly as ``CascadeMoEModel.forward``."""
    layers = model.moe_layers()
    aux = loads[0].new_zeros((), dtype=torch.float32)
    for layer, kept, load in zip(layers, pending, loads, strict=True):
        aux = aux + finish_balance(layer, kept, load)
    return aux / len(layers)
