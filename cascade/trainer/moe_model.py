"""Cascade-MOE-28M — the sparse mixture-of-experts backbone of the v2 mechanism
(DEC-CA-0050). 27.5M active / ~191M total parameters at the pinned shape.

Like ``toto2_model.py`` this module is **self-contained torch**, copied into
every checkpoint (as ``model.py``), and its bytes are folded into
``base_arch_digest`` for ``base_arch = "cascade_moe"``. It is a NEW file, not
an edit of ``toto2_model.py``: that file's bytes pin the live toto2 king's
digest, so touching it would fork every validator.

Recipe (the owner's Cascade-MOE-28M spec; Toto2 conventions kept where the
spec is silent — weightless pre-norms, PerDimScale, xPos on time blocks, u-μP
a/b residuals, the robust causal scaler, residual SiLU patch projections):

* **Backbone** — width 512, 10 blocks, each attending along one axis: time
  blocks 0-3, 5-8 (causal, full window); variate blocks 4 and 9 (across
  channels at one time position, plus a channel-ROLE embedding: target vs
  covariate). Gated attention (G1, Qiu et al. 2025): each head's output is
  multiplied elementwise by ``sigmoid(W_g · h_query)`` before the output
  projection.
* **Patch embedding** — ``[values × (1 − mask), mask]`` per patch, plus a
  parallel causal dilated-conv branch over the raw point sequence (128
  channels, kernel 7, dilations 1 and 4, mean-pooled per patch) so the
  embedding sees local shape across patch borders.
* **MoE FFN** on time blocks 1,2,3,5,6,7,8 (block 0 and the variate blocks stay
  dense, hidden 1104): 256 routed experts 512→92→512 (bias-free GELU), top-8
  per token, one always-on shared expert 512→368→512. Active width 8·92+368 =
  1104 = the dense FFN. Router: bias-free FP32 linear; DeepSeek-V3 sigmoid
  gate — select top-k on ``s + b``, mix with the selected UNBIASED ``s``
  renormalised to 1. ``b`` is a non-trained, checkpointed buffer updated after
  each step by ``b += u·sign(mean_load − load)`` (:meth:`CascadeMoEModel.
  update_routing_bias`). Complementary sequence-wise balance loss + router
  z-loss are returned via :attr:`CascadeMoEModel.aux_loss`. No capacity
  limit, no dropping; dispatch is a stable sort by expert into one padded
  batched matmul and a gather-based combine — no atomics, so it is
  bit-deterministic on a fixed device.
* **Heads** — 9 quantiles per future step (the eval grid), and in FP32 a
  rank-4 Gaussian-copula head: per step ``COPULA_RANK`` loadings + a log
  diagonal, so joint paths across channels and steps can be drawn
  (:func:`sample_copula_paths`).

Domain-expert competition hooks (DEC-CA-0050 lane B) live at the bottom:
:func:`domain_expert_affinity`, :func:`routing_topk`,
:func:`select_domain_experts`, :func:`expert_grad_mask_hooks`, :func:`merge_expert_slices`,
:func:`rebase_expert_delta`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

QUANTILE_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
COPULA_RANK = 4
Z_CLAMP = 64.0
# Channel roles for the variate blocks' role embedding.
ROLE_TARGET = 0
ROLE_COVARIATE = 1


@dataclass
class CascadeMoEConfig:
    d_model: int = 512
    num_layers: int = 10
    num_heads: int = 8
    head_dim: int = 64
    patch_size: int = 32
    num_quantiles: int = 9
    context_length: int = 4096
    horizon: int = 64
    max_patches: int = 136  # decode window in patches (context + masked horizon)
    variate_blocks: tuple[int, ...] = (4, 9)
    # time blocks whose FFN is a sparse MoE (block 0 stays dense)
    moe_blocks: tuple[int, ...] = (1, 2, 3, 5, 6, 7, 8)
    dense_ffn: int = 1104
    num_experts: int = 256
    top_k: int = 8
    expert_hidden: int = 92
    shared_hidden: int = 368
    conv_channels: int = 128
    conv_kernel: int = 7
    conv_dilations: tuple[int, ...] = (1, 4)
    copula_rank: int = COPULA_RANK
    residual_mult: float = 0.75
    cpm_c_max: int = 16
    cpm_p_max: float = 0.4
    # loss-free balancing + auxiliaries (DeepSeek-V3)
    bias_update_rate: float = 0.003
    balance_loss_coef: float = 1e-4
    router_z_loss_coef: float = 0.01

    def layer_axis(self, i: int) -> str:
        return "variate" if i in self.variate_blocks else "time"

    def is_moe(self, i: int) -> bool:
        return i in self.moe_blocks

    def to_dict(self) -> dict:
        out = {}
        for k in self.__dataclass_fields__:
            v = getattr(self, k)
            out[k] = list(v) if isinstance(v, tuple) else v
        return out

    @classmethod
    def from_contract(cls, c: object) -> CascadeMoEConfig:
        """Build from a cascade ``TrainingContractConfig`` (duck-typed). The
        contract pins the shared shape integers; the expert geometry (count,
        top-k, widths, block layout) is pinned by THIS file's bytes, which
        ``base_arch_digest`` folds in."""
        ctx = int(getattr(c, "context_length", 4096))
        hz = int(getattr(c, "horizon", 64))
        ps = int(getattr(c, "patch_size", 32))
        d_ff = int(getattr(c, "d_ff", 0) or 0)
        return cls(
            d_model=int(getattr(c, "d_model", 512)),
            num_layers=int(getattr(c, "num_layers", 10)),
            num_heads=int(getattr(c, "num_heads", 8)),
            head_dim=int(getattr(c, "head_dim", 64)),
            patch_size=ps,
            num_quantiles=int(getattr(c, "num_quantiles", 9)),
            context_length=ctx,
            horizon=hz,
            max_patches=max(8, (ctx + hz) // ps + 4),
            dense_ffn=d_ff if d_ff > 0 else cls.dense_ffn,
            cpm_c_max=int(getattr(c, "cpm_c_max", 16)),
            cpm_p_max=float(getattr(c, "cpm_p_max", 0.4)),
        )

    @classmethod
    def from_dict(cls, d: dict) -> CascadeMoEConfig:
        kw = {k: (tuple(v) if isinstance(v, list) else v)
              for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**kw)


# ── robust causal scaler (identical maths to toto2_model) ────────────────────


def causal_standardize(
    x: torch.Tensor, mask: torch.Tensor | None = None, *, min_obs: int = 8, eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-step causal location/scale over OBSERVED points only, under asinh.
    ``x``/``mask`` ``(B, L)``, mask 1 = unobserved. See toto2_model for the
    float64 shifted-moment rationale."""
    B, L = x.shape
    keep = torch.ones_like(x) if mask is None else 1.0 - mask.to(x.dtype)
    x64 = x.double()
    k64 = keep.double()
    ref = x64.gather(-1, (k64 > 0).to(torch.int64).argmax(dim=-1, keepdim=True))
    xk = (x64 - ref) * k64
    n = k64.cumsum(dim=-1)
    cnt = n.clamp_min(1.0)
    loc = xk.cumsum(dim=-1) / cnt
    var = (xk * xk).cumsum(dim=-1) / cnt - loc * loc
    loc = loc + ref
    scale = var.clamp_min(0.0).sqrt().clamp_min(eps)
    ok = n >= float(min_obs)
    has = ok.any(dim=-1)
    first = torch.where(
        has, ok.to(torch.int64).argmax(dim=-1), torch.full((B,), L - 1, device=x.device)
    )[:, None]
    loc = torch.where(ok, loc, loc.gather(-1, first))
    scale = torch.where(ok, scale, scale.gather(-1, first))
    z = torch.asinh((x64 - loc) / scale).clamp_(-Z_CLAMP, Z_CLAMP)
    return z.to(x.dtype), loc.to(x.dtype), scale.to(x.dtype)


# ── building blocks ───────────────────────────────────────────────────────────


class _ResidualMLP(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden, bias=False), nn.SiLU(), nn.Linear(hidden, dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


def _xpos(q, k, inv_freq, zeta, scale_base: float = 512.0):
    T = q.shape[-2]
    t = torch.arange(T, device=q.device, dtype=inv_freq.dtype)
    freqs = torch.outer(t, inv_freq)
    cos = freqs.cos().repeat_interleave(2, dim=-1)
    sin = freqs.sin().repeat_interleave(2, dim=-1)
    power = ((t - T // 2) / scale_base)[:, None]
    scale = (zeta[None, :] ** power).repeat_interleave(2, dim=-1)

    def rotate(x):
        return torch.stack((-x[..., 1::2], x[..., 0::2]), dim=-1).flatten(-2)

    return (q * cos + rotate(q) * sin) * scale, (k * cos + rotate(k) * sin) / scale


class DilatedConvEmbed(nn.Module):
    """Parallel causal dilated convolutions over the point sequence
    ``[values × (1 − mask), mask]``, mean-pooled to one vector per patch.
    Causal left padding: patch p's features see points ≤ its own last step,
    never a later patch."""

    def __init__(self, cfg: CascadeMoEConfig):
        super().__init__()
        self.k = cfg.conv_kernel
        self.dilations = tuple(cfg.conv_dilations)
        self.convs = nn.ModuleList(
            nn.Conv1d(2, cfg.conv_channels, cfg.conv_kernel, dilation=d, bias=False)
            for d in self.dilations
        )
        self.proj = nn.Linear(cfg.conv_channels, cfg.d_model, bias=False)
        self.patch_size = cfg.patch_size

    def forward(self, x2: torch.Tensor) -> torch.Tensor:
        """``x2``: ``(N, P, 2·ps)`` = [values ‖ mask] per patch → ``(N, P, d)``."""
        N, P, _ = x2.shape
        ps = self.patch_size
        seq = torch.stack([x2[..., :ps], x2[..., ps:]], dim=1).reshape(N, 2, P * ps)
        h = 0.0
        for conv, d in zip(self.convs, self.dilations, strict=True):
            h = h + conv(F.pad(seq, ((self.k - 1) * d, 0)))
        h = F.gelu(h)                                         # (N, ch, P·ps)
        h = h.view(N, h.shape[1], P, ps).mean(dim=-1)         # (N, ch, P)
        return self.proj(h.transpose(1, 2))


class _GatedAttention(nn.Module):
    def __init__(self, cfg: CascadeMoEConfig, axis: str):
        super().__init__()
        self.cfg = cfg
        self.axis = axis
        inner = cfg.num_heads * cfg.head_dim
        self.qkv = nn.Linear(cfg.d_model, 3 * inner, bias=True)
        self.gate = nn.Linear(cfg.d_model, inner, bias=True)   # G1: from the query token
        self.proj = nn.Linear(inner, cfg.d_model, bias=True)
        self.per_dim_scale = nn.Parameter(torch.zeros(cfg.head_dim))
        if axis == "time":
            half = cfg.head_dim // 2
            idx = torch.arange(half).float() / max(1, half)
            self.register_buffer("inv_freq", 1.0 / (10000.0**idx), persistent=False)
            self.register_buffer("zeta", (idx + 0.4) / 1.4, persistent=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        B, T, _ = h.shape
        H, hd = self.cfg.num_heads, self.cfg.head_dim
        q, k, v = self.qkv(h).view(B, T, 3, H, hd).unbind(dim=2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        if self.axis == "time":
            q, k = _xpos(q, k, self.inv_freq, self.zeta)
        q = q * (F.softplus(self.per_dim_scale) / math.log(2.0))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=(self.axis == "time"), scale=1.0 / hd)
        a = a.transpose(1, 2).reshape(B, T, H * hd)
        return self.proj(a * torch.sigmoid(self.gate(h)))


class _DenseFFN(nn.Module):
    def __init__(self, d: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden, bias=False), nn.GELU(),
                                 nn.Linear(hidden, d, bias=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MoEFFN(nn.Module):
    """Sparse MoE FFN with a shared expert, sigmoid router and loss-free
    balancing bias. Expert weights are stacked 3-D parameters
    ``w_in (E, d, h)`` / ``w_out (E, h, d)`` — NorMuon treats dim 0 as a batch
    of independent matrices (one per expert)."""

    def __init__(self, cfg: CascadeMoEConfig):
        super().__init__()
        self.cfg = cfg
        E, d, h = cfg.num_experts, cfg.d_model, cfg.expert_hidden
        self.router = nn.Linear(d, E, bias=False)
        self.w_in = nn.Parameter(torch.empty(E, d, h))
        self.w_out = nn.Parameter(torch.empty(E, h, d))
        nn.init.normal_(self.w_in, std=1.0 / math.sqrt(d))
        nn.init.normal_(self.w_out, std=1.0 / math.sqrt(h))
        self.shared = _DenseFFN(d, cfg.shared_hidden)
        # Selection-only bias: a buffer, never a parameter; checkpointed.
        self.register_buffer("route_bias", torch.zeros(E, dtype=torch.float32))
        # Per-step observations (detached) — read by the trainer/observer.
        self.last_load: torch.Tensor | None = None
        self.last_aux: torch.Tensor | None = None
        self.last_topk: torch.Tensor | None = None

    def route(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """``x (N, d)`` → ``(topk_idx (N,k), topk_w (N,k), scores (N,E), logits (N,E))``,
        all FP32."""
        with torch.autocast(device_type=x.device.type, enabled=False):
            logits = F.linear(x.float(), self.router.weight.float())
        s = torch.sigmoid(logits)
        # top-k on the biased scores (selection only)
        sel = s + self.route_bias
        idx = torch.topk(sel, self.cfg.top_k, dim=-1, sorted=True).indices
        w = s.gather(-1, idx)
        w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-20)
        return idx, w, s, logits

    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """``x``: ``(N, d)`` tokens laid out as consecutive sequences of
        ``seq_len`` tokens (for the sequence-wise balance loss)."""
        N, d = x.shape
        E, K = self.cfg.num_experts, self.cfg.top_k
        idx, w, s, logits = self.route(x)
        flat_e = idx.reshape(-1)                                  # (N·K,)
        order = torch.argsort(flat_e, stable=True)
        counts = torch.bincount(flat_e, minlength=E)              # (E,)
        cap = int(counts.max().item()) if N else 0
        # position of each sorted assignment inside its expert's slot
        starts = torch.cumsum(counts, 0) - counts
        sorted_e = flat_e[order]
        pos = torch.arange(order.numel(), device=x.device) - starts[sorted_e]
        tok = order // K
        buf = x.new_zeros(E, max(cap, 1), d)
        buf[sorted_e, pos] = x[tok]
        hid = F.gelu(torch.bmm(buf, self.w_in.to(buf.dtype)))
        out = torch.bmm(hid, self.w_out.to(buf.dtype))           # (E, cap, d)
        y_sorted = out[sorted_e, pos]                             # (N·K, d)
        inv = torch.empty_like(order)
        inv[order] = torch.arange(order.numel(), device=x.device)
        y = y_sorted[inv].view(N, K, d)
        y = (y * w.to(y.dtype).unsqueeze(-1)).sum(dim=1)
        y = y + self.shared(x)

        self.last_load = counts.detach()
        self.last_topk = idx.detach()
        if self.training:
            self.last_aux = self._aux_loss(idx, s, logits, seq_len)
        else:
            self.last_aux = None
        return y

    def _aux_loss(self, idx, s, logits, seq_len: int) -> torch.Tensor:
        E, K = self.cfg.num_experts, self.cfg.top_k
        N = idx.shape[0]
        nseq = max(1, N // max(1, seq_len))
        T = N // nseq
        # complementary sequence-wise balance loss (DeepSeek-V3 eq. 17-20)
        onehot = torch.zeros(N, E, device=s.device, dtype=s.dtype)
        onehot.scatter_(1, idx, 1.0)
        f = onehot[: nseq * T].view(nseq, T, E).sum(1) * (E / (K * T))
        p = (s / s.sum(-1, keepdim=True).clamp_min(1e-20))[: nseq * T].view(nseq, T, E).mean(1)
        bal = (f * p).sum(-1).mean()
        z = torch.logsumexp(logits, dim=-1).pow(2).mean()
        return self.cfg.balance_loss_coef * bal + self.cfg.router_z_loss_coef * z

    @torch.no_grad()
    def update_bias(self, load: torch.Tensor | None = None) -> None:
        load = self.last_load if load is None else load
        if load is None:
            return
        load = load.float()
        self.route_bias += self.cfg.bias_update_rate * torch.sign(load.mean() - load)


class _Block(nn.Module):
    def __init__(self, cfg: CascadeMoEConfig, idx: int):
        super().__init__()
        self.cfg = cfg
        self.idx = idx
        self.axis = cfg.layer_axis(idx)
        self.norm1 = nn.LayerNorm(cfg.d_model, eps=1e-4, elementwise_affine=False)
        self.attn = _GatedAttention(cfg, self.axis)
        self.norm2 = nn.LayerNorm(cfg.d_model, eps=1e-4, elementwise_affine=False)
        self.moe = cfg.is_moe(idx)
        self.ffn = MoEFFN(cfg) if self.moe else _DenseFFN(cfg.d_model, cfg.dense_ffn)
        if self.axis == "variate":
            self.role_embed = nn.Embedding(2, cfg.d_model)
        # u-μP residual weights — same scheme as toto2_model._Block.
        S = max(2.0, cfg.context_length / cfg.patch_size)
        ratio2 = S / math.log(S)
        af2 = 2.0 * cfg.residual_mult**2 / (ratio2 + 1.0)
        aa2 = ratio2 * af2
        L = 2.0 * cfg.num_layers
        t_a = aa2 / (L / 2.0 + idx * aa2 + idx * af2)
        t_m = af2 / (L / 2.0 + (idx + 1) * aa2 + idx * af2)
        self.attn_a, self.attn_b = math.sqrt(t_a / (t_a + 1)), math.sqrt(1 / (t_a + 1))
        self.mlp_a, self.mlp_b = math.sqrt(t_m / (t_m + 1)), math.sqrt(1 / (t_m + 1))

    def forward(self, x: torch.Tensor, roles: torch.Tensor | None = None) -> torch.Tensor:
        """``x``: ``(S, T, d)`` — S sequences of T tokens along this block's axis.
        ``roles``: ``(S, T)`` channel roles (variate blocks only)."""
        h = self.norm1(x)
        if self.axis == "variate" and roles is not None:
            h = h + self.role_embed(roles)
        x = self.attn_b * x + self.attn_a * self.attn(h)
        h = self.norm2(x)
        if self.moe:
            S, T, d = h.shape
            f = self.ffn(h.reshape(S * T, d), seq_len=T).view(S, T, d)
        else:
            f = self.ffn(h)
        return self.mlp_b * x + self.mlp_a * f


class CascadeMoEModel(nn.Module):
    """``forward(patches, mask, roles)`` → ``quantiles``; with
    ``with_copula=True`` → ``(quantiles, copula)``. Quantiles-only by default
    so the checkpoint wrapper's CPM decode path is the toto2 one verbatim:

    * ``quantiles`` ``(B, [C,] P, ps, num_q)`` — next-patch per-step quantiles
      (same contract as Toto2Model, so the CPM trainer/wrapper path carries over);
    * ``copula`` ``(B, [C,] P, ps, rank + 1)`` FP32 — per-step low-rank factor
      loadings ``[..., :rank]`` and log diagonal ``[..., rank]`` of the
      Gaussian copula over (channel, step).
    """

    def __init__(self, cfg: CascadeMoEConfig):
        super().__init__()
        self.cfg = cfg
        ps, d = cfg.patch_size, cfg.d_model
        self.patch_embed = nn.Linear(ps * 2, d)
        self.conv_embed = DilatedConvEmbed(cfg)
        self.embed_mlp = _ResidualMLP(d, cfg.dense_ffn)
        self.blocks = nn.ModuleList(_Block(cfg, i) for i in range(cfg.num_layers))
        self.norm = nn.LayerNorm(d, eps=1e-4, elementwise_affine=False)
        self.out_mlp = _ResidualMLP(d, cfg.dense_ffn)
        self.head = nn.Linear(d, ps * cfg.num_quantiles)
        self.copula_head = nn.Linear(d, ps * (cfg.copula_rank + 1))
        self.apply(self._init_weights)
        self.aux_loss: torch.Tensor | None = None
        self.last_hidden: torch.Tensor | None = None

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=1.0 / math.sqrt(m.weight.shape[1]))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def moe_layers(self) -> list[MoEFFN]:
        return [b.ffn for b in self.blocks if b.moe]

    def forward(self, patches: torch.Tensor, mask: torch.Tensor | None = None,
                roles: torch.Tensor | None = None, *, with_copula: bool = False):
        squeeze = patches.dim() == 3
        if squeeze:
            patches = patches[:, None]
            if mask is not None:
                mask = mask[:, None]
        B, C, P, ps = patches.shape
        if mask is None:
            mask = torch.zeros_like(patches)
        else:
            if mask.dim() == 3:
                mask = mask[..., None].expand(B, C, P, ps)
            mask = mask.to(patches.dtype)
        x2 = torch.cat([patches * (1.0 - mask), mask], dim=-1)          # (B, C, P, 2ps)
        flat = x2.reshape(B * C, P, 2 * ps)
        x = self.embed_mlp(self.patch_embed(flat) + self.conv_embed(flat)).view(B, C, P, -1)
        if roles is None:
            roles = torch.zeros(B, C, dtype=torch.long, device=patches.device)
        aux = patches.new_zeros((), dtype=torch.float32)
        for blk in self.blocks:
            if blk.axis == "time":
                x = blk(x.reshape(B * C, P, -1)).view(B, C, P, -1)
            else:
                r = roles[:, None, :].expand(B, P, C).reshape(B * P, C)
                x = (blk(x.transpose(1, 2).reshape(B * P, C, -1), r)
                     .view(B, P, C, -1).transpose(1, 2))
            if blk.moe and blk.ffn.last_aux is not None:
                aux = aux + blk.ffn.last_aux
        self.aux_loss = aux if self.training else None
        x = self.out_mlp(self.norm(x))
        # the heads' shared input — the trainer measures each loss's gradient
        # norm here to cap the copula term (training only; never at inference)
        self.last_hidden = x if self.training else None
        q = self.head(x).view(B, C, P, ps, self.cfg.num_quantiles)
        if not with_copula:
            return q[:, 0] if squeeze else q
        with torch.autocast(device_type=x.device.type, enabled=False):
            cop = self.copula_head(x.float()).view(B, C, P, ps, self.cfg.copula_rank + 1)
        if squeeze:
            return q[:, 0], cop[:, 0]
        return q, cop

    @torch.no_grad()
    def update_routing_bias(self) -> None:
        """Loss-free balancing step — call once after every optimizer step
        (and NOT during the frozen-bias half of a cooldown, nor in a lane-B
        expert leg, where the king's bias is part of the frozen routing)."""
        for layer in self.moe_layers():
            layer.update_bias()

    def routing_health(self) -> dict:
        """Observer-only routing stats from the last forward (never mutates)."""
        out = {}
        for i, layer in enumerate(self.moe_layers()):
            if layer.last_load is None:
                continue
            load = layer.last_load.float()
            mean = load.mean().clamp_min(1e-9)
            out[f"moe{i}"] = {
                "max_over_mean": float(load.max() / mean),
                "dead": int((load == 0).sum()),
                "cv": float(load.std() / mean),
            }
        return out


# ── parameter accounting ─────────────────────────────────────────────────────


def param_counts(model: CascadeMoEModel) -> dict[str, int]:
    """``{"total", "active"}``: active counts top_k of num_experts routed
    experts per MoE layer (everything else is always on)."""
    total = sum(p.numel() for p in model.parameters())
    cfg = model.cfg
    inactive = 0
    for layer in model.moe_layers():
        per_expert = layer.w_in[0].numel() + layer.w_out[0].numel()
        inactive += (cfg.num_experts - cfg.top_k) * per_expert
    return {"total": total, "active": total - inactive}


# ── copula sampling ──────────────────────────────────────────────────────────


def sample_copula_paths(quantiles: torch.Tensor, copula: torch.Tensor, n_samples: int,
                        *, generator: torch.Generator | None = None,
                        levels: tuple[float, ...] = QUANTILE_LEVELS) -> torch.Tensor:
    """Draw joint paths from marginal quantiles + a low-rank Gaussian copula.

    ``quantiles`` ``(N, D, num_q)`` and ``copula`` ``(N, D, rank+1)`` for N
    independent groups of D jointly-modelled points (flatten channels × steps
    into D). Latent ``z = V·ε + exp(½·logd)·η``, standardised per point, mapped
    to ``u = Φ(z)`` and through the piecewise-linear quantile function (flat
    extrapolation beyond the outer levels). Returns ``(n_samples, N, D)``.
    """
    q = quantiles.float()
    cop = copula.float()
    N, D, nq = q.shape
    r = cop.shape[-1] - 1
    V, logd = cop[..., :r], cop[..., r]
    eps = torch.randn(n_samples, N, r, device=q.device, generator=generator)
    eta = torch.randn(n_samples, N, D, device=q.device, generator=generator)
    z = torch.einsum("ndr,snr->snd", V, eps) + torch.exp(0.5 * logd) * eta
    sd = torch.sqrt((V * V).sum(-1) + torch.exp(logd)).clamp_min(1e-6)
    u = torch.special.ndtr(z / sd)
    lv = torch.tensor(levels, device=q.device)
    qs, _ = torch.sort(q, dim=-1)                       # enforce monotone marginals
    uc = u.clamp(lv[0], lv[-1]).contiguous()
    pos = torch.bucketize(uc, lv).clamp(1, nq - 1)
    lo, hi = lv[pos - 1], lv[pos]
    qs_b = qs.unsqueeze(0).expand(n_samples, N, D, nq)
    qlo = qs_b.gather(-1, (pos - 1).unsqueeze(-1)).squeeze(-1)
    qhi = qs_b.gather(-1, pos.unsqueeze(-1)).squeeze(-1)
    t = (uc - lo) / (hi - lo)
    return qlo + t * (qhi - qlo)


def energy_score(samples: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Energy score ``E‖X−y‖ − ½E‖X−X'‖`` per group, averaged. ``samples``
    ``(S, N, D)``, ``target`` ``(N, D)``. Uses the unbiased pairwise estimator."""
    S = samples.shape[0]
    t1 = (samples - target.unsqueeze(0)).norm(dim=-1).mean(0)
    diff = (samples.unsqueeze(0) - samples.unsqueeze(1)).norm(dim=-1)   # (S, S, N)
    t2 = diff.sum((0, 1)) / max(1, S * (S - 1))
    return (t1 - 0.5 * t2).mean()


# ── lane B: domain-expert hooks (DEC-CA-0050) ────────────────────────────────


@torch.no_grad()
def domain_expert_affinity(model: CascadeMoEModel, batches) -> torch.Tensor:
    """Routing-assignment counts ``(n_moe_layers, num_experts)`` (int64) of the
    model over ``batches`` (an iterable of ``(patches, mask)``). Run on the
    KING over a domain's affinity slice to decide which experts lane B may
    train for that domain. Eval mode, no bias update."""
    was = model.training
    model.eval()
    layers = model.moe_layers()
    acc = torch.zeros(len(layers), model.cfg.num_experts, dtype=torch.int64)
    try:
        for patches, mask in batches:
            model(patches, mask)
            for i, layer in enumerate(layers):
                acc[i] += layer.last_load.cpu().to(torch.int64)
    finally:
        model.train(was)
    return acc


@torch.no_grad()
def routing_topk(model: CascadeMoEModel, batches) -> list[torch.Tensor]:
    """The experts each token was actually routed to: per MoE layer a
    ``(N, top_k)`` int64 tensor (CPU) over every token of ``batches`` (an
    iterable of ``(patches, mask)``), in batch order. It is the forward's own
    :attr:`MoEFFN.last_topk` — top-k on the router scores PLUS the routing
    bias, i.e. the selection the expert dispatch used — so the routing maths
    is never duplicated. Run two kings over the SAME batches to compare their
    routing token by token (OLMoE router saturation). Eval mode, no bias
    update. ``bincount`` of a layer's rows equals that layer's row of
    :func:`domain_expert_affinity`."""
    was = model.training
    model.eval()
    layers = model.moe_layers()
    out: list[list[torch.Tensor]] = [[] for _ in layers]
    try:
        for patches, mask in batches:
            model(patches, mask)
            for i, layer in enumerate(layers):
                out[i].append(layer.last_topk.cpu())
    finally:
        model.train(was)
    k = model.cfg.top_k
    return [torch.cat(o) if o else torch.zeros((0, k), dtype=torch.int64) for o in out]


def select_domain_experts(affinity: torch.Tensor, *, p: float = 0.2, min_n: int = 8,
                          max_n: int = 38, exclude: dict[int, set[int]] | None = None,
                          ) -> dict[int, list[int]]:
    """ESFT-Token selection (Wang et al., EMNLP 2024, arXiv 2407.01906):
    per MoE layer, rank experts by their token-selection ratio on the domain's
    affinity slice and take them until the cumulative ratio reaches ``p``,
    clamped to ``[min_n, max_n]``. ESFT's ``p = 0.2`` trains 5-15% of experts
    per layer; defaults clamp to 8..38 of 256 (≈3-15%). Experts in
    ``exclude[layer]`` (owned by another domain's open leg) are skipped; ties
    go to the lower index. Deterministic. Returns ``{layer: sorted ids}``."""
    out: dict[int, list[int]] = {}
    E = affinity.shape[1]
    for li in range(affinity.shape[0]):
        row = affinity[li].to(torch.float64)
        total = float(row.sum())
        banned = (exclude or {}).get(li, set())
        ranked = sorted((e for e in range(E) if e not in banned),
                        key=lambda e: (-float(row[e]), e))
        picked, cum = [], 0.0
        for e in ranked:
            if len(picked) >= max_n or (len(picked) >= min_n and cum >= p):
                break
            picked.append(e)
            cum += float(row[e]) / total if total > 0 else 0.0
        out[li] = sorted(picked)
    return out


def expert_grad_mask_hooks(model: CascadeMoEModel, selection: dict[int, list[int]]) -> list:
    """Freeze everything except the selected experts' slices: every non-expert
    parameter gets ``requires_grad=False``, and the stacked expert tensors get
    a gradient hook zeroing the unselected rows. Returns the hook handles.

    The router weight and the routing bias stay frozen too (the king's routing
    decides which tokens reach the trained experts — a lane-B leg must not
    re-route other domains' traffic)."""
    for p in model.parameters():
        p.requires_grad_(False)
    handles = []
    for li, layer in enumerate(model.moe_layers()):
        keep = torch.zeros(model.cfg.num_experts, dtype=torch.bool)
        keep[list(selection.get(li, []))] = True
        if not keep.any():
            continue
        for p in (layer.w_in, layer.w_out):
            p.requires_grad_(True)
            m = keep.to(p.device).view(-1, 1, 1)
            handles.append(p.register_hook(lambda g, m=m: g * m.to(g.dtype)))
    return handles


class MergeError(ValueError):
    """A lane-B candidate changed something outside its declared selection."""


def merge_expert_slices(king: dict[str, torch.Tensor], candidate: dict[str, torch.Tensor],
                        selection: dict[int, list[int]], moe_param_names: list[tuple[str, str]],
                        ) -> dict[str, torch.Tensor]:
    """Return a copy of ``king`` with ``candidate``'s selected expert rows.

    ``moe_param_names[i]`` = (w_in name, w_out name) of MoE layer ``i`` in the
    state dict. REFUSES (``MergeError``) unless the candidate is byte-identical
    to the king everywhere outside the selected rows — the merge is exactly
    the trained delta, never a smuggled weight change. Disjoint selections from
    several domains therefore commute."""
    if king.keys() != candidate.keys():
        raise MergeError("state-dict keys differ")
    owned = {n for pair in moe_param_names for n in pair}
    for name, kt in king.items():
        ct = candidate[name]
        if kt.shape != ct.shape or kt.dtype != ct.dtype:
            raise MergeError(f"{name}: shape/dtype differs")
        if name not in owned and not torch.equal(kt, ct):
            raise MergeError(f"{name}: changed outside the expert selection")
    merged = {k: v.clone() for k, v in king.items()}
    for li, (n_in, n_out) in enumerate(moe_param_names):
        rows = torch.tensor(sorted(selection.get(li, [])), dtype=torch.long)
        for n in (n_in, n_out):
            outside = torch.ones(king[n].shape[0], dtype=torch.bool)
            outside[rows] = False
            if not torch.equal(king[n][outside], candidate[n][outside]):
                raise MergeError(f"{n}: unselected expert rows changed")
            if rows.numel():
                merged[n][rows] = candidate[n][rows]
    return merged


def rebase_expert_delta(king_now: dict[str, torch.Tensor], king_base: dict[str, torch.Tensor],
                        candidate: dict[str, torch.Tensor], selection: dict[int, list[int]],
                        moe_param_names: list[tuple[str, str]]) -> dict[str, torch.Tensor]:
    """Async merge: carry a lane-B leg trained against an OLDER king
    (``king_base``) onto the CURRENT king (``king_now``) as a diff vector on
    its selected rows — ``now + (candidate − base)`` — the fine-tuning-transfer
    / Param∆ construction (Lin et al., EMNLP 2025; Cao et al., ICLR 2025).
    Unlike row replacement it keeps whatever lane A learned on those rows since.

    The candidate is first byte-checked against ``king_base`` exactly as
    :func:`merge_expert_slices` does (it may only differ on its rows). When
    ``king_now == king_base`` the result equals the plain merge. The rebased
    model is NOT pre-approved: the verdict must be re-run on it vs ``king_now``.
    """
    merge_expert_slices(king_base, candidate, selection, moe_param_names)  # validation only
    if king_now.keys() != king_base.keys():
        raise MergeError("king_now / king_base keys differ (architecture changed)")
    out = {k: v.clone() for k, v in king_now.items()}
    for li, pair in enumerate(moe_param_names):
        rows = torch.tensor(sorted(selection.get(li, [])), dtype=torch.long)
        if not rows.numel():
            continue
        for n in pair:
            delta = candidate[n][rows].float() - king_base[n][rows].float()
            out[n][rows] = (king_now[n][rows].float() + delta).to(out[n].dtype)
    return out


def moe_param_names(model: CascadeMoEModel) -> list[tuple[str, str]]:
    """State-dict names of each MoE layer's (w_in, w_out), in layer order."""
    return [(f"blocks.{b.idx}.ffn.w_in", f"blocks.{b.idx}.ffn.w_out")
            for b in model.blocks if b.moe]
