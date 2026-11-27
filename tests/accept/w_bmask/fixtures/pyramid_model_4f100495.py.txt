"""Pyramid MoE, attention-only variate blocks and a unified quantile/copula head.

Implements the supplied 2026-10-26 pyramid_varattn_ladder specification.
Self-contained checkpoint code; historical moe_model.py remains byte-pinned.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

QUANTILE_LEVELS = (0.01, *(i / 20 for i in range(1, 20)), 0.99)
NUM_QUANTILES = len(QUANTILE_LEVELS)
MEDIAN_INDEX = QUANTILE_LEVELS.index(0.5)
COPULA_RANK = 1
HEAD_CHANNELS = NUM_QUANTILES + 1 + COPULA_RANK
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
    num_quantiles: int = NUM_QUANTILES
    context_length: int = 4096
    horizon: int = 64
    max_patches: int = 136  # decode window in patches (context + masked horizon)
    variate_blocks: tuple[int, ...] = (4, 9)
    # time blocks whose FFN is a sparse MoE (block 0 stays dense)
    moe_blocks: tuple[int, ...] = (1, 2, 3, 5, 6, 7, 8)
    dense_ffn: int = 1104
    num_experts: int = 512
    experts_per_layer: tuple[int, ...] = (128, 256, 256, 512, 512, 512, 512)
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
    router_z_loss_coef: float = 0.001

    def __post_init__(self):
        if self.num_quantiles != NUM_QUANTILES:
            raise ValueError(f"Cascade MoE Base requires {NUM_QUANTILES} quantiles")

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
        sizes = {"2m": (128, 276, 23), "7m": (256, 552, 46),
                 "28m": (512, 1104, 92), "120m": (1088, 2346, 196)}
        preset = str(getattr(c, "arch_preset", ""))
        key = preset.removeprefix("cascade-moe-v2-")
        if key not in sizes:
            raise ValueError(f"unknown pyramid size {preset!r}")
        d, ffn, hidden = sizes[key]
        expected = dict(d_model=d, num_layers=10, num_heads=d // 64,
                        head_dim=64, patch_size=32, num_quantiles=NUM_QUANTILES,
                        context_length=4096, d_ff=ffn, cpm_c_max=16, cpm_p_max=0.4)
        for name, value in expected.items():
            if getattr(c, name) != value:
                raise ValueError(f"{preset}: {name} must equal {value}")
        return cls(d_model=d, num_heads=d // 64, dense_ffn=ffn,
                   expert_hidden=hidden, shared_hidden=4 * hidden,
                   horizon=int(c.horizon))

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
    binary = (((x64.abs() < 1e-9) | ((x64 - 1).abs() < 1e-9)) | (k64 == 0)).all(-1, keepdim=True)
    loc = torch.where(binary, torch.zeros_like(loc), loc)
    scale = torch.where(binary, torch.ones_like(scale), scale)
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
            nn.Conv1d(2 if i == 0 else cfg.conv_channels, cfg.conv_channels,
                      cfg.conv_kernel, dilation=d)
            for i, d in enumerate(self.dilations)
        )
        self.proj = nn.Linear(cfg.conv_channels, cfg.d_model)
        self.patch_size = cfg.patch_size

    def forward(self, x2: torch.Tensor) -> torch.Tensor:
        n, p, _ = x2.shape
        ps = self.patch_size
        # x2 carries [values, missing]; conv branch carries [values, observed].
        h = torch.stack([x2[..., :ps], 1 - x2[..., ps:]], dim=1).reshape(n, 2, p * ps)
        for conv, dilation in zip(self.convs, self.dilations, strict=True):
            h = F.gelu(conv(F.pad(h, ((self.k - 1) * dilation, 0))))
        return self.proj(h.reshape(n, h.shape[1], p, ps).mean(-1).transpose(1, 2))


class _GatedAttention(nn.Module):
    def __init__(self, cfg: CascadeMoEConfig, axis: str):
        super().__init__()
        self.cfg = cfg
        self.axis = axis
        inner = cfg.num_heads * cfg.head_dim
        self.qkv = nn.Linear(cfg.d_model, 3 * inner, bias=True)
        self.gate = nn.Linear(cfg.d_model, inner, bias=True) if axis == "time" else None
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
        if self.gate is not None:
            a = a * (2 * torch.sigmoid(self.gate(h)))
        return self.proj(a)


class _DenseFFN(nn.Module):
    def __init__(self, d: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden, bias=False), nn.GELU(),
                                 nn.Linear(hidden, d, bias=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MoEFFN(nn.Module):
    """Sparse MoE FFN with a shared expert, sqrt-softplus router and loss-free
    balancing bias. Expert weights are stacked 3-D parameters
    ``w_in (E, d, h)`` / ``w_out (E, h, d)`` — NorMuon treats dim 0 as a batch
    of independent matrices (one per expert)."""

    def __init__(self, cfg: CascadeMoEConfig, num_experts: int):
        super().__init__()
        self.cfg = cfg
        self.num_experts = num_experts
        E, d, h = num_experts, cfg.d_model, cfg.expert_hidden
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
        s = torch.sqrt(F.softplus(logits).clamp_min(1e-30))
        # top-k on the biased scores (selection only)
        sel = s + self.route_bias
        grouped = sel.reshape(sel.shape[0], 8, self.num_experts // 8)
        groups = grouped.topk(2, dim=-1).values.sum(-1).topk(4, dim=-1).indices
        keep = torch.zeros_like(grouped[..., 0], dtype=torch.bool).scatter_(1, groups, True)
        sel = grouped.masked_fill(~keep[..., None], float("-inf")).reshape_as(sel)
        idx = torch.topk(sel, self.cfg.top_k, dim=-1, sorted=True).indices
        w = s.gather(-1, idx)
        w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-20)
        return idx, w, s, logits

    def expert_mm(self, x, weight, kind):
        return torch.bmm(x, weight)

    def expert_single(self, x, expert, kind):
        weight = self.w_in if kind == "w_in" else self.w_out
        return x @ weight[expert].to(x.dtype)

    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """``x``: ``(N, d)`` tokens laid out as consecutive sequences of
        ``seq_len`` tokens (for the sequence-wise balance loss)."""
        N, d = x.shape
        E, K = self.num_experts, self.cfg.top_k
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
        if cap * E > 2 * flat_e.numel():
            # A collapsed router must not allocate E copies of its busiest
            # expert's tokens. Each sorted assignment is still processed once.
            parts = []
            start = 0
            for expert, count in enumerate(counts.tolist()):
                if count:
                    inputs = x[tok[start:start + count]]
                    hidden = F.gelu(self.expert_single(inputs, expert, "w_in"))
                    parts.append(self.expert_single(hidden, expert, "w_out"))
                    start += count
            y_sorted = torch.cat(parts, 0)
        else:
            buf = x.new_zeros(E, max(cap, 1), d)
            buf[sorted_e, pos] = x[tok]
            hid = F.gelu(self.expert_mm(buf, self.w_in.to(buf.dtype), "w_in"))
            out = self.expert_mm(hid, self.w_out.to(buf.dtype), "w_out")
            y_sorted = out[sorted_e, pos]
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
        E = self.num_experts
        counts = torch.bincount(idx.reshape(-1), minlength=E).float()
        global_load = getattr(self, "balance_load_override", None)
        f = (counts / max(1, idx.numel()) if global_load is None
             else global_load.float() / global_load.sum().clamp_min(1))
        p = (s / s.sum(-1, keepdim=True).clamp_min(1e-20)).mean(0)
        bal = E * (f * p).sum()
        z = torch.logsumexp(logits, dim=-1).pow(2).mean()
        return self.cfg.balance_loss_coef * bal + self.cfg.router_z_loss_coef * z

    @torch.no_grad()
    def update_bias(self, load: torch.Tensor | None = None, *, rate_scale: float = 1.0) -> None:
        load = self.last_load if load is None else load
        if load is None:
            return
        load = load.float()
        mean = load.mean().clamp_min(1e-20)
        self.route_bias += self.cfg.bias_update_rate * rate_scale * ((mean - load) / mean).clamp(-1, 1)
        self.route_bias -= self.route_bias.mean()


class _Block(nn.Module):
    def __init__(self, cfg: CascadeMoEConfig, idx: int):
        super().__init__()
        self.cfg = cfg
        self.idx = idx
        self.axis = cfg.layer_axis(idx)
        self.norm1 = nn.LayerNorm(cfg.d_model, eps=1e-4, elementwise_affine=False)
        self.attn = _GatedAttention(cfg, self.axis)
        self.norm2 = (nn.LayerNorm(cfg.d_model, eps=1e-4, elementwise_affine=False)
                      if self.axis == "time" else None)
        self.moe = cfg.is_moe(idx)
        self.ffn = (MoEFFN(cfg, cfg.experts_per_layer[cfg.moe_blocks.index(idx)]) if self.moe
                    else _DenseFFN(cfg.d_model, cfg.dense_ffn) if self.axis == "time" else None)
        # u-μP residual weights — same scheme as toto2_model._Block.
        S = 128.0
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
        x = self.attn_b * x + self.attn_a * self.attn(h)
        if self.ffn is None:
            return x
        h = self.norm2(x)
        if self.moe:
            S, T, d = h.shape
            f = self.ffn(h.reshape(S * T, d), seq_len=T).view(S, T, d)
        else:
            f = self.ffn(h)
        return self.mlp_b * x + self.mlp_a * f


class CascadeMoEModel(nn.Module):
    """Q21 linear-standardized quantiles, or raw 23-channel head parameters."""

    def __init__(self, cfg: CascadeMoEConfig):
        super().__init__()
        self.cfg = cfg
        ps, d = cfg.patch_size, cfg.d_model
        self.patch_embed = nn.Linear(ps * 2, d)
        self.conv_embed = DilatedConvEmbed(cfg)
        self.embed_mlp = _ResidualMLP(d, cfg.dense_ffn)
        self.role_embed = nn.Embedding(3, d)
        self.blocks = nn.ModuleList(_Block(cfg, i) for i in range(cfg.num_layers))
        self.norm = nn.LayerNorm(d, eps=1e-4, elementwise_affine=False)
        self.out_mlp = _ResidualMLP(d, cfg.dense_ffn)
        self.head = nn.Linear(d, ps * HEAD_CHANNELS)
        # The conv branch and its projection retain PyTorch default initialization.
        for name, module in self.named_modules():
            if not name.startswith("conv_embed"):
                self._init_weights(module)
        with torch.no_grad():
            self.role_embed.weight.zero_()
            for block in self.blocks:
                if block.attn.gate is not None:
                    block.attn.gate.weight.zero_()
                    block.attn.gate.bias.zero_()
                if block.moe:
                    nn.init.normal_(block.ffn.router.weight, std=0.02)
            self.head.weight.normal_(0, 0.001)
            bias = self.head.bias.reshape(ps, HEAD_CHANNELS)
            bias.zero_()
            spacing = 0.5 * torch.special.ndtri(bias.new_tensor(QUANTILE_LEVELS)).diff()
            bias[:, 1:NUM_QUANTILES] = torch.log(torch.expm1(spacing - 1e-4))
            bias[:, NUM_QUANTILES] = math.log(math.expm1(1 - 1e-4))
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
                roles: torch.Tensor | None = None, *, with_parameters: bool = False):
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
        x = x + self.role_embed(roles)[:, :, None, :]
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
        self.aux_loss = aux / len(self.moe_layers()) if self.training else None
        x = self.out_mlp(self.norm(x))
        # the heads' shared input — the trainer measures each loss's gradient
        # norm here to cap the copula term (training only; never at inference)
        self.last_hidden = x if self.training else None
        with torch.autocast(device_type=x.device.type, enabled=False):
            params = self.head(x.float()).reshape(B, C, P, ps, HEAD_CHANNELS)
            out = params if with_parameters else quantiles(params)
        return out[:, 0] if squeeze else out

    @torch.no_grad()
    def update_routing_bias(self) -> None:
        """Update selection biases after a full-model step; expert legs freeze them."""
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
        inactive += (layer.num_experts - cfg.top_k) * per_expert
    return {"total": total, "active": total - inactive}


def quantiles(params):
    """Monotone Q21 in linear standardized units (no sorting or clipping)."""
    if params.shape[-1] != HEAD_CHANNELS:
        raise ValueError(f"expected {HEAD_CHANNELS} unified-head channels")
    gaps = F.softplus(params[..., 1:NUM_QUANTILES]) + 1e-4
    prefix = torch.cat([torch.zeros_like(gaps[..., :1]), gaps.cumsum(-1)], -1)
    return params[..., :1] + prefix - prefix[..., MEDIAN_INDEX:MEDIAN_INDEX + 1]


def _dct_project(x):
    n = x.shape[-1]
    t = torch.arange(n, device=x.device, dtype=x.dtype)
    k = torch.arange(min(8, n), device=x.device, dtype=x.dtype)
    basis = torch.cos(math.pi / n * (t[:, None] + 0.5) * k[None]) * math.sqrt(2 / n)
    basis[:, 0] = 1 / math.sqrt(n)
    return (x @ basis) @ basis.T


def sample_copula_paths(params, n_samples, *, generator=None):
    """B,C,[P,]32,23 -> B,C,[P,]S,32; rank-one noise shared over channels."""
    q = quantiles(params)
    diag = F.softplus(_dct_project(params[..., NUM_QUANTILES])) + 1e-4
    diag = diag.square().mean(-1, keepdim=True).sqrt().expand_as(diag)
    factor = 0.1 * _dct_project(params[..., NUM_QUANTILES + 1])
    budget = factor.square().mean(-1, keepdim=True)
    direction = factor / factor.abs().clamp_min(1e-6)
    tiny = torch.finfo(params.dtype).tiny
    factor = direction * (budget / direction.square().mean(-1, keepdim=True).clamp_min(tiny)).clamp_min(tiny).sqrt()
    t = torch.arange(params.shape[-2], device=params.device, dtype=params.dtype)
    kernel = 0.99 * torch.exp(-0.5 * ((t[:, None] - t[None]) / 8).square())
    kernel = kernel + 0.01 * torch.eye(t.numel(), device=t.device, dtype=t.dtype)
    chol = torch.linalg.cholesky(kernel)
    shape = (*diag.shape[:-1], n_samples, diag.shape[-1])
    noise = torch.randn(shape, device=params.device, dtype=params.dtype, generator=generator) @ chol.T
    latent_shape = list(shape[:-1]) + [1]
    latent_shape[1] = 1
    latent = torch.randn(latent_shape, device=params.device, dtype=params.dtype, generator=generator)
    z = (diag[..., None, :] * noise + factor[..., None, :] * latent)
    z = z / (diag.square() + factor.square()).sqrt()[..., None, :]
    knots = torch.special.ndtri(q.new_tensor(QUANTILE_LEVELS))
    index = torch.bucketize(z.contiguous(), knots).sub(1).clamp(0, NUM_QUANTILES - 2)
    expanded = q.unsqueeze(-3).expand(*z.shape, NUM_QUANTILES)
    lo = expanded.gather(-1, index[..., None]).squeeze(-1)
    hi = expanded.gather(-1, (index + 1)[..., None]).squeeze(-1)
    return lo + (hi - lo) * ((z - knots[index]) / (knots[index + 1] - knots[index]))


def energy_score(samples, target):
    """Fair energy score per 32-point patch; samples [...,S,32]."""
    count = samples.shape[-2]
    first = (samples - target.unsqueeze(-2)).norm(dim=-1).mean(-1)
    pairs = torch.triu_indices(count, count, 1, device=samples.device)
    second = (samples[..., pairs[0], :] - samples[..., pairs[1], :]).norm(dim=-1).mean(-1)
    return (first - 0.5 * second) / math.sqrt(target.shape[-1])


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
                acc[i, :layer.num_experts] += layer.last_load.cpu().to(torch.int64)
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
        keep = torch.zeros(layer.num_experts, dtype=torch.bool)
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
