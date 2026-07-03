"""Reference Toto2-style backbone — a patch transformer trained with
contiguous patch masking (CPM) and a multi-quantile head, from random init.

This module is **self-contained torch** and is *copied into every checkpoint*
(as ``model.py``) so the validator's ``forecast_wrapper.py`` can rebuild the
exact architecture to load the weights. Keep it dependency-light (torch only)
and free of cascade imports for that reason.

It follows the Toto 2.0 recipe (arXiv:2605.20119):

* **CPM** — a per-entry binary mask channel; training masks contiguous spans,
  inference fills the horizon with mask patches and decodes it in **one
  forward pass** (no autoregressive sampling).
* **Grouped time/variate attention** — the last layer of each group of 4
  attends over variates (full), the rest over time (causal, rotary positions);
  this matches ``Datadog/Toto-2.0-4m``'s ``layer_group_size=4`` /
  ``num_variate_layers_per_group=1`` / ``variate_layer_first=false``. cascade
  currently trains and scores univariate (``OPEN_QUESTIONS.md`` §8), so the
  variate layers run at ``C = 1`` — present and trainable, dormant until
  multivariate corpora flip on.
* **Attention details** — PerDimScale (learned per-dimension query scaling)
  with ``1/d_k`` attention scaling, biases on attention projections but not
  MLPs, ``head_dim`` fixed at 64 across the family.
* **Robust causal scaler** — per-step causal location/scale (mask-aware, with
  leading-patch backfill) under an arcsinh transform; targets are anchored at
  each patch boundary so no future value leaks into its own scaling.
* **Residual SiLU patch projections** at both ends, and a 9-level
  pinball/quantile head whose levels are exactly cascade's eval objective.

Shape and detail integers are pinned to the released ``Datadog/Toto-2.0-4m``
``config.json``: ``d_model=256``, ``num_layers=4``, ``num_heads=4``,
``qk/v_dim=64``, ``patch_size=32``, ``d_ff=688``, ``attn_bias``/no
``mlp_bias``, ``per_dim_scale``, ``use_xpos`` (γ=0.4 decay on rotary),
``norm_eps=1e-4`` with weightless norms, layer grouping, and the u-μP residual
scheme (``residual_mult=0.75``, ``residual_attn_ratio=sqrt(S/log S)≈5.14``,
applied via the unit-scaled a/b residual weights of u-μP eq. 25–31). The
optimiser orthogonalizes with Polar Express (see ``toto2_trainer.py``).
Remaining known approximations vs the release: the exact FFN inner structure
(param count 3.3M vs 4.1M) and the full u-μP init/LR width-scaling rules
(we keep fan-in init and a uniform LR). Pin ``base_arch_digest`` to whatever
you launch with.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# The 9 quantile levels 0.1..0.9 — identical to cascade's eval grid so the
# train objective equals the score objective.
QUANTILE_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


@dataclass
class Toto2Config:
    d_model: int = 256
    num_layers: int = 4
    num_heads: int = 4
    head_dim: int = 64
    patch_size: int = 32
    mlp_expansion: int = 2
    d_ff: int = 0  # exact FFN hidden width (0 ⇒ d_model × mlp_expansion); 4m ships 688
    num_quantiles: int = 9
    context_length: int = 4096
    horizon: int = 64
    max_patches: int = 256  # decode window capacity in patches (context + masked horizon)
    # layer grouping (Toto-2.0 config.json: layer_group_size=4,
    # num_variate_layers_per_group=1, variate_layer_first=false) — the last
    # layer of each group of 4 attends over variates, the rest over time.
    layer_group_size: int = 4
    # CPM training-mask distribution (Toto 2.0 §2.1 sweep optima).
    cpm_c_max: int = 16
    cpm_p_max: float = 0.4
    # u-μP residual scale α_res (released config: residual_mult = 0.75; the
    # attention/FFN ratio is derived as sqrt(S/log S) from context/patch).
    residual_mult: float = 0.75

    @property
    def ffn_hidden(self) -> int:
        return self.d_ff if self.d_ff > 0 else self.d_model * self.mlp_expansion

    @classmethod
    def from_contract(cls, c: object) -> Toto2Config:
        """Build from a cascade ``TrainingContractConfig`` (duck-typed)."""
        ctx = int(getattr(c, "context_length", 4096))
        hz = int(getattr(c, "horizon", 64))
        ps = int(getattr(c, "patch_size", 32))
        return cls(
            d_model=int(getattr(c, "d_model", 256)),
            num_layers=int(getattr(c, "num_layers", 4)),
            num_heads=int(getattr(c, "num_heads", 4)),
            head_dim=int(getattr(c, "head_dim", 64)),
            patch_size=ps,
            mlp_expansion=int(getattr(c, "mlp_expansion", 2)),
            d_ff=int(getattr(c, "d_ff", 0)),
            num_quantiles=int(getattr(c, "num_quantiles", 9)),
            context_length=ctx,
            horizon=hz,
            max_patches=max(8, (ctx + hz) // ps + 4),
            cpm_c_max=int(getattr(c, "cpm_c_max", 16)),
            cpm_p_max=float(getattr(c, "cpm_p_max", 0.4)),
        )

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    def layer_axis(self, i: int) -> str:
        """Attention axis of layer ``i``: the last layer of each group of
        ``layer_group_size`` attends over variates, the rest over time."""
        g = max(1, self.layer_group_size)
        return "variate" if i % g == g - 1 else "time"


# ── robust causal scaler ──────────────────────────────────────────────────────


def causal_standardize(
    x: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    min_obs: int = 8,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Toto 2.0's robust causal scaler: per-step causal location/scale under an
    arcsinh transform.

    ``x`` is ``(B, L)``; ``mask`` is an optional binary ``(B, L)`` with 1 =
    unobserved — masked entries are excluded from the statistics, so the stats
    carry forward unchanged across masked spans (matching inference, where
    horizon mask patches contribute nothing). Steps whose causal window holds
    fewer than ``min_obs`` observations are backfilled with the first stable
    stats (the paper's leading-patch backfill). Returns ``(z, loc, scale)``
    where ``z = arcsinh((x - loc) / scale)``; all three are ``(B, L)``.
    """
    B, L = x.shape
    keep = torch.ones_like(x) if mask is None else 1.0 - mask.to(x.dtype)
    # The cumulative E[x²]−E[x]² form cancels catastrophically once
    # mean²/var exceeds the dtype's precision (~1e7 in float32 — routine for
    # counter/gauge-style series at large levels with small fluctuations),
    # collapsing scale to eps. Accumulate in float64 and shift each row to its
    # first observation so the moments stay small regardless of series level.
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
    z = torch.asinh((x64 - loc) / scale)
    return z.to(x.dtype), loc.to(x.dtype), scale.to(x.dtype)


def patch_anchors(loc: torch.Tensor, scale: torch.Tensor, patch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Causal stats at the last step of each patch — the scaling a forecast of
    the *next* patch is anchored to. ``(B, L)`` → ``(B, P)`` each."""
    B, L = loc.shape
    P = L // patch_size
    return (
        loc.view(B, P, patch_size)[:, :, -1],
        scale.view(B, P, patch_size)[:, :, -1],
    )


def invert_standardize(z: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`causal_standardize` at a fixed anchor:
    ``x = sinh(z) * scale + loc``."""
    return torch.sinh(z) * scale + loc


# ── building blocks ───────────────────────────────────────────────────────────


class _ResidualMLP(nn.Module):
    """Two-layer SiLU MLP with a residual connection — Toto 2.0's nonlinear
    patch projection, used at both ends of the transformer. Bias-free (biases
    live on attention projections, not MLPs)."""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden, bias=False),
            nn.SiLU(),
            nn.Linear(hidden, dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


def _xpos(
    q: torch.Tensor, k: torch.Tensor, inv_freq: torch.Tensor, zeta: torch.Tensor,
    scale_base: float = 512.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """xPos (arXiv 2212.10554, ``use_xpos`` in the Toto-2.0 release): rotary
    position embedding with per-dimension exponential decay
    ``ζ̂_i = (i/(d/2) + γ)/(1 + γ)``, γ = 0.4 — queries scaled by ``ζ̂^m`` and
    keys by ``ζ̂^{-m}`` over the sequence axis of ``(B, H, T, hd)``. Follows
    the official torchscale implementation: the exponent is centered and
    divided by ``scale_base`` (512) so ``ζ̂^{±m}`` stays representable."""
    T = q.shape[-2]
    t = torch.arange(T, device=q.device, dtype=inv_freq.dtype)
    freqs = torch.outer(t, inv_freq)                      # (T, hd/2)
    cos = freqs.cos().repeat_interleave(2, dim=-1)        # (T, hd)
    sin = freqs.sin().repeat_interleave(2, dim=-1)
    power = ((t - T // 2) / scale_base)[:, None]          # (T, 1)
    scale = (zeta[None, :] ** power).repeat_interleave(2, dim=-1)  # (T, hd)

    def rotate(x):
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(-2)

    return (q * cos + rotate(q) * sin) * scale, (k * cos + rotate(k) * sin) / scale


class _Block(nn.Module):
    """Pre-norm multi-head attention + GELU MLP.

    ``axis="time"``: causal over the patch axis with rotary positions.
    ``axis="variate"``: full attention over the variate axis (no positions —
    variates are unordered). Both use PerDimScale query scaling with ``1/d_k``
    attention scaling (μP-compatible), biases on attention projections only.
    """

    def __init__(self, cfg: Toto2Config, axis: str, block_idx: int = 0):
        super().__init__()
        self.cfg = cfg
        self.axis = axis
        inner = cfg.num_heads * cfg.head_dim
        # norm_eps = 1e-4, norm_include_weight = false — per the released config.
        self.norm1 = nn.LayerNorm(cfg.d_model, eps=1e-4, elementwise_affine=False)
        self.qkv = nn.Linear(cfg.d_model, 3 * inner, bias=True)
        self.proj = nn.Linear(inner, cfg.d_model, bias=True)
        self.norm2 = nn.LayerNorm(cfg.d_model, eps=1e-4, elementwise_affine=False)
        hidden = cfg.ffn_hidden
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, hidden, bias=False),
            nn.GELU(),
            nn.Linear(hidden, cfg.d_model, bias=False),
        )
        # PerDimScale: learned per-dimension query scaling; softplus(0) = ln 2
        # normalizer so the init is an exact no-op.
        self.per_dim_scale = nn.Parameter(torch.zeros(cfg.head_dim))
        if axis == "time":
            half = cfg.head_dim // 2
            idx = torch.arange(half).float() / max(1, half)
            self.register_buffer("inv_freq", 1.0 / (10000.0**idx), persistent=False)
            self.register_buffer("zeta", (idx + 0.4) / 1.4, persistent=False)  # xPos γ=0.4

        # u-μP residual scheme (u-μP eq. 25–31; Toto 2.0 §4.4): stream and
        # branch combine as x ← b·x + a·branch with a² + b² = 1, keeping the
        # residual stream at unit scale. α_res = residual_mult = 0.75 and
        # α_res-attn-ratio = sqrt(S/log S) with S = context patches — exactly
        # the released config's residual_mult / residual_attn_ratio (≈5.136
        # at S = 128). Branches count attention and MLP separately (L = 2·layers).
        S = max(2.0, cfg.context_length / cfg.patch_size)
        ratio2 = S / math.log(S)                    # α_res-attn-ratio²
        af2 = 2.0 * cfg.residual_mult**2 / (ratio2 + 1.0)
        aa2 = ratio2 * af2
        L = 2.0 * cfg.num_layers
        i = block_idx
        tau2_attn = aa2 / (L / 2.0 + i * aa2 + i * af2)
        tau2_mlp = af2 / (L / 2.0 + (i + 1) * aa2 + i * af2)
        self.attn_a = math.sqrt(tau2_attn / (tau2_attn + 1.0))
        self.attn_b = math.sqrt(1.0 / (tau2_attn + 1.0))
        self.mlp_a = math.sqrt(tau2_mlp / (tau2_mlp + 1.0))
        self.mlp_b = math.sqrt(1.0 / (tau2_mlp + 1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).view(B, T, 3, self.cfg.num_heads, self.cfg.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # (B, H, T, hd)
        if self.axis == "time":
            q, k = _xpos(q, k, self.inv_freq, self.zeta)
        q = q * (F.softplus(self.per_dim_scale) / math.log(2.0))
        attn = F.scaled_dot_product_attention(
            q, k, v, is_causal=(self.axis == "time"), scale=1.0 / self.cfg.head_dim
        )
        attn = attn.transpose(1, 2).reshape(B, T, self.cfg.num_heads * self.cfg.head_dim)
        x = self.attn_b * x + self.attn_a * self.proj(attn)
        x = self.mlp_b * x + self.mlp_a * self.mlp(self.norm2(x))
        return x


class Toto2Model(nn.Module):
    """Patch transformer with contiguous patch masking and alternating
    time/variate attention, predicting the next patch's per-step quantiles.

    Each input patch carries a binary mask channel (1 = unobserved entry);
    masked entries are zeroed on input, so a masked patch contributes only its
    position and mask bits. Training masks random contiguous spans (CPM);
    inference appends fully-masked horizon patches and reads every horizon
    patch's quantiles from a single forward pass.
    """

    def __init__(self, cfg: Toto2Config):
        super().__init__()
        self.cfg = cfg
        # values ‖ mask channel → 2×patch_size inputs per patch.
        self.patch_embed = nn.Linear(cfg.patch_size * 2, cfg.d_model)
        self.embed_mlp = _ResidualMLP(cfg.d_model, cfg.ffn_hidden)
        # grouped layers: variate-axis attention closes each group of
        # ``layer_group_size`` (Toto-2.0's 3-time-then-1-variate pattern).
        self.blocks = nn.ModuleList(
            _Block(cfg, axis=cfg.layer_axis(i), block_idx=i)
            for i in range(cfg.num_layers)
        )
        self.norm = nn.LayerNorm(cfg.d_model, eps=1e-4, elementwise_affine=False)
        self.out_mlp = _ResidualMLP(cfg.d_model, cfg.ffn_hidden)
        # each position predicts the NEXT patch: patch_size steps × num_quantiles
        self.head = nn.Linear(cfg.d_model, cfg.patch_size * cfg.num_quantiles)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        # u-μP-flavoured init: linear weights ~ N(0, 1/fan_in); the operator can
        # swap in exact u-μP multipliers and pin base_arch_digest accordingly.
        if isinstance(m, nn.Linear):
            fan_in = m.weight.shape[1]
            nn.init.normal_(m.weight, mean=0.0, std=1.0 / math.sqrt(fan_in))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, patches: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """``patches``: ``(B, P, patch_size)`` univariate or
        ``(B, C, P, patch_size)`` multivariate; ``mask``: optional binary
        patch-level (``(B, P)`` / ``(B, C, P)``) or per-entry (same + trailing
        ``patch_size`` axis), 1 = unobserved. Returns predicted quantiles for
        each position's *next* patch, shaped like the input with a trailing
        ``num_q`` axis: ``(B, [C,] P, patch_size, num_q)``."""
        squeeze_variates = patches.dim() == 3
        if squeeze_variates:
            patches = patches[:, None]                    # (B, 1, P, ps)
            if mask is not None:
                mask = mask[:, None]
        B, C, P, ps = patches.shape
        if mask is None:
            mask = torch.zeros_like(patches)
        else:
            if mask.dim() == 3:
                mask = mask[..., None].expand(B, C, P, ps)
            mask = mask.to(patches.dtype)
        x = torch.cat([patches * (1.0 - mask), mask], dim=-1)
        x = self.embed_mlp(self.patch_embed(x))           # (B, C, P, d)
        for blk in self.blocks:
            if blk.axis == "time":
                x = blk(x.reshape(B * C, P, -1)).view(B, C, P, -1)
            else:
                x = (
                    blk(x.transpose(1, 2).reshape(B * P, C, -1))
                    .view(B, P, C, -1)
                    .transpose(1, 2)
                )
        x = self.out_mlp(self.norm(x))
        out = self.head(x)                                # (B, C, P, ps*num_q)
        out = out.view(B, C, P, ps, self.cfg.num_quantiles)
        return out[:, 0] if squeeze_variates else out


def pinball_loss(pred_q: torch.Tensor, target: torch.Tensor, levels: tuple[float, ...]) -> torch.Tensor:
    """Mean pinball (quantile) loss. ``pred_q`` ``(..., num_q)``, ``target``
    ``(...)`` broadcast over the quantile axis."""
    q = torch.tensor(levels, device=pred_q.device, dtype=pred_q.dtype)
    err = target.unsqueeze(-1) - pred_q
    return torch.maximum(q * err, (q - 1.0) * err).mean()
