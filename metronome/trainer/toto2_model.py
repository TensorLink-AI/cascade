"""Reference Toto2-style backbone — a small causal patch transformer with a
multi-quantile head, trained from random initialisation.

This module is **self-contained torch** and is *copied into every checkpoint*
(as ``model.py``) so the validator's ``forecast_wrapper.py`` can rebuild the
exact architecture to load the weights. Keep it dependency-light (torch only)
and free of metronome imports for that reason.

It follows the Toto 2.0 recipe in spirit — patch embedding (``patch_size``),
a pre-norm causal transformer (``head_dim`` fixed at 64 across the family), an
arcsinh causal input transform, and a 9-level pinball/quantile head whose levels
are exactly metronome's eval objective — at the 4M rung. It is a faithful,
runnable reference, not a byte-exact reproduction of Datadog's checkpoint; pin
``base_arch_digest`` to whatever you launch with.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# The 9 quantile levels 0.1..0.9 — identical to metronome's eval grid so the
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
    num_quantiles: int = 9
    context_length: int = 4096
    horizon: int = 64
    max_patches: int = 256  # positional table size (>= (context+horizon)/patch)

    @classmethod
    def from_contract(cls, c: object) -> Toto2Config:
        """Build from a metronome ``TrainingContractConfig`` (duck-typed)."""
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
            num_quantiles=int(getattr(c, "num_quantiles", 9)),
            context_length=ctx,
            horizon=hz,
            max_patches=max(8, (ctx + hz) // ps + 4),
        )

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def causal_standardize(x: torch.Tensor, eps: float = 1e-5) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-series location/scale standardisation (the arcsinh causal transform's
    affine part). ``x`` is ``(B, L)``. Returns ``(z, loc, scale)`` where
    ``z = arcsinh((x - loc) / scale)``; loc/scale are computed over the sequence
    (a stand-in for Toto's causal scaler — at train time the whole series is the
    context)."""
    loc = x.mean(dim=-1, keepdim=True)
    scale = x.std(dim=-1, keepdim=True).clamp_min(eps)
    z = torch.asinh((x - loc) / scale)
    return z, loc, scale


def invert_standardize(z: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`causal_standardize`: ``x = sinh(z) * scale + loc``."""
    return torch.sinh(z) * scale + loc


class _Block(nn.Module):
    """Pre-norm causal multi-head attention + GELU MLP."""

    def __init__(self, cfg: Toto2Config):
        super().__init__()
        self.cfg = cfg
        inner = cfg.num_heads * cfg.head_dim
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, 3 * inner, bias=False)
        self.proj = nn.Linear(inner, cfg.d_model, bias=False)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        hidden = cfg.d_model * cfg.mlp_expansion
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, hidden), nn.GELU(), nn.Linear(hidden, cfg.d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).view(B, T, 3, self.cfg.num_heads, self.cfg.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # (B, H, T, hd)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).reshape(B, T, self.cfg.num_heads * self.cfg.head_dim)
        x = x + self.proj(attn)
        x = x + self.mlp(self.norm2(x))
        return x


class Toto2Model(nn.Module):
    """Causal patch transformer predicting the next patch's per-step quantiles."""

    def __init__(self, cfg: Toto2Config):
        super().__init__()
        self.cfg = cfg
        self.patch_embed = nn.Linear(cfg.patch_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_patches, cfg.d_model)
        self.blocks = nn.ModuleList(_Block(cfg) for _ in range(cfg.num_layers))
        self.norm = nn.LayerNorm(cfg.d_model)
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

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """``patches``: ``(B, P, patch_size)`` → ``(B, P, patch_size, num_q)``
        predicted quantiles for each position's *next* patch."""
        B, P, _ = patches.shape
        idx = torch.arange(P, device=patches.device)
        x = self.patch_embed(patches) + self.pos(idx)[None, :, :]
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        out = self.head(x)  # (B, P, patch_size*num_q)
        return out.view(B, P, self.cfg.patch_size, self.cfg.num_quantiles)


def pinball_loss(pred_q: torch.Tensor, target: torch.Tensor, levels: tuple[float, ...]) -> torch.Tensor:
    """Mean pinball (quantile) loss. ``pred_q`` ``(..., num_q)``, ``target``
    ``(...)`` broadcast over the quantile axis."""
    q = torch.tensor(levels, device=pred_q.device, dtype=pred_q.dtype)
    err = target.unsqueeze(-1) - pred_q
    return torch.maximum(q * err, (q - 1.0) * err).mean()
