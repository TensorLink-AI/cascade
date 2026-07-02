# ============================================================================
# MODEL  ·  MiniTSFM2 — Toto-2-4m-recipe backbone with pluggable tokenizer
# decoder-only, causal time attention (index RoPE @ time-scaled positions),
# variate attention in the LAST layer, quantile head (9 levels, patch 32),
# CPM training (interior spans in the fine band + masked tail), pinball loss
# in z-space. Only fine tokens are decoded; coarse tokens are context-only.
# ============================================================================
"""The ~3.6M-param backbone plus the contiguous-patch-masking (CPM) sampler."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import STD_LEVELS_LIST
from .tokenizers import RobustScaler, apply_rope, build_tokenizer, rope_angles

QUANTILES = torch.tensor(STD_LEVELS_LIST)


class TimeAttn(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.h, self.dh = cfg.n_heads, cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.o = nn.Linear(cfg.d_model, cfg.d_model)

    def forward(self, x, ang):
        B, N, D = x.shape
        q, k, v = self.qkv(x).chunk(3, -1)
        q, k, v = [t.view(B, N, self.h, self.dh).transpose(1, 2) for t in (q, k, v)]
        q, k = apply_rope(q, ang), apply_rope(k, ang)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(out.transpose(1, 2).reshape(B, N, D))


class VariateAttn(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.h, self.dh = cfg.n_heads, cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.o = nn.Linear(cfg.d_model, cfg.d_model)

    def forward(self, x):                       # x [B*N, V, D]
        B, V, D = x.shape
        q, k, v = self.qkv(x).chunk(3, -1)
        q, k, v = [t.view(B, V, self.h, self.dh).transpose(1, 2) for t in (q, k, v)]
        out = F.scaled_dot_product_attention(q, k, v)
        return self.o(out.transpose(1, 2).reshape(B, V, D))


class Block(nn.Module):
    def __init__(self, cfg, use_variate):
        super().__init__()
        self.n1 = nn.LayerNorm(cfg.d_model); self.attn = TimeAttn(cfg)
        self.use_variate = use_variate
        if use_variate:
            self.nv = nn.LayerNorm(cfg.d_model); self.vattn = VariateAttn(cfg)
        self.n2 = nn.LayerNorm(cfg.d_model)
        self.ff = nn.Sequential(nn.Linear(cfg.d_model, 4 * cfg.d_model), nn.SiLU(),
                                nn.Linear(4 * cfg.d_model, cfg.d_model))

    def forward(self, x, ang, V, N):
        x = x + self.attn(self.n1(x), ang)
        if self.use_variate and V > 1:
            B = x.shape[0] // V
            xr = self.nv(x).view(B, V, N, -1).permute(0, 2, 1, 3).reshape(B * N, V, -1)
            mix = self.vattn(xr).view(B, N, V, -1).permute(0, 2, 1, 3).reshape(B * V, N, -1)
            x = x + mix
        return x + self.ff(self.n2(x))


class MiniTSFM2(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        P = cfg.patch
        self.tokenizer = build_tokenizer(cfg)
        self.embed = nn.Sequential(nn.Linear(2 * P + 1, cfg.d_model), nn.SiLU(),
                                   nn.Linear(cfg.d_model, cfg.d_model))
        self.mask_tok = nn.Parameter(torch.randn(cfg.d_model) * 0.02)
        self.blocks = nn.ModuleList([Block(cfg, use_variate=(layer == cfg.n_layers - 1))
                                     for layer in range(cfg.n_layers)])
        self.norm = nn.LayerNorm(cfg.d_model)
        self.q_levels = QUANTILES
        self.head = nn.Linear(cfg.d_model, P * len(STD_LEVELS_LIST))

    # ------------------------------------------------------------------ core
    def _tail_feat(self, BV, k, device, dtype):
        P = self.cfg.patch
        v = torch.zeros(BV, k, P, device=device, dtype=dtype)
        m = torch.ones(BV, k, P, device=device, dtype=dtype)
        ll = torch.zeros(BV, k, 1, device=device, dtype=dtype)
        return torch.cat([v, m, ll], dim=-1)

    def _forward_tokens(self, z_in, m_ts, k, tok_mask):
        """z_in, m_ts [BV, ctx]; k tail patches; tok_mask [BV, n_dec] (1=masked).
        Returns z-space quantiles [BV, n_dec, P, Q] for the decode region
        (fine band + tail)."""
        cfg, P = self.cfg, self.cfg.patch
        BV = z_in.shape[0]
        feat, pos = self.tokenizer(z_in, m_ts)
        tail = self._tail_feat(BV, k, z_in.device, feat.dtype)
        feat = torch.cat([feat, tail], 1)
        tail_pos = (cfg.ctx_span + torch.arange(k, device=z_in.device).float() * P) / P
        if pos.dim() == 1:
            pos = torch.cat([pos, tail_pos])
        else:
            pos = torch.cat([pos, tail_pos[None].expand(BV, -1)], 1)
        x = self.embed(feat)
        N = x.shape[1]; n_dec = cfg.zone_tokens + k
        # add learned mask token at masked decode positions
        full_mask = torch.zeros(BV, N, device=x.device, dtype=x.dtype)
        full_mask[:, -n_dec:] = tok_mask.to(x.dtype)
        x = x + full_mask[..., None] * self.mask_tok
        ang = rope_angles(pos, cfg.d_model // cfg.n_heads)
        V = getattr(self, "_V", 1)
        for blk in self.blocks:
            x = blk(x, ang, V, N)
        h = self.norm(x[:, -n_dec:])
        o = self.head(h).view(BV, n_dec, P, len(STD_LEVELS_LIST))
        return o

    # ------------------------------------------------------------------ train
    def forward_train(self, x, k, mask_patches):
        """x [B,V, ctx+k*P] raw; mask_patches [B,V, zone+k] bool (1=masked).
        Returns (loss, z-quantiles, z-targets, flat mask)."""
        cfg, P = self.cfg, self.cfg.patch
        B, V, L = x.shape
        self._V = V
        BV = B * V
        xf = x.reshape(BV, L)
        mp = mask_patches.reshape(BV, -1)                     # [BV, zone+k]
        n_dec = cfg.zone_tokens + k
        assert mp.shape[1] == n_dec
        # per-timestep mask over the decode region (last fine_span + k*P steps)
        m_ts = torch.zeros_like(xf)
        dec_span = cfg.fine_span + k * P
        m_dec = mp[..., None].expand(BV, n_dec, P).reshape(BV, dec_span).float()
        m_ts[:, -dec_span:] = m_dec
        sc = RobustScaler(xf[:, :cfg.ctx_span], m_ts[:, :cfg.ctx_span])
        z_true = sc.encode(xf)
        z_in = z_true[:, :cfg.ctx_span] * (1 - m_ts[:, :cfg.ctx_span])
        o = self._forward_tokens(z_in, m_ts[:, :cfg.ctx_span], k, mp)
        # targets: z-space patches over the decode region
        tgt = z_true[:, -dec_span:].reshape(BV, n_dec, P)
        q = self.q_levels.to(o.device)
        err = tgt[..., None] - o                              # [BV, n_dec, P, Q]
        pin = err * (q - (err < 0).float())
        w = mp.float()[..., None, None]
        loss = (pin * w).sum() / (w.sum() * P * len(STD_LEVELS_LIST) + 1e-8)
        return loss, o, tgt, mp

    # ------------------------------------------------------------------ infer
    @torch.inference_mode()
    def predict(self, ctx, k, clamp=True, valid=None):
        """ctx [B,V,C] raw (any C); returns real-space quantiles [B,V,k*P,Q].
        `valid` [B,V,C] optionally marks real (1) vs padded (0) steps when the
        caller pre-pads ragged series to a common length; padded steps are
        excluded from the scaler."""
        cfg, P = self.cfg, self.cfg.patch
        B, V, C = ctx.shape
        self._V = V
        BV = B * V
        x = ctx.reshape(BV, C)
        vld = torch.ones_like(x) if valid is None else valid.reshape(BV, C).float()
        if cfg.ctx_span > C:                                  # left-pad, edge value
            first = torch.argmax(vld, dim=1, keepdim=True)    # first real step
            edge = torch.gather(x, 1, first)
            pad = edge.expand(BV, cfg.ctx_span - C)
            xf = torch.cat([pad, x], 1)
            valid_f = torch.cat([torch.zeros(BV, cfg.ctx_span - C, device=x.device), vld], 1)
        else:
            xf = x[:, -cfg.ctx_span:]
            valid_f = vld[:, -cfg.ctx_span:]
        # padded/invalid steps carry the first-real (edge) value, mask-channel 0
        first = torch.argmax(valid_f, dim=1, keepdim=True)
        edge = torch.gather(xf, 1, first)
        xf = torch.where(valid_f > 0.5, xf, edge)
        valid = valid_f
        sc = RobustScaler(xf, m=(1 - valid))                  # scaler on real steps only
        z_in = sc.encode(xf)                                  # padded steps: edge value, mask=0
        m_ts = torch.zeros_like(z_in)
        tok_mask = torch.zeros(BV, cfg.zone_tokens + k, device=x.device)
        tok_mask[:, -k:] = 1.0
        o = self._forward_tokens(z_in, m_ts, k, tok_mask)     # [BV, zone+k, P, Q]
        o = o[:, -k:].reshape(BV, k * P, len(STD_LEVELS_LIST))
        o, _ = torch.sort(o, dim=-1)                          # de-cross
        q = (torch.sinh(o) * sc.scale[..., None] + sc.loc[..., None]).reshape(B, V, k * P, -1)
        if clamp:                                             # wide sanity clamp
            lo = ctx.min(dim=-1, keepdim=True).values[..., None]
            hi = ctx.max(dim=-1, keepdim=True).values[..., None]
            rng = (hi - lo).clamp_min(1e-3)
            q = q.clamp(lo - 10 * rng, hi + 10 * rng)
        return q

    @torch.inference_mode()
    def rollout(self, ctx, total_h, block_k=None):
        """Median-feedback block decoding past the trained tail length."""
        cfg, P = self.cfg, self.cfg.patch
        block_k = block_k or cfg.train_kmax
        cur, preds = ctx, []
        n_blocks = math.ceil(total_h / (block_k * P))
        for _ in range(n_blocks):
            q = self.predict(cur, block_k)                    # [B,V,bk*P,Q]
            med = q[..., q.shape[-1] // 2]
            preds.append(med)
            cur = torch.cat([cur, med], dim=-1)[..., -cfg.ctx_span:]
        return torch.cat(preds, dim=-1)[..., :total_h]


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# --------------------------------------------------------------------------- CPM sampler
def sample_cpm_mask(cfg, B, V, rng):
    """(k, mask_patches [B,V,zone+k]): tail span k~U{kmin..kmax} always masked,
    plus c~U{0..cmax} interior spans in the fine band, total <= pmax fraction."""
    k = int(rng.integers(cfg.train_kmin, cfg.train_kmax + 1))
    n_dec = cfg.zone_tokens + k
    mp = torch.zeros(B, V, n_dec, dtype=torch.bool)
    mp[..., -k:] = True
    budget = max(0, int(cfg.cpm_pmax * n_dec) - k)
    for b in range(B):
        for v in range(V):
            used, c = 0, int(rng.integers(0, cfg.cpm_cmax + 1))
            for _ in range(c):
                if used >= budget: break
                w = int(rng.integers(1, cfg.cpm_span_max + 1))
                w = min(w, budget - used)
                s = int(rng.integers(0, cfg.zone_tokens - w + 1))
                mp[b, v, s:s + w] = True
                used += w
    return k, mp
