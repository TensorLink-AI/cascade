# ============================================================================
# TOKENIZERS  ·  fixed | pyramid | adaptive   (+ mask-aware scaler, RoPE)
# ----------------------------------------------------------------------------
# Shared contract: tokenize(z, m) -> (feat [BV, N, 2*inner+1], pos [N] or [BV,N])
#   z    : arcsinh-scaled series [BV, L] with masked steps zeroed
#   m    : per-timestep mask channel [BV, L] (1 = masked / to-predict)
#   feat : pooled values (inner=patch) ++ pooled mask ++ log(seg_len/patch)
#   pos  : TIME-SCALED RoPE positions = patch_start_time / patch. Identical to
#          token index for the fixed tokenizer (Toto default); preserves
#          temporal geometry for non-uniform patches.
# Only the fine band (last fine_span steps, patch-size `patch`) + appended
# tail patches are ever masked or decoded — the head stays fixed patch-32.
# ============================================================================
"""Context tokenizers + the mask-aware robust scaler and RoPE helpers.

The three tokenizers share one feature contract (pooled values ++ pooled mask
++ log seg-length), so the embedding is identical and every arm has exactly the
same parameter count — no capacity confound.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RobustScaler:
    """Causal arcsinh scaler; mask-aware (masked steps excluded via nanmedian)."""

    def __init__(self, x, m=None):
        xx = x.clone()
        if m is not None:
            xx[m > 0.5] = float("nan")
        self.loc = xx.nanmedian(dim=-1, keepdim=True).values
        self.loc = torch.nan_to_num(self.loc, nan=0.0)
        self.scale = (xx - self.loc).abs().nanmedian(dim=-1, keepdim=True).values
        self.scale = torch.nan_to_num(self.scale, nan=1.0).clamp_min(1e-3)

    def encode(self, x): return torch.asinh((x - self.loc) / self.scale)

    def decode(self, z): return torch.sinh(z) * self.scale + self.loc


def rope_angles(positions, dim, base=10000.0):
    """positions [N] or [BV,N] (float ok) -> angles [.., N, dim/2]."""
    i = torch.arange(0, dim, 2, device=positions.device).float()
    inv = base ** (-i / dim)
    return positions.float().unsqueeze(-1) * inv


def apply_rope(x, ang):
    """x [B,H,N,Dh]; ang [N,Dh/2] or [B,N,Dh/2]."""
    if ang.dim() == 2:  ang = ang[None, None]          # [1,1,N,d2]
    else:               ang = ang[:, None]             # [B,1,N,d2]
    cos = torch.cos(ang).repeat_interleave(2, dim=-1)
    sin = torch.sin(ang).repeat_interleave(2, dim=-1)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    rot = torch.stack((-x2, x1), dim=-1).flatten(-2)
    return x * cos + rot * sin


def _pool_patch(seg, inner):
    """[BV, n, L] -> [BV, n, inner]: mean-pool (antialias) or repeat-upsample."""
    L = seg.shape[-1]
    if inner == L: return seg
    if inner < L:
        assert L % inner == 0
        return seg.reshape(*seg.shape[:-1], inner, L // inner).mean(-1)
    assert inner % L == 0
    return seg.repeat_interleave(inner // L, dim=-1)


def _feat(vals, msk, seg_len, inner):
    """concat pooled values, pooled mask, log(seg_len/inner) -> [BV, n, 2*inner+1]."""
    ll = torch.full((*vals.shape[:-1], 1), math.log(seg_len / inner),
                    device=vals.device, dtype=vals.dtype)
    return torch.cat([vals, msk, ll], dim=-1)


class FixedTokenizer(nn.Module):
    def __init__(self, cfg): super().__init__(); self.cfg = cfg

    def forward(self, z, m):
        P, BV = self.cfg.patch, z.shape[0]
        n = z.shape[1] // P
        v = z.reshape(BV, n, P); mm = m.reshape(BV, n, P)
        feat = _feat(v, mm, P, P)
        pos = torch.arange(n, device=z.device).float()          # == start/P
        return feat, pos


class PyramidTokenizer(nn.Module):
    def __init__(self, cfg):
        super().__init__(); self.cfg = cfg
        self.levels = cfg.pyramid_levels
        assert sum(s for s, _ in self.levels) == cfg.ctx_span
        assert self.levels[-1][1] == cfg.patch and self.levels[-1][0] == cfg.fine_span
        # level identity is carried by the log(seg_len/patch) feature + time position

    def forward(self, z, m):
        P, BV = self.cfg.patch, z.shape[0]
        feats, poss, ofs = [], [], 0
        for span, plen in self.levels:
            n = span // plen
            v = _pool_patch(z[:, ofs:ofs + span].reshape(BV, n, plen), P)
            mm = _pool_patch(m[:, ofs:ofs + span].reshape(BV, n, plen), P)
            feats.append(_feat(v, mm, plen, P))
            poss.append((torch.arange(n, device=z.device).float() * plen + ofs) / P)
            ofs += span
        return torch.cat(feats, 1), torch.cat(poss)


class AdaptiveTokenizer(nn.Module):
    """Equal-surprise segmentation of the pre-fine history into a fixed token
    budget K (deterministic per series -> eval-stable), plus the fixed fine band.
    Surprise = smoothed |dz|; boundaries at surprise-CDF quantiles with a uniform
    floor mixed in (guarantees no degenerate segments); segments are pooled to
    `patch` inner values via linear-interp gather from a mildly smoothed copy."""

    def __init__(self, cfg, floor=0.35):
        super().__init__(); self.cfg = cfg; self.floor = floor

    def forward(self, z, m):
        cfg, P = self.cfg, self.cfg.patch
        BV = z.shape[0]
        Lh = cfg.ctx_span - cfg.fine_span                    # history region
        K = cfg.adaptive_hist_tokens
        zh, mh = z[:, :Lh], m[:, :Lh]
        # --- surprise + equal-surprise boundaries ---------------------------
        s = (zh[:, 1:] - zh[:, :-1]).abs()
        s = F.avg_pool1d(s[:, None], 9, 1, 4)[:, 0]
        s = F.pad(s, (1, 0), value=0.0)
        s = (1 - self.floor) * s / (s.sum(1, keepdim=True) + 1e-8) + self.floor / Lh
        cs = s.cumsum(1)                                     # [BV, Lh] in (0,1]
        tgt = (torch.arange(1, K, device=z.device).float() / K)[None].expand(BV, -1)
        bnd = torch.searchsorted(cs.contiguous(), tgt.contiguous())      # [BV, K-1]
        bnd = bnd.clamp(1, Lh - 1)
        bnd, _ = torch.cummax(bnd, dim=1)                    # enforce monotone
        starts = torch.cat([torch.zeros(BV, 1, device=z.device, dtype=bnd.dtype), bnd], 1)
        ends = torch.cat([bnd, torch.full((BV, 1), Lh, device=z.device, dtype=bnd.dtype)], 1)
        seg_len = (ends - starts).clamp_min(1).float()       # [BV, K]
        # --- pooled inner values via interp gather (2 mip levels) ------------
        z_sm = F.avg_pool1d(zh[:, None], 17, 1, 8)[:, 0]     # antialiased copy
        use_sm = (seg_len > 2 * P)[..., None]                # long segs -> smoothed
        frac = (torch.arange(P, device=z.device).float() + 0.5) / P
        pos_f = starts[..., None].float() + seg_len[..., None] * frac    # [BV,K,P]
        i0 = pos_f.floor().long().clamp(0, Lh - 1); i1 = (i0 + 1).clamp(0, Lh - 1)
        w = (pos_f - i0.float())

        def gath(src):
            g0 = torch.gather(src[:, None].expand(-1, K, -1), 2, i0)
            g1 = torch.gather(src[:, None].expand(-1, K, -1), 2, i1)
            return g0 * (1 - w) + g1 * w
        v = torch.where(use_sm, gath(z_sm), gath(zh))
        mm = gath(mh)                                        # history is never masked, ~0
        ll = torch.log(seg_len / P)[..., None]
        feat_h = torch.cat([v, mm, ll], dim=-1)              # [BV, K, 2P+1]
        pos_h = ((starts.float() + ends.float()) * 0.5) / P  # segment-center time / P
        # --- fine band (identical to fixed) ----------------------------------
        nf = cfg.zone_tokens
        vf = z[:, Lh:].reshape(BV, nf, P); mf = m[:, Lh:].reshape(BV, nf, P)
        feat_f = _feat(vf, mf, P, P)
        pos_f2 = ((torch.arange(nf, device=z.device).float() * P + Lh) / P)[None].expand(BV, -1)
        return torch.cat([feat_h, feat_f], 1), torch.cat([pos_h, pos_f2], 1)


def build_tokenizer(cfg):
    return {"fixed": FixedTokenizer, "pyramid": PyramidTokenizer,
            "adaptive": AdaptiveTokenizer}[cfg.tokenizer](cfg)
