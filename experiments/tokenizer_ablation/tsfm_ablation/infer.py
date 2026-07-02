# ============================================================================
# INFERENCE CORE  ·  turn a MiniTSFM2 into real-space quantile forecasts for
# ragged, variable-length univariate series. Shared by both eval harnesses
# (GIFT-Eval, TIME) so they score the model through one identical path.
# ============================================================================
"""Batched quantile forecasting for arbitrary-length context series.

Every eval below reduces to the same primitive: given a list of raw 1-D history
series and a horizon H, return real-space quantiles ``(N, H, Q)``. Short
horizons (``k <= train_kmax`` patches) decode in a single pass; longer horizons
fall back to block rollout with median feedback (the same scheme the model was
built for). NaNs are forward-filled and short/ragged series are left-padded with
the first real value, marked invalid so the robust scaler ignores the padding.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .device import AMP_DTYPE, DEVICE


def _rollout_quantiles(model, ctx, valid, k):
    """Block rollout for horizons past the trained tail: decode ``train_kmax``
    patches at a time, feed the median back, and concatenate per-block quantiles."""
    kmax = model.cfg.train_kmax
    outs, cur, cv, k_left = [], ctx, valid, k
    while k_left > 0:
        kb = min(k_left, kmax)
        q = model.predict(cur, kb, valid=cv)                  # [B,1,kb*P,Q]
        outs.append(q)
        med = q[..., q.shape[-1] // 2]
        cur = torch.cat([cur, med], dim=-1)[..., -model.cfg.ctx_span:]
        cv = torch.cat([cv, torch.ones_like(med)], dim=-1)[..., -model.cfg.ctx_span:]
        k_left -= kb
    return torch.cat(outs, dim=2)


def batched_quantiles(model, series, H, batch_size=256):
    """Forecast each 1-D series independently to horizon ``H``.

    Parameters
    ----------
    model : MiniTSFM2
    series : sequence of 1-D float arrays (raw units; NaNs allowed)
    H : int, forecast horizon in timesteps
    batch_size : int, series per forward pass

    Returns
    -------
    np.ndarray of shape ``(len(series), H, len(model.q_levels))`` — real-space
    quantiles, ordered by ``model.q_levels`` (already de-crossed by ``predict``).
    """
    model.eval()
    P = model.cfg.patch
    k = math.ceil(H / P)
    Q = len(model.q_levels)
    out = np.empty((len(series), H, Q), np.float32)
    for i in range(0, len(series), batch_size):
        chunk = series[i:i + batch_size]
        Cmax = min(model.cfg.ctx_span, max(len(np.atleast_1d(s)) for s in chunk))
        Cmax = max(Cmax, P)
        ctx = np.zeros((len(chunk), 1, Cmax), np.float32)
        val = np.zeros((len(chunk), 1, Cmax), np.float32)
        for j, s in enumerate(chunk):
            y = np.asarray(s, np.float32).reshape(-1)
            # forward-fill NaNs, then keep the last Cmax real steps
            idx = np.where(~np.isnan(y), np.arange(len(y)), 0)
            np.maximum.accumulate(idx, out=idx)
            y = np.nan_to_num(y[idx], nan=0.0)
            y = y[-Cmax:]
            ctx[j, 0, -len(y):] = y
            val[j, 0, -len(y):] = 1.0
        ctxt = torch.from_numpy(ctx).to(DEVICE)
        valt = torch.from_numpy(val).to(DEVICE)
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=(DEVICE == "cuda" and AMP_DTYPE is not None)):
            if k <= model.cfg.train_kmax:
                q = model.predict(ctxt, k, valid=valt)         # [B,1,k*P,Q]
            else:
                q = _rollout_quantiles(model, ctxt, valt, k)
        out[i:i + len(chunk)] = q.float().cpu().numpy()[:, 0, :H, :]
    return out
