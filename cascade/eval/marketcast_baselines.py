"""Strong simple baselines for MarketCast (owner 2026-11-18): what a fine-tune
must beat besides the pinned base, so "beats the base" can never mean "beats
a weak model on a cell the base forecasts badly".

Both are deterministic functions of a window's CONTEXT alone (no fitting
across windows, no randomness), so the audit re-derives them from the
snapshot:

* ``rw_fhs`` — returns and the ohlcv price levels: a random walk with an
  EWMA volatility and filtered-historical-simulation innovations (the
  RiskMetrics / FHS risk baseline). One-bar returns are i.i.d. draws of
  ``σ · z`` with ``z`` the context's standardised returns; a log price
  level's ``k``-step path widens as ``σ √k`` around the last value.
* ``seasonal_ar`` — vol, range, high / low, liquidity, ohlcv volume: a
  seasonal profile (the context's median per time-of-day phase, period
  :func:`~cascade.eval.marketcast.gift_season`) plus an AR(1) on the
  deseasonalised residual with empirical one-step innovations, its spread
  growing to the residual's unconditional spread.
"""

from __future__ import annotations

import numpy as np

from ..shared.config.marketcast import base_suite
from .marketcast import gift_season

EWMA_LAMBDA = 0.97            # RiskMetrics-style decay per bar
AR_EWMA = 0.9                 # the residual level's smoothing
MIN_POINTS = 16


def _ewma_sigma(x: np.ndarray, lam: float = EWMA_LAMBDA) -> tuple[float, np.ndarray]:
    """``(σ at the end, σ_t before each x_t)`` of an EWMA of ``x²`` seeded
    with the first quarter's mean square."""
    seed = float(np.mean(x[: max(4, x.size // 4)] ** 2))
    v = np.empty(x.size)
    s = seed
    for i, xi in enumerate(x):
        v[i] = s
        s = lam * s + (1.0 - lam) * xi * xi
    return float(np.sqrt(max(s, 1e-24))), np.sqrt(np.maximum(v, 1e-24))


def _rw_fhs(ctx: np.ndarray, steps: int, levels: np.ndarray, *, level: bool) -> np.ndarray:
    c = ctx[np.isfinite(ctx)]
    if level:
        c = c[c > 0]
        if c.size < MIN_POINTS:
            return np.full((steps, levels.size), np.nan)
        lc = np.log(c)
        d = np.diff(lc)
    else:
        if c.size < MIN_POINTS:
            return np.full((steps, levels.size), np.nan)
        d = c
    sig, sig_t = _ewma_sigma(d)
    z = d / sig_t
    zq = np.quantile(z, levels) if z.size else np.zeros(levels.size)
    if not level:
        return np.tile(sig * zq, (steps, 1))
    k = np.sqrt(np.arange(1, steps + 1))[:, None]
    return np.exp(lc[-1] + sig * k * zq[None, :])


def _seasonal_ar(ctx: np.ndarray, steps: int, levels: np.ndarray, freq_minutes: int
                 ) -> np.ndarray:
    n = ctx.size
    m = gift_season(freq_minutes)
    phase = np.arange(n) % m
    prof = np.zeros(m)
    if m > 1 and n >= 2 * m:
        for p in range(m):
            v = ctx[phase == p]
            v = v[np.isfinite(v)]
            prof[p] = float(np.median(v)) if v.size else np.nan
        prof = np.where(np.isfinite(prof), prof, np.nanmedian(ctx))
    else:
        prof[:] = np.nanmedian(ctx)
    r = ctx - prof[phase]
    ok = np.isfinite(r)
    if ok.sum() < MIN_POINTS:
        return np.full((steps, levels.size), np.nan)
    rr = r[ok]
    a, b = rr[:-1], rr[1:]
    rho = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 0.0
    rho = min(max(rho if np.isfinite(rho) else 0.0, 0.0), 0.99)
    lvl = rr[0]
    for x in rr:
        lvl = AR_EWMA * lvl + (1.0 - AR_EWMA) * x
    e = b - rho * a                                        # one-step innovations
    eq = np.quantile(e - np.median(e), levels)
    uq = np.quantile(rr - np.median(rr), levels)           # the unconditional spread
    k = np.arange(1, steps + 1)
    grow = np.sqrt((1 - rho ** (2 * k)) / max(1 - rho * rho, 1e-6))[:, None]
    spread = np.where(np.abs(eq[None, :] * grow) < np.abs(uq[None, :]), eq[None, :] * grow,
                      uq[None, :])
    centre = prof[(n + k - 1) % m] + np.median(rr) + (lvl - np.median(rr)) * rho ** k
    return centre[:, None] + spread


def baseline_kind(label: str) -> str:
    """The baseline family of a window suite label."""
    s = base_suite(label)
    if s in ("returns", "alpha", "funding"):
        return "rw_fhs"
    if s == "ohlcv" and not str(label).endswith(".v"):
        return "rw_fhs_level"
    return "seasonal_ar"


def baseline_quantiles(label: str, context, steps: int, freq_minutes: int, levels
                       ) -> np.ndarray:
    """``(steps, K)`` baseline quantiles at ``levels`` for one window of
    suite label ``label`` (:func:`baseline_kind`), from its ``context``
    alone; monotone in the level; NaN when the context is too thin."""
    ctx = np.asarray(context, dtype=np.float64)
    lv = np.asarray(levels, dtype=np.float64)
    kind = baseline_kind(label)
    if kind == "rw_fhs":
        q = _rw_fhs(ctx, int(steps), lv, level=False)
    elif kind == "rw_fhs_level":
        q = _rw_fhs(ctx, int(steps), lv, level=True)
    else:
        q = _seasonal_ar(ctx, int(steps), lv, int(freq_minutes))
    return np.sort(q, axis=-1)


__all__ = ["baseline_kind", "baseline_quantiles"]
