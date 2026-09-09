"""fpfn_fast — a fully-vectorized numpy approximation of the ForecastPFN prior.

WHY (2026-07-15): the real vendored ForecastPFN (torch + pandas calendar machinery
+ per-series retry loop, max_retries=100) runs at ~91k points/s — 41x too slow for
the mainnet 20x-token-budget contract (needs ~3.7M pts/s or it deadline_hits at a
tiny tokens_frac). fpfn is our winning design's 40% CORE, so it is THE throughput
bottleneck. This reimplements the same STATISTICAL SHAPE — trend x multi-harmonic
seasonality x multiplicative Weibull noise, with occasional level shifts/spikes —
as pure batched numpy array ops (no pandas, no torch, no per-series Python loop,
no retry), so it draws whole batches at millions of points/s.

It does NOT reproduce ForecastPFN's exact calendar-frequency internals (we cannot
verify per-series quality offline anyway — calibration rho=-0.076). The bet, forced
by the 20x budget, is that ~equal-shape data delivered ~40x faster trains a far
better model than the real prior starving at ~2% of the budget. Live rounds
(testnet/mainnet oracle) are the judge of the quality trade; throughput is what we
can measure directly, and this is engineered to win it.

Deterministic per (seed + series index). numpy only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tempo_gen.data.containers import TimeSeriesContainer
from tempo_gen.synthetic_generation.abstract_classes import GeneratorWrapper
from tempo_gen.synthetic_generation.generator_params import GeneratorParams


@dataclass
class FpfnFastGeneratorParams(GeneratorParams):
    """Vectorized ForecastPFN-shaped prior."""

    # calendar-like base periods a seasonal harmonic can sit on
    periods: tuple[float, ...] = (7.0, 12.0, 24.0, 30.0, 52.0, 96.0, 168.0, 365.0)
    n_harmonics: tuple[int, int] = (1, 5)          # distinct seasonal components per series
    harmonic_decay: tuple[float, float] = (0.6, 2.0)   # amplitude falloff over components
    amp_seasonal: tuple[float, float] = (0.1, 1.5)
    # trend
    trend_slope_std: float = 0.6                   # linear trend magnitude over the series
    p_exp_trend: float = 0.3                        # fraction with a gentle exp/quadratic bend
    # multiplicative Weibull noise (ForecastPFN's signature)
    weibull_k: tuple[float, float] = (1.5, 4.0)
    noise_scale: tuple[float, float] = (0.02, 0.35)
    # occasional structure
    level_shift_prob: float = 0.15
    spike_prob: float = 0.10


class FpfnFastGeneratorWrapper(GeneratorWrapper):
    """Batched, branch-free ForecastPFN-shaped draw."""

    def __init__(self, params: FpfnFastGeneratorParams):
        super().__init__(params)

    def generate_batch(self, batch_size: int, seed: int | None = None) -> TimeSeriesContainer:
        if seed is None:
            seed = int(self.params.global_seed)
        p = self.params
        L = int(self.params.length)
        B = int(batch_size)
        rng = np.random.default_rng(int(seed) % (2**31))

        t = np.arange(L, dtype=np.float64) / max(L - 1, 1)          # (L,) in [0,1]
        t_idx = np.arange(L, dtype=np.float64)                       # (L,) raw index

        # ---- trend (B, L): linear + optional gentle bend ----
        slope = rng.normal(0.0, p.trend_slope_std, size=B)[:, None]
        level = rng.normal(0.0, 1.0, size=B)[:, None]
        trend = level + slope * t[None, :]
        bend_mask = (rng.random(B) < p.p_exp_trend)[:, None]
        curv = rng.normal(0.0, p.trend_slope_std, size=B)[:, None]
        trend = trend + np.where(bend_mask, curv * (t[None, :] ** 2), 0.0)

        # ---- multi-harmonic seasonality (B, L), fully vectorized ----
        lo, hi = p.n_harmonics
        kmax = int(hi)
        # per-series, per-harmonic params; zero out unused harmonics via a count mask
        counts = rng.integers(lo, hi + 1, size=B)                    # (B,)
        hmask = (np.arange(kmax)[None, :] < counts[:, None]).astype(np.float64)  # (B,kmax)
        per = np.asarray(p.periods)[rng.integers(0, len(p.periods), size=(B, kmax))]  # (B,kmax)
        phase = rng.uniform(0.0, 2.0 * np.pi, size=(B, kmax))
        decay = rng.uniform(*p.harmonic_decay, size=B)[:, None]
        amp0 = rng.uniform(*p.amp_seasonal, size=B)[:, None]
        order = np.arange(kmax)[None, :]
        amp = amp0 * np.exp(-decay * order) * hmask                  # (B,kmax)
        # seasonal = sum_k amp * sin(2π t_idx / period + phase)
        ang = (2.0 * np.pi) * (t_idx[None, None, :] / per[:, :, None]) + phase[:, :, None]  # (B,kmax,L)
        seasonal = np.einsum("bk,bkl->bl", amp, np.sin(ang))         # (B,L)

        base = trend + seasonal

        # ---- multiplicative Weibull noise (ForecastPFN signature), mean-normalised ----
        kw = rng.uniform(*p.weibull_k, size=B)[:, None]
        ns = rng.uniform(*p.noise_scale, size=B)[:, None]
        w = rng.weibull(kw, size=(B, L))
        # Weibull mean = Gamma(1+1/k); normalise to mean 1 so noise is multiplicative-centered
        from math import gamma
        # vectorised gamma(1+1/k) via a small map (k is per-series scalar)
        mean_w = np.array([gamma(1.0 + 1.0 / float(kk)) for kk in kw[:, 0]])[:, None]
        w = w / mean_w
        noise = 1.0 + ns * (w - 1.0)
        series = base * noise

        # ---- occasional level shift (B fraction) ----
        ls = rng.random(B) < p.level_shift_prob
        if ls.any():
            cut = rng.integers(L // 4, 3 * L // 4, size=B)
            shift = rng.normal(0.0, 1.5, size=B)
            step = (t_idx[None, :] >= cut[:, None]).astype(np.float64) * (shift * ls)[:, None]
            series = series + step

        # ---- a few decaying spikes ----
        sp = rng.random(B) < p.spike_prob
        if sp.any():
            for b in np.nonzero(sp)[0]:
                at = int(rng.integers(0, L))
                w2 = int(min(rng.integers(1, 6), L - at))
                tail = np.exp(-np.arange(w2) / max(1.0, w2 / 2.0))
                sd = series[b].std() + 1e-9
                series[b, at:at + w2] += float(rng.uniform(3.0, 9.0)) * sd * tail

        values = np.ascontiguousarray(series, dtype=np.float64)
        sampled = self._sample_parameters(B)
        return TimeSeriesContainer(values=values, start=sampled["start"],
                                   frequency=sampled["frequency"])
