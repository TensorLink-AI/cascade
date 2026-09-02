"""Realistic base family — the shape that actually WINS the field.

Empirical finding (2026-07-12): the external competitor ``plan/cascade-awesome-v1``
beats BOTH operator kings (``iris999/ares-v3`` pure-ForecastPFN and
``cascade/testnet-smoothgp`` GP/kernel-heavy) by ~2x on real-world eval, using a
dead-simple 40-line recipe: linear trend + 1-2 seasonal sinusoids + AR(1) noise +
an occasional level shift. The lesson is not "add exotic priors" — the operators
did that and lost — but "match the real data-generating process". Real series ARE
trend + seasonality + autocorrelated noise + regime shifts; a corpus drawn from
that distribution teaches the Toto2 forecaster to generalise, where ForecastPFN's
Weibull-noise harmonics and GP smears do not.

This family reproduces the awesome-v1 base EXACTLY at its defaults (so a pure
``realistic`` weight is a faithful replica / benchmark), then exposes knobs to
*out-complete* it on the structure it lacks — the reason even awesome-v1 never
clears the paired-LCB wall: it under-models heavy tails (volatility clustering in
energy/finance), richer multi-seasonality, and volatility regime switches. Turn
those on in small, realistic doses and mix with the ``intermittent`` family (the
sparse-event shape awesome-v1 has none of) to build a MORE COMPLETE realistic
distribution than anyone in the field — the actual path to a positive LCB.

Pure numpy, deterministic per ``(seed + series index)`` via ``default_rng``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tempo_gen.data.containers import TimeSeriesContainer
from tempo_gen.synthetic_generation.abstract_classes import GeneratorWrapper
from tempo_gen.synthetic_generation.generator_params import GeneratorParams


@dataclass
class RealisticGeneratorParams(GeneratorParams):
    """awesome-v1 base (defaults) + out-completion knobs.

    Defaults reproduce ``plan/cascade-awesome-v1`` exactly. Overridable via
    ``config.json`` ``family_params.realistic``.
    """

    # --- awesome-v1 base (exact defaults) ---
    n_seasonal_range: tuple[int, int] = (1, 3)      # np.random.integers(1, 3) -> 1 or 2
    seasonal_periods: tuple[float, ...] = (7.0, 12.0, 24.0, 30.0, 52.0)
    trend_slope_std: float = 0.01
    level_std: float = 1.0
    amp_range: tuple[float, float] = (0.2, 2.0)
    ar1_phi_range: tuple[float, float] = (0.0, 0.8)
    ar1_sigma_range: tuple[float, float] = (0.1, 0.5)
    level_shift_prob: float = 0.2
    level_shift_std: float = 2.0

    # --- out-completion knobs (0 => pure awesome-v1) ---
    # Fraction of series whose AR(1) innovations are Student-t (heavy tails /
    # volatility clustering) instead of Gaussian. Real energy/finance data.
    heavytail_prob: float = 0.0
    heavytail_df_range: tuple[float, float] = (3.0, 6.0)
    # Extra probability of a 3rd/4th seasonal component (richer multi-seasonality:
    # intraday x weekly x yearly) beyond awesome-v1's 1-2.
    extra_seasonal_prob: float = 0.0
    extra_seasonal_periods: tuple[float, ...] = (5.0, 96.0, 168.0, 365.0)
    # Fraction of series with a mid-series volatility regime switch (sigma jumps).
    vol_regime_prob: float = 0.0


class RealisticGeneratorWrapper(GeneratorWrapper):
    """Batch generator for the realistic base family (mirrors the vendored API)."""

    def __init__(self, params: RealisticGeneratorParams):
        super().__init__(params)

    def generate_batch(self, batch_size: int, seed: int | None = None) -> TimeSeriesContainer:
        if seed is None:
            seed = int(self.params.global_seed)
        self._set_random_seeds(seed)
        length = int(self.params.length)
        values = np.empty((batch_size, length), dtype=np.float64)
        for i in range(batch_size):
            rng = np.random.default_rng((int(seed) + i) % (2**31))
            values[i] = self._one_series(rng, length)
        sampled = self._sample_parameters(batch_size)
        return TimeSeriesContainer(
            values=values,
            start=sampled["start"],
            frequency=sampled["frequency"],
        )

    # ── model ────────────────────────────────────────────────────────────────

    def _one_series(self, rng: np.random.Generator, length: int) -> np.ndarray:
        p = self.params
        t = np.arange(length, dtype=np.float64)

        # Trend (awesome-v1: level + slope*t).
        slope = rng.normal(0.0, p.trend_slope_std)
        level = rng.normal(0.0, p.level_std)
        series = level + slope * t

        # 1-2 seasonal sinusoids (awesome-v1), + optional richer multi-seasonality.
        lo, hi = p.n_seasonal_range
        n_seas = int(rng.integers(lo, hi))
        periods = list(p.seasonal_periods)
        if p.extra_seasonal_prob > 0.0 and rng.random() < p.extra_seasonal_prob:
            n_seas += int(rng.integers(1, 3))  # 1-2 extra components
            periods = periods + list(p.extra_seasonal_periods)
        for _ in range(max(1, n_seas)):
            period = float(rng.choice(periods))
            amp = rng.uniform(*p.amp_range)
            phase = rng.uniform(0.0, 2.0 * np.pi)
            series = series + amp * np.sin(2.0 * np.pi * t / max(period, 2.0) + phase)

        # AR(1) noise (awesome-v1), with optional heavy-tailed innovations and a
        # mid-series volatility regime switch.
        phi = rng.uniform(*p.ar1_phi_range)
        sigma = rng.uniform(*p.ar1_sigma_range)
        heavy = p.heavytail_prob > 0.0 and rng.random() < p.heavytail_prob
        df = rng.uniform(*p.heavytail_df_range) if heavy else None
        # volatility regime: second-half sigma multiplier
        sig = np.full(length, sigma)
        if p.vol_regime_prob > 0.0 and rng.random() < p.vol_regime_prob:
            cut = int(rng.uniform(0.3, 0.7) * length)
            sig[cut:] *= float(np.exp(rng.uniform(np.log(0.4), np.log(3.0))))

        # Pre-draw the innovation vector (vectorised — no per-step Python call),
        # then run only the cheap AR(1) recursion in the loop.
        if heavy:
            z = rng.standard_t(df, size=length)
            if df > 2.0:
                z = z * np.sqrt((df - 2.0) / df)   # scale to unit variance
        else:
            z = rng.standard_normal(length)
        innov = z * sig
        noise = np.empty(length, dtype=np.float64)
        noise[0] = innov[0]
        for i in range(1, length):
            noise[i] = phi * noise[i - 1] + innov[i]
        series = series + noise

        # Occasional level shift (awesome-v1).
        if rng.random() < p.level_shift_prob:
            shift_at = int(rng.integers(length // 4, 3 * length // 4))
            series[shift_at:] += rng.normal(0.0, p.level_shift_std)

        return series.astype(np.float64)
