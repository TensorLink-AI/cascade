"""Counts family — weekly/daily-seasonal, heavy-tailed, zero-inflated count series.

Models the "downloads / pageviews / event-count" shape that dominates real
web/sales/ops feeds and that smooth GP priors cover poorly: non-negative
integer-ish levels over several orders of magnitude, multiplicative calendar
seasonality (weekday/weekend structure), AR(1) log-noise, self-exciting bursts
with exponential decay, occasional level shifts and saturating adoption trends,
Gamma-Poisson (negative-binomial) observation noise with wide dispersion, and
zero inflation.

Pure numpy; deterministic per (seed + series index) via np.random.default_rng —
no global RNG state, no torch, safe under the cross-process determinism
contract.
"""

from __future__ import annotations

import numpy as np

from tempo_gen.data.containers import TimeSeriesContainer
from tempo_gen.synthetic_generation.abstract_classes import GeneratorWrapper
from tempo_gen.synthetic_generation.generator_params import GeneratorParams

# Intensity cap keeps exp() finite and Poisson sampling safe.
_LOG_LAM_MAX = float(np.log(5e5))

# Candidate seasonal periods: weekly on daily data; daily cycles at common
# sub-daily cadences (hourly, 30/15/5-minute); 168 = weekly cycle on hourly data.
_PERIODS = np.array([7, 24, 48, 96, 168, 288])
_PERIOD_P = np.array([0.35, 0.20, 0.10, 0.10, 0.15, 0.10])


class CountsGeneratorParams(GeneratorParams):
    """Parameters for the counts generator (defaults tuned for 2048-length draws)."""

    # log-space magnitude band of the base level (≈ 5 .. 2e5 counts/step).
    log_base_range: tuple[float, float] = (np.log(5.0), np.log(2e5))
    # Probability a series uses a saturating (logistic) adoption trend.
    p_saturating: float = 0.25
    # Probability of a single level shift.
    p_level_shift: float = 0.30
    # Probability the series is emitted as a continuous rate (no Poisson step).
    p_continuous: float = 0.20
    # Probability zero-inflation is applied at all.
    p_zero_inflate: float = 0.40


class CountsGeneratorWrapper(GeneratorWrapper):
    """Batch generator for the counts family (mirrors the vendored wrapper API)."""

    def __init__(self, params: CountsGeneratorParams):
        super().__init__(params)

    def generate_batch(self, batch_size: int, seed: int | None = None) -> TimeSeriesContainer:
        if seed is None:
            seed = int(self.params.global_seed)
        self._set_random_seeds(seed)  # for base-class metadata sampling below
        length = int(self.params.length)
        values = np.empty((batch_size, length), dtype=np.float64)
        for i in range(batch_size):
            rng = np.random.default_rng((int(seed) + i) % (2**31))
            values[i] = self._one_series(rng, length)
        # start/frequency are presentational metadata; the cascade generator only
        # consumes `.values`. Fixed values keep the container deterministic.
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

        log_base = rng.uniform(*p.log_base_range)

        # Trend: linear drift in log space, or a logistic adoption curve.
        if rng.random() < p.p_saturating:
            k = rng.uniform(1.0, 4.0)
            x0 = rng.uniform(0.2, 0.8) * length
            width = length * rng.uniform(0.03, 0.15)
            trend = k / (1.0 + np.exp(-(t - x0) / width))
            trend -= trend[0]
        else:
            total_drift = rng.normal(0.0, 0.8)
            trend = total_drift * t / length

        # Multiplicative (log-space) calendar seasonality.
        period = int(rng.choice(_PERIODS, p=_PERIOD_P))
        if period == 7:
            # Explicit weekday pattern with a weekend dip.
            weekday = rng.normal(0.0, 0.35, 7)
            if rng.random() < 0.6:
                weekday[5:] -= rng.uniform(0.2, 1.2)
            weekday -= weekday.mean()
            seas = weekday[(t.astype(np.int64)) % 7]
        else:
            n_harm = int(rng.integers(1, 4))
            seas = np.zeros(length)
            for h in range(1, n_harm + 1):
                amp = rng.exponential(0.25) / h
                phase = rng.uniform(0.0, 2.0 * np.pi)
                seas += amp * np.cos(2.0 * np.pi * h * t / period + phase)

        # AR(1) noise in log space.
        phi = rng.uniform(0.5, 0.98)
        sig = rng.uniform(0.05, 0.45)
        e = (rng.normal(0.0, sig, length)).tolist()
        ar = np.empty(length)
        a = 0.0
        for j in range(length):
            a = phi * a + e[j]
            ar[j] = a

        # Self-exciting bursts: exponential-decay kernels at Poisson times.
        bursts = np.zeros(length)
        for _ in range(rng.poisson(1.5)):
            pos = int(rng.integers(0, length))
            amp = rng.exponential(1.2) + 0.3
            half_life = rng.uniform(1.0, 3.0 * period)
            decay = np.exp(-np.log(2.0) * np.arange(length - pos) / half_life)
            bursts[pos:] += amp * decay

        # Occasional persistent level shift.
        shift = np.zeros(length)
        if rng.random() < p.p_level_shift:
            pos = int(rng.integers(length // 8, length))
            shift[pos:] = rng.choice([-1.0, 1.0]) * rng.uniform(0.3, 1.5)

        log_lam = np.minimum(log_base + trend + seas + ar + bursts + shift, _LOG_LAM_MAX)
        lam = np.exp(log_lam)

        # Gamma-Poisson mixture => negative-binomial marginals (heavy tail).
        r = float(np.exp(rng.uniform(np.log(0.6), np.log(60.0))))
        mix = rng.gamma(shape=r, scale=lam / r)
        if rng.random() < p.p_continuous:
            y = mix  # continuous rate series (e.g. request rates, tvl)
        else:
            y = rng.poisson(np.minimum(mix, 4e5)).astype(np.float64)

        # Zero inflation (structural missing/quiet steps).
        if rng.random() < p.p_zero_inflate:
            p0 = rng.uniform(0.0, 0.25)
            y = np.where(rng.random(length) < p0, 0.0, y)

        return y
