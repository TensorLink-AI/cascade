"""regime — piecewise-stationary changepoints.

Breadth family. Mirrors the operator's reference ``cascade/testnet-changepoint``
("changepoint-regimes-v1": random changepoints redraw level/trend/seasonality/
noise; variance regimes and transient anomaly bursts) — a 2x heat finalist that,
alone, never dethroned.

The point it teaches: real series do NOT stay stationary across a 4096-step
context. Levels step, trends flip sign, seasonal amplitude collapses, variance
switches. A prior drawn only from globally-stationary processes teaches the model
to extrapolate a regime that has already ended.

Pure numpy. Deterministic per ``(seed + series index)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tempo_gen.data.containers import TimeSeriesContainer
from tempo_gen.synthetic_generation.abstract_classes import GeneratorWrapper
from tempo_gen.synthetic_generation.generator_params import GeneratorParams


@dataclass
class RegimeGeneratorParams(GeneratorParams):
    """Piecewise-stationary segments with independently redrawn dynamics."""

    n_segments_range: tuple[int, int] = (2, 7)
    seasonal_periods: tuple[float, ...] = (7.0, 12.0, 24.0, 30.0, 52.0, 96.0, 168.0)
    seasonal_prob: float = 0.7
    amp_range: tuple[float, float] = (0.2, 2.5)
    slope_std: float = 0.02
    level_jump_std: float = 2.0
    ar_phi_range: tuple[float, float] = (0.0, 0.9)
    sigma_range: tuple[float, float] = (0.05, 0.8)
    # Transient anomaly burst inside a segment.
    anomaly_prob: float = 0.25


class RegimeGeneratorWrapper(GeneratorWrapper):
    def __init__(self, params: RegimeGeneratorParams):
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
        return TimeSeriesContainer(values=values, start=sampled["start"],
                                   frequency=sampled["frequency"])

    def _one_series(self, rng: np.random.Generator, length: int) -> np.ndarray:
        p = self.params
        lo, hi = p.n_segments_range
        n_seg = int(rng.integers(lo, hi + 1))

        # Random changepoints -> segment boundaries.
        cuts = np.sort(rng.choice(np.arange(1, length), size=min(n_seg - 1, max(1, length - 2)),
                                  replace=False)) if n_seg > 1 else np.array([], dtype=int)
        bounds = np.concatenate([[0], cuts, [length]]).astype(int)

        out = np.empty(length, dtype=np.float64)
        level = float(rng.normal(0.0, 1.0))

        for s in range(len(bounds) - 1):
            a, b = int(bounds[s]), int(bounds[s + 1])
            n = b - a
            if n <= 0:
                continue
            t = np.arange(n, dtype=np.float64)

            # Each segment redraws its own dynamics.
            slope = float(rng.normal(0.0, p.slope_std))
            seg = level + slope * t

            if rng.random() < p.seasonal_prob:
                period = float(rng.choice(list(p.seasonal_periods)))
                amp = float(rng.uniform(*p.amp_range))
                phase = float(rng.uniform(0.0, 2.0 * np.pi))
                seg = seg + amp * np.sin(2.0 * np.pi * t / max(period, 2.0) + phase)

            phi = float(rng.uniform(*p.ar_phi_range))
            sigma = float(rng.uniform(*p.sigma_range))
            innov = rng.standard_normal(n) * sigma
            noise = np.empty(n, dtype=np.float64)
            noise[0] = innov[0]
            for k in range(1, n):
                noise[k] = phi * noise[k - 1] + innov[k]
            seg = seg + noise

            if rng.random() < p.anomaly_prob and n > 8:
                at = int(rng.integers(0, n))
                w = int(min(rng.integers(2, 20), n - at))
                seg[at:at + w] += rng.normal(0.0, 4.0 * sigma + 1e-9, w)

            out[a:b] = seg
            # Carry the level forward, plus a jump at the changepoint.
            level = float(seg[-1]) + float(rng.normal(0.0, p.level_jump_std))

        return out
