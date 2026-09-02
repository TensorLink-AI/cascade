"""garch — stochastic volatility / volatility clustering.

Breadth family. Mirrors the operator's own reference generator
``cascade/testnet-garch`` ("stochastic-volatility-v1": GARCH(1,1) log-levels with
Student-t shocks, Hawkes-like self-exciting bursts, OU levels with slow log-AR(1)
volatility regimes) — which reached the heat finalist slot 10 times but, as a
single-concept 100% generator, never dethroned. Carried here as a THIN component
of a broad mixture instead.

Teaches the forecaster that variance is not constant: calm and turbulent stretches
alternate. That is what widens the predictive quantiles where a fixed-variance
prior is overconfident, and CRPS pays for it.

Pure numpy. Deterministic per ``(seed + series index)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tempo_gen.data.containers import TimeSeriesContainer
from tempo_gen.synthetic_generation.abstract_classes import GeneratorWrapper
from tempo_gen.synthetic_generation.generator_params import GeneratorParams


@dataclass
class GarchGeneratorParams(GeneratorParams):
    """GARCH(1,1) + Student-t shocks + slow volatility regimes."""

    omega_range: tuple[float, float] = (1e-4, 1e-2)
    alpha_range: tuple[float, float] = (0.02, 0.20)   # shock persistence
    beta_range: tuple[float, float] = (0.70, 0.95)    # vol persistence
    df_range: tuple[float, float] = (3.0, 8.0)        # Student-t tails
    # Probability the series carries a slow log-AR(1) volatility regime on top.
    vol_regime_prob: float = 0.5
    # Probability of Hawkes-like self-exciting burst injection.
    burst_prob: float = 0.3
    # Probability the output is a cumulated (integrated) level rather than returns.
    integrate_prob: float = 0.6


class GarchGeneratorWrapper(GeneratorWrapper):
    def __init__(self, params: GarchGeneratorParams):
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
        omega = float(rng.uniform(*p.omega_range))
        alpha = float(rng.uniform(*p.alpha_range))
        beta = float(rng.uniform(*p.beta_range))
        if alpha + beta >= 0.995:                    # keep it stationary
            beta = 0.995 - alpha
        df = float(rng.uniform(*p.df_range))

        # Pre-draw unit-variance Student-t innovations (vectorised).
        z = rng.standard_t(df, size=length)
        if df > 2.0:
            z = z * np.sqrt((df - 2.0) / df)

        # Slow exogenous log-vol regime (OU on log sigma).
        if rng.random() < p.vol_regime_prob:
            phi = float(rng.uniform(0.990, 0.9995))
            s = float(rng.uniform(0.01, 0.05))
            e = rng.standard_normal(length) * s
            logv = np.empty(length)
            logv[0] = e[0]
            for t in range(1, length):
                logv[t] = phi * logv[t - 1] + e[t]
            regime = np.exp(logv - logv.mean())
        else:
            regime = np.ones(length)

        # GARCH(1,1) recursion.
        r = np.empty(length, dtype=np.float64)
        sig2 = omega / max(1e-9, 1.0 - alpha - beta)
        for t in range(length):
            sd = np.sqrt(max(sig2, 1e-12)) * regime[t]
            r[t] = sd * z[t]
            sig2 = omega + alpha * (r[t] / max(regime[t], 1e-9)) ** 2 + beta * sig2

        # Hawkes-like self-exciting bursts: a few clustered excursions.
        if rng.random() < p.burst_prob:
            n_b = int(rng.integers(1, 5))
            for _ in range(n_b):
                at = int(rng.integers(0, length))
                w = int(min(rng.integers(5, 60), length - at))
                decay = np.exp(-np.arange(w) / max(1.0, w / 3.0))
                r[at:at + w] += rng.normal(0.0, 1.0, w) * decay * float(rng.uniform(2.0, 6.0)) * (np.std(r) + 1e-9)

        if rng.random() < p.integrate_prob:
            level = float(rng.normal(0.0, 1.0))
            return (level + np.cumsum(r)).astype(np.float64)
        return r.astype(np.float64)
