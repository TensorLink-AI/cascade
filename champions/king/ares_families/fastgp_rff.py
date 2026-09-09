"""fastgp_rff — a GP prior via Random Fourier Features: no Cholesky, fully batched.

The Cholesky-grid fastgp draws at ~500k pts/s (the O(grid^3) factorisation, even at
grid 256, plus the per-Cholesky Python loop). For the mainnet 20x-token-budget
(stream_cpu), that is still a bottleneck. Random Fourier Features approximate a
stationary GP as a finite sum of cosines with random frequencies:

    f(t) = sqrt(2/D) * sum_j a_j cos(2π w_j t + b_j),   w_j ~ spectral density,
                                                        b_j ~ U[0, 2π]

Drawing an RBF-kernel GP this way is O(B * D * L) pure numpy matmul — no Cholesky,
no per-series loop — so it runs at millions of points/s while preserving the
smooth, multi-lengthscale structure the GP/kernel axis contributes. We keep the
compositional flavour (mixture of RBF lengthscales + a periodic component) that
made fastgp win, just via spectral sampling instead of a gram-matrix factorisation.

Deterministic per (seed + series index). numpy only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tempo_gen.data.containers import TimeSeriesContainer
from tempo_gen.synthetic_generation.abstract_classes import GeneratorWrapper
from tempo_gen.synthetic_generation.generator_params import GeneratorParams


@dataclass
class FastGPRFFGeneratorParams(GeneratorParams):
    """RFF GP: RBF lengthscale mixture + optional periodic component."""

    n_features: int = 64                # RFF count D (more = closer to a true GP)
    lengthscale_frac: tuple[float, float] = (0.01, 0.6)   # RBF lengthscale, fraction of series
    n_scales: tuple[int, int] = (1, 4)  # how many RBF lengthscales to superpose (compositional)
    p_periodic: float = 0.4             # fraction that add a periodic (ExpSine-like) component
    period_frac: tuple[float, float] = (0.02, 0.5)


class FastGPRFFGeneratorWrapper(GeneratorWrapper):
    def __init__(self, params: FastGPRFFGeneratorParams):
        super().__init__(params)

    def generate_batch(self, batch_size: int, seed: int | None = None) -> TimeSeriesContainer:
        if seed is None:
            seed = int(self.params.global_seed)
        p = self.params
        L = int(self.params.length)
        B = int(batch_size)
        D = int(p.n_features)
        rng = np.random.default_rng(int(seed) % (2**31))

        x = np.linspace(0.0, 1.0, L)[None, :]                       # (1, L)
        out = np.zeros((B, L), dtype=np.float64)

        # Each series superposes n_scales RBF-GP draws (compositional kernel) via RFF.
        lo, hi = p.n_scales
        n_scales = rng.integers(lo, hi + 1, size=B)                 # (B,)
        max_s = int(hi)
        for s in range(max_s):
            active = (n_scales > s)                                  # (B,) which series use this scale
            if not active.any():
                continue
            # RBF spectral density: w ~ Normal(0, 1/ell). Per-series lengthscale.
            ell = np.exp(rng.uniform(np.log(p.lengthscale_frac[0]),
                                     np.log(p.lengthscale_frac[1]), size=B))[:, None]  # (B,1)
            w = rng.standard_normal((B, D)) / ell                   # (B, D) freqs
            b = rng.uniform(0.0, 2.0 * np.pi, size=(B, D))          # (B, D) phases
            a = rng.standard_normal((B, D))                          # (B, D) weights
            # features: cos(2π w x + b) -> (B, D, L); project with a -> (B, L)
            ang = 2.0 * np.pi * (w[:, :, None] * x[:, None, :]) + b[:, :, None]  # (B,D,L)
            feat = np.cos(ang)
            draw = np.sqrt(2.0 / D) * np.einsum("bd,bdl->bl", a, feat)           # (B,L)
            out += draw * active[:, None]

        # optional periodic component (adds daily/weekly-style seasonality)
        pmask = rng.random(B) < p.p_periodic
        if pmask.any():
            per = rng.uniform(*p.period_frac, size=B)[:, None]
            amp = rng.uniform(0.3, 1.5, size=B)[:, None]
            ph = rng.uniform(0.0, 2.0 * np.pi, size=B)[:, None]
            out += (amp * np.sin(2.0 * np.pi * x / np.maximum(per, 1e-3) + ph)) * pmask[:, None]

        values = np.ascontiguousarray(out, dtype=np.float64)
        sampled = self._sample_parameters(B)
        return TimeSeriesContainer(values=values, start=sampled["start"],
                                   frequency=sampled["frequency"])
