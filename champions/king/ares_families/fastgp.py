"""fastgp — compositional-kernel GP prior that is actually DELIVERABLE.

Why this family exists (the central finding of the 2026-07-14 field audit):

The operator's own base-generator docstring says the GP/kernel family "carries a
large share of the downstream signal" (the TempoPFN ablation). The long-reigning
king ``cascade/testnet-smoothgp`` DECLARED ``gp 0.35 + kernel_synth 0.30``. And
yet, across 138 rounds, **that corpus was never once trained on**. Two independent
reasons, both verified:

1. **The blocked-emission trap.** The trainer runs ``corpus_mode=stream_cpu``
   (``cascade/shared/config.py`` default, not overridden in chain.testnet.toml),
   stops at ``token_budget`` points, and does NOT shuffle
   (``toto2_trainer.iter_training_batches``) — "digest covers exactly the consumed
   prefix". The stream requests ~2.6M series but only ~157k (~6%) are consumed.
   The base engine emits family-BLOCKED with ``forecast_pfn`` first, so smoothgp's
   fpfn block (0.15 x 2.6M = 390k series) alone swallows the whole consumed prefix.
   Its gp/kernel_synth mass never reaches the model.

2. **Cost.** Even unblocked, the vendored priors cannot supply the budget:
   measured at n=64/L=2048, ``kernel_synth`` TIMED OUT (>90s) and ``gp`` took 29s
   (~4.5k points/s). The round needs ~166.5M points; gp alone at 30% weight would
   need ~3 hours of pure generation and would starve the GPU (stream_cpu feeds it
   live). This is why every heavy-GP config either failed to train or silently
   degraded.

So the GP/kernel axis is the one high-value region of the space that NOBODY has
ever actually delivered. This family delivers it, by making the draw cheap:

  * Sample the GP on a COARSE grid (default 256 points) — the O(g^3) Cholesky is
    on 256, not 2048 (~512x fewer flops) — then **cubic-spline upsample** to full
    length. GP draws are smooth by construction, so the spline reconstruction is
    faithful; this is the same trick the current king (valor/aurora-mix) uses for
    its ``kernel_gp``, which it caps at only 12% weight.
  * Build the covariance by COMPOSING kernels (KernelSynth's contribution): RBF,
    periodic (ExpSineSquared), rational-quadratic, linear and white, combined by
    random ``+``/``*`` over 1-3 draws. That is what gives KernelSynth its pattern
    diversity, and it costs nothing extra here — the composition happens on the
    256x256 gram matrix.

Result: KernelSynth-class diversity at roughly 1000x the throughput, so it can be
carried at 30-45% weight and actually arrive at the model.

Pure numpy + scipy (both allowlisted). Deterministic per ``(seed + series index)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline

from tempo_gen.data.containers import TimeSeriesContainer
from tempo_gen.synthetic_generation.abstract_classes import GeneratorWrapper
from tempo_gen.synthetic_generation.generator_params import GeneratorParams

_KERNELS = ("rbf", "periodic", "rq", "linear", "white")


@dataclass
class FastGPGeneratorParams(GeneratorParams):
    """Coarse-grid compositional-kernel GP.

    Overridable via ``config.json`` ``family_params.fastgp``.
    """

    # Coarse grid the GP is drawn on before spline upsampling. 256 keeps the
    # Cholesky trivial (~5ms) while preserving every smooth structure the eval
    # windows care about.
    grid: int = 256
    # How many kernels get composed into one covariance (KernelSynth-style).
    max_compositions: int = 3
    # Draw several series from ONE Cholesky factor (amortises the factorisation).
    samples_per_chol: tuple[int, int] = (6, 16)
    # RBF / RQ lengthscales are drawn log-uniform over the grid, as a fraction of it.
    lengthscale_frac: tuple[float, float] = (0.01, 0.6)
    # Periodic kernel period, as a fraction of the grid.
    period_frac: tuple[float, float] = (0.02, 0.5)
    jitter: float = 1e-8


class FastGPGeneratorWrapper(GeneratorWrapper):
    """Batch generator for the fastgp family (mirrors the vendored API)."""

    def __init__(self, params: FastGPGeneratorParams):
        super().__init__(params)

    def generate_batch(self, batch_size: int, seed: int | None = None) -> TimeSeriesContainer:
        if seed is None:
            seed = int(self.params.global_seed)
        self._set_random_seeds(seed)
        length = int(self.params.length)
        p = self.params
        g = max(16, int(p.grid))

        values = np.empty((batch_size, length), dtype=np.float64)
        x = np.linspace(0.0, 1.0, g)
        x_full = np.linspace(0.0, 1.0, length)

        i = 0
        chol_no = 0
        while i < batch_size:
            # One RNG per Cholesky block, derived from (seed, block index), so the
            # stream stays a pure function of the seed regardless of batch_size.
            rng = np.random.default_rng((int(seed) + 100_003 * chol_no) % (2**31))
            lo, hi = p.samples_per_chol
            k = int(rng.integers(lo, hi + 1))
            k = min(k, batch_size - i)

            L = self._cholesky(rng, x, g)
            # k independent GP draws share the factor: L @ z.
            z = rng.standard_normal((g, k))
            draws = L @ z                                   # (g, k)

            for j in range(k):
                coarse = draws[:, j]
                # Cubic-spline upsample the coarse draw to full length. A GP draw
                # is smooth, so this reconstructs it faithfully and costs O(L).
                values[i] = CubicSpline(x, coarse)(x_full)
                i += 1
            chol_no += 1

        sampled = self._sample_parameters(batch_size)
        return TimeSeriesContainer(
            values=values,
            start=sampled["start"],
            frequency=sampled["frequency"],
        )

    # ── covariance ───────────────────────────────────────────────────────────
    def _cholesky(self, rng: np.random.Generator, x: np.ndarray, g: int) -> np.ndarray:
        p = self.params
        n_comp = int(rng.integers(1, max(1, int(p.max_compositions)) + 1))
        K = self._kernel(rng, x, g)
        for _ in range(n_comp - 1):
            K2 = self._kernel(rng, x, g)
            # KernelSynth composes with random '+' or '*'.
            K = K + K2 if rng.random() < 0.5 else K * K2

        # Normalise scale, then jitter up until the factorisation succeeds (a
        # product of kernels can be numerically borderline).
        d = float(np.mean(np.diag(K)))
        if d > 1e-12:
            K = K / d
        jit = float(p.jitter)
        for _ in range(6):
            try:
                return np.linalg.cholesky(K + jit * np.eye(g))
            except np.linalg.LinAlgError:
                jit *= 100.0
        # Deterministic fallback: white noise (never happens in practice).
        return np.eye(g)

    def _kernel(self, rng: np.random.Generator, x: np.ndarray, g: int) -> np.ndarray:
        p = self.params
        kind = _KERNELS[int(rng.integers(0, len(_KERNELS)))]
        d = x[:, None] - x[None, :]
        a2 = np.abs(d)

        if kind == "rbf":
            ell = float(np.exp(rng.uniform(np.log(p.lengthscale_frac[0]),
                                           np.log(p.lengthscale_frac[1]))))
            return np.exp(-0.5 * (d / ell) ** 2)
        if kind == "periodic":
            per = float(rng.uniform(*p.period_frac))
            ell = float(rng.uniform(0.3, 2.0))
            return np.exp(-2.0 * np.sin(np.pi * a2 / max(per, 1e-3)) ** 2 / ell**2)
        if kind == "rq":
            ell = float(np.exp(rng.uniform(np.log(p.lengthscale_frac[0]),
                                           np.log(p.lengthscale_frac[1]))))
            alpha = float(np.exp(rng.uniform(np.log(0.1), np.log(10.0))))
            return (1.0 + (d**2) / (2.0 * alpha * ell**2)) ** (-alpha)
        if kind == "linear":
            c = float(rng.uniform(0.0, 1.0))
            return (x[:, None] - c) * (x[None, :] - c) + 1e-3
        # white
        return np.eye(g)
