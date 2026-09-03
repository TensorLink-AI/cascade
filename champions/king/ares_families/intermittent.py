"""Intermittent-demand family — the renewal-process shape nobody in the field has.

This is NOT the ``counts`` family. ``counts`` is a *busy* seasonal count series
with a light zero-inflation overlay; its base level is almost always positive.
An **intermittent** series is mostly *exact zeros* with demand arriving only
occasionally, the canonical spare-parts / slow-moving-SKU / rare-event shape
(Croston 1972; Syntetos–Boylan 2005; Türkmen et al. 2019). Smooth GP / kernel
priors — which is what every competitor (king included) leans on — model this
shape very poorly: they cannot represent long flat-zero runs punctuated by
isolated spikes, and they smear probability mass across the zeros.

The generative model follows Türkmen 2019: a **renewal process** decides *when*
demand occurs (inter-arrival times drawn from a dispersion-controlled law — the
memoryless i.i.d.-geometric shortcut is deliberately avoided so lumpiness is a
free parameter), and a **separate** size law decides *how much* demand occurs
when it does. Each series is placed in one Syntetos–Boylan quadrant by its
average demand interval (ADI) and size dispersion (CV²):

    smooth       short interval, steady size
    intermittent long interval,  steady size
    erratic      short interval, variable size
    lumpy        long interval,  variable size   (hardest; GP-priors fail worst)

Optional realism overlays: a slowly time-varying arrival intensity (promotions /
seasonality in *occurrence*, not just magnitude), and end-of-life obsolescence
(demand stops partway) or ramp-up (demand begins partway). Everything is pure
numpy and deterministic per ``(seed + series index)`` via ``default_rng`` — no
global RNG, no torch — so it satisfies the cross-process determinism contract.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tempo_gen.data.containers import TimeSeriesContainer
from tempo_gen.synthetic_generation.abstract_classes import GeneratorWrapper
from tempo_gen.synthetic_generation.generator_params import GeneratorParams

# Syntetos–Boylan quadrant labels (weights live in the params dataclass).
_QUADRANTS = ("smooth", "intermittent", "erratic", "lumpy")


@dataclass
class IntermittentGeneratorParams(GeneratorParams):
    """Renewal-process intermittent-demand parameters.

    All knobs are overridable via ``config.json`` ``family_params.intermittent``
    (the params class is a real dataclass so the trainer's field filter keeps
    them). ``sparsity`` scales the *baseline* average demand interval, the single
    most useful dial for the sweep: higher => more zeros between demands.
    """

    # Baseline mean inter-arrival multiplier. 1.0 => ADI band below as-is; >1
    # stretches every gap (sparser); <1 packs demands closer.
    sparsity: float = 1.0
    # Quadrant sampling weights (smooth, intermittent, erratic, lumpy). Lumpy /
    # intermittent dominate — those are the shapes the rest of the field cannot make.
    quadrant_weights: tuple[float, float, float, float] = (0.12, 0.33, 0.20, 0.35)
    # Probability the arrival intensity is modulated by a slow seasonal/trend rate.
    p_time_varying_rate: float = 0.45
    # Probability of an obsolescence (end-of-life) or ramp-up truncation.
    p_lifecycle: float = 0.30
    # Probability demand sizes are integer counts (else continuous magnitudes).
    p_integer: float = 0.6


class IntermittentGeneratorWrapper(GeneratorWrapper):
    """Batch generator for the intermittent family (mirrors the vendored API)."""

    def __init__(self, params: IntermittentGeneratorParams):
        super().__init__(params)

    def generate_batch(self, batch_size: int, seed: int | None = None) -> TimeSeriesContainer:
        if seed is None:
            seed = int(self.params.global_seed)
        self._set_random_seeds(seed)  # base-class metadata sampling below
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
        q = str(rng.choice(_QUADRANTS, p=_norm(p.quadrant_weights)))

        # ADI (average demand interval) band and size dispersion by quadrant.
        # Short interval ≈ 1.3–3, long interval ≈ 5–40; steady CV² small, variable large.
        if q == "smooth":
            adi, cv2 = rng.uniform(1.2, 2.5), rng.uniform(0.05, 0.25)
        elif q == "intermittent":
            adi, cv2 = rng.uniform(4.0, 22.0), rng.uniform(0.05, 0.3)
        elif q == "erratic":
            adi, cv2 = rng.uniform(1.3, 3.5), rng.uniform(0.6, 2.5)
        else:  # lumpy
            adi, cv2 = rng.uniform(6.0, 40.0), rng.uniform(0.6, 3.0)
        adi = max(1.05, adi * float(p.sparsity))

        # Optional slow time-varying arrival intensity: a demand at step t is
        # thinned/boosted by a smooth positive multiplier (promotions, season).
        if rng.random() < p.p_time_varying_rate:
            rate_mult = self._slow_rate(rng, length)
        else:
            rate_mult = np.ones(length)

        # Renewal process for arrival times. Inter-arrival gaps ~ Gamma with mean
        # (adi-1) and a shape that sets dispersion: shape>1 => more regular (under-
        # dispersed) intervals, shape<1 => bursty (over-dispersed) => lumpiness.
        # This is the deliberate departure from memoryless geometric arrivals.
        gap_shape = float(np.exp(rng.uniform(np.log(0.4), np.log(3.0))))
        gap_scale = max(1e-3, (adi - 1.0) / gap_shape)

        # Size law: lognormal (continuous) whose sigma is set from the target CV².
        # For a lognormal, CV² = exp(sigma²) - 1  =>  sigma = sqrt(ln(1+CV²)).
        size_sigma = float(np.sqrt(np.log1p(cv2)))
        log_mu = rng.uniform(np.log(1.0), np.log(500.0))  # median demand size

        # Lifecycle truncation window [t0, t1): demand only occurs inside it.
        t0, t1 = 0, length
        if rng.random() < p.p_lifecycle:
            if rng.random() < 0.5:  # obsolescence: dies partway
                t1 = int(rng.uniform(0.3, 0.9) * length)
            else:                    # ramp-up: born partway
                t0 = int(rng.uniform(0.1, 0.6) * length)

        x = np.zeros(length, dtype=np.float64)
        t = t0 + int(rng.exponential(max(adi - 1.0, 0.5)))  # random phase to first demand
        while t < t1:
            # Thinning by the (normalised) slow rate: accept this demand with
            # probability rate_mult[t]; if rejected, roll forward one base gap.
            if rng.random() < min(1.0, float(rate_mult[t])):
                x[t] = float(rng.lognormal(log_mu, size_sigma))
            gap = float(rng.gamma(gap_shape, gap_scale)) + 1.0
            t += int(np.ceil(gap))

        # Optional integer counts (spare parts, unit sales) vs continuous rates.
        if rng.random() < p.p_integer:
            nz = x > 0
            x[nz] = np.maximum(1.0, np.round(x[nz]))

        return x

    # ── helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _slow_rate(rng: np.random.Generator, length: int) -> np.ndarray:
        """A smooth positive multiplier in ~[0.02, 1.5] for arrival thinning."""
        t = np.arange(length, dtype=np.float64)
        base = rng.uniform(0.4, 0.9)
        r = np.full(length, base)
        for _ in range(int(rng.integers(1, 3))):
            period = float(rng.choice([7, 12, 24, 30, 90, 180]))
            amp = rng.uniform(0.1, 0.4)
            phase = rng.uniform(0.0, 2.0 * np.pi)
            r = r + amp * np.sin(2.0 * np.pi * t / max(period, 2.0) + phase)
        r = r + rng.normal(0.0, 0.3) * t / length  # slow adoption / decline
        return np.clip(r, 0.02, 1.5)


def _norm(w) -> np.ndarray:
    a = np.asarray(w, dtype=np.float64)
    a = np.clip(a, 0.0, None)
    s = a.sum()
    return a / s if s > 0 else np.full(len(a), 1.0 / len(a))
