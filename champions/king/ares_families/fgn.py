"""fgn — long-memory fractional Gaussian noise / fractional Brownian motion.

Breadth family. Nobody in the field emits long-memory structure except the current
king (valor/aurora-mix, ``fgn`` at 5%). Real hydrology (streamflow), network
traffic and climate series are famously long-memory (Hurst H > 0.5): their
autocorrelation decays as a power law, not exponentially. Every prior in the
operator's base engine — AR(1)/OU (exponential decay), GP (smooth, short
lengthscale), ForecastPFN (harmonics + iid noise) — is short-memory. A model
trained only on those will systematically under-persist on the eval battery's
streamflow / traffic / climate windows.

Drawn by spectral synthesis: build the fGn power spectrum S(f) ~ f^(1-2H), give it
random phases, and inverse-FFT. That is O(L log L) — essentially free — so this
long-memory axis costs nothing to carry.

Pure numpy. Deterministic per ``(seed + series index)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tempo_gen.data.containers import TimeSeriesContainer
from tempo_gen.synthetic_generation.abstract_classes import GeneratorWrapper
from tempo_gen.synthetic_generation.generator_params import GeneratorParams


@dataclass
class FGNGeneratorParams(GeneratorParams):
    """Fractional Gaussian noise / fBm by spectral synthesis."""

    # Hurst exponent. >0.5 = persistent (long memory), <0.5 = anti-persistent.
    hurst_range: tuple[float, float] = (0.55, 0.95)
    # Probability of cumulating to fBm (a long-memory *level*, not increments).
    integrate_prob: float = 0.5
    # Probability of adding a seasonal component on top of the long-memory base.
    seasonal_prob: float = 0.35
    seasonal_periods: tuple[float, ...] = (24.0, 168.0, 365.0)
    amp_range: tuple[float, float] = (0.2, 1.5)


class FGNGeneratorWrapper(GeneratorWrapper):
    def __init__(self, params: FGNGeneratorParams):
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
        H = float(rng.uniform(*p.hurst_range))

        # Spectral synthesis on a padded grid (avoid FFT wrap-around correlation).
        n = 1
        while n < 2 * length:
            n *= 2
        f = np.fft.rfftfreq(n)
        f[0] = f[1] if len(f) > 1 else 1.0
        # fGn power spectrum: S(f) ~ f^(1 - 2H).
        amp = f ** (-(2.0 * H - 1.0) / 2.0)
        amp[0] = 0.0                                   # kill the DC term
        phase = rng.uniform(0.0, 2.0 * np.pi, size=f.shape)
        spec = amp * np.exp(1j * phase)
        x = np.fft.irfft(spec, n=n)[:length]

        sd = float(x.std())
        x = x / sd if sd > 1e-12 else rng.standard_normal(length)

        if rng.random() < p.integrate_prob:
            x = np.cumsum(x)
            sd = float(x.std())
            if sd > 1e-12:
                x = x / sd

        if rng.random() < p.seasonal_prob:
            t = np.arange(length, dtype=np.float64)
            period = float(rng.choice(list(p.seasonal_periods)))
            amp_s = float(rng.uniform(*p.amp_range))
            phase_s = float(rng.uniform(0.0, 2.0 * np.pi))
            x = x + amp_s * np.sin(2.0 * np.pi * t / max(period, 2.0) + phase_s)

        return x.astype(np.float64)
