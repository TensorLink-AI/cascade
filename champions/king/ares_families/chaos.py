"""Chaos family — parameterised deterministic chaos, not a fixed handful of maps.

The literature (Panda 2025; ChaosNexus 2025-26) shows synthetic chaotic ODE/map
data transfers zero-shot to *real* forecasting, but only when the training set
spans a *diverse, parameterised family* of chaotic systems — a fixed three-system
menu (logistic/Hénon/Lorenz) both under-covers the attractor space and lets the
model memorise. So each series here samples a system AND its bifurcation
parameters from continuous bands chosen to stay in (or near) the chaotic regime,
then keeps only trajectories that actually look chaotic (a lightweight spread /
non-degeneracy screen — constant or divergent runs are rejected and redrawn).

Systems: logistic map, Hénon map, the Lorenz-63 x-channel, the Rössler
x-channel, and a driven/damped duffing oscillator. Continuous ODEs are
integrated with a small fixed RK4 step and subsampled, so the emitted cadence is
comparable to the discrete maps. Output is z-scored per series (chaotic scales
are arbitrary; the shape is the signal). Pure numpy, deterministic per
``(seed + series index)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tempo_gen.data.containers import TimeSeriesContainer
from tempo_gen.synthetic_generation.abstract_classes import GeneratorWrapper
from tempo_gen.synthetic_generation.generator_params import GeneratorParams

_SYSTEMS = ("logistic", "henon", "lorenz", "rossler", "duffing")


@dataclass
class ChaosGeneratorParams(GeneratorParams):
    """Parameterised-chaos parameters (overridable via family_params.chaos)."""

    # System sampling weights (logistic, henon, lorenz, rossler, duffing).
    system_weights: tuple[float, float, float, float, float] = (0.25, 0.2, 0.2, 0.2, 0.15)
    # Burn-in steps discarded so the trajectory settles onto the attractor.
    burn_in: int = 200
    # Observation-noise sigma (fraction of series std) — small, keeps it chaotic.
    obs_noise: float = 0.02


class ChaosGeneratorWrapper(GeneratorWrapper):
    """Batch generator for the parameterised-chaos family."""

    def __init__(self, params: ChaosGeneratorParams):
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
        # Up to a few redraws if a trajectory degenerates (constant / non-finite).
        for _ in range(6):
            system = str(rng.choice(_SYSTEMS, p=_norm(p.system_weights)))
            s = self._trajectory(rng, system, length, int(p.burn_in))
            if s is not None and np.isfinite(s).all():
                sd = s.std()
                if sd > 1e-6:
                    s = (s - s.mean()) / sd
                    if float(p.obs_noise) > 0.0:
                        s = s + rng.normal(0.0, float(p.obs_noise), length)
                    return s
        # Degenerate fallback: a benign noisy sine (should be very rare).
        t = np.arange(length, dtype=np.float64)
        return np.sin(0.1 * t) + rng.normal(0.0, 0.1, length)

    def _trajectory(self, rng, system, length, burn_in):
        if system == "logistic":
            a = rng.uniform(3.7, 3.999)
            x = rng.uniform(0.1, 0.9)
            out = np.empty(length + burn_in)
            for i in range(length + burn_in):
                x = a * x * (1.0 - x)
                out[i] = x
            return out[burn_in:]

        if system == "henon":
            a, b = rng.uniform(1.2, 1.4), rng.uniform(0.2, 0.3)
            x, y = rng.uniform(-0.5, 0.5), rng.uniform(-0.2, 0.2)
            out = np.empty(length + burn_in)
            for i in range(length + burn_in):
                x, y = 1.0 - a * x * x + y, b * x
                out[i] = x
            return out[burn_in:]

        if system == "lorenz":
            sig = rng.uniform(9.0, 11.0)
            rho = rng.uniform(24.0, 32.0)
            beta = rng.uniform(2.4, 3.0)
            x, y, z = rng.uniform(-6, 6, 3)
            dt, sub = 0.01, int(rng.integers(2, 6))
            return self._ode(length, burn_in, sub, dt, (x, y, z),
                             lambda s: (sig * (s[1] - s[0]),
                                        s[0] * (rho - s[2]) - s[1],
                                        s[0] * s[1] - beta * s[2]), channel=0)

        if system == "rossler":
            a = rng.uniform(0.1, 0.3)
            b = rng.uniform(0.1, 0.3)
            c = rng.uniform(4.0, 9.0)
            x, y, z = rng.uniform(-4, 4, 3)
            dt, sub = 0.05, int(rng.integers(2, 6))
            return self._ode(length, burn_in, sub, dt, (x, y, z),
                             lambda s: (-(s[1] + s[2]),
                                        s[0] + a * s[1],
                                        b + s[2] * (s[0] - c)), channel=0)

        # duffing: x'' + delta x' - x + x^3 = gamma cos(omega t)
        delta = rng.uniform(0.1, 0.3)
        gamma = rng.uniform(0.2, 0.6)
        omega = rng.uniform(1.0, 1.4)
        x, v = rng.uniform(-1, 1), rng.uniform(-1, 1)
        dt, sub = 0.05, int(rng.integers(2, 6))
        state = np.array([x, v, 0.0])  # 3rd comp carries time for the drive

        def duff(s):
            xx, vv, tt = s
            return (vv, -delta * vv + xx - xx ** 3 + gamma * np.cos(omega * tt), 1.0)

        return self._ode(length, burn_in, sub, dt, tuple(state), duff, channel=0)

    @staticmethod
    def _ode(length, burn_in, sub, dt, init, deriv, channel):
        """RK4-integrate an ODE, subsample every ``sub`` steps, return one channel."""
        s = np.array(init, dtype=np.float64)
        total = length + burn_in
        out = np.empty(total)
        for i in range(total):
            for _ in range(sub):
                k1 = np.array(deriv(s))
                k2 = np.array(deriv(s + 0.5 * dt * k1))
                k3 = np.array(deriv(s + 0.5 * dt * k2))
                k4 = np.array(deriv(s + dt * k3))
                s = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
                if not np.isfinite(s).all():
                    return None
            out[i] = s[channel]
        return out[burn_in:]


def _norm(w) -> np.ndarray:
    a = np.asarray(w, dtype=np.float64)
    a = np.clip(a, 0.0, None)
    ssum = a.sum()
    return a / ssum if ssum > 0 else np.full(len(a), 1.0 / len(a))
