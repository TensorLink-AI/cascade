# ============================================================================
# ENSEMBLE PFN SYNTHETIC GENERATOR  ·  v10
# TempoPFN-style prior ensemble (Moroshan et al. 2025), honestly labeled:
# these are OUR vectorized implementations of the same prior FAMILIES
# (KernelSynth / GP / SDE / sawtooth / step / spikes / sine / ETS / fractal),
# not the official automl/TempoPFN code. GP-family priors use a spectral
# (random-Fourier-feature) approximation instead of O(n^3) Cholesky so that
# 16k-step series are cheap. Swap in the official TempoPFN package via
# set_official_sampler() if you install it.
# All generators: (B, L, rng) -> float32 [B, L], vectorized over B.
# ============================================================================
"""Synthetic prior ensemble. Pure numpy — the corpus builder is CPU-only."""
from __future__ import annotations

import numpy as np


def _envelope(B, L, rng, smooth=None):
    """Smooth positive amplitude envelope via low-freq cosines."""
    t = np.linspace(0, 1, L)[None, :]
    f = rng.uniform(0.5, 3.0, (B, 1))
    ph = rng.uniform(0, 2 * np.pi, (B, 1))
    e = 1.0 + rng.uniform(0.1, 0.8, (B, 1)) * np.cos(2 * np.pi * f * t + ph)
    return np.clip(e, 0.05, None)


def gen_kernelsynth_spectral(B, L, rng):
    """KernelSynth-family (Chronos): composite kernels ~ sum of periodic
    (ExpSineSquared -> harmonic stacks), stationary (RBF/RQ -> RFF), noise.
    Spectral approximation: exact for kernel ADDITION; kernel products are
    approximated by amplitude-modulating periodic parts with a smooth envelope."""
    t = np.arange(L, dtype=np.float64)[None, :]
    x = np.zeros((B, L))
    # periodic components: 1-3 base periods, few harmonics each
    for _ in range(int(rng.integers(1, 4))):
        p = np.exp(rng.uniform(np.log(6), np.log(L))) * np.ones((B, 1))
        p *= np.exp(rng.uniform(-0.1, 0.1, (B, 1)))          # jitter per series
        comp = np.zeros((B, L))
        for h in range(1, 4):
            a = rng.uniform(0, 1, (B, 1)) / h
            ph = rng.uniform(0, 2 * np.pi, (B, 1))
            comp += a * np.sin(2 * np.pi * h * t / p + ph)
        if rng.random() < 0.5:                                # ~ periodic*RBF product
            comp *= _envelope(B, L, rng)
        x += rng.uniform(0.3, 1.5, (B, 1)) * comp
    # stationary RBF/RQ via random Fourier features
    if rng.random() < 0.8:
        F = 64
        ls = np.exp(rng.uniform(np.log(L / 64), np.log(L / 2), (B, 1, 1)))
        w = rng.standard_normal((B, F, 1)) / ls               # RBF spectral: N(0, 1/ls^2)
        if rng.random() < 0.4:                                # RQ = gamma mixture of RBF
            g = rng.gamma(rng.uniform(1, 5), 1.0, (B, F, 1))
            w = w * np.sqrt(g / np.maximum(g.mean(axis=1, keepdims=True), 1e-9))
        ph = rng.uniform(0, 2 * np.pi, (B, F, 1))
        a = rng.standard_normal((B, F, 1)) * np.sqrt(2.0 / F)
        x += rng.uniform(0.3, 1.2, (B, 1)) * (a * np.cos(w * t[:, None, :] + ph)).sum(1)
    x += rng.uniform(0.0, 0.15, (B, 1)) * rng.standard_normal((B, L))   # WhiteKernel
    return x.astype(np.float32)


def gen_gp_spectral(B, L, rng):
    """GP-family (Mamba4Cast-style): up to 6 components incl. Matern (spectral
    density ~ Student-t -> sample freqs from t-dist), linear, poly-trend."""
    t = np.arange(L, dtype=np.float64)[None, :]
    x = np.zeros((B, L))
    for _ in range(int(rng.integers(2, 7))):
        kind = rng.choice(["matern", "rbf", "periodic", "linear", "poly"])
        if kind in ("matern", "rbf"):
            F = 48
            ls = np.exp(rng.uniform(np.log(L / 128), np.log(L), (B, 1, 1)))
            if kind == "matern":
                nu = rng.choice([1.5, 2.5])
                w = rng.standard_t(2 * nu, (B, F, 1)) / ls
            else:
                w = rng.standard_normal((B, F, 1)) / ls
            ph = rng.uniform(0, 2 * np.pi, (B, F, 1))
            a = rng.standard_normal((B, F, 1)) * np.sqrt(2.0 / F)
            x += rng.uniform(0.2, 1.0, (B, 1)) * (a * np.cos(w * t[:, None, :] + ph)).sum(1)
        elif kind == "periodic":
            p = np.exp(rng.uniform(np.log(8), np.log(L / 2))) * np.exp(rng.uniform(-0.1, 0.1, (B, 1)))
            x += rng.uniform(0.2, 1.2, (B, 1)) * np.sin(2 * np.pi * t / p + rng.uniform(0, 2 * np.pi, (B, 1)))
        elif kind == "linear":
            x += rng.normal(0, 1.0, (B, 1)) * (t / L - rng.uniform(0, 1, (B, 1)))
        else:
            c = rng.normal(0, 0.5, (B, 3))
            u = t / L
            x += c[:, :1] * u + c[:, 1:2] * u**2 + c[:, 2:3] * u**3
    return x.astype(np.float32)


def gen_sde_ou(B, L, rng):
    """Novel-prior family: regime-switching Ornstein-Uhlenbeck (vectorized Euler).
    theta/mu/sigma jump at 0-5 changepoints per series."""
    n_reg = 1 + rng.integers(0, 6, B)
    theta = np.exp(rng.uniform(np.log(1e-3), np.log(0.3), (B, 6)))
    mu = rng.normal(0, 1.5, (B, 6))
    sig = np.exp(rng.uniform(np.log(0.02), np.log(0.6), (B, 6)))
    # regime index per timestep
    reg = np.zeros((B, L), dtype=np.int64)
    for b in range(B):                                # cheap loop: only builds indices
        cps = np.sort(rng.choice(np.arange(1, L), size=n_reg[b] - 1, replace=False)) if n_reg[b] > 1 else []
        prev, r = 0, 0
        for cp in list(cps) + [L]:
            reg[b, prev:cp] = r; prev, r = cp, r + 1
    th = np.take_along_axis(theta, reg, 1); m = np.take_along_axis(mu, reg, 1)
    s = np.take_along_axis(sig, reg, 1)
    e = rng.standard_normal((B, L))
    x = np.zeros((B, L)); prev = rng.normal(0, 1, B)
    for i in range(L):                                # O(L) scan, vectorized over B
        prev = prev + th[:, i] * (m[:, i] - prev) + s[:, i] * e[:, i]
        x[:, i] = prev
    return x.astype(np.float32)


def gen_sawtooth(B, L, rng):
    t = np.arange(L)[None, :]
    p = np.exp(rng.uniform(np.log(8), np.log(L / 2), (B, 1)))
    ph = rng.uniform(0, 1, (B, 1))
    frac = ((t / p + ph) % 1.0)
    skew = rng.uniform(0.05, 0.95, (B, 1))            # temporal asymmetry
    x = np.where(frac < skew, frac / skew, (1 - frac) / (1 - skew)) * rng.uniform(0.5, 2, (B, 1))
    return (x - x.mean(1, keepdims=True)).astype(np.float32)


def gen_step(B, L, rng):
    x = np.zeros((B, L))
    for b in range(B):
        n = rng.integers(1, 8)
        cps = np.sort(rng.choice(np.arange(1, L), n, replace=False))
        lv, prev = 0.0, 0
        for cp in list(cps) + [L]:
            x[b, prev:cp] = lv; lv += rng.normal(0, 1.0); prev = cp
    return x.astype(np.float32)


def gen_spikes(B, L, rng):
    base = 0.05 * rng.standard_normal((B, L))
    m = rng.random((B, L)) < rng.uniform(0.005, 0.05, (B, 1))
    sgn = np.where(rng.random((B, 1)) < 0.8, 1.0, -1.0)   # mostly upper-tail (latency-like)
    return (base + m * sgn * rng.exponential(rng.uniform(0.5, 3, (B, 1)), (B, L))).astype(np.float32)


def gen_sine(B, L, rng):
    t = np.arange(L)[None, :]
    p = np.exp(rng.uniform(np.log(6), np.log(L), (B, 1)))
    return (rng.uniform(0.5, 2, (B, 1)) * np.sin(2 * np.pi * t / p + rng.uniform(0, 2 * np.pi, (B, 1)))).astype(np.float32)


def gen_ets(B, L, rng):
    """ForecastPFN-family: multiplicative Error-Trend-Seasonality."""
    t = np.arange(L)[None, :] / L
    trend = 1.0 + rng.normal(0, 0.5, (B, 1)) * t + rng.normal(0, 0.3, (B, 1)) * t**2
    seas = np.ones((B, L))
    for p0 in (7, 24, 168, 12):
        if rng.random() < 0.5:
            seas *= 1.0 + rng.uniform(0, 0.4, (B, 1)) * np.sin(2 * np.pi * np.arange(L)[None, :] / p0 + rng.uniform(0, 2 * np.pi, (B, 1)))
    err = 1.0 + rng.uniform(0.01, 0.1, (B, 1)) * rng.standard_normal((B, L))
    x = trend * np.clip(seas, 0.05, None) * err
    return (x - x.mean(1, keepdims=True)).astype(np.float32)


def gen_fractal(B, L, rng):
    """Audio-inspired multi-scale fractal: 1/f^beta spectral synthesis (rfft)."""
    beta = rng.uniform(0.5, 2.5, (B, 1))
    n_f = L // 2 + 1
    f = np.arange(1, n_f)[None, :]
    amp = f ** (-beta / 2)
    ph = rng.uniform(0, 2 * np.pi, (B, n_f - 1))
    spec = np.zeros((B, n_f), dtype=np.complex128)
    spec[:, 1:] = amp * np.exp(1j * ph)
    x = np.fft.irfft(spec, n=L, axis=1)
    x = x / (x.std(1, keepdims=True) + 1e-9)
    return x.astype(np.float32)


def gen_anomaly(B, L, rng):
    """Base signal + injected anomaly windows (level/variance bursts)."""
    x = gen_sine(B, L, rng) + 0.1 * rng.standard_normal((B, L)).astype(np.float32)
    for b in range(B):
        for _ in range(rng.integers(1, 4)):
            s = rng.integers(0, max(1, L - 20)); w = rng.integers(5, max(6, L // 20))
            if rng.random() < 0.5: x[b, s:s + w] += rng.normal(0, 2.0)
            else:                  x[b, s:s + w] *= rng.uniform(2, 5)
    return x


# --------------------------------------------------------------------------- ensemble
GEN_WEIGHTS = {          # roughly following TempoPFN's family coverage; adjust freely
    gen_kernelsynth_spectral: 0.20, gen_gp_spectral: 0.20, gen_sde_ou: 0.12,
    gen_ets: 0.12, gen_fractal: 0.10, gen_sawtooth: 0.07, gen_step: 0.07,
    gen_sine: 0.05, gen_spikes: 0.04, gen_anomaly: 0.03,
}


def _postprocess(x, rng):
    """Random compositions + global affine, mirroring PFN prior richness."""
    B, L = x.shape
    if rng.random() < 0.3:  x += gen_step(B, L, rng) * 0.7                   # level shifts
    if rng.random() < 0.3:  x += gen_spikes(B, L, rng) * 0.7                 # rare spikes
    if rng.random() < 0.5:  x += rng.normal(0, 1.0 / L, (B, 1)) * np.arange(L)[None, :]  # drift
    if rng.random() < 0.15: x = np.exp(np.clip(0.5 * x, -6, 6)) - 1.0        # positivity/skew
    x = x * np.exp(rng.uniform(np.log(0.2), np.log(5.0), (B, 1))) + rng.normal(0, 2.0, (B, 1))
    return x.astype(np.float32)


def sample_ensemble(B, L, rng):
    """Draw B series of length L from the mixed prior ensemble."""
    gens = list(GEN_WEIGHTS.keys())
    w = np.array(list(GEN_WEIGHTS.values())); w = w / w.sum()
    counts = rng.multinomial(B, w)
    outs = [g(int(c), L, rng) for g, c in zip(gens, counts) if c > 0]
    x = np.concatenate(outs, 0)
    return _postprocess(x, rng)[rng.permutation(B)]


# --- optional: official TempoPFN generators --------------------------------
# To use the official automl/TempoPFN prior instead of the native ensemble:
#   pip install "git+https://github.com/automl/TempoPFN.git"
# then adapt their generator wrappers (src/synthetic_generation/) to a callable
# (B, L, rng) -> [B, L] and register it via set_official_sampler(my_fn). The
# corpus builder prefers the official sampler when one is set.
_OFFICIAL_SAMPLER = None


def set_official_sampler(fn):
    """Register a ``(B, L, rng) -> float32[B, L]`` callable to override the
    native ensemble. Pass ``None`` to revert to the native sampler."""
    global _OFFICIAL_SAMPLER
    _OFFICIAL_SAMPLER = fn


def get_sampler():
    """The active corpus sampler: the official one if registered, else native."""
    return _OFFICIAL_SAMPLER or sample_ensemble
