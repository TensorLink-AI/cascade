from __future__ import annotations

import json
from collections.abc import Iterator
from functools import lru_cache, partial
from pathlib import Path
from queue import Full, Queue
from threading import Event, Thread

import numpy as np
from scipy.signal import lfilter

from cascade.interface import DataGenerator

_CHUNK = 2048


_STARTUP_CHUNK = 256
_RAMP_CHUNK = 1024


_SEASONAL_PERIODS = np.array(
    [4, 7, 12, 15, 24, 30, 48, 52, 60, 90, 96, 144, 168, 183, 240, 288,
     336, 365, 672, 730],
    dtype=np.float64,
)
_SEASONAL_PROBS = np.array(
    [0.01, 0.23, 0.02, 0.02, 0.07, 0.01, 0.06, 0.01, 0.08, 0.01,
     0.14, 0.07, 0.04, 0.01, 0.08, 0.08, 0.02, 0.02, 0.01, 0.01],
    dtype=np.float64,
)
_SEASONAL_PROBS /= _SEASONAL_PROBS.sum()


_SEASONAL_PAIRS = np.array(
    [[15, 60], [60, 240], [24, 168], [48, 336], [96, 672], [7, 365],
     [12, 52]],
    dtype=np.float64,
)


_FAMILIES: tuple[str, ...] = (
    "trend_seasonal_ar",
    "regime_shift",
    "multiplicative",
    "ar2",
    "integrated",
    "threshold_ar",
    "chaotic",
    "spectral_gp",
    "long_memory",
    "ou_stochastic_vol",
    "physical_sensors",
    "seasonal_counts",
    "intermittent",
    "pulse_outlier",
    "conditional_stability",
    "step_level",
    "vol_regime_switch",
    "weekly_demand",
    "tidal_harmonic",
    "flow_recession",
    "bounded_counts",
    "held_rate",
    "spiky_price",
    "epi_decay",
    "sticky_station",
    "grid_flow",
    "price_shock",
    "coastal_residual",
)


_DEFAULT_WEIGHTS: dict[str, float] = {
    "trend_seasonal_ar": 0.095,
    "regime_shift": 0.095,
    "multiplicative": 0.06,
    "ar2": 0.105,
    "integrated": 0.095,
    "threshold_ar": 0.06,
    "chaotic": 0.02,
    "spectral_gp": 0.07,
    "long_memory": 0.07,
    "ou_stochastic_vol": 0.08,
    "physical_sensors": 0.08,
    "seasonal_counts": 0.06,
    "intermittent": 0.02,
    "pulse_outlier": 0.02,
    "conditional_stability": 0.07,
    "step_level": 0.0,
    "vol_regime_switch": 0.0,
    "weekly_demand": 0.0,
    "tidal_harmonic": 0.0,
    "flow_recession": 0.0,
    "bounded_counts": 0.0,
    "held_rate": 0.0,
    "spiky_price": 0.0,
    "epi_decay": 0.0,
    "sticky_station": 0.0,
    "grid_flow": 0.0,
    "price_shock": 0.0,
    "coastal_residual": 0.0,
}



_CLEAN: frozenset[str] = frozenset({
    "held_rate",
    "step_level",
    "tidal_harmonic",
    "weekly_demand",
    "flow_recession",
})


def _step_level(rng: np.random.Generator, n: int, L: int) -> np.ndarray:
    """Piecewise-constant level with rare jumps, exactly flat in between.

    Half the rows carry literally zero within-segment noise, so the optimal
    forecast is the last value with a zero-width interval. A quantile loss
    punishes any spread there; an absolute-error loss does not notice.
    """
    jumps = rng.uniform(1.0, 8.0, size=(n, 1))
    at = rng.random((n, L)) < (jumps / max(L, 1))
    at[:, 0] = True
    size = rng.normal(0.0, 1.0, size=(n, L)) * rng.uniform(0.4, 3.0, size=(n, 1))
    level = np.cumsum(at * size, axis=1)
    scale = np.exp(rng.uniform(np.log(1.0), np.log(2000.0), size=(n, 1)))
    exact = rng.random((n, 1)) < 0.6
    sd = np.where(exact, 0.0, rng.uniform(0.002, 0.03, size=(n, 1)))
    out = (rng.uniform(-2.0, 2.0, size=(n, 1)) + level) * scale
    out = out + rng.normal(0.0, 1.0, size=(n, L)) * sd * scale
    integral = rng.random((n, 1)) < 0.65
    return np.where(integral, np.rint(out), out)


def _vol_regime_switch(rng: np.random.Generator, n: int, L: int) -> np.ndarray:
    """Persistent low/high volatility episodes with a stable level.

    conditional_stability teaches that volatility clusters, which pushes the
    model toward a wide interval everywhere. This teaches the conditional
    version: the recent window says which regime you are in, and the correct
    interval in the quiet regime is narrow. The regime is identifiable from the
    context, so a well-calibrated model can exploit it.
    """
    drive = _ar1_batch(rng.normal(0.0, 1.0, size=(n, L)),
                       rng.uniform(0.990, 0.9995, size=(n, 1)))
    # row-standardise in place; _standardize is not one of our helpers
    drive = ((drive - drive.mean(axis=1, keepdims=True))
             / np.maximum(drive.std(axis=1, keepdims=True), 1e-9))
    hot = drive > rng.uniform(0.2, 1.2, size=(n, 1))
    lo = rng.uniform(0.02, 0.20, size=(n, 1))
    ratio = rng.uniform(4.0, 25.0, size=(n, 1))
    sd = np.where(hot, lo * ratio, lo)
    x = _ar1_batch(rng.normal(0.0, 1.0, size=(n, L)) * sd,
                   rng.uniform(0.90, 0.999, size=(n, 1)))
    scale = np.exp(rng.uniform(np.log(1.0), np.log(500.0), size=(n, 1)))
    return x * scale + rng.uniform(-1.0, 1.0, size=(n, 1)) * scale


def _weekly_demand(rng: np.random.Generator, n: int, L: int) -> np.ndarray:
    """Strong deterministic weekly-plus-daily cycle under light noise.

    Near-fully forecastable once the period is read off the context, so the
    correct interval is dominated by the small noise term rather than by the
    swing of the cycle.
    """
    t = np.arange(L, dtype=np.float64)[None, :]
    day = rng.choice(np.array([24.0, 48.0, 96.0, 144.0]), size=(n, 1))
    week = day * 7.0
    amp_w = rng.uniform(0.4, 1.6, size=(n, 1))
    amp_d = rng.uniform(0.3, 1.4, size=(n, 1))
    y = amp_w * np.sin(2.0 * np.pi * t / week + rng.uniform(0, 2 * np.pi, (n, 1)))
    y = y + amp_d * np.sin(2.0 * np.pi * t / day + rng.uniform(0, 2 * np.pi, (n, 1)))
    y = y + 0.35 * amp_d * np.sin(4.0 * np.pi * t / day
                                  + rng.uniform(0, 2 * np.pi, (n, 1)))
    drift = rng.uniform(-0.3, 0.3, size=(n, 1)) * t / max(L - 1, 1)
    noise = rng.normal(0.0, 1.0, size=(n, L)) * rng.uniform(0.02, 0.15, size=(n, 1))
    base = np.exp(rng.uniform(np.log(5.0), np.log(5000.0), size=(n, 1)))
    out = base * np.exp(np.clip(y * 0.4 + drift + noise, -6.0, 6.0))
    counts = rng.random((n, 1)) < 0.45
    return np.where(counts, np.rint(out), out)


def _tidal_harmonic(rng: np.random.Generator, n: int, L: int) -> np.ndarray:
    """A few incommensurate harmonics with fixed amplitudes — near deterministic.

    The sum never repeats exactly, so it cannot be memorised as one period, but
    it is fully determined. The correct predictive interval is very narrow, which
    is precisely the case a globally-widened model gets wrong.
    """
    t = np.arange(L, dtype=np.float64)[None, :]
    k = int(rng.integers(3, 6))
    out = np.zeros((n, L), dtype=np.float64)
    base = rng.uniform(10.0, 400.0, size=(n, 1))
    for _ in range(k):
        period = base * rng.uniform(0.31, 2.7, size=(n, 1))
        out += (rng.uniform(0.2, 1.0, size=(n, 1))
                * np.sin(2.0 * np.pi * t / period
                         + rng.uniform(0, 2 * np.pi, size=(n, 1))))
    sd = rng.uniform(0.005, 0.05, size=(n, 1))
    scale = np.exp(rng.uniform(np.log(1.0), np.log(1000.0), size=(n, 1)))
    out = out + rng.normal(0.0, 1.0, size=(n, L)) * sd
    return out * scale + rng.uniform(-1.0, 1.0, size=(n, 1)) * scale


def _flow_recession(rng: np.random.Generator, n: int, L: int) -> np.ndarray:
    """Sparse recharge events, then geometric decay — a hydrograph.

    Between events the trajectory is deterministic given one decay constant, so
    most of the series is a narrow-interval problem punctuated by genuinely
    uncertain jumps. Learning to separate the two is exactly the calibration the
    quantile loss rewards.
    """
    rate = rng.uniform(1.0, 25.0, size=(n, 1)) / max(L, 1)
    hits = (rng.random((n, L)) < rate).astype(np.float64)
    mag = rng.gamma(2.0, 1.0, size=(n, L)) * rng.uniform(1.0, 12.0, size=(n, 1))
    decay = rng.uniform(0.90, 0.998, size=(n, 1))
    flow = _ar1_batch(hits * mag, decay)
    baseflow = rng.uniform(0.03, 0.6, size=(n, 1))
    scale = np.exp(rng.uniform(np.log(1.0), np.log(800.0), size=(n, 1)))
    sd = rng.uniform(0.0, 0.02, size=(n, 1))
    out = (flow + baseflow) * scale
    return np.maximum(out * (1.0 + rng.normal(0.0, 1.0, (n, L)) * sd), 0.0)


def _bounded_counts(rng: np.random.Generator, n: int, L: int) -> np.ndarray:
    """Integer occupancy inside a hard capacity, near unit root, reflecting.

    This is the shape of the pool's dominant profile — smooth, bounded, integer,
    strongly persistent, aseasonal. The difference from the bounded_occupancy
    attempt that failed is what the bounds do to the PREDICTION: reflection at 0
    and at capacity truncates the predictive interval asymmetrically near the
    edges, so the model has to learn a state-dependent spread rather than a
    global one. Matching the marginal statistics was never the point.
    """
    cap = rng.integers(4, 80, size=(n, 1)).astype(np.float64)
    step = (rng.normal(0.0, 1.0, size=(n, L))
            * rng.uniform(0.01, 0.12, size=(n, 1)) * cap)
    walk = np.cumsum(step, axis=1) + rng.uniform(0.0, 1.0, size=(n, 1)) * cap
    span = 2.0 * cap
    folded = cap - np.abs(np.mod(walk, span) - cap)
    quiet = rng.random((n, 1)) < 0.35
    folded = np.where(quiet, folded, folded + rng.normal(0.0, 0.35, size=(n, L)))
    return np.clip(np.rint(folded), 0.0, cap)


def _held_rate(rng: np.random.Generator, n: int, L: int) -> np.ndarray:
    """Near-constant series with 0-3 rare, grid-quantised steps.

    The archetype is a policy rate: flat for months, then one 25bp move. Most
    rows carry literally zero noise between steps, so the only calibrated
    forecast is "the last value, with almost no width". step_level cannot teach
    this — its jumps are too frequent and its sizes unquantised.
    """
    base = rng.uniform(-2.0, 8.0, size=(n, 1))
    grid = rng.choice(np.array([0.05, 0.1, 0.25, 0.5]), size=(n, 1))
    k = rng.integers(0, 4, size=(n, 1))                     # 0-3 steps per window
    at = rng.random((n, L)) < (k / max(L, 1))
    at[:, 0] = False
    size = grid * rng.choice(np.array([-2.,-1.,1.,2.]), size=(n, L))
    lvl = base + np.cumsum(at * size, axis=1)
    exact = rng.random((n, 1)) < 0.8
    sd = np.where(exact, 0.0, rng.uniform(0.001, 0.01, size=(n, 1)))
    out = lvl + rng.normal(0.0, 1.0, size=(n, L)) * sd
    scale = np.exp(rng.uniform(np.log(0.5), np.log(200.0), size=(n, 1)))
    return out * scale


def _spiky_price(rng: np.random.Generator, n: int, L: int) -> np.ndarray:
    """Day-ahead electricity price: daily+weekly cycle, two-sided heavy-tailed
    spikes with short persistence, volatility regimes, occasional negatives.

    The lesson is the opposite of held_rate's: a series whose recent window
    shows spikes deserves a WIDE interval, but only then — the quiet stretches
    between spike clusters are still narrow-interval territory.
    """
    t = np.arange(L, dtype=np.float64)[None, :]
    day = rng.choice(np.array([24.0, 48.0, 96.0]), size=(n, 1))
    amp = rng.uniform(0.2, 1.0, size=(n, 1))
    cyc = amp * np.sin(2*np.pi*t/day + rng.uniform(0, 2*np.pi, (n, 1)))
    cyc += 0.4*amp*np.sin(4*np.pi*t/day + rng.uniform(0, 2*np.pi, (n, 1)))
    week = 0.3*amp*np.sin(2*np.pi*t/(day*7) + rng.uniform(0, 2*np.pi, (n, 1)))
    lvl = _ar1_batch(rng.normal(0.0, 0.05, size=(n, L)),
                     rng.uniform(0.995, 0.9999, size=(n, 1)))
    hot = _ar1_batch(rng.normal(0.0, 1.0, size=(n, L)),
                     rng.uniform(0.95, 0.995, size=(n, 1)))
    hot = hot > np.quantile(hot, rng.uniform(0.7, 0.95), axis=1, keepdims=True)
    spike_p = rng.uniform(0.002, 0.02, size=(n, 1)) * (1 + 6*hot)
    hits = rng.random((n, L)) < spike_p
    mag = rng.standard_t(3, size=(n, L)) * rng.uniform(0.5, 3.0, size=(n, 1))
    spikes = _ar1_batch(hits * mag, rng.uniform(0.3, 0.8, size=(n, 1)))
    out = cyc + week + lvl + spikes
    scale = np.exp(rng.uniform(np.log(5.0), np.log(300.0), size=(n, 1)))
    shift = rng.uniform(0.0, 2.0, size=(n, 1))
    return (out + shift) * scale          # negatives possible when spikes dip


def _epi_decay(rng: np.random.Generator, n: int, L: int) -> np.ndarray:
    """Counts whose log-rate is piecewise linear: growth phases turning into
    decay, times a day-of-week multiplicative pattern. Poisson/negbin draws.

    Covers epidemic curves and campaign-style traffic: the decay slope is
    readable from the context, so a calibrated model can narrow its interval
    along the decaying tail instead of hedging against a rebound.
    """
    t = np.arange(L, dtype=np.float64)[None, :]
    k = rng.integers(2, 6)
    knots = np.sort(rng.uniform(0, L, size=(n, k)), axis=1)
    slopes = rng.uniform(-0.004, 0.003, size=(n, k+1))
    log_rate = np.zeros((n, L))
    prev = np.zeros((n, 1))
    for i in range(k+1):
        lo = prev
        hi = knots[:, i:i+1] if i < k else np.full((n, 1), float(L))
        seg = np.clip(t, lo, hi) - lo
        log_rate = log_rate + slopes[:, i:i+1] * seg
        prev = hi
    day = rng.choice(np.array([1.0, 24.0, 48.0]), size=(n, 1), p=[0.5, 0.3, 0.2])
    period = np.where(day == 1.0, 7.0, day * 7)
    dow = rng.uniform(0.1, 0.6, size=(n, 1)) * np.sin(
        2*np.pi*t/period + rng.uniform(0, 2*np.pi, (n, 1)))
    base = np.exp(rng.uniform(np.log(3.0), np.log(3000.0), size=(n, 1)))
    lam = base * np.exp(np.clip(log_rate + dow, -8.0, 6.0))
    np.clip(lam, 0.0, 5.0e6, out=lam)
    over = rng.random((n, 1)) < 0.5
    shape = rng.uniform(0.6, 4.0, size=(n, 1))
    mixed = lam * rng.gamma(shape, 1.0/shape, size=(n, L))
    return rng.poisson(np.where(over, mixed, lam)).astype(np.float64)


def _sticky_station(rng: np.random.Generator, n: int, L: int) -> np.ndarray:
    """Small-integer, extremely sticky, capacity-reflected random walk.

    Fitted to the pool's dominant profile, MEASURED from 60 GBFS
    station_status windows of block-8730000 (65% of the pool by window count):
    hold fraction p10/p50/p90 = 0.54/0.84/0.94; capacity 8/18/42; moves are
    almost all +-1 with no big rebalancing jumps at 20%-of-capacity scale;
    daily autocorrelation median ~0.0 (tides are NOT dominant); lag1 ~0.99.

    The quantile lesson: the next value IS the current value with high
    probability, and uncertainty grows only slowly with horizon - tight
    near-term quantiles on a sticky integer walk.
    """
    cap = np.exp(rng.normal(np.log(18.0), 0.55, size=(n, 1)))
    cap = np.clip(np.rint(cap), 4.0, 80.0)
    hold = rng.beta(3.2, 1.0, size=(n, 1)) * 0.42 + 0.53      # ~0.55..0.95
    step2 = rng.uniform(0.03, 0.15, size=(n, 1))              # of the moves
    move = rng.random((n, L)) >= hold
    mag = np.where(rng.random((n, L)) < step2, 2.0, 1.0)
    sgn = np.where(rng.random((n, L)) < 0.5, -1.0, 1.0)
    # weak daily modulation for a minority of rows (measured: mostly absent)
    t = np.arange(L, dtype=np.float64)[None, :]
    period = rng.choice(np.array([96.0, 144.0, 288.0]), size=(n, 1))
    tide = rng.random((n, 1)) < 0.3
    bias = np.where(tide, 0.35, 0.0) * np.sin(
        2*np.pi*t/period + rng.uniform(0, 2*np.pi, (n, 1)))
    sgn = np.where(rng.random((n, L)) < 0.5 + bias, sgn, -sgn)
    steps = move * mag * sgn
    walk = rng.uniform(0.15, 0.85, size=(n, 1)) * cap + np.cumsum(steps, axis=1)
    span = 2.0 * cap
    out = cap - np.abs(np.mod(walk, span) - cap)              # reflect at 0/cap
    return np.rint(out)


def _grid_flow(rng: np.random.Generator, n: int, L: int) -> np.ndarray:
    """Power-grid series: a hard 96-step daily profile, values free to go
    negative, heavy-tailed jumps, and a very smooth carrier.

    Fitted to the feeds our king actually LOSES on in the duel receipt
    (energy_charts public_power / day_ahead_price / co2_intensity, national
    demand): measured lag1 0.985, seasonal autocorrelation 0.79 at lag 96 for
    33 of 49 sampled windows, kurtosis 11, 1.9% of steps beyond 3 sigma, and a
    tenth of the windows spending most of their time BELOW zero.

    The last property is why an existing family could not cover this: every
    generator we ship is positive or symmetric around a positive level, so the
    corpus never taught a quantile spread on the negative side of an axis that
    real prices and cross-border flows cross constantly.
    """
    t = np.arange(L, dtype=np.float64)[None, :]
    # 96 dominates; keep a minority on the other measured cadences
    period = rng.choice(np.array([96.0, 96.0, 96.0, 24.0, 288.0, 48.0]), size=(n, 1))
    prof = np.zeros((n, L))
    for k in (1.0, 2.0, 3.0):
        prof += (rng.uniform(0.25, 1.0, size=(n, 1)) / k) * np.sin(
            2 * np.pi * k * t / period + rng.uniform(0, 2 * np.pi, size=(n, 1)))
    week = 0.25 * rng.uniform(0.2, 1.0, size=(n, 1)) * np.sin(
        2 * np.pi * t / (period * 7) + rng.uniform(0, 2 * np.pi, size=(n, 1)))
    # smooth carrier: lag1 ~0.985 comes from the level, not from the profile
    carrier = _ar1_batch(rng.normal(0.0, 1.0, size=(n, L)),
                         rng.uniform(0.995, 0.9999, size=(n, 1)))
    carrier = carrier / np.maximum(np.std(carrier, axis=1, keepdims=True), 1e-9)
    # Heavy tails, but bounded: standard_t(3) at full weight drove kurtosis to
    # 120 against a measured 11 — the spikes then dominate the series and the
    # profile the model must learn disappears underneath them.
    hit = rng.random((n, L)) < rng.uniform(0.004, 0.016, size=(n, 1))
    mag = np.clip(rng.standard_t(4, size=(n, L)), -6.0, 6.0) * rng.uniform(0.20, 0.75, size=(n, 1))
    spikes = _ar1_batch(hit * mag, rng.uniform(0.25, 0.70, size=(n, 1)))
    # The daily profile is the dominant signal (measured lag-96 autocorrelation
    # 0.79), the smooth carrier supplies lag1 0.985, noise is a garnish.
    amp = rng.uniform(1.0, 2.4, size=(n, 1))
    y = amp * (prof + week) + rng.uniform(0.15, 0.45, size=(n, 1)) * carrier + spikes
    scale = np.exp(rng.uniform(np.log(3.0), np.log(900.0), size=(n, 1)))
    # a tenth of rows live mostly below zero; the rest sit on a positive level
    below = rng.random((n, 1)) < 0.22
    level = np.where(below, rng.uniform(-1.2, 0.1, size=(n, 1)),
                            rng.uniform(0.4, 3.0, size=(n, 1)))
    return (y + level) * scale


def _price_shock(rng: np.random.Generator, n: int, L: int) -> np.ndarray:
    """A spike that MEAN-REVERTS on a known clock, not a random walk that jumps.

    The second measured property of the feeds our king loses on: an excursion
    is followed by a return, and the return has a timescale. A generator that
    only knows "jumps happen" teaches a permanently wider interval after every
    shock; one that knows "and it comes back in ~k steps" teaches the interval
    to CLOSE again. That difference is priced by a quantile loss and invisible
    to an absolute-error one.
    """
    t = np.arange(L, dtype=np.float64)[None, :]
    period = rng.choice(np.array([96.0, 96.0, 24.0, 288.0]), size=(n, 1))
    base = rng.uniform(0.4, 1.4, size=(n, 1)) * np.sin(
        2 * np.pi * t / period + rng.uniform(0, 2 * np.pi, size=(n, 1)))
    # shocks arrive, then decay back to the profile with a per-row half-life
    rate = rng.uniform(0.003, 0.020, size=(n, 1))
    hit = (rng.random((n, L)) < rate).astype(np.float64)
    sign = np.where(rng.random((n, L)) < 0.62, 1.0, -1.0)      # upward-skewed
    size = np.abs(np.clip(rng.standard_t(4, size=(n, L)), -8, 8)) * rng.uniform(0.5, 2.2, size=(n, 1))
    decay = rng.uniform(0.55, 0.97, size=(n, 1))               # the KNOWN clock
    shock = _ar1_batch(hit * sign * size, decay)
    carrier = _ar1_batch(rng.normal(0.0, 1.0, size=(n, L)),
                         rng.uniform(0.99, 0.9995, size=(n, 1)))
    carrier = carrier / np.maximum(np.std(carrier, axis=1, keepdims=True), 1e-9)
    y = base + shock + rng.uniform(0.15, 0.5, size=(n, 1)) * carrier
    scale = np.exp(rng.uniform(np.log(2.0), np.log(600.0), size=(n, 1)))
    level = np.where(rng.random((n, 1)) < 0.18,
                     rng.uniform(-1.0, 0.2, size=(n, 1)), rng.uniform(0.5, 3.0, size=(n, 1)))
    return (y + level) * scale


# Grafted from buddy777's public round-8737200 entry (their family "k21"),
# renamed. Their insight is worth adopting whole: a harmonic family emits a
# clean constituent sum, so the model learns the cycle and then has nothing to
# say about the DEPARTURE from it — and at a 64-step horizon the departure is
# the forecastable part. The construction encodes that asymmetry directly:
# a sharp rise and a long decay built from two AR(1) passes with different
# coefficients, volatility-clustered wind-sea on top, and a wandering datum.
def _coastal_residual(rng: np.random.Generator, n: int, L: int) -> np.ndarray:
    """Coastal observation: a tidal skeleton plus the weather-driven residual.

    Harmonic families emit a clean constituent sum, so a model learns the tide
    and then has nothing to say about the departure from it. Real gauges and
    buoys carry a non-tidal residual — multi-day storm surge with a sharp rise
    and a long decay, wind-sea whose amplitude clusters in time, and a slowly
    wandering datum. Those departures are the forecastable part at a 64-step
    horizon, and they are absent from a pure-harmonic prior.
    """
    if n <= 0:
        return np.empty((0, L), dtype=np.float64)
    t = np.arange(L, dtype=np.float64)[None, :]
    P0 = np.exp(rng.uniform(np.log(20.0), np.log(60.0), size=(n, 1)))
    tide = np.sin(2.0 * np.pi * t / P0 + rng.uniform(0, 2 * np.pi, size=(n, 1)))
    tide = tide + rng.uniform(0.15, 0.7, size=(n, 1)) * np.sin(
        2.0 * np.pi * t / (P0 * 1.9323) + rng.uniform(0, 2 * np.pi, size=(n, 1)))
    beat = P0 * rng.uniform(12.0, 32.0, size=(n, 1))
    tide = tide * (1.0 + rng.uniform(0.1, 0.55, size=(n, 1)) * np.sin(
        2.0 * np.pi * t / beat + rng.uniform(0, 2 * np.pi, size=(n, 1))))
    tidal_frac = rng.uniform(0.0, 1.0, size=(n, 1))
    onset = rng.random((n, L)) < rng.uniform(1.5, 8.0, size=(n, 1)) / max(L, 1)
    amp = np.exp(rng.normal(rng.uniform(-0.6, 0.9, size=(n, 1)),
                            rng.uniform(0.4, 1.0, size=(n, 1)), size=(n, L))) * onset
    rise = rng.uniform(0.55, 0.9, size=(n, 1))
    fall = rng.uniform(0.97, 0.999, size=(n, 1))
    surge = _ar1_batch(0.6 * amp, fall) - 0.55 * _ar1_batch(amp, rise)
    logvol = _ar1_batch(rng.normal(0.0, 1.0, size=(n, L))
                        * rng.uniform(0.05, 0.3, size=(n, 1)),
                        rng.uniform(0.95, 0.999, size=(n, 1)))
    sea = np.exp(np.clip(logvol, -4.0, 4.0)) * rng.normal(0.0, 1.0, size=(n, L)) \
        * rng.uniform(0.05, 0.4, size=(n, 1))
    datum = np.cumsum(rng.normal(0.0, 1.0, size=(n, L)), axis=1) \
        * rng.uniform(0.0, 0.02, size=(n, 1)) / np.sqrt(max(L, 1))
    x = tidal_frac * tide + rng.uniform(0.3, 1.6, size=(n, 1)) * surge + sea + datum
    x = (x - x.mean(axis=1, keepdims=True)) / np.maximum(x.std(axis=1, keepdims=True), 1e-9)
    scale = np.exp(rng.uniform(np.log(0.05), np.log(200.0), size=(n, 1)))
    offset = rng.normal(0.0, 3.0, size=(n, 1)) * scale
    y = x * scale + offset
    positive = rng.random((n, 1)) < 0.45
    return np.where(positive, np.abs(y) + 0.05 * scale, y)


def _validate_parameters(parameters: dict[str, float], label: str) -> None:
    if not all(np.isfinite(value) for value in parameters.values()):
        raise ValueError(f"{label} must contain only finite values")
    probability_names = {
        "sa_clean_frac",
        "integrated_heavy_frac",
        "integrated_sv_frac",
        *(key for key in parameters if key.startswith(("observation.", "augment."))),
    }
    for name in probability_names:
        if not 0.0 <= parameters[name] <= 1.0:
            raise ValueError(f"{label}.{name} must be in [0, 1]")
    for name in (
        "tr_exc_lo",
        "tr_exc_hi",
        "gr_exc_lo",
        "gr_exc_hi",
        "sa_clean_lo",
        "sa_clean_hi",
    ):
        if parameters[name] < 0.0:
            raise ValueError(f"{label}.{name} must be non-negative")
    for lo_name, hi_name in (
        ("tr_exc_lo", "tr_exc_hi"),
        ("gr_exc_lo", "gr_exc_hi"),
        ("sa_clean_lo", "sa_clean_hi"),
        ("observation.irregular_hold_prob_lo", "observation.irregular_hold_prob_hi"),
        ("observation.shock_prob_lo", "observation.shock_prob_hi"),
    ):
        if parameters[lo_name] > parameters[hi_name]:
            raise ValueError(f"{label}.{lo_name} must be <= {hi_name}")


class Generator(DataGenerator):


    def __init__(self, config_dir: str, *, seed: int) -> None:
        cfg_path = Path(config_dir) / "config.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {}
        self._cfg = cfg
        self._seed = int(seed)
        self._min_len = int(cfg.get("min_length", 64))
        self._max_len = int(cfg.get("max_length", 4096))
        if self._min_len < 1 or self._max_len < self._min_len:
            raise ValueError(f"invalid length band [{self._min_len}, {self._max_len}]")
        weights = dict(_DEFAULT_WEIGHTS)
        for k, v in dict(cfg.get("family_weights", {})).items():
            if k in weights:
                weights[k] = float(v)
        w = np.asarray([weights[f] for f in _FAMILIES], dtype=np.float64)
        if not np.all(np.isfinite(w)) or w.min() < 0 or w.sum() <= 0:
            raise ValueError("family_weights must be finite, non-negative, and not all zero")
        self._weights = w / w.sum()
        curriculum = dict(cfg.get("curriculum", {}))
        self._curriculum_enabled = bool(curriculum.get("enabled", False))
        self._expected_budget_fraction = float(
            curriculum.get("expected_budget_fraction", 1.0)
        )
        if (
            not np.isfinite(self._expected_budget_fraction)
            or not 0.0 < self._expected_budget_fraction <= 1.0
        ):
            raise ValueError("curriculum.expected_budget_fraction must be in (0, 1]")
        self._curriculum_start = float(curriculum.get("start_fraction", 0.10))
        self._curriculum_end = float(curriculum.get("end_fraction", 0.70))
        if not 0.0 <= self._curriculum_start < self._curriculum_end <= 1.0:
            raise ValueError(
                "curriculum fractions must satisfy 0 <= start_fraction < end_fraction <= 1"
            )
        start_weights = dict(weights)
        for k, v in dict(curriculum.get("start_family_weights", {})).items():
            if k in start_weights:
                start_weights[k] = float(v)
        start_w = np.asarray([start_weights[f] for f in _FAMILIES], dtype=np.float64)
        if (
            not np.all(np.isfinite(start_w))
            or start_w.min() < 0
            or start_w.sum() <= 0
        ):
            raise ValueError(
                "curriculum.start_family_weights must be finite, non-negative, "
                "and not all zero"
            )
        self._start_weights = start_w / start_w.sum()

        self._tr_hi_frac = float(cfg.get("tr_hi_frac", 0.25))
        self._prefetch_depth = int(cfg.get("prefetch_depth", 2))
        if not 1 <= self._prefetch_depth <= 4:
            raise ValueError("prefetch_depth must be in [1, 4]")
        augment = dict(cfg.get("augment", {}))
        observation = dict(cfg.get("observation", {}))
        self._parameters = {
            "tr_exc_lo": float(cfg.get("tr_exc_lo", 0.4)),
            "tr_exc_hi": float(cfg.get("tr_exc_hi", 3.0)),
            "gr_exc_lo": float(cfg.get("gr_exc_lo", 0.3)),
            "gr_exc_hi": float(cfg.get("gr_exc_hi", 2.0)),
            "sa_clean_frac": float(cfg.get("sa_clean_frac", 0.4)),
            "sa_clean_lo": float(cfg.get("sa_clean_lo", 0.02)),
            "sa_clean_hi": float(cfg.get("sa_clean_hi", 0.12)),
            "integrated_heavy_frac": float(
                cfg.get("integrated_heavy_frac", 0.25)
            ),
            "integrated_sv_frac": float(cfg.get("integrated_sv_frac", 0.30)),
            "observation.censor_rate": float(observation.get("censor_rate", 0.06)),
            "observation.quantize_rate": float(
                observation.get("quantize_rate", 0.07)
            ),
            "observation.regular_hold_rate": float(
                observation.get("regular_hold_rate", 0.04)
            ),
            "observation.irregular_hold_rate": float(
                observation.get("irregular_hold_rate", 0.0)
            ),
            "observation.irregular_hold_prob_lo": float(
                observation.get("irregular_hold_prob_lo", 0.01)
            ),
            "observation.irregular_hold_prob_hi": float(
                observation.get("irregular_hold_prob_hi", 0.10)
            ),
            "observation.shock_row_rate": float(
                observation.get("shock_row_rate", 0.0)
            ),
            "observation.shock_prob_lo": float(
                observation.get("shock_prob_lo", 0.001)
            ),
            "observation.shock_prob_hi": float(
                observation.get("shock_prob_hi", 0.015)
            ),
            "augment.tsmixup": float(augment.get("tsmixup", 0.0)),
            "augment.pad_prefix": float(augment.get("pad_prefix", 0.0)),
        }
        start_parameters = dict(curriculum.get("start_parameters", {}))
        start_observation = dict(start_parameters.pop("observation", {}))
        start_augment = dict(start_parameters.pop("augment", {}))
        known_top_level = {
            key for key in self._parameters if "." not in key
        }
        unknown = set(start_parameters) - known_top_level
        unknown.update(
            f"observation.{key}"
            for key in start_observation
            if f"observation.{key}" not in self._parameters
        )
        unknown.update(
            f"augment.{key}"
            for key in start_augment
            if f"augment.{key}" not in self._parameters
        )
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"unknown curriculum.start_parameters: {names}")
        self._start_parameters = dict(self._parameters)
        for key, value in start_parameters.items():
            self._start_parameters[key] = float(value)
        for key, value in start_observation.items():
            self._start_parameters[f"observation.{key}"] = float(value)
        for key, value in start_augment.items():
            self._start_parameters[f"augment.{key}"] = float(value)
        _validate_parameters(self._parameters, "final parameters")
        _validate_parameters(
            self._start_parameters, "curriculum.start_parameters"
        )

    @property
    def name(self) -> str:
        return str(self._cfg.get("name", "cascade-heat3-fast-learn-curriculum"))

    def _blend_at(self, token_progress: float) -> float:
        """Return the shared smoothstep blend for weights and difficulty."""
        if not self._curriculum_enabled:
            return 1.0
        position = (token_progress - self._curriculum_start) / (
            self._curriculum_end - self._curriculum_start
        )
        position = float(np.clip(position, 0.0, 1.0))
        return position * position * (3.0 - 2.0 * position)

    def _weights_at(self, token_progress: float) -> np.ndarray:
        """Blend easy-to-final family weights with a smoothstep schedule."""
        blend = self._blend_at(token_progress)
        if blend >= 1.0:
            return self._weights
        if blend <= 0.0:
            return self._start_weights
        return (1.0 - blend) * self._start_weights + blend * self._weights

    def _parameters_at(self, token_progress: float) -> dict[str, float]:
        """Blend all within-family, observation, and augmentation settings."""
        blend = self._blend_at(token_progress)
        if blend >= 1.0:
            return dict(self._parameters)
        if blend <= 0.0:
            return dict(self._start_parameters)
        return {
            key: (1.0 - blend) * self._start_parameters[key]
            + blend * final_value
            for key, final_value in self._parameters.items()
        }

    def _progress_at(self, emitted_points: float, target_points: int) -> float:
        """Calibrate nominal point progress to expected heat consumption."""
        return emitted_points / (target_points * self._expected_budget_fraction)

    def generate(self, n_series: int) -> Iterator[np.ndarray]:


        if n_series <= 0:
            return
        rng = np.random.default_rng(self._seed)
        max_len = self._max_len
        # stream_cpu requests token_budget // min_length + 2 rows. Recover the
        # budget so this fixed-length generator follows trainer token progress.
        target_points = max(1, max(n_series - 2, 1) * self._min_len)


        queue: Queue[object] = Queue(maxsize=self._prefetch_depth)
        stop = Event()
        done = object()

        def put(item: object) -> bool:
            while not stop.is_set():
                try:
                    queue.put(item, timeout=0.1)
                    return True
                except Full:
                    continue
            return False

        def produce() -> None:
            try:
                produced = 0
                emitted_points = 0
                while produced < n_series and not stop.is_set():


                    if produced == 0:
                        batch_size = _STARTUP_CHUNK
                    elif produced == _STARTUP_CHUNK:


                        batch_size = _RAMP_CHUNK
                    else:
                        batch_size = _CHUNK
                    lengths = rng.integers(
                        self._min_len, max_len + 1, size=batch_size
                    )
                    take = min(batch_size, n_series - produced)
                    chunk_points = int(lengths[:take].sum())
                    midpoint_progress = self._progress_at(
                        emitted_points + 0.5 * chunk_points, target_points
                    )
                    family_weights = self._weights_at(midpoint_progress)
                    parameters = self._parameters_at(midpoint_progress)
                    builders = (
                        partial(
                            _trend_seasonal_ar,
                            hi_frac=self._tr_hi_frac,
                            exc_lo=parameters["tr_exc_lo"],
                            exc_hi=parameters["tr_exc_hi"],
                            clean_frac=parameters["sa_clean_frac"],
                            clean_lo=parameters["sa_clean_lo"],
                            clean_hi=parameters["sa_clean_hi"],
                        ),
                        _regime_shift,
                        partial(
                            _multiplicative,
                            hi_frac=self._tr_hi_frac,
                            exc_lo=parameters["gr_exc_lo"],
                            exc_hi=parameters["gr_exc_hi"],
                        ),
                        _ar2,
                        partial(
                            _integrated,
                            heavy_frac=parameters["integrated_heavy_frac"],
                            sv_frac=parameters["integrated_sv_frac"],
                        ),
                        _threshold_ar,
                        _chaotic,
                        _spectral_gp,
                        _long_memory,
                        _ou_stochastic_vol,
                        _physical_sensors,
                        _seasonal_counts,
                        _intermittent,
                        _pulse_outlier,
                        _conditional_stability,
                        _step_level,
                        _vol_regime_switch,
                        _weekly_demand,
                        _tidal_harmonic,
                        _flow_recession,
                        _bounded_counts,
                        _held_rate,
                        _spiky_price,
                        _epi_decay,
                        _sticky_station,
                        _grid_flow,
                        _price_shock,
                        _coastal_residual,
                    )
                    current_observation = {
                        key.removeprefix("observation."): value
                        for key, value in parameters.items()
                        if key.startswith("observation.")
                    }
                    fam_ids = rng.choice(
                        len(_FAMILIES), size=batch_size, p=family_weights
                    )
                    chunk: list[np.ndarray | None] = [None] * batch_size
                    for fam in range(len(_FAMILIES)):
                        idx = np.nonzero(fam_ids == fam)[0]
                        if idx.size == 0:
                            continue
                        block = builders[fam](rng, int(idx.size), max_len)
                        family = _FAMILIES[fam]
                        preserve_nonnegative = family in {
                            "sticky_station",
                            "epi_decay",
                            "multiplicative",
                            "physical_sensors",
                            "seasonal_counts",
                            "intermittent",
                        }
                        if family in _CLEAN:
                            block = _sanitize(block)
                        else:
                          block = _sanitize(
                            _measurement_artifacts(
                                rng,
                                block,
                                preserve_nonnegative=preserve_nonnegative,
                                preserve_integers=family in {
                                    "sticky_station",
                                    "epi_decay",
                                    "seasonal_counts",
                                    "intermittent",
                                },
                                allow_reverse=family in {
                                    "trend_seasonal_ar",
                                    "multiplicative",
                                    "spectral_gp",
                                    "long_memory",
                                },
                                allow_range_artifacts=family != "integrated",
                                **current_observation,
                            )
                          )
                        for row, series_i in enumerate(idx):
                            length = int(lengths[series_i])
                            chunk[series_i] = np.ascontiguousarray(
                                block[row, :length], dtype=np.float64
                            )


                    if self._min_len == max_len:
                        mix_rate = parameters["augment.tsmixup"]
                        mixed = np.nonzero(rng.random(batch_size) < mix_rate)[0]
                        for series_i in mixed:
                            source = chunk[series_i]
                            if source is None:
                                continue
                            n_other = int(rng.integers(1, 3))
                            others = rng.integers(0, batch_size, size=n_other)
                            weights = rng.dirichlet(np.ones(n_other + 1))
                            combined = weights[0] * source
                            valid = True
                            for j, other_i in enumerate(others):
                                other = chunk[int(other_i)]
                                if other is None:
                                    valid = False
                                    break
                                combined = combined + weights[j + 1] * other
                            if valid:
                                chunk[series_i] = _sanitize(combined)


                    pad_rate = parameters["augment.pad_prefix"]
                    padded = np.nonzero(rng.random(batch_size) < pad_rate)[0]
                    for series_i in padded:
                        series = chunk[series_i]
                        if series is None or series.size < 8:
                            continue
                        cut = int(rng.integers(series.size // 8, 3 * series.size // 4))
                        series[:cut] = series[cut]
                    if not put((chunk, take)):
                        return
                    produced += take
                    emitted_points += chunk_points
            except BaseException as exc:
                put(exc)
            finally:
                put(done)

        producer = Thread(target=produce, name="cascade-generator", daemon=True)
        producer.start()
        try:
            while True:
                item = queue.get()
                if item is done:
                    break
                if isinstance(item, BaseException):
                    raise item
                chunk, take = item
                for arr in chunk[:take]:

                    if arr is None:
                        raise RuntimeError("internal: unfilled series slot")
                    yield arr
        finally:
            stop.set()
            producer.join(timeout=1.0)


def _ar1_batch(innov: np.ndarray, phi: np.ndarray) -> np.ndarray:


    n, L = innov.shape
    x = np.empty((n, L), dtype=np.float64)
    p = phi.reshape(n)
    for i in range(n):
        x[i] = lfilter([1.0], [1.0, -float(p[i])], innov[i])
    return x


def _ar2_batch(innov: np.ndarray, a1: np.ndarray, a2: np.ndarray) -> np.ndarray:

    n, L = innov.shape
    x = np.empty((n, L), dtype=np.float64)
    for i in range(n):
        x[i] = lfilter(
            [1.0], [1.0, -float(a1[i]), -float(a2[i])], innov[i]
        )
    return x


def _prefix_mean_std(
    x: np.ndarray, *, calibration_points: int = 512
) -> tuple[np.ndarray, np.ndarray]:

    prefix = x[:, : min(x.shape[1], calibration_points)]
    mean = prefix.mean(axis=1, keepdims=True)
    std = prefix.std(axis=1, keepdims=True)
    return mean, np.where(std < 1e-12, 1.0, std)


def _prefix_standardize(
    x: np.ndarray, *, center: bool = True, calibration_points: int = 512
) -> np.ndarray:
    mean, std = _prefix_mean_std(
        x, calibration_points=calibration_points
    )
    return (x - mean) / std if center else x / std


@lru_cache(maxsize=4)
def _seasonal_basis(L: int) -> tuple[np.ndarray, np.ndarray]:

    angle = (
        2.0
        * np.pi
        * np.arange(L, dtype=np.float64)[None, :]
        / _SEASONAL_PERIODS[:, None]
    )
    return np.sin(angle), np.cos(angle)


def _seasonal(rng: np.random.Generator, n: int, L: int, k_max: int = 3) -> np.ndarray:

    t = np.arange(L, dtype=np.float64)[None, :]
    sin_basis, cos_basis = _seasonal_basis(L)
    k = rng.integers(1, k_max + 1, size=n)
    pair = _SEASONAL_PAIRS[
        rng.integers(0, len(_SEASONAL_PAIRS), size=n)
    ]
    use_pair = rng.random(n) < 0.35
    out = np.zeros((n, L), dtype=np.float64)
    for j in range(k_max):
        active = np.nonzero(k > j)[0]
        per = rng.choice(
            _SEASONAL_PERIODS, size=n, p=_SEASONAL_PROBS
        )
        if j < 2:
            per = np.where(use_pair, pair[:, j], per)
        per = per[:, None]
        amp = rng.uniform(0.2, 2.0, size=n)[:, None]
        phase = rng.uniform(0.0, 2.0 * np.pi, size=n)[:, None]


        basis_idx = np.searchsorted(_SEASONAL_PERIODS, per[active, 0])
        component = amp[active] * (
            sin_basis[basis_idx] * np.cos(phase[active])
            + cos_basis[basis_idx] * np.sin(phase[active])
        )


        modulated = np.nonzero((k > j) & (rng.random(n) < 0.35))[0]
        if modulated.size:

            modulated_local = np.searchsorted(active, modulated)
            modulated_arg = (
                2.0 * np.pi * t / per[modulated] + phase[modulated]
            )
            m_per = np.clip(
                per[modulated] * rng.uniform(
                    4.0, 12.0, size=(modulated.size, 1)
                ),
                32.0,
                2.0 * L,
            )
            m_phase = rng.uniform(
                0.0, 2.0 * np.pi, size=(modulated.size, 1)
            )
            slow = np.sin(2.0 * np.pi * t / m_per + m_phase)
            amp_mod = 1.0 + rng.uniform(
                0.05, 0.45, size=(modulated.size, 1)
            ) * slow
            phase_mod = rng.uniform(
                0.05, 0.75, size=(modulated.size, 1)
            ) * np.sin(2.0 * np.pi * t / (1.7 * m_per) - m_phase)
            component[modulated_local] = (
                amp[modulated]
                * amp_mod
                * np.sin(modulated_arg + phase_mod)
            )
        out[active] += component
    return out


def _sparse_jumps(rng: np.random.Generator, n: int, L: int, rate: float, scale) -> np.ndarray:


    mask = rng.random((n, L)) < rate
    mask[:, 0] = False
    rows, cols = np.nonzero(mask)
    jumps = np.zeros((n, L), dtype=np.float64)
    if rows.size == 0:
        return jumps


    s = np.asarray(scale, dtype=np.float64)
    event_scale = s if s.ndim == 0 else s.reshape(n)[rows]
    jumps[rows, cols] = rng.normal(0.0, 1.0, size=rows.size) * event_scale
    return jumps


def _measurement_artifacts(
    rng: np.random.Generator,
    block: np.ndarray,
    *,
    preserve_nonnegative: bool,
    preserve_integers: bool = False,
    allow_reverse: bool = True,
    allow_range_artifacts: bool = True,
    censor_rate: float = 0.06,
    quantize_rate: float = 0.07,
    regular_hold_rate: float = 0.04,
    irregular_hold_rate: float = 0.0,
    irregular_hold_prob_lo: float = 0.01,
    irregular_hold_prob_hi: float = 0.10,
    shock_row_rate: float = 0.0,
    shock_prob_lo: float = 0.001,
    shock_prob_hi: float = 0.015,
) -> np.ndarray:


    original = np.asarray(block, dtype=np.float64)
    out = original.copy()
    n, L = out.shape

    reverse = (
        rng.random(n) < 0.06
        if allow_reverse
        else np.zeros(n, dtype=bool)
    )
    out[reverse] = out[reverse, ::-1]

    if not preserve_nonnegative:
        invert = rng.random(n) < 0.04
        out[invert] *= -1.0


    calibration_len = min(L, 512)
    shocked = np.nonzero(rng.random(n) < shock_row_rate)[0]
    if shocked.size and L > 1:
        diff = np.diff(out[shocked, :calibration_len], axis=1)
        center = np.median(diff, axis=1, keepdims=True)
        robust_scale = 1.4826 * np.median(
            np.abs(diff - center), axis=1, keepdims=True
        )
        fallback = np.maximum(np.std(diff, axis=1, keepdims=True), 1e-9)
        robust_scale = np.where(robust_scale > 1e-9, robust_scale, fallback)
        event_prob = rng.uniform(
            shock_prob_lo, shock_prob_hi, size=(shocked.size, 1)
        )
        event_rows, event_cols = np.nonzero(
            rng.random((shocked.size, L)) < event_prob
        )
        if event_rows.size:
            favored_sign = rng.choice(
                [-1.0, 1.0], size=(shocked.size, 1)
            )
            sign = np.where(
                rng.random(event_rows.size) < 0.75,
                favored_sign[event_rows, 0],
                -favored_sign[event_rows, 0],
            )
            magnitude = rng.lognormal(
                mean=np.log(4.0), sigma=0.6, size=event_rows.size
            )
            out[
                shocked[event_rows], event_cols
            ] += sign * magnitude * robust_scale[event_rows, 0]
        if preserve_nonnegative:
            np.maximum(out, 0.0, out=out)


    for row in np.nonzero(rng.random(n) < censor_rate)[0]:
        q = float(rng.uniform(0.03, 0.18))
        upper = rng.random() < 0.5
        if not allow_range_artifacts:
            continue
        calibration = out[row, :calibration_len]
        if upper:
            threshold = np.quantile(calibration, 1.0 - q)
            out[row] = np.minimum(out[row], threshold)
        else:
            threshold = np.quantile(calibration, q)
            out[row] = np.maximum(out[row], threshold)

    quantized = np.nonzero(rng.random(n) < quantize_rate)[0]
    if quantized.size:
        levels = rng.integers(16, 257, size=(quantized.size, 1))
        if allow_range_artifacts:
            x = out[quantized]
            calibration = x[:, :calibration_len]
            lo = calibration.min(axis=1, keepdims=True)
            hi = calibration.max(axis=1, keepdims=True)
            step = (hi - lo) / np.maximum(levels - 1, 1)
            safe_step = np.where(step < 1e-12, 1.0, step)
            clipped = np.clip(x, lo, hi)
            out[quantized] = (
                lo + np.rint((clipped - lo) / safe_step) * safe_step
            )


    held = np.nonzero(rng.random(n) < regular_hold_rate)[0]
    if held.size:
        factors = rng.choice([2, 4, 8], size=held.size, p=[0.55, 0.30, 0.15])
        for factor in (2, 4, 8):
            rows = held[factors == factor]
            if rows.size:
                out[rows] = np.repeat(
                    out[rows, ::factor], factor, axis=1
                )[:, :L]


    irregular = np.nonzero(rng.random(n) < irregular_hold_rate)[0]
    if irregular.size and L > 1:
        hold_prob = rng.uniform(
            irregular_hold_prob_lo,
            irregular_hold_prob_hi,
            size=(irregular.size, 1),
        )
        hold = rng.random((irregular.size, L)) < hold_prob
        hold[:, 0] = False
        source_index = np.where(
            ~hold, np.arange(L, dtype=np.int64)[None, :], 0
        )
        np.maximum.accumulate(source_index, axis=1, out=source_index)
        out[irregular] = np.take_along_axis(
            out[irregular], source_index, axis=1
        )
    if preserve_integers:
        out = np.maximum(np.rint(out), 0.0)


    degenerate = out[:, :calibration_len].std(axis=1) < 1e-9
    out[degenerate] = original[degenerate]
    return out


def _trend_seasonal_ar(rng: np.random.Generator, n: int, L: int, *,
                       hi_frac: float = 0.25, exc_lo: float = 0.4,
                       exc_hi: float = 3.0, clean_frac: float = 0.4,
                       clean_lo: float = 0.02, clean_hi: float = 0.12) -> np.ndarray:
    t = np.arange(L, dtype=np.float64)[None, :]
    level = rng.normal(0.0, 1.0, size=(n, 1))


    _hi = rng.random((n, 1)) < hi_frac
    exc = np.where(_hi, rng.normal(0.0, exc_hi, size=(n, 1)),
                   rng.normal(0.0, exc_lo, size=(n, 1)))
    tn = t / max(L - 1, 1)
    series = level + exc * tn + _seasonal(rng, n, L)
    phi = rng.uniform(0.0, 0.85, size=n)
    clean = rng.random((n, 1)) < clean_frac
    sigma = np.where(
        clean,
        rng.uniform(clean_lo, clean_hi, size=(n, 1)),
        rng.uniform(0.1, 0.6, size=(n, 1)),
    )
    innov = rng.normal(0.0, 1.0, size=(n, L)) * sigma
    return series + _ar1_batch(innov, phi)


def _regime_shift(rng: np.random.Generator, n: int, L: int) -> np.ndarray:


    level = np.cumsum(_sparse_jumps(rng, n, L, rate=3.0 / L, scale=2.0), axis=1)
    log_vol = np.cumsum(_sparse_jumps(rng, n, L, rate=3.0 / L, scale=0.5), axis=1)
    vol = np.exp(np.clip(log_vol, -3.0, 3.0)) * rng.uniform(0.1, 0.5, size=(n, 1))
    noise = rng.normal(0.0, 1.0, size=(n, L)) * vol
    seas = _seasonal(rng, n, L, k_max=2) * rng.uniform(0.0, 1.0, size=(n, 1))


    slope = rng.normal(0.0, 1.0 / L, size=(n, 1)) + np.cumsum(
        _sparse_jumps(rng, n, L, rate=2.0 / L, scale=4.0 / L), axis=1
    )
    piecewise_trend = np.cumsum(slope, axis=1)
    return level + piecewise_trend + seas + noise


def _multiplicative(rng: np.random.Generator, n: int, L: int, *,
                    hi_frac: float = 0.25, exc_lo: float = 0.3, exc_hi: float = 2.0) -> np.ndarray:
    t = np.arange(L, dtype=np.float64)[None, :]

    _hg = rng.random((n, 1)) < hi_frac
    gexc = np.where(_hg, rng.normal(0.0, exc_hi, size=(n, 1)),
                    rng.normal(0.0, exc_lo, size=(n, 1)))
    tn = t / max(L - 1, 1)
    base_level = np.exp(gexc * tn + rng.normal(0.0, 0.3, size=(n, 1)))
    amp = rng.uniform(0.1, 0.6, size=(n, 1))
    seasonal_shape = _seasonal(rng, n, L, k_max=1)
    seasonal_shape = _prefix_standardize(seasonal_shape, center=False)
    seas = 1.0 + amp * seasonal_shape
    noise = 1.0 + rng.normal(0.0, 1.0, size=(n, L)) * rng.uniform(0.02, 0.15, size=(n, 1))
    scale = rng.uniform(1.0, 50.0, size=(n, 1))
    return scale * base_level * np.clip(seas, 0.05, None) * np.clip(noise, 0.05, None)


def _ar2(rng: np.random.Generator, n: int, L: int) -> np.ndarray:


    p1 = rng.uniform(0.3, 0.98, size=n)
    p2 = rng.uniform(-0.6, 0.6, size=n)
    a2 = p2
    a1 = p1 * (1.0 - p2)
    sigma = rng.uniform(0.2, 0.8, size=(n, 1))
    burn = 512
    innov = rng.normal(0.0, 1.0, size=(n, L + burn)) * sigma


    return _ar2_batch(innov, a1, a2)[:, burn:]


def _integrated(
    rng: np.random.Generator,
    n: int,
    L: int,
    *,
    heavy_frac: float = 0.25,
    sv_frac: float = 0.30,
) -> np.ndarray:


    order2 = rng.random(n) < 0.35
    drift = rng.normal(0.0, 0.02, size=(n, 1))
    sigma = rng.uniform(0.2, 1.0, size=(n, 1))
    eps = rng.normal(0.0, 1.0, size=(n, L))
    heavy = np.nonzero(rng.random(n) < heavy_frac)[0]
    if heavy.size:
        df = rng.uniform(3.0, 12.0, size=(heavy.size, 1))
        eps[heavy] = rng.standard_t(df, size=(heavy.size, L)) / np.sqrt(
            df / (df - 2.0)
        )

    stochastic = np.nonzero(rng.random(n) < sv_frac)[0]
    if stochastic.size:
        phi = 0.995
        burn = 256
        vol_innov = (
            rng.standard_normal((stochastic.size, L + burn))
            * np.sqrt(1.0 - phi * phi)
        )
        log_vol = lfilter(
            [1.0],
            [1.0, -phi],
            vol_innov,
            axis=1,
        )[:, burn:]
        log_vol *= rng.uniform(0.10, 0.55, size=(stochastic.size, 1))
        eps[stochastic] *= np.exp(np.clip(log_vol, -2.0, 2.0))

    steps = eps * sigma + drift
    walk = np.cumsum(steps, axis=1)
    walk2 = np.cumsum(walk, axis=1)
    o2 = order2[:, None]


    return np.where(o2, walk2 / max(L, 1), walk)


def _threshold_ar(rng: np.random.Generator, n: int, L: int) -> np.ndarray:


    phi_hi = rng.uniform(0.3, 0.9, size=n)
    phi_lo = rng.uniform(-0.9, 0.3, size=n)
    const_hi = rng.normal(0.0, 0.3, size=n)
    const_lo = rng.normal(0.0, 0.3, size=n)
    sigma = rng.uniform(0.2, 0.7, size=(n, 1))
    burn = 256
    total = L + burn
    innov = rng.normal(0.0, 1.0, size=(n, total)) * sigma
    x = np.empty((n, total), dtype=np.float64)
    x[:, 0] = innov[:, 0]
    for t in range(1, total):
        prev = x[:, t - 1]
        hi = prev >= 0.0
        phi = np.where(hi, phi_hi, phi_lo)
        const = np.where(hi, const_hi, const_lo)
        x[:, t] = np.clip(const + phi * prev + innov[:, t], -1e6, 1e6)
    return x[:, burn:]


def _chaotic(rng: np.random.Generator, n: int, L: int) -> np.ndarray:


    use_sine = rng.random(n) < 0.5
    r_log = rng.uniform(3.6, 4.0, size=n)
    r_sin = rng.uniform(0.85, 1.0, size=n)
    x0 = rng.uniform(0.05, 0.95, size=n)
    cur = x0.copy()
    for _ in range(64):
        nxt_log = r_log * cur * (1.0 - cur)
        nxt_sin = r_sin * np.sin(np.pi * cur)
        cur = np.clip(np.where(use_sine, nxt_sin, nxt_log), 0.0, 1.0)
    x = np.empty((n, L), dtype=np.float64)
    x[:, 0] = cur
    for t in range(1, L):
        nxt_log = r_log * cur * (1.0 - cur)
        nxt_sin = r_sin * np.sin(np.pi * cur)
        cur = np.where(use_sine, nxt_sin, nxt_log)
        cur = np.clip(cur, 0.0, 1.0)
        x[:, t] = cur
    return _prefix_standardize(x)


def _spectral_gp(rng: np.random.Generator, n: int, L: int) -> np.ndarray:


    embed = 2 * L
    f = np.fft.rfftfreq(embed)[None, :]
    lengthscale = np.exp(rng.uniform(np.log(8.0), np.log(256.0), size=(n, 1)))
    envelope = np.exp(-0.5 * (2.0 * np.pi * lengthscale * f) ** 2)
    z = rng.standard_normal((n, f.shape[1])) + 1j * rng.standard_normal((n, f.shape[1]))
    z[:, 0] = 0.0
    x = np.fft.irfft(z * np.sqrt(envelope), n=embed, axis=1)[:, :L]
    return _prefix_standardize(x)


def _long_memory(rng: np.random.Generator, n: int, L: int) -> np.ndarray:


    embed = 2 * L
    f = np.fft.rfftfreq(embed)
    safe_f = np.maximum(f, 1.0 / embed)[None, :]
    beta = rng.uniform(-0.6, 2.4, size=(n, 1))
    amp = safe_f ** (-0.5 * beta)


    multiscale = rng.random((n, 1)) < 0.4
    split_idx = rng.integers(8, max(9, f.size // 3), size=(n, 1))
    split_f = np.maximum(split_idx / embed, 1.0 / embed)
    beta_hi = rng.uniform(-0.6, 2.8, size=(n, 1))
    above = np.arange(f.size)[None, :] > split_idx
    amp_hi = split_f ** (-0.5 * beta) \
        * (safe_f / split_f) ** (-0.5 * beta_hi)
    amp = np.where(multiscale & above, amp_hi, amp)
    amp[:, 0] = 0.0
    z = rng.standard_normal((n, f.size)) + 1j * rng.standard_normal((n, f.size))
    x = np.fft.irfft(z * amp, n=embed, axis=1)[:, :L]
    integrate = rng.random(n) < 0.25
    if integrate.any():
        x[integrate] = np.cumsum(x[integrate], axis=1)
    return _prefix_standardize(x)


def _ou_stochastic_vol(rng: np.random.Generator, n: int, L: int) -> np.ndarray:


    switch_rate = np.exp(rng.uniform(np.log(0.001), np.log(0.15), size=(n, 1)))
    switches = rng.random((n, L)) < switch_rate
    switches[:, 0] = rng.random(n) < 0.5
    regime = np.bitwise_and(np.cumsum(switches, axis=1), 1).astype(np.int8)


    slow = rng.random((n, 1)) < 0.5
    phi = np.where(
        slow,
        rng.uniform(0.995, 0.9995, size=(n, 1)),
        rng.uniform(0.90, 0.99, size=(n, 1)),
    )
    mu0 = rng.normal(-2.0, 1.0, size=(n, 1))
    mu1 = rng.normal(2.0, 1.0, size=(n, 1))
    mean = np.where(regime == 0, mu0, mu1)
    seasonal_on = rng.random((n, 1)) < 0.6
    mean += seasonal_on * _seasonal(rng, n, L, k_max=3) \
        * rng.uniform(0.5, 3.0, size=(n, 1))

    log_sigma0 = rng.normal(np.log(0.3), 0.3, size=(n, 1))
    log_sigma1 = rng.normal(np.log(1.5), 0.5, size=(n, 1))
    log_sigma_mean = np.where(regime == 0, log_sigma0, log_sigma1)
    vol_rho = rng.uniform(0.951, 0.995, size=(n, 1))
    vol_eta = rng.uniform(0.03, 0.20, size=(n, 1))
    vol_eps = rng.standard_normal((n, L))
    vol_drive = (
        (1.0 - vol_rho) * log_sigma_mean
        + np.sqrt(1.0 - vol_rho * vol_rho) * vol_eta * vol_eps
    )
    log_vol = np.empty((n, L), dtype=np.float64)
    log_vol[:, 0] = log_sigma_mean[:, 0]
    for i in range(n):
        rho = float(vol_rho[i, 0])
        log_vol[i, 1:] = lfilter(
            [1.0],
            [1.0, -rho],
            vol_drive[i, 1:],
            zi=[rho * log_vol[i, 0]],
        )[0]
    vol = np.exp(np.clip(log_vol, -5.0, 5.0))

    eps = rng.standard_normal((n, L))
    heavy = np.nonzero(rng.random(n) < 0.35)[0]
    if heavy.size:


        eps[heavy] = (
            rng.standard_t(4.0, size=(heavy.size, L)) / np.sqrt(2.0)
        )
    shocks = rng.random((n, L)) < (3.0 / L)
    shock_rows, shock_cols = np.nonzero(shocks)

    eps[shock_rows, shock_cols] += rng.normal(
        0.0, 5.0, size=shock_rows.size
    )

    innovation_scale = np.sqrt(np.maximum(1.0 - phi * phi, 1e-6))
    drive = (1.0 - phi) * mean + innovation_scale * vol * eps
    out = np.empty((n, L), dtype=np.float64)
    out[:, 0] = mean[:, 0] + vol[:, 0] * eps[:, 0]
    for i in range(n):
        p = float(phi[i, 0])
        out[i, 1:] = lfilter(
            [1.0], [1.0, -p], drive[i, 1:], zi=[p * out[i, 0]]
        )[0]

    scale = np.exp(rng.uniform(np.log(0.1), np.log(50.0), size=(n, 1)))
    shift = rng.uniform(-100.0, 100.0, size=(n, 1))
    return out * scale + shift


def _physical_sensors(rng: np.random.Generator, n: int, L: int) -> np.ndarray:


    seasonal = _seasonal(rng, n, L, k_max=2)
    smooth = _spectral_gp(rng, n, L)
    fronts = np.cumsum(
        _sparse_jumps(rng, n, L, rate=5.0 / L, scale=1.0), axis=1
    )
    base = (
        seasonal * rng.uniform(0.3, 2.0, size=(n, 1))
        + smooth * rng.uniform(0.2, 1.2, size=(n, 1))
        + fronts * rng.uniform(0.2, 1.0, size=(n, 1))
    )

    kind = rng.integers(0, 4, size=n)
    out = base.copy()

    bounded = kind == 1
    if bounded.any():
        gain = rng.uniform(0.8, 3.5, size=(int(bounded.sum()), 1))
        midpoint = rng.uniform(-0.8, 0.8, size=(int(bounded.sum()), 1))
        out[bounded] = 100.0 / (1.0 + np.exp(-gain * (base[bounded] - midpoint)))

    pressure = kind == 2
    if pressure.any():
        count = int(pressure.sum())


        diffusion = np.exp(
            rng.uniform(np.log(0.03), np.log(0.20), size=(count, 1))
        )
        walk = np.cumsum(
            rng.standard_normal((count, L)) * diffusion, axis=1
        )
        level = rng.uniform(900.0, 1100.0, size=(count, 1))
        out[pressure] = (
            level + walk + 2.0 * fronts[pressure] + 0.5 * seasonal[pressure]
        )

    magnitude = kind == 3
    if magnitude.any():
        count = int(magnitude.sum())
        gusts = (rng.random((count, L)) < (8.0 / L)) \
            * rng.lognormal(0.0, 0.8, size=(count, L))
        power = rng.uniform(1.0, 1.6, size=(count, 1))
        out[magnitude] = np.abs(base[magnitude]) ** power + gusts

    return out


def _seasonal_counts(rng: np.random.Generator, n: int, L: int) -> np.ndarray:


    t = np.arange(L, dtype=np.float64)[None, :]
    period = rng.choice(
        _SEASONAL_PERIODS, size=(n, 1), p=_SEASONAL_PROBS
    )
    phase = rng.uniform(0.0, 2.0 * np.pi, size=(n, 1))
    amp = rng.uniform(0.15, 0.8, size=(n, 1))
    log_rate = amp * np.sin(2.0 * np.pi * t / period + phase)
    second = rng.random((n, 1)) < 0.55
    log_rate += second * (0.5 * amp) * np.sin(
        4.0 * np.pi * t / period + rng.uniform(0.0, 2.0 * np.pi, size=(n, 1))
    )


    calendar = rng.random((n, 1)) < 0.35
    day_period = rng.choice([24, 48, 96, 144], size=(n, 1))
    day_idx = (np.floor_divide(np.arange(L)[None, :], day_period) % 7).astype(np.int64)
    day_factors = rng.normal(0.0, 0.12, size=(n, 7))
    day_factors[:, 5:] += rng.uniform(-0.8, 0.3, size=(n, 1))
    calendar_effect = np.take_along_axis(day_factors, day_idx, axis=1)
    log_rate += calendar * calendar_effect
    excursion = rng.uniform(-0.5, 0.5, size=(n, 1))
    log_rate += excursion * t / max(L - 1, 1)


    impulses = (
        (rng.random((n, L)) < (2.0 / L))
        * rng.uniform(1.0, 10.0, size=(n, L))
    )
    burst = _ar1_batch(impulses, rng.uniform(0.85, 0.995, size=(n, 1)))
    base = np.exp(rng.uniform(np.log(3.0), np.log(3000.0), size=(n, 1)))
    lam = base * np.exp(np.clip(log_rate, -5.0, 5.0)) * (1.0 + burst)
    np.clip(lam, 0.0, 1.0e7, out=lam)


    overdispersed = rng.random((n, 1)) < 0.5
    shape = rng.uniform(0.5, 4.0, size=(n, 1))
    mixed = lam * rng.gamma(shape, 1.0 / shape, size=(n, L))
    return rng.poisson(np.where(overdispersed, mixed, lam)).astype(np.float64)


def _intermittent(rng: np.random.Generator, n: int, L: int) -> np.ndarray:


    t = np.arange(L, dtype=np.float64)[None, :]
    base_p = rng.uniform(0.03, 0.35, size=(n, 1))
    period = rng.choice([7.0, 12.0, 24.0, 48.0, 168.0], size=(n, 1))
    season = rng.uniform(0.2, 1.2, size=(n, 1)) * np.sin(
        2.0 * np.pi * t / period + rng.uniform(0.0, 2.0 * np.pi, size=(n, 1))
    )
    logit = np.log(base_p / (1.0 - base_p)) + season
    p = 1.0 / (1.0 + np.exp(-logit))
    occur = (rng.random((n, L)) < p).astype(np.float64)
    magnitude = np.maximum(
        1.0,
        np.rint(
        rng.gamma(shape=2.0, scale=1.0, size=(n, L))
        * rng.uniform(1.0, 10.0, size=(n, 1))
        * np.exp(0.25 * season)
        ),
    )
    return occur * magnitude


def _pulse_outlier(rng: np.random.Generator, n: int, L: int) -> np.ndarray:


    base = _spectral_gp(rng, n, L) * rng.uniform(0.5, 2.0, size=(n, 1))
    base += _seasonal(rng, n, L, k_max=1) * rng.uniform(0.0, 1.0, size=(n, 1))
    sharp = _sparse_jumps(
        rng, n, L, rate=3.0 / L, scale=rng.uniform(3.0, 8.0, size=n)
    )
    impulses = _sparse_jumps(
        rng, n, L, rate=2.0 / L, scale=rng.uniform(2.0, 7.0, size=n)
    )
    recovery = _ar1_batch(impulses, rng.uniform(0.75, 0.995, size=n))
    series = base + sharp + recovery


    starts = rng.random((n, L)) < (2.0 / L)
    starts[:, 0] = False
    for row in range(n):
        for start in np.nonzero(starts[row])[0]:
            run = int(rng.integers(3, 65))
            end = min(int(start) + run, L)
            series[row, start:end] = series[row, start - 1]
    return series


_CS_CALM_LO = 192
_CS_CALM_HI = 1024
_CS_DYNAMIC_LO = 96
_CS_DYNAMIC_HI = 512


def _conditional_stability(
    rng: np.random.Generator, n: int, L: int
) -> np.ndarray:


    if n <= 0:
        return np.empty((0, L), dtype=np.float64)
    if L <= 0:
        return np.empty((n, 0), dtype=np.float64)

    seasonal = _seasonal(rng, n, L, k_max=2)
    ar = _ar1_batch(
        rng.normal(size=(n, L)) * rng.uniform(0.12, 0.55, size=(n, 1)),
        rng.uniform(0.35, 0.92, size=n),
    )
    smooth = _spectral_gp(rng, n, L)
    walk = np.cumsum(
        rng.normal(size=(n, L)) * rng.uniform(0.025, 0.16, size=(n, 1)),
        axis=1,
    )
    ingredients = (seasonal, ar, smooth, walk)

    kind = rng.integers(0, len(ingredients), size=n)
    calm_kind = rng.integers(0, 4, size=n)
    level = rng.normal(0.0, 2.0, size=n)
    scale = np.exp(rng.uniform(np.log(0.4), np.log(12.0), size=n))
    dynamic_amp = rng.uniform(0.6, 2.2, size=n)
    start_calm = rng.random(n) < 0.65
    out = np.empty((n, L), dtype=np.float64)

    for row in range(n):
        dynamic = ingredients[int(kind[row])][row].copy()
        calibration = dynamic[: min(L, 512)]
        calibration_mean = float(calibration.mean())
        calibration_std = float(calibration.std())
        dynamic -= calibration_mean
        if calibration_std > 1e-12:
            dynamic /= calibration_std

        current = float(level[row])
        calm = bool(start_calm[row])
        pos = 0
        segment_index = 0
        mode2_seen = False
        while pos < L:
            if calm:
                seg_len = int(rng.integers(_CS_CALM_LO, _CS_CALM_HI + 1))
            else:
                seg_len = int(
                    rng.integers(_CS_DYNAMIC_LO, _CS_DYNAMIC_HI + 1)
                )


            if segment_index == 0 and L >= 2 * _CS_DYNAMIC_LO:
                seg_len = min(seg_len, L - _CS_DYNAMIC_LO)
            end = min(pos + max(seg_len, 1), L)
            span = end - pos

            if calm:
                mode = int(calm_kind[row])
                if mode == 0:
                    values = np.full(span, current)
                elif mode == 1:


                    drift = rng.normal(0.0, 0.0025, size=span).cumsum()
                    drift += np.linspace(
                        0.0, float(rng.normal(0.0, 0.025)), span
                    )
                    values = current + drift
                elif mode == 2:

                    if not mode2_seen:
                        count_level = max(0.0, float(np.rint(abs(current) * 8.0)))
                        mode2_seen = True
                    else:
                        count_level = max(0.0, float(np.rint(current)))
                    updates = rng.random(span) < 0.025
                    changes = updates * rng.choice(
                        [-1.0, 1.0], size=span
                    )
                    values = np.maximum(
                        count_level + np.cumsum(changes), 0.0
                    )
                else:


                    events = rng.random(span) < 0.012
                    values = events * rng.gamma(1.5, 0.35, size=span)
            else:
                piece = dynamic[pos:end] * float(dynamic_amp[row])
                values = piece - piece[0] + current
                if span > 1 and np.ptp(values) < 1e-10:
                    values = current + np.linspace(0.0, 1.0, span)

            out[row, pos:end] = values
            current = float(values[-1])
            pos = end
            calm = not calm
            segment_index += 1


        row_scale = 1.0 if int(calm_kind[row]) == 2 else float(scale[row])
        out[row] *= row_scale


        if L > 1 and np.ptp(out[row]) < 1e-10:
            out[row, -1] += max(1e-3, 0.01 * row_scale)
    return out


def _sanitize(block: np.ndarray) -> np.ndarray:


    x = np.asarray(block, dtype=np.float64)
    np.nan_to_num(x, copy=False, nan=0.0, posinf=1e6, neginf=-1e6)
    if x.ndim == 1:
        peak = float(np.max(np.abs(x)))
        if peak > 1e6:
            x *= 1e6 / peak
    else:
        peak = np.max(np.abs(x), axis=1, keepdims=True)
        scale = np.where(peak > 1e6, 1e6 / np.maximum(peak, 1e-12), 1.0)
        x *= scale
    return x
