from __future__ import annotations
import json
from collections.abc import Iterator
from functools import lru_cache, partial
from pathlib import Path
from queue import Full, Queue
from threading import Event, Lock, Thread
import numpy as np
from numba import njit
from scipy.signal import lfilter
from cascade.interface import DataGenerator
_CHUNK = 2048
_PATCH_SIZE = 32
_SEASONAL_PERIODS = np.array([4, 7, 12, 15, 24, 30, 48, 52, 60, 90, 96, 144, 168, 183, 240, 288, 336, 365, 672, 730], dtype=np.float64)
_SEASONAL_PROBS = np.array([0.01, 0.23, 0.02, 0.02, 0.07, 0.01, 0.06, 0.01, 0.08, 0.01, 0.14, 0.07, 0.04, 0.01, 0.08, 0.08, 0.02, 0.02, 0.01, 0.01], dtype=np.float64)
_SEASONAL_PROBS /= _SEASONAL_PROBS.sum()
_SEASONAL_PAIRS = np.array([[15, 60], [60, 240], [24, 168], [48, 336], [96, 672], [7, 365], [12, 52]], dtype=np.float64)
_FAMILIES: tuple[str, ...] = ('k00', 'k01', 'k02', 'k03', 'k04', 'k05', 'k06', 'k07', 'k08', 'k09', 'k10', 'k11', 'k12', 'k13', 'k14', 'k15', 'k16', 'k17', 'k18', 'k19', 'k20', 'k21', 'k22')
_DEFAULT_WEIGHTS: dict[str, float] = {'k00': 0.095, 'k01': 0.095, 'k02': 0.06, 'k03': 0.105, 'k04': 0.095, 'k05': 0.06, 'k06': 0.02, 'k07': 0.07, 'k08': 0.07, 'k09': 0.08, 'k10': 0.08, 'k11': 0.06, 'k12': 0.02, 'k13': 0.02, 'k14': 0.07, 'k15': 0.0, 'k16': 0.0, 'k17': 0.0, 'k18': 0.0, 'k19': 0.0, 'k20': 0.0, 'k21': 0.0, 'k22': 0.0}
_CLEAN: frozenset[str] = frozenset({'k15', 'k18', 'k17', 'k19'})
_NUMBA_WARMED = False
_NUMBA_WARM_LOCK = Lock()

def _k15(rng: np.random.Generator, n: int, L: int) -> np.ndarray:
    jumps = rng.uniform(1.0, 8.0, size=(n, 1))
    at = rng.random((n, L)) < jumps / max(L, 1)
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

def _k16(rng: np.random.Generator, n: int, L: int) -> np.ndarray:
    drive = _ar1_batch(rng.normal(0.0, 1.0, size=(n, L)), rng.uniform(0.99, 0.9995, size=(n, 1)))
    drive = (drive - drive.mean(axis=1, keepdims=True)) / np.maximum(drive.std(axis=1, keepdims=True), 1e-09)
    hot = drive > rng.uniform(0.2, 1.2, size=(n, 1))
    lo = rng.uniform(0.02, 0.2, size=(n, 1))
    ratio = rng.uniform(4.0, 25.0, size=(n, 1))
    sd = np.where(hot, lo * ratio, lo)
    x = _ar1_batch(rng.normal(0.0, 1.0, size=(n, L)) * sd, rng.uniform(0.9, 0.999, size=(n, 1)))
    scale = np.exp(rng.uniform(np.log(1.0), np.log(500.0), size=(n, 1)))
    return x * scale + rng.uniform(-1.0, 1.0, size=(n, 1)) * scale

def _k17(rng: np.random.Generator, n: int, L: int) -> np.ndarray:
    t = np.arange(L, dtype=np.float64)[None, :]
    day = rng.choice(np.array([24.0, 48.0, 96.0, 144.0]), size=(n, 1))
    week = day * 7.0
    amp_w = rng.uniform(0.4, 1.6, size=(n, 1))
    amp_d = rng.uniform(0.3, 1.4, size=(n, 1))
    y = amp_w * np.sin(2.0 * np.pi * t / week + rng.uniform(0, 2 * np.pi, (n, 1)))
    y = y + amp_d * np.sin(2.0 * np.pi * t / day + rng.uniform(0, 2 * np.pi, (n, 1)))
    y = y + 0.35 * amp_d * np.sin(4.0 * np.pi * t / day + rng.uniform(0, 2 * np.pi, (n, 1)))
    drift = rng.uniform(-0.3, 0.3, size=(n, 1)) * t / max(L - 1, 1)
    noise = rng.normal(0.0, 1.0, size=(n, L)) * rng.uniform(0.02, 0.15, size=(n, 1))
    base = np.exp(rng.uniform(np.log(5.0), np.log(5000.0), size=(n, 1)))
    out = base * np.exp(np.clip(y * 0.4 + drift + noise, -6.0, 6.0))
    counts = rng.random((n, 1)) < 0.45
    return np.where(counts, np.rint(out), out)

def _k18(rng: np.random.Generator, n: int, L: int) -> np.ndarray:
    t = np.arange(L, dtype=np.float64)[None, :]
    k = int(rng.integers(3, 6))
    out = np.zeros((n, L), dtype=np.float64)
    base = rng.uniform(10.0, 400.0, size=(n, 1))
    for _ in range(k):
        period = base * rng.uniform(0.31, 2.7, size=(n, 1))
        out += rng.uniform(0.2, 1.0, size=(n, 1)) * np.sin(2.0 * np.pi * t / period + rng.uniform(0, 2 * np.pi, size=(n, 1)))
    sd = rng.uniform(0.005, 0.05, size=(n, 1))
    scale = np.exp(rng.uniform(np.log(1.0), np.log(1000.0), size=(n, 1)))
    out = out + rng.normal(0.0, 1.0, size=(n, L)) * sd
    return out * scale + rng.uniform(-1.0, 1.0, size=(n, 1)) * scale

def _k19(rng: np.random.Generator, n: int, L: int) -> np.ndarray:
    rate = rng.uniform(1.0, 25.0, size=(n, 1)) / max(L, 1)
    hits = (rng.random((n, L)) < rate).astype(np.float64)
    mag = rng.gamma(2.0, 1.0, size=(n, L)) * rng.uniform(1.0, 12.0, size=(n, 1))
    decay = rng.uniform(0.9, 0.998, size=(n, 1))
    flow = _ar1_batch(hits * mag, decay)
    baseflow = rng.uniform(0.03, 0.6, size=(n, 1))
    scale = np.exp(rng.uniform(np.log(1.0), np.log(800.0), size=(n, 1)))
    sd = rng.uniform(0.0, 0.02, size=(n, 1))
    out = (flow + baseflow) * scale
    return np.maximum(out * (1.0 + rng.normal(0.0, 1.0, (n, L)) * sd), 0.0)

def _k20(rng: np.random.Generator, n: int, L: int) -> np.ndarray:
    cap = rng.integers(4, 80, size=(n, 1)).astype(np.float64)
    step = rng.normal(0.0, 1.0, size=(n, L)) * rng.uniform(0.01, 0.12, size=(n, 1)) * cap
    walk = np.cumsum(step, axis=1) + rng.uniform(0.0, 1.0, size=(n, 1)) * cap
    span = 2.0 * cap
    folded = cap - np.abs(np.mod(walk, span) - cap)
    quiet = rng.random((n, 1)) < 0.35
    folded = np.where(quiet, folded, folded + rng.normal(0.0, 0.35, size=(n, L)))
    return np.clip(np.rint(folded), 0.0, cap)

def _k21(rng: np.random.Generator, n: int, L: int) -> np.ndarray:
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


def _k22(rng: np.random.Generator, n: int, L: int) -> np.ndarray:
    """Small-capacity occupancy counters: docking stations, bays, berths, slots.

    Measured against the real cluster, which is 64% of the current eval pool:
    integer capacity with a median of 8 (range 1-34), a change on only ~2% of
    steps, EVERY change exactly +/-1, plateaus running past a hundred steps,
    and roughly a fifth of all time pinned at capacity.

    That is a clipped birth-death walk, not a reflected random walk. The
    distinction is the whole point: reflection bounces off the boundary and
    keeps moving, whereas clipping PINS at it, which is where a fifth of the
    probability mass actually lives. And because the series is stationary for
    long stretches, the correct forecast is usually "exactly where it is now" —
    a prior that moves more than reality teaches the model to over-disperse,
    and over-dispersion is paid for twice, once in the quantile loss and again
    in the scaled error.
    """
    if n <= 0:
        return np.empty((0, L), dtype=np.float64)
    # capacity: small, right-skewed; median ~8
    cap = np.clip(np.rint(np.exp(rng.normal(np.log(8.0), 0.7, size=n))), 1.0, 60.0)
    # change rate per step, log-uniform around the measured 2%
    p = np.exp(rng.uniform(np.log(0.004), np.log(0.12), size=n))
    # asymmetric pressure: stations that fill and stay full vs ones that drain
    up = rng.uniform(0.35, 0.68, size=n)
    # slow demand cycle modulates activity (commute rhythm)
    t = np.arange(L, dtype=np.float64)[None, :]
    per = np.exp(rng.uniform(np.log(60.0), np.log(600.0), size=(n, 1)))
    act = 1.0 + rng.uniform(0.0, 0.8, size=(n, 1)) * np.sin(
        2.0 * np.pi * t / per + rng.uniform(0, 2 * np.pi, size=(n, 1)))
    rate = np.clip(p[:, None] * act, 0.0, 0.95)
    moves = rng.random((n, L)) < rate
    dirs = np.where(rng.random((n, L)) < up[:, None], 1.0, -1.0)
    steps = moves * dirs
    # rare rebalancing: a van tops up or empties a station in one go
    reb = rng.random((n, L)) < (rng.uniform(0.0, 3.0, size=(n, 1)) / max(L, 1))
    reb_mag = np.rint(rng.uniform(0.2, 0.6, size=(n, 1)) * cap[:, None]) \
        * np.where(rng.random((n, L)) < 0.5, 1.0, -1.0)
    steps = steps + reb * reb_mag
    # CLIPPED walk: sequential so the boundary pins rather than reflects.
    out = np.empty((n, L), dtype=np.float64)
    cur = np.rint(rng.uniform(0.0, 1.0, size=n) * cap)
    for j in range(L):
        cur = np.clip(cur + steps[:, j], 0.0, cap)
        out[:, j] = cur
    # a minority are reported as a percentage of capacity rather than a count
    pct = rng.random(n) < 0.15
    if pct.any():
        out[pct] = np.rint(out[pct] / np.maximum(cap[pct][:, None], 1.0) * 100.0)
    return out


def _validate_parameters(parameters: dict[str, float], label: str) -> None:
    if not all((np.isfinite(value) for value in parameters.values())):
        raise ValueError(f'{label} must contain only finite values')
    probability_names = {'sa_clean_frac', 'integrated_heavy_frac', 'integrated_sv_frac', *(key for key in parameters if key.startswith(('observation.', 'augment.')))}
    for name in probability_names:
        if not 0.0 <= parameters[name] <= 1.0:
            raise ValueError(f'{label}.{name} must be in [0, 1]')
    for name in ('tr_exc_lo', 'tr_exc_hi', 'gr_exc_lo', 'gr_exc_hi', 'sa_clean_lo', 'sa_clean_hi'):
        if parameters[name] < 0.0:
            raise ValueError(f'{label}.{name} must be non-negative')
    for lo_name, hi_name in (('tr_exc_lo', 'tr_exc_hi'), ('gr_exc_lo', 'gr_exc_hi'), ('sa_clean_lo', 'sa_clean_hi'), ('observation.irregular_hold_prob_lo', 'observation.irregular_hold_prob_hi'), ('observation.shock_prob_lo', 'observation.shock_prob_hi')):
        if parameters[lo_name] > parameters[hi_name]:
            raise ValueError(f'{label}.{lo_name} must be <= {hi_name}')

class Generator(DataGenerator):

    def __init__(self, config_dir: str, *, seed: int) -> None:
        cfg_path = Path(config_dir) / 'config.json'
        cfg = json.loads(cfg_path.read_text(encoding='utf-8')) if cfg_path.is_file() else {}
        self._cfg = cfg
        self._seed = int(seed)
        self._min_len = int(cfg.get('min_length', 64))
        self._max_len = int(cfg.get('max_length', 4096))
        if self._min_len < 1 or self._max_len < self._min_len:
            raise ValueError(f'invalid length band [{self._min_len}, {self._max_len}]')
        self._length_buckets = np.asarray(
            cfg.get('length_buckets', [self._max_len]), dtype=np.int64
        )
        length_cycle = dict(cfg.get('length_cycle', {}))
        self._length_peak_probs = np.asarray(
            length_cycle.get('peak_probs', [1.0]), dtype=np.float64
        )
        self._length_cycle_start = float(length_cycle.get('start_fraction', 0.0))
        self._length_cycle_end = float(length_cycle.get('end_fraction', 1.0))
        self._length_cycles = int(length_cycle.get('cycles', 1))
        self._length_expected_budget_fraction = float(
            length_cycle.get('expected_budget_fraction', 0.65)
        )
        if (
            self._length_buckets.ndim != 1
            or self._length_peak_probs.ndim != 1
            or self._length_buckets.size == 0
            or self._length_buckets.size != self._length_peak_probs.size
        ):
            raise ValueError(
                'length_buckets and length_cycle.peak_probs must be nonempty '
                '1-D arrays of equal size'
            )
        if (
            np.any(self._length_buckets < self._min_len)
            or np.any(self._length_buckets > self._max_len)
            or np.any(self._length_buckets % _PATCH_SIZE != 0)
            or np.unique(self._length_buckets).size != self._length_buckets.size
        ):
            raise ValueError(
                'length_buckets must be unique 32-point multiples inside '
                '[min_length, max_length]'
            )
        if (
            not np.all(np.isfinite(self._length_peak_probs))
            or np.any(self._length_peak_probs < 0.0)
            or self._length_peak_probs.sum() <= 0.0
        ):
            raise ValueError(
                'length_cycle.peak_probs must be finite, non-negative, '
                'and not all zero'
            )
        self._length_peak_probs /= self._length_peak_probs.sum()
        if not 0.0 <= self._length_cycle_start < self._length_cycle_end <= 1.0:
            raise ValueError(
                'length cycle fractions must satisfy '
                '0 <= start_fraction < end_fraction <= 1'
            )
        if not 1 <= self._length_cycles <= 8:
            raise ValueError('length_cycle.cycles must be in [1, 8]')
        if (
            not np.isfinite(self._length_expected_budget_fraction)
            or not 0.0 < self._length_expected_budget_fraction <= 1.0
        ):
            raise ValueError(
                'length_cycle.expected_budget_fraction must be in (0, 1]'
            )
        weights = dict(_DEFAULT_WEIGHTS)
        for k, v in dict(cfg.get('family_weights', {})).items():
            if k in weights:
                weights[k] = float(v)
        w = np.asarray([weights[f] for f in _FAMILIES], dtype=np.float64)
        if not np.all(np.isfinite(w)) or w.min() < 0 or w.sum() <= 0:
            raise ValueError('family_weights must be finite, non-negative, and not all zero')
        self._weights = w / w.sum()
        curriculum = dict(cfg.get('curriculum', {}))
        self._curriculum_enabled = bool(curriculum.get('enabled', False))
        self._expected_budget_fraction = float(
            curriculum.get(
                'expected_budget_fraction',
                self._length_expected_budget_fraction,
            )
        )
        if not np.isfinite(self._expected_budget_fraction) or not 0.0 < self._expected_budget_fraction <= 1.0:
            raise ValueError('curriculum.expected_budget_fraction must be in (0, 1]')
        if not np.isclose(
            self._expected_budget_fraction,
            self._length_expected_budget_fraction,
        ):
            raise ValueError(
                'curriculum and length_cycle expected_budget_fraction must match'
            )
        self._curriculum_start = float(curriculum.get('start_fraction', 0.1))
        self._curriculum_end = float(curriculum.get('end_fraction', 0.7))
        if not 0.0 <= self._curriculum_start < self._curriculum_end <= 1.0:
            raise ValueError('curriculum fractions must satisfy 0 <= start_fraction < end_fraction <= 1')
        start_weights = dict(weights)
        for k, v in dict(curriculum.get('start_family_weights', {})).items():
            if k in start_weights:
                start_weights[k] = float(v)
        start_w = np.asarray([start_weights[f] for f in _FAMILIES], dtype=np.float64)
        if not np.all(np.isfinite(start_w)) or start_w.min() < 0 or start_w.sum() <= 0:
            raise ValueError('curriculum.start_family_weights must be finite, non-negative, and not all zero')
        self._start_weights = start_w / start_w.sum()
        self._tr_hi_frac = float(cfg.get('tr_hi_frac', 0.25))
        self._prefetch_depth = int(cfg.get('prefetch_depth', 2))
        if not 1 <= self._prefetch_depth <= 4:
            raise ValueError('prefetch_depth must be in [1, 4]')
        augment = dict(cfg.get('augment', {}))
        observation = dict(cfg.get('observation', {}))
        self._parameters = {'tr_exc_lo': float(cfg.get('tr_exc_lo', 0.4)), 'tr_exc_hi': float(cfg.get('tr_exc_hi', 3.0)), 'gr_exc_lo': float(cfg.get('gr_exc_lo', 0.3)), 'gr_exc_hi': float(cfg.get('gr_exc_hi', 2.0)), 'sa_clean_frac': float(cfg.get('sa_clean_frac', 0.4)), 'sa_clean_lo': float(cfg.get('sa_clean_lo', 0.02)), 'sa_clean_hi': float(cfg.get('sa_clean_hi', 0.12)), 'integrated_heavy_frac': float(cfg.get('integrated_heavy_frac', 0.25)), 'integrated_sv_frac': float(cfg.get('integrated_sv_frac', 0.3)), 'observation.censor_rate': float(observation.get('censor_rate', 0.06)), 'observation.quantize_rate': float(observation.get('quantize_rate', 0.07)), 'observation.regular_hold_rate': float(observation.get('regular_hold_rate', 0.04)), 'observation.irregular_hold_rate': float(observation.get('irregular_hold_rate', 0.0)), 'observation.irregular_hold_prob_lo': float(observation.get('irregular_hold_prob_lo', 0.01)), 'observation.irregular_hold_prob_hi': float(observation.get('irregular_hold_prob_hi', 0.1)), 'observation.shock_row_rate': float(observation.get('shock_row_rate', 0.0)), 'observation.shock_prob_lo': float(observation.get('shock_prob_lo', 0.001)), 'observation.shock_prob_hi': float(observation.get('shock_prob_hi', 0.015)), 'augment.tsmixup': float(augment.get('tsmixup', 0.0)), 'augment.pad_prefix': float(augment.get('pad_prefix', 0.0))}
        start_parameters = dict(curriculum.get('start_parameters', {}))
        start_observation = dict(start_parameters.pop('observation', {}))
        start_augment = dict(start_parameters.pop('augment', {}))
        known_top_level = {key for key in self._parameters if '.' not in key}
        unknown = set(start_parameters) - known_top_level
        unknown.update((f'observation.{key}' for key in start_observation if f'observation.{key}' not in self._parameters))
        unknown.update((f'augment.{key}' for key in start_augment if f'augment.{key}' not in self._parameters))
        if unknown:
            names = ', '.join(sorted(unknown))
            raise ValueError(f'unknown curriculum.start_parameters: {names}')
        self._start_parameters = dict(self._parameters)
        for key, value in start_parameters.items():
            self._start_parameters[key] = float(value)
        for key, value in start_observation.items():
            self._start_parameters[f'observation.{key}'] = float(value)
        for key, value in start_augment.items():
            self._start_parameters[f'augment.{key}'] = float(value)
        _validate_parameters(self._parameters, 'final parameters')
        _validate_parameters(self._start_parameters, 'curriculum.start_parameters')

    @property
    def name(self) -> str:
        return str(self._cfg.get('name', ''))

    def _blend_at(self, token_progress: float) -> float:
        if not self._curriculum_enabled:
            return 1.0
        position = (token_progress - self._curriculum_start) / (self._curriculum_end - self._curriculum_start)
        position = float(np.clip(position, 0.0, 1.0))
        return position * position * (3.0 - 2.0 * position)

    def _weights_at(self, token_progress: float) -> np.ndarray:
        blend = self._blend_at(token_progress)
        if blend >= 1.0:
            return self._weights
        if blend <= 0.0:
            return self._start_weights
        return (1.0 - blend) * self._start_weights + blend * self._weights

    def _parameters_at(self, token_progress: float) -> dict[str, float]:
        blend = self._blend_at(token_progress)
        if blend >= 1.0:
            return dict(self._parameters)
        if blend <= 0.0:
            return dict(self._start_parameters)
        return {key: (1.0 - blend) * self._start_parameters[key] + blend * final_value for key, final_value in self._parameters.items()}

    def _length_probs_at(self, token_progress: float) -> np.ndarray:
        """Apply smooth short-context pulses inside a full-context envelope."""
        full_context = np.zeros_like(self._length_peak_probs)
        full_context[-1] = 1.0
        if not self._length_cycle_start < token_progress < self._length_cycle_end:
            return full_context
        phase = (token_progress - self._length_cycle_start) / (
            self._length_cycle_end - self._length_cycle_start
        )
        exposure = np.sin(np.pi * self._length_cycles * phase) ** 2
        return (
            (1.0 - exposure) * full_context
            + exposure * self._length_peak_probs
        )

    def _progress_at(self, emitted_points: float, target_points: int) -> float:
        return emitted_points / (target_points * self._expected_budget_fraction)

    def generate(self, n_series: int) -> Iterator[np.ndarray]:
        if n_series <= 0:
            return
        _warm_numba()
        rng = np.random.default_rng(self._seed)
        max_len = self._max_len
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
                while produced < n_series and (not stop.is_set()):
                    # Full batches keep series i a pure function of (seed, i).
                    batch_size = _CHUNK
                    length_progress = self._progress_at(
                        emitted_points, target_points
                    )
                    lengths = rng.choice(
                        self._length_buckets,
                        size=batch_size,
                        p=self._length_probs_at(length_progress),
                    )
                    take = min(batch_size, n_series - produced)
                    chunk_points = int(lengths[:take].sum())
                    midpoint_progress = self._progress_at(emitted_points + 0.5 * chunk_points, target_points)
                    family_weights = self._weights_at(midpoint_progress)
                    parameters = self._parameters_at(midpoint_progress)
                    builders = (partial(_k00, hi_frac=self._tr_hi_frac, exc_lo=parameters['tr_exc_lo'], exc_hi=parameters['tr_exc_hi'], clean_frac=parameters['sa_clean_frac'], clean_lo=parameters['sa_clean_lo'], clean_hi=parameters['sa_clean_hi']), _k01, partial(_k02, hi_frac=self._tr_hi_frac, exc_lo=parameters['gr_exc_lo'], exc_hi=parameters['gr_exc_hi']), _k03, partial(_k04, heavy_frac=parameters['integrated_heavy_frac'], sv_frac=parameters['integrated_sv_frac']), _k05, _k06, _k07, _k08, _k09, _k10, _k11, _k12, _k13, _k14, _k15, _k16, _k17, _k18, _k19, _k20, _k21, _k22)
                    current_observation = {key.removeprefix('observation.'): value for key, value in parameters.items() if key.startswith('observation.')}
                    fam_ids = rng.choice(len(_FAMILIES), size=batch_size, p=family_weights)
                    chunk: list[np.ndarray | None] = [None] * batch_size
                    for fam in range(len(_FAMILIES)):
                        idx = np.nonzero(fam_ids == fam)[0]
                        if idx.size == 0:
                            continue
                        block = builders[fam](rng, int(idx.size), max_len)
                        family = _FAMILIES[fam]
                        preserve_nonnegative = family in {'k02', 'k10', 'k11', 'k12'}
                        if family in _CLEAN:
                            block = _sanitize(block)
                        else:
                            block = _sanitize(_measurement_artifacts(rng, block, preserve_nonnegative=preserve_nonnegative, preserve_integers=family in {'k11', 'k12'}, allow_reverse=family in {'k00', 'k02', 'k07', 'k08'}, allow_range_artifacts=family != 'k04', **current_observation))
                        for row, series_i in enumerate(idx):
                            length = int(lengths[series_i])
                            chunk[series_i] = np.ascontiguousarray(block[row, :length], dtype=np.float64)
                    if self._min_len == max_len:
                        mix_rate = parameters['augment.tsmixup']
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
                    pad_rate = parameters['augment.pad_prefix']
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
        producer = Thread(target=produce, name='', daemon=True)
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
                        raise RuntimeError('internal: unfilled series slot')
                    yield arr
        finally:
            stop.set()
            producer.join(timeout=1.0)

@njit(cache=False, fastmath=False)
def _ar1_kernel(innov: np.ndarray, phi: np.ndarray) -> np.ndarray:
    n, length = innov.shape
    out = np.empty((n, length), dtype=np.float64)
    for row in range(n):
        state = 0.0
        for t in range(length):
            value = innov[row, t] + state
            out[row, t] = value
            state = phi[row] * value
    return out

def _ar1_batch(innov: np.ndarray, phi: np.ndarray) -> np.ndarray:
    return _ar1_kernel(
        np.ascontiguousarray(innov, dtype=np.float64),
        np.asarray(phi, dtype=np.float64).reshape(-1),
    )

def _stationary_ar1_innovations(
    standard_normal: np.ndarray,
    sigma: np.ndarray,
    phi: np.ndarray,
) -> np.ndarray:
    """Convert standard normals to exact stationary AR(1) filter inputs."""
    sigma_column = np.asarray(sigma, dtype=np.float64).reshape(-1, 1)
    phi_column = np.asarray(phi, dtype=np.float64).reshape(-1, 1)
    innov = (
        standard_normal
        * sigma_column
        * np.sqrt(np.maximum(1.0 - phi_column * phi_column, 1e-12))
    )
    innov[:, 0] = standard_normal[:, 0] * sigma_column[:, 0]
    return innov

def _preserve_gamma_intensity_totals(
    lam: np.ndarray,
    mixed: np.ndarray,
    selected: np.ndarray,
) -> np.ndarray:
    """Preserve each selected row's latent total after Gamma mixing."""
    if selected.size == 0:
        return mixed
    original_total = np.sum(
        lam[selected], axis=1, keepdims=True, dtype=np.float64
    )
    selected_mixed = mixed[selected].copy()
    mixed_total = np.sum(
        selected_mixed, axis=1, keepdims=True, dtype=np.float64
    )
    positive = original_total[:, 0] > 0.0
    if np.any(positive):
        ratio = mixed_total[positive] / original_total[positive]
        selected_mixed[positive] /= np.maximum(ratio, 1e-12)
    selected_mixed[~positive] = 0.0
    mixed[selected] = selected_mixed
    return mixed

@njit(cache=False, fastmath=False)
def _ar2_kernel(
    innov: np.ndarray,
    a1: np.ndarray,
    a2: np.ndarray,
) -> np.ndarray:
    n, length = innov.shape
    out = np.empty((n, length), dtype=np.float64)
    for row in range(n):
        state0 = 0.0
        state1 = 0.0
        for t in range(length):
            value = innov[row, t] + state0
            out[row, t] = value
            state0 = a1[row] * value + state1
            state1 = a2[row] * value
    return out

def _ar2_batch(innov: np.ndarray, a1: np.ndarray, a2: np.ndarray) -> np.ndarray:
    return _ar2_kernel(
        np.ascontiguousarray(innov, dtype=np.float64),
        np.asarray(a1, dtype=np.float64).reshape(-1),
        np.asarray(a2, dtype=np.float64).reshape(-1),
    )

@njit(cache=False, fastmath=False)
def _threshold_ar_kernel(
    innov: np.ndarray,
    phi_hi: np.ndarray,
    phi_lo: np.ndarray,
    const_hi: np.ndarray,
    const_lo: np.ndarray,
) -> np.ndarray:
    n, length = innov.shape
    out = np.empty((n, length), dtype=np.float64)
    for row in range(n):
        out[row, 0] = innov[row, 0]
        for t in range(1, length):
            prev = out[row, t - 1]
            if prev >= 0.0:
                value = const_hi[row] + phi_hi[row] * prev + innov[row, t]
            else:
                value = const_lo[row] + phi_lo[row] * prev + innov[row, t]
            out[row, t] = min(max(value, -1e6), 1e6)
    return out

def _warm_numba() -> None:
    global _NUMBA_WARMED
    if _NUMBA_WARMED:
        return
    with _NUMBA_WARM_LOCK:
        if _NUMBA_WARMED:
            return
        zeros = np.zeros((2, 3), dtype=np.float64)
        params = np.zeros(2, dtype=np.float64)
        _ar1_kernel(zeros, params)
        _ar2_kernel(zeros, params, params)
        _threshold_ar_kernel(zeros, params, params, params, params)
        _NUMBA_WARMED = True

def _prefix_mean_std(x: np.ndarray, *, calibration_points: int=512) -> tuple[np.ndarray, np.ndarray]:
    prefix = x[:, :min(x.shape[1], calibration_points)]
    mean = prefix.mean(axis=1, keepdims=True)
    std = prefix.std(axis=1, keepdims=True)
    return (mean, np.where(std < 1e-12, 1.0, std))

def _prefix_standardize(x: np.ndarray, *, center: bool=True, calibration_points: int=512) -> np.ndarray:
    mean, std = _prefix_mean_std(x, calibration_points=calibration_points)
    return (x - mean) / std if center else x / std

@lru_cache(maxsize=4)
def _seasonal_basis(L: int) -> tuple[np.ndarray, np.ndarray]:
    angle = 2.0 * np.pi * np.arange(L, dtype=np.float64)[None, :] / _SEASONAL_PERIODS[:, None]
    return (np.sin(angle), np.cos(angle))

def _seasonal(rng: np.random.Generator, n: int, L: int, k_max: int=3) -> np.ndarray:
    t = np.arange(L, dtype=np.float64)[None, :]
    sin_basis, cos_basis = _seasonal_basis(L)
    k = rng.integers(1, k_max + 1, size=n)
    pair = _SEASONAL_PAIRS[rng.integers(0, len(_SEASONAL_PAIRS), size=n)]
    use_pair = rng.random(n) < 0.35
    out = np.zeros((n, L), dtype=np.float64)
    for j in range(k_max):
        active = np.nonzero(k > j)[0]
        per = rng.choice(_SEASONAL_PERIODS, size=n, p=_SEASONAL_PROBS)
        if j < 2:
            per = np.where(use_pair, pair[:, j], per)
        per = per[:, None]
        amp = rng.uniform(0.2, 2.0, size=n)[:, None]
        phase = rng.uniform(0.0, 2.0 * np.pi, size=n)[:, None]
        basis_idx = np.searchsorted(_SEASONAL_PERIODS, per[active, 0])
        component = amp[active] * (sin_basis[basis_idx] * np.cos(phase[active]) + cos_basis[basis_idx] * np.sin(phase[active]))
        modulated = np.nonzero((k > j) & (rng.random(n) < 0.35))[0]
        if modulated.size:
            modulated_local = np.searchsorted(active, modulated)
            modulated_arg = 2.0 * np.pi * t / per[modulated] + phase[modulated]
            m_per = np.clip(per[modulated] * rng.uniform(4.0, 12.0, size=(modulated.size, 1)), 32.0, 2.0 * L)
            m_phase = rng.uniform(0.0, 2.0 * np.pi, size=(modulated.size, 1))
            slow = np.sin(2.0 * np.pi * t / m_per + m_phase)
            amp_mod = 1.0 + rng.uniform(0.05, 0.45, size=(modulated.size, 1)) * slow
            phase_mod = rng.uniform(0.05, 0.75, size=(modulated.size, 1)) * np.sin(2.0 * np.pi * t / (1.7 * m_per) - m_phase)
            component[modulated_local] = amp[modulated] * amp_mod * np.sin(modulated_arg + phase_mod)
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

def _measurement_artifacts(rng: np.random.Generator, block: np.ndarray, *, preserve_nonnegative: bool, preserve_integers: bool=False, allow_reverse: bool=True, allow_range_artifacts: bool=True, censor_rate: float=0.06, quantize_rate: float=0.07, regular_hold_rate: float=0.04, irregular_hold_rate: float=0.0, irregular_hold_prob_lo: float=0.01, irregular_hold_prob_hi: float=0.1, shock_row_rate: float=0.0, shock_prob_lo: float=0.001, shock_prob_hi: float=0.015) -> np.ndarray:
    original = np.asarray(block, dtype=np.float64)
    out = original.copy()
    n, L = out.shape
    reverse = rng.random(n) < 0.06 if allow_reverse else np.zeros(n, dtype=bool)
    out[reverse] = out[reverse, ::-1]
    if not preserve_nonnegative:
        invert = rng.random(n) < 0.04
        out[invert] *= -1.0
    calibration_len = min(L, 512)
    shocked = np.nonzero(rng.random(n) < shock_row_rate)[0]
    if shocked.size and L > 1:
        diff = np.diff(out[shocked, :calibration_len], axis=1)
        center = np.median(diff, axis=1, keepdims=True)
        robust_scale = 1.4826 * np.median(np.abs(diff - center), axis=1, keepdims=True)
        fallback = np.maximum(np.std(diff, axis=1, keepdims=True), 1e-09)
        robust_scale = np.where(robust_scale > 1e-09, robust_scale, fallback)
        event_prob = rng.uniform(shock_prob_lo, shock_prob_hi, size=(shocked.size, 1))
        event_rows, event_cols = np.nonzero(rng.random((shocked.size, L)) < event_prob)
        if event_rows.size:
            favored_sign = rng.choice([-1.0, 1.0], size=(shocked.size, 1))
            sign = np.where(rng.random(event_rows.size) < 0.75, favored_sign[event_rows, 0], -favored_sign[event_rows, 0])
            magnitude = rng.lognormal(mean=np.log(4.0), sigma=0.6, size=event_rows.size)
            out[shocked[event_rows], event_cols] += sign * magnitude * robust_scale[event_rows, 0]
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
            out[quantized] = lo + np.rint((clipped - lo) / safe_step) * safe_step
    held = np.nonzero(rng.random(n) < regular_hold_rate)[0]
    if held.size:
        factors = rng.choice([2, 4, 8], size=held.size, p=[0.55, 0.3, 0.15])
        for factor in (2, 4, 8):
            rows = held[factors == factor]
            if rows.size:
                out[rows] = np.repeat(out[rows, ::factor], factor, axis=1)[:, :L]
    irregular = np.nonzero(rng.random(n) < irregular_hold_rate)[0]
    if irregular.size and L > 1:
        hold_prob = rng.uniform(irregular_hold_prob_lo, irregular_hold_prob_hi, size=(irregular.size, 1))
        hold = rng.random((irregular.size, L)) < hold_prob
        hold[:, 0] = False
        source_index = np.where(~hold, np.arange(L, dtype=np.int64)[None, :], 0)
        np.maximum.accumulate(source_index, axis=1, out=source_index)
        out[irregular] = np.take_along_axis(out[irregular], source_index, axis=1)
    if preserve_integers:
        out = np.maximum(np.rint(out), 0.0)
    degenerate = out[:, :calibration_len].std(axis=1) < 1e-09
    out[degenerate] = original[degenerate]
    return out

def _k00(rng: np.random.Generator, n: int, L: int, *, hi_frac: float=0.25, exc_lo: float=0.4, exc_hi: float=3.0, clean_frac: float=0.4, clean_lo: float=0.02, clean_hi: float=0.12) -> np.ndarray:
    t = np.arange(L, dtype=np.float64)[None, :]
    level = rng.normal(0.0, 1.0, size=(n, 1))
    _hi = rng.random((n, 1)) < hi_frac
    exc = np.where(_hi, rng.normal(0.0, exc_hi, size=(n, 1)), rng.normal(0.0, exc_lo, size=(n, 1)))
    tn = t / max(L - 1, 1)
    series = level + exc * tn + _seasonal(rng, n, L)
    phi = rng.uniform(0.0, 0.85, size=n)
    clean = rng.random((n, 1)) < clean_frac
    sigma = np.where(clean, rng.uniform(clean_lo, clean_hi, size=(n, 1)), rng.uniform(0.1, 0.6, size=(n, 1)))
    standard_normal = rng.normal(0.0, 1.0, size=(n, L))
    innov = _stationary_ar1_innovations(standard_normal, sigma, phi)
    return series + _ar1_batch(innov, phi)

def _k01(rng: np.random.Generator, n: int, L: int) -> np.ndarray:
    level = np.cumsum(_sparse_jumps(rng, n, L, rate=3.0 / L, scale=2.0), axis=1)
    log_vol = np.cumsum(_sparse_jumps(rng, n, L, rate=3.0 / L, scale=0.5), axis=1)
    vol = np.exp(np.clip(log_vol, -3.0, 3.0)) * rng.uniform(0.1, 0.5, size=(n, 1))
    noise = rng.normal(0.0, 1.0, size=(n, L)) * vol
    seas = _seasonal(rng, n, L, k_max=2) * rng.uniform(0.0, 1.0, size=(n, 1))
    slope = rng.normal(0.0, 1.0 / L, size=(n, 1)) + np.cumsum(_sparse_jumps(rng, n, L, rate=2.0 / L, scale=4.0 / L), axis=1)
    piecewise_trend = np.cumsum(slope, axis=1)
    return level + piecewise_trend + seas + noise

def _k02(rng: np.random.Generator, n: int, L: int, *, hi_frac: float=0.25, exc_lo: float=0.3, exc_hi: float=2.0) -> np.ndarray:
    t = np.arange(L, dtype=np.float64)[None, :]
    _hg = rng.random((n, 1)) < hi_frac
    gexc = np.where(_hg, rng.normal(0.0, exc_hi, size=(n, 1)), rng.normal(0.0, exc_lo, size=(n, 1)))
    tn = t / max(L - 1, 1)
    base_level = np.exp(gexc * tn + rng.normal(0.0, 0.3, size=(n, 1)))
    amp = rng.uniform(0.1, 0.6, size=(n, 1))
    seasonal_shape = _seasonal(rng, n, L, k_max=1)
    seasonal_shape = _prefix_standardize(seasonal_shape, center=False)
    seas = 1.0 + amp * seasonal_shape
    noise = 1.0 + rng.normal(0.0, 1.0, size=(n, L)) * rng.uniform(0.02, 0.15, size=(n, 1))
    scale = rng.uniform(1.0, 50.0, size=(n, 1))
    return scale * base_level * np.clip(seas, 0.05, None) * np.clip(noise, 0.05, None)

def _k03(rng: np.random.Generator, n: int, L: int) -> np.ndarray:
    p1 = rng.uniform(0.3, 0.98, size=n)
    p2 = rng.uniform(-0.6, 0.6, size=n)
    a2 = p2
    a1 = p1 * (1.0 - p2)
    sigma = rng.uniform(0.2, 0.8, size=(n, 1))
    burn = 512
    innov = rng.normal(0.0, 1.0, size=(n, L + burn)) * sigma
    return _ar2_batch(innov, a1, a2)[:, burn:]

def _k04(rng: np.random.Generator, n: int, L: int, *, heavy_frac: float=0.25, sv_frac: float=0.3) -> np.ndarray:
    order2 = rng.random(n) < 0.35
    drift = rng.normal(0.0, 0.02, size=(n, 1))
    sigma = rng.uniform(0.2, 1.0, size=(n, 1))
    eps = rng.normal(0.0, 1.0, size=(n, L))
    heavy = np.nonzero(rng.random(n) < heavy_frac)[0]
    if heavy.size:
        df = rng.uniform(3.0, 12.0, size=(heavy.size, 1))
        eps[heavy] = rng.standard_t(df, size=(heavy.size, L)) / np.sqrt(df / (df - 2.0))
    stochastic = np.nonzero(rng.random(n) < sv_frac)[0]
    if stochastic.size:
        phi = 0.995
        burn = 256
        vol_innov = rng.standard_normal((stochastic.size, L + burn)) * np.sqrt(1.0 - phi * phi)
        log_vol = lfilter([1.0], [1.0, -phi], vol_innov, axis=1)[:, burn:]
        log_vol *= rng.uniform(0.1, 0.55, size=(stochastic.size, 1))
        eps[stochastic] *= np.exp(np.clip(log_vol, -2.0, 2.0))
    steps = eps * sigma + drift
    walk = np.cumsum(steps, axis=1)
    walk2 = np.cumsum(walk, axis=1)
    o2 = order2[:, None]
    return np.where(o2, walk2 / max(L, 1), walk)

def _k05(rng: np.random.Generator, n: int, L: int) -> np.ndarray:
    phi_hi = rng.uniform(0.3, 0.9, size=n)
    phi_lo = rng.uniform(-0.9, 0.3, size=n)
    const_hi = rng.normal(0.0, 0.3, size=n)
    const_lo = rng.normal(0.0, 0.3, size=n)
    sigma = rng.uniform(0.2, 0.7, size=(n, 1))
    burn = 256
    total = L + burn
    innov = rng.normal(0.0, 1.0, size=(n, total)) * sigma
    x = _threshold_ar_kernel(
        np.ascontiguousarray(innov, dtype=np.float64),
        np.asarray(phi_hi, dtype=np.float64),
        np.asarray(phi_lo, dtype=np.float64),
        np.asarray(const_hi, dtype=np.float64),
        np.asarray(const_lo, dtype=np.float64),
    )
    return x[:, burn:]

def _k06(rng: np.random.Generator, n: int, L: int) -> np.ndarray:
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

def _k07(rng: np.random.Generator, n: int, L: int) -> np.ndarray:
    embed = 2 * L
    f = np.fft.rfftfreq(embed)[None, :]
    lengthscale = np.exp(rng.uniform(np.log(8.0), np.log(256.0), size=(n, 1)))
    envelope = np.exp(-0.5 * (2.0 * np.pi * lengthscale * f) ** 2)
    z = rng.standard_normal((n, f.shape[1])) + 1j * rng.standard_normal((n, f.shape[1]))
    z[:, 0] = 0.0
    x = np.fft.irfft(z * np.sqrt(envelope), n=embed, axis=1)[:, :L]
    return _prefix_standardize(x)

def _k08(rng: np.random.Generator, n: int, L: int) -> np.ndarray:
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
    amp_hi = split_f ** (-0.5 * beta) * (safe_f / split_f) ** (-0.5 * beta_hi)
    amp = np.where(multiscale & above, amp_hi, amp)
    amp[:, 0] = 0.0
    z = rng.standard_normal((n, f.size)) + 1j * rng.standard_normal((n, f.size))
    x = np.fft.irfft(z * amp, n=embed, axis=1)[:, :L]
    integrate = rng.random(n) < 0.25
    if integrate.any():
        x[integrate] = np.cumsum(x[integrate], axis=1)
    return _prefix_standardize(x)

def _k09(rng: np.random.Generator, n: int, L: int) -> np.ndarray:
    switch_rate = np.exp(rng.uniform(np.log(0.001), np.log(0.15), size=(n, 1)))
    switches = rng.random((n, L)) < switch_rate
    switches[:, 0] = rng.random(n) < 0.5
    regime = np.bitwise_and(np.cumsum(switches, axis=1), 1).astype(np.int8)
    slow = rng.random((n, 1)) < 0.5
    phi = np.where(slow, rng.uniform(0.995, 0.9995, size=(n, 1)), rng.uniform(0.9, 0.99, size=(n, 1)))
    mu0 = rng.normal(-2.0, 1.0, size=(n, 1))
    mu1 = rng.normal(2.0, 1.0, size=(n, 1))
    mean = np.where(regime == 0, mu0, mu1)
    seasonal_on = rng.random((n, 1)) < 0.6
    mean += seasonal_on * _seasonal(rng, n, L, k_max=3) * rng.uniform(0.5, 3.0, size=(n, 1))
    log_sigma0 = rng.normal(np.log(0.3), 0.3, size=(n, 1))
    log_sigma1 = rng.normal(np.log(1.5), 0.5, size=(n, 1))
    log_sigma_mean = np.where(regime == 0, log_sigma0, log_sigma1)
    vol_rho = rng.uniform(0.951, 0.995, size=(n, 1))
    vol_eta = rng.uniform(0.03, 0.2, size=(n, 1))
    vol_eps = rng.standard_normal((n, L))
    vol_drive = (1.0 - vol_rho) * log_sigma_mean + np.sqrt(1.0 - vol_rho * vol_rho) * vol_eta * vol_eps
    log_vol = np.empty((n, L), dtype=np.float64)
    log_vol[:, 0] = log_sigma_mean[:, 0]
    for i in range(n):
        rho = float(vol_rho[i, 0])
        log_vol[i, 1:] = lfilter([1.0], [1.0, -rho], vol_drive[i, 1:], zi=[rho * log_vol[i, 0]])[0]
    vol = np.exp(np.clip(log_vol, -5.0, 5.0))
    eps = rng.standard_normal((n, L))
    heavy = np.nonzero(rng.random(n) < 0.35)[0]
    if heavy.size:
        eps[heavy] = rng.standard_t(4.0, size=(heavy.size, L)) / np.sqrt(2.0)
    shocks = rng.random((n, L)) < 3.0 / L
    shock_rows, shock_cols = np.nonzero(shocks)
    eps[shock_rows, shock_cols] += rng.normal(0.0, 5.0, size=shock_rows.size)
    innovation_scale = np.sqrt(np.maximum(1.0 - phi * phi, 1e-06))
    drive = (1.0 - phi) * mean + innovation_scale * vol * eps
    out = np.empty((n, L), dtype=np.float64)
    out[:, 0] = mean[:, 0] + vol[:, 0] * eps[:, 0]
    for i in range(n):
        p = float(phi[i, 0])
        out[i, 1:] = lfilter([1.0], [1.0, -p], drive[i, 1:], zi=[p * out[i, 0]])[0]
    scale = np.exp(rng.uniform(np.log(0.1), np.log(50.0), size=(n, 1)))
    shift = rng.uniform(-100.0, 100.0, size=(n, 1))
    return out * scale + shift

def _k10(rng: np.random.Generator, n: int, L: int) -> np.ndarray:
    seasonal = _seasonal(rng, n, L, k_max=2)
    smooth = _k07(rng, n, L)
    fronts = np.cumsum(_sparse_jumps(rng, n, L, rate=5.0 / L, scale=1.0), axis=1)
    base = seasonal * rng.uniform(0.3, 2.0, size=(n, 1)) + smooth * rng.uniform(0.2, 1.2, size=(n, 1)) + fronts * rng.uniform(0.2, 1.0, size=(n, 1))
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
        diffusion = np.exp(rng.uniform(np.log(0.03), np.log(0.2), size=(count, 1)))
        walk = np.cumsum(rng.standard_normal((count, L)) * diffusion, axis=1)
        level = rng.uniform(900.0, 1100.0, size=(count, 1))
        out[pressure] = level + walk + 2.0 * fronts[pressure] + 0.5 * seasonal[pressure]
    magnitude = kind == 3
    if magnitude.any():
        count = int(magnitude.sum())
        gusts = (rng.random((count, L)) < 8.0 / L) * rng.lognormal(0.0, 0.8, size=(count, L))
        power = rng.uniform(1.0, 1.6, size=(count, 1))
        out[magnitude] = np.abs(base[magnitude]) ** power + gusts
    return out

def _k11(rng: np.random.Generator, n: int, L: int) -> np.ndarray:
    t = np.arange(L, dtype=np.float64)[None, :]
    period = rng.choice(_SEASONAL_PERIODS, size=(n, 1), p=_SEASONAL_PROBS)
    phase = rng.uniform(0.0, 2.0 * np.pi, size=(n, 1))
    amp = rng.uniform(0.15, 0.8, size=(n, 1))
    log_rate = amp * np.sin(2.0 * np.pi * t / period + phase)
    second = rng.random((n, 1)) < 0.55
    log_rate += second * (0.5 * amp) * np.sin(4.0 * np.pi * t / period + rng.uniform(0.0, 2.0 * np.pi, size=(n, 1)))
    calendar = rng.random((n, 1)) < 0.35
    day_period = rng.choice([24, 48, 96, 144], size=(n, 1))
    day_idx = (np.floor_divide(np.arange(L)[None, :], day_period) % 7).astype(np.int64)
    day_factors = rng.normal(0.0, 0.12, size=(n, 7))
    day_factors[:, 5:] += rng.uniform(-0.8, 0.3, size=(n, 1))
    calendar_effect = np.take_along_axis(day_factors, day_idx, axis=1)
    log_rate += calendar * calendar_effect
    excursion = rng.uniform(-0.5, 0.5, size=(n, 1))
    log_rate += excursion * t / max(L - 1, 1)
    impulses = (rng.random((n, L)) < 2.0 / L) * rng.uniform(1.0, 10.0, size=(n, L))
    burst = _ar1_batch(impulses, rng.uniform(0.85, 0.995, size=(n, 1)))
    base = np.exp(rng.uniform(np.log(3.0), np.log(3000.0), size=(n, 1)))
    lam = base * np.exp(np.clip(log_rate, -5.0, 5.0)) * (1.0 + burst)
    np.clip(lam, 0.0, 10000000.0, out=lam)
    overdispersed = rng.random((n, 1)) < 0.5
    shape = rng.uniform(0.5, 4.0, size=(n, 1))
    mixed = lam * rng.gamma(shape, 1.0 / shape, size=(n, L))
    selected = np.flatnonzero(overdispersed[:, 0])
    mixed = _preserve_gamma_intensity_totals(lam, mixed, selected)
    intensity = np.where(overdispersed, mixed, lam)
    return rng.poisson(intensity).astype(np.float64)

def _k12(rng: np.random.Generator, n: int, L: int) -> np.ndarray:
    t = np.arange(L, dtype=np.float64)[None, :]
    base_p = rng.uniform(0.03, 0.35, size=(n, 1))
    period = rng.choice([7.0, 12.0, 24.0, 48.0, 168.0], size=(n, 1))
    season = rng.uniform(0.2, 1.2, size=(n, 1)) * np.sin(2.0 * np.pi * t / period + rng.uniform(0.0, 2.0 * np.pi, size=(n, 1)))
    logit = np.log(base_p / (1.0 - base_p)) + season
    p = 1.0 / (1.0 + np.exp(-logit))
    occur = (rng.random((n, L)) < p).astype(np.float64)
    magnitude = np.maximum(1.0, np.rint(rng.gamma(shape=2.0, scale=1.0, size=(n, L)) * rng.uniform(1.0, 10.0, size=(n, 1)) * np.exp(0.25 * season)))
    return occur * magnitude

def _k13(rng: np.random.Generator, n: int, L: int) -> np.ndarray:
    base = _k07(rng, n, L) * rng.uniform(0.5, 2.0, size=(n, 1))
    base += _seasonal(rng, n, L, k_max=1) * rng.uniform(0.0, 1.0, size=(n, 1))
    sharp = _sparse_jumps(rng, n, L, rate=3.0 / L, scale=rng.uniform(3.0, 8.0, size=n))
    impulses = _sparse_jumps(rng, n, L, rate=2.0 / L, scale=rng.uniform(2.0, 7.0, size=n))
    recovery = _ar1_batch(impulses, rng.uniform(0.75, 0.995, size=n))
    series = base + sharp + recovery
    starts = rng.random((n, L)) < 2.0 / L
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

def _k14(rng: np.random.Generator, n: int, L: int) -> np.ndarray:
    if n <= 0:
        return np.empty((0, L), dtype=np.float64)
    if L <= 0:
        return np.empty((n, 0), dtype=np.float64)
    seasonal = _seasonal(rng, n, L, k_max=2)
    ar = _ar1_batch(rng.normal(size=(n, L)) * rng.uniform(0.12, 0.55, size=(n, 1)), rng.uniform(0.35, 0.92, size=n))
    smooth = _k07(rng, n, L)
    walk = np.cumsum(rng.normal(size=(n, L)) * rng.uniform(0.025, 0.16, size=(n, 1)), axis=1)
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
        calibration = dynamic[:min(L, 512)]
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
                seg_len = int(rng.integers(_CS_DYNAMIC_LO, _CS_DYNAMIC_HI + 1))
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
                    drift += np.linspace(0.0, float(rng.normal(0.0, 0.025)), span)
                    values = current + drift
                elif mode == 2:
                    if not mode2_seen:
                        count_level = max(0.0, float(np.rint(abs(current) * 8.0)))
                        mode2_seen = True
                    else:
                        count_level = max(0.0, float(np.rint(current)))
                    updates = rng.random(span) < 0.025
                    changes = updates * rng.choice([-1.0, 1.0], size=span)
                    values = np.maximum(count_level + np.cumsum(changes), 0.0)
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
            out[row, -1] += max(0.001, 0.01 * row_scale)
    return out

def _sanitize(block: np.ndarray) -> np.ndarray:
    x = np.asarray(block, dtype=np.float64)
    np.nan_to_num(x, copy=False, nan=0.0, posinf=1000000.0, neginf=-1000000.0)
    if x.ndim == 1:
        peak = float(np.max(np.abs(x)))
        if peak > 1000000.0:
            x *= 1000000.0 / peak
    else:
        peak = np.max(np.abs(x), axis=1, keepdims=True)
        scale = np.where(peak > 1000000.0, 1000000.0 / np.maximum(peak, 1e-12), 1.0)
        x *= scale
    return x