"""cascade-fullctx-spectral-v12 — prefetched full-context mixture-of-priors generator.

This is the artifact a cascade miner actually competes with: a subclass of
``cascade.interface.DataGenerator`` that turns a single integer ``seed`` into a
corpus of univariate float series. The subnet holds the model, seeds, and
compute budget byte-identical between the king and every challenger, so the
*only* thing that moves the forecast score is the distribution this file emits.
The competitive lever is therefore **prior diversity + realism**: a corpus that
covers more of the shapes a real forecaster must handle (trend, multi-seasonal,
regime shifts, integrated/near-unit-root dynamics, smooth GP-like curves,
nonlinear/chaotic recurrences, mean-reverting stochastic volatility,
intermittent demand, event recovery, and measurement artifacts) trains a
stronger zero-shot model than the reference generator's trend+seasonal+AR(1)
mix.

Design constraints this file respects (all from the contract in
``cascade.interface``):

* **Determinism is load-bearing.** Every value is drawn from one
  ``np.random.default_rng(seed)`` in a fixed draw order, so two runs at the same
  seed produce byte-identical corpora — the property ``cascade verify`` audits
  by building the corpus twice and comparing digests.
* **Code-only.** No shipped weights, no network, no clock, no un-seeded RNG.
  Imports stay on the dependency allowlist (NumPy/SciPy only) and clear of the
  static-guard blocklist.
* **Bounded + finite.** Each series is 1-D ``(L,)`` float64, length in
  ``[min_length, max_length]``, finite (no NaN/inf). ``_sanitize`` is the last
  gate so a numerically unlucky draw can never poison a training run.

Everything is **vectorised per family** (a batched time-axis recurrence, never a
per-series Python loop over time). Compared with custom-fullctx-v4, the slow
random-Fourier GP is replaced by FFT spectral sampling, a long-memory spectral
family is added, and larger chunks amortise dispatch while staying far below the
sandbox memory limit.
"""

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

# Series generated per vectorised batch. Bounds peak memory to O(_CHUNK · max_len)
# so streaming feed modes (which request millions of series and stop early) never
# materialise the full corpus. Prefetching holds at most two completed chunks
# (current + queued) while the producer may build the next. The base block is
# 2048 × 4096 × 8 B = 64 MiB per base family block, plus temporary arrays.
# This remains comfortably below the 4 GiB sandbox cap. On the reference local
# A100 environment, 2048 rows generated ~6% more points/s than 1024 while 4096
# regressed slightly, so 2048 is the measured throughput sweet spot.
_CHUNK = 2048

# Multi-cadence seasonal bank. Full 4096-point contexts can identify several
# cycles even at 365/672/730-step periods, unlike short-crop generators.
_SEASONAL_PERIODS = np.array(
    [4, 7, 12, 24, 30, 48, 52, 90, 96, 144, 168, 183, 288, 336, 365, 672, 730],
    dtype=np.float64,
)
_SEASONAL_PROBS = np.array(
    [0.04, 0.12, 0.04, 0.16, 0.05, 0.06, 0.04, 0.03, 0.07, 0.03,
     0.13, 0.04, 0.04, 0.06, 0.07, 0.04, 0.05],
    dtype=np.float64,
)
_SEASONAL_PROBS /= _SEASONAL_PROBS.sum()

# ── family mixture ──────────────────────────────────────────────────────────
# Names are the process families the corpus mixes over; the default weights are
# a deliberate spread (no single family dominates). Override with
# ``"family_weights": {"chaotic": 0.2, ...}`` in config.json to tune the prior
# without touching code — unspecified families keep their default weight.
_FAMILIES: tuple[str, ...] = (
    "trend_seasonal_ar",   # level + slope + multi-seasonal + AR(1) noise (rich reference)
    "regime_shift",        # piecewise level/variance regimes with structural breaks
    "multiplicative",      # positive level × seasonal factor × multiplicative noise
    "ar2",                 # AR(2), stationarity-guaranteed, incl. near-unit-root
    "integrated",          # I(1)/I(2) random walks with drift
    "threshold_ar",        # SETAR — regime-switching nonlinear recurrence
    "chaotic",             # bounded chaotic maps (logistic / sine)
    "spectral_gp",         # smooth GP-like paths via batched FFT sampling
    "long_memory",         # persistent/anti-persistent power-law spectra
    "ou_stochastic_vol",   # mean-reverting regimes + clustered/heavy-tailed volatility
    "physical_sensors",    # bounded/skewed/smooth physical measurement archetypes
    "seasonal_counts",     # seasonal Poisson/NB web and demand counts with bursts
    "intermittent",        # zero-inflated / intermittent demand
    "pulse_outlier",       # sharp/decaying events, outliers, and true flat runs
)
# Dynamics-heavy composition selected by a controlled local A/B: it beat the
# prior baseline on all three validation seeds (mean geomean 0.18431 vs
# 0.19097). Config may still override these defaults, but copying generator.py
# without config now retains the measured mixture.
_DEFAULT_WEIGHTS: dict[str, float] = {
    "trend_seasonal_ar": 0.12,
    "regime_shift": 0.12,
    "multiplicative": 0.08,
    "ar2": 0.15,
    "integrated": 0.12,
    "threshold_ar": 0.08,
    "chaotic": 0.04,
    "spectral_gp": 0.07,
    "long_memory": 0.06,
    "ou_stochastic_vol": 0.10,
    "physical_sensors": 0.02,
    "seasonal_counts": 0.02,
    "intermittent": 0.01,
    "pulse_outlier": 0.01,
}


class Generator(DataGenerator):
    """A mixture-of-priors generator. Submit as ``generator.Generator``."""

    def __init__(self, config_dir: str, *, seed: int) -> None:
        cfg_path = Path(config_dir) / "config.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {}
        self._cfg = cfg
        self._seed = int(seed)
        self._min_len = int(cfg.get("min_length", 64))
        self._max_len = int(cfg.get("max_length", 4096))  # = [training] context_length (train on full context)
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
        # v3.9 length-NORMALIZED bimodal trend knobs (trend excursion is length-invariant;
        # real trend-strength is ~0.02 and length-invariant, but v2's slope*t grows with L).
        self._tr_hi_frac = float(cfg.get("tr_hi_frac", 0.25))
        self._tr_exc_lo = float(cfg.get("tr_exc_lo", 0.4))
        self._tr_exc_hi = float(cfg.get("tr_exc_hi", 3.0))
        self._gr_exc_lo = float(cfg.get("gr_exc_lo", 0.3))
        self._gr_exc_hi = float(cfg.get("gr_exc_hi", 2.0))
        self._sa_clean_frac = float(cfg.get("sa_clean_frac", 0.4))
        self._sa_clean_lo = float(cfg.get("sa_clean_lo", 0.02))
        self._sa_clean_hi = float(cfg.get("sa_clean_hi", 0.12))

    @property
    def name(self) -> str:
        return str(self._cfg.get("name", "cascade-fullctx-spectral-v12"))

    def generate(self, n_series: int) -> Iterator[np.ndarray]:
        # Lazy, chunked generation. This is REQUIRED for the streaming feed
        # modes (chain.toml ``corpus_mode = "stream_cpu"``): the trainer calls
        # ``generate(n_upper)`` with ``n_upper = token_budget // min_length + 2``
        # — often millions — and stops pulling once the token budget is hit
        # (see cascade/trainer/stream.py). Materialising all ``n_series`` up
        # front would OOM before the first yield. Generating one CHUNK at a time
        # keeps memory at O(CHUNK) and stops early when the consumer stops,
        # while a fixed draw order keeps the whole sequence seed-deterministic.
        if n_series <= 0:
            return
        rng = np.random.default_rng(self._seed)
        max_len = self._max_len
        # Bind the trend-excursion knobs as explicit builder arguments (no shared
        # module state) so the corpus is a pure function of (seed, config).
        builders = (
            partial(_trend_seasonal_ar, hi_frac=self._tr_hi_frac,
                    exc_lo=self._tr_exc_lo, exc_hi=self._tr_exc_hi,
                    clean_frac=self._sa_clean_frac,
                    clean_lo=self._sa_clean_lo, clean_hi=self._sa_clean_hi),
            _regime_shift,
            partial(_multiplicative, hi_frac=self._tr_hi_frac,
                    exc_lo=self._gr_exc_lo, exc_hi=self._gr_exc_hi),
            _ar2, _integrated, _threshold_ar, _chaotic, _spectral_gp,
            _long_memory, _ou_stochastic_vol, _physical_sensors,
            _seasonal_counts, _intermittent, _pulse_outlier,
        )
        # Generate one chunk ahead on a CPU thread while the consumer trains on
        # the current chunk. The isolation benchmark measured 21.9% of training
        # wall blocked in next(); a one-slot queue overlaps NumPy/SciPy work
        # (which releases the GIL) without changing the RNG owner or draw order.
        queue: Queue[object] = Queue(maxsize=1)
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
                while produced < n_series and not stop.is_set():
                    # Always draw a FULL _CHUNK (yielding only what's still
                    # needed), so series i remains a pure function of (seed, i).
                    lengths = rng.integers(
                        self._min_len, max_len + 1, size=_CHUNK
                    )
                    fam_ids = rng.choice(
                        len(_FAMILIES), size=_CHUNK, p=self._weights
                    )
                    chunk: list[np.ndarray | None] = [None] * _CHUNK
                    for fam in range(len(_FAMILIES)):
                        idx = np.nonzero(fam_ids == fam)[0]
                        if idx.size == 0:
                            continue
                        block = builders[fam](rng, int(idx.size), max_len)
                        # Preserve positivity for count/magnitude families.
                        preserve_nonnegative = fam in (2, 10, 11, 12)
                        block = _sanitize(
                            _measurement_artifacts(
                                rng,
                                block,
                                preserve_nonnegative=preserve_nonnegative,
                            )
                        )
                        for row, series_i in enumerate(idx):
                            length = int(lengths[series_i])
                            chunk[series_i] = np.ascontiguousarray(
                                block[row, :length], dtype=np.float64
                            )
                    take = min(_CHUNK, n_series - produced)
                    if not put((chunk, take)):
                        return
                    produced += take
            except BaseException as exc:  # propagate producer failures
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
                    # fam_ids partitions [0, _CHUNK); fail loud if that changes.
                    if arr is None:  # pragma: no cover - defensive
                        raise RuntimeError("internal: unfilled series slot")
                    yield arr
        finally:
            stop.set()
            producer.join(timeout=1.0)


# ── shared vectorised primitives ────────────────────────────────────────────


def _ar1_batch(innov: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """AR(1) filter applied along the time axis of a (n, L) innovation block.

    ``x[:, t] = phi * x[:, t-1] + innov[:, t]``. The loop is over time (L
    iterations, vectorised across the batch), never over the n series.
    """
    n, L = innov.shape
    x = np.empty((n, L), dtype=np.float64)
    p = phi.reshape(n)
    for i in range(n):
        x[i] = lfilter([1.0], [1.0, -float(p[i])], innov[i])
    return x


def _ar2_batch(innov: np.ndarray, a1: np.ndarray, a2: np.ndarray) -> np.ndarray:
    """AR(2) filter: ``x_t = a1 x_{t-1} + a2 x_{t-2} + e_t`` (batched over n)."""
    n, L = innov.shape
    x = np.empty((n, L), dtype=np.float64)
    for i in range(n):
        x[i] = lfilter(
            [1.0], [1.0, -float(a1[i]), -float(a2[i])], innov[i]
        )
    return x


@lru_cache(maxsize=4)
def _seasonal_basis(L: int) -> tuple[np.ndarray, np.ndarray]:
    """Cached unit sine/cosine waves for the fixed cadence bank."""
    angle = (
        2.0
        * np.pi
        * np.arange(L, dtype=np.float64)[None, :]
        / _SEASONAL_PERIODS[:, None]
    )
    return np.sin(angle), np.cos(angle)


def _seasonal(rng: np.random.Generator, n: int, L: int, k_max: int = 3) -> np.ndarray:
    """Sum of 1..k_max stationary or slowly modulated seasonal components."""
    t = np.arange(L, dtype=np.float64)[None, :]
    sin_basis, cos_basis = _seasonal_basis(L)
    k = rng.integers(1, k_max + 1, size=n)
    out = np.zeros((n, L), dtype=np.float64)
    for j in range(k_max):
        active = np.nonzero(k > j)[0]
        per = rng.choice(_SEASONAL_PERIODS, size=n, p=_SEASONAL_PROBS)[:, None]
        amp = rng.uniform(0.2, 2.0, size=n)[:, None]
        phase = rng.uniform(0.0, 2.0 * np.pi, size=n)[:, None]
        # Draw parameters for every row to preserve the fixed RNG sequence, but
        # evaluate only active rows. Stationary components reuse the cadence
        # bank via sin(a+b), avoiding a fresh transcendental pass over n×L.
        basis_idx = np.searchsorted(_SEASONAL_PERIODS, per[active, 0])
        component = amp[active] * (
            sin_basis[basis_idx] * np.cos(phase[active])
            + cos_basis[basis_idx] * np.sin(phase[active])
        )
        # Real seasonal strength and timing drift. TempoPFN's strongest
        # non-SDE ablation was its complex-seasonality prior, so a minority of
        # components receive slow amplitude and phase modulation while the
        # stationary baseline remains well represented.
        modulated = np.nonzero((k > j) & (rng.random(n) < 0.35))[0]
        if modulated.size:
            # Map global row indices into the active component block.
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
    """A (n, L) block of mostly-zero values with occasional N(0, scale) jumps.

    ``cumsum`` over this yields a piecewise-constant level; ``exp(cumsum)`` of a
    scaled version yields a piecewise-constant positive multiplier.
    """
    mask = rng.random((n, L)) < rate
    mask[:, 0] = False
    rows, cols = np.nonzero(mask)
    jumps = np.zeros((n, L), dtype=np.float64)
    if rows.size == 0:
        return jumps
    # Rates are O(1/L), so draw magnitudes only for actual events rather than
    # allocating and filling a second dense n×L normal array.
    s = np.asarray(scale, dtype=np.float64)
    event_scale = s if s.ndim == 0 else s.reshape(n)[rows]
    jumps[rows, cols] = rng.normal(0.0, 1.0, size=rows.size) * event_scale
    return jumps


def _measurement_artifacts(
    rng: np.random.Generator,
    block: np.ndarray,
    *,
    preserve_nonnegative: bool,
) -> np.ndarray:
    """Apply sparse, cheap real-measurement effects to a generated block.

    TempoPFN reports a 5.4% aggregate CRPS gain from its complete augmentation
    pipeline, but does not isolate optimal probabilities for Toto2. These rates
    are deliberately conservative: most rows remain untouched, and a selected
    row receives only plausible reversal/sign, censoring, quantization, or
    sample-and-hold behavior.
    """
    original = np.asarray(block, dtype=np.float64)
    out = original.copy()
    n, L = out.shape

    reverse = rng.random(n) < 0.06
    out[reverse] = out[reverse, ::-1]

    if not preserve_nonnegative:
        invert = rng.random(n) < 0.04
        out[invert] *= -1.0

    # Sensor saturation / floor effects. Existing sample values are used as
    # thresholds, avoiding artificial scales and preserving integer counts.
    for row in np.nonzero(rng.random(n) < 0.06)[0]:
        q = float(rng.uniform(0.03, 0.18))
        if rng.random() < 0.5:
            out[row] = np.minimum(out[row], np.quantile(out[row], 1.0 - q))
        else:
            out[row] = np.maximum(out[row], np.quantile(out[row], q))

    quantized = np.nonzero(rng.random(n) < 0.07)[0]
    if quantized.size:
        x = out[quantized]
        lo = x.min(axis=1, keepdims=True)
        hi = x.max(axis=1, keepdims=True)
        levels = rng.integers(16, 257, size=(quantized.size, 1))
        step = (hi - lo) / np.maximum(levels - 1, 1)
        safe_step = np.where(step < 1e-12, 1.0, step)
        out[quantized] = lo + np.rint((x - lo) / safe_step) * safe_step

    # Zero-order-hold resampling approximates telemetry gathered at a lower
    # cadence and forwarded at the nominal cadence.
    held = np.nonzero(rng.random(n) < 0.04)[0]
    if held.size:
        factors = rng.choice([2, 4, 8], size=held.size, p=[0.55, 0.30, 0.15])
        for factor in (2, 4, 8):
            rows = held[factors == factor]
            if rows.size:
                out[rows] = np.repeat(
                    out[rows, ::factor], factor, axis=1
                )[:, :L]
    # Heavy zero inflation plus upper censoring can otherwise collapse a sparse
    # row to its baseline. Such a row carries no forecasting signal.
    degenerate = out.std(axis=1) < 1e-9
    out[degenerate] = original[degenerate]
    return out


# ── family builders: each returns a (n, L) float64 block ────────────────────


def _trend_seasonal_ar(rng: np.random.Generator, n: int, L: int, *,
                       hi_frac: float = 0.25, exc_lo: float = 0.4,
                       exc_hi: float = 3.0, clean_frac: float = 0.4,
                       clean_lo: float = 0.02, clean_hi: float = 0.12) -> np.ndarray:
    t = np.arange(L, dtype=np.float64)[None, :]
    level = rng.normal(0.0, 1.0, size=(n, 1))
    # v3: bimodal trend. The total trend EXCURSION over the series is drawn directly
    # (0..exc across t/(L-1)), so the trend sits ~16x below v2's slope*t — v2's linear
    # trend was a measured ~16x too strong vs real data at production lengths.
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
    # Piecewise-constant level via cumsum of sparse jumps, plus a piecewise
    # variance regime (occasional volatility multiplier), plus mild seasonality.
    level = np.cumsum(_sparse_jumps(rng, n, L, rate=3.0 / L, scale=2.0), axis=1)
    log_vol = np.cumsum(_sparse_jumps(rng, n, L, rate=3.0 / L, scale=0.5), axis=1)
    vol = np.exp(np.clip(log_vol, -3.0, 3.0)) * rng.uniform(0.1, 0.5, size=(n, 1))
    noise = rng.normal(0.0, 1.0, size=(n, L)) * vol
    seas = _seasonal(rng, n, L, k_max=2) * rng.uniform(0.0, 1.0, size=(n, 1))
    # Piecewise-affine drift complements abrupt level jumps. Sparse slope
    # changes create ramps and recoveries without the explosive scale of an I(2)
    # process, covering TempoPFN's high-impact Step/Sawtooth structures.
    slope = rng.normal(0.0, 1.0 / L, size=(n, 1)) + np.cumsum(
        _sparse_jumps(rng, n, L, rate=2.0 / L, scale=4.0 / L), axis=1
    )
    piecewise_trend = np.cumsum(slope, axis=1)
    return level + piecewise_trend + seas + noise


def _multiplicative(rng: np.random.Generator, n: int, L: int, *,
                    hi_frac: float = 0.25, exc_lo: float = 0.3, exc_hi: float = 2.0) -> np.ndarray:
    t = np.arange(L, dtype=np.float64)[None, :]
    # v3: bimodal log-growth excursion (drawn directly), same rationale as the linear trend.
    _hg = rng.random((n, 1)) < hi_frac
    gexc = np.where(_hg, rng.normal(0.0, exc_hi, size=(n, 1)),
                    rng.normal(0.0, exc_lo, size=(n, 1)))
    tn = t / max(L - 1, 1)
    base_level = np.exp(gexc * tn + rng.normal(0.0, 0.3, size=(n, 1)))  # positive, drifting
    amp = rng.uniform(0.1, 0.6, size=(n, 1))
    seasonal_shape = _seasonal(rng, n, L, k_max=1)
    seasonal_sd = seasonal_shape.std(axis=1, keepdims=True)
    seasonal_shape /= np.where(seasonal_sd < 1e-12, 1.0, seasonal_sd)
    seas = 1.0 + amp * seasonal_shape
    noise = 1.0 + rng.normal(0.0, 1.0, size=(n, L)) * rng.uniform(0.02, 0.15, size=(n, 1))
    scale = rng.uniform(1.0, 50.0, size=(n, 1))
    return scale * base_level * np.clip(seas, 0.05, None) * np.clip(noise, 0.05, None)


def _ar2(rng: np.random.Generator, n: int, L: int) -> np.ndarray:
    # Draw partial autocorrelations in (-1, 1) and map to AR(2) coeffs via
    # Levinson-Durbin, which guarantees stationarity. Bias p1 high for
    # persistent (sometimes near-unit-root) series.
    p1 = rng.uniform(0.3, 0.98, size=n)
    p2 = rng.uniform(-0.6, 0.6, size=n)
    a2 = p2
    a1 = p1 * (1.0 - p2)
    sigma = rng.uniform(0.2, 0.8, size=(n, 1))
    innov = rng.normal(0.0, 1.0, size=(n, L)) * sigma
    x = _ar2_batch(innov, a1, a2)
    drift = rng.normal(0.0, 0.005, size=(n, 1)) * np.arange(L, dtype=np.float64)[None, :]
    return x + drift


def _integrated(rng: np.random.Generator, n: int, L: int) -> np.ndarray:
    order2 = rng.random(n) < 0.35
    drift = rng.normal(0.0, 0.02, size=(n, 1))
    sigma = rng.uniform(0.2, 1.0, size=(n, 1))
    steps = rng.normal(0.0, 1.0, size=(n, L)) * sigma + drift
    walk = np.cumsum(steps, axis=1)
    walk2 = np.cumsum(walk, axis=1)
    o2 = order2[:, None]
    # I(2) grows fast; damp it so it shares scale with the I(1) branch.
    return np.where(o2, walk2 / max(L, 1) ** 0.5, walk)


def _threshold_ar(rng: np.random.Generator, n: int, L: int) -> np.ndarray:
    # SETAR(2): coefficient flips with the sign of the previous value — a simple
    # nonlinear recurrence that produces asymmetric, regime-switching dynamics.
    phi_hi = rng.uniform(0.3, 0.9, size=n)
    phi_lo = rng.uniform(-0.9, 0.3, size=n)
    const_hi = rng.normal(0.0, 0.3, size=n)
    const_lo = rng.normal(0.0, 0.3, size=n)
    sigma = rng.uniform(0.2, 0.7, size=(n, 1))
    innov = rng.normal(0.0, 1.0, size=(n, L)) * sigma
    x = np.empty((n, L), dtype=np.float64)
    x[:, 0] = innov[:, 0]
    for t in range(1, L):
        prev = x[:, t - 1]
        hi = prev >= 0.0
        phi = np.where(hi, phi_hi, phi_lo)
        const = np.where(hi, const_hi, const_lo)
        x[:, t] = np.clip(const + phi * prev + innov[:, t], -1e6, 1e6)
    return x


def _chaotic(rng: np.random.Generator, n: int, L: int) -> np.ndarray:
    # Bounded chaotic maps: logistic x_{t+1}=r x(1-x) with r∈[3.6,4.0], and the
    # sine map r sin(pi x). Both stay in [0,1]; standardise afterwards. A random
    # observation length as a "sampling rate" adds variety across series.
    use_sine = rng.random(n) < 0.5
    r_log = rng.uniform(3.6, 4.0, size=n)
    r_sin = rng.uniform(0.85, 1.0, size=n)
    x0 = rng.uniform(0.05, 0.95, size=n)
    x = np.empty((n, L), dtype=np.float64)
    cur = x0.copy()
    x[:, 0] = cur
    for t in range(1, L):
        nxt_log = r_log * cur * (1.0 - cur)
        nxt_sin = r_sin * np.sin(np.pi * cur)
        cur = np.where(use_sine, nxt_sin, nxt_log)
        cur = np.clip(cur, 0.0, 1.0)
        x[:, t] = cur
    return x


def _spectral_gp(rng: np.random.Generator, n: int, L: int) -> np.ndarray:
    """Smooth stationary GP-like paths sampled in O(n L log L).

    An RBF kernel has a Gaussian spectral density. Drawing complex Fourier
    coefficients under that envelope and applying one batched inverse FFT
    preserves the useful smoothness/length-scale prior without the old
    48-pass cosine loop.
    """
    f = np.fft.rfftfreq(L)[None, :]
    lengthscale = np.exp(rng.uniform(np.log(8.0), np.log(256.0), size=(n, 1)))
    envelope = np.exp(-0.5 * (2.0 * np.pi * lengthscale * f) ** 2)
    z = rng.standard_normal((n, f.shape[1])) + 1j * rng.standard_normal((n, f.shape[1]))
    z[:, 0] = 0.0
    x = np.fft.irfft(z * np.sqrt(envelope), n=L, axis=1)
    sd = x.std(axis=1, keepdims=True)
    return x / np.where(sd < 1e-12, 1.0, sd)


def _long_memory(rng: np.random.Generator, n: int, L: int) -> np.ndarray:
    """Fractional power-law paths with both persistent and rough regimes.

    The spectral slope beta spans anti-persistent noise through persistent
    long-memory levels. A minority of rows are integrated once to include
    nonstationary fBm-like paths; row standardisation keeps scales bounded.
    """
    f = np.fft.rfftfreq(L)
    safe_f = np.maximum(f, 1.0 / L)[None, :]
    beta = rng.uniform(-0.6, 2.4, size=(n, 1))
    amp = safe_f ** (-0.5 * beta)
    # Some rows change roughness above a random frequency, giving smooth
    # large-scale structure and rough local variation (or the reverse) without
    # another FFT. Match amplitudes at the split to avoid a spectral jump.
    multiscale = rng.random((n, 1)) < 0.4
    split_idx = rng.integers(8, max(9, f.size // 3), size=(n, 1))
    split_f = np.maximum(split_idx / L, 1.0 / L)
    beta_hi = rng.uniform(-0.6, 2.8, size=(n, 1))
    above = np.arange(f.size)[None, :] > split_idx
    amp_hi = split_f ** (-0.5 * beta) \
        * (safe_f / split_f) ** (-0.5 * beta_hi)
    amp = np.where(multiscale & above, amp_hi, amp)
    amp[:, 0] = 0.0
    z = rng.standard_normal((n, f.size)) + 1j * rng.standard_normal((n, f.size))
    x = np.fft.irfft(z * amp, n=L, axis=1)
    integrate = rng.random(n) < 0.25
    if integrate.any():
        x[integrate] = np.cumsum(x[integrate], axis=1)
    x -= x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    return x / np.where(sd < 1e-12, 1.0, sd)


def _ou_stochastic_vol(rng: np.random.Generator, n: int, L: int) -> np.ndarray:
    """Regime-switching mean reversion with bounded stochastic volatility.

    This is a CPU-cheap discrete Euler/AR analogue of TempoPFN's highest-impact
    OU SDE prior. Regime paths, seasonal means, volatility envelopes, and
    heavy-tail masks are sampled in whole blocks; only the state recurrence
    scans time, vectorised across all rows.
    """
    # Toggle between a fast/quiet and a slow/volatile regime. A cumulative XOR
    # builds persistent Markov-like paths without a per-row Python loop.
    switch_rate = np.exp(rng.uniform(np.log(0.001), np.log(0.15), size=(n, 1)))
    switches = rng.random((n, L)) < switch_rate
    switches[:, 0] = rng.random(n) < 0.5
    regime = np.bitwise_and(np.cumsum(switches, axis=1), 1).astype(np.int8)

    # One mean-reversion speed per row lets SciPy execute the recurrence in
    # compiled code. Regime paths still switch equilibrium mean and volatility;
    # rows span both fast/quiet and slow/persistent reversion rates.
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

    sigma0 = rng.lognormal(np.log(0.3), 0.3, size=(n, 1))
    sigma1 = rng.lognormal(np.log(1.5), 0.5, size=(n, 1))
    base_sigma = np.where(regime == 0, sigma0, sigma1)
    log_vol = np.cumsum(
        _sparse_jumps(rng, n, L, rate=8.0 / L, scale=0.35), axis=1
    )
    log_vol -= log_vol.mean(axis=1, keepdims=True)
    vol = base_sigma * np.exp(np.clip(log_vol, -1.5, 1.5))

    eps = rng.standard_normal((n, L))
    heavy = np.nonzero(rng.random(n) < 0.35)[0]
    if heavy.size:
        # Replace only heavy-tailed rows; drawing Student-t noise for every row
        # previously discarded 65% of that relatively expensive work.
        eps[heavy] = (
            rng.standard_t(4.0, size=(heavy.size, L)) / np.sqrt(2.0)
        )
    shocks = rng.random((n, L)) < (3.0 / L)
    shock_rows, shock_cols = np.nonzero(shocks)
    # As with sparse jumps, draw shock magnitudes only at the O(n) events.
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
    """Generic physical measurements without matching one private dataset.

    Four row-level archetypes cover smooth signed measurements, bounded
    percentages, pressure-like wandering levels, and non-negative skewed
    magnitudes. All share multi-cadence seasonality, smooth synoptic variation,
    and sparse fronts/gusts.
    """
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
        walk = np.cumsum(rng.standard_normal((count, L)), axis=1) / np.sqrt(L)
        level = rng.uniform(900.0, 1100.0, size=(count, 1))
        out[pressure] = level + rng.uniform(2.0, 15.0, size=(count, 1)) * walk \
            + 2.0 * fronts[pressure] + 0.5 * seasonal[pressure]

    magnitude = kind == 3
    if magnitude.any():
        count = int(magnitude.sum())
        gusts = (rng.random((count, L)) < (8.0 / L)) \
            * rng.lognormal(0.0, 0.8, size=(count, L))
        power = rng.uniform(1.0, 1.6, size=(count, 1))
        out[magnitude] = np.abs(base[magnitude]) ** power + gusts

    return out


def _seasonal_counts(rng: np.random.Generator, n: int, L: int) -> np.ndarray:
    """Seasonal Poisson/negative-binomial counts with decaying bursts.

    This keeps count positivity and discreteness intact while covering
    overdispersion, cadence-linked rate variation, slow signed growth, and
    release/news-like bursts. Computation remains batched across rows.
    """
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
    # A minority carry explicit calendar interaction: intraday cadence plus
    # seven day-specific factors, with a randomized weekend dip or lift.
    calendar = rng.random((n, 1)) < 0.35
    day_period = rng.choice([24, 48, 96, 144], size=(n, 1))
    day_idx = (np.floor_divide(np.arange(L)[None, :], day_period) % 7).astype(np.int64)
    day_factors = rng.normal(0.0, 0.12, size=(n, 7))
    day_factors[:, 5:] += rng.uniform(-0.8, 0.3, size=(n, 1))
    calendar_effect = np.take_along_axis(day_factors, day_idx, axis=1)
    log_rate += calendar * calendar_effect
    excursion = rng.uniform(-0.5, 0.5, size=(n, 1))
    log_rate += excursion * t / max(L - 1, 1)

    # Sparse positive impulses filtered by row-specific decay create bursts
    # without a Python loop over timesteps.
    impulses = (
        (rng.random((n, L)) < (2.0 / L))
        * rng.uniform(1.0, 10.0, size=(n, L))
    )
    burst = _ar1_batch(impulses, rng.uniform(0.85, 0.995, size=(n, 1)))
    base = np.exp(rng.uniform(np.log(3.0), np.log(3000.0), size=(n, 1)))
    lam = base * np.exp(np.clip(log_rate, -5.0, 5.0)) * (1.0 + burst)
    np.clip(lam, 0.0, 1.0e7, out=lam)

    # A gamma-mixed Poisson is negative-binomial marginally and provides
    # realistic overdispersion. Half the rows remain ordinary Poisson.
    overdispersed = rng.random((n, 1)) < 0.5
    shape = rng.uniform(0.5, 4.0, size=(n, 1))
    mixed = lam * rng.gamma(shape, 1.0 / shape, size=(n, L))
    return rng.poisson(np.where(overdispersed, mixed, lam)).astype(np.float64)


def _intermittent(rng: np.random.Generator, n: int, L: int) -> np.ndarray:
    # Seasonal zero-inflated demand. Occurrence probabilities vary by cadence
    # instead of being iid, teaching the model forecastable sparse structure.
    t = np.arange(L, dtype=np.float64)[None, :]
    base_p = rng.uniform(0.03, 0.35, size=(n, 1))
    period = rng.choice([7.0, 12.0, 24.0, 48.0, 168.0], size=(n, 1))
    season = rng.uniform(0.2, 1.2, size=(n, 1)) * np.sin(
        2.0 * np.pi * t / period + rng.uniform(0.0, 2.0 * np.pi, size=(n, 1))
    )
    logit = np.log(base_p / (1.0 - base_p)) + season
    p = 1.0 / (1.0 + np.exp(-logit))
    occur = (rng.random((n, L)) < p).astype(np.float64)
    magnitude = (
        rng.gamma(shape=2.0, scale=1.0, size=(n, L))
        * rng.uniform(1.0, 10.0, size=(n, 1))
        * np.exp(0.25 * season)
    )
    baseline = rng.uniform(0.0, 0.5, size=(n, 1))
    return baseline + occur * magnitude


def _pulse_outlier(rng: np.random.Generator, n: int, L: int) -> np.ndarray:
    # A smooth base with isolated outliers, persistent shock/recovery responses,
    # and genuine held-constant runs.
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

    # Sparse event loops, not a time-axis scan: typically two starts per row.
    starts = rng.random((n, L)) < (2.0 / L)
    starts[:, 0] = False
    for row in range(n):
        for start in np.nonzero(starts[row])[0]:
            run = int(rng.integers(3, 65))
            end = min(int(start) + run, L)
            series[row, start:end] = series[row, start - 1]
    return series


# ── final safety gate ───────────────────────────────────────────────────────


def _sanitize(block: np.ndarray) -> np.ndarray:
    """Guarantee the contract: finite float64, no NaN/inf, bounded magnitude.

    The trainer's ``check_series`` rejects any non-finite value, which would
    fail the whole run — so this is the hard backstop after every family
    builder. Replaces non-finite values and clips to a generous bound.
    """
    x = np.asarray(block, dtype=np.float64)
    np.nan_to_num(x, copy=False, nan=0.0, posinf=1e6, neginf=-1e6)
    np.clip(x, -1e6, 1e6, out=x)
    return x
