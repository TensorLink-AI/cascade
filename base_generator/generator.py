"""cascade genesis "base generator".

A single :class:`Generator` (``cascade.interface.DataGenerator``) that adapts a
curated subset of TempoPFN's procedural time-series priors into one deterministic
corpus source. Ten families are mixed by configurable weights:

    ForecastPFN, SineWave, SawTooth, Step, Anomaly, Spikes, OrnsteinUhlenbeck,
    GP-prior, KernelSynth, CauKer

Everything is vendored under ``tempo_gen/`` (import-rewritten from TempoPFN's
``src/``). The GP-prior (gpytorch), KernelSynth (scikit-learn) and CauKer
(networkx + scikit-learn) families were added in v2: their dependencies are now
on cascade's allowlist (see ``chain.toml [dependencies]``). The TempoPFN
ablation shows this GP/kernel family carries a large share of the downstream
signal, which is why it was the priority add. The pyo-backed *audio* generators
remain excluded — pyo runs a real-time audio server and seeds via ``hash()``,
both of which break the cross-process determinism contract below.

Determinism is the load-bearing property: the emitted corpus is a pure function
of ``(seed, n_series)`` only. We seed NumPy, torch and Python ``random`` from
``seed``, run torch on CPU with deterministic algorithms, derive every
per-generator and per-series sub-seed deterministically, and use a separate
seeded RNG for length-band cropping. The upstream ``hash()``-based seed offset
(PYTHONHASHSEED-salted, not reproducible across processes) is replaced with a
stable ``zlib.crc32`` in the vendored ``abstract_classes.py``. CauKer's upstream
GP draw used ``cupy`` on the GPU; the vendored copy draws with NumPy's seeded
``multivariate_normal`` instead, keeping the path CPU-only and reproducible.
"""

from __future__ import annotations

import json
import os
import sys
import random as _py_random
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from cascade.interface import DataGenerator

# The trainer imports this file by path (importlib.spec_from_file_location), so the
# vendored ``tempo_gen`` package next to it is not on sys.path by default. Add this
# file's own directory so ``import tempo_gen`` resolves however we are loaded.
# (``os``/``sys`` are not on the static-guard blocklist; only ``os.system`` is.)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Vendored TempoPFN generator wrappers (Apache-2.0; see NOTICE).
from tempo_gen.synthetic_generation.anomalies.anomaly_generator_wrapper import (
    AnomalyGeneratorWrapper,
)
from tempo_gen.synthetic_generation.cauker.cauker_generator_wrapper import (
    CauKerGeneratorWrapper,
)
from tempo_gen.synthetic_generation.forecast_pfn_prior.forecast_pfn_generator_wrapper import (
    ForecastPFNGeneratorWrapper,
)
from tempo_gen.synthetic_generation.generator_params import (
    AnomalyGeneratorParams,
    CauKerGeneratorParams,
    ForecastPFNGeneratorParams,
    GPGeneratorParams,
    KernelGeneratorParams,
    OrnsteinUhlenbeckProcessGeneratorParams,
    SawToothGeneratorParams,
    SineWaveGeneratorParams,
    SpikesGeneratorParams,
    StepGeneratorParams,
)
from tempo_gen.synthetic_generation.gp_prior.gp_generator_wrapper import (
    GPGeneratorWrapper,
)
from tempo_gen.synthetic_generation.kernel_synth.kernel_generator_wrapper import (
    KernelGeneratorWrapper,
)
from tempo_gen.synthetic_generation.ornstein_uhlenbeck_process.ou_generator_wrapper import (
    OrnsteinUhlenbeckProcessGeneratorWrapper,
)
from tempo_gen.synthetic_generation.sawtooth.sawtooth_generator_wrapper import (
    SawToothGeneratorWrapper,
)
from tempo_gen.synthetic_generation.sine_waves.sine_wave_generator_wrapper import (
    SineWaveGeneratorWrapper,
)
from tempo_gen.synthetic_generation.spikes.spikes_generator_wrapper import (
    SpikesGeneratorWrapper,
)
from tempo_gen.synthetic_generation.steps.step_generator_wrapper import (
    StepGeneratorWrapper,
)

# Default mixing weights (need not sum to 1; they are normalised). Bias rationale:
# ForecastPFN (rich trend × multi-seasonal × Weibull-noise families), the
# regime-switching OU process (stochastic volatility + trends + seasonality) and
# the GP/kernel family (GP-prior, KernelSynth, CauKer) carry the most diverse
# downstream signal, so they get the bulk of the mass despite being the most
# expensive to draw (the GP families do an O(L^3) covariance factorisation per
# series); the cheap periodic / step / spike / anomaly families round out regime
# coverage at near-zero cost.
_DEFAULT_WEIGHTS: dict[str, float] = {
    "forecast_pfn": 0.16,
    "ornstein_uhlenbeck": 0.12,
    "gp": 0.12,
    "kernel_synth": 0.12,
    "cauker": 0.08,
    "sine_waves": 0.10,
    "steps": 0.08,
    "sawtooth": 0.08,
    "anomalies": 0.07,
    "spikes": 0.07,
}

# (wrapper class, params class) per family key.
_FAMILIES: dict[str, tuple[type, type]] = {
    "forecast_pfn": (ForecastPFNGeneratorWrapper, ForecastPFNGeneratorParams),
    "ornstein_uhlenbeck": (OrnsteinUhlenbeckProcessGeneratorWrapper, OrnsteinUhlenbeckProcessGeneratorParams),
    "gp": (GPGeneratorWrapper, GPGeneratorParams),
    "kernel_synth": (KernelGeneratorWrapper, KernelGeneratorParams),
    "cauker": (CauKerGeneratorWrapper, CauKerGeneratorParams),
    "sine_waves": (SineWaveGeneratorWrapper, SineWaveGeneratorParams),
    "steps": (StepGeneratorWrapper, StepGeneratorParams),
    "sawtooth": (SawToothGeneratorWrapper, SawToothGeneratorParams),
    "anomalies": (AnomalyGeneratorWrapper, AnomalyGeneratorParams),
    "spikes": (SpikesGeneratorWrapper, SpikesGeneratorParams),
}

# Keep per-series sub-seeds inside [0, 2**32) — the anomaly/spikes generators call
# np.random.seed(), which rejects seeds >= 2**32.
_SEED_MOD = 2_000_000_000


class Generator(DataGenerator):
    """Mix of vendored TempoPFN priors, emitted as a deterministic corpus."""

    def __init__(self, config_dir: str, *, seed: int) -> None:
        cfg_path = Path(config_dir) / "config.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {}

        self._seed = int(seed)
        self._min_len = int(cfg.get("min_length", 64))
        self._max_len = int(cfg.get("max_length", 4096))
        if not (1 <= self._min_len <= self._max_len):
            raise ValueError(f"invalid length band [{self._min_len}, {self._max_len}]")
        # Generators are drawn at this length, then random-cropped into the band.
        self._gen_len = int(cfg.get("generate_length", self._max_len))
        self._gen_len = max(self._gen_len, self._max_len)
        self._batch = max(1, int(cfg.get("batch_size", 256)))

        # Sanitisation knobs. By default we only repair non-finite values and apply
        # a generous absolute clip; we do NOT force unit scale, because varied
        # realistic scales are themselves useful signal for a from-scratch model.
        self._max_abs = float(cfg.get("max_abs_value", 1.0e6))
        self._clip_sigma = float(cfg.get("clip_sigma", 0.0))  # 0 disables sigma clip
        self._standardize = bool(cfg.get("standardize", False))

        weights = cfg.get("weights", _DEFAULT_WEIGHTS)
        # Restrict to known families with positive weight, preserve a fixed order.
        self._weights = {
            k: float(weights[k])
            for k in _FAMILIES
            if k in weights and float(weights[k]) > 0.0
        }
        if not self._weights:
            self._weights = dict(_DEFAULT_WEIGHTS)

        # Determinism flags (CPU only, no CUDA on the generate path).
        np.random.seed(self._seed % 2**31)
        _py_random.seed(self._seed)
        try:
            import torch

            torch.manual_seed(self._seed)
            torch.use_deterministic_algorithms(True)
            torch.set_num_threads(1)  # avoid nondeterministic thread reductions
        except Exception:  # pragma: no cover - torch is an allowlisted dep, but be safe
            pass

    @property
    def name(self) -> str:
        return "tempopfn-base-mix-v1"

    # ── allocation ──────────────────────────────────────────────────────────
    def _allocate(self, n_series: int) -> list[tuple[str, int]]:
        """Split ``n_series`` across families by weight (largest-remainder).

        Pure function of (weights, n_series) — no RNG — so the allocation is
        identical across processes.
        """
        keys = list(self._weights)
        total_w = sum(self._weights[k] for k in keys)
        raw = {k: n_series * self._weights[k] / total_w for k in keys}
        floor = {k: int(np.floor(raw[k])) for k in keys}
        assigned = sum(floor.values())
        remainder = n_series - assigned
        # Hand out the remaining slots to the largest fractional parts, breaking
        # ties by fixed key order.
        order = sorted(keys, key=lambda k: (-(raw[k] - floor[k]), keys.index(k)))
        for i in range(remainder):
            floor[order[i % len(order)]] += 1
        return [(k, floor[k]) for k in keys if floor[k] > 0]

    def _sub_seed(self, *parts: int) -> int:
        """Deterministic child seed in [0, _SEED_MOD) from the master seed."""
        ss = np.random.SeedSequence([self._seed, *parts])
        return int(ss.generate_state(1, dtype=np.uint32)[0]) % _SEED_MOD

    # ── raw draws ───────────────────────────────────────────────────────────
    def _raw_stream(self, family: str, base_seed: int, chunk: int) -> Iterator[np.ndarray]:
        """Yield an unbounded stream of raw full-length series for ``family``.

        Draws are made in contiguous batches of ``chunk`` series with
        non-overlapping per-series seeds, so the stream is a deterministic
        function of ``(base_seed, chunk)``. ``chunk`` is sized to the demand so we
        never generate a full batch to use only a handful of series.
        """
        wrapper_cls, params_cls = _FAMILIES[family]
        params = params_cls(global_seed=base_seed, length=self._gen_len)
        wrapper = wrapper_cls(params)
        chunk = max(1, chunk)
        batch_seed = base_seed
        while True:
            batch = wrapper.generate_batch(batch_size=chunk, seed=batch_seed % _SEED_MOD)
            values = np.asarray(batch.values)
            if values.ndim == 1:
                values = values[None, :]
            elif values.ndim == 3:
                # Multivariate families (CauKer) emit [batch, seq_len, channels].
                # Flatten each channel into its own univariate series so the
                # emitted corpus stays 1-D like every other family.
                values = np.moveaxis(values, 2, 1).reshape(-1, values.shape[1])
            for row in values:
                yield np.ascontiguousarray(row)
            # Advance past this batch's per-series seeds (wrapper uses seed + i).
            batch_seed += chunk

    # ── sanitisation ────────────────────────────────────────────────────────
    def _sanitize(self, arr: np.ndarray, length: int, fallback_rng: np.random.Generator) -> np.ndarray:
        """Return a finite float64 1-D array of exactly ``length`` samples."""
        x = np.asarray(arr, dtype=np.float64).ravel()
        if x.size != length:
            # Defensive: crop/pad to the requested length.
            if x.size > length:
                x = x[:length]
            else:
                x = np.concatenate([x, np.full(length - x.size, x[-1] if x.size else 0.0)])
        # Repair non-finite values, then clip to a trainer-safe magnitude.
        if not np.isfinite(x).all():
            x = np.nan_to_num(x, nan=0.0, posinf=self._max_abs, neginf=-self._max_abs)
        np.clip(x, -self._max_abs, self._max_abs, out=x)

        if self._standardize:
            std = x.std()
            if std > 1e-12:
                x = (x - x.mean()) / std
        if self._clip_sigma > 0.0:
            mu, sd = x.mean(), x.std()
            if sd > 1e-12:
                np.clip(x, mu - self._clip_sigma * sd, mu + self._clip_sigma * sd, out=x)

        if not np.isfinite(x).all():
            # Last-resort deterministic replacement (should not happen post-repair).
            x = fallback_rng.standard_normal(length)
        return np.ascontiguousarray(x, dtype=np.float64)

    # ── main entrypoint ───────────────────────────────────────────────────────
    def generate(self, n_series: int) -> Iterator[np.ndarray]:
        if n_series <= 0:
            return
        # Master RNG drives length-band cropping only — kept separate from every
        # generator's internal RNG so the crop sequence is order-deterministic.
        crop_rng = np.random.default_rng(self._sub_seed(0xC0FFEE))
        fallback_rng = np.random.default_rng(self._sub_seed(0xFA11BACC))

        allocation = self._allocate(n_series)
        emitted = 0
        for fam_idx, (family, count) in enumerate(allocation):
            base_seed = self._sub_seed(fam_idx + 1)
            main_chunk = min(self._batch, count)
            main = self._raw_stream(family, base_seed, main_chunk)
            regen = self._raw_stream(family, (base_seed + _SEED_MOD // 2) % _SEED_MOD, min(self._batch, 16))
            for _ in range(count):
                # Draw the crop window first so master-RNG state advances exactly
                # once per emitted series, regardless of any repair path.
                length = int(crop_rng.integers(self._min_len, self._max_len + 1))
                max_off = self._gen_len - length
                offset = int(crop_rng.integers(0, max_off + 1)) if max_off > 0 else 0

                raw = next(main)
                window = raw[offset:offset + length]
                series = self._sanitize(window, length, fallback_rng)
                if not np.isfinite(series).all():
                    raw2 = next(regen)
                    window = raw2[offset:offset + length]
                    series = self._sanitize(window, length, fallback_rng)
                yield series
                emitted += 1

        # Allocation sums to n_series by construction, but guard the contract.
        if emitted != n_series:  # pragma: no cover
            raise RuntimeError(f"emitted {emitted} series; expected {n_series}")
