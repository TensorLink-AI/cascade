"""Reference multivariate generator (DEC-CA-0041): causally-coupled channels
in the king's synthetic style, for the settling ablation (does multivariate
training beat univariate on the MV-aware eval, DEC-CA-0026).

Coupling follows the CauKer / Chronos-2-Synth recipe — a *random temporal
causal graph*: each series draws a small DAG over its channels, and a child
channel adds a lagged, weighted copy of each parent's PAST. That lag is the
whole point: a parent's history genuinely predicts the child's future, so a
joint forecaster that conditions across the variate axis lowers the child's
per-window error while a univariate model cannot — which is the only thing that
earns reward under per-variate GIFT scoring. The per-channel base (trend +
seasonality + noise) mirrors the deployed synthetic style, so at C=1 this is an
ordinary king-style univariate corpus.

Importable as ``generator.Generator`` (the trainer's contract). Emits ``(C, L)``
arrays; deterministic from ``seed`` and ``n_series`` alone. Until
``[generator] max_channels`` is raised the trainer rejects C>1 — this generator
exists to be run in the ablation once it is.
"""
from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pathlib import Path

import numpy as np

from .generator import DataGenerator


class Generator(DataGenerator):
    """Causally-coupled multivariate reference generator."""

    def __init__(self, config_dir: str, *, seed: int) -> None:
        self._seed = int(seed)
        cfg: dict = {}
        cfg_path = Path(config_dir) / "config.json" if config_dir else None
        if cfg_path is not None and cfg_path.is_file():
            try:
                cfg = json.loads(cfg_path.read_text(encoding="utf-8")) or {}
            except Exception:  # noqa: BLE001 — a bad config falls back to defaults
                cfg = {}
        self._max_channels = int(cfg.get("max_channels", 4))
        self._min_channels = int(cfg.get("min_channels", 2))
        self._length = int(cfg.get("length", 1024))
        self._max_delay = int(cfg.get("max_coupling_delay", 24))
        self._coupling = float(cfg.get("coupling_strength", 0.6))

    def generate(self, n_series: int) -> Iterator[np.ndarray | Mapping]:
        for i in range(int(n_series)):
            # per-series RNG derived from the ctor seed and the index only
            rng = np.random.default_rng((self._seed & 0xFFFFFFFFFFFF) * 1_000_003 + i)
            yield self._one_series(rng)

    # ------------------------------------------------------------------ core

    def _base_channel(self, rng: np.random.Generator, length: int) -> np.ndarray:
        """King-style univariate base: level + linear trend + a seasonal term +
        AR(1)-coloured noise. Finite by construction."""
        t = np.arange(length, dtype=np.float64)
        level = rng.normal(0.0, 3.0)
        trend = rng.normal(0.0, 0.01)
        period = float(rng.integers(8, 64))
        amp = rng.uniform(0.5, 3.0)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        seasonal = amp * np.sin(2.0 * np.pi * t / period + phase)
        # AR(1) innovations for a bit of autocorrelated structure
        phi = float(rng.uniform(0.0, 0.6))
        eps = rng.normal(0.0, 0.5, size=length)
        noise = np.empty(length)
        noise[0] = eps[0]
        for k in range(1, length):
            noise[k] = phi * noise[k - 1] + eps[k]
        return level + trend * t + seasonal + noise

    def _one_series(self, rng: np.random.Generator) -> np.ndarray:
        c = int(rng.integers(self._min_channels, self._max_channels + 1))
        length = self._length
        base = np.stack([self._base_channel(rng, length) for _ in range(c)], axis=0)
        out = base.copy()
        # random temporal causal DAG: channel k may draw parents from 0..k-1
        for k in range(1, c):
            n_parents = int(rng.integers(1, k + 1))
            parents = rng.choice(k, size=n_parents, replace=False)
            for p in parents:
                delay = int(rng.integers(1, self._max_delay + 1))
                w = self._coupling * float(rng.normal(1.0, 0.3))
                # child's value at t adds a weighted copy of parent p at t-delay:
                # the parent's PAST drives the child's FUTURE (cross-predictive).
                out[k, delay:] += w * out[p, :-delay]
        arr = out.astype(np.float64)
        # keep magnitudes tame (the trainer standardises, but stay well finite)
        return arr / max(1.0, float(np.abs(arr).max()) / 1e4)

    @property
    def name(self) -> str:
        return "mv-coupled-reference"
