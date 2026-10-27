"""Lane A e2e test generator: coupled 32-channel synthetic series, numpy only.

Each series is a driver (seasonal sines, a random-walk trend and AR(1) noise)
plus 31 responders that depend on it: lagged copies, exponential smoothings at
several rates, differences and an inverse response, each lightly noised.
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from cascade.interface import DataGenerator

_C = 32
_LAGS = (1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31)
_ALPHAS = (0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7)


class Generator(DataGenerator):
    def __init__(self, config_dir: str, *, seed: int) -> None:
        cfg_path = Path(config_dir) / "config.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {}
        # The e2e runner writes a per-scenario salt so every submission has its own
        # corpus digest (the trainer refuses duplicate source).
        self._seed = int(seed) ^ int(cfg.get("salt", 0))
        self._min_len = 256
        self._max_len = 1024

    @property
    def name(self) -> str:
        return "lane-a-e2e-mv32"

    def _driver(self, rng: np.random.Generator, L: int) -> np.ndarray:
        t = np.arange(L, dtype=np.float64)
        x = np.zeros(L)
        for _ in range(int(rng.integers(1, 4))):
            period = float(rng.choice([7.0, 12.0, 24.0, 52.0, 168.0]))
            x += rng.uniform(0.2, 2.0) * np.sin(2 * np.pi * t / period + rng.uniform(0, 2 * np.pi))
        x += np.cumsum(rng.standard_normal(L)) * rng.uniform(0.0, 0.1)
        phi = rng.uniform(0.0, 0.9)
        e = rng.standard_normal(L) * rng.uniform(0.05, 0.5)
        ar = np.empty(L)
        ar[0] = e[0]
        for i in range(1, L):
            ar[i] = phi * ar[i - 1] + e[i]
        return x + ar + rng.uniform(-5, 5)

    def _couple(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        L = x.shape[0]
        s = float(np.std(x)) or 1.0
        rows = [x]
        for lag in _LAGS:
            r = np.empty(L)
            r[:lag] = x[0]
            r[lag:] = x[:-lag]
            rows.append(0.8 * r + 0.05 * s * rng.standard_normal(L))
        alphas = np.asarray(_ALPHAS)
        ema = np.empty((alphas.size, L))
        acc = np.full(alphas.size, x[0])
        for i in range(L):
            acc = alphas * x[i] + (1.0 - alphas) * acc
            ema[:, i] = acc
        rows.extend(ema)
        d = np.diff(x, prepend=x[0])
        rows.append(d)
        rows.append(np.cumsum(x - x.mean()) / L)
        rows.append(-0.5 * x + 0.1 * s * rng.standard_normal(L))
        rows.append(np.abs(d) + 0.01 * s * rng.standard_normal(L))
        while len(rows) < _C:
            rows.append(rows[1 + len(rows) % 11] * 0.5 + 0.1 * s * rng.standard_normal(L))
        return np.stack(rows[:_C])

    def generate(self, n_series: int) -> Iterator[np.ndarray]:
        rng = np.random.default_rng(self._seed)
        for _ in range(max(0, int(n_series))):
            L = int(rng.integers(self._min_len, self._max_len + 1))
            yield np.ascontiguousarray(self._couple(self._driver(rng, L), rng))
