"""Lane A e2e test generator: coupled 8-channel synthetic series, numpy only.

Each series is a driver (sum of seasonal sines, a random walk trend and AR(1)
noise) plus seven responders that depend on it: lagged copies, exponential
smoothings, an integral and an inverse response.
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from cascade.interface import DataGenerator

_C = 8


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
        return "lane-a-e2e-mv8"

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
        out = np.empty((_C, L))
        out[0] = x
        for k, lag in enumerate((1, 3, 7), start=1):
            out[k, :lag] = x[0]
            out[k, lag:] = x[:-lag]
            out[k] = 0.8 * out[k] + 0.05 * s * rng.standard_normal(L)
        for k, a in ((4, 0.1), (5, 0.3)):
            acc = x[0]
            for i in range(L):
                acc = a * x[i] + (1.0 - a) * acc
                out[k, i] = acc
        out[6] = np.cumsum(x - x.mean()) / L
        out[7] = -0.5 * x + 0.1 * s * rng.standard_normal(L)
        return out

    def generate(self, n_series: int) -> Iterator[np.ndarray]:
        rng = np.random.default_rng(self._seed)
        for _ in range(max(0, int(n_series))):
            L = int(rng.integers(self._min_len, self._max_len + 1))
            yield np.ascontiguousarray(self._couple(self._driver(rng, L), rng))
