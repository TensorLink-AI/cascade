"""Reference cascade generator — a template miners can fork.

Emits deterministic synthetic univariate series: a mix of trend, multi-seasonal
sinusoids, AR(1) noise, and occasional level shifts. Everything is driven by the
single ``seed`` passed in, so two runs at the same seed produce byte-identical
corpora — the determinism the trainer audits.

The submitted class MUST be named ``Generator`` and subclass
``cascade.interface.DataGenerator``. In a real submission you only need the
``cascade`` package importable to subclass it; the heavy lifting is your data
process.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from cascade.interface import DataGenerator


class Generator(DataGenerator):
    def __init__(self, config_dir: str, *, seed: int) -> None:
        cfg_path = Path(config_dir) / "config.json"
        self._cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {}
        self._seed = int(seed)
        self._min_len = int(self._cfg.get("min_length", 128))
        self._max_len = int(self._cfg.get("max_length", 1024))

    @property
    def name(self) -> str:
        return "reference-trend-seasonal-ar1"

    def generate(self, n_series: int) -> Iterator[np.ndarray]:
        rng = np.random.default_rng(self._seed)
        for _ in range(n_series):
            length = int(rng.integers(self._min_len, self._max_len + 1))
            t = np.arange(length, dtype=np.float64)

            # Trend.
            slope = rng.normal(0.0, 0.01)
            level = rng.normal(0.0, 1.0)
            series = level + slope * t

            # One or two seasonal components.
            for _ in range(int(rng.integers(1, 3))):
                period = float(rng.choice([7, 12, 24, 30, 52]))
                amp = rng.uniform(0.2, 2.0)
                phase = rng.uniform(0.0, 2.0 * np.pi)
                series += amp * np.sin(2.0 * np.pi * t / period + phase)

            # AR(1) noise.
            phi = rng.uniform(0.0, 0.8)
            sigma = rng.uniform(0.1, 0.5)
            noise = np.empty(length, dtype=np.float64)
            noise[0] = rng.normal(0.0, sigma)
            for i in range(1, length):
                noise[i] = phi * noise[i - 1] + rng.normal(0.0, sigma)
            series += noise

            # Occasional level shift.
            if rng.random() < 0.2:
                shift_at = int(rng.integers(length // 4, 3 * length // 4))
                series[shift_at:] += rng.normal(0.0, 2.0)

            yield series.astype(np.float64)
