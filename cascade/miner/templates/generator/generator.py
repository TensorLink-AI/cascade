"""Reference cascade generator — a template miners can fork.

Emits deterministic synthetic multivariate series: a mix of trend, multi-seasonal
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
        self._channels = int(self._cfg.get("channels", 8))
        if not 8 <= self._channels <= 32:
            raise ValueError("v2 generators require 8–32 channels per series")

    @property
    def name(self) -> str:
        return "reference-trend-seasonal-ar1"

    def generate(self, n_series: int) -> Iterator[np.ndarray | dict]:
        rng = np.random.default_rng(self._seed)
        for _ in range(n_series):
            length = int(rng.integers(self._min_len, self._max_len + 1))
            packing = self._cfg.get("packing", {})
            if not packing.get("enabled", False):
                yield self._draw_values(rng, length, self._channels)
                continue
            groups = self._draw_groups(rng, packing)
            values = np.empty((self._channels, length), dtype=np.float32)
            for label in range(int(groups.max()) + 1):
                channels = groups == label
                # A fresh independent driver per group; common time length only.
                values[channels] = self._draw_values(rng, length, int(channels.sum()))
            yield {"values": values, "groups": groups}

    def _draw_groups(self, rng, packing):
        bins = packing.get("size_bins", [[1, 1], [2, 4], [5, 8], [9, 16], [17, 32]])
        shares = np.asarray(packing.get("shares", [30, 25, 25, 15, 5]), dtype=float)
        if len(bins) != len(shares) or not np.isfinite(shares).all() or (shares < 0).any() or shares.sum() <= 0:
            raise ValueError("packing shares must be finite nonnegative weights matching size_bins")
        labels = []
        label = 0
        while len(labels) < self._channels:
            lo, hi = bins[int(rng.choice(len(bins), p=shares / shares.sum()))]
            if not 1 <= lo <= hi <= 32:
                raise ValueError("packing size bins must stay within 1–32")
            size = min(int(rng.integers(lo, hi + 1)), self._channels - len(labels))
            labels.extend([label] * size)
            label += 1
        return np.asarray(labels, dtype=np.uint8)

    @staticmethod
    def _draw_values(rng, length, channels):
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

        # Shared delayed drivers plus independent channel innovations.
        delays = rng.integers(0, 8, size=(channels, 1))
        times = np.maximum(np.arange(length)[None, :] - delays, 0)
        loading = rng.uniform(0.5, 1.5, size=(channels, 1))
        independent = rng.normal(0.0, 0.2, size=(channels, length))
        rho = rng.uniform(0.1, 0.7, size=channels)
        for i in range(1, length):
            independent[:, i] += rho * independent[:, i - 1]
        values = loading * series[times] + independent
        return values.astype(np.float64)
