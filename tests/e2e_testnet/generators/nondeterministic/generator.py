"""Rejection fixture: ignores the seed, so two runs disagree."""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from cascade.interface import DataGenerator


class Generator(DataGenerator):
    def __init__(self, config_dir: str, *, seed: int) -> None:
        self._rng = np.random.default_rng()  # Deliberately unseeded, without a blocked import.

    @property
    def name(self) -> str:
        return "e2e-reject-nondeterministic"

    def generate(self, n_series: int) -> Iterator[np.ndarray]:
        rng = self._rng
        for _ in range(max(0, int(n_series))):
            x = np.cumsum(rng.standard_normal(512))
            yield np.stack([x + 0.1 * k * rng.standard_normal(512) for k in range(32)])
