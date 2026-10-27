"""Rejection fixture: one channel per yield (v2 Lane A requires 8-32)."""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from cascade.interface import DataGenerator


class Generator(DataGenerator):
    def __init__(self, config_dir: str, *, seed: int) -> None:
        self._seed = int(seed)

    @property
    def name(self) -> str:
        return "e2e-reject-univariate"

    def generate(self, n_series: int) -> Iterator[np.ndarray]:
        rng = np.random.default_rng(self._seed)
        for _ in range(max(0, int(n_series))):
            yield np.cumsum(rng.standard_normal(512))
