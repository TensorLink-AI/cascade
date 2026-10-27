"""Rejection fixture: opens a network connection while generating."""
from __future__ import annotations

import socket
from collections.abc import Iterator

import numpy as np

from cascade.interface import DataGenerator


class Generator(DataGenerator):
    def __init__(self, config_dir: str, *, seed: int) -> None:
        self._seed = int(seed)

    @property
    def name(self) -> str:
        return "e2e-reject-net-access"

    def generate(self, n_series: int) -> Iterator[np.ndarray]:
        with socket.create_connection(("1.1.1.1", 443), timeout=5):
            pass
        rng = np.random.default_rng(self._seed)
        for _ in range(max(0, int(n_series))):
            x = np.cumsum(rng.standard_normal(512))
            yield np.stack([x * (k + 1) for k in range(32)])
