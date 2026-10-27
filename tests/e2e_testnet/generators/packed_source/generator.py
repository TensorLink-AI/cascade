"""Rejection fixture: generator code kept in a string constant and exec'd."""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from cascade.interface import DataGenerator

# Long enough (>= static_guard.MIN_PACKED_CHARS) for the packed-source scan to decode it.
_SOURCE = """
import numpy as np


def make(rng, length):
    \"\"\"Eight channels driven by one random walk, each scaled and lightly noised.\"\"\"
    drivers = np.cumsum(rng.standard_normal(length))
    noise = 0.05 * rng.standard_normal((8, length))
    return np.stack([drivers * (k + 1) for k in range(8)]) + noise
"""

_NS: dict = {}
exec(compile(_SOURCE, "<packed>", "exec"), _NS)


class Generator(DataGenerator):
    def __init__(self, config_dir: str, *, seed: int) -> None:
        self._seed = int(seed)

    @property
    def name(self) -> str:
        return "e2e-reject-packed-source"

    def generate(self, n_series: int) -> Iterator[np.ndarray]:
        rng = np.random.default_rng(self._seed)
        for _ in range(max(0, int(n_series))):
            yield _NS["make"](rng, 512)
