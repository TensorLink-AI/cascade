"""Gap probe: a packed module SHORTER than static_guard.MIN_PACKED_CHARS hiding a
blocked import. The static scan does not decode it; only the runtime sandbox
stands between it and the network. Reported as WARN while it verifies."""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from cascade.interface import DataGenerator

_NS: dict = {}
exec("import socket\ndef probe():\n    return socket.gethostname()\n", _NS)


class Generator(DataGenerator):
    def __init__(self, config_dir: str, *, seed: int) -> None:
        self._seed = int(seed)

    @property
    def name(self) -> str:
        return "e2e-gap-short-packed"

    def generate(self, n_series: int) -> Iterator[np.ndarray]:
        _NS["probe"]()
        rng = np.random.default_rng(self._seed)
        for _ in range(max(0, int(n_series))):
            x = np.cumsum(rng.standard_normal(512))
            yield np.stack([x * (k + 1) + 0.05 * rng.standard_normal(512) for k in range(8)])
