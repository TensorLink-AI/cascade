"""Synthetic source — deterministic, offline, network-free.

Not a real-world feed and **not for a production pool** (it is exactly the kind
of synthetic, gameable distribution the eval avoids). Its job is to exercise the
full build → write → load path in tests and in an offline ``metronome-pool
build --sources synthetic`` smoke run, with byte-stable output for a given
``as_of`` + count.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from ..source import FetchJson, HarvestContext, HarvestedSeries


class SyntheticSource:
    name = "synthetic"

    def __init__(self, n_series: int = 64) -> None:
        self.n_series = n_series

    def harvest(self, fetch: FetchJson, ctx: HarvestContext) -> Iterable[HarvestedSeries]:
        length = ctx.context_length + ctx.horizon
        n = min(self.n_series, ctx.max_series)
        # Seed from the as_of ordinal so a given day reproduces, but rotates daily.
        base_seed = ctx.as_of.toordinal()
        t = np.arange(length, dtype=np.float64)
        for i in range(n):
            rng = np.random.default_rng(base_seed * 100003 + i)
            period = float(rng.integers(12, 48))
            trend = rng.normal(0, 0.01)
            level = rng.normal(10, 3)
            signal = (
                level
                + trend * t
                + rng.uniform(1, 5) * np.sin(2 * np.pi * t / period)
                + rng.normal(0, 0.5, size=length)
            )
            yield HarvestedSeries(
                series_id=f"synthetic__s{i:05d}",
                values=signal,
                freq="H",
                domain="synthetic",
                seasonal_period=24,
                attrs={"period": period},
            )
