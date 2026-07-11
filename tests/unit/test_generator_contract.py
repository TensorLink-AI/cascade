"""DataGenerator output checks: check_series + drain_generator."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

from cascade.interface.generator import (
    CAST_SAFE_MAX_FLOAT32,
    DataGenerator,
    check_series,
    drain_generator,
)


class _Good(DataGenerator):
    def __init__(self, config_dir: str = ".", *, seed: int = 0) -> None:
        self._seed = seed

    @property
    def name(self) -> str:
        return "good"

    def generate(self, n_series: int) -> Iterator[np.ndarray]:
        rng = np.random.default_rng(self._seed)
        for _ in range(n_series):
            yield rng.standard_normal(100)


def test_check_series_accepts_valid():
    check_series(np.zeros(100), min_length=10, max_length=200)


@pytest.mark.parametrize(
    "arr",
    [
        [1.0, 2.0],                       # not ndarray
        np.zeros((2, 2)),                 # 2-D
        np.arange(100),                   # int dtype
        np.zeros(5),                      # too short
        np.zeros(500),                    # too long
        np.array([np.nan, 1.0, 2.0] * 10),  # non-finite
    ],
)
def test_check_series_rejects(arr):
    with pytest.raises(ValueError):
        check_series(arr, min_length=10, max_length=200)


def test_drain_generator_returns_exact_count():
    out = drain_generator(_Good(seed=1), 5, min_length=10, max_length=200, max_total_points=10_000)
    assert len(out) == 5
    assert all(a.dtype == np.float64 for a in out)


def test_drain_generator_determinism():
    a = drain_generator(_Good(seed=7), 4, min_length=10, max_length=200, max_total_points=10_000)
    b = drain_generator(_Good(seed=7), 4, min_length=10, max_length=200, max_total_points=10_000)
    for x, y in zip(a, b, strict=True):
        assert np.array_equal(x, y)


def test_drain_generator_enforces_point_cap():
    with pytest.raises(ValueError):
        drain_generator(_Good(seed=1), 5, min_length=10, max_length=200, max_total_points=150)


class _WrongCount(DataGenerator):
    def __init__(self, config_dir: str = ".", *, seed: int = 0) -> None:
        pass

    @property
    def name(self) -> str:
        return "wrong"

    def generate(self, n_series: int) -> Iterator[np.ndarray]:
        yield np.zeros(50)  # yields 1 regardless of n_series


def test_drain_generator_rejects_wrong_count():
    with pytest.raises(ValueError):
        drain_generator(_WrongCount(), 5, min_length=10, max_length=200, max_total_points=10_000)


# ── MV-ready schema (channel axis, default univariate) ──────────────────────


def test_check_series_promotes_1d_and_caps_channels():
    # 1-D is one channel and always allowed.
    check_series(np.zeros(100), min_length=10, max_length=200)
    # 2-D within the channel cap is allowed.
    check_series(np.zeros((3, 100)), min_length=10, max_length=200, max_channels=3)
    # Over the cap (default cap is 1 → univariate) is rejected.
    with pytest.raises(ValueError):
        check_series(np.zeros((2, 100)), min_length=10, max_length=200)
    # Length band applies to the time axis, not the channel axis.
    with pytest.raises(ValueError):
        check_series(np.zeros((1, 5)), min_length=10, max_length=200)


class _Multivariate(DataGenerator):
    def __init__(self, config_dir: str = ".", *, seed: int = 0) -> None:
        self._seed = seed

    @property
    def name(self) -> str:
        return "mv"

    def generate(self, n_series: int) -> Iterator[np.ndarray]:
        rng = np.random.default_rng(self._seed)
        for _ in range(n_series):
            yield rng.standard_normal((2, 100))


def test_drain_canonicalises_to_channel_first_and_counts_all_points():
    # Univariate generators are promoted to (1, L).
    uni = drain_generator(_Good(seed=1), 3, min_length=10, max_length=200, max_total_points=10_000)
    assert all(a.shape == (1, 100) for a in uni)

    # Multivariate is preserved when the channel cap allows it; the point budget
    # counts every emitted value (C * L), so 3 * (2*100) = 600 > 500 trips the cap.
    mv = drain_generator(
        _Multivariate(seed=1), 3, min_length=10, max_length=200,
        max_total_points=10_000, max_channels=2,
    )
    assert all(a.shape == (2, 100) for a in mv)
    with pytest.raises(ValueError):
        drain_generator(
            _Multivariate(seed=1), 3, min_length=10, max_length=200,
            max_total_points=500, max_channels=2,
        )


# ── cheap data-quality gates (all opt-in; default no-op) ────────────────────


def test_max_abs_gate_rejects_extreme_magnitude():
    # An extreme raw magnitude is what could drive the trainer's eps-floored
    # standardization ratio past the asinh overflow into NaN loss; the cap rejects
    # it well before that. (Finite in float64, so the existing checks pass it.)
    arr = np.full(100, 1e300)
    # No cap by default → accepted (unchanged behaviour).
    check_series(arr, min_length=10, max_length=200)
    # Under the cast-safe ceiling it is rejected.
    with pytest.raises(ValueError, match="cast-safe"):
        check_series(arr, min_length=10, max_length=200, max_abs=CAST_SAFE_MAX_FLOAT32)
    # A normal series passes the same cap.
    check_series(np.arange(100.0), min_length=10, max_length=200, max_abs=CAST_SAFE_MAX_FLOAT32)


def test_reject_constant_gate():
    flat = np.zeros(100)
    # Default: a constant series is still accepted (opt-in gate).
    check_series(flat, min_length=10, max_length=200)
    with pytest.raises(ValueError, match="constant"):
        check_series(flat, min_length=10, max_length=200, reject_constant=True)
    # A series with any variation passes.
    varied = np.arange(100.0)
    check_series(varied, min_length=10, max_length=200, reject_constant=True)


class _Constant(DataGenerator):
    def __init__(self, config_dir: str = ".", *, seed: int = 0) -> None:
        pass

    @property
    def name(self) -> str:
        return "constant"

    def generate(self, n_series: int) -> Iterator[np.ndarray]:
        for _ in range(n_series):
            yield np.full(100, 3.0)


def test_drain_reject_constant_flows_through():
    kw = dict(min_length=10, max_length=200, max_total_points=10_000)
    # Off by default.
    assert len(drain_generator(_Constant(), 4, **kw)) == 4
    with pytest.raises(ValueError, match="constant"):
        drain_generator(_Constant(), 4, reject_constant=True, **kw)


class _NCopies(DataGenerator):
    """Yields ``distinct`` unique series, then repeats the first to fill n_series."""

    def __init__(self, config_dir: str = ".", *, seed: int = 0, distinct: int = 1) -> None:
        self._seed = seed
        self._distinct = distinct

    @property
    def name(self) -> str:
        return "ncopies"

    def generate(self, n_series: int) -> Iterator[np.ndarray]:
        rng = np.random.default_rng(self._seed)
        uniques = [rng.standard_normal(100) for _ in range(self._distinct)]
        for i in range(n_series):
            yield uniques[i] if i < self._distinct else uniques[0]


def test_dup_fraction_gate():
    kw = dict(min_length=10, max_length=200, max_total_points=10_000)
    # 1 unique + 9 copies → dup fraction 0.9.
    gen = lambda: _NCopies(distinct=1)  # noqa: E731
    # Disabled by default and at 1.0.
    assert len(drain_generator(gen(), 10, **kw)) == 10
    assert len(drain_generator(gen(), 10, max_dup_fraction=1.0, **kw)) == 10
    # A tight cap trips on the copies.
    with pytest.raises(ValueError, match="duplicate-series fraction"):
        drain_generator(gen(), 10, max_dup_fraction=0.5, **kw)
    # All-distinct series never trip the cap.
    out = drain_generator(_NCopies(distinct=10), 10, max_dup_fraction=0.0, **kw)
    assert len(out) == 10
