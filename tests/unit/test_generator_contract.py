"""DataGenerator output checks: check_series + drain_generator."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

from cascade.interface.generator import DataGenerator, check_series, drain_generator


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
