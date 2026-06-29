"""Contract tests for the cascade base generator.

Adapted from cascade's ``tests/unit/test_generator_contract.py`` pattern, but
pointed at THIS repo's ``Generator``. Run from the repo root with:

    PYTHONPATH=/path/to/cascade pytest base_generator/tests -q

(``cascade`` must be importable; the generator's own deps — numpy/scipy/pandas/
torch — must be installed.)
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from cascade.interface.generator import check_series, drain_generator

REPO_DIR = Path(__file__).resolve().parents[1]

# Mirror chain.toml [generator] bounds.
MIN_LEN = 64
MAX_LEN = 2048
MAX_TOTAL_POINTS = 2_000_000_000
MAX_CHANNELS = 1


def _load_generator_cls():
    spec = importlib.util.spec_from_file_location("base_generator_under_test", REPO_DIR / "generator.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Generator


@pytest.fixture(scope="module")
def Generator():
    return _load_generator_cls()


def _drain(Generator, n, seed=0):
    gen = Generator(str(REPO_DIR), seed=seed)
    return drain_generator(
        gen, n,
        min_length=MIN_LEN, max_length=MAX_LEN,
        max_total_points=MAX_TOTAL_POINTS, max_channels=MAX_CHANNELS,
    )


def test_subclasses_datagenerator(Generator):
    from cascade.interface import DataGenerator

    gen = Generator(str(REPO_DIR), seed=0)
    assert isinstance(gen, DataGenerator)
    assert isinstance(gen.name, str) and gen.name


def test_yields_exact_count(Generator):
    out = _drain(Generator, 40, seed=3)
    assert len(out) == 40


def test_every_series_passes_check_series(Generator):
    gen = Generator(str(REPO_DIR), seed=5)
    n = 0
    for i, arr in enumerate(gen.generate(40)):
        check_series(arr, min_length=MIN_LEN, max_length=MAX_LEN, max_channels=MAX_CHANNELS, index=i)
        n += 1
    assert n == 40


def test_canonicalised_shape_and_dtype(Generator):
    out = _drain(Generator, 24, seed=7)
    # drain_generator canonicalises 1-D -> (1, L) float64.
    assert all(a.ndim == 2 and a.shape[0] == 1 for a in out)
    assert all(a.dtype == np.float64 for a in out)
    assert all(np.isfinite(a).all() for a in out)
    assert all(MIN_LEN <= a.shape[-1] <= MAX_LEN for a in out)


def test_deterministic_same_seed(Generator):
    a = _drain(Generator, 32, seed=11)
    b = _drain(Generator, 32, seed=11)
    assert len(a) == len(b)
    for x, y in zip(a, b, strict=True):
        assert np.array_equal(x, y)


def test_different_seed_differs(Generator):
    a = _drain(Generator, 16, seed=0)
    b = _drain(Generator, 16, seed=1)
    # At least one series should differ (overwhelmingly all do).
    assert any(x.shape != y.shape or not np.array_equal(x, y) for x, y in zip(a, b))


def test_length_band_diversity(Generator):
    out = _drain(Generator, 64, seed=2)
    lengths = {a.shape[-1] for a in out}
    # Random-cropping should produce many distinct lengths within the band.
    assert len(lengths) > 10
    assert min(a.shape[-1] for a in out) >= MIN_LEN
    assert max(a.shape[-1] for a in out) <= MAX_LEN


def test_point_cap_enforced(Generator):
    # A tiny cap must trip drain_generator's global point budget.
    gen = Generator(str(REPO_DIR), seed=0)
    with pytest.raises(ValueError):
        drain_generator(
            gen, 64,
            min_length=MIN_LEN, max_length=MAX_LEN,
            max_total_points=100, max_channels=MAX_CHANNELS,
        )
