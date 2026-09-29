from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]


def _generator():
    spec = importlib.util.spec_from_file_location("cascade_v2_generator", REPO / "generator.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.Generator(str(REPO), seed=123)


def _module():
    spec = importlib.util.spec_from_file_location("cascade_v2_module", REPO / "generator.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_exact_count_full_context_and_finite():
    rows = list(_generator().generate(1031))
    assert len(rows) == 1031
    assert all(row.shape == (4096,) for row in rows)
    assert all(row.dtype == np.float64 for row in rows)
    assert all(np.isfinite(row).all() and row.std() > 1e-9 for row in rows)


def test_deterministic_across_instances():
    left = list(_generator().generate(1030))
    right = list(_generator().generate(1030))
    assert all(np.array_equal(a, b) for a, b in zip(left, right, strict=True))


def test_requested_size_does_not_change_prefix():
    short = list(_generator().generate(17))
    long = list(_generator().generate(1030))
    assert all(np.array_equal(a, b) for a, b in zip(short, long[:17], strict=True))


def test_nonpositive_request_is_empty():
    assert list(_generator().generate(0)) == []


def test_ou_stochastic_vol_is_deterministic_finite_and_active():
    module = _module()
    left = module._ou_stochastic_vol(np.random.default_rng(77), 16, 512)
    right = module._ou_stochastic_vol(np.random.default_rng(77), 16, 512)
    assert np.array_equal(left, right)
    assert np.isfinite(left).all()
    assert (left.std(axis=1) > 1e-9).all()
    assert "ou_stochastic_vol" in module._FAMILIES
    assert module._DEFAULT_WEIGHTS["ou_stochastic_vol"] > 0


def test_physical_sensors_are_deterministic_finite_and_active():
    module = _module()
    left = module._physical_sensors(np.random.default_rng(91), 32, 512)
    right = module._physical_sensors(np.random.default_rng(91), 32, 512)
    assert np.array_equal(left, right)
    assert np.isfinite(left).all()
    assert (left.std(axis=1) > 1e-9).all()
    assert "physical_sensors" in module._FAMILIES
    assert module._DEFAULT_WEIGHTS["physical_sensors"] > 0


def test_seasonal_counts_are_deterministic_integer_and_active():
    module = _module()
    left = module._seasonal_counts(np.random.default_rng(109), 32, 512)
    right = module._seasonal_counts(np.random.default_rng(109), 32, 512)
    assert np.array_equal(left, right)
    assert np.isfinite(left).all()
    assert (left >= 0).all()
    assert np.array_equal(left, np.round(left))
    assert (left.std(axis=1) > 1e-9).all()
    assert "seasonal_counts" in module._FAMILIES
    assert module._DEFAULT_WEIGHTS["seasonal_counts"] > 0


def test_event_family_has_real_flat_runs_and_recovery():
    module = _module()
    left = module._pulse_outlier(np.random.default_rng(131), 32, 512)
    right = module._pulse_outlier(np.random.default_rng(131), 32, 512)
    assert np.array_equal(left, right)
    assert np.isfinite(left).all()
    assert (left.std(axis=1) > 1e-9).all()
    assert np.any(np.diff(left, axis=1) == 0.0)


def test_measurement_artifacts_are_deterministic_finite_and_shape_safe():
    module = _module()
    raw = module._trend_seasonal_ar(np.random.default_rng(149), 64, 512)
    left = module._measurement_artifacts(
        np.random.default_rng(151), raw, preserve_nonnegative=False
    )
    right = module._measurement_artifacts(
        np.random.default_rng(151), raw, preserve_nonnegative=False
    )
    assert np.array_equal(left, right)
    assert left.shape == raw.shape
    assert np.isfinite(left).all()
    assert (left.std(axis=1) > 1e-9).all()
