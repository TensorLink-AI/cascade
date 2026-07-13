"""TIME parity: Seasonal-Naive quantile baseline + ratio→shifted-geomean.

These are the pure-numpy pieces of the TIME suite (no timebench/gluonts needed),
so they run in the sidecar env without the optional TIME data.
"""

from __future__ import annotations

import math

import numpy as np
from cascade_benchmark.aggregate import (
    normalize_time,
    seasonal_naive_quantiles,
    shifted_gmean,
)


def test_seasonal_naive_repeats_last_season():
    # target [0..5], season 3, horizon 4 → phases 3,4,5 then wrap to 3.
    sn = seasonal_naive_quantiles(np.arange(6.0), horizon=4, season=3, n_quantiles=9)
    assert sn.shape == (9, 1, 4)                       # (num_q, V, H)
    np.testing.assert_array_equal(sn[0, 0], [3.0, 4.0, 5.0, 3.0])
    # A point forecast: every quantile level coincides.
    for q in range(9):
        np.testing.assert_array_equal(sn[q, 0], sn[0, 0])


def test_seasonal_naive_multivariate_and_short_context():
    # 2 variates; season longer than context clamps into range (no negative index).
    t = np.array([[10.0, 11.0], [20.0, 21.0]])
    sn = seasonal_naive_quantiles(t, horizon=3, season=5, n_quantiles=9)
    assert sn.shape == (9, 2, 3)
    assert np.isfinite(sn).all()


def test_normalize_time_is_ratio_then_shifted_gmean():
    # model is 2x better than Seasonal-Naive on every task/metric ⇒ ratio 0.5.
    model = [{"MASE": 0.5, "CRPS": 0.4}, {"MASE": 1.0, "CRPS": 0.8}]
    base = [{"MASE": 1.0, "CRPS": 0.8}, {"MASE": 2.0, "CRPS": 1.6}]
    out = normalize_time(model, base)
    assert math.isclose(out["mase"], shifted_gmean(np.array([0.5, 0.5])), rel_tol=1e-9)
    assert math.isclose(out["crps"], shifted_gmean(np.array([0.5, 0.5])), rel_tol=1e-9)


def test_normalize_time_case_insensitive_and_wql_alias():
    # Upper-case model keys, a WQL-named CRPS on the baseline: still matched.
    model = [{"mase": 2.0, "wql": 1.0}]
    base = [{"MASE": 1.0, "WQL": 2.0}]
    out = normalize_time(model, base)
    assert math.isclose(out["mase"], 2.0, rel_tol=1e-6)
    assert math.isclose(out["crps"], 0.5, rel_tol=1e-6)


def test_normalize_time_model_equals_naive_is_one():
    model = [{"MASE": 1.0, "CRPS": 0.5}, {"MASE": 2.0, "CRPS": 1.0}]
    out = normalize_time(model, model)
    assert math.isclose(out["mase"], 1.0, rel_tol=1e-6)
    assert math.isclose(out["crps"], 1.0, rel_tol=1e-6)


# ── Seasonal-Naive baseline cache (checkpoint-independent → compute once) ─────


def test_baseline_cache_round_trips(tmp_path, monkeypatch):
    from cascade_benchmark import cache

    monkeypatch.setenv("CASCADE_BENCH_TIME_BASELINE_CACHE", str(tmp_path))
    monkeypatch.delenv("CASCADE_BENCH_NO_CACHE", raising=False)
    d = cache.baseline_cache_dir()
    assert d == tmp_path
    assert cache.load_baseline(d, "WUI_Global", "short", 24, 9) is None   # miss
    cache.store_baseline(d, "WUI_Global", "short", 24, 9, {"MASE": 1.0, "CRPS": 0.5})
    assert cache.load_baseline(d, "WUI_Global", "short", 24, 9) == {"MASE": 1.0, "CRPS": 0.5}
    # A different task (or grid) is a distinct key.
    assert cache.load_baseline(d, "WUI_Global", "long", 24, 9) is None
    assert cache.load_baseline(d, "WUI_Global", "short", 48, 9) is None


def test_baseline_cache_disabled(tmp_path, monkeypatch):
    from cascade_benchmark import cache

    monkeypatch.setenv("CASCADE_BENCH_TIME_BASELINE_CACHE", str(tmp_path))
    monkeypatch.setenv("CASCADE_BENCH_NO_CACHE", "1")
    assert cache.baseline_cache_dir() is None
    # store/load are no-ops (and never raise) when disabled.
    cache.store_baseline(None, "x", "short", 1, 9, {"MASE": 1.0})
    assert cache.load_baseline(None, "x", "short", 1, 9) is None


def test_baseline_cache_key_is_filesystem_safe(tmp_path):
    from cascade_benchmark import cache

    # Real TIME config keys contain slashes; the cache file name must stay flat.
    p = cache.cache_key(tmp_path, "WUI_Global/region", "short", 24, 9)
    assert p.parent == tmp_path
    assert "/" not in p.name[:-5]  # basename minus ".json"
