"""Aggregation math — the official Seasonal-Naive normalized shifted-geomean.

Pure numpy, no benchmark deps. Run from the sidecar env: ``pytest benchmarks``.
"""

from __future__ import annotations

import numpy as np
from cascade_benchmark.aggregate import official_aggregate, shifted_gmean


def test_shifted_gmean_matches_geometric_mean():
    assert shifted_gmean(np.array([1.0, 1.0, 1.0])) == 1.0
    assert abs(shifted_gmean(np.array([1.0, 4.0])) - 2.0) < 1e-3


def test_model_equal_to_baseline_normalizes_to_one():
    baseline = {f"d{i}/H/short": {"MASE": 1.0 + i, "MAE": 2.0, "CRPS": 0.5 + i} for i in range(5)}
    rows = [{"full": k, **v} for k, v in baseline.items()]
    agg = official_aggregate(rows, baseline)
    assert agg["n_scored"] == 5 and agg["n_nonzero"] == 5
    assert abs(agg["crps"] - 1.0) < 1e-6
    assert abs(agg["mase"] - 1.0) < 1e-6


def test_twice_as_bad_aggregates_to_two():
    baseline = {f"d{i}/H/short": {"MASE": 1.0 + i, "MAE": 2.0, "CRPS": 0.5 + i} for i in range(5)}
    rows = [{"full": k, "MASE": v["MASE"] * 2, "MAE": v["MAE"], "CRPS": v["CRPS"] * 2}
            for k, v in baseline.items()]
    agg = official_aggregate(rows, baseline)
    assert abs(agg["mase"] - 2.0) < 1e-6
    assert abs(agg["crps"] - 2.0) < 1e-6


def test_zero_inflated_and_low_variance_split_out():
    baseline = {
        "a/H/short": {"MASE": 1.0, "MAE": 2.0, "CRPS": 0.5},   # main pool
        "b/H/short": {"MASE": 0.0, "MAE": 2.0, "CRPS": 0.5},   # zero MASE → zero pool
        "lv/H/short": {"MASE": 1.0, "MAE": 2.0, "CRPS": 0.5},  # low-variance → zero pool
    }
    rows = [{"full": k, **v} for k, v in baseline.items()]
    agg = official_aggregate(rows, baseline, low_variance=frozenset({"lv"}))
    assert agg["n_nonzero"] == 1
    assert agg["n_zero"] == 2
    assert "mae_zero" in agg


def test_unmatched_rows_are_skipped_not_fatal():
    baseline = {"a/H/short": {"MASE": 1.0, "MAE": 2.0, "CRPS": 0.5}}
    rows = [{"full": "a/H/short", "MASE": 1.0, "MAE": 2.0, "CRPS": 0.5},
            {"full": "missing/H/short", "MASE": 9.0, "MAE": 9.0, "CRPS": 9.0}]
    agg = official_aggregate(rows, baseline)
    assert agg["n_scored"] == 1 and agg["n_skipped"] == 1
