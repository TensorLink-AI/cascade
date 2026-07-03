"""gift-eval per-config ratio rows for the consensus gate — Seasonal-Naive
normalization and the zero-inflated MASE→MAE substitution, without needing
gluonts/torch (the ratio math is pure)."""

from __future__ import annotations

from cascade_benchmark.suites.gifteval import _ratio_rows, _safe_ratio


def test_safe_ratio_guards_zero_and_none():
    assert _safe_ratio(2.0, 4.0) == 0.5
    assert _safe_ratio(1.0, 0) is None
    assert _safe_ratio(None, 4.0) is None
    assert _safe_ratio(1.0, None) is None


def test_ratio_rows_normalize_against_baseline():
    rows = [{"full": "m4/D/short", "MASE": 1.0, "MAE": 2.0, "CRPS": 0.5}]
    baseline = {"m4/D/short": {"MASE": 2.0, "MAE": 4.0, "CRPS": 1.0}}
    (r,) = _ratio_rows(rows, baseline)
    assert r["full"] == "m4/D/short"
    assert r["crps_ratio"] == 0.5   # 0.5 / 1.0
    assert r["mase_ratio"] == 0.5   # 1.0 / 2.0 (non-zero pool → MASE ratio)


def test_zero_mase_config_substitutes_mae_ratio():
    # Seasonal-Naive MASE == 0 ⇒ the config is in the zero pool; the MASE ratio
    # column carries the MAE ratio instead (matches official_aggregate).
    rows = [{"full": "flat/D/short", "MASE": 3.0, "MAE": 1.0, "CRPS": 0.4}]
    baseline = {"flat/D/short": {"MASE": 0.0, "MAE": 4.0, "CRPS": 0.8}}
    (r,) = _ratio_rows(rows, baseline)
    assert r["crps_ratio"] == 0.5   # 0.4 / 0.8
    assert r["mase_ratio"] == 0.25  # MAE ratio 1.0 / 4.0


def test_rows_without_baseline_key_are_dropped():
    rows = [{"full": "unknown/D/short", "MASE": 1.0, "MAE": 2.0, "CRPS": 0.5}]
    assert _ratio_rows(rows, {}) == []
