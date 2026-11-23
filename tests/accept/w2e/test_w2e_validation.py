"""W2e: type and range validation of the new keys and the cross-key checks
(max_windows_per_settlement >= min_windows; the rev6.9 backtest inequality)."""

from __future__ import annotations

import pytest

from .helpers import (
    ACCEPTED,
    MUTATIONS,
    REFUSED,
    SHIPPED,
    field,
    inequality_holds,
    load_ok,
    raw_with,
    refused,
    same,
)

CASES = [(k, v) for k, vals in REFUSED.items() for v in vals]
OK_CASES = [(k, v) for k, vals in ACCEPTED.items() for v in vals]


@pytest.mark.parametrize("name", SHIPPED)
@pytest.mark.parametrize(("key", "value"), CASES, ids=[f"{k}={v!r}" for k, v in CASES])
def test_w2e_invalid_value_is_refused_at_load(name, key, value):
    raw = raw_with(name, {**MUTATIONS[key][2], key: value})
    assert refused(raw), f"[lane_d] {key} = {value!r} loaded; it must be refused"


@pytest.mark.parametrize("name", SHIPPED)
@pytest.mark.parametrize(("key", "value"), OK_CASES, ids=[f"{k}={v!r}" for k, v in OK_CASES])
def test_w2e_valid_value_loads(name, key, value):
    ld = load_ok(raw_with(name, {**MUTATIONS[key][2], key: value}),
                 f" with {key}={value!r}").lane_d
    assert same(field(ld, key), value)


@pytest.mark.parametrize("name", SHIPPED)
def test_w2e_max_windows_below_min_windows_is_refused(name):
    """The cap may equal min_windows, never fall below it, whichever moved."""
    ok = load_ok(raw_with(name, {"min_windows": 1500, "max_windows_per_settlement": 1500}))
    assert field(ok.lane_d, "max_windows_per_settlement") == 1500
    assert refused(raw_with(name, {"min_windows": 1500, "max_windows_per_settlement": 1499}))
    # min_windows raised above the default 5000 cap alone is refused too.
    assert refused(raw_with(name, {"min_windows": 6000, "max_windows_per_settlement": None}))
    load_ok(raw_with(name, {"min_windows": 6000, "max_windows_per_settlement": 6000}))


# (backtest_weight_cap, backtest_ratio_floor, d_margin) -> loads?
INEQUALITY = [
    (0.05, 0.85, 0.01),     # the rev6.9 defaults: 1 - 0.85**0.05 ~ 0.0081 < 0.01
    (0.05, 0.80, 0.01),     # ~ 0.0111: refused
    (0.10, 0.85, 0.01),     # ~ 0.0161: refused
    (0.05, 0.85, 0.005),    # d_margin lowered by B1, floor kept: refused
    (0.05, 0.70, 0.02),     # ~ 0.0177 < 0.02: loads
    (0.05, 0.95, 0.005),    # ~ 0.0026 < 0.005: loads
    (0.0, 0.50, 0.01),      # cap 0: off, no check
    (0.0, 0.85, 0.002),     # cap 0: off, no check
]


@pytest.mark.parametrize("name", SHIPPED)
@pytest.mark.parametrize(("cap", "floor", "d_margin"), INEQUALITY,
                         ids=[f"cap{c}-floor{f}-d{d}" for c, f, d in INEQUALITY])
def test_w2e_backtest_inequality_at_load(name, cap, floor, d_margin):
    """rev6.9 (a): cap > 0 loads only if 1 - floor ** cap < d_margin."""
    raw = raw_with(name, {"backtest_weight_cap": cap, "backtest_ratio_floor": floor,
                          "d_margin": d_margin})
    if inequality_holds(cap, floor, d_margin):
        ld = load_ok(raw, f" (cap={cap}, floor={floor}, d_margin={d_margin})").lane_d
        assert same(field(ld, "backtest_weight_cap"), cap)
        assert same(field(ld, "backtest_ratio_floor"), floor)
    else:
        assert refused(raw), (f"cap={cap}, floor={floor}, d_margin={d_margin} loaded: "
                              f"1 - floor**cap = {1 - floor ** cap:.5f} >= d_margin")
