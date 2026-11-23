"""W2e: every new key is overlay-settable lane D tuning (§14.2), and the
cross-key checks re-run at overlay adoption (rev6.9 (a): "the check runs at
every overlay adoption, so an overlay that breaks it is refused")."""

from __future__ import annotations

import pytest

from .helpers import (
    ARM,
    OVERLAY_VALUE,
    SHIPPED,
    W2E_KEYS,
    field,
    load_ok,
    overrides_refusal,
    raw_with,
    same,
    shipped_raw,
    validate_ok,
    validate_refused,
)


@pytest.mark.parametrize("key", W2E_KEYS)
def test_w2e_every_key_passes_check_overrides(key):
    why = overrides_refusal({"lane_d": {key: OVERLAY_VALUE[key]}})
    assert why is None, f"[lane_d] {key} refused by check_overrides: {why}"


@pytest.mark.parametrize("name", SHIPPED)
@pytest.mark.parametrize("key", W2E_KEYS)
def test_w2e_every_key_overlay_takes_effect(name, key):
    b = OVERLAY_VALUE[key]
    base = load_ok(shipped_raw(name))
    out = validate_ok(base, {"lane_d": {**ARM, key: b}})
    got = field(out.lane_d, key)
    assert same(got, b), f"overlay [lane_d] {key}={b!r} built {got!r}"


@pytest.mark.parametrize("name", SHIPPED)
def test_w2e_overlay_breaking_the_backtest_inequality_is_refused(name):
    base = load_ok(shipped_raw(name))
    assert validate_refused(base, {"lane_d": {**ARM, "backtest_weight_cap": 0.10}}), (
        "an overlay arming backtest_weight_cap 0.10 at floor 0.85 / d_margin 0.01 "
        "(1 - 0.85**0.1 ~ 0.016) was adopted")
    assert validate_refused(base, {"lane_d": {**ARM, "backtest_weight_cap": 0.05,
                                              "backtest_ratio_floor": 0.80}})


@pytest.mark.parametrize("name", SHIPPED)
def test_w2e_overlay_lowering_d_margin_under_an_armed_cap_is_refused(name):
    """A config already carrying cap 0.05 / floor 0.85: an overlay that
    lowers d_margin to 0.005 breaks the inequality and is refused; one that
    also raises the floor to 0.95 is adopted."""
    base = load_ok(raw_with(name, {"backtest_weight_cap": 0.05}))
    assert same(field(base.lane_d, "backtest_weight_cap"), 0.05)
    assert validate_refused(base, {"lane_d": {**ARM, "d_margin": 0.005}})
    out = validate_ok(base, {"lane_d": {**ARM, "d_margin": 0.005,
                                        "backtest_ratio_floor": 0.95}})
    assert same(field(out.lane_d, "backtest_ratio_floor"), 0.95)


@pytest.mark.parametrize("name", SHIPPED)
def test_w2e_overlay_max_windows_below_min_windows_is_refused(name):
    base = load_ok(shipped_raw(name))
    assert validate_refused(base, {"lane_d": {**ARM, "max_windows_per_settlement": 999}})
    assert validate_refused(base, {"lane_d": {**ARM, "min_windows": 6000}})
