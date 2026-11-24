"""W2e: every new key is overlay-settable lane D tuning (§14.2), and the
cross-key checks re-run at overlay adoption (rev6.18 D24: the rev6.9 (a)
backtest inequality is deleted; public_weight / window_ratio_floor carry
their own ranges only)."""

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
def test_w2e_overlay_public_weight_and_clamp_have_no_cross_key_rule(name):
    """rev6.18 A3 / D24 (correction of the two rev6.9 (a) overlay-inequality
    tests: the backtest keys and their rule are deleted, owner 2026-11-24).
    public_weight and window_ratio_floor are overlay-settable and an
    overlay lowering d_margin is adopted at any w; their own ranges are
    re-checked at adoption."""
    base = load_ok(shipped_raw(name))
    out = validate_ok(base, {"lane_d": {**ARM, "public_weight": 0.9, "d_margin": 0.001}})
    assert same(field(out.lane_d, "public_weight"), 0.9)
    assert validate_refused(base, {"lane_d": {**ARM, "public_weight": 1.0}})
    assert validate_refused(base, {"lane_d": {**ARM, "window_ratio_floor": 1.0}})
    # A retired key in an overlay changes nothing, so it is refused as having
    # no effect (the overlay rule for an unknown key).
    assert validate_refused(base, {"lane_d": {**ARM, "backtest_weight_cap": 0.5}})


@pytest.mark.parametrize("name", SHIPPED)
def test_w2e_overlay_max_windows_below_min_windows_is_refused(name):
    base = load_ok(shipped_raw(name))
    assert validate_refused(base, {"lane_d": {**ARM, "max_windows_per_settlement": 999}})
    assert validate_refused(base, {"lane_d": {**ARM, "min_windows": 6000}})
