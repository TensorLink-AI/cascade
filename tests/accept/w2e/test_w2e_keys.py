"""W2e: every new [lane_d] key is parsed, has its plan default, is carried by
both shipped files, survives a TOML round-trip and is mutation-tested."""

from __future__ import annotations

import pytest

from .helpers import (
    MUTATIONS,
    RETIRED,
    SHIPPED,
    W2E_DEFAULTS,
    W2E_KEYS,
    field,
    load_ok,
    load_text,
    raw_with,
    same,
    shipped_raw,
    splice_keys,
)


@pytest.mark.parametrize("name", SHIPPED)
@pytest.mark.parametrize("key", W2E_KEYS)
def test_w2e_key_default(name, key):
    """Absent from the table, the key takes the plan default."""
    ld = load_ok(raw_with(name, drop=(key,)), f" without {key}").lane_d
    got = field(ld, key)
    assert same(got, W2E_DEFAULTS[key]), (
        f"[lane_d] {key} default is {got!r}, plan says {W2E_DEFAULTS[key]!r}")


@pytest.mark.parametrize("name", SHIPPED)
def test_w2e_max_windows_default_is_5000_not_2000(name):
    """rev6.11 (a): the owner's 5000 replaces rev6.10's recommended 2000."""
    ld = load_ok(raw_with(name, drop=("max_windows_per_settlement",))).lane_d
    v = field(ld, "max_windows_per_settlement")
    assert type(v) is int and v == 5000, f"max_windows_per_settlement default {v!r}"


@pytest.mark.parametrize("name", SHIPPED)
def test_w2e_backtest_and_lowfreq_keys_are_retired(name):
    """rev6.18 D24 (owner 2026-11-24: backtest dropped) and A7 (owner
    2026-11-23: A10 removed). Correction of the rev6.15 (b) test that pinned
    backtest_weight_cap = 0.05: the keys are deleted. The shipped files carry
    none of them; a deployed table that still does loads and ignores them,
    even at values the deleted rev6.9 inequality refused; the loader lists
    them in RETIRED_KEYS["lane_d"]."""
    from cascade.shared.config import RETIRED_KEYS

    table = shipped_raw(name).get("lane_d") or {}
    assert not set(RETIRED) & set(table), f"{name} [lane_d] still carries {set(RETIRED) & set(table)}"
    ld = load_ok(raw_with(name, {"backtest_weight_cap": 0.5, "backtest_ratio_floor": 0.1,
                                 "lowfreq_freqs": ["D"]}), " with the retired keys").lane_d
    for k in RETIRED:
        assert not hasattr(ld, k), f"LaneDConfig still parses the retired key {k!r}"
    assert set(RETIRED) <= set(RETIRED_KEYS.get("lane_d") or ()), (
        "RETIRED_KEYS['lane_d'] does not list the rev6.18 F4 keys")


@pytest.mark.parametrize("name", SHIPPED)
def test_w2e_public_weight_is_not_tied_to_d_margin(name):
    """rev6.18 A3: public_weight is NOT subject to the deleted backtest loader
    rule: any w in [0, 1) loads at any d_margin."""
    ld = load_ok(raw_with(name, {"public_weight": 0.95, "d_margin": 0.0001,
                                 "window_ratio_floor": 0.01})).lane_d
    assert same(field(ld, "public_weight"), 0.95)
    assert same(field(ld, "window_ratio_floor"), 0.01)


@pytest.mark.parametrize("name", SHIPPED)
def test_w2e_forward_margin_hours_stays_6(name):
    """rev6.10 (a): "delay eval by 6 hours" is the existing default."""
    ld = load_ok(raw_with(name, drop=("forward_margin_hours",))).lane_d
    assert float(ld.forward_margin_hours) == 6.0


@pytest.mark.parametrize("name", SHIPPED)
def test_w2e_shipped_table_carries_every_key_at_its_default(name):
    table = shipped_raw(name).get("lane_d") or {}
    missing = [k for k in W2E_KEYS if k not in table]
    assert not missing, f"{name} [lane_d] lacks {missing}"
    for k in W2E_KEYS:
        assert same(table[k], W2E_DEFAULTS[k]), (
            f"{name} [lane_d] {k} = {table[k]!r}, default {W2E_DEFAULTS[k]!r}")
    assert table["from_block"] == 0, f"{name} arms lane D"
    # rev6.12 (b): the rev6.11 share cap is replaced, never shipped.
    assert "embargo_window_share_cap" not in table
    ld = load_ok(shipped_raw(name)).lane_d
    for k in W2E_KEYS:
        assert same(field(ld, k), W2E_DEFAULTS[k]), k


@pytest.mark.parametrize("name", SHIPPED)
@pytest.mark.parametrize("key", sorted(MUTATIONS))
def test_w2e_key_mutation_changes_the_parsed_value(name, key):
    a, b, co = MUTATIONS[key]
    ld_a = load_ok(raw_with(name, {**co, key: a}), f" with {key}={a!r}").lane_d
    ld_b = load_ok(raw_with(name, {**co, key: b}), f" with {key}={b!r}").lane_d
    va, vb = field(ld_a, key), field(ld_b, key)
    assert same(va, a), f"[lane_d] {key}={a!r} parsed as {va!r}"
    assert same(vb, b), f"[lane_d] {key}={b!r} parsed as {vb!r}"
    assert ld_a != ld_b, f"[lane_d] {key}: {a!r} and {b!r} parse to the same config"


ROUND_TRIP = {k: b for k, (_a, b, _co) in MUTATIONS.items()}


@pytest.mark.parametrize("name", SHIPPED)
def test_w2e_round_trip_load_dump_load_equal(name, tmp_path):
    """The keys written into the file load back to the same values, and the
    same config as the in-memory parse of that table."""
    cfg1 = load_ok(raw_with(name, ROUND_TRIP), " (round-trip table)")
    text = splice_keys(name, {k: field(cfg1.lane_d, k) for k in W2E_KEYS})
    cfg2 = load_text(text, tmp_path, name)
    assert cfg2.lane_d == cfg1.lane_d, "[lane_d] changed across load -> dump -> load"
    for k in W2E_KEYS:
        assert same(field(cfg2.lane_d, k), ROUND_TRIP[k]), k
