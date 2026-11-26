"""W-S1: the four ``[v2]`` sentinel keys (§16 "Consensus and rollout"): loader
parse, plan default, validation, both shipped files at the default (nothing
armed), mutation and TOML round-trip."""

from __future__ import annotations

import pytest

from .helpers import (
    ACCEPTED,
    MUTATIONS,
    REFUSED,
    SHIPPED,
    WS1_DEFAULTS,
    WS1_KEYS,
    comparable,
    field,
    load_ok,
    load_text,
    norm,
    raw_with,
    refused,
    shipped_raw,
    splice_v2,
    toml_value,
)


@pytest.mark.parametrize("name", SHIPPED)
@pytest.mark.parametrize("key", WS1_KEYS)
def test_ws1_key_default(name, key):
    """Absent from [v2], each key takes its plan default: from_block 0 (off),
    the four suites, ratchet on (D34), tau 0.01."""
    v2 = load_ok(raw_with(name, drop=(key,)), f" without {key}").v2
    assert norm(field(v2, key)) == norm(WS1_DEFAULTS[key]), (
        f"[v2] {key} default is {field(v2, key)!r}, plan says {WS1_DEFAULTS[key]!r}")


@pytest.mark.parametrize("name", SHIPPED)
def test_ws1_shipped_v2_carries_every_key_at_its_default(name):
    """Both shipped files carry the keys at their defaults; sentinel_from_block
    stays 0. Arming is one signed overlay (D45), never a chain.toml edit."""
    table = shipped_raw(name).get("v2") or {}
    missing = [k for k in WS1_KEYS if k not in table]
    assert not missing, f"{name} [v2] lacks {missing}"
    assert table["sentinel_from_block"] == 0, f"{name} arms the sentinel"
    v2 = load_ok(shipped_raw(name)).v2
    for k in WS1_KEYS:
        assert norm(field(v2, k)) == norm(WS1_DEFAULTS[k]), (k, field(v2, k))


@pytest.mark.parametrize("name", SHIPPED)
@pytest.mark.parametrize("key", WS1_KEYS)
def test_ws1_key_mutation_changes_the_parsed_value(name, key):
    a, b = MUTATIONS[key]
    va = field(load_ok(raw_with(name, {key: a}), f" with {key}={a!r}").v2, key)
    vb = field(load_ok(raw_with(name, {key: b}), f" with {key}={b!r}").v2, key)
    assert norm(va) == norm(a), f"[v2] {key}={a!r} parsed as {va!r}"
    assert norm(vb) == norm(b), f"[v2] {key}={b!r} parsed as {vb!r}"
    assert norm(va) != norm(vb)


@pytest.mark.parametrize("name", SHIPPED)
@pytest.mark.parametrize("key", WS1_KEYS)
def test_ws1_refused_values(name, key):
    """Types and ranges: from_block an int >= 0 on an era boundary; suites a
    non-empty list of unique bench suite names (gift, boom, time, fev); ratchet
    a bool; tau a number >= 0."""
    accepted = [v for v in REFUSED[key] if not refused(raw_with(name, {key: v}))]
    assert not accepted, f"[v2] {key}: the loader accepted {accepted!r}"


@pytest.mark.parametrize("name", SHIPPED)
@pytest.mark.parametrize("key", WS1_KEYS)
def test_ws1_accepted_values_are_carried_as_written(name, key):
    for v in ACCEPTED[key]:
        got = field(load_ok(raw_with(name, {key: v}), f" with {key}={v!r}").v2, key)
        assert norm(got) == norm(v), f"[v2] {key}={v!r} parsed as {got!r}"


ROUND_TRIP = {"sentinel_from_block": MUTATIONS["sentinel_from_block"][1],
              "sentinel_suites": MUTATIONS["sentinel_suites"][1],
              "sentinel_ratchet": False, "sentinel_tau": 0.05}


@pytest.mark.parametrize("name", SHIPPED)
@pytest.mark.parametrize("table", ["non-default", "shipped"])
def test_ws1_round_trip_load_dump_load_equal(name, table, tmp_path):
    raw = raw_with(name, ROUND_TRIP) if table == "non-default" else shipped_raw(name)
    cfg1 = load_ok(raw, f" ({table} table)")
    lines = {k: toml_value(list(v) if isinstance(v, tuple) else v)
             for k in WS1_KEYS for v in [field(cfg1.v2, k)]}
    cfg2 = load_text(splice_v2(name, lines), tmp_path, name)
    assert cfg2.v2 == cfg1.v2, "[v2] changed across load -> dump -> load"
    if table == "shipped":
        assert comparable(cfg2) == comparable(cfg1)
    else:
        for k, v in ROUND_TRIP.items():
            assert norm(field(cfg2.v2, k)) == norm(v), k
