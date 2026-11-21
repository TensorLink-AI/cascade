"""WB10: every new [lane_b] key is parsed by the loader, has its plan default,
is carried by both shipped files at that default (nothing armed), survives a
TOML round-trip and is MUTATION-tested (changing the key changes the parsed
value, so an unread key fails)."""

from __future__ import annotations

import pytest

from .helpers import (
    MUTATIONS,
    SHIPPED,
    SINGLE_VALUED,
    WB10_KEYS,
    as_table,
    comparable,
    default_of,
    dump_wb10,
    field,
    load_ok,
    load_text,
    norm,
    raw_with,
    round_skus,
    same_value,
    shipped_raw,
    splice_lane_b,
)


@pytest.mark.parametrize("name", SHIPPED)
@pytest.mark.parametrize("key", WB10_KEYS)
def test_wb10_key_default(name, key):
    """Absent from the table, the key takes its plan default (§12b, §17.4)."""
    lb = load_ok(raw_with(name, drop=(key,)), f" without {key}").lane_b
    want = default_of(name, key)
    assert same_value(key, field(lb, key), want), (
        f"[lane_b] {key} default is {field(lb, key)!r}, plan says {want!r}")


@pytest.mark.parametrize("name", SHIPPED)
def test_wb10_funded_pod_skus_default_is_lane_a_list_and_separate(name):
    """D42: [lane_b] funded_pod_skus starts as today's lane A list but is its
    own key: setting it never changes [round] funded_pod_skus."""
    lb = load_ok(raw_with(name, drop=("funded_pod_skus",))).lane_b
    assert norm(field(lb, "funded_pod_skus")) == round_skus(name)
    cfg = load_ok(raw_with(name, {"funded_pod_skus": ["8xH100"]}))
    assert norm(field(cfg.lane_b, "funded_pod_skus")) == ("8xH100",)
    assert tuple(cfg.round.funded_pod_skus) == round_skus(name), (
        "[lane_b] funded_pod_skus leaked into [round] funded_pod_skus")


@pytest.mark.parametrize("name", SHIPPED)
def test_wb10_contamination_min_hits_default_is_one_exact_three_normalised(name):
    """§12b.1: a refusal at >= 1 exact or >= 3 normalised sub-window hits."""
    lb = load_ok(raw_with(name, drop=("contamination_min_hits",))).lane_b
    assert as_table(field(lb, "contamination_min_hits")) == {"exact": 1, "normalised": 3}


@pytest.mark.parametrize("name", SHIPPED)
def test_wb10_shipped_table_carries_every_key_at_its_default(name):
    """Both shipped files carry every WB10 key at its default; nothing armed:
    suite_targets false, no hash set pinned, lane B from_block / share as on
    main."""
    table = shipped_raw(name).get("lane_b") or {}
    missing = [k for k in WB10_KEYS if k not in table]
    assert not missing, f"{name} [lane_b] lacks {missing}"
    assert table["suite_targets"] is False, f"{name} arms suite targets"
    assert table["contamination_hashset"] == "", f"{name} pins a hash set"
    lb = load_ok(shipped_raw(name)).lane_b
    for k in WB10_KEYS:
        assert same_value(k, field(lb, k), default_of(name, k)), (
            f"{name} [lane_b] {k} = {field(lb, k)!r}, default {default_of(name, k)!r}")


@pytest.mark.parametrize("name", SHIPPED)
@pytest.mark.parametrize("key", sorted(MUTATIONS))
def test_wb10_key_mutation_changes_the_parsed_value(name, key):
    """Two different values give two different parsed values (an unread key
    would give the same config twice)."""
    a, b, co = MUTATIONS[key]
    lb_a = load_ok(raw_with(name, {**co, key: a}), f" with {key}={a!r}").lane_b
    lb_b = load_ok(raw_with(name, {**co, key: b}), f" with {key}={b!r}").lane_b
    va, vb = field(lb_a, key), field(lb_b, key)
    assert same_value(key, va, a), f"[lane_b] {key}={a!r} parsed as {va!r}"
    assert same_value(key, vb, b), f"[lane_b] {key}={b!r} parsed as {vb!r}"
    assert lb_a != lb_b, f"[lane_b] {key}: {a!r} and {b!r} parse to the same config"


def test_wb10_every_key_is_mutation_or_single_valued():
    assert set(MUTATIONS) | set(SINGLE_VALUED) == set(WB10_KEYS)


# Every WB10 key at a non-default value: one table that must load, dump and
# load back equal.
ROUND_TRIP = {
    **{k: b for k, (_a, b, _co) in MUTATIONS.items()},
    "suite_targets": True,
    "router_rows": "frozen",
    "sentinel_target_rule": "defer",
}


@pytest.mark.parametrize("name", SHIPPED)
def test_wb10_round_trip_load_dump_load_equal(name, tmp_path):
    cfg1 = load_ok(raw_with(name, ROUND_TRIP), " (round-trip table)")
    text = splice_lane_b(name, dump_wb10(cfg1.lane_b))
    cfg2 = load_text(text, tmp_path, name)
    assert cfg2.lane_b == cfg1.lane_b, "[lane_b] changed across load -> dump -> load"
    for k in WB10_KEYS:
        assert same_value(k, field(cfg2.lane_b, k), ROUND_TRIP[k]), k


@pytest.mark.parametrize("name", SHIPPED)
def test_wb10_round_trip_of_the_shipped_defaults(name, tmp_path):
    cfg1 = load_ok(shipped_raw(name))
    text = splice_lane_b(name, dump_wb10(cfg1.lane_b))
    cfg2 = load_text(text, tmp_path, name)
    assert cfg2.lane_b == cfg1.lane_b
    assert comparable(cfg2) == comparable(cfg1)
