"""W2: every [lane_d] key is parsed by the loader, has its plan default, is
carried by both shipped files, survives a TOML round-trip and is MUTATION-
tested (changing the key changes the parsed value, so an unread key fails).
Legacy / DEC-CA-0062 placeholder keys keep parsing."""

from __future__ import annotations

import pytest

from .helpers import (
    ARM,
    ARM_BLOCK_2,
    LEGACY_KEYS,
    LEGACY_VALUES,
    MUTATIONS,
    SHIPPED,
    UNTIL_BLOCK_2,
    W2_DEFAULTS,
    W2_KEYS,
    co_for,
    comparable,
    dump_lane_d,
    field,
    load_ok,
    load_text,
    norm,
    raw_with,
    shipped_raw,
    splice_lane_d,
    weights,
)

PINNED_DEFAULTS = [k for k, v in W2_DEFAULTS.items() if v is not None]


@pytest.mark.parametrize("name", SHIPPED)
@pytest.mark.parametrize("key", PINNED_DEFAULTS)
def test_w2_key_default(name, key):
    """Absent from the table, the key takes the plan default (§6 / §17.4)."""
    ld = load_ok(raw_with(name, drop=(key,)), f" without {key}").lane_d
    assert norm(field(ld, key)) == norm(W2_DEFAULTS[key]), (
        f"[lane_d] {key} default is {field(ld, key)!r}, plan says {W2_DEFAULTS[key]!r}")


@pytest.mark.parametrize("name", SHIPPED)
def test_w2_gate_weights_default_is_equal_over_gate_suites(name):
    """D2b: one gate suite per crowning, drawn with weights that start equal."""
    ld = load_ok(raw_with(name, drop=("gate_weights", "gate_suites"))).lane_d
    w = weights(field(ld, "gate_weights"))
    assert set(w) == set(norm(field(ld, "gate_suites"))), w
    assert len(set(w.values())) == 1, f"default gate_weights not equal: {w}"
    # follows gate_suites when only the suites are set
    ld2 = load_ok(raw_with(name, {"gate_suites": ["gift", "fev"], "gate_weights": None})).lane_d
    assert weights(field(ld2, "gate_weights")) == {"gift": 0.5, "fev": 0.5}


@pytest.mark.parametrize("name", SHIPPED)
def test_w2_lowfreq_freqs_default_is_daily_and_slower(name):
    """A10: the low-frequency slice is "daily and slower"."""
    ld = load_ok(raw_with(name, drop=("lowfreq_freqs",))).lane_d
    v = norm(field(ld, "lowfreq_freqs"))
    assert isinstance(v, tuple) and v, f"lowfreq_freqs default {v!r}"
    assert all(isinstance(f, str) and f for f in v), v
    assert "D" in v, f"lowfreq_freqs default {v!r} lacks daily"
    assert not {"H", "min", "T", "S"} & set(v), f"lowfreq_freqs default {v!r} has sub-daily"


@pytest.mark.parametrize("name", SHIPPED)
def test_w2_ready_ttl_blocks_default_is_a_positive_block_count(name):
    ld = load_ok(raw_with(name, drop=("ready_ttl_blocks",))).lane_d
    v = field(ld, "ready_ttl_blocks")
    assert type(v) is int and v > 0, f"ready_ttl_blocks default {v!r}"


@pytest.mark.parametrize("name", SHIPPED)
def test_w2_shipped_table_carries_every_key_at_its_default(name):
    """Both shipped files carry every W2 key at its default; nothing armed."""
    table = shipped_raw(name).get("lane_d") or {}
    missing = [k for k in W2_KEYS if k not in table]
    assert not missing, f"{name} [lane_d] lacks {missing}"
    assert table["from_block"] == 0, f"{name} arms lane D (from_block={table['from_block']})"
    assert table["until_block"] == 0 and table["share"] == 0.0
    ld = load_ok(shipped_raw(name)).lane_d
    for k in PINNED_DEFAULTS:
        assert norm(field(ld, k)) == norm(W2_DEFAULTS[k]), (
            f"{name} [lane_d] {k} = {field(ld, k)!r}, default {W2_DEFAULTS[k]!r}")


@pytest.mark.parametrize("name", SHIPPED)
@pytest.mark.parametrize("key", sorted(MUTATIONS))
def test_w2_key_mutation_changes_the_parsed_value(name, key):
    """Two different values give two different parsed values (an unread key
    would give the same config twice)."""
    a, b, _co = MUTATIONS[key]
    ld_a = load_ok(raw_with(name, {**co_for(key, a), key: a}), f" with {key}={a!r}").lane_d
    ld_b = load_ok(raw_with(name, {**co_for(key, b), key: b}), f" with {key}={b!r}").lane_d
    va, vb = field(ld_a, key), field(ld_b, key)
    if key == "gate_weights":
        assert weights(va) == weights(a) and weights(vb) == weights(b), (va, vb)
    else:
        assert norm(va) == norm(a), f"[lane_d] {key}={a!r} parsed as {va!r}"
        assert norm(vb) == norm(b), f"[lane_d] {key}={b!r} parsed as {vb!r}"
    assert ld_a != ld_b, f"[lane_d] {key}: {a!r} and {b!r} parse to the same config"


# Every W2 key at a non-default value together with every legacy key: one
# table that must load, dump and load back equal.
ROUND_TRIP = {
    **{k: b for k, (_a, b, _co) in MUTATIONS.items()},
    **ARM,
    "from_block": ARM_BLOCK_2,
    "until_block": UNTIL_BLOCK_2,
    "share": 0.25,
    "takes_from": "lane_a",
    "remainder": "burn",
    "dtypes": ["BF16"],
    "gate_suites": ["gift", "boom", "time"],
    "gate_weights": {"gift": 1.0, "boom": 2.0, "time": 1.0},
    **LEGACY_VALUES,
    "notes": {"owner": "w2 round-trip"},
}
# rev6.23: one court weight per slot of the mutated reward_prior_kings.
ROUND_TRIP["court_weights"] = [0.9] + [0.1 / ROUND_TRIP["reward_prior_kings"]] * \
    ROUND_TRIP["reward_prior_kings"]
DUMP_KEYS = W2_KEYS + LEGACY_KEYS + ("notes",)


@pytest.mark.parametrize("name", SHIPPED)
def test_w2_round_trip_load_dump_load_equal(name, tmp_path):
    cfg1 = load_ok(raw_with(name, table=ROUND_TRIP), " (round-trip table)")
    text = splice_lane_d(name, dump_lane_d(cfg1.lane_d, DUMP_KEYS))
    cfg2 = load_text(text, tmp_path, name)
    assert cfg2.lane_d == cfg1.lane_d, "[lane_d] changed across load -> dump -> load"
    for k in W2_KEYS:
        if k == "gate_weights":
            assert weights(field(cfg2.lane_d, k)) == weights(ROUND_TRIP[k])
        else:
            assert norm(field(cfg2.lane_d, k)) == norm(ROUND_TRIP[k]), k


@pytest.mark.parametrize("name", SHIPPED)
def test_w2_round_trip_of_the_shipped_defaults(name, tmp_path):
    cfg1 = load_ok(shipped_raw(name))
    text = splice_lane_d(name, dump_lane_d(cfg1.lane_d, DUMP_KEYS))
    cfg2 = load_text(text, tmp_path, name)
    assert cfg2.lane_d == cfg1.lane_d
    assert comparable(cfg2) == comparable(cfg1)


@pytest.mark.parametrize("name", SHIPPED)
@pytest.mark.parametrize("key", LEGACY_KEYS)
def test_w2_legacy_and_placeholder_keys_still_parse(name, key):
    """The deployed file carries targets, teacher_*, gain_margin, ...: each
    still parses and is carried as written, armed or not."""
    co = {"teacher_models": ["chronos-2"]} if key == "teacher_mode" else {}
    for extra in ({}, ARM):
        ld = load_ok(raw_with(name, {**extra, **co, key: LEGACY_VALUES[key]}),
                     f" with legacy {key}").lane_d
        assert norm(field(ld, key)) == norm(LEGACY_VALUES[key]), (
            f"legacy [lane_d] {key} not carried: {field(ld, key)!r}")


@pytest.mark.parametrize("name", SHIPPED)
def test_w2_shipped_legacy_values_unchanged(name):
    """The shipped legacy values parse to today's values."""
    ld = load_ok(shipped_raw(name)).lane_d
    assert field(ld, "pool_prefix") == "pool_d" and field(ld, "teacher_api") == "ephemeris"
    assert field(ld, "teacher_mode") == "route" and field(ld, "gain_margin") == 0.01
    assert norm(field(ld, "data_sources")) == ("real", "prior_corpus")
    assert field(ld, "expert_choice") == "miner" and field(ld, "max_context_length") == 32768
