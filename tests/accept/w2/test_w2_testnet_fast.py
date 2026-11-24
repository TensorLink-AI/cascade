"""TESTNET-FAST profile (owner 2026-11-24): chain.testnet.toml scales the
[lane_d] waits and window caps to its 30-min settlement grid so lane D runs
end-to-end on testnet in hours. The rule logic is unchanged; only these
values move. Mainnet (chain.toml) carries none of them, and neither file arms
lane D (testnet arms it by the signed W-LAUNCH overlay)."""

from __future__ import annotations

import re

import pytest

from .helpers import (
    TESTNET,
    TESTNET_FAST_LANE_D,
    W2_DEFAULTS,
    _raw_text,
    field,
    file_raw,
    load_ok,
    norm,
    shipped_raw,
)


@pytest.mark.parametrize("key", sorted(TESTNET_FAST_LANE_D))
def test_testnet_file_carries_the_fast_value(key):
    default, fast = TESTNET_FAST_LANE_D[key]
    assert default != fast, key
    if key in W2_DEFAULTS:
        assert norm(default) == norm(W2_DEFAULTS[key]), key
    table = file_raw(TESTNET)["lane_d"]
    assert table[key] == fast, f"{TESTNET} [lane_d] {key} = {table[key]!r}, fast {fast!r}"
    ld = load_ok(file_raw(TESTNET), " (testnet-fast profile)").lane_d
    assert norm(field(ld, key)) == norm(fast), key


@pytest.mark.parametrize("key", sorted(TESTNET_FAST_LANE_D))
def test_mainnet_file_keeps_the_release_default(key):
    default, _fast = TESTNET_FAST_LANE_D[key]
    ld = load_ok(file_raw("chain.toml")).lane_d
    assert norm(field(ld, key)) == norm(default), key


@pytest.mark.parametrize("key", sorted(TESTNET_FAST_LANE_D))
def test_fast_line_names_the_mainnet_value(key):
    default, _fast = TESTNET_FAST_LANE_D[key]
    text = _raw_text(TESTNET)
    body = text[text.index("\n[lane_d]\n"):]
    body = body[:body.index("\n[", 1)]
    line = next(ln for ln in body.splitlines() if re.match(rf"^{re.escape(key)}\s*=", ln))
    m = re.search(r"# TESTNET-FAST \(2026-11-24\): (\S+)", line)
    assert m, f"{TESTNET} [lane_d] {key} has no TESTNET-FAST comment: {line!r}"
    assert float(m.group(1)) == float(default), line


def test_testnet_fast_keeps_lane_d_unarmed_and_inside_the_overlay_envelope():
    from cascade.shared.netconfig_bounds import value_refusals

    table = file_raw(TESTNET)["lane_d"]
    assert table["from_block"] == 0 and table["share"] == 0.0 and table["until_block"] == 0
    fast = {k: v for k, (_d, v) in TESTNET_FAST_LANE_D.items()}
    assert value_refusals({"lane_d": fast}) == []
    ld = load_ok(file_raw(TESTNET)).lane_d
    # the crown bar still falls start -> d_margin, now over 4 settlements
    assert ld.crown_bar(0) == ld.d_margin_start and ld.crown_bar(4) == ld.d_margin
    assert ld.forward_margin_hours <= ld.eval_delay_hours
    assert ld.min_windows <= ld.max_windows_per_settlement
    # restoring the profile gives the release defaults (what the W2 tests pin)
    base = load_ok(shipped_raw(TESTNET)).lane_d
    for k, (default, _f) in TESTNET_FAST_LANE_D.items():
        assert norm(field(base, k)) == norm(default), k
