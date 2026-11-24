"""W-LAUNCH (owner 2026-11-24): every new competition starts at the Wed
2026-11-25 launch era start. Lane C and MarketCast are typed in chain.toml;
lane B and lane D arm by a ready-to-sign two-step overlay that folds in the
A 0 / B 35 / C 15 / D 50 split, passes every loader / overlay check and the
R3 lint."""

from __future__ import annotations

import json
import math

import pytest

from .helpers import MAINNET_LAUNCH, PAIRS, TESTNET_LAUNCH, bodies, overrides, shipped_cfg


def _era(cfg, block: int) -> int:
    from cascade.shared.era import era_length_blocks

    return int(era_length_blocks(cfg.round, block))


def test_w_launch_blocks_are_era_starts_on_the_shipped_grids():
    from cascade.shared import netconfig as nc

    main, test = shipped_cfg("chain.toml"), shipped_cfg("chain.testnet.toml")
    assert _era(main, MAINNET_LAUNCH) == 3600 and MAINNET_LAUNCH % 3600 == 0
    assert _era(test, TESTNET_LAUNCH) == 600 and TESTNET_LAUNCH % 600 == 0
    assert nc.is_era_start(main, MAINNET_LAUNCH) and nc.is_era_start(test, TESTNET_LAUNCH)
    # the first era start after Wed 2026-11-25 18:00 UTC (12 s/block):
    # mainnet from 9573138 @ 2026-11-24 00:55:39 UTC, testnet from
    # 8511976 @ 2026-11-24 02:06:48 UTC
    for launch, anchor, t0 in ((MAINNET_LAUNCH, 9573138, 1795481739),
                               (TESTNET_LAUNCH, 8511976, 1795486008)):
        at_18 = anchor + (1795629600 - t0) / 12.0       # 2026-11-25T18:00:00Z
        era = 3600 if launch == MAINNET_LAUNCH else 600
        assert launch == (math.floor(at_18) // era + 1) * era
    assert int(main.v2.from_block) <= MAINNET_LAUNCH
    assert int(test.v2.from_block) <= TESTNET_LAUNCH
    # every launch overlay arms lane D at exactly its network's launch block
    for _, _, full, launch in PAIRS.values():
        assert overrides(full)["lane_d"]["from_block"] == launch


def test_w_launch_chain_toml_types_lane_c_and_marketcast():
    main = shipped_cfg("chain.toml")
    assert int(main.lane_c.from_block) == MAINNET_LAUNCH
    assert int(main.marketcast.from_block) == MAINNET_LAUNCH
    # lane B / lane D arm by overlay, never by a repo default
    assert int(main.lane_b.from_block) == 0 and int(main.lane_d.from_block) == 0
    assert not main.lane_c.active(MAINNET_LAUNCH)          # the settlement AT it is pre-lane-C
    assert main.lane_c.active(MAINNET_LAUNCH + 1)
    assert not main.marketcast.active(MAINNET_LAUNCH - 1) and main.marketcast.active(MAINNET_LAUNCH)


@pytest.mark.parametrize("net", sorted(PAIRS))
def test_w_launch_overlays_validate_and_chain(net):
    from cascade.shared import netconfig as nc

    base, b1, b2, launch = bodies(net)
    p1 = nc.plan_overlay(base, b1["overrides"], launch, None)
    assert p1.ok, p1.render()
    p2 = nc.plan_overlay(base, b2["overrides"], launch, b1)
    assert p2.ok, p2.render()
    assert b2["seq"] == b1["seq"] + 1 and b2["prev"] == nc.body_digest(b1)
    assert "lane_d" not in b1["overrides"]
    out = nc.build_config(base, b2)
    assert (out.lane_b.share, out.lane_c.share, out.lane_d.share) == (0.35, 0.15, 0.5)
    assert out.lane_c.takes_from == "lane_a" and out.lane_d.takes_from == "lane_a"
    assert int(out.lane_d.from_block) == launch
    assert not out.lane_d.active(launch) and out.lane_d.active(launch + 1)
    if net == "mainnet":
        # the loader stores lane B's first settlement (typed era start + 1)
        assert int(base.lane_b.from_block) == 0
        assert int(out.lane_b.from_block) == launch + 1
        assert int(out.lane_c.from_block) == int(out.marketcast.from_block) == launch
    else:
        assert int(out.lane_b.from_block) == int(base.lane_b.from_block)    # already armed
        assert "from_block" not in b1["overrides"].get("lane_b", {})
    # step 2 is a byte-identical restatement of step 1 outside [lane_d]
    rest = {t: kv for t, kv in b2["overrides"].items() if t != "lane_d"}
    assert json.dumps(rest, sort_keys=True) == json.dumps(b1["overrides"], sort_keys=True)


@pytest.mark.parametrize("net", sorted(PAIRS))
def test_w_launch_overlays_pass_the_r3_lint(net):
    from cascade.laned import overlay_lint as L

    base, b1, b2, _ = bodies(net)
    assert L.lint(b1, None, base, None) == []
    assert L.lint(b2, b1, base, None) == []
    # the single-shot body (step 2 at seq 1) is what the lint refuses: the
    # reason the launch ships as two steps
    from cascade.shared import netconfig as nc

    solo = nc.next_overlay(b2["overrides"], b2["effective_block"], None,
                           network=nc.network_identity(base))
    assert any("non-lane-d" in x for x in L.lint(solo, None, base, None))


@pytest.mark.parametrize("net", sorted(PAIRS))
def test_w_launch_overlays_restate_the_split(net):
    """#859's split-a0-b35-c15-d50 share keys, every one restated (overlays
    are snapshots)."""
    _, pre, full, _ = PAIRS[net]
    want = {"lane_b": {"share": 0.35},
            "lane_c": {"share": 0.15, "takes_from": "lane_a"},
            "lane_d": {"share": 0.5, "takes_from": "lane_a"}}
    got = overrides(full)
    for table, kv in want.items():
        for k, v in kv.items():
            assert got[table][k] == v, (net, table, k)
    for table in ("lane_b", "lane_c"):
        for k, v in want[table].items():
            assert overrides(pre)[table][k] == v


@pytest.mark.parametrize("net", sorted(PAIRS))
def test_w_launch_overlay_refused_before_one_era_ahead(net):
    """Published late, the plan warns: an overlay locked in after its
    effective block applies at the next era start, and its switch blocks
    (lane B / D from_block = the launch) would then lie before it."""
    from cascade.shared import netconfig as nc

    base, b1, b2, launch = bodies(net)
    era = _era(base, launch)
    plan = nc.plan_overlay(base, b2["overrides"], launch, b1, head_block=launch - era + 1)
    assert any("less than one era" in w for w in plan.warnings)
    late = nc.plan_overlay(base, b2["overrides"], launch + era, b1)
    assert not late.ok and any("lies before" in e for e in late.errors)


def test_w_launch_suite_targets_stay_unarmed():
    """[lane_b] suite_targets needs the release-only contamination_hashset
    pin; no launch overlay may arm it."""
    from cascade.shared import netconfig as nc

    for _, pre, full, _ in PAIRS.values():
        for rel in (pre, full):
            assert "suite_targets" not in overrides(rel).get("lane_b", {})
    base = shipped_cfg("chain.toml")
    body = nc.next_overlay({"lane_b": {"suite_targets": True}}, MAINNET_LAUNCH, None,
                           network=nc.network_identity(base))
    with pytest.raises(nc.NetConfigError, match="contamination_hashset"):
        nc.validate_overlay(base, body)
