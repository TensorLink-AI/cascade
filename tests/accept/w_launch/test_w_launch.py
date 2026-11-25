"""W-LAUNCH (owner 2026-11-24): every new competition starts at the Wed
2026-11-25 launch era start. Lane C and MarketCast are typed in chain.toml;
lane B and lane D arm by a ready-to-sign two-step overlay that folds in the
A 0 / B 35 / C 15 / D 50 split, passes every loader / overlay check and the
R3 lint."""

from __future__ import annotations

import json
import math

import pytest

from .helpers import MAINNET_LAUNCH, PAIRS, ROOT, TESTNET_LAUNCH, bodies, overrides, shipped_cfg


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


@pytest.mark.parametrize("net", sorted(PAIRS))
def test_w_launch_lane_d_launches_live_only(net):
    """#894 (owner 2026-11-24): step 2 sets ``[lane_d] public_weight = 0.0``
    (step 1 never carries a lane D key), the built config judges on the live
    half alone, and with that weight the lane D judge and its inputs source
    need NO public inputs: nothing is drawn, no index / eval host / block
    hash is read (at the shipped 0.5 the same calls fail closed)."""
    from dataclasses import replace
    from types import SimpleNamespace

    from cascade.validator.lane_d import _public_half
    from cascade.validator.lane_d_inputs import LaneDInputsError, LaneDInputsSource

    base, b1, b2, launch = bodies(net)
    assert overrides(PAIRS[net][2])["lane_d"].get("public_weight") == 0.0
    assert "lane_d" not in b1["overrides"]
    assert float(base.lane_d.public_weight) == 0.5          # the shipped default
    out = nc_build(base, b2)
    assert float(out.lane_d.public_weight) == 0.0

    class _Boom:
        def block_hash(self, _b):
            raise AssertionError("public_weight = 0 must not read the block hash")

    def inputs(cfg):
        return SimpleNamespace(cfg=cfg, _public_indexes={}, _eval_host=None)

    # the judge: no public half, no indexes / scores / block hash needed
    assert _public_half(out.lane_d, None, None, None, 0, ["d" * 64]) is None
    # the inputs source: (None, None) without touching the reader or the host
    assert LaneDInputsSource._public(inputs(out), _Boom(), launch + 1, {}, frozenset(),
                                     ()) == (None, None)
    # control: at 0.5 the same missing inputs fail closed
    half = replace(out, lane_d=replace(out.lane_d, public_weight=0.5))
    with pytest.raises(LaneDInputsError, match="public"):
        LaneDInputsSource._public(inputs(half), _Boom(), launch + 1, {}, frozenset(), ())
    from cascade.validator.lane_d import LaneDPublicError

    with pytest.raises(LaneDPublicError):
        _public_half(half.lane_d, None, None, None, 0, ["d" * 64])


@pytest.mark.parametrize("net", sorted(PAIRS))
def test_w_launch_lane_d_embargo_is_24h_inside_the_bound(net):
    """Owner 2026-11-25 (leak mitigation, OPSLOG + cascade#935): lane A pool
    data up to snapshot 9579600 is already revealed on HF, so step 2 sets
    ``[lane_d] embargo_hours = 24`` (shipped 48) and no lane D embargo window
    overlaps it. Restore 48 by overlay after ~2026-11-27. Step 1 carries no
    lane D key; the value sits inside the W-OV bound and reaches the built
    config."""
    from cascade.shared.netconfig_bounds import BOUNDS

    base, b1, b2, _ = bodies(net)
    assert overrides(PAIRS[net][2])["lane_d"]["embargo_hours"] == 24.0
    assert "lane_d" not in b1["overrides"]
    bound = BOUNDS[("lane_d", "embargo_hours")]
    assert bound.lo <= 24.0 <= bound.hi
    assert float(nc_build(base, b2).lane_d.embargo_hours) == 24.0
    assert "restore 48" in (ROOT / PAIRS[net][2]).read_text().lower()


def nc_build(base, body):
    from cascade.shared import netconfig as nc

    return nc.build_config(base, body)
