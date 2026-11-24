"""W-CMP (DEC-CA-0075): a lane-C competition is a spec composed only of
registry blocks, selected by overlay; MarketCast is one such spec and its
spec restates today's config byte-identically; unknown blocks are a typed
refusal whose share burns — never a crash or a fork."""

from __future__ import annotations

import pytest

from .helpers import SHIPPED, overlay, parse, shipped_raw, with_specs


def test_w_cmp_registry_lists_the_blocks():
    from cascade.shared import competition_spec as CS

    reg = CS.registry()
    assert set(reg["series_metrics"]) >= {"returns", "ohlcv", "vol", "range", "high", "low",
                                          "liquidity"}
    assert reg["scores"] == ["crps"] and set(reg["guards"]) == {"mase", "qlike", "tail"}
    assert reg["menu"]["test_families"] == ["plain_lcb", "maxt", "bonferroni"]
    assert reg["menu"]["pay_curves"] == ["tiered_decay", "fixed_term",
                                         "indefinite_until_beaten"]
    assert reg["runners"] == {"marketcast": ["mkt"]}
    assert "cascade-std-pool/1" in reg["pool_schemas"]
    from cascade.shared import netconfig as nc

    assert nc.overlay_policy()["competition_registry"] == reg


@pytest.mark.parametrize("name", SHIPPED)
def test_w_cmp_marketcast_spec_restates_the_shipped_competition(name):
    """``mkt:market`` as a spec is byte-identical: every MarketCast field, the
    lane-C competitions and the split are unchanged."""
    from cascade.shared.competition_spec import marketcast_spec
    from cascade.shared.lane_c import lane_c_split

    raw = shipped_raw(name)
    base = parse(raw)
    spec = marketcast_spec(base)
    got = parse(with_specs(raw, {"mkt": spec}))
    assert got.marketcast == base.marketcast
    assert got.lane_c.competition_list == base.lane_c.competition_list
    assert got.lane_c.targets == base.lane_c.targets
    assert got.lane_c.spec("mkt") is not None and not got.lane_c.spec_refusals
    for blk in (base.marketcast.from_block or 10**6, 10**9):
        assert lane_c_split(got.lane_c, blk, 3600, got.marketcast) == lane_c_split(
            base.lane_c, blk, 3600, base.marketcast)
    assert marketcast_spec(got) == spec


def test_w_cmp_spec_changes_reach_the_marketcast_judge():
    raw = shipped_raw()
    got = parse(with_specs(raw, {"mkt": {
        "runner": "marketcast", "gain_margin": 0.02, "alpha": 0.04,
        "test_family": "plain_lcb", "pay_curve": "indefinite_until_beaten",
        "embargo_days": 40, "embargo_eval_days": 30,
        "pool": {"prefix": "pool_mkt_v2", "signer": "owner", "schema": "cascade-std-pool/1"},
        "select": {"domain": "market"}}}))
    mc = got.marketcast
    assert (mc.gain_margin, mc.alpha, mc.test_family, mc.pay_curve) == (
        0.02, 0.04, "plain_lcb", "indefinite_until_beaten")
    assert (mc.embargo_days, mc.embargo_eval_days, mc.pool_prefix) == (40, 30, "pool_mkt_v2")
    assert (mc.pool_signer, mc.pool_schema, mc.pool_select) == (
        "", "cascade-std-pool/1", (("domain", "market"),))


@pytest.mark.parametrize("spec,code", [
    ({"runner": "marketcast", "metrics": ["returns", "sharpe9"]}, "unknown_block"),
    ({"runner": "marketcast", "score": "pinball"}, "unknown_block"),
    ({"runner": "marketcast", "test_family": "holm"}, "unknown_block"),
    ({"runner": "marketcast", "pay_curve": "lottery"}, "unknown_block"),
    ({"runner": "marketcast", "pool": {"schema": "cascade-std-pool/7"}}, "unknown_block"),
    ({"runner": "python", "code": "x"}, "unknown_key"),
    ({"runner": "gpu-script"}, "unknown_block"),
    ({"runner": "marketcast", "alpha": 0.4}, "out_of_bounds"),
    ({"runner": "marketcast", "gain_margin": 0.0}, "out_of_bounds"),
])
def test_w_cmp_unknown_or_unbounded_blocks_are_typed_refusals(spec, code):
    from cascade.shared.competition_spec import SpecRefusal, parse_spec

    r = parse_spec("mkt", spec)
    assert isinstance(r, SpecRefusal) and r.code == code, r


def test_w_cmp_refused_marketcast_spec_burns_never_crashes():
    from cascade.shared.lane_c import lane_c_split

    raw = shipped_raw()
    cfg = parse(with_specs(raw, {"mkt": {"runner": "marketcast", "metrics": ["sharpe9"]}}))
    assert cfg.marketcast.refused == "unknown_block"
    assert not cfg.marketcast.active(10**9)
    comps, slot = lane_c_split(cfg.lane_c, 10**9, 3600, cfg.marketcast)
    assert slot == {} and [c["state"] for c in comps] == ["pending"]
    rec = cfg.lane_c.specs_record()
    assert rec["refused"][0]["id"] == "mkt" and rec["refused"][0]["refused"] == "unknown_block"


def test_w_cmp_new_competition_without_a_runner_burns_its_share():
    """A spec for a new id this release cannot host is refused
    ``runner_capacity``; its declared weight stays in the split, so its part
    burns and MarketCast's part is unchanged in absolute terms."""
    from cascade.shared.lane_c import lane_c_split

    raw = shipped_raw()
    base = parse(raw)
    cfg = parse(with_specs(raw, {"energy": {"runner": "marketcast", "weight": 1.0,
                                            "select": {"domain": "energy"}}}))
    assert cfg.lane_c.spec_refusal("energy").code == "runner_capacity"
    assert cfg.lane_c.refused_weight == 1.0
    blk = 10**9
    _c0, s0 = lane_c_split(base.lane_c, blk, 3600, base.marketcast)
    _c1, s1 = lane_c_split(cfg.lane_c, blk, 3600, cfg.marketcast)
    assert set(s0) == set(s1)
    for t in s0:
        assert s1[t] == pytest.approx(s0[t] / 2)


def test_w_cmp_overlay_sets_a_spec_and_refuses_out_of_bounds():
    from cascade.shared.config import load_chain_config

    from .helpers import ROOT

    base = load_chain_config(ROOT / "chain.toml")
    out, why = overlay(base, {"lane_c": {"specs": {"mkt": {
        "runner": "marketcast", "gain_margin": 0.03, "pay_curve": "tiered_decay"}}}})
    assert why is None, why
    assert out.marketcast.gain_margin == 0.03 and out.marketcast.pay_curve == "tiered_decay"
    _o, why = overlay(base, {"lane_c": {"specs": {"mkt": {"runner": "marketcast",
                                                           "alpha": 0.3}}}})
    assert why is not None and "out of bounds" in why and "alpha" in why
    # an unknown NAME is not an overlay refusal: every node adopts it and the
    # competition is refused (typed) — a node one release behind burns the
    # same share instead of rejecting the whole overlay
    out, why = overlay(base, {"lane_c": {"specs": {"mkt": {"runner": "marketcast",
                                                            "metrics": ["sharpe9"]}}}})
    assert why is None, why
    assert out.marketcast.refused == "unknown_block"
