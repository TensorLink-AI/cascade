"""DEC-CA-0056 [lane_c] config: defaults, guards, the typed-block checks and
the MarketCast-only competition registry (owner 2026-10-20)."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from cascade.shared.config import load_chain_config, parse_chain_config
from cascade.shared.config.lane_c import (
    COMPETITION_KINDS,
    DEFAULT_TARGETS,
    FORWARD_ONLY_REQUIRED,
    LaneCConfig,
    check_target,
    parse_lane_c,
    parse_target,
    register_competition_kind,
)
from cascade.shared.lane_c import active_targets

SUITES = ("returns", "vol")

ROOT = Path(__file__).resolve().parents[2]


def test_shipped_configs_are_inert_with_the_defaults():
    for f in ("chain.toml", "chain.testnet.toml"):
        lc = load_chain_config(ROOT / f).lane_c
        assert lc.from_block == 0 and not lc.active(10**12)
        assert lc.share == 0.25 and lc.remainder == "burn"
        assert lc.takes_from == ("lane_a" if f == "chain.toml" else "lane_b")
        assert lc.targets == DEFAULT_TARGETS == ("mkt:returns", "mkt:vol")
        assert (lc.full_pay_tiers, lc.half_life_rounds, lc.floor, lc.win_ttl_eras) == (
            ((0.05, 10), (0.02, 5), (0.01, 2)), 1.0, 0.1, 28)
        assert (lc.alpha, lc.bootstrap_B, lc.gain_margin, lc.min_target_windows) == \
            (0.05, 10_000, 0.01, 100)
        assert lc.guard_public == () and lc.tau_public == 0.01
        assert (lc.leg_hours, lc.legs_per_target, lc.max_base_age_eras) == (1.0, 2, 2)
        assert (lc.max_lr_scale, lc.max_steps_fraction) == (1.0, 1.0)
        assert not hasattr(lc, "pool_prefix")


def test_absent_section_is_the_default():
    assert parse_lane_c({}) == LaneCConfig(replay_settlements=64)


def test_c_from_a_cannot_overallocate_the_remaining_pool():
    cfg = parse_lane_c({"takes_from": "lane_a", "share": 0.25}, lane_b_share=0.25)
    assert cfg.takes_from == "lane_a"
    with pytest.raises(ValueError, match="available lane_a share"):
        parse_lane_c({"takes_from": "lane_a", "share": 0.8}, lane_b_share=0.25)
    mainnet = load_chain_config(ROOT / "chain.toml")
    assert mainnet.lane_b.share == 0.25
    assert mainnet.lane_c.share == 0.25 and mainnet.lane_c.takes_from == "lane_a"


def test_shipped_configs_load_with_marketcast_off_and_zero_active_slots():
    """MarketCast off (from_block 0): the shipped mkt: targets load, count no
    active slot at any block, so an armed lane C would burn its whole share."""
    for f in ("chain.toml", "chain.testnet.toml"):
        cfg = load_chain_config(ROOT / f)
        assert cfg.marketcast.from_block == 0
        assert tuple(cfg.marketcast.suites) == SUITES
        for block in (0, 7201, 10**9):
            assert active_targets(cfg.lane_c, block, cfg.marketcast) == ()


def test_target_specs():
    assert parse_target("mkt:returns", mkt_suites=SUITES) == ("mkt", "returns")
    assert check_target("mkt:vol", mkt_suites=SUITES) == "mkt:vol"
    for bad in ("energy", "mkt:", "mkt:Returns", "mkt:range", "region:eu", ""):
        with pytest.raises(ValueError):
            parse_target(bad, mkt_suites=SUITES)


@pytest.mark.parametrize("spec", ["domain:energy", "freq:sub_hourly", "horizon:720"])
def test_generic_targets_are_refused_forward_only_required(spec):
    with pytest.raises(ValueError, match="forward-only eval") as e:
        parse_target(spec, mkt_suites=SUITES)
    assert FORWARD_ONLY_REQUIRED in str(e.value) and "DEC-CA-0056" in str(e.value)
    with pytest.raises(ValueError, match=r"\[lane_c\] .*only MarketCast \(mkt:\) has one"):
        parse_lane_c({"targets": ["mkt:returns", spec]})


def test_competition_registry():
    assert COMPETITION_KINDS["mkt"].forward_only is True
    assert {k for k, c in COMPETITION_KINDS.items() if not c.forward_only} == \
        {"domain", "freq", "horizon"}
    for bad in (None, 1, "yes"):
        with pytest.raises(ValueError, match="forward_only"):
            register_competition_kind("future", eval="some eval", forward_only=bad)
    with pytest.raises(TypeError):
        register_competition_kind("future", eval="some eval")      # no forward_only at all
    with pytest.raises(ValueError, match="already registered"):
        register_competition_kind("mkt", eval="x", forward_only=True)
    assert "future" not in COMPETITION_KINDS


@pytest.mark.parametrize("kv, msg", [
    ({"share": 1.5}, "share"),
    ({"takes_from": "lane_z"}, "takes_from"),
    ({"remainder": "lane_a"}, "remainder"),
    ({"targets": ["mkt:returns", "mkt:returns"]}, "unique"),
    ({"targets": "mkt:returns"}, "list"),
    ({"targets": ["domain:energy"]}, "forward-only eval"),
    ({"gain_margin": 0.0}, "gain_margin"),
    ({"floor": 0.0}, "floor"),
    ({"win_ttl_eras": 3}, "win_ttl_eras"),
    ({"n_windows": 50}, "min_target_windows"),
    ({"max_steps_fraction": 1.5}, "max_steps_fraction"),
    ({"pool_prefix": "pool_c"}, "pool_prefix is retired"),
    ({"guard_public": {"mkt:returns": ["gift"]}}, "public bench guards are off"),
    ({"guard_public": {"domain:energy": ["gift"]}}, "forward-only eval"),
    ({"bootstrap_B": 1.5}, "integer"),
])
def test_guards(kv, msg):
    with pytest.raises(ValueError, match=msg):
        parse_lane_c(kv)


def test_armed_block_checks():
    era = 1800 * 4
    ok = parse_lane_c({"from_block": 3 * era}, v2_from_block=2 * era, era_blocks=era,
                      lane_b_share=0.5)
    # the era STARTING at from_block is lane C's: its first settlement is
    # strictly after it (DEC-CA-0043), the one AT it stays pre-lane-C
    assert not ok.active(3 * era) and ok.active(3 * era + 1800)
    assert ok.era_active(3 * era) and not ok.era_active(3 * era - 1)
    with pytest.raises(ValueError, match="era boundary"):
        parse_lane_c({"from_block": 3 * era + 1800}, v2_from_block=era, era_blocks=era)
    with pytest.raises(ValueError, match="needs \\[v2\\]"):
        parse_lane_c({"from_block": 3 * era}, v2_from_block=0, era_blocks=era)
    with pytest.raises(ValueError, match="precedes"):
        parse_lane_c({"from_block": era}, v2_from_block=2 * era, era_blocks=era)
    with pytest.raises(ValueError, match="exceeds"):
        parse_lane_c({"from_block": 3 * era, "share": 0.6}, v2_from_block=era,
                     era_blocks=era, lane_b_share=0.5)
    # dormant: none of the armed-only checks bite…
    parse_lane_c({}, v2_from_block=0, era_blocks=era, lane_b_share=0.5)
    # …except the share bound, checked always so an overlay cannot stage it
    with pytest.raises(ValueError, match="exceeds"):
        parse_lane_c({}, v2_from_block=0, era_blocks=era, lane_b_share=0.1)


def test_loader_arms_lane_c_on_an_era_boundary(tmp_path):
    cfg = load_chain_config(ROOT / "chain.toml")
    era = cfg.round.epoch_blocks * cfg.round.era_settlements
    v2 = 10_000 * era
    raw = copy.deepcopy(cfg.raw)
    raw["v2"]["from_block"] = v2
    # DEC-CA-0063 needs [netconfig] owner_hotkey once v2 is typed; lane C only
    raw["v2"]["bench_source"] = "trainer"
    # Lane C takes its allocation from B, which now launches independently.
    raw["lane_b"]["from_block"] = v2
    raw["lane_c"]["from_block"] = v2 + era
    armed = parse_chain_config(raw)
    assert armed.lane_c.from_block == v2 + era
    # The raw launch boundary is aligned; the effective B gate excludes
    # the final legacy settlement at that very block.
    assert armed.lane_b.from_block == v2 + 1
    raw["lane_c"]["from_block"] = v2 + era + 1
    with pytest.raises(ValueError, match="era boundary"):
        parse_chain_config(raw)
