"""Lane composition (A / B / C shares and lane A's internal split) is set by
config and therefore by a signed overlay (DEC-CA-0057) — no validator update."""

from dataclasses import replace

import pytest

from cascade.shared.config import load_chain_config
from cascade.shared.config.v2 import LADDER_DEFAULTS, parse_v2_economics
from cascade.shared.emission import (
    COURT_BLOCKS,
    LADDER_CAP_BP,
    LADDER_FAIL_BP,
    LADDER_START_LEVEL_BP,
    LADDER_STEP_BP,
    ladder_params,
    ladder_step,
)
from cascade.shared.netconfig import build_config


def _base():
    return load_chain_config("chain.toml")


def test_defaults_equal_the_shipped_constants():
    lp = ladder_params(_base())
    assert lp == {"ladder_start_level_bp": LADDER_START_LEVEL_BP,
                  "ladder_step_bp": LADDER_STEP_BP,
                  "ladder_cap_bp": LADDER_CAP_BP, "ladder_fail_bp": LADDER_FAIL_BP,
                  "key_slash_bp": 100, "court_bp": 500, "court_blocks": COURT_BLOCKS}
    assert LADDER_DEFAULTS["court_blocks"] == COURT_BLOCKS


def test_ladder_follows_config():
    cfg = _base()
    cfg = replace(cfg, v2=replace(cfg.v2, ladder_start_level_bp=4000, ladder_step_bp=500,
                                  ladder_cap_bp=9000, ladder_fail_bp=2000))
    assert ladder_step(None, crowned=True, check=None, cfg=cfg) == 4000
    assert ladder_step(8800, crowned=False, check="pass", cfg=cfg) == 9000
    assert ladder_step(6000, crowned=False, check="fail", cfg=cfg) == 2000
    # without a config: the constants (bit-identical to before)
    assert ladder_step(None, crowned=True, check=None) == LADDER_START_LEVEL_BP


@pytest.mark.parametrize("raw", [
    {"ladder_fail_bp": 6000},                       # fail above start
    {"ladder_cap_bp": 9800},                        # cap + court > 100 %
    {"ladder_start_level_bp": 9600},                # start above cap
    {"court_blocks": -1},
    {"key_slash_bp": "100"},
])
def test_ladder_refusals(raw):
    with pytest.raises(ValueError):
        parse_v2_economics(raw)


def test_overlay_sets_lane_shares_and_ladder():
    base = _base()
    body = {"protocol": "cascade-netconfig/2", "network": {"netuid": 91, "chain": "bittensor:finney"}, "seq": 1, "prev": "", "effective_block": 0,
            "overrides": {"lane_b": {"share": 0.6},
                          "lane_c": {"share": 0.3, "takes_from": "lane_b"},
                          "v2": {"ladder_cap_bp": 9000, "court_bp": 1000,
                                 "king_check_margin": 0.3}}}
    cfg = build_config(base, body)
    # A = 1 − lane_b.share, B = lane_b.share − lane_c.share, C = lane_c.share
    assert (round(1 - cfg.lane_b.share, 6), round(cfg.lane_b.share - cfg.lane_c.share, 6),
            cfg.lane_c.share) == (0.4, 0.3, 0.3)
    assert ladder_params(cfg)["ladder_cap_bp"] == 9000
    assert cfg.v2.king_check_margin == 0.3


def test_overlay_refuses_lane_c_above_lane_b():
    body = {"protocol": "cascade-netconfig/2", "network": {"netuid": 91, "chain": "bittensor:finney"}, "seq": 1, "prev": "", "effective_block": 0,
            "overrides": {"lane_b": {"share": 0.2},
                          "lane_c": {"share": 0.3, "takes_from": "lane_b"}}}
    with pytest.raises(ValueError, match="exceeds"):
        build_config(_base(), body)


def test_overlay_mainnet_lane_c_takes_from_lane_a():
    from cascade.shared.lane_c import lane_pool_shares

    body = {"protocol": "cascade-netconfig/2", "network": {"netuid": 91, "chain": "bittensor:finney"}, "seq": 1, "prev": "", "effective_block": 0,
            "overrides": {"lane_b": {"share": 0.2}, "lane_c": {"share": 0.3}}}
    cfg = build_config(_base(), body)
    b, c = lane_pool_shares(cfg.lane_b, cfg.lane_c)
    assert (round(1 - b - c, 6), b, c) == (0.5, 0.2, 0.3)
