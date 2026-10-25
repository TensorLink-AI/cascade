"""Lane D (DEC-CA-0062) is wired as an inert placeholder."""

import pytest

from cascade.shared.config import load_chain_config
from cascade.shared.config.lane_d import LaneDConfig, parse_lane_d


@pytest.mark.parametrize("toml", ["chain.toml", "chain.testnet.toml"])
def test_shipped_tables_parse_inert(toml):
    ld = load_chain_config(toml).lane_d
    assert ld.from_block == 0 and ld.share == 0.0 and ld.pool_prefix == "pool_d"
    assert not ld.active(10**9)


def test_arming_is_refused_until_implemented():
    with pytest.raises(ValueError, match="not implemented"):
        parse_lane_d({"from_block": 9_000_000})


@pytest.mark.parametrize("raw", [
    {"share": 1.0}, {"share": -0.1}, {"takes_from": "lane_z"}, {"gain_margin": 0.0},
    {"targets": "energy"}, {"from_block": -1}, {"data_sources": []},
    {"data_sources": ["synthetic"]}, {"expert_choice": "router"},
    {"max_context_length": 65536}, {"teacher_mode": "explicit"},
])
def test_placeholder_keys_are_validated(raw):
    with pytest.raises(ValueError):
        parse_lane_d(raw)


def test_missing_table_is_the_default():
    assert parse_lane_d(None) == LaneDConfig()
