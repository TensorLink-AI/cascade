"""Lane D (LANE_D_PLAN W2): the shipped tables load unarmed; arming is an
era-boundary ``from_block`` (by signed overlay). The DEC-CA-0062 keys and
``confirm_on_king_change`` are retired (rev6.18 B11): a table that still
carries them loads, and they change nothing."""

import pytest

from cascade.shared.config import load_chain_config
from cascade.shared.config.lane_d import LaneDConfig, parse_lane_d


@pytest.mark.parametrize("toml", ["chain.toml", "chain.testnet.toml"])
def test_shipped_tables_parse_inert(toml):
    ld = load_chain_config(toml).lane_d
    assert ld.from_block == 0 and ld.share == 0.0
    assert not ld.active(10**9)


def test_arming_needs_an_era_boundary():
    era = lambda _b: 3600  # noqa: E731
    ld = parse_lane_d({"from_block": 9_000_000}, era_blocks_at=era)
    assert ld.from_block == 9_000_000
    assert not ld.active(9_000_000) and ld.active(9_000_900)
    with pytest.raises(ValueError, match="era boundary"):
        parse_lane_d({"from_block": 9_000_001}, era_blocks_at=era)


@pytest.mark.parametrize("raw", [
    {"share": 1.0}, {"share": -0.1}, {"takes_from": "lane_z"}, {"from_block": -1},
])
def test_live_keys_are_validated(raw):
    with pytest.raises(ValueError):
        parse_lane_d(raw)


@pytest.mark.parametrize("raw", [
    {"gain_margin": 0.0}, {"targets": "energy"}, {"data_sources": []},
    {"data_sources": ["synthetic"]}, {"expert_choice": "router"},
    {"max_context_length": 65536}, {"teacher_mode": "explicit"},
    {"confirm_on_king_change": "both"},
])
def test_retired_keys_load_and_change_nothing(raw):
    from cascade.shared.config import RETIRED_KEYS

    assert set(raw) <= RETIRED_KEYS["lane_d"]
    assert parse_lane_d(raw) == LaneDConfig()


def test_missing_table_is_the_default():
    assert parse_lane_d(None) == LaneDConfig()
