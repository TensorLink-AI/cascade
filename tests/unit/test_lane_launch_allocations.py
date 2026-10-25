"""Staged lane launches: the validator's actual weights and audit replay."""

import tomllib
from dataclasses import replace
from pathlib import Path

import pytest

from cascade.audit.checks import check_weights
from cascade.shared.config import load_chain_config
from cascade.shared.lane_b import DomainWin
from cascade.validator.state import ChampionState

from .test_dec55_validator import KING, OLD, _Client, _manifest, _receipt, _runner

ROOT = Path(__file__).resolve().parents[2]


def _config(tmp_path):
    raw = tomllib.loads((ROOT / "chain.testnet.toml").read_text())
    start = raw["v2"]["from_block"]
    era = raw["round"]["epoch_blocks"] * raw["round"]["era_settlements"]
    raw["v2"]["king_ladder"] = True
    raw["lane_b"]["from_block"] = start + 2 * era
    raw["lane_b"]["share"] = 0.25
    raw["lane_c"]["from_block"] = start + 4 * era
    raw["lane_c"]["takes_from"] = "lane_a"
    raw["validator"]["state_db_path"] = str(tmp_path / "state.json")
    return load_chain_config(raw=raw), start, era


@pytest.mark.parametrize("phase", ["a_only", "b_boundary", "a_b", "c_boundary", "a_b_c"])
def test_staged_weights_keep_ladder_and_burn_unused_launched_slots(tmp_path, phase):
    cfg, start, era = _config(tmp_path)
    grid = cfg.round.epoch_blocks
    b_start, c_start = start + 2 * era, start + 4 * era
    boundary = {"a_only": start + grid, "b_boundary": b_start,
                "a_b": b_start + grid, "c_boundary": c_start,
                "a_b_c": c_start + grid}[phase]
    b_live, c_live = boundary > b_start, boundary > c_start
    b_share = 0.25 if b_live else 0.0
    state = ChampionState(king_hotkey=KING, king_uid=3, king_since_block=start + grid,
                          king_share_bp=5000, court_hotkey=OLD,
                          court_until_block=boundary + 100)
    r = _runner(cfg, state)
    m = _manifest(boundary)
    r._v2_settlement_economics(m, [], 1, None, KING)
    if b_live:
        r._lane_b().record(DomainWin(domain="energy", hotkey="expert", uid=7,
            round_id="old", merge_block=boundary, target_lcb=0.05, tier=0.05, full_eras=10))
    client = _Client(boundary, {KING: 3, OLD: 4, "expert": 7})
    r._lane_b_block = r._lane_c_block = boundary
    vector = r._apply_weights(client, m.round_id, [3, 4], boundary=boundary)
    a_share = 0.5 if c_live else (0.75 if b_live else 1.0)
    assert vector[3] == pytest.approx(0.5 * a_share)
    assert vector[4] == pytest.approx(0.05 * a_share)
    assert vector[7] == pytest.approx(b_share / len(cfg.lane_b.targets))
    assert vector[0] == pytest.approx(1 - sum(vector[u] for u in (3, 4, 7)))
    assert sum(vector) == pytest.approx(1)
    assert (r._last_lane_b_pay is not None) is b_live
    assert (r._last_lane_c_pay is not None) is c_live
    if c_live:
        assert r._last_lane_b_pay["share"] == 0.25
        assert r._last_lane_c_pay["share"] == 0.25
    receipt = _receipt(r, m, boundary, [3, 4], vector)
    receipt = replace(receipt, lane_c_pay=r._last_lane_c_pay,
                      lane_c=[] if c_live else None)
    assert check_weights(receipt, cfg).status == "PASS"


@pytest.mark.parametrize("king_bp", [9500, 1000, 100])
def test_a_only_cap_failed_check_and_key_slash_scale_to_full_pool(tmp_path, king_bp):
    cfg, start, _ = _config(tmp_path)
    boundary = start + cfg.round.epoch_blocks
    state = ChampionState(king_hotkey=KING, king_uid=3, king_since_block=boundary,
                          king_share_bp=9500 if king_bp == 100 else king_bp,
                          king_slashed=king_bp == 100)
    r = _runner(cfg, state)
    client = _Client(boundary, {KING: 3})
    r._lane_b_block = boundary
    vector = r._apply_weights(client, "standing", [3], boundary=boundary)
    assert vector[3] == pytest.approx(king_bp / 10000)
    assert vector[0] == pytest.approx(1 - king_bp / 10000)


def test_lane_c_cannot_launch_before_or_without_b(tmp_path):
    cfg, start, era = _config(tmp_path)
    for b_start in (0, start + 5 * era):
        raw = {**cfg.raw, "lane_b": {**cfg.raw["lane_b"], "from_block": b_start}}
        with pytest.raises(ValueError, match="needs lane B live"):
            load_chain_config(raw=raw)
