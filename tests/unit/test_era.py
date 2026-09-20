"""Era arithmetic + the DEC-CA-0043 rollover knobs (cascade.shared.era)."""
from __future__ import annotations

import re
import tomllib
from dataclasses import replace
from pathlib import Path

import pytest

from cascade.shared import era as E
from cascade.shared.config import load_chain_config

REPO = Path(__file__).resolve().parents[2]


def _armed(cfg, rollover: int, *, settlements: int = 4, prev: int = 0):
    round_cfg = replace(cfg.round, rolling_from_block=rollover,
                        era_settlements=settlements,
                        epoch_blocks_prev=prev,
                        epoch_activation_block=rollover if prev else 0)
    scoring = replace(cfg.scoring, era_king_from_block=rollover,
                      tenure_blocks_from_block=rollover,
                      cohort_maxt_from_block=1, increment_from_block=1,
                      cohort_maxt_increment_from_block=1)
    return replace(cfg, round=round_cfg, scoring=scoring)


# ── gates ────────────────────────────────────────────────────────────────────

def test_gates_are_off_at_zero_and_flip_at_the_block(cfg):
    assert not E.rolling_active(cfg.round, 10**9)
    assert not E.era_king_active(cfg.scoring, 10**9)
    assert not E.tenure_blocks_active(cfg.scoring, 10**9)
    eb = cfg.round.epoch_blocks
    armed = _armed(cfg, eb * 10)
    assert not E.rolling_active(armed.round, eb * 10 - 1)
    assert E.rolling_active(armed.round, eb * 10)
    assert E.era_king_active(armed.scoring, eb * 10)
    assert not E.era_king_active(armed.scoring, None)
    assert E.tenure_blocks_active(armed.scoring, eb * 10)


# ── era grid ─────────────────────────────────────────────────────────────────

def test_era_boundaries_are_grid_boundaries_and_seed_from_previous_era(cfg):
    eb = cfg.round.epoch_blocks
    armed = _armed(cfg, eb * 8, settlements=4)
    length = eb * 4
    assert E.era_length_blocks(armed.round, eb * 8) == length
    for block in (eb * 8, eb * 9, eb * 11, eb * 11 + eb - 1):
        era = E.era_for_block(armed.round, block)
        assert era.index == block // length
        assert era.start_block == era.index * length
        assert era.start_block % eb == 0
        assert era.seed_block == era.start_block - length
    assert E.era_for_block(armed.round, eb * 12).index == E.era_for_block(armed.round, eb * 8).index + 1
    assert E.next_era_start(armed.round, eb * 9) == eb * 12
    assert E.next_era_start(armed.round, eb * 12) == eb * 16


def test_era_seed_block_floors_at_zero_for_the_first_era(cfg):
    armed = _armed(cfg, cfg.round.epoch_blocks * 4)
    assert E.era_for_block(armed.round, 0).seed_block == 0


def test_era_length_follows_the_grid_in_force_at_the_block(cfg):
    eb = cfg.round.epoch_blocks
    # 4× faster grid from the rollover: eras stay 4 settlements long, so an
    # era after the switch is a quarter of the blocks of one before it.
    armed = _armed(cfg, eb * 4, prev=eb * 4)
    armed = replace(armed, round=replace(armed.round, epoch_blocks=eb))
    assert E.era_length_blocks(armed.round, eb * 4 - 1) == eb * 4 * 4
    assert E.era_length_blocks(armed.round, eb * 4) == eb * 4


def test_member_rotation_is_era_index_mod_k():
    assert [E.member_index_for_era(i, 3) for i in range(7)] == [0, 1, 2, 0, 1, 2, 0]
    assert E.member_index_for_era(5, 0) == 0


def test_min_effective_era_is_one_full_era_of_notice(cfg):
    eb = cfg.round.epoch_blocks
    armed = _armed(cfg, eb * 4)
    length = eb * 4
    # created anywhere inside era 3 ⇒ effective no earlier than era 5
    for created in (3 * length, 3 * length + 1, 4 * length - 1):
        assert E.min_effective_era(armed.round, created) == 5
    assert E.min_effective_era(armed.round, 4 * length) == 6


# ── tenure / ripeness re-denomination ────────────────────────────────────────

def test_margin_warmup_rounds_unchanged_before_the_gate(cfg):
    eb = cfg.round.epoch_blocks
    armed = _armed(cfg, eb * 10)
    armed = replace(armed, scoring=replace(armed.scoring, margin_warmup_blocks=eb * 8))
    assert armed.koth_params(block=eb * 9).margin_warmup_rounds == cfg.scoring.margin_warmup_rounds
    assert armed.koth_params(block=None).margin_warmup_rounds == cfg.scoring.margin_warmup_rounds


def test_margin_warmup_blocks_keeps_wall_time_across_a_grid_switch(cfg):
    eb = cfg.round.epoch_blocks          # the NEW (fast) grid
    prev = eb * 4                        # the old grid
    rollover = prev * 5
    armed = _armed(cfg, rollover, prev=prev)
    armed = replace(armed, scoring=replace(
        armed.scoring, margin_warmup_rounds=8, margin_warmup_blocks=8 * prev))
    # before: 8 rounds of the old grid; after: 32 settlements of the new grid
    assert armed.koth_params(block=rollover - prev).margin_warmup_rounds == 8
    assert armed.koth_params(block=rollover).margin_warmup_rounds == 32
    assert E.effective_reign_threshold_rounds(
        armed.round, replace(armed.scoring, cascade_reign_blocks=5 * prev), rollover) == 20
    assert E.effective_reign_threshold_rounds(
        armed.round, replace(armed.scoring, cascade_reign_blocks=5 * prev), rollover - 1
    ) == cfg.scoring.cascade_reign_days


def test_tenure_in_blocks_keeps_a_king_crowned_on_the_old_grid_whole(cfg):
    eb = cfg.round.epoch_blocks
    prev = eb * 4
    rollover = prev * 5
    armed = _armed(cfg, rollover, prev=prev)
    # pre-gate: the counter, verbatim
    assert E.tenure_rounds_at(armed.round, armed.scoring, block=rollover - prev,
                              tenure_rounds=14, king_since_block=None) == 14
    # first settlement after the switch, legacy king with no crowning block:
    # 14 old rounds are 56 new settlements — the margin stays decayed
    assert E.tenure_rounds_at(armed.round, armed.scoring, block=rollover,
                              tenure_rounds=14, king_since_block=None) == 56
    # a recorded crowning block is used directly
    assert E.tenure_rounds_at(armed.round, armed.scoring, block=rollover + eb * 3,
                              tenure_rounds=99, king_since_block=rollover) == 3
    assert E.tenure_rounds_at(armed.round, armed.scoring, block=rollover,
                              tenure_rounds=0, king_since_block=rollover) == 0


# ── EraSpec round-trip ───────────────────────────────────────────────────────

def test_era_spec_json_round_trip():
    spec = E.EraSpec(index=7, start_block=25200, seed_block=21600, generation=9, member_index=1)
    assert E.EraSpec.from_json(spec.to_json()) == spec
    with pytest.raises(ValueError):
        E.EraSpec.from_json({"index": 1})
    with pytest.raises(ValueError):
        E.EraSpec.from_json("nope")


# ── loader ───────────────────────────────────────────────────────────────────

def _toml_with(path: Path, tmp_path: Path, **overrides) -> Path:
    text = path.read_text()
    raw = tomllib.loads(text)
    out = []
    for line in text.splitlines():
        key = line.split("=", 1)[0].strip() if "=" in line else ""
        if key in overrides:
            continue
        out.append(line)
    # append overrides at the end of the named sections by re-emitting the
    # sections we touch (round / scoring) with the override lines appended
    body = "\n".join(out) + "\n"
    for section, keys in (("round", ("rolling_from_block", "era_settlements",
                                     "epoch_blocks", "epoch_blocks_prev",
                                     "epoch_activation_block")),
                          ("scoring", ("era_king_from_block", "tenure_blocks_from_block",
                                       "cohort_maxt_from_block",
                                       "cohort_maxt_increment_from_block"))):
        extra = [f"{k} = {overrides[k]!r}" for k in keys if k in overrides]
        if extra:
            body = re.sub(rf"^\[{section}\]\n", f"[{section}]\n" + "\n".join(extra) + "\n",
                          body, count=1, flags=re.MULTILINE)
    assert raw  # parsed the source fine
    p = tmp_path / path.name
    p.write_text(body)
    return p


def test_shipped_tomls_parse_the_knobs():
    main = load_chain_config(REPO / "chain.toml")
    assert main.round.rolling_from_block == 0
    assert main.scoring.era_king_from_block == 0
    assert main.scoring.tenure_blocks_from_block == 0
    assert main.round.era_settlements == 4
    assert main.scoring.margin_warmup_blocks == 8 * main.round.epoch_blocks
    assert main.scoring.cascade_reign_blocks == main.scoring.cascade_reign_days * main.round.epoch_blocks
    test = load_chain_config(REPO / "chain.testnet.toml")
    eb = test.round.epoch_blocks
    assert test.round.rolling_from_block == eb
    assert test.scoring.era_king_from_block == eb
    assert test.scoring.tenure_blocks_from_block == eb
    assert test.round.era_settlements == 4
    assert test.scoring.margin_warmup_blocks == test.scoring.margin_warmup_rounds * eb
    assert test.scoring.cascade_reign_blocks == test.scoring.cascade_reign_days * eb


def test_loader_rejects_unequal_rollover_keys(tmp_path):
    p = _toml_with(REPO / "chain.toml", tmp_path, rolling_from_block=7200,
                   era_king_from_block=10800, tenure_blocks_from_block=7200,
                   cohort_maxt_from_block=1, cohort_maxt_increment_from_block=1)
    with pytest.raises(ValueError, match="same block"):
        load_chain_config(p)


def test_loader_rejects_a_partially_set_rollover(tmp_path):
    p = _toml_with(REPO / "chain.toml", tmp_path, rolling_from_block=7200,
                   cohort_maxt_from_block=1, cohort_maxt_increment_from_block=1)
    with pytest.raises(ValueError, match="ONE block"):
        load_chain_config(p)


def test_loader_rejects_a_misaligned_rollover(tmp_path):
    p = _toml_with(REPO / "chain.toml", tmp_path, rolling_from_block=3601,
                   era_king_from_block=3601, tenure_blocks_from_block=3601,
                   cohort_maxt_from_block=1, cohort_maxt_increment_from_block=1)
    with pytest.raises(ValueError, match="boundary"):
        load_chain_config(p)


def test_loader_requires_the_rollover_on_both_grids(tmp_path):
    # rollover 10800 is a boundary of 3600 but the grid switch to 900 must
    # land on the same block
    p = _toml_with(REPO / "chain.toml", tmp_path, rolling_from_block=10800,
                   era_king_from_block=10800, tenure_blocks_from_block=10800,
                   epoch_blocks=900, epoch_blocks_prev=3600,
                   epoch_activation_block=14400,
                   cohort_maxt_from_block=1, cohort_maxt_increment_from_block=1)
    with pytest.raises(ValueError, match="must equal"):
        load_chain_config(p)


def test_loader_requires_the_stack_order(tmp_path):
    p = _toml_with(REPO / "chain.toml", tmp_path, rolling_from_block=7200,
                   era_king_from_block=7200, tenure_blocks_from_block=7200,
                   cohort_maxt_from_block=1, cohort_maxt_increment_from_block=14400)
    with pytest.raises(ValueError, match="stacks on"):
        load_chain_config(p)


def test_loader_accepts_the_rollover_recipe(tmp_path):
    p = _toml_with(REPO / "chain.toml", tmp_path, rolling_from_block=14400,
                   era_king_from_block=14400, tenure_blocks_from_block=14400,
                   epoch_blocks=900, epoch_blocks_prev=3600,
                   epoch_activation_block=14400,
                   cohort_maxt_from_block=3600, cohort_maxt_increment_from_block=7200)
    cfg = load_chain_config(p)
    assert cfg.round.rolling_from_block == 14400
    assert E.era_length_blocks(cfg.round, 14400) == 3600
    assert E.era_length_blocks(cfg.round, 14399) == 14400
    assert cfg.koth_params(block=14400).margin_warmup_rounds == 32
    assert cfg.koth_params(block=14399).margin_warmup_rounds == 8
