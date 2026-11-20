"""[scoring] crowning_anchor_from_block: a king crowned after the
tenure-blocks gate keeps its recorded crowning block. Before the fix the
gate's one-time legacy re-anchor also fired at every NEW king's first
settlement (a dethrone resets ``tenure_anchor_gate``) and imputed the anchor
from a 0-round counter — that settlement's own block — so post-gate kings'
tenure started one handled settlement late (mainnet 2026-11-18: crowned at
9530100, recorded 9531900)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from cascade.shared.config import load_chain_config
from cascade.shared.era import crowned_post_gate, crowning_anchor_active
from cascade.validator.state import ChampionState
from tests.unit.test_era_validator import _manifest, _noisy, _runner, _scores

REPO = Path(__file__).resolve().parents[2]


def test_loader_parses_the_knob(tmp_path):
    src = (REPO / "chain.toml").read_text()
    assert "crowning_anchor_from_block = 0\n" in src
    assert load_chain_config(REPO / "chain.toml").scoring.crowning_anchor_from_block == 0
    p = tmp_path / "chain.toml"
    p.write_text(src.replace("crowning_anchor_from_block = 0\n",
                             "crowning_anchor_from_block = 9554400\n"))
    assert load_chain_config(p).scoring.crowning_anchor_from_block == 9554400


def test_gate_helpers(cfg):
    s = replace(cfg.scoring, tenure_blocks_from_block=1000, crowning_anchor_from_block=5000)
    assert not crowning_anchor_active(s, 4999) and crowning_anchor_active(s, 5000)
    assert not crowning_anchor_active(replace(s, crowning_anchor_from_block=0), 10**9)
    assert crowned_post_gate(s, 1000) and not crowned_post_gate(s, 999)
    assert not crowned_post_gate(s, None)
    assert not crowned_post_gate(replace(s, tenure_blocks_from_block=0), 10**6)


def _base(cfg, *, fix_from):
    eb_old = cfg.round.epoch_blocks
    eb_new = eb_old // 4
    rollover = eb_old * 8
    base = replace(cfg, round=replace(cfg.round, epoch_blocks=eb_new, epoch_blocks_prev=eb_old,
                                      epoch_activation_block=rollover, rolling_from_block=rollover,
                                      era_settlements=4),
                   scoring=replace(cfg.scoring, era_king_from_block=rollover,
                                   tenure_blocks_from_block=rollover,
                                   margin_warmup_blocks=cfg.scoring.margin_warmup_rounds * eb_old,
                                   crowning_anchor_from_block=fix_from(rollover)))
    return base, rollover, eb_new


def _new_king_first_settlement(base, rollover, eb_new):
    """The validator's state after a post-gate dethrone at ``crowned``: the
    anchor gate was reset with the fresh reign (as on the dethrone path)."""
    crowned = rollover + 4 * eb_new
    old = ChampionState(king_hotkey="old_hk", king_uid=1, tenure_rounds=9,
                        king_since_block=rollover - 9 * base.round.epoch_blocks_prev,
                        tenure_anchor_gate=rollover)
    state = replace(old, king_hotkey="king_hk", king_uid=0, tenure_rounds=0,
                    king_since_block=crowned, tenure_anchor_gate=0)
    king = _scores(1.0, 0)
    r = _runner(base, king_scores=king, chal={"chal_hk": _noisy(king, 1.2, 9)}, state=state)
    nxt = crowned + 2 * eb_new
    out = r.process_settlement(_manifest(base, nxt), windows=[], base_seed=nxt)
    return r, out, crowned


def test_pre_fix_behaviour_is_unchanged_while_the_gate_is_off(cfg):
    base, rollover, eb_new = _base(cfg, fix_from=lambda _: 0)
    r, out, crowned = _new_king_first_settlement(base, rollover, eb_new)
    # the known offset: re-anchored to the first handled settlement's block
    assert r.state.king_since_block == crowned + 2 * eb_new
    assert out.king_tenure_rounds == 0
    assert r.state.tenure_anchor_gate == rollover


@pytest.mark.parametrize("fix_at", ["rollover", "later"])
def test_post_gate_king_keeps_its_crowning_block(cfg, fix_at):
    base, rollover, eb_new = _base(
        cfg, fix_from=(lambda r: r) if fix_at == "rollover" else (lambda r: r + 4 * eb_new))
    r, out, crowned = _new_king_first_settlement(base, rollover, eb_new)
    assert r.state.king_since_block == crowned          # the real crowning, kept
    assert out.king_tenure_rounds == 2                  # 2 settlements reigned
    assert r.state.tenure_anchor_gate == rollover       # marked: never revisited
    nxt = crowned + 3 * eb_new
    out2 = r.process_settlement(_manifest(base, nxt), windows=[], base_seed=nxt)
    assert out2.king_tenure_rounds == 3 and r.state.king_since_block == crowned


def test_legacy_king_is_still_re_anchored_at_the_gate_with_the_fix_on(cfg):
    # a king crowned BEFORE the gate (or with no recorded block) keeps the
    # DEC-CA-0043 counter re-anchor — the fleet-agreed quantity
    base, rollover, eb_new = _base(cfg, fix_from=lambda r: r)
    eb_old = base.round.epoch_blocks_prev
    early = ChampionState(king_hotkey="king_hk", king_uid=0, tenure_rounds=14,
                          king_since_block=rollover - 2 * eb_old)
    king = _scores(1.0, 0)
    r = _runner(base, king_scores=king, chal={"chal_hk": _noisy(king, 1.2, 9)}, state=early)
    out = r.process_settlement(_manifest(base, rollover), windows=[], base_seed=rollover)
    assert r.state.king_since_block == rollover - 14 * eb_old
    assert out.king_tenure_rounds == 56
