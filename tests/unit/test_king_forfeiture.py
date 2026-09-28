"""DEC-CA-0048: king forfeiture (validator, consensus-gated) + admission denylist (trainer).

* `apply_forfeit`: a forfeited king abdicates to the NAMED successor
  (`[scoring] forfeit_successor_hotkey`) when one is set, else to the most
  recent eligible former king (tenure/streaks reset); neither ⇒ vacant throne;
  forfeited former kings leave the court; no-op when nothing is listed or
  nothing matches.
* `forfeited_hotkeys(scoring, block)` is empty before `forfeit_from_block` and
  when the list is empty — the gate every validator applies at the same block.
* Loader parses both knobs (`[scoring] forfeit_*`, `[round] blocked_*`).
* `RoundConfig.blocked_at(block)` gates the admission denylist.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from cascade.shared.config import load_chain_config
from cascade.shared.era import era_first_settlement, forfeit_successor, forfeited_hotkeys
from cascade.validator.state import ChampionState, apply_forfeit

K, P, Q, X = "KINGKINGKINGKING", "PRIORPRIORPRIOR", "OLDEROLDEROLDER", "XXXXXXXXXXXXXXXX"
S = "SUCCESSORSUCCESSOR"


def _state(king=K, court=(P, Q), tenure=5):
    return ChampionState(king_hotkey=king, king_uid=105, tenure_rounds=tenure,
                         streaks={"CHAL": 2}, rounds_seen=40, former_kings=court)


def test_forfeited_king_abdicates_to_most_recent_eligible_former_king():
    t = apply_forfeit(_state(), forfeited={K}, keep_former_kings=4)
    assert t is not None and t.dethroned and t.new_king_hotkey == P
    assert t.state.king_hotkey == P and t.state.king_uid is None
    assert t.state.former_kings == (Q,) and t.state.tenure_rounds == 0 and t.state.streaks == {}
    assert t.state.rounds_seen == 40
    assert t.note.startswith("forfeit:KINGKINGKING")


def test_forfeited_king_and_forfeited_heir_skip_to_the_next():
    t = apply_forfeit(_state(), forfeited={K, P}, keep_former_kings=4)
    assert t.state.king_hotkey == Q and t.state.former_kings == ()


def test_no_successor_leaves_the_throne_vacant():
    t = apply_forfeit(_state(court=()), forfeited={K}, keep_former_kings=4)
    assert t.dethroned and t.new_king_hotkey is None and t.state.king_hotkey is None


def test_forfeited_former_king_leaves_the_court_only():
    t = apply_forfeit(_state(), forfeited={Q}, keep_former_kings=4)
    assert t is not None and not t.dethroned and t.new_king_hotkey == K
    assert t.state.king_hotkey == K and t.state.former_kings == (P,) and t.state.tenure_rounds == 5


def test_noop_when_nothing_matches_or_nothing_listed():
    assert apply_forfeit(_state(), forfeited={X}, keep_former_kings=4) is None
    assert apply_forfeit(_state(), forfeited=set(), keep_former_kings=4) is None


def test_idempotent():
    t = apply_forfeit(_state(), forfeited={K}, keep_former_kings=4)
    assert apply_forfeit(t.state, forfeited={K}, keep_former_kings=4) is None


def test_court_cap_respected_after_abdication():
    t = apply_forfeit(_state(court=(P, Q, X)), forfeited={K}, keep_former_kings=1)
    assert t.state.king_hotkey == P and t.state.former_kings == (Q,)


# ── the named successor ───────────────────────────────────────

def test_named_successor_is_crowned_fresh_and_the_forfeited_king_leaves_the_court():
    st = replace(_state(), king_pointer="ptr-old", era_index=2545, king_since_block=9_100_000,
                 resync_holds=2, last_resync_round_id="r1", last_handled_round_id="r1",
                 last_handled_manifest_sha="abc")
    t = apply_forfeit(st, forfeited={K}, keep_former_kings=4, successor=S, crowned_block=9_190_800)
    assert t is not None and t.dethroned and t.new_king_hotkey == S
    n = t.state
    assert n.king_hotkey == S and n.king_uid is None and n.tenure_rounds == 0 and n.streaks == {}
    assert n.former_kings == (P, Q)                     # the court stays; the forfeited king is NOT retired into it
    assert n.king_since_block == 9_190_800 and n.king_pointer == "" and n.era_index is None
    assert n.resync_holds == 0 and n.last_resync_round_id is None
    assert n.last_handled_round_id == "r1" and n.last_handled_manifest_sha == "abc"   # loop position kept
    assert n.rounds_seen == 40
    assert t.note == f"forfeit:{K[:12]}→{S[:12]}"


def test_named_successor_already_in_the_court_moves_to_the_throne():
    t = apply_forfeit(_state(court=(P, S, Q)), forfeited={K}, keep_former_kings=4, successor=S)
    assert t.state.king_hotkey == S and t.state.former_kings == (P, Q)


def test_a_forfeited_successor_falls_back_to_the_court_rule():
    t = apply_forfeit(_state(), forfeited={K, S}, keep_former_kings=4, successor=S)
    assert t.state.king_hotkey == P and t.state.former_kings == (Q,)


def test_successor_without_a_forfeited_king_only_trims_the_court():
    t = apply_forfeit(_state(), forfeited={Q}, keep_former_kings=4, successor=S)
    assert not t.dethroned and t.state.king_hotkey == K and t.state.former_kings == (P,)


def test_successor_crowning_is_idempotent():
    t = apply_forfeit(_state(), forfeited={K}, keep_former_kings=4, successor=S, crowned_block=1)
    assert apply_forfeit(t.state, forfeited={K}, keep_former_kings=4, successor=S, crowned_block=2) is None


def test_forfeit_successor_gate(cfg):
    s = replace(cfg.scoring, forfeit_hotkeys=(K,), forfeit_from_block=9_190_800,
                forfeit_successor_hotkey=S)
    assert forfeit_successor(s, 9_190_799) == "" and forfeit_successor(s, None) == ""
    assert forfeit_successor(s, 9_190_800) == S
    assert forfeit_successor(replace(s, forfeit_hotkeys=(K, S)), 9_190_800) == ""     # listed ⇒ court rule
    assert forfeit_successor(replace(s, forfeit_successor_hotkey=""), 9_190_800) == ""
    assert forfeit_successor(cfg.scoring, 10**9) == ""


def test_era_first_settlement_is_one_grid_step_after_the_start(cfg):
    r = replace(cfg.round, epoch_blocks=900)
    assert era_first_settlement(r, 9_162_000) == 9_162_900


# ── the consensus gate ────────────────────────────────────────────────────────

def test_forfeited_hotkeys_gate(cfg):
    s = replace(cfg.scoring, forfeit_hotkeys=(K,), forfeit_from_block=9_190_800)
    assert forfeited_hotkeys(s, 9_190_799) == frozenset()
    assert forfeited_hotkeys(s, None) == frozenset()
    assert forfeited_hotkeys(s, 9_190_800) == frozenset({K})
    assert forfeited_hotkeys(replace(s, forfeit_from_block=0), 10**9) == frozenset()   # no block ⇒ inert
    assert forfeited_hotkeys(replace(s, forfeit_hotkeys=()), 10**9) == frozenset()
    assert forfeited_hotkeys(cfg.scoring, 10**9) == frozenset()                        # shipped default


# ── admission denylist gate ──────────────────────────────────────────────────

def test_blocked_at(cfg):
    r = replace(cfg.round, blocked_hotkeys=(K, X), blocked_from_block=100)
    assert r.blocked_at(99) == frozenset() and r.blocked_at(None) == frozenset()
    assert r.blocked_at(100) == frozenset({K, X})
    assert replace(r, blocked_from_block=0).blocked_at(1) == frozenset({K, X})          # 0 = immediately
    assert cfg.round.blocked_at(10**9) == frozenset()


# ── loader ───────────────────────────────────────────────────────────────────

def test_loader_parses_both_knobs(tmp_path):
    root = Path(__file__).resolve().parents[2]
    text = (root / "chain.toml").read_text()
    import re
    text = re.sub(r"(?m)^\[scoring\]$", f'[scoring]\nforfeit_hotkeys = ["{K}"]\nforfeit_from_block = 9190800\nforfeit_successor_hotkey = "{S}"', text, count=1)
    text = re.sub(r"(?m)^\[round\]$", f'[round]\nblocked_hotkeys = ["{K}", "{X}"]\nblocked_from_block = 9190800', text, count=1)
    p = tmp_path / "chain.toml"
    p.write_text(text)
    c = load_chain_config(p)
    assert c.scoring.forfeit_hotkeys == (K,) and c.scoring.forfeit_from_block == 9_190_800
    assert c.scoring.forfeit_successor_hotkey == S
    assert c.round.blocked_hotkeys == (K, X) and c.round.blocked_from_block == 9_190_800
    base = load_chain_config(root / "chain.toml")
    assert base.scoring.forfeit_hotkeys == () and base.scoring.forfeit_from_block == 0
    assert base.scoring.forfeit_successor_hotkey == ""
    assert base.round.blocked_hotkeys == () and base.round.blocked_from_block == 0


def test_loader_refuses_a_successor_that_is_itself_forfeited(tmp_path):
    import re

    import pytest
    root = Path(__file__).resolve().parents[2]
    text = (root / "chain.toml").read_text()
    text = re.sub(r"(?m)^\[scoring\]$", f'[scoring]\nforfeit_hotkeys = ["{K}", "{S}"]\nforfeit_successor_hotkey = "{S}"', text, count=1)
    p = tmp_path / "chain.toml"
    p.write_text(text)
    with pytest.raises(ValueError, match="forfeit_successor_hotkey"):
        load_chain_config(p)


# ── the resync valve never crowns a forfeited king back ──────────────────────

def test_resync_valve_holds_instead_of_adopting_a_forfeited_trained_king(cfg):
    from cascade.shared.manifest import (
        TrainedEntry,
        TrainingManifest,
        contract_digest,
        format_trained_pointer,
    )
    from cascade.validator.loop import ValidatorRunner
    from tests.unit.test_validator_round import CID, CID2

    gate = 9_190_800                                                  # a 3600-grid boundary
    entries = [TrainedEntry("king_hk", 0, "king", CID, format_trained_pointer(CID2), "d", gate + 5),
               TrainedEntry("chal_hk", 1, "challenger", CID, format_trained_pointer(CID2), "d", gate + 5)]
    manifest = TrainingManifest(round_id="1", created_block=gate + 5,
                                contract_digest=contract_digest(cfg.training),
                                base_arch_digest=cfg.training.base_arch_digest,
                                eval_dataset=cfg.eval.eval_dataset, entries=entries)
    forf = replace(cfg, scoring=replace(cfg.scoring, king_resync_max_rounds=5,
                                        forfeit_hotkeys=("king_hk",), forfeit_from_block=gate,
                                        forfeit_successor_hotkey=S))
    state = ChampionState(king_hotkey=S, king_uid=None, tenure_rounds=0, resync_holds=4)
    runner = ValidatorRunner(cfg=forf, state=state, evaluate_fn=lambda e, w: [], verify_signatures=False)
    new_state, reason = runner._resync_step(manifest)
    assert new_state.king_hotkey == S                                 # held, not demoted to the forfeited king
    assert not reason.startswith("king_resync_demoted")
    # the same manifest without the forfeiture trips the valve as before
    plain = ValidatorRunner(cfg=replace(cfg, scoring=replace(cfg.scoring, king_resync_max_rounds=5)),
                            state=state, evaluate_fn=lambda e, w: [], verify_signatures=False)
    demoted, reason2 = plain._resync_step(manifest)
    assert demoted.king_hotkey == "king_hk" and reason2.startswith("king_resync_demoted")
