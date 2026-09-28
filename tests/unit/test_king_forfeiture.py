"""DEC-CA-0048: king forfeiture (validator, consensus-gated) + admission denylist (trainer).

* `apply_forfeit`: a forfeited king abdicates to the most recent eligible former
  king (tenure/streaks reset); no successor ⇒ vacant throne; forfeited former
  kings leave the court; no-op when nothing is listed or nothing matches.
* `forfeited_hotkeys(scoring, block)` is empty before `forfeit_from_block` and
  when the list is empty — the gate every validator applies at the same block.
* Loader parses both knobs (`[scoring] forfeit_*`, `[round] blocked_*`).
* `RoundConfig.blocked_at(block)` gates the admission denylist.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from cascade.shared.config import load_chain_config
from cascade.shared.era import forfeited_hotkeys
from cascade.validator.state import ChampionState, apply_forfeit

K, P, Q, X = "KINGKINGKINGKING", "PRIORPRIORPRIOR", "OLDEROLDEROLDER", "XXXXXXXXXXXXXXXX"


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
    text = re.sub(r"(?m)^\[scoring\]$", f'[scoring]\nforfeit_hotkeys = ["{K}"]\nforfeit_from_block = 9190800', text, count=1)
    text = re.sub(r"(?m)^\[round\]$", f'[round]\nblocked_hotkeys = ["{K}", "{X}"]\nblocked_from_block = 9190800', text, count=1)
    p = tmp_path / "chain.toml"
    p.write_text(text)
    c = load_chain_config(p)
    assert c.scoring.forfeit_hotkeys == (K,) and c.scoring.forfeit_from_block == 9_190_800
    assert c.round.blocked_hotkeys == (K, X) and c.round.blocked_from_block == 9_190_800
    base = load_chain_config(root / "chain.toml")
    assert base.scoring.forfeit_hotkeys == () and base.scoring.forfeit_from_block == 0
    assert base.round.blocked_hotkeys == () and base.round.blocked_from_block == 0
