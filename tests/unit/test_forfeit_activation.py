"""DEC-CA-0048 forfeiture DECIDED ON CHAIN through the DEC-CA-0045 machinery.

* The readiness note carries one segment per feature; a note with no
  forfeiture configured is byte-identical to the single-segment v1 note.
* The forfeit feature's name hashes the list AND the successor.
* The resolver locks the forfeit feature in independently of the rollover;
  the applied block is the first settlement of the next era.
* Validator runner: signals both segments, locks in, writes
  ``forfeit_from_block``, persists and restores; the trainer adopts the same
  block from the notes and never signals.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import cascade.shared.activation as A
from cascade.shared.config import load_chain_config
from cascade.shared.era import forfeit_successor, forfeited_hotkeys

REPO = Path(__file__).resolve().parents[2]
PRIMARY = "rolling-era-king"
GRID = 3600
B0 = 9046800 + GRID
K, S = "KINGKINGKINGKING", "SUCCESSORSUCCESSOR"


@pytest.fixture
def cfg():
    """The shipped chain.toml with its DEC-CA-0048 forfeiture and DEC-CA-0049
    margin-v2 bar DISARMED: these tests model the plain fleet (neither
    configured) and arm one explicitly."""
    c = load_chain_config(REPO / "chain.toml")
    return replace(c, scoring=replace(c.scoring, forfeit_hotkeys=(), forfeit_from_block=0,
                                      forfeit_successor_hotkey="", win_margin_start_v2=0.0,
                                      win_margin_end_v2=0.0, margin_warmup_blocks_v2=0,
                                      margin_v2_from_block=0))


def _typed(cfg):
    """Rolling eras already armed by the owner (the deployed shape today) so
    only the forfeiture is left to decide on chain."""
    return replace(cfg, round=replace(cfg.round, rolling_from_block=B0, epoch_blocks=900,
                                      epoch_blocks_prev=3600, epoch_activation_block=B0),
                   scoring=replace(cfg.scoring, era_king_from_block=B0, tenure_blocks_from_block=B0,
                                   cohort_maxt_increment_from_block=B0))


def _forfeit(cfg, *, block=0, successor=S):
    return replace(cfg, scoring=replace(cfg.scoring, forfeit_hotkeys=(K,), forfeit_from_block=block,
                                        forfeit_successor_hotkey=successor))


def _v(hk, stake):
    return A.ValidatorStake(hotkey=hk, stake=stake, permit=True, last_update=0)


def _fleet(*stakes):
    return [_v(f"v{i}", s) for i, s in enumerate(stakes, 1)]


class FakeChain:
    def __init__(self, validators, signals, *, block, hotkey="v1"):
        self.validators = list(validators)
        self.signals = dict(signals)
        self.block = block
        self.hotkey = hotkey
        self.written: list[str] = []

    def current_block(self):
        return self.block

    def validator_stakes(self, block=None):
        return list(self.validators)

    def read_plain_commitments(self, block=None):
        return dict(self.signals)

    def set_plain_commitment(self, payload):
        self.written.append(payload)
        self.signals[self.hotkey] = payload

    def hotkey_ss58(self):
        return self.hotkey


# ── the note ───────────────────────────────────────────────────────

def test_single_segment_note_is_unchanged_and_multi_segment_notes_round_trip():
    one = A.format_signal(PRIMARY, lock_block=10, activation_block=20)
    assert one == "cascade-ready:1:rolling-era-king:10:20"
    assert A.parse_signals(one) == (A.ReadySignal(PRIMARY, 10, 20),)
    assert A.parse_signal(one) == A.ReadySignal(PRIMARY, 10, 20)
    two = A.format_signals(A.ReadySignal(PRIMARY, 10, 20), A.ReadySignal("forfeit-abcd1234"))
    assert two == "cascade-ready:1:rolling-era-king:10:20:forfeit-abcd1234:0:0"
    assert A.parse_signals(two) == (A.ReadySignal(PRIMARY, 10, 20), A.ReadySignal("forfeit-abcd1234"))
    assert A.parse_signal(two) == A.ReadySignal(PRIMARY, 10, 20)          # primary reading survives
    assert A.signal_for(two, "forfeit-abcd1234") == A.ReadySignal("forfeit-abcd1234")
    assert A.signal_for(two, "other") is None
    # malformed: a dangling segment, a bad pair, a duplicate feature, a v2 note
    assert A.parse_signals("cascade-ready:1:a:0:0:b:0") is None
    assert A.parse_signals("cascade-ready:1:a:0:0:b:5:3") is None
    assert A.parse_signals("cascade-ready:1:a:0:0:a:0:0") is None
    assert A.parse_signals("cascade-ready:2:a:0:0") is None
    with pytest.raises(ValueError):
        A.format_signals(A.ReadySignal("a"), A.ReadySignal("a"))


def test_tally_counts_a_feature_wherever_it_sits_in_the_note():
    both = A.format_signals(A.ReadySignal(PRIMARY), A.ReadySignal("forfeit-x"))
    sig = {"v1": both, "v2": A.format_signal(PRIMARY), "v3": "cascade-ready:1:a:0:0:b:0"}
    t = A.tally("forfeit-x", _fleet(50, 30, 20), sig, threshold=0.51, block=B0)
    assert t.signed == ("v1",) and t.signed_stake == 50 and not t.locked
    t2 = A.tally(PRIMARY, _fleet(50, 30, 20), sig, threshold=0.51, block=B0)
    assert t2.signed == ("v1", "v2") and t2.locked                        # the malformed v3 counts as unsigned


# ── the feature ────────────────────────────────────────────────────

def test_forfeit_feature_name_hashes_list_and_successor(cfg):
    a = A.forfeit_feature_name((K, "B"), S)
    assert a.startswith("forfeit-") and len(a) == len("forfeit-") + 8
    assert A.forfeit_feature_name(("B", K), S) == a                         # order-free
    assert A.forfeit_feature_name((K, "B"), "OTHER") != a                   # a different successor is a new vote
    assert A.forfeit_feature_name((K,), S) != a
    assert A.forfeit_feature_name((), S) == ""
    spec = A.forfeit_feature(_forfeit(cfg))
    assert spec is not None and spec.decided_on_chain and spec.name == A.forfeit_feature_name((K,), S)
    assert A.forfeit_feature(_forfeit(cfg, block=B0)).typed_block == B0     # typed ⇒ not decided on chain
    assert not A.forfeit_feature(_forfeit(cfg, block=B0)).decided_on_chain
    assert A.forfeit_feature(cfg) is None                                    # nothing listed
    assert A.forfeit_feature(replace(_forfeit(cfg), activation=replace(cfg.activation, feature=""))) is None


def test_own_note_carries_the_forfeit_segment_only_when_one_is_decided_on_chain(cfg):
    rec = A.ActivationRecord()
    assert A.own_signal_payload(cfg, rec) == A.format_signal(PRIMARY)       # today's note, byte-identical
    assert A.own_signal_payload(_typed(cfg), rec) is None                    # typed rollover, no forfeiture: inert
    name = A.forfeit_feature(_forfeit(cfg)).name
    assert A.own_signal_payload(_forfeit(cfg), rec) == f"cascade-ready:1:{PRIMARY}:0:0:{name}:0:0"
    # a typed rollover with a forfeiture to decide: primary segment plain, forfeit segment live
    assert A.own_signal_payload(_forfeit(_typed(cfg)), rec) == f"cascade-ready:1:{PRIMARY}:0:0:{name}:0:0"
    locked = A.ActivationRecord(feature=name, lock_block=B0, activation_block=B0 + 900)
    assert A.own_signal_payload(_forfeit(_typed(cfg)), rec, locked) == (
        f"cascade-ready:1:{PRIMARY}:0:0:{name}:{B0}:{B0 + 900}")
    assert A.own_signal_payload(_forfeit(_typed(cfg), block=B0), rec) is None   # typed forfeiture too: inert


def test_apply_forfeit_activation_applies_at_the_boundary_after_the_lock(cfg):
    c = _forfeit(_typed(cfg))
    # the forfeiture lands at the resolved rollover itself — no era of notice (owner 2026-09-28)
    c1 = A.apply_forfeit_activation(c, B0 + 3600)
    assert c1.scoring.forfeit_from_block == B0 + 3600 and c1.activation.resolved_forfeit_block == B0 + 3600
    assert forfeited_hotkeys(c1.scoring, B0 + 3599) == frozenset()
    assert forfeited_hotkeys(c1.scoring, B0 + 3600) == frozenset({K})
    assert forfeit_successor(c1.scoring, B0 + 3600) == S
    # a rollover inside an era lands mid-era, on that boundary
    assert A.apply_forfeit_activation(c, B0 + 900).scoring.forfeit_from_block == B0 + 900
    assert A.apply_forfeit_activation(c, B0 + 4500).scoring.forfeit_from_block == B0 + 4500
    # idempotent on the same block; one-way on another; typed block untouched
    assert A.apply_forfeit_activation(c1, B0 + 3600) is c1
    with pytest.raises(ValueError):
        A.apply_forfeit_activation(c1, B0 + 7200)
    typed = _forfeit(_typed(cfg), block=B0 + 900)
    assert A.apply_forfeit_activation(typed, B0 + 3600) is typed
    assert A.configured_forfeit_block(typed) == B0 + 900 and A.configured_forfeit_block(c1) == 0
    assert A.apply_forfeit_activation(_typed(cfg), B0 + 3600).scoring.forfeit_from_block == 0   # nothing listed


def test_resolver_locks_the_forfeit_feature_independently_of_the_rollover(cfg):
    c = _forfeit(_typed(cfg))
    spec = A.forfeit_feature(c)
    note = A.format_signals(A.ReadySignal(PRIMARY), A.ReadySignal(spec.name))
    chain = FakeChain(_fleet(60, 40), {"v1": note}, block=B0 + 3600 + 5)
    res = A.resolve_activation(c, chain, now_block=B0 + 3600 + 5, record=A.ActivationRecord(), feature=spec)
    assert res.record.locked and res.record.feature == spec.name and res.record.source == "tally"
    assert res.record.lock_block == B0 + 3600 and res.record.activation_block == B0 + 3600 + 900
    # a stale record for another forfeit list is blanked, never applied
    other = A.ActivationRecord(feature="forfeit-deadbeef", lock_block=1, activation_block=2)
    assert A.record_for(c, other, spec) == A.ActivationRecord()


# ── validator runner ───────────────────────────────────────────────

def test_validator_runner_decides_the_forfeiture_on_chain_and_crowns_at_the_block(cfg, tmp_path):
    from cascade.validator.loop import ValidatorRunner

    c = _forfeit(_typed(cfg))
    spec = A.forfeit_feature(c)
    runner = ValidatorRunner(cfg=c, activation_store=A.ActivationStore(tmp_path / "act.json"))
    chain = FakeChain(_fleet(60, 40), {}, block=B0 + 3600 + 1, hotkey="v1")
    runner._activation_startup(chain)
    # the FIRST note carries both segments (primary plain: the rollover is typed); own 60% locks in the
    # forfeiture at boundary B0+3600; the note is then rewritten with the agreed pair
    assert chain.written[0] == f"cascade-ready:1:{PRIMARY}:0:0:{spec.name}:0:0"
    assert chain.written[-1] == f"cascade-ready:1:{PRIMARY}:0:0:{spec.name}:{B0 + 3600}:{B0 + 3600 + 900}"
    assert runner._forfeit.locked and runner._forfeit.activation_block == B0 + 3600 + 900
    assert runner.cfg.scoring.forfeit_from_block == B0 + 3600 + 900         # the boundary after the lock, mid-era
    assert runner.activation_block == 0                                      # the typed rollover records nothing
    assert A.ActivationStore(tmp_path / "activation_forfeit_state.json").load().activation_block == B0 + 3600 + 900
    # a restart restores the decision without a chain read
    fresh = ValidatorRunner(cfg=c, activation_store=A.ActivationStore(tmp_path / "act.json"))
    dead = FakeChain([], {}, block=B0 + 9000)
    dead.validator_stakes = lambda block=None: (_ for _ in ()).throw(RuntimeError("down"))
    fresh._activation_startup(dead)
    assert fresh.cfg.scoring.forfeit_from_block == B0 + 3600 + 900
    # status carries the forfeit view
    view = A.summary(fresh.cfg, fresh._activation, None, fresh._forfeit, None)
    assert view["forfeit"]["feature"] == spec.name and view["forfeit"]["activation_block"] == B0 + 3600 + 900


def test_validator_runner_is_silent_when_the_forfeiture_is_typed(cfg):
    from cascade.validator.loop import ValidatorRunner

    c = _forfeit(_typed(cfg), block=B0 + 900)
    runner = ValidatorRunner(cfg=c)
    chain = FakeChain(_fleet(60, 40), {}, block=B0 + 1, hotkey="v1")
    runner._activation_startup(chain)
    assert chain.written == [] and runner.cfg.scoring.forfeit_from_block == B0 + 900


def test_validator_crowns_the_successor_on_the_block_clock_without_a_manifest(cfg, tmp_path):
    from cascade.validator import state as state_mod
    from cascade.validator.loop import WEIGHTS_RATE_LIMIT_BLOCKS, ValidatorRunner
    from cascade.validator.state import ChampionState

    c = _forfeit(_typed(cfg), block=B0 + 900)
    c = replace(c, validator=replace(c.validator, state_db_path=str(tmp_path / "state.json")))
    state = ChampionState(king_hotkey=K, king_uid=105, tenure_rounds=3, former_kings=("OLD1", "OLD2"))
    runner = ValidatorRunner(cfg=c, state=state, evaluate_fn=lambda e, w: [], verify_signatures=False)
    runner._last_weight_block = B0 + 800
    # one block short of the gate: nothing moves
    runner._forfeit_tick(FakeChain([], {}, block=B0 + 899))
    assert runner.state.king_hotkey == K and runner._last_weight_block == B0 + 800
    # the gate reached between manifests: crowned at once, stamped with the GATE block (not the
    # poll block), court untouched, the standing weights pulled forward to the rate limit
    runner._forfeit_tick(FakeChain([], {}, block=B0 + 903))
    st = runner.state
    assert st.king_hotkey == S and st.king_uid is None and st.tenure_rounds == 0
    assert st.king_since_block == B0 + 900 and st.former_kings == ("OLD1", "OLD2")
    interval = int(c.validator.weight_set_interval_blocks)
    assert runner._last_weight_block == B0 + 800 - (interval - WEIGHTS_RATE_LIMIT_BLOCKS)
    assert state_mod.loads((tmp_path / "state.json").read_text()).king_hotkey == S   # persisted
    # idempotent afterwards; a chain hiccup is swallowed
    assert runner._forfeit_on_block(now_block=B0 + 950) is False
    dead = FakeChain([], {}, block=B0 + 950)
    dead.current_block = lambda: (_ for _ in ()).throw(RuntimeError("down"))
    runner._forfeit_tick(dead)
    assert runner.state.king_hotkey == S


# ── trainer ────────────────────────────────────────────────────────

def test_trainer_tick_adopts_the_forfeiture_block_from_the_notes_and_never_signals(cfg, tmp_path):
    from cascade.trainer.loop import TrainerRunner

    c = _forfeit(_typed(cfg))
    spec = A.forfeit_feature(c)
    runner = TrainerRunner(cfg=c, base_trainer=None, work_root=tmp_path,
                           activation_store=A.ActivationStore(tmp_path / "act.json"))
    runner.promotion = SimpleNamespace(round_cfg=c.round, scoring_cfg=c.scoring)
    note = A.format_signals(A.ReadySignal(PRIMARY), A.ReadySignal(spec.name, B0 + 3600, B0 + 3600 + 900))
    chain = FakeChain(_fleet(60, 40), {"v1": note}, block=B0 + 3600 + 100)
    runner._activation_tick(chain, B0 + 3600 + 100)
    assert runner.cfg.scoring.forfeit_from_block == B0 + 3600 + 900
    assert runner.promotion.scoring_cfg is runner.cfg.scoring
    assert chain.written == []
    assert A.ActivationStore(tmp_path / "activation_forfeit_state.json").load().source == "signals"
