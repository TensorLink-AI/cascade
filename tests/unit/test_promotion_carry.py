"""DEC-CA-0050 staged-promotion carry, DECIDED ON CHAIN through the DEC-CA-0045
machinery.

The 2026-10-14 wedge: gen 14 was STAGED for era 2577, a dethrone at block
9275400 (after the record fired at 9271800, before era 2577 began) rebuilt the
Cascade state and dropped ``pending_*``; the record was then re-judged against
the NEW king's reign clock ("fired before the reign clock ripened") and every
era-2577 manifest declaring gen 14 was rejected ``era_init_mismatch``.

* Before ``promotion_carry_from_block`` every decision is bit-identical to the
  old rule (the wedge replays exactly).
* From it: ``crown`` carries the staged promotion and records the previous
  reign's anchor; a record that fired under an earlier reign is timed against
  that reign's clock (when observed) or not at all; a record fired within the
  current reign is judged by the current clock as before; a promotion
  re-staged after its effective era began installs at the next settlement.
* The rule is decided on chain (segment ``promotion-carry``, next settlement
  boundary after lock-in), persisted, restored, stamped on receipts
  (``promotion_carry_block``, drop-when-default) and checked by the audit.
"""
from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import cascade.shared.activation as A
from cascade.shared.bench_report import bench_report_key, dump_bench_report, load_bench_report
from cascade.shared.config import DEFAULT_CHAIN_TOML, load_chain_config, promotion_carry_active
from cascade.shared.era import member_index_for_era, min_effective_era, settlement_era
from cascade.validator import cascade as CA
from cascade.validator.cascade import CascadeController, CascadeState
from tests.unit.test_era_validator import (
    M1A,
    M1B,
    M2A,
    _armed,
    _manifest,
    _promo_store,
    _rollover,
    _runner,
)
from tests.unpinned import load_unpinned, strip_pins

REPO = Path(__file__).resolve().parents[2]
PRIMARY = "rolling-era-king"
PC = A.PROMOTION_CARRY_FEATURE
B0 = 9046800 + 3600


# ── shared fixtures ──────────────────────────────────────────────────────────

def _carry_on(cfg, block):
    return replace(cfg, scoring=replace(cfg.scoring, promotion_carry_from_block=int(block)))


def _cas(armed, tmp_path, **state):
    base = dict(king_hotkey="king_hk", reign_start_block=0, clock_observed=True,
                generation=1, members=(M1A, M1B))
    base.update(state)
    return CascadeController(
        reign_days=armed.scoring.cascade_reign_days, round_cfg=armed.round,
        state=CascadeState(**base), state_path=tmp_path / "cascade_state.json")


def _timeline(cfg):
    """The incident's shape on the test grid: the record fires inside era n−2
    (effective era n = the earliest allowed), is staged at a settlement of era
    n−1, and the dethrone lands at a LATER settlement of era n−1 — after the
    record fired, before era n starts."""
    armed = _armed(cfg)
    eb = cfg.round.epoch_blocks
    length = eb * int(armed.round.era_settlements)
    n_block = _rollover(armed) + length * 3 + eb       # era n's first settlement
    era_n = settlement_era(armed.round, n_block)
    fired = era_n.start_block - 2 * length + eb        # inside era n−2
    assert min_effective_era(armed.round, fired) == era_n.index
    stage_block = era_n.start_block - 2 * eb           # a settlement of era n−1
    dethrone_block = era_n.start_block - eb            # a later one, still era n−1
    assert fired < stage_block < dethrone_block < era_n.start_block
    return SimpleNamespace(armed=armed, eb=eb, length=length, n_block=n_block, era_n=era_n,
                           fired=fired, stage_block=stage_block, dethrone_block=dethrone_block)


def _old_member(armed, block):
    era = settlement_era(armed.round, block)
    idx = member_index_for_era(era.index, 2)
    return (M1A, M1B)[idx], idx


def _dethrone(v, armed, block, new_king="chal_hk"):
    """Drive the REAL cascade step for a dethrone settlement at ``block``."""
    outcome = SimpleNamespace(transition=SimpleNamespace(dethroned=True,
                                                         new_king_hotkey=new_king))
    v._cascade_round(_manifest(armed, block), outcome)


def _incident(cfg, tmp_path, *, carry_block):
    t = _timeline(cfg)
    armed = _carry_on(t.armed, carry_block) if carry_block else t.armed
    store = _promo_store([(M2A, 0.4)], generation=2, effective_era=t.era_n.index,
                         fired_block=t.fired)
    v = _runner(armed, cascade=_cas(armed, tmp_path), store=store)
    # staged at a settlement of era n−1 (the 04:05:49 "staged for era" line)
    member, idx = _old_member(armed, t.stage_block)
    assert v.check_manifest(_manifest(armed, t.stage_block, warm_start=member, generation=1,
                                      member_index=idx)) is None
    assert v.cascade.state.pending_generation == 2
    # the dethrone (16:13:57 "reign clock reset for new king … (observed)")
    _dethrone(v, armed, t.dethrone_block)
    assert v.cascade.state.king_hotkey == "chal_hk"
    assert v.cascade.state.reign_start_block == t.dethrone_block
    m_n = _manifest(armed, t.n_block, warm_start=M2A, generation=2, member_index=0)
    return t, v, m_n


# ── the incident replay ─────────────────────────────────────────────────────

def test_incident_replay_rejected_before_activation(cfg, tmp_path):
    t, v, m_n = _incident(cfg, tmp_path, carry_block=0)
    # the old crown dropped the staged promotion …
    assert v.cascade.state.pending_generation == 0
    # … and the re-evaluation judges the record on the NEW king's clock
    reason = v.check_manifest(m_n)
    assert reason is not None and reason.startswith("era_init_mismatch")
    assert "generation 2 member 0, the ledger names generation 1" in reason
    assert v.cascade.state.generation == 1 and v.cascade.state.pending_generation == 0


def test_incident_replay_accepted_and_installed_when_active(cfg, tmp_path):
    t, v, m_n = _incident(cfg, tmp_path, carry_block=_rollover(_armed(cfg)))
    st = v.cascade.state
    assert (st.pending_generation, st.pending_members, st.pending_effective_era) == (
        2, (M2A,), t.era_n.index)
    assert st.prev_reign_start_block == 0 and st.prev_clock_observed
    assert v.check_manifest(m_n) is None
    assert v.process_settlement(m_n, windows=[], base_seed=t.n_block) is not None
    assert v.cascade.state.generation == 2 and v.cascade.state.members == (M2A,)
    assert v.cascade.state.pending_generation == 0
    # the king persists across the install; the old init is now refused
    assert v.cascade.state.king_hotkey == "chal_hk"
    member, idx = _old_member(t.armed, t.n_block)
    assert v.check_manifest(_manifest(t.armed, t.n_block, warm_start=member, generation=1,
                                      member_index=idx)).startswith("era_init_mismatch")


def test_gate_is_judged_per_block_a_dethrone_before_it_keeps_the_old_rule(cfg, tmp_path):
    # Activation AFTER the dethrone: the dethrone itself still runs the old
    # rule (drops the staged promotion) — every validator agrees on that
    # block — and the later settlement re-stages under the new rule.
    t = _timeline(cfg)
    gate = t.era_n.start_block                       # a settlement boundary after the dethrone
    assert promotion_carry_active(_carry_on(t.armed, gate).scoring, t.n_block)
    assert not promotion_carry_active(_carry_on(t.armed, gate).scoring, t.dethrone_block)
    t, v, m_n = _incident(cfg, tmp_path, carry_block=gate)
    assert v.cascade.state.pending_generation == 0
    assert v.cascade.state.prev_reign_start_block is None
    assert v.check_manifest(m_n) is None             # prior-reign record, timing unattestable
    assert v.cascade.state.pending_generation == 2


# ── crown / serialisation ───────────────────────────────────────────────────

def _staged_state():
    return CascadeState(king_hotkey="A", reign_start_block=3600, clock_observed=True,
                        generation=13, members=("m0", "m1"), pending_generation=14,
                        pending_members=("n0", "n1", "n2"), pending_effective_era=2577)


def test_crown_without_carry_is_the_old_rule_bit_identical():
    st = CA.crown(_staged_state(), king_hotkey="B", block=9275400)
    assert st == CascadeState(king_hotkey="B", reign_start_block=9275400, checkpoints=(),
                              generation=13, members=("m0", "m1"), clock_observed=True)
    assert CA.crown(_staged_state(), king_hotkey="B", block=9275400,
                    carry_pending=False) == st
    assert "prev_reign_start_block" not in CA.dumps(st)


def test_crown_with_carry_keeps_the_staged_promotion_and_the_previous_anchor():
    st = CA.crown(_staged_state(), king_hotkey="B", block=9275400, carry_pending=True)
    assert (st.king_hotkey, st.reign_start_block, st.checkpoints) == ("B", 9275400, ())
    assert (st.generation, st.members) == (13, ("m0", "m1"))
    assert (st.pending_generation, st.pending_members, st.pending_effective_era) == (
        14, ("n0", "n1", "n2"), 2577)
    assert st.prev_reign_start_block == 3600 and st.prev_clock_observed
    # an adopted or unanchored outgoing clock is recorded as NOT attesting
    adopted = CA.crown(replace(_staged_state(), clock_observed=False), king_hotkey="B",
                       block=9275400, carry_pending=True)
    assert adopted.prev_reign_start_block == 3600 and not adopted.prev_clock_observed
    unanchored = CA.crown(replace(_staged_state(), reign_start_block=None), king_hotkey="B",
                          block=9275400, carry_pending=True)
    assert unanchored.prev_reign_start_block is None and not unanchored.prev_clock_observed
    # an accepted promotion starts a fresh reign: no previous anchor carried
    acc = CA.accept_promotion(st, generation=14, members=("n0",), block=9277200)
    assert acc.prev_reign_start_block is None and acc.pending_generation == 0


def test_state_serialisation_is_drop_when_default_and_round_trips():
    plain = CA.crown(_staged_state(), king_hotkey="B", block=9275400)
    obj = json.loads(CA.dumps(plain))
    assert "prev_reign_start_block" not in obj and "prev_clock_observed" not in obj
    assert CA.loads(CA.dumps(plain)) == plain
    carried = CA.crown(_staged_state(), king_hotkey="B", block=9275400, carry_pending=True)
    obj = json.loads(CA.dumps(carried))
    assert obj["prev_reign_start_block"] == 3600 and obj["prev_clock_observed"] is True
    assert CA.loads(CA.dumps(carried)) == carried
    # a pre-DEC-CA-0050 state file loads with no previous reign
    legacy = json.dumps({"king_hotkey": "B", "reign_start_block": 5, "generation": 1,
                         "members": ["x"], "clock_observed": True, "checkpoints": []})
    st = CA.loads(legacy)
    assert st.prev_reign_start_block is None and not st.prev_clock_observed


def test_controller_note_dethrone_carries_only_when_asked(tmp_path):
    c = CascadeController(reign_days=5, state=_staged_state(), state_path=tmp_path / "s.json")
    c.note_dethrone("B", block=9275400)
    assert c.state.pending_generation == 0
    c = CascadeController(reign_days=5, state=_staged_state(), state_path=tmp_path / "s.json")
    c.note_dethrone("B", block=9275400, carry_pending=True)
    assert c.state.pending_generation == 14
    assert CA.load_state(tmp_path / "s.json") == c.state           # persisted


# ── ripeness of a prior-reign record ────────────────────────────────────────

def _prior_reign_case(cfg, tmp_path, *, prev, prev_observed, active=True, fired=None,
                      reign_start=None, store_fired=None):
    t = _timeline(cfg)
    armed = _carry_on(t.armed, _rollover(t.armed)) if active else t.armed
    fired = t.fired if fired is None else fired
    cas = _cas(armed, tmp_path, king_hotkey="chal_hk",
               reign_start_block=t.dethrone_block if reign_start is None else reign_start,
               prev_reign_start_block=prev, prev_clock_observed=prev_observed)
    store = _promo_store([(M2A, 0.4)], generation=2, effective_era=t.era_n.index,
                         fired_block=fired if store_fired is None else store_fired)
    v = _runner(armed, cascade=cas, store=store)
    m_n = _manifest(armed, t.n_block, warm_start=M2A, generation=2, member_index=0)
    return t, v, m_n


def test_prior_reign_record_ripe_on_the_previous_observed_clock_is_staged(cfg, tmp_path):
    t, v, m_n = _prior_reign_case(cfg, tmp_path, prev=0, prev_observed=True)
    assert v.cascade.prior_reign_anchor(t.fired) == 0
    assert v.check_manifest(m_n) is None
    assert v.cascade.state.pending_generation == 2


def test_prior_reign_record_unripe_on_the_previous_observed_clock_is_ignored(cfg, tmp_path):
    t = _timeline(cfg)
    prev = t.fired - t.eb                          # one round of reign: not ripe (5 needed)
    t, v, m_n = _prior_reign_case(cfg, tmp_path, prev=prev, prev_observed=True)
    assert v.cascade.prior_reign_anchor(t.fired) == prev
    assert v.check_manifest(m_n).startswith("era_init_mismatch")
    assert v.cascade.state.pending_generation == 0


@pytest.mark.parametrize("prev,observed", [(None, False), (None, True), ("unripe", False)])
def test_prior_reign_record_without_an_attesting_clock_skips_timing(cfg, tmp_path, prev,
                                                                    observed):
    t = _timeline(cfg)
    if prev == "unripe":
        prev = t.fired - t.eb                      # would fail if it were attested
    t, v, m_n = _prior_reign_case(cfg, tmp_path, prev=prev, prev_observed=observed)
    assert v.cascade.prior_reign_anchor(t.fired) is None
    assert v.check_manifest(m_n) is None
    assert v.cascade.state.pending_generation == 2


def test_record_fired_before_the_previous_reign_is_not_attestable(cfg, tmp_path):
    t = _timeline(cfg)
    t, v, m_n = _prior_reign_case(cfg, tmp_path, prev=t.fired + t.eb, prev_observed=True)
    assert v.cascade.prior_reign_anchor(t.fired) is None
    assert v.check_manifest(m_n) is None


def test_prior_reign_record_is_rejected_before_activation(cfg, tmp_path):
    t, v, m_n = _prior_reign_case(cfg, tmp_path, prev=0, prev_observed=True, active=False)
    assert v.check_manifest(m_n).startswith("era_init_mismatch")
    assert v.cascade.state.pending_generation == 0


def test_record_fired_within_the_current_reign_is_judged_by_the_current_clock(cfg, tmp_path):
    # Fired AFTER the current anchor: the DEC-CA-0050 branch never applies,
    # active or not — an unripe current clock still ignores it, a ripe one
    # stages it.
    t = _timeline(cfg)
    for active in (False, True):
        unripe_start = t.fired - t.eb
        _, v, m_n = _prior_reign_case(cfg, tmp_path, prev=0, prev_observed=True, active=active,
                                      reign_start=unripe_start)
        assert not v.cascade.fired_in_prior_reign(t.fired)
        assert v.check_manifest(m_n).startswith("era_init_mismatch")
        ripe_start = t.fired - 6 * t.eb
        _, v, m_n = _prior_reign_case(cfg, tmp_path, prev=None, prev_observed=False,
                                      active=active, reign_start=ripe_start)
        assert v.check_manifest(m_n) is None and v.cascade.state.pending_generation == 2


def test_prior_reign_record_is_reign_scoped_to_its_own_reign(cfg, tmp_path):
    # Attested on the previous clock, the members' provenance is scoped to the
    # PREVIOUS reign: a report benched before that reign began fails.
    t = _timeline(cfg)
    prev = t.fired - 6 * t.eb
    _, v, m_n = _prior_reign_case(cfg, tmp_path, prev=prev, prev_observed=True,
                                  store_fired=None)
    assert v.check_manifest(m_n) is None           # report at fired >= prev: in scope
    stale = _promo_store([(M2A, 0.4)], generation=2, effective_era=t.era_n.index,
                         fired_block=t.fired)
    rec_key = bench_report_key("r9")
    rep = load_bench_report(stale.texts[rec_key])
    stale.texts[rec_key] = dump_bench_report(replace(rep, created_block=prev - 1))
    _, v2, _ = _prior_reign_case(cfg, tmp_path, prev=prev, prev_observed=True)
    v2.bench_report_store = stale
    v2._bench_report_cache.clear()
    assert v2.check_manifest(m_n).startswith("era_init_mismatch")
    assert v2.cascade.state.pending_generation == 0


def test_the_new_reigns_log_never_tightens_a_prior_reign_records_floor(cfg, tmp_path):
    # The new king's reign log may hold a checkpoint far better than the
    # promoted members — it was never their candidate, so it stays out of the
    # floor when the record fired under the previous reign.
    t, v, m_n = _prior_reign_case(cfg, tmp_path, prev=0, prev_observed=True)
    v.cascade.record_checkpoint("cascade/new-reign@x", gifteval_crps=0.1,
                                gifteval_mase=0.1, boom_crps=0.1, boom_mase=0.1,
                                time_crps=0.1, time_mase=0.1, now=0.0)
    record = v._fetch_latest_promotion_record()
    # the pre-DEC-CA-0050 floor (current log included) would refuse it …
    assert v._verify_members(record, enforce_reign_scope=False, carried_exempt=True
                             ).startswith("warm_start_member_below_floor")
    # … the prior-reign path does not, end to end
    assert v.check_manifest(m_n) is None
    assert v.cascade.state.pending_generation == 2


# ── late re-stage: the stuck network recovers on its own ─────────────────────

def _lost_state_case(cfg, tmp_path, *, active):
    """The live state after the wedge: ledger gen 1, nothing staged, no
    previous anchor (the old crown erased both), reign anchored at the
    dethrone, observed."""
    t = _timeline(cfg)
    armed = _carry_on(t.armed, _rollover(t.armed)) if active else t.armed
    cas = _cas(armed, tmp_path, king_hotkey="chal_hk", reign_start_block=t.dethrone_block)
    store = _promo_store([(M2A, 0.4)], generation=2, effective_era=t.era_n.index,
                         fired_block=t.fired)
    return t, armed, _runner(armed, cascade=cas, store=store)


def test_late_restage_installs_at_the_next_settlement_when_active(cfg, tmp_path):
    t, armed, v = _lost_state_case(cfg, tmp_path, active=False)
    later = t.n_block + t.eb                         # era n already running
    assert settlement_era(armed.round, later).index == t.era_n.index
    m_late = _manifest(armed, later, warm_start=M2A, generation=2, member_index=0)
    assert v.check_manifest(m_late).startswith("era_init_mismatch")      # still wedged

    t, armed, v = _lost_state_case(cfg, tmp_path, active=True)
    m_late = _manifest(armed, later, warm_start=M2A, generation=2, member_index=0)
    assert v.check_manifest(m_late) is None
    assert v.cascade.state.pending_generation == 2 and v.cascade.state.generation == 1
    assert v.process_settlement(m_late, windows=[], base_seed=later) is not None
    assert v.cascade.state.generation == 2 and v.cascade.state.pending_generation == 0
    # the next era rotates over the installed set (k = 1)
    nxt = t.n_block + t.length
    era_next = settlement_era(armed.round, nxt)
    assert era_next.index == t.era_n.index + 1
    v.state = replace(v.state, last_handled_round_id=None)
    assert v.check_manifest(_manifest(armed, nxt, warm_start=M2A, generation=2,
                                      member_index=0)) is None


def test_late_restage_first_seen_in_a_later_era_still_installs(cfg, tmp_path):
    t, armed, v = _lost_state_case(cfg, tmp_path, active=True)
    nxt = t.n_block + t.length                     # era n+1 > effective_era n
    m = _manifest(armed, nxt, warm_start=M2A, generation=2, member_index=0)
    assert v.check_manifest(m) is None
    assert v.process_settlement(m, windows=[], base_seed=nxt) is not None
    assert v.cascade.state.generation == 2


# ── config ──────────────────────────────────────────────────────────────────

def test_config_key_parses_and_round_trips(tmp_path):
    c = load_unpinned(DEFAULT_CHAIN_TOML)
    assert c.scoring.promotion_carry_from_block == 0
    bare = re.sub(r"^promotion_carry_from_block\s*=.*$", "", strip_pins(DEFAULT_CHAIN_TOML.read_text()),
                  flags=re.M)
    p = tmp_path / "chain.toml"
    p.write_text(bare)
    assert load_chain_config(p).scoring.promotion_carry_from_block == 0
    p.write_text(bare.replace("\n[scoring]\n", "\n[scoring]\npromotion_carry_from_block = "
                              "9281700\n", 1))
    assert load_chain_config(p).scoring.promotion_carry_from_block == 9281700
    p.write_text(bare.replace("\n[scoring]\n", "\n[scoring]\npromotion_carry_from_block = -1\n",
                              1))
    with pytest.raises(ValueError, match="promotion_carry_from_block"):
        load_chain_config(p)


def test_gate_helper():
    sc = SimpleNamespace(promotion_carry_from_block=0)
    assert not promotion_carry_active(sc, 10 ** 8)
    sc = SimpleNamespace(promotion_carry_from_block=9281700)
    assert not promotion_carry_active(sc, None)
    assert not promotion_carry_active(sc, 9281700 - 900)
    assert promotion_carry_active(sc, 9281700) and promotion_carry_active(sc, 9282600)


# ── the on-chain decision ───────────────────────────────────────────────────

@pytest.fixture
def acfg():
    """The shipped chain.toml with the forfeiture and margin-v2 disarmed and
    the DEC-CA-0043 rollover typed (the deployed end state), so the promotion
    carry is the only feature decided on chain."""
    c = load_unpinned(REPO / "chain.toml")
    return replace(
        c,
        round=replace(c.round, rolling_from_block=B0, epoch_blocks=900, epoch_blocks_prev=3600,
                      epoch_activation_block=B0),
        scoring=replace(c.scoring, forfeit_hotkeys=(), forfeit_from_block=0,
                        forfeit_successor_hotkey="", win_margin_start_v2=0.0,
                        win_margin_end_v2=0.0, margin_warmup_blocks_v2=0,
                        margin_v2_from_block=0, era_king_from_block=B0,
                        tenure_blocks_from_block=B0, cohort_maxt_increment_from_block=B0))


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


def test_feature_presence_and_typed_override(acfg):
    spec = A.promotion_carry_feature(acfg)
    assert spec is not None and spec.name == PC and spec.align == "boundary"
    assert spec.decided_on_chain
    off = replace(acfg, scoring=replace(acfg.scoring, cascade_enabled=False))
    assert A.promotion_carry_feature(off) is None
    no_act = replace(acfg, activation=replace(acfg.activation, feature=""))
    assert A.promotion_carry_feature(no_act) is None
    typed = _carry_on(acfg, B0 + 3600)
    assert A.promotion_carry_feature(typed).typed_block == B0 + 3600
    assert not A.promotion_carry_feature(typed).decided_on_chain


def test_note_appends_the_segment_last_and_keeps_the_earlier_bytes(acfg):
    rec = A.ActivationRecord()
    assert A.own_signal_payload(acfg, rec) == f"cascade-ready:1:{PRIMARY}:0:0:{PC}:0:0"
    locked = A.ActivationRecord(feature=PC, lock_block=B0 + 900, activation_block=B0 + 1800)
    note = A.own_signal_payload(acfg, rec, promotion_carry_record=locked)
    assert note == f"cascade-ready:1:{PRIMARY}:0:0:{PC}:{B0 + 900}:{B0 + 1800}"
    assert A.signal_for(note, PC) == A.ReadySignal(PC, B0 + 900, B0 + 1800)
    assert A.parse_signal(note) == A.ReadySignal(PRIMARY)          # primary unchanged
    # without the feature the note is exactly today's bytes
    off = replace(acfg, scoring=replace(acfg.scoring, cascade_enabled=False))
    assert A.own_signal_payload(off, rec) is None                   # typed rollover: inert
    assert A.own_signal_payload(_carry_on(acfg, B0 + 900), rec) is None


def test_a_note_that_would_overflow_128_bytes_drops_only_the_new_segment():
    # The deployed fleet's note today: all three earlier features locked in,
    # 7-digit blocks — 116 bytes; with a locked promotion-carry segment it
    # would be 148 and the write would be refused.
    c = load_unpinned(REPO / "chain.toml")
    c = replace(c, scoring=replace(c.scoring, forfeit_hotkeys=("K",),
                                   forfeit_successor_hotkey="S"))
    f, m = A.forfeit_feature(c).name, A.margin_v2_feature(c).name
    lp = A.ActivationRecord(feature=PRIMARY, lock_block=9133200, activation_block=9136800)
    lf = A.ActivationRecord(feature=f, lock_block=9165600, activation_block=9166500)
    lm = A.ActivationRecord(feature=m, lock_block=9190800, activation_block=9194400)
    lc = A.ActivationRecord(feature=PC, lock_block=9281700, activation_block=9282600)
    today = A.format_signals(A.ReadySignal(PRIMARY, 9133200, 9136800),
                             A.ReadySignal(f, 9165600, 9166500),
                             A.ReadySignal(m, 9190800, 9194400))
    assert len(today) == 116
    assert A.own_signal_payload(c, lp, lf, lm) == today
    assert A.own_signal_payload(c, lp, lf, lm, lc) == today
    # typing the decided blocks frees the room (a typed feature has no segment)
    typed = replace(c, scoring=replace(c.scoring, forfeit_from_block=9166500,
                                       margin_v2_from_block=9194400))
    note = A.own_signal_payload(typed, lp, lf, lm, lc)
    assert note == f"cascade-ready:1:{PRIMARY}:9133200:9136800:{PC}:9281700:9282600"
    assert len(note.encode()) <= A.NOTE_MAX_BYTES


def test_resolver_locks_in_at_the_next_settlement_boundary(acfg):
    spec = A.promotion_carry_feature(acfg)
    note = A.format_signals(A.ReadySignal(PRIMARY), A.ReadySignal(PC))
    lock = B0 + 4500                                   # a mid-era settlement boundary
    chain = FakeChain(_fleet(60, 40), {"v1": note}, block=lock + 5)
    res = A.resolve_activation(acfg, chain, now_block=lock + 5, record=A.ActivationRecord(),
                               feature=spec)
    assert res.record.locked and res.record.lock_block == lock
    assert res.record.activation_block == lock + 900   # next settlement, not next era
    below = FakeChain(_fleet(40, 60), {"v1": note}, block=lock + 5)
    assert not A.resolve_activation(acfg, below, now_block=lock + 5,
                                    record=A.ActivationRecord(), feature=spec).record.locked
    # late joiner adopts the fleet's note; an era-aligned pair is inadmissible
    good = A.format_signals(A.ReadySignal(PRIMARY), A.ReadySignal(PC, lock, lock + 900))
    assert A.agreed_activation(PC, _fleet(60, 40), {"v1": good}, threshold=0.51,
                               block=lock + 100, round_cfg=acfg.round) == (lock, lock + 900)
    bad = A.format_signals(A.ReadySignal(PRIMARY), A.ReadySignal(PC, lock, B0 + 7200))
    assert A.agreed_activation(PC, _fleet(60, 40), {"v1": bad}, threshold=0.51,
                               block=lock + 100, round_cfg=acfg.round) is None


def test_apply_is_one_way_boundary_checked_and_respects_the_typed_override(acfg):
    c1 = A.apply_promotion_carry_activation(acfg, B0 + 5400)
    assert c1.scoring.promotion_carry_from_block == B0 + 5400
    assert A.resolved_promotion_carry_block(c1) == B0 + 5400
    assert A.configured_promotion_carry_block(c1) == 0
    assert A.apply_promotion_carry_activation(c1, B0 + 5400) is c1
    with pytest.raises(ValueError, match="one-way"):
        A.apply_promotion_carry_activation(c1, B0 + 6300)
    with pytest.raises(ValueError, match="settlement boundary"):
        A.apply_promotion_carry_activation(acfg, B0 + 5400 + 17)
    typed = _carry_on(acfg, B0 + 900)
    assert A.apply_promotion_carry_activation(typed, B0 + 5400) is typed


def test_validator_runner_decides_on_chain_persists_restores_and_stamps(acfg, tmp_path):
    from cascade.validator.loop import ValidatorRunner

    runner = ValidatorRunner(cfg=acfg, activation_store=A.ActivationStore(tmp_path / "act.json"))
    lock = B0 + 4500
    chain = FakeChain(_fleet(60, 40), {}, block=lock + 1, hotkey="v1")
    runner._activation_startup(chain)
    assert chain.written[0] == f"cascade-ready:1:{PRIMARY}:0:0:{PC}:0:0"
    assert chain.written[-1] == f"cascade-ready:1:{PRIMARY}:0:0:{PC}:{lock}:{lock + 900}"
    assert runner._promotion_carry.locked
    assert runner.cfg.scoring.promotion_carry_from_block == lock + 900
    assert runner.promotion_carry_block == lock + 900
    assert not runner._promotion_carry_active(lock)
    assert runner._promotion_carry_active(lock + 900)
    saved = A.ActivationStore(tmp_path / "activation_promotion_carry_state.json").load()
    assert saved.activation_block == lock + 900
    again = ValidatorRunner(cfg=acfg, activation_store=A.ActivationStore(tmp_path / "act.json"))
    dead = SimpleNamespace(current_block=lambda: (_ for _ in ()).throw(RuntimeError("down")),
                           hotkey_ss58=lambda: "v1")
    again._activation_startup(dead)
    assert again.cfg.scoring.promotion_carry_from_block == lock + 900
    assert "promotion_carry" in A.summary(again.cfg, A.ActivationRecord(), None)


def test_validator_runner_is_silent_on_a_typed_block(acfg, tmp_path):
    from cascade.validator.loop import ValidatorRunner

    runner = ValidatorRunner(cfg=_carry_on(acfg, B0 + 900),
                             activation_store=A.ActivationStore(tmp_path / "act.json"))
    chain = FakeChain(_fleet(60, 40), {}, block=B0 + 4501)
    runner._activation_startup(chain)
    assert chain.written == []
    assert runner.promotion_carry_block == 0
    assert runner._promotion_carry_active(B0 + 900)


# ── receipts + audit ─────────────────────────────────────────────────────────

def test_receipt_field_is_drop_when_default_and_round_trips():
    from cascade.shared import receipt as R

    base = R.RoundReceipt(round_id="1", status="rejected", epoch_start_block=B0,
                          epoch_block_hash="0x", base_seed=1, generation_seed=1, training_seed=1,
                          manifest={}, participants=(), validator_hotkey="v1", reject_reason="x")
    assert b"promotion_carry_block" not in base.canonical_body()
    stamped = replace(base, promotion_carry_block=B0 + 5400)
    assert (b'"promotion_carry_block":' + str(B0 + 5400).encode()) in stamped.canonical_body()
    assert R.load_receipt(R.dump_receipt(stamped)).promotion_carry_block == B0 + 5400
    assert R.load_receipt(R.dump_receipt(base)).promotion_carry_block == 0


def test_audit_check_replays_the_stamp(acfg):
    from cascade.audit.checks import check_promotion_carry, run_tier0

    gate = B0 + 5400
    pre = SimpleNamespace(activation_block=0, margin_v2_block=0, promotion_carry_block=0)
    post = SimpleNamespace(activation_block=0, margin_v2_block=0, promotion_carry_block=gate)
    assert A.apply_receipt_activation(acfg, pre) is acfg
    replayed = A.apply_receipt_activation(acfg, post)
    assert replayed.scoring.promotion_carry_from_block == gate
    assert check_promotion_carry(pre, acfg).status == "PASS"
    assert check_promotion_carry(post, replayed).status == "WARN"            # no chain
    note = A.format_signals(A.ReadySignal(PRIMARY), A.ReadySignal(PC, gate - 900, gate))
    assert check_promotion_carry(post, replayed, FakeChain(
        _fleet(60, 40), {"v1": note}, block=gate + 10)).status == "PASS"
    other = A.format_signals(A.ReadySignal(PRIMARY), A.ReadySignal(PC, gate, gate + 900))
    assert check_promotion_carry(post, replayed, FakeChain(
        _fleet(60, 40), {"v1": other}, block=gate + 10)).status == "FAIL"
    assert check_promotion_carry(post, acfg).status == "FAIL"                # not applied
    off = replace(acfg, scoring=replace(acfg.scoring, cascade_enabled=False))
    assert check_promotion_carry(post, off).status == "FAIL"
    assert check_promotion_carry(post, _carry_on(acfg, gate)).status == "PASS"
    assert check_promotion_carry(post, _carry_on(acfg, gate + 900)).status == "FAIL"
    import inspect
    assert "check_promotion_carry(receipt, cfg, client)" in inspect.getsource(run_tier0)


# ── pinned lock-ins (the 128-byte note budget) ───────────────────────────────

LIVE_NOTE = ("cascade-ready:1:rolling-era-king:9133200:9136800:forfeit-42b1a486:9165600:"
             "9166500:margin-v2-d30eb443:9190800:9194400")          # mainnet, old release
PINS = {"rolling-era-king": (9133200, 9136800), "forfeit-42b1a486": (9165600, 9166500),
        "margin-v2-d30eb443": (9190800, 9194400)}


def _pinned():
    return load_chain_config(REPO / "chain.toml")


def _tallied():
    """The unpinned release config after the three lock-ins applied the way a
    validator that tallied them applies them (rollover first)."""
    c = load_unpinned(REPO / "chain.toml")
    c = A.apply_activation(c, 9136800)
    c = A.apply_forfeit_activation(c, 9166500)
    return A.apply_margin_v2_activation(c, 9194400)


def _diff(a, b, path="cfg"):
    import dataclasses
    out = []
    if dataclasses.is_dataclass(a):
        for f in dataclasses.fields(a):
            if f.name in ("raw", "pinned"):
                continue
            out += _diff(getattr(a, f.name), getattr(b, f.name), f"{path}.{f.name}")
    elif a != b:
        out.append((path, a, b))
    return out


def test_shipped_pins_name_the_live_lock_ins():
    c = _pinned()
    assert {f: (lk, ac) for f, lk, ac in c.activation.pinned} == PINS
    assert A.forfeit_feature(c).name == "forfeit-42b1a486"
    assert A.margin_v2_feature(c).name == "margin-v2-d30eb443"
    for spec in (A.primary_feature(c), A.forfeit_feature(c), A.margin_v2_feature(c)):
        assert spec.pinned and spec.decided_on_chain and not spec.signals
    assert not A.promotion_carry_feature(c).pinned and A.promotion_carry_feature(c).signals


def test_pinned_config_is_field_for_field_the_tallied_lock_in():
    pinned, tallied = _pinned(), _tallied()
    assert _diff(pinned, tallied) == []
    assert (pinned.activation.resolved_block, pinned.activation.resolved_forfeit_block,
            pinned.activation.resolved_margin_v2_block) == (9136800, 9166500, 9194400)
    assert (A.configured_rollover(pinned), A.configured_forfeit_block(pinned),
            A.configured_margin_v2_block(pinned)) == (0, 0, 0)          # not "typed"
    from cascade.shared.config import effective_epoch_blocks
    from cascade.shared.era import era_king_active
    for b in (9133200, 9135900, 9136800, 9137700, 9165600, 9166500, 9167400, 9190800,
              9193500, 9194400, 9195300, 9277200, 9280800):
        assert pinned.koth_params(block=b) == tallied.koth_params(block=b), b
        assert effective_epoch_blocks(pinned.round, b) == effective_epoch_blocks(tallied.round, b)
        assert era_king_active(pinned.scoring, b) == era_king_active(tallied.scoring, b)


def test_pinned_receipt_stamps_and_audit_match_the_tallied_lock_in(tmp_path):
    from cascade.audit.checks import check_activation, check_margin_v2
    from cascade.shared import receipt as R
    from cascade.validator.loop import ValidatorRunner

    pinned, tallied = _pinned(), _tallied()
    rp = ValidatorRunner(cfg=pinned)
    rt = ValidatorRunner(cfg=tallied)
    assert (rp.activation_block, rp.margin_v2_block, rp.promotion_carry_block) == (
        rt.activation_block, rt.margin_v2_block, rt.promotion_carry_block) == (
        9136800, 9194400, 0)
    base = R.RoundReceipt(round_id="1", status="rejected", epoch_start_block=9280800,
                          epoch_block_hash="0x", base_seed=1, generation_seed=1,
                          training_seed=1, manifest={}, participants=(),
                          validator_hotkey="v1", reject_reason="x")
    bodies = {replace(base, activation_block=r.activation_block,
                      margin_v2_block=r.margin_v2_block).canonical_body() for r in (rp, rt)}
    assert len(bodies) == 1
    receipt = replace(base, activation_block=9136800, margin_v2_block=9194400)
    # the pinned auditor replays exactly the tallied config, and the pin is
    # the agreement (the notes no longer carry these lock-ins)
    assert A.apply_receipt_activation(pinned, receipt) is pinned
    replayed = A.apply_receipt_activation(load_unpinned(REPO / "chain.toml"), receipt)
    # (receipts carry no forfeiture stamp — the forfeiture never replayed from
    # a receipt, before or after pinning)
    assert {d[0] for d in _diff(pinned, replayed)} == {
        "cfg.scoring.forfeit_from_block", "cfg.activation.resolved_forfeit_block"}
    assert check_activation(receipt, pinned).status == "PASS"
    assert check_margin_v2(receipt, pinned).status == "PASS"
    # a stamp that disagrees with the pin still fails
    assert check_activation(replace(receipt, activation_block=9140400), pinned).status == "FAIL"
    assert check_margin_v2(replace(receipt, margin_v2_block=9198000), pinned).status == "FAIL"
    # a pre-rollover receipt (no stamps) audits exactly as before
    assert check_activation(base, pinned).status == "PASS"


def test_pinned_validator_posts_a_note_that_fits_before_and_after_lock_in(tmp_path):
    from cascade.validator.loop import ValidatorRunner

    store = A.ActivationStore(tmp_path / "activation_state.json")
    # the live box's persisted tallied records survive untouched
    live = {
        "activation_state.json": A.ActivationRecord("rolling-era-king", 9133200, 9136800,
                                                    9133200, "tally"),
        "activation_forfeit_state.json": A.ActivationRecord("forfeit-42b1a486", 9165600,
                                                            9166500, 9165600, "signals"),
        "activation_margin_v2_state.json": A.ActivationRecord("margin-v2-d30eb443", 9190800,
                                                              9194400, 9190800, "tally"),
    }
    for name, rec in live.items():
        A.ActivationStore(tmp_path / name).save(rec)
    before = {n: (tmp_path / n).read_bytes() for n in live}
    runner = ValidatorRunner(cfg=_pinned(), activation_store=store)
    lock = 9281700
    assert lock % 900 == 0
    chain = FakeChain(_fleet(60, 40), {"v2": LIVE_NOTE}, block=lock + 1, hotkey="v1")
    runner._activation_startup(chain)
    first, last = chain.written[0], chain.written[-1]
    assert first == f"cascade-ready:1:{PRIMARY}:0:0:{PC}:0:0"
    assert len(first.encode()) == 56
    assert last == f"cascade-ready:1:{PRIMARY}:0:0:{PC}:{lock}:{lock + 900}"
    assert len(last.encode()) == 68 <= A.NOTE_MAX_BYTES
    assert len(f"cascade-ready:1:{PRIMARY}:0:0:{PC}:99999000:99999900") == 70
    assert runner.cfg.scoring.promotion_carry_from_block == lock + 900
    assert runner.promotion_carry_block == lock + 900
    assert {n: (tmp_path / n).read_bytes() for n in live} == before
    assert runner._activation == live["activation_state.json"]
    assert runner._margin_v2 == live["activation_margin_v2_state.json"]
    assert runner._forfeit == live["activation_forfeit_state.json"]


def test_mixed_fleet_old_notes_never_count_for_promotion_carry():
    c = _pinned()
    spec = A.promotion_carry_feature(c)
    new = A.own_signal_payload(c, A.ActivationRecord())
    signals = {"v1": new, "v2": LIVE_NOTE, "v3": LIVE_NOTE}
    t = A.tally(PC, _fleet(45, 45, 10), signals, threshold=0.51, block=9281700)
    assert t.signed == ("v1",) and not t.locked                      # 45 %: old notes not signed
    t = A.tally(PC, _fleet(60, 30, 10), signals, threshold=0.51, block=9281700)
    assert t.locked
    chain = FakeChain(_fleet(45, 45, 10), signals, block=9281701)
    assert not A.resolve_activation(c, chain, now_block=9281701, record=A.ActivationRecord(),
                                    feature=spec).record.locked
    # an old-release node still reads both notes and its own features' segments
    assert A.parse_signals(new) is not None and A.parse_signals(LIVE_NOTE) is not None
    assert A.signal_for(LIVE_NOTE, "margin-v2-d30eb443") == A.ReadySignal(
        "margin-v2-d30eb443", 9190800, 9194400)
    assert A.signal_for(new, "margin-v2-d30eb443") is None


def test_pinned_features_never_re_tally_or_read_the_chain():
    class NoChain:
        def __getattr__(self, name):
            raise AssertionError(f"pinned resolver touched the chain ({name})")

    c = _pinned()
    for spec in (A.primary_feature(c), A.forfeit_feature(c), A.margin_v2_feature(c)):
        lock, act = PINS[spec.name]
        res = A.resolve_activation(c, NoChain(), now_block=9281701, record=A.ActivationRecord(),
                                   feature=spec)
        assert (res.record.lock_block, res.record.activation_block, res.record.source) == (
            lock, act, "pinned")
        kept = A.ActivationRecord(spec.name, lock, act, lock, "tally")
        res = A.resolve_activation(c, NoChain(), now_block=9281701, record=kept, feature=spec)
        assert res.record == kept and not res.changed
        assert A.record_for(c, A.ActivationRecord(), spec).activation_block == act
    # the primary head "rolling-era-king:0:0" in every new note never relocks
    note = A.own_signal_payload(c, A.ActivationRecord())
    res = A.resolve_activation(c, NoChain(), now_block=9281701, record=A.ActivationRecord())
    assert A.signal_for(note, PRIMARY) == A.ReadySignal(PRIMARY)
    assert res.record.activation_block == 9136800


@pytest.mark.parametrize("edit,match", [
    (lambda t: t.replace("lock_block       = 9133200", "lock_block       = 9129600"),
     "not a pair"),
    (lambda t: t.replace("activation_block = 9194400", "activation_block = 9193500"),
     "not a pair"),
    (lambda t: t.replace('feature          = "forfeit-42b1a486"',
                         'feature          = "forfeit-deadbeef"'), "stale pin"),
    (lambda t: re.sub(r"^margin_v2_from_block\s*=.*$", "margin_v2_from_block = 9194400", t,
                      flags=re.M), "pin OR type"),
    (lambda t: re.sub(r"^rolling_from_block\s*=.*$", "rolling_from_block = 9136800", t,
                      flags=re.M), "rollover"),       # the loader's own check refuses first
    (lambda t: re.sub(r'^feature = "rolling-era-king"$', 'feature = ""', t, flags=re.M),
     "activation"),
    (lambda t: t.replace("lock_block       = 9165600", "lock_block       = 9166500"),
     "lock_block < activation_block"),
    (lambda t: t + '\n[[activation.pinned]]\nfeature = "rolling-era-king"\nlock_block = 1\n'
                   'activation_block = 2\n', "twice"),
    (lambda t: t + '\n[[activation.pinned]]\nfeature = "x"\nlock_block = 1\n'
                   'activation_block = 2\nnote = 1\n', "unknown key"),
])
def test_loader_refuses_a_bad_pin(tmp_path, edit, match):
    p = tmp_path / "chain.toml"
    p.write_text(edit((REPO / "chain.toml").read_text()))
    with pytest.raises(ValueError, match=match):
        load_chain_config(p)


def test_strip_pins_leaves_the_rest_of_the_file_alone():
    text = (REPO / "chain.toml").read_text()
    stripped = strip_pins(text)
    assert "[[activation.pinned]]" not in stripped
    assert len(text.splitlines()) - len(stripped.splitlines()) == 12
    assert load_unpinned(REPO / "chain.toml").activation.pinned == ()


def test_a_pin_and_a_typed_block_for_the_same_feature_is_refused():
    t = _tallied()
    typed = replace(t, activation=replace(t.activation, resolved_block=0,
                                          pinned=(("rolling-era-king", 9133200, 9136800),)))
    assert A.configured_rollover(typed) == 9136800
    with pytest.raises(ValueError, match="pin OR type"):
        A.apply_pinned_activations(typed)
