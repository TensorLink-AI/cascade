"""DEC-CA-0067 rule 2 as amended ("the v2b switch keeps the running seed
epoch", ``image-decoupled/3``): the applied ``[v2] image_decoupled_from_block``
era changes the CHECK a v2 settlement's entries pass and nothing else — the
seed epoch, its seeds, its init and its king pointer continue across it, so
the carried king and the finished legs are judged under the decoupled check
and nothing retrains. The vote still takes effect two eras after the lock-in
era (a promotion's notice). ``/2`` made that era open a seed epoch (new seeds,
a forced king retrain) and was never signalled.
"""
from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import cascade.shared.activation as A
from cascade.shared.config import effective_epoch_blocks, load_chain_config
from cascade.shared.era import (
    EraSpec,
    era_for_block,
    era_length_blocks,
    v2_era_seed_block,
    v2_first_era,
    v2_first_warm_era,
    v2_init_epoch_start,
    v2_seed_epoch_start,
    v2b_switch_era,
)

ROOT = Path(__file__).resolve().parents[2]

MAIN, TEST = "v2b-69630f92", "v2b-c1ffa858"            # rule /3, quantile head on
MAIN_NOQ, TEST_NOQ = "v2b-1abeaeb4", "v2b-c31c2c8e"    # rule /3, knob missing/false


@pytest.fixture
def base():
    """Mainnet chain.toml with v2 at a fixed era (ladder on, as shipped)."""
    cfg = load_chain_config(ROOT / "chain.toml")
    era = era_length_blocks(cfg.round, 0)
    gate = 10_000 * era
    cfg = replace(cfg, v2=replace(cfg.v2, from_block=gate, image_decoupled_from_block=0),
                  training=replace(cfg.training, v2_from_block=gate))   # the vote still open
    return cfg, era


def _switch(cfg, era_index: int, era: int):
    return replace(cfg, v2=replace(cfg.v2, image_decoupled_from_block=era_index * era))


# ── the pure rule ────────────────────────────────────────────────────────────


def test_switch_era_keeps_the_running_seed_epoch(base):
    cfg, era = base
    first, warm = v2_first_era(cfg), v2_first_warm_era(cfg)
    sw = warm + 6
    on = _switch(cfg, sw, era)
    assert v2b_switch_era(on) == sw and v2b_switch_era(cfg) is None
    for k in (-1, 0, 1, 5):
        # the switch opens no epoch: seeds, init epoch and seed block continue
        assert v2_seed_epoch_start(on, sw + k, []) == first
        assert v2_init_epoch_start(on, sw + k, []) == first
        e = era_for_block(on.round, (sw + k) * era)
        assert v2_era_seed_block(on, e, []) == (first - 1) * era
    # inside the random-init window too
    early = _switch(cfg, first + 3, era)
    assert v2_seed_epoch_start(early, first + 5, []) == first
    assert v2_init_epoch_start(early, first + 5, []) == first


def test_promotions_open_epochs_exactly_as_without_a_switch(base):
    cfg, era = base
    sw = v2_first_warm_era(cfg) + 6
    on, w = _switch(cfg, sw, era), sw + 4
    assert v2_seed_epoch_start(on, w - 1, [w]) == v2_first_era(cfg)
    assert v2_seed_epoch_start(on, w + 2, [w]) == w == v2_init_epoch_start(on, w + 2, [w])
    # a promotion from before the switch keeps its epoch across it
    w0 = sw - 3
    for idx in (sw - 1, sw, sw + 1):
        assert v2_seed_epoch_start(on, idx, [w0]) == w0 == v2_init_epoch_start(on, idx, [w0])
    # a promotion landing ON the switch era is just a promotion
    for k in (0, 3):
        assert v2_seed_epoch_start(on, sw + k, [sw]) == sw == v2_init_epoch_start(on, sw + k, [sw])
        e = era_for_block(on.round, (sw + k) * era)
        assert v2_era_seed_block(on, e, [sw]) == (sw - 1) * era


def test_any_switch_is_exactly_the_warm_start_rule(base):
    cfg, era = base
    first, warm = v2_first_era(cfg), v2_first_warm_era(cfg)
    for c in (cfg, _switch(cfg, first, era), _switch(cfg, first + 2, era),
              _switch(cfg, warm + 6, era)):
        for warms in ([], [warm], [warm, warm + 7]):
            for idx in range(first - 2, warm + 12):
                want = (None if idx < first else
                        max([first] + [w for w in warms if first < w <= idx]))
                assert v2_seed_epoch_start(c, idx, warms) == want
                assert v2_init_epoch_start(c, idx, warms) == want


# ── the vote: two eras of notice, a fresh feature name ───────────────────────


def test_feature_name_is_the_rule_3_bundle():
    assert A.V2B_RULE == "image-decoupled/3"
    for n, want in ((4000, MAIN_NOQ), (300, TEST_NOQ)):
        body = f"image-decoupled/3|train_image_digest,continuation_protocol|eval_n_windows_after={n}"
        assert want == "v2b-" + hashlib.sha256(body.encode()).hexdigest()[:8]
        assert A.v2b_feature_name(n) == want == A.v2b_feature_name(n, False)
    for n, want in ((4000, MAIN), (300, TEST)):
        body = (f"image-decoupled/3|train_image_digest,continuation_protocol|"
                f"eval_n_windows_after={n}|quantile_head_after=1")
        assert want == "v2b-" + hashlib.sha256(body.encode()).hexdigest()[:8]
        assert A.v2b_feature_name(n, True) == want != A.v2b_feature_name(n)
    # the /1 and /2 names are retired (never signalled)
    for old in ("v2b-150ec0d0", "v2b-90bc5237", "v2b-b10591aa", "v2b-3a259561", "v2b-f2eefcb3"):
        assert old not in {A.v2b_feature_name(4000), A.v2b_feature_name(4000, True),
                           A.v2b_feature_name(300), A.v2b_feature_name(300, True)}
    main = load_chain_config(ROOT / "chain.toml")
    test = load_chain_config(ROOT / "chain.testnet.toml")
    assert main.v2.quantile_head_after and test.v2.quantile_head_after
    assert main.v2.image_decoupled_feature == MAIN == A.v2b_feature_name_for(main.v2)
    assert test.v2.image_decoupled_feature == TEST == A.v2b_feature_name_for(test.v2)


def test_lock_in_takes_effect_two_eras_after_its_era(base):
    cfg, era = base
    rc = cfg.round
    eb = int(effective_epoch_blocks(rc, 100 * era))
    for lock in (100 * era, 100 * era + eb, 101 * era - eb):
        act = A.activation_block_for(rc, lock, "epoch")
        assert act == 102 * era and act % era == 0
        assert A.valid_pair(rc, lock, act, "epoch")
        # the /1 pair (next era start) is no longer admissible
        assert not A.valid_pair(rc, lock, 101 * era, "epoch")
    spec = A.image_decoupled_feature(replace(cfg, v2=replace(cfg.v2, launch_at_utc="",
                                                              from_block=10_000 * era)))
    assert spec is not None and spec.align == "epoch"


def test_other_features_keep_their_alignment(base):
    cfg, era = base
    rc = cfg.round
    eb = int(effective_epoch_blocks(rc, 100 * era))
    lock = 100 * era + eb
    assert A.activation_block_for(rc, lock) == lock + eb                       # boundary
    assert A.activation_block_for(rc, lock, "era") == 101 * era               # era
    assert A.primary_feature(cfg).align == "boundary"
    mv2 = A.margin_v2_feature(cfg)
    assert mv2 is None or mv2.align == "era"
    for spec in (A.forfeit_feature(cfg), A.prelaunch_burn_feature(cfg)):
        assert spec is None or spec.align == "boundary"


def test_restart_and_late_joiner_adopt_the_same_gated_block(base, tmp_path):
    from .test_activation import FakeChain, _v

    cfg, era = base
    rc = cfg.round
    v2cfg = replace(cfg, v2=replace(cfg.v2, launch_at_utc="", from_block=10_000 * era))
    spec = A.image_decoupled_feature(v2cfg)
    lock = 10_010 * era
    act = A.activation_block_for(rc, lock, spec.align)
    assert act == 10_012 * era
    # restart: the persisted record re-applies the same block
    store = A.ActivationStore(tmp_path / "activation_image_decoupled_state.json")
    rec = A.ActivationRecord(feature=spec.name, lock_block=lock, activation_block=act,
                             last_checked_boundary=lock, source="tally")
    store.save(rec)
    assert A.record_for(v2cfg, store.load(), spec) == rec
    armed = A.apply_image_decoupled_activation(v2cfg, act)
    assert v2b_switch_era(armed) == 10_012
    # late joiner: adopts the pair the stake majority names
    note = A.format_signals(A.ReadySignal(str(cfg.activation.feature or "x")),
                            A.ReadySignal(spec.name, lock, act))
    chain = FakeChain([_v("v1", 60), _v("v2", 40)], {"v1": note}, block=act + 5)
    chain.fail_as_of = True
    res = A.resolve_activation(v2cfg, chain, now_block=act + 5, record=A.ActivationRecord(),
                               feature=spec)
    assert (res.record.lock_block, res.record.activation_block) == (lock, act)
    # a typed override is any era start; off-grid is refused
    with pytest.raises(ValueError, match="era start"):
        A.apply_image_decoupled_activation(v2cfg, act + int(effective_epoch_blocks(rc, act)))


# ── the validator ────────────────────────────────────────────────────────────


def _validator(cfg, era, tmp_path, *, members=(), effective=0):
    from cascade.validator.cascade import CascadeController, CascadeState
    from cascade.validator.loop import ValidatorRunner
    from cascade.validator.state import ChampionState

    cfg = replace(cfg, validator=replace(cfg.validator, state_db_path=str(tmp_path / "s.json")))
    r = ValidatorRunner(cfg=cfg, state=ChampionState(king_hotkey="5King"))
    r.cascade = CascadeController(reign_days=1.0, state=CascadeState(
        king_hotkey="5King", reign_start_block=int(cfg.v2.from_block), clock_observed=True,
        generation=1 if members else 0, members=tuple(members),
        generation_effective_era=int(effective)))
    r._fetch_latest_promotion_record = lambda: None
    return r


def test_validator_seed_block_pointer_era_and_init_epoch_continue_across_the_switch(base,
                                                                                    tmp_path):
    cfg, era = base
    first = v2_first_era(cfg)
    sw = v2_first_warm_era(cfg) + 6
    r = _validator(_switch(cfg, sw, era), era, tmp_path)
    plain = _validator(cfg, era, tmp_path / "p")
    for idx in (sw - 1, sw, sw + 2):
        e = era_for_block(r.cfg.round, idx * era)
        assert r._era_seed_block(e) == (first - 1) * era == plain._era_seed_block(e)
        # one king pointer per seed epoch — the switch does not re-key it, so
        # the carried king's pointer is the one the post-switch settlements name
        assert r._pointer_era(e) == first == plain._pointer_era(e)
        assert r._epoch_start(e) == first == r._init_epoch_start(e)


def test_validator_refuses_an_init_change_mid_epoch_switch_or_not(base, tmp_path):
    from cascade.shared.manifest import TrainingManifest, format_trained_pointer

    cfg, era = base
    w = v2_first_warm_era(cfg) + 2
    sw = w + 4
    kp = format_trained_pointer("cascade/king@sha256:" + "b" * 64)
    r = _validator(_switch(cfg, sw, era), era, tmp_path, members=(kp,), effective=w)
    r.state = replace(r.state, epoch_init={"epoch": w, "init": kp})
    spec = replace(era_for_block(r.cfg.round, sw * era), generation=1, member_index=0)
    spec = replace(spec, seed_block=r._era_seed_block(spec))
    assert spec.seed_block == (w - 1) * era                       # the epoch's seed block
    m = TrainingManifest(round_id="1", created_block=sw * era + 5, contract_digest="c",
                         base_arch_digest="d", eval_dataset="e", warm_start_ckpt="merged",
                         warm_start_base=kp, warm_start_merges=[{"m": 1}], era=spec.to_json())
    why = r._check_era_init(m, EraSpec.from_json(spec.to_json()))
    assert why is not None and "only at a warm start" in why
    # a different init without merges is refused the same way
    m2 = replace(m, warm_start_ckpt=format_trained_pointer("x/y@sha256:" + "c" * 64),
                 warm_start_base="", warm_start_merges=[])
    assert "era_init_mismatch" in (r._check_era_init(m2, EraSpec.from_json(spec.to_json())) or "")
    # the running init is accepted
    m3 = replace(m, warm_start_ckpt=kp, warm_start_base="", warm_start_merges=[])
    assert r._check_era_init(m3, EraSpec.from_json(spec.to_json())) is None


def test_lane_b_acceptance_block_is_the_init_epochs(base):
    from cascade.validator.merge_acceptance import acceptance_block

    cfg, era = base
    w = v2_first_warm_era(cfg) + 2
    sw = w + 4
    e = era_for_block(cfg.round, (sw + 1) * era)
    e = replace(e, seed_block=(w - 1) * era)                      # the epoch continues
    m = SimpleNamespace(era=e.to_json())
    runner = SimpleNamespace(_init_epoch_start=lambda spec: w)
    for c in (cfg, _switch(cfg, sw, era)):
        assert acceptance_block(c, m) == w * era
        assert acceptance_block(c, m, runner) == w * era
    # a /2-era stamp (the switch era as a seed epoch) still resolves to the
    # init epoch's start through the runner
    old = SimpleNamespace(era=replace(e, seed_block=(sw - 1) * era).to_json())
    assert acceptance_block(_switch(cfg, sw, era), old, runner) == w * era


# ── the trainer ──────────────────────────────────────────────────────────────


def test_trainer_carries_the_king_across_the_switch_era(cfg, tmp_path):
    """The whole point of the amendment: no king leg, no init re-bench, the
    same seeds — the switch era is a later era of the running epoch."""
    from tests.unit.test_dec55_trainer_epochs import EpochOps, _join, _step, _tick_era, _v2
    from tests.unit.test_rolling import Clock, FakeClient, _sched

    v2cfg, v2_block, L = _v2(cfg)
    first = era_for_block(v2cfg.round, v2_block).index
    sw_block = v2_block + 2 * L
    sw = era_for_block(v2cfg.round, sw_block).index
    assert sw == first + 2
    v2cfg = replace(v2cfg, v2=replace(v2cfg.v2, image_decoupled_from_block=sw_block))
    clock = Clock()
    ops = EpochOps(tmp_path, clock)
    sched, _ = _sched(v2cfg, tmp_path, clock, ops=ops)
    client = FakeClient()
    sched.tick(client, v2_block + 5)
    _join(sched)
    _step(clock, ops, sched, client, from_block=v2_block + 5, to_block=v2_block + 6)
    init0, base0 = sched.state.current.warm_start_ckpt, sched.state.current.base_seed
    seed0 = sched.state.current.seed_block
    last = v2_block + 6
    for k in (1, 2, 3):
        last = _tick_era(clock, ops, sched, client, v2_block + k * L, L, last=last)
        cur = sched.state.current
        assert cur.epoch_start == first
    cur = sched.state.current
    assert cur.index == sw + 1 and cur.warm_start_ckpt == init0      # same init
    assert cur.seed_block == seed0 and cur.base_seed == base0         # same seeds
    assert [c[1] for c in ops.king_calls] == [first]                  # never retrained
    assert (sw - 1, sw) in ops.carried and (sw, sw + 1) in ops.carried
    assert sw not in ops.retired and sw - 1 not in ops.retired
    assert ops.init_benches == [first]                                # benched once


def test_trainer_lane_b_reuses_the_init_epochs_merged_init_at_the_switch(cfg, tmp_path,
                                                                          monkeypatch):
    from tests.unit.test_dec55_trainer_epochs import EpochOps, _v2
    from tests.unit.test_rolling import Clock, FakeClient, _sched

    v2cfg, v2_block, L = _v2(cfg)
    first = era_for_block(v2cfg.round, v2_block).index
    w, sw = first + 3, first + 5
    v2cfg = replace(v2cfg, v2=replace(v2cfg.v2,
                                      image_decoupled_from_block=v2_block + (sw - first) * L))

    class Drv:
        """Mirrors ``LaneBDriver.init_for``: building persists the merged init
        per era; ``apply_pending=False`` READS the one built for that era."""

        def __init__(self):
            self.calls, self.built = [], {}

        def init_for(self, era_index, ckpt, work_dir, apply_pending=True, era_start=None):
            self.calls.append((era_index, apply_pending))
            key = str(era_index)
            if key in self.built:
                return self.built[key]
            if not apply_pending:
                return ckpt, []
            self.built[key] = (f"merged-{era_index}", [{"m": 1}])
            return self.built[key]

    clock = Clock()
    sched, _ = _sched(v2cfg, tmp_path, clock, ops=EpochOps(tmp_path, clock))
    sched.lane_b = Drv()
    monkeypatch.setattr(type(v2cfg.lane_b), "active", lambda self, b: True)
    sched.state.generations = {"1": {"members": [["M", "s"]], "effective_era": w}}
    at_w = sched._new_era_state(FakeClient(), era_for_block(v2cfg.round, v2_block + 3 * L))
    at_sw = sched._new_era_state(FakeClient(), era_for_block(v2cfg.round, v2_block + 5 * L))
    assert at_w.warm_start_ckpt == f"merged-{w}" and at_w.warm_start_base == "M"
    # the switch era is a later era of the warm-start epoch: same epoch, same
    # seeds, the MERGED init — no merge pending since applied
    assert at_sw.epoch_start == w and at_sw.base_seed == at_w.base_seed
    assert at_sw.seed_block == at_w.seed_block
    assert at_sw.warm_start_ckpt == f"merged-{w}" and at_sw.warm_start_base == "M"
    assert at_sw.warm_start_merges == at_w.warm_start_merges == [{"m": 1}]
    assert sched.lane_b.calls == [(w, True), (w, False)]


def test_trainer_keeps_a_prepared_era_across_the_switch(cfg, tmp_path):
    from tests.unit.test_dec55_trainer_epochs import EpochOps, _v2
    from tests.unit.test_rolling import Clock, FakeClient, _sched

    v2cfg, v2_block, L = _v2(cfg)
    first = era_for_block(v2cfg.round, v2_block).index
    clock = Clock()
    ops = EpochOps(tmp_path, clock)
    sched, _ = _sched(v2cfg, tmp_path, clock, ops=ops)
    nxt = sched._new_era_state(FakeClient(), era_for_block(v2cfg.round, v2_block + 2 * L))
    assert nxt.epoch_start == first
    sched.state.next = nxt
    sched.update_config(v2cfg)                                  # nothing changed: kept
    assert sched.state.next is nxt
    on = replace(v2cfg, v2=replace(v2cfg.v2, image_decoupled_from_block=v2_block + 2 * L))
    sched.update_config(on)                                     # the switch moves no epoch
    assert sched.state.next is nxt and sched.state.next.epoch_start == first
    assert ops.retired == []
    sched._reconcile_prepared_epoch()                           # what every tick runs first
    assert sched.state.next is nxt


def test_restart_keeps_a_persisted_switch_era_and_its_carried_king(cfg, tmp_path):
    """A typed block (or a late resolve) arrives with a RESTART — the
    scheduler is built from persisted state whose prepared ``next`` (the
    switch era) carries the king leg. Under /3 it is kept as prepared and the
    king is carried into it; nothing is discarded or retrained."""
    from cascade.shared.manifest import TrainedEntry
    from cascade.trainer.rolling import _entry_to_json
    from tests.unit.test_dec55_trainer_epochs import EpochOps, _v2
    from tests.unit.test_rolling import Clock, FakeClient, _ptr, _sched

    v2cfg, v2_block, L = _v2(cfg)
    first = era_for_block(v2cfg.round, v2_block).index
    clock = Clock()
    sched, _ = _sched(v2cfg, tmp_path, clock, ops=EpochOps(tmp_path, clock))
    cur = sched._new_era_state(FakeClient(), era_for_block(v2cfg.round, v2_block + L))
    nxt = sched._new_era_state(FakeClient(), era_for_block(v2cfg.round, v2_block + 2 * L))
    king = _entry_to_json(TrainedEntry("KING", 1, "king", "king/gen@sha256:" + "0" * 64,
                                       _ptr(f"KKING-{first}"), "d", v2_block))
    for era in (cur, nxt):
        era.king_hotkey, era.king_entry, era.king_init = "KING", dict(king), era.warm_start_ckpt
    sched.state.current, sched.state.next = cur, nxt
    sched._save()
    assert nxt.epoch_start == cur.epoch_start == first
    # restart with the switch at the prepared era (typed in chain.toml)
    sw_block = v2_block + 2 * L
    on = replace(v2cfg, v2=replace(v2cfg.v2, image_decoupled_from_block=sw_block))
    ops = EpochOps(tmp_path, clock)
    again, _ = _sched(on, tmp_path, clock, ops=ops)
    assert again.state.next is not None and again.state.next.epoch_start == first
    again._reconcile_prepared_epoch()
    assert again.state.next is not None and ops.retired == []
    assert again.state.current.index == first + 1
    again._roll_era(FakeClient(), era_for_block(on.round, sw_block), sw_block + 1)
    new = again.state.current
    assert new.index == first + 2 and new.epoch_start == first
    assert new.seed_block == nxt.seed_block and new.base_seed == nxt.base_seed
    assert new.king_entry == king and new.king_hotkey == "KING"    # carried, not retrained


# ── a discarded prepared era never costs the miner (the safety net) ──────────


DISCARD = "the seed epoch of this prepared era moved"


def _in_flight_leg(sched, ops, era, *, fail=None):
    import threading

    from cascade.trainer.rolling import ResolvedGen
    from tests.unit.test_rolling import REF

    q = ops.queue()
    q.add("ALFA", REF["ALFA"], reveal_block=1)
    assert q.mark_in_flight("ALFA", REF["ALFA"], target_boundary=era.start_block + 900,
                            era_index=era.index, started_block=era.start_block - 10)
    attempt = q.get("ALFA").attempt_id
    ops.hold["ALFA"] = threading.Event()
    if fail is not None:
        ops.failures["ALFA"] = fail
    sched._launch_leg(ResolvedGen("ALFA", 2, REF["ALFA"], 1), era, era.start_block - 10,
                      era.start_block + 900, "challenger")
    return q, attempt


def _discard_with_leg_in_flight(cfg, tmp_path, fail=None):
    import json

    from tests.unit.test_dec55_trainer_epochs import EpochOps, _join, _v2
    from tests.unit.test_rolling import Clock, FakeClient, _sched

    v2cfg, v2_block, L = _v2(cfg)
    clock = Clock()
    ops = EpochOps(tmp_path, clock)
    sched, _ = _sched(v2cfg, tmp_path, clock, ops=ops)
    nxt = sched._new_era_state(FakeClient(), era_for_block(v2cfg.round, v2_block + 2 * L))
    sched.state.next = nxt
    q, attempt = _in_flight_leg(sched, ops, nxt, fail=fail)
    sched._discard_prepared_era("next", DISCARD)
    assert sched.state.next is None and nxt._superseded
    rows = [json.loads(x) for x in
            (tmp_path / "make_good_review.jsonl").read_text().splitlines()]
    assert rows == [{"hotkey": "ALFA", "attempt_id": attempt, "payer": "miner key",
                     "ref": rows[0]["ref"], "era_index": nxt.index,
                     "reason": DISCARD, "at": rows[0]["at"]}]
    ops.hold["ALFA"].set()
    _join(sched)
    return q.get("ALFA")


def test_in_flight_miner_leg_of_a_discarded_era_is_requeued_unburned_for_make_good(cfg,
                                                                                    tmp_path):
    """Our discard never costs the miner — the in-flight leg is recorded for
    make-good (payer + attempt) and requeued unburned when it ends."""
    from cascade.funding.queue import OPERATOR_BLOCKED_CLASS

    e = _discard_with_leg_in_flight(cfg, tmp_path)
    assert e.status == "queued" and e.attempts == 0
    assert e.last_error_class == OPERATOR_BLOCKED_CLASS
    assert DISCARD in e.last_error


def test_a_superseded_legs_failure_is_never_charged_to_the_miner(cfg, tmp_path):
    from cascade.funding.queue import OPERATOR_BLOCKED_CLASS

    e = _discard_with_leg_in_flight(cfg, tmp_path, fail=("bad", True, "generator", True))
    assert e.status == "queued" and e.attempts == 0
    assert e.last_error_class == OPERATOR_BLOCKED_CLASS


# ── a typed switch needs the vote's notice ───────────────────────────────────


@pytest.fixture
def vcfg():
    """test_image_decoupled_contract's ``vcfg``: the shipped chain.toml with
    the rollover typed and the other secondary features disarmed."""
    from .test_image_decoupled_contract import B0

    c = load_chain_config(ROOT / "chain.toml")
    return replace(
        c,
        round=replace(c.round, rolling_from_block=B0, epoch_blocks=900,
                      epoch_blocks_prev=3600, epoch_activation_block=B0),
        scoring=replace(c.scoring, era_king_from_block=B0, tenure_blocks_from_block=B0,
                        cohort_maxt_increment_from_block=B0, forfeit_hotkeys=(),
                        forfeit_from_block=0, forfeit_successor_hotkey="",
                        win_margin_start_v2=0.0, win_margin_end_v2=0.0,
                        margin_warmup_blocks_v2=0, margin_v2_from_block=0),
        v2=replace(c.v2, prelaunch_burn=0.0, image_decoupled_from_block=0),  # vote open
        eval=replace(c.eval, shared_ladder_windows=0))


def _typed(c, block):
    return replace(c, v2=replace(c.v2, image_decoupled_from_block=block))


def test_typed_switch_needs_two_eras_of_notice_judged_once(vcfg):
    from .test_image_decoupled_contract import B0

    L = era_length_blocks(vcfg.round, B0)
    T = B0 + 4 * L
    on = _typed(vcfg, T)
    assert A.v2b_typed_notice_problem(on, None, T - 2 * L) == ""          # exactly enough
    why = A.v2b_typed_notice_problem(on, None, T - L + 5)
    assert "too little notice" in why and str(T + L) in why
    # judged against the FIRST-SEEN block: a node that passed never refuses itself
    seen = A.stamp_typed_image_decoupled(on, A.ActivationRecord(), T - 2 * L)
    assert (seen.source, seen.activation_block, seen.lock_block) == ("config", T, T - 2 * L)
    assert A.stamp_typed_image_decoupled(on, seen, T - 1) is seen
    assert A.v2b_typed_notice_problem(on, seen, T - 1) == ""
    # a changed typed block is a fresh first sight
    later = _typed(vcfg, T + L)
    assert A.stamp_typed_image_decoupled(later, seen, T - 1).lock_block == T - 1
    # once the switch era started there is nothing to give notice for
    assert A.v2b_typed_notice_problem(on, None, T) == ""
    assert A.v2b_typed_notice_problem(vcfg, None, T - 1) == ""            # nothing typed


def test_validator_and_trainer_refuse_to_start_on_a_short_typed_notice(vcfg, tmp_path):
    from cascade.trainer.loop import TrainerRunner
    from cascade.validator.activation_mixin import ActivationStartupRefused
    from cascade.validator.loop import ValidatorRunner

    from .test_activation import FakeChain, _v
    from .test_image_decoupled_contract import B0

    L = era_length_blocks(vcfg.round, B0)
    T = B0 + 4 * L
    on = _typed(vcfg, T)
    head = T - L + 5
    chain = FakeChain([_v("v1", 60), _v("v2", 40)], {}, block=head, hotkey="v1")
    runner = ValidatorRunner(cfg=on, activation_store=A.ActivationStore(tmp_path / "act.json"))
    with pytest.raises(ActivationStartupRefused, match="too little notice"):
        runner._activation_startup(chain)

    (tmp_path / "t").mkdir()
    trainer = TrainerRunner.__new__(TrainerRunner)
    trainer.cfg, trainer._cfg_base, trainer.netconfig = on, None, None
    trainer.activation_store = A.ActivationStore(tmp_path / "t" / "activation_state.json")
    trainer._image_decoupled = None
    with pytest.raises(RuntimeError, match="too little notice"):
        trainer._image_decoupled_startup(chain)

    # both nodes first loaded it in time (persisted first-seen): they start
    for path in (tmp_path / "activation_image_decoupled_state.json",
                 tmp_path / "t" / "activation_image_decoupled_state.json"):
        A.ActivationStore(path).save(A.stamp_typed_image_decoupled(on, A.ActivationRecord(),
                                                                   T - 3 * L))
    ok = ValidatorRunner(cfg=on, activation_store=A.ActivationStore(tmp_path / "act.json"))
    ok._activation_startup(chain)
    assert ok._image_decoupled.lock_block == T - 3 * L
    trainer._image_decoupled_startup(chain)
    assert trainer._image_decoupled.lock_block == T - 3 * L


# ── the audit ────────────────────────────────────────────────────────────────


def test_audit_epoch_shape_ignores_the_switch(base):
    from cascade.audit.checks import _epoch_start_for_seed

    cfg, era = base
    first, warm = v2_first_era(cfg), v2_first_warm_era(cfg)
    sw = warm + 6
    on = _switch(cfg, sw, era)
    e = era_for_block(cfg.round, (sw + 3) * era)
    # the running epoch's seed is legal at and after the switch
    assert _epoch_start_for_seed(on, replace(e, seed_block=(first - 1) * era)) == first
    assert _epoch_start_for_seed(cfg, replace(e, seed_block=(first - 1) * era)) == first
    # any warm-start SHAPE is legal, switch or not (whether a promotion fired
    # is the record's to show)
    for c in (cfg, on):
        assert _epoch_start_for_seed(c, replace(e, seed_block=(warm + 1) * era)) == warm + 2
        assert _epoch_start_for_seed(c, replace(e, seed_block=(sw - 1) * era)) == sw
    # inside the random-init window nothing but the first era is a start —
    # a switch there makes no epoch
    early = _switch(cfg, first + 3, era)
    e3 = era_for_block(cfg.round, (first + 5) * era)
    for c in (cfg, early):
        assert _epoch_start_for_seed(c, replace(e3, seed_block=(first + 2) * era)) is None
        assert _epoch_start_for_seed(c, replace(e3, seed_block=(first - 1) * era)) == first
    # before the switch the rule is the same
    e0 = era_for_block(cfg.round, (sw - 1) * era)
    assert _epoch_start_for_seed(on, replace(e0, seed_block=(first - 1) * era)) == first


def test_audit_check_era_across_the_switch(base):
    from cascade.audit import checks as C

    cfg, era = base
    first = v2_first_era(cfg)
    sw = v2_first_warm_era(cfg) + 6
    on = _switch(cfg, sw, era)
    on = replace(on, scoring=replace(on.scoring, era_king_from_block=int(on.v2.from_block)))
    start = (sw + 2) * era
    boundary = start + int(effective_epoch_blocks(on.round, start))
    e = era_for_block(on.round, start)

    def receipt(seed_block):
        stamp = replace(e, seed_block=seed_block).to_json()
        return SimpleNamespace(status="scored", epoch_start_block=boundary,
                               era_start_block=start, era_base_seed=7,
                               manifest={"era": stamp, "entries": []})

    # the running epoch's seed block (what the trainer stamps) passes with the
    # switch applied — exactly as without it
    plain = replace(on, v2=replace(on.v2, image_decoupled_from_block=0))
    for c in (on, plain):
        ok = C.check_era(receipt((first - 1) * era), c)
        assert ok.status != C.FAIL, ok.detail
        # a warm-start shape passes too
        assert C.check_era(receipt((sw - 1) * era), c).status != C.FAIL
        # a seed inside the random-init window that is not the first era fails
        assert C.check_era(receipt((first + 1) * era), c).status == C.FAIL
