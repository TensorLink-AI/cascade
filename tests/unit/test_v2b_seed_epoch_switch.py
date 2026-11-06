"""DEC-CA-0067 rule 2 ("the v2b switch era opens a seed epoch",
``image-decoupled/2``): the applied ``[v2] image_decoupled_from_block`` era is
a v2 seed-epoch start — new seeds and a forced king retrain, the SAME init —
derived from config alone, so every node agrees; the vote takes effect two
eras after the lock-in era (a promotion's notice). Without an applied switch
everything is exactly the warm-start rule.
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


@pytest.fixture
def base():
    """Mainnet chain.toml with v2 at a fixed era (ladder on, as shipped)."""
    cfg = load_chain_config(ROOT / "chain.toml")
    era = era_length_blocks(cfg.round, 0)
    gate = 10_000 * era
    cfg = replace(cfg, v2=replace(cfg.v2, from_block=gate),
                  training=replace(cfg.training, v2_from_block=gate))
    return cfg, era


def _switch(cfg, era_index: int, era: int):
    return replace(cfg, v2=replace(cfg.v2, image_decoupled_from_block=era_index * era))


# ── the pure rule ────────────────────────────────────────────────────────────


def test_switch_era_is_an_epoch_start_and_later_eras_share_its_seed_block(base):
    cfg, era = base
    first, warm = v2_first_era(cfg), v2_first_warm_era(cfg)
    sw = warm + 6
    on = _switch(cfg, sw, era)
    assert v2b_switch_era(on) == sw and v2b_switch_era(cfg) is None
    assert v2_seed_epoch_start(on, sw - 1, []) == first          # before: the running epoch
    for k in (0, 1, 5):
        assert v2_seed_epoch_start(on, sw + k, []) == sw
        # the init epoch is unchanged: no promotion is implied
        assert v2_init_epoch_start(on, sw + k, []) == first
        e = era_for_block(on.round, (sw + k) * era)
        assert v2_era_seed_block(on, e, []) == (sw - 1) * era   # the switch era's seed block
    # inside the random-init window too (a switch is not a warm start)
    early = _switch(cfg, first + 3, era)
    assert v2_seed_epoch_start(early, first + 5, []) == first + 3
    assert v2_init_epoch_start(early, first + 5, []) == first


def test_a_promotion_after_the_switch_opens_its_own_epoch(base):
    cfg, era = base
    sw = v2_first_warm_era(cfg) + 6
    on, w = _switch(cfg, sw, era), sw + 4
    assert v2_seed_epoch_start(on, w - 1, [w]) == sw
    assert v2_seed_epoch_start(on, w + 2, [w]) == w == v2_init_epoch_start(on, w + 2, [w])
    # and one before it is superseded by the switch (seeds only)
    w0 = sw - 3
    assert v2_seed_epoch_start(on, sw + 1, [w0]) == sw
    assert v2_init_epoch_start(on, sw + 1, [w0]) == w0


def test_a_promotion_on_the_switch_era_is_one_epoch(base):
    cfg, era = base
    sw = v2_first_warm_era(cfg) + 6
    on = _switch(cfg, sw, era)
    for k in (0, 3):
        assert v2_seed_epoch_start(on, sw + k, [sw]) == sw == v2_init_epoch_start(on, sw + k, [sw])
        e = era_for_block(on.round, (sw + k) * era)
        assert v2_era_seed_block(on, e, [sw]) == v2_era_seed_block(on, e, []) == (sw - 1) * era


def test_no_switch_applied_is_exactly_the_warm_start_rule(base):
    cfg, era = base
    first, warm = v2_first_era(cfg), v2_first_warm_era(cfg)
    at_first = _switch(cfg, first, era)          # a switch AT the first v2 era adds nothing
    for c in (cfg, at_first):
        for warms in ([], [warm], [warm, warm + 7]):
            for idx in range(first - 2, warm + 12):
                want = (None if idx < first else
                        max([first] + [w for w in warms if first < w <= idx]))
                assert v2_seed_epoch_start(c, idx, warms) == want
                assert v2_init_epoch_start(c, idx, warms) == want


# ── the vote: two eras of notice, a fresh feature name ───────────────────────


def test_feature_name_is_the_rule_2_bundle():
    assert A.V2B_RULE == "image-decoupled/2"
    # without the quantile-head knob the name is exactly the pre-knob one
    for n, want in ((4000, "v2b-90bc5237"), (300, "v2b-3a259561")):
        body = f"image-decoupled/2|train_image_digest,continuation_protocol|eval_n_windows_after={n}"
        assert want == "v2b-" + hashlib.sha256(body.encode()).hexdigest()[:8]
        assert A.v2b_feature_name(n) == want == A.v2b_feature_name(n, False)
    # quantile_head_after = true is part of the bundle: a fresh name
    for n, want in ((4000, "v2b-b10591aa"), (300, "v2b-f2eefcb3")):
        body = (f"image-decoupled/2|train_image_digest,continuation_protocol|"
                f"eval_n_windows_after={n}|quantile_head_after=1")
        assert want == "v2b-" + hashlib.sha256(body.encode()).hexdigest()[:8]
        assert A.v2b_feature_name(n, True) == want != A.v2b_feature_name(n)
    assert A.v2b_feature_name(4000) != "v2b-150ec0d0"           # the /1 name is retired
    main = load_chain_config(ROOT / "chain.toml")
    test = load_chain_config(ROOT / "chain.testnet.toml")
    assert main.v2.quantile_head_after and test.v2.quantile_head_after
    assert main.v2.image_decoupled_feature == "v2b-b10591aa" == A.v2b_feature_name_for(main.v2)
    assert test.v2.image_decoupled_feature == "v2b-f2eefcb3" == A.v2b_feature_name_for(test.v2)


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
    # a typed override is any era start (it opens its own epoch); off-grid is refused
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


def test_validator_seed_block_pointer_era_and_init_epoch(base, tmp_path):
    cfg, era = base
    first = v2_first_era(cfg)
    sw = v2_first_warm_era(cfg) + 6
    r = _validator(_switch(cfg, sw, era), era, tmp_path)
    e = era_for_block(r.cfg.round, (sw + 2) * era)
    assert r._era_seed_block(e) == (sw - 1) * era
    assert r._pointer_era(e) == sw                   # the king is held (and retrains) per epoch
    assert r._epoch_start(e) == sw and r._init_epoch_start(e) == first
    plain = _validator(cfg, era, tmp_path / "p")
    assert plain._era_seed_block(e) == (first - 1) * era and plain._pointer_era(e) == first


def test_validator_refuses_an_init_change_at_the_switch_era(base, tmp_path):
    from cascade.shared.manifest import TrainingManifest, format_trained_pointer

    cfg, era = base
    w = v2_first_warm_era(cfg) + 2
    sw = w + 4
    kp = format_trained_pointer("cascade/king@sha256:" + "b" * 64)
    r = _validator(_switch(cfg, sw, era), era, tmp_path, members=(kp,), effective=w)
    r.state = replace(r.state, epoch_init={"epoch": w, "init": kp})
    spec = replace(era_for_block(r.cfg.round, sw * era), generation=1, member_index=0)
    spec = replace(spec, seed_block=r._era_seed_block(spec))
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
    e = replace(e, seed_block=(sw - 1) * era)
    m = SimpleNamespace(era=e.to_json())
    runner = SimpleNamespace(_init_epoch_start=lambda spec: w)
    assert acceptance_block(cfg, m) == sw * era                    # no switch: unchanged
    assert acceptance_block(cfg, m, runner) == sw * era           # no switch: runner ignored
    on = _switch(cfg, sw, era)
    assert acceptance_block(on, m) == sw * era
    assert acceptance_block(on, m, runner) == w * era             # the init epoch's start


# ── the trainer ──────────────────────────────────────────────────────────────


def test_trainer_retrains_the_king_at_the_switch_era_with_the_same_init(cfg, tmp_path):
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
    last = v2_block + 6
    for k in (1, 2, 3):
        last = _tick_era(clock, ops, sched, client, v2_block + k * L, L, last=last)
        cur = sched.state.current
        assert cur.epoch_start == (first if k < 2 else sw)
    cur = sched.state.current
    assert cur.index == sw + 1 and cur.warm_start_ckpt == init0      # same init
    assert cur.seed_block == era_for_block(v2cfg.round, sw_block).seed_block
    assert cur.base_seed != base0                                     # new seeds
    assert [c[1] for c in ops.king_calls] == [first, sw]              # one forced retrain
    assert (sw, sw + 1) in ops.carried                                # then carried again
    assert ops.init_benches == [first, sw]                            # re-benched per epoch


def test_trainer_lane_b_reuses_the_init_epochs_merged_init_at_the_switch(cfg, tmp_path,
                                                                          monkeypatch):
    from tests.unit.test_dec55_trainer_epochs import EpochOps, _v2
    from tests.unit.test_rolling import Clock, FakeClient, _sched

    v2cfg, v2_block, L = _v2(cfg)
    first = era_for_block(v2cfg.round, v2_block).index
    w, sw = first + 3, first + 5
    v2cfg = replace(v2cfg, v2=replace(v2cfg.v2, image_decoupled_from_block=v2_block + 5 * L))

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
    assert at_sw.epoch_start == sw and at_sw.base_seed != at_w.base_seed
    # the switch era trains from the init epoch's MERGED init — the same init,
    # no merge pending since applied
    assert at_sw.warm_start_ckpt == f"merged-{w}" and at_sw.warm_start_base == "M"
    assert at_sw.warm_start_merges == at_w.warm_start_merges == [{"m": 1}]
    assert sched.lane_b.calls == [(w, True), (w, False)]


def test_trainer_discards_a_prepared_era_the_switch_moved(cfg, tmp_path):
    from tests.unit.test_dec55_trainer_epochs import EpochOps, _v2
    from tests.unit.test_rolling import Clock, FakeClient, _sched

    v2cfg, v2_block, L = _v2(cfg)
    first = era_for_block(v2cfg.round, v2_block).index
    clock = Clock()
    sched, _ = _sched(v2cfg, tmp_path, clock, ops=EpochOps(tmp_path, clock))
    nxt = sched._new_era_state(FakeClient(), era_for_block(v2cfg.round, v2_block + 2 * L))
    assert nxt.epoch_start == first
    sched.state.next = nxt
    sched.update_config(v2cfg)                                  # nothing changed: kept
    assert sched.state.next is nxt
    sched.update_config(replace(v2cfg, v2=replace(v2cfg.v2,
                                                  image_decoupled_from_block=v2_block + 2 * L)))
    assert sched.state.next is None                             # re-prepared on the new epoch


def test_restart_discards_a_persisted_stale_switch_era_and_reprepares_it(cfg, tmp_path):
    """Review #1: a typed block (or a late resolve) arrives with a RESTART —
    the scheduler is built from persisted state whose prepared ``next`` (the
    switch era) still sits in the old epoch with the carried king leg. It is
    discarded at the first tick's reconcile and re-prepared on the new
    epoch's seeds with no carried king; the running era is never discarded."""
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
    again._reconcile_prepared_epoch()                  # what every tick runs first
    assert again.state.next is None and ops.retired == [first + 2]
    assert again.state.current.index == first + 1      # the running era is kept
    again._reconcile_prepared_epoch()                  # idempotent, logs once
    assert again.state.current.index == first + 1
    again._roll_era(FakeClient(), era_for_block(on.round, sw_block), sw_block + 1)
    new = again.state.current
    assert new.index == first + 2 and new.epoch_start == first + 2
    assert new.seed_block == era_for_block(on.round, sw_block).seed_block
    assert new.base_seed != nxt.base_seed and new.king_entry is None   # retrains


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
    sched.update_config(replace(v2cfg, v2=replace(v2cfg.v2,
                                                  image_decoupled_from_block=v2_block + 2 * L)))
    assert sched.state.next is None and nxt._superseded
    rows = [json.loads(x) for x in
            (tmp_path / "make_good_review.jsonl").read_text().splitlines()]
    assert rows == [{"hotkey": "ALFA", "attempt_id": attempt, "payer": "miner key",
                     "ref": rows[0]["ref"], "era_index": nxt.index,
                     "reason": "the v2b switch opens a seed epoch at this era",
                     "at": rows[0]["at"]}]
    ops.hold["ALFA"].set()
    _join(sched)
    return q.get("ALFA")


def test_in_flight_miner_leg_of_a_discarded_era_is_requeued_unburned_for_make_good(cfg,
                                                                                    tmp_path):
    """Review #6: our discard never costs the miner — the in-flight leg is
    recorded for make-good (payer + attempt) and requeued unburned when it
    ends."""
    from cascade.funding.queue import OPERATOR_BLOCKED_CLASS

    e = _discard_with_leg_in_flight(cfg, tmp_path)
    assert e.status == "queued" and e.attempts == 0
    assert e.last_error_class == OPERATOR_BLOCKED_CLASS
    assert "seed epoch" in e.last_error


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
        v2=replace(c.v2, prelaunch_burn=0.0))


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


def test_audit_epoch_shape_accepts_the_switch_and_rejects_a_seed_across_it(base):
    from cascade.audit.checks import _epoch_start_for_seed

    cfg, era = base
    first, warm = v2_first_era(cfg), v2_first_warm_era(cfg)
    sw = warm + 6
    on = _switch(cfg, sw, era)
    e = era_for_block(cfg.round, (sw + 3) * era)
    assert _epoch_start_for_seed(on, replace(e, seed_block=(sw - 1) * era)) == sw
    # a seed from before the switch is mid-epoch across the rule change
    assert _epoch_start_for_seed(on, replace(e, seed_block=(first - 1) * era)) is None
    assert _epoch_start_for_seed(on, replace(e, seed_block=(warm + 1) * era)) is None
    # unchanged without a switch
    assert _epoch_start_for_seed(cfg, replace(e, seed_block=(first - 1) * era)) == first
    # (era sw is past the random-init window: without a switch it is still a
    # legal warm-start SHAPE — whether a promotion fired is the record's to show)
    assert _epoch_start_for_seed(cfg, replace(e, seed_block=(sw - 1) * era)) == sw
    # ...but inside the random-init window only the switch makes it legal
    early = _switch(cfg, first + 3, era)
    e3 = era_for_block(cfg.round, (first + 5) * era)
    assert _epoch_start_for_seed(cfg, replace(e3, seed_block=(first + 2) * era)) is None
    assert _epoch_start_for_seed(early, replace(e3, seed_block=(first + 2) * era)) == first + 3
    # a promotion after the switch is still legal
    e2 = era_for_block(cfg.round, (sw + 9) * era)
    assert _epoch_start_for_seed(on, replace(e2, seed_block=(sw + 4) * era)) == sw + 5
    # before the switch the old rule holds
    e0 = era_for_block(cfg.round, (sw - 1) * era)
    assert _epoch_start_for_seed(on, replace(e0, seed_block=(first - 1) * era)) == first


def test_audit_check_era_on_a_switch_epoch_receipt(base):
    from cascade.audit import checks as C

    cfg, era = base
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

    ok = C.check_era(receipt((sw - 1) * era), on)
    assert ok.status != C.FAIL, ok.detail
    bad = C.check_era(receipt((v2_first_era(cfg) - 1) * era), on)
    assert bad.status == C.FAIL
    # the same pre-switch seed passes without a switch (the warm-start rule)
    plain = replace(on, v2=replace(on.v2, image_decoupled_from_block=0))
    assert C.check_era(receipt((v2_first_era(cfg) - 1) * era), plain).status != C.FAIL
