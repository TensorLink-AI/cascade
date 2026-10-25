"""Auxiliary settlement progress, shared admission and recovery billing."""

from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest

from cascade.funding.queue import FundedQueue, LaneBQueue
from cascade.shared.bench_report import BenchReport, dump_bench_report
from cascade.trainer.admission import FairSlots
from cascade.trainer.loop import ResolvedGenerator, TrainerRunner
from cascade.trainer.runner_funded_legs import funded_leg_scope

from .test_lane_c_trainer_rolling import DIGEST, _LaneC
from .test_rolling import EB, Clock, FakeClient, _advance, _armed, _join, _sched, _validator


class _LaneB:
    def __init__(self):
        self.settlements = []

    def tick(self, *args):
        return []

    def settle(self, **kwargs):
        self.settlements.append(kwargs)

    def expert_manifest_digest(self, rid):
        return "ab" * 32

    def live_hotkeys(self):
        return set()


@pytest.mark.parametrize("lane", ["b", "c", "reports"])
def test_no_lane_a_challengers_does_not_stop_other_lanes(cfg, tmp_path, lane):
    armed = _armed(cfg)
    if lane == "b":
        armed = replace(armed, lane_b=replace(armed.lane_b, from_block=1))
    if lane == "c":
        armed = replace(armed, lane_c=replace(armed.lane_c, from_block=1))
    clock = Clock()
    sched, ops = _sched(armed, tmp_path, clock)
    events = []
    if lane == "b":
        sched.lane_b = _LaneB()
    if lane == "c":
        sched.lane_c = _LaneC(events)
    text = dump_bench_report(BenchReport(round_id="123", created_block=1))
    if lane == "reports":
        sched.state.vbench_pending = {"123": 0}
        ops.read_text = lambda key: text
    start = armed.round.rolling_from_block + 5
    client = FakeClient()
    sched.tick(client, start)
    _join(sched)
    _advance(clock, ops, sched, client, from_block=start,
             to_block=armed.round.rolling_from_block + EB + 3)
    assert len(ops.manifests) == 1
    m = ops.manifests[0]
    assert [e.role for e in m.entries] == ["king"]
    assert not ops.burnt and not ops.leg_calls
    assert _validator(armed, ops).check_manifest(m) is None
    if lane == "b":
        assert len(sched.lane_b.settlements) == 1
        assert m.expert_manifest_sha256 == "ab" * 32
    elif lane == "c":
        assert len(events) == 1 and m.lane_c_digest == DIGEST
    else:
        assert m.validator_bench_reports == {"123": hashlib.sha256(text.encode()).hexdigest()}
        assert "123" not in sched.state.vbench_pending


def test_recovery_billing_survives_restart_and_no_capacity_requeue(tmp_path):
    path = tmp_path / "queue.json"
    q = FundedQueue(path)
    q.add("miner", "ref", 1)
    q.mark_in_flight("miner", "ref", target_boundary=10, era_index=1, started_block=2)
    assert q.requeue("miner", error="completed leg aged out on our side",
                     error_class="no_capacity", burn_attempt=False, operator_billed=True)
    recovered = FundedQueue(path)
    assert recovered.get("miner").operator_billed and recovered.get("miner").attempts == 0
    recovered.requeue("miner", error="operator GPU unavailable", error_class="operator",
                      burn_attempt=False)
    assert recovered.get("miner").operator_billed
    assert recovered.public_view()["entries"][0]["operator_billed"]


def test_lane_context_is_thread_local_and_king_does_not_borrow_a_queue(cfg, tmp_path):
    cfg = replace(cfg, round=replace(cfg.round, funded_mode="required"))
    r = TrainerRunner(cfg=cfg, base_trainer=object(), work_root=tmp_path, trainer_spec="m:C")
    lane_b = LaneBQueue(tmp_path / "b.json")
    reduced = cfg.training.for_hours(0.5)
    barrier = threading.Barrier(2)
    seen = []

    def worker():
        with funded_leg_scope(r, lane_b, reduced):
            barrier.wait(timeout=3)
            seen.append(r._funded_queue().path)
            seen.append(r._leg_wall_seconds())
            barrier.wait(timeout=3)

    t = threading.Thread(target=worker)
    t.start()
    barrier.wait(timeout=3)
    assert r._funded_queue().path != lane_b.path
    barrier.wait(timeout=3)
    t.join(timeout=3)
    assert not t.is_alive() and seen == [lane_b.path, reduced.max_train_seconds]
    with funded_leg_scope(r, lane_b, reduced, king_leg=True):
        assert r._funded_queue() is None
        assert not r._king_pending()
    assert r._funded_queue().path != lane_b.path


def test_deadline_less_leg_defers_while_king_needs_capacity(cfg, tmp_path):
    r = TrainerRunner(cfg=cfg, base_trainer=object(), work_root=tmp_path, trainer_spec="m:C")
    r.cfg = replace(cfg, round=replace(cfg.round, funded_king_rent=True, funded_pods="rent"))
    r._funded_king_host, r._king_rent_done = None, False
    assert r._king_pending()
    assert r._yield_to_king("lane B or C") is False
    with funded_leg_scope(r, None, cfg.training, king_leg=True):
        assert r._yield_to_king("king's own funded leg") is True


def test_operator_king_also_gets_cpu_generation_priority(cfg, tmp_path, monkeypatch):
    from cascade.trainer.rolling import LegOps

    r = TrainerRunner(cfg=cfg, base_trainer=object(), work_root=tmp_path, trainer_spec="m:C")
    seen = []
    monkeypatch.setattr(r, "_rolling_train_king", lambda *a, **kw:
                        seen.append(r._leg_local.funded_is_king))
    LegOps(r).train_king(None, SimpleNamespace(start_block=1), 1, end_wall=2)
    assert seen == [True] and not getattr(r._leg_local, "funded_is_king", False)


@pytest.mark.parametrize("where", ["runtime", "allocated_rent"])
def test_paid_operator_failure_credits_the_right_lane_and_blocks_payer_retry(
        cfg, tmp_path, monkeypatch, where):
    from cascade.trainer import runner_funded_legs as rfl
    from cascade.trainer.faults import _FundedLegSkip, _OperatorFault

    cfg = replace(cfg, round=replace(cfg.round, funded_mode="required",
                                    funded_operator_fallback=False))
    r = TrainerRunner(cfg=cfg, base_trainer=object(), work_root=tmp_path, trainer_spec="m:C")
    from cascade.trainer.admission import remember_owners
    remember_owners(r, [SimpleNamespace(hotkey=h, coldkey="owner-" + h, commit_block=1)
                        for h in ['miner']])
    b = LaneBQueue(tmp_path / "b.json")
    b.add_lane_b("miner", ("energy",), "ref", 1)
    gen = ResolvedGenerator("miner", 1, "ref")
    monkeypatch.setattr(rfl, "leg_corpus", lambda *a, **kw: {})
    monkeypatch.setattr(rfl, "_reupload_pending", lambda *a, **kw: None)
    with monkeypatch.context() as m:
        def rent(*args, **kwargs):
            r._leg_local.funded_allocated = True
            if where == "allocated_rent":
                r._record_funded_failure("miner", "credential setup failed", miner_fault=False,
                                         error_class="operator", burn=False)
                raise _OperatorFault("credential setup failed")
            return object(), object()

        m.setattr(r, "_rent_funded_host", rent)
        m.setattr(r, "_funded_pod_identity_mismatch", lambda p: None)
        m.setattr(r, "_teardown_funded_pod", lambda p: None)
        m.setattr("cascade.trainer.remote.probe_worker_runtime", lambda host: "stale image")
        with pytest.raises(_OperatorFault):
            r._run_funded_leg(None, gen, SimpleNamespace(base_seed=1), 1, cfg.training,
                              "-xenergy", warm_start_ref="base", funded_queue=b)
    assert b.get("miner").operator_billed and b.get("miner").operator_faults == 1
    assert r._funded_queue().get("miner") is None
    # No operator lane or rental is configured, so no make-good can run: the
    # retry rents a new pod on the payer's key (bounded by the fault cap).
    class _PayerRent(Exception):
        pass

    def payer_vault():
        raise _PayerRent

    monkeypatch.setattr(r, "_payer_vault", payer_vault)
    with pytest.raises((_PayerRent, _FundedLegSkip)):
        r._run_funded_leg(None, gen, SimpleNamespace(base_seed=1), 1, cfg.training,
                          "-xenergy", warm_start_ref="base", funded_queue=b)
    assert not b.get("miner").operator_billed


def test_shared_job_admission_bounds_lanes_and_reserves_king(cfg, tmp_path, monkeypatch):
    from cascade.trainer.runner_funded_legs import _FundedLegMixin

    cfg = replace(cfg, round=replace(cfg.round, funded_field_cap=1))
    r = TrainerRunner(cfg=cfg, base_trainer=object(), work_root=tmp_path, trainer_spec="m:C")
    from cascade.trainer.admission import remember_owners
    remember_owners(r, [SimpleNamespace(hotkey=h, coldkey="owner-" + h, commit_block=1)
                        for h in ['A', 'B']])
    entered, release = threading.Event(), threading.Event()
    order, errors = [], []

    def work(self, disp, gen, *args, **kwargs):
        order.append(gen.hotkey)
        if gen.hotkey == "A":
            entered.set()
            assert release.wait(5)
        return gen.hotkey

    monkeypatch.setattr(_FundedLegMixin, "_run_funded_leg_scoped", work)

    def run(hotkey, *, king=False):
        try:
            r._run_funded_leg(None, ResolvedGenerator(hotkey, 1, "ref"),
                              SimpleNamespace(base_seed=1), 1,
                              cfg.training, "", warm_start_ref=None, king_leg=king)
        except Exception as e:
            errors.append(e)

    a = threading.Thread(target=run, args=("A",))
    a.start()
    assert entered.wait(3)
    b = threading.Thread(target=run, args=("B",))
    b.start()
    gate = r.__dict__["_funded_job_slots"]
    with gate.condition:
        assert gate.condition.wait_for(lambda: len(gate.waiters) == 1, timeout=3)
    run("KING", king=True)
    assert order == ["A", "KING"] and gate.active == 1
    release.set()
    for t in (a, b):
        t.join(timeout=5)
        assert not t.is_alive()
    assert not errors and order == ["A", "KING", "B"] and gate.active == 0


def test_verified_upload_recovery_bypasses_full_compute_admission(cfg, tmp_path, monkeypatch):
    from cascade.trainer import runner_funded_legs as rfl

    r = TrainerRunner(cfg=cfg, base_trainer=object(), work_root=tmp_path, trainer_spec="m:C")
    gate = r.__dict__["_funded_job_slots"] = FairSlots()
    r._leg_local.end_wall = time.time() - 1
    monkeypatch.setattr(rfl, "_reupload_pending", lambda *a: "recovered-checkpoint")
    with gate.slot(1):
        result = r._run_funded_leg(None, ResolvedGenerator("miner", 1, "ref"),
                                   SimpleNamespace(base_seed=1), 1, cfg.training, "",
                                   warm_start_ref=None)
        assert result == "recovered-checkpoint" and gate.active == 1


def test_cpu_vm_admission_waits_before_provisioning(tmp_path, monkeypatch):
    from .test_generation_stage import FakeProvider, FakeStore, _stage

    stage, _index, _calls = _stage(tmp_path, FakeProvider(), FakeStore())
    stage.cfg = replace(stage.cfg, generation=replace(stage.cfg.generation, max_parallel_jobs=1))
    entered, release = threading.Event(), threading.Event()
    order, errors = [], []
    monkeypatch.setattr(stage, "existing", lambda **kw: None)

    def generate(**kwargs):
        order.append(kwargs["hotkey"])
        if kwargs["hotkey"] == "first":
            entered.set()
            assert release.wait(5)
        return kwargs["hotkey"]

    monkeypatch.setattr(stage, "_generate_admitted", generate)

    def run(hotkey):
        try:
            stage.generate(round_id="1", hotkey=hotkey, uid=1, era=1,
                           gen_ref="ref", repo_dir=tmp_path, seed=1)
        except Exception as e:
            errors.append(e)

    first = threading.Thread(target=run, args=("first",))
    first.start()
    assert entered.wait(3)
    second = threading.Thread(target=run, args=("second",))
    second.start()
    gate = stage.__dict__["_slots"]
    with gate.condition:
        assert gate.condition.wait_for(lambda: len(gate.waiters) == 1, timeout=3)
    assert order == ["first"]
    release.set()
    for t in (first, second):
        t.join(timeout=5)
        assert not t.is_alive()
    assert not errors and order == ["first", "second"] and gate.active == 0


def test_fair_slots_priority_fifo_and_release_after_failure():
    gate = FairSlots()
    order = []
    errors = []

    def wait(label, priority=False):
        try:
            with gate.slot(1, deadline=time.time() + 5, priority=priority) as acquired:
                assert acquired
                order.append(label)
                if label == "king":
                    raise ValueError("simulated launch failure")
        except ValueError:
            errors.append(label)

    with gate.slot(1) as acquired:
        assert acquired
        a = threading.Thread(target=wait, args=("first",))
        a.start()
        with gate.condition:
            assert gate.condition.wait_for(lambda: len(gate.waiters) == 1, timeout=3)
        b = threading.Thread(target=wait, args=("second",))
        b.start()
        with gate.condition:
            assert gate.condition.wait_for(lambda: len(gate.waiters) == 2, timeout=3)
        king = threading.Thread(target=wait, args=("king", True))
        king.start()
        with gate.condition:
            assert gate.condition.wait_for(lambda: len(gate.waiters) == 3, timeout=3)
        assert gate.active == 1 and not order
    for t in (a, b, king):
        t.join(timeout=5)
        assert not t.is_alive()
    assert order == ["king", "first", "second"] and errors == ["king"]
    assert gate.active == 0 and not gate.waiters
    with gate.slot(1, deadline=time.time() - 1) as acquired:
        assert acquired is False
    assert gate.active == 0 and not gate.waiters
