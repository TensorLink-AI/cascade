"""2026-10-31 cgen VM churn: offer quarantine on boot failure, the per-hotkey
rent cap, teardown on every exit path (including process stop), admission
before generation, and corpus reuse on requeue."""

from __future__ import annotations

import json
import logging
import signal
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest

from cascade.generation import stage as stage_mod
from cascade.generation.stage import (
    GenerationInfraError,
    OfferQuarantine,
    RentLimiter,
    terminate_owned_vms,
)
from cascade.provision.core import ProvisionError
from cascade.provision.cpu import CpuShape, select_cpu_offer

from .test_generation_stage import HK, FakeProvider, FakeStore, _go, _stage


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class ErrorBootProvider(FakeProvider):
    """Shadeform 2026-10-31: every instance of one offer enters 'error'."""

    name = "shadeform"

    def __init__(self, bad_regions=("dallas-usa-1",), regions=("dallas-usa-1",)):
        super().__init__()
        self.bad, self.regions = set(bad_regions), list(regions)
        self.by_vm: dict[str, str] = {}

    def offers(self, vcpus, price):
        return [CpuShape(vcpus=32, memory_gb=128, price_per_hour=1.02, provider="shadeform",
                         type_id="cpu_32", region=r, cloud="latitude") for r in self.regions]

    def launch(self, shape, *, name, ssh_pubkey, image=""):
        vm = f"vm-{len(self.launched) + 1}"
        self.launched.append(name)
        self.by_vm[vm] = shape.region
        return vm

    def wait_ready(self, vm, *, timeout):
        if self.by_vm[vm] in self.bad:
            raise ProvisionError(f"shadeform instance {vm} entered 'error'")
        return True


def _churn_stage(tmp_path, prov, clock):
    st, idx, calls = _stage(tmp_path, prov, FakeStore())
    st.clock = clock
    st.ledger_path = tmp_path / "generation_vms.json"
    return st, idx


# ── offer quarantine ────────────────────────────────────────────────────────


def test_a_boot_failure_quarantines_the_offer_so_the_requeue_rents_nothing(tmp_path, caplog):
    clock, prov = Clock(), ErrorBootProvider()
    st, _ = _churn_stage(tmp_path, prov, clock)
    with (caplog.at_level(logging.ERROR, logger="cascade.generation.stage"),
          pytest.raises(GenerationInfraError, match="never came up")):
        _go(st, tmp_path)
    assert prov.launched and prov.terminated == ["vm-1"]
    assert "offer quarantined for 1800s (strike 1)" in caplog.text
    assert json.loads(st.ledger_path.read_text()) == []          # torn down, row dropped
    # The requeued leg's next tick: the only offer is quarantined → no rent.
    with pytest.raises(GenerationInfraError, match="no CPU capacity"):
        _go(st, tmp_path)
    assert len(prov.launched) == 1
    # Cooldown over: one more try; a second strike doubles the cooldown.
    clock.t += 1801
    with pytest.raises(GenerationInfraError, match="never came up"):
        _go(st, tmp_path)
    assert len(prov.launched) == 2
    assert st.offer_quarantine().blocked(prov, prov.offers(32, 2)[0])
    clock.t += 1801
    assert st.offer_quarantine().blocked(prov, prov.offers(32, 2)[0])   # 3600 s now
    clock.t += 1800
    assert not st.offer_quarantine().blocked(prov, prov.offers(32, 2)[0])


def test_a_quarantined_offer_falls_through_to_the_next_one(tmp_path):
    clock = Clock()
    prov = ErrorBootProvider(bad_regions=("dallas-usa-1",),
                             regions=("dallas-usa-1", "ashburn-usa-1"))
    st, idx = _churn_stage(tmp_path, prov, clock)
    with pytest.raises(GenerationInfraError, match="never came up"):
        _go(st, tmp_path)
    res = _go(st, tmp_path)                        # requeue rents the healthy region
    assert res.index["corpus_digest"] == idx["corpus_digest"]
    assert [prov.by_vm[v] for v in prov.terminated] == ["dallas-usa-1", "ashburn-usa-1"]


def test_a_vm_that_boots_clears_its_offer():
    clock = Clock()
    q = OfferQuarantine(cooldown_s=10, clock=clock)
    shape = CpuShape(vcpus=32, memory_gb=1, price_per_hour=1, provider="p", type_id="t")
    assert q.strike("p", shape) == (1, 10)
    assert q.strike("p", shape) == (2, 20)
    q.clear("p", shape)
    assert not q.blocked("p", shape)
    assert q.strike("p", shape) == (1, 10)


def test_select_cpu_offer_skips_excluded_offers():
    prov = ErrorBootProvider(regions=("a", "b"))
    cfg = SimpleNamespace(vcpus=32, max_price_per_hour=2.0)
    picked = select_cpu_offer([prov], cfg, exclude=lambda p, o: o.region == "a")
    assert picked[1].region == "b"
    assert select_cpu_offer([prov], cfg, exclude=lambda p, o: True) is None


# ── the per-hotkey rent cap ─────────────────────────────────────────────────


def test_a_hotkey_cannot_rent_past_the_cap_and_it_alarms(tmp_path, caplog):
    clock = Clock()
    prov = ErrorBootProvider(bad_regions=(), regions=("r",))
    prov.wait_ready = lambda vm, *, timeout: False            # "not ready" every time
    st, _ = _churn_stage(tmp_path, prov, clock)
    st.boot_fail_cooldown_s = 0.0                             # isolate the cap
    for _ in range(3):
        with pytest.raises(GenerationInfraError, match="not ready"):
            _go(st, tmp_path)
        clock.t += 60
    with (caplog.at_level(logging.ERROR, logger="cascade.generation.stage"),
          pytest.raises(GenerationInfraError, match="rent cap")):
        _go(st, tmp_path)
    assert len(prov.launched) == 3
    assert "generation ALARM" in caplog.text and "refusing another rent" in caplog.text
    clock.t += 3600                                           # the window slides
    with pytest.raises(GenerationInfraError, match="not ready"):
        _go(st, tmp_path)
    assert len(prov.launched) == 4


def test_the_king_alarms_at_the_cap_but_is_never_refused(tmp_path, caplog):
    clock = Clock()
    prov = FakeProvider()
    st, idx = _churn_stage(tmp_path, prov, clock)
    st.existing = lambda **kw: None                           # force a fresh rent
    for _ in range(3):
        st.rent_limiter().record(HK)
    (tmp_path / "repo").mkdir(exist_ok=True)
    with caplog.at_level(logging.ERROR, logger="cascade.generation.stage"):
        res = st.generate(round_id="77", hotkey=HK, uid=4, era=9, gen_ref="g@sha256:1",
                          repo_dir=tmp_path / "repo", seed=123, priority=True)
    assert res.index["corpus_digest"] == idx["corpus_digest"]
    assert "the king is never refused" in caplog.text


def test_rent_limiter_window():
    clock = Clock()
    lim = RentLimiter(limit=2, window_s=100, clock=clock)
    assert lim.record("a") == 1 and lim.record("a") == 2 and lim.recent("b") == 0
    clock.t += 101
    assert lim.recent("a") == 0


# ── teardown on every exit path ─────────────────────────────────────────────


class _Boom(BaseException):
    """Not an Exception: a KeyboardInterrupt/SystemExit-class unwind."""


@pytest.mark.parametrize("step,exc", [
    ("_bootstrap", GenerationInfraError("bootstrap failed")),
    ("_run", RuntimeError("ssh died")),
    ("_uploaded_index", GenerationInfraError("index unreadable")),
    ("_bootstrap", _Boom()),
])
def test_every_exit_path_terminates_the_vm_and_drops_its_row(tmp_path, monkeypatch, step, exc):
    prov = FakeProvider()
    st, _ = _churn_stage(tmp_path, prov, Clock())

    def fail(*a, **k):
        raise exc

    monkeypatch.setattr(st, step, fail)
    with pytest.raises(type(exc)):
        _go(st, tmp_path)
    assert prov.terminated == ["vm-1"]
    assert json.loads(st.ledger_path.read_text()) == []
    assert not stage_mod._LIVE


def test_shutdown_terminates_only_this_processes_vms(tmp_path, monkeypatch):
    ledger = tmp_path / "generation_vms.json"
    ledger.write_text(json.dumps([
        {"provider": "targon", "vm_id": "vm-mine", "name": "cgen-mine"},
        {"provider": "targon", "vm_id": "vm-other", "name": "cgen-other"},
        {"provider": "targon", "vm_id": "", "name": "cgen-launching"},
    ]))
    monkeypatch.setattr(stage_mod, "_LIVE", {"cgen-mine", "cgen-launching"})
    prov = FakeProvider()
    assert terminate_owned_vms(ledger, [prov]) == ["vm-mine"]
    assert prov.terminated == ["vm-mine"]
    left = {r["name"] for r in json.loads(ledger.read_text())}
    assert left == {"cgen-other", "cgen-launching"}           # the reaper's to resolve


def test_shutdown_keeps_the_row_when_terminate_fails(tmp_path, monkeypatch):
    ledger = tmp_path / "generation_vms.json"
    ledger.write_text(json.dumps([{"provider": "targon", "vm_id": "vm-1", "name": "cgen-x"}]))
    monkeypatch.setattr(stage_mod, "_LIVE", {"cgen-x"})
    prov = FakeProvider()

    def boom(vm):
        raise RuntimeError("api down")

    prov.terminate = boom
    assert terminate_owned_vms(ledger, [prov]) == []
    assert len(json.loads(ledger.read_text())) == 1


def test_sigterm_handler_tears_down_then_defers_to_the_previous_handler():
    from cascade.trainer.runner_generation import install_generation_shutdown

    events = []
    sig = signal.SIGUSR1
    old = signal.signal(sig, lambda s, f: events.append("previous"))
    try:
        runner = SimpleNamespace(_shutdown_generation_vms=lambda: events.append("teardown"))
        assert install_generation_shutdown(runner, signals=(sig,), deadline_s=5) == [sig]
        signal.raise_signal(sig)
        deadline = time.time() + 5
        while len(events) < 2 and time.time() < deadline:
            time.sleep(0.01)
        assert events == ["teardown", "previous"]
    finally:
        signal.signal(sig, old)


def test_an_ignored_signal_stays_ignored():
    from cascade.trainer.runner_generation import install_generation_shutdown

    sig = signal.SIGUSR2
    old = signal.signal(sig, signal.SIG_IGN)
    try:
        assert install_generation_shutdown(SimpleNamespace(), signals=(sig,)) == []
        assert signal.getsignal(sig) == signal.SIG_IGN
    finally:
        signal.signal(sig, old)


def test_runner_shutdown_uses_the_stage_ledger(tmp_path, monkeypatch):
    from cascade.trainer.runner_generation import _GenerationMixin

    seen = []
    monkeypatch.setattr(stage_mod, "terminate_owned_vms",
                        lambda ledger, providers: seen.append((ledger, providers)) or ["v"])
    stage = SimpleNamespace(ledger_path=tmp_path / "l.json", providers=("p",))
    runner = SimpleNamespace(__dict__={})
    runner.__dict__["_generation_stage_obj"] = stage
    assert _GenerationMixin._shutdown_generation_vms(runner) == ["v"]
    assert seen == [(tmp_path / "l.json", ["p"])]
    assert _GenerationMixin._shutdown_generation_vms(SimpleNamespace()) == []


# ── reuse on requeue ────────────────────────────────────────────────────────


def _go_seed3(st, tmp_path):
    # The fake run's corpus.json carries seed 3 (see test_generation_stage._corpus).
    (tmp_path / "repo").mkdir(exist_ok=True)
    return st.generate(round_id="77", hotkey=HK, uid=4, era=9, gen_ref="g@sha256:1",
                       repo_dir=tmp_path / "repo", seed=3)


def test_a_requeued_leg_reuses_its_verified_corpus_without_a_vm(tmp_path):
    prov = FakeProvider()
    st, idx = _churn_stage(tmp_path, prov, Clock())
    first = _go_seed3(st, tmp_path)
    again = _go_seed3(st, tmp_path)
    assert len(prov.launched) == 1
    assert again.index["corpus_digest"] == first.index["corpus_digest"] == idx["corpus_digest"]


@pytest.mark.parametrize("field,value", [("corpus_digest", "0" * 64), ("seed", 4),
                                         ("generator_ref", "g@sha256:2")])
def test_a_corpus_that_does_not_match_is_not_reused(tmp_path, field, value):
    prov = FakeProvider()
    st, idx = _churn_stage(tmp_path, prov, Clock())
    _go_seed3(st, tmp_path)
    probe = dict(round_id="77", hotkey=HK, gen_ref="g@sha256:1", seed=3)
    assert st.existing(**probe) is not None
    key = "corpora/round-77/5Fminer/corpus.json"
    tampered = json.loads(st.store.objs[key])
    tampered[field] = value
    st.store.objs[key] = json.dumps(tampered)
    assert st.existing(**probe) is None


# ── admission before generation ─────────────────────────────────────────────


def test_a_leg_refused_by_shared_admission_never_reaches_generation(cfg, tmp_path,
                                                                    monkeypatch):
    """The no_capacity refusals at 04:36–04:46 rented nothing: the shared
    compute admission gate is taken BEFORE the leg's corpus is built."""
    from cascade.trainer import runner_funded_legs as rfl
    from cascade.trainer.admission import FairSlots
    from cascade.trainer.faults import _FundedLegSkip
    from cascade.trainer.loop import ResolvedGenerator, TrainerRunner

    cfg = replace(cfg, round=replace(cfg.round, funded_field_cap=1))
    r = TrainerRunner(cfg=cfg, base_trainer=object(), work_root=tmp_path, trainer_spec="m:C")
    gate = r.__dict__["_funded_job_slots"] = FairSlots()
    r._leg_local.end_wall = time.time() - 1                  # latest safe start passed
    built = []
    monkeypatch.setattr(rfl, "leg_corpus", lambda *a, **k: built.append(a) or {})
    monkeypatch.setattr(r, "_leg_corpus_ready", lambda *a: False)
    failures = []
    monkeypatch.setattr(r, "_record_funded_failure",
                        lambda hk, msg, **kw: failures.append((hk, msg, kw)))
    contract = replace(cfg.training, corpus_mode="shards")
    with gate.slot(1), pytest.raises(_FundedLegSkip):
        r._run_funded_leg(None, ResolvedGenerator("miner", 1, "ref"),
                          SimpleNamespace(base_seed=1, generation_seed=2), 1,
                          contract, "", warm_start_ref=None)
    assert built == []
    assert failures and failures[0][2]["error_class"] == "no_capacity"


def test_latest_safe_start_uses_the_wall_admission_budgeted(cfg, tmp_path, monkeypatch):
    """2026-10-31: admission targeted a boundary with the measured SKU wall, but
    this deadline subtracted the 5 h max_train_seconds cap, so no-corpus legs
    hit 'past the latest safe start' for ~1.2 h of every 3 h cycle while the CPU
    tier idled. Both now budget the same wall."""
    from contextlib import contextmanager

    from cascade.trainer import runner_funded_legs as rfl
    from cascade.trainer.faults import _FundedLegSkip
    from cascade.trainer.loop import ResolvedGenerator, TrainerRunner

    r = TrainerRunner(cfg=cfg, base_trainer=object(), work_root=tmp_path, trainer_spec="m:C")
    seen = []

    class _Gate:
        @contextmanager
        def slot(self, limit, *, deadline=None, **kw):
            seen.append(deadline)
            yield False

    r.__dict__["_funded_job_slots"] = _Gate()
    from contextlib import nullcontext

    from cascade.trainer import admission as adm
    monkeypatch.setattr(adm, "owner_admission", lambda *a, **k: nullcontext())
    end = time.time() + 40_000
    r._leg_local.end_wall = end
    monkeypatch.setattr(rfl, "leg_corpus", lambda *a, **k: {})
    monkeypatch.setattr(r, "_leg_corpus_ready", lambda *a: False)
    monkeypatch.setattr(r, "_leg_wall_seconds", lambda sku=None: 13_600.0)
    monkeypatch.setattr(r, "_record_funded_failure", lambda *a, **k: None)
    contract = replace(cfg.training, corpus_mode="shards", max_train_seconds=18_000)
    with pytest.raises(_FundedLegSkip):
        r._run_funded_leg(None, ResolvedGenerator("miner", 1, "ref"),
                          SimpleNamespace(base_seed=1, generation_seed=2), 1,
                          contract, "", warm_start_ref=None)
    want = end - 13_600.0 - r.FUNDED_PUBLISH_MARGIN_SECONDS - r._generation_lead_seconds()
    assert seen == [pytest.approx(want)]


def test_generation_lead_is_the_window_plus_measured_overhead(cfg):
    from cascade.trainer.runner_generation import _GenerationMixin

    assert _GenerationMixin.GENERATION_OVERHEAD_SECONDS == 1800.0
