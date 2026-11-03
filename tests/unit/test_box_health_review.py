"""Second review of #417 (N1-N6, F7, F9, the F1 residual): the box-health
breaker never fails closed on a plain capacity shortfall, never fails open,
never holds the king for a non-disk halt, and a lost kept pod is decided
BEFORE any rent on the miner's key."""
from __future__ import annotations

import errno
import json
from dataclasses import replace as dc_replace
from types import SimpleNamespace

import pytest

from cascade.funding.operator_approvals import OperatorApprovals
from cascade.funding.queue import FundedQueue
from cascade.provision import funded as funded_mod
from cascade.shared.config import RoundConfig
from cascade.trainer import box_health, harvest_guard, money_guard, runner_common
from cascade.trainer.box_health import (
    HARVEST,
    LOCAL_DISK,
    STORAGE,
    BoxHealth,
    RentsHalted,
    rent_halt_refusal,
)
from cascade.trainer.faults import _FundedLegSkip
from tests.unit.test_funded_pod_wiring import _challenger, _rent_ok, _runner, _vault

GIB = 1 << 30


def _bare(tmp_path, **kw):
    rnd = RoundConfig(funded_mode="required", min_free_disk_gb=30.0,
                      expected_checkpoint_gb=15.0, **kw)
    return SimpleNamespace(cfg=SimpleNamespace(round=rnd), work_root=tmp_path)


def _free(monkeypatch, gib):
    from cascade.shared import disk_space

    for mod in (runner_common, disk_space):
        monkeypatch.setattr(mod, "DISK_USAGE",
                            lambda p: SimpleNamespace(total=1, used=0, free=int(gib * GIB)))


# ── N2: a shortfall refuses that rent only; only real disk errors halt ──────


def test_a_reservation_shortfall_refuses_only_that_rent_and_never_halts(tmp_path, monkeypatch):
    r = _bare(tmp_path)
    _free(monkeypatch, 40)                                   # < 30 + 15
    assert runner_common.disk_guard_refusal(r).startswith("operator disk")
    assert BoxHealth.for_runner(r).halt() is None
    assert rent_halt_refusal(r, king=True) == ""             # the king is untouched


def test_a_failed_canary_halts_and_holds_the_king(tmp_path, monkeypatch):
    r = _bare(tmp_path)
    _free(monkeypatch, 500)
    monkeypatch.setattr(box_health, "harvest_canary", lambda runner: "canary failed: EIO")
    assert runner_common.disk_guard_refusal(r)
    assert BoxHealth.for_runner(r).halt()["cls"] == LOCAL_DISK
    r.__dict__.pop("_box_health_checked", None)
    assert rent_halt_refusal(r, king=True)                   # disk ⇒ the king waits too


def test_the_health_check_sizes_for_pod_holding_legs_not_the_whole_queue(tmp_path, monkeypatch):
    r = _bare(tmp_path)
    q = FundedQueue(tmp_path / "funded_queue.json")
    for i in range(20):                                      # a deep queue, nothing holds a pod
        q.add(f"hk{i}", f"r@{i}", 10 + i)
    _free(monkeypatch, 50)                                   # 30 + 15 x 1 fits; x 20 would not
    BoxHealth.for_runner(r).record(LOCAL_DISK, "hk0", "ENOSPC")
    assert rent_halt_refusal(r) == "" and BoxHealth.for_runner(r).halt() is None


# ── N3: the halt fails closed — storage probe, latch ────────────────────────


def test_a_storage_halt_clears_only_on_the_registry_probe(tmp_path, monkeypatch):
    r = _bare(tmp_path)
    _free(monkeypatch, 500)                                  # the DISK is fine
    bh = BoxHealth.for_runner(r)
    bh.record(STORAGE, "hkA", "registry push failed")
    bh.record(STORAGE, "hkB", "registry push failed")
    assert bh.halt()["cls"] == STORAGE
    r._storage_probe = lambda: "storage canary failed: 503"
    assert rent_halt_refusal(r)                              # disk check alone never clears it
    assert rent_halt_refusal(r, king=True) == ""             # ... and never holds the king (N4)
    r.__dict__.pop("_box_health_checked", None)
    r._storage_probe = lambda: ""
    assert rent_halt_refusal(r) == "" and bh.halt() is None


def test_the_halt_holds_in_process_when_its_state_write_fails(tmp_path, monkeypatch):
    import cascade.shared.atomic as atomic

    r = _bare(tmp_path)

    def full(*a, **k):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(atomic, "write_text_atomic", full)
    bh = BoxHealth.for_runner(r)
    bh.record(LOCAL_DISK, "hkA", "ENOSPC")
    assert not (tmp_path / "box_health.json").exists()
    assert bh.halt() is not None                             # latched for this process
    monkeypatch.undo()
    bh.clear("test")


# ── N4: an own-key king leg is held only by a disk halt ─────────────────────


def test_an_own_key_king_leg_ignores_a_harvest_halt(tmp_path, monkeypatch):
    import threading

    r = _bare(tmp_path)
    _free(monkeypatch, 500)
    bh = BoxHealth.for_runner(r)
    bh.record(STORAGE, "hkA", "x")
    bh.record(STORAGE, "hkB", "x")
    r._storage_probe = lambda: "storage down"
    loc = r.__dict__.setdefault("_leg_local_obj", threading.local())
    loc.funded_is_king = True
    assert runner_common.disk_guard_refusal(r) == ""
    loc.funded_is_king = False
    r.__dict__.pop("_box_health_checked", None)
    assert runner_common.disk_guard_refusal(r)


# ── N1 / N6: a lost kept pod is decided BEFORE any rent ─────────────────────


def _lost_runner(tmp_path, monkeypatch, *, live, kept=True):
    r = _runner(tmp_path)
    _vault(tmp_path, "hkA")
    q = FundedQueue(tmp_path / "funded_queue.json")
    q.add("hkA", "r@1", 10)
    q.mark_in_flight("hkA", "r@1", target_boundary=1, era_index=1, started_block=1)
    r._funded_queue = lambda: q
    r._ledger_add(_rent_ok("hkA").pod)                        # the ledgered (kept) pod
    BoxHealth.for_runner(r).note_leg("hkA", {"trained": True, "kept": kept,
                                             "ref": _challenger("hkA").ref, "cls": LOCAL_DISK})
    r._payer_pod_liveness = lambda pod_id: live
    rents = []
    monkeypatch.setattr(funded_mod, "rent_funded_pod",
                        lambda **kw: rents.append(kw) or dc_replace(_rent_ok(), adopted=False))
    return r, q, rents


@pytest.mark.parametrize("kept,live", [(True, (None, True)), (False, (None, True)),
                                       (False, (object(), True))])
def test_a_lost_run_parks_for_approval_before_any_rent(tmp_path, monkeypatch, kept, live):
    r, q, rents = _lost_runner(tmp_path, monkeypatch, live=live, kept=kept)
    with pytest.raises(_FundedLegSkip):
        r._rent_funded_host("777", _challenger("hkA"))
    assert rents == []                                        # the miner paid NOTHING more
    assert r._funded_leg_failures["hkA"][2] == "awaiting_operator_approval"
    (req,) = OperatorApprovals.for_work_root(tmp_path).pending()
    assert req["kind"] == "retrain_after_loss"


def test_an_approved_retrain_rents_once_as_an_ordinary_leg(tmp_path, monkeypatch):
    r, q, rents = _lost_runner(tmp_path, monkeypatch, live=(None, True))
    with pytest.raises(_FundedLegSkip):
        r._rent_funded_host("777", _challenger("hkA"))
    q.requeue("hkA", error="awaiting", error_class="awaiting_operator_approval",
              burn_attempt=False)
    from cascade.trainer.operator_legs_cli import main as cli

    assert cli(["approve-operator-leg", "hkA", "--work-root", str(tmp_path)]) == 0
    q.mark_in_flight("hkA", "r@1", target_boundary=1, era_index=1, started_block=2)
    host, pod = r._rent_funded_host("777", _challenger("hkA"))
    assert len(rents) == 1 and host.host == "10.9.9.9"        # one rent, used normally
    assert BoxHealth.for_runner(r).leg("hkA") is None


def test_unknown_kept_pod_liveness_neither_rents_nor_parks(tmp_path, monkeypatch):
    r, q, rents = _lost_runner(tmp_path, monkeypatch, live=(None, False))
    with pytest.raises(_FundedLegSkip):
        r._rent_funded_host("777", _challenger("hkA"))
    assert rents == [] and OperatorApprovals.for_work_root(tmp_path).pending() == []
    assert r._funded_leg_failures["hkA"][2] == "operator_transient"


def test_a_stale_ledger_row_does_not_bypass_the_halt(tmp_path, monkeypatch):
    r = _runner(tmp_path)
    _vault(tmp_path, "hkA")
    r._ledger_add(_rent_ok("hkA").pod)                        # stale write-ahead row
    r._payer_pod_liveness = lambda pod_id: (None, True)       # not live
    BoxHealth.for_runner(r).record(LOCAL_DISK, "hkZ", "ENOSPC")
    monkeypatch.setattr(box_health, "harvest_canary", lambda runner: "canary failed: EROFS")
    rents = []
    monkeypatch.setattr(funded_mod, "rent_funded_pod", lambda **kw: rents.append(kw))
    with pytest.raises(_FundedLegSkip):
        r._rent_funded_host("777", _challenger("hkA"))
    assert rents == [] and r._funded_leg_failures["hkA"][2] == "operator_disk"


# ── F1 residual: a liveness blip is never "gone" ────────────────────────────


def test_a_liveness_blip_is_retried_and_never_reads_as_gone():
    calls, sleeps = [], []

    def live(pod_id):
        calls.append(pod_id)
        if len(calls) < 3:
            raise RuntimeError("lium API 502")
        return SimpleNamespace(ip="1.2.3.4", ssh_port=22)

    addr, definitive = runner_common.probe_pod_live(live, "p-0", sleep=sleeps.append)
    assert addr is not None and definitive and len(calls) == 3 and len(sleeps) == 2

    def down(pod_id):
        raise RuntimeError("lium API 502")

    assert runner_common.probe_pod_live(down, "p-0", sleep=lambda s: None) == (None, False)
    assert runner_common.probe_pod_live(lambda p: None, "p-0") == (None, True)


# ── N5: any successful harvest ends a lost-run record ───────────────────────


def test_the_r2_fallback_clears_the_leg_record(tmp_path):
    r = _bare(tmp_path, harvest_disk_retries=0)
    r.cfg.round = RoundConfig(min_free_disk_gb=0.0, expected_checkpoint_gb=0.0,
                              harvest_disk_retries=0)
    r._harvest_retry_sleep = lambda s: None
    BoxHealth.for_runner(r).note_leg("hkA", {"trained": True, "kept": True})

    def attempt():
        raise OSError(errno.ENOSPC, "No space left on device")

    assert harvest_guard.guarded_harvest(r, tmp_path / "d", attempt, label="hkA",
                                         hotkey="hkA", r2_fallback=lambda: "r2") == "r2"
    assert BoxHealth.for_runner(r).leg("hkA") is None


# ── F9: a denial binds to its kind ──────────────────────────────────────────


def test_a_make_good_denial_never_decides_a_later_retrain(tmp_path):
    s = OperatorApprovals(tmp_path / "a.json")
    s.request("q", "hkA", ref="r@1", attempt_id="A", kind="make_good")
    s.decide("hkA", approve=False)
    assert s.check("q", "hkA", ref="r@1", attempt_id="B", kind="make_good") == "denied"
    assert s.check("q", "hkA", ref="r@1", attempt_id="B", kind="retrain_after_loss") == "none"
    # and the new kind can be requested and decided on its own
    assert s.request("q", "hkA", ref="r@1", attempt_id="B", kind="retrain_after_loss")
    assert s.decide("hkA", approve=True)["kind"] == "retrain_after_loss"


def test_unverifiable_evidence_is_never_evicted(tmp_path):
    from cascade.shared import disk_space

    seed = tmp_path / "_funded_harvest" / "777"
    item = seed / "hkA--j"
    item.mkdir(parents=True)
    (item / "w").write_bytes(b"x")
    (seed / "hkA--j.unverifiable.json").write_text(json.dumps({"box_copy": str(item)}))
    src = disk_space.harvest_source(tmp_path / "_funded_harvest", set)
    assert not src.reclaimable(item)


# ── third review: R1-R6 ─────────────────────────────────────────────────────


def test_an_owner_resume_from_another_process_clears_this_one(tmp_path):
    """R1: a persisted halt is never latched — `resume-rents` (another process,
    another instance) clears it here too."""
    from cascade.trainer.operator_legs_cli import main as cli

    r = _bare(tmp_path)
    BoxHealth.for_runner(r).record(LOCAL_DISK, "hkA", "ENOSPC")
    assert BoxHealth.for_runner(r).halt() is not None
    assert cli(["resume-rents", "--reason", "disk freed", "--work-root", str(tmp_path)]) == 0
    assert BoxHealth.for_runner(r).halt() is None
    assert rent_halt_refusal(r, king=True) == ""


def test_an_unpersisted_latch_yields_to_a_later_clear(tmp_path, monkeypatch):
    import cascade.shared.atomic as atomic

    r = _bare(tmp_path)
    real = atomic.write_text_atomic

    def full(*a, **k):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(atomic, "write_text_atomic", full)
    BoxHealth.for_runner(r).record(LOCAL_DISK, "hkA", "ENOSPC")
    assert BoxHealth.for_runner(r).halt() is not None          # latched: the save failed
    monkeypatch.setattr(atomic, "write_text_atomic", real)
    other = BoxHealth(tmp_path / "box_health.json")
    other.record(LOCAL_DISK, "hkB", "ENOSPC")                  # now persisted
    assert other.clear("owner: resumed") is True
    assert BoxHealth.for_runner(r).halt() is None


def test_the_king_waits_out_a_halt_without_probing_or_failing(tmp_path, monkeypatch):
    """R2: no provider call while halted, a sleep per pass, and at the latest
    safe start a RentsHalted the era loop never counts or rotates on."""
    from cascade.trainer import rolling
    from cascade.trainer.runner_operator_pods import _wait_out_rent_halt

    answers = ["halted", "halted", ""]
    monkeypatch.setattr(box_health, "rent_halt_refusal", lambda runner, king=False: answers.pop(0))
    sleeps, removed = [], []
    r = SimpleNamespace(_rent_wait_sleep=sleeps.append, _rent_wait_now=lambda: 0.0,
                        _funded_rent_wait_deadline=lambda: 100.0,
                        FUNDED_RENT_RETRY_SECONDS=60.0, _ledger_remove=removed.append)
    _wait_out_rent_halt(r, "king-x")
    assert sleeps == [60.0, 60.0] and removed == []
    monkeypatch.setattr(box_health, "rent_halt_refusal", lambda runner, king=False: "halted")
    r._rent_wait_now = lambda: 101.0
    with pytest.raises(RentsHalted) as ei:
        _wait_out_rent_halt(r, "king-x")
    assert removed == ["king-x-0"]
    wrapped = RuntimeError("king leg failed")
    wrapped.__cause__ = ei.value
    assert rolling._rents_halted(wrapped) and not rolling._rents_halted(RuntimeError("x"))


def test_the_registry_probe_pushes_one_fixed_canary_once_with_a_deadline(tmp_path, monkeypatch):
    """R3/D1: the STORAGE probe pushes the SAME tiny file to ONE fixed Hub repo
    (content-addressed: no litter), one attempt, a short deadline — never the
    retrying checkpoint upload, never HF / R2."""
    import time as _time
    from pathlib import Path

    import cascade.shared.hippius as hip

    monkeypatch.setattr(hip, "upload_dir_to_hub",
                        lambda *a, **k: pytest.fail("never the retrying upload"))
    monkeypatch.setattr(hip, "upload_dir_to_hf",
                        lambda *a, **k: pytest.fail("the probe must never touch HF"))
    pushes = []
    r = _bare(tmp_path)
    r.hub = lambda: SimpleNamespace(namespace="cascade")
    r._hub_canary_push = lambda d, repo: pushes.append(
        (repo, sorted((p.name, p.read_text()) for p in Path(d).iterdir())))
    assert box_health.storage_probe(r) == "" and box_health.storage_probe(r) == ""
    assert pushes[0] == pushes[1] and pushes[0][0] == "cascade/box-health-canary"

    def down(d, repo):
        raise hip.StorageError("hub 503")

    r._hub_canary_push = down
    assert "failed" in box_health.storage_probe(r)
    monkeypatch.setattr(box_health, "STORAGE_PROBE_TIMEOUT_SECONDS", 0.2)
    r._hub_canary_push = lambda d, repo: _time.sleep(2)
    t0 = _time.monotonic()
    assert "timed out" in box_health.storage_probe(r) and _time.monotonic() - t0 < 1.5
    assert "resume-rents" in box_health.storage_probe(_bare(tmp_path / "nohub"))


def test_the_in_lock_halt_check_never_runs_the_probe(tmp_path, monkeypatch):
    """D1: under the rent lock only the standing verdict is read."""
    r = _bare(tmp_path)
    BoxHealth.for_runner(r).record(STORAGE, "hkA", "x")
    BoxHealth.for_runner(r).record(STORAGE, "hkB", "x")
    monkeypatch.setattr(box_health, "health_check",
                        lambda *a, **k: pytest.fail("no probe under the rent lock"))
    assert rent_halt_refusal(r, cached_only=True)
    assert rent_halt_refusal(r, king=True, cached_only=True) == ""


def test_an_own_key_king_leg_held_by_a_disk_halt_raises_rents_halted(tmp_path, monkeypatch):
    """D2: never a _FundedLegSkip the era loop would count and rotate on."""
    import threading

    r = _runner(tmp_path)
    _vault(tmp_path, "hkA")
    loc = r.__dict__.setdefault("_leg_local_obj", threading.local())
    loc.funded_is_king = True
    BoxHealth.for_runner(r).record(LOCAL_DISK, "hkZ", "ENOSPC")
    monkeypatch.setattr(box_health, "harvest_canary", lambda runner: "canary failed: EIO")
    rents = []
    monkeypatch.setattr(funded_mod, "rent_funded_pod", lambda **kw: rents.append(kw))
    with pytest.raises(RentsHalted):
        r._rent_funded_host("777", _challenger("hkA"))
    assert rents == [] and "hkA" not in r._funded_leg_failures


def test_a_harvest_halt_clears_on_the_disk_check(tmp_path, monkeypatch):
    r = _bare(tmp_path)
    _free(monkeypatch, 500)
    bh = BoxHealth.for_runner(r)
    bh.record(HARVEST, "hkA", "x")
    bh.record(HARVEST, "hkB", "x")
    r._storage_probe = lambda: pytest.fail("a harvest halt is a disk check, not storage")
    assert rent_halt_refusal(r) == "" and bh.halt() is None


def test_two_approved_legs_fit_the_daily_cap_at_the_landed_price(tmp_path):
    """R4: project with the best live offer, book the LANDED $/h — the $200
    cap admits more than one leg a day."""
    from tests.unit.test_money_safety import GEN, HK, _approve, _gate_runner

    r, q = _gate_runner(tmp_path)                              # ceiling $20/h, 3.25 h
    r._operator_offer_rate = lambda skus: 12.0
    total = 0.0
    for _ in range(2):
        _approve(r, q)
        assert money_guard.operator_leg_gate(r, GEN, kind="expired_key", rent=True) is None
        money_guard.note_operator_launch(r, HK, landed_rate=11.0)
        total += 11.0 * 3.25
        q.requeue(HK, error="lost", error_class="no_capacity", burn_attempt=False)
        q.mark_in_flight(HK, q.get(HK).ref, target_boundary=1, era_index=1, started_block=9)
    rows = money_guard.OperatorSpend.for_runner(r).rows()
    assert money_guard.OperatorSpend.for_runner(r).total() == pytest.approx(total)
    assert rows[0]["projected_usd"] == pytest.approx(12.0 * 3.25)


def test_off_path_disk_errors_halt_and_a_miners_artifact_is_not_ours(tmp_path):
    from cascade.shared.hippius import StorageError
    from cascade.trainer.faults import StorageFault

    box_health.note_disk_error(tmp_path, OSError(errno.ENOSPC, "No space"), "funded pod ledger")
    assert BoxHealth(tmp_path / "box_health.json").halt()["cls"] == LOCAL_DISK
    assert box_health.our_side_fault_class(StorageError("generator artifact unreachable")) is None
    assert box_health.our_side_fault_class(StorageFault("upload failed")) == STORAGE


def test_unknown_liveness_without_a_lost_leg_still_gets_every_guard(tmp_path, monkeypatch):
    r = _runner(tmp_path)
    _vault(tmp_path, "hkA")
    r._ledger_add(_rent_ok("hkA").pod)
    r._payer_pod_liveness = lambda pod_id: (None, False)      # unknown (no key / provider)
    BoxHealth.for_runner(r).record(LOCAL_DISK, "hkZ", "ENOSPC")
    monkeypatch.setattr(box_health, "harvest_canary", lambda runner: "canary failed: EIO")
    rents = []
    monkeypatch.setattr(funded_mod, "rent_funded_pod", lambda **kw: rents.append(kw))
    with pytest.raises(_FundedLegSkip):
        r._rent_funded_host("777", _challenger("hkA"))
    assert rents == [] and r._funded_leg_failures["hkA"][2] == "operator_disk"


def test_prune_never_ages_out_mismatch_evidence(tmp_path):
    import os
    import time

    seed = tmp_path / "_funded_harvest" / "777"
    item = seed / "hkA--j"
    item.mkdir(parents=True)
    proof = seed / "hkA--j.unverifiable.json"
    proof.write_text("{}")
    old = time.time() - 10 * 86400
    for p in (item, proof):
        os.utime(p, (old, old))
    assert runner_common.prune_harvest_dirs(tmp_path) == 0
    assert item.is_dir() and proof.is_file()


def test_an_operator_leg_adopt_blip_neither_rents_nor_parks(tmp_path, monkeypatch):
    """F1 residual, the operator-leg adopt path itself: the provider's
    liveness probe errors — retried; still unknown ⇒ requeued transient with
    the ledger row kept: no launch, no approval request."""
    import cascade.provision.core as core_mod
    from cascade.provision.state import PodInstance
    from tests.unit.test_funded_expired_key_operator_rent import _fake_lium, _rent_runner

    launched, _torn = _fake_lium(monkeypatch)
    calls = []

    def flaky(self, pod_id):
        calls.append(pod_id)
        raise RuntimeError("lium API 502")

    monkeypatch.setattr(core_mod.LiumProvider, "live_pod_address", flaky)
    r = _rent_runner(tmp_path, monkeypatch)
    r._rent_wait_sleep = lambda s: None
    pod_id = "cascade-n91-777-funded-hka-op-0"
    r._ledger_add(PodInstance(provider="lium", instance_id=pod_id, stage="funded",
                              rented_at_iso="", payer_hotkey=""))
    with pytest.raises(_FundedLegSkip):
        r._rent_operator_leg_host("777", _challenger("hkA"))
    assert launched == [] and len(calls) == 3
    assert r._funded_leg_failures["hkA"][2] == "operator_transient"
    assert any(p.instance_id == pod_id for p in r._load_funded_ledger())   # kept
    assert OperatorApprovals.for_work_root(tmp_path).pending() == []

    # the blip passes: the SAME pod is adopted, nothing launched
    from cascade.provision.core import PodAddress

    seq = iter([RuntimeError("502"), PodAddress(ip="7.7.7.7", ssh_port=40001)])

    def recovers(self, pid):
        v = next(seq)
        if isinstance(v, Exception):
            raise v
        return v

    monkeypatch.setattr(core_mod.LiumProvider, "live_pod_address", recovers)
    host, got = r._rent_operator_leg_host("777", _challenger("hkA"))
    assert (host.host, got, launched) == ("7.7.7.7", pod_id, [])
