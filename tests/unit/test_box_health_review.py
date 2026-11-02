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
from cascade.trainer import box_health, harvest_guard, runner_common
from cascade.trainer.box_health import HARVEST, LOCAL_DISK, BoxHealth, rent_halt_refusal
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


def test_a_harvest_halt_clears_only_on_the_storage_probe(tmp_path, monkeypatch):
    r = _bare(tmp_path)
    _free(monkeypatch, 500)                                  # the DISK is fine
    bh = BoxHealth.for_runner(r)
    bh.record(HARVEST, "hkA", "harvest fault")
    bh.record(HARVEST, "hkB", "harvest fault")
    assert bh.halt()["cls"] == HARVEST
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
    bh.record(HARVEST, "hkA", "x")
    bh.record(HARVEST, "hkB", "x")
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
