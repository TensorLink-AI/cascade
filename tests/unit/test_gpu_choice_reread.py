"""A miner's re-fund GPU change reaches a leg already WAITING for capacity
(incident 2026-11-03: 5G6jNi6Y re-funded 8xH200 → 8xH100 at 10:32 UTC, the
queue entry was updated, but the in-flight leg kept polling for 8xH200 until
its 13:39 deadline — only a trainer restart picked the new choice up).

Every provider call is mocked: no rentals, no network."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import cascade.provision.funded as funded_mod
from cascade.funding.gpu_choice import GpuChoice
from cascade.trainer.faults import _FundedLegSkip
from cascade.trainer.loop import TrainerRunner
from cascade.trainer.runner_funded_pods import _changed_gpu_choice
from tests.unit.test_funded_pod_wiring import REF as WREF
from tests.unit.test_funded_pod_wiring import _rent_ok
from tests.unit.test_miner_chosen_gpu import _challenger, _no_capacity, _rent_runner

H100_WALL = 6000
RTX4090_WALL = 13500


def _in_flight(r, choice):
    q = r._funded_queue()
    q.add("hkA", WREF, 10, gpu_choice=choice)
    assert q.mark_in_flight("hkA", WREF, target_boundary=1, era_index=1, started_block=1)
    return q


def test_a_re_fund_mid_wait_switches_the_next_rent_to_the_new_type_and_cap(
        tmp_path, monkeypatch):
    r = _rent_runner(tmp_path)
    q = _in_flight(r, GpuChoice("L40S", False, 30.0))
    outcomes = [_no_capacity(), _rent_ok()]
    calls, waits = [], []
    monkeypatch.setattr(funded_mod, "rent_funded_pod",
                        lambda **kw: calls.append(kw) or outcomes.pop(0))

    def wait(skus, **kw):
        waits.append((skus, kw))
        assert kw["choice_changed"]() is False                # nothing changed yet
        # the re-fund lands while the leg polls (intake: "already-queued")
        assert q.add("hkA", WREF, 10, gpu_choice=GpuChoice("H100", False, 20.0)) \
            == "already-queued"
        assert kw["choice_changed"]() is True
        return "gpu_changed"

    r._wait_for_funded_capacity = wait
    r._rent_funded_host("777", _challenger())
    assert [kw["skus"] for kw in calls] == [("L40S",), ("H100",)]
    assert calls[0]["price_caps"]["max_price_per_hour"] == 30.0
    assert calls[1]["price_caps"]["max_price_per_hour"] == 20.0
    assert calls[1]["price_caps"]["max_leg_cost_usd"] == 0.0     # the miner's cap, as before
    (skus, kw), = waits
    assert skus == ("L40S",) and kw["price_caps"]["max_price_per_hour"] == 30.0


def test_the_real_wait_leaves_on_a_changed_choice_and_survives_a_raising_check(tmp_path):
    from tests.unit.test_funded_rent_wait import _arm_wait

    r = _rent_runner(tmp_path)
    _arm_wait(r, deadline_offsets=3600, capacity_seq=[0, 0, 0, 0])
    seq = [False, True]
    assert r._wait_for_funded_capacity("H100", describe="x",
                                       choice_changed=lambda: seq.pop(0)) == "gpu_changed"
    assert seq == []                                     # polled once, then left

    def boom():
        raise OSError("torn queue file")

    _arm_wait(r, deadline_offsets=3600, capacity_seq=[0, 3])
    assert r._wait_for_funded_capacity("H100", describe="x", choice_changed=boom) is True


def test_an_unreadable_queue_keeps_the_current_choice(tmp_path, monkeypatch):
    cur = GpuChoice("8xH200", False, 30.0)

    def broken():
        raise ValueError("funded_queue.json: not json")

    r = SimpleNamespace(_funded_queue=broken)
    assert _changed_gpu_choice(r, "hkA", WREF, "", cur) is None
    r = SimpleNamespace(_funded_queue=lambda: SimpleNamespace(
        get=lambda hk: (_ for _ in ()).throw(OSError("EIO"))))
    assert _changed_gpu_choice(r, "hkA", WREF, "", cur) is None

    # and through the rent loop: the queue breaks after the leg started
    rr = _rent_runner(tmp_path)
    _in_flight(rr, GpuChoice("L40S", False, 2.0))
    outcomes = [_no_capacity(), _rent_ok()]
    calls = []
    monkeypatch.setattr(funded_mod, "rent_funded_pod",
                        lambda **kw: calls.append(kw) or outcomes.pop(0))
    import cascade.trainer.runner_funded_pods as rfp

    real = rfp._changed_gpu_choice

    def wait(skus, **kw):
        # every re-read now fails inside the helper: the wait keeps its type
        monkeypatch.setattr(rfp, "_changed_gpu_choice",
                            lambda runner, *a: real(SimpleNamespace(_funded_queue=broken), *a))
        assert kw["choice_changed"]() is False
        return True

    rr._wait_for_funded_capacity = wait
    rr._rent_funded_host("777", _challenger())
    assert [kw["skus"] for kw in calls] == [("L40S",), ("L40S",)]
    assert calls[1]["price_caps"]["max_price_per_hour"] == 2.0


def test_only_the_same_in_flight_attempt_counts(tmp_path):
    r = _rent_runner(tmp_path)
    q = _in_flight(r, GpuChoice("L40S", False, 2.0))
    attempt = q.get("hkA").attempt_id
    cur = GpuChoice("L40S", False, 2.0)
    assert _changed_gpu_choice(r, "hkA", WREF, attempt, cur) is None       # unchanged
    q.add("hkA", WREF, 10, gpu_choice=GpuChoice("l40s", False, 2.0))
    assert _changed_gpu_choice(r, "hkA", WREF, attempt, cur) is None       # case only
    q.add("hkA", WREF, 10, gpu_choice=GpuChoice("H100", True, 4.0))
    assert _changed_gpu_choice(r, "hkA", WREF, attempt, cur) == GpuChoice("H100", True, 4.0)
    assert _changed_gpu_choice(r, "hkA", WREF, "another-attempt", cur) is None
    assert _changed_gpu_choice(r, "hkA", "other@sha256:" + "e" * 64, attempt, cur) is None
    # a re-fund of a NEW ref queues behind the flight: its choice is not this leg's
    q.add("hkA", "other@sha256:" + "e" * 64, 11, gpu_choice=GpuChoice("A6000"))
    assert _changed_gpu_choice(r, "hkA", WREF, attempt, cur) is None


def _arm_target(r, *, seconds_left):
    """A leg with a target settlement ``seconds_left`` away from a fake now."""
    now = 1_000_000.0
    leg = SimpleNamespace(end_wall=now + seconds_left,
                          funded_contract=SimpleNamespace(max_train_seconds=18000))
    r._leg_local_obj = leg
    r._leg_local = leg
    r._rent_wait_now = lambda: now
    r.FUNDED_PUBLISH_MARGIN_SECONDS = TrainerRunner.FUNDED_PUBLISH_MARGIN_SECONDS
    return now


def test_a_new_type_that_no_longer_fits_the_target_is_never_rented(tmp_path, monkeypatch):
    r = _rent_runner(tmp_path)
    # H100 (6000 s) still lands before the target; RTX4090 (13500 s) cannot
    _arm_target(r, seconds_left=H100_WALL + TrainerRunner.FUNDED_PUBLISH_MARGIN_SECONDS
                + 3000)
    q = _in_flight(r, GpuChoice("H100", False, 3.0))
    calls = []
    monkeypatch.setattr(funded_mod, "rent_funded_pod",
                        lambda **kw: calls.append(kw) or _no_capacity())

    def wait(skus, **kw):
        q.add("hkA", WREF, 10, gpu_choice=GpuChoice("RTX4090", False, 1.0))
        return "gpu_changed"

    r._wait_for_funded_capacity = wait
    with pytest.raises(_FundedLegSkip):
        r._rent_funded_host("777", _challenger())
    assert [kw["skus"] for kw in calls] == [("H100",)]           # RTX4090 never rented
    msg, miner_fault, error_class, burn = r._funded_leg_failures["hkA"]
    assert "doomed leg" in msg and "RTX4090" in msg
    assert (miner_fault, error_class, burn) == (False, "no_capacity", False)
    assert r._load_funded_ledger() == []


def test_a_leg_without_a_miner_choice_is_never_re_steered(tmp_path, monkeypatch):
    r = _rent_runner(tmp_path)
    q = _in_flight(r, GpuChoice())                               # default path
    outcomes = [_no_capacity(), _rent_ok()]
    calls = []
    monkeypatch.setattr(funded_mod, "rent_funded_pod",
                        lambda **kw: calls.append(kw) or outcomes.pop(0))

    def wait(skus, **kw):
        assert "choice_changed" not in kw
        q.add("hkA", WREF, 10, gpu_choice=GpuChoice("H100", False, 3.0))
        return True

    r._wait_for_funded_capacity = wait
    r._rent_funded_host("777", _challenger())
    assert [kw["skus"] for kw in calls] == [calls[0]["skus"]] * 2
    assert calls[0]["skus"] != ("H100",)                         # default path throughout


def test_an_explicit_choice_is_never_re_steered(tmp_path, monkeypatch):
    # the king's default and lane B / marketcast choices are passed in, not
    # read from the lane-A entry: a lane-A re-fund never moves them
    r = _rent_runner(tmp_path)
    q = _in_flight(r, GpuChoice("L40S", False, 2.0))
    outcomes = [_no_capacity(), _rent_ok()]
    calls = []
    monkeypatch.setattr(funded_mod, "rent_funded_pod",
                        lambda **kw: calls.append(kw) or outcomes.pop(0))

    def wait(skus, **kw):
        assert "choice_changed" not in kw
        q.add("hkA", WREF, 10, gpu_choice=GpuChoice("RTX4090", False, 1.0))
        return True

    r._wait_for_funded_capacity = wait
    r._rent_funded_host("777", _challenger(), gpu_choice=GpuChoice("H100", False, 3.0))
    assert [kw["skus"] for kw in calls] == [("H100",), ("H100",)]
