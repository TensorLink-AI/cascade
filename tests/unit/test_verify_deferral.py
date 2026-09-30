"""Trainer-side payer-bench verification: defer instead of drop (2026-09-30),
and no king-pod teardown under a running sweep."""
from __future__ import annotations

import dataclasses
import time
from types import SimpleNamespace

from cascade.shared.manifest import BenchScores, TrainedEntry, format_trained_pointer
from cascade.trainer import rolling as R
from cascade.trainer.bench_hook import host_bench_lock
from cascade.trainer.loop import TrainerRunner
from cascade.trainer.remote import RemoteHost

KING_POD = RemoteHost(name="king", host="8.8.8.8", workdir="/root/cascade", cuda_device="0")
SCORES = BenchScores(gifteval_crps=0.54, gifteval_mase=0.79, boom_crps=0.39, boom_mase=0.65,
                     time_crps=0.57, time_mase=0.73)


def _entry(hk, role):
    ptr = format_trained_pointer(f"cascade/ckpt-{hk.lower()}@sha256:" + "b" * 64)
    return TrainedEntry(hk, 1, role, f"{hk.lower()}/g@sha256:" + "a" * 64, ptr, "d", 100)


def _fake(isolated=True, verify=("ok", None)):
    host = RemoteHost(name="payer", host="9.9.9.9", workdir="/root/cascade", cuda_device="0",
                      isolated=isolated)
    contract = SimpleNamespace(arch_preset="toto2-4m")
    f = SimpleNamespace(
        cfg=SimpleNamespace(throne_contracts=lambda: [contract],
                            telemetry=SimpleNamespace(funded_bench_verify_top=1,
                                                      funded_bench_verify_tolerance=0.02)),
        _final_role_hosts={("challenger", "toto2-4m", "ALFA"): host,
                           ("king", "toto2-4m", "KING"): KING_POD},
        cascade_bench_plan=object(),
        _remote_bench_scores=lambda *a, **k: SCORES,
        verify_calls=[],
    )
    f._rolling_verify_host = lambda king, era: TrainerRunner._rolling_verify_host(f, king, era)

    def _verify(entry, payer, king, era):
        f.verify_calls.append(entry.miner_hotkey)
        return verify
    f._rolling_verify_payer = _verify
    return f


ERA = SimpleNamespace(base_seed=7, index=3)


def test_no_king_leg_yet_defers_instead_of_dropping():
    f = _fake()
    out = TrainerRunner._rolling_bench_challenger(f, _entry("ALFA", "challenger"), None, ERA)
    assert R.VERIFY_PENDING in out and out[R.VERIFY_PENDING]["gifteval_crps"] == 0.54
    assert f.verify_calls == []


def test_busy_king_pod_defers_instead_of_preempting():
    f = _fake()
    lock = host_bench_lock(KING_POD)
    lock.acquire()
    try:
        out = TrainerRunner._rolling_bench_challenger(f, _entry("ALFA", "challenger"),
                                                      _entry("KING", "king"), ERA)
    finally:
        lock.release()
    assert out[R.VERIFY_PENDING]["_reason"] == "king pod busy" and f.verify_calls == []


def test_verified_now_returns_operator_numbers_and_forged_is_dropped():
    ours = dict(dataclasses.asdict(SCORES), gifteval_crps=0.545)
    f = _fake(verify=("ok", ours))
    assert TrainerRunner._rolling_bench_challenger(
        f, _entry("ALFA", "challenger"), _entry("KING", "king"), ERA) == ours
    f = _fake(verify=("forged", None))
    assert TrainerRunner._rolling_bench_challenger(
        f, _entry("ALFA", "challenger"), _entry("KING", "king"), ERA) is None
    f = _fake(verify=("failed", {"error": "x"}))
    out = TrainerRunner._rolling_bench_challenger(
        f, _entry("ALFA", "challenger"), _entry("KING", "king"), ERA)
    assert R.VERIFY_PENDING in out


def test_operator_lane_numbers_need_no_verification():
    f = _fake(isolated=False)
    out = TrainerRunner._rolling_bench_challenger(f, _entry("ALFA", "challenger"), None, ERA)
    assert out == dataclasses.asdict(SCORES)


def test_king_pod_teardown_waits_for_a_running_sweep():
    retired = []
    f = SimpleNamespace(_rolling_king_hosts={3: KING_POD})
    f._rolling_retire_king_pod_now = lambda era: retired.append((era.index, time.time()))
    lock = host_bench_lock(KING_POD)
    lock.acquire()
    TrainerRunner._rolling_retire_king_pod(f, ERA)
    time.sleep(0.1)
    assert retired == []                      # deferred while the sweep runs
    t_release = time.time()
    lock.release()
    end = time.time() + 5
    while not retired and time.time() < end:
        time.sleep(0.01)
    assert retired and retired[0][1] >= t_release


def test_idle_king_pod_is_torn_down_at_once():
    retired = []
    f = SimpleNamespace(_rolling_king_hosts={3: KING_POD})
    f._rolling_retire_king_pod_now = lambda era: retired.append(era.index)
    TrainerRunner._rolling_retire_king_pod(f, ERA)
    assert retired == [3]
