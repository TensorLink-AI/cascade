"""Miner spend is booked per RENT, never per reused pod name (2026-11-03, rc20,
5EvqPCMN).

Live: a pod on ec4ab881 ($30/h) trained from 04:05, went unreachable at
04:29; a restart at 04:35 found it "not live" and rented fresh under the SAME
pod name on 63cadadb ($44.64/h), which was a short pod (1 GPU) rejected two
minutes later. The fresh rent's record was the lost pod's still-open record
(matched by name): the 2-minute reject booked ~42 min ($31.35 "make-good
owed"), the lost pod $12.83 waste, and the owed amount tripped the $30 waste
cap. Each rent now has its own record, the lost pod closes at its last-seen
time as infra waste, and a make-good owed never pauses the miner.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from cascade.funding.miner_spend import MAKE_GOOD, WASTE, MinerSpend
from cascade.funding.queue import FundedQueue
from cascade.shared.config import RoundConfig
from cascade.trainer import miner_spend_guard as g

HK = "5EvqPCMNrdlyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
REF = "vault/direct@sha256:" + "e" * 64
POD = "cascade-n91-777-funded-5evqpcmnrdly-0"
T0 = 1_762_142_700.0                       # 04:05 UTC


def _setup(tmp_path, clock):
    q = FundedQueue(tmp_path / "funded_queue.json")
    q.add(HK, REF, 10)
    r = SimpleNamespace(cfg=SimpleNamespace(round=RoundConfig(miner_waste_cap_usd=30.0)),
                        work_root=tmp_path, _funded_leg_failures={})
    r._funded_queue = lambda: q

    def record(hk, msg, *, miner_fault, error_class, burn):
        r._funded_leg_failures[hk] = (msg, miner_fault, error_class, burn)

    r._record_funded_failure = record
    led = MinerSpend.beside(q.path, clock=lambda: clock[0])
    return r, q, led


def test_the_5evqpcmn_sequence_books_each_rent_on_its_own(tmp_path, monkeypatch):
    clock = [T0]
    r, q, led = _setup(tmp_path, clock)
    monkeypatch.setattr(g, "_ledger", lambda runner: led)
    monkeypatch.setattr(g, "_any_ledger", lambda runner: led)
    # 04:05 rent on ec4ab881 at $30/h, training; last seen alive 04:29
    g.spend_start(r, HK, POD, rate=30.0, new_rent=True, rent_id="ec4ab881@1")
    clock[0] = T0 + 24 * 60
    led.touch(HK, POD)
    # 04:35 restart: the ledgered pod is not live — closed at its last-seen time
    clock[0] = T0 + 30 * 60
    g.spend_end(r, HK, POD, outcome=WASTE, at_last_seen=True, reason="lost: not live")
    # 04:36:03 fresh rent of the SAME pod name on 63cadadb at $44.64/h ...
    clock[0] = T0 + 31 * 60 + 3
    g.spend_start(r, HK, POD, rate=44.64, new_rent=True, rent_id="63cadadb@2")
    # ... 04:38:00 a short pod, rejected
    clock[0] = T0 + 33 * 60
    g.spend_end(r, HK, POD, outcome=MAKE_GOOD, reason="post-rent reject: short pod")
    s = led.summary(HK, ref=REF, funded_at=q.get(HK).funded_at)
    assert s.waste_usd == pytest.approx(30.0 * 24 / 60, abs=0.01)          # $12 lost pod
    assert s.make_good_usd == pytest.approx(44.64 * 117 / 3600, abs=0.01)  # ≈ $1.45 owed
    assert [p.get("rent_id") for p in s.pods] == ["ec4ab881@1", "63cadadb@2"]
    # the owed amount is OUR debt: no pause at the $30 cap
    g.waste_cap_check(r, SimpleNamespace(hotkey=HK, ref=REF, uid=232))
    assert HK not in r._funded_leg_failures


def test_a_new_rent_closes_a_forgotten_open_rent_at_its_last_seen_time(tmp_path):
    clock = [T0]
    led = MinerSpend(tmp_path / "miner_spend.json", clock=lambda: clock[0])
    led.start(HK, ref=REF, funded_at=None, pod_id=POD, rate=30.0, new_rent=True)
    clock[0] = T0 + 600
    led.touch(HK, POD)
    clock[0] = T0 + 3600                                   # never closed (a crash)
    led.start(HK, ref=REF, funded_at=None, pod_id=POD, rate=40.0, new_rent=True)
    # an update of the CURRENT rent (the landed price) never opens a third
    led.start(HK, ref=REF, funded_at=None, pod_id=POD, rate=41.0)
    pods = led.summary(HK, ref=REF, funded_at=None).pods
    assert len(pods) == 2
    assert pods[0]["end"] == T0 + 600 and pods[0]["outcome"] == WASTE
    assert pods[1]["start"] == T0 + 3600 and pods[1]["rate"] == 41.0 and pods[1]["end"] is None


def test_a_seeded_blacklist_would_have_kept_63cadadb_out(tmp_path, monkeypatch):
    from cascade.provision import core, short_pods

    rows = [{"id": "63cadadb-b379-4c45-8a01-04a81b135933", "gpu_count": 8,
             "price_per_hour": 44.64, "price_per_gpu_hour": 5.58},
            {"id": "ec4ab881-0000-4000-8000-000000000000", "gpu_count": 8,
             "price_per_hour": 30.0, "price_per_gpu_hour": 3.75}]
    prov = core.LiumProvider()
    monkeypatch.setattr(prov, "_cli", lambda args: SimpleNamespace(stdout=json.dumps(rows)))
    monkeypatch.setattr(prov, "_steer_hosts", lambda sku, shaped: shaped)
    monkeypatch.setattr(core, "cooling_executors", lambda: set())
    f = tmp_path / "_train_work" / short_pods.FILE_NAME
    short_pods.set_path(f)
    try:
        short_pods.block("63cadadb-b379-4c45-8a01-04a81b135933", seconds=7 * 86400, file=f)
        assert [e["id"] for e in prov._list_executors("8xH200")] == [rows[1]["id"]]
    finally:
        short_pods.set_path(None)
