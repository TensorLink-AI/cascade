"""King-pod rotation must act on the FAILED pod only (2026-10-22 19:43–21:02:
a rotation that waited on the rent lock quarantined and tore down four fresh
king pods seconds after each came up — the era-2595 king never trained)."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

from cascade.trainer import loop as L
from cascade.trainer.loop import TrainerRunner


def _host(ip, port=20311):
    return SimpleNamespace(host=ip, port=port, name="funded-king")


def _runner(current, monkeypatch):
    r = TrainerRunner.__new__(TrainerRunner)
    r._funded_king_lock = threading.Lock()
    r._rolling_king_host_era = 2595
    r._funded_king_host = current
    r.retired, r.quarantined = [], []
    r._rolling_retire_king_pod = lambda era: r.retired.append(era.index)
    monkeypatch.setattr(L, "_quarantine_lane_host",
                        lambda host, reason: r.quarantined.append(host.host))
    monkeypatch.setattr(L, "_note_pod_loss", lambda *a, **k: None)
    return r


ERA = SimpleNamespace(index=2595)


def test_normal_rotation_quarantines_and_retires_the_failed_pod(monkeypatch):
    r = _runner(_host("1.1.1.1"), monkeypatch)
    TrainerRunner._rolling_rotate_king_pod(r, ERA, "pod unreachable for 906s")
    assert r.quarantined == ["1.1.1.1"] and r.retired == [2595]


def test_rotation_never_blocks_on_a_rent_in_progress(monkeypatch):
    r = _runner(_host("1.1.1.1"), monkeypatch)
    r._funded_king_lock.acquire()                    # a re-rent holds the lock
    t0 = time.monotonic()
    TrainerRunner._rolling_rotate_king_pod(r, ERA, "pod unreachable")
    assert time.monotonic() - t0 < 1.0               # returned at once
    assert r.quarantined == ["1.1.1.1"]              # the FAILED host, snapshotted
    assert r.retired == []                           # no teardown under the rent
    # the rent completes with a REPLACEMENT pod: the deferred check must spare it
    r._funded_king_host = _host("2.2.2.2", 40303)
    r._funded_king_lock.release()
    for _ in range(50):
        if not any(t.name == "rotate-king-2595" for t in threading.enumerate()):
            break
        time.sleep(0.05)
    assert r.retired == [] and "2.2.2.2" not in r.quarantined


def test_deferred_rotation_retires_when_the_failed_pod_is_still_there(monkeypatch):
    h = _host("1.1.1.1")
    r = _runner(h, monkeypatch)
    r._funded_king_lock.acquire()                    # busy for an unrelated reason
    TrainerRunner._rolling_rotate_king_pod(r, ERA, "x")
    r._funded_king_lock.release()                    # same pod still holds the slot
    for _ in range(50):
        if r.retired:
            break
        time.sleep(0.05)
    assert r.retired == [2595]


def test_slot_check_matches_by_address():
    r = TrainerRunner.__new__(TrainerRunner)
    r._funded_king_lock = threading.Lock()
    r._rolling_king_host_era = 2595
    r._funded_king_host = _host("1.1.1.1")
    assert TrainerRunner._king_slot_still_holds(r, ERA, _host("1.1.1.1")) is True
    assert TrainerRunner._king_slot_still_holds(r, ERA, _host("9.9.9.9")) is False
    r._funded_king_host = None
    assert TrainerRunner._king_slot_still_holds(r, ERA, _host("1.1.1.1")) is True   # gone
    r._funded_king_lock.acquire()
    assert TrainerRunner._king_slot_still_holds(r, ERA, None) is None               # busy
