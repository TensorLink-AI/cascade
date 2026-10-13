"""First-come-first-served funded rents (owner 2026-10-13).

2026-10-13 07:10: two H100s appeared after a dry spell; uids 242/254 (seated
04:14) saw them first but lost both to legs dispatched at 04:57 and 07:01,
because the rent slot was a plain lock granted in scheduler order.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from cascade.trainer.rent_queue import (
    KING_PRIORITY,
    CapacityBeacon,
    SeniorityRentLock,
    leg_priority,
)


def _race(lock: SeniorityRentLock, requests: list[tuple[str, tuple, float]],
          hold: float = 0.02) -> list[str]:
    """Start one thread per (name, priority, delay); return grant order."""
    order: list[str] = []
    guard = threading.Lock()

    def run(name: str, prio: tuple, delay: float) -> None:
        time.sleep(delay)
        with lock.slot(prio):
            with guard:
                order.append(name)
            time.sleep(hold)

    threads = [threading.Thread(target=run, args=r) for r in requests]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert all(not t.is_alive() for t in threads)
    return order


def test_senior_waiters_win_regardless_of_arrival():
    lock = SeniorityRentLock(settle_seconds=0.0)
    lock.acquire(KING_PRIORITY)                       # slot busy while they queue
    results: list[str] = []

    def run(name, prio):
        with lock.slot(prio):
            results.append(name)

    ts = [threading.Thread(target=run, args=a) for a in
          (("late", leg_priority(300)), ("first", leg_priority(100)),
           ("mid", leg_priority(200)))]
    for t in ts:
        t.start()
        time.sleep(0.02)                              # arrive late, first, mid
    while lock.waiting() < 3:
        time.sleep(0.01)
    lock.release()
    for t in ts:
        t.join(5)
    assert results == ["first", "mid", "late"]


def test_settle_window_ranks_near_simultaneous_arrivals():
    # Junior arrives first at a FREE slot; the senior arrives inside the
    # settle window and must still be granted first.
    lock = SeniorityRentLock(settle_seconds=0.3)
    time.sleep(0.35)                                  # slot has been free > settle
    lock = SeniorityRentLock(settle_seconds=0.3)      # fresh: free_since = now
    order = _race(lock, [("junior", leg_priority(500), 0.0),
                         ("senior", leg_priority(100), 0.1)])
    assert order == ["senior", "junior"]


def test_king_always_first():
    lock = SeniorityRentLock(settle_seconds=0.2)
    order = _race(lock, [("leg", leg_priority(1), 0.0),
                         ("king", KING_PRIORITY, 0.05)])
    assert order == ["king", "leg"]


def test_unknown_seniority_ranks_last_and_plain_with_still_works():
    lock = SeniorityRentLock(settle_seconds=0.2)
    order: list[str] = []

    def plain():
        with lock:
            order.append("plain")

    def known():
        time.sleep(0.05)
        with lock.slot(leg_priority(42)):
            order.append("known")

    ts = [threading.Thread(target=plain), threading.Thread(target=known)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(5)
    assert order == ["known", "plain"]
    assert leg_priority(None) > leg_priority(10**9)


def test_release_unheld_raises():
    with pytest.raises(RuntimeError):
        SeniorityRentLock().release()


def test_replay_2026_10_13_0710_two_h100s_four_legs():
    """The four legs ask for the slot within one settle window, in the order
    the scheduler actually served them; seniority must reorder them so the two
    longest-seated legs (uids 242, 254) take the two GPUs."""
    lock = SeniorityRentLock(settle_seconds=0.3)
    reveal = {"5HYvBFhbi8fM": 9271499, "5FjuCBUTNuoP": 9272605,
              "5GQcrnxPgNYM(uid242)": 9266855, "5GRPKCJJuBk2(uid254)": 9271359}
    arrivals = ["5HYvBFhbi8fM", "5FjuCBUTNuoP",
                "5GQcrnxPgNYM(uid242)", "5GRPKCJJuBk2(uid254)"]
    order = _race(lock, [(n, leg_priority(reveal[n]), 0.03 * i)
                         for i, n in enumerate(arrivals)])
    assert order[:2] == ["5GQcrnxPgNYM(uid242)", "5GRPKCJJuBk2(uid254)"]
    assert order == ["5GQcrnxPgNYM(uid242)", "5GRPKCJJuBk2(uid254)",
                     "5HYvBFhbi8fM", "5FjuCBUTNuoP"]


def test_capacity_beacon_wakes_sleepers():
    b = CapacityBeacon()
    assert b.wait(0.05) is False                      # times out quietly
    woke: list[bool] = []
    t = threading.Thread(target=lambda: woke.append(b.wait(5.0)))
    t.start()
    time.sleep(0.05)
    t0 = time.monotonic()
    b.notify()
    t.join(2)
    assert woke == [True] and time.monotonic() - t0 < 1.0


def test_loop_rank_comes_from_funded_queue_reveal_block():
    from cascade.trainer.loop import TrainerRunner

    entries = {"hkA": SimpleNamespace(reveal_block=9266855)}
    fake = SimpleNamespace(
        _funded_queue=lambda: SimpleNamespace(get=lambda hk: entries.get(hk)))
    rank = TrainerRunner._leg_rent_priority
    assert rank(fake, "hkA") == leg_priority(9266855)
    assert rank(fake, "hkB") == leg_priority(None)   # unknown → last
    broken = SimpleNamespace(_funded_queue=lambda: None)
    assert rank(broken, "hkA") == leg_priority(None)


def test_loop_capacity_sleep_prefers_injected_sleep_then_beacon():
    from cascade.trainer.loop import TrainerRunner

    slept: list[float] = []
    fake = SimpleNamespace(_rent_wait_sleep=slept.append)
    TrainerRunner._capacity_sleep(fake, 90.0)
    assert slept == [90.0]

    beacon = CapacityBeacon()
    fake2 = SimpleNamespace(_rent_wait_sleep=None, _funded_capacity_beacon=beacon)
    threading.Timer(0.05, beacon.notify).start()
    t0 = time.monotonic()
    TrainerRunner._capacity_sleep(fake2, 30.0)
    assert time.monotonic() - t0 < 2.0               # woken, not 30 s
