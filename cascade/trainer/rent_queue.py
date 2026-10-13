"""First-come-first-served ordering for funded pod rents.

Every funded rent (the JIT king, each payer-funded challenger leg, each
operator-billed leg) is serialized through one slot so a later ``lium up`` sees
the executors an earlier one claimed (2026-09-02: concurrent pairs picked the
same top executor and the loser 429'd). A plain ``threading.Lock`` serialized
them in *arbitrary* order, so when the marketplace was dry and a couple of GPUs
appeared, whichever thread the scheduler woke first took them
(2026-10-13 07:10: two H100s appeared; uids 242/254, seated since 04:14, saw
them first and lost both to legs dispatched at 04:57 and 07:01).

:class:`SeniorityRentLock` grants the slot to the most senior waiter instead:
``priority`` tuples compare ascending (the king's tier sorts before every
challenger; challengers order by reveal block — the funded queue's own
seniority rule, "earliest reveal drains first"), ties broken by arrival. When
the slot is free a new request waits ``settle_seconds`` before it is granted so
requests that arrive together (several legs woken by the same capacity
sighting) are ranked together rather than first-thread-wins.

:class:`CapacityBeacon` closes the other half of the gap: a leg sleeping out its
90 s capacity poll would miss a GPU another leg just spotted. A leg that sees
capacity calls :meth:`CapacityBeacon.notify`; every polling leg's
:meth:`CapacityBeacon.wait` returns at once, so they re-probe and join the
line inside the settle window.
"""

from __future__ import annotations

import heapq
import itertools
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

# Tier 0 sorts first: the king is required for the round and always rents
# before any challenger (matches the existing yield-to-king rule).
KING_PRIORITY: tuple = (0,)
# A waiter with no seniority information ranks after every known one.
UNKNOWN_SENIORITY = 1 << 62


def leg_priority(reveal_block: int | None) -> tuple:
    """Rent priority of a challenger leg: earliest reveal block first."""
    rb = int(reveal_block) if reveal_block else UNKNOWN_SENIORITY
    return (1, rb)


class SeniorityRentLock:
    """A mutex granted to the most senior waiter, not the first thread woken.

    Use :meth:`slot` with a priority; ``with lock:`` (plain context manager)
    is kept for callers without seniority information and ranks them last.
    """

    def __init__(self, settle_seconds: float = 5.0,
                 clock=time.monotonic) -> None:
        self._cond = threading.Condition()
        self._held = False
        self._waiting: list[tuple] = []          # heap of tickets
        self._seq = itertools.count()
        self._free_since = clock()
        self._settle = float(settle_seconds)
        self._clock = clock

    def _grantable(self, ticket: tuple) -> bool:
        if self._held or self._waiting[0] != ticket:
            return False
        # Free slot: hold a fresh request for the settle window so concurrent
        # arrivals are ranked together (the head may change meanwhile).
        return self._clock() - self._free_since >= self._settle

    def acquire(self, priority: tuple = (UNKNOWN_SENIORITY,)) -> None:
        ticket = (*priority, next(self._seq))
        with self._cond:
            heapq.heappush(self._waiting, ticket)
            try:
                while not self._grantable(ticket):
                    remaining = self._settle - (self._clock() - self._free_since)
                    timeout = max(0.05, remaining) if not self._held else None
                    self._cond.wait(timeout)
            except BaseException:
                self._waiting.remove(ticket)
                heapq.heapify(self._waiting)
                self._cond.notify_all()
                raise
            heapq.heappop(self._waiting)
            self._held = True

    def release(self) -> None:
        with self._cond:
            if not self._held:
                raise RuntimeError("release of an unheld SeniorityRentLock")
            self._held = False
            self._free_since = self._clock()
            self._cond.notify_all()

    @contextmanager
    def slot(self, priority: tuple = (UNKNOWN_SENIORITY,)) -> Iterator[None]:
        self.acquire(priority)
        try:
            yield
        finally:
            self.release()

    def __enter__(self) -> SeniorityRentLock:
        self.acquire()
        return self

    def __exit__(self, *exc) -> None:
        self.release()

    def waiting(self) -> int:
        with self._cond:
            return len(self._waiting)


class CapacityBeacon:
    """Broadcast "a GPU just appeared" to every leg polling for capacity."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._gen = 0

    def notify(self) -> None:
        with self._cond:
            self._gen += 1
            self._cond.notify_all()

    def wait(self, timeout: float) -> bool:
        """Sleep up to ``timeout``; True when woken by :meth:`notify`."""
        with self._cond:
            gen = self._gen
            return self._cond.wait_for(lambda: self._gen != gen, timeout=max(0.0, timeout))
