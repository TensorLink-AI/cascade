"""Bounded, FIFO admission for shared operator compute; no provider I/O."""

from __future__ import annotations

import threading
import time
from collections import deque
from contextlib import contextmanager


class FairSlots:
    """Share a live limit across lanes, releasing slots even after failure.

    Priority is reserved for king work; it jumps queued miners, never
    interrupts an active job. ``clock`` and deadlines use wall seconds.
    """

    def __init__(self, *, clock=time.time):
        self.clock = clock
        self.condition = threading.Condition()
        self.waiters = deque()
        self.active = 0
        self.owners = {}

    @contextmanager
    def slot(self, limit, *, deadline=None, priority=False, owner=None, owner_limit=0):
        token = object()
        acquired = False
        with self.condition:
            if priority:
                self.waiters.appendleft((token, owner, owner_limit))
            else:
                self.waiters.append((token, owner, owner_limit))
            self.condition.notify_all()
            try:
                while True:
                    cap = max(1, int(limit() if callable(limit) else limit))
                    remaining = None if deadline is None else deadline - self.clock()
                    if remaining is not None and remaining <= 0:
                        break
                    eligible = next((item for item in self.waiters
                                     if not item[2] or self.owners.get(item[1], 0) < item[2]), None)
                    if eligible is not None and eligible[0] is token and self.active < cap:
                        self.waiters.remove(eligible)
                        self.active += 1
                        if owner_limit:
                            self.owners[owner] = self.owners.get(owner, 0) + 1
                        acquired = True
                        break
                    self.condition.wait(1.0 if remaining is None else min(1.0, remaining))
            finally:
                if not acquired:
                    self.waiters.remove((token, owner, owner_limit))
                self.condition.notify_all()
        try:
            yield acquired
        finally:
            if acquired:
                with self.condition:
                    self.active -= 1
                    if owner_limit:
                        self.owners[owner] -= 1
                        if not self.owners[owner]:
                            del self.owners[owner]
                    self.condition.notify_all()


def remember_owners(runner, rows):
    """Only chain-resolved current owners count; never infer an owner from a hotkey."""
    lock = runner.__dict__.setdefault("_owner_identity_lock", threading.Lock())
    latest = {}
    for row in rows:
        hotkey = str(getattr(row, "hotkey", "") or "")
        if hotkey and (hotkey not in latest or getattr(row, "commit_block", 0) >= getattr(latest[hotkey], "commit_block", 0)):
            latest[hotkey] = row
    with lock:
        identities = runner.__dict__.setdefault("_coldkey_identities", {})
        for hotkey, row in latest.items():
            owner = str(getattr(row, "coldkey", "") or "")
            identities[hotkey] = (owner, time.monotonic())
    return rows


def owner_of(runner, hotkey):
    owner, timestamp = runner.__dict__.get("_coldkey_identities", {}).get(hotkey, ("", 0))
    return owner if owner and time.monotonic() - timestamp <= 300 else ""


def owner_busy(runner, hotkey):
    limit = int(getattr(runner.cfg.round, "ops_max_active_per_coldkey", 0))
    if not limit:
        return False
    owner = owner_of(runner, hotkey)
    if not owner:
        return True
    sched = runner.__dict__.get("_rolling_sched")
    live = set(getattr(sched, "_threads", {}))
    for name in ("lane_b", "lane_c"):
        driver = getattr(sched, name, None)
        if driver is not None:
            live.update(driver.live_hotkeys())
    return sum(owner_of(runner, hk) == owner for hk in live if hk != hotkey) >= limit


@contextmanager
def owner_admission(runner, hotkey, *, exempt=False, deadline=None):
    """A shared, reentrant owner slot covers generation, training and fallback."""
    limit = int(getattr(runner.cfg.round, "ops_max_active_per_coldkey", 0))
    if exempt or not limit:
        yield
        return
    from .faults import _FundedLegSkip

    owner = owner_of(runner, hotkey)
    if not owner:
        runner._record_funded_failure(hotkey, "coldkey identity is unavailable; waiting for fresh chain metadata",
                                      miner_fault=False, error_class="no_capacity", burn=False)
        raise _FundedLegSkip(hotkey)
    local = runner.__dict__.setdefault("_owner_admission_local", threading.local())
    if getattr(local, "held", None) == (owner, hotkey):
        yield
        return
    gate = runner.__dict__.setdefault("_owner_slots", FairSlots())
    if deadline is None:
        deadline = time.time() + 1800
    with gate.slot(2**31, owner=owner, owner_limit=limit, deadline=deadline) as acquired:
        if not acquired:
            runner._record_funded_failure(hotkey, "coldkey compute slot remained busy; requeued before rent",
                                          miner_fault=False, error_class="no_capacity", burn=False)
            raise _FundedLegSkip(hotkey)
        previous = getattr(local, "held", None)
        local.held = (owner, hotkey)
        try:
            yield
        finally:
            local.held = previous
