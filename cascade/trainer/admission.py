"""Bounded, FIFO admission for shared operator compute; no provider I/O."""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from contextlib import contextmanager
from pathlib import Path

log = logging.getLogger(__name__)

# How long a chain-resolved hotkey -> coldkey identity stays usable for a NEW
# admission. Wall-clock seconds (the cache is persisted across restarts).
IDENTITY_TTL_SECONDS = 300.0
# Persisted next to the rolling state in the trainer's work_root, so a restart
# does not come up with a cold cache (2026-10-31: every re-attach after a
# restart failed "coldkey identity is unavailable" and was requeued).
IDENTITY_FILE = "coldkey_identities.json"


class FairSlots:
    """Share a live limit across lanes, releasing slots even after failure.

    Priority is reserved for king work; it jumps queued miners, never
    interrupts an active job. ``clock`` and deadlines use wall seconds.

    ``background`` work (opportunistic corpus pre-generation) only fills idle
    capacity and yields to real legs: it takes a slot only while no
    foreground job is waiting and the total is under the limit, and the slots
    it holds never count against a foreground job's limit — so a real leg
    never waits behind background work (at most ``2 x limit`` jobs run while
    background work drains). ``cancel()`` true abandons a still-waiting
    request (``acquired`` False); it never interrupts an active job.
    """

    def __init__(self, *, clock=time.time):
        self.clock = clock
        self.condition = threading.Condition()
        self.waiters = deque()
        self.active = 0
        self.background_active = 0
        self.owners = {}

    def _owner_ok(self, item):
        return not item[2] or self.owners.get(item[1], 0) < item[2]

    @contextmanager
    def slot(self, limit, *, deadline=None, priority=False, owner=None, owner_limit=0,
             background=False, cancel=None, force=False):
        """``force`` — take the slot NOW, over the limits, still counted: an
        already-admitted leg re-attached after a restart (it held its slot in
        the previous process; refusing it would cut a running leg)."""
        token = object()
        item = (token, owner, owner_limit, bool(background))
        acquired = False
        if force:
            with self.condition:
                self.active += 1
                if owner_limit:
                    self.owners[owner] = self.owners.get(owner, 0) + 1
                acquired = True
            try:
                yield acquired
            finally:
                with self.condition:
                    self.active -= 1
                    if owner_limit:
                        self.owners[owner] -= 1
                        if not self.owners[owner]:
                            del self.owners[owner]
                    self.condition.notify_all()
            return
        with self.condition:
            if priority and not background:
                self.waiters.appendleft(item)
            else:
                self.waiters.append(item)
            self.condition.notify_all()
            try:
                while True:
                    cap = max(1, int(limit() if callable(limit) else limit))
                    remaining = None if deadline is None else deadline - self.clock()
                    if remaining is not None and remaining <= 0:
                        break
                    if cancel is not None and cancel():
                        break
                    if background:
                        free = (self.active < cap
                                and not any(not w[3] for w in self.waiters))
                    else:
                        free = self.active - self.background_active < cap
                    eligible = next((w for w in self.waiters
                                     if w[3] == item[3] and self._owner_ok(w)), None)
                    if eligible is not None and eligible[0] is token and free:
                        self.waiters.remove(eligible)
                        self.active += 1
                        if background:
                            self.background_active += 1
                        if owner_limit:
                            self.owners[owner] = self.owners.get(owner, 0) + 1
                        acquired = True
                        break
                    self.condition.wait(1.0 if remaining is None else min(1.0, remaining))
            finally:
                if not acquired:
                    self.waiters.remove(item)
                self.condition.notify_all()
        try:
            yield acquired
        finally:
            if acquired:
                with self.condition:
                    self.active -= 1
                    if background:
                        self.background_active -= 1
                    if owner_limit:
                        self.owners[owner] -= 1
                        if not self.owners[owner]:
                            del self.owners[owner]
                    self.condition.notify_all()


def _identity_path(runner) -> Path | None:
    root = getattr(runner, "work_root", None)
    return Path(root) / IDENTITY_FILE if root else None


def _parse_identities(text: str) -> dict:
    raw = json.loads(text)
    if not isinstance(raw, dict):
        raise ValueError("coldkey identity cache is not an object")
    out = {}
    for hotkey, row in raw.items():
        owner, ts = row
        out[str(hotkey)] = (str(owner or ""), float(ts))
    return out


def _identities(runner) -> dict:
    """The live hotkey -> ``(coldkey, wall time resolved)`` cache; loaded from
    the persisted file the first time a process asks (a restart starts warm).
    Caller holds the identity lock."""
    d = runner.__dict__
    cache = d.get("_coldkey_identities")
    if cache is not None:
        return cache
    cache = {}
    path = _identity_path(runner)
    if path is not None:
        from ..shared.atomic import read_text_with_backup

        try:
            rows, source = read_text_with_backup(path, _parse_identities)
        except Exception as e:  # noqa: BLE001 — a cold cache is the old behaviour
            rows, source = None, f"error: {e}"
        if rows:
            cache.update(rows)
        elif source not in ("absent", "main", "backup"):
            log.warning("coldkey identity cache %s unreadable (%s) — starting cold", path, source)
    d["_coldkey_identities"] = cache
    return cache


def _persist_identities(runner, cache: dict) -> None:
    path = _identity_path(runner)
    if path is None:
        return
    from ..shared.atomic import write_text_atomic

    try:
        write_text_atomic(path, json.dumps({hk: [o, ts] for hk, (o, ts) in sorted(cache.items())},
                                           indent=0), backup=True)
    except OSError as e:
        log.warning("coldkey identity cache not persisted (%s): %s", path, e)


def _identity_lock(runner):
    return runner.__dict__.setdefault("_owner_identity_lock", threading.Lock())


def remember_owners(runner, rows):
    """Only chain-resolved current owners count; never infer an owner from a hotkey."""
    latest = {}
    for row in rows:
        hotkey = str(getattr(row, "hotkey", "") or "")
        if hotkey and (hotkey not in latest or getattr(row, "commit_block", 0) >= getattr(latest[hotkey], "commit_block", 0)):
            latest[hotkey] = row
    now = time.time()
    with _identity_lock(runner):
        identities = _identities(runner)
        for hotkey, row in latest.items():
            owner = str(getattr(row, "coldkey", "") or "")
            identities[hotkey] = (owner, now)
        if latest:
            _persist_identities(runner, identities)
    return rows


def owner_of(runner, hotkey, *, max_age: float | None = IDENTITY_TTL_SECONDS):
    """``hotkey``'s chain-resolved coldkey; ``max_age`` None = the last
    persisted identity whatever its age (the per-coldkey limit's view)."""
    with _identity_lock(runner):
        owner, timestamp = _identities(runner).get(hotkey, ("", 0))
    if not owner:
        return ""
    return owner if max_age is None or time.time() - timestamp <= max_age else ""


def _operator_paid(runner, hotkey) -> bool:
    """Whether ``hotkey``'s leg spends OPERATOR money beyond its CPU tier:
    its lane-A entry is operator-billed (a make-good / an operator rent).
    Unknown ⇒ False (a miner-key leg is never held by the coldkey limit)."""
    try:
        q = runner._funded_queue()
        e = q.get(hotkey) if q is not None else None
        return bool(e is not None and getattr(e, "operator_billed", False))
    except Exception:  # noqa: BLE001
        return False


# ── already-admitted legs (restart re-attach) ───────────────────────────────

def mark_admitted(runner, hotkey) -> None:
    """``hotkey``'s leg was admitted BEFORE this process started (an in-flight
    queue entry being re-attached): its owner slot is honoured even when the
    identity cache has not caught up yet. Cleared by :func:`clear_admitted`
    when the leg thread ends."""
    with _identity_lock(runner):
        runner.__dict__.setdefault("_admitted_legs", set()).add(str(hotkey))


def clear_admitted(runner, hotkey) -> None:
    with _identity_lock(runner):
        runner.__dict__.setdefault("_admitted_legs", set()).discard(str(hotkey))


def is_admitted(runner, hotkey) -> bool:
    with _identity_lock(runner):
        return str(hotkey) in runner.__dict__.get("_admitted_legs", set())


def owner_busy(runner, hotkey):
    """Money safety II (audit F): ``ops_max_active_per_coldkey`` bounds only
    OPERATOR-paid work — an operator-billed leg (operator rent / lane) —
    never a leg on the miner's own key; an unknown owner uses the last
    persisted identity and, when there is none, never blocks."""
    limit = int(getattr(runner.cfg.round, "ops_max_active_per_coldkey", 0))
    if not limit or not _operator_paid(runner, hotkey):
        return False
    owner = owner_of(runner, hotkey, max_age=None)
    if not owner:
        return False
    sched = runner.__dict__.get("_rolling_sched")
    live = set(getattr(sched, "_threads", {}))
    for name in ("lane_b", "lane_c"):
        driver = getattr(sched, name, None)
        if driver is not None:
            live.update(driver.live_hotkeys())
    return sum(owner_of(runner, hk, max_age=None) == owner and _operator_paid(runner, hk)
               for hk in live if hk != hotkey) >= limit


@contextmanager
def owner_admission(runner, hotkey, *, exempt=False, deadline=None, cpu_tier=False):
    """A shared, reentrant owner slot over OPERATOR-paid work (money safety
    II, audit F): an operator-billed leg (operator rent / lane) end to end,
    and the CPU-VM generation of ANY leg (``cpu_tier=True`` — the operator
    pays for the VM). A leg on the miner's own key takes no slot for its GPU
    rent; king legs are exempt. The owner is the last PERSISTED identity (no
    TTL for the limit); with none at all the leg proceeds (logged), never
    blocked by a slow chain read.

    Only a NEW admission needs a fresh coldkey identity. A leg marked
    :func:`mark_admitted` (re-attached after a restart) was admitted by the
    previous process: an unknown identity lets it through (logged), a known
    one takes its slot at once, over the limit, so the count stays right."""
    limit = int(getattr(getattr(getattr(runner, "cfg", None), "round", None),
                        "ops_max_active_per_coldkey", 0) or 0)
    if exempt or not limit:
        yield
        return
    from .faults import _FundedLegSkip

    local = runner.__dict__.setdefault("_owner_admission_local", threading.local())
    held = getattr(local, "held", None)
    if held is not None and held[1] == hotkey:
        yield                               # reentrant: this thread already holds it
        return
    if not cpu_tier and not _operator_paid(runner, hotkey):
        yield                               # a miner-key GPU leg: no operator slot
        return
    owner = owner_of(runner, hotkey, max_age=None)
    admitted = is_admitted(runner, hotkey)
    if not owner:                           # unknown never blocks (audit F)
        log.warning("owner admission: %s has no coldkey identity on record%s — "
                    "proceeding (the limit never blocks on an unknown owner)", str(hotkey)[:12],
                    " (re-attached)" if admitted else "")
        local.held = ("", hotkey)
        try:
            yield
        finally:
            local.held = held
        return
    gate = runner.__dict__.setdefault("_owner_slots", FairSlots())
    if deadline is None:
        deadline = time.time() + 1800
    if not admitted:
        with gate.condition:
            holding = int(gate.owners.get(owner, 0))
        if holding >= limit:
            # Say so when the wait STARTS: a leg that sat 30 min on a busy
            # coldkey slot used to log nothing until it was requeued
            # (2026-11-07, two hotkeys of one coldkey).
            log.info("owner admission: %s waits for a coldkey compute slot — coldkey %s… "
                     "holds %d/%d (%s); requeued if none frees within %.0f s",
                     str(hotkey)[:12], str(owner)[:12], holding, limit,
                     "CPU generation" if cpu_tier else "operator leg",
                     max(0.0, float(deadline) - time.time()))
    with gate.slot(2**31, owner=owner, owner_limit=limit, deadline=deadline,
                   force=admitted) as acquired:
        if not acquired:
            runner._record_funded_failure(hotkey, "coldkey compute slot remained busy; requeued before rent",
                                          miner_fault=False, error_class="no_capacity", burn=False)
            raise _FundedLegSkip(hotkey)
        local.held = (owner, hotkey)
        try:
            yield
        finally:
            local.held = held
