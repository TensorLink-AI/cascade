"""The operator fallback lane pool and pod-host helpers for the trainer runner."""

from __future__ import annotations

import logging
import queue
import threading
import time

from .faults import _LaneDeadlinePassed, _NoOperatorLanes

log = logging.getLogger("cascade.trainer")


REMOVED_FROM_FILE = "removed from the hosts file"


class _FinalLanePool(queue.Queue):
    """The operator fallback lanes (``final``/``any`` hosts on file) as a
    queue of IDLE lanes that discovers lanes added after it was built.

    ``get()`` (the blocking, no-timeout form the dispatch path uses) refreshes
    membership from a ``refresh_fn`` on a timeout loop: hosts whose NAMES are
    new enter the rotation through the join gate. Liveness stays
    FAILURE-driven (a dead lane fails its dispatch and the retry policy moves
    on). With ``sync_from_file`` the hosts file is also authoritative for each
    known lane's host object and membership:

    * an edited entry (same name, ANY field: chain_toml, port, ssh options,
      ...) re-joins THROUGH THE GATE with the fresh host object — an IDLE
      lane at the next refresh, a BUSY one once its leg hands it back
      (:meth:`put`). So the gate's work (ssh/runtime probe, the deployed
      chain.toml push) runs on edits too; a rejected edit leaves the
      rotation and is re-gated after ``REJECT_RETRY_S``.
    * a lane no longer on file leaves the rotation once idle (a busy lane
      finishes its leg first and is dropped on :meth:`put`). A file that
      parses but lists ZERO lanes drains them all the same way.

    A refresh that RAISES (hosts file missing, unreadable, unparsable, or
    with no ``[[host]]`` entry at all — ``load_hosts``) keeps the current
    set. 2026-11-01: edits never reached dispatch, and lanes removed from
    the file (all of them) kept taking legs until ssh failed.
    """

    REFRESH_INTERVAL_S = 60.0
    # A lane that failed the join gate or was drained as dead is re-gated no
    # sooner than this (a pod still booting, or one that came back).
    REJECT_RETRY_S = 600.0
    # A lane the circuit breaker took out stays out this long (it failed
    # several different legs in a row — an operator config/host fault that a
    # 10-minute re-gate would only feed the next legs).
    BREAKER_HOLD_S = 6 * 3600.0

    def __init__(self, initial_hosts: list, refresh_fn, gate_fn=None, *,
                 sync_from_file: bool = False, breaker_threshold: int = 0):
        super().__init__()
        # True: the refresh also updates/removes KNOWN lanes (class doc);
        # False keeps the join-only refresh.
        self._sync = bool(sync_from_file)
        # BUSY lanes' pending file changes, applied when the lane is handed
        # back: name -> why it leaves the rotation (removed / edited).
        self._pending: dict[str, str] = {}
        self._refresh_fn = refresh_fn          # () -> list[RemoteHost]; may raise
        self._gate_fn = gate_fn                # (RemoteHost) -> reason | None; may raise
        self._absorb_lock = threading.Lock()
        self._known: dict[str, object] = {}
        self._rejected: dict[str, float] = {}  # lane name -> earliest re-gate time
        # Lane circuit breaker ([round] operator_lane_breaker_failures; 0 =
        # off): lane name -> the DISTINCT legs it failed operator-side since
        # its last successful leg.
        self.breaker_threshold = int(breaker_threshold or 0)
        self._lane_failures: dict[str, list[str]] = {}
        # Lanes the breaker took out: name -> the host object it tripped on.
        # With ``sync_from_file`` an EDITED entry (the operator fixed the lane)
        # ends the hold at once and re-joins through the gate; an unchanged
        # entry waits out BREAKER_HOLD_S.
        self._breaker_held: dict[str, object] = {}
        # The initial lanes are taken as given; the gate guards lanes that
        # JOIN later.
        for h in initial_hosts:
            self._known[getattr(h, "name", str(h))] = h
            super().put(h)

    def _admit(self, host) -> bool:
        """The join gate: a lane that joined blind could take a leg's only
        retry with a dead sshd. A rejected lane is remembered and re-gated
        after ``REJECT_RETRY_S``."""
        if self._gate_fn is None:
            return True
        name = getattr(host, "name", str(host))
        try:
            reason = self._gate_fn(host)
        except Exception as e:  # noqa: BLE001 — a gate crash is a rejection, never a wedge
            reason = f"gate error: {e}"
        if reason:
            self._rejected[name] = time.time() + self.REJECT_RETRY_S
            log.warning("final lane pool: lane %s NOT admitted (%s); re-gated in %.0f min",
                        name, reason, self.REJECT_RETRY_S / 60)
            return False
        self._rejected.pop(name, None)
        return True

    @staticmethod
    def same_pod(a, b) -> bool:
        """Two lanes on one pod (same ssh endpoint) share its fate."""
        return (getattr(a, "host", None), getattr(a, "port", None)) == (
            getattr(b, "host", None), getattr(b, "port", None))

    def mark_dead(self, host, reason: str, *, lane_only: bool = False,
                  hold_s: float | None = None) -> list[str]:
        """Drain every lane on ``host``'s pod: a transport failure (ssh refused,
        detached worker unreachable) is a POD fault, and its sibling lanes
        would only feed the next leg's single retry the same dead endpoint.
        The lanes leave membership; the hosts file re-offers them after
        ``REJECT_RETRY_S`` (or ``hold_s``) through the gate. ``lane_only``
        drains just ``host``'s own lane (the circuit breaker: siblings on the
        same pod get their own count). Returns the drained names."""
        with self._absorb_lock:
            name = getattr(host, "name", str(host))
            dead = [] if lane_only else [n for n, h in self._known.items()
                                         if self.same_pod(h, host)]
            if name not in dead:
                dead.append(name)
            until = time.time() + (self.REJECT_RETRY_S if hold_s is None else float(hold_s))
            for n in dead:
                self._known.pop(n, None)
                self._pending.pop(n, None)
                self._rejected[n] = until
            kept = []
            while True:
                try:
                    h = super().get(block=False)
                except queue.Empty:
                    break
                if getattr(h, "name", str(h)) not in dead:
                    kept.append(h)
            for h in kept:
                super().put(h)
        if dead:
            log.warning("final lane pool: %d lane(s) on %s:%s drained as dead (%s): %s",
                        len(dead), getattr(host, "host", "?"), getattr(host, "port", "?"),
                        reason, ", ".join(dead))
        return dead

    def record_leg_outcome(self, host, leg: str, *, ok: bool,
                           operator_fault: bool = True, reason: str = "") -> bool:
        """The lane circuit breaker. A successful leg resets ``host``'s lane;
        an operator-side failure of a leg it has not failed yet counts one;
        miner faults count nothing either way. At ``breaker_threshold``
        DIFFERENT legs failed in a row the lane leaves the rotation for
        ``BREAKER_HOLD_S`` with one ERROR "LANE BREAKER" line — 2026-11-01:
        one lane failed ~30 legs in seconds each (rc=2, its config), every one
        of them accruing requeues and backoff. Returns True when it tripped."""
        name = getattr(host, "name", str(host))
        with self._absorb_lock:
            if ok:
                self._lane_failures.pop(name, None)
                return False
            if not operator_fault or self.breaker_threshold <= 0:
                return False
            legs = self._lane_failures.setdefault(name, [])
            if leg not in legs:
                legs.append(leg)
            n = len(legs)
            if n < self.breaker_threshold:
                return False
            self._lane_failures.pop(name, None)
            self._breaker_held[name] = host
        log.error("LANE BREAKER: lane %s (%s:%s) failed %d different legs in a row "
                  "operator-side with no successful leg between (last: %s) — taken out of "
                  "rotation for %.1f h; fix the lane, then it re-joins through the gate",
                  name, getattr(host, "host", "?"), getattr(host, "port", "?"), n,
                  reason or "?", self.BREAKER_HOLD_S / 3600)
        self.mark_dead(host, f"LANE BREAKER: {n} different legs failed in a row",
                       lane_only=True, hold_s=self.BREAKER_HOLD_S)
        return True

    def _live_lanes(self) -> list[str]:
        """Known lanes that can still serve: a BUSY lane already removed from
        the hosts file leaves on :meth:`put` and will never serve a waiter."""
        return [n for n in list(self._known) if self._pending.get(n) != REMOVED_FROM_FILE]

    def _raise_if_no_lanes(self) -> None:
        if not self._live_lanes():
            held = sorted(n for n, t in self._rejected.items() if t > time.time())
            raise _NoOperatorLanes(
                "no operator lanes left — none live on file"
                + (f" (out of rotation: {', '.join(held)})" if held else ""))

    def unavailable_names(self) -> set[str]:
        """Lane names out of the rotation RIGHT NOW (drained as dead, refused
        at the gate, taken out by the breaker) and not live again — callers
        deciding "is there an operator lane?" must not count them."""
        # Lock-free snapshot: a refresh holds the lock through a join gate's
        # ssh probe, and this is asked on every capacity poll.
        now = time.time()
        known = set(self._known)
        return ({n for n, t in list(self._rejected.items()) if t > now and n not in known}
                | {n for n, why in list(self._pending.items()) if why == REMOVED_FROM_FILE})

    def known_hosts(self) -> list:
        """Current membership (for lane-count geometry): grows as lanes join,
        shrinks as dead (or, with ``sync_from_file``, removed) lanes leave."""
        return list(self._known.values())

    def _get_where(self, match, timeout: float):
        """An idle lane ``match(host)`` accepts, removed from the queue; every
        other idle lane STAYS IDLE for other legs (never held while waiting).
        ``match`` may probe the lane (ssh), so it runs OUTSIDE the queue lock
        on a snapshot; the chosen lane is then taken only if still idle.
        ``timeout`` 0 = no wait. Raises ``queue.Empty``."""
        end = time.time() + max(0.0, float(timeout))
        rejected: set[int] = set()
        while True:
            with self.mutex:
                snapshot = [h for h in self.queue if id(h) not in rejected]
            for h in snapshot:
                try:
                    ok = match(h) is True
                except Exception:  # noqa: BLE001 — an unprovable lane never matches
                    ok = False
                if not ok:
                    rejected.add(id(h))
                    continue
                with self.mutex:
                    for i, cur in enumerate(self.queue):
                        if cur is h:
                            del self.queue[i]
                            self.not_full.notify()
                            return h
                # taken by another leg meanwhile: rescan
            remaining = end - time.time()
            if remaining <= 0:
                raise queue.Empty
            with self.not_empty:
                # woken by any put (notify_all below); re-check the new lanes
                if all(id(h) in rejected for h in self.queue):
                    self.not_empty.wait(min(remaining, 5.0))
            with self.mutex:
                live = {id(h) for h in self.queue}
            rejected &= live      # a lane handed back is re-judged (re-bootstrap)

    def put(self, item, block: bool = True, timeout: float | None = None) -> None:
        """Hand a lane back (a leg finished with it). A file change recorded
        while it was busy applies now: the lane leaves the rotation — removed
        from the file, or edited, in which case the next refresh (every
        ``get`` runs one first) re-joins it through the gate."""
        name = getattr(item, "name", None)
        if name is None or name not in self._pending:
            # Fast path, no lock: a refresh can hold the lock through a join
            # gate's ssh probe. A change recorded right after this check is
            # applied to the then-idle lane by the next refresh.
            super().put(item, block, timeout)
            self._wake_all()
            return
        with self._absorb_lock:
            why = self._pending.pop(name, None)
            if why is not None:
                self._known.pop(name, None)
                log.info("lane pool: lane %s left the rotation after its leg (%s)",
                         name, why)
                return
        super().put(item, block, timeout)
        self._wake_all()

    def _wake_all(self) -> None:
        """Every waiter re-checks: a leg filtering by image must see a lane
        that another leg's waiter was notified for."""
        with self.not_empty:
            self.not_empty.notify_all()

    def _sync_known(self, fresh: list) -> None:
        """Apply the file to KNOWN lanes (caller holds ``_absorb_lock``): a lane
        removed or edited leaves the rotation — idle now, busy on :meth:`put`.
        An edited lane is still on file, so the join loop re-gates it with the
        fresh host object (in the same refresh for an idle lane). ``fresh``
        may be empty: a file that PARSED with zero lanes drains them all."""
        by_name = {getattr(h, "name", None): h for h in fresh}
        by_name.pop(None, None)
        changes: dict[str, str] = {}             # name -> why it leaves
        for name, cur in self._known.items():
            new = by_name.get(name)
            if new is None:
                changes[name] = REMOVED_FROM_FILE
            elif new is not cur and new != cur:
                changes[name] = "hosts file entry changed; re-gating"
        for name in list(self._pending):         # the file reverted the change
            if name not in changes:
                self._pending.pop(name, None)
        for name in changes:                     # an edited/removed entry: fresh count
            self._lane_failures.pop(name, None)
        if not changes:
            return
        idle = []
        while True:
            try:
                idle.append(super().get(block=False))
            except queue.Empty:
                break
        idle_names = set()
        for h in idle:
            name = getattr(h, "name", str(h))
            idle_names.add(name)
            if name not in changes:
                super().put(h)
                continue
            self._known.pop(name, None)
            self._pending.pop(name, None)
            log.info("lane pool: idle lane %s left the rotation (%s)", name, changes[name])
        for name, why in changes.items():
            if name not in idle_names and name in self._known:
                self._pending[name] = why        # busy: applied on put()

    def _absorb_new(self) -> None:
        try:
            fresh = list(self._refresh_fn())
        except Exception as e:  # noqa: BLE001 — a torn/absent file keeps the last set
            log.debug("final lane pool refresh failed (keeping current set): %s", e)
            return
        with self._absorb_lock:
            if self._sync:
                self._sync_known(fresh)
                self._release_edited_breaker_holds(fresh)
            now = time.time()
            for h in fresh:
                name = getattr(h, "name", None)
                if not name or name in self._known:
                    continue
                if self._rejected.get(name, 0.0) > now:
                    continue                      # rejected/drained: not yet due
                if not self._admit(h):
                    continue                      # _admit logs the rejection
                self._known[name] = h
                self._breaker_held.pop(name, None)
                self._lane_failures.pop(name, None)
                super().put(h)
                log.info("lane pool: new lane %s joined", name)

    def _release_edited_breaker_holds(self, fresh: list) -> None:
        """A breaker-held lane whose hosts-file entry CHANGED since it tripped
        is re-gated now instead of after ``BREAKER_HOLD_S`` (the operator
        fixed it); an unchanged or absent entry stays held. Caller holds
        ``_absorb_lock``."""
        by_name = {getattr(h, "name", None): h for h in fresh}
        for name, tripped in list(self._breaker_held.items()):
            new = by_name.get(name)
            if new is None or new is tripped or new == tripped:
                continue
            self._breaker_held.pop(name, None)
            self._rejected.pop(name, None)
            log.warning("lane pool: lane %s was edited after its LANE BREAKER trip — "
                        "re-gating it now", name)

    def _deadline_of(self, deadline, host) -> float | None:
        """Per-lane form: ``deadline`` may be a callable ``host -> float|None``
        (a lane's OWN latest safe start — [round] funded_sku_wall_seconds
        makes a fast lane usable later than a slow one) or one number."""
        if callable(deadline):
            return deadline(host)
        return deadline

    def _latest_deadline(self, deadline) -> float | None:
        """The last moment ANY known lane could still take this leg: the
        wait's bound. None when unbounded (no deadline, or a lane without
        one)."""
        if not callable(deadline):
            return deadline
        latest = None
        for h in self.known_hosts():
            d = self._deadline_of(deadline, h)
            if d is None:
                return None
            latest = d if latest is None else max(latest, d)
        return latest

    def get(self, block: bool = True, timeout: float | None = None, *,
            deadline=None, match=None):
        """``deadline`` (wall-clock, blocking form only): the latest moment a
        lane may still be handed out — past it the wait ends with
        ``_LaneDeadlinePassed`` and a lane that frees later stays in the pool
        for a leg that CAN still fit. None = wait forever. A callable ``host -> deadline`` gives every lane its own (per-SKU)
        latest safe start: a lane freed past ITS deadline goes back for a
        leg that fits it; the wait itself lasts until the latest of them.

        ``match`` (``host -> bool``, DEC-CA-0067): only a lane it accepts is
        handed out (a leg's era image); other lanes stay idle for other legs
        while this one waits. No KNOWN lane matching ⇒ ``_NoOperatorLanes``
        (nothing that could ever serve this leg)."""
        if match is not None:
            return self._get_matching(deadline, match, block=block, timeout=timeout)
        if not block or timeout is not None:
            return super().get(block=block, timeout=timeout)
        while True:
            latest = self._latest_deadline(deadline)
            if latest is not None:
                remaining = latest - time.time()
                if remaining <= 0:
                    # Past the deadline a lane that is FREE RIGHT NOW is still
                    # handed out (a rented lane idling is the waste); only
                    # WAITING for one to free would run the leg late.
                    self._absorb_new()
                    try:
                        return super().get(block=False)
                    except queue.Empty:
                        self._raise_if_no_lanes()
                        raise _LaneDeadlinePassed(
                            f"no operator lane free at the leg's latest safe start "
                            f"({time.strftime('%H:%M:%SZ', time.gmtime(latest))})"
                        ) from None
            self._absorb_new()
            # ZERO live lanes (none on file, all removed by the hosts-file
            # sync, drained as dead, refused at the gate or taken out by the
            # breaker): nothing can ever free — never wait for one.
            # 2026-11-01: 8 legs sat here ~02:20→06:41 after every lane went.
            if not self._live_lanes():
                try:
                    return super().get(block=False)   # a lane handed back untracked
                except queue.Empty:
                    self._raise_if_no_lanes()
            wait = self.REFRESH_INTERVAL_S
            if latest is not None:
                wait = max(0.0, min(wait, latest - time.time()))
            try:
                host = super().get(timeout=wait)
            except queue.Empty:
                continue
            own = self._deadline_of(deadline, host)
            if own is not None and time.time() >= own:
                super().put(host)                # freed too late for THIS leg
                if callable(deadline):
                    # Another lane type may still fit: keep waiting for one,
                    # but never spin on the one we just put back.
                    time.sleep(min(1.0, max(0.0, (latest or 0) - time.time())) or 0.05)
                continue                         # next pass raises
            return host


    def _no_matching_lane(self, match) -> None:
        self._raise_if_no_lanes()
        known = self.known_hosts()

        def state(h):
            # True / False / None (unknown — e.g. unreachable right now: the
            # leg waits and retries, it is never told "no lane")
            try:
                return match(h)
            except Exception:  # noqa: BLE001
                return None
        if known and all(state(h) is False for h in known):
            raise _NoOperatorLanes("no operator lane on this leg's era image (DEC-CA-0067) "
                                   "— none of " + ", ".join(
                                       sorted(getattr(h, "name", "?") for h in known)))

    def _get_matching(self, deadline, match, *, block=True, timeout=None):
        """:meth:`get` restricted to lanes ``match`` accepts (same deadline
        semantics, never holding a non-matching lane)."""
        if not block:
            return self._get_where(match, 0.0)
        if timeout is not None:
            return self._get_where(match, timeout)
        while True:
            latest = self._latest_deadline(deadline)
            self._absorb_new()
            self._no_matching_lane(match)
            if latest is not None and latest - time.time() <= 0:
                try:
                    return self._get_where(match, 0.0)
                except queue.Empty:
                    raise _LaneDeadlinePassed(
                        f"no operator lane on this leg's image free at its latest safe "
                        f"start ({time.strftime('%H:%M:%SZ', time.gmtime(latest))})") from None
            wait = self.REFRESH_INTERVAL_S
            if latest is not None:
                wait = max(0.0, min(wait, latest - time.time()))

            def fits(h):
                own = self._deadline_of(deadline, h)
                return (own is None or time.time() < own) and match(h) is True
            try:
                return self._get_where(fits, wait)
            except queue.Empty:
                continue


def _lium_provider(rnd, *, price_caps: dict | None = None, **kw):
    """An operator-key :class:`LiumProvider` carrying ``[round]
    funded_cpu_blocklist`` and the price guards (imported at call time so
    tests can swap the class)."""
    from ..provision.core import LiumProvider
    from ..provision.funded import apply_cpu_blocklist, apply_price_caps

    prov = apply_cpu_blocklist(LiumProvider(**kw),
                               tuple(getattr(rnd, "funded_cpu_blocklist", ()) or ()))
    return apply_price_caps(prov, **(price_caps or {}))


def host_cannot_bench(runner, host) -> str:
    """Why ``host`` cannot run the public-benchmark sidecar, or ``""``.

    Only checked when ``runner`` has benches configured. The NVIDIA driver
    must be at least :data:`~cascade.trainer.remote.BENCH_MIN_DRIVER_MAJOR`;
    a probe that cannot read it passes (never sink a pod on an ssh hiccup).
    Cached per endpoint on the runner — a pod's driver does not change."""
    if getattr(runner, "cascade_bench_plan", None) is None or host is None:
        return ""
    from .remote import BENCH_MIN_DRIVER_MAJOR, probe_driver_major

    key = (str(getattr(host, "host", "")), int(getattr(host, "port", 0) or 0))
    cache = runner.__dict__.setdefault("_bench_driver_cache", {})
    if key not in cache:
        major, err = probe_driver_major(host)
        if major is None:
            log.warning("host %s: driver probe unavailable (%s) — bench capability "
                        "assumed", key[0], err)
        cache[key] = major
    major = cache[key]
    if major is not None and major < BENCH_MIN_DRIVER_MAJOR:
        return f"driver {major} < {BENCH_MIN_DRIVER_MAJOR}: cannot run the public benchmarks"
    return ""


def king_pod_name_prefix(netuid: int, round_id: str) -> str:
    """Name prefix of an era's JIT king pod (``<prefix>-0``; ``round_id`` is
    the era's base seed). The n<netuid> token keeps it OUT of both the
    provisioner's reaper scheme (which must never touch trainer-ledgered pods)
    and any co-hosted deployment's sweep; the seed lets a restart recognise
    its own pod."""
    return f"cascade-n{netuid}-{round_id}-funded-king"
