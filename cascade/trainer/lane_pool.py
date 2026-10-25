"""The operator fallback lane pool and pod-host helpers for the trainer runner."""

from __future__ import annotations

import logging
import queue
import threading
import time

from .faults import _LaneDeadlinePassed

log = logging.getLogger("cascade.trainer")


class _FinalLanePool(queue.Queue):
    """The operator fallback lanes (``final``/``any`` hosts on file) as a
    queue of IDLE lanes that discovers lanes added after it was built.

    ``get()`` (the blocking, no-timeout form the dispatch path uses) refreshes
    membership from a ``refresh_fn`` on a timeout loop: hosts whose NAMES are
    new enter the rotation through the join gate. Removal stays
    FAILURE-driven (a dead lane fails its dispatch and the retry policy moves
    on): the hosts file is authoritative for membership, never for liveness.
    """

    REFRESH_INTERVAL_S = 60.0
    # A lane that failed the join gate or was drained as dead is re-gated no
    # sooner than this (a pod still booting, or one that came back).
    REJECT_RETRY_S = 600.0

    def __init__(self, initial_hosts: list, refresh_fn, gate_fn=None):
        super().__init__()
        self._refresh_fn = refresh_fn          # () -> list[RemoteHost]; may raise
        self._gate_fn = gate_fn                # (RemoteHost) -> reason | None; may raise
        self._absorb_lock = threading.Lock()
        self._known: dict[str, object] = {}
        self._rejected: dict[str, float] = {}  # lane name -> earliest re-gate time
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

    def mark_dead(self, host, reason: str) -> list[str]:
        """Drain every lane on ``host``'s pod: a transport failure (ssh refused,
        detached worker unreachable) is a POD fault, and its sibling lanes
        would only feed the next leg's single retry the same dead endpoint.
        The lanes leave membership; the hosts file re-offers them after
        ``REJECT_RETRY_S`` through the gate. Returns the drained names."""
        with self._absorb_lock:
            dead = [n for n, h in self._known.items() if self.same_pod(h, host)]
            name = getattr(host, "name", str(host))
            if name not in dead:
                dead.append(name)
            until = time.time() + self.REJECT_RETRY_S
            for n in dead:
                self._known.pop(n, None)
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

    def known_hosts(self) -> list:
        """Current membership (for lane-count geometry) — grows, never shrinks."""
        return list(self._known.values())

    def _absorb_new(self) -> None:
        try:
            fresh = self._refresh_fn()
        except Exception as e:  # noqa: BLE001 — a torn/absent file keeps the last set
            log.debug("final lane pool refresh failed (keeping current set): %s", e)
            return
        with self._absorb_lock:
            now = time.time()
            for h in fresh:
                name = getattr(h, "name", None)
                if not name or name in self._known:
                    continue
                if self._rejected.get(name, 0.0) > now:
                    continue                      # rejected/drained: not yet due
                if not self._admit(h):
                    continue
                self._known[name] = h
                super().put(h)
                log.info("lane pool: new lane %s joined", name)

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
            deadline=None):
        """``deadline`` (wall-clock, blocking form only): the latest moment a
        lane may still be handed out — past it the wait ends with
        ``_LaneDeadlinePassed`` and a lane that frees later stays in the pool
        for a leg that CAN still fit. None = wait forever. A callable ``host -> deadline`` gives every lane its own (per-SKU)
        latest safe start: a lane freed past ITS deadline goes back for a
        leg that fits it; the wait itself lasts until the latest of them."""
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
                        raise _LaneDeadlinePassed(
                            f"no operator lane free at the leg's latest safe start "
                            f"({time.strftime('%H:%M:%SZ', time.gmtime(latest))})"
                        ) from None
            self._absorb_new()
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
