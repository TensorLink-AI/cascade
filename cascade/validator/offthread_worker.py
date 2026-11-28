"""Lane D work off the weight/poll thread (LANE_D_PLAN W19, invariant 6).

:class:`LaneDWorker` runs one lane D settlement step at a time on its own
daemon thread. The validator's poll loop submits the armed settlement
boundary each poll and never waits for the step: a multi-hour lane D bench
sweep can no longer delay a due weight re-assert ("weights never wait for
lane D").

Semantics (tests/accept/w19/README.md):

- ``submit(boundary)`` starts ``step(boundary)`` on a new daemon thread and
  returns ``True`` at once, unless a step is already in flight or that
  boundary has completed (then ``False``, nothing started: no pile-up).
- A boundary whose step returned is completed and never run again by this
  worker; one whose step raised is NOT completed (fail-closed: a later
  submit retries it) and the exception is kept as :attr:`last_error` until a
  later step succeeds.
- The step gets only the boundary: it must open its own chain access, never
  share the poll thread's (non-thread-safe) client.
- Moving past an unsettled boundary (2026-11-27): when a boundary whose step
  raised is followed by a submit of a LATER boundary, that boundary is given
  up (never retried) and, before the later step, ``on_abandon(boundary)``
  runs on the new step's thread (the lane D step publishes its absent
  ``infra_deferred`` record there). A raising ``on_abandon`` is logged; the
  later step still runs.

- Memory guard (W-OFFBOX, 2026-11-28): while a step runs, the process RSS is
  sampled (:func:`process_rss_bytes`, every :data:`RSS_SAMPLE_SECONDS`); the
  step's start / peak / growth are logged when it ends (kept as
  :attr:`last_rss`), and a growth over ``rss_budget_bytes``
  (:data:`STEP_RSS_BUDGET_BYTES`) is logged as an ERROR naming the budget. The
  lane D step loaded every pinned pool_d snapshot into this process and the
  box OOM-killed the validator twice (04:39, 05:41) with no line saying why.

Generic thread infrastructure, deliberately not named ``lane_d_*`` and
importing no lane D module: ``loop.py`` imports it, and the audit's import
closure (which reaches ``loop.py``) must load no lane D producer module (W9
invariant 12).
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

from ..shared.rss_budget import RSS_SAMPLE_SECONDS, RssWatch, process_rss_bytes, report_rss

log = logging.getLogger(__name__)

# The most a lane D step may grow the validator's RSS (the box runs the
# validator, trainer and intake in 7 GB): the step orchestrates, the lane D
# eval host loads and scores (W-OFFBOX). The sampler is the shared one
# (cascade.shared.rss_budget, retro P2).
STEP_RSS_BUDGET_BYTES = 512 * 1024 * 1024
_RssWatch = RssWatch            # the pre-P2 private name (marketcast, tests)


class LaneDWorker:
    """One lane D step at a time, on a daemon thread the caller never joins
    implicitly."""

    def __init__(self, step: Callable[[int], object],
                 on_abandon: Callable[[int], object] | None = None, *,
                 rss_budget_bytes: int = STEP_RSS_BUDGET_BYTES,
                 rss_probe: Callable[[], int | None] = process_rss_bytes,
                 rss_interval: float = RSS_SAMPLE_SECONDS):
        self.step = step
        self.on_abandon = on_abandon
        self.rss_budget_bytes = int(rss_budget_bytes)
        self._rss_probe = rss_probe
        self._rss_interval = float(rss_interval)
        # {"start", "peak", "growth"} bytes of the last step (None: unreadable)
        self.last_rss: dict | None = None
        self._failed: int | None = None
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._busy = False
        self._completed: set[int] = set()
        self.last_error: BaseException | None = None

    @property
    def running(self) -> bool:
        """A step is in flight."""
        with self._lock:
            return self._busy

    def submit(self, boundary: int) -> bool:
        """Start ``step(boundary)`` off this thread; ``False`` (nothing
        started) while a step is running or once ``boundary`` completed."""
        b = int(boundary)
        with self._lock:
            if self._busy or b in self._completed:
                return False
            abandoned = None
            if self._failed is not None and self._failed < b:
                abandoned, self._failed = self._failed, None
                self._completed.add(abandoned)
            self._busy = True
            # The sampler starts here, before the step's thread, so the step
            # starts as promptly as before (a thread start waits for the new
            # thread to run).
            watch = _RssWatch(self._rss_probe, self._rss_interval)
            t = threading.Thread(target=self._run, args=(b, abandoned, watch),
                                 name=f"lane-d-step-{b}", daemon=True)
            self._thread = t
        try:
            watch.start()
            t.start()
        except RuntimeError as e:
            watch.stop()
            with self._lock:
                self._busy = False
                self._thread = None
                if abandoned is not None:
                    self._completed.discard(abandoned)
                    self._failed = abandoned
            log.warning("lane D: could not start the step thread for boundary %d (%s)", b, e)
            return False
        log.info("lane D: step for settlement %d started off the poll thread", b)
        return True

    def join(self, timeout: float | None = None) -> bool:
        """Wait at most ``timeout`` seconds; ``True`` iff no step is in flight
        afterwards."""
        with self._lock:
            t = self._thread
        if t is not None:
            t.join(timeout)
        return not self.running

    def _run(self, boundary: int, abandoned: int | None = None,
             watch: _RssWatch | None = None) -> None:
        if abandoned is not None:
            log.warning("lane D: settlement %d was never settled and the head moved on to "
                        "%d; it is given up", abandoned, boundary)
            if self.on_abandon is not None:
                try:
                    self.on_abandon(abandoned)
                except BaseException as e:  # noqa: BLE001 — never blocks the next step
                    log.warning("lane D: recording settlement %d as abandoned failed (%s)",
                                abandoned, e)
        if watch is None:
            watch = _RssWatch(self._rss_probe, self._rss_interval).start()
        try:
            self.step(boundary)
        except BaseException as e:  # noqa: BLE001 — a step never escapes its thread
            self._rss_report(boundary, watch.stop())
            with self._lock:
                self.last_error = e
                self._busy = False
                if self._failed is None or boundary > self._failed:
                    self._failed = boundary
            log.warning("lane D: step for settlement %d failed (%s: %s); retried at a "
                        "later tick", boundary, type(e).__name__, e)
            return
        self._rss_report(boundary, watch.stop())
        with self._lock:
            self._completed.add(boundary)
            if self._failed == boundary:
                self._failed = None
            self.last_error = None
            self._busy = False
        log.info("lane D: step for settlement %d completed", boundary)

    def _rss_report(self, boundary: int, rss: dict | None) -> None:
        self.last_rss = rss
        report_rss(f"lane D: step for settlement {boundary}", rss, self.rss_budget_bytes,
                   logger=log, hint="pools, windows and checkpoints must stay on the lane D "
                   "eval host (W-OFFBOX, docs/INCIDENT_RULES.md IR010)")


__all__ = ["RSS_SAMPLE_SECONDS", "STEP_RSS_BUDGET_BYTES", "LaneDWorker", "process_rss_bytes"]
