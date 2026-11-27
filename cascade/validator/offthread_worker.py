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

Generic thread infrastructure, deliberately not named ``lane_d_*`` and
importing no lane D module: ``loop.py`` imports it, and the audit's import
closure (which reaches ``loop.py``) must load no lane D producer module (W9
invariant 12).
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

log = logging.getLogger(__name__)


class LaneDWorker:
    """One lane D step at a time, on a daemon thread the caller never joins
    implicitly."""

    def __init__(self, step: Callable[[int], object],
                 on_abandon: Callable[[int], object] | None = None):
        self.step = step
        self.on_abandon = on_abandon
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
            t = threading.Thread(target=self._run, args=(b, abandoned),
                                 name=f"lane-d-step-{b}", daemon=True)
            self._thread = t
        try:
            t.start()
        except RuntimeError as e:
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

    def _run(self, boundary: int, abandoned: int | None = None) -> None:
        if abandoned is not None:
            log.warning("lane D: settlement %d was never settled and the head moved on to "
                        "%d; it is given up", abandoned, boundary)
            if self.on_abandon is not None:
                try:
                    self.on_abandon(abandoned)
                except BaseException as e:  # noqa: BLE001 — never blocks the next step
                    log.warning("lane D: recording settlement %d as abandoned failed (%s)",
                                abandoned, e)
        try:
            self.step(boundary)
        except BaseException as e:  # noqa: BLE001 — a step never escapes its thread
            with self._lock:
                self.last_error = e
                self._busy = False
                if self._failed is None or boundary > self._failed:
                    self._failed = boundary
            log.warning("lane D: step for settlement %d failed (%s: %s); retried at a "
                        "later tick", boundary, type(e).__name__, e)
            return
        with self._lock:
            self._completed.add(boundary)
            if self._failed == boundary:
                self._failed = None
            self.last_error = None
            self._busy = False
        log.info("lane D: step for settlement %d completed", boundary)


__all__ = ["LaneDWorker"]
