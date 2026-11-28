"""Per-step peak-RSS sampler shared by every off-thread box step (retro P2 iii).

#1000 (lane D step, W-OFFBOX) and #1005 (MarketCast round, W-MKTOFF) each
grew their own sampler after the validator was OOM-killed with no line saying
why. This is the one implementation: :func:`step_rss` (context manager) /
:func:`rss_budgeted` (decorator) sample this process's RSS on a daemon thread
while a step runs, log ``<label>: rss start S MB, peak P MB (+G MB)`` when it
ends and an ERROR carrying :data:`LOG_TOKEN` when the growth is over the
step's budget, so cascade-ops can alert on one pattern for every step.

The box (4 cores, 7 GB) runs the validator, trainer, intake and provisioners
together: a step's growth is the memory it took from all of them. RSS is
process-wide, so concurrent steps see each other's growth (an over-count,
never an under-count). Pure stdlib.
"""

from __future__ import annotations

import contextlib
import functools
import logging
import os
import threading
from collections.abc import Callable, Iterator

log = logging.getLogger(__name__)

RSS_SAMPLE_SECONDS = 0.25
DEFAULT_STEP_BUDGET_BYTES = 512 * 1024 * 1024
LOG_TOKEN = "box_budget_exceeded"
_MB = 1024 * 1024


def process_rss_bytes() -> int | None:
    """This process's resident set size (``/proc/self/statm``), ``None`` where
    it cannot be read."""
    try:
        with open("/proc/self/statm", encoding="ascii") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return None


class RssWatch:
    """Samples the RSS on a daemon thread from ``start()`` to ``stop()``."""

    def __init__(self, probe: Callable[[], int | None] = process_rss_bytes,
                 interval: float = RSS_SAMPLE_SECONDS):
        self._probe = probe
        self._interval = float(interval)
        self._stop = threading.Event()
        self.start_rss = probe()
        self.peak = self.start_rss
        self._thread = threading.Thread(target=self._loop, name="rss-watch", daemon=True)

    def _sample(self) -> None:
        v = self._probe()
        if v is not None and (self.peak is None or v > self.peak):
            self.peak = v

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            self._sample()

    def start(self) -> RssWatch:
        self._thread.start()
        return self

    def stop(self) -> dict | None:
        self._stop.set()
        if self._thread.ident is not None:
            self._thread.join(5.0)
        self._sample()
        if self.start_rss is None or self.peak is None:
            return None
        return {"start": int(self.start_rss), "peak": int(self.peak),
                "growth": int(self.peak - self.start_rss)}


def report_rss(label: str, rss: dict | None, budget_bytes: int, *,
               logger: logging.Logger | None = None, hint: str = "") -> bool:
    """Log one step's RSS; ``True`` iff it grew over ``budget_bytes`` (then an
    ERROR naming the budget and ``hint``, the place the work belongs)."""
    lg = logger or log
    if rss is None:
        lg.warning("%s: process RSS unreadable", label)
        return False
    lg.info("%s: rss start %d MB, peak %d MB (+%d MB)", label,
            rss["start"] // _MB, rss["peak"] // _MB, rss["growth"] // _MB)
    if rss["growth"] <= int(budget_bytes):
        return False
    lg.error("%s: %s grew the process RSS by %d MB, over the %d MB box budget (peak %d MB)%s",
             LOG_TOKEN, label, rss["growth"] // _MB, int(budget_bytes) // _MB,
             rss["peak"] // _MB, f": {hint}" if hint else "")
    return True


class StepRss:
    """What :func:`step_rss` yields: ``rss`` / ``over`` are set on exit."""

    rss: dict | None = None
    over: bool = False


@contextlib.contextmanager
def step_rss(label: str, budget_bytes: int = DEFAULT_STEP_BUDGET_BYTES, *,
             logger: logging.Logger | None = None, hint: str = "",
             probe: Callable[[], int | None] = process_rss_bytes,
             interval: float = RSS_SAMPLE_SECONDS,
             watch: RssWatch | None = None) -> Iterator[StepRss]:
    """Sample the RSS around the ``with`` body (also when it raises) and
    report it. ``watch`` passes a sampler already started (a caller that
    must start sampling before it spawns the step's thread)."""
    out = StepRss()
    w = watch if watch is not None else RssWatch(probe, interval).start()
    try:
        yield out
    finally:
        out.rss = w.stop()
        out.over = report_rss(label, out.rss, budget_bytes, logger=logger, hint=hint)


def rss_budgeted(label: str, budget_bytes: int = DEFAULT_STEP_BUDGET_BYTES, *,
                 hint: str = "", logger: logging.Logger | None = None):
    """Decorator form of :func:`step_rss` for an off-thread step function."""
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*a, **k):
            with step_rss(label, budget_bytes, logger=logger, hint=hint):
                return fn(*a, **k)
        return wrapper
    return deco


__all__ = ["DEFAULT_STEP_BUDGET_BYTES", "LOG_TOKEN", "RSS_SAMPLE_SECONDS", "RssWatch",
           "StepRss", "process_rss_bytes", "report_rss", "rss_budgeted", "step_rss"]
