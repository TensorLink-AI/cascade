"""A finished run is never lost to OUR disk (2026-11-02).

From 15:32 that day the checkpoint volume was full: every finished leg failed
its harvest with ``[Errno 28] No space left on device``, was classed
``[infra]`` (a burned MINER attempt) and — worse — had its pod torn down by
the leg's ``finally`` before anything was saved, so a completed, paid run
was gone. The class-level fixes here:

* :func:`local_disk_error` — ANY local-disk ``OSError`` (ENOSPC, EDQUOT,
  EROFS, EIO, EFBIG) anywhere in a leg is the OPERATOR's fault: never a
  miner burn (``faults.classify_leg_exception`` consults it first).
* :func:`guarded_harvest` — before each harvest attempt the box must hold the
  expected checkpoint + the ``min_free_disk_gb`` reserve (the space manager
  evicts reclaimable data first); a local-disk error mid-write drops the
  partial copy, frees space and retries, bounded
  (``[round] harvest_disk_retries`` with doubling waits); a copy that still
  differs from the pod's own listing after its re-harvest is retried too.
  Past the budget the checkpoint goes pod → R2 directly when R2 is
  configured (the pod-side safety net), else :class:`HarvestKeepPod` is
  raised — an operator fault that tells the leg paths to KEEP the pod (its
  ledger row stays; the requeued leg re-attaches and harvests it, the sweep
  keeps a finished-unharvested run for ``SWEEP_UNHARVESTED_KEEP_SECONDS``).
* the pod is torn down only after a verified harvest (``harvest_release``)
  or when the run is proven failed — never because OUR side failed.
"""

from __future__ import annotations

import errno
import logging
import shutil
import threading
import time
from collections.abc import Callable
from pathlib import Path

from .faults import _OperatorFault

log = logging.getLogger("cascade.trainer")

LOCAL_DISK_ERRNOS = frozenset({errno.ENOSPC, errno.EDQUOT, errno.EROFS, errno.EIO,
                               errno.EFBIG})


class HarvestKeepPod(_OperatorFault):
    """The harvest failed on the operator's side with the run complete on its
    pod: keep the pod (never torn down, never re-billed as a new rent), the
    leg requeues unburned and re-attaches."""

    keep_pod = True


class HarvestDiskFault(HarvestKeepPod):
    """Our disk could not hold the checkpoint within the retry budget."""


class HarvestIncomplete(HarvestKeepPod):
    """The box copy still differs from the pod's own listing (size/sha256)."""


def local_disk_error(exc: BaseException | None) -> OSError | None:
    """The local-disk ``OSError`` in ``exc``'s cause chain, or ``None``."""
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, OSError) and exc.errno in LOCAL_DISK_ERRNOS:
            return exc
        exc = exc.__cause__ or exc.__context__
    return None


def keeps_pod(exc: BaseException | None) -> bool:
    """Whether a leg failure says: keep the pod (see :class:`HarvestKeepPod`)."""
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if getattr(exc, "keep_pod", False):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


def keep_cost_line(runner, pod_id: str) -> str:
    """One log line pricing a kept pod: worst case at the price ceiling until
    the sweep releases a finished-unharvested run."""
    rnd = runner.cfg.round
    rate = max(float(getattr(rnd, "funded_max_price_per_hour", 0.0) or 0.0),
               float(getattr(rnd, "operator_max_price_per_hour", 0.0) or 0.0))
    hours = float(getattr(type(runner), "SWEEP_UNHARVESTED_KEEP_SECONDS", 6 * 3600.0)) / 3600.0
    return (f"pod {pod_id} KEPT after an operator-side harvest failure: it keeps billing at "
            f"≤ ${rate:.2f}/h until the requeued leg re-attaches and harvests it, or the "
            f"sweep releases it after {hours:.0f} h (worst case ≈ ${rate * hours:.2f})")


def _released(runner) -> bool:
    loc = runner.__dict__.get("_leg_local_obj")
    return bool(getattr(loc, "harvest_released", False))


def _drop_partial(dest: Path) -> None:
    """Remove a partial box copy (never one a pending upload still needs)."""
    from ..shared.disk_space import PENDING_SUFFIX

    if dest.with_name(dest.name + PENDING_SUFFIX).exists():
        return
    shutil.rmtree(dest, ignore_errors=True)
    for p in dest.parent.glob(f"{dest.name}.partial-*"):
        shutil.rmtree(p, ignore_errors=True)


def _room(runner, dest: Path, why: str) -> int:
    """Make room for one checkpoint at ``dest``; bytes still short (0 = ok)."""
    from ..shared.disk_space import GIB, free_bytes
    from .runner_common import reclaim_box_space

    rnd = runner.cfg.round
    need = int(float(getattr(rnd, "expected_checkpoint_gb", 0.0) or 0.0) * GIB)
    floor = int(float(getattr(rnd, "min_free_disk_gb", 0.0) or 0.0) * GIB)
    if need + floor <= 0:
        return 0
    free = free_bytes(dest)
    if free is None or free >= need + floor:
        return 0
    reclaim_box_space(runner, dest, need_bytes=need, floor_bytes=floor, why=why)
    free = free_bytes(dest)
    return 0 if free is None else max(0, need + floor - free)


def guarded_harvest(runner, dest: Path | str, attempt: Callable[[], object], *,
                    label: str, pod_id: str = "",
                    r2_fallback: Callable[[], object] | None = None):
    """Run ``attempt`` (one full ssh harvest + verify + upload) under the
    disk policy above; returns its result."""
    from .money_guard import alert, clear_alert

    dest = Path(dest)
    rnd = runner.cfg.round
    retries = max(0, int(getattr(rnd, "harvest_disk_retries", 3) or 0))
    wait = float(getattr(rnd, "harvest_disk_retry_seconds", 60) or 0)
    sleep = getattr(runner, "_harvest_retry_sleep", None) or time.sleep
    active = runner.__dict__.setdefault("_active_harvest_dirs", set())
    key = str(dest.resolve())
    active.add(key)
    loc = runner.__dict__.setdefault("_leg_local_obj", threading.local())
    loc.harvest_released = False
    last: BaseException | None = None
    try:
        for i in range(retries + 1):
            short = _room(runner, dest, f"harvest of {label}")
            if short:
                last = HarvestDiskFault(f"{short / 1e9:.1f} GB short of room for the checkpoint "
                                        "after evicting everything reclaimable")
                alert(runner, "harvest_disk", label[:16],
                      f"harvest of {label} blocked: {short / 1e9:.1f} GB short on {dest.parent}")
            else:
                try:
                    result = attempt()
                    clear_alert(runner, "harvest_disk", label[:16])
                    return result
                except HarvestIncomplete as e:
                    last = e
                except Exception as e:
                    err = local_disk_error(e)
                    if err is None:
                        raise
                    if _released(runner):
                        # The pod is gone (released after a verified copy):
                        # the box copy is the only one — keep it for the
                        # pending-upload retry, never delete it.
                        raise _OperatorFault(f"local disk fault after the pod was released "
                                             f"(the verified box copy is kept): {err}") from e
                    last = e
                    _drop_partial(dest)
                    alert(runner, "harvest_disk", label[:16],
                          f"harvest of {label} hit a local disk error: {err}")
            if i < retries:
                delay = wait * (2 ** i)
                log.error("harvest of %s failed on the operator's side (try %d/%d: %s) — %s; "
                          "retrying in %.0f s after freeing space", label, i + 1, retries + 1,
                          str(last)[:200], keep_cost_line(runner, pod_id or "?"), delay)
                sleep(delay)
        if r2_fallback is not None:
            try:
                log.warning("harvest of %s: box-side harvest exhausted its budget — pushing "
                            "pod → R2 directly (the pod-side safety net)", label)
                result = r2_fallback()
                clear_alert(runner, "harvest_disk", label[:16])
                return result
            except Exception as e:  # noqa: BLE001 — fall through: keep the pod
                from .faults import _FundedDiverged, _FundedTamper

                if isinstance(e, (_FundedTamper, _FundedDiverged)):
                    raise                          # a verdict on the run, not our disk
                log.error("harvest of %s: pod → R2 fallback failed too (%s)", label,
                          str(e)[:200])
        raise HarvestDiskFault(
            f"harvest of {label} failed on the operator's side after {retries + 1} tries "
            f"({last}); the pod is KEPT for the re-attach — nothing burned") from last
    finally:
        active.discard(key)
