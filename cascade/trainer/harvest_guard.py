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


def trained_seconds_of(runner, rec: dict | None = None) -> float:
    """How long the finished run a kept pod protects trained: the recorded
    figure, else the leg's expected wall (a finished leg trained ~its wall)."""
    v = (rec or {}).get("trained_seconds")
    if v:
        return float(v)
    try:
        return float(runner._leg_wall_seconds())
    except Exception:  # noqa: BLE001
        return 0.0


def keep_budget_seconds(runner, rec: dict | None = None) -> float:
    """The PRICED keep (money safety II): a pod kept after our harvest fault
    is worth keeping while keep cost < ``harvest_keep_cost_fraction`` x
    rate x trained hours — the rate cancels, so keep seconds < fraction x
    trained seconds — and never past ``harvest_keep_max_seconds`` (0 = no
    hard cap). 0 = do not keep."""
    rnd = runner.cfg.round
    frac = float(getattr(rnd, "harvest_keep_cost_fraction", 0.5) or 0.0)
    cap = float(getattr(rnd, "harvest_keep_max_seconds", 10800) or 0.0)
    priced = frac * trained_seconds_of(runner, rec)
    if priced <= 0:
        return 0.0
    return min(priced, cap) if cap > 0 else priced


def keep_cost_line(runner, pod_id: str) -> str:
    """One log line pricing a kept pod: worst case at the price ceiling until
    the sweep releases a finished-unharvested run."""
    rnd = runner.cfg.round
    rate = max(float(getattr(rnd, "funded_max_price_per_hour", 0.0) or 0.0),
               float(getattr(rnd, "operator_max_price_per_hour", 0.0) or 0.0))
    hours = keep_budget_seconds(runner) / 3600.0
    return (f"pod {pod_id} KEPT after an operator-side harvest failure: it keeps billing at "
            f"≤ ${rate:.2f}/h until the requeued leg re-attaches and harvests it, or it is "
            f"released after the priced keep of {hours:.1f} h (worst case ≈ "
            f"${rate * hours:.2f})")


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
                    label: str, pod_id: str = "", hotkey: str = "",
                    r2_fallback: Callable[[], object] | None = None):
    """Run ``attempt`` (one full ssh harvest + verify + upload) under the
    disk policy above; returns its result.

    The first local-disk error is recorded in the box health at once (it
    halts every new rent while this leg still retries). A copy that differs
    from a MINER-controlled pod's own listing twice is tamper — the payer can
    reach their pod — never a free operator-side retry (review F7)."""
    from .box_health import LOCAL_DISK, BoxHealth
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
    loc.leg_trained = True                  # a harvest only follows a finished run
    payer = bool(getattr(loc, "harvest_pod_payer", False))
    halted = False
    mismatches = 0
    last: BaseException | None = None
    try:
        for i in range(retries + 1):
            short = _room(runner, dest, f"harvest of {label}")
            if short:
                last = HarvestDiskFault(f"{short / 1e9:.1f} GB short of room for the checkpoint "
                                        "after evicting everything reclaimable")
                # a room shortfall is not a disk error: alert, no halt (N2)
                alert(runner, "harvest_disk", label[:16],
                      f"harvest of {label} blocked: {short / 1e9:.1f} GB short on {dest.parent}",
                      severity="critical")
            else:
                try:
                    result = attempt()
                    clear_alert(runner, "harvest_disk", label[:16])
                    if hotkey:
                        BoxHealth.for_runner(runner).forget_leg(hotkey)
                    return result
                except HarvestIncomplete as e:
                    last = e
                    mismatches += 1
                    if payer and mismatches >= 2:
                        raise _unverifiable(runner, dest, label, e, hotkey) from e
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
                          f"harvest of {label} hit a local disk error: {err}",
                          severity="critical")
                    if not halted:
                        # halt every new rent NOW, while this leg retries
                        BoxHealth.for_runner(runner).record(LOCAL_DISK, hotkey or label, str(err))
                        halted = True
            if i < retries:
                delay = wait * (2 ** i)
                log.error("harvest of %s failed on the operator's side (try %d/%d: %s) — %s; "
                          "retrying in %.0f s after freeing space", label, i + 1, retries + 1,
                          str(last)[:200], keep_cost_line(runner, pod_id or "?"), delay)
                sleep(delay)
        loc.harvest_r2_tried = r2_fallback is not None
        if r2_fallback is None:
            # Money safety II: the pod-side safety net is unavailable — the
            # kept pod is the ONLY copy of a finished, paid run. Say so loudly.
            log.critical("harvest of %s: R2 is NOT configured — no pod → R2 push possible; "
                         "the kept pod is the only copy of this finished run (configure the "
                         "R2 backup store to close this)", label)
            alert(runner, "harvest_no_r2", label[:16],
                  f"harvest of {label} failed on our side and R2 is not configured — the kept "
                  "pod is the only copy of the finished run", severity="critical")
        if r2_fallback is not None:
            try:
                log.warning("harvest of %s: box-side harvest exhausted its budget — pushing "
                            "pod → R2 directly (the pod-side safety net)", label)
                result = r2_fallback()
                clear_alert(runner, "harvest_disk", label[:16])
                if hotkey:                      # harvested: no stale lost-run record (N5)
                    BoxHealth.for_runner(runner).forget_leg(hotkey)
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


def _unverifiable(runner, dest: Path, label: str, exc: BaseException, hotkey: str):
    """Keep the evidence of a persistent payer-pod mismatch (the box copy and
    both manifests) and return the terminal, UNBURNED verdict (review F7)."""
    import json

    from .faults import CheckpointUnverifiable
    from .money_guard import alert

    loc = runner.__dict__.get("_leg_local_obj")
    ev = getattr(loc, "harvest_mismatch_evidence", None) or {}
    proof = dest.with_name(dest.name + ".unverifiable.json")
    try:
        proof.write_text(json.dumps({
            "hotkey": hotkey, "label": label, "box_copy": str(dest), "at": time.time(),
            "pod_manifest": ev.get("pod"), "box_manifest": ev.get("box"),
            "mismatches": ev.get("mismatches")}, default=list, indent=1), encoding="utf-8")
    except OSError as e:
        log.error("harvest of %s: could not write the mismatch evidence (%s)", label, e)
    alert(runner, "checkpoint_unverifiable", (hotkey or label)[:16],
          f"payer pod checkpoint of {label} differs from the pod's own listing after a retry — "
          f"failed UNBURNED; evidence kept: {dest} and {proof} (any burn is the owner's call)",
          severity="critical")
    log.critical("harvest of %s: checkpoint UNVERIFIABLE (%s) — evidence kept at %s and %s; "
                 "terminal, nothing burned", label, str(exc)[:200], dest, proof)
    return CheckpointUnverifiable(f"checkpoint unverifiable after a retry: {exc}")


def handle_our_side_fault(runner, gen, exc: BaseException, *, pod_id: str, payer: bool,
                          trained: bool, released: bool) -> bool:
    """One leg failure seen by the leg paths: record any our-side fault in
    the box health (may halt every rent; one critical alert), and decide
    whether to KEEP the pod. Returns True = keep it (never torn down now).

    A pod is kept only while the finished run is still on it (trained, not
    yet released, a keep-pod or local-disk fault) and for at most
    ``[round] harvest_keep_max_seconds`` from the first keep — a payer pod
    bills the MINER while kept (review F6). Every keep and every lost run is
    priced in one critical alert so the owner can make it good."""
    from ..funding.queue import KEPT_POD_CLASS
    from .box_health import BoxHealth, miner_lost_usd, record_leg_fault
    from .money_guard import alert

    hk = gen.hotkey
    ours = keeps_pod(exc) or local_disk_error(exc) is not None
    keep = bool(ours and trained and not released)
    if keep:
        bh = BoxHealth.for_runner(runner)
        prev = bh.leg(hk) or {}
        since = prev.get("kept_since") if prev.get("pod_id") == pod_id else None
        now = time.time()
        trained_s = trained_seconds_of(runner, prev if prev.get("pod_id") == pod_id else None)
        limit = keep_budget_seconds(runner, {"trained_seconds": trained_s})
        if since is None:
            since = now
        if now - since >= limit:
            keep = False                  # the priced keep is spent (0 = never keep)
        loc = runner.__dict__.get("_leg_local_obj")
        record_leg_fault(runner, hk, exc, trained=True, pod_id=pod_id, payer=payer, kept=keep,
                         ref=gen.ref, extra={"kept_since": since, "trained_seconds": trained_s,
                                             "r2_tried": bool(getattr(loc, "harvest_r2_tried",
                                                                      False))}
                         if keep else {})
    else:
        record_leg_fault(runner, hk, exc, trained=trained and ours, pod_id=pod_id, payer=payer,
                         kept=False, ref=gen.ref)
    who = "MINER" if payer else "operator"
    if keep:
        runner._record_funded_failure(hk, f"operator-side harvest fault, pod kept for the "
                                      f"re-attach: {exc}", miner_fault=False,
                                      error_class=KEPT_POD_CLASS, burn=False)
        line = keep_cost_line(runner, pod_id)
        log.error("leg %s: %s", hk[:12], line)
        if payer:
            alert(runner, "miner_billed_keep", hk[:16], f"{line} — billed to the {who}; make "
                  "it good if the run is lost", severity="critical")
        return True
    if ours and trained and not released:
        lost = miner_lost_usd(runner)
        loc = runner.__dict__.get("_leg_local_obj")
        r2 = ("the pod → R2 push was tried and failed" if getattr(loc, "harvest_r2_tried", False)
              else "R2 is NOT configured (no pod → R2 push was possible)")
        alert(runner, "miner_billed_loss" if payer else "operator_billed_loss", hk[:16],
              f"finished run of {hk[:12]} LOST to an operator-side fault (pod {pod_id} "
              f"released after the priced keep; {r2}): ≤ ${lost:.2f} billed to the {who}; its "
              "retrain is an operator make-good (owner approval)", severity="critical")
    return False
