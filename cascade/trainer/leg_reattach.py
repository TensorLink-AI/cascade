"""Lane B / lane C leg re-attach after a trainer restart (owner 2026-11-26).

Lane A detached legs survive a trainer restart: the rolling scheduler re-runs
the in-flight leg, the funded rent path ADOPTS the leg's ledgered pod (its
name is a function of the seed epoch and the hotkey) and the detached
dispatcher ATTACHES to the worker already running there
(:func:`cascade.trainer.remote.find_detached_run`). Lane B and lane C legs
used to requeue every persisted open leg on restart instead, wasting the
miner-paid pod and its partial run.

The drivers now re-run a persisted open leg through the SAME funded path in
*attach-only* mode (:func:`reattach_only`, a per-thread flag):

* the rent path refuses anything but adopting a ledgered pod confirmed live
  (never a fresh rent: :func:`rent_refusal`);
* the detached dispatcher refuses to launch a second worker when no live or
  cleanly finished run of the leg is on the pod
  (:class:`cascade.trainer.remote.ReattachMissing`).

Either refusal is a :data:`REATTACH_REFUSED` verdict, which the drivers
requeue exactly as before (uncharged, ``no_capacity``, operator-billed). The
leg's funded queue entry is never marked in flight again, so its attempt id
and its charge stay those of the original dispatch. A leg whose heartbeat is
older than ``[round] dispatch_reattach_grace_seconds`` is requeued without a
re-attach attempt.

Pure stdlib; nothing here touches a pod, the chain or a key.
"""

from __future__ import annotations

import contextlib
import threading
import time

REATTACH_REFUSED = "re-attach refused"
DEFAULT_GRACE_SECONDS = 900
# A heartbeat is written at most this often (a tick saves the driver state
# anyway; this bounds the extra writes when ticks are fast).
HEARTBEAT_SECONDS = 60.0

_LOCAL = threading.local()


def _wall() -> float:
    return time.time()


@contextlib.contextmanager
def reattach_only():
    """While active on this thread, the funded rent path may only adopt the
    leg's live ledgered pod and the detached dispatcher may only attach."""
    prev = bool(getattr(_LOCAL, "on", False))
    _LOCAL.on = True
    try:
        yield
    finally:
        _LOCAL.on = prev


def attach_only() -> bool:
    """True inside :func:`reattach_only` on this thread."""
    return bool(getattr(_LOCAL, "on", False))


def grace_seconds(cfg) -> int:
    """``[round] dispatch_reattach_grace_seconds`` (the lane A dispatcher's
    re-attach grace), default 900."""
    rnd = getattr(cfg, "round", None)
    try:
        return max(0, int(getattr(rnd, "dispatch_reattach_grace_seconds",
                                  DEFAULT_GRACE_SECONDS)))
    except (TypeError, ValueError):
        return DEFAULT_GRACE_SECONDS


def fresh(rec, now: float, grace: int) -> bool:
    """A persisted re-attach record whose heartbeat is within ``grace``."""
    if not isinstance(rec, dict):
        return False
    try:
        seen = float(rec.get("seen_at", 0.0) or 0.0)
    except (TypeError, ValueError):
        return False
    return seen > 0 and 0 <= float(now) - seen <= float(grace)


def rent_refusal(*, prior_alloc: bool, pod_live) -> str:
    """The attach-only rent verdict: ``""`` (adopt the live ledgered pod) or
    why nothing may be rented. Unknown liveness refuses too: an attach-only
    leg never risks a fresh rent of the same pod name."""
    if not attach_only():
        return ""
    if not prior_alloc:
        return f"{REATTACH_REFUSED}: the leg's pod is not on the funded ledger"
    if pod_live is not True:
        state = "gone" if pod_live is False else "of unknown liveness"
        return f"{REATTACH_REFUSED}: the leg's ledgered pod is {state}"
    return ""


def refusal_failure(why: str):
    """The driver verdict for a refused re-attach: requeued uncharged and
    operator-billed, exactly as a restart requeue was before re-attach."""
    from .lane_c_driver import LegFailure

    msg = str(why or "")
    if not msg.startswith(REATTACH_REFUSED):
        msg = f"{REATTACH_REFUSED}: {msg}"
    f = LegFailure(msg + " — requeued", miner_fault=False, error_class="no_capacity",
                   burn=False)
    f.operator_billed = True
    return f


def is_refusal(failure_or_text) -> bool:
    msg = str(getattr(failure_or_text, "msg", failure_or_text) or "")
    return REATTACH_REFUSED in msg


def chain_refusal(exc: BaseException | None) -> str:
    """The refusal text anywhere in ``exc``'s cause chain, else ``""``."""
    seen = 0
    while exc is not None and seen < 16:
        if is_refusal(str(exc)):
            return str(exc)
        exc = exc.__cause__ or exc.__context__
        seen += 1
    return ""
