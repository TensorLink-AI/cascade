"""Best-effort public stage events for the exact admitted funded attempt."""

import logging
import threading
from contextlib import contextmanager

log = logging.getLogger(__name__)


def funded_attempt_id(queue, hotkey, ref, started_block):
    """Optional telemetry identity; a failed extra read must not stop training."""
    try:
        entry = queue.get(hotkey)
        if entry.status == "in_flight" and entry.ref == ref and entry.started_block == started_block:
            return entry.attempt_id
    except Exception:  # noqa: BLE001
        log.warning("submission attempt telemetry unavailable")
    return None


@contextmanager
def progress_scope(runner, gen):
    """Keep A's progress identity across miner-rent to operator-fallback handoff."""
    local = runner.__dict__.setdefault("_leg_local_obj", threading.local())
    old = getattr(local, "funded_progress", None)
    local.funded_progress = progress_binding(runner, gen)
    try:
        yield
    finally:
        local.funded_progress = old


def progress_binding(runner, gen):
    try:
        q = runner._funded_queue()
        e = q.get(gen.hotkey) if q is not None else None
        if (getattr(e, "status", None) == "in_flight"
                and e.ref.rsplit("|", 1)[-1] == gen.ref):
            return q, gen.hotkey, e.ref, e.started_block, e.funded_at, e.attempt_id
    except Exception:  # noqa: BLE001 — telemetry never changes the work verdict
        log.warning("submission progress binding unavailable")
    return None


def report_progress(runner, phase: str, reason: str = "") -> None:
    local = runner.__dict__.get("_leg_local_obj")
    binding = getattr(local, "funded_progress", None)
    if binding is None:
        return
    q, hotkey, ref, started, funded, attempt = binding
    try:
        q.set_phase(hotkey, phase, reason=reason, expect_ref=ref,
                    expect_started_block=started, expect_funded_at=funded,
                    expect_attempt_id=attempt)
    except Exception:  # noqa: BLE001 — never publish an exception containing credentials
        log.warning("submission progress update unavailable (%s)", phase)
