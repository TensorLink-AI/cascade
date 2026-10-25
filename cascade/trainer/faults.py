"""Leg fault classes and failure classifiers for the trainer runner."""

from __future__ import annotations

import logging
import re
import subprocess

from ..shared.hippius import StorageError

log = logging.getLogger("cascade.trainer")


# _funded_pod_identity_mismatch's reason when the payer's key is gone from the
# vault: the leg paths class it _FundedKeyLost (auth), not _FundedTamper.
KEY_LOST_REASON = "payer key unavailable to re-identify the pod"


class _FundedTamper(Exception):
    """A funded pod failed its identity pin (replaced under the same name, or
    a different container answering at its address). Miner fault, terminal,
    hotkey spent."""


class _FundedKeyLost(Exception):
    """The payer's Lium key is no longer in the vault while their pod is live
    (aged out mid-leg, or removed by the intake): the pod can be neither
    re-identified nor stopped from here. NOT tamper — nothing about the pod
    changed — and never infra: the miner's own credential lapsed. Classed
    ``auth`` like the seating-time check, unburned."""


class _FundedDiverged(Exception):
    """A leg came back with a checkpoint whose weights are NaN/inf
    (:class:`cascade.eval.checkpoint_guard.CheckpointDiverged`): the run
    diverged on the miner's corpus. A challenger's is a generator fault
    (their shot, like the worker's own rc=3 on the same condition — see
    ``toto2_trainer.check_loss_finite``); the king's fails the king leg,
    since a NaN entry would break every validator's verdict."""


class _OperatorFault(RuntimeError):
    """A funded leg lost to the OPERATOR's side (or its provider's) while the
    miner's pod and run were fine: the checkpoint upload failing after every
    retry, a stale worker image the provider booted, the CPU generation tier.
    Infra, requeued, and it never burns one of the entry's attempts — that
    budget is for failures on the miner's own pod (DEC-CA-0036)."""


class _FundedLegSkip(Exception):
    """A funded challenger leg that settled its own queue verdict (rent/setup
    failure) and must simply be dropped from the round — never retried on the
    operator fleet, which would silently move the bill."""


class _FundedOperatorFallback(Exception):
    """A funded leg (or the JIT king) that could not rent on the marketplace
    and — with ``[round] funded_operator_fallback`` ON and operator final
    lanes on file — runs on an operator lane instead, operator-billed. The
    explicit, owner-armed exception to _FundedLegSkip's rule."""


class _FundedOperatorRent(_FundedOperatorFallback):
    """An operator-billed rolling leg with NO operator lane on file: rent a pod
    on the OPERATOR's Lium account for it (``_run_operator_rented_leg``).
    Raised only where the rolling challenger path opted in; every other
    handler sees the parent class and behaves exactly as before."""


class _LaneDeadlinePassed(Exception):
    """No operator lane came free before the leg's latest safe start: a leg
    started now could not land before its target boundary. Raised by the lane
    pool's deadline form so the leg requeues instead of blocking on the pool
    and starting hours late."""


def _http_status_in_chain(exc: BaseException | None) -> int | None:
    """First HTTP status code found walking an exception's cause chain."""
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if isinstance(status, int):
            return status
        exc = exc.__cause__ or exc.__context__
    return None


# Fault class a funded leg gets the FIRST time its generator stalls on a pod
# ("generator_stalled: no series for Ns"). Not a burn: a stall is also what an
# environment fault (a host running a stale worker image) looks like from the
# orchestrator. The entry requeues unburned (attempts untouched); the SAME
# hotkey stalling again on its next leg is then the miner's fault.
STALL_CLASS = "stall"
_STALL_MARKER = "generator_stalled"


def classify_funded_worker_failure(rc: int | None, text: str, *,
                                   stalled_before: bool) -> tuple[bool, str, bool]:
    """``(miner_fault, error_class, burn_attempt)`` for a funded worker exit.

    ``rc == 3`` is the worker's "miner submission rejected" exit (CorpusError).
    A rejection whose text is a stream STALL is classed :data:`STALL_CLASS`
    (infra-side, unburned) the first time and ``"generator"`` (the miner's
    shot) only when this hotkey already stalled once before. Every other
    rejection stays ``"generator"``; any other exit is ``"infra"`` and burns
    one of the entry's retry attempts, as before.
    """
    if rc == 3:
        if _STALL_MARKER in (text or "") and not stalled_before:
            return False, STALL_CLASS, False
        return True, "generator", False
    return False, "infra", True


# Storage-layer failure fingerprints: a run that died inside the artifact /
# checkpoint fetch path (registry 401/403, Hippius chunk errors) failed at OUR
# storage boundary, not in the generator. Registry blips are global, so the
# lane retry backs off before its second attempt instead of hitting the same
# blip on another host.
_STORAGE_FAILURE_MARKERS = (
    "storageerror", "hippius", "hf_hub_download", "file_download",
    "generator_artifact_unreachable",
)


def _transport_failure(exc: BaseException) -> bool:
    """A dispatch failure that says the POD is unreachable rather than the
    leg wrong: an ssh transport exit (255 — refused, unreachable, sshd dead),
    a detached worker that stopped answering polls, or a failed detached
    launch. Miner faults (import error, NaN, OOM) are never this."""
    from .remote import RemoteDispatchError

    if isinstance(exc, RemoteDispatchError):
        if getattr(exc, "returncode", None) == 255:
            return True
        msg = str(exc)
        return ("pod unreachable" in msg or "detached launch failed" in msg
                or "Connection refused" in msg)
    if isinstance(exc, subprocess.CalledProcessError):
        return exc.returncode == 255
    return "Connection refused" in str(exc)


def _quarantine_lane_host(host, reason: str) -> None:
    """Quarantine the pod's host IP (Lium relists the same machine under new
    executor ids) after a transport failure; loopback/profile hosts skipped.
    Best-effort — never lets a quarantine write sink the dispatch path."""
    ip = str(getattr(host, "host", "") or "")
    if not ip or ip.startswith("127.") or ip in ("localhost", "::1"):
        return
    try:
        from ..provision.core import record_host_quarantine

        record_host_quarantine(ip, reason)
    except Exception as e:  # noqa: BLE001
        log.warning("host quarantine for %s failed (ignored): %s", ip, e)


def _storage_failure(exc: BaseException) -> bool:
    """True when an exception (or, for remote failures, the stderr tail in its
    message) points at the storage layer rather than the challenger's code."""
    if isinstance(exc, StorageError):
        return True
    msg = str(exc).lower()
    return any(m in msg for m in _STORAGE_FAILURE_MARKERS)


# Text markers of a payer key that cannot pay: Lium's "insufficient balance"
# family (classified "infra" by the rent taxonomy) and HTTP 402.
_NO_FUNDS_MARKERS = ("insufficient balance", "insufficient funds", "not enough balance",
                     "balance is too low", "out of funds", "payment required",
                     "top up", "low balance")
_NO_FUNDS_CODE_RE = re.compile(r"(?<![0-9a-z])402(?![0-9a-z])")


def king_key_failed(verdict: tuple | None, exc: BaseException | None = None) -> bool:
    """DEC-CA-0055 item 6: did a king leg on the king's OWN key fail because
    of the key — rejected/revoked (``auth``), lost mid-leg, absent/expired in
    the vault, or out of funds — rather than the market or our side? Only
    then does the operator take the leg over (and the king is slashed).
    ``verdict`` is the leg's recorded ``(msg, miner_fault, error_class,
    burn)``; ``exc`` the raised exception (its text is checked for no-funds
    markers too)."""
    msg, cls = "", ""
    if verdict:
        msg = str(verdict[0] or "")
        cls = str(verdict[2] or "") if len(verdict) > 2 else ""
    if cls in ("auth", "tamper"):
        return True
    text = (msg + " " + (str(exc) if exc is not None else "")).lower()
    return any(m in text for m in _NO_FUNDS_MARKERS) or bool(_NO_FUNDS_CODE_RE.search(text))
