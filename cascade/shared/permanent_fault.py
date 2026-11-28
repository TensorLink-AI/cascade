"""Permanent faults: errors that will not heal by retrying, and the page.

Retro 2026-11-28 class C4 ("silent fallback or degrade instead of a loud
failure"): the default error path of every settlement step is "defer and
retry". A permanent fault (a library missing from the process that runs the
step, a GPU the pinned torch has no kernels for, a contract or image pin that
does not match) therefore looked like a transient one for days: lane C's
timestral ImportError was deferred for ~58 h as "MarketCast report absent".

This module is stdlib-only (no torch, no bittensor) and does two things:

* :func:`classify` names a permanent fault from the exception itself (its
  type, then a short list of message markers). Anything it does not name is
  ``None`` = transient: the existing retry/defer path is unchanged.
* :class:`RepeatTracker` names the second kind: the SAME error (digits and
  hex normalised away) at the same site ``threshold`` times in a row.

:func:`page` writes the one log line cascade-ops matches
(``PERMANENT FAULT [<class>] <site>: <detail> — retries stopped; owner paged``)
as a CRITICAL event, so the owner is paged through the cascade-ops alert
sinks (webhook / Telegram from ``/etc/cascade-ops.env``; ``alerts.jsonl``
always). It never raises.

Paging is classification and alerting only: callers stop their IN-WINDOW
retries for a permanent fault (those hold the lane-A settlement for up to an
hour and cannot succeed), and keep every consensus-visible outcome (deferral
rows, report bytes) exactly as before.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

__all__ = [
    "PAGE_MARKER",
    "PermanentFault",
    "RepeatTracker",
    "classify",
    "classify_text",
    "normalise",
    "page",
]

log = logging.getLogger("cascade.permanent_fault")

PAGE_MARKER = "PERMANENT FAULT"

# Exception TYPE names (anywhere in the cause chain) that a retry cannot fix:
# the code or environment of the process that ran the step is wrong.
_PERMANENT_TYPES: dict[str, str] = {
    "ImportError": "missing_library",
    "ModuleNotFoundError": "missing_library",
    "TimestralUnavailable": "missing_library",
    "CodePathMismatch": "code_path_mismatch",
    "LegImageMismatch": "image_mismatch",
    "TrainImageMismatch": "image_mismatch",
    "HandshakeMismatch": "contract_mismatch",
    "HashsetMismatch": "contract_mismatch",
}

# Message markers (lower-cased substring) → class. Ordered; first match wins.
# Each one is a fault seen this week whose retry could only fail the same way.
_PERMANENT_MARKERS: tuple[tuple[str, str], ...] = (
    # 2026-11-28 01:44: Blackwell sm_120 under torch 2.4.1+cu124
    ("no kernel image is available for execution on the device", "gpu_arch_unsupported"),
    ("is not compatible with the current pytorch installation", "gpu_arch_unsupported"),
    ("sm_120", "gpu_arch_unsupported"),
    # 2026-11-28 08:05: timestral imported in the validator process
    ("no module named", "missing_library"),
    ("cannot import name", "missing_library"),
    # contract / config / arch pins that do not match the release
    ("contract_digest mismatch", "contract_mismatch"),
    ("contract mismatch", "contract_mismatch"),
    ("base_arch_digest", "contract_mismatch"),
    ("worker_code_fingerprint", "image_mismatch"),
    ("image mismatch", "image_mismatch"),
    ("lock drift", "dependency_drift"),
    ("uv.lock", "dependency_drift"),
)

_TRANSIENT_FIRST = ("timed out", "timeout", "connection reset", "connection refused",
                    "temporarily unavailable", "rate limit", "too many requests")
_TRANSIENT_CODE_RE = re.compile(r"(?<![0-9a-z])(?:429|50[234])(?![0-9a-z])")


@dataclass(frozen=True)
class PermanentFault:
    """``kind`` names the class; ``rule`` is the matching type name or marker
    (the evidence cited in the page); ``detail`` the error text, trimmed."""

    kind: str
    rule: str
    detail: str


def _chain(exc: BaseException | None):
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        yield exc
        exc = exc.__cause__ or exc.__context__


def classify_text(text: str) -> PermanentFault | None:
    """A permanent fault named from error TEXT alone (a pod judge's remote
    traceback line, a worker's stderr tail), else ``None``. A transient
    marker (timeout, refused connection, 5xx) wins: a network error that
    happens to mention a module is not a missing library."""
    low = (text or "").lower()
    if (not low or any(m in low for m in _TRANSIENT_FIRST)
            or _TRANSIENT_CODE_RE.search(low)):
        return None
    for name, kind in _PERMANENT_TYPES.items():
        if re.search(r"(?<![A-Za-z0-9_])" + re.escape(name) + r"(?![A-Za-z0-9_])", text or ""):
            return PermanentFault(kind, name, (text or "")[:300])
    for marker, kind in _PERMANENT_MARKERS:
        if marker in low:
            return PermanentFault(kind, marker, (text or "")[:300])
    return None


def classify(exc: BaseException | None) -> PermanentFault | None:
    """A :class:`PermanentFault` when ``exc`` (or anything in its cause chain)
    is one a retry cannot fix; ``None`` (transient: retry/defer as before)
    otherwise."""
    for e in _chain(exc):
        for cls in type(e).__mro__:
            kind = _PERMANENT_TYPES.get(cls.__name__)
            if kind is not None:
                return PermanentFault(kind, cls.__name__, f"{type(e).__name__}: {e}"[:300])
    if exc is None:
        return None
    return classify_text(f"{type(exc).__name__}: {exc}")


_NORM_RE = re.compile(r"0x[0-9a-f]+|[0-9a-f]{12,}|\d+(?:\.\d+)?", re.IGNORECASE)


def normalise(text: str) -> str:
    """The error text with numbers, hashes and hex ids folded to ``#`` so the
    same fault in a new round compares equal."""
    return _NORM_RE.sub("#", (text or "").strip())[:240]


class RepeatTracker:
    """``threshold`` identical (normalised) errors in a row at one site =
    permanent (``kind="repeated"``). A success, or a different error, resets
    the site. In memory: a restart starts the count again (bounded)."""

    def __init__(self, threshold: int = 3) -> None:
        self.threshold = max(2, int(threshold))
        self._last: dict[str, tuple[str, int]] = {}

    def failed(self, site: str, text: str) -> PermanentFault | None:
        key = normalise(text)
        prev, n = self._last.get(site, ("", 0))
        n = n + 1 if prev == key else 1
        self._last[site] = (key, n)
        if n >= self.threshold:
            return PermanentFault("repeated", f"same error {n}x in a row",
                                  (text or "")[:300])
        return None

    def ok(self, site: str) -> None:
        self._last.pop(site, None)

    def count(self, site: str) -> int:
        return self._last.get(site, ("", 0))[1]


def page(site: str, fault: PermanentFault, *, action: str = "retries stopped") -> str:
    """Log the cascade-ops page line (CRITICAL) and return it. Never raises."""
    try:
        line = (f"{PAGE_MARKER} [{fault.kind}] {site}: {fault.detail} (rule: {fault.rule}) "
                f"— {action}; owner paged")
        line = line.replace("\n", " ")
        log.critical("%s", line)
        return line
    except Exception:  # noqa: BLE001 — paging never breaks the caller
        return ""
