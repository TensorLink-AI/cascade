"""LANE_D_PLAN §1.3b "Held-out suite = the cross-lane rotating sentinel", §16
rules 1, 2 and 4, T7b (W3b3): the era's sentinel, swept and published for
every new lane D king.

For every NEW KING_D (a ``lane-d-report/1`` whose ``crowned`` is set), the
era's §16 sentinel suite — :func:`cascade.shared.sentinel.sentinel_for_era`
over ``[v2] sentinel_suites``, never a fixed suite — is swept for the new
king, the previous KING_D and the house in the settlement's bench
:class:`~cascade.validator.lane_d_bench.SweepSession` (one deduplicated host
call; rows the session already holds are re-used). A ``lane-d-sentinel/1``
record is built and published under :func:`sentinel_key`.

The sentinel NEVER gates: the step runs after ``settle_lane_d`` on its
returned report, reads its inputs without mutating them and its only output
is the published record. A host error or a faulty model is recorded (status
``"error"`` / ``"fault"``), never raised; a store failure is returned as
``published: False`` with the exact text, so the caller can retry the same
bytes.

Publishing is idempotent: identical text at the key writes nothing; a
published non-error record is never replaced; an ``error`` record (or an
unreadable object) at the key is replaced by a later sweep (the typed
retry). The record is strict canonical JSON with no
wall-clock or process-local field, so every validator writes the same bytes.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from cascade.shared.hippius import ObjectNotFound, StorageError
from cascade.shared.sentinel import sentinel_for_era
from cascade.validator.lane_d_bench import METRICS, _suite_rows

SCHEMA = "lane-d-sentinel/1"
ROLES = ("king", "previous_king", "house")
STATUSES = ("ok", "fault", "error")
KEY_PREFIX = "lane-d/sentinel"
CONTENT_TYPE = "application/json"
PUBLIC_ACL = "public-read"


def sentinel_key(era_block: int, king_digest: str) -> str:
    """The deterministic object key of a king's sentinel record for an era."""
    return f"{KEY_PREFIX}/era-{int(era_block)}/{king_digest}.json"


def sentinel_text(record: Mapping) -> str:
    """Strict canonical JSON (sorted keys, compact, NaN/Infinity refused)."""
    return json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sentinel_bytes(record: Mapping) -> bytes:
    """The published bytes of ``record``: :func:`sentinel_text`, UTF-8."""
    return sentinel_text(record).encode("utf-8")


# ── argument checks ──────────────────────────────────────────────────────────


def _check_era_block(era_block: object) -> int:
    if isinstance(era_block, bool) or not isinstance(era_block, int):
        raise TypeError(f"era_block must be an int; got {era_block!r}")
    if era_block < 0:
        raise ValueError(f"era_block must be >= 0; got {era_block!r}")
    return era_block


def _check_suites(sentinel_suites: object) -> tuple[str, ...]:
    if isinstance(sentinel_suites, (str, bytes)) or not isinstance(sentinel_suites, Sequence):
        raise TypeError(f"sentinel_suites must be a sequence of names; got {sentinel_suites!r}")
    names = tuple(sentinel_suites)
    if not names:
        raise ValueError("sentinel_suites is empty")
    for s in names:
        if not isinstance(s, str) or not s:
            raise ValueError(f"sentinel suite {s!r} is not a non-empty str")
    if len(set(names)) != len(names):
        raise ValueError(f"sentinel_suites has duplicates: {list(names)}")
    return names


def _digest(report: Mapping, path: tuple[str, ...]) -> str:
    v: Any = report
    for k in path:
        if not isinstance(v, Mapping):
            raise ValueError(f"report[{']['.join(path)}] is missing")
        v = v.get(k)
    if not isinstance(v, str) or not v:
        raise ValueError(f"report[{']['.join(path)}] must be a non-empty str; got {v!r}")
    return v


def _roles(report: Mapping) -> dict[str, str]:
    return {"king": _digest(report, ("crowned",)),
            "previous_king": _digest(report, ("king_before", "digest")),
            "house": _digest(report, ("house", "digest"))}


# ── per-model numbers ────────────────────────────────────────────────────────


def _geomean(values: list[float]) -> float:
    return math.exp(math.fsum(math.log(v) for v in values) / len(values))


def _model(digest: str, rec: Mapping, suite: str) -> tuple[dict, str | None]:
    """``(model entry, fault reason or None)`` for one role's sweep record."""
    rows, fault = _suite_rows(rec, suite)
    if fault is None:
        for item in sorted(rows):
            for j, m in enumerate(METRICS):
                if rows[item][j] <= 0.0 and fault is None:
                    fault = f"suite {suite!r} item {item!r} has a non-positive {m!r}"
    if fault is not None:
        return {"digest": digest, "status": "fault", "items": None,
                "crps": None, "mase": None}, fault
    ids = sorted(rows)
    out = {"digest": digest, "status": "ok", "items": len(ids)}
    for j, m in enumerate(METRICS):
        out[m] = _geomean([rows[i][j] for i in ids])
    return out, None


def _error_models(roles: Mapping[str, str]) -> dict[str, dict]:
    return {role: {"digest": roles[role], "status": "error", "items": None,
                   "crps": None, "mase": None} for role in ROLES}


# ── publishing ───────────────────────────────────────────────────────────────


def _replaceable(existing: str) -> bool:
    """An object at the key may be replaced only when it is an ``error``
    record of this schema, or not a readable record at all."""
    try:
        old = json.loads(existing)
    except ValueError:
        return True
    if not isinstance(old, Mapping) or old.get("schema") != SCHEMA:
        return True
    return old.get("status") == "error"


def _put(store: Any, key: str, text: str) -> None:
    """PUBLIC-READ put; a backend without canned ACLs publishes private
    rather than not at all (the bench-report convention)."""
    try:
        store.put_text(key, text, content_type=CONTENT_TYPE, acl=PUBLIC_ACL)
    except StorageError:
        store.put_text(key, text, content_type=CONTENT_TYPE)


def _publish(store: Any, key: str, text: str) -> bool:
    """``True`` when the key holds a published record after the call: this
    text, or a non-error record already there (never replaced). ``False``
    when the store could not be read or written."""
    try:
        existing = store.get_text(key)
    except ObjectNotFound:
        existing = None
    except StorageError:
        return False
    if existing is not None and (existing == text or not _replaceable(existing)):
        return True
    try:
        _put(store, key, text)
    except StorageError:
        return False
    return True


# ── the step ─────────────────────────────────────────────────────────────────


def sweep_and_publish(report: Mapping | None, *, era_block: int, era_block_hash: str,
                      sentinel_suites: Sequence[str], gate_suite: str, bench: Any,
                      store: Any) -> dict | None:
    """Sweep the era's sentinel for a new lane D king and publish the record.

    Returns ``None`` (no host call, no store write) when ``report`` is
    ``None`` or crowned nobody; otherwise ``{"key", "record", "text",
    "published"}``. Raises ``ValueError`` / ``TypeError`` before any host
    call or store write on a malformed ``era_block_hash``, an empty or
    malformed ``sentinel_suites``, a drawn sentinel equal to ``gate_suite``
    or a malformed report. Never raises on a host error or a store failure.
    """
    if report is None:
        return None
    if not isinstance(report, Mapping):
        raise TypeError(f"report must be a mapping or None; got {type(report).__name__}")
    if report.get("crowned") is None:
        return None
    block = _check_era_block(era_block)
    names = _check_suites(sentinel_suites)
    suite = sentinel_for_era(era_block_hash, names)
    if not isinstance(gate_suite, str) or not gate_suite:
        raise TypeError(f"gate_suite must be a non-empty str; got {gate_suite!r}")
    if suite == gate_suite:
        raise ValueError(f"the era's sentinel {suite!r} equals the gate suite: the "
                         "sentinel is held out of the gate and never gates")
    roles = _roles(report)
    session_id = bench.session_id
    if not isinstance(session_id, str) or not session_id:
        raise TypeError(f"bench.session_id must be a non-empty str; got {session_id!r}")

    status, reason = "ok", None
    try:
        recs = bench.paired(roles["king"], roles["previous_king"], roles["house"],
                            suites=(suite,))
        by_role = dict(zip(ROLES, recs, strict=True))
    except Exception as e:  # the pod is unreachable or reset: recorded, never raised
        status, reason = "error", f"sweep_failed: {type(e).__name__}: {e}"
        models = _error_models(roles)
    else:
        models, faults = {}, []
        for role in ROLES:
            models[role], fault = _model(roles[role], by_role[role], suite)
            if fault is not None:
                faults.append(f"{role}: {fault}")
        if faults:
            status, reason = "fault", "; ".join(faults)

    record = {"schema": SCHEMA, "boundary": report.get("boundary"),
              "era": {"block": block, "block_hash": era_block_hash},
              "suite": suite, "sentinel_suites": list(names), "gate_suite": gate_suite,
              "king_digest": roles["king"], "session": session_id,
              "status": status, "reason": reason, "gating": False, "models": models}
    text = sentinel_text(record)
    key = sentinel_key(block, roles["king"])
    return {"key": key, "record": record, "text": text,
            "published": _publish(store, key, text)}


__all__ = ["CONTENT_TYPE", "KEY_PREFIX", "ROLES", "SCHEMA", "STATUSES", "sentinel_bytes",
           "sentinel_for_era", "sentinel_key", "sentinel_text", "sweep_and_publish"]
