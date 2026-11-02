"""Per-leg owner approvals for legs billed to the OPERATOR (2026-11-02).

Owner rule: "in future we do not train miner submissions [on our account]
without my permission" — and the owner verifies EACH such leg personally.
Every path that would bill a miner leg to the operator (an expired payer key,
``funded_operator_fallback``, operator lanes, ``ops_paid_fallback``, a
make-good after our own fault) — and a RETRAIN after a finished run was lost
on our side, even on the miner's key (the miner already paid once) — asks
this store first.

Lifecycle of one record (one per queue + hotkey):

* ``pending`` — filed by the trainer (hotkey, uid, ref, kind, SKU, projected
  cost, why); the entry parks ``awaiting_operator_approval``.
* ``approved`` — ``cascade-trainer approve-operator-leg <hotkey>`` (also
  WITHOUT a pending request: a pre-approval for the hotkey's next leg). Bound
  to the entry's ref and funding (``funded_at``) and to the request's kind
  (a pre-approval may name ``--kind``; "" = any kind).
* ``launched`` — set by the trainer only once the pod is actually rented (or
  the operator lane dispatched), with that attempt's id. A no-capacity /
  lemon / rent error before that keeps the approval for the next attempt.
  While launched, the SAME attempt (a restart re-attach, a re-dispatch of the
  leg) passes again; a DIFFERENT attempt (a retry after the leg failed)
  needs a NEW approval.
* ``denied`` — single use: the next gate answers "denied" once.

A re-fund changes ``funded_at``; any record bound to the old funding no
longer matches (cleared on use). Records hold hotkeys, refs and numbers only
— never key material. Flock-serialised, reload-before-act, exactly like
:class:`cascade.funding.queue.FundedQueue` (the trainer and the owner's CLI
both write it).
"""

from __future__ import annotations

import fcntl
import json
import os
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

APPROVALS_FILE = "operator_approvals.json"

PENDING, APPROVED, LAUNCHED, DENIED, CONSUMED = (
    "pending", "approved", "launched", "denied", "consumed")


def _key(queue_name: str, hotkey: str) -> str:
    return f"{queue_name}:{hotkey}"


def _bound(r: dict, *, ref: str, funded_at: float | None) -> bool:
    if r.get("ref") != ref:
        return False
    fa = r.get("funded_at")
    return fa is None or funded_at is None or float(fa) == float(funded_at)


def _kind_ok(r: dict, kind: str) -> bool:
    return not r.get("kind") or not kind or r.get("kind") == kind


class OperatorApprovals:
    """``<work_root>/operator_approvals.json``: one record per (queue, hotkey)."""

    def __init__(self, path: Path | str, *, clock: Callable[[], float] = time.time) -> None:
        self.path = Path(path)
        self.clock = clock

    @classmethod
    def for_work_root(cls, work_root: Path | str, **kw) -> OperatorApprovals:
        return cls(Path(work_root) / APPROVALS_FILE, **kw)

    @contextmanager
    def _locked(self, *, exclusive: bool = True) -> Iterator[dict]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        with open(lock_path, "w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
            try:
                yield self._load()
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)

    def _load(self) -> dict:
        if not self.path.is_file():
            return {}
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        recs = raw.get("records", {})
        return dict(recs) if isinstance(recs, dict) else {}

    def _save(self, records: dict) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps({"records": records}, indent=2, sort_keys=True),
                       encoding="utf-8")
        os.replace(tmp, self.path)

    # ── reads ────────────────────────────────────────────────────────────────

    def records(self) -> list[dict]:
        with self._locked(exclusive=False) as recs:
            return [dict(r) for _, r in sorted(recs.items())]

    def pending(self) -> list[dict]:
        return [r for r in self.records() if r.get("state") == PENDING]

    def get(self, queue_name: str, hotkey: str) -> dict | None:
        with self._locked(exclusive=False) as recs:
            r = recs.get(_key(queue_name, hotkey))
            return dict(r) if r else None

    def check(self, queue_name: str, hotkey: str, *, ref: str, attempt_id: str, kind: str,
              funded_at: float | None = None, projected_cost_usd: float = 0.0) -> str:
        """The owner's verdict for this leg, WITHOUT using it up:
        ``"approved"`` (an approval for this ref/funding/kind, or this very
        attempt already launched under one), ``"over_ceiling"``, ``"denied"``
        (the trainer then calls :meth:`take_denial`) or ``"none"``."""
        r = self.get(queue_name, hotkey)
        if not r or not _bound(r, ref=ref, funded_at=funded_at):
            return "none"
        state = r.get("state")
        if state == LAUNCHED:
            return "approved" if r.get("launched_attempt_id") == attempt_id else "none"
        if state == DENIED:
            return "denied"
        if state != APPROVED or not _kind_ok(r, kind):
            return "none"
        ceiling = r.get("max_cost_usd")
        if ceiling is not None and float(projected_cost_usd) > float(ceiling):
            return "over_ceiling"
        return "approved"

    def peek_approved(self, queue_name: str, hotkey: str, ref: str, *,
                      funded_at: float | None = None, kind: str = "") -> bool:
        """An unlaunched approval for this hotkey's current leg (no state change)."""
        r = self.get(queue_name, hotkey)
        return bool(r and r.get("state") == APPROVED and _bound(r, ref=ref, funded_at=funded_at)
                    and _kind_ok(r, kind))

    # ── trainer side ─────────────────────────────────────────────────────────

    def request(self, queue_name: str, hotkey: str, *, ref: str, attempt_id: str,
                queue_path: str = "", uid: int | None = None, sku: str = "",
                rate_per_hour: float = 0.0, expected_hours: float = 0.0,
                projected_cost_usd: float = 0.0, kind: str = "", reason: str = "",
                funded_at: float | None = None, miner_lost_usd: float = 0.0) -> bool:
        """File (or refresh) a PENDING request; True only when it is NEW —
        the caller alerts once per request, never per poll. Never overwrites
        an owner decision that still applies to this leg."""
        with self._locked() as recs:
            k = _key(queue_name, hotkey)
            prev = recs.get(k) or {}
            now = self.clock()
            if prev.get("state") in (APPROVED, DENIED) and _bound(prev, ref=ref,
                                                                  funded_at=funded_at):
                return False
            fresh = not (prev.get("state") == PENDING and _bound(prev, ref=ref,
                                                                 funded_at=funded_at)
                         and prev.get("kind") == kind)
            recs[k] = {
                "queue": queue_name, "queue_path": str(queue_path), "hotkey": hotkey,
                "ref": ref, "funded_at": funded_at, "uid": uid, "sku": sku, "kind": kind,
                "reason": reason[:400],
                "rate_per_hour": round(float(rate_per_hour), 4),
                "expected_hours": round(float(expected_hours), 3),
                "projected_cost_usd": round(float(projected_cost_usd), 2),
                "miner_lost_usd": round(float(miner_lost_usd), 2),
                "state": PENDING,
                "request_attempt_id": attempt_id if fresh else prev.get("request_attempt_id", ""),
                "requested_at": now if fresh else prev.get("requested_at", now),
                "last_seen_at": now,
            }
            self._save(recs)
            return fresh

    def mark_launched(self, queue_name: str, hotkey: str, *, ref: str, attempt_id: str,
                      cost_usd: float = 0.0) -> bool:
        """The pod was rented (or the lane dispatched) under an approval: from
        now on only THIS attempt passes; a later retry needs a new approval."""
        with self._locked() as recs:
            k = _key(queue_name, hotkey)
            r = recs.get(k)
            if not r or r.get("ref") != ref or r.get("state") not in (APPROVED, LAUNCHED):
                return False
            recs[k] = {**r, "state": LAUNCHED, "launched_attempt_id": attempt_id,
                       "launched_at": self.clock(), "launched_cost_usd": round(cost_usd, 2)}
            self._save(recs)
            return True

    def take_denial(self, queue_name: str, hotkey: str) -> None:
        """A denial is single use."""
        with self._locked() as recs:
            k = _key(queue_name, hotkey)
            r = recs.get(k)
            if r and r.get("state") == DENIED:
                recs[k] = {**r, "state": CONSUMED, "decision": "denied",
                           "consumed_at": self.clock()}
                self._save(recs)

    def clear(self, queue_name: str, hotkey: str) -> None:
        with self._locked() as recs:
            if recs.pop(_key(queue_name, hotkey), None) is not None:
                self._save(recs)

    # ── owner side ───────────────────────────────────────────────────────────

    def decide(self, hotkey: str, *, approve: bool, attempt_id: str | None = None,
               max_cost_usd: float | None = None, queue_name: str | None = None,
               preapprove: dict | None = None) -> dict:
        """Approve / deny ``hotkey``'s PENDING request (one per queue; pass
        ``queue_name`` when several queues have one). With no pending request
        and ``preapprove`` given (``{"queue", "queue_path", "ref",
        "funded_at", "kind"}`` of the hotkey's live entry) an approval is
        recorded for its NEXT operator-billed leg. ``attempt_id``, when given,
        must match the pending request's. Raises ``LookupError`` /
        ``ValueError`` with an owner-readable reason; returns the record."""
        if max_cost_usd is not None and float(max_cost_usd) <= 0:
            raise ValueError("--max-cost must be positive")
        with self._locked() as recs:
            matches = [(k, r) for k, r in recs.items()
                       if r.get("hotkey") == hotkey and r.get("state") == PENDING
                       and (queue_name is None or r.get("queue") == queue_name)]
            if len(matches) > 1:
                raise LookupError(f"{hotkey} has pending legs in several queues "
                                  f"({', '.join(r['queue'] for _, r in matches)}); pass --queue")
            if matches:
                k, r = matches[0]
                if attempt_id and attempt_id != r.get("request_attempt_id"):
                    raise ValueError(f"attempt {attempt_id} is not the pending request "
                                     f"({r.get('request_attempt_id')})")
            elif preapprove is not None and approve:
                k = _key(preapprove["queue"], hotkey)
                r = {"queue": preapprove["queue"], "queue_path": preapprove.get("queue_path", ""),
                     "hotkey": hotkey, "ref": preapprove["ref"],
                     "funded_at": preapprove.get("funded_at"),
                     "kind": preapprove.get("kind", ""), "reason": "pre-approved by the owner",
                     "request_attempt_id": "", "requested_at": self.clock()}
            else:
                raise LookupError(f"no pending operator-billed leg for {hotkey}")
            rec = {**r, "state": APPROVED if approve else DENIED, "decided_at": self.clock(),
                   "max_cost_usd": (float(max_cost_usd) if approve and max_cost_usd is not None
                                    else None)}
            recs[k] = rec
            self._save(recs)
            return dict(rec)
