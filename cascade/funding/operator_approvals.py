"""One-shot owner approvals for legs billed to the OPERATOR (2026-11-02).

Owner rule: "in future we do not train miner submissions [on our account]
without my permission" — and the owner verifies EACH such leg personally.
So every path that would bill a miner leg to the operator (an expired payer
key, ``funded_operator_fallback``, operator lanes, ``ops_paid_fallback``, a
make-good after our own fault) first asks this store:

* no decision → a PENDING request is filed (hotkey, uid, ref, SKU, projected
  cost, why) and the leg parks in the queue as ``awaiting_operator_approval``
  (unburned, never re-dispatched, no hot loop);
* ``cascade-trainer approve-operator-leg <hotkey>`` turns the pending request
  into a single-use APPROVAL (optionally with a cost ceiling) and returns the
  entry to the queue; the next operator-billed dispatch of that hotkey
  CONSUMES it — exactly one dispatch. A retry after a later failure needs a
  NEW approval (the operator-fault cap still applies on top);
* ``cascade-trainer deny-operator-leg <hotkey>`` records a single-use DENIAL:
  the entry returns to the queue on the miner's own key, or — with no usable
  key — fails terminally, unburned ("re-fund needed").

Records hold hotkeys, refs and numbers only — never key material. The file
lives under the trainer's work root and is flock-serialised, reload-before-
act, exactly like :class:`cascade.funding.queue.FundedQueue` (the trainer and
the owner's CLI both write it).
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

PENDING, APPROVED, DENIED, CONSUMED = "pending", "approved", "denied", "consumed"


def _key(queue_name: str, hotkey: str) -> str:
    return f"{queue_name}:{hotkey}"


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

    def peek_approved(self, queue_name: str, hotkey: str, ref: str) -> bool:
        """An unconsumed approval for this hotkey's CURRENT ref (no state change)."""
        r = self.get(queue_name, hotkey)
        return bool(r and r.get("state") == APPROVED and r.get("ref") == ref)

    # ── trainer side ─────────────────────────────────────────────────────────

    def request(self, queue_name: str, hotkey: str, *, ref: str, attempt_id: str,
                queue_path: str = "", uid: int | None = None, sku: str = "",
                rate_per_hour: float = 0.0, expected_hours: float = 0.0,
                projected_cost_usd: float = 0.0, kind: str = "", reason: str = "") -> bool:
        """File (or refresh) a PENDING request; True only when it is NEW —
        the caller alerts once per request, never per poll."""
        with self._locked() as recs:
            k = _key(queue_name, hotkey)
            prev = recs.get(k) or {}
            now = self.clock()
            fresh = not (prev.get("state") == PENDING and prev.get("ref") == ref)
            recs[k] = {
                "queue": queue_name, "queue_path": str(queue_path), "hotkey": hotkey,
                "ref": ref, "uid": uid, "sku": sku, "kind": kind, "reason": reason[:400],
                "rate_per_hour": round(float(rate_per_hour), 4),
                "expected_hours": round(float(expected_hours), 3),
                "projected_cost_usd": round(float(projected_cost_usd), 2),
                "state": PENDING,
                "request_attempt_id": attempt_id if fresh else prev.get("request_attempt_id", ""),
                "requested_at": now if fresh else prev.get("requested_at", now),
                "last_seen_at": now,
            }
            self._save(recs)
            return fresh

    def consume(self, queue_name: str, hotkey: str, *, ref: str, attempt_id: str,
                projected_cost_usd: float) -> str:
        """Use the owner's decision for ONE dispatch: ``"approved"`` /
        ``"denied"`` (both single-use: the record turns ``consumed``), else
        ``"none"`` (no decision for this ref) or ``"over_ceiling"`` (the
        projected cost exceeds the approved ceiling — a new approval is
        needed; the record turns pending again)."""
        with self._locked() as recs:
            k = _key(queue_name, hotkey)
            r = recs.get(k)
            if not r or r.get("ref") != ref or r.get("state") not in (APPROVED, DENIED):
                return "none"
            now = self.clock()
            if r["state"] == APPROVED:
                ceiling = r.get("max_cost_usd")
                if ceiling is not None and float(projected_cost_usd) > float(ceiling):
                    recs[k] = {**r, "state": PENDING, "projected_cost_usd":
                               round(float(projected_cost_usd), 2), "requested_at": now,
                               "request_attempt_id": attempt_id,
                               "reason": (f"projected ${projected_cost_usd:.2f} exceeds the "
                                          f"approved ceiling ${float(ceiling):.2f}")}
                    self._save(recs)
                    return "over_ceiling"
            verdict = "approved" if r["state"] == APPROVED else "denied"
            recs[k] = {**r, "state": CONSUMED, "decision": verdict, "consumed_at": now,
                       "consumed_attempt_id": attempt_id}
            self._save(recs)
            return verdict

    # ── owner side ───────────────────────────────────────────────────────────

    def decide(self, hotkey: str, *, approve: bool, attempt_id: str | None = None,
               max_cost_usd: float | None = None, queue_name: str | None = None) -> dict:
        """Approve / deny the PENDING request of ``hotkey`` (one per queue; pass
        ``queue_name`` when several queues have one). ``attempt_id``, when
        given, must match the request's. Raises ``LookupError`` / ``ValueError``
        with an owner-readable reason; returns the updated record."""
        with self._locked() as recs:
            matches = [(k, r) for k, r in recs.items()
                       if r.get("hotkey") == hotkey and r.get("state") == PENDING
                       and (queue_name is None or r.get("queue") == queue_name)]
            if not matches:
                raise LookupError(f"no pending operator-billed leg for {hotkey}")
            if len(matches) > 1:
                raise LookupError(f"{hotkey} has pending legs in several queues "
                                  f"({', '.join(r['queue'] for _, r in matches)}); pass --queue")
            k, r = matches[0]
            if attempt_id and attempt_id != r.get("request_attempt_id"):
                raise ValueError(f"attempt {attempt_id} is not the pending request "
                                 f"({r.get('request_attempt_id')})")
            if max_cost_usd is not None and float(max_cost_usd) <= 0:
                raise ValueError("--max-cost must be positive")
            rec = {**r, "state": APPROVED if approve else DENIED, "decided_at": self.clock(),
                   "max_cost_usd": (float(max_cost_usd) if approve and max_cost_usd is not None
                                    else None)}
            recs[k] = rec
            self._save(recs)
            return dict(rec)
