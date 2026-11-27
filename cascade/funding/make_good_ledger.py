"""The persistent make-good debt ledger: what the operator OWES miners, and
what was paid back (owner 2026-11-27).

A pod on a MINER's key that our side rejected right after the rent (short
pod, stale image, slow host, NCCL) or that never came up (a lemon) is booked
``make_good`` in :mod:`cascade.funding.miner_spend` — the miner paid, the
operator owes it back. Until now that debt lived only in the trainer's
in-memory ``miner_make_good_owed`` alert (dropped after 6 h unrefreshed, lost
on every restart) and in a per-submission spend record a re-fund replaces.

This file (``<work root>/make_good_ledger.json``) keeps every owed pod rent
once (keyed by its rent id, so re-recording is idempotent) and every payment
the owner records (``cascade-trainer make-good-paid``). Outstanding = owed - paid, per miner and in total;
the trainer publishes the totals in ``status/v2.json`` (``make_good``).

Hotkeys, refs, pod names and dollars only — never key material. flock'd and
written atomically (temp file + rename + fsync). An unreadable file raises
:class:`MakeGoodUnknown` and is never overwritten.
"""

from __future__ import annotations

import fcntl
import json
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

from ..shared.atomic import write_text_atomic

MAKE_GOOD_LEDGER_FILE = "make_good_ledger.json"
SCHEMA = 1


class MakeGoodUnknown(RuntimeError):
    """The ledger exists but cannot be read: nothing is written over it."""


def _cents(x: float) -> float:
    return round(float(x), 2)


class MakeGoodLedger:
    """``{"schema": 1, "owed": {rent_id: row}, "paid": [payment, ...]}``."""

    def __init__(self, path: Path | str, *, clock: Callable[[], float] = time.time) -> None:
        self.path = Path(path)
        self.clock = clock

    @classmethod
    def for_work_root(cls, work_root: Path | str, **kw) -> MakeGoodLedger:
        return cls(Path(work_root) / MAKE_GOOD_LEDGER_FILE, **kw)

    @contextmanager
    def _locked(self) -> Iterator[dict]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path.with_suffix(self.path.suffix + ".lock"), "w") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            try:
                yield self._load()
            finally:
                fcntl.flock(lf, fcntl.LOCK_UN)

    def _load(self) -> dict:
        try:
            doc = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {"schema": SCHEMA, "owed": {}, "paid": []}
        except (OSError, ValueError) as e:
            raise MakeGoodUnknown(f"make-good ledger unreadable: {e}") from e
        if (not isinstance(doc, dict) or not isinstance(doc.get("owed"), dict)
                or not isinstance(doc.get("paid"), list)):
            raise MakeGoodUnknown("make-good ledger malformed")
        return doc

    def _save(self, doc: dict) -> None:
        doc["schema"] = SCHEMA
        write_text_atomic(self.path, json.dumps(doc, indent=1, sort_keys=True))

    # ── writes ───────────────────────────────────────────────────────────────

    def record_owed(self, hotkey: str, *, rent_id: str, usd: float, pod_id: str = "",
                    ref: str = "", funded_at: float | None = None, reason: str = "") -> bool:
        """One pod rent on ``hotkey``'s key the operator owes back (``usd``).
        Idempotent per ``rent_id``: a re-record updates the amount (a landed
        price replacing an estimate), never adds a second row. A $0 row (a
        create the platform refused) is not recorded. True when the ledger
        changed."""
        usd = _cents(usd)
        if not hotkey or not rent_id or usd <= 0:
            return False
        with self._locked() as doc:
            row = doc["owed"].get(rent_id)
            if row is not None and row.get("usd") == usd:
                return False
            if row is None:
                row = {"hotkey": hotkey, "pod_id": pod_id, "ref": ref, "funded_at": funded_at,
                       "reason": str(reason)[:200], "at": self.clock()}
            row["usd"] = usd
            doc["owed"][rent_id] = row
            self._save(doc)
            return True

    def mark_paid(self, hotkey: str, usd: float | None = None, *, note: str,
                  by: str = "owner") -> float:
        """The owner paid ``hotkey`` back ``usd`` (None = everything
        outstanding). Recorded as one payment; returns the amount recorded
        (0.0 when nothing was outstanding and no amount was given)."""
        if not hotkey:
            raise ValueError("a hotkey is required")
        if usd is not None and (not isinstance(usd, (int, float)) or usd <= 0):
            raise ValueError("the paid amount must be > 0")
        with self._locked() as doc:
            amount = (_cents(usd) if usd is not None
                      else self._miner(doc, hotkey)["outstanding_usd"])
            if amount <= 0:
                return 0.0
            doc["paid"].append({"hotkey": hotkey, "usd": amount, "note": str(note)[:300],
                                "by": str(by)[:64], "at": self.clock()})
            self._save(doc)
            return amount

    # ── reads ────────────────────────────────────────────────────────────────

    @staticmethod
    def _miner(doc: dict, hotkey: str) -> dict:
        owed = sorted((r for r in doc["owed"].values() if r.get("hotkey") == hotkey),
                      key=lambda r: float(r.get("at") or 0.0))
        paid = [p for p in doc["paid"] if p.get("hotkey") == hotkey]
        o = _cents(sum(float(r.get("usd") or 0.0) for r in owed))
        p = _cents(sum(float(x.get("usd") or 0.0) for x in paid))
        return {"hotkey": hotkey, "owed_usd": o, "paid_usd": p,
                "outstanding_usd": _cents(max(0.0, o - p)), "pods": len(owed),
                "last_owed_at": owed[-1].get("at") if owed else None,
                "last_paid_at": max((float(x.get("at") or 0.0) for x in paid), default=None)}

    def view(self) -> dict:
        """``{owed_usd, paid_usd, outstanding_usd, miners: [...]}`` — totals
        and one row per miner (outstanding first). Raises
        :class:`MakeGoodUnknown` on an unreadable file."""
        with self._locked() as doc:
            hotkeys = sorted({r.get("hotkey") for r in doc["owed"].values()}
                             | {p.get("hotkey") for p in doc["paid"]} - {None, ""})
            miners = [self._miner(doc, hk) for hk in hotkeys]
        miners.sort(key=lambda m: (-m["outstanding_usd"], m["hotkey"]))
        return {"schema": SCHEMA,
                "owed_usd": _cents(sum(m["owed_usd"] for m in miners)),
                "paid_usd": _cents(sum(m["paid_usd"] for m in miners)),
                "outstanding_usd": _cents(sum(m["outstanding_usd"] for m in miners)),
                "miners": miners}

    def rows(self, hotkey: str | None = None) -> tuple[list[dict], list[dict]]:
        """``(owed rows, payments)`` (optionally one miner's), oldest first."""
        with self._locked() as doc:
            owed = [{"rent_id": k, **r} for k, r in doc["owed"].items()
                    if hotkey is None or r.get("hotkey") == hotkey]
            paid = [dict(p) for p in doc["paid"] if hotkey is None or p.get("hotkey") == hotkey]
        owed.sort(key=lambda r: float(r.get("at") or 0.0))
        return owed, paid


__all__ = ["MAKE_GOOD_LEDGER_FILE", "MakeGoodLedger", "MakeGoodUnknown"]
