"""Miner spend tracking and the waste cap (owner, 2026-11-03).

Every pod rented on a MINER's payer key is recorded per hotkey, bound to the
submission it served (ref + ``funded_at``; a re-fund starts a fresh count):
its landed $/h, start, end and outcome — it produced a harvested, verified
checkpoint (``result``) or it did not (``waste``: a lemon, a lost or
unreachable host, a crash, an our-side fault, a pod kept during our
re-harvest that never produced). ``miner_waste_usd`` is the cost of the
waste pods; the successful attempt's pod is never waste.

Before every rent on a miner's key the trainer reads this ledger: waste at
or above ``[round] miner_waste_cap_usd`` pauses the entry
(``paused_spend_cap`` — never failed, burned, expired or rejected) and, when
the waste is infra / our side, files a ``spend_cap_make_good`` owner
approval to finish the leg on the operator's account.

Hotkeys, refs and dollars only — never key material. flock'd,
reload-before-act, like :class:`cascade.funding.queue.FundedQueue`. An
unreadable ledger raises :class:`SpendUnknown`: new miner-key rents fail
CLOSED (the entry pauses with that reason), never open.
"""

from __future__ import annotations

import fcntl
import json
import os
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

MINER_SPEND_FILE = "miner_spend.json"
RESULT, WASTE = "result", "waste"
# A pod on the miner's key that OUR side rejected right after the rent (a
# short pod, a stale image, a slow host, NCCL) or that never came up (a
# lemon): the miner paid, the operator owes it back. Spend, never waste — it
# is excluded from the miner's waste view and the waste cap, and summed as
# ``make_good_usd`` for the owner (money safety II).
MAKE_GOOD = "make_good"
INFRA, MINER = "infra", "miner"
# A pod that vanished mid-leg with NO proof of whose doing (``payer_pod_lost``):
# the miner's money is gone, but it is neither an operator-side infra loss
# (no early make-good filed from it) nor the miner's own fault. The owner
# sees it in the spend view and decides (2026-11-04, 5GhPUmHV).
UNPROVEN = "unproven"


class SpendUnknown(RuntimeError):
    """The spend ledger exists but cannot be read: rents fail closed."""


@dataclass(frozen=True)
class Spend:
    spend_usd: float
    waste_usd: float
    infra_waste_usd: float
    pods: tuple[dict, ...]
    capped: bool
    make_good_usd: float = 0.0
    make_good_requested: bool = False


def _cost(p: dict, now: float) -> float:
    end = p.get("end") or now
    return max(0.0, float(end) - float(p.get("start") or end)) / 3600.0 * float(p.get("rate") or 0)


class MinerSpend:
    """``<dir>/miner_spend.json``: one record per hotkey."""

    def __init__(self, path: Path | str, *, clock: Callable[[], float] = time.time) -> None:
        self.path = Path(path)
        self.clock = clock

    @classmethod
    def beside(cls, queue_path: Path | str, **kw) -> MinerSpend:
        """The ledger beside the funding queues (the trainer's work root)."""
        return cls(Path(queue_path).with_name(MINER_SPEND_FILE), **kw)

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
            return {}
        except (OSError, ValueError) as e:
            raise SpendUnknown(f"miner spend ledger unreadable: {e}") from e
        recs = doc.get("records") if isinstance(doc, dict) else None
        if not isinstance(recs, dict):
            raise SpendUnknown("miner spend ledger malformed")
        return recs

    def _save(self, recs: dict) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps({"records": recs}, indent=1, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self.path)

    @staticmethod
    def _bound(r: dict | None, ref: str, funded_at: float | None) -> bool:
        return bool(r) and r.get("ref") == ref and (
            funded_at is None or r.get("funded_at") is None
            or float(r["funded_at"]) == float(funded_at))

    # Records are keyed by (hotkey, entry ref): lane A and lane B / C entries
    # of one hotkey never share (or erase) each other's history (review S4).

    @staticmethod
    def _key(hotkey: str, ref: str) -> str:
        return f"{hotkey}::{ref}"

    @staticmethod
    def _of(recs: dict, hotkey: str | None):
        for k, r in recs.items():
            if hotkey is None or not hotkey or k.split("::", 1)[0] == hotkey:
                yield k, r

    # ── the rent / teardown sites ────────────────────────────────────────────

    def start(self, hotkey: str, *, ref: str, funded_at: float | None, pod_id: str,
              rate: float, new_rent: bool = False, rent_id: str = "") -> None:
        """A pod was allocated on the miner's key.

        Pod NAMES are reused across rents (``…-funded-<slug>-0``), so a record
        is one RENT, never one name (2026-11-03 5EvqPCMN: a 2-minute short-pod
        reject inherited the start of the lost pod before it and booked 42
        min). ``new_rent`` (a fresh create): any record still open under this
        name is closed at its LAST-SEEN-ALIVE time as infra waste (a host
        loss) and a NEW record (``rent_id``) opens now. Otherwise the call is
        an update of the open rent (a known landed rate replaces an estimate)."""
        with self._locked() as recs:
            k = self._key(hotkey, ref)
            r = recs.get(k)
            if not self._bound(r, ref, funded_at):
                r = {"hotkey": hotkey, "ref": ref, "funded_at": funded_at, "pods": [],
                     "capped": False}
            now = self.clock()
            open_ = [p for p in r["pods"] if p["pod_id"] == pod_id and p.get("end") is None]
            if new_rent:
                for p in open_:
                    p.update(end=p.get("seen") or p.get("start"), outcome=WASTE, kind=INFRA,
                             reason="lost before the next rent of this pod name (closed at "
                                    "its last-seen-alive time)")
                open_ = []
            if open_:
                p = open_[-1]
                if rate:
                    p["rate"] = round(float(rate), 4)
                p["seen"] = now
            else:
                r["pods"].append({"pod_id": pod_id, "rent_id": rent_id or f"{pod_id}@{now:.3f}",
                                  "rate": round(float(rate or 0), 4),
                                  "start": now, "seen": now, "end": None, "outcome": None})
            recs[k] = r
            self._save(recs)

    def end(self, hotkey: str | None, pod_id: str, *, outcome: str, reason: str = "",
            kind: str = INFRA, at_last_seen: bool = False) -> None:
        """Close the open record of ``pod_id`` (the CURRENT rent of that name;
        any of the hotkey's submissions; ``hotkey`` None/"" = any hotkey) now,
        or at its last-seen-alive time (``at_last_seen``: a pod found lost).
        No-op when none is open."""
        with self._locked() as recs:
            changed = False
            for _k, r in self._of(recs, hotkey):
                for p in r["pods"]:
                    if p["pod_id"] == pod_id and p.get("end") is None:
                        end = (p.get("seen") or p.get("start")) if at_last_seen else self.clock()
                        p.update(end=end, outcome=outcome, reason=reason[:200], kind=kind)
                        changed = True
            if changed:
                self._save(recs)

    def touch(self, hotkey: str, pod_id: str) -> None:
        """The pod was seen alive now (the cap for a later vanish)."""
        with self._locked() as recs:
            changed = False
            for _k, r in self._of(recs, hotkey):
                for p in r["pods"]:
                    if p["pod_id"] == pod_id and p.get("end") is None:
                        p["seen"] = self.clock()
                        changed = True
            if changed:
                self._save(recs)

    def close_missing(self, alive_ids) -> int:
        """Close open records whose pod no longer exists (not on the funded
        ledger): ended at its LAST-SEEN-ALIVE time, never ``now`` — a vanished
        pod's cost never grows (review S1). Returns how many were closed."""
        alive = set(alive_ids)
        with self._locked() as recs:
            n = 0
            for _k, r in recs.items():
                for p in r["pods"]:
                    if p.get("end") is None and p["pod_id"] not in alive:
                        p.update(end=p.get("seen") or p.get("start"), outcome=WASTE,
                                 reason="pod gone without a result", kind=INFRA)
                        n += 1
            if n:
                self._save(recs)
            return n

    def mark_make_good_requested(self, hotkey: str, ref: str) -> bool:
        """Mark the early make-good request as filed; False if it already was."""
        with self._locked() as recs:
            r = recs.get(self._key(hotkey, ref))
            if r is None or r.get("make_good_requested"):
                return False
            r["make_good_requested"] = True
            self._save(recs)
            return True

    def mark_capped(self, hotkey: str, ref: str, capped: bool = True) -> None:
        with self._locked() as recs:
            k = self._key(hotkey, ref)
            if k in recs:
                recs[k]["capped"] = bool(capped)
                self._save(recs)

    def reset(self, hotkey: str) -> None:
        """The owner's resume (``--reset-spend``): a fresh count."""
        with self._locked() as recs:
            gone = [k for k, _r in self._of(recs, hotkey)]
            for k in gone:
                recs.pop(k, None)
            if gone:
                self._save(recs)

    # ── reads ────────────────────────────────────────────────────────────────

    def summary(self, hotkey: str, *, ref: str, funded_at: float | None) -> Spend:
        with self._locked() as recs:
            r = recs.get(self._key(hotkey, ref))
        if not self._bound(r, ref, funded_at):
            return Spend(0.0, 0.0, 0.0, (), False)
        now = self.clock()
        pods = tuple(dict(p) for p in r["pods"])
        spend = sum(_cost(p, now) for p in pods)
        waste = [p for p in pods if p.get("outcome") == WASTE]
        owed = [p for p in pods if p.get("outcome") == MAKE_GOOD]
        return Spend(round(spend, 2), round(sum(_cost(p, now) for p in waste), 2),
                     round(sum(_cost(p, now) for p in waste if p.get("kind", INFRA) == INFRA), 2),
                     pods, bool(r.get("capped")),
                     make_good_usd=round(sum(_cost(p, now) for p in owed), 2),
                     make_good_requested=bool(r.get("make_good_requested")))

    def public(self) -> dict[tuple[str, str], dict]:
        """``{(hotkey, ref): {miner_spend_usd, miner_waste_usd, funded_at}}``."""
        try:
            with self._locked() as recs:
                rows = [(r.get("hotkey") or k.split("::", 1)[0], r.get("ref"), r.get("funded_at"))
                        for k, r in recs.items()]
        except SpendUnknown:
            return {}
        out = {}
        for hk, ref, fa in rows:
            s = self.summary(hk, ref=ref, funded_at=fa)
            out[(hk, ref)] = {"miner_spend_usd": round(s.spend_usd),
                              "miner_waste_usd": round(s.waste_usd), "funded_at": fa}
        return out
