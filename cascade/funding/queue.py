"""The funded-challenger queue: who has paid for a lane, in what order.

The queue is the demand signal for elastic rounds (DEC-CA-0036): entries are
drained into round fields ``cap`` at a time, ordered by **reveal block** —
earliest commit first, the one seniority claim the repo trusts (a UID is not
one; see NOTE-ca-operational-invariants). Queue depth divided by the cap is
what sizes the day's round cadence (:func:`rounds_needed`).

Entries reference hotkeys and generator refs only — **never API keys** (those
live in :class:`cascade.funding.vault.PayerKeyVault`). The file is therefore
safe to publish as the transparency feed miners watch, and safe to read from
the trainer, the provisioner, and the intake service alike.

TWO processes write this file — the long-lived ``cascade-intake`` service and
the trainer (plus the intake's own handler threads) — so every operation is
**reload-before-act under an exclusive ``flock``** on a sibling ``.lock``
file: a mutation loads the current file, applies one change, and writes back
atomically while holding the lock; a read reloads first. An in-memory
snapshot is never trusted across operations — the first cut of this class
cached state at construction, and a stale intake instance would have
resurrected entries the trainer had already settled (audit 2026-08-29).

One live entry per hotkey (the PRISM 1-max rule): funding again while queued
replaces the entry's ref; a terminal entry (``done``/``failed``/``withdrawn``)
frees the slot for a fresh fund. Queued entries that outlive the payer
vault's TTL expire terminally (:meth:`expire_stale`) — their key is gone, so
they could only squat in the depth count and hold boundaries open. Nothing
here burns a submission — burn semantics stay with the trainer's machinery.
"""

from __future__ import annotations

import fcntl
import json
import math
import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from .vault import DEFAULT_TTL_SECONDS

__all__ = [
    "DEFAULT_MAX_ATTEMPTS",
    "FundedEntry",
    "FundedQueue",
    "rounds_needed",
    "select_field",
]

# Bounded auto-retry for infra faults, the PRISM default. Sold-out/rate-limit
# requeues do NOT count against this (should_recover's no-burn classes).
DEFAULT_MAX_ATTEMPTS = 3

_STATUSES = ("pending_reveal", "queued", "in_round", "done", "failed", "withdrawn")


@dataclass(frozen=True)
class FundedEntry:
    """One funded submission: identity, seniority, and lifecycle state."""

    hotkey: str
    ref: str                       # generator ref (repo@digest), matched to the reveal
    reveal_block: int              # seniority: earliest reveal drains first
    funded_at: float               # unix seconds, from the queue's clock
    status: str = "queued"
    attempts: int = 0              # infra-fault retries consumed (bounded)
    last_error: str = ""
    last_error_class: str = ""     # cascade.funding.faults class, "" = none
    # Refreshed by every touch that proves the entry is alive (fund, requeue,
    # promotion) — expiry keys off THIS, not funded_at, so an entry actively
    # cycling through no_capacity requeues waits as long as the drought lasts
    # ("sold out is not Score(0)", no time bound) while a genuinely abandoned
    # one still dies at the TTL. 0.0 = pre-field files: fall back to funded_at.
    last_active: float = 0.0

    @property
    def active_at(self) -> float:
        return self.last_active or self.funded_at


def select_field(entries: Iterable[FundedEntry], cap: int) -> list[FundedEntry]:
    """The next round's funded field: queued entries, earliest reveal first.

    Ties on reveal block break on hotkey for determinism. ``cap <= 0`` means
    no cap — the trainer's field filter drains this ordering and applies its
    own cap after ref-matching, so both sides share ONE ordering rule.
    """
    queued = sorted(
        (e for e in entries if e.status == "queued"),
        key=lambda e: (e.reveal_block, e.hotkey),
    )
    return queued[:cap] if cap > 0 else queued


def rounds_needed(queue_depth: int, cap: int, *, min_rounds: int = 1, max_rounds: int = 4) -> int:
    """Elastic cadence: rounds required to drain ``queue_depth`` at ``cap`` per round.

    Clamped to ``[min_rounds, max_rounds]`` — the cap per round is a statistics
    constant (DEC-CA-0012's alpha/k was sized for it), so demand raises the
    *number of rounds*, never the per-round k. Overflow beyond
    ``max_rounds × cap`` simply waits for the next day.
    """
    if cap <= 0:
        raise ValueError("cap must be positive")
    needed = math.ceil(max(queue_depth, 0) / cap)
    return max(min_rounds, min(needed, max_rounds))


class FundedQueue:
    """JSON-file queue shared across processes: flock-serialised, reload-first."""

    def __init__(self, path: Path | str, *, clock: Callable[[], float] = time.time,
                 entry_ttl_seconds: float = DEFAULT_TTL_SECONDS) -> None:
        self.path = Path(path)
        self.clock = clock
        # Must track the payer vault's TTL (cascade-intake --ttl-hours /
        # [round] funded_entry_ttl_hours): expiring earlier kills paid entries
        # whose key still works; later leaves keyless entries holding the
        # skip-floor open. The runbook pins the two knobs together.
        self.entry_ttl_seconds = float(entry_ttl_seconds)
        self._entries: dict[str, FundedEntry] = {}
        self._load()

    # ── persistence ──────────────────────────────────────────────────────────

    @contextmanager
    def _locked(self) -> Iterator[None]:
        """Exclusive cross-process (and cross-thread) lock + fresh reload.

        flock on a sibling ``.lock`` file serialises the intake service, its
        handler threads, and the trainer; the reload inside the lock is what
        makes read-modify-write safe — the mutation applies to the CURRENT
        file, never to a snapshot from this instance's past.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        with open(lock_path, "w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                self._load()
                yield
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)

    def _load(self) -> None:
        self._entries = {}
        if not self.path.is_file():
            return
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        for item in raw.get("entries", []):
            entry = FundedEntry(
                hotkey=str(item["hotkey"]),
                ref=str(item["ref"]),
                reveal_block=int(item["reveal_block"]),
                funded_at=float(item["funded_at"]),
                status=str(item.get("status", "queued")),
                attempts=int(item.get("attempts", 0)),
                last_error=str(item.get("last_error", "")),
                last_error_class=str(item.get("last_error_class", "")),
                last_active=float(item.get("last_active", 0.0)),
            )
            self._entries[entry.hotkey] = entry

    def _save(self) -> None:
        import os

        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "entries": [asdict(e) for e in sorted(
                self._entries.values(), key=lambda e: (e.reveal_block, e.hotkey))],
        }
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self.path)

    # ── reads (always against the current file, never a cached snapshot) ─────

    def entries(self) -> list[FundedEntry]:
        self._load()
        return sorted(self._entries.values(), key=lambda e: (e.reveal_block, e.hotkey))

    def get(self, hotkey: str) -> FundedEntry | None:
        self._load()
        return self._entries.get(hotkey)

    def queued_depth(self) -> int:
        self._load()
        return sum(1 for e in self._entries.values() if e.status == "queued")

    # ── lifecycle ────────────────────────────────────────────────────────────

    def add(self, hotkey: str, ref: str, reveal_block: int) -> str:
        """Fund (or re-fund) a submission; returns the outcome for the caller.

        ``"already-queued"`` — same hotkey, same ref, still live: idempotent
        no-op (the PRISM 200 path). ``"replaced"`` — a live entry updated to a
        new ref (a re-reveal before its round). ``"queued"`` — fresh entry,
        including over a terminal one (a resubmit after done/failed frees the
        slot: entry cost is the compute funding, not a lifetime bar).
        """
        with self._locked():
            now = self.clock()
            prev = self._entries.get(hotkey)
            if prev is not None and prev.status in ("pending_reveal", "queued", "in_round"):
                if prev.ref == ref and prev.status != "pending_reveal":
                    # Idempotent, but a re-fund is proof of life — refresh so
                    # an actively-tended entry never TTL-expires under you.
                    self._entries[hotkey] = replace(prev, last_active=now)
                    self._save()
                    return "already-queued"
                if prev.status == "in_round":
                    # Mid-round the manifest may already reference the old ref —
                    # never mutate an entry a live round is training.
                    return "already-queued"
                self._entries[hotkey] = replace(
                    prev, ref=ref, reveal_block=int(reveal_block),
                    funded_at=now, last_active=now, status="queued")
                self._save()
                return "replaced" if prev.ref != ref else "queued"
            self._entries[hotkey] = FundedEntry(
                hotkey=hotkey, ref=ref, reveal_block=int(reveal_block),
                funded_at=now, last_active=now)
            self._save()
            return "queued"

    def add_pending(self, hotkey: str, ref: str) -> str:
        """A submit-with-key entry whose reveal has not landed on chain yet.

        The one-request flow: the ZIP and the Lium key arrive together, before
        the miner's chain commit reveals — so the entry parks as
        ``pending_reveal`` (no reveal block, no seniority, never selected) and
        :meth:`promote_pending` flips it to ``queued`` the moment the reveal
        resolves. Idempotent per (hotkey, ref); a live queued/in_round entry
        is left alone (they already have a better state).
        """
        with self._locked():
            prev = self._entries.get(hotkey)
            if prev is not None and prev.status in ("queued", "in_round"):
                return "already-queued"
            now = self.clock()
            if prev is not None and prev.status == "pending_reveal" and prev.ref == ref:
                self._entries[hotkey] = replace(prev, last_active=now)
                self._save()
                return "already-pending"
            self._entries[hotkey] = FundedEntry(
                hotkey=hotkey, ref=ref, reveal_block=0,
                funded_at=now, last_active=now, status="pending_reveal")
            self._save()
            return "pending_reveal"

    def promote_pending(self, resolve_reveal) -> int:
        """Flip pending entries whose reveal now resolves; count promoted.

        ``resolve_reveal(hotkey, ref) -> int | None`` is the intake's oracle.
        Resolution stamps the REAL reveal block, so seniority is chain truth,
        never upload order. Unresolvable entries stay pending until they
        resolve or :meth:`expire_stale` reaps them at the key TTL.
        """
        with self._locked():
            promoted = 0
            for hk, e in list(self._entries.items()):
                if e.status != "pending_reveal":
                    continue
                block = resolve_reveal(hk, e.ref)
                if block is None:
                    continue
                self._entries[hk] = replace(
                    e, status="queued", reveal_block=int(block),
                    last_active=self.clock())
                promoted += 1
            if promoted:
                self._save()
            return promoted

    def mark_in_round(self, selections: Iterable) -> list[str]:
        """Flip selected entries to ``in_round``; returns the hotkeys CONFIRMED.

        ``selections`` is ``(hotkey, ref)`` pairs (bare hotkeys accepted for
        callers that cannot know the ref). The ref is re-checked INSIDE the
        lock: the trainer selects from a snapshot, and an intake ref-replace
        landing in the select→mark window would otherwise get the NEW entry
        consumed by a round that trained the OLD ref (review 2026-08-29).
        A selection whose entry changed underneath is skipped — the caller
        must drop that challenger from the round.
        """
        with self._locked():
            confirmed: list[str] = []
            for sel in selections:
                hk, ref = sel if isinstance(sel, tuple) else (sel, None)
                e = self._entries.get(hk)
                if e is None or e.status != "queued":
                    continue
                if ref is not None and e.ref != ref:
                    continue
                self._entries[hk] = replace(e, status="in_round")
                confirmed.append(hk)
            if confirmed:
                self._save()
            return confirmed

    def recover_in_round(self) -> int:
        """Return every ``in_round`` entry to ``queued``; count recovered.

        Called by the trainer at round START, before selection: a completed
        round marks its entries ``done``, so anything still ``in_round`` at
        the next round's entry is a torn round (crash/restart mid-round) —
        the entries re-enter the field un-burned, mirroring the trainer's
        burn-after-heat rule ("the field simply re-enters the retried round").
        """
        with self._locked():
            stale = [hk for hk, e in self._entries.items() if e.status == "in_round"]
            for hk in stale:
                self._entries[hk] = replace(self._entries[hk], status="queued")
            if stale:
                self._save()
            return len(stale)

    def expire_stale(self, max_age_seconds: float | None = None) -> int:
        """Terminally expire idle queued entries past the payer-key TTL.

        An entry idle past the vault TTL has no key left to rent with — but
        it still counted toward ``queued_depth``, so one never-enterable fund
        (never revealed, hotkey deregistered) could hold every boundary open
        and bill the operator a king leg per epoch indefinitely (audit
        2026-08-29). "Idle" is measured from ``active_at``: a requeue or
        re-fund is proof of life, so a sold-out entry actively cycling waits
        as long as the drought lasts. Expiry is terminal-with-reason: the
        miner re-funds to re-enter, exactly as after any other terminal state.
        """
        if max_age_seconds is None:
            max_age_seconds = self.entry_ttl_seconds
        with self._locked():
            now = self.clock()
            expired = [hk for hk, e in self._entries.items()
                       if e.status in ("queued", "pending_reveal")
                       and now - e.active_at > max_age_seconds]
            for hk in expired:
                self._entries[hk] = replace(
                    self._entries[hk], status="failed",
                    last_error="funding expired: entry outlived the payer-key TTL "
                               "without entering a round — fund again to re-enter",
                    last_error_class="funding_expired")
            if expired:
                self._save()
            return len(expired)

    def mark_done(self, hotkey: str) -> None:
        with self._locked():
            e = self._entries.get(hotkey)
            if e is not None:
                self._entries[hotkey] = replace(e, status="done")
                self._save()

    def withdraw(self, hotkey: str) -> bool:
        """Miner-initiated exit while queued/pending; False once a round has it."""
        with self._locked():
            e = self._entries.get(hotkey)
            if e is None or e.status not in ("queued", "pending_reveal"):
                return False
            self._entries[hotkey] = replace(e, status="withdrawn")
            self._save()
            return True

    def requeue(self, hotkey: str, *, error: str, error_class: str,
                burn_attempt: bool, max_attempts: int = DEFAULT_MAX_ATTEMPTS) -> bool:
        """Return a failed leg to the queue (the no-burn path).

        ``burn_attempt`` follows the fault taxonomy: infra faults consume one
        of ``max_attempts``; sold-out and rate-limited do not (they are the
        market's fault, not anyone's retry budget). Exhausted attempts turn
        the entry terminal ``failed`` — returns False, and the miner's next
        ``fund`` starts fresh. The entry itself is NEVER silently dropped:
        the miner paid for visibility into where their money went.
        """
        with self._locked():
            e = self._entries.get(hotkey)
            if e is None:
                return False
            attempts = e.attempts + (1 if burn_attempt else 0)
            if burn_attempt and attempts > max_attempts:
                self._entries[hotkey] = replace(
                    e, status="failed", attempts=attempts,
                    last_error=error[:500], last_error_class=error_class)
                self._save()
                return False
            self._entries[hotkey] = replace(
                e, status="queued", attempts=attempts,
                last_error=error[:500], last_error_class=error_class,
                last_active=self.clock())
            self._save()
            return True

    def fail(self, hotkey: str, *, error: str, error_class: str = "") -> None:
        """Terminal failure (e.g. an auth-class key the miner must replace)."""
        with self._locked():
            e = self._entries.get(hotkey)
            if e is not None:
                self._entries[hotkey] = replace(
                    e, status="failed", last_error=error[:500], last_error_class=error_class)
                self._save()

    # ── transparency feed ────────────────────────────────────────────────────

    def public_view(self) -> dict:
        """The queue as miners may see it — no key material, no sealed field.

        ``pending_reveal`` entries are REDACTED to a bare count: their chain
        commit is still timelock-encrypted, so listing (hotkey, ref) here
        would leak exactly what the timed reveal hides — who is entering the
        next round, hours early (review 2026-08-29). They join the listing
        the moment their reveal resolves, which is when the chain shows them
        anyway.
        """
        current = self.entries()
        return {
            "queued_depth": sum(1 for e in current if e.status == "queued"),
            "pending_reveal_count": sum(1 for e in current
                                        if e.status == "pending_reveal"),
            "entries": [
                {
                    "hotkey": e.hotkey,
                    "ref": e.ref,
                    "reveal_block": e.reveal_block,
                    "status": e.status,
                    "attempts": e.attempts,
                    "last_error_class": e.last_error_class,
                }
                for e in current if e.status != "pending_reveal"
            ],
        }
