"""The funded-challenger queue: who has paid for a lane, in what order.

The queue is the rolling intake (DEC-CA-0036 / DEC-CA-0043): entries start
legs ``funded_field_cap`` at a time, ordered by **reveal block** — earliest
commit first, the one seniority claim the repo trusts (a UID is not one; see
NOTE-ca-operational-invariants).

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
import re
import time
import uuid
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from .vault import DEFAULT_TTL_SECONDS

__all__ = [
    "COMPETITION_ENDED_CLASS",
    "DEFAULT_MAX_ATTEMPTS",
    "OPERATOR_FAULT_CLASS",
    "FundedEntry",
    "FundedQueue",
    "LABEL_MAX_CHARS",
    "normalize_label",
    "select_field",
]

# Bounded auto-retry for infra faults, the PRISM default. Sold-out/rate-limit
# requeues do NOT count against this (should_recover's no-burn classes).
DEFAULT_MAX_ATTEMPTS = 3
# At this many operator-side losses, pause further payer rentals when no
# operator make-good is configured. Includes the initial lost leg.
OPERATOR_FAULT_RETRIES = 3

# The error class of a leg lost to the OPERATOR's side (checkpoint storage,
# the CPU generation tier, a provider booting a stale image) while the miner's
# pod was fine: never burns an attempt, and backs off — the entry may not
# start again before ``not_before``, doubling per consecutive such requeue up
# to the cap — so a long outage waits instead of re-renting and re-generating
# every poll.
OPERATOR_FAULT_CLASS = "operator"
# The terminal class of a lane-C entry whose competition ended before it
# could train (concurrent competitions, owner 2026-10-20): no pod was rented,
# so nothing was charged, and it is NOT the miner's fault.
COMPETITION_ENDED_CLASS = "competition_ended"
INFRA_BACKOFF_BASE_SECONDS = 120.0
INFRA_BACKOFF_MAX_SECONDS = 1800.0

# Public, bounded lifecycle telemetry. Never store provider error text or keys.
PHASES = frozenset({"admitted", "waiting_admission", "preparing_data", "waiting_gpu",
                    "booting", "training", "harvesting", "evaluating",
                    "awaiting_settlement", "waiting_operator", "queued", "done",
                    "failed", "withdrawn"})
PHASE_REASONS = frozenset({"", "availability_check", "market_capacity", "king_priority",
                          "operator_recovery"})



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
    # When this entry FIRST hit a rate_limited requeue in its current streak
    # (0.0 = not in one; any non-rate_limited outcome resets it). The
    # taxonomy's RECOVERY_WINDOW applies to rate limits only — a key 429ing
    # for six hours is not going to clear by waiting, and without the bound a
    # hostile payer could occupy a funded seat forever (review 2026-09-02).
    rate_limited_since: float = 0.0
    # Miner-chosen display name for this submission (``cascade fund --label``,
    # funded UX). PRESENTATIONAL ONLY: it rides the public queue view and the
    # funded roster so a miner can pick their entry out
    # of a dashboard by name. It is never identity (hotkey + ref are), never
    # signed, never in a manifest or receipt, and never read by any scoring
    # path. Normalised by :func:`normalize_label`; "" = unlabelled.
    label: str = ""
    # Rolling intake (DEC-CA-0043): an ``in_flight`` leg trains NOW, outside
    # any round, toward ``target_boundary`` (the settlement it is expected to
    # land in) under era ``era_index``'s seeds and init; ``started_block`` is
    # the chain height at leg start — the block its ref is judged as of. A
    # re-commit while the leg is in flight parks as ``queued_ref`` /
    # ``queued_reveal_block`` and becomes the hotkey's next queued entry when
    # the flight settles (one live entry per hotkey; the running leg is never
    # rebound). All 0/"" for boundary-synchronous entries.
    target_boundary: int = 0
    era_index: int = 0
    started_block: int = 0
    training_init: str | None = None
    training_contract_digest: str = ""
    # Owner make-good (2026-09-24): a leg the trainer's own fault cost the
    # miner is rerun on compute the OPERATOR pays for — the rent path routes a
    # flagged entry to an operator lane and never touches the payer's key.
    # Set by the operator (never by intake); cleared with the entry's next
    # ``fund``.
    queued_ref: str = ""
    queued_reveal_block: int = 0
    operator_billed: bool = False
    # Times work on this entry was lost on our side (each time the flag above
    # turns on). Without operator capacity the leg retries on the payer's key
    # with a new pod until this reaches the trainer's cap, then pauses.
    operator_faults: int = 0
    # Operator-fault requeue backoff (OPERATOR_FAULT_CLASS, INFRA_BACKOFF_*): the leg does not start
    # before ``not_before`` (queue clock); ``infra_streak`` counts consecutive
    # such requeues. Any other requeue outcome resets both.
    not_before: float = 0.0
    infra_streak: int = 0
    # DEC-CA-0060 miner-chosen GPU type (``X-Gpu-Type``): "" = the default
    # path (cheapest fitting type under the operator caps). A chosen type is
    # rented under the miner's own $/h cap (0 = none) and, with
    # ``gpu_allow_fallback`` false (the default), WAITS for that type —
    # requeued like any capacity miss — instead of renting another. Set by
    # every fund (the latest request wins; one without the header clears it).
    gpu: str = ""
    gpu_allow_fallback: bool = False
    gpu_max_price_per_hour: float = 0.0
    phase: str = ""
    phase_at: float = 0.0
    phase_reason: str = ""
    phase_history: tuple[dict, ...] = ()
    requeues: int = 0
    attempt_id: str = ""

    @property
    def active_at(self) -> float:
        return self.last_active or self.funded_at

    @property
    def gpu_choice(self):
        """The entry's :class:`~cascade.funding.gpu_choice.GpuChoice`."""
        from .gpu_choice import GpuChoice

        return GpuChoice(gpu=self.gpu, allow_fallback=self.gpu_allow_fallback,
                         max_price_per_hour=self.gpu_max_price_per_hour)


def _phase(entry: FundedEntry, phase: str, now: float, reason: str = "", *, force=False) -> FundedEntry:
    if not force and (entry.phase, entry.phase_reason) == (phase, reason):
        return entry
    event = {"phase": phase, "at": now, "reason": reason, "error_class": entry.last_error_class,
             "attempt_id": entry.attempt_id,
             "target_boundary": entry.target_boundary, "era_index": entry.era_index}
    return replace(entry, phase=phase, phase_at=now, phase_reason=reason,
                   phase_history=(*entry.phase_history, event)[-16:])


# DEC-CA-0060 fields, written to the queue file only when set: an entry
# without a GPU choice serialises exactly as before the fields existed.
_GPU_FIELD_DEFAULTS = {"gpu": "", "gpu_allow_fallback": False, "gpu_max_price_per_hour": 0.0}


def _gpu_fields(choice) -> dict:
    """``replace()`` kwargs for a fund's GpuChoice; ``{}`` when None (keep)."""
    if choice is None:
        return {}
    return {"gpu": str(choice.gpu or ""),
            "gpu_allow_fallback": bool(choice.allow_fallback) if choice.gpu else False,
            "gpu_max_price_per_hour": float(choice.max_price_per_hour) if choice.gpu else 0.0}


LABEL_MAX_CHARS = 32
_LABEL_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def normalize_label(raw: object) -> str:
    """The stored form of a submission label, or ``ValueError`` when unusable.

    Whitespace-stripped; empty means "no label". Otherwise 1–``LABEL_MAX_CHARS``
    characters from ``[A-Za-z0-9._-]`` — a slug, not free text: the label is
    echoed into public JSON documents and terminal renderers, so no spaces,
    no unicode, no markup, and nothing that could pass as another hotkey (the
    charset has no ``5…`` SS58 ambiguity at 32 chars, and renderers always
    print it beside the hotkey, never instead of it).
    """
    text = str(raw or "").strip()
    if not text:
        return ""
    if len(text) > LABEL_MAX_CHARS:
        raise ValueError(f"label longer than {LABEL_MAX_CHARS} characters")
    if not _LABEL_RE.match(text):
        raise ValueError("label may only contain letters, digits, '.', '_' and '-'")
    return text


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

    # ── persistence ──────────────────────────────────────────────────────────
    #
    # There is NO cached ``self._entries``: this instance is shared across the
    # intake's handler threads and the trainer process, so a per-instance dict
    # would race (an unlocked read rebinding it mid-write could serialise a
    # torn or empty map, dropping a just-202'd fund — review 2026-08-29). Every
    # operation instead loads a FRESH local dict under a flock and never shares
    # it. Writers hold LOCK_EX; readers hold LOCK_SH so they cannot observe a
    # write mid-flight, yet concurrent readers do not block each other.

    @contextmanager
    def _locked(self, *, exclusive: bool = True) -> Iterator[dict[str, FundedEntry]]:
        """flock the sibling ``.lock``, yield a freshly-loaded local dict.

        Writers pass ``exclusive=True`` (LOCK_EX) and call :meth:`_save` with
        the dict they mutated; readers pass ``exclusive=False`` (LOCK_SH) and
        only read. The dict is local to this call — never stored on ``self`` —
        so nothing another thread does can corrupt it.

        NOT re-entrant: each call opens its OWN fd, and flock conflicts across
        fds even within one thread, so calling a public read (``get``/
        ``entries``/``queued_depth``) or another mutator from INSIDE a
        ``with self._locked()`` block self-deadlocks. Every method here works
        the yielded ``entries`` dict directly for exactly this reason — keep it
        that way; never nest a public queue call inside a lock.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        with open(lock_path, "w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
            try:
                yield self._load()
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)

    def _load(self) -> dict[str, FundedEntry]:
        entries: dict[str, FundedEntry] = {}
        if not self.path.is_file():
            return entries
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
                rate_limited_since=float(item.get("rate_limited_since", 0.0)),
                label=str(item.get("label", "")),
                target_boundary=int(item.get("target_boundary", 0) or 0),
                era_index=int(item.get("era_index", 0) or 0),
                started_block=int(item.get("started_block", 0) or 0),
                training_init=item.get("training_init"),
                training_contract_digest=str(item.get("training_contract_digest", "") or ""),
                queued_ref=str(item.get("queued_ref", "") or ""),
                queued_reveal_block=int(item.get("queued_reveal_block", 0) or 0),
                operator_billed=bool(item.get("operator_billed", False)),
                operator_faults=int(item.get("operator_faults", 0) or 0),
                not_before=float(item.get("not_before", 0.0) or 0.0),
                infra_streak=int(item.get("infra_streak", 0) or 0),
                gpu=str(item.get("gpu", "") or ""),
                gpu_allow_fallback=bool(item.get("gpu_allow_fallback", False)),
                gpu_max_price_per_hour=float(item.get("gpu_max_price_per_hour", 0.0) or 0.0),
                phase=str(item.get("phase", "") or ""),
                phase_at=float(item.get("phase_at", 0.0) or 0.0),
                phase_reason=str(item.get("phase_reason", "") or ""),
                phase_history=tuple(item.get("phase_history") or ())[-16:],
                requeues=int(item.get("requeues", 0) or 0),
                attempt_id=str(item.get("attempt_id", "") or ""),
            )
            entries[entry.hotkey] = entry
        return entries

    def _save(self, entries: dict[str, FundedEntry]) -> None:
        import os

        self.path.parent.mkdir(parents=True, exist_ok=True)
        def _row(e: FundedEntry) -> dict:
            row = asdict(e)
            for k, default in _GPU_FIELD_DEFAULTS.items():
                if row.get(k) == default:
                    row.pop(k, None)
            return row

        payload = {
            "entries": [_row(e) for e in sorted(
                entries.values(), key=lambda e: (e.reveal_block, e.hotkey))],
        }
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self.path)

    # ── reads (shared lock, fresh local dict — never a cached snapshot) ──────

    def entries(self) -> list[FundedEntry]:
        with self._locked(exclusive=False) as entries:
            return sorted(entries.values(), key=lambda e: (e.reveal_block, e.hotkey))

    def get(self, hotkey: str) -> FundedEntry | None:
        with self._locked(exclusive=False) as entries:
            return entries.get(hotkey)

    # ── lifecycle ────────────────────────────────────────────────────────────

    def add(self, hotkey: str, ref: str, reveal_block: int, *,
            label: str = "", gpu_choice=None) -> str:
        """Fund (or re-fund) a submission; returns the outcome for the caller.

        ``label`` (already normalised) names the entry for dashboards; an
        empty label on a re-fund keeps the one already stored, a non-empty
        one replaces it — the label is the miner's to change at any time
        because nothing downstream keys on it.

        ``gpu_choice`` (DEC-CA-0060 ``GpuChoice``; the intake always passes
        one) REPLACES the entry's choice on every accepted outcome — the
        latest fund wins, a request without a GPU resets to the default. It
        steers the NEXT rent only: an in-flight leg's pod is never touched.
        ``None`` keeps whatever is stored.

        ``"already-queued"`` — same hotkey, same ref, still live: idempotent
        no-op (the PRISM 200 path). ``"replaced"`` — a live entry updated to a
        new ref (a re-reveal before its round). ``"queued"`` — fresh entry,
        including over a terminal one (a resubmit after done/failed frees the
        slot: entry cost is the compute funding, not a lifetime bar).
        """
        with self._locked() as entries:
            now = self.clock()
            prev = entries.get(hotkey)
            if prev is not None and prev.status == "in_flight":
                # DEC-CA-0043: the running leg is never rebound. Same ref ⇒
                # proof of life; a new ref queues BEHIND the flight and
                # becomes the next queued entry when it settles.
                if prev.ref == ref:
                    entries[hotkey] = replace(prev, last_active=now, queued_ref="",
                                              queued_reveal_block=0,
                                              label=label or prev.label,
                                              **_gpu_fields(gpu_choice))
                    self._save(entries)
                    return "already-queued"
                entries[hotkey] = replace(prev, queued_ref=ref,
                                          queued_reveal_block=int(reveal_block),
                                          last_active=now, label=label or prev.label,
                                          **_gpu_fields(gpu_choice))
                self._save(entries)
                return "queued-behind"
            if prev is not None and prev.status in ("pending_reveal", "queued", "in_round"):
                if prev.ref == ref and prev.status != "pending_reveal":
                    # Idempotent, but a re-fund is proof of life — refresh so
                    # an actively-tended entry never TTL-expires under you.
                    entries[hotkey] = replace(prev, last_active=now,
                                              label=label or prev.label,
                                              **_gpu_fields(gpu_choice))
                    self._save(entries)
                    return "already-queued"
                if prev.status == "in_round":
                    # Mid-round the manifest may already reference the old ref —
                    # never mutate an entry a live round is training.
                    return "already-queued"
                entries[hotkey] = replace(
                    prev, ref=ref, reveal_block=int(reveal_block),
                    funded_at=now, last_active=now, status="queued",
                    label=label or prev.label, **_gpu_fields(gpu_choice))
                self._save(entries)
                return "replaced" if prev.ref != ref else "queued"
            entries[hotkey] = FundedEntry(
                hotkey=hotkey, ref=ref, reveal_block=int(reveal_block),
                funded_at=now, last_active=now, label=label, **_gpu_fields(gpu_choice))
            self._save(entries)
            return "queued"

    def add_pending(self, hotkey: str, ref: str, *, label: str = "") -> str:
        """A submit-with-key entry whose reveal has not landed on chain yet.

        ``label`` follows :meth:`add`'s rule (empty keeps, non-empty replaces).

        The one-request flow: the ZIP and the Lium key arrive together, before
        the miner's chain commit reveals — so the entry parks as
        ``pending_reveal`` (no reveal block, no seniority, never selected) and
        :meth:`promote_pending` flips it to ``queued`` the moment the reveal
        resolves. Idempotent per (hotkey, ref); a live queued/in_round entry
        is left alone (they already have a better state).
        """
        with self._locked() as entries:
            prev = entries.get(hotkey)
            if prev is not None and prev.status in ("queued", "in_round", "in_flight"):
                return "already-queued"
            now = self.clock()
            if prev is not None and prev.status == "pending_reveal" and prev.ref == ref:
                entries[hotkey] = replace(prev, last_active=now,
                                          label=label or prev.label)
                self._save(entries)
                return "already-pending"
            entries[hotkey] = FundedEntry(
                hotkey=hotkey, ref=ref, reveal_block=0,
                funded_at=now, last_active=now, status="pending_reveal",
                label=label)
            self._save(entries)
            return "pending_reveal"

    def promote_pending(self, resolve_reveal) -> int:
        """Flip pending entries whose reveal now resolves; count promoted.

        ``resolve_reveal(hotkey, ref) -> int | None`` is the intake's oracle.
        Resolution stamps the REAL reveal block, so seniority is chain truth,
        never upload order. Unresolvable entries stay pending until they
        resolve or :meth:`expire_stale` reaps them at the key TTL.

        The chain resolver is called OUTSIDE the flock: a substrate poll can
        hang with no timeout, and holding the exclusive lock across it would
        stall every other queue op — including the trainer's admission. So: snapshot the pending set under a
        SHARED lock, resolve unlocked, then apply the resolutions under the
        exclusive lock, re-checking each entry is still the same pending ref.
        """
        with self._locked(exclusive=False) as snapshot:
            pending = {hk: e.ref for hk, e in snapshot.items()
                       if e.status == "pending_reveal"}
        if not pending:
            return 0
        resolved = {hk: resolve_reveal(hk, ref) for hk, ref in pending.items()}
        resolved = {hk: b for hk, b in resolved.items() if b is not None}
        if not resolved:
            return 0
        with self._locked() as entries:
            promoted = 0
            for hk, block in resolved.items():
                e = entries.get(hk)
                # Re-check under the lock: the entry must still be the same
                # pending ref we resolved (a re-fund could have replaced it).
                if e is None or e.status != "pending_reveal" or e.ref != pending[hk]:
                    continue
                entries[hk] = replace(
                    e, status="queued", reveal_block=int(block),
                    last_active=self.clock())
                promoted += 1
            if promoted:
                self._save(entries)
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
        with self._locked() as entries:
            confirmed: list[str] = []
            for sel in selections:
                hk, ref = sel if isinstance(sel, tuple) else (sel, None)
                e = entries.get(hk)
                if e is None or e.status != "queued":
                    continue
                if ref is not None and e.ref != ref:
                    continue
                entries[hk] = replace(e, status="in_round")
                confirmed.append(hk)
            if confirmed:
                self._save(entries)
            return confirmed

    def mark_in_flight(self, hotkey: str, ref: str, *, target_boundary: int,
                       era_index: int, started_block: int, training_init: str | None = None,
                       training_contract_digest: str = "") -> bool:
        """Flip a ``queued`` entry to ``in_flight`` (DEC-CA-0043 rolling
        intake): its leg starts now toward ``target_boundary`` under era
        ``era_index``. Ref re-checked inside the lock like
        :meth:`mark_in_round`; False when the entry moved underneath, or
        while it backs off after an operator-fault requeue (``not_before``)."""
        with self._locked() as entries:
            e = entries.get(hotkey)
            if e is None or e.status != "queued" or e.ref != ref:
                return False
            if e.not_before and self.clock() < e.not_before:
                return False
            entries[hotkey] = _phase(replace(
                e, status="in_flight", target_boundary=int(target_boundary),
                era_index=int(era_index), started_block=int(started_block),
                training_init=training_init, training_contract_digest=training_contract_digest,
                attempt_id=uuid.uuid4().hex,
                last_active=self.clock()), "admitted", self.clock())
            self._save(entries)
            return True

    def in_flight(self) -> list[FundedEntry]:
        """Admitted legs, including preparation, capacity waits and judging."""
        return [e for e in self.entries() if e.status == "in_flight"]

    def set_phase(self, hotkey: str, phase: str, *, expect_ref: str,
                  expect_started_block: int, expect_funded_at: float,
                  expect_attempt_id: str,
                  reason: str = "") -> bool:
        """Update only the same running attempt; late callbacks cannot relabel a retry."""
        if phase not in PHASES - {"queued", "done", "failed", "withdrawn"} or reason not in PHASE_REASONS:
            raise ValueError("unknown public phase/reason")
        with self._locked() as entries:
            e = entries.get(hotkey)
            if (e is None or e.status != "in_flight" or e.ref != expect_ref
                    or e.started_block != expect_started_block or e.funded_at != expect_funded_at
                    or e.attempt_id != expect_attempt_id):
                return False
            updated = _phase(e, phase, self.clock(), reason)
            if updated != e:
                entries[hotkey] = updated
                self._save(entries)
            return True

    def set_operator_billed(self, hotkey: str, flag: bool) -> bool:
        """Mark (or unmark) an entry as operator-billed: its leg runs on an
        operator lane, never on the payer's key (owner make-good for a leg
        the trainer's own fault cost). Any status; False when unknown."""
        with self._locked() as entries:
            e = entries.get(hotkey)
            if e is None:
                return False
            entries[hotkey] = replace(
                e, operator_billed=bool(flag),
                operator_faults=e.operator_faults + (1 if flag and not e.operator_billed else 0))
            self._save(entries)
            return True

    def restamp_flight(self, hotkey: str, *, target_boundary: int, era_index: int,
                       started_block: int) -> bool:
        """Replace an ``in_flight`` entry's era / target / start stamp — the
        restart re-attach found it unsound (zeroed by a requeue, stamped by a
        trainer that predates the fields) and re-derived it. 2026-09-24: a leg
        re-attached from a zeroed stamp published ``train_block 0`` and
        validators rejected the settlement whole."""
        with self._locked() as entries:
            e = entries.get(hotkey)
            if e is None or e.status != "in_flight":
                return False
            entries[hotkey] = replace(
                e, target_boundary=int(target_boundary), era_index=int(era_index),
                started_block=int(started_block))
            self._save(entries)
            return True

    def _after_flight(self, entries: dict[str, FundedEntry], hotkey: str) -> None:
        """A settled flight with a re-commit parked behind it: the parked ref
        becomes the hotkey's fresh queued entry (attempts reset — it is a new
        submission). Called INSIDE the lock, after the flight's own outcome
        was written."""
        e = entries.get(hotkey)
        if e is None or not e.queued_ref:
            return
        now = self.clock()
        entries[hotkey] = FundedEntry(
            hotkey=hotkey, ref=e.queued_ref, reveal_block=int(e.queued_reveal_block),
            funded_at=now, last_active=now, label=e.label, gpu=e.gpu,
            gpu_allow_fallback=e.gpu_allow_fallback,
            gpu_max_price_per_hour=e.gpu_max_price_per_hour)

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
        with self._locked() as entries:
            now = self.clock()
            expired = [hk for hk, e in entries.items()
                       if e.status in ("queued", "pending_reveal")
                       and now - e.active_at > max_age_seconds]
            for hk in expired:
                entries[hk] = _phase(replace(
                    entries[hk], status="failed",
                    last_error="funding expired: entry outlived the payer-key TTL "
                               "without entering a round — fund again to re-enter",
                    last_error_class="funding_expired"), "failed", now)
            if expired:
                self._save(entries)
            return len(expired)

    def mark_done(self, hotkey: str, *, expect_ref: str | None = None,
                  expect_started_block: int | None = None) -> None:
        """Complete only the named flight when called by a settlement replay.

        A previous call may already have promoted a parked re-commit. Never
        mark that next entry done when replaying the old settlement.
        """
        with self._locked() as entries:
            e = entries.get(hotkey)
            if expect_ref is not None and (e is None or e.ref != expect_ref
                                           or e.status != "in_flight"):
                return
            if expect_started_block is not None and (
                    e is None or e.started_block != int(expect_started_block)):
                return
            if e is not None:
                was_flight = e.status == "in_flight"
                entries[hotkey] = _phase(replace(e, status="done"), "done", self.clock())
                if was_flight:
                    self._after_flight(entries, hotkey)
                self._save(entries)

    def withdraw(self, hotkey: str) -> bool:
        """Miner-initiated exit while queued/pending; False once a round has it."""
        with self._locked() as entries:
            e = entries.get(hotkey)
            if e is None or e.status not in ("queued", "pending_reveal"):
                return False
            entries[hotkey] = _phase(replace(e, status="withdrawn"), "withdrawn", self.clock())
            self._save(entries)
            return True

    def requeue(self, hotkey: str, *, error: str, error_class: str,
                burn_attempt: bool, max_attempts: int = DEFAULT_MAX_ATTEMPTS,
                operator_billed: bool = False) -> bool:
        """Return a failed leg to the queue (the no-burn path).

        ``burn_attempt`` follows the fault taxonomy: infra faults consume one
        of ``max_attempts``; sold-out and rate-limited do not (they are the
        market's fault, not anyone's retry budget). An operator-side fault
        (:data:`OPERATOR_FAULT_CLASS`, never burned) backs off instead: the
        entry cannot start again (:meth:`mark_in_flight`) before ``not_before``. Exhausted attempts turn
        the entry terminal ``failed`` — returns False, and the miner's next
        ``fund`` starts fresh. The entry itself is NEVER silently dropped:
        the miner paid for visibility into where their money went.

        ``operator_billed=True`` marks work lost on our side. Configured operator
        capacity handles recovery; otherwise payer retries stop at the loss cap.

        Rate limits get the taxonomy's RECOVERY_WINDOW: an unbroken streak of
        ``rate_limited`` requeues longer than it turns terminal — otherwise a
        key pinned at its rate limit occupies a funded seat and triggers a
        fresh rent attempt every round, forever (review 2026-09-02).
        """
        from .faults import RECOVERY_WINDOW_SECONDS

        with self._locked() as entries:
            e = entries.get(hotkey)
            if e is None:
                return False
            now = self.clock()
            rl_since = 0.0
            if error_class == "rate_limited":
                rl_since = e.rate_limited_since or now
                if now - rl_since > RECOVERY_WINDOW_SECONDS:
                    entries[hotkey] = _phase(replace(
                        e, status="failed",
                        last_error=(error[:400] + " [rate-limited past the "
                                    "recovery window — fix the key's rate "
                                    "limit, then fund again]"),
                        last_error_class=error_class), "failed", now)
                    self._save(entries)
                    return False
            attempts = e.attempts + (1 if burn_attempt else 0)
            was_flight = e.status == "in_flight"
            if burn_attempt and attempts > max_attempts:
                entries[hotkey] = _phase(replace(
                    e, status="failed", attempts=attempts,
                    last_error=error[:500], last_error_class=error_class), "failed", now)
                if was_flight:
                    self._after_flight(entries, hotkey)
                self._save(entries)
                return False
            streak, not_before = 0, 0.0
            if error_class == OPERATOR_FAULT_CLASS and not burn_attempt:
                streak = e.infra_streak + 1
                not_before = now + min(INFRA_BACKOFF_MAX_SECONDS,
                                       INFRA_BACKOFF_BASE_SECONDS * 2 ** (streak - 1))
            entries[hotkey] = _phase(replace(
                e, status="queued", attempts=attempts,
                last_error=error[:500], last_error_class=error_class,
                last_active=now, rate_limited_since=rl_since,
                target_boundary=0, era_index=0, started_block=0,
                training_init=None, training_contract_digest="",
                operator_billed=e.operator_billed or bool(operator_billed),
                operator_faults=e.operator_faults + (
                    1 if operator_billed and not e.operator_billed else 0),
                not_before=not_before, infra_streak=streak, requeues=e.requeues + 1),
                "queued", now, force=True)
            if was_flight and e.queued_ref:
                # The re-commit supersedes the requeue: the miner moved on.
                self._after_flight(entries, hotkey)
            self._save(entries)
            return True

    def touch(self, hotkeys: Iterable[str]) -> int:
        """Refresh ``last_active`` for queued entries; count touched.

        Proof-of-life without a state change: an entry the ADMISSION side held
        back (capacity clamp, more-senior entries out-capping it) never rents,
        so nothing else refreshes it — without this it TTL-expires while
        actively waiting through no fault of its own (review 2026-09-02).
        """
        with self._locked() as entries:
            now = self.clock()
            touched = 0
            for hk in hotkeys:
                e = entries.get(hk)
                if e is not None and e.status == "queued":
                    entries[hk] = replace(e, last_active=now)
                    touched += 1
            if touched:
                self._save(entries)
            return touched

    def fail(self, hotkey: str, *, error: str, error_class: str = "",
             expect_ref: str | None = None) -> bool:
        """Terminal failure (e.g. an auth-class key the miner must replace).

        ``expect_ref`` guards against the select→fail race: a caller that
        decided to fail an entry off a snapshot passes the ref it saw, and the
        fail is SKIPPED if a re-fund replaced the entry with a different ref in
        the window — otherwise the miner's fresh, correctly-funded entry would
        be terminally failed for the OLD ref's reason (review 2026-08-29).
        Returns True iff the entry was failed.
        """
        with self._locked() as entries:
            e = entries.get(hotkey)
            if e is None:
                return False
            if expect_ref is not None and e.ref != expect_ref:
                return False
            was_flight = e.status == "in_flight"
            entries[hotkey] = _phase(replace(
                e, status="failed", last_error=error[:500], last_error_class=error_class),
                "failed", self.clock())
            if was_flight:
                self._after_flight(entries, hotkey)
            self._save(entries)
            return True

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
            "as_of": datetime.fromtimestamp(self.clock(), UTC).isoformat(),
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
                    "funded_at": e.funded_at,
                    "last_active": e.last_active,
                    "not_before": e.not_before,
                    "requeues": e.requeues,
                    "attempt_id": e.attempt_id,
                    "phase": (e.phase if e.status == "in_flight" else e.status),
                    "phase_at": e.phase_at,
                    "phase_reason": e.phase_reason,
                    "phase_history": list(e.phase_history),
                    **({"queued_ref": e.queued_ref,
                        "queued_reveal_block": e.queued_reveal_block}
                       if e.queued_ref and e.queued_reveal_block > 0 else {}),
                    **({"operator_billed": True} if e.operator_billed else {}),
                    **({"operator_faults": e.operator_faults} if e.operator_faults else {}),
                    "label": e.label,
                    # DEC-CA-0060: "" = the default (cheapest fitting type)
                    "gpu": e.gpu,
                    **({"gpu_allow_fallback": e.gpu_allow_fallback,
                        "gpu_max_price_per_hour": e.gpu_max_price_per_hour}
                       if e.gpu else {}),
                    **({"target_boundary": e.target_boundary, "era_index": e.era_index,
                        "started_block": e.started_block}
                       if e.status == "in_flight" else {}),
                }
                for e in current if e.status != "pending_reveal"
            ],
        }


# ── lane C (DEC-CA-0056): the fine-tune queue ────────────────────────────────
#
# A SEPARATE file beside the funded-challenger queue, with the same semantics
# (flock'd reload-before-act, one live entry per hotkey, infra requeue with
# the bounded attempt budget, operator-fault backoff, TTL expiry, the
# in-flight re-commit parking). Its entries are keyed by hotkey like the
# challenger queue's and carry the miner's declared TARGET with the ref: the
# entry's ``ref`` field holds ``"<target>|<ref>"`` (neither a target spec nor
# a Hub ref can contain ``|``), so a re-fund for another target or another
# ref replaces the entry exactly like a re-fund for another ref does in lane
# A, and the in-flight ref re-check covers the target too. Nothing here burns
# a lane-A submission: lane C never touches the one-submission-per-hotkey
# ledger.

LANE_C_QUEUE_FILE = "lane_c_queue.json"
_LANE_C_SEP = "|"


def lane_c_queue_path(funded_queue_path: Path | str) -> Path:
    """The lane-C queue file: ``lane_c_queue.json`` beside the challenger
    queue. Derived (never configured separately) so the intake and the trainer
    can never split-brain on it: both resolve it from the same
    ``--queue-path`` / ``[round] funded_queue_path``."""
    return Path(funded_queue_path).with_name(LANE_C_QUEUE_FILE)


def lane_c_entry_ref(target: str, ref: str) -> str:
    target, ref = str(target).strip(), str(ref).strip()
    if not target or not ref or _LANE_C_SEP in target or _LANE_C_SEP in ref:
        raise ValueError(f"bad lane-C entry (target={target!r}, ref={ref!r})")
    return f"{target}{_LANE_C_SEP}{ref}"


def split_lane_c_ref(entry_ref: str) -> tuple[str, str]:
    """``(target, ref)`` of a lane-C queue entry's ``ref`` field."""
    target, sep, ref = str(entry_ref).partition(_LANE_C_SEP)
    if not sep:
        raise ValueError(f"not a lane-C queue ref: {entry_ref!r}")
    return target, ref


class LaneCQueue(FundedQueue):
    """The lane-C fine-tune queue (DEC-CA-0056): :class:`FundedQueue` over
    ``"<target>|<ref>"`` entry refs (see the module note above)."""

    def add_lane_c(self, hotkey: str, target: str, ref: str, reveal_block: int, *,
                   label: str = "", gpu_choice=None) -> str:
        return self.add(hotkey, lane_c_entry_ref(target, ref), int(reveal_block), label=label,
                        gpu_choice=gpu_choice)

    def queued(self, *, now: float | None = None) -> list[FundedEntry]:
        """Startable entries, earliest reveal first: ``queued`` and not
        backing off (``not_before``)."""
        t = self.clock() if now is None else float(now)
        return [e for e in select_field(self.entries(), 0)
                if not (e.not_before and t < e.not_before)]

    def public_view(self) -> dict:
        view = super().public_view()
        for row in view.get("entries", []):
            try:
                row["target"], row["ref"] = split_lane_c_ref(row["ref"])
                if row.get("queued_ref"):
                    row["queued_target"], row["queued_ref"] = split_lane_c_ref(row["queued_ref"])
            except ValueError:
                row["target"] = ""
        view["lane"] = "c"
        return view


__all__ += ["LANE_C_QUEUE_FILE", "LaneCQueue", "lane_c_entry_ref", "lane_c_queue_path",
            "split_lane_c_ref"]


# Lane-B funding queue (DEC-CA-0059), separate from the trainer's LaneBQueue
# that tracks expert-row reservations. Entry refs bind slot-ordered targets
# and the generator: "<t1>,<t2>|<ref>". Re-funds follow lane-A replacement
# semantics; in-flight identity includes the targets. This queue does not
# consume the lane-A lifetime-submission ledger.

LANE_B_QUEUE_FILE = "lane_b_queue.json"
_LANE_B_TARGET_SEP = ","


def lane_b_queue_path(funded_queue_path: Path | str) -> Path:
    """The lane-B queue file: ``lane_b_queue.json`` beside the challenger
    queue — derived, never configured, so the intake and the trainer agree."""
    return Path(funded_queue_path).with_name(LANE_B_QUEUE_FILE)


def lane_b_entry_ref(targets, ref: str) -> str:
    """``"<t1>,<t2>|<ref>"`` for the declared ``targets`` (one or more, in the
    order given — callers pass the slot order) and ``ref``."""
    ts = [str(t).strip() for t in (targets or ())]
    ref = str(ref).strip()
    bad = [t for t in ts if not t or _LANE_B_TARGET_SEP in t or _LANE_C_SEP in t]
    if not ts or bad or len(set(ts)) != len(ts) or not ref or _LANE_C_SEP in ref:
        raise ValueError(f"bad lane-B entry (targets={list(targets or ())!r}, ref={ref!r})")
    return f"{_LANE_B_TARGET_SEP.join(ts)}{_LANE_C_SEP}{ref}"


def split_lane_b_ref(entry_ref: str) -> tuple[tuple[str, ...], str]:
    """``(targets, ref)`` of a lane-B queue entry's ``ref`` field."""
    head, sep, ref = str(entry_ref).partition(_LANE_C_SEP)
    targets = tuple(t for t in head.split(_LANE_B_TARGET_SEP))
    if not sep or not ref or not head or any(not t for t in targets):
        raise ValueError(f"not a lane-B queue ref: {entry_ref!r}")
    return targets, ref


class LaneBQueue(FundedQueue):
    """The lane-B domain-expert FUNDING queue (DEC-CA-0059 item 8):
    :class:`FundedQueue` over ``"<targets>|<ref>"`` entry refs (see the
    module note above)."""

    def add_lane_b(self, hotkey: str, targets, ref: str, reveal_block: int, *,
                   label: str = "", gpu_choice=None) -> str:
        return self.add(hotkey, lane_b_entry_ref(targets, ref), int(reveal_block), label=label,
                        gpu_choice=gpu_choice)

    def queued(self, *, now: float | None = None) -> list[FundedEntry]:
        """Startable entries, earliest reveal first: ``queued`` and not
        backing off (``not_before``)."""
        t = self.clock() if now is None else float(now)
        return [e for e in select_field(self.entries(), 0)
                if not (e.not_before and t < e.not_before)]

    def public_view(self) -> dict:
        view = super().public_view()
        for row in view.get("entries", []):
            try:
                targets, row["ref"] = split_lane_b_ref(row["ref"])
                row["targets"] = list(targets)
                if row.get("queued_ref"):
                    queued_targets, row["queued_ref"] = split_lane_b_ref(row["queued_ref"])
                    row["queued_targets"] = list(queued_targets)
            except ValueError:
                row["targets"] = []
        view["lane"] = "b"
        return view


__all__ += ["LANE_B_QUEUE_FILE", "LaneBQueue", "lane_b_entry_ref", "lane_b_queue_path",
            "split_lane_b_ref"]
