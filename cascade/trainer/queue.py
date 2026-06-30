"""Trainer-side submission queue with cheap anti-duplicate checks.

cascade trains the king and a challenger together under one shared
:class:`~cascade.trainer.contract.RoundSeeds` — a single round is the
expensive resource (~3h of GPU per generator). This module sits in front of that
expense as a FIFO backlog of *challenger* generators discovered on-chain, with a
handful of O(1)/O(n) checks that keep guaranteed-wasted runs off the GPU. It is
the cascade analogue of teutonic's validator queue (FIFO drain + copy gate),
adapted to cascade's content-addressed submissions.

The headline check is **duplicate-of-king**: because a generator is pinned by its
immutable Hub ``repo@digest``, a challenger whose generator is byte-identical to
the reigning king's (same ref) would burn a full controlled-experiment round on a
generator that can only tie the king (a challenger must clear the win margin to
dethrone — an identical run never will), so it is dropped for free with a ref
equality test, never fetched, never run.

A stronger gate sits in front of all of these — **1 hotkey, 1 eval**:

* **hotkey-already-used** — once a hotkey's challenger has been enqueued, that
  hotkey can *never* be enqueued again for the lifetime of the trainer state.
  This is cascade's port of teutonic's permanent per-hotkey ``seen`` set: a
  miner gets exactly one fair shot per hotkey registration, and the only way to
  get another is to register a *fresh* hotkey on subnet and commit from it. The
  burn happens **at enqueue, not at train** (mirroring teutonic's "one reveal =
  one shot"): a crash between enqueue and the training round still consumes the
  shot, so a miner cannot retry by replaying the same commitment. Unlike the
  per-reign ``trained_refs`` cache below, ``seen_hotkeys`` is a *lifetime* burn —
  it is never cleared by :meth:`note_king`, and it persists across reigns and
  process restarts.

Three more cheap checks ride along at :meth:`SubmissionQueue.enqueue`:

* **already-queued** — the same ref is already waiting (idempotent re-discovery
  of an unchanged commitment across polls).
* **already-trained** — the ref was already trained *this reign*. The cache
  resets when the king changes (:meth:`note_king`), mirroring teutonic's
  per-cycle ``evaluated_repos``: under single-round dethroning (the shipped
  ``[scoring] dethrone_cp = 1``) a challenger that has had its one fair shot
  against the current king should not be re-trained every round until the throne
  turns over.
* **latest-commit-wins** — a new ref from a hotkey already in the queue
  supersedes that hotkey's older queued entry (miners re-deploy by committing a
  new ref), matching :func:`cascade.trainer.loop.resolve_commitments`. Under
  the lifetime hotkey burn this now applies only to a hotkey's **first**,
  still-pending submission: a re-deploy that arrives before the original was ever
  drained can still replace it, but once that submission is enqueued+seen the
  hotkey is done for good and a later re-deploy is rejected by the hotkey gate
  before it can supersede anything.

The queue is a small JSON-serialisable record so the trainer can persist it to
``[queue] state_db_path`` and keep the backlog (the per-reign dedup and the
lifetime hotkey burn) across restarts, the same way the validator persists its
champion state.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass(frozen=True)
class QueuedSubmission:
    """One challenger generator waiting for a training round.

    ``ref`` is the generator's Hippius Hub ``repo@digest`` (the OCI digest is the
    content hash); it is the key every dedup check turns on. ``commit_block`` is
    the chain block the miner revealed the commitment at (FIFO ordering uses
    arrival into the queue, not this block, so a late re-deploy does not jump the
    line)."""

    hotkey: str
    uid: int
    ref: str
    commit_block: int


# Skip-reason strings returned by enqueue() (None ⇒ accepted). Kept as constants
# so callers/tests can match on them without hardcoding the message text.
SKIP_DUPLICATE_OF_KING = "duplicate_of_king"
SKIP_ALREADY_QUEUED = "already_queued"
SKIP_ALREADY_TRAINED = "already_trained"
SKIP_HOTKEY_ALREADY_USED = "hotkey_already_used"


@dataclass
class SubmissionQueue:
    """FIFO backlog of challenger submissions with per-reign duplicate dedup.

    Not frozen: enqueue/select/mark_trained mutate ``pending`` and the trained
    cache in place. Use :func:`dumps`/:func:`loads` to persist.

    Attributes:
        pending: challengers awaiting training, oldest-first (FIFO).
        king_ref: the reigning king's generator ref the ``trained_refs`` cache is
            scoped to. When :meth:`note_king` sees a different ref the cache is
            cleared (a new reign — every challenger deserves a fresh shot).
        trained_refs: refs already trained during the current reign, most-recent
            last, capped at ``max_trained_cache`` (a ring buffer).
        max_trained_cache: cap on ``trained_refs`` (``[queue] trained_cache_size``).
        seen_hotkeys: every hotkey ever enqueued — the **1 hotkey, 1 eval** burn
            set. Unlike ``trained_refs`` this is *not* per-reign: it is never
            cleared by :meth:`note_king` and persists across restarts, so a
            hotkey that has had its one shot is rejected forever (the miner must
            register a fresh hotkey to try again). See the module docstring.
    """

    pending: list[QueuedSubmission] = field(default_factory=list)
    king_ref: str | None = None
    trained_refs: list[str] = field(default_factory=list)
    max_trained_cache: int = 256
    seen_hotkeys: set[str] = field(default_factory=set)

    # ── reign tracking ───────────────────────────────────────────────────────

    def note_king(self, king_ref: str | None) -> bool:
        """Record the reigning king's ref; reset the per-reign dedup on a change.

        Returns True if the reign turned over (the king ref changed), so a caller
        can log it. Also drops any pending entry that now *is* the king (a queued
        challenger that won the throne, or a copy of the new king). The cache is
        only reset on a genuine change to a non-None ref, so a transient missing
        king does not wipe the dedup mid-reign.
        """
        changed = king_ref is not None and king_ref != self.king_ref
        if changed:
            self.king_ref = king_ref
            self.trained_refs = []
        if king_ref is not None:
            self.pending = [q for q in self.pending if q.ref != king_ref]
        return changed

    # ── intake ───────────────────────────────────────────────────────────────

    def enqueue(self, sub: QueuedSubmission) -> str | None:
        """Add a challenger to the backlog, or return why it was skipped.

        Returns None when the submission was accepted (now pending), otherwise one
        of the ``SKIP_*`` reason strings. Cheap by construction: a membership test
        against the lifetime hotkey burn set, a ref equality test against the
        king, a membership test against the per-reign trained cache, and a linear
        scan of ``pending`` (which holds at most one entry per hotkey).

        The **1 hotkey, 1 eval** gate runs first: a hotkey that has *ever* been
        enqueued is rejected forever (``SKIP_HOTKEY_ALREADY_USED``), regardless of
        what ref it commits now — even a brand-new, king-distinct, never-trained
        generator. The only way back in is a fresh hotkey. Because this gate
        rejects any already-seen hotkey before the latest-commit-wins step below,
        a seen hotkey can no longer supersede its own queued entry with a new ref;
        latest-commit-wins therefore applies only to a hotkey's first,
        still-pending submission (a re-deploy that arrives before the original was
        ever drained). Once a submission is enqueued and the hotkey is burned, the
        hotkey is done for good — by design, under the lifetime burn.
        """
        # 1-hotkey-1-eval (lifetime): the strongest gate, checked first. A hotkey
        # that has already had its shot — even one lost to a crash between enqueue
        # and training — cannot enqueue again until the miner registers a fresh
        # hotkey on subnet. Mirrors teutonic's permanent ``seen`` set.
        if sub.hotkey in self.seen_hotkeys:
            return SKIP_HOTKEY_ALREADY_USED
        if self.king_ref is not None and sub.ref == self.king_ref:
            return SKIP_DUPLICATE_OF_KING
        if sub.ref in self.trained_refs:
            return SKIP_ALREADY_TRAINED
        if any(q.ref == sub.ref for q in self.pending):
            return SKIP_ALREADY_QUEUED
        # latest-commit-wins: a fresh ref from a hotkey already in the queue
        # supersedes that hotkey's older entry (the miner re-deployed). Under the
        # lifetime burn this only ever fires for a hotkey's first, still-pending
        # submission — a re-deploy from an already-seen hotkey is rejected above
        # before it reaches here, so a seen hotkey cannot replace its own entry.
        self.pending = [q for q in self.pending if q.hotkey != sub.hotkey]
        self.pending.append(sub)
        # Burn the hotkey at enqueue (not at train): one accepted submission spends
        # the hotkey's single eval slot for the lifetime of this state.
        self.seen_hotkeys.add(sub.hotkey)
        return None

    def prune_to_field(self, field_refs: set[str]) -> list[QueuedSubmission]:
        """Drop pending entries whose ref is no longer in the resolved on-chain
        field (the miner re-deployed to a new ref, or deregistered). Returns the
        removed entries so a caller can log them."""
        keep, drop = [], []
        for q in self.pending:
            (keep if q.ref in field_refs else drop).append(q)
        self.pending = keep
        return drop

    # ── selection / completion ───────────────────────────────────────────────

    def select(self, n: int) -> list[QueuedSubmission]:
        """The front ``n`` still-eligible challengers (FIFO), without removing them.

        Re-applies the duplicate-of-king and already-trained checks at selection
        time (the king may have changed since enqueue), so a stale entry never
        reaches the GPU. Entries stay pending until :meth:`mark_trained` confirms
        an attempt, so a crash between select and train just re-selects next round.
        """
        if n <= 0:
            return []
        out: list[QueuedSubmission] = []
        for q in self.pending:
            if len(out) >= n:
                break
            if self.king_ref is not None and q.ref == self.king_ref:
                continue
            if q.ref in self.trained_refs:
                continue
            out.append(q)
        return out

    def mark_trained(self, ref: str) -> None:
        """Record a ref as trained this reign and remove it from the backlog.

        Called once per challenger the trainer *attempted* this round (win, loss,
        or a generator that failed to train) — an attempt consumes the challenger's
        shot for this reign, so a broken or losing generator is not re-run every
        round. The cache is a bounded ring buffer."""
        self.pending = [q for q in self.pending if q.ref != ref]
        if ref in self.trained_refs:
            return
        self.trained_refs.append(ref)
        if len(self.trained_refs) > self.max_trained_cache:
            self.trained_refs = self.trained_refs[-self.max_trained_cache :]


def dumps(queue: SubmissionQueue) -> str:
    return json.dumps(
        {
            "pending": [
                {"hotkey": q.hotkey, "uid": q.uid, "ref": q.ref, "commit_block": q.commit_block}
                for q in queue.pending
            ],
            "king_ref": queue.king_ref,
            "trained_refs": list(queue.trained_refs),
            "max_trained_cache": queue.max_trained_cache,
            # sorted() so the lifetime burn set serialises deterministically.
            "seen_hotkeys": sorted(queue.seen_hotkeys),
        },
        sort_keys=True,
    )


def loads(text: str) -> SubmissionQueue:
    obj = json.loads(text)
    return SubmissionQueue(
        pending=[
            QueuedSubmission(
                hotkey=str(p["hotkey"]),
                uid=int(p["uid"]),
                ref=str(p["ref"]),
                commit_block=int(p.get("commit_block", 0)),
            )
            for p in (obj.get("pending") or [])
        ],
        king_ref=obj.get("king_ref"),
        trained_refs=[str(c) for c in (obj.get("trained_refs") or [])],
        max_trained_cache=int(obj.get("max_trained_cache", 256)),
        seen_hotkeys={str(h) for h in (obj.get("seen_hotkeys") or [])},
    )
