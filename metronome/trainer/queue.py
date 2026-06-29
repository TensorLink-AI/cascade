"""Trainer-side submission queue with cheap anti-duplicate checks.

metronome trains the king and a challenger together under one shared
:class:`~metronome.trainer.contract.RoundSeeds` — a single round is the
expensive resource (~3h of GPU per generator). This module sits in front of that
expense as a FIFO backlog of *challenger* generators discovered on-chain, with a
handful of O(1)/O(n) checks that keep guaranteed-wasted runs off the GPU. It is
the metronome analogue of teutonic's validator queue (FIFO drain + copy gate),
adapted to metronome's content-addressed submissions.

The headline check is **duplicate-of-king**: because a generator is pinned by its
content CID, a challenger whose generator is byte-identical to the reigning
king's has the *same CID*. Training it would burn a full controlled-experiment
round on a generator that can only tie the king (a challenger must clear the win
margin to dethrone — an identical run never will), so it is dropped for free with
a CID equality test, never fetched, never run.

Three more cheap checks ride along at :meth:`SubmissionQueue.enqueue`:

* **already-queued** — the same CID is already waiting (idempotent re-discovery
  of an unchanged commitment across polls).
* **already-trained** — the CID was already trained *this reign*. The cache
  resets when the king changes (:meth:`note_king`), mirroring teutonic's
  per-cycle ``evaluated_repos``: under single-round dethroning (the shipped
  ``[scoring] dethrone_cp = 1``) a challenger that has had its one fair shot
  against the current king should not be re-trained every round until the throne
  turns over.
* **latest-commit-wins** — a new CID from a hotkey already in the queue
  supersedes that hotkey's older queued entry (miners re-deploy by committing a
  new CID), matching :func:`metronome.trainer.loop.resolve_commitments`.

The queue is a small JSON-serialisable record so the trainer can persist it to
``[queue] state_db_path`` and keep the backlog (and per-reign dedup) across
restarts, the same way the validator persists its champion state.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass(frozen=True)
class QueuedSubmission:
    """One challenger generator waiting for a training round.

    ``cid`` is the generator's Hippius registry CID (content hash); it is the key
    every dedup check turns on. ``commit_block`` is the chain block the miner
    revealed the commitment at (FIFO ordering uses arrival into the queue, not
    this block, so a late re-deploy does not jump the line)."""

    hotkey: str
    uid: int
    cid: str
    commit_block: int


# Skip-reason strings returned by enqueue() (None ⇒ accepted). Kept as constants
# so callers/tests can match on them without hardcoding the message text.
SKIP_DUPLICATE_OF_KING = "duplicate_of_king"
SKIP_ALREADY_QUEUED = "already_queued"
SKIP_ALREADY_TRAINED = "already_trained"


@dataclass
class SubmissionQueue:
    """FIFO backlog of challenger submissions with per-reign duplicate dedup.

    Not frozen: enqueue/select/mark_trained mutate ``pending`` and the trained
    cache in place. Use :func:`dumps`/:func:`loads` to persist.

    Attributes:
        pending: challengers awaiting training, oldest-first (FIFO).
        king_cid: the reigning king's generator CID the ``trained_cids`` cache is
            scoped to. When :meth:`note_king` sees a different CID the cache is
            cleared (a new reign — every challenger deserves a fresh shot).
        trained_cids: CIDs already trained during the current reign, most-recent
            last, capped at ``max_trained_cache`` (a ring buffer).
        max_trained_cache: cap on ``trained_cids`` (``[queue] trained_cache_size``).
    """

    pending: list[QueuedSubmission] = field(default_factory=list)
    king_cid: str | None = None
    trained_cids: list[str] = field(default_factory=list)
    max_trained_cache: int = 256

    # ── reign tracking ───────────────────────────────────────────────────────

    def note_king(self, king_cid: str | None) -> bool:
        """Record the reigning king's CID; reset the per-reign dedup on a change.

        Returns True if the reign turned over (the king CID changed), so a caller
        can log it. Also drops any pending entry that now *is* the king (a queued
        challenger that won the throne, or a copy of the new king). The cache is
        only reset on a genuine change to a non-None CID, so a transient missing
        king does not wipe the dedup mid-reign.
        """
        changed = king_cid is not None and king_cid != self.king_cid
        if changed:
            self.king_cid = king_cid
            self.trained_cids = []
        if king_cid is not None:
            self.pending = [q for q in self.pending if q.cid != king_cid]
        return changed

    # ── intake ───────────────────────────────────────────────────────────────

    def enqueue(self, sub: QueuedSubmission) -> str | None:
        """Add a challenger to the backlog, or return why it was skipped.

        Returns None when the submission was accepted (now pending), otherwise one
        of the ``SKIP_*`` reason strings. Cheap by construction: a CID equality
        test against the king, a membership test against the per-reign trained
        cache, and a linear scan of ``pending`` (which holds at most one entry per
        hotkey).
        """
        if self.king_cid is not None and sub.cid == self.king_cid:
            return SKIP_DUPLICATE_OF_KING
        if sub.cid in self.trained_cids:
            return SKIP_ALREADY_TRAINED
        if any(q.cid == sub.cid for q in self.pending):
            return SKIP_ALREADY_QUEUED
        # latest-commit-wins: a fresh CID from a hotkey already in the queue
        # supersedes that hotkey's older entry (the miner re-deployed).
        self.pending = [q for q in self.pending if q.hotkey != sub.hotkey]
        self.pending.append(sub)
        return None

    def prune_to_field(self, field_cids: set[str]) -> list[QueuedSubmission]:
        """Drop pending entries whose CID is no longer in the resolved on-chain
        field (the miner re-deployed to a new CID, or deregistered). Returns the
        removed entries so a caller can log them."""
        keep, drop = [], []
        for q in self.pending:
            (keep if q.cid in field_cids else drop).append(q)
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
            if self.king_cid is not None and q.cid == self.king_cid:
                continue
            if q.cid in self.trained_cids:
                continue
            out.append(q)
        return out

    def mark_trained(self, cid: str) -> None:
        """Record a CID as trained this reign and remove it from the backlog.

        Called once per challenger the trainer *attempted* this round (win, loss,
        or a generator that failed to train) — an attempt consumes the challenger's
        shot for this reign, so a broken or losing generator is not re-run every
        round. The cache is a bounded ring buffer."""
        self.pending = [q for q in self.pending if q.cid != cid]
        if cid in self.trained_cids:
            return
        self.trained_cids.append(cid)
        if len(self.trained_cids) > self.max_trained_cache:
            self.trained_cids = self.trained_cids[-self.max_trained_cache :]


def dumps(queue: SubmissionQueue) -> str:
    return json.dumps(
        {
            "pending": [
                {"hotkey": q.hotkey, "uid": q.uid, "cid": q.cid, "commit_block": q.commit_block}
                for q in queue.pending
            ],
            "king_cid": queue.king_cid,
            "trained_cids": list(queue.trained_cids),
            "max_trained_cache": queue.max_trained_cache,
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
                cid=str(p["cid"]),
                commit_block=int(p.get("commit_block", 0)),
            )
            for p in (obj.get("pending") or [])
        ],
        king_cid=obj.get("king_cid"),
        trained_cids=[str(c) for c in (obj.get("trained_cids") or [])],
        max_trained_cache=int(obj.get("max_trained_cache", 256)),
    )
