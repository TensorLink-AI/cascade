"""Submission queue + cheap anti-duplicate checks (pure, no GPU/chain/Hippius).

Covers the standalone :mod:`metronome.trainer.queue` (FIFO, dup-of-king,
already-queued, already-trained, per-reign reset, persistence) and the
duplicate-of-king / same-CID filters baked into
:func:`metronome.trainer.loop.plan_round`.
"""

from __future__ import annotations

from metronome.trainer.loop import ResolvedGenerator, plan_round
from metronome.trainer.queue import (
    SKIP_ALREADY_QUEUED,
    SKIP_ALREADY_TRAINED,
    SKIP_DUPLICATE_OF_KING,
    QueuedSubmission,
    SubmissionQueue,
)
from metronome.trainer.queue import dumps as dump_queue
from metronome.trainer.queue import loads as load_queue

KING = "cidKING"
A, B, C = "cidA", "cidB", "cidC"


def _sub(hotkey: str, uid: int, cid: str, block: int = 1) -> QueuedSubmission:
    return QueuedSubmission(hotkey=hotkey, uid=uid, cid=cid, commit_block=block)


# ── plan_round filters ────────────────────────────────────────────────────────


def test_plan_round_drops_challenger_identical_to_king():
    # uid 2 committed the king's exact generator CID — a copy; it must not be
    # planned as a challenger (it could only tie, never dethrone).
    resolved = [
        ResolvedGenerator(hotkey="k", uid=0, cid=KING),
        ResolvedGenerator(hotkey="copy", uid=2, cid=KING),
        ResolvedGenerator(hotkey="real", uid=1, cid=A),
    ]
    plan = plan_round(resolved, king_hotkey="k")
    assert plan.king.hotkey == "k"
    assert [c.hotkey for c in plan.challengers] == ["real"]


def test_plan_round_dedups_challengers_sharing_a_cid():
    # Two miners committed the same generator CID — train it once (lowest UID).
    resolved = [
        ResolvedGenerator(hotkey="k", uid=0, cid=KING),
        ResolvedGenerator(hotkey="late", uid=5, cid=A),
        ResolvedGenerator(hotkey="early", uid=1, cid=A),
    ]
    plan = plan_round(resolved, king_hotkey="k")
    assert [c.hotkey for c in plan.challengers] == ["early"]


def test_plan_round_interim_king_when_king_absent():
    # No reigning king present ⇒ lowest-UID generator is the interim king and is
    # not also a challenger.
    resolved = [
        ResolvedGenerator(hotkey="b", uid=3, cid=B),
        ResolvedGenerator(hotkey="a", uid=1, cid=A),
    ]
    plan = plan_round(resolved, king_hotkey=None)
    assert plan.king.hotkey == "a"
    assert [c.hotkey for c in plan.challengers] == ["b"]


# ── enqueue cheap checks ──────────────────────────────────────────────────────


def test_enqueue_accepts_then_rejects_duplicate_cid():
    q = SubmissionQueue()
    assert q.enqueue(_sub("a", 1, A)) is None
    # same CID again (idempotent re-discovery) is rejected
    assert q.enqueue(_sub("a", 1, A)) == SKIP_ALREADY_QUEUED
    assert [s.cid for s in q.pending] == [A]


def test_enqueue_rejects_duplicate_of_king():
    q = SubmissionQueue()
    q.note_king(KING)
    assert q.enqueue(_sub("copy", 9, KING)) == SKIP_DUPLICATE_OF_KING
    assert q.pending == []


def test_enqueue_rejects_already_trained_this_reign():
    q = SubmissionQueue()
    q.note_king(KING)
    assert q.enqueue(_sub("a", 1, A)) is None
    q.mark_trained(A)
    # A had its shot this reign; re-discovery is skipped and it leaves the backlog
    assert q.enqueue(_sub("a", 1, A)) == SKIP_ALREADY_TRAINED
    assert q.pending == []


def test_latest_commit_supersedes_same_hotkey():
    q = SubmissionQueue()
    assert q.enqueue(_sub("a", 1, A, block=1)) is None
    assert q.enqueue(_sub("a", 1, B, block=2)) is None  # a re-deployed to CID B
    assert [s.cid for s in q.pending] == [B]


# ── FIFO selection / completion ───────────────────────────────────────────────


def test_select_is_fifo_and_non_destructive():
    q = SubmissionQueue()
    for hk, cid in [("a", A), ("b", B), ("c", C)]:
        q.enqueue(_sub(hk, 1, cid))
    assert [s.cid for s in q.select(2)] == [A, B]
    # select does not remove — the same picks are still pending
    assert [s.cid for s in q.pending] == [A, B, C]
    q.mark_trained(A)
    assert [s.cid for s in q.select(2)] == [B, C]


def test_select_skips_king_and_trained_at_selection_time():
    q = SubmissionQueue()
    for hk, cid in [("a", A), ("b", B)]:
        q.enqueue(_sub(hk, 1, cid))
    # A became the king after it was queued — it must not be selected.
    q.king_cid = A
    assert [s.cid for s in q.select(2)] == [B]


def test_select_zero_or_negative_returns_empty():
    q = SubmissionQueue()
    q.enqueue(_sub("a", 1, A))
    assert q.select(0) == []
    assert q.select(-1) == []


# ── per-reign cache + pruning ─────────────────────────────────────────────────


def test_note_king_resets_trained_cache_on_new_reign():
    q = SubmissionQueue()
    q.note_king(KING)
    q.mark_trained(A)
    assert A in q.trained_cids
    changed = q.note_king("cidNEWKING")  # throne turned over
    assert changed is True
    assert q.trained_cids == []  # every challenger gets a fresh shot
    # same king CID again is a no-op (cache preserved)
    q.mark_trained(B)
    assert q.note_king("cidNEWKING") is False
    assert B in q.trained_cids


def test_note_king_removes_pending_copy_of_new_king():
    q = SubmissionQueue()
    q.enqueue(_sub("a", 1, A))
    q.enqueue(_sub("b", 2, B))
    q.note_king(A)  # the queued challenger A just won the throne
    assert [s.cid for s in q.pending] == [B]


def test_prune_to_field_drops_redeployed_cids():
    q = SubmissionQueue()
    q.enqueue(_sub("a", 1, A))
    q.enqueue(_sub("b", 2, B))
    dropped = q.prune_to_field({B})  # A no longer in the on-chain field
    assert [d.cid for d in dropped] == [A]
    assert [s.cid for s in q.pending] == [B]


def test_trained_cache_is_bounded_ring_buffer():
    q = SubmissionQueue(max_trained_cache=3)
    for i in range(5):
        q.mark_trained(f"cid{i}")
    assert q.trained_cids == ["cid2", "cid3", "cid4"]


# ── persistence round-trip ────────────────────────────────────────────────────


def test_dumps_loads_round_trip():
    q = SubmissionQueue(max_trained_cache=7)
    q.note_king(KING)
    q.enqueue(_sub("a", 1, A, block=10))
    q.enqueue(_sub("b", 2, B, block=11))
    q.mark_trained(C)

    back = load_queue(dump_queue(q))
    assert back.king_cid == KING
    assert back.max_trained_cache == 7
    assert [(s.hotkey, s.uid, s.cid, s.commit_block) for s in back.pending] == [
        ("a", 1, A, 10),
        ("b", 2, B, 11),
    ]
    assert back.trained_cids == [C]


def test_loads_tolerates_empty_payload():
    q = load_queue("{}")
    assert q.pending == []
    assert q.king_cid is None
    assert q.trained_cids == []
