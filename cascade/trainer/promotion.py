"""Trainer-side promotion engine — selection authority for Cascade warm-start.

Under propose-and-verify (DEC-CA-0012) the TRAINER selects which reign
checkpoints become the next warm-start generation: it authors the bench numbers
selection runs on, trains every model, and already declares each round's init in
the signed manifest — fleet re-derivation only ever re-checked the trainer's
arithmetic on the trainer's own data. Validators verify the declaration against
a small envelope (:mod:`cascade.validator.loop`); the selection policy here is
therefore trainer policy, free to evolve without fleet lockstep — as long as
every selected member stays inside the envelope (a benched reign checkpoint
within ``cascade_quality_epsilon`` of the reign's best, at most
``cascade_top_k`` members, promoted only on a ripe reign clock).

The v1 policy is structural diversity over a quality-gated candidate pool
(DEC-CA-0012 discussion): the pool is every benched duel checkpoint of the reign
— the king's AND the challengers' (different generators are genuinely different
data distributions, the deepest diversity available; the checkpoint's owner
earns NOTHING from promotion, by design) — the geomean-best checkpoint anchors
the set, and remaining slots greedily prefer a *different generator* first, then
*maximal round spacing* (reign checkpoints are same-init same-step siblings, so
spacing diversifies the data regime they trained on, not depth). Fancier
policies (eval-profile dispersion, per-window error decorrelation) can replace
this without touching consensus.

Per-round allocation across the live members is likewise policy: v1 rotates
deterministically by epoch index. Validators accept ANY live member, so
adaptive allocation (dropping a losing lineage mid-generation) is a pure
engine change.

The engine keys its reign clock off the on-chain king the trainer already
trains as king (``highest_incentive_hotkey``), persists across restarts, and
grandfathers a pre-DEC-CA-0012 single-pointer install as generation 1.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..shared.promotion import PromotedMember, PromotionRecord
from ..validator.cascade import CascadeState as _ClockState
from ..validator.cascade import cascade_score, reign_rounds

log = logging.getLogger("cascade.trainer.promotion")


def _member_to_json(m: PromotedMember) -> dict:
    return {"checkpoint_id": m.checkpoint_id, "size": m.size,
            "source_round": m.source_round, "score": m.score}


def _member_from_json(m: dict) -> PromotedMember:
    return PromotedMember(
        checkpoint_id=str(m["checkpoint_id"]), size=str(m.get("size", "")),
        source_round=str(m.get("source_round", "")),
        score=float(m.get("score", float("nan"))),
    )


@dataclass(frozen=True)
class Candidate:
    """One benched duel checkpoint, eligible for promotion selection."""

    checkpoint_id: str
    size: str
    hotkey: str
    role: str
    round_id: str
    epoch_index: int
    score: float


def select_members(
    candidates: list[Candidate],
    *,
    k_max: int,
    quality_epsilon: float,
    min_round_spacing: int = 1,
) -> list[Candidate]:
    """The v1 selection policy: quality gate, then structural diversity.

    Eligible = candidates whose score sits within ``(1 + quality_epsilon)`` of
    the pool's best (lower is better) — diversity is only ever arbitrated
    WITHIN the near-frontier set, never against it. The geomean-best candidate
    anchors the set (top-k strictly contains top-1); each remaining slot
    greedily picks the eligible candidate that (a) satisfies
    ``min_round_spacing`` from every already-selected candidate of the SAME
    generator — same-generator reign checkpoints are same-init same-step
    siblings, so adjacent rounds are near-duplicates, while a different
    generator's checkpoint from the very same round is genuinely different
    data — preferring (b) a generator hotkey not yet in the set, then (c)
    maximal minimum round distance, then (d) score, with ``checkpoint_id`` as
    the final deterministic tie-break. Returns fewer than ``k_max`` when the
    eligible pool can't fill the slots — the set is never padded with worse or
    adjacent checkpoints.
    """
    if not candidates or k_max < 1:
        return []
    best = min(c.score for c in candidates)
    floor = best * (1.0 + float(quality_epsilon))
    eligible = sorted(
        (c for c in candidates if c.score <= floor),
        key=lambda c: (c.score, c.checkpoint_id),
    )
    chosen = [eligible[0]]

    def _spaced(c: Candidate) -> bool:
        same = [s for s in chosen if s.hotkey == c.hotkey]
        return all(abs(c.epoch_index - s.epoch_index) >= int(min_round_spacing)
                   for s in same)

    while len(chosen) < int(k_max):
        taken = {s.checkpoint_id for s in chosen}
        pool = [c for c in eligible if c.checkpoint_id not in taken and _spaced(c)]
        if not pool:
            break
        def _rank(c: Candidate):
            new_generator = all(c.hotkey != s.hotkey for s in chosen)
            spacing = min(abs(c.epoch_index - s.epoch_index) for s in chosen)
            return (0 if new_generator else 1, -spacing, c.score, c.checkpoint_id)
        chosen.append(min(pool, key=_rank))
    return chosen


@dataclass
class TrainerPromotion:
    """The engine: reign clock + candidate log + live member set, persisted.

    ``reign_threshold`` is ripeness in ROUNDS (``[scoring]
    cascade_reign_rounds``); ``k_max``/``quality_epsilon`` mirror the fleet's
    envelope knobs (``cascade_top_k`` / ``cascade_quality_epsilon``) — the
    engine must select inside the envelope validators verify. ``pointer_path``
    is the warm-start pointer file the training loop reads
    (:meth:`TrainerRunner._load_warm_start`); ``state_path`` persists the
    engine across restarts. ``round_cfg`` is the RoundConfig the clock divides
    by. Thread-safe: ``record_bench`` runs on the post-publish bench thread.
    """

    reign_threshold: float
    k_max: int
    quality_epsilon: float
    min_round_spacing: int = 1
    state_path: Path | None = None
    pointer_path: Path | None = None
    round_cfg: object | None = None

    generation: int = 0
    members: tuple[PromotedMember, ...] = ()
    king_hotkey: str | None = None
    reign_start_block: int | None = None
    candidates: tuple[Candidate, ...] = ()
    # The last fired promotion's record, held (and persisted) until the caller
    # confirms it published: state advances at fire time, so a publish failure
    # must be retried from here — losing the record would leave the fleet with
    # no way to verify the generation the pointer file already rotates over.
    pending_record: PromotionRecord | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # ── lifecycle ────────────────────────────────────────────────────────────

    @classmethod
    def load(
        cls,
        *,
        reign_threshold: float,
        k_max: int,
        quality_epsilon: float,
        state_path: Path,
        pointer_path: Path,
        round_cfg: object | None = None,
        min_round_spacing: int = 1,
    ) -> TrainerPromotion:
        """Restore the engine from ``state_path`` (fresh when absent/corrupt),
        then grandfather a pre-DEC-CA-0012 pointer file — the single winner the
        old validator-side Cascade installed — as generation 1, so an armed
        deployment upgrades without a round of ``warm_start_mismatch``."""
        engine = cls(
            reign_threshold=reign_threshold, k_max=k_max,
            quality_epsilon=quality_epsilon, min_round_spacing=min_round_spacing,
            state_path=state_path, pointer_path=pointer_path, round_cfg=round_cfg,
        )
        if state_path.is_file():
            try:
                engine._restore(json.loads(state_path.read_text(encoding="utf-8")))
            except Exception as e:  # noqa: BLE001 — corrupt state ⇒ fresh engine
                log.warning("trainer promotion state %s unreadable (%s); starting fresh",
                            state_path, e)
        engine._adopt_legacy_pointer()
        return engine

    def _restore(self, obj: dict) -> None:
        self.generation = int(obj.get("generation", 0) or 0)
        self.members = tuple(
            _member_from_json(m) for m in (obj.get("members") or ())
        )
        self.king_hotkey = obj.get("king_hotkey") or None
        rsb = obj.get("reign_start_block")
        self.reign_start_block = None if rsb is None else int(rsb)
        self.candidates = tuple(
            Candidate(
                checkpoint_id=str(c["checkpoint_id"]), size=str(c.get("size", "")),
                hotkey=str(c.get("hotkey", "")), role=str(c.get("role", "")),
                round_id=str(c.get("round_id", "")),
                epoch_index=int(c.get("epoch_index", 0)),
                score=float(c["score"]),
            )
            for c in (obj.get("candidates") or ())
        )
        pr = obj.get("pending_record")
        if pr:
            self.pending_record = PromotionRecord(
                generation=int(pr["generation"]),
                king_hotkey=str(pr.get("king_hotkey", "")),
                fired_round=str(pr.get("fired_round", "")),
                fired_block=int(pr.get("fired_block", 0)),
                members=tuple(_member_from_json(m) for m in (pr.get("members") or ())),
            )

    def _adopt_legacy_pointer(self) -> None:
        """Grandfather a pre-existing pointer file when the engine has no state
        of its own: the pre-DEC-CA-0012 single winner becomes generation 1, and
        a member-set file (this engine's own schema — the state file was lost
        or corrupted while the pointer survived) is re-adopted at its recorded
        generation, so a state-file loss degrades to a resumable engine rather
        than one that rejects every candidate forever."""
        if self.generation != 0 or self.pointer_path is None or not self.pointer_path.is_file():
            return
        try:
            obj = json.loads(self.pointer_path.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001 — a broken legacy file is surfaced, not adopted
            log.warning("legacy warm-start pointer %s unreadable (%s); NOT adopted — "
                        "the training loop will fail loud on it", self.pointer_path, e)
            return
        members = [m for m in (obj.get("members") or ()) if m.get("checkpoint_id")]
        if members:
            self.generation = max(1, int(obj.get("generation", 1) or 1))
            self.members = tuple(_member_from_json(m) for m in members)
            self._persist()
            log.info("trainer promotion: re-adopted member-set pointer file as "
                     "generation %d (%d member(s); engine state was missing)",
                     self.generation, len(self.members))
            return
        cid = str(obj.get("checkpoint_id") or "")
        if not cid:
            return
        self.generation = 1
        self.members = (PromotedMember(
            checkpoint_id=cid, size=str(obj.get("size", "")), source_round="",
            score=float(obj.get("score") or float("nan"))),)
        self._persist()
        self._write_pointer()
        log.info("trainer promotion: adopted legacy warm-start pointer %s as "
                 "generation 1", cid)

    # ── per-round hooks ──────────────────────────────────────────────────────

    def note_round(self, king_hotkey: str | None, *, epoch_block: int) -> None:
        """Track the reign at each round boundary: a king change (the trainer's
        view — on-chain highest incentive, the same hotkey it trains as king)
        resets the clock and clears the candidate log; an unanchored clock is
        anchored here. The live member set carries over either way — a
        promotion outlives the reign that produced it."""
        with self._lock:
            if king_hotkey != self.king_hotkey:
                self.king_hotkey = king_hotkey
                self.reign_start_block = int(epoch_block)
                self.candidates = ()
                log.info("trainer promotion: reign clock reset for king %s at block %d",
                         (king_hotkey or "?")[:12], int(epoch_block))
            elif self.reign_start_block is None and king_hotkey is not None:
                self.reign_start_block = int(epoch_block)
            self._persist()

    def record_bench(self, manifest: object, report: object) -> int:
        """Log a round's benched duel checkpoints as promotion candidates.
        Called from the post-publish bench thread with the round's manifest and
        its published bench report. Admissible only when the round trained from
        the CURRENT generation (its ``warm_start_ckpt`` is a live member, or
        random init in the random-init era) — a late report from a previous
        generation's round must not seed the new one. Returns how many
        candidates were added."""
        ws = str(getattr(manifest, "warm_start_ckpt", "") or "")
        with self._lock:
            live = {m.checkpoint_id for m in self.members}
            if ws not in live and not (ws == "" and self.generation == 0):
                log.info("trainer promotion: round=%s trained from %r (not the live "
                         "generation); its bench contributes no candidates",
                         getattr(manifest, "round_id", "?"), ws or "<random init>")
                return 0
            epoch_index = self._epoch_index(int(getattr(report, "created_block", 0)))
            known = {c.checkpoint_id for c in self.candidates}
            added = 0
            for e in getattr(report, "entries", ()):
                if e.trained_pointer in known:
                    continue
                s = e.scores
                self.candidates = (*self.candidates, Candidate(
                    checkpoint_id=e.trained_pointer, size=e.size,
                    hotkey=e.miner_hotkey, role=e.role,
                    round_id=str(getattr(report, "round_id", "")),
                    epoch_index=epoch_index,
                    score=cascade_score(
                        s.gifteval_crps, s.gifteval_mase, s.boom_crps,
                        s.boom_mase, s.time_crps, s.time_mase),
                ))
                known.add(e.trained_pointer)
                added += 1
            if added:
                self._persist()
                log.info("trainer promotion: recorded %d candidate(s) from round=%s; "
                         "%d this reign", added, getattr(report, "round_id", "?"),
                         len(self.candidates))
            return added

    def maybe_promote(self, *, epoch_block: int, round_id: str) -> PromotionRecord | None:
        """Fire a promotion when the reign clock is ripe and the reign has
        candidates: select the member set, advance the generation, reset the
        clock (the king persists — DEC-CA-0004), clear the candidate log, and
        write the pointer file. The record is retained (persisted) as
        :attr:`pending_record` until :meth:`mark_record_published` — state
        advances at fire time, so the caller retries the publish from
        :meth:`unpublished_record` every round until it lands."""
        with self._lock:
            if self.king_hotkey is None or self.reign_start_block is None:
                return None
            clock = _ClockState(king_hotkey=self.king_hotkey,
                                reign_start_block=self.reign_start_block)
            elapsed = reign_rounds(clock, int(epoch_block), self.round_cfg)
            if elapsed is None or elapsed < float(self.reign_threshold):
                return None
            if not self.candidates:
                log.warning("trainer promotion: clock ripe (%.2f ≥ %.2f rounds) but no "
                            "benched candidate this reign; holding", elapsed,
                            float(self.reign_threshold))
                return None
            selected = select_members(
                list(self.candidates), k_max=self.k_max,
                quality_epsilon=self.quality_epsilon,
                min_round_spacing=self.min_round_spacing,
            )
            self.generation += 1
            self.members = tuple(
                PromotedMember(checkpoint_id=c.checkpoint_id, size=c.size,
                               source_round=c.round_id, score=c.score)
                for c in selected
            )
            self.candidates = ()
            self.reign_start_block = int(epoch_block)
            record = PromotionRecord(
                generation=self.generation,
                king_hotkey=self.king_hotkey or "",
                fired_round=str(round_id),
                fired_block=int(epoch_block),
                members=self.members,
            )
            self.pending_record = record
            self._persist()
            self._write_pointer()
            log.info(
                "PROMOTION fired: generation=%d reign=%.2f rounds members=[%s]; "
                "king %s persists, reign clock reset",
                self.generation, elapsed,
                ", ".join(f"{m.checkpoint_id} ({m.score:.5f})" for m in self.members),
                (self.king_hotkey or "?")[:12],
            )
            return record

    def unpublished_record(self) -> PromotionRecord | None:
        """The fired-but-unpublished promotion record, or ``None``. The caller
        publishes it (signed) and confirms with :meth:`mark_record_published`;
        until then it survives restarts and is re-offered every round."""
        with self._lock:
            return self.pending_record

    def mark_record_published(self) -> None:
        with self._lock:
            self.pending_record = None
            self._persist()

    def init_for_epoch(self, epoch_index: int) -> tuple[str, str] | None:
        """The member this epoch's round trains from — deterministic rotation
        (v1 allocation policy; validators accept any live member, so this can
        become adaptive without touching consensus). ``None`` in the
        random-init era."""
        with self._lock:
            if not self.members:
                return None
            m = self.members[int(epoch_index) % len(self.members)]
            return m.checkpoint_id, m.size

    # ── persistence ──────────────────────────────────────────────────────────

    def _epoch_index(self, block: int) -> int:
        if self.round_cfg is None:
            return int(block) // 7_200
        from ..shared.config import effective_epoch_blocks

        return int(block) // effective_epoch_blocks(self.round_cfg, int(block))

    def _persist(self) -> None:
        if self.state_path is None:
            return
        body = {
            "generation": self.generation,
            "members": [_member_to_json(m) for m in self.members],
            "king_hotkey": self.king_hotkey,
            "reign_start_block": self.reign_start_block,
            "candidates": [
                {"checkpoint_id": c.checkpoint_id, "size": c.size, "hotkey": c.hotkey,
                 "role": c.role, "round_id": c.round_id,
                 "epoch_index": c.epoch_index, "score": c.score}
                for c in self.candidates
            ],
            "pending_record": None if self.pending_record is None else {
                "generation": self.pending_record.generation,
                "king_hotkey": self.pending_record.king_hotkey,
                "fired_round": self.pending_record.fired_round,
                "fired_block": self.pending_record.fired_block,
                "members": [_member_to_json(m) for m in self.pending_record.members],
            },
        }
        # Atomic (tmp + rename): a crash mid-write must not corrupt the state
        # file — a corrupted file restarts the engine at generation 0.
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
            tmp.write_text(json.dumps(body, sort_keys=True), encoding="utf-8")
            tmp.replace(self.state_path)
        except Exception as e:  # noqa: BLE001 — persistence must never abort a round
            log.warning("trainer promotion: failed to persist state to %s: %s",
                        self.state_path, e)

    def _write_pointer(self) -> None:
        """Write the warm-start pointer file the training loop reads: the live
        member set plus a legacy single-pointer mirror (``checkpoint_id`` /
        ``size`` = the first member) so pre-multi-member readers — including a
        co-hosted validator's migration shim — stay compatible."""
        if self.pointer_path is None or not self.members:
            return
        first = self.members[0]
        body = {
            "generation": self.generation,
            "selection": "epoch_rotation",
            "members": [
                {"checkpoint_id": m.checkpoint_id, "size": m.size,
                 "source_round": m.source_round, "score": m.score}
                for m in self.members
            ],
            "checkpoint_id": first.checkpoint_id,
            "size": first.size,
            "installed_at": time.time(),
        }
        try:
            self.pointer_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.pointer_path.with_suffix(self.pointer_path.suffix + ".tmp")
            tmp.write_text(json.dumps(body, sort_keys=True), encoding="utf-8")
            tmp.replace(self.pointer_path)
            log.info("trainer promotion: warm-start pointer written to %s "
                     "(generation %d, %d member(s))",
                     self.pointer_path, self.generation, len(self.members))
        except Exception as e:  # noqa: BLE001
            log.warning("trainer promotion: failed to write pointer file %s: %s",
                        self.pointer_path, e)
