"""Trainer-side promotion engine — selection authority for Cascade warm-start.

Under propose-and-verify (DEC-CA-0013) the TRAINER selects which reign
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
(DEC-CA-0013 discussion): the pool is every benched duel checkpoint of the reign
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

ALL-TIME RULE (DEC-CA-0044, block-gated ``[scoring] cascade_alltime_from_block``):
from the activation block the member set is ONE fixed population — the
all-time top-k benched checkpoints across every reign and generation, ranked
by the suite-WEIGHTED score (GIFT-Eval : BOOM : TIME = 50 : 25 : 25 by
default; :func:`cascade.validator.cascade.weighted_cascade_score`), still
within ``cascade_quality_epsilon`` of the best. The leaderboard is a bounded
sorted list: a better checkpoint is inserted at its rank, the members below it
slide down one place and the last one drops out. A change to the member set is
never installed on the spot: the engine ANNOUNCES it (``pending_change``,
surfaced on the status docs and the public leaderboard doc) and fires the
signed record ``notice_blocks`` later (24h at 12 s/block), so miners get the
notice period to prepare against the exact announced checkpoint. The
announced set is FROZEN for the window — a better checkpoint arriving during
the notice waits for the following change — unless the frozen set fell outside
the envelope (a much better arrival moved the floor), in which case the
current target is re-announced. Rotation across members is unchanged. The
no-downgrade guard is implied: the leaderboard only ever improves.

The engine keys its reign clock off whatever king the runner resolves for it —
the signed receipt trail's verdict king when readable (prompt: validators reset
their clocks at the dethrone verdict), the on-chain incentive king as fallback
(it lags a dethrone by 1-2 epochs). It persists across restarts and
grandfathers a pre-DEC-CA-0013 pointer file (single winner OR member set) at
its recorded generation.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..shared.promotion import (
    PromotedMember,
    PromotionRecord,
    member_from_json,
    member_to_json,
)
from ..validator.cascade import (
    DEFAULT_SUITE_WEIGHTS,
    cascade_score,
    reign_rounds,
    weighted_cascade_score,
)
from ..validator.cascade import CascadeState as _ClockState

log = logging.getLogger("cascade.trainer.promotion")


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


@dataclass(frozen=True)
class LeaderEntry:
    """One row of the all-time leaderboard (DEC-CA-0044): a benched checkpoint
    with its six signed numbers and the suite-weighted ``score`` it is ranked
    on. ``source_round`` is the round whose signed bench report scored it —
    the provenance a validator re-reads. ``created_block`` is that report's
    manifest block (observability; ordering ties)."""

    checkpoint_id: str
    size: str
    hotkey: str
    role: str
    source_round: str
    created_block: int
    gifteval_crps: float
    gifteval_mase: float
    boom_crps: float
    boom_mase: float
    time_crps: float
    time_mase: float
    score: float

    def to_json(self) -> dict:
        return {
            "checkpoint_id": self.checkpoint_id, "size": self.size,
            "hotkey": self.hotkey, "role": self.role,
            "source_round": self.source_round, "created_block": self.created_block,
            "gifteval_crps": self.gifteval_crps, "gifteval_mase": self.gifteval_mase,
            "boom_crps": self.boom_crps, "boom_mase": self.boom_mase,
            "time_crps": self.time_crps, "time_mase": self.time_mase,
            "score": self.score,
        }

    @classmethod
    def from_json(cls, o: dict) -> LeaderEntry:
        return cls(
            checkpoint_id=str(o["checkpoint_id"]), size=str(o.get("size", "")),
            hotkey=str(o.get("hotkey", "")), role=str(o.get("role", "")),
            source_round=str(o.get("source_round", "")),
            created_block=int(o.get("created_block", 0) or 0),
            gifteval_crps=float(o["gifteval_crps"]), gifteval_mase=float(o["gifteval_mase"]),
            boom_crps=float(o["boom_crps"]), boom_mase=float(o["boom_mase"]),
            time_crps=float(o["time_crps"]), time_mase=float(o["time_mase"]),
            score=float(o["score"]),
        )

    def rescored(self, weights: tuple[float, float, float]) -> LeaderEntry:
        """The same row under different suite weights."""
        from dataclasses import replace

        return replace(self, score=weighted_cascade_score(
            self.gifteval_crps, self.gifteval_mase, self.boom_crps, self.boom_mase,
            self.time_crps, self.time_mase, weights=weights))


@dataclass(frozen=True)
class PendingChange:
    """An ANNOUNCED member-set change (DEC-CA-0044): the next generation's
    frozen member list, the round/block it was announced at, and the block it
    may fire at (``announced_block + notice_blocks``). Surfaced to miners the
    whole window; the signed record only publishes when it fires."""

    generation: int
    members: tuple[PromotedMember, ...]
    announced_round: str
    announced_block: int
    effective_block: int

    def to_json(self) -> dict:
        return {
            "generation": self.generation,
            "members": [member_to_json(m) for m in self.members],
            "announced_round": self.announced_round,
            "announced_block": self.announced_block,
            "effective_block": self.effective_block,
        }

    @classmethod
    def from_json(cls, o: dict) -> PendingChange:
        return cls(
            generation=int(o["generation"]),
            members=tuple(member_from_json(m) for m in (o.get("members") or ())),
            announced_round=str(o.get("announced_round", "")),
            announced_block=int(o.get("announced_block", 0) or 0),
            effective_block=int(o.get("effective_block", 0) or 0),
        )

    def member_ids(self) -> tuple[str, ...]:
        return tuple(m.checkpoint_id for m in self.members)


def admit_leader(
    board: tuple[LeaderEntry, ...], entry: LeaderEntry, k: int,
) -> tuple[tuple[LeaderEntry, ...], int | None]:
    """Insert ``entry`` into the rank-ordered all-time leaderboard (pure).

    Returns ``(new_board, rank)`` — ``rank`` is 1-based when the entry made
    the board, ``None`` when it did not (worse than every member of a full
    board, or already listed). Strictly-better only: an equal score never
    displaces the incumbent (the earlier checkpoint keeps its rank, so a
    re-bench of an identical artefact cannot churn the set). The board holds
    at most ``k`` rows: the member the newcomer outranks slides down one
    place, and the row pushed past ``k`` drops out — "if it beats 2nd, it
    takes 2nd and 2nd becomes 3rd". Non-finite scores never enter."""
    k = int(k)
    if k < 1 or not math.isfinite(entry.score):
        return board, None
    if any(e.checkpoint_id == entry.checkpoint_id for e in board):
        return board, None
    pos = len(board)
    for i, e in enumerate(board):
        if entry.score < e.score:
            pos = i
            break
    if pos >= k:
        return board, None
    new = (*board[:pos], entry, *board[pos:])
    return new[:k], pos + 1


def alltime_members(
    board: tuple[LeaderEntry, ...], *, k_max: int, quality_epsilon: float,
) -> list[LeaderEntry]:
    """The member set the all-time rule declares: the board's top ``k_max``
    rows that sit within ``(1 + quality_epsilon)`` of the best (the envelope
    validators verify — a 3rd-best that trails the best by more than the
    epsilon is not a legal member and is left out, never padded). Rank order
    is the rotation order."""
    if not board or k_max < 1:
        return []
    best = board[0].score
    floor = best * (1.0 + float(quality_epsilon))
    return [e for e in board[:int(k_max)] if e.score <= floor]


def error_correlations(
    vectors: dict[str, list[float]],
) -> dict[tuple[str, str], float]:
    """Pairwise Pearson correlation of per-window error RESIDUALS.

    ``vectors`` maps checkpoint_id → per-window error scores (same battery,
    same window order, all positive). Raw error vectors correlate near 1.0
    for ANY two competent models because shared window difficulty dominates
    (the same reason DEC-CA-0006 rejected UCB ranking), so each vector is
    log-transformed and centered PER WINDOW across the pool first — what is
    correlated is each checkpoint's relative strengths and weaknesses, the
    trajectory-diversity signal promotion wants. Pairs are keyed both ways;
    ids with mismatched lengths or degenerate residuals are simply absent.
    """
    ids = [i for i, v in vectors.items() if v]
    if len(ids) < 2:
        return {}
    n = min(len(vectors[i]) for i in ids)
    logs = {i: [math.log(max(float(x), 1e-12)) for x in vectors[i][:n]] for i in ids}
    col_mean = [sum(logs[i][w] for i in ids) / len(ids) for w in range(n)]
    resid = {i: [logs[i][w] - col_mean[w] for w in range(n)] for i in ids}
    out: dict[tuple[str, str], float] = {}
    for a_pos, a in enumerate(ids):
        for b in ids[a_pos + 1:]:
            ra, rb = resid[a], resid[b]
            ma, mb = sum(ra) / n, sum(rb) / n
            da, db = [x - ma for x in ra], [x - mb for x in rb]
            va = math.sqrt(sum(x * x for x in da))
            vb = math.sqrt(sum(x * x for x in db))
            if va <= 0.0 or vb <= 0.0:
                continue
            r = sum(x * y for x, y in zip(da, db, strict=True)) / (va * vb)
            out[(a, b)] = out[(b, a)] = r
    return out


def select_members(
    candidates: list[Candidate],
    *,
    k_max: int,
    quality_epsilon: float,
    min_round_spacing: int = 1,
    error_vectors: dict[str, list[float]] | None = None,
) -> list[Candidate]:
    """Selection policy: quality gate, then error-decorrelation diversity.

    Eligible = candidates whose score sits within ``(1 + quality_epsilon)`` of
    the pool's best (lower is better) — diversity is only ever arbitrated
    WITHIN the near-frontier set, never against it. The geomean-best candidate
    anchors the set (top-k strictly contains top-1); each remaining slot
    greedily picks the eligible candidate that satisfies ``min_round_spacing``
    from every already-selected candidate of the SAME generator (same-generator
    adjacent reign checkpoints are same-init same-step near-duplicates).

    Among spaced candidates the slot goes to, in order:

    * When ``error_vectors`` covers the candidate AND at least one chosen
      member: the candidate whose **maximum error correlation** against the
      chosen set is lowest (see :func:`error_correlations`) — trajectory
      diversity measured on errors, not inferred from structure. Ties break
      by score then id.
    * Otherwise (no vectors supplied, or this candidate/chosen pair not
      covered): the v1 structural policy — prefer a generator hotkey not yet
      in the set, then maximal minimum round distance, then score, then
      ``checkpoint_id``. Vector-covered candidates always outrank vectorless
      ones — measured diversity beats guessed diversity.

    Returns fewer than ``k_max`` when the eligible pool can't fill the slots —
    the set is never padded with worse or adjacent checkpoints.
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
    corr = error_correlations(
        {c.checkpoint_id: (error_vectors or {}).get(c.checkpoint_id) or []
         for c in eligible})

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
            pairs = [corr[(c.checkpoint_id, s.checkpoint_id)] for s in chosen
                     if (c.checkpoint_id, s.checkpoint_id) in corr]
            if pairs:  # measured trajectory diversity
                return (0, max(pairs), c.score, c.checkpoint_id)
            new_generator = all(c.hotkey != s.hotkey for s in chosen)
            spacing = min(abs(c.epoch_index - s.epoch_index) for s in chosen)
            return (1, 0.0 if new_generator else 1.0, -spacing, c.score)

        chosen.append(min(pool, key=_rank))
    return chosen


def reign_tail(
    rows: list, validator_hotkey: str = "",
) -> tuple[str, int, list[str]] | None:
    """The current reign per a receipts-index ``rounds`` list: ``(king_hotkey,
    reign_start_block, round_ids oldest→newest)`` — the unbroken tail of scored
    rounds whose ``post_round_king_hotkey`` is the newest round's king. Pure.

    This is the deploy-time backfill's view of "how long has this king already
    reigned": validators anchor their clocks at the dethrone verdict, and the
    dethrone round is exactly where the tail breaks. Rows from other validators
    are ignored when ``validator_hotkey`` is given (the index carries one row
    per validator per round); rejected/incomplete rows never carry a
    ``post_round_king_hotkey`` and are skipped. ``None`` when no usable row
    exists — the caller falls back to anchoring at the next boundary.
    """
    usable = [
        r for r in rows
        if isinstance(r, dict) and r.get("status") == "scored"
        and r.get("post_round_king_hotkey")
        and int(r.get("epoch_start_block") or 0) > 0
        and (not validator_hotkey or r.get("validator_hotkey") == validator_hotkey)
    ]
    usable.sort(key=lambda r: int(r["epoch_start_block"]), reverse=True)
    if not usable:
        return None
    king = str(usable[0]["post_round_king_hotkey"])
    tail = []
    for r in usable:
        if str(r["post_round_king_hotkey"]) != king:
            break
        tail.append(r)
    start = int(tail[-1]["epoch_start_block"])
    return king, start, [str(r.get("round_id") or "") for r in reversed(tail)]


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
    # Optional {checkpoint_id: [per-window error scores]} JSON cache feeding
    # select_members' error-decorrelation policy. Best-effort: absent/stale
    # entries just fall back to structural diversity for those candidates.
    error_vectors_path: Path | None = None
    # All-time rule (DEC-CA-0044): the activation block (0 = never), the
    # notice period in blocks between announcing a member-set change and
    # firing it, and the suite weights the leaderboard ranks on. The
    # leaderboard itself accumulates from every signed bench report the
    # engine sees, activation or not — at the block it is already populated.
    alltime_from_block: int = 0
    notice_blocks: int = 7200
    suite_weights: tuple[float, float, float] = DEFAULT_SUITE_WEIGHTS

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
    # All-time leaderboard state: the rank-ordered top-k population, the
    # announced-but-not-fired change, and whether the deploy-time history
    # backfill (every published bench report, not just the reign's) has run.
    leaderboard: tuple[LeaderEntry, ...] = ()
    pending_change: PendingChange | None = None
    leaderboard_seeded: bool = False
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
        error_vectors_path: Path | None = None,
        alltime_from_block: int = 0,
        notice_blocks: int = 7200,
        suite_weights: tuple[float, float, float] = DEFAULT_SUITE_WEIGHTS,
    ) -> TrainerPromotion:
        """Restore the engine from ``state_path`` (fresh when absent/corrupt),
        then grandfather a pre-DEC-CA-0013 pointer file — the single winner the
        old validator-side Cascade installed — as generation 1, so an armed
        deployment upgrades without a round of ``warm_start_mismatch``."""
        engine = cls(
            reign_threshold=reign_threshold, k_max=k_max,
            quality_epsilon=quality_epsilon, min_round_spacing=min_round_spacing,
            state_path=state_path, pointer_path=pointer_path, round_cfg=round_cfg,
            error_vectors_path=error_vectors_path,
            alltime_from_block=int(alltime_from_block), notice_blocks=int(notice_blocks),
            suite_weights=tuple(suite_weights),
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
            member_from_json(m) for m in (obj.get("members") or ())
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
        board = []
        for o in obj.get("leaderboard") or ():
            try:
                board.append(LeaderEntry.from_json(o))
            except (KeyError, TypeError, ValueError):
                continue  # a malformed row is dropped, never the whole board
        # Re-rank under the CURRENT weights: a weight change in chain.toml
        # re-sorts the persisted population instead of freezing stale ranks.
        self.leaderboard = tuple(sorted(
            (e.rescored(self.suite_weights) for e in board),
            key=lambda e: (e.score, e.created_block, e.checkpoint_id)))
        pc = obj.get("pending_change")
        self.pending_change = PendingChange.from_json(pc) if pc else None
        self.leaderboard_seeded = bool(obj.get("leaderboard_seeded", False))
        pr = obj.get("pending_record")
        if pr:
            self.pending_record = PromotionRecord(
                generation=int(pr["generation"]),
                king_hotkey=str(pr.get("king_hotkey", "")),
                fired_round=str(pr.get("fired_round", "")),
                fired_block=int(pr.get("fired_block", 0)),
                members=tuple(member_from_json(m) for m in (pr.get("members") or ())),
            )

    def _adopt_legacy_pointer(self) -> None:
        """Grandfather a pre-existing pointer file when the engine has no state
        of its own: the pre-DEC-CA-0013 single winner becomes generation 1, and
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
            self.members = tuple(member_from_json(m) for m in members)
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

    def seed_reign(self, king_hotkey: str, reign_start_block: int) -> bool:
        """Anchor an engine that has never seen a king to an ALREADY-RUNNING
        reign (deploy-time backfill). A first deployment mid-reign must count
        the rounds the king has already survived: validators anchored their
        clocks at the dethrone verdict and keep counting across our restarts,
        so an engine that re-anchors "now" would fire the promotion
        ``reign_threshold`` rounds later than every validator expects — and
        the fleet, whose ripeness check uses ITS clock, would have accepted
        the earlier fire. No-op (``False``) once the engine has a king: an
        engine with history trusts its own persisted state, and
        :meth:`note_round` owns the clock from then on."""
        with self._lock:
            if self.king_hotkey is not None or self.reign_start_block is not None:
                return False
            self.king_hotkey = str(king_hotkey)
            self.reign_start_block = int(reign_start_block)
            self._persist()
            log.info("trainer promotion: seeded reign clock for king %s at block %d "
                     "(deploy-time backfill)", king_hotkey[:12], int(reign_start_block))
            return True

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
        # The all-time leaderboard admits every signed bench (any generation,
        # any reign): the population is all-time by definition.
        self.admit_report(report)
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

    def admit_report(self, report: object) -> int:
        """Offer every entry of a trainer-signed bench report to the all-time
        leaderboard (DEC-CA-0044). Returns how many entries made the board.
        Idempotent (a listed pointer is skipped) and reign-agnostic — the
        deploy-time history backfill and the per-round bench thread both
        feed it. Thread-safe."""
        with self._lock:
            k = max(1, int(self.k_max))
            block = int(getattr(report, "created_block", 0) or 0)
            admitted = 0
            for e in getattr(report, "entries", ()):
                sc = e.scores
                entry = LeaderEntry(
                    checkpoint_id=e.trained_pointer, size=e.size,
                    hotkey=e.miner_hotkey, role=e.role,
                    source_round=str(getattr(report, "round_id", "")),
                    created_block=block,
                    gifteval_crps=float(sc.gifteval_crps),
                    gifteval_mase=float(sc.gifteval_mase),
                    boom_crps=float(sc.boom_crps), boom_mase=float(sc.boom_mase),
                    time_crps=float(sc.time_crps), time_mase=float(sc.time_mase),
                    score=weighted_cascade_score(
                        sc.gifteval_crps, sc.gifteval_mase, sc.boom_crps,
                        sc.boom_mase, sc.time_crps, sc.time_mase,
                        weights=self.suite_weights),
                )
                self.leaderboard, rank = admit_leader(self.leaderboard, entry, k)
                if rank is not None:
                    admitted += 1
                    log.info("trainer promotion: all-time leaderboard #%d ← %s "
                             "(%.5f weighted, round=%s %s)", rank, entry.checkpoint_id,
                             entry.score, entry.source_round, entry.role or "duel")
            if admitted:
                self._persist()
            return admitted

    def mark_leaderboard_seeded(self) -> None:
        with self._lock:
            self.leaderboard_seeded = True
            self._persist()

    def alltime_active(self, block: int) -> bool:
        """Whether a round at ``block`` selects under the all-time rule."""
        return self.alltime_from_block > 0 and int(block) >= int(self.alltime_from_block)

    def upcoming(self) -> dict | None:
        """The announced-but-unfired member-set change as the status-doc
        ``warm_start.upcoming`` block, or ``None``."""
        with self._lock:
            return None if self.pending_change is None else self.pending_change.to_json()

    def leaderboard_rows(self) -> list[dict]:
        """The rank-ordered leaderboard as public-doc rows."""
        with self._lock:
            return [{"rank": i + 1, **e.to_json()} for i, e in enumerate(self.leaderboard)]

    def _maybe_promote_alltime(self, *, epoch_block: int, round_id: str) -> PromotionRecord | None:
        """The all-time rule's boundary step (caller holds the lock).

        Target = the leaderboard's top-k within the quality epsilon. When it
        differs from the live set and nothing is announced, ANNOUNCE it
        (effective ``notice_blocks`` later) and hold. When a change is
        announced and its effective block has arrived — and at least
        ``notice_blocks`` passed since the reign anchor (a dethrone during the
        window resets that, and the validators' spacing check mirrors it) —
        FIRE the frozen set. A frozen set that no longer sits inside the
        envelope (a much better arrival moved the floor) is re-announced as
        the current target rather than fired into a rejection."""
        target = alltime_members(self.leaderboard, k_max=self.k_max,
                                 quality_epsilon=self.quality_epsilon)
        if not target:
            return None
        target_ids = [e.checkpoint_id for e in target]
        live_ids = {m.checkpoint_id for m in self.members}
        pending = self.pending_change

        def _announce() -> None:
            self.pending_change = PendingChange(
                generation=self.generation + 1,
                members=tuple(PromotedMember(
                    checkpoint_id=e.checkpoint_id, size=e.size,
                    source_round=e.source_round, score=e.score) for e in target),
                announced_round=str(round_id), announced_block=int(epoch_block),
                effective_block=int(epoch_block) + int(self.notice_blocks),
            )
            self._persist()
            log.info("PROMOTION ANNOUNCED (all-time rule): generation %d will train from "
                     "[%s] from block %d (%d blocks' notice); announced at round=%s",
                     self.generation + 1,
                     ", ".join(f"{e.checkpoint_id} ({e.score:.5f})" for e in target),
                     int(epoch_block) + int(self.notice_blocks), int(self.notice_blocks),
                     round_id)

        if pending is None:
            if set(target_ids) == live_ids:
                return None
            _announce()
            return None
        if set(pending.member_ids()) == live_ids:
            # The announced set is already live (state restored from a pointer
            # the fire had written): nothing left to fire.
            self.pending_change = None
            self._persist()
            return None
        # Envelope re-check on the frozen set: every announced member must
        # still sit within epsilon of the CURRENT all-time best, or validators
        # reject the record on their (better) floor.
        best = self.leaderboard[0].score if self.leaderboard else float("nan")
        floor = best * (1.0 + float(self.quality_epsilon))
        by_id = {e.checkpoint_id: e for e in self.leaderboard}
        frozen_ok = all(
            (m.checkpoint_id in by_id and by_id[m.checkpoint_id].score <= floor)
            or (m.checkpoint_id not in by_id and math.isfinite(m.score) and m.score <= floor)
            for m in pending.members)
        if not frozen_ok:
            log.warning("trainer promotion: announced set [%s] fell outside the quality "
                        "envelope (all-time best now %.5f); re-announcing the current "
                        "target", ", ".join(pending.member_ids()), best)
            _announce()
            return None
        if int(epoch_block) < int(pending.effective_block):
            log.info("trainer promotion: announced generation %d takes effect at block %d "
                     "(%d blocks to go); holding", pending.generation,
                     pending.effective_block, int(pending.effective_block) - int(epoch_block))
            return None
        if (self.reign_start_block is not None
                and int(epoch_block) - int(self.reign_start_block) < int(self.notice_blocks)):
            log.info("trainer promotion: announced generation %d is due but the reign "
                     "anchor moved (block %d) inside the notice period; holding until "
                     "block %d", pending.generation, int(self.reign_start_block),
                     int(self.reign_start_block) + int(self.notice_blocks))
            return None
        self.generation += 1
        self.members = tuple(pending.members)
        self.candidates = ()
        self.pending_change = None
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
            "PROMOTION fired (all-time rule): generation=%d members=[%s]; announced at "
            "round=%s block %d; king %s persists, reign clock reset",
            self.generation,
            ", ".join(f"{m.checkpoint_id} ({m.score:.5f})" for m in self.members),
            pending.announced_round, pending.announced_block,
            (self.king_hotkey or "?")[:12],
        )
        return record

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
            if self.alltime_active(epoch_block):
                return self._maybe_promote_alltime(epoch_block=epoch_block, round_id=round_id)
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
            # No-downgrade guard: a ripe clock says a promotion MAY fire, never
            # that it must. If the best candidate this reign benches WORSE than
            # the live generation's best member (lower = better), installing it
            # would ratchet the whole field's shared init downhill — the basin
            # DEC-CA-0014 exists to escape must never be entered by promotion
            # itself. Hold instead: the live generation keeps training, the
            # clock stays ripe, candidates keep accumulating, and the promotion
            # fires the first round a candidate at least matches the incumbent
            # init's bench. Pure trainer policy (DEC-CA-0013: declining to
            # declare a generation is always envelope-legal); members without a
            # finite recorded score — legacy pointer adoptions — cannot anchor
            # the comparison and never block a firing.
            best_member = min((m.score for m in self.members
                               if math.isfinite(m.score) and m.score > 0),
                              default=None)
            best_candidate = min((c.score for c in self.candidates
                                  if math.isfinite(c.score)), default=None)
            if (best_member is not None and best_candidate is not None
                    and best_candidate > best_member):
                log.warning(
                    "trainer promotion: clock ripe (%.2f rounds) but the best "
                    "candidate benches %.5f vs the live generation's best member "
                    "%.5f — holding the current generation (no-downgrade guard); "
                    "%d candidate(s) logged, retrying as new rounds bench",
                    elapsed, best_candidate, best_member, len(self.candidates))
                return None
            selected = select_members(
                list(self.candidates), k_max=self.k_max,
                quality_epsilon=self.quality_epsilon,
                min_round_spacing=self.min_round_spacing,
                error_vectors=self._load_error_vectors(),
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

    def _load_error_vectors(self) -> dict[str, list[float]] | None:
        """The error-vector cache for select_members, or ``None``. Best-effort:
        selection must fire on a ripe clock whether or not vectors exist —
        a missing/corrupt cache just means structural-diversity fallback."""
        if self.error_vectors_path is None:
            return None
        try:
            obj = json.loads(self.error_vectors_path.read_text(encoding="utf-8"))
            vectors = {str(k): [float(x) for x in v]
                       for k, v in obj.items() if isinstance(v, list) and v}
            log.info("trainer promotion: error-vector cache %s covers %d checkpoint(s)",
                     self.error_vectors_path, len(vectors))
            return vectors or None
        except FileNotFoundError:
            return None
        except Exception as e:  # noqa: BLE001 — never let the cache block a firing
            log.warning("trainer promotion: error-vector cache %s unreadable (%s); "
                        "structural fallback", self.error_vectors_path, e)
            return None

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
            "members": [member_to_json(m) for m in self.members],
            "king_hotkey": self.king_hotkey,
            "reign_start_block": self.reign_start_block,
            "candidates": [
                {"checkpoint_id": c.checkpoint_id, "size": c.size, "hotkey": c.hotkey,
                 "role": c.role, "round_id": c.round_id,
                 "epoch_index": c.epoch_index, "score": c.score}
                for c in self.candidates
            ],
            "leaderboard": [e.to_json() for e in self.leaderboard],
            "pending_change": (None if self.pending_change is None
                               else self.pending_change.to_json()),
            "leaderboard_seeded": self.leaderboard_seeded,
            "pending_record": None if self.pending_record is None else {
                "generation": self.pending_record.generation,
                "king_hotkey": self.pending_record.king_hotkey,
                "fired_round": self.pending_record.fired_round,
                "fired_block": self.pending_record.fired_block,
                "members": [member_to_json(m) for m in self.pending_record.members],
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
            "rule": ("alltime_top_k" if self.alltime_active(self.reign_start_block or 0)
                     else "reign_scoped"),
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
