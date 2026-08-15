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
from ..validator.cascade import CascadeState as _ClockState
from ..validator.cascade import cascade_score, reign_rounds

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
        error_vectors_path: Path | None = None,
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
