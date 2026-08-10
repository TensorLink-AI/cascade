"""Cascade — king-reign promotion for the warm-start loop (validator half).

The daily king-of-the-hill loop runs underneath: challengers train fresh models
from the shared init and dethrone the king when they clear the margin
(:mod:`cascade.eval.koth` + :mod:`cascade.validator.state`). Cascade sits *on
top* of that loop and answers a different question — *when has one king held the
throne long enough that its reign's best checkpoints should become the new floor
the whole field trains up from?*

Promotion is **propose-and-verify** (DEC-CA-0013). The trainer — which authors
the bench numbers, trains every model, and already declares each round's init in
the signed manifest — *selects* the promoted member set and publishes a signed
:class:`~cascade.shared.promotion.PromotionRecord`. The validator never selects:
it *verifies* the declaration against a deterministic envelope and tracks two
things here:

* **The reign clock.** A block-anchored count of blocks since the current king
  last took the throne, anchored to the round's epoch-start block (derived
  identically by every validator from the signed manifest's ``created_block`` —
  never local wall-clock). Every dethrone re-crowns and resets the clock
  (Cascade reuses the KOTH dethrone signal via
  :meth:`CascadeController.note_dethrone` — it never re-implements dethroning).
  The clock is the envelope's *ripeness* predicate: a new promotion generation
  is only acceptable once the reign has survived ``cascade_reign_rounds``
  challenges.

* **The reign log.** Every duel checkpoint benched during the reign — the
  king's AND the challenger's (both are candidates; different generators are
  the deepest diversity the promoted set can draw on) — scored
  ``geomean(gifteval_crps, gifteval_mase, boom_crps, boom_mase, time_crps,
  time_mase)`` from the trainer-signed bench report. The log is the envelope's
  *provenance and quality-floor* evidence: a declared member must be a benched
  reign checkpoint whose score sits within ``cascade_quality_epsilon`` of the
  reign's best.

On a VERIFIED promotion the king PERSISTS on the throne — only the reign clock
and checkpoint log reset, so the next generation needs a fresh full reign
(DEC-CA-0004: the throne is never vacated). The accepted generation number and
member set survive re-crowns and dethrones alike: a promotion outlives the reign
that produced it — the field keeps training from the live set no matter who
holds the throne.

:class:`CascadeState` is JSON-serialisable and persisted next to the champion
state — the clock, the log, and the accepted generation must survive process
restarts. A persisted state predating the block anchor (or missing one) is
re-anchored at the next observed round's block instead of firing immediately, so
stale state can never cause a spurious ripeness verdict on restart.

The pure core (the dataclasses + transition functions) carries no I/O and is
unit-tested directly; :class:`CascadeController` binds it to persistence.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, replace
from pathlib import Path

log = logging.getLogger("cascade.validator.cascade")

# The reign clock counts blocks — every validator reads the same block from the
# signed manifest, so the clock is identical across the fleet and across
# restarts.
#
# It is measured in ROUNDS, not wall-clock days. A reign is "survived N
# challenges", and a challenge happens once per round; tying it to a day made
# the threshold silently mean different things at different cadences (at 24h
# rounds 7 days WAS 7 rounds; at 12h it would be 14). Round-relative keeps the
# competitive meaning fixed under any future cadence change.
#
# Retained only so an old ``reign_days``-style reading can still be derived for
# logs: 7200 blocks x 12 s = one real day. NOT used by the clock.
BLOCKS_PER_DAY = 7_200

# A tiny floor so a zero/negative eval number can't collapse the geomean product
# to 0 (or NaN under a fractional power). Mirrors eval.scoring.global_geomean.
_EPS = 1e-12


def geomean(*values: float) -> float:
    """Geometric mean of its arguments — lower is better for the eval metrics
    Cascade scores on. Each value is clamped to a tiny positive floor so a single
    zero (or a spurious negative) can't zero out or NaN the product."""
    if not values:
        return float("nan")
    prod = 1.0
    for v in values:
        prod *= max(float(v), _EPS)
    return float(prod ** (1.0 / len(values)))


def cascade_score(
    gifteval_crps: float,
    gifteval_mase: float,
    boom_crps: float,
    boom_mase: float,
    time_crps: float,
    time_mase: float,
) -> float:
    """The Cascade checkpoint score: geomean of the six public-benchmark numbers
    (GIFT-Eval / BOOM / TIME CRPS+MASE). Lower is better."""
    return geomean(gifteval_crps, gifteval_mase, boom_crps, boom_mase, time_crps, time_mase)


@dataclass(frozen=True)
class CheckpointRecord:
    """One duel checkpoint benched during the reign, scored for Cascade.

    ``checkpoint_id`` identifies the artifact (the trained-pointer / registry
    ref). The six eval numbers are the CRPS and MASE the checkpoint scored on
    GIFT-Eval, BOOM, and TIME; ``score`` is their :func:`cascade_score`
    (geomean, lower is better); ``timestamp`` is wall-clock epoch seconds when
    it was recorded (local observability only); ``size`` is the arch preset the
    checkpoint was trained at, so a promoted init is only ever loaded into a
    matching model. ``role`` records which side of the duel produced it
    ("king" | "challenger"; "" on records from before the field existed) —
    observability, never a join key.
    """

    checkpoint_id: str
    gifteval_crps: float
    gifteval_mase: float
    boom_crps: float
    boom_mase: float
    time_crps: float
    time_mase: float
    score: float
    timestamp: float
    size: str = ""
    role: str = ""

    @classmethod
    def scored(
        cls,
        checkpoint_id: str,
        *,
        gifteval_crps: float,
        gifteval_mase: float,
        boom_crps: float,
        boom_mase: float,
        time_crps: float,
        time_mase: float,
        timestamp: float,
        size: str = "",
        role: str = "",
    ) -> CheckpointRecord:
        """Build a record, computing ``score`` from the six eval numbers so the
        geomean convention lives in exactly one place."""
        return cls(
            checkpoint_id=checkpoint_id,
            gifteval_crps=float(gifteval_crps),
            gifteval_mase=float(gifteval_mase),
            boom_crps=float(boom_crps),
            boom_mase=float(boom_mase),
            time_crps=float(time_crps),
            time_mase=float(time_mase),
            score=cascade_score(gifteval_crps, gifteval_mase, boom_crps, boom_mase, time_crps, time_mase),
            timestamp=float(timestamp),
            size=str(size),
            role=str(role),
        )


@dataclass(frozen=True)
class CascadeState:
    """The reign clock, the reign's checkpoint log, and the accepted promotion.

    Attributes:
        king_hotkey: the reigning king Cascade is timing. ``None`` when the throne
            is vacant (before genesis).
        reign_start_block: epoch-start block when ``king_hotkey`` took the
            throne — the zero of the reign clock, read identically by every
            validator from the signed manifest. ``None`` when no king reigns OR
            the state predates the block anchor (legacy wall-clock state); an
            unanchored reign is re-anchored at the next observed round, never
            judged ripe.
        checkpoints: every :class:`CheckpointRecord` benched this reign, in
            record order. Cleared on every re-crown so the envelope's evidence
            only ever spans the current reign; persisted so a mid-reign restart
            keeps verification a lookup rather than forcing a re-eval.
        generation: the accepted warm-start generation (0 = random-init era, no
            promotion accepted). Strictly monotonic; survives re-crowns.
        members: the accepted generation's member checkpoint pointers — the set
            a manifest's ``warm_start_ckpt`` must come from. Survives re-crowns
            and dethrones: a promotion outlives the reign that produced it.
        clock_observed: whether the reign anchor comes from an OBSERVED
            transition (a scored dethrone verdict, or an accepted promotion)
            rather than an adoption/re-anchor (validator joined or restarted
            mid-reign, legacy state re-anchored). Only an observed anchor can
            attest reign timing — an adopted clock measures its own uptime, and
            treating it as authoritative would reject honest promotions.
    """

    king_hotkey: str | None = None
    reign_start_block: int | None = None
    checkpoints: tuple[CheckpointRecord, ...] = ()
    generation: int = 0
    members: tuple[str, ...] = ()
    clock_observed: bool = False


# ── pure transitions over CascadeState ───────────────────────────────────────


def crown(state: CascadeState, *, king_hotkey: str, block: int,
          observed: bool = True) -> CascadeState:
    """Re-crown for a newly-throned king: start the reign clock at ``block`` and
    clear the previous reign's checkpoint log. Called on every dethrone (Cascade
    reuses KOTH's dethrone signal rather than reimplementing it) and on every
    accepted promotion (the persisting king starts a fresh reign). The accepted
    promotion (``generation``/``members``) is deliberately CARRIED OVER — the
    field trains from the live set no matter who holds the throne. ``observed``
    records whether this crowning comes from a transition the validator actually
    watched (attesting clock) or an adoption (mid-reign join/restart)."""
    return CascadeState(
        king_hotkey=king_hotkey,
        reign_start_block=int(block),
        checkpoints=(),
        generation=state.generation,
        members=state.members,
        clock_observed=bool(observed),
    )


def record_checkpoint(state: CascadeState, record: CheckpointRecord) -> CascadeState:
    """Append a benched checkpoint to the current reign's log (pure)."""
    return replace(state, checkpoints=(*state.checkpoints, record))


def accept_promotion(
    state: CascadeState, *, generation: int, members: tuple[str, ...], block: int
) -> CascadeState:
    """Adopt a VERIFIED promotion: install the new generation + member set and
    re-crown the same king (fresh clock, cleared log) so the next generation
    needs a fresh full reign. Pure — verification happened before this. An
    accepted promotion is an observed transition: the fresh clock can attest
    the next generation's timing."""
    return CascadeState(
        king_hotkey=state.king_hotkey,
        reign_start_block=int(block),
        checkpoints=(),
        generation=int(generation),
        members=tuple(members),
        clock_observed=True,
    )


def reign_blocks(state: CascadeState, block: int) -> int | None:
    """Blocks the current king has reigned, or ``None`` if vacant/unanchored."""
    if state.reign_start_block is None:
        return None
    return int(block) - state.reign_start_block


def reign_rounds(
    state: CascadeState, block: int, round_cfg: object | None = None
) -> float | None:
    """Rounds the current king has reigned, or ``None`` if vacant/unanchored.

    Epoch-relative: the divisor is the round length in force, not a fixed day.
    ``round_cfg`` is a :class:`~cascade.shared.config.RoundConfig`; ``None``
    falls back to a 7200-block round, which reproduces the historical
    day-denominated behaviour exactly.

    A scheduled cadence change is handled exactly rather than approximated: the
    reign is split at the activation block and each segment divided by its own
    length, so a reign spanning the seam counts the rounds that actually ran.
    """
    b = reign_blocks(state, block)
    if b is None:
        return None
    if b <= 0:
        return 0.0
    start = int(state.reign_start_block or 0)
    end = int(block)
    if round_cfg is None:
        return b / BLOCKS_PER_DAY

    from ..shared.config import effective_epoch_blocks

    act = int(getattr(round_cfg, "epoch_activation_block", 0) or 0)
    if act and start < act < end:
        pre = (act - start) / effective_epoch_blocks(round_cfg, start)
        post = (end - act) / effective_epoch_blocks(round_cfg, act)
        return pre + post
    return b / effective_epoch_blocks(round_cfg, start)


def reign_days(state: CascadeState, block: int) -> float | None:
    """Wall-clock days the current king has reigned (reign blocks / 7200).

    Observability only — ripeness is round-based (:func:`reign_rounds`). Kept
    because a human reading a log wants days.
    """
    b = reign_blocks(state, block)
    return None if b is None else b / BLOCKS_PER_DAY


def best_score(state: CascadeState) -> float | None:
    """The reign log's lowest (best) score, or ``None`` on an empty log — the
    reference the envelope's quality floor is measured against."""
    if not state.checkpoints:
        return None
    return min(r.score for r in state.checkpoints)


def log_record_for(state: CascadeState, checkpoint_id: str) -> CheckpointRecord | None:
    """The reign log's record for ``checkpoint_id``, else ``None``."""
    for r in state.checkpoints:
        if r.checkpoint_id == checkpoint_id:
            return r
    return None


# ── persistence (JSON, alongside the champion state DB) ───────────────────────


def dumps(state: CascadeState) -> str:
    return json.dumps(
        {
            "king_hotkey": state.king_hotkey,
            "reign_start_block": state.reign_start_block,
            "generation": state.generation,
            "members": list(state.members),
            "clock_observed": state.clock_observed,
            "checkpoints": [
                {
                    "checkpoint_id": r.checkpoint_id,
                    "gifteval_crps": r.gifteval_crps,
                    "gifteval_mase": r.gifteval_mase,
                    "boom_crps": r.boom_crps,
                    "boom_mase": r.boom_mase,
                    "time_crps": r.time_crps,
                    "time_mase": r.time_mase,
                    "score": r.score,
                    "timestamp": r.timestamp,
                    "size": r.size,
                    "role": r.role,
                }
                for r in state.checkpoints
            ],
        },
        sort_keys=True,
    )


def loads(text: str) -> CascadeState:
    obj = json.loads(text)
    checkpoints = tuple(
        CheckpointRecord(
            checkpoint_id=str(c["checkpoint_id"]),
            gifteval_crps=float(c["gifteval_crps"]),
            gifteval_mase=float(c["gifteval_mase"]),
            boom_crps=float(c["boom_crps"]),
            boom_mase=float(c["boom_mase"]),
            time_crps=float(c["time_crps"]),
            time_mase=float(c["time_mase"]),
            score=float(c["score"]),
            timestamp=float(c["timestamp"]),
            size=str(c.get("size", "")),
            role=str(c.get("role", "")),
        )
        for c in (obj.get("checkpoints") or ())
    )
    # Legacy wall-clock state files carry "reign_start" (epoch seconds) and no
    # block anchor: load them UNANCHORED (reign_start_block=None) so the clock is
    # re-anchored at the next observed round instead of misread — a stale
    # wall-clock value must never translate into an instant ripe clock. States
    # from before the promotion fields default to generation 0 / no members (the
    # legacy single-pointer adoption shim in the validator loop upgrades them).
    start_block = obj.get("reign_start_block")
    return CascadeState(
        king_hotkey=obj.get("king_hotkey"),
        reign_start_block=None if start_block is None else int(start_block),
        checkpoints=checkpoints,
        generation=int(obj.get("generation", 0) or 0),
        members=tuple(str(m) for m in (obj.get("members") or ())),
        # Pre-upgrade states default to an UNOBSERVED clock: they skip ripeness
        # attestation once, until the next watched dethrone/acceptance —
        # conservative toward accepting an honest promotion, never rejecting one.
        clock_observed=bool(obj.get("clock_observed", False)),
    )


# ── stateful controller ───────────────────────────────────────────────────────


@dataclass
class CascadeController:
    """Binds the pure Cascade core to persistence.

    ``reign_days`` is the ripeness threshold, in ROUNDS (``[scoring]
    cascade_reign_rounds``; field name kept for call-site compat). ``round_cfg``
    is the :class:`RoundConfig` the clock divides by; leave it ``None`` and the
    divisor is a fixed 7200-block round, reproducing the historical
    day-denominated behaviour exactly. ``state_path`` is where
    :class:`CascadeState` is persisted (the clock and the accepted promotion
    must survive restarts); ``None`` keeps it in memory only.

    The controller never selects or installs a promotion — the trainer selects
    (:mod:`cascade.trainer.promotion`); the validator loop verifies the signed
    declaration against the envelope and calls :meth:`note_promotion` on
    acceptance.
    """

    reign_days: float          # ripeness threshold, in ROUNDS
    state: CascadeState = field(default_factory=CascadeState)
    state_path: Path | None = None
    round_cfg: object | None = None   # RoundConfig; None ⇒ fixed 7200-block rounds

    def note_dethrone(self, new_king: str, *, block: int, observed: bool = True) -> None:
        """Reset the reign clock for a fresh king. Call this on — and only on — a
        KOTH dethrone (``StateTransition.dethroned``); it re-crowns Cascade's view
        so the reign starts at the epoch block the throne changed hands. The
        accepted promotion carries over — the field keeps training from the live
        member set under the new king. Pass ``observed=False`` when the crowning
        is an ADOPTION (validator joined/restarted mid-reign and inherited the
        champion) rather than a watched verdict — an adopted clock measures its
        own uptime and must not attest reign timing."""
        self.state = crown(self.state, king_hotkey=new_king, block=block,
                           observed=observed)
        self._persist()
        log.info("cascade: reign clock reset for new king %s at block %d (%s)",
                 (new_king or "?")[:12], int(block),
                 "observed" if observed else "adopted")

    def observe_round(self, *, block: int) -> None:
        """Re-anchor an unanchored reign at the observed round's epoch block.

        A reigning king with no block anchor (legacy wall-clock state, or state
        written before the anchor existed) starts its clock fresh here rather
        than reading as instantly ripe — stale state must never validate a
        premature promotion."""
        state = self.state
        if state.king_hotkey is None or state.reign_start_block is not None:
            return
        self.state = replace(state, reign_start_block=int(block), clock_observed=False)
        self._persist()
        log.info(
            "cascade: reign for king %s re-anchored at block %d (state had no "
            "block anchor); clock starts fresh",
            (state.king_hotkey or "?")[:12], int(block),
        )

    def record_checkpoint(
        self,
        checkpoint_id: str,
        *,
        gifteval_crps: float,
        gifteval_mase: float,
        boom_crps: float,
        boom_mase: float,
        time_crps: float,
        time_mase: float,
        now: float,
        size: str = "",
        role: str = "",
    ) -> CheckpointRecord | None:
        """Score and log a benched duel checkpoint. No-op (returns ``None``)
        when the throne is vacant — the log only accumulates within a reign —
        or when the pointer is already logged (the pending-bench re-probe can
        offer a round twice)."""
        if self.state.king_hotkey is None:
            return None
        if log_record_for(self.state, checkpoint_id) is not None:
            return None
        rec = CheckpointRecord.scored(
            checkpoint_id,
            gifteval_crps=gifteval_crps,
            gifteval_mase=gifteval_mase,
            boom_crps=boom_crps,
            boom_mase=boom_mase,
            time_crps=time_crps,
            time_mase=time_mase,
            timestamp=now,
            size=size,
            role=role,
        )
        self.state = record_checkpoint(self.state, rec)
        self._persist()
        log.info(
            "cascade: recorded %s checkpoint %s score=%.5f (gift crps=%.5f mase=%.5f, "
            "boom crps=%.5f mase=%.5f, time crps=%.5f mase=%.5f); %d checkpoint(s) this reign",
            rec.role or "duel", rec.checkpoint_id, rec.score, rec.gifteval_crps,
            rec.gifteval_mase, rec.boom_crps, rec.boom_mase, rec.time_crps,
            rec.time_mase, len(self.state.checkpoints),
        )
        return rec

    def is_ripe(self, *, block: int) -> bool:
        """Whether the reign clock has reached the ripeness threshold — the
        envelope's timing predicate for accepting the NEXT generation. False
        when the throne is vacant or the clock is unanchored (an unanchored
        clock can't attest anything; the caller decides whether ripeness is
        verifiable at all — see the envelope's bootstrap rules)."""
        state = self.state
        if state.king_hotkey is None or state.reign_start_block is None:
            return False
        elapsed = reign_rounds(state, block, self.round_cfg)
        return elapsed is not None and elapsed >= self.reign_days

    def can_verify_ripeness(self) -> bool:
        """Whether this validator's clock is in a position to judge reign
        timing at all: an anchored reign whose anchor comes from an OBSERVED
        transition (a watched dethrone verdict or an accepted promotion). An
        adopted or re-anchored clock — validator joined or restarted
        mid-reign, legacy state — cannot distinguish "promotion too early"
        from "my clock started late", so the envelope skips the timing checks
        rather than rejecting a whole reign's rounds on a clock that only
        measures its own uptime. Unlike a generation gate, this lets a genesis
        validator that watched the ENTIRE random-init era attest even the
        first promotion (generation 0 → 1)."""
        return self.state.reign_start_block is not None and self.state.clock_observed

    def adopt_member_set(self, *, generation: int, members: tuple[str, ...]) -> None:
        """Migration shim: adopt a member set recovered from a trainer-written
        pointer file (owner box, pre-record deploys) as the accepted
        generation. Runs only from the random-init era; the clock is untouched
        (whatever its provenance), so this never manufactures attestation."""
        if self.state.generation != 0 or not members:
            return
        self.state = replace(
            self.state, generation=max(1, int(generation)),
            members=tuple(str(m) for m in members))
        self._persist()
        log.info(
            "cascade: adopted trainer-written warm-start set (%d member(s)) as "
            "generation %d", len(members), max(1, int(generation)),
        )

    def note_promotion(self, *, generation: int, members: tuple[str, ...], block: int) -> None:
        """Adopt a VERIFIED promotion (the validator loop verified the signed
        record against the envelope before calling): install the generation +
        member set, re-crown the SAME king, fresh clock, cleared log."""
        self.state = accept_promotion(
            self.state, generation=generation, members=members, block=block
        )
        self._persist()
        log.info(
            "cascade: promotion generation=%d accepted (%d member(s)); king %s "
            "persists, reign clock reset",
            int(generation), len(members), (self.state.king_hotkey or "?")[:12],
        )

    def adopt_legacy_pointer(self, checkpoint_id: str) -> None:
        """Migration shim: grandfather a pre-DEC-CA-0013 single-pointer install
        (the old ``warm_start_init.json`` this validator's own Cascade wrote) as
        generation 1. Runs once, only from the random-init era — after this the
        signed-record path owns all transitions."""
        if self.state.generation != 0 or not checkpoint_id:
            return
        self.state = replace(self.state, generation=1, members=(str(checkpoint_id),))
        self._persist()
        log.info(
            "cascade: adopted legacy warm-start pointer %s as promotion "
            "generation 1 (pre-record install)", checkpoint_id,
        )

    # ── I/O helpers ──────────────────────────────────────────────────────────

    def _persist(self) -> None:
        if self.state_path is None:
            return
        try:
            self.state_path.write_text(dumps(self.state), encoding="utf-8")
        except Exception as e:  # noqa: BLE001 — persistence must never abort a round
            log.warning("cascade: failed to persist state to %s: %s", self.state_path, e)


def load_state(path: str | Path) -> CascadeState:
    """Load persisted Cascade state (JSON), or a fresh state if the file is
    absent/unreadable — the reign clock resumes across restarts."""
    p = Path(path)
    if not p.is_file():
        return CascadeState()
    try:
        return loads(p.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        log.warning("cascade: could not load state from %s (%s); starting fresh", p, e)
        return CascadeState()
