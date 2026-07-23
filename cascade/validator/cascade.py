"""Cascade — king-reign promotion for the warm-start loop.

The daily king-of-the-hill loop runs underneath: challengers train fresh models
from the shared init and dethrone the king when they clear the margin
(:mod:`cascade.eval.koth` + :mod:`cascade.validator.state`). Cascade sits *on
top* of that loop and answers a different question — *when has one king held the
throne long enough that its best checkpoint should become the new floor the
whole field trains up from?*

The mechanism has three moving parts:

* **Trigger.** A wall-clock *reign clock* counts days since the current king last
  took the throne. Every dethrone re-crowns and resets the clock to zero (Cascade
  reuses the KOTH dethrone signal via :meth:`CascadeController.note_dethrone` — it
  never re-implements dethroning). When a king reigns ``cascade_reign_days``
  consecutive days undethroned, Cascade fires.

* **Checkpoint selection.** Every checkpoint the king produces during its reign is
  evaluated on the three public suites — GIFT-Eval, BOOM, and TIME — each yielding
  a CRPS and a MASE. The checkpoint is scored ``geomean(gifteval_crps,
  gifteval_mase, boom_crps, boom_mase, time_crps, time_mase)`` (lower is better)
  and the six numbers + score are kept in a per-reign log
  (:class:`CheckpointRecord`). On a Cascade event the reign's lowest-score
  checkpoint is selected — a lookup over the log, never a re-eval.

  The six numbers are produced once by the (trusted, owner-operated) trainer via
  the benchmark sidecar and published on the king's manifest entry
  (:class:`cascade.shared.manifest.BenchScores`), so every validator records the
  *same signed numbers* — Cascade is deterministic across validators rather than
  each re-running a non-bit-reproducible GPU sweep. The validator can still score
  a checkpoint itself as a fallback when the manifest lacks the numbers.

* **Action.** The selected checkpoint is installed as the warm-start init for all
  subsequent rounds (promoted **as-is** — never retrained or fine-tuned). The king
  PERSISTS on the throne — only its reign clock and checkpoint log reset, so the
  next promotion needs a fresh ``cascade_reign_days`` reign. The throne is never
  vacated: both roles train from the shared init, so promotion confers no relative
  advantage to give back, and a vacated throne could only be refilled through a
  dethrone — an incumbent that kept winning would leave it vacant and the clock
  dead forever (DEC-CA-0004).

The reign clock is wall-clock driven, so :class:`CascadeState` (king identity,
reign start, and the reign's checkpoint log) is JSON-serialisable and persisted
next to the champion state — Cascade must survive process restarts and pick the
clock back up where it left off.

The pure core (the dataclasses + transition functions) carries no I/O and is
unit-tested directly; :class:`CascadeController` binds it to persistence and the
checkpoint installer, and exposes the single per-round entry point
:meth:`CascadeController.cascade_check`.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

log = logging.getLogger("cascade.validator.cascade")

SECONDS_PER_DAY = 86_400.0

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
    """One checkpoint the king produced during its reign, scored for Cascade.

    ``checkpoint_id`` identifies the artifact to promote (the trained-pointer /
    registry ref). The six eval numbers are the CRPS and MASE the checkpoint
    scored on GIFT-Eval, BOOM, and TIME; ``score`` is their :func:`cascade_score`
    (geomean, lower is better); ``timestamp`` is wall-clock epoch seconds when it
    was recorded (the selection tiebreak, and reign observability).
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
        )


@dataclass(frozen=True)
class CascadeState:
    """The reign clock and the current reign's checkpoint log.

    Attributes:
        king_hotkey: the reigning king Cascade is timing. ``None`` when the throne
            is vacant (before genesis, or just after a Cascade fired and re-opened
            the competition).
        reign_start: wall-clock epoch seconds when ``king_hotkey`` took the
            throne — the zero of the reign clock. ``None`` iff no king reigns.
        checkpoints: every :class:`CheckpointRecord` the king produced this reign,
            in record order. Cleared on every re-crown so selection only ever sees
            the current reign; persisted so a mid-reign restart keeps selection a
            lookup rather than forcing a re-eval.
    """

    king_hotkey: str | None = None
    reign_start: float | None = None
    checkpoints: tuple[CheckpointRecord, ...] = ()


@dataclass(frozen=True)
class CascadeEvent:
    """A fired Cascade: the record of what was promoted and why.

    ``old_king`` reigned ``reign_days`` (wall-clock) before Cascade fired — the
    name is historical; the king persists on the throne with a fresh clock.
    ``winner`` is the reign's lowest-score checkpoint, now installed as the
    warm-start init; ``timestamp`` is when the event fired (epoch seconds).
    """

    old_king: str | None
    reign_days: float
    winner: CheckpointRecord
    timestamp: float


# ── pure transitions over CascadeState ───────────────────────────────────────


def crown(state: CascadeState, *, king_hotkey: str, now: float) -> CascadeState:
    """Re-crown for a newly-throned king: start the reign clock at ``now`` and
    clear the previous reign's checkpoint log. Called on every dethrone (Cascade
    reuses KOTH's dethrone signal rather than reimplementing it) and on every
    fired Cascade (the persisting king starts a fresh reign). Idempotent for a
    king that is already reigning *from the same instant* — but a genuine re-crown
    of the same hotkey (it lost and retook the throne) correctly restarts its
    clock, which is what a reign is."""
    return CascadeState(king_hotkey=king_hotkey, reign_start=float(now), checkpoints=())


def record_checkpoint(state: CascadeState, record: CheckpointRecord) -> CascadeState:
    """Append a scored checkpoint to the current reign's log (pure)."""
    return replace(state, checkpoints=(*state.checkpoints, record))


def reign_seconds(state: CascadeState, now: float) -> float | None:
    """Wall-clock seconds the current king has reigned, or ``None`` if vacant."""
    if state.reign_start is None:
        return None
    return float(now) - state.reign_start


def reign_days(state: CascadeState, now: float) -> float | None:
    """Wall-clock days the current king has reigned, or ``None`` if vacant."""
    s = reign_seconds(state, now)
    return None if s is None else s / SECONDS_PER_DAY


def select_winner(state: CascadeState) -> CheckpointRecord | None:
    """The reign's lowest-score checkpoint (earliest wins ties, for determinism),
    or ``None`` when the reign produced no scored checkpoint."""
    if not state.checkpoints:
        return None
    return min(state.checkpoints, key=lambda r: (r.score, r.timestamp))


def should_cascade(state: CascadeState, now: float, reign_days_threshold: float) -> bool:
    """Whether the reign clock has reached the threshold *and* there is at least
    one checkpoint to promote. A ripe clock with an empty log is not a Cascade —
    there is nothing to select — so the king simply holds until it produces one."""
    if state.king_hotkey is None or state.reign_start is None:
        return False
    if not state.checkpoints:
        return False
    days = reign_days(state, now)
    return days is not None and days >= reign_days_threshold


# ── persistence (JSON, alongside the champion state DB) ───────────────────────


def dumps(state: CascadeState) -> str:
    return json.dumps(
        {
            "king_hotkey": state.king_hotkey,
            "reign_start": state.reign_start,
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
        )
        for c in (obj.get("checkpoints") or ())
    )
    reign_start = obj.get("reign_start")
    return CascadeState(
        king_hotkey=obj.get("king_hotkey"),
        reign_start=None if reign_start is None else float(reign_start),
        checkpoints=checkpoints,
    )


# ── stateful controller: the single per-round entry point ─────────────────────

# Installs the winning checkpoint as the warm-start init for all subsequent
# rounds. Promotes AS-IS (no retrain/fine-tune); raising aborts the Cascade so it
# is retried next round with the throne (and clock) untouched.
InstallFn = Callable[[CheckpointRecord], None]


@dataclass
class CascadeController:
    """Binds the pure Cascade core to persistence and the checkpoint installer.

    ``reign_days`` is the trigger threshold (``[scoring] cascade_reign_days``).
    ``install_fn`` promotes the selected checkpoint to the warm-start init; when
    unset the selection is logged but not installed (a plumbing warning, never a
    silent no-op). ``state_path`` is where :class:`CascadeState` is persisted (the
    reign clock must survive restarts); ``None`` keeps it in memory only.
    """

    reign_days: float
    state: CascadeState = field(default_factory=CascadeState)
    install_fn: InstallFn | None = None
    state_path: Path | None = None

    def note_dethrone(self, new_king: str, *, now: float) -> None:
        """Reset the reign clock for a fresh king. Call this on — and only on — a
        KOTH dethrone (``StateTransition.dethroned``); it re-crowns Cascade's view
        so the wall-clock reign starts at the moment the throne changed hands."""
        self.state = crown(self.state, king_hotkey=new_king, now=now)
        self._persist()
        log.info("cascade: reign clock reset for new king %s", (new_king or "?")[:12])

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
    ) -> CheckpointRecord | None:
        """Score and log a checkpoint the reigning king produced. No-op (returns
        ``None``) when the throne is vacant — Cascade only records within a reign,
        so a checkpoint that arrives before any king is crowned is ignored."""
        if self.state.king_hotkey is None:
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
        )
        self.state = record_checkpoint(self.state, rec)
        self._persist()
        log.info(
            "cascade: recorded checkpoint %s score=%.5f (gift crps=%.5f mase=%.5f, "
            "boom crps=%.5f mase=%.5f, time crps=%.5f mase=%.5f); %d checkpoint(s) this reign",
            rec.checkpoint_id, rec.score, rec.gifteval_crps, rec.gifteval_mase,
            rec.boom_crps, rec.boom_mase, rec.time_crps, rec.time_mase, len(self.state.checkpoints),
        )
        return rec

    def cascade_check(self, now: float) -> CascadeEvent | None:
        """The single per-round entry point: check the reign clock and, if it has
        reached ``reign_days`` with a checkpoint to promote, perform the Cascade —
        install the reign's best checkpoint as the warm-start init, then re-crown
        the SAME king so its next promotion needs a fresh full reign. Returns the
        :class:`CascadeEvent` when it fires, else ``None`` (clock not yet ripe,
        throne vacant, or nothing to promote).

        Install happens *before* the re-crown, so if ``install_fn`` raises the
        reign (and its clock) is left intact for a clean retry next round.
        """
        state = self.state
        if state.king_hotkey is None or state.reign_start is None:
            return None
        days = reign_days(state, now)
        if days is None or days < self.reign_days:
            return None
        winner = select_winner(state)
        if winner is None:
            # Clock is ripe but the reign logged no checkpoint yet: hold the throne
            # (there is nothing to promote) until one is recorded.
            log.warning(
                "cascade: clock ripe (king=%s reigned %.2fd ≥ %.2fd) but no checkpoint "
                "recorded this reign; holding until one is",
                (state.king_hotkey or "?")[:12], days, self.reign_days,
            )
            return None

        event = CascadeEvent(
            old_king=state.king_hotkey, reign_days=days, winner=winner, timestamp=float(now)
        )
        # Install first — a failure here must not touch the reign (retried next
        # round with the clock intact).
        self._install(winner)
        self.state = crown(self.state, king_hotkey=state.king_hotkey, now=now)
        self._persist()
        log.info(
            "CASCADE fired: king=%s reign=%.2fd winner=%s score=%.5f "
            "(gift crps=%.5f mase=%.5f, boom crps=%.5f mase=%.5f, time crps=%.5f mase=%.5f); "
            "installed as warm-start init, king persists, reign clock reset",
            (event.old_king or "?")[:12], event.reign_days, winner.checkpoint_id,
            winner.score, winner.gifteval_crps, winner.gifteval_mase,
            winner.boom_crps, winner.boom_mase, winner.time_crps, winner.time_mase,
        )
        return event

    # ── I/O helpers ──────────────────────────────────────────────────────────

    def _install(self, winner: CheckpointRecord) -> None:
        if self.install_fn is not None:
            self.install_fn(winner)
        else:
            log.warning(
                "cascade: no install_fn wired; checkpoint %s selected but NOT installed "
                "as warm-start init", winner.checkpoint_id,
            )

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
