"""Champion state and the sticky dethrone machine.

The per-round statistical verdict lives in :mod:`cascade.eval.koth`; this
module owns the *stateful* part: a challenger must win ``dethrone_cp``
**consecutive** rounds before it takes the throne (``dethrone_cp = 1`` makes
dethroning single-round). A single lost or inconclusive round resets its streak.
The king's ``tenure_rounds`` feeds the margin schedule (when warmup is enabled,
an entrenched king is harder to displace). Dethroned kings are remembered in
``former_kings`` so reward routing can split weight across the recent court of
champions (teutonic-style payout).

State is a small, JSON-serialisable record so it can be persisted to the
validator's state DB (``[validator] state_db_path``) and survive restarts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace

from ..eval.koth import RoundResult


@dataclass(frozen=True)
class ChampionState:
    """The validator's view of the throne.

    Attributes:
        king_hotkey / king_uid: the reigning champion. None before genesis.
        tenure_rounds: rounds the current king has held the throne.
        streaks: per-challenger-hotkey count of *consecutive* round wins.
        rounds_seen: total rounds processed (monotonic; used for logging).
        former_kings: previous distinct kings, most-recent-first, capped at the
            reward window (``[scoring] reward_prior_kings``). These share equal
            weight with the current king while they stay registered (teutonic-
            style payout). Empty when reward routing is winner-take-all.
    """

    king_hotkey: str | None = None
    king_uid: int | None = None
    tenure_rounds: int = 0
    streaks: dict[str, int] = field(default_factory=dict)
    rounds_seen: int = 0
    former_kings: tuple[str, ...] = ()
    # Consecutive rounds the validator has held the throne for a champion whose
    # *trained* king disagrees (the trainer's incentive lags a dethrone — see the
    # validator loop's king-resync branch). Reset to 0 on any normally-scored
    # round. When it reaches ``[scoring] king_resync_max_rounds`` the safety valve
    # gives up and adopts the trained king (:func:`demote_to_trained`), so a
    # champion that can never be trained as king (e.g. no usable commitment) does
    # not wedge the subnet forever. Persisted so the cap survives restarts.
    resync_holds: int = 0
    # The round_id of the last manifest that bumped ``resync_holds``. A restart
    # re-gates the SAME stale manifest, and before this field each re-gate
    # advanced the counter — five restarts during a pause tripped the valve and
    # demoted a healthy champion (2026-07-22). The counter only advances when
    # the un-synced round is a NEW round_id; same-round re-gates hold without
    # counting. Cleared with ``resync_holds`` on any normally-scored round.
    last_resync_round_id: str | None = None
    # Restart re-entry guard (the trainer's PR #157 twin — OPSLOG 2026-07-25):
    # the (round_id, manifest sha256) the loop last handled (scored, rejected,
    # or resync-held). Persisted so a restart does NOT re-judge the already-
    # published round — under a changed scoring rule that re-judgement can
    # flip a settled verdict (live 2026-07-28: DEC-CA-0009 would have retro-
    # dethroned round 10447… at boot). The sha keeps the 2026-07-15 semantics
    # across restarts: a re-published manifest with DIFFERENT content for the
    # same round id is still re-judged, never silently skipped.
    last_handled_round_id: str | None = None
    last_handled_manifest_sha: str | None = None


@dataclass(frozen=True)
class StateTransition:
    state: ChampionState
    dethroned: bool
    new_king_hotkey: str | None
    note: str


def genesis(king_hotkey: str, king_uid: int) -> ChampionState:
    """Seed the throne with an initial king (the owner baseline or first miner)."""
    return ChampionState(king_hotkey=king_hotkey, king_uid=king_uid)


def _roll_former_kings(
    state: ChampionState, *, new_king: str, keep: int
) -> tuple[str, ...]:
    """Court of recent champions after ``new_king`` takes the throne.

    The outgoing king moves to the front, followed by the prior court, with the
    new king removed (it is the reigning king now, not a *former* one) and
    duplicates dropped. Capped at ``keep`` (``[scoring] reward_prior_kings``);
    ``keep <= 0`` ⇒ winner-take-all, so no court is kept.
    """
    if keep <= 0:
        return ()
    ordered: list[str] = []
    for hk in (state.king_hotkey, *state.former_kings):
        if hk is not None and hk != new_king and hk not in ordered:
            ordered.append(hk)
    return tuple(ordered[:keep])


def apply_round(
    state: ChampionState,
    *,
    challenger_hotkey: str,
    challenger_uid: int,
    result: RoundResult,
    dethrone_cp: int,
    keep_former_kings: int = 0,
    defeated_hotkeys: tuple[str, ...] = (),
) -> StateTransition:
    """Fold one round's result into the champion state.

    Pure: returns a new :class:`ChampionState`. The streak only advances on a
    conclusive win; an inconclusive round (too few windows) leaves the streak
    untouched but still counts the king's tenure. On a dethrone, the outgoing
    king is rolled into ``former_kings`` (capped at ``keep_former_kings``) so
    reward routing can pay the recent court of champions.

    ``challenger_hotkey`` is the round's *decided* challenger — the crowned
    margin-clearer, or the best of the cohort when none cleared.
    ``defeated_hotkeys`` are the other challengers the validator duelled this
    round (DEC-CA-0010); their streaks reset. That includes a challenger which
    cleared the margin but was not crowned: a streak is a claim on the throne and
    only one challenger can hold that claim per round, so a non-crowned clearer
    does not bank progress toward ``dethrone_cp``. Dormant at ``dethrone_cp = 1``.
    An inconclusive round leaves every streak untouched, the cohort included —
    no duel in it produced a decision.
    """
    rounds_seen = state.rounds_seen + 1

    if result.inconclusive:
        # No decision: king holds, tenure advances, streaks unchanged.
        return StateTransition(
            state=replace(state, tenure_rounds=state.tenure_rounds + 1, rounds_seen=rounds_seen),
            dethroned=False,
            new_king_hotkey=state.king_hotkey,
            note="inconclusive",
        )

    streaks = dict(state.streaks)
    for hk in defeated_hotkeys:
        if hk != challenger_hotkey:
            streaks.pop(hk, None)
    if result.challenger_wins_round:
        streaks[challenger_hotkey] = streaks.get(challenger_hotkey, 0) + 1
    else:
        streaks.pop(challenger_hotkey, None)

    if streaks.get(challenger_hotkey, 0) >= dethrone_cp:
        # Dethrone: challenger becomes king; tenure and all streaks reset. The
        # outgoing king joins the rewarded court of former kings.
        return StateTransition(
            state=ChampionState(
                king_hotkey=challenger_hotkey,
                king_uid=challenger_uid,
                tenure_rounds=0,
                streaks={},
                rounds_seen=rounds_seen,
                former_kings=_roll_former_kings(
                    state, new_king=challenger_hotkey, keep=keep_former_kings
                ),
                last_handled_round_id=state.last_handled_round_id,
                last_handled_manifest_sha=state.last_handled_manifest_sha,
            ),
            dethroned=True,
            new_king_hotkey=challenger_hotkey,
            note=(
                f"dethroned after {dethrone_cp} consecutive win(s)"
            ),
        )

    return StateTransition(
        state=replace(
            state,
            tenure_rounds=state.tenure_rounds + 1,
            streaks=streaks,
            rounds_seen=rounds_seen,
        ),
        dethroned=False,
        new_king_hotkey=state.king_hotkey,
        note=("win" if result.challenger_wins_round else "loss"),
    )


def demote_to_trained(
    state: ChampionState, *, trained_hotkey: str, trained_uid: int
) -> ChampionState:
    """Abandon a stuck champion and crown the trainer's trained king.

    The validator's king-resync safety valve: when the crowned champion has
    stayed out of sync with the trainer for ``king_resync_max_rounds`` consecutive
    rounds — it can never become the king the trainer trains (e.g. it has no
    usable on-chain commitment) — holding the throne for it starves the subnet.
    This resynchronises the validator to ground truth (the incentive-driven king
    the trainer actually trained) by crowning it fresh: tenure and streaks reset
    and the resync counter clears. The abandoned champion is **not** rolled into
    ``former_kings`` — it is being dropped, not honourably retired, so it must not
    keep drawing reward. ``rounds_seen`` still advances (a round was processed).
    """
    return ChampionState(
        king_hotkey=trained_hotkey,
        king_uid=trained_uid,
        tenure_rounds=0,
        streaks={},
        rounds_seen=state.rounds_seen + 1,
        former_kings=(),
        resync_holds=0,
        last_resync_round_id=None,
        # The throne is being reset, not the loop's position in the manifest
        # stream — dropping the marker here would re-judge the round that
        # triggered the demotion on the next restart.
        last_handled_round_id=state.last_handled_round_id,
        last_handled_manifest_sha=state.last_handled_manifest_sha,
    )


def dumps(state: ChampionState) -> str:
    return json.dumps(
        {
            "king_hotkey": state.king_hotkey,
            "king_uid": state.king_uid,
            "tenure_rounds": state.tenure_rounds,
            "streaks": state.streaks,
            "rounds_seen": state.rounds_seen,
            "former_kings": list(state.former_kings),
            "resync_holds": state.resync_holds,
            "last_resync_round_id": state.last_resync_round_id,
            "last_handled_round_id": state.last_handled_round_id,
            "last_handled_manifest_sha": state.last_handled_manifest_sha,
        },
        sort_keys=True,
    )


def loads(text: str) -> ChampionState:
    obj = json.loads(text)
    return ChampionState(
        king_hotkey=obj.get("king_hotkey"),
        king_uid=obj.get("king_uid"),
        tenure_rounds=int(obj.get("tenure_rounds", 0)),
        streaks={str(k): int(v) for k, v in (obj.get("streaks") or {}).items()},
        rounds_seen=int(obj.get("rounds_seen", 0)),
        former_kings=tuple(str(k) for k in (obj.get("former_kings") or ())),
        resync_holds=int(obj.get("resync_holds", 0)),
        last_resync_round_id=(
            str(obj["last_resync_round_id"])
            if obj.get("last_resync_round_id") is not None else None
        ),
        last_handled_round_id=(
            str(obj["last_handled_round_id"])
            if obj.get("last_handled_round_id") is not None else None
        ),
        last_handled_manifest_sha=(
            str(obj["last_handled_manifest_sha"])
            if obj.get("last_handled_manifest_sha") is not None else None
        ),
    )
