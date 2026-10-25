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
    # re-gates the SAME stale manifest; counting each re-gate would let a few
    # restarts during a pause trip the valve and demote a healthy champion.
    # The counter only advances when
    # the un-synced round is a NEW round_id; same-round re-gates hold without
    # counting. Cleared with ``resync_holds`` on any normally-scored round.
    last_resync_round_id: str | None = None
    # Restart re-entry guard (mirrors the trainer's): the (round_id, manifest
    # sha256) the loop last handled (scored, rejected, or resync-held).
    # Persisted so a restart does NOT re-judge the already-published round —
    # under a changed scoring rule that re-judgement can flip a settled verdict
    # (e.g. DEC-CA-0009). The sha keeps content-dedup semantics across
    # restarts: a re-published manifest with DIFFERENT content for the
    # same round id is still re-judged, never silently skipped.
    last_handled_round_id: str | None = None
    last_handled_manifest_sha: str | None = None
    # Era king (DEC-CA-0043). ``king_since_block``: the epoch boundary of the
    # settlement that crowned the current king. None for a king crowned
    # before the field existed — the validator imputes one from the counter
    # at that king's FIRST settlement past tenure_blocks_from_block
    # (era.legacy_king_anchor) and persists it here; it is never re-imputed
    # (the counter keeps advancing on the new grid, so re-imputing would
    # inflate the tenure 4× per settlement). ``king_pointer``: the checkpoint pointer this validator
    # judged in the crowning settlement, or the era's first king leg — the
    # trainer cannot substitute another checkpoint of the same generator
    # within an era. ``era_index``: the era ``king_pointer`` was adopted in.
    # All drop-when-default in the state file.
    king_since_block: int | None = None
    king_pointer: str = ""
    era_index: int | None = None
    # CONSENSUS: the ``tenure_blocks_from_block`` gate at which this
    # validator imputed ``king_since_block`` from the ``tenure_rounds``
    # counter (era.legacy_king_anchor). At the first settlement past the
    # gate EVERY validator re-anchors the reigning king from the counter —
    # also one that recorded the real crowning block before the gate — so
    # the fleet counts tenure from ONE anchor whether a validator upgraded
    # before or after that king's crowning. 0 = not yet; drop-when-default.
    tenure_anchor_gate: int = 0
    # Loop bookkeeping beside the handled marker (all drop-when-default):
    # ``last_handled_block``: the handled manifest's ``created_block`` — the
    # handled position never moves backwards (an old signed manifest
    # re-published as latest is skipped, never recorded). ``reject_config_id``:
    # the validator config identity a REJECT of the last handled round was
    # latched under ("" = that round was not a reject); a different identity at
    # a later poll re-judges it. ``pre_round``: ``{"round_id", "state",
    # "cascade"}`` — this state (and the Cascade state) as they were BEFORE the
    # last handled round was applied, so a re-judgement of that round
    # (re-publish, changed config, unfinished follow-up after a restart)
    # REPLACES its effects instead of stacking them. ``followup_round_id``: the
    # handled round whose weights / receipt / Cascade step have not completed
    # yet; the next poll finishes it before anything newer is judged.
    last_handled_block: int | None = None
    reject_config_id: str = ""
    pre_round: dict | None = None
    followup_round_id: str | None = None
    # DEC-CA-0055 lane-A economics (v2 eras, ``[v2] king_ladder``; all
    # drop-when-default, so the ``pre_round`` snapshot carries them and a
    # re-judged settlement replaces — never stacks — a ladder step):
    # ``king_share_bp`` — the reigning king's ladder share of lane A in basis
    # points (a duel crowning sets its margin-tiered start share,
    # :func:`cascade.shared.emission.crown_start`; None = a DEC-CA-0048
    # successor not moved yet, read as ``[v2] ladder_start_level_bp``). ``court_hotkey`` / ``court_until_block`` — the IMMEDIATE
    # former king and the block its 5 % of lane A stops (dethroning boundary
    # + COURT_BLOCKS). ``king_slashed`` — the last settlement paid the king
    # ``key_slash_bp`` (the re-assert repeats that settlement's vector).
    # ``king_key_lapses`` / ``king_lapse_open`` / ``king_slash_until_block`` —
    # the reign's escalating key-slash state
    # (:func:`cascade.shared.emission.key_slash_step`; reset by a crowning).
    # ``king_check_pass`` — the last PASSING king check this validator judged
    # (``{"round_id", "block", "king_pointer", "king_hotkey"}``): the only
    # evidence a v2 promotion is accepted on. ``epoch_init`` — ``{"epoch",
    # "init"}``: the init the current v2 seed epoch trains from (carried
    # across crownings: an epoch outlives the reign that started it).
    king_share_bp: int | None = None
    court_hotkey: str | None = None
    court_until_block: int = 0
    king_slashed: bool = False
    king_key_lapses: int = 0
    king_lapse_open: bool = False
    king_slash_until_block: int = 0
    king_check_pass: dict | None = None
    epoch_init: dict | None = None


def _loop_fields(state: ChampionState) -> dict:
    """The loop-position fields every fresh-state constructor carries over
    (the throne is reset, not the loop's place in the manifest stream) —
    plus the v2 seed epoch's init, which belongs to the era stream, not the
    reign."""
    return {
        "last_handled_round_id": state.last_handled_round_id,
        "last_handled_manifest_sha": state.last_handled_manifest_sha,
        "last_handled_block": state.last_handled_block,
        "reject_config_id": state.reject_config_id,
        "pre_round": state.pre_round,
        "followup_round_id": state.followup_round_id,
        "epoch_init": state.epoch_init,
    }


@dataclass(frozen=True)
class StateTransition:
    state: ChampionState
    dethroned: bool
    new_king_hotkey: str | None
    note: str


def genesis(king_hotkey: str, king_uid: int) -> ChampionState:
    """Seed the throne with an initial king (the owner baseline or first miner)."""
    return ChampionState(king_hotkey=king_hotkey, king_uid=king_uid)


def apply_forfeit(
    state: ChampionState, *, forfeited: frozenset[str] | set[str], keep_former_kings: int = 0,
    successor: str | None = None, crowned_block: int | None = None,
) -> StateTransition | None:
    """DEC-CA-0048: strip forfeited hotkeys from the throne and the court.

    A forfeited KING abdicates to ``successor`` when one is named (``[scoring]
    forfeit_successor_hotkey``, never itself forfeited), else to the most
    recent former king that is not forfeited; with neither the throne is
    EMPTY until the next duel decides one. The new king starts a fresh reign:
    tenure and streaks reset, ``king_since_block`` = ``crowned_block`` (the
    forfeiture boundary), the era king pointer is cleared so the successor's
    first king leg is adopted (:meth:`_adopt_era_king`), and ``king_uid`` is
    re-resolved by the vote path. Forfeited former kings simply leave the
    court; a named successor already in the court leaves it too (it is the
    king now). Every other field — loop position, re-entry guards — is kept.
    Returns ``None`` when nothing changes — the caller neither persists nor
    logs. Pure; consensus-critical: every validator applies the same list,
    successor and block or weights fork."""
    forfeited = frozenset(forfeited)
    if not forfeited:
        return None
    succ = str(successor or "").strip()
    if succ in forfeited:
        succ = ""
    court = tuple(hk for hk in state.former_kings if hk not in forfeited)
    if state.king_hotkey is not None and state.king_hotkey in forfeited:
        if succ:
            new_king: str | None = succ
            rest = tuple(hk for hk in court if hk != succ)
        else:
            new_king = court[0] if court else None
            rest = court[1:] if court else ()
        new = replace(
            state, king_hotkey=new_king, king_uid=None, tenure_rounds=0, streaks={},
            former_kings=rest[:max(0, keep_former_kings)],
            resync_holds=0, last_resync_round_id=None,
            king_since_block=(int(crowned_block) if crowned_block is not None and new_king else None),
            king_pointer="", era_index=None,
            # DEC-CA-0055: a fresh reign — the ladder restarts, no check
            # carries over, and a forfeited hotkey is never the paid court
            king_share_bp=None, king_check_pass=None, king_slashed=False,
            king_key_lapses=0, king_lapse_open=False, king_slash_until_block=0,
            court_hotkey=(None if state.court_hotkey in forfeited else state.court_hotkey),
            court_until_block=(0 if state.court_hotkey in forfeited
                               else state.court_until_block),
        )
        return StateTransition(
            state=new, dethroned=True, new_king_hotkey=new_king,
            note=f"forfeit:{state.king_hotkey[:12]}→{(new_king or 'vacant')[:12]}")
    if state.court_hotkey and state.court_hotkey in forfeited:
        state = replace(state, court_hotkey=None, court_until_block=0)
        return StateTransition(state=replace(state, former_kings=court), dethroned=False,
                               new_king_hotkey=state.king_hotkey, note="forfeit:court")
    if court != state.former_kings:
        return StateTransition(state=replace(state, former_kings=court), dethroned=False,
                               new_king_hotkey=state.king_hotkey, note="forfeit:court")
    return None


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
    crowned_block: int | None = None,
    crowned_pointer: str = "",
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
    round (DEC-CA-0012); their streaks reset. That includes a challenger which
    cleared the margin but was not crowned: a streak is a claim on the throne and
    only one challenger can hold that claim per round, so a non-crowned clearer
    does not bank progress toward ``dethrone_cp``. Dormant at ``dethrone_cp = 1``.
    An inconclusive round leaves every streak untouched, the cohort included —
    no duel in it produced a decision.

    ``crowned_block`` / ``crowned_pointer`` (DEC-CA-0043) are recorded on a
    dethrone only: the deciding settlement's epoch boundary and the winner's
    trained pointer become ``king_since_block`` / ``king_pointer``; the era
    index carries over (the era does not end on a dethrone).
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
                **_loop_fields(state),
                king_since_block=crowned_block,
                king_pointer=str(crowned_pointer or ""),
                era_index=state.era_index,
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
    state: ChampionState, *, trained_hotkey: str, trained_uid: int,
    trained_block: int | None = None, trained_pointer: str = "",
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
        **_loop_fields(state),
        king_since_block=trained_block,
        king_pointer=str(trained_pointer or ""),
        era_index=state.era_index,
    )


def dumps(state: ChampionState) -> str:
    era_fields = {}
    if state.king_since_block is not None:
        era_fields["king_since_block"] = int(state.king_since_block)
    if state.king_pointer:
        era_fields["king_pointer"] = state.king_pointer
    if state.era_index is not None:
        era_fields["era_index"] = int(state.era_index)
    if state.tenure_anchor_gate:
        era_fields["tenure_anchor_gate"] = int(state.tenure_anchor_gate)
    if state.last_handled_block is not None:
        era_fields["last_handled_block"] = int(state.last_handled_block)
    if state.reject_config_id:
        era_fields["reject_config_id"] = state.reject_config_id
    if state.pre_round is not None:
        era_fields["pre_round"] = state.pre_round
    if state.followup_round_id is not None:
        era_fields["followup_round_id"] = state.followup_round_id
    if state.king_share_bp is not None:
        era_fields["king_share_bp"] = int(state.king_share_bp)
    if state.court_hotkey:
        era_fields["court_hotkey"] = state.court_hotkey
        era_fields["court_until_block"] = int(state.court_until_block)
    if state.king_slashed:
        era_fields["king_slashed"] = True
    if state.king_key_lapses:
        era_fields["king_key_lapses"] = int(state.king_key_lapses)
    if state.king_lapse_open:
        era_fields["king_lapse_open"] = True
    if state.king_slash_until_block:
        era_fields["king_slash_until_block"] = int(state.king_slash_until_block)
    if state.king_check_pass is not None:
        era_fields["king_check_pass"] = state.king_check_pass
    if state.epoch_init is not None:
        era_fields["epoch_init"] = state.epoch_init
    return json.dumps(
        {
            **era_fields,
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
        king_since_block=(
            int(obj["king_since_block"]) if obj.get("king_since_block") is not None else None
        ),
        king_pointer=str(obj.get("king_pointer", "") or ""),
        era_index=(int(obj["era_index"]) if obj.get("era_index") is not None else None),
        tenure_anchor_gate=int(obj.get("tenure_anchor_gate", 0) or 0),
        last_handled_block=(
            int(obj["last_handled_block"]) if obj.get("last_handled_block") is not None else None
        ),
        reject_config_id=str(obj.get("reject_config_id", "") or ""),
        pre_round=(dict(obj["pre_round"]) if isinstance(obj.get("pre_round"), dict) else None),
        followup_round_id=(
            str(obj["followup_round_id"]) if obj.get("followup_round_id") is not None else None
        ),
        king_share_bp=(int(obj["king_share_bp"]) if obj.get("king_share_bp") is not None
                       else None),
        court_hotkey=(str(obj["court_hotkey"]) if obj.get("court_hotkey") else None),
        court_until_block=int(obj.get("court_until_block", 0) or 0),
        king_slashed=bool(obj.get("king_slashed", False)),
        king_key_lapses=int(obj.get("king_key_lapses", 0) or 0),
        king_lapse_open=bool(obj.get("king_lapse_open", False)),
        king_slash_until_block=int(obj.get("king_slash_until_block", 0) or 0),
        king_check_pass=(dict(obj["king_check_pass"])
                         if isinstance(obj.get("king_check_pass"), dict) else None),
        epoch_init=(dict(obj["epoch_init"]) if isinstance(obj.get("epoch_init"), dict)
                    else None),
    )
