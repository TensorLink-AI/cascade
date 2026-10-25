"""Emission rules around the v2 go-live (CONSENSUS on the weight vector,
DEC-CA-0055).

Every rule is applied AFTER the usual lanes are built — lane A (the court
vector, ``decayed_share_vector``) and, once lane B is live, lane B (the
domain slots) — by every validator, and re-derived by the audit from the
receipt's ``emission`` record:

* ``[v2] prelaunch_burn`` — from the fleet-decided block (``[v2]
  prelaunch_burn_from_block``, written by the DEC-CA-0045 vote) every
  settlement that is not a v2 settlement sends this fraction of the WHOLE
  vector to ``burn_uid``; the rest keeps its usual split, scaled down.
* ``[v2] carryover_king_share`` — from the first v2 settlement until a king is
  crowned at a v2 settlement, lane A (the carried king and its court) is paid
  at most this fraction of the total; the excess burns.
* ``[v2] king_ladder`` — a king crowned at a v2 settlement is paid a SHARE of
  lane A's pool from the ladder (:func:`ladder_step`: the crowning starts at
  a MARGIN-TIERED share — :func:`crown_start`, owner 2026-10-20: the start_bp
  of the highest ``[v2] ladder_start_tiers`` threshold the crowning
  challenger's increment-unit LCB meets (80 / 65 / 50 / 25 % at 0.50 / 0.25 /
  0.10 / 0.01), a level-unit crowning ``[v2] ladder_start_level_bp`` (50 %) —
  then +10 % per passed king check up to 95 %, a failed check drops it to 10 %);
  the immediate former king is paid 5 % of the pool for ``COURT_BLOCKS`` from
  the boundary that dethroned it; a settlement whose king leg the operator had
  to fund (manifest ``funded_by == "operator"``) pays the king 1 % instead
  (the ladder itself is unchanged). The slash ESCALATES within a reign
  (:func:`key_slash_step`, owner 2026-10-20): the first operator-paid lapse
  holds it for at least ``[v2] key_slash_min_eras`` eras (then until a
  king-funded leg), a second lapse holds it for the rest of the reign; a
  crowning resets the count. Everything unpaid burns.

The cap, the ladder and the slash touch lane A's portion ONLY: a lane-A uid
that also holds a lane-B domain win keeps that pay untouched
(:func:`lane_parts` keeps the two lanes apart; :func:`emission_vector`
recombines them).

Pure (no chain I/O): :func:`emission_policy` decides from the config, the
settlement boundary and the ladder state; :func:`emission_vector` transforms
the lanes; :func:`emission_record` is what the receipt stamps.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from .config import BLOCK_SECONDS
from .config.v2 import LADDER_START_TIERS
from .era import era_length_blocks, v2_settlement_active

# Ladder steps in BASIS POINTS of lane A's pool — integers, so the persisted
# state and every replay agree exactly (no float drift across +0.10 steps).
# A king crowned at a v2 settlement in LEVEL units (or with no duel LCB);
# an increment-unit crowning starts at its LADDER_START_TIERS tier instead.
LADDER_START_LEVEL_BP = 5000
LADDER_STEP_BP = 1000       # each passed king check
LADDER_CAP_BP = 9500
LADDER_FAIL_BP = 1000       # a failed king check drops the share TO this
KEY_SLASH_BP = 100          # the king leg ran on the operator's key
COURT_BP = 500              # the immediate former king, for COURT_BLOCKS
# 5 days in blocks (12 s blocks: 36000), counted from the dethroning boundary.
COURT_BLOCKS = int(5 * 86400 / BLOCK_SECONDS)

# The escalating key slash (owner 2026-10-20): the FIRST operator-paid lapse
# of a reign holds the slash for at least this many eras (6 = one 3-day
# round) from the settlement that recorded it.
KEY_SLASH_MIN_ERAS = 6

CHECK_RESULTS = ("pass", "fail", "deferred")
# Why a settlement's king is at ``key_slash_bp`` ("" = not slashed): its king
# leg is operator-paid now / inside the first lapse's minimum window / a
# second (or later) lapse this reign — slashed until the reign ends.
SLASH_REASONS = ("operator_paid", "min_window", "repeat_lapse")
# The units of a crowning LCB (:func:`crown_units`): the tiers apply to
# "increment" only — a level-unit LCB is a different statistic.
CROWN_UNITS = ("increment", "level")


def ladder_params(cfg: Any = None) -> dict[str, int]:
    """The lane-A split in force (``[v2] ladder_*`` / ``key_slash_bp`` /
    ``court_*``, overlay-settable), else the constants above."""
    v2 = getattr(cfg, "v2", None)
    base = {"ladder_start_level_bp": LADDER_START_LEVEL_BP, "ladder_step_bp": LADDER_STEP_BP,
            "ladder_cap_bp": LADDER_CAP_BP, "ladder_fail_bp": LADDER_FAIL_BP,
            "key_slash_bp": KEY_SLASH_BP, "court_bp": COURT_BP, "court_blocks": COURT_BLOCKS}
    return {k: int(getattr(v2, k, d)) for k, d in base.items()} if v2 is not None else base


def ladder_start_tiers(cfg: Any = None) -> tuple[tuple[float, int], ...]:
    """``[v2] ladder_start_tiers`` (overlay-settable; descending thresholds,
    validated at load), else :data:`LADDER_START_TIERS`."""
    raw = getattr(getattr(cfg, "v2", None), "ladder_start_tiers", None) or LADDER_START_TIERS
    return tuple((float(t), int(bp)) for t, bp in raw)


def crown_units(*, margin_mode: str, has_baseline: bool, maxt: bool,
                maxt_increment: bool) -> str:
    """CONSENSUS: the units of the crowned challenger's recorded LCB.
    "increment" iff the settlement was judged in increment units (``margin_mode``
    "increment" AND the init baseline was scored) AND, when that LCB is the
    cohort's joint max-T bound (``maxt``), the bound itself was taken in
    increment units (``[scoring] cohort_maxt_increment_from_block`` reached —
    before it the joint bound is the LEVEL statistic). Else "level". The
    validator and the audit both call this."""
    if margin_mode != "increment" or not has_baseline:
        return "level"
    if maxt and not maxt_increment:
        return "level"
    return "increment"


@dataclass(frozen=True)
class CrownStart:
    """A crowning's starting ladder share (owner 2026-10-20). ``lcb`` — the
    crowned challenger's verdict LCB (None = not finite); ``units`` — one of
    :data:`CROWN_UNITS`; ``tier`` — the threshold whose start_bp applies (None
    in level units); ``below_floor`` — an increment LCB under the lowest
    threshold (cannot clear the v2 bar's 0.01 floor; a config mismatch) paid
    at the lowest tier."""
    lcb: float | None
    units: str
    tier: float | None
    start_bp: int
    below_floor: bool = False


def crown_start(lcb: float | None, units: str, cfg: Any = None) -> CrownStart:
    """CONSENSUS: the start share of a king crowned with verdict LCB ``lcb``
    in ``units``. Increment units: the start_bp of the HIGHEST threshold with
    ``lcb >= threshold`` (:func:`ladder_start_tiers`); below the lowest (or a
    non-finite LCB) ⇒ the lowest tier's, ``below_floor``. Level units: ``[v2]
    ladder_start_level_bp``. Pure — the audit re-derives it from the recorded
    LCB."""
    if units not in CROWN_UNITS:
        raise ValueError(f"crown units must be one of {CROWN_UNITS}; got {units!r}")
    x = None if lcb is None or not math.isfinite(float(lcb)) else float(lcb)
    if units == "level":
        return CrownStart(lcb=x, units=units, tier=None,
                          start_bp=ladder_params(cfg)["ladder_start_level_bp"])
    tiers = ladder_start_tiers(cfg)
    for t, bp in tiers:
        if x is not None and x >= t:
            return CrownStart(lcb=x, units=units, tier=t, start_bp=bp)
    t, bp = tiers[-1]
    return CrownStart(lcb=x, units=units, tier=t, start_bp=bp, below_floor=True)


def crown_record(crown: CrownStart) -> dict:
    """The crowning fields the ladder record stamps."""
    return {"crown_lcb": crown.lcb, "crown_units": crown.units, "crown_tier": crown.tier,
            "crown_start_bp": int(crown.start_bp),
            "crown_below_floor": bool(crown.below_floor)}


def key_slash_min_eras(cfg: Any = None) -> int:
    """``[v2] key_slash_min_eras`` (overlay-settable), else the constant."""
    return int(getattr(getattr(cfg, "v2", None), "key_slash_min_eras", KEY_SLASH_MIN_ERAS))


@dataclass(frozen=True)
class KeySlash:
    """The reign's key-slash state after one settlement (persisted on the
    validator's ChampionState, stamped on the ladder record). ``lapses`` —
    operator-paid lapses this reign (consecutive operator-paid settlements
    are ONE lapse); ``lapse_open`` — the last lapse has not been closed by a
    king-funded leg; ``slash_until_block`` — the first lapse's minimum window
    ends at this boundary (exclusive; 0 = none); ``reason`` — one of
    :data:`SLASH_REASONS`, "" = not slashed."""
    lapses: int = 0
    lapse_open: bool = False
    slash_until_block: int = 0
    reason: str = ""

    @property
    def slashed(self) -> bool:
        return bool(self.reason)


def key_slash_step(*, funded_by: str, boundary: int, lapses: int, lapse_open: bool,
                   slash_until_block: int, cfg: Any = None) -> KeySlash:
    """CONSENSUS (DEC-CA-0055, escalating slash): the key-slash state after
    the scored settlement at ``boundary`` whose manifest king entry records
    ``funded_by``, from the reign's state before it. Only ``"operator"``
    opens a lapse (``"operator_fault"`` — our side — never counts); only
    ``"king"`` closes one. The first lapse starts a window of ``[v2]
    key_slash_min_eras`` eras (the era length in force at ``boundary``); the
    second makes the slash last the rest of the reign. Slashed = operator-paid
    now, inside the window, or a repeat lapse. Pure; a crowning resets the
    state to :class:`KeySlash` () before this runs."""
    b, n, open_, until = int(boundary), int(lapses), bool(lapse_open), int(slash_until_block)
    op = str(funded_by or "") == "operator"
    if op and not open_:
        n, open_ = n + 1, True
        if n == 1:
            until = b + key_slash_min_eras(cfg) * era_length_blocks(cfg.round, b)
    elif str(funded_by or "") == "king":
        open_ = False
    reason = ("repeat_lapse" if n >= 2 else "operator_paid" if op
              else "min_window" if b < until else "")
    return KeySlash(lapses=n, lapse_open=open_, slash_until_block=until, reason=reason)


@dataclass(frozen=True)
class EmissionPolicy:
    """How one settlement's vector is shaped. Fractions: ``prelaunch_burn`` and
    ``lane_a_cap`` of the TOTAL; ``king_share`` / ``former_share`` /
    ``king_slash`` of lane A's POOL. ``king_share`` set = the v2 ladder is in
    force (lane A is paid exactly king + former, the rest burns);
    ``king_slash`` set = the carried king's lane-A weight is capped at that
    fraction (the key slash outside the ladder)."""
    prelaunch_burn: float = 0.0
    lane_a_cap: float | None = None
    king_share: float | None = None
    former_share: float = 0.0
    king_slash: float | None = None

    @property
    def inert(self) -> bool:
        return (self.prelaunch_burn <= 0.0 and self.lane_a_cap is None
                and self.king_share is None and self.king_slash is None)

    def as_record(self) -> dict:
        def _f(x):
            return None if x is None else float(x)
        return {"prelaunch_burn": float(self.prelaunch_burn),
                "lane_a_cap": _f(self.lane_a_cap),
                "king_share": _f(self.king_share),
                "former_share": float(self.former_share),
                "king_slash": _f(self.king_slash)}

    @classmethod
    def from_record(cls, rec: dict) -> EmissionPolicy:
        def _f(x):
            return None if x is None else float(x)
        return cls(prelaunch_burn=float(rec.get("prelaunch_burn", 0.0) or 0.0),
                   lane_a_cap=_f(rec.get("lane_a_cap")),
                   king_share=_f(rec.get("king_share")),
                   former_share=float(rec.get("former_share", 0.0) or 0.0),
                   king_slash=_f(rec.get("king_slash")))


@dataclass(frozen=True)
class LadderState:
    """The lane-A ladder inputs of one settlement (persisted on the
    validator's ChampionState, stamped on the receipt). ``king_share_bp``
    None = a v2 king crowned with no duel LCB (a DEC-CA-0048 successor) whose
    ladder has not moved yet (reads as ``[v2] ladder_start_level_bp``); a
    duel crowning stores its :func:`crown_start` share. ``court_until_block`` > boundary = the immediate
    former king (``court_hotkey``) is still paid. ``slashed`` = this
    settlement's king leg was operator-funded."""
    king_share_bp: int | None = None
    court_hotkey: str | None = None
    court_until_block: int = 0
    slashed: bool = False


def ladder_on(cfg: Any) -> bool:
    """``[v2] king_ladder`` (DEC-CA-0055 items 4–6) — tolerant of configs
    without the key."""
    return bool(getattr(getattr(cfg, "v2", None), "king_ladder", False))


def crowned_in_v2(cfg: Any, king_since_block: int | None) -> bool:
    """Was the current king crowned at a v2 settlement?"""
    return king_since_block is not None and v2_settlement_active(cfg, int(king_since_block))


def ladder_step(prev_bp: int | None, *, crowned: bool, check: str | None,
                cfg: Any = None, start_bp: int | None = None) -> int:
    """CONSENSUS: the king's ladder share (bp of lane A) after one settlement.
    ``crowned`` (a dethrone / new v2 king at this settlement) restarts at
    ``start_bp`` (the :func:`crown_start` share; None = ``[v2]
    ladder_start_level_bp``); else a ``"pass"`` adds 10 % (capped at 95 %), a
    ``"fail"`` drops it TO 10 %, and anything else (``"deferred"``, no check)
    leaves it."""
    lp = ladder_params(cfg)
    if crowned:
        return lp["ladder_start_level_bp"] if start_bp is None else int(start_bp)
    base = lp["ladder_start_level_bp"] if prev_bp is None else int(prev_bp)
    if check == "pass":
        return min(lp["ladder_cap_bp"], base + lp["ladder_step_bp"])
    if check == "fail":
        return lp["ladder_fail_bp"]
    return base


def court_active(ladder: LadderState | None, boundary: int) -> bool:
    return bool(ladder is not None and ladder.court_hotkey
                and int(boundary) < int(ladder.court_until_block))


def emission_policy(cfg: Any, *, boundary: int, king_since_block: int | None,
                    prelaunch_from_block: int | None = None,
                    ladder: LadderState | None = None) -> EmissionPolicy:
    """The policy for the settlement at ``boundary``. ``king_since_block`` is
    the boundary of the settlement that crowned the current king (None = a
    king crowned before the field existed, i.e. not a v2 crowning).
    ``prelaunch_from_block`` is the pre-launch burn's start (default: ``[v2]
    prelaunch_burn_from_block`` — the typed block, or the one the on-chain
    vote wrote there); 0 = not decided, no burn."""
    v2 = getattr(cfg, "v2", None)
    if v2 is None:
        return EmissionPolicy()
    if not v2_settlement_active(cfg, int(boundary)):
        start = (int(getattr(v2, "prelaunch_burn_from_block", 0) or 0)
                 if prelaunch_from_block is None else int(prelaunch_from_block or 0))
        burn = float(getattr(v2, "prelaunch_burn", 0.0) or 0.0)
        if burn > 0.0 and 0 < start <= int(boundary):
            return EmissionPolicy(prelaunch_burn=burn)
        return EmissionPolicy()
    on = ladder_on(cfg)
    slashed = bool(on and ladder is not None and ladder.slashed)
    if crowned_in_v2(cfg, king_since_block):
        if not on:
            return EmissionPolicy()
        lp = ladder_params(cfg)
        bp = lp["key_slash_bp"] if slashed else (
            lp["ladder_start_level_bp"] if ladder is None or ladder.king_share_bp is None
            else int(ladder.king_share_bp))
        former = lp["court_bp"] if court_active(ladder, boundary) else 0
        return EmissionPolicy(king_share=bp / 10000.0, former_share=former / 10000.0)
    share = float(getattr(v2, "carryover_king_share", 0.0) or 0.0)
    return EmissionPolicy(lane_a_cap=share if share > 0.0 else None,
                          king_slash=(ladder_params(cfg)["key_slash_bp"] / 10000.0
                                      if slashed else None))


def lane_parts(court_uids: Sequence[int], n_uids: int, *, king_decay: float, burn_uid: int,
               lane_b_cfg: Any = None, wins: Iterable[Any] = (),
               lane_c_cfg: Any = None, c_wins: Iterable[Any] = (),
               c_slot_frac: dict[str, float] | None = None
               ) -> tuple[list[float], list[float]]:
    """``(lane_a, lane_b)`` — the lanes of the vector KEPT APART (their sum
    is :func:`cascade.shared.lane_b.combined_weights`'s vector). ``lane_a`` is
    the court vector scaled to lane A's pool (``1 − share``, plus the unpaid
    lane-B part under ``remainder = "lane_a"``); ``lane_b`` holds each paid
    domain slot (``share / n_domains × multiplier``) and the burned lane-B
    remainder on ``burn_uid``. ``lane_b_cfg`` None = no lane B (pool 1).

    ``lane_c_cfg`` (DEC-CA-0056, only when lane C is ACTIVE at the
    settlement): lane C's ``share`` is taken from its configured source (A
    on mainnet, B in legacy configs), and lane C's paid target slots
    (``c_wins``, ``share × slot_frac[target] ×
    multiplier``) plus its burned remainder are ADDED to the second element,
    which every caller treats as the untouched non-lane-A part. ``None`` =
    exactly the two-lane result. ``c_slot_frac`` is the settlement's
    per-target fraction of lane C's share (its ``lane_c_pay`` record's
    ``slot_frac``, :func:`cascade.shared.lane_c.lane_c_split` — competition
    weight, then equal per target; review L9 — an ``mkt:`` target holds a
    slot only once MarketCast is live).

    Lane C WITHOUT lane B is refused: the launch order is A, B, then C.
    The shared allocation helper also drives receipt/audit share records."""
    from .chain import decayed_share_vector
    from .lane_c import lane_pool_shares

    court = decayed_share_vector(list(court_uids), n_uids, decay=king_decay, burn_uid=burn_uid)
    b = [0.0] * n_uids
    pool_a = 1.0
    share_b, share_c = lane_pool_shares(lane_b_cfg, lane_c_cfg)
    if lane_b_cfg is not None:
        # DEC-CA-0059: one slot per lane-B target (domains + horizon targets)
        slot = share_b / max(1, len(getattr(lane_b_cfg, "targets", None)
                                    or lane_b_cfg.domains))
        paid_b = 0.0
        for w in wins:
            amt = slot * min(1.0, max(0.0, float(w.multiplier)))
            b[int(w.uid)] += amt
            paid_b += amt
        left_b = share_b - paid_b
        pool_a = 1.0 - share_b - share_c
        if lane_b_cfg.remainder == "lane_a":
            pool_a += left_b
        elif left_b > 0:
            b[burn_uid] += left_b
    if lane_c_cfg is not None:
        from .lane_c import lane_c_vector

        c = lane_c_vector(n_uids, cfg=lane_c_cfg, wins=list(c_wins), burn_uid=burn_uid,
                          share=share_c, slot_frac=c_slot_frac)
        b = [x + y for x, y in zip(b, c, strict=True)]
    return [pool_a * w for w in court], b


def emission_vector(lane_a: Sequence[float], lane_b: Sequence[float] | None,
                    policy: EmissionPolicy, *, burn_uid: int,
                    lane_a_uids: Sequence[int] = ()) -> list[float]:
    """The vector under ``policy``. ``lane_a_uids`` are lane A's reward UIDs
    (king first; under the ladder ``[king, former king]``). The cap, the
    ladder and the slash re-shape ``lane_a`` only; ``lane_b`` is added back
    untouched; the pre-launch burn then scales the whole vector. The result
    sums to the inputs' total."""
    a = [float(w) for w in lane_a]
    n = len(a)
    if not (0 <= burn_uid < n):
        raise ValueError(f"burn_uid {burn_uid} out of range [0,{n})")
    uids = [int(u) for u in lane_a_uids if 0 <= int(u) < n and int(u) != burn_uid]
    pool = sum(a)
    if policy.king_share is not None:
        new = [0.0] * n
        if uids:
            new[uids[0]] += float(policy.king_share) * pool
        if len(uids) > 1 and uids[1] != uids[0] and policy.former_share > 0.0:
            new[uids[1]] += float(policy.former_share) * pool
        new[burn_uid] += pool - sum(new)
        a = new
    if policy.king_slash is not None and uids:
        k, top = uids[0], float(policy.king_slash) * pool
        if a[k] > top:
            a[burn_uid] += a[k] - top
            a[k] = top
    if policy.lane_a_cap is not None:
        court = list(dict.fromkeys(uids))
        paid = sum(a[u] for u in court)
        if paid > policy.lane_a_cap:
            scale = policy.lane_a_cap / paid
            for u in court:
                a[u] *= scale
            a[burn_uid] += paid - policy.lane_a_cap
    out = a if lane_b is None else [x + float(y) for x, y in zip(a, lane_b, strict=True)]
    burn = min(1.0, max(0.0, float(policy.prelaunch_burn)))
    if burn > 0.0:
        total = sum(out)
        out = [w * (1.0 - burn) for w in out]
        out[burn_uid] += burn * total
    return out


def apply_emission_policy(vec: list[float], policy: EmissionPolicy, *, burn_uid: int,
                          lane_a_uids: list[int] | tuple[int, ...] = ()) -> list[float]:
    """``vec`` under ``policy``, the whole vector read as lane A (no lane B
    kept apart — use :func:`emission_vector` with :func:`lane_parts` for a
    lane-B vector). ``lane_a_uids`` are the court's reward UIDs (king first)."""
    out = [float(w) for w in vec]
    if policy.inert:
        return out
    return emission_vector(out, None, policy, burn_uid=burn_uid, lane_a_uids=lane_a_uids)


def emission_record(policy: EmissionPolicy, *, boundary: int, king_since_block: int | None,
                    ladder: dict | None = None) -> dict:
    """The receipt's ``emission`` record: the boundary the policy was decided
    at, the crowning boundary it was decided from, the policy, and (v2
    ladder / key slash) the ladder transition (:func:`ladder_record`)."""
    out = {"boundary": int(boundary),
           "king_since_block": None if king_since_block is None else int(king_since_block),
           "policy": policy.as_record()}
    if ladder is not None:
        out["ladder"] = dict(ladder)
    return out


def ladder_record(*, prev_bp: int | None, bp: int | None, event: str,
                  state: LadderState, funded_by: str = "", boundary: int = 0,
                  key: KeySlash | None = None, prev_key: KeySlash | None = None,
                  crown: CrownStart | None = None) -> dict:
    """The ladder part of the emission record. ``event``: ``"crowned"`` (a
    dethrone at this settlement; ``crown`` stamps its LCB, units, tier and
    start share — :func:`crown_record`), ``"pass"`` / ``"fail"`` / ``"deferred"`` (a
    king check), ``"none"``, or ``"carried"`` (no v2 crowning yet — the ladder
    does not run, only the slash). ``key`` / ``prev_key``: the escalating
    key-slash state after / before this settlement (``prev_key`` None at a
    crowning, which resets it): ``slash_reason`` and ``slash_until_block``
    (None for ``repeat_lapse`` — until the reign ends) say why and until when."""
    key = key or KeySlash()
    b = int(boundary)
    out = {"prev_bp": prev_bp, "bp": bp, "event": str(event),
           "court_hotkey": state.court_hotkey,
           "court_until_block": int(state.court_until_block),
           "slashed": bool(state.slashed), "funded_by": str(funded_by or ""),
           "key_lapses": int(key.lapses), "lapse_open": bool(key.lapse_open),
           "slash_until_block": int(key.slash_until_block), "slash_reason": str(key.reason),
           # the earliest boundary the slash can lift: the window's end, or
           # None = with a king-funded leg (operator_paid past the window) /
           # never this reign (repeat_lapse) / not slashed
           "slash_until": (int(key.slash_until_block)
                           if key.reason in ("operator_paid", "min_window")
                           and int(key.slash_until_block) > b else None)}
    if crown is not None:
        out.update(crown_record(crown))
    if prev_key is not None:
        out.update(prev_key_lapses=int(prev_key.lapses),
                   prev_lapse_open=bool(prev_key.lapse_open),
                   prev_slash_until_block=int(prev_key.slash_until_block))
    return out
