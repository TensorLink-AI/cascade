"""Lane D forward-only window eligibility (LANE_D_PLAN §1.3, rev3 + rev5
Proposed design changes 1 and 2; T8; §3.2 invariant 4).

For an entry E and the king K it is judged against, the bound is

    bound = max(T_E, T_K) + 3600 * cfg.forward_margin_hours

(unix seconds; T is the reveal time). A ``pool_d/1`` window is eligible only
if, in this order (the first failing rule names the refusal):

1. ``missing_clock``: ``target_ts`` and ``first_seen_ts`` each hold
   ``cfg.horizon`` points and ``end_ts`` is set and equals ``target_ts[-1]``;
1a. ``unlisted_feed``: the window's feed (its ``feed``, else its ``source``)
   has an entry in the pool's top-level ``feeds`` block. An unlisted feed, or
   a pool with no ``feeds`` block, cannot be judged on rules 2 and 3, so it
   fails closed;
2. ``forecast_product``: the window's feed is not a forecast product;
3. ``over_lag``: the feed's measured ``mirror_lag_hours`` is
   ``<= cfg.max_mirror_lag_hours``. A feed the snapshot LISTS with
   ``mirror_lag_hours: null`` is declared unmeasured and is not a refusal
   (W6fs measures lag only where the feed provides a publish stamp; RV3
   note 7: a policy choice for the decision register);
4. both clocks after the bound, strictly, for every target point.
   ``before_margin`` when both fail, ``publish_ahead`` when only first-seen
   fails (future-stamped values the mirror saw before the bound),
   ``late_mirror`` when only data time fails (values public before T that
   reached the mirror after the bound);
5. ``absent_from_s0``: S0, the earliest ``frontier`` record whose ``as_of_ts``
   is after the bound, exists (a series absent from S0 starts its clock at
   first appearance: the first later snapshot holding it is its S0);
6. ``backfill``: ``min(target_ts) > end_ts(S0)``.

Fail closed (RV3 BLOCKING 1): rules 5 and 6 apply to every series. A series
whose ``frontier`` is absent or ``null`` has no evidence of being in S0
(change 1(a): "present in S0 with a recorded ``end_ts``"), so it is
``absent_from_s0``, exactly like an empty list. Likewise a missing ``feeds``
entry is ``unlisted_feed``, never "lag unmeasured". Missing metadata never
switches a rule off.

Only ``max(T_E, T_K)`` enters the rule, so swapping the two T values yields
identical verdicts. The snapshot-absence rule of the v2 plan is not used
here: it survives only as W1's report-only upper bound.

Light imports only: this module is on the validator package's import path.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum


class RefusalReason(StrEnum):
    """Typed refusal reasons, declared in precedence order."""

    MISSING_CLOCK = "missing_clock"
    UNLISTED_FEED = "unlisted_feed"
    FORECAST_PRODUCT = "forecast_product"
    OVER_LAG = "over_lag"
    BEFORE_MARGIN = "before_margin"
    PUBLISH_AHEAD = "publish_ahead"
    LATE_MIRROR = "late_mirror"
    ABSENT_FROM_S0 = "absent_from_s0"
    BACKFILL = "backfill"


REFUSAL_REASONS: tuple[RefusalReason, ...] = tuple(RefusalReason)


@dataclass(frozen=True)
class WindowVerdict:
    """One pool window's §1.3 verdict: ``reason`` is ``None`` iff eligible."""

    window_id: str
    eligible: bool
    reason: RefusalReason | None
    bound: float


def forward_bound(t_entry: int, t_king: int, cfg) -> float:
    """``max(T_E, T_K) + forward_margin`` in unix seconds."""
    return float(max(int(t_entry), int(t_king))) + 3600.0 * float(cfg.forward_margin_hours)


def _clocks_recorded(w: dict, horizon: int) -> bool:
    ts, seen, end = w["target_ts"], w["first_seen_ts"], w.get("end_ts")
    return (len(ts) == horizon and len(seen) == horizon and end is not None
            and int(end) == int(ts[-1]))


def _s0_end(frontier, bound: float) -> float | None:
    """``end_ts`` of S0, the earliest frontier record with ``as_of_ts > bound``."""
    after = sorted((float(a), float(e)) for a, e in frontier if float(a) > bound)
    return after[0][1] if after else None


def window_reason(w: dict, feeds: dict, bound: float, cfg) -> RefusalReason | None:
    """The first failing §1.3 rule for window ``w`` at ``bound``, or ``None``."""
    if not _clocks_recorded(w, int(cfg.horizon)):
        return RefusalReason.MISSING_CLOCK
    feed = feeds.get(w.get("feed") or w["source"])
    if not isinstance(feed, dict):
        return RefusalReason.UNLISTED_FEED
    if bool(feed.get("forecast_product", False)):
        return RefusalReason.FORECAST_PRODUCT
    lag = feed.get("mirror_lag_hours")
    if lag is not None and float(lag) > float(cfg.max_mirror_lag_hours):
        return RefusalReason.OVER_LAG
    data_ok = min(w["target_ts"]) > bound
    seen_ok = min(w["first_seen_ts"]) > bound
    if not data_ok and not seen_ok:
        return RefusalReason.BEFORE_MARGIN
    if not seen_ok:
        return RefusalReason.PUBLISH_AHEAD
    if not data_ok:
        return RefusalReason.LATE_MIRROR
    # An absent or null frontier is no evidence of S0 presence: fail closed.
    s0_end = _s0_end(w.get("frontier") or (), bound)
    if s0_end is None:
        return RefusalReason.ABSENT_FROM_S0
    if min(w["target_ts"]) <= s0_end:
        return RefusalReason.BACKFILL
    return None


def eligible_windows(pool: dict, *, t_entry: int, t_king: int, cfg) -> Iterator[WindowVerdict]:
    """One :class:`WindowVerdict` per window of ``pool`` (what
    :func:`cascade.validator.lane_d.load_pool` returns), in pool order.

    Reads ``cfg.forward_margin_hours``, ``cfg.max_mirror_lag_hours`` and
    ``cfg.horizon``. Fails closed: a series with an absent or ``null``
    ``frontier`` is ``absent_from_s0`` and a feed missing from ``feeds`` (or
    a pool with no ``feeds`` block) is ``unlisted_feed``."""
    bound = forward_bound(t_entry, t_king, cfg)
    feeds = pool.get("feeds") or {}
    for w in pool["windows"]:
        reason = window_reason(w, feeds, bound, cfg)
        yield WindowVerdict(window_id=str(w["window_id"]), eligible=reason is None,
                            reason=reason, bound=bound)


__all__ = ["REFUSAL_REASONS", "RefusalReason", "WindowVerdict", "eligible_windows",
           "forward_bound", "window_reason"]
