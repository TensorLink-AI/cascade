"""Explicit finite royalty and separate lane payout accounting, in integer units."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class RewardPolicy:
    lane_b: str
    continuation_of_a: str
    royalty_of_duel: str
    royalty_expiry: int
    royalty_history: int
    empty_royalty: str
    paused_throne: str
    # Stale king (DEC-CA-0052 §10): a king whose generator has not produced a
    # new promotion within ``stale_after`` (journal clock) has ``stale_burn`` of
    # its throne share burned. 0 disables it.
    stale_after: int = 0
    stale_burn: str = "0"

    def __post_init__(self):
        for value in (self.lane_b, self.continuation_of_a, self.royalty_of_duel, self.stale_burn):
            if not 0 <= Fraction(value) <= 1:
                raise ValueError("reward fraction outside [0,1]")
        if type(self.stale_after) is not int or self.stale_after < 0:
            raise ValueError("stale_after must be a nonnegative integer")
        if self.royalty_expiry < 0 or self.royalty_history < 0:
            raise ValueError("royalties require finite expiry/history")
        if self.empty_royalty not in ("burn", "current") or self.paused_throne not in ("burn", "current"):
            raise ValueError("explicit empty/pause payout policy required")


def allocate_boundary(policy, *, emission, boundary, king, history, paused=False, stale=None):
    """Reserve continuation/lane-B pools separately; never renormalise payouts.

    history rows carry distinct contribution IDs and original accepted boundary.
    Duplicate/re-entry rows do not extend expiry. Integer dust burns.
    ``stale`` (round journals) decides the stale-king burn; None falls back to
    ``policy.stale_after`` on the journal clock.
    """
    if type(emission) is not int or emission < 0 or boundary < 0:
        raise ValueError("invalid payout boundary")
    b = int(emission * Fraction(policy.lane_b))
    a = emission - b
    continuation = int(a * Fraction(policy.continuation_of_a))
    duel = a - continuation
    royalties = int(duel * Fraction(policy.royalty_of_duel))
    current = duel - royalties
    contributions = {}
    for row in sorted(history, key=lambda x: (x["boundary"], x["contribution"])):
        if row["boundary"] > boundary or row["boundary"] < 0:
            raise ValueError("invalid royalty provenance")
        contributions.setdefault(row["contribution"], row)
    eligible = [r for r in contributions.values() if r["payer"] != king
                and boundary - r["boundary"] < policy.royalty_expiry]
    eligible = eligible[-policy.royalty_history:] if policy.royalty_history else []
    paid, burn = {}, 0
    if eligible:
        each = royalties // len(eligible)
        for row in eligible:
            paid[row["payer"]] = paid.get(row["payer"], 0) + each
        burn += royalties - each * len(eligible)
    elif policy.empty_royalty == "current":
        current += royalties
    else:
        burn += royalties
    stale_units = 0
    if king and (not paused or policy.paused_throne == "current"):
        improved = max((r["boundary"] for r in contributions.values() if r["payer"] == king),
                       default=None)
        if stale is None:
            stale_king = bool(policy.stale_after) and (
                improved is None or boundary - improved >= policy.stale_after)
        else:
            stale_king = stale
        if stale_king:
            stale_units = int(current * Fraction(policy.stale_burn))
        paid[king] = paid.get(king, 0) + current - stale_units
        burn += stale_units
    else:
        burn += current
    assert sum(paid.values()) + burn + b + continuation == emission
    out = {"throne_royalties": paid, "burn": burn, "lane_b_pool": b,
           "continuation_pool": continuation}
    if stale_units:
        out["stale_burn"] = stale_units
    return out


def continuation_shares(rewards, paid, pool):
    """Split one boundary's continuation pool across earned, unpaid entitlements.

    Pro rata to each entitlement's outstanding amount, floored, never above it:
    an entitlement is a finite claim settled through emissions over as many
    boundaries as it takes, and a dethroned payer keeps theirs. Undistributed
    units (dust, or a pool larger than what is owed) burn.
    """
    if type(pool) is not int or pool < 0:
        raise ValueError("invalid continuation pool")
    outstanding = {job: r["amount"] - paid.get(job, 0) for job, r in sorted(rewards.items())}
    outstanding = {job: left for job, left in outstanding.items() if left > 0}
    total = sum(outstanding.values())
    if not total:
        return {}
    if total <= pool:
        return outstanding
    return {job: left * pool // total for job, left in outstanding.items() if left * pool // total}
