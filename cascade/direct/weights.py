"""Direct-MoE rewards as validator weights: emissions, never TAO transfers.

A settled payout boundary in the replayed journal names integer reward units
per hotkey (throne + finite royalties, continuation entitlements) plus burns.
Validators turn those units into the normalised weight vector they already
set for lanes A and B; a hotkey that is no longer registered forfeits its
slice to ``burn_uid`` rather than redistributing it.
"""

from __future__ import annotations


def boundary_units(state, boundary):
    """``{"hotkeys": {hotkey: units}, "burn": units, "lane_b_pool": units}``."""
    alloc = state["payout_boundaries"][str(boundary)]
    if "continuation_paid" not in alloc:
        raise ValueError("payout boundary not settled in the journal yet")
    units = dict(alloc["throne_royalties"])
    for row in alloc["continuation_paid"].values():
        units[row["payer"]] = units.get(row["payer"], 0) + row["amount"]
    return {"hotkeys": {hk: u for hk, u in sorted(units.items()) if u > 0},
            "burn": alloc["burn"] + alloc["continuation_burn"],
            "lane_b_pool": alloc["lane_b_pool"]}


def latest_settled_boundary(state):
    settled = [int(b) for b, a in state["payout_boundaries"].items() if "continuation_paid" in a]
    return str(max(settled)) if settled else None


def weight_vector(units, *, uid_of, n_uids, burn_uid, lane_b=None):
    """Normalised weights (index = uid). ``lane_b`` optionally maps the lane-B
    pool onto hotkeys; absent, the pool burns (the direct lane-B payout is not
    wired yet)."""
    if n_uids <= 0 or not 0 <= burn_uid < n_uids:
        raise ValueError("invalid metagraph size or burn uid")
    vec = [0.0] * n_uids
    total = units["burn"] + units["lane_b_pool"] + sum(units["hotkeys"].values())
    if total <= 0:
        vec[burn_uid] = 1.0
        return vec
    vec[burn_uid] += units["burn"]
    shares = dict(units["hotkeys"])
    if lane_b:
        for hk, u in lane_b.items():
            shares[hk] = shares.get(hk, 0) + u
    else:
        vec[burn_uid] += units["lane_b_pool"]
    for hotkey, u in shares.items():
        uid = uid_of(hotkey)
        vec[uid if uid is not None and 0 <= uid < n_uids else burn_uid] += u
    s = sum(vec)
    return [v / s for v in vec]
