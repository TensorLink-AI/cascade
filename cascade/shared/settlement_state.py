"""The settlement state a receipt carries for satellite continuity (DEC-CA-0058).

A validator's judgement of settlement ``n`` depends on the state the previous
settlements left: the king and its tenure, the court (``former_kings``), the
DEC-CA-0055 ladder, and the lane-B / lane-C slots and pending entries. The
main validator stamps the digest of that state BEFORE the settlement
(``state_prior_sha256``) and the state AFTER it (``state_post``) on its
receipt; a satellite checks every receipt's prior digest against the
previous receipt's posterior, so a main validator whose state was reset or
edited between settlements is caught.

Pure and torch-free: duck-typed over :class:`cascade.validator.state.
ChampionState`, :class:`cascade.shared.lane_b.LaneBState` and
:class:`cascade.shared.lane_c.LaneCState`.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def settlement_state(champion: Any, *, lane_b: Any = None, lane_c: Any = None) -> dict:
    """The compact settlement state. ``lane_b`` / ``lane_c`` None (the lane
    is not configured) drop their keys."""
    out: dict = {
        "king": getattr(champion, "king_hotkey", None),
        "king_since_block": getattr(champion, "king_since_block", None),
        "tenure_rounds": int(getattr(champion, "tenure_rounds", 0) or 0),
        "former_kings": [str(h) for h in (getattr(champion, "former_kings", ()) or ())],
        "ladder": {
            "bp": getattr(champion, "king_share_bp", None),
            "court_hotkey": getattr(champion, "court_hotkey", None),
            "court_until_block": int(getattr(champion, "court_until_block", 0) or 0),
            "slashed": bool(getattr(champion, "king_slashed", False)),
            # DEC-CA-0055 escalating key slash (the reign's lapse state)
            "key_lapses": int(getattr(champion, "king_key_lapses", 0) or 0),
            "lapse_open": bool(getattr(champion, "king_lapse_open", False)),
            "slash_until_block": int(getattr(champion, "king_slash_until_block", 0) or 0),
        },
    }
    # Each lane-B / lane-C slot carries its gain-tiered full-pay period (owner
    # 2026-10-20: gain LCB, tier, full eras — fixed at the win), so the
    # receipt chain binds a recorded tier across settlements.
    if lane_b is not None:
        out["lane_b"] = {
            str(d): [str(w.hotkey), int(w.uid), int(w.merge_block), str(w.round_id),
                     *_full_pay_fields(w, "target_lcb")]
            for d, w in sorted((getattr(lane_b, "wins", None) or {}).items())}
    if lane_c is not None:
        out["lane_c"] = {
            "slots": {str(t): [str(w.hotkey), int(w.uid), int(w.win_block), str(w.round_id),
                               *_full_pay_fields(w, "gain_lcb")]
                      for t, w in sorted((getattr(lane_c, "slots", None) or {}).items())},
            "pending": sorted([str(p.target), str(p.kind), str(p.hotkey), str(p.round_id),
                               int(p.block)] for p in (getattr(lane_c, "pending", None) or ())),
        }
    return out


def _full_pay_fields(w: Any, lcb_attr: str) -> list:
    tier, n = getattr(w, "tier", None), getattr(w, "full_eras", None)
    return [float(getattr(w, lcb_attr, 0.0) or 0.0), None if tier is None else float(tier),
            None if n is None else int(n)]


def slot_full_pay(entry: list | None) -> dict | None:
    """``{gain_lcb, tier, full_eras}`` of one ``state_post`` slot entry (None
    when absent or untiered)."""
    if not entry or len(entry) < 7:
        return None
    return {"gain_lcb": entry[4], "tier": entry[5], "full_eras": entry[6]}


def state_digest(state: dict | None) -> str:
    """sha256 hex of the canonical JSON of ``state`` ("" for None)."""
    if state is None:
        return ""
    body = json.dumps(state, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def paid_hotkeys(state: dict | None) -> set[str]:
    """Every hotkey ``state`` may legitimately pay: the king, the court, the
    ladder's former king and each lane-B / lane-C slot holder."""
    if not state:
        return set()
    out = {str(state["king"])} if state.get("king") else set()
    out |= {str(h) for h in state.get("former_kings") or ()}
    court = (state.get("ladder") or {}).get("court_hotkey")
    if court:
        out.add(str(court))
    out |= {str(v[0]) for v in (state.get("lane_b") or {}).values()}
    out |= {str(v[0]) for v in ((state.get("lane_c") or {}).get("slots") or {}).values()}
    return out


__all__ = ["paid_hotkeys", "settlement_state", "slot_full_pay", "state_digest"]
