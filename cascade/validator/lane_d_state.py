"""``lane_d_state.json`` (LANE_D_PLAN §1.5 "State"; W7c deepens it).

Holds KING_D, the prior miner kings, the house, and every entry's progress
(status, typed reason, the windows it has been judged on, its confirmation
record, and ``frozen_set``: the W5b accumulating/frozen window set from
:func:`~cascade.validator.lane_d_frozen.advance_frozen_set`). Written atomically; protected like ``cascade_state.json`` (snapshot
before restarts, never hand-edited). A missing file is the genesis state: the
house holds KING_D.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from ..shared.atomic import write_text_atomic
from . import lane_d_report

STATE_SCHEMA = "lane-d-state/1"
# §1.5 typed statuses and reasons an entry can carry. One source of truth: the
# W7a report enums (``lane_d_report``, equal to the schema file), plus the
# reasons state can hold before the report schema lists them.
ENTRY_STATUSES = tuple(lane_d_report.ENTRY_STATUSES)
# ``lane_closed`` (T18, W5b): every open entry closes at ``[lane_d] until_block``.
STATE_ONLY_REASONS = ("lane_closed",)
ENTRY_REASONS = tuple(lane_d_report.ENTRY_REASONS) + tuple(
    r for r in STATE_ONLY_REASONS if r not in lane_d_report.ENTRY_REASONS)


def genesis(house: dict) -> dict:
    """The state before any lane D settlement: the house holds KING_D."""
    h = {"digest": str(house["digest"]), "reveal_ts": int(house["reveal_ts"])}
    return {"schema": STATE_SCHEMA, "boundary": 0, "house": dict(h),
            "king": {**h, "hotkey": None, "house": True},
            "prior_kings": [], "entries": {}}


def load_state(path: str | os.PathLike, house: dict) -> dict:
    p = Path(path)
    if not p.is_file():
        return genesis(house)
    state = json.loads(p.read_text(encoding="utf-8"))
    if state.get("schema") != STATE_SCHEMA:
        raise ValueError(f"{p}: unknown lane D state schema {state.get('schema')!r}")
    return state


def save_state(path: str | os.PathLike, state: dict) -> None:
    write_text_atomic(path, json.dumps(state, sort_keys=True, indent=1, allow_nan=False))


def refresh_house(state: dict, house: dict) -> None:
    """§1.4: the house follows lane A's king digest. While the house holds
    KING_D the throne moves with it; a miner KING_D keeps its throne."""
    h = {"digest": str(house["digest"]), "reveal_ts": int(house["reveal_ts"])}
    if state["house"]["digest"] == h["digest"]:
        return
    state["house"] = dict(h)
    if state["king"].get("house"):
        state["king"] = {**h, "hotkey": None, "house": True}


__all__ = ["ENTRY_REASONS", "ENTRY_STATUSES", "STATE_ONLY_REASONS", "STATE_SCHEMA", "genesis",
           "load_state", "refresh_house", "save_state"]
