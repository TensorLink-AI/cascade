"""Undo journal for the validator's separate local settlement state files.

Persist before any gate/catch-up can mutate a lane. Commit after the lane
folds and champion handled marker are durable, before weights or receipts.
Recovery is repeatable if interrupted. Requires a single validator writer.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..shared.atomic import write_text_atomic
from . import state as champion


def journal_path(runner) -> Path:
    return Path(str(runner.cfg.validator.state_db_path) + ".settlement.json")


def begin(runner) -> None:
    casc = None
    if runner.cascade is not None:
        from . import cascade

        casc = cascade.dumps(runner.cascade.state)
    record = {"schema": 1, "phase": "prepared", "state": champion.dumps(runner.state),
              "cascade": casc, "lane_b": runner._lane_b_snapshot(),
              "lane_c": runner._lane_c_snapshot(),
              "lane_b_block": runner._lane_b_block, "lane_c_block": runner._lane_c_block}
    write_text_atomic(journal_path(runner), json.dumps(record, allow_nan=False))


def commit(runner) -> None:
    # A durable tombstone prevents an old undo record resurrecting after a
    # power failure; unlink without a directory fsync would not guarantee it.
    write_text_atomic(journal_path(runner), '{"schema":1,"phase":"committed"}')


def recover(runner) -> bool:
    path = journal_path(runner)
    if not path.exists():
        return False
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("schema") != 1 or record.get("phase") not in ("prepared", "committed"):
        raise ValueError(f"unreadable settlement recovery journal: {path}")
    if record["phase"] == "committed":
        return False
    state = champion.loads(record["state"])
    if record.get("lane_b") is not None:
        runner._lane_b_restore(record["lane_b"])
        runner._persist_lane_b()
    if record.get("lane_c") is not None:
        from ..shared.lane_c import LaneCState

        runner._lane_c_state = LaneCState.from_json(record["lane_c"])
        runner._persist_lane_c()
    if record.get("cascade") is not None:
        from . import cascade

        if runner.cascade is None:
            raise ValueError("settlement recovery requires the recorded Cascade controller")
        runner.cascade.state = cascade.loads(record["cascade"])
        runner.cascade._persist()
    runner.state = state
    runner._lane_b_block = record.get("lane_b_block")
    runner._lane_c_block = record.get("lane_c_block")
    runner._followup = None
    runner._persist_state()
    commit(runner)
    return True
