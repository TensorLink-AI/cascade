#!/usr/bin/env python3
"""Quiet-window rules for the lane D restart scripts (plan §10.5 S5, §13).

    window.py --root ROOT --now 2026-11-16T12:00:00Z [--chain-toml FILE] [--ack RULE ...]

Read-only. Prints one ``REFUSE [<rule>]: <why>`` line per rule that blocks a
restart now, ``NOTE`` lines otherwise, and exits 0 (GO), 1 (refused) or 2
(usage). Every rule applies to every service the S5 scripts restart:

  freeze      Wed 22:00 - Thu 08:00 UTC
  clock       no ``block N`` sample in validator.log / trainer.log (or a stale one)
  settlement  within 20 min before or 15 min after a settlement boundary, or a
              settlement manifest pending publish. The grid is the chain
              config's [round] epoch_blocks (--chain-toml, else
              <root>/chain.toml; mainnet 900, testnet 150); with neither, the
              mainnet 900-block grid is assumed and a NOTE says so
  scoring     the validator has a ``new manifest round=R`` with no weights set yet
  in_flight   funded legs on pods, or open king/operator rents
  harvest     harvests awaiting verification

The log, block and window readers are deploy_gate's (scripts/deploy_gate.py in
this checkout), so the restart scripts and ``deploy_gate.py check`` place the
windows the same way. Unlike ``check``, a state file that exists but can't be
parsed refuses (a restart guard fails closed).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import deploy_gate as dg  # noqa: E402  (path set above; run with python -P)

# A block sample older than this can't place a boundary to the minute
# (12 s is the nominal block time, not a guarantee).
MAX_SAMPLE_AGE_MIN = 60.0
NOW_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def _load_json(path: Path) -> tuple[dict | None, str]:
    """(parsed object, "") ; (None, "") when absent ; (None, why) when unreadable."""
    if not path.exists():
        return None, ""
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        return None, f"{path} is unreadable ({e.__class__.__name__}) — cannot rule it out"
    if not isinstance(data, dict):
        return None, f"{path} is not a JSON object — cannot rule it out"
    return data, ""


def legs_in_flight(work: Path) -> tuple[list[str], str]:
    q, err = _load_json(work / "funded_queue.json")
    if err:
        return [], err
    out = []
    entries = (q or {}).get("entries", {})
    for e in (entries.values() if isinstance(entries, dict) else entries):
        if not isinstance(e, dict):
            continue
        if e.get("status") == "in_flight" and str(e.get("phase") or "") not in dg.POD_PHASES_FREE:
            out.append(f"{str(e.get('hotkey', '?'))[:12]} ({e.get('phase')})")
    return out, ""


def rules(root: Path, now: datetime,
          grid: dg.Grid = dg.DEFAULT_GRID) -> tuple[list[tuple[str, str]], list[str]]:
    refuse: list[tuple[str, str]] = []
    notes: list[str] = []
    if dg.freeze_window(now):
        refuse.append(("freeze", "Wed 22:00 - Thu 08:00 UTC: no restarts"))

    validator_lines = dg.tail(root / "validator.log")
    trainer_lines = dg.tail(root / "trainer.log")
    sample = dg.last_block_sample(validator_lines) or dg.last_block_sample(trainer_lines)
    if sample is None:
        refuse.append(("clock", "no `block N` line in validator.log or trainer.log — "
                                "cannot place the settlement window"))
    else:
        age = (now - sample[0]).total_seconds() / 60
        if age > MAX_SAMPLE_AGE_MIN:
            refuse.append(("clock", f"the last block sample (block {sample[1]}) is {age:.0f} min "
                                    f"old — cannot place the settlement window"))
        else:
            to_next, since = dg.minutes_to_next_settlement(sample, now, grid)
            if to_next < dg.PRE_SETTLEMENT_MIN:
                refuse.append(("settlement", f"settlement in {to_next:.0f} min (no restarts in "
                                             f"the last {dg.PRE_SETTLEMENT_MIN} min before a "
                                             "boundary)"))
            elif since < dg.POST_SETTLEMENT_MIN:
                refuse.append(("settlement", f"settlement {since:.0f} min ago — manifest publish "
                                             "and pick-up still in progress"))
            else:
                notes.append(f"next settlement in {to_next:.0f} min")

    rid = dg.validator_scoring(validator_lines)
    if rid:
        refuse.append(("scoring", f"validator is still judging round {rid} "
                                  "(wait for its weights set)"))

    work = root / "_train_work"
    legs, err = legs_in_flight(work)
    if err:
        refuse.append(("in_flight", err))
    elif legs:
        refuse.append(("in_flight", "legs on pods would be killed or re-rented: "
                                    + "; ".join(legs)))
    era, err = _load_json(work / "era_state.json")
    if err:
        refuse.append(("harvest", err))
    elif era:
        if era.get("rents"):
            refuse.append(("in_flight", f"{len(era['rents'])} king/operator rent(s) open"))
        if era.get("pending_publish"):
            refuse.append(("settlement", "a settlement manifest is pending publish"))
        if era.get("verify_queue"):
            refuse.append(("harvest", f"{len(era['verify_queue'])} harvest(s) awaiting "
                                      "verification"))
    return refuse, notes


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--now", required=True, help=f"UTC time, {NOW_FORMAT}")
    ap.add_argument("--ack", action="append", default=[])
    ap.add_argument("--chain-toml", type=Path, default=None,
                    help="chain config for the settlement grid (default <root>/chain.toml; "
                         "with neither, the mainnet 900-block grid, noted)")
    a = ap.parse_args(argv)
    try:
        now = datetime.strptime(a.now, NOW_FORMAT).replace(tzinfo=UTC)
    except ValueError:
        print(f"window.py: --now must look like {NOW_FORMAT}, got {a.now!r}", file=sys.stderr)
        return 2
    if not a.root.is_dir():
        print(f"window.py: --root {a.root} is not a directory", file=sys.stderr)
        return 2
    try:
        grid, grid_note = dg.resolve_grid(a.chain_toml, a.root)
    except dg.GridError as e:
        print(f"window.py: {e}", file=sys.stderr)
        return 2
    refuse, notes = rules(a.root, now, grid)
    notes.insert(0, grid_note)
    acks = set(a.ack)
    for n in notes:
        print(f"NOTE {n}")
    for rule, why in refuse:
        if rule in acks:
            print(f"NOTE acked [{rule}] by the owner: {why}")
    blocking = [(r, w) for r, w in refuse if r not in acks]
    for rule, why in blocking:
        print(f"REFUSE [{rule}]: {why}")
    return 1 if blocking else 0


if __name__ == "__main__":
    sys.exit(main())
