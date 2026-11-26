"""`cascade lane-b --help` carries runnable examples and points at
`cascade rules b` and the lane B guide (owner 2026-11-24: "make the miner CLI
up to date"). Every example must parse with today's parser, so the help can
never advertise a flag that does not exist.

Nothing here imports a module in PATHS at import time.
"""

from __future__ import annotations

import argparse
import shlex


def _lane_b():
    from cascade.miner.cli import build_parser

    p = build_parser()
    action = next(a for a in p._actions if isinstance(a, argparse._SubParsersAction))
    return p, action.choices["lane-b"]


def test_lane_b_help_examples_parse():
    parser, lane_b = _lane_b()
    epilog = lane_b.epilog or ""
    examples = [line.strip() for line in epilog.splitlines()
                if line.strip().startswith("cascade lane-b ")]
    assert examples, "lane-b --help has no examples"
    seen = set()
    for line in examples:
        try:
            args = parser.parse_args(shlex.split(line)[1:])
        except SystemExit as e:
            raise AssertionError(f"help example does not parse: {line!r} (exit {e.code})") from None
        seen.add(args.lane_b_cmd)
    assert {"fund", "status", "verify-repo", "test"} <= seen, seen


def test_lane_b_help_points_at_rules_and_guide():
    _, lane_b = _lane_b()
    epilog = lane_b.epilog or ""
    assert "cascade rules b" in epilog
    assert "MINER_LANE_B.md" in epilog
