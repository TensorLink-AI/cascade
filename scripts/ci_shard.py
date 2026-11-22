#!/usr/bin/env python3
"""Deterministic split of tests/unit into N CI shards (stdlib only).

Used by .github/workflows/ci.yml on pull requests. Every shard runs on the
same checkout, so every shard computes the same plan. The plan is a partition:
each collected test file goes to exactly one shard (``--verify`` asserts it).

Collection matches pytest's default ``python_files`` (test_*.py, *_test.py),
since pyproject's [tool.pytest.ini_options] does not override it. Files are
assigned longest-first (LPT bin packing) by WEIGHTS, the measured seconds of
the slowest files on a GitHub ubuntu-latest runner. Unlisted files weigh
DEFAULT_WEIGHT. Ties break on the path, so the plan has no randomness. A stale
or missing weight only unbalances the shards. It never drops a test.

  python3 scripts/ci_shard.py --shard 2 --of 4   # print shard 2's files
  python3 scripts/ci_shard.py --verify --of 4    # partition check + plan
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UNIT = ROOT / "tests" / "unit"

# Measured seconds per file (pytest --durations, ubuntu-latest), slowest only.
WEIGHTS: dict[str, float] = {}
DEFAULT_WEIGHT = 1.0


def test_files() -> list[str]:
    found = {
        p
        for pattern in ("test_*.py", "*_test.py")
        for p in UNIT.rglob(pattern)
        if p.is_file() and "__pycache__" not in p.parts
    }
    return sorted(p.relative_to(ROOT).as_posix() for p in found)


def plan(n: int) -> list[list[str]]:
    files = test_files()
    order = sorted(files, key=lambda f: (-WEIGHTS.get(f, DEFAULT_WEIGHT), f))
    shards: list[list[str]] = [[] for _ in range(n)]
    loads = [0.0] * n
    for f in order:
        i = min(range(n), key=lambda k: (loads[k], k))
        shards[i].append(f)
        loads[i] += WEIGHTS.get(f, DEFAULT_WEIGHT)
    return [sorted(s) for s in shards]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--of", type=int, required=True, help="number of shards")
    ap.add_argument("--shard", type=int, help="1-based shard index to print")
    ap.add_argument("--verify", action="store_true", help="assert the plan is a partition")
    a = ap.parse_args()
    if a.of < 1:
        ap.error("--of must be >= 1")
    shards = plan(a.of)
    if a.verify:
        files = test_files()
        flat = [f for s in shards for f in s]
        if sorted(flat) != files or len(set(flat)) != len(flat):
            print("SHARD PLAN IS NOT A PARTITION of tests/unit", file=sys.stderr)
            return 1
        if not files:
            print("no test files found under tests/unit", file=sys.stderr)
            return 1
        for i, s in enumerate(shards, 1):
            w = sum(WEIGHTS.get(f, DEFAULT_WEIGHT) for f in s)
            print(f"shard {i}/{a.of}: {len(s)} files, weight {w:.0f}")
        print(f"OK: {len(files)} test files, each in exactly one of {a.of} shards")
        return 0
    if a.shard is None or not 1 <= a.shard <= a.of:
        ap.error("--shard must be in 1..--of")
    if not shards[a.shard - 1]:
        print(f"shard {a.shard}/{a.of} is empty", file=sys.stderr)
        return 1
    print("\n".join(shards[a.shard - 1]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
