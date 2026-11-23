#!/usr/bin/env python3
"""Deterministic split of tests/unit into N CI shards (stdlib only).

Used by .github/workflows/ci.yml on pull requests. Every shard runs on the
same checkout, so every shard computes the same plan. The plan is a partition:
each test runs in exactly one shard (``--verify`` asserts it).

Units. A test file is one unit, except a file listed under "split" in
scripts/ci_shard_weights.json: that file is cut into one unit per top-level
test function or Test class (found with ``ast``; pytest selector
``file::name``, which runs every parametrization of that name). Split files
are the ones too slow to fit one shard whole.

Assignment. Units are placed longest-first (LPT bin packing) by the measured
seconds in ci_shard_weights.json (ubuntu-latest, pytest --durations). A unit
with no measurement weighs DEFAULT_WEIGHT, so a new file lands in the lightest
shard and moves only other unmeasured units. Every unit also costs
UNIT_OVERHEAD (import/collection). Ties break on the selector, so the plan has
no randomness. A stale or missing weight only unbalances the shards. It never
drops a test.

Collection matches pytest's defaults (python_files test_*.py / *_test.py,
python_functions test*, python_classes Test*), since pyproject's
[tool.pytest.ini_options] does not override them.

  python3 scripts/ci_shard.py --shard 2 --of 6        # print shard 2's selectors
  python3 scripts/ci_shard.py --verify --of 6         # partition check + plan
  python3 scripts/ci_shard.py --verify --of 6 --collected ids.txt
      # also check pytest's own collection (``pytest tests/unit --collect-only
      # -q`` output): every collected test is covered by exactly one unit
  python3 scripts/ci_shard.py --refresh-weights LOG...
      # rewrite ci_shard_weights.json from shard logs (--durations=0 output)
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UNIT = ROOT / "tests" / "unit"
WEIGHTS_FILE = Path(__file__).resolve().with_name("ci_shard_weights.json")

DEFAULT_WEIGHT = 0.3  # seconds for a unit with no measurement
UNIT_OVERHEAD = 0.15  # seconds of import/collection per unit
SPLIT_ABOVE = 40.0  # --refresh-weights splits files slower than this


def load_weights() -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    if not WEIGHTS_FILE.exists():
        return {}, {}
    data = json.loads(WEIGHTS_FILE.read_text())
    return dict(data.get("files", {})), {k: dict(v) for k, v in data.get("split", {}).items()}


def test_files() -> list[str]:
    found = {
        p
        for pattern in ("test_*.py", "*_test.py")
        for p in UNIT.rglob(pattern)
        if p.is_file() and "__pycache__" not in p.parts
    }
    return sorted(p.relative_to(ROOT).as_posix() for p in found)


def top_level_tests(rel: str) -> list[str]:
    """Top-level test functions and Test classes of a file, in source order."""
    tree = ast.parse((ROOT / rel).read_text(), filename=rel)
    names = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("test"):
                names.append(node.name)
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            names.append(node.name)
    return names


def units() -> dict[str, float]:
    """Every unit selector -> weight in seconds."""
    files, split = load_weights()
    out: dict[str, float] = {}
    for f in test_files():
        if f in split:
            names = top_level_tests(f)
            if not names:
                raise SystemExit(f"split file {f} has no top-level tests")
            for name in names:
                out[f"{f}::{name}"] = split[f].get(name, DEFAULT_WEIGHT) + UNIT_OVERHEAD
        else:
            out[f] = files.get(f, DEFAULT_WEIGHT) + UNIT_OVERHEAD
    return out


def plan(n: int) -> tuple[list[list[str]], list[float], dict[str, float]]:
    weights = units()
    order = sorted(weights, key=lambda u: (-weights[u], u))
    shards: list[list[str]] = [[] for _ in range(n)]
    loads = [0.0] * n
    for u in order:
        i = min(range(n), key=lambda k: (loads[k], k))
        shards[i].append(u)
        loads[i] += weights[u]
    return [sorted(s) for s in shards], loads, weights


def top_name(rest: str) -> str:
    """'test_x[a]' / 'TestC::test_y' -> the top-level name pytest selects."""
    return re.split(r"::|\[", rest, maxsplit=1)[0]


def unit_of(nodeid: str, split: dict[str, dict[str, float]]) -> str:
    f, _, rest = nodeid.partition("::")
    return f"{f}::{top_name(rest)}" if f in split else f


def verify(n: int, collected: str | None) -> int:
    shards, loads, weights = plan(n)
    flat = [u for s in shards for u in s]
    if not flat:
        print("no test files found under tests/unit", file=sys.stderr)
        return 1
    if sorted(flat) != sorted(weights) or len(set(flat)) != len(flat):
        print("SHARD PLAN IS NOT A PARTITION of tests/unit", file=sys.stderr)
        return 1
    covered = {u.partition("::")[0] for u in flat}
    if covered != set(test_files()):
        print("SHARD PLAN DOES NOT COVER EVERY TEST FILE", file=sys.stderr)
        return 1
    files, split = load_weights()
    stale = sorted(set(files) - covered) + sorted(set(split) - covered)
    if stale:
        print(f"note: {len(stale)} weight entries name no current file (harmless): {stale}")
    if collected is not None:
        ids = [
            ln.strip()
            for ln in Path(collected).read_text().splitlines()
            if ln.startswith("tests/unit/") and "::" in ln
        ]
        if not ids:
            print(f"no collected test ids in {collected}", file=sys.stderr)
            return 1
        owner = {u: i for i, s in enumerate(shards, 1) for u in s}
        missing = [i for i in ids if unit_of(i, split) not in owner]
        if missing:
            print(f"{len(missing)} COLLECTED TESTS ARE IN NO SHARD:", file=sys.stderr)
            print("\n".join(missing[:50]), file=sys.stderr)
            return 1
        hit = {unit_of(i, split) for i in ids}
        dead = sorted(u for u in owner if "::" in u and u not in hit)
        if dead:
            print(f"SPLIT SELECTORS MATCH NO COLLECTED TEST: {dead}", file=sys.stderr)
            return 1
        print(f"OK: {len(ids)} collected tests, each in exactly one shard")
    for i, (s, w) in enumerate(zip(shards, loads), 1):
        print(f"shard {i}/{n}: {len(s)} units, weight {w:.0f}s")
    print(f"OK: {len(covered)} test files ({len(flat)} units), each unit in exactly one of {n} shards")
    return 0


DURATION = re.compile(r"(\d+\.\d+)s (?:call|setup|teardown)\s+(tests/unit/\S+?\.py)::(\S+)")


def refresh(logs: list[str]) -> int:
    per_file: dict[str, float] = defaultdict(float)
    per_name: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for log in logs:
        for line in Path(log).read_text(errors="replace").splitlines():
            m = DURATION.search(line)
            if m:
                secs, f, rest = float(m.group(1)), m.group(2), m.group(3)
                per_file[f] += secs
                per_name[f][top_name(rest)] += secs
    if not per_file:
        print("no pytest --durations lines found", file=sys.stderr)
        return 1
    data = {
        "_doc": "Measured seconds per tests/unit file (sum of pytest --durations lines) "
        "for scripts/ci_shard.py. Regenerate: ci_shard.py --refresh-weights <shard logs>.",
        "files": {f: round(s, 2) for f, s in sorted(per_file.items())},
        "split": {
            f: {n: round(s, 2) for n, s in sorted(per_name[f].items())}
            for f in sorted(per_file)
            if per_file[f] > SPLIT_ABOVE
        },
    }
    WEIGHTS_FILE.write_text(json.dumps(data, indent=1) + "\n")
    print(f"wrote {WEIGHTS_FILE}: {len(per_file)} files, {sum(per_file.values()):.0f}s total")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--of", type=int, help="number of shards")
    ap.add_argument("--shard", type=int, help="1-based shard index to print")
    ap.add_argument("--verify", action="store_true", help="assert the plan is a partition")
    ap.add_argument("--collected", help="pytest --collect-only -q output to check against")
    ap.add_argument("--refresh-weights", nargs="+", metavar="LOG", help="rebuild weights")
    a = ap.parse_args()
    if a.refresh_weights:
        return refresh(a.refresh_weights)
    if a.of is None or a.of < 1:
        ap.error("--of must be >= 1")
    if a.verify:
        return verify(a.of, a.collected)
    if a.shard is None or not 1 <= a.shard <= a.of:
        ap.error("--shard must be in 1..--of")
    shards, _, _ = plan(a.of)
    if not shards[a.shard - 1]:
        print(f"shard {a.shard}/{a.of} is empty", file=sys.stderr)
        return 1
    print("\n".join(shards[a.shard - 1]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
