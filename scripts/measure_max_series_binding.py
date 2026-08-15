"""Measure whether HarvestContext.max_series binds during a tsbench_forge harvest.

``emitted`` in TsbenchForgeSource.harvest is a budget across the WHOLE catalog
(source.py documents ``max_series`` as per-source, and tsbench_forge is one
source spanning ~3k catalog entries), so when it trips, every later catalog
entry is silently dropped. This script runs the real harvest against a synced
forge mirror and reports where — if anywhere — the budget binds, in catalog
order, so the answer is a measurement rather than an inference from the forge
index.

Three passes over the same mirror (read-only; nothing is published):

  A. production-exact: per-feed cap 25 (the publish workflow's
     --max-panel-series-per-feed) + budget 10_000 — the daily publish path;
     if the budget binds here the warning fires and the run stops early.
  B. production caps, unbounded budget — the full per-entry emission curve;
     the binding index for ANY budget falls out of the cumulative sum.
  C. source-default per-feed cap 200, unbounded — the docs/EVAL_POOL.md
     manual path, where binding is far more likely.

Output: landmark summary on stdout (small enough for log tails) plus the full
per-entry table as JSON at --out for artifact upload.

Usage: TSFORGE_DIR=./tsforge python scripts/measure_max_series_binding.py --out report.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path

from cascade.pool.source import HarvestContext
from cascade.pool.sources.tsbench_forge import TsbenchForgeSource

# Mirror the production publish context (publish-pool.yml + chain.toml [eval]).
PROD_CONTEXT = dict(span_days=210, context_length=4096, horizon=64)
PROD_BUDGET = 10_000
PROD_PER_FEED_CAP = 25
DEFAULT_PER_FEED_CAP = 200
UNBOUNDED = 10**9


def catalog_order(forge_dir: Path) -> list[str]:
    import yaml

    with open(forge_dir / "sources.yaml", encoding="utf-8") as f:
        return [e["id"] for e in (yaml.safe_load(f) or []) if e.get("id")]


def run_pass(forge_dir: Path, *, per_feed_cap: int, budget: int, as_of: dt.date) -> Counter:
    """Consume the real harvest; return emitted-series counts per catalog id."""
    src = TsbenchForgeSource(forge_dir, max_series_per_source=per_feed_cap)
    ctx = HarvestContext(as_of=as_of, max_series=budget, **PROD_CONTEXT)
    counts: Counter = Counter()
    for hs in src.harvest(lambda url, params=None: None, ctx):
        counts[hs.source] += 1
    return counts


def landmarks(order: list[str], counts: Counter, budget: int) -> dict:
    """Where a global budget of ``budget`` would trip on this emission curve."""
    total = 0
    for idx, sid in enumerate(order):
        n = counts.get(sid, 0)
        if total + n > budget:
            # The (budget+1)-th series would come from this entry: the harvest
            # returns mid-entry here and drops everything after.
            dropped_entries = sum(1 for s in order[idx + 1 :] if counts.get(s, 0) > 0)
            dropped_series = (total + n - budget) + sum(
                counts.get(s, 0) for s in order[idx + 1 :]
            )
            return {
                "binds": True,
                "index": idx,
                "source_id": sid,
                "emitted_before_entry": total,
                "dropped_contributing_entries_after": dropped_entries,
                "dropped_series": dropped_series,
            }
        total += n
    return {"binds": False, "total_emitted": total}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--forge-dir", type=Path, default=Path(os.environ.get("TSFORGE_DIR", "./tsforge")))
    ap.add_argument("--out", type=Path, default=Path("max_series_binding_report.json"))
    ap.add_argument("--as-of", default=None, help="YYYY-MM-DD (default: today, like the publish job)")
    ap.add_argument("--new-source-start-index", type=int, default=2392,
                    help="Catalog index where the 2026-08-13 additions begin (measured from forge git history)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    as_of = dt.date.fromisoformat(args.as_of) if args.as_of else dt.date.today()
    order = catalog_order(args.forge_dir)
    print(f"catalog entries: {len(order)}; as_of={as_of}; "
          f"new-source tail starts at index {args.new_source_start_index}", flush=True)

    report: dict = {"as_of": as_of.isoformat(), "catalog_entries": len(order),
                    "new_source_start_index": args.new_source_start_index, "passes": {}}

    # A: production-exact (budget live — the warning fires here if it binds).
    print("\n== pass A: per-feed cap 25, budget 10_000 (production publish path)", flush=True)
    counts_a = run_pass(args.forge_dir, per_feed_cap=PROD_PER_FEED_CAP, budget=PROD_BUDGET, as_of=as_of)
    total_a = sum(counts_a.values())
    last_sid = next((sid for sid in reversed(order) if counts_a.get(sid, 0) > 0), None)
    print(f"pass A emitted {total_a} series from {len(counts_a)} catalog entries; "
          f"last emitting entry: {last_sid} (index {order.index(last_sid) if last_sid else -1})", flush=True)
    report["passes"]["A_production"] = {
        "per_feed_cap": PROD_PER_FEED_CAP, "budget": PROD_BUDGET,
        "emitted": total_a, "contributing_entries": len(counts_a),
        "last_emitting_index": order.index(last_sid) if last_sid else None,
        "budget_hit": total_a >= PROD_BUDGET,
    }

    # B: production caps, unbounded — the full curve.
    print("\n== pass B: per-feed cap 25, unbounded budget (full emission curve)", flush=True)
    counts_b = run_pass(args.forge_dir, per_feed_cap=PROD_PER_FEED_CAP, budget=UNBOUNDED, as_of=as_of)
    lb = landmarks(order, counts_b, PROD_BUDGET)
    tail_series = sum(n for sid, n in counts_b.items()
                      if order.index(sid) >= args.new_source_start_index)
    tail_entries = sum(1 for sid, n in counts_b.items()
                       if n > 0 and order.index(sid) >= args.new_source_start_index)
    print(f"pass B emitted {sum(counts_b.values())} series total from "
          f"{len(counts_b)} entries; budget-10k landmark: {lb}", flush=True)
    print(f"new-source tail (index >= {args.new_source_start_index}): "
          f"{tail_series} series from {tail_entries} entries", flush=True)
    report["passes"]["B_uncapped"] = {
        "per_feed_cap": PROD_PER_FEED_CAP, "emitted": sum(counts_b.values()),
        "contributing_entries": len(counts_b), "budget_10k": lb,
        "new_source_tail_series": tail_series, "new_source_tail_entries": tail_entries,
    }

    # C: source-default per-feed cap (manual build path), unbounded.
    print("\n== pass C: per-feed cap 200 (source default), unbounded budget", flush=True)
    counts_c = run_pass(args.forge_dir, per_feed_cap=DEFAULT_PER_FEED_CAP, budget=UNBOUNDED, as_of=as_of)
    lc = landmarks(order, counts_c, PROD_BUDGET)
    print(f"pass C emitted {sum(counts_c.values())} series total from "
          f"{len(counts_c)} entries; budget-10k landmark: {lc}", flush=True)
    report["passes"]["C_default_cap"] = {
        "per_feed_cap": DEFAULT_PER_FEED_CAP, "emitted": sum(counts_c.values()),
        "contributing_entries": len(counts_c), "budget_10k": lc,
    }

    report["per_entry"] = [
        {"index": i, "id": sid,
         "prod_cap25": counts_b.get(sid, 0), "cap200": counts_c.get(sid, 0)}
        for i, sid in enumerate(order)
    ]
    args.out.write_text(json.dumps(report, indent=1))
    print(f"\nfull per-entry table written to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
