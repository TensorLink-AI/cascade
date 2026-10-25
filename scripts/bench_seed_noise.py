#!/usr/bin/env python3
"""Calibrate the DEC-CA-0061 rule-v2 bench gate against training-seed noise.

REPORT-ONLY. Nothing here is on a consensus path; it reads published bench
rows / reports and prints statistics.

Two checkpoints trained with the SAME recipe from the SAME init and differing
only in the training seed are, by construction, equally good: any difference
between them on the public suites is seed noise (bench forecasts of a
9-quantile checkpoint are deterministic, so it is training noise across
tasks, not sampling). This tool runs every ORDERED pair of the given
replicas through the gate exactly as a validator would
(:func:`cascade.eval.bench_gate.evaluate_bench_gate`, rule
"avg_noninf_winrate") and the dropped 2026-10-19 strict rule
(:func:`cascade.eval.bench_gate.strict_eight`), and prints:

(a) how often seed noise alone flips an old strict comparison (per number,
    and the share of ordered pairs the strict rule would have blocked);
(b) how often seed noise alone fails the new rule, per condition. NOTE: the
    AVERAGE condition fails about half the ordered pairs by construction
    (of A-vs-B and B-vs-A at most one has a ratio below 1) — it is the
    point estimate, the private pool supplies the significance. The two
    conditions to calibrate are the REGRESSION one (``tolerance``) and the
    WIN RATE one;
(c) the distribution of the per-suite ratio LCBs (``1 - ucb`` at alpha/4)
    and the smallest ``tolerance`` under which no seed pair would have been
    flagged as a significant regression.

Owner item before arming DEC-CA-0061 on mainnet: run it on >= 3 seed
replicas of the v2 recipe and set ``[scoring] bench_gate_tolerance`` above
the observed seed-noise LCB excess (with margin).

Inputs (two or more), each one checkpoint:

* ``rows.json`` — a published bench-rows document (``benchmarks/rows/...``:
  ``{suite: {"status", "rows": [{"item", "crps", "mase"}]}}``). Its
  suite-level numbers (for the strict comparison) are the geomean of its
  task CRPS / MASE ("rows-derived");
* ``report.json::POINTER`` — a signed bench report and the entry's trained
  pointer; the rows are read from the entry's ``rows_key`` (sha256-checked)
  under ``--rows-root`` (default: the report's directory), and the
  suite-level numbers are the entry's signed scores;
* either form prefixed ``store:`` (``store:benchmarks/rows/x.json``,
  ``store:benchmarks/round-7.json::POINTER``) reads the object from the
  manifest bucket of ``--config`` (anonymous-first, the audit's reader).

Usage:
    uv run python scripts/bench_seed_noise.py seed1.json seed2.json seed3.json
    uv run python scripts/bench_seed_noise.py --json \\
        store:benchmarks/round-7.json::metro-v1:trained:... store:...
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from cascade.eval.bench_gate import (
    DEFAULT_MIN_WIN_RATE,
    DEFAULT_TOLERANCE,
    RULE_V2,
    SCORE_PREFIX,
    SUITES_V2,
    evaluate_bench_gate,
    strict_eight,
)

DEFAULT_ALPHA = 0.05
DEFAULT_B = 1000
DEFAULT_MIN_ITEMS = 10


# ── pure functions ───────────────────────────────────────────────────────────


def rows_suite_scores(rows: Mapping[str, Mapping]) -> dict[str, float]:
    """Rows-derived suite-level numbers: per suite the geomean of the usable
    (finite, > 0) task CRPS and MASE, under the BenchScores keys
    (``gifteval_crps`` …). A suite with no usable value is left out."""
    out: dict[str, float] = {}
    for suite in SUITES_V2:
        block = rows.get(suite) or {}
        if block.get("status") != "ok":
            continue
        for m in ("crps", "mase"):
            vals = [float(r[m]) for r in block.get("rows") or ()
                    if isinstance(r.get(m), (int, float)) and not isinstance(r.get(m), bool)
                    and math.isfinite(float(r[m])) and float(r[m]) > 0]
            if vals:
                out[f"{SCORE_PREFIX[suite]}_{m}"] = math.exp(
                    math.fsum(math.log(v) for v in vals) / len(vals))
    return out


def pair_stats(king: Mapping, chal: Mapping, *, alpha: float = DEFAULT_ALPHA,
               B: int = DEFAULT_B, seed: str = "seed-noise",
               min_items: int = DEFAULT_MIN_ITEMS, tolerance: float = DEFAULT_TOLERANCE,
               min_win_rate: float = DEFAULT_MIN_WIN_RATE) -> dict:
    """One ordered pair (``king`` = reference, ``chal`` = challenger), each
    ``{"name", "rows", "scores"}``: the new rule's three numbers, the
    per-condition failures, the per-suite ratio LCBs, and the old strict
    outcome (``strict`` None when a suite-level number is missing)."""
    g = evaluate_bench_gate(king["rows"], chal["rows"], king.get("scores"),
                            chal.get("scores"), alpha=alpha, B=B, seed=seed,
                            min_items=min_items, rule=RULE_V2, tolerance=tolerance,
                            min_win_rate=min_win_rate)
    strict = strict_eight(king.get("scores"), chal.get("scores"))
    out = {"king": king.get("name", ""), "chal": chal.get("name", ""),
           "computed": g.computed, "passed": g.passed, "reason": g.reason,
           "strict": strict,
           "strict_passed": None if strict is None else all(strict.values())}
    if not g.computed:
        return out
    lcb = {s: float(g.suites[s]["ratio_lcb"]) for s in SUITES_V2}
    out.update(avg_ratio=g.avg_ratio, worst_lcb=g.worst_lcb, win_rate=g.win_rate,
               suite_lcb=lcb,
               fail_average=not g.avg_ratio < 1.0,
               fail_regression=any(v > 1.0 + tolerance for v in lcb.values()),
               fail_win_rate=not g.win_rate >= min_win_rate)
    return out


def _quantiles(vals: Sequence[float]) -> dict[str, float]:
    v = sorted(vals)
    if not v:
        return {}
    p90 = v[min(len(v) - 1, max(0, math.ceil(0.9 * len(v)) - 1))]
    return {"min": v[0], "p50": statistics.median(v), "p90": p90, "max": v[-1]}


def summarize(results: Sequence[Mapping]) -> dict:
    """Aggregate :func:`pair_stats` rows: (a) strict flip rates, (b) new-rule
    failure rates per condition, (c) the per-suite ratio-LCB distribution
    and the smallest tolerance no seed pair would exceed."""
    done = [r for r in results if r.get("computed")]
    strict = [r for r in results if r.get("strict") is not None]
    keys = sorted({k for r in strict for k in r["strict"]})
    n = len(done)
    lcbs = {s: [r["suite_lcb"][s] for r in done] for s in SUITES_V2}
    all_lcb = [x for v in lcbs.values() for x in v]
    return {
        "pairs": len(results), "computed": n,
        "uncomputable": [f"{r['king']} vs {r['chal']}: {r['reason']}"
                         for r in results if not r.get("computed")],
        "strict": {
            "pairs": len(strict),
            "blocked_share": (sum(1 for r in strict if not r["strict_passed"]) / len(strict)
                              if strict else float("nan")),
            "flip_rate": {k: sum(1 for r in strict if not r["strict"][k]) / len(strict)
                          for k in keys},
        },
        "new_rule": {
            "fail_share": (sum(1 for r in done if not r["passed"]) / n) if n else float("nan"),
            **{f"fail_{c}": (sum(1 for r in done if r[f"fail_{c}"]) / n) if n else float("nan")
               for c in ("average", "regression", "win_rate")},
            "win_rate": _quantiles([r["win_rate"] for r in done]),
        },
        "ratio_lcb": {s: _quantiles(v) for s, v in lcbs.items()},
        "min_clean_tolerance": max(0.0, max(all_lcb) - 1.0) if all_lcb else float("nan"),
    }


def ordered_pairs(checkpoints: Sequence[Mapping], **kw) -> list[dict]:
    """:func:`pair_stats` for every ordered pair of distinct replicas."""
    return [pair_stats(a, b, **kw) for a, b in itertools.permutations(checkpoints, 2)]


# ── inputs ───────────────────────────────────────────────────────────────────


def load_checkpoint(ref: str, read_text: Callable[[str], str], *,
                    rows_root: str | None = None) -> dict:
    """``{"name", "rows", "scores"}`` for one input reference (module
    docstring). ``read_text`` reads a path / store key."""
    from cascade.shared.bench_report import load_bench_report, load_bench_rows

    if "::" not in ref:
        rows = json.loads(read_text(ref))
        return {"name": ref, "rows": rows, "scores": rows_suite_scores(rows)}
    report_ref, pointer = ref.split("::", 1)
    report = load_bench_report(read_text(report_ref))
    e = report.entry_for_pointer(pointer)
    if e is None:
        raise ValueError(f"{report_ref}: no entry for pointer {pointer!r}")
    if not e.rows_key:
        raise ValueError(f"{report_ref}: entry {pointer!r} carries no rows_key")
    if rows_root is not None:
        key = str(Path(rows_root) / e.rows_key)
    elif report_ref.startswith("store:"):
        key = "store:" + e.rows_key
    else:
        key = str(Path(report_ref).parent / e.rows_key)
    import dataclasses

    scores = {k: v for k, v in dataclasses.asdict(e.scores).items() if v is not None}
    return {"name": f"{report_ref}::{pointer[-16:]}",
            "rows": load_bench_rows(read_text(key), e.rows_sha256), "scores": scores}


def _reader(config: str | None) -> Callable[[str], str]:
    bucket = None

    def read(ref: str) -> str:
        nonlocal bucket
        if not ref.startswith("store:"):
            return Path(ref).read_text(encoding="utf-8")
        if bucket is None:
            from cascade.audit.main import _BucketReader
            from cascade.shared.config import load_chain_config

            bucket = _BucketReader(load_chain_config(config or "chain.toml"))
        return bucket.get_text(ref.removeprefix("store:"))

    return read


def _fmt(x) -> str:
    return "n/a" if not isinstance(x, (int, float)) or not math.isfinite(x) else f"{x:.4f}"


def print_report(summary: Mapping, *, tolerance: float, min_win_rate: float,
                 out=sys.stdout) -> None:
    p = print
    p(f"seed-noise calibration: {summary['pairs']} ordered pairs "
      f"({summary['computed']} computable)", file=out)
    for u in summary["uncomputable"]:
        p(f"  uncomputable: {u}", file=out)
    st = summary["strict"]
    p(f"\n(a) old strict rule (8 comparisons): blocks {_fmt(st['blocked_share'])} of "
      f"{st['pairs']} pairs", file=out)
    for k, v in st["flip_rate"].items():
        p(f"    {k:<14} not strictly better in {_fmt(v)}", file=out)
    nr = summary["new_rule"]
    p(f"\n(b) new rule {RULE_V2} (tolerance {tolerance}, min win rate {min_win_rate}): "
      f"fails {_fmt(nr['fail_share'])}", file=out)
    p(f"    average >= 1     {_fmt(nr['fail_average'])}  (~0.5 expected by construction)",
      file=out)
    p(f"    regression       {_fmt(nr['fail_regression'])}  <- calibrate tolerance", file=out)
    p(f"    win rate < min   {_fmt(nr['fail_win_rate'])}", file=out)
    q = nr["win_rate"]
    if q:
        p("    win rate dist    " + "  ".join(f"{k} {_fmt(v)}" for k, v in q.items()), file=out)
    p("\n(c) per-suite ratio LCB (alpha/4):", file=out)
    for s, qq in summary["ratio_lcb"].items():
        p(f"    {s:<5} " + "  ".join(f"{k} {_fmt(v)}" for k, v in qq.items()), file=out)
    p(f"\n    smallest tolerance no seed pair exceeds: "
      f"{_fmt(summary['min_clean_tolerance'])}", file=out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("inputs", nargs="+", help="rows.json | report.json::POINTER | store:…")
    ap.add_argument("--rows-root", default=None,
                    help="directory the report entries' rows_key resolve under")
    ap.add_argument("--config", default=None, help="chain toml for store: reads")
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    ap.add_argument("--B", type=int, default=DEFAULT_B)
    ap.add_argument("--seed", default="seed-noise")
    ap.add_argument("--min-items", type=int, default=DEFAULT_MIN_ITEMS)
    ap.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    ap.add_argument("--min-win-rate", type=float, default=DEFAULT_MIN_WIN_RATE)
    ap.add_argument("--json", action="store_true", help="print the pairs + summary as JSON")
    args = ap.parse_args(argv)
    if len(args.inputs) < 2:
        ap.error("need two or more seed replicas")
    read = _reader(args.config)
    cks = [load_checkpoint(r, read, rows_root=args.rows_root) for r in args.inputs]
    pairs = ordered_pairs(cks, alpha=args.alpha, B=args.B, seed=args.seed,
                          min_items=args.min_items, tolerance=args.tolerance,
                          min_win_rate=args.min_win_rate)
    summary = summarize(pairs)
    if args.json:
        print(json.dumps({"pairs": pairs, "summary": summary}, indent=2, default=str))
    else:
        print_report(summary, tolerance=args.tolerance, min_win_rate=args.min_win_rate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
