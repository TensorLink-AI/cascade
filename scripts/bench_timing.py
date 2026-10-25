#!/usr/bin/env python3
"""Time the public-benchmark sweep sequential vs concurrent (report-only).

Runs the ``benchmarks/`` sidecar on ONE checkpoint once with ``--workers 1``
and once per other worker setting (``auto`` and/or explicit counts), prints the
wall time per suite and the speedup over the sequential run, and FAILS (exit 1)
unless every report is byte-identical to the sequential one — the concurrent
sweep's contract (``benchmarks/cascade_benchmark/parallel.py``).

Nothing here is on a consensus path. Use it on the main validator's sweep box
(or the eval-offload pod) to measure what the DEC-CA-0063 sweep really costs,
then re-set ``[v2] bench_sweep_hours`` / ``bench_budget_hours`` from the
measured times (docs/OPERATOR.md "Sizing the sweep").

    python scripts/bench_timing.py /path/to/checkpoint --data-dir /data/bench \\
        --device cuda --workers auto,4 [--suites gift-eval,boom,time,fev] \\
        [--batch-size 512] [--max-series 0] [--extra-checkpoint /path/to/king]

``--extra-checkpoint`` times the PAIRED pass instead (both checkpoints in one
sweep, as the validator runs challenger + king); identity is checked for
every report of the pass.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def parse_worker_list(text: str) -> list[str]:
    """``"auto,4"`` → ``["1", "auto", "4"]``: the sequential baseline first,
    then each other setting once, in the order given."""
    out = ["1"]
    for part in str(text).split(","):
        w = part.strip().lower()
        if not w:
            continue
        if w != "auto":
            n = int(w)                       # ValueError on junk
            if n < 1:
                raise ValueError(f"worker count must be >= 1; got {w!r}")
            w = str(n)
        if w not in out:
            out.append(w)
    return out


def build_command(uv: str, project: str, checkpoint: str, out_json: str, *, workers: str,
                  timings_json: str, suites: str, device: str, batch_size: int,
                  max_series: int, num_samples: int, data_dir: str | None,
                  extra: list[tuple[str, str]] = (), fev_manifest: str | None = None) -> list[str]:
    extras = [a for s in ("time", "fev") if s in suites.split(",") for a in ("--extra", s)]
    cmd = [uv, "run", "--frozen", "--project", project, *extras,
           "cascade-benchmark", checkpoint, out_json,
           "--suites", suites, "--device", device, "--batch-size", str(batch_size),
           "--max-series", str(max_series), "--num-samples", str(num_samples),
           "--workers", workers, "--timings-json", timings_json]
    if data_dir:
        cmd += ["--data-dir", data_dir]
    if fev_manifest:
        cmd += ["--fev-manifest", fev_manifest]
    for ck, out in extra:
        cmd += ["--extra-checkpoint", ck, out]
    return cmd


def identical(baseline: list[str], other: list[str]) -> bool:
    """Every report text of a run equals the sequential run's, byte for byte."""
    return len(baseline) == len(other) and all(a == b for a, b in zip(baseline, other,
                                                                      strict=True))


def validate_report(report: dict, suites: list[str]) -> None:
    """A fast missing/partial benchmark is never a successful qualification."""
    rows = report.get("suites", [])
    if len(rows) != len(suites) or {r.get("suite") for r in rows} != set(suites):
        raise ValueError("benchmark suite coverage mismatch")
    for result in rows:
        name = result["suite"]
        revision = report.get("data_revisions", {}).get(name)
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", revision):
            raise ValueError(f"{name}: missing/unknown/partial data pin")
        if result.get("status") != "ok":
            raise ValueError(f"{name}: benchmark did not finish successfully")
        skipped = result.get("skipped", [])
        # FEV's declared univariate-only scope is reported, never confused
        # with an actual load/scoring failure. It is NOT full FEV coverage.
        if any(name != "fev" or r.get("stage") != "scope" for r in skipped):
            raise ValueError(f"{name}: required benchmark tasks were skipped")
        n = result.get("n_series", 0)
        items = result.get("rows", [])
        if (n <= 0 or n != len(items) or n + len(skipped) != result.get("n_expected")
                or len({r.get("full") for r in items}) != n):
            raise ValueError(f"{name}: incomplete or duplicate benchmark rows")
        for metric in ("crps", "mase"):
            value = result.get("metrics", {}).get(metric)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0):
                raise ValueError(f"{name}: invalid {metric}")


def speedup_table(runs: dict[str, dict]) -> list[str]:
    """Rows of ``suite | workers=1 wall | workers=W wall (×speedup) …`` from
    ``{workers: timings-json}`` (``"1"`` first), plus a total row and each
    run's plan line."""
    order = list(runs)
    base = runs[order[0]]
    suites = list(base.get("suites", {}))
    head = ["suite".ljust(12)] + [f"workers={w}".rjust(22) for w in order]
    lines = [" ".join(head)]

    def cell(t: float, ref: float, first: bool) -> str:
        if first or not t:
            return f"{t:10.1f}s".rjust(22)
        return f"{t:10.1f}s ({ref / t:4.2f}x)".rjust(22)

    for s in suites:
        ref = float(base["suites"][s].get("wall_s", 0.0))
        row = [s.ljust(12)]
        for i, w in enumerate(order):
            t = float(runs[w].get("suites", {}).get(s, {}).get("wall_s", 0.0))
            row.append(cell(t, ref, i == 0))
        lines.append(" ".join(row))
    ref = float(base.get("total_s", 0.0))
    lines.append(" ".join(["total".ljust(12)] + [
        cell(float(runs[w].get("total_s", 0.0)), ref, i == 0) for i, w in enumerate(order)]))
    lines += [f"  workers={w}: {runs[w].get('plan', '')}" for w in order]
    return lines


def main(argv: list[str] | None = None) -> int:  # pragma: no cover — needs the sidecar + data
    p = argparse.ArgumentParser(prog="bench_timing", description=__doc__.split("\n\n")[0])
    p.add_argument("checkpoint")
    p.add_argument("--extra-checkpoint", action="append", default=[],
                   help="time a paired pass with this checkpoint too (repeatable)")
    p.add_argument("--workers", default="auto", help="comma list compared against 1")
    p.add_argument("--suites", default="gift-eval,boom,time,fev")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--max-series", type=int, default=0)
    p.add_argument("--num-samples", type=int, default=100)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--fev-manifest", default=None)
    p.add_argument("--project", default=str(REPO / "benchmarks"))
    p.add_argument("--uv", default=shutil.which("uv") or "uv")
    p.add_argument("--keep", default=None, help="keep the reports/timings in this dir")
    args = p.parse_args(argv)

    requested = [s.strip() for s in args.suites.split(",") if s.strip()]
    if not requested or len(requested) != len(set(requested)):
        p.error("choose unique, nonempty benchmark suites")
    if args.max_series:
        p.error("qualification requires --max-series 0; use the benchmark CLI for smoke runs")
    for name in ("CASCADE_BENCH_GIFTEVAL_DATASETS", "CASCADE_BENCH_BOOM_DATASETS",
                 "CASCADE_BENCH_TIME_DATASETS", "CASCADE_BENCH_TIME_RAW"):
        if os.environ.get(name):
            p.error(f"unset {name}: qualification must use the full declared scope")

    workers = parse_worker_list(args.workers)
    runs: dict[str, dict] = {}
    texts: dict[str, list[str]] = {}
    with tempfile.TemporaryDirectory(prefix="bench-timing-") as tmp:
        root = Path(args.keep or tmp)
        root.mkdir(parents=True, exist_ok=True)
        for w in workers:
            d = root / f"w{w}"
            d.mkdir(exist_ok=True)
            outs = [str(d / "report-0.json")] + [str(d / f"report-{i + 1}.json")
                                                 for i in range(len(args.extra_checkpoint))]
            cmd = build_command(
                args.uv, args.project, args.checkpoint, outs[0], workers=w,
                timings_json=str(d / "timings.json"), suites=args.suites, device=args.device,
                batch_size=args.batch_size, max_series=args.max_series,
                num_samples=args.num_samples, data_dir=args.data_dir,
                extra=list(zip(args.extra_checkpoint, outs[1:], strict=True)),
                fev_manifest=args.fev_manifest)
            print(f"[bench-timing] workers={w}: {' '.join(cmd)}", file=sys.stderr)
            t0 = time.monotonic()
            proc = subprocess.run(cmd, check=False)
            if proc.returncode != 0:
                print(f"[bench-timing] workers={w} FAILED (exit {proc.returncode})",
                      file=sys.stderr)
                return 2
            runs[w] = json.loads((d / "timings.json").read_text(encoding="utf-8"))
            runs[w].setdefault("total_s", time.monotonic() - t0)
            texts[w] = [Path(o).read_text(encoding="utf-8") for o in outs]
            try:
                for text in texts[w]:
                    validate_report(json.loads(text), requested)
            except ValueError as exc:
                print(f"[bench-timing] INVALID qualification: {exc}", file=sys.stderr)
                return 2
    print("\n".join(speedup_table(runs)))
    bad = [w for w in workers[1:] if not identical(texts["1"], texts[w])]
    if bad:
        print(f"[bench-timing] REPORTS DIFFER from workers=1 for workers={','.join(bad)}",
              file=sys.stderr)
        return 1
    print("[bench-timing] every requested suite completed its declared scope; "
          "every report is byte-identical to the sequential sweep")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
