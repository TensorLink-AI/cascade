"""``cascade-benchmark`` entry point.

    cascade-benchmark <checkpoint_dir> <out.json> \
        [--suites gift-eval,boom,time] [--num-samples 100] \
        [--max-series N] [--device cpu|cuda]

Loads the checkpoint, runs each requested suite (errors are captured per-suite,
never fatal), and writes a :class:`~cascade_benchmark.results.BenchmarkReport`
as JSON. Exit code is 0 whenever the report is written — the validator decides
what to do with skipped/errored suites — and non-zero only on argument or
filesystem failure, so the caller can distinguish "ran, some suites skipped"
from "could not run at all".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .results import BenchmarkReport
from .suites import SUITES


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="cascade-benchmark")
    p.add_argument("checkpoint_dir", help="trained checkpoint dir (has forecast_wrapper.py)")
    p.add_argument("out_json", help="path to write the results JSON")
    p.add_argument(
        "--suites",
        default="gift-eval,boom,time",
        help="comma-separated subset of: " + ", ".join(SUITES),
    )
    p.add_argument("--num-samples", type=int, default=100)
    p.add_argument(
        "--max-series",
        type=int,
        default=0,
        help="cap the number of datasets/tasks evaluated, for a fast smoke run "
        "(0 = full benchmark)",
    )
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)

    ckpt = Path(args.checkpoint_dir)
    if not (ckpt / "forecast_wrapper.py").is_file():
        print(f"error: {ckpt}/forecast_wrapper.py not found", file=sys.stderr)
        return 2

    requested = [s.strip() for s in args.suites.split(",") if s.strip()]
    unknown = [s for s in requested if s not in SUITES]
    if unknown:
        print(f"error: unknown suite(s): {unknown}; known: {list(SUITES)}", file=sys.stderr)
        return 2

    max_series = args.max_series or None
    report = BenchmarkReport(checkpoint=str(ckpt))
    for name in requested:
        result = SUITES[name](
            str(ckpt),
            num_samples=args.num_samples,
            max_series=max_series,
            device=args.device,
        )
        report.suites.append(result)
        print(f"[{name}] {result.status} {result.metrics or result.detail}", file=sys.stderr)

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    print(f"wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
