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
    p.add_argument(
        "--gifteval-datasets",
        default=None,
        help="comma/space-separated gift-eval config subset (e.g. the pinned "
        "consensus-gate subset); sets CASCADE_BENCH_GIFTEVAL_DATASETS. Omit to "
        "run the full 97-config battery.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="series forwarded through the checkpoint per batch (quantile-head "
        "wrappers only; legacy sample wrappers run one series at a time)",
    )
    p.add_argument(
        "--data-dir",
        default=None,
        help="root dir for benchmark datasets (each suite reads <data-dir>/<suite>). "
        "Wires GIFT_EVAL/BOOM/CASCADE_BENCH_TIME_DATASET automatically; with "
        "--download, fetches any missing dataset first. Overrides those env vars.",
    )
    p.add_argument(
        "--download",
        action="store_true",
        help="download each requested suite's FULL dataset into --data-dir before "
        "scoring (requires --data-dir; skips data already present).",
    )
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

    if args.download and not args.data_dir:
        print("error: --download requires --data-dir", file=sys.stderr)
        return 2

    import os

    if args.gifteval_datasets:
        os.environ["CASCADE_BENCH_GIFTEVAL_DATASETS"] = args.gifteval_datasets

    from .datasets import DATASETS, apply_env, ensure_datasets, recorded_revision

    if args.data_dir:
        env = ensure_datasets(requested, args.data_dir, download=args.download)
        apply_env(env)
        for name in requested:
            spec = DATASETS.get(name)
            if spec and spec.env_var not in env:
                print(f"[{name}] no data under {args.data_dir}/{name} "
                      "(use --download to fetch it) — will skip", file=sys.stderr)

    def _actual_revision(name: str) -> str | None:
        """Provenance of the data this suite will actually read: the download
        marker of the wired dir, or 'unknown' for marker-less (hand-managed /
        bare-env-var) data. None when the suite has no data at all."""
        spec = DATASETS.get(name)
        if spec is None:
            return None
        path = os.environ.get(spec.env_var)
        if not path:
            return None
        return recorded_revision(path) or "unknown"

    max_series = args.max_series or None
    report = BenchmarkReport(
        checkpoint=str(ckpt),
        data_revisions={
            name: rev for name in requested if (rev := _actual_revision(name)) is not None
        },
    )
    for name in requested:
        result = SUITES[name](
            str(ckpt),
            num_samples=args.num_samples,
            max_series=max_series,
            device=args.device,
            batch_size=args.batch_size,
        )
        report.suites.append(result)
        print(f"[{name}] {result.status} {result.metrics or result.detail}", file=sys.stderr)

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    print(f"wrote {out}", file=sys.stderr)
    return 0


def download_main(argv: list[str] | None = None) -> int:
    """``cascade-benchmark-download`` — fetch the full benchmark datasets.

        cascade-benchmark-download --data-dir ./bench_data [--suites gift-eval,boom,time]

    Pulls each benchmark's HuggingFace dataset repo into ``<data-dir>/<suite>`` and
    prints the env vars to export (or just pass the same ``--data-dir`` to
    ``cascade-benchmark``). Set ``HF_TOKEN`` for gated repos.
    """
    from .datasets import DATASETS, download_suite

    p = argparse.ArgumentParser(
        prog="cascade-benchmark-download",
        description="Download the full GIFT-Eval / BOOM / TIME datasets for the scorer.",
    )
    p.add_argument("--data-dir", required=True, help="root dir to download into (<data-dir>/<suite>)")
    p.add_argument(
        "--suites",
        default=",".join(DATASETS),
        help="comma-separated subset of: " + ", ".join(DATASETS),
    )
    args = p.parse_args(argv)

    suites = [s.strip() for s in args.suites.split(",") if s.strip()]
    unknown = [s for s in suites if s not in DATASETS]
    if unknown:
        print(f"error: unknown suite(s): {unknown}; known: {list(DATASETS)}", file=sys.stderr)
        return 2

    for s in suites:
        dest = Path(args.data_dir) / s
        print(f"downloading {DATASETS[s].hf_repo} → {dest} …", file=sys.stderr)
        download_suite(s, dest)
        print(f"  {DATASETS[s].env_var}={dest}", file=sys.stderr)
    print(
        f"done — run: cascade-benchmark CKPT out.json --data-dir {args.data_dir}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
