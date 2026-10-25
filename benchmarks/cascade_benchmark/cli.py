"""``cascade-benchmark`` entry point.

    cascade-benchmark <checkpoint_dir> <out.json> \
        [--suites gift-eval,boom,time,fev] [--num-samples 100] \
        [--max-series N] [--device cpu|cuda] [--workers auto|N] \
        [--extra-checkpoint <dir> <out.json>]...

Loads the checkpoint, runs each requested suite (errors are captured per-suite,
never fatal), and writes a :class:`~cascade_benchmark.results.BenchmarkReport`
as JSON. ``--workers`` runs the suites' work units concurrently
(``cascade_benchmark.parallel``; ``auto`` sizes to the GPU, see
``cascade_benchmark.sizing``); ``--extra-checkpoint`` sweeps further
checkpoints in the SAME pass (each dataset built once) and writes one report
each. Qualify identical reports against ``--workers 1`` on the chosen hardware.
FEV requires a local hash manifest. Exit code is 0 whenever the report is written — the validator decides
what to do with skipped/errored suites — and non-zero only on argument or
filesystem failure, so the caller can distinguish "ran, some suites skipped"
from "could not run at all".
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from .results import BenchmarkReport
from .suites import SUITES


def report_json(checkpoint, results, data_revisions: dict) -> str:
    """The exact report text the CLI writes for one checkpoint."""
    report = BenchmarkReport(checkpoint=str(checkpoint), data_revisions=dict(data_revisions))
    report.suites.extend(results)
    return json.dumps(report.to_dict(), indent=2)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cascade-benchmark")
    p.add_argument("checkpoint_dir", help="trained checkpoint dir (has forecast_wrapper.py)")
    p.add_argument("out_json", help="path to write the results JSON")
    p.add_argument(
        "--suites",
        default="gift-eval,boom,time,fev",
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
    p.add_argument(
        "--workers",
        default="1",
        help="worker processes for the work units: 'auto' (sized to GPU memory and "
        "CPU cores, cascade_benchmark.sizing) or N. 1 = in-process sequential. The "
        "report is identical for every value.",
    )
    p.add_argument("--max-workers", type=int, default=None,
                   help="cap on --workers auto (default cascade_benchmark.sizing.DEFAULT_CAP)")
    p.add_argument("--worker-vram-gb", type=float, default=None,
                   help="per-worker VRAM for --workers auto instead of measuring it")
    p.add_argument("--worker-ram-gb", type=float, default=None,
                   help="per-worker host RAM allowance (auto otherwise: model bytes + data)")
    p.add_argument("--fev-manifest", type=Path,
                   help="hash-pinned local FEV tasks and datasets; defaults to "
                   "<data-dir>/fev/manifest.json or CASCADE_BENCH_FEV_MANIFEST")
    p.add_argument("--allow-unpinned-fev", action="store_true",
                   help="archived/diagnostic runs only; FEV provenance is recorded as unknown")
    p.add_argument(
        "--extra-checkpoint",
        nargs=2,
        action="append",
        default=[],
        metavar=("CHECKPOINT_DIR", "OUT_JSON"),
        help="also sweep this checkpoint in the same pass (paired sweep; repeatable)",
    )
    p.add_argument("--timings-json", default=None,
                   help="write per-suite wall times + the worker plan here (never in the report)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    ckpt = Path(args.checkpoint_dir)
    targets = [(ckpt, Path(args.out_json))] + [
        (Path(c), Path(o)) for c, o in args.extra_checkpoint]
    for c, _o in targets:
        if not (c / "forecast_wrapper.py").is_file():
            print(f"error: {c}/forecast_wrapper.py not found", file=sys.stderr)
            return 2
    from .sizing import DEFAULT_CAP, GIB, parse_workers

    try:
        workers = parse_workers(args.workers)
    except ValueError as e:
        print(f"error: --workers: {e}", file=sys.stderr)
        return 2

    requested = [s.strip() for s in args.suites.split(",") if s.strip()]
    unknown = [s for s in requested if s not in SUITES]
    if unknown:
        print(f"error: unknown suite(s): {unknown}; known: {list(SUITES)}", file=sys.stderr)
        return 2
    if not requested or len(set(requested)) != len(requested) or args.max_series < 0:
        print("error: choose unique, nonempty suites and a non-negative --max-series", file=sys.stderr)
        return 2

    if args.download and not args.data_dir:
        print("error: --download requires --data-dir", file=sys.stderr)
        return 2

    import os

    for name in ("batch_size", "num_samples"):
        if getattr(args, name) < 1:
            print(f"error: --{name.replace('_', '-')} must be positive", file=sys.stderr)
            return 2
    for name in ("max_workers", "worker_vram_gb", "worker_ram_gb"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            print(f"error: --{name.replace('_', '-')} must be finite and positive", file=sys.stderr)
            return 2

    fev_revision = None
    if "fev" in requested:
        manifest = (args.fev_manifest or os.environ.get("CASCADE_BENCH_FEV_MANIFEST")
                    or (Path(args.data_dir) / "fev/manifest.json" if args.data_dir else None))
        if manifest is not None:
            try:
                from .provenance import verify_fev_manifest

                fev_revision = verify_fev_manifest(manifest)
            except Exception as e:  # invalid data is infra, never a miner's scientific loss
                print(f"error: FEV manifest verification failed: {e}", file=sys.stderr)
                return 3
        elif not args.allow_unpinned_fev:
            print("error: FEV requires --fev-manifest (or <data-dir>/fev/manifest.json). "
                  "Use --allow-unpinned-fev only for archived/diagnostic runs.", file=sys.stderr)
            return 3
        else:
            print("[fev] WARNING: unpinned diagnostic run; not comparable across revisions",
                  file=sys.stderr)
            fev_revision = "unknown"

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

    from .parallel import SweepTimings, WorkerCrashed, run_sweep

    max_series = args.max_series or None
    revisions = {name: rev for name in requested if (rev := _actual_revision(name)) is not None}
    if fev_revision is not None:
        revisions["fev"] = fev_revision
    if "fev" in requested and not args.allow_unpinned_fev:
        for name in requested:
            if name in DATASETS and revisions.get(name) != DATASETS[name].revision:
                print(f"error: {name} needs the full dataset at the release's pinned revision; "
                      "missing, partial or mismatched data cannot enter a four-suite sweep",
                      file=sys.stderr)
                return 3
    timings = SweepTimings()
    try:
        per_ckpt = run_sweep(
            requested, [str(c) for c, _o in targets],
            num_samples=args.num_samples, max_series=max_series, device=args.device,
            batch_size=args.batch_size, workers=workers,
            max_workers=args.max_workers or DEFAULT_CAP,
            worker_vram_bytes=(int(args.worker_vram_gb * GIB) if args.worker_vram_gb else None),
            per_worker_ram_bytes=(int(args.worker_ram_gb * GIB) if args.worker_ram_gb else None),
            timings=timings,
        )
    except WorkerCrashed as e:
        print(f"error: {e}", file=sys.stderr)
        return 3
    for (c, out), results in zip(targets, per_ckpt, strict=True):
        for name, result in zip(requested, results, strict=True):
            result.max_series = args.max_series
            tag = f"[{name}]" if len(targets) == 1 else f"[{name}:{c.name}]"
            print(f"{tag} {result.status} {result.metrics or result.detail}", file=sys.stderr)
            if result.skipped:
                # A partial sweep is not the benchmark — say so where the pod log
                # (and the trainer's stderr capture) will show it.
                print(f"{tag} PARTIAL: scored {result.n_series} of {result.n_expected} "
                      f"configs; skipped {len(result.skipped)}:", file=sys.stderr)
                for s in result.skipped:
                    print(f"{tag}   {s['full']} ({s['stage']}): {s['reason']}", file=sys.stderr)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report_json(c, results, revisions), encoding="utf-8")
        print(f"wrote {out}", file=sys.stderr)
    for name, t in timings.suites.items():
        print(f"[timing] {name}: {t['units']} units, wall {t['wall_s']:.1f}s, "
              f"unit-sum {t['unit_s']:.1f}s", file=sys.stderr)
    print(f"[timing] total {timings.total_s:.1f}s ({timings.plan})", file=sys.stderr)
    if args.timings_json:
        tp = Path(args.timings_json)
        tp.parent.mkdir(parents=True, exist_ok=True)
        tp.write_text(json.dumps(timings.to_dict(), indent=2), encoding="utf-8")
    return 0


def download_main(argv: list[str] | None = None) -> int:
    """``cascade-benchmark-download`` — fetch the full benchmark datasets.

        cascade-benchmark-download --data-dir ./bench_data [--suites gift-eval,boom,time,fev]

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
