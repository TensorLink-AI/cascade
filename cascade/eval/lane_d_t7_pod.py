"""The T7 pod run (LANE_D_PLAN §4 T7, §1.1 gate 2; W5d): measure every lane D
context variant on the eval host that scores lane D and write the
``laned_t7/1`` document the box-side ``ops_jobs evidence`` job installs.

    python -P -m cascade.eval.lane_d_t7_pod \\
        --checkpoint-dir DIR (--histories windows.npz | --synthetic-seed N) \\
        --host NAME --gpu "H100 80GB" --mem-gb 80 [--variants 4096,8192,16384] \\
        [--window-counts 1000,5000] [--n-models 10] [--horizon 64] \\
        [--settlement-blocks 900] [--device cuda] --out evidence/t7.json

Per variant, the scorer is timed (:func:`cascade.eval.lane_d_t7.measure_variant`:
load + forecast, projected to ``--n-models`` per settlement, the variant's own
``cuda_max_allocated`` peak) on each window count of ``--window-counts``:
``[lane_d] min_windows`` (1000, the floor a settlement can score on) and
``max_windows_per_settlement`` (5000, the per-bin draw cap: the most a
settlement scores). The ROW that gates arming is the measurement at the
largest count (the worst case a settlement can hit); every measurement is
kept under ``window_sweep`` (variant -> rows, ascending count), outside the
rows the scorer reads. The short-window check (a 4k checkpoint vs its byte
copy declared at 128k on ≤ 4096-point windows) runs on the largest count's
windows cut to their last 4096 points.

Windows: ``--histories`` is an ``.npz`` of 1-D float arrays (a pool_d
snapshot's histories, keys taken in sorted order; only arrays at least the
variant long are used for that variant, never a shorter one, RV3), or
``--synthetic-seed`` builds deterministic random-walk windows (numpy
``default_rng(seed)``, overlapping slices of one walk), each exactly as long
as the largest variant: T7 times
the model at a fixed input length, so the values do not change the timing.

Only a CUDA device produces pod evidence (``source = "pod"``); ``--device
cpu`` writes ``source = "local"``, which the evidence job refuses. Needs torch
(the scorer): a POD command, never the production box. Exit 0 written, 1
refused (nothing written), 2 usage.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

from .lane_d_t7 import (
    LAUNCH_VARIANTS,
    MIN_MODELS,
    POD_SOURCE,
    TRAINED_CONTEXT,
    build_evidence,
    settlement_budget,
)

DEFAULT_WINDOW_COUNTS = (1000, 5000)
LOCAL_SOURCE = "local"
COMMAND = "cascade.eval.lane_d_t7_pod"
REPO = Path(__file__).resolve().parents[2]


class T7RunError(ValueError):
    """The run cannot produce honest evidence; nothing is written."""


def _ints(text: str, what: str) -> tuple[int, ...]:
    try:
        out = tuple(sorted({int(x) for x in str(text).split(",") if x.strip()}))
    except ValueError:
        raise argparse.ArgumentTypeError(f"{what} must be comma-separated integers") from None
    if not out or out[0] < 1:
        raise argparse.ArgumentTypeError(f"{what} must name at least one positive integer")
    return out


def synthetic_histories(seed: int, n: int, length: int) -> list[np.ndarray]:
    """``n`` deterministic random-walk series of exactly ``length`` points:
    window ``i`` is points ``[i, i + length)`` of one walk (read-only views,
    so 5000 windows of 16384 points cost one array of ``length + n``)."""
    rng = np.random.default_rng(int(seed))
    walk = np.cumsum(rng.standard_normal(int(length) + int(n))) + 100.0
    walk.setflags(write=False)
    return [walk[i:i + int(length)] for i in range(int(n))]


def load_histories(path: str | os.PathLike) -> list[np.ndarray]:
    """The 1-D histories of an ``.npz`` (sorted keys), as float64."""
    with np.load(Path(path), allow_pickle=False) as z:
        out = [np.asarray(z[k], dtype=np.float64).reshape(-1) for k in sorted(z.files)]
    if not out:
        raise T7RunError(f"{path} holds no histories")
    return out


def pick_windows(histories: list[np.ndarray], *, variant: int, count: int) -> list[np.ndarray]:
    """The first ``count`` histories at least ``variant`` long; fewer is a
    refusal (a variant is never timed on fewer or shorter windows)."""
    long = [h for h in histories if h.size >= variant]
    if len(long) < count:
        raise T7RunError(f"variant {variant}: {len(long)} windows hold >= {variant} points, "
                         f"{count} needed")
    return long[:count]


def _git_sha() -> str:
    try:
        out = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=30, check=True)
    except (OSError, subprocess.SubprocessError):
        raise T7RunError("no --tool-git-sha and git rev-parse failed: the evidence must name "
                         "the code that measured it") from None
    return out.stdout.strip()


def run(args, *, measure: Callable | None = None, short: Callable | None = None,
        torch_version: str | None = None) -> dict:
    """Measure and return the ``laned_t7/1`` document (``measure`` / ``short``
    / ``torch_version`` default to the real kit and torch)."""
    from . import lane_d_t7 as t7

    measure = t7.measure_variant if measure is None else measure
    short = t7.short_window_check if short is None else short
    if torch_version is None:
        import torch

        torch_version = str(torch.__version__)
    variants, counts = tuple(args.variants), tuple(args.window_counts)
    top = max(counts)
    if args.histories is not None:
        histories = load_histories(args.histories)
    else:
        histories = synthetic_histories(args.synthetic_seed, top, max(variants))
    budget = settlement_budget(args.mem_gb, settlement_blocks=args.settlement_blocks)
    source = POD_SOURCE if str(args.device).startswith("cuda") else LOCAL_SOURCE
    rows, sweep = [], {}
    for v in variants:
        measured = []
        for n in counts:
            windows = pick_windows(histories, variant=v, count=n)
            measured.append(measure(args.checkpoint_dir, windows, context=v,
                                    horizon=args.horizon, host=args.host, gpu=args.gpu,
                                    n_models=args.n_models, budget=budget, device=args.device))
        measured.sort(key=lambda r: r["n_windows"])
        if [r["n_windows"] for r in measured] != list(counts):
            raise T7RunError(f"variant {v}: measured window counts "
                             f"{[r['n_windows'] for r in measured]} are not {list(counts)}")
        sweep[str(v)] = measured
        rows.append(measured[-1])                  # the largest count gates arming
    base = pick_windows(histories, variant=1, count=top)
    sw = short(args.checkpoint_dir, [h[-TRAINED_CONTEXT:] for h in base], horizon=args.horizon,
               device=args.device)
    tool = {"command": COMMAND, "git_sha": args.tool_git_sha or _git_sha(),
            "window_counts": list(counts), "gate_window_count": top}
    doc = build_evidence(rows, short_window=sw, tool=tool, torch_version=torch_version,
                         source=source)
    doc["window_sweep"] = sweep
    return doc


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=f"python -P -m {COMMAND}",
                                description="Measure lane D T7 on the eval host (pod side).")
    p.add_argument("--checkpoint-dir", type=Path, required=True,
                   help="a lane D checkpoint (an entry directory) to time")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--histories", type=Path, help=".npz of 1-D histories (a pool_d snapshot)")
    src.add_argument("--synthetic-seed", type=int, help="deterministic random-walk windows")
    p.add_argument("--host", required=True, help="the eval host's name ([lane_d_ops] eval_host)")
    p.add_argument("--gpu", required=True, help="the GPU model, as nvidia-smi names it")
    p.add_argument("--mem-gb", type=float, required=True, help="the GPU's memory (budget)")
    p.add_argument("--variants", type=lambda s: _ints(s, "--variants"),
                   default=LAUNCH_VARIANTS, help="contexts to measure (default 4096,8192,16384)")
    p.add_argument("--window-counts", type=lambda s: _ints(s, "--window-counts"),
                   default=DEFAULT_WINDOW_COUNTS,
                   help="window counts per variant (default 1000,5000; the largest gates)")
    p.add_argument("--n-models", type=int, default=MIN_MODELS,
                   help="models per settlement (default 10: 8 entries + KING_D + house)")
    p.add_argument("--horizon", type=int, default=64, help="[lane_d] horizon")
    p.add_argument("--settlement-blocks", type=int, default=900,
                   help="blocks per settlement (the budget is half of one)")
    p.add_argument("--device", default="cuda", help="torch device (cuda for pod evidence)")
    p.add_argument("--tool-git-sha", default=None, help="the release commit (default git HEAD)")
    p.add_argument("--out", type=Path, required=True, help="the laned_t7/1 file to write")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.n_models < MIN_MODELS:
        print(f"lane_d_t7_pod: --n-models must be >= {MIN_MODELS}", file=sys.stderr)
        return 2
    try:
        doc = run(args)
    except (T7RunError, ValueError, OSError) as e:
        print(f"lane_d_t7_pod: refused: {e}; nothing written", file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(f".{args.out.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(doc, sort_keys=True, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, args.out)
    print(json.dumps({"out": str(args.out), "armed": doc["armed"], "passed": doc["passed"],
                      "source": doc["source"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["DEFAULT_WINDOW_COUNTS", "T7RunError", "build_arg_parser", "load_histories", "main",
           "pick_windows", "run", "synthetic_histories"]
