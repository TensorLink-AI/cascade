"""Post-round public-benchmark telemetry — LOG-ONLY, never touches KOTH state.

After the trainer publishes a round's manifest, the orchestrator can fire the
benchmark sidecar (GIFT-Eval / BOOM / TIME) at the round's **king** checkpoint
on the (now idle) GPU pod. Validators keep scoring rounds exclusively on the
private eval pool — these numbers exist so the operator gets a round-over-round
time series of what the champion generator produces at the standard budget.
They must never feed miner scores, weights, or the throne decision: the
benchmark data is public (a Goodhart target for generators) and GPU sweeps are
not bit-reproducible across SKUs (unauditable as a consensus input).

Failure semantics mirror ``cascade.eval.benchmarks``: the run happens on a
daemon thread and every failure path logs and returns — a broken, slow, or
missing benchmark must never delay or fail a round. Training always wins the
GPU: each launch first kills any still-running benchmark on the pod, and the
next round's training dispatch simply contends ahead of a straggler (size the
suites to the round cadence: full battery ≈ 1h on a 4090 — fine at 24h rounds;
use ``max_series``/a suite subset on fast testnet rounds).

Command construction is pure and unit-tested; only the launcher shells out.
"""

from __future__ import annotations

import json
import logging
import shlex
import threading
from dataclasses import dataclass
from pathlib import Path

# PREEMPT_BENCHMARKS also serves as this hook's kill-any-previous-sweep prefix
# (never let a stale sweep pile up behind fast rounds) — one pattern, one
# place, see remote.py for the anchoring rationale.
from ..eval.benchmarks import format_report
from .remote import PREEMPT_BENCHMARKS, RemoteHost, build_ssh_argv, run_ssh

log = logging.getLogger("cascade.trainer.bench")


@dataclass(frozen=True)
class BenchPlan:
    """What to run after each round. ``suites``/``max_series`` size the sweep
    to the round cadence; ``data_dir`` must hold the pinned benchmark data on
    the pod (``cascade-benchmark-download --data-dir …``)."""

    suites: str = "gift-eval,boom,time"
    max_series: int = 0            # 0 = full benchmark
    batch_size: int = 512
    device: str = "cuda"
    data_dir: str = "/root/bench_data"
    uv_bin: str = "~/.local/bin/uv"  # uv on the pod (runs the sidecar's own env)
    timeout_seconds: int = 2 * 3600
    # Decouple telemetry cadence from round cadence: skip launching when the
    # last launch was under this many seconds ago (0 = benchmark every round).
    # The right setting when rounds are tighter than the sweep: pick an
    # interval > sweep duration and telemetry samples every Nth king instead
    # of racing (and being preempted by) every round's training.
    min_interval_seconds: int = 0


def king_paths(host: RemoteHost, round_id: str, arch_preset: str) -> tuple[str, str]:
    """(checkpoint dir, report path) of a round's king on the pod — the layout
    ``cascade.trainer.worker`` writes under ``<workdir>/_train_work``."""
    base = f"{host.workdir}/_train_work/{round_id}/{arch_preset}/king"
    return f"{base}/checkpoint", f"{base}/benchmark_report.json"


def build_bench_remote_command(host: RemoteHost, round_id: str, arch_preset: str,
                               plan: BenchPlan) -> tuple[str, str]:
    """The remote shell string that benchmarks the round's king, plus the
    report path it writes. Pure — safe to unit test."""
    ckpt, report = king_paths(host, round_id, arch_preset)
    argv = [
        "cascade-benchmark", ckpt, report,
        "--suites", plan.suites,
        "--device", plan.device,
        "--batch-size", str(plan.batch_size),
        "--data-dir", plan.data_dir,
    ]
    if plan.max_series:
        argv += ["--max-series", str(plan.max_series)]
    quoted = " ".join(shlex.quote(a) for a in argv)
    prefix = ""
    if host.cuda_device is not None:
        prefix = f"CUDA_VISIBLE_DEVICES={shlex.quote(host.cuda_device)} "
    cmd = (
        PREEMPT_BENCHMARKS
        + prefix
        + f"{plan.uv_bin} run --project {shlex.quote(f'{host.workdir}/benchmarks')} "
        + quoted
    )
    return cmd, report


def run_post_round_benchmark(host: RemoteHost, round_id: str, arch_preset: str,
                             plan: BenchPlan, *, work_root: Path | None = None,
                             runner=None) -> dict | None:
    """Benchmark the round's king on ``host`` and return the parsed report.

    Blocking (call it from :func:`launch_post_round_benchmark`'s thread).
    Returns ``None`` on any failure — this path must never raise into a round.
    """
    try:
        remote_cmd, report_path = build_bench_remote_command(host, round_id, arch_preset, plan)
        ssh = build_ssh_argv(host, remote_cmd)
        run = runner or run_ssh
        proc = run(ssh, plan.timeout_seconds)
        if proc.returncode != 0:
            log.warning("post-round benchmark failed on %s (exit %s): %s",
                        host.name, proc.returncode, (proc.stderr or "")[-400:])
            return None
        cat = run(build_ssh_argv(host, f"cat {shlex.quote(report_path)}"), 120)
        if cat.returncode != 0:
            log.warning("post-round benchmark report missing on %s: %s",
                        host.name, (cat.stderr or "")[-200:])
            return None
        report = json.loads(cat.stdout)
        if work_root is not None:
            local = Path(work_root) / round_id / arch_preset / "king-benchmark_report.json"
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_text(json.dumps(report, indent=2), encoding="utf-8")
        log.info("bench round=%s %s", round_id, format_report(report))
        return report
    except Exception as e:  # noqa: BLE001 — log-only telemetry must never raise
        log.warning("post-round benchmark errored (ignored): %s", e)
        return None


_last_launch: dict[str, float] = {}  # host.name → monotonic() of last launch


def launch_post_round_benchmark(host: RemoteHost, round_id: str, arch_preset: str,
                                plan: BenchPlan, *, work_root: Path | None = None
                                ) -> threading.Thread | None:
    """Fire-and-forget wrapper: runs the benchmark on a daemon thread so the
    round loop moves straight on to polling for the next epoch. Returns None
    (skipped) when the last launch on this host was under
    ``plan.min_interval_seconds`` ago."""
    import time

    now = time.monotonic()
    last = _last_launch.get(host.name)
    if plan.min_interval_seconds and last is not None and (now - last) < plan.min_interval_seconds:
        log.info("post-round benchmark skipped for round=%s (last launch %.0fs ago < %ds interval)",
                 round_id, now - last, plan.min_interval_seconds)
        return None
    _last_launch[host.name] = now
    t = threading.Thread(
        target=run_post_round_benchmark,
        args=(host, round_id, arch_preset, plan),
        kwargs={"work_root": work_root},
        name=f"bench-{round_id}",
        daemon=True,
    )
    t.start()
    log.info("post-round benchmark launched for round=%s king (%s) on %s [suites=%s]",
             round_id, arch_preset, host.name, plan.suites)
    return t
