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
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path

# PREEMPT_BENCHMARKS also serves as this hook's kill-any-previous-sweep prefix
# (never let a stale sweep pile up behind fast rounds) — one pattern, one
# place, see remote.py for the anchoring rationale.
from ..eval.benchmarks import format_report
from .remote import PREEMPT_BENCHMARKS, RemoteHost, build_ssh_argv, run_ssh

# a completed suite download's marker file (cascade_benchmark.datasets._MARKER)
DATA_MARKER = "_cascade_revision.json"

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
    # uv on the pod (runs the sidecar's own env). Image-booted pods bake it at
    # /bin/uv (deploy/Dockerfile); ~/.local/bin/uv was the rsync/bootstrap-era
    # installer path — override for a bootstrap-provisioned fleet.
    uv_bin: str = "/bin/uv"
    timeout_seconds: int = 2 * 3600
    # fence for the dataset download's wedge mode (2026-08-12: unauthenticated
    # HF pulls deadlocked twice); authenticated pulls finish in ~3 min.
    download_timeout_seconds: int = 2700
    # Decouple telemetry cadence from round cadence: skip launching when the
    # last launch was under this many seconds ago (0 = benchmark every round).
    # The right setting when rounds are tighter than the sweep: pick an
    # interval > sweep duration and telemetry samples every Nth king instead
    # of racing (and being preempted by) every round's training.
    min_interval_seconds: int = 0
    # Benchmark data dir on the TRAINER box to sideload to the pod (tar over
    # ssh) before each sweep — None/absent-dir = old behavior (the pod's own
    # HF download, fenced by download_timeout_seconds, remains the fallback
    # either way). r45–r48: four rounds straight lost their reports to the
    # on-pod fetch (forecast_wrapper/exit-2, then r48's exit-124 stall at
    # ~4975 files); the manual sideload of the same 4.4G battery streams in
    # ~75s and succeeded every time. Marker-guarded, so sequential roles on
    # one pod stage exactly once.
    local_data_dir: str | None = None
    # Fence on the tar|ssh stream — its own stall guard, separate from (and
    # much tighter than) the bench/download timeouts: the manual stream takes
    # ~75s, so a stream still running at 15 min is wedged, and failing it
    # early leaves the whole bench window to the download fallback.
    sideload_timeout_seconds: int = 900


def _suite_list(suites: str) -> list[str]:
    return [s.strip() for s in str(suites).split(",") if s.strip()]


def suite_markers(data_dir: str, suites: list[str]) -> str:
    """The ``test`` expression asserting every suite's COMPLETION marker under
    ``data_dir`` (`-f m1 -a -f m2 …`) — the one definition of "data is staged"
    shared by the bench data guard, the prewarm, and the sideload."""
    return " -a ".join(
        f"-f {shlex.quote(f'{data_dir}/{s}/{DATA_MARKER}')}" for s in suites)


def build_prewarm_remote_command(workdir: str, *,
                                 suites: str = "gift-eval,boom,time",
                                 uv_bin: str = "/bin/uv",
                                 download_timeout_seconds: int = 2700,
                                 hf_token: bool = False) -> str:
    """The remote shell string that pre-warms a fresh FINAL pod's benchmark
    data in the background. Pure — safe to unit test.

    JIT-rented final pods boot from the worker image, which deliberately bakes
    no benchmark data — the bench's data guard downloads on first use, which
    put a cold ~4.4G pull inside the bench window and cost the king leg its
    report three rounds running (r18, r26, r27: exit 124 mid-download). Fired
    right after the health gate, the download overlaps the multi-hour final
    training instead, and the guard's marker test finds warm data at bench
    time. Detached (``nohup``/``&`` with all stdio redirected) so the
    launching SSH returns immediately; marker-guarded so a re-boot of a
    half-warmed pod resumes rather than restarts; ``timeout``-fenced against
    the download's wedge mode exactly like the guard. Targets
    ``<workdir>/bench_data`` — the same path the trainer's bench plan derives
    (main.py builds ``data_dir=f"{wd}/bench_data"`` from the fleet workdir).

    ``hf_token`` arms the stdin-env sourcing prefix (the credential itself
    never enters the command string — it travels on stdin like every other
    forwarded cred); the exported var is inherited by the detached child.
    """
    data_dir = f"{workdir}/bench_data"
    project = shlex.quote(f"{workdir}/benchmarks")
    markers = suite_markers(data_dir, _suite_list(suites))
    inner = (f"timeout {int(download_timeout_seconds)} "
             f"{uv_bin} run --extra time --project {project} "
             f"cascade-benchmark-download --data-dir {shlex.quote(data_dir)}")
    env_source = "set -a && . /dev/stdin && set +a && " if hf_token else ""
    return (
        env_source
        + f"test {markers} || {{ nohup sh -c {shlex.quote(inner)} "
        + f">> {shlex.quote(f'{workdir}/bench_prewarm.log')} 2>&1 < /dev/null & }}"
    )


def role_paths(host: RemoteHost, round_id: str, arch_preset: str,
               role: str = "king") -> tuple[str, str]:
    """(checkpoint dir, report path) of a round's final ``role`` on the pod.
    ``role`` is the work-DIR name: bare (``king``, ``challenger``) for
    pre-cohort finals, uid-suffixed (``challenger-u<uid>``) when a
    DEC-CA-0012 cohort trained several challengers — callers pass the
    suffixed form (see ``loop._bench_role_dir``). Matches the layout
    ``cascade.trainer.worker`` writes under ``<workdir>/_train_work``."""
    base = f"{host.workdir}/_train_work/{round_id}/{arch_preset}/{role}"
    return f"{base}/checkpoint", f"{base}/benchmark_report.json"


def king_paths(host: RemoteHost, round_id: str, arch_preset: str) -> tuple[str, str]:
    """(checkpoint dir, report path) of a round's king on the pod."""
    return role_paths(host, round_id, arch_preset, "king")


def build_bench_remote_command(host: RemoteHost, round_id: str, arch_preset: str,
                               plan: BenchPlan, *, role: str = "king",
                               hf_token: str | None = None) -> tuple[str, str]:
    """The remote shell string that benchmarks the round's final ``role``
    checkpoint, plus the report path it writes. Pure — safe to unit test.

    ``hf_token`` arms the stdin-env sourcing prefix (the credential itself
    NEVER enters the command string — it travels on stdin exactly like the
    worker dispatch's forwarded creds, see ``remote.build_remote_command``):
    unauthenticated HF dataset pulls are rate-limited and wedged two fresh-pod
    downloads on 2026-08-12."""
    ckpt, report = role_paths(host, round_id, arch_preset, role)
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
    project = shlex.quote(f"{host.workdir}/benchmarks")
    # Image-booted pods deliberately bake no benchmark data (4.4G would slow
    # every heat boot for nothing) — self-provision on first bench instead.
    # && -chained so a failed download fails the sweep (best-effort upstream)
    # rather than benching against an empty data dir. Same-pod launches are
    # serialized by the caller (grouped by pod address), so the guard never
    # races itself.
    #
    # The guard tests each requested suite's COMPLETION marker, not the bare
    # data dir: `test -d` turned one interrupted download into a permanent
    # per-pod TIME skip (the dir existed, the download never re-ran — the
    # r13–r15 bench drought). The download is marker-aware and resumable, so
    # re-running over partial data is cheap; `timeout` fences the wedge mode
    # (thread-pool deadlock with no sockets, seen twice on 2026-08-12) so a
    # hung download fails this sweep instead of eating the bench window.
    markers = suite_markers(plan.data_dir, _suite_list(plan.suites))
    data_guard = (
        f"{{ test {markers} || "
        f"timeout {int(plan.download_timeout_seconds)} "
        f"{plan.uv_bin} run --extra time --project {project} cascade-benchmark-download "
        f"--data-dir {shlex.quote(plan.data_dir)}; }} && "
    )
    # --extra time on EVERY uv run: timebench is an optional extra, and uv run
    # syncs the env exactly — a run without the flag would UNINSTALL it again
    # (its absence on the pods was the root cause of the TIME-skip era,
    # 2026-08-05 bootstrap rework).
    env_source = "set -a && . /dev/stdin && set +a && " if hf_token else ""
    cmd = (
        PREEMPT_BENCHMARKS
        + env_source
        + data_guard
        + prefix
        + f"{plan.uv_bin} run --extra time --project {project} "
        + quoted
    )
    return cmd, report


def build_sideload_remote_command(data_dir: str) -> str:
    """The remote shell string that receives the sideload tar stream on its
    stdin and unpacks it into ``data_dir``. Pure — safe to unit test."""
    dd = shlex.quote(data_dir)
    return f"mkdir -p {dd} && tar -C {dd} -xf -"


def _stream_tar(local_dir: str, ssh_argv: list[str], timeout: int) -> int:
    """``tar -C local_dir -cf - . | ssh …`` — stream the local benchmark data
    into the pod-side unpack command. Returns the pipeline's exit status
    (remote/ssh failure wins; a clean remote surfaces the local tar's rc)."""
    tar = subprocess.Popen(
        ["tar", "-C", local_dir, "-cf", "-", "."],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        proc = subprocess.run(ssh_argv, stdin=tar.stdout, capture_output=True,
                              text=True, timeout=timeout)
    except Exception:
        tar.kill()
        tar.wait()
        raise
    finally:
        tar.stdout.close()  # EPIPEs tar if ssh died early — no orphaned writer
    tar_rc = tar.wait(timeout=60)
    return proc.returncode if proc.returncode != 0 else tar_rc


def sideload_bench_data(host: RemoteHost, plan: BenchPlan, *,
                        runner=None, streamer=None) -> bool:
    """Stage ``plan.local_data_dir`` (the trainer box's benchmark data) into
    ``plan.data_dir`` on the pod before a sweep, so the bench command's data
    guard finds warm markers instead of gambling on the on-pod HF download
    (r45–r48: four consecutive rounds lost their reports to that fetch; the
    manual tar-over-ssh of the same data succeeded every time, ~75s for 4.4G).

    Returns True when the pod's data dir holds every requested suite's
    completion marker afterwards; False means "not staged here" and is always
    non-fatal — the bench command's marker-guarded download remains the
    fallback, unchanged. Idempotent: the pod-side marker check runs first, so
    the sequential per-role sweeps on one pod stream the data exactly once.
    Skipped (False, no ssh) when the knob is unset or the LOCAL copy lacks any
    requested suite's marker — never sideload data we can't vouch complete.

    ``runner`` is the ``(ssh_argv, timeout)`` test seam shared with
    :func:`run_post_round_benchmark`; ``streamer`` doubles for
    :func:`_stream_tar` as ``(local_dir, ssh_argv, timeout) -> rc``.
    """
    try:
        if not plan.local_data_dir:
            return False
        suites = _suite_list(plan.suites)
        if not suites:
            return False
        local = Path(plan.local_data_dir)
        missing = [s for s in suites if not (local / s / DATA_MARKER).is_file()]
        if missing:
            log.info("bench sideload skipped: %s missing suite marker(s) %s — "
                     "pod falls back to the on-pod download", local, missing)
            return False
        run = runner or run_ssh
        check = build_ssh_argv(host, f"test {suite_markers(plan.data_dir, suites)}")
        if run(check, 120).returncode == 0:
            log.info("bench data already staged on %s (%s) — sideload skipped",
                     host.name, plan.data_dir)
            return True
        log.info("sideloading bench data %s → %s:%s", local, host.name, plan.data_dir)
        unpack = build_ssh_argv(host, build_sideload_remote_command(plan.data_dir))
        rc = (streamer or _stream_tar)(str(local), unpack,
                                       plan.sideload_timeout_seconds)
        if rc != 0:
            log.warning("bench sideload to %s failed (rc=%s) — pod falls back "
                        "to the on-pod download", host.name, rc)
            return False
        staged = run(check, 120).returncode == 0
        if staged:
            log.info("bench sideload to %s complete (markers verified)", host.name)
        else:
            log.warning("bench sideload to %s streamed but markers still missing "
                        "under %s — pod falls back to the on-pod download",
                        host.name, plan.data_dir)
        return staged
    except Exception as e:  # noqa: BLE001 — bench prep must never raise into a round
        log.warning("bench sideload errored (pod falls back to download): %s", e)
        return False


def run_post_round_benchmark(host: RemoteHost, round_id: str, arch_preset: str,
                             plan: BenchPlan, *, work_root: Path | None = None,
                             runner=None, role: str = "king") -> dict | None:
    """Benchmark the round's final ``role`` checkpoint on ``host`` and return
    the parsed report.

    Blocking (call it from :func:`launch_post_round_benchmark`'s thread).
    Returns ``None`` on any failure — this path must never raise into a round.
    """
    try:
        import os

        # Guaranteed-first data staging: stream the trainer box's local copy to
        # the pod (marker-guarded, no-op when unset/absent/already staged). The
        # bench command's own download guard stays in place as the fallback,
        # so a failed/skipped sideload degrades to exactly the old behavior.
        sideload_bench_data(host, plan, runner=runner)
        token = os.environ.get("HF_TOKEN") or None
        remote_cmd, report_path = build_bench_remote_command(
            host, round_id, arch_preset, plan, role=role, hf_token=token)
        ssh = build_ssh_argv(host, remote_cmd)
        if runner is not None:  # test seam: doubles take (argv, timeout) only
            proc = runner(ssh, plan.timeout_seconds)
        else:
            payload = f"HF_TOKEN={shlex.quote(token)}\n" if token else None
            proc = run_ssh(ssh, plan.timeout_seconds, stdin_text=payload)
        if proc.returncode != 0:
            log.warning("post-round benchmark failed on %s (exit %s): %s",
                        host.name, proc.returncode, (proc.stderr or "")[-400:])
            return None
        cat = (runner or run_ssh)(build_ssh_argv(host, f"cat {shlex.quote(report_path)}"), 120)
        if cat.returncode != 0:
            log.warning("post-round benchmark report missing on %s: %s",
                        host.name, (cat.stderr or "")[-200:])
            return None
        report = json.loads(cat.stdout)
        if work_root is not None:
            local = Path(work_root) / round_id / arch_preset / f"{role}-benchmark_report.json"
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_text(json.dumps(report, indent=2), encoding="utf-8")
        log.info("bench round=%s role=%s %s", round_id, role, format_report(report))
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
