"""Public-benchmark sweeps (GIFT-Eval / BOOM / TIME) on the pod that holds a
checkpoint — the numbers behind the trainer's signed bench reports.

The rolling scheduler benches each checkpoint at completion on the pod that
trained it (DEC-CA-0043); the six numbers feed promotion (DEC-CA-0013), never
the duel verdict: the benchmark data is public (a Goodhart target for
generators) and GPU sweeps are not bit-reproducible across SKUs.

Failure semantics mirror ``cascade.eval.benchmarks``: every failure path logs
and returns ``None`` — a broken, slow, or missing benchmark never fails a leg.
Training always wins the GPU (a training dispatch kills any straggling sweep),
and sweeps on one pod are serialized (:func:`host_bench_lock`).

Command construction is pure and unit-tested; only the runner shells out.
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
    """What a sweep runs. ``suites``/``max_series`` size it; ``data_dir`` must
    hold the pinned benchmark data on the pod (``cascade-benchmark-download
    --data-dir …``)."""

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
    # Detached bench (see remote.run_detached): the sweep runs in its own
    # session on the pod and the orchestrator polls, so an ssh drop mid-bench
    # no longer loses the report. Off by default (test fixture shape); the
    # live trainer arms it from [round] detached_dispatch.
    detached: bool = False
    poll_seconds: int = 30
    reattach_grace_seconds: int = 900
    # fence for the dataset download's wedge mode (unauthenticated HF pulls
    # can deadlock); authenticated pulls finish in ~3 min.
    download_timeout_seconds: int = 2700
    # Benchmark data dir on the TRAINER box to sideload to the pod (tar over
    # ssh) before each sweep — None/absent-dir = old behavior (the pod's own
    # HF download, fenced by download_timeout_seconds, remains the fallback
    # either way). The on-pod fetch is the flaky leg; the 4.4G battery
    # streams in ~75s. Marker-guarded, so sequential roles on one pod stage
    # exactly once.
    local_data_dir: str | None = None
    # Fence on the tar|ssh stream — its own stall guard, separate from (and
    # much tighter than) the bench/download timeouts: the manual stream takes
    # ~75s, so a stream still running at 15 min is wedged, and failing it
    # early leaves the whole bench window to the download fallback.
    sideload_timeout_seconds: int = 900


def _suite_list(suites: str) -> list[str]:
    return [s.strip() for s in str(suites).split(",") if s.strip()]


# All four suites can be sideloaded. FEV is operator-staged and hash-pinned;
# the downloader handles the other three and never fetches FEV implicitly.
STAGED_SUITES = ("gift-eval", "boom", "time", "fev")


def _data_marker(suite: str) -> str:
    return "manifest.json" if suite == "fev" else DATA_MARKER


def _staged(suites: list[str]) -> list[str]:
    return [s for s in suites if s in STAGED_SUITES]


def _uv_extras(plan: BenchPlan) -> str:
    """The sidecar extras every ``uv run`` passes: ``time`` always, ``fev``
    when the sweep runs FEV (``uv run`` syncs exactly — a run without the
    flag would uninstall fev and the suite would report ``skipped``)."""
    return "--frozen --extra time" + (" --extra fev" if "fev" in _suite_list(plan.suites) else "")


def bench_suites_for(cfg, suites: str) -> str:
    """The trainer's sweep suites: ``suites`` (the ``--bench-suites`` flag),
    plus ``fev`` once the DEC-CA-0061 four-suite rule can apply at any block
    (``[v2] bench_gate_v2_from_block``). Unchanged otherwise."""
    from ..shared.era import bench_gate_v2_configured

    names = _suite_list(suites)
    if bench_gate_v2_configured(cfg) and "fev" not in names:
        names.append("fev")
    return ",".join(names)


def suite_markers(data_dir: str, suites: list[str]) -> str:
    """The ``test`` expression asserting every suite's COMPLETION marker under
    ``data_dir`` (`-f m1 -a -f m2 …`) — the one definition of "data is staged"
    shared by the bench data guard and the sideload."""
    return " -a ".join(
        f"-f {shlex.quote(f'{data_dir}/{s}/{_data_marker(s)}')}" for s in suites)



def role_paths(host: RemoteHost, round_id: str, arch_preset: str,
               role: str = "king") -> tuple[str, str]:
    """(checkpoint dir, report path) of a round's final ``role`` on the pod.
    ``role`` is the work-DIR name the leg trained under — the role plus any
    repo suffix (e.g. ``challenger-u<uid>``); callers pass the full form.
    Matches the layout
    ``cascade.trainer.worker`` writes under ``<workdir>/_train_work``."""
    base = f"{host.workdir}/_train_work/{round_id}/{arch_preset}/{role}"
    return f"{base}/checkpoint", f"{base}/benchmark_report.json"


def build_bench_remote_command(host: RemoteHost, round_id: str, arch_preset: str,
                               plan: BenchPlan, *, role: str = "king",
                               hf_token: str | None = None) -> tuple[str, str]:
    """The remote shell string that benchmarks the round's final ``role``
    checkpoint, plus the report path it writes. Pure — safe to unit test.

    ``hf_token`` arms the stdin-env sourcing prefix (the credential itself
    NEVER enters the command string — it travels on stdin exactly like the
    worker dispatch's forwarded creds, see ``remote.build_remote_command``):
    unauthenticated HF dataset pulls are rate-limited and can wedge."""
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
    # every boot) — self-provision on first bench instead.
    # && -chained so a failed download fails the sweep (best-effort upstream)
    # rather than benching against an empty data dir. Same-pod launches are
    # serialized by the caller (grouped by pod address), so the guard never
    # races itself.
    #
    # The guard tests each requested suite's COMPLETION marker, not the bare
    # data dir: `test -d` would turn one interrupted download into a
    # permanent per-pod suite skip. The download is marker-aware and
    # resumable, so re-running over partial data is cheap; `timeout` fences
    # the wedge mode (thread-pool deadlock with no sockets) so a hung
    # download fails this sweep instead of eating the bench window.
    staged = [s for s in _staged(_suite_list(plan.suites)) if s != "fev"]
    markers = suite_markers(plan.data_dir, staged)
    extras = _uv_extras(plan)
    data_guard = (
        f"{{ test {markers} || "
        f"timeout {int(plan.download_timeout_seconds)} "
        f"{plan.uv_bin} run {extras} --project {project} cascade-benchmark-download "
        f"--data-dir {shlex.quote(plan.data_dir)}; }} && "
    ) if staged else ""
    # --extra time on EVERY uv run: timebench is an optional extra, and uv run
    # syncs the env exactly — a run without the flag would UNINSTALL it again
    # and silently skip the TIME suite.
    env_source = "set -a && . /dev/stdin && set +a && " if hf_token else ""
    fev_guard = (f"test -f {shlex.quote(plan.data_dir + '/fev/manifest.json')} && "
                 if "fev" in _suite_list(plan.suites) else "")
    cmd = (
        PREEMPT_BENCHMARKS
        + env_source
        + data_guard
        + fev_guard
        + prefix
        + f"{plan.uv_bin} run {extras} --project {project} "
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


def push_checkpoint(host: RemoteHost, local_checkpoint: str | Path, round_id: str,
                    arch_preset: str, role: str, *, timeout: int = 600,
                    runner=None) -> str:
    """Stage a LOCAL checkpoint directory at ``role``'s ``_train_work``
    checkpoint path on ``host`` (tar over ssh; the target is wiped first so a
    stale sibling can never be benched by mistake). Returns the pod-side
    checkpoint dir — exactly what :func:`role_paths` derives, so a following
    :func:`run_post_round_benchmark` at the same ``role`` benches it.

    This is how the operator's king pod re-benches a checkpoint it did not
    train (DEC-CA-0036 payer-bench verification): the orchestrator already
    holds the guard-verified copy, and pushing ~64MB is cheaper and safer
    than teaching the pod to pull from the Hub."""
    ckpt, _ = role_paths(host, round_id, arch_preset, role)
    remote = (f"rm -rf {shlex.quote(ckpt)} && mkdir -p {shlex.quote(ckpt)} && "
              f"tar -C {shlex.quote(ckpt)} -xf -")
    argv = build_ssh_argv(host, remote)
    if runner is not None:  # test seam
        proc = runner(argv, timeout)
        rc = proc.returncode
    else:
        rc = _stream_tar(str(local_checkpoint), argv, timeout)
    if rc != 0:
        raise RuntimeError(f"checkpoint push to {host.name}:{ckpt} failed (rc={rc})")
    return ckpt


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
        suites = _staged(_suite_list(plan.suites))
        if not suites:
            return False
        local = Path(plan.local_data_dir)
        missing = [s for s in suites if not (local / s / _data_marker(s)).is_file()]
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


# One benchmark sweep per POD at a time. Every launch pre-empts any running
# `cascade-benchmark` on its host (PREEMPT_BENCHMARKS — it clears strays a
# restart left behind), so two concurrent sweeps on one pod would kill each
# other. With the lock the pre-emption only ever hits a stray from another
# process.
_HOST_LOCKS: dict[str, threading.Lock] = {}
_HOST_LOCKS_GUARD = threading.Lock()


def _host_key(host) -> str:
    return f"{getattr(host, 'host', '')}:{getattr(host, 'port', '')}"


def host_bench_lock(host) -> threading.Lock:
    """The lock that serializes benchmark sweeps on ``host``'s pod."""
    key = _host_key(host)
    with _HOST_LOCKS_GUARD:
        lock = _HOST_LOCKS.get(key)
        if lock is None:
            lock = _HOST_LOCKS[key] = threading.Lock()
        return lock


def host_busy(host) -> bool:
    """True while a sweep holds ``host``'s pod (callers defer rather than wait)."""
    return host_bench_lock(host).locked()


def run_post_round_benchmark(host: RemoteHost, round_id: str, arch_preset: str,
                             plan: BenchPlan, *, work_root: Path | None = None,
                             runner=None, role: str = "king") -> dict | None:
    """Benchmark the round's final ``role`` checkpoint on ``host`` and return
    the parsed report.

    Blocking. Returns ``None`` on any failure — this path must never raise into a round.
    Serialized per pod (:func:`host_bench_lock`): a second sweep on the same
    pod waits for the first instead of pre-empting it.
    """
    with host_bench_lock(host):
        return _run_post_round_benchmark(host, round_id, arch_preset, plan,
                                         work_root=work_root, runner=runner, role=role)


def _run_post_round_benchmark(host: RemoteHost, round_id: str, arch_preset: str,
                              plan: BenchPlan, *, work_root: Path | None = None,
                              runner=None, role: str = "king") -> dict | None:
    try:
        import os

        # Guaranteed-first data staging: stream the trainer box's local copy to
        # the pod (marker-guarded, no-op when unset/absent/already staged). The
        # bench command's own download guard stays in place as the fallback,
        # so a failed/skipped sideload degrades to exactly the old behavior.
        sideload_bench_data(host, plan, runner=runner)
        # An ISOLATED host (a payer's pod, DEC-CA-0036) gets no operator
        # credential of any kind — the sideload above is what stages its
        # data; a failed sideload degrades to a token-less download there.
        token = (None if getattr(host, "isolated", False)
                 else os.environ.get("HF_TOKEN") or None)
        payload = f"HF_TOKEN={shlex.quote(token)}\n" if token else None
        if plan.detached:
            # The launcher sources the token from stdin (exported ⇒ inherited by
            # the detached session); the body itself carries no env prefix.
            from .remote import PREEMPT_BENCHMARKS, detached_run_dir, run_detached

            body, report_path = build_bench_remote_command(
                host, round_id, arch_preset, plan, role=role, hf_token=None)
            body = body[len(PREEMPT_BENCHMARKS):] if body.startswith(PREEMPT_BENCHMARKS) else body

            def _call(argv, t, stdin=None):
                return runner(argv, t) if runner is not None else run_ssh(argv, t, stdin)

            proc = run_detached(
                host, body, payload, detached_run_dir(host, f"bench-{role}-{round_id}"),
                timeout=plan.timeout_seconds, poll_seconds=plan.poll_seconds,
                grace_seconds=plan.reattach_grace_seconds, runner=_call,
                describe=f"bench {role} on {host.name}")
        else:
            remote_cmd, report_path = build_bench_remote_command(
                host, round_id, arch_preset, plan, role=role, hf_token=token)
            ssh = build_ssh_argv(host, remote_cmd)
            if runner is not None:  # test seam: doubles take (argv, timeout) only
                proc = runner(ssh, plan.timeout_seconds)
            else:
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
