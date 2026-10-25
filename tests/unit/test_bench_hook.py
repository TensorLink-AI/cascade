"""Post-round benchmark hook — pure command construction plus the log-only
failure contract: nothing in this path may ever raise into the round loop."""

from __future__ import annotations

import json
from pathlib import Path
from subprocess import CompletedProcess

from cascade.trainer.bench_hook import (
    DATA_MARKER,
    BenchPlan,
    build_bench_remote_command,
    build_sideload_remote_command,
    role_paths,
    run_post_round_benchmark,
    sideload_bench_data,
)
from cascade.trainer.remote import PREEMPT_BENCHMARKS, RemoteHost

HOST = RemoteHost(name="pod", host="1.2.3.4", workdir="/root/cascade", cuda_device="0")


def test_role_paths_match_worker_layout():
    ckpt, report = role_paths(HOST, "42", "toto2-4m", "king")
    assert ckpt == "/root/cascade/_train_work/42/toto2-4m/king/checkpoint"
    assert report == "/root/cascade/_train_work/42/toto2-4m/king/benchmark_report.json"


def test_build_bench_remote_command():
    cmd, report = build_bench_remote_command(HOST, "42", "toto2-4m", BenchPlan())
    # bracketed pattern: kills previous benchmarks without self-matching this shell
    assert cmd.startswith(PREEMPT_BENCHMARKS)
    assert "CUDA_VISIBLE_DEVICES=0" in cmd
    assert "--project /root/cascade/benchmarks" in cmd
    assert "--suites gift-eval,boom,time" in cmd and "--device cuda" in cmd
    assert "--max-series" not in cmd  # 0 = full benchmark
    assert report.endswith("king/benchmark_report.json")
    capped, _ = build_bench_remote_command(
        HOST, "42", "toto2-4m", BenchPlan(max_series=8, suites="gift-eval"))
    assert "--max-series 8" in capped and "--suites gift-eval" in capped


def test_data_guard_tests_per_suite_markers_not_the_bare_dir():
    # `test -d` turned ONE interrupted download into a permanent per-pod TIME
    # skip (the r13–r15 bench-report drought): the dir existed, so the
    # marker-aware resumable download never re-ran. The guard now demands every
    # requested suite's completion marker, and the download runs under
    # `timeout` so its wedge mode (2026-08-12: thread-pool deadlock, no open
    # sockets) fails the sweep instead of eating the bench window.
    cmd, _ = build_bench_remote_command(HOST, "42", "toto2-4m", BenchPlan())
    assert "test -d" not in cmd
    for suite in ("gift-eval", "boom", "time"):
        assert f"/root/bench_data/{suite}/_cascade_revision.json" in cmd
    assert "timeout 2700" in cmd
    one, _ = build_bench_remote_command(
        HOST, "42", "toto2-4m", BenchPlan(suites="gift-eval"))
    assert "boom/_cascade_revision.json" not in one


def test_uv_runs_carry_the_time_extra():
    # timebench is an OPTIONAL extra and `uv run` syncs the env EXACTLY, so
    # every invocation must ask for it — a single bare run uninstalls it again
    # (its absence was the TIME-skip root cause, 2026-08-05 bootstrap rework).
    cmd, _ = build_bench_remote_command(HOST, "42", "toto2-4m", BenchPlan())
    runs = [seg for seg in cmd.split("&&") if " run " in seg]
    assert runs and all("--extra time" in seg for seg in runs)


def test_hf_token_arms_stdin_sourcing_but_never_enters_the_command():
    plain, _ = build_bench_remote_command(HOST, "42", "toto2-4m", BenchPlan())
    assert "/dev/stdin" not in plain
    armed, _ = build_bench_remote_command(
        HOST, "42", "toto2-4m", BenchPlan(), hf_token="hf_secret123")
    assert "set -a && . /dev/stdin && set +a" in armed
    assert "hf_secret123" not in armed  # credential travels on stdin only






def test_run_post_round_benchmark_saves_and_returns_report(tmp_path: Path):
    report = {"checkpoint": "x", "suites": [
        {"suite": "gift-eval", "status": "ok", "metrics": {"crps": 0.5}, "n_series": 3}]}
    calls = []

    def runner(argv, timeout):
        calls.append(argv)
        out = json.dumps(report) if len(calls) > 1 else ""  # 1st = run, 2nd = cat
        return CompletedProcess(argv, 0, stdout=out, stderr="")

    got = run_post_round_benchmark(
        HOST, "42", "toto2-4m", BenchPlan(), work_root=tmp_path, runner=runner)
    assert got == report
    saved = tmp_path / "42" / "toto2-4m" / "king-benchmark_report.json"
    assert json.loads(saved.read_text()) == report


def test_run_post_round_benchmark_never_raises():
    def boom(argv, timeout):
        raise OSError("ssh exploded")

    assert run_post_round_benchmark(HOST, "42", "toto2-4m", BenchPlan(), runner=boom) is None

    def fails(argv, timeout):
        return CompletedProcess(argv, 1, stdout="", stderr="cuda OOM")

    assert run_post_round_benchmark(HOST, "42", "toto2-4m", BenchPlan(), runner=fails) is None


def _make_local_data(tmp_path: Path, suites=("gift-eval", "boom", "time")) -> Path:
    local = tmp_path / "bench_data"
    for s in suites:
        (local / s).mkdir(parents=True)
        (local / s / DATA_MARKER).write_text("{}")
    return local


def test_sideload_noop_when_unconfigured_or_local_incomplete(tmp_path: Path):
    def never(argv, timeout):  # pragma: no cover — must not be reached
        raise AssertionError("no ssh may run for a skipped sideload")

    # default plan: knob unset ⇒ old behavior, zero ssh traffic
    assert sideload_bench_data(HOST, BenchPlan(), runner=never) is False
    # dir missing entirely
    plan = BenchPlan(local_data_dir=str(tmp_path / "nope"))
    assert sideload_bench_data(HOST, plan, runner=never) is False
    # dir present but a requested suite's completion marker missing: never
    # sideload data we can't vouch complete — the pod download tops up instead
    local = _make_local_data(tmp_path, suites=("gift-eval", "boom"))
    plan = BenchPlan(local_data_dir=str(local))  # requests time too
    assert sideload_bench_data(HOST, plan, runner=never) is False


def test_sideload_skips_when_pod_already_staged(tmp_path: Path):
    # Idempotence: multiple bench roles run sequentially on one pod — only the
    # first may stream; the rest must hit the marker check and return.
    local = _make_local_data(tmp_path)
    checks = []

    def runner(argv, timeout):
        checks.append(argv)
        return CompletedProcess(argv, 0, stdout="", stderr="")

    def never_stream(local_dir, ssh_argv, timeout):  # pragma: no cover
        raise AssertionError("staged pod must not be re-streamed")

    plan = BenchPlan(local_data_dir=str(local))
    assert sideload_bench_data(HOST, plan, runner=runner, streamer=never_stream) is True
    assert len(checks) == 1
    # the pod-side check tests every requested suite's completion marker
    remote = checks[0][-1]
    for suite in ("gift-eval", "boom", "time"):
        assert f"/root/bench_data/{suite}/{DATA_MARKER}" in remote


def test_sideload_streams_then_verifies(tmp_path: Path):
    local = _make_local_data(tmp_path)
    calls = []

    def runner(argv, timeout):
        calls.append(("check", argv))
        # 1st check: cold pod (rc 1); 2nd check (post-stream verify): staged
        return CompletedProcess(argv, 1 if len(calls) == 1 else 0, stdout="", stderr="")

    streams = []

    def streamer(local_dir, ssh_argv, timeout):
        streams.append((local_dir, ssh_argv, timeout))
        calls.append(("stream", ssh_argv))
        return 0

    plan = BenchPlan(local_data_dir=str(local))
    assert sideload_bench_data(HOST, plan, runner=runner, streamer=streamer) is True
    assert [kind for kind, _ in calls] == ["check", "stream", "check"]
    local_dir, ssh_argv, timeout = streams[0]
    assert local_dir == str(local)
    assert timeout == plan.sideload_timeout_seconds
    # the receiving end mkdir-p's and untars into the pod's data_dir
    assert ssh_argv[-1] == "mkdir -p /root/bench_data && tar -C /root/bench_data -xf -"
    assert ssh_argv[-1] == build_sideload_remote_command(plan.data_dir)


def test_sideload_failure_is_nonfatal_fallback_to_download(tmp_path: Path):
    local = _make_local_data(tmp_path)
    plan = BenchPlan(local_data_dir=str(local))

    def cold(argv, timeout):
        return CompletedProcess(argv, 1, stdout="", stderr="")

    # stream fails (e.g. its own timeout fence, rc 124) ⇒ False, never raises
    assert sideload_bench_data(
        HOST, plan, runner=cold, streamer=lambda *a: 124) is False
    # stream "succeeds" but the verify still misses markers ⇒ False
    assert sideload_bench_data(
        HOST, plan, runner=cold, streamer=lambda *a: 0) is False

    def boom(argv, timeout):
        raise OSError("ssh exploded")

    assert sideload_bench_data(HOST, plan, runner=boom) is False


def test_run_post_round_benchmark_sideloads_before_the_sweep(tmp_path: Path):
    # With the knob set and the pod already staged, the first ssh the run makes
    # is the sideload's marker check — the data can never again be a gamble
    # inside the bench window (r45–r48).
    local = _make_local_data(tmp_path)
    report = {"checkpoint": "x", "suites": []}
    calls = []

    def runner(argv, timeout):
        calls.append(argv[-1])
        # 1st = sideload marker check, 2nd = bench run, 3rd = cat report
        out = json.dumps(report) if len(calls) == 3 else ""
        return CompletedProcess(argv, 0, stdout=out, stderr="")

    plan = BenchPlan(local_data_dir=str(local))
    got = run_post_round_benchmark(HOST, "42", "toto2-4m", plan,
                                   work_root=tmp_path, runner=runner)
    assert got == report
    assert len(calls) == 3
    assert calls[0].startswith("test -f") and DATA_MARKER in calls[0]
    assert "cascade-benchmark" in calls[1]


def test_training_dispatch_preempts_benchmarks():
    from cascade.trainer.remote import build_remote_command

    cmd, _ = build_remote_command(HOST, ["python", "-m", "cascade.trainer.worker"], {})
    assert cmd.startswith(PREEMPT_BENCHMARKS)  # training always wins
    assert "cd /root/cascade &&" in cmd


# ── DEC-CA-0036: benching on a payer's (isolated) pod ─────────────────────────

def test_isolated_host_gets_no_hf_token(monkeypatch):
    """A payer pod is credential-free: the operator's HF_TOKEN never arms the
    stdin sourcing there, however the environment is set."""
    monkeypatch.setenv("HF_TOKEN", "hf_operator_secret")
    from dataclasses import replace

    seen = []

    def runner(argv, timeout):
        seen.append(argv[-1])
        return CompletedProcess(argv, 1, stdout="", stderr="")

    run_post_round_benchmark(replace(HOST, isolated=True), "42", "toto2-4m",
                             BenchPlan(), runner=runner)
    assert seen and "/dev/stdin" not in seen[0]
    seen.clear()
    run_post_round_benchmark(HOST, "42", "toto2-4m", BenchPlan(), runner=runner)
    assert "/dev/stdin" in seen[0]   # the operator's own pod still sources it


def test_push_checkpoint_stages_at_the_role_path_and_wipes_first(tmp_path: Path):
    from cascade.trainer.bench_hook import push_checkpoint, role_paths

    calls = []

    def runner(argv, timeout):
        calls.append(argv[-1])
        return CompletedProcess(argv, 0, stdout="", stderr="")

    dest = push_checkpoint(HOST, tmp_path, "42", "toto2-4m", "challenger-u3-verify",
                           runner=runner)
    ckpt, _ = role_paths(HOST, "42", "toto2-4m", "challenger-u3-verify")
    assert dest == ckpt
    (remote,) = calls
    assert remote.startswith(f"rm -rf {ckpt} && mkdir -p {ckpt} && tar -C {ckpt} -xf -")


def test_push_checkpoint_raises_on_a_failed_stream(tmp_path: Path):
    import pytest

    from cascade.trainer.bench_hook import push_checkpoint

    def runner(argv, timeout):
        return CompletedProcess(argv, 255, stdout="", stderr="")

    with pytest.raises(RuntimeError, match="checkpoint push"):
        push_checkpoint(HOST, tmp_path, "42", "toto2-4m", "king-verify", runner=runner)


def test_sweeps_on_one_pod_are_serialized_and_other_pods_run_in_parallel():
    """Every launch pre-empts running sweeps on its host (PREEMPT_BENCHMARKS),
    so two sweeps on ONE pod must never overlap (2026-09-29: verification
    re-benches killed each other, rc 143); different pods stay concurrent."""
    import threading
    import time

    from cascade.trainer.bench_hook import host_busy

    report = {"checkpoint": "x", "suites": []}
    active: dict[str, int] = {}
    peak: dict[str, int] = {}
    guard = threading.Lock()

    def runner_for(key):
        def runner(argv, timeout):
            if "cat " in " ".join(argv):
                return CompletedProcess(argv, 0, stdout=json.dumps(report), stderr="")
            with guard:
                active[key] = active.get(key, 0) + 1
                peak[key] = max(peak.get(key, 0), active[key])
            time.sleep(0.15)
            with guard:
                active[key] -= 1
            return CompletedProcess(argv, 0, stdout="", stderr="")
        return runner

    other = RemoteHost(name="pod2", host="5.6.7.8", workdir="/root/cascade", cuda_device="0")
    threads = [threading.Thread(target=run_post_round_benchmark,
                                args=(h, "42", "toto2-4m", BenchPlan()),
                                kwargs={"runner": runner_for(h.host), "role": f"r{i}"})
               for i, h in enumerate([HOST, HOST, HOST, other, other])]
    for t in threads:
        t.start()
    time.sleep(0.05)
    assert host_busy(HOST)
    for t in threads:
        t.join(5)
    assert peak[HOST.host] == 1 and peak[other.host] == 1
    assert not host_busy(HOST) and not host_busy(other)
