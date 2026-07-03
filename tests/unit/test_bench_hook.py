"""Post-round benchmark hook — pure command construction plus the log-only
failure contract: nothing in this path may ever raise into the round loop."""

from __future__ import annotations

import json
from pathlib import Path
from subprocess import CompletedProcess

from cascade.trainer.bench_hook import (
    BenchPlan,
    build_bench_remote_command,
    king_paths,
    launch_post_round_benchmark,
    run_post_round_benchmark,
)
from cascade.trainer.remote import RemoteHost

HOST = RemoteHost(name="pod", host="1.2.3.4", workdir="/root/cascade", cuda_device="0")


def test_king_paths_match_worker_layout():
    ckpt, report = king_paths(HOST, "42", "toto2-4m")
    assert ckpt == "/root/cascade/_train_work/42/toto2-4m/king/checkpoint"
    assert report == "/root/cascade/_train_work/42/toto2-4m/king/benchmark_report.json"


def test_build_bench_remote_command():
    cmd, report = build_bench_remote_command(HOST, "42", "toto2-4m", BenchPlan())
    # bracketed pattern: kills previous benchmarks without self-matching this shell
    assert cmd.startswith("pkill -f 'bin/cascade[-]benchmark'")
    assert "CUDA_VISIBLE_DEVICES=0" in cmd
    assert "--project /root/cascade/benchmarks" in cmd
    assert "--suites gift-eval,boom,time" in cmd and "--device cuda" in cmd
    assert "--max-series" not in cmd  # 0 = full benchmark
    assert report.endswith("king/benchmark_report.json")
    capped, _ = build_bench_remote_command(
        HOST, "42", "toto2-4m", BenchPlan(max_series=8, suites="gift-eval"))
    assert "--max-series 8" in capped and "--suites gift-eval" in capped


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


def test_training_dispatch_preempts_benchmarks():
    from cascade.trainer.remote import build_remote_command

    cmd = build_remote_command(HOST, ["python", "-m", "cascade.trainer.worker"], {})
    assert cmd.startswith("pkill -f 'bin/cascade[-]benchmark'")  # training always wins
    assert "cd /root/cascade &&" in cmd


def test_min_interval_skips_back_to_back_launches(monkeypatch):
    import cascade.trainer.bench_hook as bh

    monkeypatch.setattr(bh, "_last_launch", {})
    monkeypatch.setattr(bh, "run_post_round_benchmark", lambda *a, **k: None)
    plan = BenchPlan(min_interval_seconds=3600)
    t1 = launch_post_round_benchmark(HOST, "1", "toto2-4m", plan)
    assert t1 is not None
    t1.join(timeout=5)
    assert launch_post_round_benchmark(HOST, "2", "toto2-4m", plan) is None  # too soon
    assert launch_post_round_benchmark(HOST, "3", "toto2-4m", BenchPlan()) is not None


def test_launch_is_fire_and_forget(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("must be swallowed inside run_post_round_benchmark")

    monkeypatch.setattr("cascade.trainer.bench_hook.build_ssh_argv", boom)
    t = launch_post_round_benchmark(HOST, "42", "toto2-4m", BenchPlan())
    t.join(timeout=10)
    assert not t.is_alive()
