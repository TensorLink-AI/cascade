"""The cascade side of the concurrent / paired benchmark sweep: the sidecar
argv (``--workers``, ``--extra-checkpoint``), the remote command, the paired
remote pass, and the report-only timing script's pure parts."""

from __future__ import annotations

import importlib.util
import json
import shlex
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import cascade.eval.benchmarks as bench
from cascade.trainer.remote import RemoteHost
from cascade.validator.eval_offload import build_bench_remote_command, run_bench_via_host_many

_spec = importlib.util.spec_from_file_location(
    "bench_timing", Path(__file__).resolve().parents[2] / "scripts" / "bench_timing.py")
timing = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(timing)


def _host():
    return RemoteHost(name="gpu1", host="1.2.3.4", user="root", port=22, workdir="/w")


def test_bench_workers_arg_keeps_the_sequential_argv():
    assert bench.bench_workers_arg(None) == [] and bench.bench_workers_arg("1") == []
    assert bench.bench_workers_arg(1) == []
    assert bench.bench_workers_arg("auto") == ["--workers", "auto"]
    assert bench.bench_workers_arg(4) == ["--workers", "4"]


def test_paired_local_pass_is_one_sidecar_call(monkeypatch, tmp_path):
    ckpts = []
    for n in ("chal", "king"):
        d = tmp_path / n
        d.mkdir()
        (d / "forecast_wrapper.py").write_text("x", encoding="utf-8")
        ckpts.append(d)
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    monkeypatch.setattr("cascade.eval.checkpoint_guard.verify_checkpoint_code", lambda c: None)
    seen = []

    def fake_run(cmd, **_k):
        seen.append(cmd)
        outs = [cmd[cmd.index("cascade-benchmark") + 2]] + [cmd[i + 2] for i, a in enumerate(cmd) if a == "--extra-checkpoint"]
        for k, o in enumerate(outs):
            Path(o).write_text(json.dumps({"checkpoint": str(k), "suites": []}),
                               encoding="utf-8")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    got = bench.run_bench_reports(ckpts, project_dir=tmp_path, suites=("gift-eval",),
                                  workers="auto", uv_bin="uv")
    assert len(seen) == 1
    cmd = seen[0]
    assert cmd[cmd.index("cascade-benchmark") + 1] == str(ckpts[0]) and cmd[cmd.index("--extra-checkpoint") + 1] == str(ckpts[1])
    assert cmd[cmd.index("--workers") + 1] == "auto"
    assert [r["checkpoint"] for r in got] == ["0", "1"]


def test_a_refused_checkpoint_leaves_the_pass_to_the_others(monkeypatch, tmp_path):
    good = tmp_path / "good"
    good.mkdir()
    (good / "forecast_wrapper.py").write_text("x", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    monkeypatch.setattr("cascade.eval.checkpoint_guard.verify_checkpoint_code", lambda c: None)

    def fake_run(cmd, **_k):
        assert "--extra-checkpoint" not in cmd
        Path(cmd[cmd.index("cascade-benchmark") + 2]).write_text("{}", encoding="utf-8")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    got = bench.run_bench_reports([tmp_path / "missing", good], project_dir=tmp_path,
                                  uv_bin="uv")
    assert got == [None, {}]


def test_remote_command_carries_workers_and_extra_pairs():
    cmd = build_bench_remote_command(_host(), "/w/c0", "/w/r0.json", suites="gift-eval,boom",
                                     workers="auto", extra=(("/w/c1", "/w/r1.json"),))
    argv = shlex.split(cmd)
    assert argv[argv.index("--workers") + 1] == "auto"
    i = argv.index("--extra-checkpoint")
    assert argv[i + 1:i + 3] == ["/w/c1", "/w/r1.json"]
    plain = shlex.split(build_bench_remote_command(_host(), "/w/c", "/w/r", suites="boom"))
    assert "--workers" not in plain and "--extra-checkpoint" not in plain


def test_paired_remote_pass_scps_each_and_cats_each_report(tmp_path):
    calls = []

    def runner(argv, _timeout):
        calls.append(argv)
        text = " ".join(argv)
        if "cat " in text:
            k = text.rsplit("report-", 1)[1].split(".json")[0]
            return SimpleNamespace(returncode=0, stdout=json.dumps({"k": int(k)}), stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    got = run_bench_via_host_many(_host(), [tmp_path / "a", tmp_path / "b"], suites="time",
                                  workers="3", runner=runner)
    assert got == [{"k": 0}, {"k": 1}]
    assert sum(1 for c in calls if c[0] == "scp") == 2
    bench_cmd = next(c[-1] for c in calls if "cascade-benchmark" in c[-1])
    assert "--extra-checkpoint" in bench_cmd and "--workers 3" in bench_cmd


def test_timing_script_pure_parts():
    assert timing.parse_worker_list("auto,4") == ["1", "auto", "4"]
    assert timing.parse_worker_list("1, 2,2,AUTO") == ["1", "2", "auto"]
    with pytest.raises(ValueError):
        timing.parse_worker_list("0")
    with pytest.raises(ValueError):
        timing.parse_worker_list("many")
    assert timing.identical(["a", "b"], ["a", "b"])
    assert not timing.identical(["a"], ["a", "b"]) and not timing.identical(["a"], ["b"])
    cmd = timing.build_command("uv", "bm", "/c", "/o.json", workers="auto", timings_json="/t",
                               suites="time", device="cuda", batch_size=512, max_series=0,
                               num_samples=100, data_dir="/d", extra=[("/k", "/k.json")])
    assert cmd[cmd.index("--workers") + 1] == "auto" and "--extra-checkpoint" in cmd
    runs = {"1": {"total_s": 100.0, "plan": "sequential",
                  "suites": {"gift-eval": {"wall_s": 60.0}, "time": {"wall_s": 40.0}}},
            "auto": {"total_s": 25.0, "plan": "workers=4",
                     "suites": {"gift-eval": {"wall_s": 15.0}, "time": {"wall_s": 20.0}}}}
    lines = timing.speedup_table(runs)
    assert "4.00x" in lines[1] and "2.00x" in lines[2] and "4.00x" in lines[3]
    assert lines[-1].strip().startswith("workers=auto: workers=4")
