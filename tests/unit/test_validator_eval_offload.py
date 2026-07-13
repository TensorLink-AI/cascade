"""The validator's GIFT-Eval gate can be offloaded to a GPU pod.

Only the gift-eval compute crosses to the pod (scp the fetched checkpoint →
run ``cascade-benchmark --suites gift-eval`` → pull the report); the paired
bootstrap and every consensus decision stay on the orchestrator. Failures
return ``None`` (gate uncomputable), never raise.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from cascade.eval.benchmarks import gift_rows_from_report
from cascade.trainer.remote import RemoteHost
from cascade.validator.eval_offload import (
    bench_scores_via_host,
    build_bench_remote_command,
    build_scp_argv,
    gift_rows_via_host,
)


def _host(cuda_device="0"):
    return RemoteHost(
        name="eval-pod", host="9.9.9.9", port=40123, user="root",
        key_path="~/.ssh/k", remote_python="/root/cascade/.venv/bin/python",
        workdir="/root/cascade", cuda_device=cuda_device, stage="final",
    )


_REPORT = {
    "checkpoint": "/root/cascade/_eval_offload/ckpt/checkpoint",
    "data_revisions": {"gift-eval": "abc123"},
    "suites": [
        {"suite": "gift-eval", "status": "ok",
         "rows": [{"full": "m4_hourly", "crps_ratio": 0.9, "mase_ratio": 0.95}]},
    ],
}


# A full 3-suite report for the cascade bench (extract_bench_scores reads
# each suite's `metrics` crps/mase).
_BENCH_REPORT = {
    "checkpoint": "/root/cascade/_eval_offload/ckpt/checkpoint",
    "suites": [
        {"suite": "gift-eval", "status": "ok", "metrics": {"crps": 0.42, "mase": 0.81}},
        {"suite": "boom", "status": "ok", "metrics": {"crps": 0.55, "mase": 0.90}},
        {"suite": "time", "status": "ok", "metrics": {"crps": 0.38, "mase": 0.77}},
    ],
}


@dataclass
class _Proc:
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""


class _FakeRunner:
    """Records every argv; returns ``report`` JSON for the `cat` step, ok else."""

    def __init__(self, *, report=None, bench_returncode=0, scp_returncode=0):
        self.calls: list[list[str]] = []
        self.report = report if report is not None else _REPORT
        self.bench_returncode = bench_returncode
        self.scp_returncode = scp_returncode

    def __call__(self, argv, timeout):
        self.calls.append(argv)
        joined = " ".join(argv)
        if argv and argv[0] == "scp":
            return _Proc(returncode=self.scp_returncode)
        if "cascade-benchmark" in joined:
            return _Proc(returncode=self.bench_returncode)
        if "cat " in joined:
            return _Proc(stdout=json.dumps(self.report))
        return _Proc()  # prep / cleanup

    def bench_cmd(self):
        return next(" ".join(c) for c in self.calls if "cascade-benchmark" in " ".join(c))


# ── pure command builders ────────────────────────────────────────────────────

def test_build_scp_argv_uses_capital_P_port_and_key():
    argv = build_scp_argv(_host(), "/local/ckpt/.", "/root/cascade/_eval_offload/ckpt/checkpoint")
    assert argv[0] == "scp" and "-r" in argv
    assert "-P" in argv and argv[argv.index("-P") + 1] == "40123"  # scp uses -P, not -p
    assert "-i" in argv
    assert argv[-2] == "/local/ckpt/."
    assert argv[-1] == "root@9.9.9.9:/root/cascade/_eval_offload/ckpt/checkpoint"


def test_build_bench_remote_command_gift_gate_single_suite():
    cmd = build_bench_remote_command(
        _host(), "/r/ckpt", "/r/out.json", suites="gift-eval",
        datasets="m4_hourly", num_samples=20, data_dir="/root/cascade/bench_data")
    assert "cascade-benchmark /r/ckpt /r/out.json" in cmd
    assert "--suites gift-eval" in cmd            # gate never runs boom/time
    assert "--device cuda" in cmd
    assert "--gifteval-datasets m4_hourly" in cmd
    assert "--data-dir /root/cascade/bench_data" in cmd
    assert "--max-series" not in cmd              # datasets path, not max_series
    assert cmd.startswith("CUDA_VISIBLE_DEVICES=0 ")  # pins the pod's device
    assert "--project /root/cascade/benchmarks" in cmd


def test_build_bench_remote_command_cascade_all_suites_with_max_series():
    cmd = build_bench_remote_command(
        _host(), "/r/ckpt", "/r/out.json", suites="gift-eval,boom,time",
        num_samples=20, max_series=3)
    assert "--suites gift-eval,boom,time" in cmd   # cascade bench runs all three
    assert "--max-series 3" in cmd
    assert "--gifteval-datasets" not in cmd


def test_build_bench_remote_command_omits_device_prefix_when_unset():
    cmd = build_bench_remote_command(_host(cuda_device=None), "/r/ckpt", "/r/out.json",
                                     suites="gift-eval")
    assert not cmd.startswith("CUDA_VISIBLE_DEVICES")


# ── dispatch: gift-eval gate ─────────────────────────────────────────────────

def test_gift_rows_via_host_dispatches_and_parses(tmp_path):
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    runner = _FakeRunner()
    rows = gift_rows_via_host(
        _host(), ckpt, datasets="", num_samples=20,
        data_dir="/root/cascade/bench_data", runner=runner)
    # Parsed the same shape the local sidecar returns.
    assert rows == {"status": "ok", "revision": "abc123",
                    "rows": [{"full": "m4_hourly", "crps_ratio": 0.9, "mase_ratio": 0.95}]}
    # It scp'd the checkpoint and ran ONLY gift-eval on the pod.
    assert any(c[0] == "scp" for c in runner.calls)
    assert "--suites gift-eval " in runner.bench_cmd() + " "


def test_gift_rows_via_host_returns_none_when_benchmark_fails(tmp_path):
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    runner = _FakeRunner(bench_returncode=1)   # gift-eval errored on the pod
    rows = gift_rows_via_host(_host(), ckpt, runner=runner)
    assert rows is None                        # ⇒ caller treats the gate as uncomputable


def test_gift_rows_via_host_returns_none_when_scp_fails(tmp_path):
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    runner = _FakeRunner(scp_returncode=1)
    assert gift_rows_via_host(_host(), ckpt, runner=runner) is None


# ── dispatch: cascade bench (GIFT-Eval + BOOM + TIME) ────────────────────────

def test_bench_scores_via_host_dispatches_all_suites_and_parses(tmp_path):
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    runner = _FakeRunner(report=_BENCH_REPORT)
    scores = bench_scores_via_host(_host(), ckpt, num_samples=20, max_series=3,
                                   data_dir="/root/cascade/bench_data", runner=runner)
    assert scores == {
        "gifteval_crps": 0.42, "gifteval_mase": 0.81,
        "boom_crps": 0.55, "boom_mase": 0.90,
        "time_crps": 0.38, "time_mase": 0.77,
    }
    assert "--suites gift-eval,boom,time" in runner.bench_cmd()  # all three suites
    assert "--max-series 3" in runner.bench_cmd()


def test_bench_scores_via_host_returns_none_on_incomplete_report(tmp_path):
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    # Only gift-eval present ⇒ extract_bench_scores wants all three ⇒ None.
    runner = _FakeRunner(report=_REPORT)
    assert bench_scores_via_host(_host(), ckpt, runner=runner) is None


# ── shared parse helper ──────────────────────────────────────────────────────

def test_gift_rows_from_report_parses_and_handles_missing():
    assert gift_rows_from_report(_REPORT)["status"] == "ok"
    assert gift_rows_from_report(_REPORT)["revision"] == "abc123"
    assert gift_rows_from_report(None) is None
    assert gift_rows_from_report({"suites": []}) is None  # no gift-eval suite
