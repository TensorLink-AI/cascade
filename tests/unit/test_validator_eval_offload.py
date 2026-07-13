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
    build_gift_remote_command,
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


@dataclass
class _Proc:
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""


class _FakeRunner:
    """Records every argv; returns the report JSON for the `cat` step, ok else."""

    def __init__(self, gift_returncode=0, scp_returncode=0):
        self.calls: list[list[str]] = []
        self.gift_returncode = gift_returncode
        self.scp_returncode = scp_returncode

    def __call__(self, argv, timeout):
        self.calls.append(argv)
        joined = " ".join(argv)
        if argv and argv[0] == "scp":
            return _Proc(returncode=self.scp_returncode)
        if "cascade-benchmark" in joined:
            return _Proc(returncode=self.gift_returncode)
        if "cat " in joined:
            return _Proc(stdout=json.dumps(_REPORT))
        return _Proc()  # prep / cleanup


# ── pure command builders ────────────────────────────────────────────────────

def test_build_scp_argv_uses_capital_P_port_and_key():
    argv = build_scp_argv(_host(), "/local/ckpt/.", "/root/cascade/_eval_offload/ckpt/checkpoint")
    assert argv[0] == "scp" and "-r" in argv
    assert "-P" in argv and argv[argv.index("-P") + 1] == "40123"  # scp uses -P, not -p
    assert "-i" in argv
    assert argv[-2] == "/local/ckpt/."
    assert argv[-1] == "root@9.9.9.9:/root/cascade/_eval_offload/ckpt/checkpoint"


def test_build_gift_remote_command_single_suite_on_gpu():
    cmd = build_gift_remote_command(
        _host(), "/r/ckpt", "/r/out.json",
        datasets="m4_hourly", num_samples=20, data_dir="/root/cascade/bench_data")
    assert "cascade-benchmark /r/ckpt /r/out.json" in cmd
    assert "--suites gift-eval" in cmd            # never runs boom/time
    assert "--device cuda" in cmd
    assert "--gifteval-datasets m4_hourly" in cmd
    assert "--data-dir /root/cascade/bench_data" in cmd
    assert cmd.startswith("CUDA_VISIBLE_DEVICES=0 ")  # pins the pod's device
    assert "--project /root/cascade/benchmarks" in cmd


def test_build_gift_remote_command_omits_device_prefix_when_unset():
    cmd = build_gift_remote_command(_host(cuda_device=None), "/r/ckpt", "/r/out.json")
    assert not cmd.startswith("CUDA_VISIBLE_DEVICES")


# ── dispatch ────────────────────────────────────────────────────────────────

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
    # It scp'd the checkpoint and ran gift-eval on the pod.
    assert any(c[0] == "scp" for c in runner.calls)
    assert any("cascade-benchmark" in " ".join(c) for c in runner.calls)


def test_gift_rows_via_host_returns_none_when_benchmark_fails(tmp_path):
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    runner = _FakeRunner(gift_returncode=1)   # gift-eval errored on the pod
    rows = gift_rows_via_host(_host(), ckpt, runner=runner)
    assert rows is None                        # ⇒ caller treats the gate as uncomputable


def test_gift_rows_via_host_returns_none_when_scp_fails(tmp_path):
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    runner = _FakeRunner(scp_returncode=1)
    assert gift_rows_via_host(_host(), ckpt, runner=runner) is None


# ── shared parse helper ──────────────────────────────────────────────────────

def test_gift_rows_from_report_parses_and_handles_missing():
    assert gift_rows_from_report(_REPORT)["status"] == "ok"
    assert gift_rows_from_report(_REPORT)["revision"] == "abc123"
    assert gift_rows_from_report(None) is None
    assert gift_rows_from_report({"suites": []}) is None  # no gift-eval suite
