"""A pod rented as an N-GPU layout must expose N GPUs: Lium sold an 8xH200
executor with 6 (testnet 2026-10-30). The check rides the host-bench gate, so
the king and funded rent loops release a short pod and rent again."""
import subprocess

import pytest

from cascade.trainer import remote as remote_mod
from cascade.trainer.remote import probe_gpu_count
from tests.unit.test_funded_pod_wiring import _runner
from tests.unit.test_host_bench_floor import _host


def _smi(n):
    return "".join(f"GPU {i}: NVIDIA H200 NVL (UUID: GPU-{i})\n" for i in range(n))


def test_probe_counts_the_listed_gpus(monkeypatch):
    seen = {}

    def fake_run(argv, **kw):
        seen["cmd"] = argv[-1]
        return subprocess.CompletedProcess(argv, 0, stdout=_smi(6), stderr="")

    monkeypatch.setattr(remote_mod.subprocess, "run", fake_run)
    assert probe_gpu_count(_host()) == (6, "")
    assert "nvidia-smi -L" in seen["cmd"]


@pytest.mark.parametrize("outcome, needle", [
    ("rc", "failed rc=9"), ("timeout", "timed out"), ("empty", "listed no GPUs")])
def test_probe_reports_failures_instead_of_raising(monkeypatch, outcome, needle):
    def fake_run(argv, **kw):
        if outcome == "timeout":
            raise subprocess.TimeoutExpired(argv, 1)
        rc = 9 if outcome == "rc" else 0
        return subprocess.CompletedProcess(argv, rc, stdout="", stderr="no driver")

    monkeypatch.setattr(remote_mod.subprocess, "run", fake_run)
    n, err = probe_gpu_count(_host())
    assert n is None and needle in err


@pytest.mark.parametrize("sku, got, short", [
    ("8xH200", 6, True), ("8xH200", 8, False), ("4xH100", 3, True),
    ("4xH100", 4, False), ("8xH200", None, False)])
def test_short_layout_is_rejected_even_with_the_floor_off(tmp_path, monkeypatch, sku, got, short):
    runner = _runner(tmp_path)                                  # no bench floor
    monkeypatch.setattr(remote_mod, "probe_gpu_count",
                        lambda host, **kw: (got, "" if got else "timed out"))
    monkeypatch.setattr(remote_mod, "probe_host_bench",
                        lambda host, **kw: pytest.fail("bench probed with the floor off"))
    why = runner._host_bench_below_floor(_host(), sku, "king pod k-0")
    assert why.startswith("short pod: 6 GPU(s) visible, rented as 8xH200") if (sku, got) == ("8xH200", 6) \
        else (bool(why) == short)


def test_single_gpu_sku_is_never_probed(tmp_path, monkeypatch):
    runner = _runner(tmp_path)
    monkeypatch.setattr(remote_mod, "probe_gpu_count",
                        lambda host, **kw: pytest.fail("probed a 1-GPU sku"))
    assert runner._host_bench_below_floor(_host(), "H100", "x") == ""
