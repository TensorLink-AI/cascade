"""The validator's GIFT-Eval gate can be offloaded to a GPU pod.

Only the gift-eval compute crosses to the pod (scp the fetched checkpoint →
run ``cascade-benchmark --suites gift-eval`` → pull the report); the paired
bootstrap and every consensus decision stay on the orchestrator. Failures
return ``None`` (gate uncomputable), never raise.

The pod itself is elastic (provisioner-rented per round manifest), so the
host is re-resolved from the hosts file AT EACH offloaded eval — see the
``make_eval_host_fn`` section.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from cascade.eval.benchmarks import gift_rows_from_report
from cascade.trainer.remote import RemoteHost
from cascade.validator.eval_offload import (
    bench_scores_via_host,
    build_bench_remote_command,
    build_scp_argv,
    gift_rows_via_host,
    make_eval_host_fn,
)
from cascade.validator.loop import ValidatorRunner


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


# ── make_eval_host_fn: lazy per-eval host resolution (elastic pod) ───────────

def _hosts_toml(name="cascade-900-eval-0", ip="9.9.9.9", stage=""):
    stage_line = f'stage = "{stage}"\n' if stage else ""
    return (f'[[host]]\nname = "{name}"\nhost = "{ip}"\nport = 40123\n'
            f'key_path = "~/.ssh/k"\nworkdir = "/root/cascade"\n{stage_line}')


def test_eval_host_fn_reresolves_per_call_across_pod_lifecycle(tmp_path):
    """The elastic-pod lifecycle: no file → pod published → file cleared →
    a NEW pod published. Every transition is picked up without a restart."""
    path = tmp_path / "eval_hosts.toml"
    fn = make_eval_host_fn(path)
    assert fn() is None                                     # not rented yet ⇒ local
    path.write_text(_hosts_toml(), encoding="utf-8")
    host = fn()
    assert host is not None and host.host == "9.9.9.9"      # pod appeared ⇒ offload resumes
    path.write_text("# cascade-provisioner: no fleet\n", encoding="utf-8")
    assert fn() is None                                     # torn down ⇒ local fallback
    path.write_text(_hosts_toml(name="cascade-901-eval-0", ip="9.9.9.10"), encoding="utf-8")
    assert fn().host == "9.9.9.10"                          # next round's pod


def test_eval_host_fn_missing_file_after_delete_falls_back_local(tmp_path):
    path = tmp_path / "eval_hosts.toml"
    path.write_text(_hosts_toml(), encoding="utf-8")
    fn = make_eval_host_fn(path)
    assert fn() is not None
    path.unlink()                                           # file disappears entirely
    assert fn() is None


def test_eval_host_fn_unparseable_file_warns_and_falls_back_local(tmp_path, caplog):
    path = tmp_path / "eval_hosts.toml"
    path.write_text("this is [not valid toml", encoding="utf-8")
    fn = make_eval_host_fn(path)
    with caplog.at_level(logging.WARNING, logger="cascade.validator.eval_offload"):
        assert fn() is None                                 # degraded, never crashed
    assert any("unreadable" in r.message for r in caplog.records)


def test_eval_host_fn_ignores_heat_only_hosts(tmp_path):
    path = tmp_path / "eval_hosts.toml"
    path.write_text(_hosts_toml(stage="heat"), encoding="utf-8")
    assert make_eval_host_fn(path)() is None                # no final/any host ⇒ local


def test_eval_host_fn_logs_transitions_once_not_per_eval(tmp_path, caplog):
    path = tmp_path / "eval_hosts.toml"
    path.write_text(_hosts_toml(), encoding="utf-8")
    fn = make_eval_host_fn(path)
    with caplog.at_level(logging.INFO, logger="cascade.validator.eval_offload"):
        for _ in range(3):                                  # static file: one 'now' log
            fn()
        path.unlink()
        fn()
        fn()                                                # one 'gone' log
    infos = [r.message for r in caplog.records if r.levelno == logging.INFO]
    assert len([m for m in infos if "host now" in m]) == 1
    assert len([m for m in infos if "host gone" in m]) == 1


def test_runner_resolves_eval_host_freshly_per_eval(cfg):
    """The runner asks the injected resolver at every eval — a host that
    appears mid-run is used, one that disappears stops being used."""
    answers = [None, _host(), None]
    fn = lambda: answers.pop(0)  # noqa: E731
    runner = ValidatorRunner(cfg=cfg, eval_host_fn=fn, verify_signatures=False)
    assert runner._eval_host() is None                      # pod not rented yet
    assert runner._eval_host().host == "9.9.9.9"            # rented: offload
    assert runner._eval_host() is None                      # torn down: local again


def test_runner_without_resolver_stays_local(cfg):
    assert ValidatorRunner(cfg=cfg, verify_signatures=False)._eval_host() is None
