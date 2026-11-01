"""Host admission guards (2026-10-31 / 11-01 incidents).

A. the 93.120.231.186 8xH100 trained at ~0.2 steps/s on two legs; the
   throughput watchdog judged only at its 20-min grace and the host was
   re-rented in between → an EARLY judge (optimistic rate) and a host
   quarantine on every watchdog abort;
B. ``[round] funded_host_blocklist`` — permanent, honoured by Lium rent
   selection and the operator-lane join gate, merged with the quarantine file;
C. hosts that cannot do multi-GPU (NCCL "Cuda failure 999" before step 1, a
   hung 8-GPU all_reduce, comm-bound 8xH200 VMs) → an N-rank NCCL all_reduce
   preflight before a leg is admitted, a bus-bandwidth floor, and a
   first-step deadline;
D. one operator lane failed ~30 different legs in seconds each (rc=2) →
   the "LANE BREAKER".

No GPUs, no ssh: scripted runners and fakes throughout.
"""
from __future__ import annotations

import contextlib
import logging
import re
import subprocess
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from cascade.provision import core as core_mod
from cascade.provision.core import (
    HOST_BLOCKLIST_REASON,
    LiumProvider,
    quarantined_hosts,
    record_host_quarantine,
    set_host_blocklist,
)
from cascade.shared.config import RoundConfig, load_chain_config
from cascade.trainer import remote as remote_mod
from cascade.trainer.cost_guards import (
    FIRST_STEP_ABORT,
    THROUGHPUT_ABORT,
    ProgressWatchdog,
    leg_watchdog,
)
from cascade.trainer.faults import _OperatorFault
from cascade.trainer.host_preflight import (
    NCCL_PREFLIGHT,
    host_blocked_reason,
    multi_gpu_preflight_failure,
    preflight_gpu_count,
)
from cascade.trainer.loop import TrainerRunner, _FinalLanePool
from cascade.trainer.remote import (
    LegAborted,
    NcclPreflight,
    RemoteDispatchError,
    RemoteHost,
    nccl_preflight_command,
    nccl_preflight_script,
    probe_nccl_allreduce,
)

ROOT = Path(__file__).resolve().parents[2]
BAD = "93.120.231.186"


class _Errors(logging.Handler):
    """ERROR records of the ``cascade.trainer`` logger, captured on the logger
    itself (caplog's root handler misses them when an earlier test in the
    session disabled propagation)."""

    def __init__(self):
        super().__init__(level=logging.ERROR)
        self.lines: list[str] = []

    def emit(self, record):
        self.lines.append(record.getMessage())


@pytest.fixture
def errors():
    lg = logging.getLogger("cascade.trainer")
    h = _Errors()
    lg.addHandler(h)
    old = lg.level
    if lg.getEffectiveLevel() > logging.ERROR:
        lg.setLevel(logging.ERROR)
    yield h.lines
    lg.removeHandler(h)
    lg.setLevel(old)


def _with_round(text: str, extra: str) -> str:
    return re.sub(r"(?m)^\[round\]$", "[round]\n" + extra, text, count=1)


def _host(**kw) -> RemoteHost:
    base = dict(name="lane-x", host=BAD, port=32308, user="root", key_path="/k",
                remote_python="/root/cascade/.venv/bin/python", workdir="/root/cascade")
    base.update(kw)
    return RemoteHost(**base)


# ── config: every new knob is parsed by the loader (round-trip) ──────────────

def test_new_knobs_default_and_round_trip_through_the_loader(tmp_path):
    base = load_chain_config(ROOT / "chain.toml").round
    assert base.progress_watchdog_early_seconds == 600            # safe on-value
    assert base.progress_first_step_seconds == 0                  # off (image-dependent)
    assert base.funded_host_blocklist == ()
    assert base.funded_host_nccl_timeout_seconds == 90            # preflight on
    assert base.funded_host_nccl_busbw_floor == ()                # floor off
    assert base.operator_lane_breaker_failures == 3
    text = _with_round((ROOT / "chain.toml").read_text(), "\n".join([
        "progress_watchdog_early_seconds = 300",
        "progress_first_step_seconds = 2700",
        f'funded_host_blocklist = ["{BAD}", " 10.0.0.7 ", "{BAD}"]',
        "funded_host_nccl_timeout_seconds = 60",
        'funded_host_nccl_busbw_floor = { "8xH200" = 300, "8xH100" = 250.5 }',
        "operator_lane_breaker_failures = 5"]))
    p = tmp_path / "chain.toml"
    p.write_text(text)
    rnd = load_chain_config(p).round
    assert rnd.progress_watchdog_early_seconds == 300
    assert rnd.progress_first_step_seconds == 2700
    assert rnd.funded_host_blocklist == (BAD, "10.0.0.7")         # stripped, de-duplicated
    assert rnd.funded_host_nccl_timeout_seconds == 60
    assert rnd.funded_host_nccl_busbw_floor == (("8xH100", 250.5), ("8xH200", 300.0))
    assert rnd.operator_lane_breaker_failures == 5


@pytest.mark.parametrize("line", [
    "progress_watchdog_early_seconds = -1",
    'progress_first_step_seconds = "x"',
    f'funded_host_blocklist = "{BAD}"',                          # a bare string, not a list
    "funded_host_blocklist = [1]",
    "funded_host_nccl_timeout_seconds = true",
    "funded_host_nccl_busbw_floor = 300",
    'funded_host_nccl_busbw_floor = { "8xH200" = -1 }',
    "operator_lane_breaker_failures = 1.5",
])
def test_bad_new_knob_values_fail_the_load(tmp_path, line):
    p = tmp_path / "chain.toml"
    p.write_text(_with_round((ROOT / "chain.toml").read_text(), line))
    with pytest.raises(ValueError):
        load_chain_config(p)


# ── A. the early throughput judge + host quarantine ──────────────────────────

class _Clock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t


def _watch(clock, **kw):
    kw.setdefault("label", "funded leg hkA")
    kw.setdefault("hotkey", "hkA")
    kw.setdefault("pod_name", "cascade-n91-1-funded-hka-0")
    kw.setdefault("training_steps", 23841)
    kw.setdefault("max_train_seconds", 18000)                     # 5 h, the incident contract
    kw.setdefault("end_wall", None)
    kw.setdefault("grace_seconds", 1200)
    kw.setdefault("early_seconds", 600)
    return ProgressWatchdog(clock=clock, **kw)


def test_early_judge_aborts_the_incident_rate_at_ten_minutes(errors):
    clock = _Clock()
    w = _watch(clock, host=BAD)
    assert w({"step": 1}) is None
    clock.t = 300
    assert w({"step": 51}) is None                                # before the early judge
    clock.t = 600
    why = w({"step": 101})                                    # 0.2 steps/s, observed
    assert why and why.startswith(THROUGHPUT_ABORT)
    assert "early judge" in why and f"host={BAD}" in why
    assert "projected training wall" in why
    assert w.kind == "throughput"
    assert len(errors) == 1 and THROUGHPUT_ABORT in errors[0]


def test_early_judge_never_aborts_an_on_budget_leg_whose_record_lags():
    # 0.9 steps/s: 23841 / 0.9 = 7.36 h < 1.5 x 5 h — on budget, barely. The
    # record is written every 50 steps, so it can lag the true step by 49:
    # the optimistic rate absorbs that.
    clock = _Clock()
    w = _watch(clock)
    w({"step": 50})
    clock.t = 600
    assert w({"step": 50 + int(0.9 * 600) - 49}) is None
    clock.t = 1200                                                # the full judge agrees
    assert w({"step": 50 + int(0.9 * 1200)}) is None


def test_early_judge_off_waits_for_the_grace():
    clock = _Clock()
    w = _watch(clock, early_seconds=0)
    w({"step": 1})
    clock.t = 600
    assert w({"step": 121}) is None
    clock.t = 1200
    assert (w({"step": 241}) or "").startswith(THROUGHPUT_ABORT)


def test_grace_zero_keeps_the_throughput_judge_off_even_with_an_early_value():
    clock = _Clock()
    w = _watch(clock, grace_seconds=0, early_seconds=600)
    w({"step": 1})
    for t in (600, 1200, 5000):
        clock.t = t
        assert w({"step": 2}) is None


def test_leg_watchdog_reads_the_new_knobs(tmp_path):
    from tests.unit.test_funded_pod_wiring import _challenger, _runner

    runner = _runner(tmp_path, progress_watchdog_early_seconds=420,
                     progress_first_step_seconds=2700)
    runner.__dict__["_leg_local_obj"] = SimpleNamespace(end_wall=None)
    contract = SimpleNamespace(training_steps=100, max_train_seconds=10)
    w = leg_watchdog(runner, gen=_challenger("hkA"), pod=SimpleNamespace(instance_id="p-0"),
                     contract=contract, host=BAD)
    assert (w.early, w.first_step, w.host) == (420.0, 2700.0, BAD)
    # first-step deadline alone keeps a watchdog alive with the throughput judge off
    off = _runner(tmp_path, progress_watchdog_grace_seconds=0, payer_key_check_seconds=0,
                  progress_first_step_seconds=900)
    off.__dict__["_leg_local_obj"] = SimpleNamespace(end_wall=None)
    assert leg_watchdog(off, gen=_challenger("hkA"), pod=SimpleNamespace(instance_id="p-0"),
                        contract=contract) is not None


def test_a_watchdog_abort_quarantines_the_host_for_a_day():
    from cascade.trainer.runner_funded_legs import _dispatch_watched

    class _AbortDisp:
        def dispatch(self, host, **kw):
            # "step 401/…" must not read as an HTTP 401 (account fault ⇒ no quarantine)
            raise LegAborted(f"remote challenger: {THROUGHPUT_ABORT}: hotkey=hkA host={BAD} "
                             "rate=0.205 steps/s (step 401/23841 after 1212s observed)")

    with pytest.raises(_OperatorFault, match=THROUGHPUT_ABORT):
        _dispatch_watched(_AbortDisp(), _host())
    q = quarantined_hosts()
    assert BAD in q and THROUGHPUT_ABORT in q[BAD]
    import json
    import os
    rec = json.loads(Path(os.environ[core_mod.LIUM_QUARANTINE_FILE_ENV]).read_text())[BAD]
    assert rec["until"] - __import__("time").time() > 23 * 3600


def test_operator_rented_legs_carry_the_watchdog_too(tmp_path, monkeypatch):
    """An operator-billed leg pod (the 93.120.231.186 op-0 pods) is watched
    like a payer pod: _run_operator_rented_leg dispatches through
    _dispatch_watched."""
    import inspect

    from cascade.trainer.runner_operator_pods import _OperatorPodMixin

    src = inspect.getsource(_OperatorPodMixin._run_operator_rented_leg)
    assert "_with_leg_watchdog" in src and "_dispatch_watched(" in src


# ── C. first-step deadline ───────────────────────────────────────────────────

def test_first_step_deadline_aborts_a_leg_that_never_trains(errors):
    clock = _Clock(100.0)
    w = _watch(clock, first_step_seconds=1800, host=BAD)
    assert w(None) is None                                        # worker still starting
    clock.t = 100 + 1799
    assert w({"tokens": 0}) is None
    clock.t = 100 + 1800
    why = w(None)
    assert why and why.startswith(FIRST_STEP_ABORT) and f"host={BAD}" in why
    assert w.kind == "first_step"
    assert len(errors) == 1 and FIRST_STEP_ABORT in errors[0]


def test_first_step_deadline_is_off_by_default_and_stops_once_training_starts():
    clock = _Clock()
    off = _watch(clock, first_step_seconds=0)
    clock.t = 99999
    assert off(None) is None
    clock.t = 0
    w = _watch(clock, first_step_seconds=600, grace_seconds=0)
    w({"step": 1})                                                # trained before the deadline
    clock.t = 5000
    assert w(None) is None                                        # a missed poll, not a hang


# ── B. permanent host blocklist ──────────────────────────────────────────────

def test_blocklist_merges_into_the_quarantine_view_and_is_replaceable():
    record_host_quarantine("10.0.0.1", "never ready", seconds=600)
    set_host_blocklist([BAD, " ", BAD])
    assert quarantined_hosts() == {"10.0.0.1": "never ready", BAD: HOST_BLOCKLIST_REASON}
    set_host_blocklist(())                                        # removed from the knob
    assert quarantined_hosts() == {"10.0.0.1": "never ready"}


def test_lium_listing_skips_a_blocklisted_host(monkeypatch, tmp_path):
    monkeypatch.setenv(core_mod.LIUM_COOLDOWN_FILE_ENV, str(tmp_path / "cooldown.json"))
    def _run(argv):
        out = ('[{"id": "ex-a", "gpu_count": 8}, {"id": "ex-b", "gpu_count": 8}]'
               if "ls" in argv else "[]")
        return types.SimpleNamespace(returncode=0, stdout=out, stderr="")

    prov = LiumProvider(bin="lium", _run=_run)
    monkeypatch.setattr(prov, "_executor_hosts", lambda: {"ex-a": BAD, "ex-b": "10.0.0.2"})
    assert [e["id"] for e in prov._list_executors("8xH100")] == ["ex-a", "ex-b"]
    from cascade.trainer.host_preflight import sync_host_blocklist

    sync_host_blocklist(SimpleNamespace(cfg=SimpleNamespace(
        round=RoundConfig(funded_host_blocklist=(BAD,)))))
    assert [e["id"] for e in prov._list_executors("8xH100")] == ["ex-b"]


def _gate(monkeypatch, rnd, *, ssh_ok=True):
    def run_ssh(argv, timeout=None, stdin_text=None):
        if not ssh_ok:
            pytest.fail("ssh probed a blocked lane")
        return SimpleNamespace(returncode=0, stdout="LANE_OK\n", stderr="")

    monkeypatch.setattr("cascade.trainer.remote.run_ssh", run_ssh)
    monkeypatch.setattr("cascade.trainer.remote.probe_worker_runtime", lambda h, **k: "")
    fake = SimpleNamespace(cfg=SimpleNamespace(round=rnd),
                           _host_bench_below_floor=lambda host, sku, label: "")
    return TrainerRunner._operator_lane_gate.__get__(fake)


def test_lane_gate_refuses_a_blocklisted_or_quarantined_host_without_ssh(monkeypatch):
    rnd = SimpleNamespace(funded_pod_sku="RTX4090", funded_host_blocklist=(BAD,))
    why = _gate(monkeypatch, rnd, ssh_ok=False)(_host())
    assert why and "blocked" in why and HOST_BLOCKLIST_REASON in why
    rnd2 = SimpleNamespace(funded_pod_sku="RTX4090", funded_host_blocklist=())
    record_host_quarantine("10.0.0.5", "NCCL PREFLIGHT failed", seconds=600, host_fault=True)
    assert "NCCL PREFLIGHT" in _gate(monkeypatch, rnd2, ssh_ok=False)(_host(host="10.0.0.5"))
    assert _gate(monkeypatch, rnd2)(_host(host="10.0.0.6")) is None


def test_rolling_pool_leaves_a_blocklisted_round_start_lane_out(tmp_path):
    hosts_toml = tmp_path / "hosts.toml"

    def _lane(name, host):
        return (f'[[host]]\nname = "{name}"\nhost = "{host}"\nport = 22\nuser = "root"\n'
                f'key_path = "~/.ssh/k"\nremote_python = "python"\nworkdir = "/root/cascade"\n'
                f'cuda_device = "0"\nstage = "final"\n')
    hosts_toml.write_text(_lane("lane-bad", BAD) + _lane("lane-ok", "10.0.0.2"))
    fake = SimpleNamespace(remote_hosts_path=hosts_toml, remote_hosts=(),
                           cfg=SimpleNamespace(round=RoundConfig(funded_host_blocklist=(BAD,),
                                                                 operator_lane_breaker_failures=4)),
                           _operator_lane_gate=lambda h: None)
    fake._rolling_final_hosts = TrainerRunner._rolling_final_hosts.__get__(fake)
    pool = TrainerRunner._rolling_lane_pool.__get__(fake)()
    assert [h.name for h in pool.known_hosts()] == ["lane-ok"]
    assert pool.breaker_threshold == 4


# ── C. the NCCL preflight probe ──────────────────────────────────────────────

def test_preflight_script_compiles_and_command_runs_n_ranks_under_a_hard_timeout():
    compile(nccl_preflight_script(), "<nccl>", "exec")            # valid python, no torch import
    assert "dist.all_reduce" in nccl_preflight_script()
    cmd = nccl_preflight_command(_host(cuda_device="0,1,2,3,4,5,6,7"), 8, 90)
    assert "--nproc_per_node=8" in cmd and "timeout -k 10 90" in cmd
    assert "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7" in cmd
    assert "-m torch.distributed.run --standalone" in cmd and 'rm -f "$f"' in cmd


def _ran(rc=0, stdout="", stderr=""):
    return SimpleNamespace(returncode=rc, stdout=stdout, stderr=stderr)


_GOOD = ('NCCLCHECK {"world": 8, "bytes": 268435456, "seconds": 0.0012, '
         '"algbw_gbps": 223.7, "busbw_gbps": 391.4, "ok": true}\n')


def test_probe_reports_bandwidth_and_ships_the_script_on_stdin():
    seen = {}

    def run(argv, timeout, stdin):
        seen.update(argv=argv, timeout=timeout, stdin=stdin)
        return _ran(stdout="W1101 torchrun banner\n" + _GOOD)

    res = probe_nccl_allreduce(_host(), nprocs=8, timeout=60, runner=run)
    assert res == NcclPreflight(ok=True, world=8, busbw_gbps=391.4)
    assert "dist.all_reduce" in seen["stdin"] and seen["timeout"] == 120


@pytest.mark.parametrize("outcome, needle, inconclusive", [
    (_ran(124, stderr="..."), "hung", False),                     # the pod-side timeout fired
    (_ran(137), "hung", False),
    ("timeout", "hung", False),                                   # the local one fired
    (_ran(1, stderr="NCCL WARN Cuda failure 999 'unknown error'"), "Cuda failure 999", False),
    (_ran(0, stdout=_GOOD.replace('"ok": true', '"ok": false')), "wrong sum", False),
    (_ran(0, stdout=_GOOD.replace('"world": 8', '"world": 6')), "expected 8", False),
    (_ran(0, stdout="nothing\n"), "no result", False),
    (_ran(255, stderr="Connection refused"), "ssh rc=255", True),
])
def test_probe_failures_are_verdicts_and_only_transport_is_inconclusive(outcome, needle,
                                                                        inconclusive):
    def run(argv, timeout, stdin):
        if outcome == "timeout":
            raise subprocess.TimeoutExpired(argv, timeout)
        return outcome

    res = probe_nccl_allreduce(_host(), nprocs=8, timeout=60, runner=run)
    assert not res.ok and needle in res.error and res.inconclusive is inconclusive


# ── C. the preflight verdict ─────────────────────────────────────────────────

def _rr(**kw):
    return SimpleNamespace(cfg=SimpleNamespace(round=RoundConfig(**kw)))


def test_preflight_skips_single_gpu_and_off():
    probe = lambda host, **kw: pytest.fail("probed")             # noqa: E731
    assert preflight_gpu_count(_host(cuda_device="3"), "8xH100") == 1
    assert preflight_gpu_count(_host(), "8xH200") == 8
    assert preflight_gpu_count(_host(), "RTX4090") == 1
    assert multi_gpu_preflight_failure(_rr(), _host(), "RTX4090", "x", probe=probe) == ""
    assert multi_gpu_preflight_failure(_rr(funded_host_nccl_timeout_seconds=0), _host(),
                                       "8xH200", "x", probe=probe) == ""


def test_a_hung_collective_rejects_and_quarantines_the_host(errors):
    seen = {}

    def probe(host, *, nprocs, timeout):
        seen.update(nprocs=nprocs, timeout=timeout)
        return NcclPreflight(ok=False, world=8, error="NCCL all_reduce hung: killed")

    why = multi_gpu_preflight_failure(_rr(), _host(), "8xH200", "funded pod p-0",
                                      probe=probe)
    assert why.startswith(NCCL_PREFLIGHT) and "hung" in why
    assert any(NCCL_PREFLIGHT in e and "quarantined" in e for e in errors)
    assert seen == {"nprocs": 8, "timeout": 90.0}
    assert BAD in quarantined_hosts()


def test_bandwidth_floor_rejects_a_comm_bound_host_only_when_armed():
    probe = lambda host, **kw: NcclPreflight(ok=True, world=8, busbw_gbps=40.0)  # noqa: E731
    assert multi_gpu_preflight_failure(_rr(), _host(), "8xH200", "x", probe=probe) == ""
    assert quarantined_hosts() == {}
    armed = _rr(funded_host_nccl_busbw_floor=(("8xH200", 300.0),))
    why = multi_gpu_preflight_failure(armed, _host(), "8xh200", "x", probe=probe)
    assert "comm-bound" in why and "40.0 GB/s < floor 300" in why
    assert BAD in quarantined_hosts()
    fast = lambda host, **kw: NcclPreflight(ok=True, world=8, busbw_gbps=390.0)  # noqa: E731
    assert multi_gpu_preflight_failure(armed, _host(host="10.0.0.8"), "8xH200", "x",
                                       probe=fast) == ""


def test_an_unreachable_probe_passes_with_a_warning():
    probe = lambda host, **kw: NcclPreflight(ok=False, error="ssh rc=255",  # noqa: E731
                                             inconclusive=True)
    assert multi_gpu_preflight_failure(_rr(), _host(), "8xH200", "x", probe=probe) == ""
    assert quarantined_hosts() == {}


def test_lane_gate_refuses_a_lane_that_fails_the_preflight(monkeypatch):
    monkeypatch.setattr(remote_mod, "probe_nccl_allreduce",
                        lambda host, **kw: NcclPreflight(ok=False, world=4,
                                                         error="NCCL all_reduce failed rc=1"))
    gate = _gate(monkeypatch, RoundConfig(funded_pod_sku="RTX4090"))
    why = gate(_host(cuda_device="0,1,2,3"))
    assert why and why.startswith(NCCL_PREFLIGHT)
    assert BAD in quarantined_hosts()
    assert host_blocked_reason(SimpleNamespace(cfg=SimpleNamespace(round=RoundConfig())), BAD)


def test_a_funded_pod_failing_the_preflight_is_released_and_the_leg_rents_again(
        tmp_path, monkeypatch):
    from cascade.provision import funded as funded_mod
    from cascade.trainer import runner_funded_pods as rfp
    from tests.unit.test_funded_pod_wiring import _challenger, _rent_ok, _runner, _vault

    runner = _runner(tmp_path)
    _vault(tmp_path, "hkA")
    rents, torn = [], []
    monkeypatch.setattr(funded_mod, "rent_funded_pod", lambda **kw: rents.append(1) or _rent_ok())
    monkeypatch.setattr(funded_mod, "teardown_funded",
                        lambda pods, vault, **kw: (torn.extend(p.instance_id for p in pods), [])[1])
    runner._funded_pod_code_mismatch = lambda result, profile: ""
    runner._funded_pod_too_slow = lambda result, profile: ""
    verdicts = [f"{NCCL_PREFLIGHT} failed on 8 GPU(s): NCCL all_reduce hung", ""]
    monkeypatch.setattr(rfp, "funded_pod_comm_failure",
                        lambda r, result, profile: verdicts.pop(0))
    host, pod = runner._rent_funded_host("777", _challenger("hkA"))
    assert len(rents) == 2 and torn == [pod.instance_id]
    assert "hkA" not in runner._funded_leg_failures                # never the miner's


def test_funded_pod_comm_failure_sees_every_gpu_of_the_layout(tmp_path, monkeypatch):
    from dataclasses import replace

    from cascade.trainer.runner_funded_pods import funded_pod_comm_failure
    from tests.unit.test_funded_pod_wiring import _rent_ok, _runner

    seen = {}

    def probe(host, *, nprocs, timeout):
        seen.update(dev=host.cuda_device, n=nprocs, key=host.pinned_host_key)
        return NcclPreflight(ok=True, world=nprocs, busbw_gbps=380.0)

    monkeypatch.setattr(remote_mod, "probe_nccl_allreduce", probe)
    result = _rent_ok()
    result = replace(result, pod=replace(result.pod, sku="8xH200"))
    profile = SimpleNamespace(user="root", key_path="/k", remote_python="/p",
                              workdir="/root/cascade", ssh_options=())
    assert funded_pod_comm_failure(_runner(tmp_path), result, profile) == ""
    assert seen == {"dev": "0,1,2,3,4,5,6,7", "n": 8, "key": result.host_key}


# ── D. the lane circuit breaker ──────────────────────────────────────────────

class _Lane:
    def __init__(self, name, host="104.218.237.44", port=22):
        self.name, self.host, self.port, self.stage = name, host, port, "final"


def _rc2(name):
    return RemoteDispatchError(f"remote challenger on {name} failed (rc=2): "
                               "error: argument --contract-block: invalid", returncode=2)


class _FailingDisp:
    def __init__(self, exc_for):
        self.exc_for = exc_for
        self.calls = []

    def dispatch(self, host, **kw):
        self.calls.append((host.name, kw.get("hotkey")))
        exc = self.exc_for(host, kw)
        if exc is not None:
            raise exc
        return SimpleNamespace(host=host.name)


def test_breaker_trips_after_k_different_legs_and_logs_one_error(errors):
    lane = _Lane("targon-lane-2")
    pool = _FinalLanePool([lane], lambda: [], breaker_threshold=3)
    disp = _FailingDisp(lambda h, kw: _rc2(h.name))
    raised = []
    for hk in ("hkA", "hkA", "hkB", "hkC"):                    # hkA twice: one leg
        with pytest.raises(RemoteDispatchError) as ei:
            TrainerRunner._dispatch_on_free_lane(disp, pool, [lane], describe="x",
                                                 hotkey=hk, role="challenger")
        raised.append(getattr(ei.value, "lane_breaker", False))
    assert raised == [False, False, False, True]
    assert pool.known_hosts() == [] and pool.empty()               # out of rotation
    assert len(errors) == 1 and "LANE BREAKER" in errors[0] and "targon-lane-2" in errors[0]
    # the last leg did not block on the now-empty pool: one attempt, no retry
    assert disp.calls[-1] == ("targon-lane-2", "hkC") and disp.calls.count(
        ("targon-lane-2", "hkC")) == 1
    # held for hours, not the 10-min re-gate
    assert pool._rejected["targon-lane-2"] - __import__("time").time() > 3600


def test_a_success_resets_the_count_and_miner_faults_never_count(monkeypatch):
    from cascade.trainer import loop as loop_mod

    monkeypatch.setattr(loop_mod, "STORAGE_RETRY_BACKOFF_SECONDS", 0)
    lane = _Lane("lane-a")
    pool = _FinalLanePool([lane], lambda: [], breaker_threshold=2)
    script = {"hkA": _rc2("lane-a"),
              "hkB": None,                                         # a good leg
              "hkC": RemoteDispatchError("rc=3 generator rejected", returncode=3),
              "hkD": RemoteDispatchError("hippius StorageError 503", returncode=1),
              "hkE": _rc2("lane-a")}
    disp = _FailingDisp(lambda h, kw: script[kw["hotkey"]])
    for hk in ("hkA", "hkB", "hkC", "hkD", "hkE"):
        with contextlib.suppress(RemoteDispatchError):
            TrainerRunner._dispatch_on_free_lane(disp, pool, [lane], describe="x",
                                                 hotkey=hk, role="challenger")
    assert [h.name for h in pool.known_hosts()] == ["lane-a"]      # never tripped
    assert pool._lane_failures == {"lane-a": ["hkE|challenger|"]}


def test_breaker_off_never_drains():
    lane = _Lane("lane-a")
    pool = _FinalLanePool([lane], lambda: [])                      # threshold 0
    disp = _FailingDisp(lambda h, kw: _rc2(h.name))
    for hk in ("a", "b", "c", "d", "e"):
        with pytest.raises(RemoteDispatchError):
            TrainerRunner._dispatch_on_free_lane(disp, pool, [lane], describe="x", hotkey=hk)
    assert [h.name for h in pool.known_hosts()] == ["lane-a"]


def test_a_tripped_lane_hands_the_retry_to_a_healthy_one():
    bad, good = _Lane("bad", host="10.0.0.1"), _Lane("good", host="10.0.0.2")
    pool = _FinalLanePool([bad, good], lambda: [], breaker_threshold=1)
    disp = _FailingDisp(lambda h, kw: _rc2(h.name) if h.name == "bad" else None)
    out = TrainerRunner._dispatch_on_free_lane(disp, pool, [bad, good], describe="x",
                                               hotkey="hkA", role="challenger")
    assert out.host == "good"
    assert [h.name for h in pool.known_hosts()] == ["good"]


def test_breaker_victim_requeues_without_the_operator_backoff(tmp_path):
    from cascade.trainer.runner_rolling import record_lane_breaker_victim
    from tests.unit.test_funded_pod_wiring import _runner

    runner = _runner(tmp_path)
    plain = RuntimeError("x")
    assert record_lane_breaker_victim(runner, "hkA", plain) is False
    victim = RuntimeError("remote challenger failed (rc=2)")
    victim.lane_breaker = True
    assert record_lane_breaker_victim(runner, "hkA", victim) is True
    msg, miner_fault, cls, burn = runner._funded_leg_failures["hkA"]
    assert (miner_fault, cls, burn) == (False, "no_capacity", False) and "LANE BREAKER" in msg


def test_lane_b_converts_a_breaker_victim_to_no_capacity():
    from cascade.trainer.lane_b_driver import lane_breaker_failure

    assert lane_breaker_failure(RuntimeError("x")) == {}
    victim = RuntimeError("remote expert failed (rc=2)")
    victim.lane_breaker = True
    f = lane_breaker_failure(victim)["failure"]
    assert (f.miner_fault, f.error_class, f.burn) == (False, "no_capacity", False)
    assert "LANE BREAKER" in f.msg
