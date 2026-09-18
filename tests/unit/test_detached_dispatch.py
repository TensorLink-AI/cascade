"""Detached dispatch (2026-09-18): a remote leg's fate no longer rides one ssh session.

The attached form (``build_remote_command`` + one long ``run_ssh``) turned every transport
drop into a failed leg while the worker kept training on the pod: 5Co2Te 09-13 after
1h39m, 5DoJQ 09-17 after 1h12m, and at 02:52 on 09-18 all seven Oslo lanes at once (8 legs
lost and re-queued). Pod-side benches died the same way (u201 on three pods, exit 255).

Detached: the launch ssh starts the worker under ``setsid nohup`` in its own session with
stdout/stderr/pid/exit_code in a per-leg run dir and returns; the orchestrator polls with
short ssh calls, tolerates an unreachable pod for a grace period, then fetches the receipt
and hands the caller the same (rc, stdout, stderr) triple the attached form produced.
"""
from __future__ import annotations

import shlex
import subprocess
import types
from pathlib import Path

import pytest

import cascade.trainer.remote as remote_mod
from cascade.shared.manifest import format_trained_pointer
from cascade.trainer.bench_hook import BenchPlan, run_post_round_benchmark
from cascade.trainer.remote import (
    DETACHED_LAUNCH_TOKEN,
    DETACHED_STDERR_MARK,
    RECEIPT_SENTINEL,
    RemoteDispatcher,
    RemoteDispatchError,
    RemoteHost,
    build_detached_command,
    build_remote_command,
    run_detached,
)


def _host(**kw):
    kw.setdefault("name", "pod-a")
    kw.setdefault("host", "10.0.0.1")
    kw.setdefault("workdir", "/root/cascade")
    return RemoteHost(**kw)


def _proc(rc=0, stdout="", stderr=""):
    return types.SimpleNamespace(returncode=rc, stdout=stdout, stderr=stderr)


def _remote_cmd(argv):
    return argv[-1]


class _Scripted:
    """A runner double: ``steps`` is what each successive ssh call returns — a proc, or
    an Exception instance to raise (a transport failure). Records every call."""

    def __init__(self, steps):
        self.steps = list(steps)
        self.calls: list[tuple[str, str | None]] = []

    def __call__(self, argv, timeout, stdin=None):
        self.calls.append((_remote_cmd(argv), stdin))
        step = self.steps.pop(0) if self.steps else _proc(rc=0, stdout="RUNNING\n")
        if isinstance(step, Exception):
            raise step
        return step


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(remote_mod, "_SLEEP", lambda s: None)


# ── the launcher ─────────────────────────────────────────────────────────────

def test_launcher_detaches_the_worker_and_keeps_credentials_off_the_command_line():
    cmd = build_detached_command(_host(), "bash -c 'python -m x'", "/root/cascade/_train_work/_dispatch/leg-1",
                                 stdin_env_present=True)
    assert cmd.startswith("cd /root/cascade && set -a && . /dev/stdin && set +a && ")
    assert "setsid nohup bash -c " in cmd
    assert ">/root/cascade/_train_work/_dispatch/leg-1/stdout" in cmd
    assert "2>/root/cascade/_train_work/_dispatch/leg-1/stderr" in cmd
    assert cmd.rstrip().endswith(f"echo {DETACHED_LAUNCH_TOKEN}")
    inner = shlex.split(cmd.split("setsid nohup ", 1)[1])[2]
    assert inner.startswith("echo $$ > /root/cascade/_train_work/_dispatch/leg-1/pid; ")
    assert inner.endswith("; rc=$?; echo $rc > /root/cascade/_train_work/_dispatch/leg-1/exit_code")
    assert "HF_TOKEN" not in cmd                       # never inline


def test_launcher_without_credentials_skips_the_stdin_source():
    cmd = build_detached_command(_host(), "true", "/root/cascade/_train_work/_dispatch/leg-2",
                                 stdin_env_present=False)
    assert "/dev/stdin" not in cmd


def test_attached_command_is_unchanged_by_the_refactor():
    cmd, stdin_env = build_remote_command(_host(cuda_device="1"), ["python", "-m", "x"],
                                          {"HF_TOKEN": "t"}, lane_count=2)
    inner = shlex.split(cmd.split("bash -c ", 1)[1])[0]
    assert "python -m x </dev/null & w=$!; wait $w; rc=$?; kill -KILL -- -$w 2>/dev/null; exit $rc" in inner
    assert stdin_env == "HF_TOKEN=t\n"


# ── run_detached ─────────────────────────────────────────────────────────────

def _receipt_stdout(role="challenger"):
    import json
    entry = {"role": role, "miner_uid": 7, "miner_hotkey": "hk", "size": "toto2-4m",
             "trained_pointer": format_trained_pointer("cascade/ckpt-r1-king@sha256:" + "a" * 64),
             "corpus_digest": "d", "gen_ref": "g", "train_block": 1, "bench_scores": None,
             "duel_rank": None, "gpu_name": "", "warm_started": False}
    return RECEIPT_SENTINEL + json.dumps(entry) + "\n"


def test_run_detached_launches_polls_and_fetches():
    runner = _Scripted([
        _proc(stdout=DETACHED_LAUNCH_TOKEN + "\n"),          # launch
        _proc(stdout="RUNNING\n"),                            # poll 1
        _proc(stdout="RUNNING\n"),                            # poll 2
        _proc(stdout="EXIT:0\n"),                             # poll 3
        _proc(stdout="OUT-LINE\n" + DETACHED_STDERR_MARK + "\nERR-TAIL\n"),  # fetch
    ])
    proc = run_detached(_host(), "true", "K=v\n", "/root/cascade/_train_work/_dispatch/leg",
                        timeout=3600, poll_seconds=1, grace_seconds=60, runner=runner)
    assert proc.returncode == 0
    assert proc.stdout.strip() == "OUT-LINE"
    assert proc.stderr.strip() == "ERR-TAIL"
    launch_cmd, launch_stdin = runner.calls[0]
    assert DETACHED_LAUNCH_TOKEN in launch_cmd and launch_stdin == "K=v\n"
    assert all(stdin == "" for _, stdin in runner.calls[1:])   # creds only at launch
    assert "exit_code" in runner.calls[1][0] and "kill -0" in runner.calls[1][0]


def test_run_detached_survives_transport_drops_inside_the_grace_period():
    runner = _Scripted([
        _proc(stdout=DETACHED_LAUNCH_TOKEN + "\n"),
        _proc(stdout="RUNNING\n"),
        subprocess.TimeoutExpired(cmd="ssh", timeout=60),     # pod unreachable
        _proc(rc=255, stderr="ssh: connect to host 10.0.0.1 port 22: Connection reset"),
        _proc(rc=255, stderr="Connection closed by remote host"),
        _proc(stdout="EXIT:0\n"),                             # back, and done
        _proc(stdout="ok\n" + DETACHED_STDERR_MARK + "\n"),
    ])
    proc = run_detached(_host(), "true", None, "/root/cascade/_train_work/_dispatch/leg",
                        timeout=3600, poll_seconds=1, grace_seconds=900, runner=runner)
    assert proc.returncode == 0 and proc.stdout.strip() == "ok"
    assert len(runner.calls) == 7


def test_run_detached_declares_the_leg_lost_only_past_the_grace_period(monkeypatch):
    clock = {"t": 0.0}

    def fake_time():
        clock["t"] += 100.0                                   # each poll = 100 s apart
        return clock["t"]

    monkeypatch.setattr(remote_mod, "time", types.SimpleNamespace(time=fake_time))
    runner = _Scripted([_proc(stdout=DETACHED_LAUNCH_TOKEN + "\n")]
                       + [_proc(rc=255, stderr="down")] * 20)
    with pytest.raises(RemoteDispatchError) as ei:
        run_detached(_host(), "true", None, "/root/cascade/_train_work/_dispatch/leg",
                     timeout=36000, poll_seconds=1, grace_seconds=250, runner=runner)
    assert ei.value.returncode == 255
    assert "unreachable" in str(ei.value)
    # launch + 3 polls (100/200 s inside the grace, the 300 s one past it)
    assert len(runner.calls) == 4


def test_run_detached_exit_codes_and_vanished_worker():
    runner = _Scripted([
        _proc(stdout=DETACHED_LAUNCH_TOKEN + "\n"),
        _proc(stdout="EXIT:3\n"),
        _proc(stdout=DETACHED_STDERR_MARK + "\nCorpusError: generator_stalled\n"),
    ])
    proc = run_detached(_host(), "true", None, "/rd", timeout=60, poll_seconds=1, runner=runner)
    assert proc.returncode == 3 and "generator_stalled" in proc.stderr

    runner = _Scripted([
        _proc(stdout=DETACHED_LAUNCH_TOKEN + "\n"),
        _proc(stdout="GONE\n"),
        _proc(stdout=DETACHED_STDERR_MARK + "\nlast log line\n"),
    ])
    proc = run_detached(_host(), "true", None, "/rd", timeout=60, poll_seconds=1, runner=runner)
    assert proc.returncode == 1 and "vanished" in proc.stderr and "last log line" in proc.stderr


def test_run_detached_launch_failure_and_fetch_retry():
    runner = _Scripted([_proc(rc=255, stderr="ssh: no route")])
    with pytest.raises(RemoteDispatchError) as ei:
        run_detached(_host(), "true", None, "/rd", timeout=60, poll_seconds=1, runner=runner)
    assert "launch failed" in str(ei.value) and ei.value.returncode == 255

    runner = _Scripted([
        _proc(stdout=DETACHED_LAUNCH_TOKEN + "\n"),
        _proc(stdout="EXIT:0\n"),
        _proc(rc=255, stderr="blip"),                          # fetch 1 fails
        subprocess.TimeoutExpired(cmd="ssh", timeout=180),     # fetch 2 fails
        _proc(stdout="late\n" + DETACHED_STDERR_MARK + "\n"),  # fetch 3
    ])
    proc = run_detached(_host(), "true", None, "/rd", timeout=60, poll_seconds=1, runner=runner)
    assert proc.returncode == 0 and proc.stdout.strip() == "late"


def test_run_detached_timeout_kills_the_group(monkeypatch):
    clock = {"t": 0.0}

    def fake_time():
        clock["t"] += 1000.0
        return clock["t"]

    monkeypatch.setattr(remote_mod, "time", types.SimpleNamespace(time=fake_time))
    runner = _Scripted([_proc(stdout=DETACHED_LAUNCH_TOKEN + "\n"), _proc(stdout="RUNNING\n"),
                        _proc(stdout="")])
    with pytest.raises(RemoteDispatchError) as ei:
        run_detached(_host(), "true", None, "/rd", timeout=1500, poll_seconds=1, runner=runner)
    assert "timed out" in str(ei.value)
    assert any("kill -KILL" in cmd for cmd, _ in runner.calls)


# ── RemoteDispatcher in detached mode ────────────────────────────────────────

def test_dispatcher_detached_returns_the_entry_from_the_fetched_receipt():
    runner = _Scripted([
        _proc(stdout=DETACHED_LAUNCH_TOKEN + "\n"),
        _proc(stdout="RUNNING\n"),
        _proc(stdout="EXIT:0\n"),
        _proc(stdout=_receipt_stdout() + DETACHED_STDERR_MARK + "\nlogs\n"),
    ])
    disp = RemoteDispatcher(trainer_spec="m:C", detached=True, poll_seconds=1,
                            reattach_grace_seconds=60, _runner=runner)
    entry = disp.dispatch(_host(cuda_device="0"), gen_ref="g", uid=7, hotkey="hk" * 24,
                          role="challenger", base_seed=1, block=1)
    assert entry.role == "challenger" and entry.miner_uid == 7
    launch_cmd, _ = runner.calls[0]
    assert "_train_work/_dispatch/challenger-hkhkhkhkhkhk-1-" in launch_cmd
    assert "set -m; CUDA_VISIBLE_DEVICES=0 " in shlex.split(launch_cmd.split("setsid nohup ", 1)[1])[2]


def test_dispatcher_detached_maps_rc3_to_a_rejection():
    runner = _Scripted([
        _proc(stdout=DETACHED_LAUNCH_TOKEN + "\n"),
        _proc(stdout="EXIT:3\n"),
        _proc(stdout=DETACHED_STDERR_MARK + "\nCorpusError: generator_stalled: no series for 1800s\n"),
    ])
    disp = RemoteDispatcher(trainer_spec="m:C", detached=True, poll_seconds=1, _runner=runner)
    with pytest.raises(RemoteDispatchError) as ei:
        disp.dispatch(_host(), gen_ref="g", uid=7, hotkey="hk", role="challenger", base_seed=1, block=1)
    assert ei.value.returncode == 3 and "miner submission rejected" in str(ei.value)


def test_dispatcher_default_is_still_attached():
    out = _receipt_stdout()
    disp = RemoteDispatcher(trainer_spec="m:C", _runner=lambda argv, t, s=None: _proc(stdout=out))
    entry = disp.dispatch(_host(), gen_ref="g", uid=7, hotkey="hk", role="challenger", base_seed=1, block=1)
    assert entry.miner_uid == 7 and disp.detached is False


# ── the pod-side bench ───────────────────────────────────────────────────────

def test_bench_detached_polls_then_reads_the_report(tmp_path: Path):
    import json
    report = {"suites": [{"suite": "gift-eval", "status": "ok"}], "scores": {}}
    calls = []

    steps = [
        _proc(stdout=DETACHED_LAUNCH_TOKEN + "\n"),           # launch (no sideload ssh: local_data_dir unset)
        _proc(stdout="RUNNING\n"),
        _proc(rc=255, stderr="dropped"),                      # blip
        _proc(stdout="EXIT:0\n"),
        _proc(stdout="bench log\n" + DETACHED_STDERR_MARK + "\n"),   # fetch
        _proc(stdout=json.dumps(report)),                     # cat report
    ]

    def runner(argv, timeout):
        calls.append(_remote_cmd(argv))
        step = steps.pop(0)
        return step

    plan = BenchPlan(detached=True, poll_seconds=1, reattach_grace_seconds=60, timeout_seconds=600,
                     local_data_dir=None)
    got = run_post_round_benchmark(_host(), "r1", "toto2-4m", plan, work_root=tmp_path,
                                   runner=runner, role="king")
    assert got == report
    assert any("cascade-benchmark" in c and "setsid nohup" in c for c in calls)
    assert (tmp_path / "r1" / "toto2-4m" / "king-benchmark_report.json").is_file()


def test_bench_attached_path_unchanged(tmp_path: Path):
    import json
    report = {"suites": [], "scores": {}}
    steps = [_proc(stdout=""), _proc(stdout=json.dumps(report))]
    got = run_post_round_benchmark(_host(), "r1", "toto2-4m", BenchPlan(local_data_dir=None),
                                   work_root=tmp_path, runner=lambda a, t: steps.pop(0), role="king")
    assert got == report


# ── config ───────────────────────────────────────────────────────────────────

def test_chain_toml_arms_detached_dispatch_and_the_loader_parses_it(tmp_path: Path):
    from cascade.shared.config import load_chain_config

    repo = Path(__file__).resolve().parents[2] / "chain.toml"
    r = load_chain_config(repo).round
    assert r.detached_dispatch is True
    assert r.dispatch_poll_seconds == 30 and r.dispatch_reattach_grace_seconds == 900
    src = repo.read_text(encoding="utf-8")
    edited = src.replace("detached_dispatch = true", "detached_dispatch = false") \
                .replace("dispatch_poll_seconds = 30", "dispatch_poll_seconds = 2") \
                .replace("dispatch_reattach_grace_seconds = 900", "dispatch_reattach_grace_seconds = 0")
    assert edited != src
    (tmp_path / "chain.toml").write_text(edited, encoding="utf-8")
    r2 = load_chain_config(tmp_path / "chain.toml").round
    assert r2.detached_dispatch is False
    assert r2.dispatch_poll_seconds == 5           # floor
    assert r2.dispatch_reattach_grace_seconds == 0
