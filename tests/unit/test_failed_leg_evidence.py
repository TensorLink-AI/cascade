"""A failed leg's evidence survives the pod and names the real cause.

Failure class (2026-11-01 03:18, 5CMfsDbJ on an operator 8xH100 at step
~16500/23841; same at 00:03 for five legs): "leg failed [operator]" logged only
the stderr TAIL — bash's job notice ``[1]+ Exit N  CASCADE_PROGRESS_FILE=...
< /dev/null`` — and the pod was removed seconds before the failure was
logged, so the cause was gone. Now the job notice is stripped, the last
Traceback and error-signal lines lead the failure reason with the exit code
and last progress step, and the fetched tails are saved on the box before the
failure reaches any teardown.
"""
from __future__ import annotations

import json
import os
import types

import pytest

import cascade.trainer.remote as remote_mod
from cascade.trainer.remote import (
    DETACHED_FETCH_PROGRESS_MARK,
    DETACHED_LAUNCH_TOKEN,
    DETACHED_STDERR_MARK,
    DETACHED_STDERR_TAIL_BYTES,
    FAILED_LEG_LOG_MAX_BYTES,
    FAILED_LEG_LOGS_KEEP,
    RemoteDispatcher,
    RemoteDispatchError,
    RemoteHost,
    _detached_fetch_command,
    error_tail,
    last_traceback,
    run_detached,
    save_failed_leg_evidence,
    split_job_notices,
    summarize_failure,
)

HK = "5CMfsDbJxyzAbcdefghij"
JOB = ("[1]+  Exit 1                  CASCADE_PROGRESS_FILE=/root/cascade/_train_work/_dispatch/"
       "challenger-5CMfsDbJxyzA-9-ab/progress.json python -m cascade.trainer.worker --x < /dev/null")
STDERR = "\n".join([
    "INFO step 16450 loss 1.23",
    "Traceback (most recent call last):",
    '  File "/root/cascade/cascade/trainer/moe_trainer.py", line 700, in train',
    "    loss.backward()",
    "RuntimeError: NCCL watchdog timeout: collective ALLREDUCE ran 600s",
    "ERROR worker: training aborted",
    "INFO error_class=infra is just a field name",
    "INFO flushing logs",
    JOB,
])


def _host(**kw):
    kw.setdefault("name", "op-8xh100")
    kw.setdefault("host", "10.0.0.7")
    kw.setdefault("port", 2201)
    kw.setdefault("workdir", "/root/cascade")
    return RemoteHost(**kw)


def _proc(rc=0, stdout="", stderr=""):
    return types.SimpleNamespace(returncode=rc, stdout=stdout, stderr=stderr)


class _Scripted:
    def __init__(self, steps):
        self.steps = list(steps)
        self.calls: list[str] = []

    def __call__(self, argv, timeout, stdin=None):
        self.calls.append(argv[-1])
        step = self.steps.pop(0) if self.steps else _proc(stdout="RUNNING\n")
        if isinstance(step, Exception):
            raise step
        return step


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(remote_mod, "_SLEEP", lambda s: None)


# ── the reason ───────────────────────────────────────────────────────────────

def test_job_notices_are_stripped_but_their_status_kept():
    lines, statuses = split_job_notices(STDERR + "\n[2]-  Killed   python -m x")
    assert all("CASCADE_PROGRESS_FILE" not in ln for ln in lines)
    assert statuses == ["Exit 1", "Killed"]


def test_last_traceback_runs_through_its_exception_line():
    lines, _ = split_job_notices(STDERR)
    tb = last_traceback(lines)
    assert tb[0] == "Traceback (most recent call last):"
    assert tb[-1].startswith("RuntimeError: NCCL watchdog timeout")
    assert last_traceback(["no", "traceback"]) == []


def test_summary_leads_with_exit_code_step_and_the_real_cause():
    cause, detail = summarize_failure(STDERR, rc=1, progress={"step": 16500, "tokens": 9})
    assert cause.startswith("exit_code=1 job=Exit 1 last_step=16500: RuntimeError: NCCL watchdog")
    assert "CASCADE_PROGRESS_FILE" not in cause + detail          # the job notice is gone
    assert "--- last traceback ---" in detail and "loss.backward()" in detail
    signals = detail.split("--- error signals ---", 1)[1].split("--- stderr tail ---", 1)[0]
    assert "ERROR worker: training aborted" in signals
    assert "error_class=infra" not in signals                      # lowercase field ≠ a level


def test_summary_without_a_traceback_uses_the_last_signal_or_line():
    cause, _ = summarize_failure("loading\nKilled\n" + JOB, rc=137)
    assert cause == "exit_code=137 job=Exit 1: Killed"
    cause, _ = summarize_failure("only a log line\n" + JOB, rc=1)
    assert cause.endswith(": only a log line")
    cause, detail = summarize_failure("", rc=1)
    assert cause == "exit_code=1: (no stderr)" and detail == ""


def test_the_cause_survives_error_tail_truncation():
    cause, detail = summarize_failure(STDERR + "\n" + "x" * 5000, rc=1, progress={"step": 3})
    msg = f"remote challenger on op failed (rc=1) {cause}\n{detail}"
    assert "NCCL watchdog timeout" in error_tail(msg, 500)         # the queue's last_error


# ── the evidence file ────────────────────────────────────────────────────────

def test_evidence_is_written_capped_and_pruned(tmp_path):
    root = tmp_path / "_failed_legs"
    p = save_failed_leg_evidence(root, HK, header={"exit_code": 1}, stdout="out\n",
                                 stderr=STDERR, clock=lambda: 1_761_966_000.0)
    assert p is not None and os.path.basename(p).startswith(HK[:12] + "-2025")
    text = open(p).read()
    assert json.loads(text.splitlines()[0]) == {"exit_code": 1}
    assert "=== stderr (tail) ===" in text and "NCCL watchdog timeout" in text and JOB in text
    big = save_failed_leg_evidence(root, HK, header={}, stdout="o" * 10**6,
                                   stderr="e" * 10**6, clock=lambda: 1_761_966_001.0)
    assert os.path.getsize(big) <= FAILED_LEG_LOG_MAX_BYTES
    for i in range(FAILED_LEG_LOGS_KEEP + 5):
        save_failed_leg_evidence(root, f"hk{i}", header={}, stdout="", stderr="",
                                 clock=lambda i=i: 1_761_970_000.0 + i)
    assert len(list(root.glob("*.log"))) == FAILED_LEG_LOGS_KEEP


def test_evidence_never_masks_the_failure(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("not a dir")
    assert save_failed_leg_evidence(blocker / "sub", HK, header={}, stdout="", stderr="") is None
    assert save_failed_leg_evidence(None, HK, header={}, stdout="", stderr="") is None


# ── the detached fetch ───────────────────────────────────────────────────────

def test_fetch_takes_a_large_stderr_tail_and_the_progress_record():
    cmd = _detached_fetch_command(_host(), "/root/cascade/_train_work/_dispatch/leg")
    assert f"tail -c {DETACHED_STDERR_TAIL_BYTES} " in cmd and DETACHED_STDERR_TAIL_BYTES >= 131072
    assert DETACHED_FETCH_PROGRESS_MARK in cmd and "progress.json" in cmd


def test_run_detached_carries_the_last_progress_step():
    poll = _proc(stdout='PROGRESS:{"step":16450,"tokens":1,"t":1}\nRUNNING\n')
    # the fetch has no progress section (older pod): the last polled record
    runner = _Scripted([_proc(stdout=DETACHED_LAUNCH_TOKEN + "\n"), poll,
                        _proc(stdout="EXIT:1\n"),
                        _proc(stdout="out\n" + DETACHED_STDERR_MARK + "\n" + STDERR + "\n")])
    proc = run_detached(_host(), "true", None, "/rd", timeout=3600, poll_seconds=1, runner=runner)
    assert proc.returncode == 1 and proc.progress["step"] == 16450
    assert DETACHED_FETCH_PROGRESS_MARK not in proc.stderr
    # the fetched record (the worker's final one) wins over the polled one
    runner = _Scripted([_proc(stdout=DETACHED_LAUNCH_TOKEN + "\n"), poll,
                        _proc(stdout="EXIT:1\n"),
                        _proc(stdout=("out\n" + DETACHED_STDERR_MARK + "\n" + STDERR + "\n"
                                      + DETACHED_FETCH_PROGRESS_MARK + "\n"
                                      + '{"step":16500,"tokens":2,"t":2}'))])
    proc = run_detached(_host(), "true", None, "/rd", timeout=3600, poll_seconds=1, runner=runner)
    assert proc.progress["step"] == 16500 and proc.stderr.rstrip().endswith(JOB)


def test_dispatch_failure_saves_evidence_and_reports_the_cause(tmp_path):
    runner = _Scripted([
        _proc(stdout=""),                                          # no prior run on the pod
        _proc(stdout=DETACHED_LAUNCH_TOKEN + "\n"),
        _proc(stdout='PROGRESS:{"step":16400,"tokens":1,"t":1}\nRUNNING\n'),
        _proc(stdout="EXIT:1\n"),
        _proc(stdout=("OUT\n" + DETACHED_STDERR_MARK + "\n" + STDERR + "\n"
                      + DETACHED_FETCH_PROGRESS_MARK + "\n" + '{"step":16500,"tokens":2,"t":2}')),
    ])
    root = tmp_path / "_failed_legs"
    disp = RemoteDispatcher(trainer_spec="m:C", detached=True, poll_seconds=1, _runner=runner,
                            evidence_dir=str(root))
    with pytest.raises(RemoteDispatchError) as ei:
        disp.dispatch(_host(), gen_ref="g", uid=7, hotkey=HK, role="challenger",
                      base_seed=9, block=1)
    msg = str(ei.value)
    first = msg.splitlines()[0]
    assert ei.value.returncode == 1
    assert "exit_code=1" in first and "last_step=16500" in first
    assert "RuntimeError: NCCL watchdog timeout" in first and "[evidence: " in first
    assert "CASCADE_PROGRESS_FILE" not in msg
    # the evidence is on the box when the failure surfaces — before any teardown
    (saved,) = list(root.glob(f"{HK[:12]}-*.log"))
    text = saved.read_text()
    header = json.loads(text.splitlines()[0])
    assert header["exit_code"] == 1 and header["progress"]["step"] == 16500
    assert header["address"] == "10.0.0.7:2201" and "OUT" in text and JOB in text


def test_dispatch_rejection_keeps_its_reason_and_also_saves_evidence(tmp_path):
    runner = _Scripted([
        _proc(stdout=""),
        _proc(stdout=DETACHED_LAUNCH_TOKEN + "\n"),
        _proc(stdout="EXIT:3\n"),
        _proc(stdout=DETACHED_STDERR_MARK + "\nCorpusError: generator_stalled: no series\n"),
    ])
    root = tmp_path / "_failed_legs"
    disp = RemoteDispatcher(trainer_spec="m:C", detached=True, poll_seconds=1, _runner=runner,
                            evidence_dir=str(root))
    with pytest.raises(RemoteDispatchError) as ei:
        disp.dispatch(_host(), gen_ref="g", uid=7, hotkey=HK, role="challenger",
                      base_seed=9, block=1)
    assert ei.value.returncode == 3 and "miner submission rejected" in str(ei.value)
    assert "generator_stalled" in str(ei.value)
    assert len(list(root.glob("*.log"))) == 1


def test_dispatcher_without_an_evidence_dir_still_reports_the_cause():
    runner = _Scripted([
        _proc(stdout=""),
        _proc(stdout=DETACHED_LAUNCH_TOKEN + "\n"),
        _proc(stdout="EXIT:1\n"),
        _proc(stdout=DETACHED_STDERR_MARK + "\n" + STDERR + "\n"),
    ])
    disp = RemoteDispatcher(trainer_spec="m:C", detached=True, poll_seconds=1, _runner=runner)
    with pytest.raises(RemoteDispatchError) as ei:
        disp.dispatch(_host(), gen_ref="g", uid=7, hotkey=HK, role="challenger",
                      base_seed=9, block=1)
    assert "NCCL watchdog timeout" in str(ei.value).splitlines()[0]
    assert "[evidence" not in str(ei.value)
