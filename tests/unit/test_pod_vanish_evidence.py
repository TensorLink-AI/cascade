"""Pod resource samples + vanish post-mortem (2026-11-13).

Three legs died with the pod dark 6-15 min and back with its run dir gone,
at steps 14400-16450. Every detached poll now reads the pod's resources; a
vanish runs a read-only post-mortem before teardown; the samples PROVE a
container restart or a host reboot. The failure TEXT the fault classifier
bills on is unchanged — evidence rides only into the saved log.
"""
from __future__ import annotations

import subprocess

import cascade.trainer.remote as R
from cascade.trainer.remote import (
    DETACHED_LAUNCH_TOKEN,
    DETACHED_RESOURCES_MARK,
    DETACHED_STDERR_MARK,
    RemoteHost,
    parse_resources,
    resource_verdict,
    run_detached,
)


def _proc(rc=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(args=[], returncode=rc, stdout=stdout, stderr=stderr)


class _Scripted:
    def __init__(self, steps):
        self.steps = list(steps)
        self.argvs = []

    def __call__(self, argv, timeout, stdin):
        self.argvs.append(argv)
        step = self.steps.pop(0) if self.steps else _proc(stdout="RUNNING\n")
        if isinstance(step, BaseException):
            raise step
        return step


def _host():
    return RemoteHost(name="p", host="h", workdir="/w")


def _res(mem=500_000_000, cg=60 * 2**30, up=1000.0, pid1=4242, disk=900_000_000):
    return f"{DETACHED_RESOURCES_MARK}{mem} {cg} max {up} {pid1} {disk}\n"


def test_poll_command_emits_a_parseable_resource_line(tmp_path):
    rd = tmp_path / "run"
    rd.mkdir()
    host = RemoteHost(name="x", host="h", workdir=str(tmp_path))
    out = subprocess.run(["bash", "-c", R._detached_poll_command(host, str(rd))],
                         capture_output=True, text=True).stdout.splitlines()
    assert out[-1] == "GONE"                                   # no pid file
    r = parse_resources(out)
    assert r is not None and set(r) == {"mem_avail_kb", "cg_mem_bytes", "cg_max", "uptime_s",
                                        "pid1_start", "disk_free_kb"}
    assert isinstance(r["uptime_s"], float) and r["uptime_s"] > 0   # Linux CI
    assert isinstance(r["disk_free_kb"], int)


def test_parse_resources_tolerates_missing_fields():
    assert parse_resources(["RUNNING"]) is None
    r = parse_resources([f"{DETACHED_RESOURCES_MARK}123  max  "])
    assert r["mem_avail_kb"] == 123 and r["cg_mem_bytes"] is None and r["cg_max"] == "max"
    assert r["uptime_s"] is None and r["pid1_start"] is None and r["disk_free_kb"] is None


def test_resource_verdict_proves_restart_and_reboot():
    a = parse_resources([_res(up=5000.0, pid1=4242)])
    assert resource_verdict([a]) == ""
    assert resource_verdict([a, parse_resources([_res(up=5600.0, pid1=4242)])]) == ""
    assert "CONTAINER RESTARTED" in resource_verdict(
        [a, parse_resources([_res(up=5600.0, pid1=99999)])])
    assert "HOST REBOOTED" in resource_verdict(
        [a, parse_resources([_res(up=300.0, pid1=12)])])


def test_vanish_runs_a_postmortem_and_keeps_the_classified_text_unchanged():
    runner = _Scripted([
        _proc(stdout=DETACHED_LAUNCH_TOKEN + "\n"),
        _proc(stdout=_res(up=8000.0, pid1=4242) + "RUNNING\n"),
        _proc(rc=255, stderr="ssh: timeout"),                    # dark
        _proc(stdout=_res(up=8400.0, pid1=777777) + "GONE\n"),   # back, worker gone
        _proc(stdout=DETACHED_STDERR_MARK + "\nlast log line\n"),  # log fetch
        _proc(stdout="run dir: MISSING\n[ 1.0] Out of memory: Killed process 9 (python)\n"),
    ])
    proc = run_detached(_host(), "true", None, "/rd", timeout=600, poll_seconds=1,
                        grace_seconds=600, runner=runner)
    assert proc.returncode == 1
    # the billed-on text: exactly the pre-2026-11-13 shape — no dmesg, no verdict
    assert proc.stderr.rstrip().endswith(
        "last log line\n[detached worker process vanished without an exit code]")
    assert "Killed process" not in proc.stderr and "RESTARTED" not in proc.stderr
    # the evidence carries the verdict, the postmortem and the samples
    assert "CONTAINER RESTARTED" in proc.postmortem
    assert "run dir: MISSING" in proc.postmortem and "Killed process" in proc.postmortem
    assert "last 2 poll(s)" in proc.postmortem
    # the postmortem ran AFTER the log fetch
    cmds = [" ".join(a) for a in runner.argvs]
    assert "run dir: present" in cmds[-1] and "tail -c" in cmds[-2]


def test_postmortem_failure_never_fails_differently():
    runner = _Scripted([
        _proc(stdout=DETACHED_LAUNCH_TOKEN + "\n"),
        _proc(stdout="GONE\n"),
        _proc(stdout=DETACHED_STDERR_MARK + "\nx\n"),
        OSError("ssh died"),
    ])
    proc = run_detached(_host(), "true", None, "/rd", timeout=60, poll_seconds=1, runner=runner)
    assert proc.returncode == 1 and "vanished" in proc.stderr
    assert "postmortem failed: ssh died" in proc.postmortem
    assert "not proven" in proc.postmortem


def test_clean_exit_has_no_postmortem_and_no_extra_ssh():
    runner = _Scripted([
        _proc(stdout=DETACHED_LAUNCH_TOKEN + "\n"),
        _proc(stdout=_res() + "RUNNING\n"),
        _proc(stdout=_res() + "EXIT:0\n"),
        _proc(stdout="out\n" + DETACHED_STDERR_MARK + "\n"),
    ])
    proc = run_detached(_host(), "true", None, "/rd", timeout=60, poll_seconds=1, runner=runner)
    assert proc.returncode == 0 and proc.postmortem == ""
    assert len(runner.argvs) == 4


def test_evidence_file_gets_the_postmortem(tmp_path):
    from cascade.trainer.remote import save_failed_leg_evidence

    p = save_failed_leg_evidence(tmp_path, "5Ctyc5w8gwif", header={"exit_code": 1},
                                 stdout="", stderr="tail\n=== verdict: X ===\nrun dir: MISSING")
    from pathlib import Path

    text = Path(p).read_text(encoding="utf-8")
    assert "=== verdict: X ===" in text and "run dir: MISSING" in text
