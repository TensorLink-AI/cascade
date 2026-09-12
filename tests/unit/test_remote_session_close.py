"""The remote worker command closes the SSH session when the worker exits (2026-09-12).

A lingering child of the worker (the generator sandbox) holding the session's
stdout/stderr keeps sshd waiting for EOF, so the orchestrator never sees the worker's
exit: the king's worker exited rc=3 at 11:08 and the dispatch returned at 14:51. The
worker now runs in its own process group and the wrapper reaps the group as soon as the
worker exits, preserving the worker's exit code.
"""
from __future__ import annotations

import shlex
import subprocess
import sys
import time

import pytest

from cascade.trainer.remote import PREEMPT_BENCHMARKS, RemoteHost, build_remote_command


def _host(**kw):
    kw.setdefault("name", "pod-a")
    kw.setdefault("host", "10.0.0.1")
    kw.setdefault("workdir", "/root/cascade")
    return RemoteHost(**kw)


def test_command_wraps_the_worker_in_a_reaped_process_group():
    cmd, stdin_env = build_remote_command(_host(cuda_device="1"), ["python", "-m", "x"],
                                          {"HF_TOKEN": "t"}, lane_count=2)
    assert cmd.startswith(PREEMPT_BENCHMARKS)
    assert "cd /root/cascade && set -a && . /dev/stdin && set +a && bash -c " in cmd
    inner = shlex.split(cmd.split("bash -c ", 1)[1])[0]
    assert inner.startswith("set -m; ")
    assert "CUDA_VISIBLE_DEVICES=1 CASCADE_LANE_COUNT=2" in inner or \
        "CASCADE_LANE_COUNT=2" in inner and "CUDA_VISIBLE_DEVICES=1" in inner
    assert "python -m x </dev/null & w=$!; wait $w; rc=$?; kill -KILL -- -$w 2>/dev/null; exit $rc" in inner
    assert stdin_env == "HF_TOKEN=t\n"                     # credentials still on stdin only


@pytest.mark.skipif(sys.platform != "linux", reason="needs a POSIX shell with job control")
def test_wrapper_returns_when_the_worker_exits_even_with_a_lingering_child(tmp_path):
    # A "worker" that spawns a child holding stdout open for a long time, then
    # exits 3. Without the group reap, reading the pipe blocks until the child
    # exits (20 s); with it the wrapper returns at once with rc=3.
    worker = tmp_path / "worker.sh"
    worker.write_text("#!/bin/bash\nsleep 20 &\necho __RECEIPT__\nexit 3\n", encoding="utf-8")
    worker.chmod(0o755)
    cmd, _ = build_remote_command(_host(workdir=str(tmp_path)), ["bash", str(worker)], {})
    cmd = cmd.replace(PREEMPT_BENCHMARKS, "")          # no benchmarks to preempt here
    t0 = time.monotonic()
    proc = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=15,
                          input="")
    assert time.monotonic() - t0 < 10.0                  # did not wait for the 20 s child
    assert proc.returncode == 3                          # worker's rc preserved
    assert "__RECEIPT__" in proc.stdout
