"""2026-11-27: the eval-host request is bounded ON the host and serialised.

``subprocess.run(timeout=)`` kills only the local ``ssh``: the remote
``lane_d_forecasts public|forecast`` kept running and every retry stacked
another one on the GPU (three on the 4090). The remote command must be
``timeout -k 30 <T-30> …`` and the GPU requests must hold an exclusive
``flock`` on the cache root, so a retry never runs beside an orphan.

The behavioural tests run the built remote command in a local ``bash``
(``flock`` / ``timeout`` from util-linux / coreutils) with a fake
``remote_python``; no ssh, no torch, no GPU.
"""

from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import threading
import time
from types import SimpleNamespace

import pytest

from . import helpers as h


def _hosts(tmp_path, *, python: str = "python", workdir: str = ".") -> object:
    hosts = tmp_path / "hosts.toml"
    hosts.write_text('[[host]]\nname = "laned-eval"\nhost = "10.0.0.9"\nstage = "final"\n'
                     f'remote_python = "{python}"\nworkdir = "{workdir}"\n', encoding="utf-8")
    return hosts


def _public_reply() -> str:
    from cascade.eval.lane_d_forecasts import PUBLIC_REPLY_SCHEMA

    return json.dumps({"schema": PUBLIC_REPLY_SCHEMA, "ok": True, "scores": {}, "dropped": {}})


def _captured(tmp_path, *, timeout_s: float = 3600.0, cache_dir: str = "/c") -> dict:
    """{request: remote command} for hello, forecast and public."""
    from cascade.eval.lane_d_forecasts import REPLY_SCHEMA

    m = h.inputs_mod()
    cmds: dict[str, str] = {}

    def run(argv, *, input, capture_output, timeout, check):
        words = argv[-1].split()
        kind = next(k for k in ("hello", "forecast", "public") if k in words)
        cmds[kind] = argv[-1]
        if kind == "hello":
            out = json.dumps(h.hello())
        elif kind == "public":
            out = _public_reply()
        else:
            out = json.dumps({"schema": REPLY_SCHEMA, "ok": True, "quantiles": []})
        return SimpleNamespace(returncode=0, stdout=out.encode(), stderr=b"")

    host = m.SSHEvalHost(_hosts(tmp_path), cache_dir=cache_dir, budget_bytes=10,
                         contexts=h.CONTEXTS, run=run, timeout_s=timeout_s)
    host.hello()
    host.public_scores("d" * 64, context=128, horizon=8, draws=[], block_hash="0x00",
                       max_channels=4)
    host.forecast("d" * 64, context=128, horizon=8, windows=[])
    return cmds


def _before_python(cmd: str) -> list[str]:
    words = shlex.split(cmd)
    return words[:words.index("python")]


def test_every_remote_request_is_bounded_on_the_host(tmp_path):
    cmds = _captured(tmp_path, timeout_s=3600)
    for kind, cmd in cmds.items():
        pre = _before_python(cmd)
        assert "timeout" in pre, f"{kind}: the remote process is not bounded on the host: {cmd}"
        i = pre.index("timeout")
        assert pre[i + 1:i + 4] == ["-k", "30", "3570"], \
            f"{kind}: want timeout -k 30 <T-30> (T = 3600), got {pre[i:i + 4]}"


def test_gpu_requests_hold_the_cache_root_lock_and_hello_does_not(tmp_path):
    cmds = _captured(tmp_path, cache_dir="/root/lane_d_cache")
    for kind in ("forecast", "public"):
        pre = _before_python(cmds[kind])
        assert "flock" in pre, f"{kind}: no lock: a retry would run beside an orphan"
        assert pre[pre.index("flock") + 1] == "/root/lane_d_cache/.lane_d_eval.lock"
        assert pre.index("timeout") < pre.index("flock"), \
            "the lock wait must sit inside the host-side bound"
    assert "flock" not in _before_python(cmds["hello"]), "hello needs no GPU lock"


_FAKE = """#!/bin/bash
# the fake eval host: $3 is the request (python -m <module> <request> ...)
if [ "$3" = "hello" ]; then echo '{hello}'; exit 0; fi
echo "start $$" >> {log}
sleep {sleep}
echo "end $$" >> {log}
echo '{reply}'
"""


def _local_host(tmp_path, *, sleep: float, timeout_s: float):
    if not (shutil.which("flock") and shutil.which("timeout") and shutil.which("bash")):
        pytest.skip("flock / timeout / bash not on this box")
    m = h.inputs_mod()
    log = tmp_path / "runs.log"
    fake = tmp_path / "fake_python"
    fake.write_text(_FAKE.replace("{hello}", json.dumps(h.hello()))
                    .replace("{reply}", _public_reply()).replace("{log}", str(log))
                    .replace("{sleep}", str(sleep)), encoding="utf-8")
    fake.chmod(0o755)

    def run(argv, *, input, capture_output, timeout, check):
        # the remote command, run here instead of over ssh
        return subprocess.run(["bash", "-c", argv[-1]], input=input,
                              capture_output=capture_output, timeout=timeout, check=check)

    host = m.SSHEvalHost(_hosts(tmp_path, python=str(fake), workdir=str(tmp_path)),
                         cache_dir=str(tmp_path / "cache"), budget_bytes=10,
                         contexts=h.CONTEXTS, run=run, timeout_s=timeout_s)
    host.hello()
    return host, log


def _public(host):
    return host.public_scores("d" * 64, context=128, horizon=8, draws=[], block_hash="0x00",
                              max_channels=4)


def test_concurrent_requests_never_overlap_on_the_host(tmp_path):
    host, log = _local_host(tmp_path, sleep=0.6, timeout_s=120)
    errors: list[BaseException] = []

    def call():
        try:
            _public(host)
        except BaseException as e:  # noqa: BLE001 — reported below
            errors.append(e)

    threads = [threading.Thread(target=call) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    assert not errors, errors
    events = [ln.split()[0] for ln in log.read_text().splitlines()]
    assert events == ["start", "end"] * 3, \
        f"requests ran side by side on the eval host (stacked retries): {events}"


def test_a_request_past_its_bound_is_killed_on_the_host(tmp_path):
    # T = 31 s ⇒ the host-side bound is 1 s; the fake would run 20 s.
    host, log = _local_host(tmp_path, sleep=20, timeout_s=31)
    t0 = time.monotonic()
    try:
        _public(host)
        failed = False
    except Exception:  # noqa: BLE001 — any failure; the bound is what is checked
        failed = True
    assert time.monotonic() - t0 < 15, "the remote process outlived its host-side bound"
    assert "end" not in log.read_text(), "the remote request ran to completion"
    assert failed, "a request killed on the host must fail, never answer"


def test_the_source_reads_the_house_alone_for_an_absent_record():
    src_cls = h.inputs_mod().LaneDInputsSource
    assert callable(getattr(src_cls, "house", None)), \
        "LaneDInputsSource.house() is missing: an abandoned boundary cannot be recorded absent"
    src = object.__new__(src_cls)
    src._house = lambda: {"digest": "h" * 64}
    assert src.house() == {"digest": "h" * 64}
