"""#903 (W3b6a): the offloaded bench sweep finds uv wherever the sweep host has
it. The default used to be the hard-coded ``~/.local/bin/uv``; a host whose
image puts uv elsewhere (the Lium template: ``/usr/local/bin/uv``) failed
every offloaded lane A sweep as ``infra_failed``. The default is now resolved
ON the host (PATH first, ``~/.local/bin/uv`` as the fallback); an explicit
``uv_bin`` is used as given. The command still runs the sidecar's checked-in
lock (``--frozen``), so the report is unchanged."""
from __future__ import annotations

import shlex
import subprocess
from types import SimpleNamespace

from cascade.validator import eval_offload as EO


def _host():
    return SimpleNamespace(workdir="/root/cascade", cuda_device=None)


def _uv_token(cmd: str) -> str:
    """The shell word that names uv in the remote command."""
    head, sep, _ = cmd.partition(" run --project ")
    assert sep, cmd
    return head


def _on_host(word: str, *, home, path) -> str:
    """What ``word`` expands to in the host's bash (HOME / PATH as given)."""
    out = subprocess.run(["/bin/bash", "-c", f"echo {word}"], env={"HOME": str(home),
                         "PATH": str(path)}, capture_output=True, text=True, check=True)
    return out.stdout.strip()


def _fake_uv(tmp_path):
    fake = tmp_path / "usr-local-bin" / "uv"
    fake.parent.mkdir()
    fake.write_text("#!/bin/sh\n", encoding="utf-8")
    fake.chmod(0o755)
    return fake


def test_w3b6a_default_uv_resolves_from_path_then_falls_back(tmp_path):
    fake = _fake_uv(tmp_path)
    empty = tmp_path / "empty-bin"
    empty.mkdir()
    word = _uv_token(EO.build_bench_remote_command(_host(), "/w/c", "/w/r", suites="gift-eval"))
    assert _on_host(word, home=tmp_path, path=fake.parent) == str(fake), (
        "#903: the default uv is not resolved from the host's PATH")
    assert _on_host(word, home=tmp_path, path=empty) == str(tmp_path / ".local/bin/uv")


def test_w3b6a_explicit_uv_bin_is_used_as_given(tmp_path):
    fake = _fake_uv(tmp_path)
    cmd = EO.build_bench_remote_command(_host(), "/w/c", "/w/r", suites="gift-eval",
                                        uv_bin="/opt/uv/bin/uv")
    assert shlex.split(cmd)[0] == "/opt/uv/bin/uv"
    assert _on_host(_uv_token(cmd), home=tmp_path, path=fake.parent) == "/opt/uv/bin/uv"
    default = EO.build_bench_remote_command(_host(), "/w/c", "/w/r", suites="gift-eval")
    assert _on_host(_uv_token(default), home=tmp_path, path=fake.parent) == str(fake), (
        "#903: the default uv is not resolved from the host's PATH")


def test_w3b6a_resolved_uv_still_runs_the_frozen_lock_with_a_timeout(tmp_path):
    fake = _fake_uv(tmp_path)
    cmd = EO.build_bench_remote_command(
        SimpleNamespace(workdir="/root/cascade", cuda_device="1"), "/w/c", "/w/r",
        suites="gift-eval,time", timeout_s=600)
    assert cmd.startswith("CUDA_VISIBLE_DEVICES=1 timeout --signal=TERM --kill-after=15s 600s ")
    word = _uv_token(cmd).split("600s ", 1)[1]
    assert _on_host(word, home=tmp_path, path=fake.parent) == str(fake), (
        "#903: the default uv is not resolved from the host's PATH")
    tail = cmd.partition(" run --project ")[2]
    assert tail.startswith("/root/cascade/benchmarks --frozen --extra time cascade-benchmark ")
