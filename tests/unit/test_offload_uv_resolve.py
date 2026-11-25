"""#903: the offloaded bench sweep finds uv wherever the host has it."""
from __future__ import annotations

import os
import shlex
import stat
import subprocess
from types import SimpleNamespace

from cascade.validator.eval_offload import DEFAULT_UV_BIN, UV_RESOLVE, build_bench_remote_command


def _host():
    return SimpleNamespace(workdir="/root/cascade", cuda_device=None)


def test_default_uv_is_resolved_on_the_host():
    cmd = build_bench_remote_command(_host(), "/w/c", "/w/r", suites="gift-eval")
    assert UV_RESOLVE in cmd and f"{DEFAULT_UV_BIN} run" not in cmd.replace(UV_RESOLVE, "")


def test_explicit_uv_bin_is_used_as_given():
    cmd = build_bench_remote_command(_host(), "/w/c", "/w/r", suites="gift-eval", uv_bin="/bin/uv")
    assert shlex.split(cmd)[0] == "/bin/uv" and UV_RESOLVE not in cmd


def test_resolver_prefers_path_then_falls_back(tmp_path):
    fake = tmp_path / "bin" / "uv"
    fake.parent.mkdir()
    fake.write_text("#!/bin/sh\n")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    script = f"echo {UV_RESOLVE}"
    env = {"HOME": str(tmp_path), "PATH": f"{fake.parent}:/usr/bin:/bin"}
    assert subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True).stdout.strip() == str(fake)
    env["PATH"] = "/usr/bin:/bin"
    got = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True).stdout.strip()
    want = ("/usr/bin/uv" if os.access("/usr/bin/uv", os.X_OK) else
            "/bin/uv" if os.access("/bin/uv", os.X_OK) else os.path.join(str(tmp_path), ".local/bin/uv"))
    assert got == want
