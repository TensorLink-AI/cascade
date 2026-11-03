"""T3: a stalled or trickling checkpoint harvest fails (RemoteDispatchError)
instead of hanging the leg forever."""

from __future__ import annotations

import io
import sys
import tarfile
import time

import pytest

from cascade.trainer import remote as remote_mod
from cascade.trainer.remote import RemoteDispatchError, RemoteHost


def _tar_bytes() -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        data = b"W" * 4096
        info = tarfile.TarInfo("./weights.safetensors")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _py_ssh(monkeypatch, script: str) -> None:
    monkeypatch.setattr(remote_mod, "build_ssh_argv",
                        lambda host, cmd: [sys.executable, "-c", script])


def test_a_stalled_stream_is_killed_and_raises(tmp_path, monkeypatch):
    tarball = tmp_path / "ok.tar"
    tarball.write_bytes(_tar_bytes())
    # Half the archive, then silence (a pod that stopped sending).
    _py_ssh(monkeypatch, (
        "import sys,time\n"
        f"d=open({str(tarball)!r},'rb').read()\n"
        "sys.stdout.buffer.write(d[:1024]); sys.stdout.buffer.flush()\n"
        "time.sleep(60)\n"))
    t0 = time.monotonic()
    with pytest.raises(RemoteDispatchError, match="stalled"):
        remote_mod.harvest_remote_dir(RemoteHost(name="p", host="h"), "/w/ck",
                                      tmp_path / "out", timeout=60, stall_timeout=0.5)
    assert time.monotonic() - t0 < 20


def test_a_trickling_stream_hits_the_overall_deadline(tmp_path, monkeypatch):
    tarball = tmp_path / "ok.tar"
    tarball.write_bytes(_tar_bytes())
    # One byte every 50 ms: never stalls, never finishes in time.
    _py_ssh(monkeypatch, (
        "import sys,time\n"
        f"d=open({str(tarball)!r},'rb').read()\n"
        "for b in d:\n"
        "    sys.stdout.buffer.write(bytes([b])); sys.stdout.buffer.flush(); time.sleep(0.05)\n"))
    t0 = time.monotonic()
    with pytest.raises(RemoteDispatchError, match="no complete archive within 1s"):
        remote_mod.harvest_remote_dir(RemoteHost(name="p", host="h"), "/w/ck",
                                      tmp_path / "out", timeout=1, stall_timeout=5)
    assert time.monotonic() - t0 < 20


def test_a_healthy_stream_still_extracts_under_the_watchdog(tmp_path, monkeypatch):
    tarball = tmp_path / "ok.tar"
    tarball.write_bytes(_tar_bytes())
    monkeypatch.setattr(remote_mod, "build_ssh_argv", lambda host, cmd: ["cat", str(tarball)])
    dest = remote_mod.harvest_remote_dir(RemoteHost(name="p", host="h"), "/w/ck",
                                         tmp_path / "out", timeout=30, stall_timeout=5)
    assert (dest / "weights.safetensors").read_bytes() == b"W" * 4096


def test_a_slow_but_moving_harvest_outlives_the_soft_deadline():
    """2026-11-03: 14.68 GB of a 14.8 GB make-good checkpoint arrived at ~2.6 MB/s
    when the fixed 3600 s deadline killed it; a stream above the rate floor goes on."""
    breach = remote_mod.harvest_total_breach
    gb = 1_000_000_000
    assert breach(3601, int(14.68 * gb), 3600) is None            # ~4 MB/s: keep going
    assert breach(5400, int(14.0 * gb), 3600) is None             # ~2.6 MB/s: keep going
    assert "floor" in breach(3601, 100 * 1_000_000, 3600)          # ~28 kB/s: fail as before
    assert breach(100, 0, 3600) is None                             # inside the soft deadline
    assert "hard cap" in breach(remote_mod.HARVEST_HARD_CAP_SECONDS + 1, 10**15, 3600)
