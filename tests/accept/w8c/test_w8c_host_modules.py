"""W8c eval-host startup self-check (2026-11-26).

A fresh lane D eval host synced with ``--extra train`` only had no
``pyarrow``; the public half then failed mid-settlement with "mirror read
failed (ModuleNotFoundError: No module named 'pyarrow')". ``public`` and
``forecast`` now look their modules up (``importlib.util.find_spec``, nothing
imported) and refuse at start, before stdin is read, naming the missing
module and the ``uv sync`` that installs it. Torch-free: find_spec is faked.
"""

from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path

import pytest

from . import helpers as h

REPO = Path(__file__).resolve().parents[3]


def _fake_find_spec(monkeypatch, missing):
    real = importlib.util.find_spec

    def find_spec(name, *a, **k):
        if name.split(".")[0] in missing:
            return None
        if name in ("torch", "safetensors", "pyarrow", "boto3"):
            return object()          # present: never import the real one
        return real(name, *a, **k)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)


def _check(m):
    fn = getattr(m, "check_host_modules", None)
    assert fn is not None, (
        "lane_d_forecasts has no check_host_modules (no startup self-check: a host "
        "without pyarrow fails the public half mid-settlement)")
    return fn


class _Stdin:
    def __init__(self):
        self.read_called = False
        outer = self

        class _Buf(io.BytesIO):
            def read(self, *a):
                outer.read_called = True
                return b""

        self.buffer = _Buf()


def _run(m, monkeypatch, capsys, argv):
    stdin = _Stdin()
    monkeypatch.setattr(sys, "stdin", stdin)
    rc = m.main(argv)
    return rc, capsys.readouterr(), stdin.read_called


def test_public_without_pyarrow_refuses_at_start(monkeypatch, capsys, tmp_path):
    m = h.forecasts_mod()
    _check(m)
    _fake_find_spec(monkeypatch, {"pyarrow"})
    rc, out, read = _run(m, monkeypatch, capsys,
                         ["public", "--cache-dir", str(tmp_path), "--budget-bytes", "10",
                          "--public-root", str(tmp_path)])
    assert rc == getattr(m, "EXIT_MISSING_MODULES", None) and rc not in (0, 2), rc
    assert not read, "the request was read before the self-check"
    assert out.out == ""
    assert "pyarrow" in out.err and "pool-forge" in out.err, out.err
    assert "uv sync --frozen" in out.err and "--extra pool-forge" in out.err, out.err


def test_hippius_mirror_needs_boto3_local_root_does_not(monkeypatch):
    m = h.forecasts_mod()
    check = _check(m)
    _fake_find_spec(monkeypatch, {"boto3"})
    assert check("public", hippius_mirror=False) is None
    msg = check("public", hippius_mirror=True)
    assert msg and "boto3" in msg and "hippius" in msg, msg
    assert check("forecast") is None


def test_forecast_without_torch_refuses_and_hello_never_checks(monkeypatch, capsys, tmp_path):
    m = h.forecasts_mod()
    check = _check(m)
    _fake_find_spec(monkeypatch, {"torch"})
    msg = check("forecast")
    assert msg and "torch" in msg and "train" in msg, msg
    rc, out, read = _run(m, monkeypatch, capsys,
                         ["forecast", "--cache-dir", str(tmp_path), "--budget-bytes", "10"])
    assert rc == m.EXIT_MISSING_MODULES and not read and "torch" in out.err, (rc, out.err)


def test_every_module_present_passes(monkeypatch):
    m = h.forecasts_mod()
    check = _check(m)
    _fake_find_spec(monkeypatch, set())
    for cmd in ("public", "forecast"):
        assert check(cmd, hippius_mirror=True) is None, cmd


@pytest.mark.parametrize("cmd", ["public", "forecast"])
def test_named_extras_are_the_eval_host_profile_extras(cmd):
    """Every extra the refusal can name is in the launch profile's uv_extras."""
    import tomllib

    m = h.forecasts_mod()
    _check(m)
    prof = tomllib.loads((REPO / "scripts/evalhost/profiles/launch_eval.toml").read_text())
    needed = {m.MODULE_EXTRAS[x] for x in m.HOST_MODULES[cmd]} | {
        m.MODULE_EXTRAS[x] for x in m.MIRROR_MODULES}
    assert needed <= set(prof["uv_extras"]), (needed, prof["uv_extras"])
