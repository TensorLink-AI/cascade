"""The release code-path guard (2026-11-04): a service refuses to start when
the imported ``cascade`` is not the declared release — a stale copy on the
cwd shadowed PYTHONPATH and ran Oct-30 intake code for a day."""

from __future__ import annotations

import types
from pathlib import Path

import pytest

from cascade.shared import codepath


def _pkg(tmp_path: Path, name: str):
    root = tmp_path / name
    (root / "cascade").mkdir(parents=True)
    init = root / "cascade" / "__init__.py"
    init.write_text("")
    return types.SimpleNamespace(__file__=str(init)), root


def test_declared_release_prefers_explicit_env_then_first_pythonpath(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    assert codepath.declared_release_root({"PYTHONPATH": f"{a}:{b}"}) == a.resolve()
    assert codepath.declared_release_root(
        {"PYTHONPATH": str(a), codepath.RELEASE_PATH_ENV: str(b)}) == b.resolve()
    assert codepath.declared_release_root({}) is None
    assert codepath.declared_release_root({"PYTHONPATH": ""}) is None


def test_matching_release_starts_and_logs(tmp_path, caplog):
    pkg, root = _pkg(tmp_path, "rc28")
    exits: list[int] = []
    with caplog.at_level("INFO", logger="cascade.codepath"):
        got = codepath.verify_release_code_path(
            "cascade-intake", package=pkg, environ={"PYTHONPATH": str(root)},
            exit_fn=exits.append)
    assert got == root.resolve() and not exits
    assert any("cascade-intake: code" in r.getMessage() for r in caplog.records)


def test_shadowed_release_refuses_to_start(tmp_path, caplog):
    stale_pkg, _stale = _pkg(tmp_path, "cwd-copy")
    _good_pkg, release = _pkg(tmp_path, "rc28")
    exits: list[int] = []
    with caplog.at_level("CRITICAL", logger="cascade.codepath"), pytest.raises(
            codepath.CodePathMismatch):
        codepath.verify_release_code_path(
            "cascade-intake", package=stale_pkg, environ={"PYTHONPATH": str(release)},
            exit_fn=exits.append)
    assert exits == [codepath.EXIT_CODE_PATH_MISMATCH]
    msg = " ".join(r.getMessage() for r in caplog.records)
    assert "REFUSING TO START" in msg and "cwd-copy" in msg and "rc28" in msg


def test_shadowed_release_may_be_allowed_explicitly(tmp_path, caplog):
    stale_pkg, stale = _pkg(tmp_path, "cwd-copy")
    _good_pkg, release = _pkg(tmp_path, "rc28")
    exits: list[int] = []
    with caplog.at_level("CRITICAL", logger="cascade.codepath"):
        got = codepath.verify_release_code_path(
            "cascade-trainer", package=stale_pkg,
            environ={"PYTHONPATH": str(release), codepath.ALLOW_MISMATCH_ENV: "1"},
            exit_fn=exits.append)
    assert got == stale.resolve() and not exits
    assert any("continuing because" in r.getMessage() for r in caplog.records)


def test_no_declared_release_is_not_a_mismatch(tmp_path):
    pkg, root = _pkg(tmp_path, "venv-install")
    exits: list[int] = []
    assert codepath.verify_release_code_path("cascade-validator", package=pkg, environ={},
                                             exit_fn=exits.append) == root.resolve()
    assert not exits


def test_every_service_main_checks_its_code_path_first():
    """Each entry point calls the guard before anything else can misbehave."""
    import inspect

    from cascade.funding import main as funding_main
    from cascade.provision import main as provision_main
    from cascade.trainer import main as trainer_main
    from cascade.validator import main as validator_main

    for mod, svc in ((funding_main, "cascade-intake"), (trainer_main, "cascade-trainer"),
                     (validator_main, "cascade-validator"),
                     (provision_main, "cascade-provisioner")):
        src = inspect.getsource(mod.main)
        assert f'verify_release_code_path("{svc}")' in src, svc
