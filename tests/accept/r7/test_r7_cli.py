"""R7: ``overlay`` joins the ``cascade-laned`` dispatcher, and the wrapper's
source never touches keys, signing or storage itself.

Contract: tests/accept/r7/README.md ("Where it lives", "Keys").
"""

from __future__ import annotations

import importlib

from .helpers import FORBIDDEN_IN_SOURCE, REPO, describe, run_overlay

MODULE = REPO / "cascade" / "laned" / "overlay.py"


def test_r7_overlay_registered_beside_blocks():
    cli = importlib.import_module("cascade.laned.cli")
    assert cli.COMMANDS.get("overlay") == "cascade.laned.overlay", (
        f"cascade-laned has no overlay command: {cli.COMMANDS}")
    assert cli.COMMANDS.get("blocks") == "cascade.laned.blocks", (
        "R1's blocks command was dropped from the dispatcher")
    mod = importlib.import_module("cascade.laned.overlay")
    assert callable(getattr(mod, "add_parser", None)), "cascade.laned.overlay has no add_parser"


def test_r7_overlay_help_lists_commands(tmp_path):
    proc = run_overlay(tmp_path, "--help")
    assert proc.returncode == 0, describe(proc)
    for name in ("benign", "refused", "kill", "publish"):
        assert name in proc.stdout, f"overlay --help does not list {name!r}\n{proc.stdout}"


def test_r7_wrapper_source_never_touches_keys():
    assert MODULE.is_file(), f"{MODULE} does not exist"
    text = MODULE.read_text(encoding="utf-8")
    low = text.lower()
    for needle in FORBIDDEN_IN_SOURCE:
        assert needle.lower() not in low, (
            f"{MODULE.relative_to(REPO)} contains {needle!r}: the wrapper only prints "
            "(and, approved, runs) cascade-validator; it never reads keys, signs or writes "
            "storage itself")
