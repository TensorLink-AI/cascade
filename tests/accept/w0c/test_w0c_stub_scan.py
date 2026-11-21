"""W0c: scripts/laned/stub_scan.py fails on each planted violation and passes
on clean code (build rule 3)."""

from __future__ import annotations

import re

import pytest

from .helpers import FIXTURES, copy_fixture, output, run_tool, write

STUB = FIXTURES / "stub_scan"

VIOLATIONS = {
    "todo": "todo",
    "fixme": "fixme",
    "pass_body": "pass-body",
    "not_implemented": "not-implemented",
    "ellipsis_body": "ellipsis-body",
    "sample_value_word": "sample-value",
    "sample_value_digest": "sample-value",
    "mock_unittest": "mock-import",
    "mock_module": "mock-import",
}


def _clean_tree(root):
    for src in sorted((STUB / "clean").glob("*.py.txt")):
        copy_fixture(src, root, f"lanedpkg/{src.name[:-len('.txt')]}")
    return root


@pytest.mark.parametrize("name,rule", sorted(VIOLATIONS.items()))
def test_stub_scan_fails_on_planted_violation(tmp_path, name, rule):
    root = _clean_tree(tmp_path / "repo")
    rel = f"lanedpkg/{name}.py"
    copy_fixture(STUB / "violations" / f"{name}.py.txt", root, rel)
    proc = run_tool("stub_scan.py", "--root", root, "lanedpkg/")
    assert proc.returncode == 1, output(proc)
    assert re.search(rf"^{re.escape(rel)}:\d+: {re.escape(rule)}\b", proc.stdout, re.M), proc.stdout
    # Clean files in the same tree are not reported.
    assert "scorer.py" not in proc.stdout and "intake.py" not in proc.stdout, proc.stdout


def test_stub_scan_passes_on_clean_code(tmp_path):
    root = _clean_tree(tmp_path / "repo")
    proc = run_tool("stub_scan.py", "--root", root, "lanedpkg/")
    assert proc.returncode == 0, output(proc)


def test_stub_scan_exempts_tests_directory(tmp_path):
    root = _clean_tree(tmp_path / "repo")
    for name in ("todo", "pass_body", "mock_unittest", "sample_value_word"):
        copy_fixture(STUB / "violations" / f"{name}.py.txt", root, f"tests/accept/w9/test_{name}.py")
    proc = run_tool("stub_scan.py", "--root", root, "lanedpkg/", "tests/")
    assert proc.returncode == 0, output(proc)


def test_stub_scan_reads_paths_file_with_dirs_and_globs(tmp_path):
    root = _clean_tree(tmp_path / "repo")
    copy_fixture(STUB / "violations" / "todo.py.txt", root, "tools/laned_todo.py")
    copy_fixture(STUB / "violations" / "fixme.py.txt", root, "unlisted/fixme.py")
    write(root, "tests/accept/w9/PATHS", "# W9 implementation paths\nlanedpkg/\n\ntools/laned_*.py\n")
    proc = run_tool("stub_scan.py", "--root", root, "--paths-file", root / "tests/accept/w9/PATHS")
    assert proc.returncode == 1, output(proc)
    assert re.search(r"^tools/laned_todo\.py:\d+: todo\b", proc.stdout, re.M), proc.stdout
    # A path not listed in PATHS is out of scope.
    assert "unlisted/fixme.py" not in proc.stdout, proc.stdout
