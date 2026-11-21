"""W0c (rev6, ii): `scripts/laned/accept <WP> --expect-fail` passes only when
the WP's tests are collected and fail for the right reason (an assertion
failure, or a missing module/attribute named in the WP's PATHS), never on a
syntax, fixture or import error elsewhere."""

from __future__ import annotations

import pytest

from .helpers import commit, init_repo, output, run_tool, write

ASSERT_FAIL = "import pkg_old\n\n\ndef test_value():\n    assert pkg_old.VALUE == 2\n"
PASSING = "def test_ok():\n    assert 1 + 1 == 2\n"

CASES = {
    # name: ({test file name: source}, expected to pass expect-fail)
    "assertion_failure": ({"test_a.py": ASSERT_FAIL}, True),
    "missing_module_in_paths": (
        {"test_a.py": "from pkg_new.core import run\n\n\ndef test_run():\n    assert run() == 1\n"}, True),
    "missing_attribute_in_paths": (
        {"test_a.py": "import pkg_old\n\n\ndef test_compute():\n    assert pkg_old.compute() == 3\n"}, True),
    "cannot_import_name_in_paths": (
        {"test_a.py": "from pkg_old import compute\n\n\ndef test_compute():\n    assert compute() == 3\n"},
        True),
    "some_pass_some_fail_right": ({"test_a.py": ASSERT_FAIL, "test_b.py": PASSING}, True),
    "syntax_error": ({"test_a.py": ASSERT_FAIL, "test_b.py": "def test_broken(:\n    assert True\n"}, False),
    "unrelated_import_error": (
        {"test_a.py": "import laned_no_such_dependency\n\n\ndef test_x():\n    assert laned_no_such_dependency\n"},
        False),
    "attribute_error_outside_paths": (
        {"test_a.py": ASSERT_FAIL,
         "test_b.py": "import json\n\n\ndef test_x():\n    assert json.laned_no_such_attr() == 1\n"},
        False),
    "fixture_error": (
        {"test_a.py": ASSERT_FAIL,
         "test_b.py": "def test_x(laned_no_such_fixture):\n    assert laned_no_such_fixture == 1\n"},
        False),
    "all_pass": ({"test_a.py": PASSING}, False),
    "nothing_collected": ({"test_a.py": "VALUE = 1\n"}, False),
}


def _repo(tmp_path, files):
    repo = init_repo(tmp_path / "repo")
    write(repo, "pkg_old.py", "VALUE = 1\n")
    write(repo, "tests/accept/w9/PATHS", "pkg_new/\npkg_old.py\n")
    for name, src in files.items():
        write(repo, f"tests/accept/w9/{name}", src)
    commit(repo, "tests-only", ["pkg_old.py", "tests"])
    return repo


@pytest.mark.parametrize("case", sorted(CASES))
def test_expect_fail(tmp_path, case):
    files, passes = CASES[case]
    proc = run_tool("accept", "W9", "--expect-fail", "--root", _repo(tmp_path, files))
    if passes:
        assert proc.returncode == 0, output(proc)
    else:
        assert proc.returncode != 0, output(proc)


def test_expect_fail_also_enforces_invariant10(tmp_path):
    src = ("import pkg_old\n\n\n"
           "def test_receipt_" + "unchanged_when_unarmed():\n    assert pkg_old.VALUE == 2\n")
    proc = run_tool("accept", "W9", "--expect-fail", "--root", _repo(tmp_path, {"test_a.py": src}))
    assert proc.returncode != 0, output(proc)
    assert "FAIL invariant10" in output(proc), output(proc)
