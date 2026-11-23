"""W0c (rev6, iii; rev6.17): `scripts/laned/accept <WP> --check-pr` lets a PR
change the WP's tests/accept/<wp>/ together with code inside its PATHS
(MODE combined, plan rev6.17 rule 1) and fails one that touches a path
outside the WP's PATHS, with or without tests."""

from __future__ import annotations

import re

import pytest

from .helpers import commit, git, init_repo, output, run_tool, write


@pytest.fixture
def pr_repo(tmp_path):
    repo = init_repo(tmp_path / "repo")
    write(repo, "tests/accept/w9/PATHS", "lanedpkg/w9.py\nlanedpkg/w9_*.py\nlanedpkg/w9data/\n")
    write(repo, "tests/accept/w9/test_w9.py", "def test_spec():\n    assert True\n")
    write(repo, "tests/accept/w8/PATHS", "lanedpkg/w8.py\n")
    write(repo, "lanedpkg/__init__.py", "")
    write(repo, "README.md", "base\n")
    commit(repo, "base", ["tests", "lanedpkg", "README.md"])
    return repo


def _branch(repo, name, files):
    git(repo, "checkout", "-q", "-b", name, "main")
    for rel, text in files.items():
        write(repo, rel, text)
    commit(repo, name, list(files))
    git(repo, "checkout", "-q", "main")
    return name


def _check(repo, head, wp="W9"):
    return run_tool("accept", wp, "--check-pr", "--base", "main", "--head", head, "--root", repo)


CASES = {
    # name: (files, passes, mode)
    "tests_only": ({"tests/accept/w9/test_more.py": "def test_more():\n    assert False\n"},
                   True, "tests-only"),
    # rev6.17 rule 1: tests + code together is one feature PR
    "tests_plus_impl": ({"tests/accept/w9/test_more.py": "def test_more():\n    assert False\n",
                         "lanedpkg/w9.py": "VALUE = 9\n"}, True, "combined"),
    "rule1_tests_plus_docs": ({"tests/accept/w9/test_more.py": "def test_more():\n    assert False\n",
                               "README.md": "edited\n"}, False, None),
    "tests_for_other_wp": ({"tests/accept/w8/test_w8.py": "def test_w8():\n    assert False\n"},
                           False, None),
    "impl_in_paths": ({"lanedpkg/w9.py": "VALUE = 9\n", "lanedpkg/w9_extra.py": "EXTRA = 1\n",
                       "lanedpkg/w9data/table.json": "{}\n"}, True, "implementation"),
    "impl_outside_paths": ({"lanedpkg/w9.py": "VALUE = 9\n", "lanedpkg/other.py": "OTHER = 1\n"},
                           False, None),
    "impl_touches_laned_tooling": ({"lanedpkg/w9.py": "VALUE = 9\n",
                                    "scripts/laned/accept": "edited\n"}, False, None),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_check_pr(pr_repo, case):
    files, passes, mode = CASES[case]
    head = _branch(pr_repo, f"pr-{case}", files)
    proc = _check(pr_repo, head)
    if passes:
        assert proc.returncode == 0, output(proc)
        assert re.search(rf"^MODE {mode}$", proc.stdout, re.M), proc.stdout
    else:
        assert proc.returncode != 0, output(proc)


def test_check_pr_tests_plus_outside_path_is_named(pr_repo):
    # rev6.17: tests may ride with code, never with a path outside PATHS
    files, _, _ = CASES["rule1_tests_plus_docs"]
    head = _branch(pr_repo, "pr-named", files)
    proc = _check(pr_repo, head)
    assert proc.returncode != 0
    assert "FAIL paths" in output(proc), output(proc)
    assert "README.md" in output(proc), output(proc)


def test_check_pr_paths_failure_is_named(pr_repo):
    files, _, _ = CASES["impl_outside_paths"]
    head = _branch(pr_repo, "pr-paths", files)
    proc = _check(pr_repo, head)
    assert proc.returncode != 0
    assert "FAIL paths" in output(proc), output(proc)
    assert "lanedpkg/other.py" in output(proc), output(proc)
