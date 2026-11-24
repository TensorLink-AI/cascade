"""W0c (plan rev6.17, owner 2026-11-23): one PR per feature.

- rule 1 relaxed: a PR may change a WP's tests/accept/<wp>/ together with
  code in its PATHS (`--check-pr` prints MODE combined); `--combined` runs
  the WP's NEW tests from the PR head against the base's code, where each
  must fail for a right reason (a new test passing there proves nothing,
  unless the WP is a declared verification WP), then the full Accept on HEAD;
- PATHS ride along: entries the PR adds count for its scope;
- feature batches: any number of laned:<WP> labels, overlapping PATHS;
- anti-gaming: deleting or weakening existing assertions needs the flag
  label laned:tests-correction (`--correction`);
- LABELLED L2 rule: a squash merge `(#N)` is a WP's L2 commit only when PR N
  carried laned:<WP> and changed a non-test path in the WP's PATHS.
"""

from __future__ import annotations

import json
import re

import pytest

from .helpers import commit, git, init_repo, output, run_tool, write

SIX = [f"F{n}" for n in range(1, 7)]

PLAN = "# fixture plan\n\n| WP | What | Depends | Runs in | Accept |\n|---|---|---|---|---|\n" + "".join(
    f"| {w} | {w} | — | ci | `scripts/laned/accept {w}` |\n" for w in ("W1", "V1", "N1", "G1", *SIX))


@pytest.fixture
def repo(tmp_path):
    repo = init_repo(tmp_path / "repo")
    files = {
        "docs/plans/LANE_D_PLAN.md": PLAN, "README.md": "base\n", ".gitignore": "__pycache__/\n",
        "pkg/__init__.py": "", "pkg/core.py": "VALUE = 1\n", "pkg/w1.py": "VALUE = 1\n",
        "pkg/v1.py": "VALUE = 1\n", "scripts/laned/verification_wps": "V1\n",
    }
    for rel, text in files.items():
        write(repo, rel, text)
    commit(repo, "seed", list(files))
    tests = {
        "tests/accept/w1/PATHS": "pkg/w1.py\n",
        "tests/accept/w1/test_w1.py": "import pkg.w1\n\n\ndef test_w1():\n    assert pkg.w1.VALUE >= 1\n"
                                      "    assert isinstance(pkg.w1.VALUE, int)\n",
        "tests/accept/v1/PATHS": "pkg/v1.py\n",
        "tests/accept/v1/test_v1.py": "import pkg.v1\n\n\ndef test_v1():\n    assert pkg.v1.VALUE\n",
        "tests/accept/g1/PATHS": "pkg/core.py\npkg/g1.py\n",
        "tests/accept/g1/test_g1.py": "import pkg.core\n\n\ndef test_g1():\n    assert pkg.core.VALUE\n",
    }
    for w in SIX:
        d = w.lower()
        tests[f"tests/accept/{d}/PATHS"] = f"pkg/core.py\npkg/{d}.py\n"
        tests[f"tests/accept/{d}/test_{d}.py"] = f"import pkg.{d}\n\n\ndef test_{d}():\n    assert pkg.{d}.VALUE\n"
    for rel, text in tests.items():
        write(repo, rel, text)
    commit(repo, "tests(laned): W1 V1 G1 F1-F6", list(tests))
    write(repo, "pkg/w1.py", "VALUE = 2\n")
    commit(repo, "feat(laned): W1", ["pkg/w1.py"])
    return repo


W1_MORE = "import pkg.w1\n\n\ndef test_w1_three():\n    assert pkg.w1.VALUE == 3\n"


def _branch(repo, name, files, start="main"):
    git(repo, "checkout", "-q", "-b", name, start)
    for rel, text in files.items():
        if text is None:
            git(repo, "rm", "-q", rel)
        else:
            write(repo, rel, text)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", name)
    git(repo, "checkout", "-q", "main")
    return name


def _check(repo, head, labels, wp, *extra):
    return run_tool("accept", wp, "--check-pr", "--base", "main", "--head", head,
                    "--pr-wps", ",".join(labels), *extra, "--root", repo)


def _combined(repo, head, wp, *extra):
    git(repo, "checkout", "-q", head)
    try:
        return run_tool("accept", wp, "--combined", "--base", "main", "--head", head, *extra, "--root", repo)
    finally:
        git(repo, "checkout", "-q", "main")


def _mode(proc):
    m = re.search(r"^MODE (\S+)$", proc.stdout, re.M)
    return m.group(1) if m else None


def test_tests_plus_code_passes_when_the_new_test_fails_on_base(repo):
    head = _branch(repo, "pr", {"pkg/w1.py": "VALUE = 3\n", "tests/accept/w1/test_w1_more.py": W1_MORE})
    proc = _check(repo, head, ["W1"], "W1")
    assert proc.returncode == 0 and _mode(proc) == "combined", output(proc)
    proc = _combined(repo, head, "W1")
    assert proc.returncode == 0, output(proc)
    assert "PASS W1 combined" in proc.stdout and "LEDGER-MODE combined" in proc.stdout, output(proc)
    assert re.search(r"^TESTS-TREE [0-9a-f]{40}$", proc.stdout, re.M), proc.stdout


def test_tests_plus_code_fails_when_head_does_not_pass(repo):
    head = _branch(repo, "pr", {"pkg/w1.py": "VALUE = 4\n", "tests/accept/w1/test_w1_more.py": W1_MORE})
    proc = _combined(repo, head, "W1")
    assert proc.returncode != 0 and "PASS W1 combined" not in proc.stdout, output(proc)


def test_new_test_already_passing_on_base_is_refused(repo):
    head = _branch(repo, "pr", {
        "pkg/w1.py": "VALUE = 3\n", "tests/accept/w1/test_w1_more.py": W1_MORE,
        "tests/accept/w1/test_w1_old.py": "import pkg.w1\n\n\ndef test_positive():\n    assert pkg.w1.VALUE > 0\n"})
    proc = _combined(repo, head, "W1")
    assert proc.returncode != 0, output(proc)
    assert "already pass on BASE" in proc.stdout, output(proc)


def test_verification_wp_may_add_tests_passing_on_base(repo):
    head = _branch(repo, "pr", {
        "pkg/v1.py": "VALUE = 1  # documented\n",
        "tests/accept/v1/test_v1_more.py": "import pkg.v1\n\n\ndef test_int():\n    assert isinstance(pkg.v1.VALUE, int)\n"})
    proc = _combined(repo, head, "V1")
    assert proc.returncode == 0 and "PASS V1 combined" in proc.stdout, output(proc)


def test_wrong_reason_failure_on_base_is_refused(repo):
    head = _branch(repo, "pr", {
        "pkg/w1.py": "VALUE = 3\n",
        "tests/accept/w1/test_w1_more.py": "import laned_no_such_dep\n\n\ndef test_x():\n    assert laned_no_such_dep\n"})
    proc = _combined(repo, head, "W1")
    assert proc.returncode != 0 and "wrong reason" in proc.stdout, output(proc)


def test_paths_ride_along(repo):
    files = {"tests/accept/w1/PATHS": "pkg/w1.py\npkg/w1_extra.py\n", "pkg/w1_extra.py": "EXTRA = 5\n",
             "tests/accept/w1/test_w1_extra.py":
                 "import pkg.w1_extra\n\n\ndef test_extra():\n    assert pkg.w1_extra.EXTRA == 5\n"}
    head = _branch(repo, "pr", files)
    proc = _check(repo, head, ["W1"], "W1")
    assert proc.returncode == 0 and _mode(proc) == "combined", output(proc)
    assert "PATHS-ADDED W1 pkg/w1_extra.py" in proc.stdout, output(proc)
    proc = _combined(repo, head, "W1")
    assert proc.returncode == 0, output(proc)
    outside = _branch(repo, "pr-outside", {**files, "pkg/other.py": "X = 1\n"})
    proc = _check(repo, outside, ["W1"], "W1")
    assert proc.returncode != 0 and "FAIL paths" in proc.stdout and "pkg/other.py" in proc.stdout, output(proc)


def test_new_wp_tests_paths_and_code_in_one_pr(repo):
    head = _branch(repo, "pr", {
        "tests/accept/n1/PATHS": "pkg/n1.py\n", "pkg/n1.py": "VALUE = 7\n",
        "tests/accept/n1/test_n1.py": "import pkg.n1\n\n\ndef test_n1():\n    assert pkg.n1.VALUE == 7\n"})
    proc = _check(repo, head, ["N1"], "N1")
    assert proc.returncode == 0 and _mode(proc) == "combined", output(proc)
    proc = _combined(repo, head, "N1")
    assert proc.returncode == 0 and "PASS N1 combined" in proc.stdout, output(proc)


WEAKENED = {
    "fewer_assertions": {"tests/accept/w1/test_w1.py":
                         "import pkg.w1\n\n\ndef test_w1():\n    assert pkg.w1.VALUE >= 1\n"},
    "removed_test_file": {"tests/accept/w1/test_w1.py": None},
    "skip_added": {"tests/accept/w1/test_w1.py":
                   "import pytest\nimport pkg.w1\n\n\n@pytest.mark.skip\ndef test_w1():\n"
                   "    assert pkg.w1.VALUE >= 1\n    assert isinstance(pkg.w1.VALUE, int)\n"},
}


@pytest.mark.parametrize("case", sorted(WEAKENED))
def test_weakened_tests_need_the_tests_correction_label(repo, case):
    files = {"pkg/w1.py": "VALUE = 3\n", "tests/accept/w1/test_w1_more.py": W1_MORE, **WEAKENED[case]}
    head = _branch(repo, f"pr-{case}", files)
    proc = _check(repo, head, ["W1"], "W1")
    assert proc.returncode != 0 and "FAIL weakened" in proc.stdout, output(proc)
    assert _mode(proc) is None, proc.stdout
    proc = _check(repo, head, ["W1", "tests-correction"], "W1", "--correction")
    assert proc.returncode == 0 and _mode(proc) == "combined", output(proc)


def _feature_files():
    files = {"pkg/core.py": "VALUE = 2\n"}
    files.update({f"pkg/{w.lower()}.py": "from pkg.core import VALUE  # noqa: F401\n" for w in SIX})
    files["tests/accept/f1/test_f1_core.py"] = "import pkg.core\n\n\ndef test_two():\n    assert pkg.core.VALUE == 2\n"
    return files


def test_six_wp_overlapping_feature_batch(repo):
    head = _branch(repo, "pr-six", _feature_files())
    for wp in SIX:
        proc = _check(repo, head, SIX, wp)
        assert proc.returncode == 0, output(proc)
        assert f"BATCH {' '.join(SIX)}" in proc.stdout, proc.stdout
        assert _mode(proc) == ("combined" if wp == "F1" else "implementation"), proc.stdout
    proc = _combined(repo, head, "F1")
    assert proc.returncode == 0, output(proc)


def _board(repo, tmp_path, runs, labels=None):
    ci = tmp_path / "ci.json"
    ci.write_text(json.dumps({"runs": runs}))
    env = {}
    if labels is not None:
        f = tmp_path / "pr_labels.json"
        f.write_text(json.dumps([{"number": n, "labels": [{"name": f"laned:{w}"} for w in ws]}
                                 for n, ws in labels.items()]))
        env["LANED_PR_LABELS"] = str(f)
    out = tmp_path / "board"
    proc = run_tool("status", "--root", repo, "--ref", "main", "--ci-results", ci, "--out", out, env=env)
    assert proc.returncode == 0, output(proc)
    return json.loads((out / "status" / "laned_build.json").read_text())["wps"]


def test_labelled_l2_rule(repo, tmp_path):
    """The squash merge (#77) labelled F1-F6 is each one's L2; G1 (PATHS
    pkg/core.py, not labelled) is not, nor from #79, a merge labelled F1 that
    only changes pkg/core.py (inside G1's PATHS: the old git-only rule said
    L2, the #817 / W5i case)."""
    _branch(repo, "pr-six", _feature_files())
    git(repo, "merge", "-q", "--squash", "pr-six")
    git(repo, "commit", "-q", "-m", "feat(laned): F1-F6 (#77)")
    write(repo, "pkg/core.py", "VALUE = 3\n")
    commit(repo, "fix(laned): F1 core (#79)", ["pkg/core.py"])
    xf = [{"workflow": "laned-accept", "event": "pull_request", "mode": "expect-fail", "wp": w,
           "head_sha": "0" * 40, "conclusion": "success"} for w in [*SIX, "G1"]]
    rows = [{"workflow": "laned-accept", "event": "pull_request", "mode": "accept", "wp": w, "pr": 77,
             "head_sha": "1" * 40, "conclusion": "success", "created_at": "2026-11-23T00:00:00Z"} for w in SIX]
    board = _board(repo, tmp_path, xf + rows, labels={77: SIX, 79: ["F1"]})
    for w in SIX:
        assert board[w]["level"] == "L2", (w, board[w])
    assert board["G1"]["level"] == "L1", board["G1"]
    # labels unknown (no file, no PR rows): the git-only rule still applies
    legacy = _board(repo, tmp_path, xf)
    assert legacy["G1"]["level"] == "L2", legacy["G1"]


def test_ledger_records_pr_and_combined(tmp_path):
    f = tmp_path / "results.json"
    proc = run_tool("ci_ledger.py", "record", "--file", f, "--wp", "W1", "--mode", "combined",
                    "--event", "pull_request", "--head-sha", "a" * 40, "--conclusion", "success",
                    "--pr", "12", "--tests-tree", "b" * 40)
    assert proc.returncode == 0, output(proc)
    row = json.loads(f.read_text())["runs"][-1]
    assert row.get("pr") == 12 and row.get("mode") == "combined", row
