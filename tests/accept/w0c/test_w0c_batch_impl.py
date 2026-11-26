"""W0c (plan rev6.3 (b), rev6.17): feature PRs over several WPs.

`scripts/laned/accept <WP> --check-pr --pr-wps A,B,C` lets one PR carry any
number of laned:<WP> labels (rev6.17, one PR per feature): overlapping PATHS
and in-batch Depends are allowed; every changed file must be in some
labelled WP's PATHS and every label must own a changed file. Each WP is still
accepted on its own, and the squash merge is each WP's L2 commit (status board
and `built_on`). A tests-only batch has no size cap and may mix built and
unbuilt WPs (MODE per WP). A single label keeps the one-WP rules.
"""

from __future__ import annotations

import json
import re

import pytest

from .helpers import commit, git, init_repo, output, run_tool, write

PLAN = """# fixture plan

## 3. Implementation

### 3.0 Work packages

| WP | What | Depends | Runs in | Accept |
|---|---|---|---|---|
| B1 | batch one | — | ci | `scripts/laned/accept B1` |
| B2 | batch two | — | ci | `scripts/laned/accept B2` |
| B3 | batch three | — | ci | `scripts/laned/accept B3` |
| B4 | batch four | — | ci | `scripts/laned/accept B4` |
| D1 | depends on B1 | B1 | ci | `scripts/laned/accept D1` |
| D2 | depends on D1 | D1 **(rev6)** | ci | `scripts/laned/accept D2` |
| O1 | overlaps B1 | — | ci | `scripts/laned/accept O1` |
| N1 | tests not on main | — | ci | `scripts/laned/accept N1` |

| WP | Stage | Runs in | Accept |
|---|---|---|---|
| X1 | S9 | ci | `scripts/laned/accept X1` |
"""

# WP -> (PATHS, implementation files {path: text}, test module)
WPS = {
    "B1": ("pkg/b1.py\n", {"pkg/b1.py": "VALUE = 1\n"},
           "import pkg.b1\n\n\ndef test_b1():\n    assert pkg.b1.VALUE == 1\n"),
    "B2": ("pkg/b2/\n", {"pkg/b2/__init__.py": "", "pkg/b2/mod.py": "VALUE = 2\n"},
           "import pkg.b2.mod\n\n\ndef test_b2():\n    assert pkg.b2.mod.VALUE == 2\n"),
    "B3": ("pkg/b3_*.py\n", {"pkg/b3_core.py": "VALUE = 3\n"},
           "import pkg.b3_core\n\n\ndef test_b3():\n    assert pkg.b3_core.VALUE == 3\n"),
    "B4": ("pkg/b4.py\n", {"pkg/b4.py": "VALUE = 4\n"},
           "import pkg.b4\n\n\ndef test_b4():\n    assert pkg.b4.VALUE == 4\n"),
    "D1": ("pkg/d1.py\n", {"pkg/d1.py": "VALUE = 5\n"},
           "import pkg.d1\n\n\ndef test_d1():\n    assert pkg.d1.VALUE == 5\n"),
    "D2": ("pkg/d2.py\n", {"pkg/d2.py": "VALUE = 6\n"},
           "import pkg.d2\n\n\ndef test_d2():\n    assert pkg.d2.VALUE == 6\n"),
    "O1": ("pkg/\n", {"pkg/o1.py": "VALUE = 7\n"},
           "import pkg.o1\n\n\ndef test_o1():\n    assert pkg.o1.VALUE == 7\n"),
    "X1": ("pkg/x1.py\n", {"pkg/x1.py": "VALUE = 8\n"},
           "import pkg.x1\n\n\ndef test_x1():\n    assert pkg.x1.VALUE == 8\n"),
}


def _impl(*wps: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for wp in wps:
        out.update(WPS[wp][1])
    return out


@pytest.fixture
def repo(tmp_path):
    repo = init_repo(tmp_path / "repo")
    write(repo, "docs/plans/LANE_D_PLAN.md", PLAN)
    write(repo, "README.md", "base\n")
    write(repo, ".gitignore", "__pycache__/\n")
    write(repo, "pkg/__init__.py", "")
    commit(repo, "plan", ["docs", "README.md", ".gitignore", "pkg"])
    for wp, (paths, _, test) in WPS.items():  # tests-only merges (N1 has none on main)
        d = wp.lower()
        write(repo, f"tests/accept/{d}/PATHS", paths)
        write(repo, f"tests/accept/{d}/test_{d}.py", test)
        commit(repo, f"tests(laned): {wp}", [f"tests/accept/{d}"])
    return repo


def _branch(repo, name, files, start="main"):
    git(repo, "checkout", "-q", "-b", name, start)
    for rel, text in files.items():
        write(repo, rel, text)
    commit(repo, name, list(files))
    git(repo, "checkout", "-q", "main")
    return name


def _check(repo, head, labels, wp=None):
    args = ["--pr-wps", ",".join(labels)] if labels else []
    return run_tool("accept", wp or labels[0], "--check-pr", "--base", "main", "--head", head,
                    *args, "--root", repo)


def _squash(repo, branch, msg):
    git(repo, "merge", "-q", "--squash", branch)
    git(repo, "commit", "-q", "-m", msg)
    return git(repo, "rev-parse", "HEAD")


@pytest.mark.parametrize("labels", [["B1", "B2"], ["B1", "B2", "B3"]], ids=["two", "three"])
def test_batch_implementation_passes_for_every_label(repo, labels):
    head = _branch(repo, "pr-" + "-".join(labels), _impl(*labels))
    for wp in labels:
        proc = _check(repo, head, labels, wp)
        assert proc.returncode == 0, output(proc)
        lines = [ln for ln in proc.stdout.splitlines() if ln.startswith(("BATCH ", "MODE "))]
        assert lines == ["BATCH " + " ".join(labels), "MODE implementation"], proc.stdout
        for w in labels:
            assert re.search(rf"^SCOPE {w} \d+ file\(s\)$", proc.stdout, re.M), proc.stdout


PASSES = {
    # rev6.17 FEATURE rule: none of these is refused any more
    "four_wps": ["B1", "B2", "B3", "B4"],
    "overlapping_paths": ["B1", "O1"],
    "in_batch_dependency": ["B1", "D1"],
    "transitive_dependency": ["D2", "B1"],
    "deps_unknown": ["B1", "X1"],
}


@pytest.mark.parametrize("case", sorted(PASSES))
def test_feature_batch_passes(repo, case):
    labels = PASSES[case]
    head = _branch(repo, f"pr-{case}", _impl(*labels))
    for wp in labels:
        proc = _check(repo, head, labels, wp)
        assert proc.returncode == 0, output(proc)
        lines = [ln for ln in proc.stdout.splitlines() if ln.startswith(("BATCH ", "MODE "))]
        assert lines == ["BATCH " + " ".join(labels), "MODE implementation"], proc.stdout


REFUSED = {
    # name: (labels, files, step, reason)
    "tests_not_on_main": (["B1", "N1"], {**_impl("B1"), "pkg/n1.py": "VALUE = 9\n"}, "rule1",
                          "batch-no-tests"),
    "file_in_no_wp": (["B1", "B2"], {**_impl("B1", "B2"), "README.md": "edited\n"}, "paths",
                      "batch-out-of-scope"),
    "laned_tooling": (["B1", "B2"], {**_impl("B1", "B2"), "scripts/laned/accept": "x\n"}, "paths",
                      "batch-out-of-scope"),
    "label_without_changes": (["B1", "B2"], _impl("B1"), "rule1", "batch-label-without-changes"),
    "review_label": (["B1", "RV9"], _impl("B1"), "rule1", "batch-non-wp-label"),
}


@pytest.mark.parametrize("case", sorted(REFUSED))
def test_batch_implementation_refused(repo, case):
    labels, files, step, reason = REFUSED[case]
    head = _branch(repo, f"pr-{case}", files)
    for wp in labels:
        if wp.startswith("RV"):
            continue  # its own check-pr is the review rule
        proc = _check(repo, head, labels, wp)
        assert proc.returncode != 0, output(proc)
        assert f"FAIL {step}: {reason}" in proc.stdout, output(proc)
        assert "MODE " not in proc.stdout, proc.stdout


def test_dependent_batch_is_one_feature_pr(repo):
    # rev6.17: D2 depends on B1 (transitively); one feature PR carries both
    head = _branch(repo, "pr-dep-feature", _impl("B1", "D2"))
    for wp in ("B1", "D2"):
        proc = _check(repo, head, ["B1", "D2"], wp)
        assert proc.returncode == 0, output(proc)
        assert "depends on" not in proc.stdout, output(proc)


def test_file_in_two_batch_wps_is_owned_by_each(repo):
    # B1 and B4 both claim pkg/shared.py: rev6.17 counts the changed file
    # for each WP whose PATHS match it (no overlap / ambiguity refusal)
    write(repo, "tests/accept/b4/PATHS", "pkg/b4.py\npkg/shared.py\n")
    write(repo, "tests/accept/b1/PATHS", "pkg/b1.py\npkg/shared.py\n")
    commit(repo, "tests(laned): widen B1/B4", ["tests/accept/b1/PATHS", "tests/accept/b4/PATHS"])
    head = _branch(repo, "pr-shared", {**_impl("B1", "B4"), "pkg/shared.py": "S = 1\n"})
    proc = _check(repo, head, ["B1", "B4"])
    assert proc.returncode == 0, output(proc)
    assert re.search(r"^SCOPE B1 2 file\(s\)$", proc.stdout, re.M), proc.stdout
    assert re.search(r"^SCOPE B4 2 file\(s\)$", proc.stdout, re.M), proc.stdout


def test_single_label_implementation_unchanged(repo):
    # one label: another WP's file is outside its PATHS (classic message)
    head = _branch(repo, "pr-single", _impl("B1", "B2"))
    for labels in (None, ["B1"]):
        proc = run_tool("accept", "B1", "--check-pr", "--base", "main", "--head", head,
                        *(["--pr-wps", ",".join(labels)] if labels else []), "--root", repo)
        assert proc.returncode != 0, output(proc)
        assert "FAIL paths: outside the WP's PATHS: pkg/b2/__init__.py pkg/b2/mod.py" in proc.stdout, \
            output(proc)
    ok = _branch(repo, "pr-single-ok", _impl("B1"))
    proc = run_tool("accept", "B1", "--check-pr", "--base", "main", "--head", ok, "--root", repo)
    assert proc.returncode == 0, output(proc)
    assert proc.stdout == "MODE implementation\n", proc.stdout


def test_batch_merge_is_l2_for_each_wp(repo, tmp_path):
    """The full path: check-pr per label, each WP's own Accept and ledger row,
    squash merge; then the merge is every WP's L2 commit (status board), the
    WPs count as built (a tests-only PR is a tests-correction), and a batch of
    corrections for up to 3 of them runs per WP."""
    labels = ["B1", "B2", "B3"]
    head = _branch(repo, "pr-batch", _impl(*labels))
    ledger = tmp_path / "results.json"
    git(repo, "checkout", "-q", head)
    for wp in labels:
        proc = _check(repo, head, labels, wp)
        assert proc.returncode == 0 and "MODE implementation" in proc.stdout, output(proc)
        proc = run_tool("accept", wp, "--root", repo)
        assert proc.returncode == 0 and f"PASS {wp}" in proc.stdout, output(proc)
        proc = run_tool("ci_ledger.py", "record", "--file", ledger, "--wp", wp, "--mode", "accept",
                        "--event", "pull_request", "--head-sha", git(repo, "rev-parse", "HEAD"),
                        "--conclusion", "success")
        assert proc.returncode == 0, output(proc)
    rows = json.loads(ledger.read_text())["runs"]
    assert [r["wp"] for r in rows] == labels
    git(repo, "checkout", "-q", "main")
    squash = _squash(repo, "pr-batch", "feat: B1 B2 B3 (#9)")

    # status board: the squash merge is each WP's L2 commit
    ci = tmp_path / "ci.json"
    ci.write_text(json.dumps({"runs": [
        {"workflow": "laned-accept", "event": "pull_request", "mode": "expect-fail", "wp": w,
         "head_sha": squash, "conclusion": "success"} for w in [*labels, "B4"]]}))
    out = tmp_path / "board"
    proc = run_tool("status", "--root", repo, "--ref", "main", "--ci-results", ci, "--out", out)
    assert proc.returncode == 0, output(proc)
    board = json.loads((out / "status" / "laned_build.json").read_text())["wps"]
    for wp in labels:
        assert board[wp]["level"] == "L2", (wp, board[wp])
    assert board["B4"]["level"] == "L1", board["B4"]

    # built_on sees the batch merge: a tests-only PR for B1 is a tests-correction
    more = {"tests/accept/b1/test_b1_more.py": "import pkg.b1\n\n\ndef test_more():\n    assert pkg.b1.VALUE\n"}
    h = _branch(repo, "pr-b1-tests", more)
    proc = run_tool("accept", "B1", "--check-pr", "--base", "main", "--head", h, "--root", repo)
    assert proc.returncode == 0 and proc.stdout == "MODE tests-correction\n", output(proc)

    # a batch of tests-corrections (B1, B2 built): BATCH + MODE tests-correction per WP
    corr = {**more, "tests/accept/b2/test_b2_more.py":
            "import pkg.b2.mod\n\n\ndef test_more():\n    assert pkg.b2.mod.VALUE == 2\n"}
    h = _branch(repo, "pr-corrections", corr)
    git(repo, "checkout", "-q", h)
    for wp in ("B1", "B2"):
        proc = _check(repo, h, ["B1", "B2"], wp)
        assert proc.returncode == 0, output(proc)
        lines = [ln for ln in proc.stdout.splitlines() if ln.startswith(("BATCH ", "MODE "))]
        assert lines == ["BATCH B1 B2", "MODE tests-correction"], proc.stdout
        proc = run_tool("accept", wp, "--tests-correction", "--base", "main", "--head", h, "--root", repo)
        assert proc.returncode == 0 and f"PASS {wp} tests-correction" in proc.stdout, output(proc)
    git(repo, "checkout", "-q", "main")

    # rev6.17: a built WP and an unbuilt one in one tests batch, each its own MODE
    mixed = {**more, "tests/accept/d1/test_d1_more.py": "def test_more():\n    assert False\n"}
    h = _branch(repo, "pr-mixed", mixed)
    for wp, mode in (("B1", "tests-correction"), ("D1", "tests-only")):
        proc = _check(repo, h, ["B1", "D1"], wp)
        assert proc.returncode == 0 and re.search(rf"^MODE {mode}$", proc.stdout, re.M), output(proc)

    # rev6.17: four built WPs in one corrections batch (no cap)
    _branch(repo, "pr-b4", _impl("B4"))
    _squash(repo, "pr-b4", "feat: B4 (#10)")
    four = {f"tests/accept/{w.lower()}/test_{w.lower()}_more.py": "def test_more():\n    assert True\n"
            for w in ("B1", "B2", "B3", "B4")}
    h = _branch(repo, "pr-four-corrections", four)
    proc = _check(repo, h, ["B1", "B2", "B3", "B4"])
    assert proc.returncode == 0 and re.search(r"^MODE tests-correction$", proc.stdout, re.M), output(proc)


def test_batch_impl_squash_with_foreign_file_is_not_l2(repo, tmp_path):
    """A merge touching two WPs' PATHS plus a file in no WP's PATHS is no L2
    commit for either (the pre-batch rule for a stray file is unchanged)."""
    _branch(repo, "stray", {**_impl("B1", "B2"), "README.md": "edited\n"})
    sha = _squash(repo, "stray", "stray")
    ci = tmp_path / "ci.json"
    ci.write_text(json.dumps({"runs": [
        {"workflow": "laned-accept", "event": "pull_request", "mode": "expect-fail", "wp": w,
         "head_sha": sha, "conclusion": "success"} for w in ("B1", "B2")]}))
    out = tmp_path / "board"
    proc = run_tool("status", "--root", repo, "--ref", "main", "--ci-results", ci, "--out", out)
    assert proc.returncode == 0, output(proc)
    board = json.loads((out / "status" / "laned_build.json").read_text())["wps"]
    assert board["B1"]["level"] == "L1" and board["B2"]["level"] == "L1", board
