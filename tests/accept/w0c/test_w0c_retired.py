"""W0c (owner decision D24, plan rev6.18, 2026-11-24): RETIRED WPs.

W5i (slow-cadence backtest windows) was dropped by D24 and is never built,
but tests/accept/w5i/ (PATHS over lane_d.py, lane_d_inputs.py, ...) made
every PR touching those files carry laned:W5i and fail W5i's Accept. The
RETIRED rule (accept --help) lists such WPs in scripts/laned/retired_wps,
read at BASE: their PATHS never count for scope, their label is inert (MODE
retired), their tests dir may only be deleted (laned:tests-correction), they
are never a CROSS or merge-group run, and the status board shows RETIRED.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from .helpers import LANED, REPO, commit, git, init_repo, output, run_tool, write
from .test_w0c_status import _run_status, fixture_repo  # noqa: F401  (pytest fixture)

WORKFLOW = REPO / ".github" / "workflows" / "laned-accept.yml"
NO_PYC = {"PYTHONDONTWRITEBYTECODE": "1"}
RETIRED_FILE = "scripts/laned/retired_wps"
R_TEST = "import pkg_r\n\n\ndef test_r():\n    assert pkg_r.R == 1\n"


def _repo(tmp_path: Path, retire: str | None = "R  owner decision D1 (fixture)\n") -> tuple[Path, str]:
    """A: live WP over pkg_a.py + shared.py. R: built (git-only impl commit)
    over pkg_r.py + shared.py, then retired (when ``retire``). Returns BASE."""
    repo = init_repo(tmp_path / "repo")
    files = {"README.md": "x\n", ".gitignore": "__pycache__/\n", "pkg_a.py": "A = 0\n",
             "shared.py": "S = 1\n",
             "tests/accept/a/PATHS": "pkg_a.py\nshared.py\n",
             "tests/accept/a/test_a.py": "import pkg_a\n\n\ndef test_a():\n    assert pkg_a.A == 1\n",
             "tests/accept/r/PATHS": "pkg_r.py\nshared.py\n",
             "tests/accept/r/test_r.py": R_TEST}
    for rel, text in files.items():
        write(repo, rel, text)
    commit(repo, "tests", list(files))
    write(repo, "pkg_a.py", "A = 1\n")
    commit(repo, "a impl", ["pkg_a.py"])
    write(repo, "pkg_r.py", "R = 1\n")
    base = commit(repo, "r impl", ["pkg_r.py"])
    if retire is not None:
        write(repo, RETIRED_FILE, "# retired WPs\n" + retire)
        base = commit(repo, "retire R", [RETIRED_FILE])
    return repo, base


def _pr(repo: Path, base: str, files: dict[str, str | None], msg: str = "pr") -> str:
    git(repo, "checkout", "-q", "--detach", base)
    for rel, text in files.items():
        if text is None:
            git(repo, "rm", "-q", "--", rel)
        else:
            write(repo, rel, text)
            git(repo, "add", "--", rel)
    git(repo, "commit", "-q", "-m", msg)
    return git(repo, "rev-parse", "HEAD")


def _check(repo: Path, wp: str, base: str, head: str, labels: str, *extra: str):
    return run_tool("accept", wp, "--check-pr", "--base", base, "--head", head,
                    "--pr-wps", labels, *extra, "--root", repo)


def _mode(proc) -> str | None:
    return next((ln[5:] for ln in proc.stdout.splitlines() if ln.startswith("MODE ")), None)


def test_retired_wps_file_lists_w5i_with_the_d24_reference():
    f = LANED / "retired_wps"
    assert f.is_file(), "scripts/laned/retired_wps missing"
    rows = [ln.split("#", 1)[0].strip() for ln in f.read_text().splitlines()]
    rows = [r for r in rows if r]
    by_wp = {r.split(None, 1)[0]: r.split(None, 1)[1] if len(r.split(None, 1)) > 1 else "" for r in rows}
    assert "W5i" in by_wp, rows
    assert "D24" in by_wp["W5i"] and "rev6.18" in by_wp["W5i"], by_wp["W5i"]
    plan = (REPO / "docs" / "plans" / "LANE_D_PLAN.md").read_text()
    assert "| W5i | **RETIRED" in plan, "the plan's W5i row is not RETIRED"


def test_retired_paths_never_need_the_label(tmp_path):
    repo, base = _repo(tmp_path)
    # shared.py is in A's and R's PATHS: laned:A alone is enough (the #871 shape)
    head = _pr(repo, base, {"shared.py": "S = 2\n"})
    alone = _check(repo, "A", base, head, "A")
    assert alone.returncode == 0 and _mode(alone) == "implementation", output(alone)
    # a leftover laned:R label is inert: A still passes, R prints MODE retired
    both_a = _check(repo, "A", base, head, "A,R")
    assert both_a.returncode == 0 and _mode(both_a) == "implementation", output(both_a)
    both_r = _check(repo, "R", base, head, "A,R")
    assert both_r.returncode == 0 and _mode(both_r) == "retired", output(both_r)
    assert "RETIRED R owner decision D1 (fixture)" in both_r.stdout, output(both_r)


def test_retired_paths_never_give_scope(tmp_path):
    repo, base = _repo(tmp_path)
    # pkg_r.py is only in R's PATHS: no live label covers it
    head = _pr(repo, base, {"pkg_r.py": "R = 3\n", "pkg_a.py": "A = 1  # touched\n"})
    a = _check(repo, "A", base, head, "A,R")
    assert a.returncode != 0 and "FAIL paths" in a.stdout and "pkg_r.py" in a.stdout, output(a)
    # a PR labelled only with a retired WP may not change code at all
    only = _check(repo, "R", base, _pr(repo, base, {"pkg_r.py": "R = 3\n"}), "R")
    assert only.returncode != 0 and "retired-only" in only.stdout, output(only)


def test_retired_tests_dir_may_only_be_deleted_with_tests_correction_read_at_base(tmp_path):
    repo, base = _repo(tmp_path)
    gone = _pr(repo, base, {"tests/accept/r/PATHS": None, "tests/accept/r/test_r.py": None})
    ok = _check(repo, "R", base, gone, "R", "--correction")
    assert ok.returncode == 0 and _mode(ok) == "retired", output(ok)
    weak = _check(repo, "R", base, gone, "R")
    assert weak.returncode != 0 and "FAIL weakened" in weak.stdout, output(weak)
    # deleting it under another label only: unlabelled
    unl = _check(repo, "A", base, _pr(repo, base, {"tests/accept/r/test_r.py": None,
                                                    "pkg_a.py": "A = 1  # x\n"}), "A")
    assert unl.returncode != 0 and "unlabelled-tests" in unl.stdout, output(unl)
    # editing a retired WP's tests is refused even with tests-correction
    edit = _pr(repo, base, {"tests/accept/r/test_r.py": R_TEST + "\n\ndef test_more():\n    assert True\n"})
    bad = _check(repo, "R", base, edit, "R", "--correction")
    assert bad.returncode != 0 and "retired-tests" in bad.stdout, output(bad)

    # retirement is read at BASE, never from the PR: a PR cannot retire R itself
    repo, base = _repo(tmp_path / "self", retire=None)
    head = _pr(repo, base, {RETIRED_FILE: "R  self-declared\n", "tests/accept/r/PATHS": None,
                            "tests/accept/r/test_r.py": None})
    r = _check(repo, "R", base, head, "R", "--correction")
    assert _mode(r) != "retired" and "RETIRED R" not in r.stdout, output(r)


def test_retired_wps_line_without_a_reference_fails_closed(tmp_path):
    repo, base = _repo(tmp_path, retire="R\n")
    head = _pr(repo, base, {"shared.py": "S = 2\n"})
    proc = _check(repo, "A", base, head, "A")
    assert proc.returncode != 0 and "FAIL retired" in proc.stdout, output(proc)
    assert "no owner decision reference" in proc.stdout, output(proc)


def test_retired_wp_is_never_a_cross_wp(tmp_path):
    # sanity: R is built, so before retirement a shared.py change selects it
    live_repo, live_base = _repo(tmp_path / "live", retire=None)
    live = run_tool("accept", "A", "--cross", "--base", live_base, "--head",
                    _pr(live_repo, live_base, {"shared.py": "S = 2\n"}), "--root", live_repo, env=NO_PYC)
    assert "cross r:" in live.stdout.lower(), output(live)
    repo, base = _repo(tmp_path / "retired")
    proc = run_tool("accept", "A", "--cross", "--base", base, "--head",
                    _pr(repo, base, {"shared.py": "S = 2\n"}), "--root", repo, env=NO_PYC)
    assert proc.returncode == 0, output(proc)
    assert "cross r:" not in proc.stdout.lower() and "CROSS r " not in proc.stdout \
        and "CROSS R " not in proc.stdout, output(proc)


def test_merge_group_skips_a_retired_label(tmp_path):
    repo, base = _repo(tmp_path)
    head = _pr(repo, base, {"shared.py": "S = 2\n"}, "shared (#5)")
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({"5": ["laned:A", "laned:R"]}))
    proc = run_tool("accept", "--merge-group", "--base", base, "--head", head,
                    "--group-labels", labels, "--root", repo, env=NO_PYC)
    assert proc.returncode == 0, output(proc)
    assert "GROUP R retired SKIP (#5)" in proc.stdout, output(proc)
    assert "GROUP A accept PASS (#5)" in proc.stdout and "GROUP PASS" in proc.stdout, output(proc)


def test_status_board_shows_retired(tmp_path, fixture_repo):  # noqa: F811
    repo, shas = fixture_repo
    # W1 is L2 and W2 L1 (blocked:owner) on the fixture; W5 depends on W2
    write(repo, RETIRED_FILE, "W1  owner decision D1 (fixture)\nW2  owner decision D2 (fixture)\n")
    shas["docs"] = commit(repo, "retire W1 W2", [RETIRED_FILE])
    data, out = _run_status(tmp_path, repo, shas)
    for wp, ref in (("W1", "owner decision D1 (fixture)"), ("W2", "owner decision D2 (fixture)")):
        v = data["wps"][wp]
        assert v["level"] == "L0" and v["side_state"] == "retired" and v["retired"] == ref, v
    # a retired dependency blocks nobody
    assert data["wps"]["W5"]["side_state"] != "blocked:dependency", data["wps"]["W5"]
    md = (out / "docs" / "plans" / "STATUS.md").read_text()
    assert "| W1 | RETIRED | retired: owner decision D1 (fixture) |" in md, md
    # and counts toward no stage
    for c in data["stages"]["S2"]["requirements"]:
        below = c["check"].split("(below: ", 1)[1].rstrip(")").split(", ") if "(below: " in c["check"] else []
        assert "W1" not in below and "W2" not in below, c


def test_workflow_skips_a_retired_mode():
    wf = yaml.safe_load(WORKFLOW.read_text())
    steps = wf["jobs"]["pr"]["steps"]
    step = next(s for s in steps if s.get("name", "").startswith("Accept (check-pr"))
    run = step["run"]
    assert '"$mode" = "retired"' in run, "the PR job has no MODE retired branch"
    branch = run.split('"$mode" = "retired"', 1)[1].split("fi\n", 1)[0]
    assert "SKIP" in branch and "continue" in branch and "ci_ledger" not in branch, branch
