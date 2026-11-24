"""W0c (owner 2026-11-24): the GitHub merge queue is the freshness gate.

- `accept --freshness --warn-only` (the pull_request job): a stale base is a
  WARNING and exits 0; without `--warn-only` it still fails.
- `accept --merge-group` (the merge_group job): every queued PR's OWN WP runs
  its Accept on the combined ref, so a PR green on an older base cannot land
  broken; it fails closed on unknown PRs.
- the workflow wires both.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from .helpers import REPO, commit, git, init_repo, output, run_tool, write

WORKFLOW = REPO / ".github" / "workflows" / "laned-accept.yml"
# fixture modules change within one second at equal size: no stale .pyc
NO_PYC = {"PYTHONDONTWRITEBYTECODE": "1"}

W3_TEST = ("import pkg_lib\nimport pkg_w3\n\n\n"
           "def test_w3():\n    assert pkg_w3.W3 == pkg_lib.LIB\n")


def _seed(tmp_path: Path) -> tuple[Path, str]:
    repo = init_repo(tmp_path / "repo")
    files = {"README.md": "x\n", ".gitignore": "__pycache__/\n", "pkg_lib.py": "LIB = 1\n",
             "pkg_w3.py": "W3 = 0\n", "tests/accept/w3/PATHS": "pkg_w3.py\n",
             "tests/accept/w3/test_w3.py": W3_TEST}
    for rel, text in files.items():
        write(repo, rel, text)
    return repo, commit(repo, "seed", list(files))


def _on(repo: Path, start: str, prs: list[tuple[dict[str, str], str]]) -> str:
    git(repo, "checkout", "-q", "--detach", start)
    head = start
    for files, msg in prs:
        for rel, text in files.items():
            write(repo, rel, text)
        head = commit(repo, msg, list(files))
    return head


def _labels(tmp_path: Path, labels: dict[int, list[str]]) -> Path:
    f = tmp_path / "group-labels.json"
    f.write_text(json.dumps({str(k): v for k, v in labels.items()}))
    return f


def test_freshness_warn_only_reports_a_stale_base_without_failing(tmp_path):
    repo, base0 = _seed(tmp_path)
    # main moved: pkg_lib changed, which W3's tests import (the #593 shape)
    main1 = _on(repo, base0, [({"pkg_lib.py": "LIB = 22\n"}, "lib (#1)")])
    pr = _on(repo, base0, [({"pkg_w3.py": "W3 = 1\n"}, "w3 impl")])
    warned = run_tool("accept", "--freshness", "--base", base0, "--head", pr, "--main", main1,
                      "--warn-only", "--root", repo)
    assert warned.returncode == 0, output(warned)
    assert "WARNING STALE BASE" in warned.stdout, output(warned)
    assert "STALE BASE: update-branch and re-run" not in warned.stdout, output(warned)
    strict = run_tool("accept", "--freshness", "--base", base0, "--head", pr, "--main", main1,
                      "--root", repo)
    assert strict.returncode != 0, output(strict)
    assert "STALE BASE: update-branch and re-run" in strict.stdout, output(strict)
    fresh = run_tool("accept", "--freshness", "--base", base0, "--head", pr, "--main", base0,
                     "--warn-only", "--root", repo)
    assert fresh.returncode == 0 and "FRESH base" in fresh.stdout, output(fresh)
    assert "WARNING STALE BASE" not in fresh.stdout, output(fresh)


def test_merge_group_runs_each_queued_prs_own_wp_on_the_combined_ref(tmp_path):
    repo, base0 = _seed(tmp_path)
    impl = ({"pkg_w3.py": "W3 = 1\n"}, "w3 impl (#7)")
    alone = _on(repo, base0, [impl])
    ok = run_tool("accept", "--merge-group", "--base", base0, "--head", alone,
                  "--group-labels", _labels(tmp_path, {7: ["laned:W3"]}), "--root", repo, env=NO_PYC)
    assert ok.returncode == 0, output(ok)
    assert "GROUP-OWN W3" in ok.stdout, output(ok)
    assert "GROUP W3 accept PASS (#7)" in ok.stdout, output(ok)
    # a PR queued ahead (or merged to main since W3's PR base) breaks W3:
    # the CROSS excludes W3, so only its own run on the combined ref sees it
    broken = _on(repo, base0, [({"pkg_lib.py": "LIB = 22\n"}, "lib bump (#6)"), impl])
    bad = run_tool("accept", "--merge-group", "--base", base0, "--head", broken,
                   "--group-labels", _labels(tmp_path, {6: [], 7: ["laned:W3"]}), "--root", repo, env=NO_PYC)
    assert bad.returncode != 0, output(bad)
    assert "GROUP W3 accept FAIL (#7)" in bad.stdout, output(bad)
    assert "GROUP FAIL" in bad.stdout, output(bad)


def test_merge_group_fails_closed_without_labels_or_pr_number(tmp_path):
    repo, base0 = _seed(tmp_path)
    alone = _on(repo, base0, [({"pkg_w3.py": "W3 = 1\n"}, "w3 impl (#7)")])
    unknown = run_tool("accept", "--merge-group", "--base", base0, "--head", alone,
                       "--group-labels", _labels(tmp_path, {}), "--root", repo, env=NO_PYC)
    assert unknown.returncode != 0, output(unknown)
    assert "no labels for #7" in unknown.stdout, output(unknown)
    direct = _on(repo, base0, [({"pkg_w3.py": "W3 = 1\n"}, "direct push")])
    nopr = run_tool("accept", "--merge-group", "--base", base0, "--head", direct,
                    "--group-labels", _labels(tmp_path, {}), "--root", repo, env=NO_PYC)
    assert nopr.returncode != 0, output(nopr)
    assert "names no PR" in nopr.stdout, output(nopr)


def _steps() -> dict[str, dict]:
    doc = yaml.safe_load(WORKFLOW.read_text())
    steps = doc["jobs"]["pr"]["steps"]
    return {s.get("name", ""): s for s in steps}


def test_workflow_pr_freshness_is_a_warning_and_merge_group_runs_own_wps():
    steps = _steps()
    fresh = steps.get("freshness")
    assert fresh, "the pr job has no freshness step"
    run = fresh["run"]
    assert "--warn-only" in run, "freshness must run with --warn-only on pull_request"
    assert run.rstrip().endswith("exit 0"), "the freshness step must never fail the PR"
    assert "::warning::" in run and "GITHUB_STEP_SUMMARY" in run
    group = [s for n, s in steps.items() if "erge-group" in n]
    assert group, "the pr job has no merge-group step"
    text = "\n".join(s["run"] for s in group)
    assert all(s.get("if") == "github.event_name == 'merge_group'" for s in group)
    assert "--merge-group" in text and "--group-labels" in text, "merge_group must run each queued WP"
    assert "--cross" in text and "GROUP-OWN" in text, "merge_group must still run the CROSS"
