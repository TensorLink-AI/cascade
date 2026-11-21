"""W0c (rev6, §17): scripts/laned/status on a fixture repo.

The fixture repo (helpers.build_status_repo) has real git history, the CI
results come from fixtures/status/ci_results.json (GitHub Actions is the only
mocked service) and the evidence from fixtures/status/evidence/. README.md
explains why each expected level holds.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from .helpers import (
    STATUS_FIXTURE,
    build_status_repo,
    git,
    git_bare,
    materialise_ci_results,
    output,
    run_tool,
    write,
)

EXPECTED = {k: v for k, v in json.loads((STATUS_FIXTURE / "expected_levels.json").read_text()).items()
            if not k.startswith("_")}

ALL_STAGES = ("S0", "S0a", "S0b", "S0c", "S1", "S2", "S2b", "S3", "S4", "S5", "S6",
              "SB0", "SB1", "SB2", "SB3", "SB4", "SB5", "S7", "S8")
MILESTONES = ("Build-complete", "Launch-ready", "Live", "Done")


def _run_status(tmp_path: Path, repo: Path, shas: dict[str, str], ref: str = "main",
                out_name: str = "out", extra: tuple[str, ...] = ()) -> tuple[dict, Path]:
    ci = materialise_ci_results(shas, tmp_path / "ci_results.json")
    out = tmp_path / out_name
    proc = run_tool("status", "--root", repo, "--ref", ref, "--ci-results", ci,
                    "--out", out, "--release-tag-glob", "cap-*", *extra)
    assert proc.returncode == 0, output(proc)
    board = out / "status" / "laned_build.json"
    assert board.is_file(), f"status did not write {board}"
    assert (out / "docs" / "plans" / "STATUS.md").is_file(), "status did not write STATUS.md"
    return json.loads(board.read_text()), out


@pytest.fixture
def fixture_repo(tmp_path):
    return build_status_repo(tmp_path)


def test_status_exact_level_for_every_fixture_wp(tmp_path, fixture_repo):
    repo, shas = fixture_repo
    data, _ = _run_status(tmp_path, repo, shas)
    assert data["schema"] == "laned_build/1"
    assert data["ref"] == "main"
    assert data["commit"] == shas["docs"]
    assert set(data["wps"]) == set(EXPECTED)
    got = {wp: {"level": v["level"], "side_state": v["side_state"]} for wp, v in data["wps"].items()}
    assert got == EXPECTED


def test_status_regression_l3_to_l2_after_red_main_run(tmp_path, fixture_repo):
    repo, shas = fixture_repo
    # At the evidence commit the red run (on the later 'docs' commit) is not in
    # history yet: W1 is accepted.
    before, _ = _run_status(tmp_path, repo, shas, ref=shas["evidence"], out_name="before")
    assert before["wps"]["W1"]["level"] == "L3"
    after, _ = _run_status(tmp_path, repo, shas, out_name="after")
    assert after["wps"]["W1"]["level"] == "L2"


def test_status_reads_ref_tree_not_working_tree(tmp_path, fixture_repo):
    repo, shas = fixture_repo
    # Uncommitted evidence that would raise R2x to L4 and W0 to L4 is ignored.
    write(repo, "evidence/r2x.json", json.dumps({"wp": "R2x", "passed": True}))
    write(repo, "evidence/w0.json", json.dumps({"wp": "W0", "passed": True}))
    data, _ = _run_status(tmp_path, repo, shas)
    assert data["wps"]["R2x"]["level"] == "L3"
    assert data["wps"]["W0"]["level"] == "L3"


def test_status_decision_register_and_owner_inputs(tmp_path, fixture_repo):
    repo, shas = fixture_repo
    data, _ = _run_status(tmp_path, repo, shas)
    d = data["decisions"]
    assert d["D1"]["status"] == "decided"
    assert d["D1"]["class"] == "O"
    assert d["D1"]["blocking"] is False
    assert d["D99"]["status"] == "open"
    assert d["D99"]["class"] == "S"
    assert d["D99"]["blocking"] is True
    oa = data["owner_inputs"]
    assert oa["OA8"]["evidence"] == "evidence/owner/oa8.json"
    assert oa["OA8"]["present"] is True
    assert oa["OA9"]["evidence"] == "evidence/owner/oa9.json"
    assert oa["OA9"]["present"] is False


def test_status_stages_and_milestones(tmp_path, fixture_repo):
    repo, shas = fixture_repo
    data, _ = _run_status(tmp_path, repo, shas)
    assert set(ALL_STAGES) <= set(data["stages"])
    for stage in ALL_STAGES:
        assert data["stages"][stage]["state"] in ("complete", "incomplete")
    # D99 is open, so S0 is not complete even though W0 is at L3 and the gate
    # evidence passes.
    assert data["stages"]["S0"]["state"] == "incomplete"
    assert data["stages"]["S3"]["state"] == "incomplete"
    assert set(data["milestones"]) == set(MILESTONES)
    assert all(v == "incomplete" for v in data["milestones"].values())


def test_status_s0_complete_when_register_closed(tmp_path):
    repo, shas = build_status_repo(tmp_path, d99_open=False)
    data, _ = _run_status(tmp_path, repo, shas)
    assert data["decisions"]["D99"]["status"] == "decided"
    assert data["decisions"]["D99"]["blocking"] is False
    assert data["wps"]["W6"]["side_state"] is None
    assert data["wps"]["W0"]["level"] == "L3"
    assert data["stages"]["S0"]["state"] == "complete"


def test_status_coverage_counters(tmp_path, fixture_repo):
    repo, shas = fixture_repo
    data, _ = _run_status(tmp_path, repo, shas)
    c = data["counters"]
    assert c["t_rows"] == {"passed": 1, "total": 3}
    # test_inv10_* (W0c, L7) and test_inv9_* (W0, L3) count; test_inv1_* is
    # in W1, which is only at L2.
    assert c["invariants"] == {"covered": [9, 10], "total": 12}
    assert c["orphans"] == ["T3"]
    blocked = {row["reason"]: row for row in c["blocked"]}
    assert set(blocked) == {"owner", "dependency", "D99", "external"}
    assert blocked["owner"]["wps"] == ["W2"]
    assert blocked["dependency"]["wps"] == ["W5"]
    assert blocked["D99"]["wps"] == ["W6"]
    assert blocked["external"]["wps"] == ["W7"]
    assert all(row["count"] == len(row["wps"]) for row in c["blocked"])


def test_status_md_content(tmp_path, fixture_repo):
    repo, shas = fixture_repo
    _, out = _run_status(tmp_path, repo, shas)
    md = (out / "docs" / "plans" / "STATUS.md").read_text()
    assert re.search(r"^\|\s*WP\s*\|\s*Level\s*\|\s*State\s*\|", md, re.M), md
    for wp, exp in EXPECTED.items():
        state = exp["side_state"] or "-"
        row = rf"^\|\s*{re.escape(wp)}\s*\|\s*{exp['level']}\s*\|\s*{re.escape(state)}\s*\|"
        assert re.search(row, md, re.M), f"no STATUS.md row for {wp} {exp}"
    assert re.search(r"T-rows.*\b1/3\b", md), "STATUS.md lacks the T-rows counter"


def test_status_push_to_build_state_branch_only(tmp_path, fixture_repo):
    repo, shas = fixture_repo
    bare = tmp_path / "remote.git"
    git(tmp_path, "clone", "-q", "--bare", str(repo), str(bare))
    remote_main = git_bare(bare, "rev-parse", "main").stdout.strip()

    # Without --push nothing reaches the remote.
    _run_status(tmp_path, repo, shas, out_name="local")
    assert git_bare(bare, "rev-parse", "--verify", "-q", "laned/build-state").returncode != 0

    data, out = _run_status(tmp_path, repo, shas, out_name="pushed", extra=("--push", str(bare)))
    shown = git_bare(bare, "show", "laned/build-state:status/laned_build.json")
    assert shown.returncode == 0, shown.stderr
    assert json.loads(shown.stdout) == data
    md = git_bare(bare, "show", "laned/build-state:docs/plans/STATUS.md")
    assert md.returncode == 0, md.stderr
    assert md.stdout == (out / "docs" / "plans" / "STATUS.md").read_text()

    # main is never touched, locally or on the remote.
    assert git_bare(bare, "rev-parse", "main").stdout.strip() == remote_main
    assert git(repo, "rev-parse", "main") == shas["docs"]

    # A second publish succeeds without force.
    _run_status(tmp_path, repo, shas, out_name="pushed2", extra=("--push", str(bare)))
    assert git_bare(bare, "rev-parse", "--verify", "-q", "laned/build-state").returncode == 0
