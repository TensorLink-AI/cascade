"""W0c (rev6, §17.2): `scripts/laned/accept --gate <stage>` verifies a stage's
exit-gate evidence at evidence/gates/<stage lowercased>.json."""

from __future__ import annotations

import json

import pytest

from .helpers import STATUS_FIXTURE, commit, copy_fixture, init_repo, output, run_tool, write


@pytest.fixture
def gate_repo(tmp_path):
    repo = init_repo(tmp_path / "repo")
    copy_fixture(STATUS_FIXTURE / "evidence/gates/s0.json", repo, "evidence/gates/s0.json")
    copy_fixture(STATUS_FIXTURE / "evidence/gates/s0b.json", repo, "evidence/gates/s0b.json")
    write(repo, "evidence/gates/s2.json", "{not json")
    write(repo, "evidence/gates/s4.json", json.dumps(
        {"stage": "S3", "passed": True, "checks": [{"name": "t_rows", "passed": True}]}))
    write(repo, "evidence/gates/s5.json", json.dumps({"stage": "S5", "passed": True, "checks": []}))
    commit(repo, "gates", ["evidence"])
    return repo


def test_gate_passes_on_valid_evidence(gate_repo):
    proc = run_tool("accept", "--gate", "S0", "--root", gate_repo)
    assert proc.returncode == 0, output(proc)


@pytest.mark.parametrize("stage", [
    "S0b",  # a check recorded passed: false (top-level passed: true is not enough)
    "S1",   # no evidence file
    "S2",   # not JSON
    "S4",   # evidence names a different stage
    "S5",   # no checks recorded
])
def test_gate_fails_on_bad_or_missing_evidence(gate_repo, stage):
    proc = run_tool("accept", "--gate", stage, "--root", gate_repo)
    assert proc.returncode != 0, output(proc)
