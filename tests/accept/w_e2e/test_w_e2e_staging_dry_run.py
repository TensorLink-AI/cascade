"""W-E2E staging tier (retro P1b): the testnet runner is runnable and
dry-runnable, and its guards keep it on testnet with one cheap Lium GPU.

The dry run parses every command it would run with the real `cascade` CLI
parser, so a renamed or removed flag fails in CI instead of on testnet. No
network, chain, pod or key is touched."""

from __future__ import annotations

import json

import pytest

from tests.e2e_lanes.harness import box_guard
from tests.e2e_testnet import lanes as staging


@pytest.mark.parametrize("lane", staging.LANES)
def test_dry_run_plans_every_stage_and_the_cli_accepts_every_command(lane, tmp_path):
    report = tmp_path / f"{lane}.json"
    with box_guard():                               # the dry run is torch-free
        rc = staging.main(["--lane", lane, "--report", str(report)])
    doc = json.loads(report.read_text(encoding="utf-8"))
    assert doc["mode"] == "dry-run"
    stages = {c["stage"] for c in doc["checks"]}
    assert set(staging.STAGES) <= stages, stages
    cli = [c for c in doc["checks"] if c["name"].startswith("cli_")]
    assert cli and all(c["status"] == "PASS" for c in cli), cli
    paid = [p for p in doc["plan"] if p["paid"]]
    assert paid, "the plan names no paid step"
    assert not any(c["status"] == "PASS" and c["stage"] == "submit" for c in doc["checks"]), \
        "a dry run executed a submit step"
    assert rc == 0, [c for c in doc["checks"] if c["status"] == "FAIL"]


def _guards(*extra):
    return staging.guard_errors(staging.build_argparser().parse_args(["--lane", "b", *extra]))


def test_guards_keep_the_run_on_testnet_with_one_cheap_lium_gpu(tmp_path):
    assert _guards() == []
    assert any("testnet only" in e for e in _guards("--network", "finney"))
    main_toml = staging.ROOT / "chain.toml"
    assert any("not the testnet subnet" in e for e in _guards("--chain-toml", str(main_toml)))
    assert any("one cheap single-GPU" in e for e in _guards("--gpu", "8xH200"))
    assert any("--max-price" in e for e in _guards("--max-price", "0"))
    assert any("--max-price" in e for e in _guards("--max-price", "30"))


def test_a_spend_run_stops_at_a_failed_preflight(tmp_path, monkeypatch):
    """--spend on a mainnet config never reaches a paid stage."""
    ran = []
    monkeypatch.setattr(staging, "_run_cli", lambda argv, timeout: ran.append(argv) or (0, {}, ""))
    report = tmp_path / "r.json"
    rc = staging.main(["--lane", "c", "--spend", "--chain-toml", str(staging.ROOT / "chain.toml"),
                       "--report", str(report)])
    doc = json.loads(report.read_text(encoding="utf-8"))
    assert rc == 1
    assert ran == []
    assert any(c["name"] == "blocked" for c in doc["checks"])
