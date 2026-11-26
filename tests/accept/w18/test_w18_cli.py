"""W18: the CLI, ``python -P -m cascade.laned.vault_sweeper``, over the fs
adapter (network refused, boto3/botocore unimportable). Contract: README.md."""

from __future__ import annotations

import json
from pathlib import Path

from .helpers import (
    CHAIN_TOML,
    DUE,
    EVIDENCE_SCHEMA,
    NOW,
    WP,
    evidence_problems,
    fs_seed,
    iso,
    ledger,
    log_problems,
    run_cli,
    scenario_entries,
    strict_loads,
    tree,
    write_plan,
)


def _setup(tmp_path: Path):
    entries = scenario_entries()
    vroot, aroot = tmp_path / "vault", tmp_path / "archive"
    fs_seed(vroot, entries)
    led = tmp_path / "ledger.json"
    led.write_text(json.dumps(ledger(entries)), encoding="utf-8")
    base = ["--vault-dir", str(vroot), "--archive-dir", str(aroot), "--ledger", str(led),
            "--config", str(CHAIN_TOML), "--log-dir", str(tmp_path / "logs"),
            "--now", iso(NOW)]
    return entries, vroot, aroot, base


def _out(proc):
    try:
        return strict_loads(proc.stdout)
    except ValueError as e:
        raise AssertionError(f"stdout is not one strict JSON object ({e}): {proc.stdout[-2000:]!r}"
                             f"\nstderr: {proc.stderr[-2000:]}") from None


def test_cli_dry_run_prints_the_log_and_changes_nothing(tmp_path):
    entries, vroot, aroot, base = _setup(tmp_path)
    before = tree(vroot)
    proc = run_cli(tmp_path, *base, "--dry-run")
    assert proc.returncode == 0, proc.stderr[-3000:]
    log = _out(proc)
    assert log_problems(log) == []
    assert log["mode"] == "dry-run"
    assert log["policy"]["loser_ttl_eras"] == 4, "read from --config [lane_d] loser_ttl_eras"
    assert tree(vroot) == before and not aroot.exists()
    files = list((tmp_path / "logs").iterdir())
    assert [p.name for p in files] == [f"vault-sweep-{log['run_id']}.json"]
    assert strict_loads(files[0].read_text(encoding="utf-8")) == log


def test_cli_needs_exactly_one_mode(tmp_path):
    _, vroot, _, base = _setup(tmp_path)
    before = tree(vroot)
    for extra in ([], ["--dry-run", "--live"]):
        proc = run_cli(tmp_path, *base, *extra)
        assert proc.returncode == 2, (extra, proc.returncode, proc.stderr[-1000:])
    assert tree(vroot) == before


def test_cli_live_refused_without_d16_decided(tmp_path):
    _, vroot, aroot, base = _setup(tmp_path)
    before = tree(vroot)
    plan = write_plan(tmp_path / "plan.md", "open")
    proc = run_cli(tmp_path, *base, "--live", "--plan", str(plan))
    assert proc.returncode == 2, (proc.returncode, proc.stdout[-1000:], proc.stderr[-1000:])
    assert _out(proc) == {"refused": "d16_not_decided"}
    assert tree(vroot) == before and not aroot.exists()


def test_cli_live_writes_evidence_the_verifier_reads(tmp_path):
    entries, vroot, aroot, base = _setup(tmp_path)
    plan = write_plan(tmp_path / "plan.md", "decided")
    ev_path = tmp_path / "w18.json"
    proc = run_cli(tmp_path, *base, "--live", "--plan", str(plan), "--evidence-out", str(ev_path))
    assert proc.returncode == 0, proc.stderr[-3000:]
    log = _out(proc)
    assert log["mode"] == "live" and log["d16"] == "decided"
    for tag in DUE:
        for rel in entries[tag].rels():
            assert not (vroot / rel).exists() and (aroot / rel).is_file()
    ev = strict_loads(ev_path.read_text(encoding="utf-8"))
    assert ev == {"schema": EVIDENCE_SCHEMA, "wp": WP, "passed": True, "log": log}
    problems = evidence_problems(ev)
    assert problems and any("not the live s3 vault" in x for x in problems), (
        "an fs-adapter run is CI evidence, never L4 owner evidence")


def test_cli_live_exit_1_when_an_action_failed(tmp_path):
    entries, vroot, aroot, base = _setup(tmp_path)
    aroot.mkdir(mode=0o700)
    (aroot / entries["lost_due"].hotkey).write_bytes(b"blocks the archive directory")
    plan = write_plan(tmp_path / "plan.md", "decided")
    ev_path = tmp_path / "w18.json"
    proc = run_cli(tmp_path, *base, "--live", "--plan", str(plan), "--evidence-out", str(ev_path))
    assert proc.returncode == 1, (proc.returncode, proc.stderr[-2000:])
    log = _out(proc)
    assert log["counts"]["failed"] >= 1
    assert strict_loads(ev_path.read_text(encoding="utf-8"))["passed"] is False
