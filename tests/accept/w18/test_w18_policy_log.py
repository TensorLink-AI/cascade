"""W18: the sweep policy (the [lane_d] TTL key), the D16 register reader and
the strict JSON action log written once per run. Contract: README.md."""

from __future__ import annotations

import dataclasses
import json

from .helpers import (
    ARCHIVE_BUCKET,
    ARCHIVE_PREFIX,
    CHAIN_TOML,
    ERA,
    LOG_SCHEMA,
    PLAN,
    VAULT_BUCKET,
    VAULT_PREFIX,
    ledger,
    log_problems,
    run_sweep,
    s3_stores,
    scenario_entries,
    seeded_fake,
    strict_loads,
    vs,
    write_plan,
)


def test_policy_reads_loser_ttl_eras_from_the_loaded_chain_config():
    from cascade.shared.config import load_chain_config
    m = vs()
    cfg = load_chain_config(CHAIN_TOML)
    pol = m.policy_from_config(cfg)
    assert pol.loser_ttl_eras == cfg.lane_d.loser_ttl_eras == 4
    assert pol.archive_retention_days == m.ARCHIVE_RETENTION_DAYS == 14, "D8: archive 14 days"
    mutated = dataclasses.replace(cfg, lane_d=dataclasses.replace(cfg.lane_d, loser_ttl_eras=7))
    assert m.policy_from_config(mutated).loser_ttl_eras == 7, (
        "the TTL must come from [lane_d] loser_ttl_eras (overlay-settable), not a constant")


def test_d16_status_reads_the_decision_register_on_main():
    assert vs().d16_status(PLAN) == "decided", (
        "plan §17.4 register: D16 is decided (owner, 2026-11-21)")


def test_d16_status_reads_only_the_register_row(tmp_path):
    m = vs()
    assert m.d16_status(write_plan(tmp_path / "a.md", "open")) == "open"
    assert m.d16_status(write_plan(tmp_path / "b.md", "decided")) == "decided"
    assert m.d16_status(write_plan(tmp_path / "c.md", None)) is None, (
        "no D16 row in the register: the §6 recommendation table is not a decision")
    assert m.d16_status(write_plan(tmp_path / "d.md", None, register=False)) is None
    assert m.d16_status(tmp_path / "missing.md") is None


def test_action_log_is_strict_json_written_once_per_run(tmp_path):
    entries = scenario_entries()
    fake = seeded_fake(entries)
    vault, archive = s3_stores(fake)
    logs = tmp_path / "logs"
    log1 = run_sweep(vault, archive, ledger(entries), tmp_path, live=False, plan=None,
                     log_dir=logs)
    log2 = run_sweep(vault, archive, ledger(entries), tmp_path, log_dir=logs)
    assert log1["run_id"] != log2["run_id"]
    files = sorted(p.name for p in logs.iterdir())
    assert files == sorted(f"vault-sweep-{x['run_id']}.json" for x in (log1, log2)), files
    for log in (log1, log2):
        text = (logs / f"vault-sweep-{log['run_id']}.json").read_text(encoding="utf-8")
        parsed = strict_loads(text)
        assert parsed == log, "the file is the returned log"
        assert log_problems(parsed) == [], log_problems(parsed)
        assert parsed["schema"] == LOG_SCHEMA
        assert parsed["ledger_era"] == ERA
        assert parsed["vault"] == {"kind": "s3", "bucket": VAULT_BUCKET, "prefix": VAULT_PREFIX}
        assert parsed["archive"] == {"kind": "s3", "bucket": ARCHIVE_BUCKET,
                                     "prefix": ARCHIVE_PREFIX}
    assert log1["mode"] == "dry-run" and log1["d16"] is None
    assert log2["mode"] == "live" and log2["d16"] == "decided"


def test_action_log_is_complete_one_record_per_decision(tmp_path):
    entries = scenario_entries()
    fake = seeded_fake(entries)
    vault, archive = s3_stores(fake)
    log = run_sweep(vault, archive, ledger(entries), tmp_path)
    vault_keys = {a["key"] for a in log["actions"] if a["store"] == "vault"}
    want = {k.removeprefix(VAULT_PREFIX + "/") for k in fake_initial_keys(entries)}
    assert vault_keys == want, (
        f"every vault object gets at least one record: missing {sorted(want - vault_keys)}, "
        f"extra {sorted(vault_keys - want)}")
    json.dumps(log, allow_nan=False)


def fake_initial_keys(entries):
    from .helpers import all_vault_objects
    return {f"{VAULT_PREFIX}/{k}" for k in all_vault_objects(entries)}
