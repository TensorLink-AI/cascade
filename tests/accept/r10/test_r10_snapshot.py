"""R10 snapshot: a timestamped bundle with a sha256 manifest, allowlist only."""

from __future__ import annotations

import hashlib
import re

from .helpers import (
    MAINNET_TOML,
    SOURCE_BLOCK,
    SOURCE_NAMES,
    build_source,
    expected_source_names,
    manifest,
    refused,
    run_cli,
    sha256_file,
    snapshot,
    tree_digest,
)


def test_snapshot_writes_timestamped_bundle_with_manifest(tmp_path):
    src = build_source(tmp_path)
    out = tmp_path / "bundles"
    bundle, printed_sha = snapshot(src, out)
    assert bundle.is_dir() and bundle.parent == out.resolve()
    assert re.fullmatch(r"state-\d{8}T\d{6}Z", bundle.name), bundle.name
    m = manifest(bundle)
    assert m["schema"] == "laned_state_bundle/1"
    assert re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", m["created_at"]), m["created_at"]
    assert m["source"]["netuid"] == 91
    assert m["source"]["network_id"] == "bittensor:finney"
    assert m["source"]["block"] == SOURCE_BLOCK
    assert printed_sha == hashlib.sha256((bundle / "manifest.json").read_bytes()).hexdigest()


def test_snapshot_manifest_lists_every_state_file_with_its_sha256(tmp_path):
    src = build_source(tmp_path)
    bundle, _ = snapshot(src, tmp_path / "bundles")
    entries = manifest(bundle)["files"]
    assert {e["source_name"] for e in entries} == expected_source_names()
    for e in entries:
        copied = bundle / e["path"]
        assert copied.is_file(), e
        assert e["sha256"] == sha256_file(copied) == sha256_file(src / e["source_name"]), e
        assert e["bytes"] == copied.stat().st_size, e
    roles = {e["source_name"]: e["role"] for e in entries}
    for role, name in SOURCE_NAMES.items():
        assert roles[name] == role
    assert roles[SOURCE_NAMES["validator_state"] + ".bak"] == "validator_state"
    assert roles["activation_forfeit_state.json"] == "activation"


def test_snapshot_never_bundles_keys_env_configs_or_logs(tmp_path):
    src = build_source(tmp_path)
    bundle, _ = snapshot(src, tmp_path / "bundles")
    names = {p.name for p in bundle.rglob("*") if p.is_file()}
    for forbidden in ("coldkey", "default", ".env", "chain.toml", "validator.log",
                      "submissions.db", "validator_state.db.settlement.json"):
        assert forbidden not in names, f"{forbidden} was bundled"
    assert not any("wallets" in p.parts or "hotkeys" in p.parts for p in bundle.rglob("*"))


def test_snapshot_never_writes_to_the_source(tmp_path):
    src = build_source(tmp_path)
    before = tree_digest(src)
    snapshot(src, tmp_path / "bundles")
    assert tree_digest(src) == before


def test_snapshot_refuses_an_in_flight_settlement(tmp_path):
    src = build_source(tmp_path)
    (src / (SOURCE_NAMES["validator_state"] + ".settlement.json")).write_text(
        '{"schema": 1, "phase": "prepared", "state": "{}", "cascade": null}')
    ok_src = build_source(tmp_path / "control")
    snapshot(ok_src, tmp_path / "control-bundles")
    out = tmp_path / "bundles"
    proc = run_cli("snapshot", "--source", src, "--config", MAINNET_TOML, "--out", out,
                   "--block", SOURCE_BLOCK)
    refused(proc, "settlement")
    assert not out.exists() or not any(out.iterdir())
