"""R10 restore: byte-identical placement by the target config; refusals."""

from __future__ import annotations

import json
import shlex
from pathlib import Path

from .helpers import (
    ACTIVATION_NAMES,
    MAINNET_TOML,
    SOURCE_BLOCK,
    SOURCE_NAMES,
    TARGET_NAMES,
    TESTNET_TOML,
    build_source,
    manifest,
    ok,
    refused,
    restore,
    restored_source,
    run_cli,
    sha256_file,
    snapshot,
)


def _placements() -> dict[str, str]:
    """source name -> target path (relative to the target dir)."""
    out = {SOURCE_NAMES[r]: TARGET_NAMES[r] for r in SOURCE_NAMES}
    out[SOURCE_NAMES["validator_state"] + ".bak"] = TARGET_NAMES["validator_state"] + ".bak"
    out.update({n: n for n in ACTIVATION_NAMES})
    return out


def test_restore_places_files_by_target_config_byte_identical(tmp_path):
    src, _, target = restored_source(tmp_path)
    for source_name, target_name in _placements().items():
        restored = target / target_name
        assert restored.is_file(), f"{source_name} not restored as {target_name}"
        assert restored.read_bytes() == (src / source_name).read_bytes(), target_name
    expected = set(_placements().values()) | {"laned_restore.json"}
    actual = {str(p.relative_to(target)) for p in target.rglob("*") if p.is_file()}
    assert actual == expected


def test_restore_writes_restore_record_with_rewritten_network_identity(tmp_path):
    _, bundle, target = restored_source(tmp_path)
    rec = json.loads((target / "laned_restore.json").read_text())
    assert rec["schema"] == "laned_state_restore/1"
    assert rec["bundle_sha256"] == sha256_file(bundle / "manifest.json")
    assert rec["source"] == {"netuid": 91, "network_id": "bittensor:finney",
                             "block": SOURCE_BLOCK}
    assert rec["target"]["netuid"] == 259
    assert rec["target"]["network_id"] == "bittensor:test"
    assert Path(rec["target"]["dir"]) == target.resolve()
    listed = {f["source_name"]: f for f in rec["files"]}
    assert set(listed) == set(_placements())
    by_source = {e["source_name"]: e for e in manifest(bundle)["files"]}
    for source_name, target_name in _placements().items():
        f = listed[source_name]
        assert f["target_path"] == target_name
        assert f["sha256"] == by_source[source_name]["sha256"] == sha256_file(target / target_name)
        assert f["role"] == by_source[source_name]["role"]


def test_restore_never_rewrites_signed_netconfig_network_binding(tmp_path):
    src, _, target = restored_source(tmp_path)
    restored = json.loads((target / TARGET_NAMES["netconfig_state"]).read_text())
    assert restored["history"][0]["envelope"]["body"]["network"] == {
        "chain": "bittensor:finney", "netuid": 91}
    assert restored == json.loads((src / SOURCE_NAMES["netconfig_state"]).read_text())


def test_restore_refuses_non_empty_target_unless_forced(tmp_path):
    src = build_source(tmp_path)
    bundle, _ = snapshot(src, tmp_path / "bundles")
    target = tmp_path / "testnet-node"
    target.mkdir()
    (target / "leftover.txt").write_text("from an earlier run")
    refused(restore(bundle, target), "--force-target")
    assert sorted(p.name for p in target.iterdir()) == ["leftover.txt"]
    ok(restore(bundle, target, "--force-target"))
    restored = target / TARGET_NAMES["validator_state"]
    assert restored.read_bytes() == (src / SOURCE_NAMES["validator_state"]).read_bytes()


def test_restore_into_an_empty_existing_target(tmp_path):
    src = build_source(tmp_path)
    bundle, _ = snapshot(src, tmp_path / "bundles")
    target = tmp_path / "testnet-node"
    target.mkdir()
    ok(restore(bundle, target))
    assert (target / "laned_restore.json").is_file()


def test_restore_refuses_a_bundle_that_does_not_verify(tmp_path):
    src = build_source(tmp_path)
    bundle, _ = snapshot(src, tmp_path / "bundles")
    ok(run_cli("verify", "--bundle", bundle))
    e = next(x for x in manifest(bundle)["files"] if x["source_name"] == "validator_state.db")
    path = bundle / e["path"]
    path.write_text(path.read_text().replace('"tenure_rounds": 6', '"tenure_rounds": 60'))
    target = tmp_path / "testnet-node"
    refused(restore(bundle, target), path.name)
    assert not target.exists() or not any(target.iterdir())


def test_restore_refuses_a_mainnet_target(tmp_path):
    src = build_source(tmp_path)
    bundle, _ = snapshot(src, tmp_path / "bundles")
    ok(run_cli("verify", "--bundle", bundle))
    target = tmp_path / "mainnet-again"
    proc = run_cli("restore", "--bundle", bundle, "--target", target, "--config", MAINNET_TOML)
    refused(proc, "91")
    assert not target.exists() or not any(target.iterdir())


def test_restore_start_cmd_prints_release_command_without_running(tmp_path):
    src = build_source(tmp_path)
    bundle, _ = snapshot(src, tmp_path / "bundles")
    target = tmp_path / "testnet-node"
    proc = ok(restore(bundle, target, "--start-cmd"))
    lines = proc.stdout.splitlines()
    cwd = [ln[len("START_CWD "):] for ln in lines if ln.startswith("START_CWD ")]
    cmd = [ln[len("START "):] for ln in lines if ln.startswith("START ")]
    assert cwd == [str(target.resolve())], proc.stdout
    assert len(cmd) == 1, proc.stdout
    argv = shlex.split(cmd[0])
    via_module = "-P" in argv and "-m" in argv and \
        argv[argv.index("-m") + 1] == "cascade.validator.main"
    via_entry = Path(argv[0]).name == "cascade-validator"
    assert via_module or via_entry, argv
    assert argv[argv.index("--chain-toml") + 1] == str(TESTNET_TOML.resolve()), argv
    assert argv[argv.index("--network") + 1] == "test", argv
    # dry-run: nothing started, so the target holds only what restore wrote
    expected = set(_placements().values()) | {"laned_restore.json"}
    assert {str(p.relative_to(target)) for p in target.rglob("*")} == expected
