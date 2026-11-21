"""R10 check: the restored state loads cleanly through the real loaders."""

from __future__ import annotations

from .helpers import (
    ACTIVATION_NAMES,
    SOURCE_NAMES,
    TARGET_NAMES,
    TESTNET_TOML,
    ActivationStore,
    build_source,
    casc,
    champion,
    nc,
    ok,
    refused,
    restore,
    restored_source,
    run_cli,
    snapshot,
)


def _check(target):
    return run_cli("check", "--target", target, "--config", TESTNET_TOML)


def _restore_from(src, tmp_path):
    bundle, _ = snapshot(src, tmp_path / "bundles")
    target = tmp_path / "testnet-node"
    ok(restore(bundle, target))
    return target


def test_check_reports_clean_start_and_real_loaders_agree(tmp_path):
    src, _, target = restored_source(tmp_path)
    proc = ok(_check(target))
    assert "CLEAN START" in proc.stdout
    # the real loaders see exactly the mainnet state at the testnet paths
    assert champion.loads((target / TARGET_NAMES["validator_state"]).read_text()) == \
        champion.loads((src / SOURCE_NAMES["validator_state"]).read_text())
    assert casc.load_state(target / TARGET_NAMES["cascade_state"]) == \
        casc.load_state(src / SOURCE_NAMES["cascade_state"])
    assert nc.load_state(target / TARGET_NAMES["netconfig_state"]) == \
        nc.load_state(src / SOURCE_NAMES["netconfig_state"])
    for name in ACTIVATION_NAMES:
        assert ActivationStore(target / name).load() == ActivationStore(src / name).load()


def _clean_then_break(tmp_path, source_name, text):
    """A control restore checks clean; then the same source with one file
    unparseable (no usable .bak) restores byte-identically and must not."""
    ok(_check(restored_source(tmp_path / "control")[2]))
    src = build_source(tmp_path / "broken")
    (src / source_name).write_text(text)
    return _restore_from(src, tmp_path / "broken")


def test_check_refuses_unparseable_cascade_state(tmp_path):
    target = _clean_then_break(tmp_path, SOURCE_NAMES["cascade_state"], '{"checkpoints": [')
    text = refused(_check(target), TARGET_NAMES["cascade_state"])
    assert "CLEAN START" not in text


def test_check_refuses_unparseable_netconfig_state_even_when_netconfig_off(tmp_path):
    # chain.testnet.toml has [netconfig] owner_hotkey = "": the check still loads it
    target = _clean_then_break(tmp_path, SOURCE_NAMES["netconfig_state"], '{"history": 7')
    refused(_check(target), TARGET_NAMES["netconfig_state"])


def test_check_refuses_unparseable_activation_state(tmp_path):
    target = _clean_then_break(tmp_path, "activation_forfeit_state.json", "not json")
    refused(_check(target), "activation_forfeit_state.json")


def test_check_refuses_unparseable_warm_start_init(tmp_path):
    target = _clean_then_break(tmp_path, SOURCE_NAMES["warm_start_init"], "[1, 2")
    refused(_check(target), TARGET_NAMES["warm_start_init"])


def test_check_refuses_a_backup_fallback(tmp_path):
    # torn main validator state, valid .bak: the loader would resume from .bak
    target = _clean_then_break(tmp_path, SOURCE_NAMES["validator_state"], '{"king_hotkey": ')
    assert champion.loads((target / (TARGET_NAMES["validator_state"] + ".bak")).read_text())
    refused(_check(target), ".bak")


def test_check_refuses_drift_after_restore(tmp_path):
    _, _, target = restored_source(tmp_path)
    ok(_check(target))
    path = target / TARGET_NAMES["cascade_state"]
    st = casc.load_state(path)
    path.write_text(casc.dumps(casc.CascadeState(king_hotkey=st.king_hotkey,
                                                 reign_start_block=st.reign_start_block,
                                                 generation=st.generation + 1,
                                                 members=st.members)))
    refused(_check(target), TARGET_NAMES["cascade_state"])


def test_check_refuses_without_restore_record(tmp_path):
    _, _, target = restored_source(tmp_path)
    ok(_check(target))
    (target / "laned_restore.json").unlink()
    refused(_check(target), "laned_restore.json")


def test_check_refuses_missing_validator_state(tmp_path):
    _, _, target = restored_source(tmp_path)
    ok(_check(target))
    (target / TARGET_NAMES["validator_state"]).unlink()
    (target / (TARGET_NAMES["validator_state"] + ".bak")).unlink()
    refused(_check(target), TARGET_NAMES["validator_state"])
