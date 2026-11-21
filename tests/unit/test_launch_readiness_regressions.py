"""Release/runtime identities and calendar-aware replay at the launch boundary."""
from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from cascade.provision.codeprint import canonical_fingerprint
from cascade.shared.calendar_launch import launch_timestamp, receipt_calendar_config
from cascade.shared.config import load_chain_config, parse_chain_config
from cascade.shared.hippius import StorageError
from cascade.shared.manifest import TrainedEntry, contract_digest
from cascade.shared.netconfig import effective_raw
from cascade.trainer.launch_preflight import check_worker_release
from cascade.trainer.remote import receipt_to_entry


def test_uploaded_receipt_retains_identity_and_old_receipts_remain_replayable():
    receipt = dict(miner_hotkey="king", miner_uid=1, role="king", gen_ref="fixture/gen",
        trained_pointer="metro-v1:trained:hippius:fixture/ckpt@sha256:" + "a" * 64,
        corpus_digest="d", train_block=123, init_pointer="", training_contract_digest="b" * 64)
    entry = receipt_to_entry(receipt)
    assert entry.init_pointer == "" and entry.training_contract_digest == "b" * 64
    receipt.pop("init_pointer")
    receipt.pop("training_contract_digest")
    old = receipt_to_entry(receipt)
    assert old.init_pointer is None and old.training_contract_digest == ""


def test_stale_worker_pin_is_rejected_before_a_paid_launch():
    cfg = load_chain_config()
    cfg = replace(cfg, round=replace(cfg.round, worker_code_fingerprint="0" * 64))
    with pytest.raises(ValueError, match="rebuild from this source"):
        check_worker_release(cfg)


def test_after_launch_the_trainer_tree_need_not_match_the_image_pin(cfg, tmp_path):
    """2026-11-12: every trainer release after the pinned worker image blocked
    the king leg (only king legs pass the launch preflight). After a verified
    launch root exists, the pin names the image (pods verify themselves); the
    image pins must still name one release."""
    from cascade.trainer import launch_preflight as lp

    stale = replace(cfg, v2=replace(cfg.v2, from_block=43200),
                    manifest=replace(cfg.manifest, validator_hotkey="5Fixture"),
                    round=replace(cfg.round, worker_code_fingerprint="0" * 64,
                                  funded_pod_image="fixture/image@sha256:" + "a" * 64),
                    training=replace(cfg.training, v2_contract=replace(
                        cfg.training.v2_contract, train_image_digest="sha256:" + "a" * 64)))
    with pytest.raises(ValueError, match="rebuild from this source"):
        check_worker_release(stale)                               # launch day: refused
    check_worker_release(stale, launched=True)                    # after launch: the image is pinned
    with pytest.raises(ValueError, match="same release"):
        check_worker_release(replace(stale, round=replace(
            stale.round, funded_pod_image="fixture/image@sha256:" + "b" * 64)), launched=True)

    # ready(): the launch root's presence is what marks "after launch".
    seen = []
    real = lp.check_worker_release
    lp_check = lambda c, *, launched=False: seen.append(launched) or real(c, launched=launched)  # noqa: E731
    try:
        lp.check_worker_release = lp_check
        with pytest.raises(ValueError, match="rebuild from this source"):
            lp.ready(stale, "fixture/gen@sha256:" + "c" * 64, tmp_path)
        (tmp_path / "v2-launch.json").write_text("{}")
        assert lp.ready(stale, "fixture/gen@sha256:" + "c" * 64, tmp_path) is False   # no proof match
    finally:
        lp.check_worker_release = real
    assert seen == [False, True]


def test_worker_release_requires_matching_fingerprint_and_image_pins(cfg):
    root = Path(__file__).resolve().parents[2]
    fingerprint = canonical_fingerprint((str(p.relative_to(root)), hashlib.sha256(p.read_bytes()).hexdigest())
        for p in sorted((root / "cascade").rglob("*.py")))
    cfg = replace(cfg, v2=replace(cfg.v2, from_block=43200),
        round=replace(cfg.round, worker_code_fingerprint=fingerprint,
                      funded_pod_image="fixture/image@sha256:" + "a" * 64),
        training=replace(cfg.training, v2_contract=replace(cfg.training.v2_contract,
                          train_image_digest="sha256:" + "a" * 64)))
    check_worker_release(cfg)
    with pytest.raises(ValueError, match="same release"):
        check_worker_release(replace(cfg, round=replace(cfg.round,
            funded_pod_image="fixture/image@sha256:" + "b" * 64)))


def test_calendar_replay_uses_finalized_time_not_the_receipts_claim(tmp_path):
    from tests.unit.predecision_main import predecision_main
    from tests.unit.test_calendar_launch import _chain

    cfg = predecision_main(tmp_path)  # dated-launch scenario: the pre-2026-11-13 file
    raw = effective_raw(cfg)
    raw["netconfig"]["owner_hotkey"] = cfg.manifest.validator_hotkey
    cfg = parse_chain_config(raw)
    target = launch_timestamp(cfg.v2.launch_at_utc)
    receipt = SimpleNamespace(epoch_start_block=50000, v2_block=99999)
    assert receipt_calendar_config(cfg, receipt, _chain(target, 100000)).v2.from_block == 43200
    resolved = receipt_calendar_config(cfg, receipt, _chain(target, 100000))
    from cascade.audit.checks import check_v2, receipt_training

    replay = SimpleNamespace(epoch_start_block=54000, v2_block=43200, status="rejected")
    assert receipt_training(replay, resolved).base_arch == "cascade_pyramid"
    assert check_v2(replay, resolved).status == "PASS"
    with pytest.raises(ValueError, match="chain connection"):
        receipt_calendar_config(cfg, receipt)
    with pytest.raises(ValueError, match="not finalized"):
        receipt_calendar_config(cfg, receipt, _chain(target, 30000))
    before = receipt_calendar_config(cfg, SimpleNamespace(epoch_start_block=20000), _chain(target, 30000))
    assert before.v2.from_block == 0 and not before.v2.launch_at_utc


@pytest.mark.parametrize("bad", ["missing", "wrong_init", "wrong_contract"])
def test_v2_manifest_refuses_unbound_or_incompatible_worker_entries(cfg, tmp_path, bad):
    from cascade.trainer.rolling import EraState
    from tests.unit.test_dec55_trainer_epochs import _v2
    from tests.unit.test_rolling import Clock, _ptr, _sched

    cfg, start, _ = _v2(cfg)
    scheduler, _ = _sched(cfg, tmp_path, Clock())
    era = EraState(10, start, start, 1)
    contract = cfg.training.at_block(start)
    entry = TrainedEntry("king", 1, "king", "fixture/gen", _ptr("king"), "d", start,
        init_pointer="", training_contract_digest=contract_digest(contract))
    changes = {"missing": {"init_pointer": None, "training_contract_digest": ""},
               "wrong_init": {"init_pointer": _ptr("other")},
               "wrong_contract": {"training_contract_digest": "f" * 64}}
    with pytest.raises(StorageError, match="refusing v2 manifest"):
        scheduler._build_manifest(era, replace(entry, **changes[bad]), [], "50000", start,
                                  pin_block=start)
    manifest = scheduler._build_manifest(era, entry, [], "50000", start, pin_block=start)
    assert manifest.entries[0].training_contract_digest == manifest.contract_digest


def test_runtime_only_contract_changes_keep_a_verified_king_proof(cfg, tmp_path, monkeypatch):
    """2026-11-15: arming ``continuation_protocol`` (an IMAGE_RUNTIME_FIELD,
    numerics unchanged) moved the full contract digest; the per-king proof
    carried the old one and the era-2642 king leg was refused for 5.5 h."""
    import json

    from cascade.shared.manifest import numerics_contract_digest
    from cascade.trainer import launch_preflight as lp

    monkeypatch.setattr(lp, "check_worker_release", lambda c, *, launched=False: None)
    ref = "fixture/king@sha256:" + "c" * 64
    base = replace(cfg, manifest=replace(cfg.manifest, validator_hotkey="5Fixture"),
                   training=replace(cfg.training, v2_contract=replace(
                       cfg.training.v2_contract, continuation_protocol="")))
    report = {"ok": True, "scope": "full", "runtime_skipped": False, "corpus_digest": "d" * 64}
    legacy = {k: v for k, v in lp.identity(base, ref).items() if k != "numerics_contract_digest"}
    path = lp.proof_path(tmp_path, ref)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({**legacy, "verification": report}))

    armed = replace(base, training=replace(base.training, v2_contract=replace(
        base.training.v2_contract, continuation_protocol="cascade-continuation/1")))
    assert lp.identity(armed, ref)["contract_digest"] != legacy["contract_digest"]
    # a legacy proof (full digest only) cannot prove a different full digest…
    assert lp.ready(armed, ref, tmp_path) is False
    # …but one successful match under its own contract stamps the numerics digest
    assert lp.ready(base, ref, tmp_path) is True
    stamped = json.loads(path.read_text())
    assert stamped["numerics_contract_digest"] == numerics_contract_digest(base.training.v2_contract)
    (tmp_path / "v2-launch.json").unlink()
    # and from then on a runtime-only change keeps the king's proof valid
    assert lp.ready(armed, ref, tmp_path) is True
    repinned = replace(armed, training=replace(armed.training, v2_contract=replace(
        armed.training.v2_contract, train_image_digest="sha256:" + "e" * 64)))
    (tmp_path / "v2-launch.json").unlink()
    assert lp.ready(repinned, ref, tmp_path) is True

    # a numerics change is still refused
    c = armed.training.v2_contract
    changed = replace(armed, training=replace(armed.training, v2_contract=replace(
        c, weight_decay=c.weight_decay * 2)))
    assert numerics_contract_digest(changed.training.v2_contract) != stamped["numerics_contract_digest"]
    (tmp_path / "v2-launch.json").unlink()
    assert lp.ready(changed, ref, tmp_path) is False
