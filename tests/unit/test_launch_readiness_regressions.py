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


def test_calendar_replay_uses_finalized_time_not_the_receipts_claim():
    from tests.unit.test_calendar_launch import _chain

    cfg = load_chain_config()
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
