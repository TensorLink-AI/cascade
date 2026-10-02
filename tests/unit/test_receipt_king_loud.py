"""The trainer's receipt-king read is never silent.

Failure class (2026-10-02 04:12): the validator crowned uid 124; the trainer,
running pre-#345 code, could not verify receipts carrying the new
margin_v2_block stamp, silently kept the last king it had read (the deposed
uid 135) and never adopted the dethrone until restarted. Receipts from newer
code must now verify, and an unverifiable/unreadable receipt must be logged.
"""
from __future__ import annotations

import json
import logging
from dataclasses import replace

import pytest

from cascade.shared.hippius import RECEIPT_LATEST_KEY, receipt_latest_key
from cascade.shared.receipt import dump_receipt, sign_receipt
from cascade.trainer import loop as loop_mod
from cascade.trainer.loop import TrainerRunner

from .receipt_fixture import make_scored_receipt

bt = pytest.importorskip("bittensor")
ALICE = bt.Keypair.create_from_uri("//Alice")
BOB = bt.Keypair.create_from_uri("//Bob")


class _Store:
    def __init__(self, texts: dict[str, str]):
        self.texts = texts

    def get_text(self, key: str) -> str:
        if key not in self.texts:
            raise KeyError(key)
        return self.texts[key]


def _signed_text(signer, *, future_field: bool = False) -> tuple[str, str]:
    receipt, _, _ = make_scored_receipt(validator_hotkey=ALICE.ss58_address)
    obj = json.loads(dump_receipt(sign_receipt(receipt, ALICE)))
    if future_field:
        obj["some_future_stamp"] = 9194400
    body = {k: v for k, v in obj.items() if k != "signature"}
    raw = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False)
    text = json.dumps(dict(body, signature=signer.sign(raw.encode()).hex()), indent=2)
    return text, receipt.verdict.king_hotkey


def _runner(cfg, tmp_path, store):
    cfg = replace(cfg, manifest=replace(cfg.manifest, validator_hotkey=ALICE.ss58_address))
    runner = TrainerRunner(cfg=cfg, base_trainer=object(), work_root=tmp_path,
                           use_sandbox=False)
    runner.manifest_store = lambda: store  # type: ignore[method-assign]
    return runner


def _logs(monkeypatch):
    seen: list[tuple[int, str]] = []
    monkeypatch.setattr(loop_mod.log, "log",
                        lambda lvl, msg, *a, **k: seen.append((lvl, msg % a if a else msg)))
    return seen


def test_receipt_from_newer_code_is_adopted_with_a_restart_warning(cfg, tmp_path, monkeypatch):
    text, king = _signed_text(ALICE, future_field=True)
    store = _Store({receipt_latest_key(ALICE.ss58_address): text})
    runner = _runner(cfg, tmp_path, store)
    seen = _logs(monkeypatch)
    assert runner._receipt_king() == king
    assert runner._receipt_king() == king
    warns = [m for lvl, m in seen if lvl == logging.WARNING]
    assert len(warns) == 1 and "restart the trainer" in warns[0], seen


def test_unverifiable_receipt_keeps_the_sticky_king_loudly_once(cfg, tmp_path, monkeypatch):
    good, king = _signed_text(ALICE)
    key = receipt_latest_key(ALICE.ss58_address)
    store = _Store({key: good})
    runner = _runner(cfg, tmp_path, store)
    seen = _logs(monkeypatch)
    assert runner._receipt_king() == king and seen == []

    forged, _ = _signed_text(BOB)          # same body, wrong signer
    store.texts = {key: forged, RECEIPT_LATEST_KEY: forged}
    assert runner._receipt_king() == king  # sticky
    assert runner._receipt_king() == king
    errors = [m for lvl, m in seen if lvl == logging.ERROR]
    assert len(errors) == 1, seen          # once per distinct problem, not per tick
    assert "may be STALE" in errors[0] and "signature" in errors[0]

    store.texts = {key: good}              # recovers; a later failure logs again
    assert runner._receipt_king() == king
    store.texts = {}
    runner._receipt_king()
    assert len([m for lvl, m in seen if lvl == logging.ERROR]) == 2, seen
