"""A latest receipt that names no king (a rejected king re-sync or a
no-contest) must not hide the crowning: the trainer walks the settlement
chain back to the newest SCORED receipt (2026-11-04: the crowning's receipt
was superseded 21 s later by a rejected re-sync receipt, and a freshly
restarted trainer, with no sticky king, kept naming the dethroned king)."""
from __future__ import annotations

from dataclasses import replace

import pytest

from cascade.shared.hippius import manifest_round_key, receipt_latest_key, receipt_round_key
from cascade.shared.manifest import dump_manifest
from cascade.shared.receipt import dump_receipt, sign_receipt
from cascade.trainer.loop import TrainerRunner

from .receipt_fixture import make_manifest, make_rejected_receipt, make_scored_receipt

bt = pytest.importorskip("bittensor.sp_core")
ALICE = bt.Keypair.create_from_uri("//Alice")
BOB = bt.Keypair.create_from_uri("//Bob")


class _Store:
    def __init__(self, texts):
        self.texts = texts

    def get_text(self, key):
        if key not in self.texts:
            raise KeyError(key)
        return self.texts[key]


def _runner(cfg, tmp_path, store):
    cfg = replace(cfg, manifest=replace(cfg.manifest, validator_hotkey=ALICE.ss58_address))
    r = TrainerRunner(cfg=cfg, base_trainer=object(), work_root=tmp_path, use_sandbox=False)
    r.manifest_store = lambda: store  # type: ignore[method-assign]
    return r


def _manifest(rid, prev):
    return dump_manifest(replace(make_manifest(base_seed=1), round_id=rid, prev_round_id=prev))


def _chain(signer=ALICE):
    scored, _, _ = make_scored_receipt(validator_hotkey=ALICE.ss58_address)
    scored = replace(scored, round_id="100")
    rejected = replace(make_rejected_receipt(reason="king_resyncing",
                                             validator_hotkey=ALICE.ss58_address),
                       round_id="101")
    a = ALICE.ss58_address
    texts = {
        receipt_latest_key(a): dump_receipt(sign_receipt(rejected, ALICE)),
        receipt_round_key("101", a): dump_receipt(sign_receipt(rejected, ALICE)),
        receipt_round_key("100", a): dump_receipt(sign_receipt(scored, signer)),
        manifest_round_key("101"): _manifest("101", "100"),
        manifest_round_key("100"): _manifest("100", ""),
    }
    return texts, scored.verdict.king_hotkey


def test_rejected_latest_receipt_walks_back_to_the_crowning(cfg, tmp_path):
    texts, king = _chain()
    assert king
    assert _runner(cfg, tmp_path, _Store(texts))._receipt_king() == king


def test_walk_back_ignores_a_receipt_with_a_foreign_signature(cfg, tmp_path):
    texts, _ = _chain(signer=BOB)
    assert _runner(cfg, tmp_path, _Store(texts))._receipt_king() is None


def test_walk_back_stops_on_a_broken_chain(cfg, tmp_path):
    texts, _ = _chain()
    del texts[manifest_round_key("101")]
    assert _runner(cfg, tmp_path, _Store(texts))._receipt_king() is None


def test_walk_back_is_bounded(cfg, tmp_path):
    texts, _ = _chain()
    r = _runner(cfg, tmp_path, _Store(texts))
    assert r._walk_back_receipt_king(_Store(texts), ALICE.ss58_address, "101", limit=0) is None
    assert r._walk_back_receipt_king(_Store(texts), ALICE.ss58_address, "101", limit=1)[1] == "100"
