"""First-boot champion inheritance from the signed receipt trail.

A validator with no local state must adopt the throne recorded by the signed
public receipts (``_bootstrap_state_from_receipts``) rather than judging the
next manifest blind — and must adopt NOTHING that does not verify against the
pinned signer. See OPSLOG 2026-07-17 (king-sync pause) for the incident that
motivated this.
"""

from __future__ import annotations

import json
from dataclasses import replace

import bittensor as bt

from cascade.shared.hippius import (
    RECEIPT_INDEX_KEY,
    RECEIPT_LATEST_KEY,
    receipt_latest_key,
    receipt_round_key,
)
from cascade.shared.receipt import dump_receipt, sign_receipt
from cascade.validator.loop import _bootstrap_state_from_receipts

from .receipt_fixture import make_rejected_receipt, make_scored_receipt

VALIDATOR_KP = bt.Keypair.create_from_uri("//Validator")
OTHER_KP = bt.Keypair.create_from_uri("//Mallory")
ANCHOR = VALIDATOR_KP.ss58_address


class FakeStore:
    def __init__(self, objects: dict[str, str]):
        self.objects = dict(objects)

    def get_text(self, key: str) -> str:
        if key not in self.objects:
            raise KeyError(key)
        return self.objects[key]


def _signed_scored():
    receipt, _, _ = make_scored_receipt(validator_hotkey=ANCHOR)
    return sign_receipt(receipt, VALIDATOR_KP)


def test_adopts_champion_from_signed_latest_receipt():
    receipt = _signed_scored()
    store = FakeStore({receipt_latest_key(ANCHOR): dump_receipt(receipt)})
    state = _bootstrap_state_from_receipts(store, ANCHOR)
    assert state is not None
    assert state.king_hotkey == receipt.verdict.king_hotkey
    assert state.king_uid == receipt.verdict.king_uid
    # inherited state starts a fresh local history — nothing else carried over
    assert state.rounds_seen == 0 and state.resync_holds == 0


def test_legacy_shared_pointer_is_a_fallback():
    receipt = _signed_scored()
    store = FakeStore({RECEIPT_LATEST_KEY: dump_receipt(receipt)})
    state = _bootstrap_state_from_receipts(store, ANCHOR)
    assert state is not None and state.king_hotkey == receipt.verdict.king_hotkey


def test_wrong_signer_adopts_nothing():
    receipt, _, _ = make_scored_receipt(validator_hotkey=OTHER_KP.ss58_address)
    forged = sign_receipt(receipt, OTHER_KP)  # valid signature, wrong identity
    store = FakeStore({receipt_latest_key(ANCHOR): dump_receipt(forged)})
    assert _bootstrap_state_from_receipts(store, ANCHOR) is None


def test_tampered_receipt_adopts_nothing():
    receipt = _signed_scored()
    tampered = replace(receipt, round_id=receipt.round_id + "0")  # body != signature
    store = FakeStore({receipt_latest_key(ANCHOR): dump_receipt(tampered)})
    assert _bootstrap_state_from_receipts(store, ANCHOR) is None


def test_genesis_throne_adopts_nothing():
    receipt, _, _ = make_scored_receipt(validator_hotkey=ANCHOR)
    receipt = replace(receipt, verdict=replace(receipt.verdict,
                                               king_hotkey=None, king_uid=None))
    signed = sign_receipt(receipt, VALIDATOR_KP)
    store = FakeStore({receipt_latest_key(ANCHOR): dump_receipt(signed)})
    assert _bootstrap_state_from_receipts(store, ANCHOR) is None


def test_hold_latest_falls_back_to_indexed_scored_round():
    # latest.json is a hold/rejected receipt (verdict-less) — the shape every
    # resync round writes. The index points at the earlier scored round; only
    # the SIGNED per-round receipt it points to is trusted.
    scored = _signed_scored()
    hold = make_rejected_receipt(reason="king_resyncing: champion x != trained y",
                                 validator_hotkey=ANCHOR)
    hold = sign_receipt(hold, VALIDATOR_KP)
    index = {"schema": 2, "rounds": [
        {"round_id": scored.round_id, "status": "scored", "validator_hotkey": ANCHOR},
        {"round_id": hold.round_id, "status": "rejected", "validator_hotkey": ANCHOR},
    ]}
    store = FakeStore({
        receipt_latest_key(ANCHOR): dump_receipt(hold),
        RECEIPT_INDEX_KEY: json.dumps(index),
        receipt_round_key(scored.round_id, ANCHOR): dump_receipt(scored),
    })
    state = _bootstrap_state_from_receipts(store, ANCHOR)
    assert state is not None and state.king_hotkey == scored.verdict.king_hotkey


def test_index_pointer_alone_grants_nothing():
    # A (possibly attacker-writable) index row pointing at a round whose
    # receipt is missing or unsigned must not mint a champion.
    scored, _, _ = make_scored_receipt(validator_hotkey=ANCHOR)  # NOT signed
    index = {"schema": 2, "rounds": [
        {"round_id": scored.round_id, "status": "scored", "validator_hotkey": ANCHOR},
    ]}
    store = FakeStore({
        RECEIPT_INDEX_KEY: json.dumps(index),
        receipt_round_key(scored.round_id, ANCHOR): dump_receipt(scored),
    })
    assert _bootstrap_state_from_receipts(store, ANCHOR) is None


def test_empty_store_or_anchor_starts_blank():
    assert _bootstrap_state_from_receipts(FakeStore({}), ANCHOR) is None
    assert _bootstrap_state_from_receipts(FakeStore({}), "") is None
