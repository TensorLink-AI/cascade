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


# ── court (former_kings) seeding ─────────────────────────────────────────────
#
# A joining validator that adopts only the throne votes the whole reward mass
# on one uid while established validators split it across the court, capping
# its vtrust at the top slot's consensus share (observed live 2026-07-31:
# an external validator converged on the right king yet pinned at 0.52 —
# exactly the court head's consensus weight).


def _signed_reign(king_hotkey: str, king_uid: int, round_id: str, kp=VALIDATOR_KP):
    """A signed scored receipt recording ``king_hotkey`` on the throne."""
    receipt, _, _ = make_scored_receipt(validator_hotkey=kp.ss58_address)
    receipt = replace(receipt, round_id=round_id,
                      verdict=replace(receipt.verdict,
                                      king_hotkey=king_hotkey, king_uid=king_uid))
    return sign_receipt(receipt, kp)


def _trail(*reigns, kp=VALIDATOR_KP, latest=True):
    """Store holding a chronological reign trail (oldest first) + index.

    ``reigns`` are ``(king_hotkey, king_uid)`` per round; round ids are
    ``r0…rN`` in order. ``latest=True`` also writes the newest receipt to the
    anchor's ``latest.json``.
    """
    anchor = kp.ss58_address
    objects, rows = {}, []
    receipts = [_signed_reign(hk, uid, f"r{i}", kp)
                for i, (hk, uid) in enumerate(reigns)]
    for r in receipts:
        objects[receipt_round_key(r.round_id, anchor)] = dump_receipt(r)
        rows.append({"round_id": r.round_id, "status": "scored",
                     "validator_hotkey": anchor})
    if latest and receipts:
        objects[receipt_latest_key(anchor)] = dump_receipt(receipts[-1])
    objects[RECEIPT_INDEX_KEY] = json.dumps({"schema": 2, "rounds": rows})
    return FakeStore(objects)


def test_court_seeded_by_recency_of_reign():
    # Reigns A → B → C (chronological); a fresh validator should adopt C with
    # the court [B, A] — the exact state apply_round would have accumulated.
    store = _trail(("A_hk", 10), ("B_hk", 11), ("C_hk", 12))
    state = _bootstrap_state_from_receipts(store, ANCHOR, keep_former_kings=4)
    assert state is not None
    assert (state.king_hotkey, state.king_uid) == ("C_hk", 12)
    assert state.former_kings == ("B_hk", "A_hk")


def test_court_capped_at_keep():
    store = _trail(("A_hk", 10), ("B_hk", 11), ("C_hk", 12))
    state = _bootstrap_state_from_receipts(store, ANCHOR, keep_former_kings=1)
    assert state is not None and state.former_kings == ("B_hk",)


def test_keep_zero_preserves_throne_only_adoption():
    store = _trail(("A_hk", 10), ("B_hk", 11))
    state = _bootstrap_state_from_receipts(store, ANCHOR)
    assert state is not None
    assert state.king_hotkey == "B_hk" and state.former_kings == ()


def test_recrowned_king_appears_once():
    # A reigned, was dethroned by B, retook the throne, then C won: the trail
    # C, A, B, A (newest-first) must yield court [A, B] — A once, by its most
    # recent reign, never duplicated and never listed as its own predecessor.
    store = _trail(("A_hk", 10), ("B_hk", 11), ("A_hk", 10), ("C_hk", 12))
    state = _bootstrap_state_from_receipts(store, ANCHOR, keep_former_kings=4)
    assert state is not None
    assert state.king_hotkey == "C_hk"
    assert state.former_kings == ("A_hk", "B_hk")


def test_unsigned_trail_rows_contribute_no_court():
    # Mallory-signed receipts mid-trail are skipped: the court comes only from
    # rounds that verify against the anchor.
    anchor = VALIDATOR_KP.ss58_address
    good = _trail(("A_hk", 10), ("C_hk", 12))
    forged = _signed_reign("EVIL_hk", 66, "rF", kp=OTHER_KP)
    good.objects[receipt_round_key("rF", anchor)] = dump_receipt(forged)
    index = json.loads(good.objects[RECEIPT_INDEX_KEY])
    index["rounds"].insert(1, {"round_id": "rF", "status": "scored",
                               "validator_hotkey": anchor})
    good.objects[RECEIPT_INDEX_KEY] = json.dumps(index)
    state = _bootstrap_state_from_receipts(good, anchor, keep_former_kings=4)
    assert state is not None
    assert state.king_hotkey == "C_hk" and state.former_kings == ("A_hk",)


def test_champion_and_court_both_from_index_when_latest_is_hold():
    # latest.json holds a rejected (verdict-less) receipt — the resync shape.
    # Champion AND court must still assemble from the indexed rounds.
    store = _trail(("A_hk", 10), ("B_hk", 11), latest=False)
    hold = make_rejected_receipt(reason="king_resyncing: champion x != trained y",
                                 validator_hotkey=ANCHOR)
    store.objects[receipt_latest_key(ANCHOR)] = dump_receipt(sign_receipt(hold, VALIDATOR_KP))
    state = _bootstrap_state_from_receipts(store, ANCHOR, keep_former_kings=4)
    assert state is not None
    assert state.king_hotkey == "B_hk" and state.former_kings == ("A_hk",)


def test_walk_is_bounded():
    # An index of one long unchanged reign stops at the read cap instead of
    # fetching every receipt; the champion still adopts.
    from cascade.validator import loop as loop_mod

    reads = 0
    store = _trail(*[("A_hk", 10)] * 5, ("B_hk", 11))
    real_get = store.get_text

    def counting_get(key):
        nonlocal reads
        reads += 1
        return real_get(key)

    store.get_text = counting_get
    old_cap = loop_mod._BOOTSTRAP_MAX_RECEIPT_READS
    loop_mod._BOOTSTRAP_MAX_RECEIPT_READS = 2
    try:
        state = _bootstrap_state_from_receipts(store, ANCHOR, keep_former_kings=4)
    finally:
        loop_mod._BOOTSTRAP_MAX_RECEIPT_READS = old_cap
    assert state is not None and state.king_hotkey == "B_hk"
    # latest.json + index + at most 2 per-round receipts
    assert reads <= 4
    assert state.former_kings == ("A_hk",)
