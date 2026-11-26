"""W14: the lane A round receipt carries the lane D record
(`cascade/shared/receipt.py`, `cascade/validator/receipts.py`).

Plan W14 row: "a `lane_d: {status: "absent"}` receipt state"; §3.2
invariant 2: "the receipt gains lane D fields"; invariant 1: "With
`from_block = 0`, every receipt ... matches today's". W7c README "Not
pinned": "How a lane A `RoundReceipt` embeds the lane D receipt record. That
wiring belongs to emission (W8) and the absent state (W14)."
"""

from __future__ import annotations

import dataclasses
import hashlib
import json

import pytest

from .helpers import Chain, absent, has_param

# sha256 of canonical_body() of tests/unit/receipt_fixture.py's byte-stable
# (cfg=None) receipts, recorded on main at 567f3d3.
SCORED_BODY_SHA = "14da7493f4c39a8226867391d246903c166a7985b05d614e034fb393fc345f24"
REJECTED_BODY_SHA = "4e4151ab26ce0395a5938f0759110a453d41b7ec8e5e3b03b29dc8249be10f62"


def _fixture():
    from tests.unit import receipt_fixture

    return receipt_fixture


def _require_field():
    from cascade.shared.receipt import RoundReceipt, build_receipt

    names = {f.name for f in dataclasses.fields(RoundReceipt)}
    assert "lane_d" in names, "RoundReceipt has no lane_d field"
    assert has_param(build_receipt, "lane_d"), "build_receipt takes no lane_d keyword"


def _absent_record(tmp_path):
    chain = Chain(tmp_path)
    sp = chain.state_path("r")
    chain.settle(1, sp)
    return absent(chain, 2, sp).receipt


def _present_record(tmp_path):
    chain = Chain(tmp_path)
    return chain.settle(1, chain.state_path("p")).receipt


# ── invariant 1 ─────────────────────────────────────────────────────────────


def test_inv1_receipt_unchanged_when_unarmed():
    """Receipts built without a lane D record keep today's exact bytes, and
    an explicit ``lane_d=None`` is the same receipt."""
    from cascade.shared.receipt import dump_receipt, load_receipt

    f = _fixture()
    scored = f.make_scored_receipt(None)[0]
    rejected = f.make_rejected_receipt(None)
    assert hashlib.sha256(scored.canonical_body()).hexdigest() == SCORED_BODY_SHA
    assert hashlib.sha256(rejected.canonical_body()).hexdigest() == REJECTED_BODY_SHA
    assert "lane_d" not in json.loads(scored.canonical_body())
    back = load_receipt(dump_receipt(scored))
    assert back.canonical_body() == scored.canonical_body()
    if "lane_d" in {fl.name for fl in dataclasses.fields(type(scored))}:
        assert scored.lane_d is None
        none = dataclasses.replace(scored, lane_d=None)
        assert none.canonical_body() == scored.canonical_body()


def test_inv1_receipt_armed_nonempty_lane_d_effect(tmp_path):
    _require_field()
    rec = _absent_record(tmp_path)
    scored = _fixture().make_scored_receipt(None)[0]
    stamped = dataclasses.replace(scored, lane_d=rec)
    body = json.loads(stamped.canonical_body())
    assert body["lane_d"] == rec
    assert body["lane_d"]["status"] == "absent"
    assert stamped.canonical_body() != scored.canonical_body()


# ── round trip ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("kind", ["absent", "present"])
def test_the_lane_d_record_round_trips(tmp_path, kind):
    from cascade.shared.receipt import dump_receipt, load_receipt, receipt_has_unknown_fields

    _require_field()
    rec = _absent_record(tmp_path) if kind == "absent" else _present_record(tmp_path)
    stamped = dataclasses.replace(_fixture().make_scored_receipt(None)[0], lane_d=rec)
    back = load_receipt(dump_receipt(stamped))
    assert back.lane_d == rec
    assert back.canonical_body() == stamped.canonical_body()
    # This code models the field: it is not a newer-code stamp.
    assert receipt_has_unknown_fields(back) is False


def test_build_receipt_takes_the_record(tmp_path):
    from cascade.shared.receipt import build_receipt

    _require_field()
    rec = _absent_record(tmp_path)
    scored = _fixture().make_scored_receipt(None)[0]
    fields = {fl.name: getattr(scored, fl.name) for fl in dataclasses.fields(scored)}
    rebuilt = build_receipt(
        round_id=scored.round_id, status=scored.status,
        epoch_start_block=scored.epoch_start_block, epoch_block_hash=scored.epoch_block_hash,
        base_seed=scored.base_seed,
        seeds=type("Seeds", (), {"generation_seed": fields["generation_seed"],
                                 "training_seed": fields["training_seed"]})(),
        manifest=scored.load_embedded_manifest(), participants=scored.participants,
        eval_context=scored.eval_context, entry_scores=scored.entry_scores,
        verdict=scored.verdict, reward_uids=scored.reward_uids, weights=scored.weights,
        lane_d=rec)
    assert json.loads(rebuilt.canonical_body())["lane_d"] == rec
    assert rebuilt.canonical_body() == dataclasses.replace(scored, lane_d=rec).canonical_body()


def test_a_signed_receipt_covers_the_record(tmp_path):
    """The record is inside the signed body: editing it after signing fails
    verification."""
    from cascade.shared.receipt import (
        dump_receipt,
        load_receipt,
        sign_receipt,
        verify_receipt_signature,
    )
    from tests.accept.w7c.helpers import fixture_key

    _require_field()
    rec = _absent_record(tmp_path)
    key = fixture_key("w14-main")

    class _Wallet:
        hotkey = key

    signed = sign_receipt(dataclasses.replace(_fixture().make_scored_receipt(None)[0],
                                              validator_hotkey=key.ss58_address,
                                              lane_d=rec), _Wallet())
    assert verify_receipt_signature(load_receipt(dump_receipt(signed)), key.ss58_address)
    obj = json.loads(dump_receipt(signed))
    obj["lane_d"]["reason"] = "infra_deferred"
    assert not verify_receipt_signature(load_receipt(json.dumps(obj)), key.ss58_address)


def test_build_round_receipt_passes_the_record_through():
    from cascade.validator.receipts import ReceiptMixin

    assert has_param(ReceiptMixin.build_round_receipt, "lane_d"), \
        "build_round_receipt takes no lane_d keyword"
