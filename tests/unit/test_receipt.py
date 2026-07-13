"""Round-receipt schema: round-trip, golden stability, signing, tamper tests.

Convention (same as the manifest tests): every signed schema gets a round-trip
test, a golden fixture pin (accidental serialisation drift breaks loudly), and
tamper tests — mutate one field and the signature over ``canonical_body`` must
no longer verify.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from cascade.shared.receipt import (
    RECEIPT_VERSION,
    RoundReceipt,
    dump_receipt,
    load_receipt,
    sign_receipt,
    verify_receipt_signature,
)

from .receipt_fixture import make_rejected_receipt, make_scored_receipt

GOLDEN_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "round_receipt_v3.json"


# ── round-trip ────────────────────────────────────────────────────────────────


def test_scored_receipt_round_trips_byte_identically():
    receipt, _, _ = make_scored_receipt()
    loaded = load_receipt(dump_receipt(receipt))
    assert loaded == receipt
    assert loaded.canonical_body() == receipt.canonical_body()


def test_rejected_receipt_round_trips():
    receipt = make_rejected_receipt(reason="contract_digest_mismatch: x != y")
    loaded = load_receipt(dump_receipt(receipt))
    assert loaded == receipt
    assert loaded.status == "rejected"
    assert loaded.reject_reason == "contract_digest_mismatch: x != y"
    assert loaded.eval_context is None and loaded.verdict is None


def test_embedded_manifest_recoverable_verbatim():
    receipt, _, _ = make_scored_receipt()
    m = receipt.load_embedded_manifest()
    assert m.round_id == receipt.round_id
    assert m.signature == "00ff" * 16  # the trainer's signature travels inside
    assert {e.role for e in m.entries} == {"king", "challenger"}


def test_score_records_round_trip_to_window_scores():
    receipt, king_scores, _ = make_scored_receipt()
    king_rec = next(e for e in receipt.entry_scores if e.role == "king")
    back = [r.to_score() for r in king_rec.scores]
    assert len(back) == len(king_scores)
    for a, b in zip(back, king_scores, strict=True):
        assert a.series_id == b.series_id and a.mase == b.mase
        assert a.abs_target == b.abs_target
        assert (a.qloss_per_q == b.qloss_per_q).all()


def test_wrong_version_rejected():
    receipt = make_rejected_receipt()
    body = json.loads(dump_receipt(receipt))
    body["receipt_version"] = RECEIPT_VERSION + 1
    with pytest.raises(ValueError, match="receipt_version"):
        load_receipt(json.dumps(body))


def test_rejected_status_requires_reason():
    receipt = make_rejected_receipt()
    with pytest.raises(ValueError, match="reject_reason"):
        replace(receipt, reject_reason=None)


def test_unknown_status_rejected():
    receipt = make_rejected_receipt()
    with pytest.raises(ValueError, match="status"):
        replace(receipt, status="draft")


def test_canonical_body_is_strict_json():
    # NaN would break third-party parsers; the schema maps it to null.
    receipt, _, _ = make_scored_receipt()
    body = receipt.canonical_body().decode("utf-8")
    assert "NaN" not in body and "Infinity" not in body
    json.loads(body)  # strict-parseable


def test_inconclusive_nan_lcb_becomes_null():
    from cascade.eval.koth import RoundResult
    from cascade.shared.receipt import VerdictRecord
    from cascade.validator.state import StateTransition, genesis

    result = RoundResult(
        challenger_wins_round=False, lcb=float("nan"), margin=0.02, n_windows=3,
        king_geomean=float("nan"), chal_geomean=1.0, inconclusive=True,
    )
    transition = StateTransition(
        state=genesis("king_hk", 0), dethroned=False,
        new_king_hotkey="king_hk", note="inconclusive",
    )
    receipt, _, _ = make_scored_receipt()
    v = VerdictRecord.from_round(result, transition,
                                 params=receipt.verdict and _params_of(receipt), bootstrap_seed=1)
    assert v.lcb is None and v.king_geomean is None and v.chal_geomean == 1.0


def _params_of(receipt: RoundReceipt):
    from cascade.eval.koth import KothParams

    return KothParams(**receipt.verdict.params)


# ── golden fixture ────────────────────────────────────────────────────────────


def test_golden_fixture_matches_schema():
    """The committed fixture pins receipt_version 1's exact serialisation.

    If this fails you changed the wire format: bump RECEIPT_VERSION and
    regenerate the fixture (see the module docstring of receipt_fixture.py)
    rather than silently breaking published receipts.
    """
    receipt, _, _ = make_scored_receipt()
    golden = GOLDEN_PATH.read_text(encoding="utf-8")
    assert dump_receipt(receipt) == golden
    assert load_receipt(golden) == receipt


def test_golden_fixture_canonical_digest_pinned():
    receipt, _, _ = make_scored_receipt()
    digest = hashlib.sha256(receipt.canonical_body()).hexdigest()
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    golden.pop("signature")
    want = hashlib.sha256(
        json.dumps(golden, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    assert digest == want


# ── signing + tamper ──────────────────────────────────────────────────────────


class _FakeHotkey:
    def sign(self, body: bytes) -> bytes:
        return b"SIG:" + body[:8]


class _FakeWallet:
    hotkey = _FakeHotkey()


def test_sign_receipt_signs_canonical_body():
    receipt, _, _ = make_scored_receipt()
    signed = sign_receipt(receipt, _FakeWallet())
    assert signed.signature == (b"SIG:" + receipt.canonical_body()[:8]).hex()
    # the signature is excluded from the signed body
    assert signed.canonical_body() == receipt.canonical_body()


def test_verify_rejects_missing_signature_or_hotkey():
    receipt, _, _ = make_scored_receipt()
    assert verify_receipt_signature(receipt, "5Fhotkey") is False
    assert verify_receipt_signature(replace(receipt, signature="abcd"), "") is False


def _mutations(receipt: RoundReceipt) -> dict[str, RoundReceipt]:
    """One tampered copy per receipt section (the signature must die on each)."""
    manifest = dict(receipt.manifest)
    manifest["entries"] = [dict(manifest["entries"][0]), *manifest["entries"][1:]]
    manifest["entries"][0]["gen_ref"] = "mallory/gen@sha256:" + "f" * 64
    scores0 = receipt.entry_scores[0]
    tampered_score = replace(scores0.scores[0], mase=scores0.scores[0].mase * 0.5)
    tampered_scores = replace(
        scores0, scores=(tampered_score, *scores0.scores[1:])
    )
    return {
        "round_id": replace(receipt, round_id="999"),
        "epoch_block_hash": replace(receipt, epoch_block_hash="0x" + "cd" * 32),
        "base_seed": replace(receipt, base_seed=receipt.base_seed + 1),
        "training_seed": replace(receipt, training_seed=receipt.training_seed + 1),
        "manifest": replace(receipt, manifest=manifest),
        "participants": replace(
            receipt,
            participants=(replace(receipt.participants[0],
                                  commit_block=receipt.epoch_start_block + 5),
                          *receipt.participants[1:]),
        ),
        "window_ids": replace(
            receipt,
            eval_context=replace(receipt.eval_context,
                                 window_ids=("hacked",) + receipt.eval_context.window_ids[1:]),
        ),
        "scores": replace(receipt, entry_scores=(tampered_scores, *receipt.entry_scores[1:])),
        "verdict": replace(
            receipt, verdict=replace(receipt.verdict, challenger_wins_round=False)
        ),
        "weights": replace(receipt, weights=(1.0,) + receipt.weights[1:]),
        "validator_hotkey": replace(receipt, validator_hotkey="5MalloryHotkey"),
    }


def test_every_field_mutation_changes_canonical_body():
    receipt, _, _ = make_scored_receipt()
    body = receipt.canonical_body()
    for name, tampered in _mutations(receipt).items():
        assert tampered.canonical_body() != body, f"mutation {name!r} did not change the body"


def test_real_signature_dies_on_any_tamper():
    bt = pytest.importorskip("bittensor")
    kp = bt.Keypair.create_from_uri("//Alice")
    receipt, _, _ = make_scored_receipt(validator_hotkey=kp.ss58_address)
    signed = sign_receipt(receipt, kp)
    assert verify_receipt_signature(signed, kp.ss58_address) is True
    # wrong signer address ⇒ untrusted
    other = bt.Keypair.create_from_uri("//Bob")
    assert verify_receipt_signature(signed, other.ss58_address) is False
    # every single-field tamper kills the signature
    for name, tampered in _mutations(signed).items():
        assert verify_receipt_signature(tampered, kp.ss58_address) is False, (
            f"mutation {name!r} still verified"
        )


def test_signature_survives_dump_load():
    bt = pytest.importorskip("bittensor")
    kp = bt.Keypair.create_from_uri("//Alice")
    receipt, _, _ = make_scored_receipt(validator_hotkey=kp.ss58_address)
    signed = sign_receipt(receipt, kp)
    loaded = load_receipt(dump_receipt(signed))
    assert verify_receipt_signature(loaded, kp.ss58_address) is True
