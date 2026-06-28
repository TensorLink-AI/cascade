"""Manifest signing/verification — the signing path with a fake bittensor wallet
and the no-signature / no-hotkey rejections (real ss58 verify needs bittensor)."""

from __future__ import annotations

from metronome.shared.manifest import (
    TrainedEntry,
    TrainingManifest,
    sign_manifest,
    verify_signature,
)

CID = "QmYwAPJzv5CZsnA625s3Xf2nemtYgPpHdWEz79ojWnPbdG"
CID_T = "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi"


def _manifest(sig=None):
    return TrainingManifest(
        round_id="1",
        created_block=10,
        contract_digest="c" * 64,
        base_arch_digest="a" * 64,
        eval_dataset="metronome-private-v1",
        entries=[TrainedEntry("hk", 0, "king", CID, f"metro-v1:trained:hippius:{CID_T}", "d", 1)],
        signature=sig,
    )


class _FakeHotkey:
    """Signs by returning the bytes back (deterministic) — exercises the wiring,
    not real crypto."""

    def sign(self, body: bytes) -> bytes:
        return b"SIG:" + body[:8]


class _FakeWallet:
    hotkey = _FakeHotkey()


def test_sign_manifest_signs_canonical_body():
    m = _manifest()
    assert m.signature is None
    signed = sign_manifest(m, _FakeWallet())
    assert signed.signature is not None
    # signature is hex of the fake signer's output over the canonical body
    expected = (b"SIG:" + m.canonical_body()[:8]).hex()
    assert signed.signature == expected
    # signing does not mutate the canonical body (signature is excluded from it)
    assert signed.canonical_body() == m.canonical_body()


def test_verify_rejects_missing_signature_or_hotkey():
    assert verify_signature(_manifest(sig=None), "5Fhotkey") is False
    assert verify_signature(_manifest(sig="abcd"), "") is False
