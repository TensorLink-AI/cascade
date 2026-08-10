"""Bench report — the trainer's post-publish public-benchmark artifact:
build/sign/parse round-trip, entry lookups, and the signature gates (real ss58
verify needs bittensor, so signing runs on the fake-wallet wiring, exactly like
test_manifest_signing)."""

from __future__ import annotations

import json

import pytest

from cascade.shared.bench_report import (
    BENCH_REPORT_VERSION,
    BenchEntry,
    BenchReport,
    bench_report_key,
    dump_bench_report,
    load_bench_report,
    publish_bench_report,
    sign_bench_report,
    verify_bench_report_signature,
)
from cascade.shared.hippius import StorageError
from cascade.shared.manifest import BenchScores

PTR_K = "metro-v1:trained:hippius:cascade/ckpt-r1-king@sha256:" + "c" * 64
PTR_C = "metro-v1:trained:hippius:cascade/ckpt-r1-challenger@sha256:" + "d" * 64

KING_SCORES = BenchScores(
    gifteval_crps=0.42, gifteval_mase=0.81, boom_crps=0.55,
    boom_mase=0.90, time_crps=0.38, time_mase=0.77,
)
CHALL_SCORES = BenchScores(
    gifteval_crps=0.44, gifteval_mase=0.83, boom_crps=0.57,
    boom_mase=0.92, time_crps=0.40, time_mase=0.79,
)


def _report(entries=None, sig=None):
    if entries is None:
        entries = (
            BenchEntry(role="king", size="toto2-4m", miner_hotkey="king_hk",
                       miner_uid=1, trained_pointer=PTR_K, scores=KING_SCORES),
            BenchEntry(role="challenger", size="toto2-4m", miner_hotkey="chal_hk",
                       miner_uid=2, trained_pointer=PTR_C, scores=CHALL_SCORES),
        )
    return BenchReport(round_id="777", created_block=9000,
                       entries=tuple(entries), signature=sig)


class _FakeHotkey:
    def sign(self, body: bytes) -> bytes:
        return b"SIG:" + body[:8]


class _FakeWallet:
    hotkey = _FakeHotkey()


def test_round_trip_preserves_everything():
    signed = sign_bench_report(_report(), _FakeWallet())
    loaded = load_bench_report(dump_bench_report(signed))
    assert loaded == signed
    assert loaded.round_id == "777" and loaded.created_block == 9000
    assert loaded.entry_for("king").scores == KING_SCORES
    assert loaded.entry_for("challenger").scores == CHALL_SCORES
    # The signature survives the round trip AND the canonical body is stable.
    assert loaded.signature == signed.signature
    assert loaded.canonical_body() == signed.canonical_body()


def test_sign_covers_canonical_body_and_excludes_signature():
    r = _report()
    signed = sign_bench_report(r, _FakeWallet())
    assert signed.signature == (b"SIG:" + r.canonical_body()[:8]).hex()
    assert signed.canonical_body() == r.canonical_body()


def test_entry_lookups():
    r = _report()
    assert r.entry_for("king", size="toto2-4m").miner_hotkey == "king_hk"
    assert r.entry_for("king", size="toto2-22m") is None
    assert r.entry_for_pointer(PTR_C).role == "challenger"
    assert r.entry_for_pointer("metro-v1:trained:hippius:x/y@sha256:" + "0" * 64) is None


def test_partial_report_is_legitimate():
    only_king = (BenchEntry(role="king", size="toto2-4m", miner_hotkey="king_hk",
                            miner_uid=1, trained_pointer=PTR_K, scores=KING_SCORES),)
    loaded = load_bench_report(dump_bench_report(_report(entries=only_king)))
    assert loaded.entry_for("king") is not None
    assert loaded.entry_for("challenger") is None


def test_load_rejects_unknown_version():
    body = json.loads(dump_bench_report(_report()))
    body["report_version"] = BENCH_REPORT_VERSION + 1
    with pytest.raises(ValueError, match="report_version"):
        load_bench_report(json.dumps(body))


def test_entry_rejects_bad_role():
    with pytest.raises(ValueError, match="role"):
        BenchEntry(role="usurper", size="toto2-4m", miner_hotkey="hk",
                   miner_uid=0, trained_pointer=PTR_K, scores=KING_SCORES)


def test_verify_rejects_missing_signature_or_hotkey():
    assert verify_bench_report_signature(_report(sig=None), "5Fhotkey") is False
    assert verify_bench_report_signature(_report(sig="abcd"), "") is False


class _Store:
    """Minimal put_text sink recording the canned ACL each write carried."""

    def __init__(self, acl_raises: bool = False):
        self.texts: dict[str, str] = {}
        self.acls: list[str | None] = []
        self._acl_raises = acl_raises

    def put_text(self, key, text, *, content_type="text/plain", acl=None):
        self.acls.append(acl)
        if acl is not None and self._acl_raises:
            raise StorageError("acl unsupported")
        self.texts[key] = text


def test_publish_writes_the_round_key():
    store = _Store()
    text = dump_bench_report(_report())
    key = publish_bench_report(store, text, "777")
    assert key == bench_report_key("777") == "benchmarks/round-777.json"
    assert store.texts[key] == text


def test_publish_is_public_read():
    """The scoreboard fetches this object anonymously from the browser, and the
    manifest bucket is private by default — a report written without the canned
    ACL reads 403, which the page cannot tell apart from "never benched"."""
    store = _Store()
    publish_bench_report(store, dump_bench_report(_report()), "777")
    assert store.acls == ["public-read"]


def test_publish_falls_back_to_private_when_acl_unsupported():
    """Same fallback as the receipt publisher: a backend without canned ACLs
    publishes private rather than not at all."""
    store = _Store(acl_raises=True)
    text = dump_bench_report(_report())
    key = publish_bench_report(store, text, "777")
    assert store.acls == ["public-read", None]
    assert store.texts[key] == text
