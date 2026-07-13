"""The dashboard-facing receipts index: the compact per-round summary and the
public-read ``receipts/index.json`` rolling window the notebook reads.

Presentational only (no signing, no chain, no boto3) — exercised on the
self-consistent receipt fixture and a fake S3 store.
"""

from __future__ import annotations

import json

import cascade.shared.hippius as hippius
from cascade.shared.receipt import summarize_receipt

from .receipt_fixture import (
    GEN_REF_CHAL,
    GEN_REF_KING,
    make_rejected_receipt,
    make_scored_receipt,
)


class _FakeS3Store:
    def __init__(self):
        self.objects: dict[str, str] = {}
        self.acls: dict[str, str | None] = {}

    def put_text(self, key, text, *, content_type="text/plain", acl=None):
        self.objects[key] = text
        self.acls[key] = acl

    def get_text(self, key):
        # Mirror the real S3Store: a missing object is a StorageError, not KeyError.
        if key not in self.objects:
            raise hippius.StorageError(f"s3_get_failed: {key}: missing")
        return self.objects[key]


# ── summarize_receipt ────────────────────────────────────────────────────────


def test_summarize_scored_receipt_pulls_identities_and_verdict():
    receipt, _king, _chal = make_scored_receipt()
    s = summarize_receipt(receipt)

    assert s["round_id"] == receipt.round_id
    assert s["status"] == "scored"
    assert s["king_hotkey"] == "king_hk" and s["king_uid"] == 0
    assert s["chal_hotkey"] == "chal_hk" and s["chal_uid"] == 1
    assert s["king_gen_ref"] == GEN_REF_KING
    assert s["chal_gen_ref"] == GEN_REF_CHAL
    assert s["n_participants"] == 2
    assert s["n_windows"] == receipt.eval_context.n_windows
    # unlabeled windows ⇒ each is its own cluster ⇒ n_clusters == n_windows
    assert s["n_clusters"] == len(receipt.entry_scores[0].scores)
    # the fixture's challenger is strictly better ⇒ finite geomeans, a decision
    assert s["king_geomean"] is not None and s["chal_geomean"] is not None
    assert s["chal_geomean"] < s["king_geomean"]
    assert s["challenger_wins_round"] is True
    assert s["reject_reason"] is None
    # every value must be JSON-round-trippable (no numpy/NaN leaking through)
    assert json.loads(json.dumps(s)) == s


def test_summarize_rejected_receipt_has_reason_and_empty_verdict():
    receipt = make_rejected_receipt(reason="signature_invalid")
    s = summarize_receipt(receipt)

    assert s["status"] == "rejected"
    assert s["reject_reason"] == "signature_invalid"
    # a rejected round carries no scores/verdict
    assert s["king_geomean"] is None and s["chal_geomean"] is None
    assert s["challenger_wins_round"] is None and s["dethroned"] is None
    assert s["n_windows"] is None
    # generator refs still come from the embedded (gated) manifest
    assert s["king_gen_ref"] == GEN_REF_KING and s["chal_gen_ref"] == GEN_REF_CHAL
    assert s["n_clusters"] is None   # no scores ⇒ no cluster count


def test_summarize_counts_distinct_clusters_from_sources():
    # When windows carry upstream-feed ``source`` labels, n_clusters collapses
    # to the distinct feeds (what the cluster bootstrap actually resampled),
    # derived from the SIGNED per-window keys — not the raw window count.
    from dataclasses import replace

    receipt, _k, _c = make_scored_receipt()
    king = receipt.entry_scores[0]
    labelled = tuple(
        replace(sc, source=f"feed{i % 4}") for i, sc in enumerate(king.scores)
    )
    receipt = replace(
        receipt,
        entry_scores=(replace(king, scores=labelled), *receipt.entry_scores[1:]),
    )
    s = summarize_receipt(receipt)
    assert s["n_clusters"] == 4                       # 4 distinct feeds
    assert s["n_windows"] == len(labelled) and s["n_windows"] > 4


# ── update_receipt_index ─────────────────────────────────────────────────────


def test_update_receipt_index_writes_public_read_with_header():
    store = _FakeS3Store()
    receipt, _, _ = make_scored_receipt()
    entry = hippius.update_receipt_index(
        store, summarize_receipt(receipt),
        updated_at="2026-07-03T00:00:00+00:00",
        subnet={"netuid": 7, "name": "cascade"},
    )

    assert entry["receipt_key"] == f"receipts/round-{receipt.round_id}.json"
    assert entry["published_at"] == "2026-07-03T00:00:00+00:00"
    assert store.acls[hippius.RECEIPT_INDEX_KEY] == "public-read"

    doc = json.loads(store.objects[hippius.RECEIPT_INDEX_KEY])
    assert doc["schema"] == hippius.RECEIPT_INDEX_SCHEMA
    assert doc["subnet"] == {"netuid": 7, "name": "cascade"}
    assert doc["updated_at"] == "2026-07-03T00:00:00+00:00"
    assert [r["round_id"] for r in doc["rounds"]] == [receipt.round_id]


def test_update_receipt_index_idempotent_per_round():
    store = _FakeS3Store()
    receipt, _, _ = make_scored_receipt()
    summary = summarize_receipt(receipt)
    hippius.update_receipt_index(store, summary)
    hippius.update_receipt_index(store, summary)  # same round again

    doc = json.loads(store.objects[hippius.RECEIPT_INDEX_KEY])
    assert len(doc["rounds"]) == 1  # replaced, not duplicated


def test_update_receipt_index_sorts_by_epoch_and_caps():
    store = _FakeS3Store()
    # synthetic summaries across epochs, inserted out of order
    for blk in (300, 100, 200):
        hippius.update_receipt_index(
            store, {"round_id": f"r{blk}", "epoch_start_block": blk, "status": "scored"},
            max_keep=2,
        )
    doc = json.loads(store.objects[hippius.RECEIPT_INDEX_KEY])
    # sorted ascending by epoch, then capped to the 2 most-recent
    assert [r["round_id"] for r in doc["rounds"]] == ["r200", "r300"]


def test_read_receipt_index_empty_when_absent_or_malformed():
    store = _FakeS3Store()
    assert hippius.read_receipt_index(store) == {"schema": hippius.RECEIPT_INDEX_SCHEMA, "rounds": []}
    store.objects[hippius.RECEIPT_INDEX_KEY] = "{not json"
    assert hippius.read_receipt_index(store)["rounds"] == []
    store.objects[hippius.RECEIPT_INDEX_KEY] = json.dumps({"schema": 1})  # no rounds list
    assert hippius.read_receipt_index(store)["rounds"] == []


# ── per-validator prefixed index ─────────────────────────────────────────────

HK = "5ValidatorHotkey"


class _ScopedS3Store(_FakeS3Store):
    """Writes allowed only under one validator's prefix — models the
    owner-issued credential scoped to ``receipts/<hotkey>/*``."""

    def __init__(self, allow_prefix: str):
        super().__init__()
        self.allow_prefix = allow_prefix

    def put_text(self, key, text, *, content_type="text/plain", acl=None):
        if not key.startswith(self.allow_prefix):
            raise hippius.StorageError(f"s3_put_failed: {key}: AccessDenied")
        super().put_text(key, text, content_type=content_type, acl=acl)


def test_update_receipt_index_prefixed_mirrors_legacy():
    store = _FakeS3Store()
    receipt, _, _ = make_scored_receipt()
    entry = hippius.update_receipt_index(
        store, summarize_receipt(receipt), validator_hotkey=HK,
    )

    # the entry points at the validator's own per-round receipt
    assert entry["receipt_key"] == f"receipts/{HK}/round-{receipt.round_id}.json"
    prefixed = hippius.receipt_index_key(HK)
    assert prefixed == f"receipts/{HK}/index.json"
    assert store.acls[prefixed] == "public-read"
    # broad (owner) credential ⇒ the legacy shared index stays fed for old readers
    assert store.objects[hippius.RECEIPT_INDEX_KEY] == store.objects[prefixed]


def test_update_receipt_index_scoped_key_skips_legacy_mirror():
    store = _ScopedS3Store(f"receipts/{HK}/")
    hippius.update_receipt_index(
        store, {"round_id": "r1", "epoch_start_block": 1, "status": "scored"},
        validator_hotkey=HK,
    )
    assert hippius.receipt_index_key(HK) in store.objects
    # the denied legacy mirror is skipped, not fatal
    assert hippius.RECEIPT_INDEX_KEY not in store.objects


def test_read_receipt_index_seeds_from_legacy_history():
    store = _FakeS3Store()
    # pre-prefix history lives only in the shared index
    hippius.update_receipt_index(
        store, {"round_id": "r1", "epoch_start_block": 1, "status": "scored"},
    )
    doc = hippius.read_receipt_index(store, HK)   # no prefixed index yet
    assert [r["round_id"] for r in doc["rounds"]] == ["r1"]
    # the first prefixed update carries that history forward
    hippius.update_receipt_index(
        store, {"round_id": "r2", "epoch_start_block": 2, "status": "scored"},
        validator_hotkey=HK,
    )
    doc = json.loads(store.objects[hippius.receipt_index_key(HK)])
    assert [r["round_id"] for r in doc["rounds"]] == ["r1", "r2"]
