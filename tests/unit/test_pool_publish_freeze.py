"""Hourly pool republish freeze — the consensus guard for lane A.

With hourly publishes the pending index entry for one ``effective_block`` is
replaced several times. The trainer signs the entry's ``(key, sha256)`` into
the manifest (``eval_pool_key`` / ``eval_pool_sha256``) and every validator
re-resolves the index and must match it (``check_pool_pin``), then scores from
a fresh index read. These tests pin that an entry within one epoch of the
chain head can never change, so no two readers of the same epoch can see
different bytes.
"""

from __future__ import annotations

import datetime as dt
import json
from types import SimpleNamespace

import pytest

from cascade.pool.cli import chain_head_from_status, estimate_chain_head
from cascade.shared import hippius
from cascade.shared.hippius import (
    POOL_SENTINEL_MIN_BLOCK,
    ObjectNotFound,
    PoolEntryFrozen,
    publish_pool_snapshot,
    read_pool_index,
    tar_cid_digest,
)


class _Store:
    def __init__(self):
        self.objects: dict[str, bytes] = {}
        self.writes: list[str] = []

    def put_bytes(self, key, data, *, content_type="application/octet-stream", acl=None):
        self.objects[key] = bytes(data)
        self.writes.append(key)

    def put_text(self, key, text, *, content_type="text/plain", acl=None):
        self.put_bytes(key, text.encode("utf-8"))

    def get_bytes(self, key):
        if key not in self.objects:
            raise ObjectNotFound(key)
        return self.objects[key]

    def get_text(self, key):
        return self.get_bytes(key).decode("utf-8")


def _pub(store, data: bytes, block: int, **kw):
    return publish_pool_snapshot(store, data, effective_block=block, as_of="d", n_series=1,
                                 context_length=8, horizon=2, **kw)


def test_pending_entry_outside_the_horizon_is_replaced():
    s = _Store()
    _pub(s, b"v1", 9000, frozen_before_block=8000)
    _pub(s, b"v2", 9000, frozen_before_block=8000)
    (entry,) = read_pool_index(s)
    assert entry.sha256 == tar_cid_digest(b"v2")
    assert s.objects["pool/snapshots/block-9000.tar"] == b"v2"


def test_entry_within_the_horizon_is_final_and_nothing_is_written():
    s = _Store()
    _pub(s, b"v1", 9000, frozen_before_block=8000)
    s.writes.clear()
    with pytest.raises(PoolEntryFrozen, match="pool_entry_frozen"):
        _pub(s, b"v2", 9000, frozen_before_block=9001)
    assert s.writes == []  # neither the shared tar key nor the index was touched
    assert s.objects["pool/snapshots/block-9000.tar"] == b"v1"
    assert read_pool_index(s)[0].sha256 == tar_cid_digest(b"v1")


def test_identical_bytes_on_a_frozen_entry_is_a_noop():
    s = _Store()
    first = _pub(s, b"v1", 9000)
    s.writes.clear()
    assert _pub(s, b"v1", 9000, frozen_before_block=99999) == first
    assert s.writes == []


def test_new_entry_below_the_horizon_is_refused():
    s = _Store()
    with pytest.raises(PoolEntryFrozen, match="new entry"):
        _pub(s, b"v1", 9000, frozen_before_block=9500)
    assert s.objects == {}


def test_unknown_head_never_replaces_but_still_adds():
    s = _Store()
    _pub(s, b"v1", 9000, allow_replace=False)  # new entry: allowed (pre-PR behaviour)
    with pytest.raises(PoolEntryFrozen, match="chain head is unknown"):
        _pub(s, b"v2", 9000, allow_replace=False)
    assert read_pool_index(s)[0].sha256 == tar_cid_digest(b"v1")


def test_sentinel_is_never_frozen():
    s = _Store()
    _pub(s, b"v1", POOL_SENTINEL_MIN_BLOCK + 1, frozen_before_block=10**13, allow_replace=False)
    _pub(s, b"v2", POOL_SENTINEL_MIN_BLOCK + 1, frozen_before_block=10**13, allow_replace=False)
    assert read_pool_index(s)[0].sha256 == tar_cid_digest(b"v2")


def test_trainer_pin_and_validator_resolution_agree_across_a_refused_republish():
    """The binding: trainer pins (key, sha) for the settlement block; a validator
    re-resolving the index after a late hourly run still gets the same pair."""
    from cascade.shared.config import load_chain_config
    from cascade.validator.pool import BucketWindowSource

    s = _Store()
    _pub(s, b"scored", 9000, frozen_before_block=8000)
    src = BucketWindowSource(load_chain_config(None), s)
    trainer_pin = src.provenance_for_round(0, block=9000)
    with pytest.raises(PoolEntryFrozen):
        _pub(s, b"late-rebuild", 9000, frozen_before_block=9000 + 900)
    assert src.provenance_for_round(0, block=9000) == trainer_pin
    assert trainer_pin == ("pool/snapshots/block-9000.tar", tar_cid_digest(b"scored"))


# ── chain-head estimate (chain-free; validator status/chain.json) ────────────


NOW = dt.datetime(2026, 11, 24, 6, 0, tzinfo=dt.UTC)


def test_status_anchor_is_extrapolated_at_the_block_time():
    doc = {"current_block": 1000, "as_of": "2026-11-24T05:00:00+00:00"}
    assert chain_head_from_status(doc, NOW) == 1000 + 3600 // 12
    assert chain_head_from_status({"current_block": 5}, NOW) is None
    future = {"current_block": 1000, "as_of": "2026-11-24T07:00:00+00:00"}
    assert chain_head_from_status(future, NOW) == 1000  # clock skew never subtracts


def test_estimate_takes_the_greater_of_status_and_latest_manifest(monkeypatch):
    s = _Store()
    s.put_text("status/chain.json",
               json.dumps({"current_block": 1000, "as_of": "2026-11-24T06:00:00+00:00"}))
    monkeypatch.setattr(hippius, "read_latest_manifest", lambda store: "m")
    import cascade.shared.manifest as manifest_mod

    monkeypatch.setattr(manifest_mod, "load_manifest",
                        lambda text: SimpleNamespace(created_block=1500))
    assert estimate_chain_head(object(), store=s, now=NOW) == 1500


def test_estimate_without_a_wall_clock_anchor_is_unknown(monkeypatch):
    s = _Store()  # no status/chain.json
    monkeypatch.setattr(hippius, "read_latest_manifest", lambda store: "m")
    import cascade.shared.manifest as manifest_mod

    monkeypatch.setattr(manifest_mod, "load_manifest",
                        lambda text: SimpleNamespace(created_block=1500))
    # created_block is only a lower bound: it must never loosen the freeze
    assert estimate_chain_head(object(), store=s, now=NOW) is None
