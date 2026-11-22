"""W17: the W-S3 sentinel board embedded in `status/lane_d.json` (§16 rule 4,
§1.3b Progress board). Contract: README.md. The board itself is W-S3's
`build_sentinel_board` (on main); the record is published by W3b3's real
`sweep_and_publish` (via the W-S3 fixture)."""

from __future__ import annotations

import pytest

from tests.accept.w_s3 import helpers as ws3

from . import helpers as h

ERA2_BLOCK = h.ARM_BLOCK + h.ERA      # the era B2_NEXT_ERA settles in
ERA2_HASH = h.era_hash("era-2")
ERA1_HASH = h.era_hash("era-1")


@pytest.fixture(scope="module")
def crowned(tmp_path_factory):
    return h.crowned(tmp_path_factory.mktemp("w17-sentinel-crowned"))


@pytest.fixture(scope="module")
def rich(tmp_path_factory):
    return h.rich_b1(tmp_path_factory.mktemp("w17-sentinel-rich"))


def _published_store(sc) -> h.MemoryStore:
    store = h.MemoryStore()
    house = sc["state"]["house"]["digest"]
    ws3.publish_record(store, king=sc["fx"].x["digest"], previous=house, house=house,
                       era_block=ERA2_BLOCK, block_hash=ERA2_HASH, boundary=h.B2_NEXT_ERA)
    return store


def _copy_store(store: h.MemoryStore) -> h.MemoryStore:
    out = h.MemoryStore()
    out.objects = dict(store.objects)
    return out


def _board(cfg, *, era_block, era_block_hash, kings, store):
    from cascade.shared.sentinel_board import build_sentinel_board

    return build_sentinel_board(cfg, era_block=era_block, era_block_hash=era_block_hash,
                                kings=kings, store=store)


def test_sentinel_board_embedded_for_the_eras_new_king(crowned):
    cfg = h.chain_cfg(sentinel=True)
    store = _published_store(crowned)
    want = _board(cfg, era_block=ERA2_BLOCK, era_block_hash=ERA2_HASH,
                  kings=[crowned["fx"].x["digest"]], store=_copy_store(store))
    assert want["kings"][0]["state"] == "ok", "fixture: the W3b3 record is valid"
    store.gets.clear()
    store.puts.clear()
    doc = h.status(cfg, state=crowned["state"], report=crowned["report"],
                   block=h.B2_NEXT_ERA + 5, era_block=ERA2_BLOCK, era_block_hash=ERA2_HASH,
                   store=store)
    assert doc["sentinel"] == want
    assert doc["sentinel"]["gating"] is False
    from cascade.validator.lane_d_sentinel import sentinel_key

    assert store.gets == [sentinel_key(ERA2_BLOCK, crowned["fx"].x["digest"])]
    assert store.puts == [], "the build writes nothing"


def test_sentinel_board_has_no_king_crowned_in_another_era(crowned):
    cfg = h.chain_cfg(sentinel=True)
    store = _published_store(crowned)
    doc = h.status(cfg, state=crowned["state"], report=crowned["report"],
                   block=h.B2_NEXT_ERA + 5, era_block=h.ARM_BLOCK, era_block_hash=ERA1_HASH,
                   store=store)
    want = _board(cfg, era_block=h.ARM_BLOCK, era_block_hash=ERA1_HASH, kings=[],
                  store=h.MemoryStore())
    assert doc["sentinel"] == want
    assert doc["sentinel"]["kings"] == []
    assert doc["sentinel"]["suite"] == want["suite"], "the era's suite is still shown"


def test_sentinel_board_has_no_king_while_the_house_reigns(rich):
    cfg = h.chain_cfg(sentinel=True, max_judged_per_coldkey_per_settlement=1)
    doc = h.status(cfg, state=rich["state"], report=rich["report"], block=h.B1 + 5,
                   era_block=h.ARM_BLOCK, era_block_hash=ERA1_HASH, store=h.MemoryStore())
    assert doc["sentinel"] is not None
    assert doc["sentinel"]["kings"] == []


def test_sentinel_board_absent_when_w_s1_unarmed(crowned):
    cfg = h.chain_cfg(sentinel=False)
    assert cfg.v2.sentinel_from_block == 0
    store = _published_store(crowned)
    store.gets.clear()
    store.puts.clear()
    doc = h.status(cfg, state=crowned["state"], report=crowned["report"],
                   block=h.B2_NEXT_ERA + 5, era_block=ERA2_BLOCK, era_block_hash=ERA2_HASH,
                   store=store)
    assert doc is not None
    assert doc["sentinel"] is None
    assert store.calls() == 0, "unarmed W-S1: no store read"


def test_sentinel_board_absent_without_an_era_block(crowned):
    cfg = h.chain_cfg(sentinel=True)
    store = h.MemoryStore()
    doc = h.status(cfg, state=crowned["state"], report=crowned["report"],
                   block=h.B2_NEXT_ERA + 5, store=store)
    assert doc["sentinel"] is None
    assert store.calls() == 0
