"""``status/chain.json`` — the dashboard's live chain-status document."""

from __future__ import annotations

from cascade.shared.chain import Commitment
from cascade.shared.chain_status import (
    CHAIN_STATUS_KEY,
    build_chain_status,
    publish_chain_status,
)
from cascade.shared.hippius import StorageError


def _commit(uid, hotkey, block, payload=None):
    payload = payload if payload is not None else (
        f"metro-v1:gen:hippius:ns/gen-{uid}@sha256:{'a' * 64}"
    )
    return Commitment(uid=uid, hotkey=hotkey, coldkey=None,
                      payload=payload, commit_block=block)


def test_build_chain_status_document(cfg):
    from dataclasses import replace

    c = replace(cfg, round=replace(cfg.round, commit_floor_block=100))
    status = build_chain_status(
        c,
        current_block=14_500,
        commitments=[
            _commit(3, "hk-early", 14_000),
            _commit(7, "hk-late", 14_450),
            _commit(9, "hk-bad", 14_460, payload="garbage"),   # malformed → dropped
            _commit(1, "hk-prelaunch", 90),                    # below floor → dropped
        ],
        network="test",
        as_of="2026-07-19T00:00:00+00:00",
    )
    eb = c.round.epoch_blocks
    assert status["schema"] == 1
    assert status["network"] == "test"
    assert status["netuid"] == c.subnet.netuid
    assert status["current_block"] == 14_500
    assert status["epoch_blocks"] == eb
    assert status["epoch_start_block"] == (14_500 // eb) * eb
    assert status["block_time_s"] == 12.0  # the chain clock, not round_hours/epoch_blocks
    assert "stage_windows" not in status
    # newest first; only the two valid, post-floor commits survive
    assert [(s["uid"], s["commit_block"]) for s in status["submissions"]] == [
        (7, 14_450), (3, 14_000)]
    assert status["submissions"][0]["gen_ref"] == "ns/gen-7@sha256:" + "a" * 64
    # no economics passed → no key (older publishers keep producing valid docs)
    assert "economics" not in status


def test_build_chain_status_carries_economics(cfg):
    econ = {"alpha_price_tao": 0.0142,
            "tao_in_emission_per_day": 128.4, "as_of_block": 14_500}
    status = build_chain_status(
        cfg, current_block=14_500, commitments=[], economics=econ)
    assert status["economics"] == econ
    # None (chain flake / shape drift) degrades to the no-key document
    status = build_chain_status(
        cfg, current_block=14_500, commitments=[], economics=None)
    assert "economics" not in status


class _Store:
    def __init__(self, fail_acl=False):
        self.fail_acl = fail_acl
        self.writes = []

    def put_text(self, key, text, *, content_type="", acl=None):
        if acl and self.fail_acl:
            raise StorageError("acl unsupported")
        self.writes.append((key, text, acl))


def test_publish_chain_status_public_read_with_acl_fallback():
    store = _Store()
    assert publish_chain_status(store, {"schema": 1}) == CHAIN_STATUS_KEY
    assert store.writes[0][0] == CHAIN_STATUS_KEY
    assert store.writes[0][2] == "public-read"
    # a backend that rejects canned ACLs still gets the object (private)
    store = _Store(fail_acl=True)
    publish_chain_status(store, {"schema": 1})
    assert store.writes == [(CHAIN_STATUS_KEY, store.writes[0][1], None)]
