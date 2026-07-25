"""``status/chain.json`` — the dashboard's live chain-status document."""

from __future__ import annotations

from datetime import UTC

import pytest

from cascade.miner.dashboard import RoundTimeline
from cascade.shared.chain import Commitment
from cascade.shared.chain_status import (
    CHAIN_STATUS_KEY,
    ROUND_STATUS_KEY,
    build_chain_status,
    build_round_status,
    live_round_stage,
    publish_chain_status,
    publish_round_status,
    stage_windows,
)
from cascade.shared.hippius import StorageError


def _commit(uid, hotkey, block, payload=None):
    payload = payload if payload is not None else (
        f"metro-v1:gen:hippius:ns/gen-{uid}@sha256:{'a' * 64}"
    )
    return Commitment(uid=uid, hotkey=hotkey, coldkey=None,
                      payload=payload, commit_block=block)


def test_stage_windows_single_source_with_cli_timeline(cfg):
    heat_s, duel_s = stage_windows(cfg)
    tl = RoundTimeline.from_chain_config(cfg)
    assert (tl.heat_seconds, tl.duel_seconds) == (heat_s, duel_s)
    assert 0 < heat_s < duel_s


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
    assert status["block_time_s"] == c.round.round_hours * 3600.0 / eb
    heat_s, duel_s = stage_windows(c)
    assert status["stage_windows"] == {"heat_seconds": heat_s, "duel_seconds": duel_s}
    # newest first; only the two valid, post-floor commits survive
    assert [(s["uid"], s["commit_block"]) for s in status["submissions"]] == [
        (7, 14_450), (3, 14_000)]
    assert status["submissions"][0]["gen_ref"] == "ns/gen-7@sha256:" + "a" * 64


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


# ── trainer-reported round stage (status/round.json) ─────────────────────────


def _round_doc(**over):
    doc = build_round_status(
        round_id="99", epoch_start_block=14_400, stage="heat",
        as_of="2026-07-22T10:00:00+00:00", heat_done=12, heat_total=65,
    )
    doc.update(over)
    return doc


def test_build_round_status_document():
    doc = _round_doc()
    assert doc["schema"] == 1
    assert doc["round_id"] == "99"
    assert doc["epoch_start_block"] == 14_400
    assert doc["stage"] == "heat"
    assert (doc["heat_done"], doc["heat_total"]) == (12, 65)
    assert "finalists" not in doc  # optional fields stay out when unset
    with pytest.raises(ValueError):
        build_round_status(round_id="99", epoch_start_block=0,
                           stage="settled", as_of="")  # settled is receipt-only


def test_publish_round_status_public_read_with_acl_fallback():
    store = _Store()
    assert publish_round_status(store, _round_doc()) == ROUND_STATUS_KEY
    assert store.writes[0][:1] == (ROUND_STATUS_KEY,)
    assert store.writes[0][2] == "public-read"
    store = _Store(fail_acl=True)
    publish_round_status(store, _round_doc())
    assert store.writes == [(ROUND_STATUS_KEY, store.writes[0][1], None)]


def test_live_round_stage_accepts_fresh_matching_doc():
    now = 1_784_000_000.0  # any epoch-seconds anchor
    from datetime import datetime

    as_of = datetime.fromtimestamp(now - 60, tz=UTC).isoformat()
    doc = _round_doc(as_of=as_of)
    live = live_round_stage(doc, epoch_start_block=14_400, now_s=now)
    assert live is not None and live["stage"] == "heat"


def test_live_round_stage_rejects_stale_mismatched_and_malformed():
    now = 1_784_000_000.0
    from datetime import datetime

    old = datetime.fromtimestamp(now - 4000, tz=UTC).isoformat()
    future = datetime.fromtimestamp(now + 4000, tz=UTC).isoformat()
    ok = dict(epoch_start_block=14_400, now_s=now)
    assert live_round_stage(_round_doc(as_of=old), **ok) is None       # stale
    assert live_round_stage(_round_doc(as_of=future), **ok) is None    # skewed
    assert live_round_stage(_round_doc(), epoch_start_block=21_600, now_s=now) is None  # other round
    assert live_round_stage(_round_doc(stage="settled"), **ok) is None  # unknown stage
    assert live_round_stage(_round_doc(as_of="not-a-time"), **ok) is None
    assert live_round_stage(_round_doc(as_of="2026-07-22T10:00:00"), **ok) is None  # naive ts
    assert live_round_stage(None, **ok) is None
    assert live_round_stage("garbage", **ok) is None
