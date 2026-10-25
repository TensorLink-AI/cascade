"""Fresh discovery must not turn missing, stale or future state into admission."""
from __future__ import annotations

import copy
import json
from datetime import UTC, datetime, timedelta

import pytest

from cascade.miner import cli
from cascade.miner.live_status import status_snapshot

NOW = datetime.now(UTC)
PTR = "metro-v1:trained:hippius:cascade/init@sha256:" + "a" * 64


def docs():
    return {"chain": {"as_of": NOW.isoformat(), "current_block": 500, "era_blocks": 100},
            "competition": {"as_of": NOW.isoformat(), "block": 500,
                            "warm_start": {"pointer": PTR, "mode": "checkpoint",
                                           "era_start_block": 500},
                            "policy": {"new_rule": {"budget": 42}},
                            "lane_b": {"active": False}}}


def test_live_snapshot_preserves_future_policy_and_missing_queues():
    data = status_snapshot(docs(), now=NOW, max_age=180)
    assert data["fresh"] and data["ready_for_training"]
    assert data["policy"]["new_rule"]["budget"] == 42
    assert data["queues"] == {"a": None, "b": None, "c": None}
    assert data["lanes"]["b"]["active"] is False


@pytest.mark.parametrize("seconds", [-300, 60])
def test_stale_and_future_dated_feeds_are_not_current(seconds):
    d = docs()
    d["competition"]["as_of"] = (NOW + timedelta(seconds=seconds)).isoformat()
    data = status_snapshot(d, now=NOW, max_age=180)
    assert not data["fresh"] and not data["ready_for_training"]


@pytest.mark.parametrize("change", [{"era_start_block": 400}, {"era_start_block": 600},
                                     {"pointer": "bad"}, {"pointer": ""}])
def test_old_future_or_invalid_checkpoint_is_not_a_current_init(change):
    d = docs()
    d["competition"]["warm_start"].update(change)
    assert not status_snapshot(d, now=NOW, max_age=180)["ready_for_training"]


def test_explicit_random_init_and_unknown_are_distinct():
    d = docs()
    d["competition"]["warm_start"].update(pointer="", mode="random_init")
    assert status_snapshot(d, now=NOW, max_age=180)["ready_for_training"]
    d["competition"].pop("warm_start")
    assert not status_snapshot(d, now=NOW, max_age=180)["ready_for_training"]


@pytest.mark.parametrize("missing", [None, "yesterday", "2026-01-01T00:00:00"])
def test_unusable_timestamp_never_looks_fresh(missing):
    d = docs()
    d["chain"]["as_of"] = missing
    assert not status_snapshot(d, now=NOW, max_age=180)["fresh"]


@pytest.mark.parametrize("available", [True, False])
def test_status_cli_json_reads_live_without_a_wallet(monkeypatch, capsys, available):
    d = docs() if available else {}
    for doc in d.values():
        doc["as_of"] = datetime.now(UTC).isoformat()
    by_path = {"status/chain.json": d.get("chain"), "status/v2.json": d.get("competition")}
    reads = []

    def fetch(storage, key, **_):
        reads.append(key)
        return copy.deepcopy(by_path.get(key))

    monkeypatch.setattr("cascade.miner.dashboard.fetch_public_json", fetch)
    rc = cli.main(["status", "--json"])
    out = json.loads(capsys.readouterr().out)
    assert rc == (0 if available else 3)
    assert out["command"] == "status" and out["data"]["ready_for_training"] == available
    assert len(reads) == 5
    if not available:
        assert out["error"]["code"] == "live_status_unavailable"


def test_each_invocation_refetches_public_state(monkeypatch, capsys):
    reads = []
    monkeypatch.setattr("cascade.miner.dashboard.fetch_public_json",
                        lambda storage, key, **kw: reads.append(key))
    cli.main(["status", "--json"])
    cli.main(["status", "--json"])
    assert len(reads) == 10


def test_live_scoring_prefers_published_current_init(monkeypatch, cfg):
    from cascade.miner.score import resolve_live_warm_start
    from cascade.shared.manifest import contract_digest, contract_payload

    d = docs()
    for doc in d.values():
        doc["as_of"] = datetime.now(UTC).isoformat()
    d["competition"]["training_contract"] = {"schema": 1, "era_start_block": 500,
        "body": contract_payload(cfg.training.primary_size), "digest": contract_digest(cfg.training.primary_size)}
    seen = []

    def fetch(storage, key, **kw):
        seen.append(key)
        return d["competition"] if key == "status/v2.json" else d["chain"]

    monkeypatch.setattr("cascade.miner.dashboard.fetch_public_json", fetch)
    assert resolve_live_warm_start(cfg) == PTR
    assert len(seen) == 5 and "receipts/latest.json" not in seen


def test_live_scoring_never_falls_back_when_new_feed_is_stale(monkeypatch, cfg):
    from cascade.miner.live_status import LiveStateUnavailable
    from cascade.miner.score import resolve_live_warm_start

    d = docs()
    d["competition"]["as_of"] = (NOW - timedelta(hours=1)).isoformat()
    monkeypatch.setattr("cascade.miner.dashboard.fetch_public_json",
                        lambda storage, key, **kw: d["competition"] if key == "status/v2.json" else d["chain"])
    with pytest.raises(LiveStateUnavailable, match="stale"):
        resolve_live_warm_start(cfg)


def test_live_scoring_network_failure_is_not_random_init(monkeypatch, cfg):
    from cascade.miner.live_status import LiveStateUnavailable
    from cascade.miner.score import resolve_live_warm_start

    monkeypatch.setattr("cascade.miner.dashboard.fetch_public_json", lambda *a, **k: None)
    with pytest.raises(LiveStateUnavailable, match="stale or unconfirmed"):
        resolve_live_warm_start(cfg)
