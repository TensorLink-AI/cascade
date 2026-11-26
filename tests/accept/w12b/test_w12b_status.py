"""W12b: `cascade status --json` lane D fields. Contract: README.md."""

from __future__ import annotations

import copy
import json

import pytest

from . import helpers as h


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    return tmp_path_factory.mktemp("w12b-status")


@pytest.fixture(scope="module")
def doc(root):
    return h.lane_d_doc(root)


def _armed(doc, *, age=40):
    d = h.docs(current_block=doc["block"] + age)
    d["lane_d_queue"] = copy.deepcopy(doc)
    return d


# ── the source ──────────────────────────────────────────────────────────────


def test_lane_d_source_is_the_w17_status_key():
    from cascade.miner.live_status import SOURCES
    from cascade.validator.lane_d_status import LANE_D_STATUS_KEY

    assert SOURCES.get("lane_d_queue") == LANE_D_STATUS_KEY == h.LANE_D_KEY


def test_todays_sources_are_kept():
    from cascade.miner.live_status import SOURCES

    assert {k: SOURCES.get(k) for k in h.TODAY_SOURCES} == h.TODAY_SOURCES
    assert set(SOURCES) == set(h.TODAY_SOURCES) | {"lane_d_queue"}


# ── armed: the queue, the lane summary, the block-clock freshness ───────────


def test_queue_d_is_the_published_document_verbatim(doc):
    q = h.queue_d(h.snapshot(_armed(doc)))
    assert q == doc
    assert [r["digest"] for r in q["queue"]] == [r["digest"] for r in doc["queue"]]


def test_lane_d_summary_restates_the_document(doc):
    lane = h.lane_d(h.snapshot(_armed(doc)))
    assert isinstance(lane, dict)
    assert set(lane) >= h.LANE_D_SUMMARY_KEYS, sorted(h.LANE_D_SUMMARY_KEYS - set(lane))
    assert lane["active"] is True
    for k in ("lane", "schema", "block", "era", "boundary", "king", "house", "floors"):
        assert lane[k] == doc[k], k


def test_a_closed_lane_is_not_active(root):
    doc = h.closed_doc(root)
    snap = h.snapshot(_armed(doc))
    lane = h.lane_d(snap)
    assert isinstance(lane, dict)
    assert lane["active"] is False and lane["lane"] == "closed"
    assert h.queue_d(snap) == doc


def test_lane_d_freshness_is_on_the_block_clock(doc):
    """W17's document carries no wall-clock stamp (by design), so lane D
    freshness is measured in blocks against status/chain.json's head."""
    fr = h.lane_d_freshness(h.snapshot(_armed(doc, age=40)))
    assert set(fr) >= h.LANE_D_FRESH_KEYS
    assert {k: fr[k] for k in h.LANE_D_FRESH_KEYS} == {
        "available": True, "block": doc["block"], "age_blocks": 40, "fresh": True}


@pytest.mark.parametrize("age", [0, 1, h.ERA_BLOCKS])
def test_a_document_up_to_one_era_old_is_fresh(doc, age):
    fr = h.lane_d_freshness(h.snapshot(_armed(doc, age=age)))
    assert fr.get("fresh") is True and fr.get("age_blocks") == age


@pytest.mark.parametrize("age", [h.ERA_BLOCKS + 1, -1, -900])
def test_an_older_or_future_document_is_not_fresh(doc, age):
    fr = h.lane_d_freshness(h.snapshot(_armed(doc, age=age)))
    assert fr.get("available") is True and fr.get("fresh") is False
    assert fr.get("age_blocks") == age


def test_no_chain_head_means_not_fresh(doc):
    d = _armed(doc)
    d["chain"].pop("current_block")
    fr = h.lane_d_freshness(h.snapshot(d))
    assert fr.get("available") is True and fr.get("fresh") is False
    assert fr.get("age_blocks") is None


def test_a_stale_lane_d_document_never_makes_the_snapshot_stale(doc):
    """Lane D is optional discovery: the top-level ``fresh`` /
    ``ready_for_training`` stay lane A's (chain + competition)."""
    snap = h.snapshot(_armed(doc, age=10 * h.ERA_BLOCKS))
    assert h.lane_d_freshness(snap).get("fresh") is False
    assert snap["fresh"] is True and snap["ready_for_training"] is True


def test_snapshot_is_deterministic_and_does_not_mutate_inputs(doc):
    from cascade.miner.live_status import status_snapshot

    d = _armed(doc)
    before = json.dumps(d, sort_keys=True)
    a = status_snapshot(d, now=h.NOW, max_age=180.0)
    b = status_snapshot(d, now=h.NOW, max_age=180.0)
    assert json.dumps(d, sort_keys=True) == before
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert h.queue_d(a) == doc


# ── unarmed: nothing published, today's snapshot ────────────────────────────


def test_status_snapshot_unchanged_when_unarmed():
    """W17 publishes nothing while unarmed: lane D reads as absent and every
    field today's snapshot has keeps today's value."""
    from cascade.miner.live_status import freshness

    d = h.docs(current_block=h.DOC_BLOCK)
    snap = h.snapshot(d)
    assert snap["fresh"] is True and snap["ready_for_training"] is True
    for name in h.TODAY_SOURCES:
        assert snap["freshness"][name] == freshness(d.get(name), now=h.NOW, max_age=180.0), name
    assert {k: snap["queues"][k] for k in ("a", "b", "c")} == {"a": None, "b": None, "c": None}
    assert snap["lanes"]["b"] == {"active": False} and snap["lanes"]["c"] == {"active": False}
    assert snap["queues"].get("d") is None and snap["lanes"].get("d") is None
    fr = snap["freshness"].get("lane_d_queue")
    assert fr is None or fr.get("fresh") is False


def test_status_snapshot_armed_nonempty_lane_d_effect(doc):
    snap = h.snapshot(_armed(doc))
    q = h.queue_d(snap)
    assert q == doc and q["queue"]
    assert h.lane_d(snap)["active"] is True
    assert h.lane_d_freshness(snap).get("fresh") is True
    # lane A's view is untouched by lane D
    unarmed = h.snapshot(h.docs(current_block=doc["block"] + 40))
    for k in ("fresh", "ready_for_training", "warm_start", "chain", "king"):
        assert snap[k] == unarmed[k], k
    assert {x: snap["queues"][x] for x in "abc"} == {x: unarmed["queues"][x] for x in "abc"}


# ── the CLI: `cascade status --json` ────────────────────────────────────────


def _run_status(monkeypatch, capsys, published: dict) -> tuple[int, dict, list[str]]:
    from cascade.miner import cli

    reads: list[str] = []

    def fetch(storage, key, **_):
        reads.append(key)
        return copy.deepcopy(published.get(key))

    monkeypatch.setattr("cascade.miner.dashboard.fetch_public_json", fetch)
    rc = cli.main(["status", "--json"])
    return rc, json.loads(capsys.readouterr().out), reads


def _published(doc, *, current_block):
    from datetime import UTC, datetime

    d = h.docs(current_block=current_block, now=datetime.now(UTC))
    out = {"status/chain.json": d["chain"], "status/v2.json": d["competition"]}
    if doc is not None:
        out[h.LANE_D_KEY] = doc
    return out


def test_status_json_reads_and_reports_lane_d(monkeypatch, capsys, doc):
    rc, out, reads = _run_status(monkeypatch, capsys,
                                 _published(doc, current_block=doc["block"] + 40))
    assert rc == 0, out
    assert h.LANE_D_KEY in reads
    data = out["data"]
    assert h.queue_d(data) == doc
    assert h.lane_d(data)["active"] is True
    assert h.lane_d_freshness(data).get("fresh") is True
    assert str(data["sources"].get("lane_d_queue", "")).endswith("/" + h.LANE_D_KEY)


def test_status_json_without_lane_d_still_succeeds(monkeypatch, capsys):
    rc, out, _ = _run_status(monkeypatch, capsys, _published(None, current_block=h.DOC_BLOCK))
    assert rc == 0, out
    assert out["data"]["fresh"] is True
    assert out["data"]["queues"].get("d") is None
    assert out["data"]["lanes"].get("d") is None
