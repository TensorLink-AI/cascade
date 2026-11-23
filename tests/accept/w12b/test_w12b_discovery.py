"""W12b: `cascade discovery-check` lane D fields. Contract: README.md."""

from __future__ import annotations

import copy
import json

import pytest

from . import helpers as h


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    return tmp_path_factory.mktemp("w12b-discovery")


@pytest.fixture(scope="module")
def doc(root):
    return h.lane_d_doc(root)


def _checks(all_docs):
    from cascade.miner.discovery_check import check_snapshot

    return check_snapshot(h.snapshot(all_docs))


def _with(doc, *, age=40):
    d = h.docs(current_block=doc["block"] + age)
    d["lane_d_queue"] = copy.deepcopy(doc)
    return d


def test_active_lane_d_requires_a_fresh_lane_d_queue(doc):
    checks = _checks(_with(doc))
    assert checks.get("lane_d_queue") is True


def test_stale_lane_d_queue_fails_the_release_check(doc):
    checks = _checks(_with(doc, age=h.ERA_BLOCKS + 1))
    assert checks.get("lane_d_queue") is False


def test_future_dated_lane_d_queue_fails_the_release_check(doc):
    checks = _checks(_with(doc, age=-1))
    assert checks.get("lane_d_queue") is False


def test_closed_lane_d_is_not_checked(root):
    checks = _checks(_with(h.closed_doc(root)))
    assert "lane_d_queue" not in checks


def test_discovery_checks_unchanged_when_unarmed(doc):
    """No status/lane_d.json (W17 publishes nothing unarmed): the check set
    and every value are today's."""
    from cascade.miner.discovery_check import check_snapshot

    d = h.docs(current_block=h.DOC_BLOCK)
    checks = check_snapshot(h.snapshot(d))
    assert "lane_d_queue" not in checks
    assert set(checks) == {"fresh_feeds", "current_init", "effective_contract", "agent_protocol",
                           "submission_tracking", "lane_a_queue"}


def test_discovery_checks_armed_nonempty_lane_d_effect(doc):
    from cascade.miner.discovery_check import check_snapshot

    unarmed = check_snapshot(h.snapshot(h.docs(current_block=doc["block"] + 40)))
    armed = check_snapshot(h.snapshot(_with(doc)))
    assert set(armed) == set(unarmed) | {"lane_d_queue"}
    assert {k: armed[k] for k in unarmed} == unarmed
    assert armed["lane_d_queue"] is True


def test_discovery_check_cli_reports_lane_d_remaining(monkeypatch, capsys, doc):
    """End to end: a stale lane D queue lands in ``remaining`` (and the
    command reports the release incomplete); no write is performed."""
    from cascade.miner import cli

    snap = h.snapshot(_with(doc, age=h.ERA_BLOCKS + 1))
    monkeypatch.setattr("cascade.miner.discovery_check.fetch_snapshot",
                        lambda *a, **kw: copy.deepcopy(snap))
    monkeypatch.setattr("cascade.miner.discovery_check.read_guide", lambda *a, **kw: "")
    rc = cli.main(["discovery-check", "--json"])
    out = json.loads(capsys.readouterr().out)
    assert rc != 0
    data = out.get("data") or {}
    assert data.get("checks", {}).get("lane_d_queue") is False
    assert "lane_d_queue" in data.get("remaining", [])
    assert data.get("writes_performed") is False
