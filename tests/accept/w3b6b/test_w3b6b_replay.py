"""W3b6b, re-derive: W7d's replay checks every bound bench record against
its digest, its role's digest and the result's session, and recomputes the
four LCBs of every pass / gate_regression result from the published bytes
(``check_gate_rows``, wired into ``replay_chain(..., bench_rows=)``). A
crowned pass without ``rows`` is refused. Mismatches are paths
``gate.results[i].rows`` / ``gate.results[i].lcb``. Contract: README.md."""

from __future__ import annotations

import copy
import inspect
import json

import pytest

from . import helpers as h


@pytest.fixture(scope="module")
def gated(tmp_path_factory):
    r, rep = h.gated(tmp_path_factory.mktemp("w3b6b_replay"))
    return r, rep, h.persisted_rows(r)


def _check(rep, rows, r):
    return list(h.check_rows_fn()(rep, rows, r.cfg.lane_d))


def _bound(rep) -> dict:
    (res,) = h.gate_results(rep)
    assert isinstance(res.get("rows"), dict), "W3b6b: the gate result binds its records"
    return res["rows"]


def _hit(paths, path) -> bool:
    return any(p == path or p.startswith(path + ".") for p in paths)


def test_replay_chain_takes_bench_rows():
    p = inspect.signature(h.replay_mod().replay_chain).parameters
    assert "bench_rows" in p and p["bench_rows"].default is None, \
        "W3b6b: replay_chain(..., bench_rows=None)"


def test_honest_rows_match(gated, tmp_path):
    r, rep, rows = gated
    _bound(rep)
    assert _check(rep, rows, r) == []
    d = tmp_path / "rows"
    d.mkdir()
    for dg, data in rows.items():
        (d / f"{dg}.json").write_bytes(data)
    assert _check(rep, d, r) == [], "a directory of <sha256>.json files works too"


def test_tampered_record_is_a_rows_mismatch(gated):
    r, rep, rows = gated
    rows = dict(rows)
    cand = _bound(rep)["candidate"]
    rec = json.loads(rows[cand])
    rec["suites"][rep["gate"]["suite"]]["rows"][0]["crps"] *= 0.5
    rows[cand] = h.json_text(rec).encode("ascii")
    assert _hit(_check(rep, rows, r), "gate.results[0].rows")


def test_rebound_forged_record_is_an_lcb_mismatch(gated):
    """Doctored rows AND a report re-binding their digest: the recomputed LCBs
    disagree with the signed ones."""
    r, rep, rows = gated
    rows, rep = dict(rows), copy.deepcopy(rep)
    bound = _bound(rep)
    rec = json.loads(rows[bound["candidate"]])
    for x in rec["suites"][rep["gate"]["suite"]]["rows"]:
        x["crps"] *= 0.7
    text = h.json_text(rec)
    rows[h.sha(text)] = text.encode("ascii")
    rep["gate"]["results"][0]["rows"]["candidate"] = h.sha(text)
    assert _hit(_check(rep, rows, r), "gate.results[0].lcb")


def test_doctored_lcb_with_honest_rows_is_a_mismatch(gated):
    r, rep, rows = gated
    rep = copy.deepcopy(rep)
    _bound(rep)
    rep["gate"]["results"][0]["lcb"]["king_d"]["crps"] += 0.001
    assert _hit(_check(rep, rows, r), "gate.results[0].lcb")


@pytest.mark.parametrize("how", ["dropped", "unknown_digest", "wrong_role", "session"])
def test_missing_or_misplaced_rows_are_refused(gated, how):
    r, rep, rows = gated
    rep, rows = copy.deepcopy(rep), dict(rows)
    bound = _bound(rep)
    res = rep["gate"]["results"][0]
    assert rep.get("crowned") == res["digest"], "the fixture crowns the gated candidate"
    if how == "dropped":
        res.pop("rows")
    elif how == "unknown_digest":
        res["rows"]["house"] = "e" * 64
    elif how == "wrong_role":
        res["rows"]["king_d"], res["rows"]["candidate"] = bound["candidate"], bound["king_d"]
    else:
        res["session"] = "another-session"
    assert _hit(_check(rep, rows, r), "gate.results[0].rows")


def test_check_reads_only(gated):
    r, rep, rows = gated
    _bound(rep)
    snap_rep, snap_rows = copy.deepcopy(rep), copy.deepcopy(rows)
    _check(rep, rows, r)
    assert rep == snap_rep and rows == snap_rows, "pure: no input is mutated"
