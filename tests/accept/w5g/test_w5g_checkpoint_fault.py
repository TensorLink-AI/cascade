"""W5g: `checkpoint_fault` closes the entry as the miner's fault and nothing
gates it again (plan rev6 §1.3b "Faults", §10.4 "`checkpoint_fault`: entry
refused; evidence kept", T17 "`checkpoint_fault`; KING_D not held").
README.md has the contract.

Three ways in: the candidate's rows on the drawn suite are NaN at its
confirmation (a judged candidate), the same on the re-gate of a held clear
(a typed skip), and malformed eval forecasts at selection (a typed skip).
In every case the entry is `closed` / `checkpoint_fault` in state and
report, never `bench_hold`, never crowned, and at later settlements it is
neither selected, judged, skipped again nor swept.
"""

from __future__ import annotations

import numpy as np

from .helpers import (
    T_X,
    Flow,
    boundary,
    entry,
    entry_of,
    forecasts,
    gate_inputs,
    house,
    lane_d_cfg,
    read_state,
    rows_of,
    settle,
    skips_of,
    snapshot,
    uids,
    verdict_of,
)


def _assert_closed_fault(path, report, dg):
    rec = read_state(path)["entries"][dg]
    assert (rec["status"], rec["reason"]) == ("closed", "checkpoint_fault"), rec
    assert not rec.get("gate_hold"), "a faulted entry keeps no held clear"
    e = entry_of(report, dg)
    assert (e["status"], e["reason"]) == ("closed", "checkpoint_fault")
    assert report["crowned"] != dg


def _assert_never_again(f: Flow, snap: int):
    """A later settlement with fresh windows and a healthy bench: the faulted
    entry is not selected, judged, skipped or swept, and its record stands."""
    before = read_state(f.path)["entries"][f.x["digest"]]
    report = f.next("pass", snap=snap)
    dg = f.x["digest"]
    assert dg not in report["judged"] and verdict_of(report, dg) is None
    assert rows_of(report, dg) == set()
    assert skips_of(report, dg) == [], "a closed entry is not considered again"
    host = f.hosts[f.n]
    assert dg not in host.swept_digests(), "a faulted entry was gated again"
    assert read_state(f.path)["entries"][dg] == before
    assert entry_of(report, dg)["reason"] == "checkpoint_fault"
    assert report["crowned"] is None
    assert f.king() == f.h0["digest"]


def test_t17_fault_at_the_confirmation_closes_and_is_never_gated_again(tmp_path):
    f = Flow(tmp_path, mode2="candidate_fault")
    _assert_closed_fault(f.path, f.reports[2], f.x["digest"])
    assert skips_of(f.reports[2], f.x["digest"]) == [], \
        "a judged candidate's fault is its entry reason, not a skip"
    assert f.king() == f.h0["digest"], "T17: KING_D is not held by a candidate fault"
    _assert_never_again(f, snap=3)


def test_fault_on_the_regate_of_a_held_clear_closes(tmp_path):
    f = Flow(tmp_path, mode2="no_bench")
    assert f.state_entry()["reason"] == "bench_hold"
    report = f.next("candidate_fault")
    _assert_closed_fault(f.path, report, f.x["digest"])
    assert skips_of(report, f.x["digest"]) == ["checkpoint_fault"], \
        "a held clear re-gated (not judged) that faults is a typed skip"
    assert verdict_of(report, f.x["digest"]) is None
    _assert_never_again(f, snap=3)


def test_malformed_forecasts_close_at_selection(tmp_path):
    cfg = lane_d_cfg()
    h0 = house()
    x = entry("x", reveal_ts=T_X)
    path = tmp_path / "state" / "lane_d_state.json"
    models = {h0["digest"]: ("bad", "house-0"), x["digest"]: ("sharp", "x")}
    pool1, recs1 = snapshot(tmp_path, 1)
    fc = forecasts(recs1, models)
    first = sorted(fc[x["digest"]])[0]
    fc[x["digest"]][first] = np.full_like(fc[x["digest"]][first], np.nan)
    hash_, bench, _ = gate_inputs("pass", x["digest"], h0["digest"], "b1")
    r1 = settle(cfg, boundary(1), pool1, [x], h0, fc, uids(x), path, hash_=hash_, bench=bench)
    _assert_closed_fault(path, r1.report, x["digest"])
    assert skips_of(r1.report, x["digest"]) == ["checkpoint_fault"]
    assert verdict_of(r1.report, x["digest"]) is None
    before = read_state(path)["entries"][x["digest"]]
    pool2, recs2 = snapshot(tmp_path, 2)
    hash_, bench, host = gate_inputs("pass", x["digest"], h0["digest"], "b2")
    r2 = settle(cfg, boundary(2), pool2, [x], h0, forecasts(recs2, models), uids(x), path,
                hash_=hash_, bench=bench)
    assert verdict_of(r2.report, x["digest"]) is None and skips_of(r2.report, x["digest"]) == []
    assert x["digest"] not in host.swept_digests()
    assert read_state(path)["entries"][x["digest"]] == before
