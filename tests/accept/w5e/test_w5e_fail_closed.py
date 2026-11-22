"""W5e (RV5 correction): an ARMED crowning fails closed without its gate
inputs (plan rev6 §1.3b, §3 invariants; review evidence/rv5.md "settle_lane_d
without gate kwargs"). README.md has the contract.

- Armed, a member that clears its confirmation duel is a crowning candidate.
  If the settlement has no `bench` session or no `boundary_block_hash`, the
  §1.3b gate cannot run, so the candidate is not crowned: it ends
  `bench_hold` (open, not lost, not the miner's fault) and the report carries
  no crown. The legacy call (no gate keywords at all) is the same case.
- A crowned report without a gate record (`gate` absent or `null`) is
  refused by `validate_report`, so `verify_report` (the satellites' accept
  path) refuses it too.
- Unarmed, the legacy call is unchanged: `report=None, pay=None`, nothing
  written.

On main the gate keywords don't exist: the gated calls fail on the
AssertionError in `helpers.settle_gated`, and the legacy armed call crowns
without a gate, which the assertions refuse.
"""

from __future__ import annotations

import copy

import pytest

from .helpers import (
    ARM_BLOCK,
    T_X,
    BenchHost,
    bench_session,
    bench_table,
    block_hash,
    boundary,
    entry,
    entry_of,
    fixture_key,
    house,
    lane_d_cfg,
    model_forecasts,
    passing_bench,
    read_state,
    rows_of,
    settle,
    settle_crowning,
    settle_gated,
    snapshot,
    uids,
    verdict_of,
)


def _fc(recs, models: dict) -> dict:
    return {dg: model_forecasts(recs, q, name) for dg, (q, name) in models.items()}


def _first_clear(tmp_path):
    """X first-clears against the genesis house at boundary(1). That is not a
    crowning candidate, so the legacy (ungated) call is still allowed here."""
    cfg = lane_d_cfg()
    h0 = house()
    x = entry("x", reveal_ts=T_X)
    path = tmp_path / "state" / "lane_d_state.json"
    models = {h0["digest"]: ("bad", "house-0"), x["digest"]: ("sharp", "x")}
    pool1, recs1 = snapshot(tmp_path, 1)
    s1 = settle(cfg, boundary(1), pool1, [x], h0, _fc(recs1, models), uids(x), path)
    v = verdict_of(s1.report, x["digest"])
    assert v is not None and v["stage"] == "first" and v["outcome"] == "confirm_pending", \
        f"fixture: X did not first-clear: {v}"
    return cfg, h0, x, path, models


def _missing_inputs(variant, cfg, b, pool, ents, h0, fc, uid_of, path, x, h0d):
    """Settle boundary `b` with the gate inputs `variant` leaves out."""
    if variant == "legacy_call":
        return settle(cfg, b, pool, ents, h0, fc, uid_of, path)
    healthy = passing_bench([x], [h0d], "sess-b2")
    if variant == "bench_none":
        return settle_gated(cfg, b, pool, ents, h0, fc, uid_of, path,
                            hash_=block_hash("b2"), bench=None)
    if variant == "hash_none":
        return settle_gated(cfg, b, pool, ents, h0, fc, uid_of, path, hash_=None, bench=healthy)
    assert variant == "both_none"
    return settle_gated(cfg, b, pool, ents, h0, fc, uid_of, path, hash_=None, bench=None)


VARIANTS = ("legacy_call", "bench_none", "hash_none", "both_none")


@pytest.mark.parametrize("variant", VARIANTS)
def test_armed_crowning_without_bench_or_block_hash_is_bench_hold(tmp_path, variant):
    cfg, h0, x, path, models = _first_clear(tmp_path)
    assert cfg.active(boundary(2)), "fixture: the settlement must be armed"
    pool2, recs2 = snapshot(tmp_path, 2)
    s2 = _missing_inputs(variant, cfg, boundary(2), pool2, [x], h0, _fc(recs2, models),
                         uids(x), path, x["digest"], h0["digest"])
    rep = s2.report
    v = verdict_of(rep, x["digest"])
    assert v is not None and v["stage"] == "confirm" and v["cleared"] is True, \
        f"fixture: X did not clear its confirmation duel: {v}"
    # No crown: the §1.3b gate could not run.
    assert rep["crowned"] is None, \
        f"{variant}: an armed settlement crowned without the §1.3b gate inputs"
    assert rep["king_after"]["digest"] == rep["king_before"]["digest"] == h0["digest"]
    assert v["outcome"] not in ("crowned", "lost"), v["outcome"]
    gate = rep.get("gate")
    if gate is not None:
        passed = [r for r in gate.get("results") or [] if r.get("outcome") == "pass"]
        assert not passed, f"{variant}: a gate result passed without a bench or a block hash"
    assert s2.pay["court_uids"][0] is None, "the house still holds KING_D (its slot burns)"
    state = read_state(path)
    assert state["king"]["digest"] == h0["digest"] and state["king"]["house"]
    # bench_hold: typed, open, not the miner's fault.
    rec = state["entries"][x["digest"]]
    assert rec["reason"] == "bench_hold", (rec["status"], rec["reason"])
    assert rec["status"] in ("hold", "confirm_pending"), \
        f"{variant}: a missing gate input is not the miner's fault, the entry must stay open"
    e = entry_of(rep, x["digest"])
    assert (e["status"], e["reason"]) == (rec["status"], "bench_hold")
    # Not charged: with the inputs back and no fresh window, the held clear is
    # re-gated on the next boundary's draw and crowns without a new duel.
    hash3 = block_hash("b3")
    host3 = BenchHost(bench_table({x["digest"]: 0.9, h0["digest"]: 1.0}))
    s3 = settle_gated(cfg, boundary(3), pool2, [x], h0, _fc(recs2, models), uids(x), path,
                      hash_=hash3, bench=bench_session(host3, "sess-b3"))
    assert s3.report["crowned"] == x["digest"], \
        f"{variant}: the hold burned the candidate's chance"
    assert rows_of(s3.report, x["digest"]) == set(), \
        "a held candidate was re-duelled instead of re-gated"
    assert x["digest"] in host3.swept_digests()
    assert read_state(path)["king"]["digest"] == x["digest"]


def test_unarmed_legacy_call_is_unchanged(tmp_path):
    """The unarmed path takes no gate inputs and still returns nothing."""
    h0 = house()
    x = entry("x", reveal_ts=T_X)
    models = {h0["digest"]: ("bad", "house-0"), x["digest"]: ("sharp", "x")}
    pool1, recs1 = snapshot(tmp_path, 1)
    fc = _fc(recs1, models)
    cases = {"from_block_0": (lane_d_cfg(from_block=0), boundary(2)),
             "at_from_block": (lane_d_cfg(), ARM_BLOCK)}
    for name, (cfg, b) in cases.items():
        assert not cfg.active(b), f"fixture: {name} must be unarmed"
        path = tmp_path / name / "lane_d_state.json"
        s = settle(cfg, b, pool1, [x], h0, fc, uids(x), path)
        assert (s.report, s.pay) == (None, None), name
        assert not path.exists(), f"{name}: an unarmed settlement wrote state"


def _crowned_report(tmp_path):
    cfg, h0, x, path, models = _first_clear(tmp_path)
    pool2, recs2 = snapshot(tmp_path, 2)
    s2 = settle_crowning(cfg, boundary(2), pool2, [x], h0, _fc(recs2, models), uids(x), path,
                         candidates=[x["digest"]], references=[h0["digest"]], tag="b2")
    assert s2.report["crowned"] == x["digest"], "fixture: the gated settlement did not crown"
    assert isinstance(s2.report.get("gate"), dict), "a gated crowning carries its gate record"
    return s2.report


@pytest.mark.parametrize("mutation", ("absent", "null"))
def test_a_crowned_report_without_a_gate_record_is_refused(tmp_path, mutation):
    from cascade.validator.lane_d_report import (
        ReportSchemaError,
        sign_report,
        validate_report,
        verify_report,
    )

    rep = copy.deepcopy(_crowned_report(tmp_path))
    if mutation == "absent":
        rep.pop("gate")
    else:
        rep["gate"] = None
    signer = fixture_key("signer")
    forged = sign_report(rep, signer)
    refused = None
    try:
        validate_report(forged)
    except ReportSchemaError as e:
        refused = e
    assert refused is not None, \
        f"a crowned report with gate {mutation} validated: an ungated crown must be refused"
    assert verify_report(forged, signer.ss58_address) is False


def test_an_uncrowned_report_needs_no_gate_record(tmp_path):
    """The refusal is tied to the crown: a settlement without a crowning
    candidate has no gate and validates. Here X's confirmation fails."""
    from cascade.validator.lane_d_report import validate_report

    cfg, h0, x, path, _ = _first_clear(tmp_path)
    pool2, recs2 = snapshot(tmp_path, 2)
    models = {h0["digest"]: ("fair", "house-0"), x["digest"]: ("bad", "x")}
    host = BenchHost(bench_table({x["digest"]: 0.9, h0["digest"]: 1.0}))
    s2 = settle_gated(cfg, boundary(2), pool2, [x], h0, _fc(recs2, models), uids(x), path,
                      hash_=block_hash("b2"), bench=bench_session(host, "sess-b2"))
    assert s2.report["crowned"] is None and host.calls == []
    assert s2.report.get("gate") is None
    validate_report(s2.report)
