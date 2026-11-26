"""W5e (RV5 correction, rev6.11): an ARMED crowning fails closed without its
gate inputs (plan rev6 §1.3b, §3 invariants; review evidence/rv5.md
"settle_lane_d without gate kwargs"). README.md has the contract.

- Armed, a member that clears at its first verdict is a crowning candidate
  (rev6.11 (b): no confirmation duel). If the settlement has no `bench`
  session or no `boundary_block_hash`, the §1.3b gate cannot run, so the
  candidate is not crowned: it ends `bench_hold` (open, not lost, not the
  miner's fault) and the report carries no crown. The legacy call (no gate
  keywords at all) is the same case.
- A crowned report without a gate record (`gate` absent or `null`) is
  refused by `validate_report`, so `verify_report` (the satellites' accept
  path) refuses it too.
- Unarmed, the legacy call is unchanged: `report=None, pay=None`, nothing
  written.

On main a first clear is `confirm_pending`, never a candidate, so the
assertions on the first-verdict hold and crown fail.
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


def _base(tmp_path):
    cfg = lane_d_cfg()
    h0 = house()
    x = entry("x", reveal_ts=T_X)
    path = tmp_path / "state" / "lane_d_state.json"
    models = {h0["digest"]: ("bad", "house-0"), x["digest"]: ("sharp", "x")}
    pool1, recs1 = snapshot(tmp_path, 1)
    return cfg, h0, x, path, models, pool1, recs1


def _missing_inputs(variant, cfg, b, pool, ents, h0, fc, uid_of, path, x, h0d):
    """Settle boundary `b` with the gate inputs `variant` leaves out."""
    if variant == "legacy_call":
        return settle(cfg, b, pool, ents, h0, fc, uid_of, path)
    healthy = passing_bench([x], [h0d], "sess-b1")
    if variant == "bench_none":
        return settle_gated(cfg, b, pool, ents, h0, fc, uid_of, path,
                            hash_=block_hash("b1"), bench=None)
    if variant == "hash_none":
        return settle_gated(cfg, b, pool, ents, h0, fc, uid_of, path, hash_=None, bench=healthy)
    assert variant == "both_none"
    return settle_gated(cfg, b, pool, ents, h0, fc, uid_of, path, hash_=None, bench=None)


VARIANTS = ("legacy_call", "bench_none", "hash_none", "both_none")


@pytest.mark.parametrize("variant", VARIANTS)
def test_armed_crowning_without_bench_or_block_hash_is_bench_hold(tmp_path, variant):
    cfg, h0, x, path, models, pool1, recs1 = _base(tmp_path)
    assert cfg.active(boundary(1)), "fixture: the settlement must be armed"
    s1 = _missing_inputs(variant, cfg, boundary(1), pool1, [x], h0, _fc(recs1, models),
                         uids(x), path, x["digest"], h0["digest"])
    rep = s1.report
    v = verdict_of(rep, x["digest"])
    assert v is not None and v.get("stage", "first") == "first" and v["cleared"] is True, \
        f"fixture: X did not clear at its first verdict: {v}"
    # No crown: the §1.3b gate could not run.
    assert rep["crowned"] is None, \
        f"{variant}: an armed settlement crowned without the §1.3b gate inputs"
    assert rep["king_after"]["digest"] == rep["king_before"]["digest"] == h0["digest"]
    assert v["outcome"] not in ("crowned", "lost"), v["outcome"]
    gate = rep.get("gate")
    if gate is not None:
        passed = [r for r in gate.get("results") or [] if r.get("outcome") == "pass"]
        assert not passed, f"{variant}: a gate result passed without a bench or a block hash"
    assert s1.pay["court_uids"][0] is None, "the house still holds KING_D (its slot burns)"
    state = read_state(path)
    assert state["king"]["digest"] == h0["digest"] and state["king"]["house"]
    # bench_hold: typed, open, not the miner's fault, and not confirm_pending.
    rec = state["entries"][x["digest"]]
    assert (rec["status"], rec["reason"]) == ("hold", "bench_hold"), \
        f"{variant}: a missing gate input holds the first clear open: {rec['status']}"
    e = entry_of(rep, x["digest"])
    assert (e["status"], e["reason"]) == ("hold", "bench_hold")
    # Not charged: with the inputs back and no fresh window, the held first
    # clear is re-gated on the next boundary's draw and crowns without a new duel.
    hash2 = block_hash("b2")
    host2 = BenchHost(bench_table({x["digest"]: 0.9, h0["digest"]: 1.0}))
    s2 = settle_gated(cfg, boundary(2), pool1, [x], h0, _fc(recs1, models), uids(x), path,
                      hash_=hash2, bench=bench_session(host2, "sess-b2"))
    assert s2.report["crowned"] == x["digest"], \
        f"{variant}: the hold burned the candidate's chance"
    assert rows_of(s2.report, x["digest"]) == set(), \
        "a held candidate was re-duelled instead of re-gated"
    assert x["digest"] in host2.swept_digests()
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
    cfg, h0, x, path, models, pool1, recs1 = _base(tmp_path)
    s1 = settle_crowning(cfg, boundary(1), pool1, [x], h0, _fc(recs1, models), uids(x), path,
                         candidates=[x["digest"]], references=[h0["digest"]], tag="b1")
    assert s1.report["crowned"] == x["digest"], "fixture: the gated settlement did not crown"
    assert isinstance(s1.report.get("gate"), dict), "a gated crowning carries its gate record"
    return s1.report


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
    candidate has no gate and validates. Here X does not clear."""
    from cascade.validator.lane_d_report import validate_report

    cfg, h0, x, path, _, pool1, recs1 = _base(tmp_path)
    models = {h0["digest"]: ("fair", "house-0"), x["digest"]: ("bad", "x")}
    host = BenchHost(bench_table({x["digest"]: 0.9, h0["digest"]: 1.0}))
    s1 = settle_gated(cfg, boundary(1), pool1, [x], h0, _fc(recs1, models), uids(x), path,
                      hash_=block_hash("b1"), bench=bench_session(host, "sess-b1"))
    assert s1.report["crowned"] is None and host.calls == []
    assert s1.report.get("gate") is None
    validate_report(s1.report)
