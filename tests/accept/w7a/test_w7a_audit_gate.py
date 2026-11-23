"""W7a x W5e: `check_lane_d` understands the §1.3b gate (plan rev6 §1.3b,
§1.5, §14.3 items 1-2: satellites and `cascade-audit` re-derive every verdict
from signed fields; W5e README "Fail closed": "The audit must refuse it
too").

rev6.11 (b): the confirmation duel is DROPPED. A crown is decided at the
FIRST verdict: first clear (max-T LCB vs KING_D) ∧ A10 ∧ the §1.3b gate. The
audit (`cascade/audit/lane_d.py`, in W7a's PATHS) reads the signed `gate`
record and re-derives, from the report and the config alone (invariant 12:
never importing `cascade.validator.lane_d*`):

* the draw: `gate.block == boundary`, `gate.eligible` and `gate.suite` are
  `gate_suite_for(gate.block_hash, gate_suites, gate_weights, gate.sentinel)`;
* each result's outcome from its own LCBs: `pass` iff all four LCBs
  (`king_d`/`house` x `crps`/`mase`) are `>= -tau_public`, `gate_regression`
  iff one is below, with `blocked_by` the refs that fall below;
* the verdict each gate outcome implies for a cleared first verdict: `pass`
  -> eligible to crown, `gate_regression` / `checkpoint_fault` -> `lost`,
  `bench_hold` -> held (neither crowned nor lost); without a gate record
  (the gate could not run) a cleared verdict is held, never crowned;
* the crown: the highest-LCB cleared verdict whose gate result is `pass`,
  or a held candidate re-gated `pass` with no verdict in the report. A
  crowned report without a `gate` dict, or whose results hold no `pass`
  for the crowned digest, FAILs;
* a report carrying a confirm stage (`stage: "confirm"` on a cohort or a
  verdict) FAILs.

Every report here comes from a real `settle_lane_d` with the W5e gate inputs
(W5e's fixtures, `tests/accept/w5e/helpers.py`: synthetic chain, recorded
bench rows through a real `SweepSession`, no torch). A doctored report is
re-signed by the validator's own key, so only the audit's re-derivation can
catch it.

On main (W5e still has the confirm stage) a first clear is
`confirm_pending` and never gated, so the scenario fixtures and the honest
PASS cases fail on an `assert`.
"""

from __future__ import annotations

import copy

import pytest


def _h():
    """W5e's fixture helpers, imported in test bodies only."""
    from tests.accept.w5e import helpers as w5e

    return w5e


# ── settlements (real settle_lane_d, gate inputs) ───────────────────────────


def _fc(recs, models: dict) -> dict:
    h = _h()
    return {dg: h.model_forecasts(recs, q, name) for dg, (q, name) in models.items()}


def _verdict(report, dg):
    return next((v for v in report["verdicts"] if v["digest"] == dg), None)


def _entry(report, dg) -> dict:
    (e,) = [e for e in report["entries"] if e["digest"] == dg]
    return e


def _result(report, dg) -> dict:
    gate = report.get("gate")
    assert isinstance(gate, dict), f"the report carries no gate record (gate={gate!r})"
    rs = [r for r in gate["results"] if r["digest"] == dg]
    assert len(rs) == 1, f"want one gate result for {dg[:12]}; got {len(rs)}"
    return rs[0]


def _single(tmp, spec1, *, sentinel=None, gated=True) -> dict:
    """rev6.11 (b): X clears against the genesis house H0 at its FIRST
    verdict, boundary(1), where the bench (``spec1``: `bench_table` spec
    over X and H0) decides the gate. ``gated=False`` makes the legacy
    (ungated) call."""
    h = _h()
    cfg = h.lane_d_cfg()
    h0, x = h.house(), h.entry("x", reveal_ts=h.T_X)
    path = tmp / "state" / "lane_d_state.json"
    models = {h0["digest"]: ("bad", "house-0"), x["digest"]: ("sharp", "x")}
    out = {"cfg": cfg, "h0": h0, "x": x, "path": path, "models": models}
    pool, recs = h.snapshot(tmp, 1)
    args = (cfg, h.boundary(1), pool, [x], h0, _fc(recs, models), h.uids(x), path)
    if not gated:
        s = h.settle(*args)
    else:
        s = h.settle_gated(*args, hash_=h.block_hash("b1"), sentinel=sentinel,
                           bench=h.bench_session(
                               h.BenchHost(h.bench_table(spec1(x["digest"], h0["digest"]))),
                               "sess-1"))
    out["s1"], out["pool1"], out["recs1"] = s.report, pool, recs
    v = _verdict(out["s1"], x["digest"])
    assert v is not None and v.get("stage", "first") == "first" and v["cleared"] is True, \
        f"fixture: X did not clear at its first verdict: {v}"
    return out


SENTINEL = "boom"


def _build(name: str, tmp):
    """One scenario, checked for its rev6.11 shape (an `assert`, so a main
    without the change fails here for a right reason)."""
    run = _build_raw(name, tmp)
    for key in ("s1", "s2"):
        r = run.get(key)
        if r is None:
            continue
        if name not in ("legacy", "losers"):
            assert isinstance(r.get("gate"), dict), \
                f"fixture {name}.{key}: no gate record (the first clear must be gated)"
    if name in ("pass", "sentinel"):
        assert run["s1"]["crowned"] == run["x"]["digest"], \
            f"fixture {name}: X is not crowned at its first verdict"
    if name == "hold":
        assert run["s2"]["crowned"] == run["x"]["digest"], "fixture hold: no re-gated crown"
    if name == "legacy":
        e = _entry(run["s1"], run["x"]["digest"])
        assert (e["status"], e["reason"]) == ("hold", "bench_hold"), f"fixture legacy: {e}"
    return run


def _build_raw(name: str, tmp):
    h = _h()
    if name == "pass":
        return _single(tmp, lambda x, h0: {x: 0.9, h0: 1.0})
    if name == "sentinel":
        return _single(tmp, lambda x, h0: {x: {**{s: 0.9 for s in h.GATE_SUITES},
                                                SENTINEL: 1.5}, h0: 1.0},
                       sentinel=SENTINEL)
    if name == "regression":
        return _single(tmp, lambda x, h0: {x: 1.10, h0: 1.0})
    if name == "fault":
        return _single(tmp, lambda x, h0: {x: {s: "nan" for s in h.GATE_SUITES}, h0: 1.0})
    if name == "hold":
        run = _single(tmp, lambda x, h0: {x: 0.9, h0: {s: "nan" for s in h.GATE_SUITES}})
        # boundary(2): the bench is back, no fresh window: X's held first clear
        # is re-gated (a new draw) and crowned without a duel (no verdict, no rows).
        x, h0 = run["x"], run["h0"]
        host = h.BenchHost(h.bench_table({x["digest"]: 0.9, h0["digest"]: 1.0}))
        s2 = h.settle_gated(run["cfg"], h.boundary(2), run["pool1"], [x], h0,
                            _fc(run["recs1"], run["models"]), h.uids(x), run["path"],
                            hash_=h.block_hash("b2"), bench=h.bench_session(host, "sess-2"))
        run["s2"] = s2.report
        return run
    if name == "legacy":
        return _single(tmp, None, gated=False)
    if name == "runner_up":
        cfg = h.lane_d_cfg()
        h0 = h.house()
        x1, x2 = h.entry("x1", reveal_ts=h.T_X), h.entry("x2", reveal_ts=h.T_X)
        path = tmp / "state" / "lane_d_state.json"
        models = {h0["digest"]: ("bad", "house-0"), x1["digest"]: ("sharp", "x1"),
                  x2["digest"]: ("good", "x2")}
        out = {"cfg": cfg, "h0": h0, "x1": x1, "x2": x2}
        spec = {x1["digest"]: 1.10, x2["digest"]: 0.9, h0["digest"]: 1.0}
        pool, recs = h.snapshot(tmp, 1)
        s = h.settle_gated(cfg, h.boundary(1), pool, [x1, x2], h0, _fc(recs, models),
                           h.uids(x1, x2), path, hash_=h.block_hash("b1"),
                           bench=h.bench_session(h.BenchHost(h.bench_table(spec)), "sess-1"))
        out["s1"] = s.report
        v1, v2 = _verdict(out["s1"], x1["digest"]), _verdict(out["s1"], x2["digest"])
        assert v1["cleared"] and v2["cleared"], f"fixture: {v1} {v2}"
        assert v1["lcb"] > v2["lcb"], "fixture: x1 must be the max-T best"
        return out
    if name == "losers":
        # Legacy (ungated) call with nothing to gate: Z and W both lose.
        cfg = h.lane_d_cfg()
        h0 = h.house()
        z, w = h.entry("z", reveal_ts=h.T_X), h.entry("w", reveal_ts=h.T_X)
        path = tmp / "state" / "lane_d_state.json"
        models = {h0["digest"]: ("fair", "house-0"), z["digest"]: ("bad", "z"),
                  w["digest"]: ("bad", "w")}
        pool, recs = h.snapshot(tmp, 1)
        s1 = h.settle(cfg, h.boundary(1), pool, [z, w], h0, _fc(recs, models), h.uids(z, w),
                      path)
        return {"cfg": cfg, "h0": h0, "z": z, "w": w, "s1": s1.report}
    raise KeyError(name)


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """Each scenario settled once (the box is small)."""
    cache: dict[str, dict] = {}

    def get(name: str) -> dict:
        if name not in cache:
            cache[name] = _build(name, tmp_path_factory.mktemp(f"w7a-audit-{name}"))
        return cache[name]
    return get


# ── the audit ───────────────────────────────────────────────────────────────


def _audit(report, cfg, signer_name: str = "signer", helpers=None):
    pytest.importorskip("bittensor.sp_core")
    from cascade.audit.checks import CheckResult
    from cascade.audit.lane_d import check_lane_d

    key = (helpers or _h()).fixture_key(signer_name)
    res = check_lane_d(report, cfg, signer_hotkey=key.ss58_address)
    assert isinstance(res, CheckResult), f"check_lane_d returned {type(res).__name__}"
    return res


def _passes(report, cfg, what: str, **kw) -> None:
    res = _audit(report, cfg, **kw)
    assert res.status == "PASS", f"{what}: check_lane_d {res.status}: {res.detail}"


def _fails(report, cfg, what: str) -> None:
    res = _audit(report, cfg)
    assert res.status == "FAIL", \
        f"{what}: a re-signed doctored report got check_lane_d {res.status}: {res.detail}"


def _resign(report: dict) -> dict:
    """The validator itself re-signs a doctored body: the signature proves
    nothing about the gate, only the re-derivation does."""
    from cascade.validator.lane_d_report import rows_sha256, sign_report

    out = copy.deepcopy(report)
    out["rows_sha256"] = rows_sha256(out["rows"])
    return sign_report(out, _h().fixture_key("signer"))


def _tau(cfg) -> float:
    return float(cfg.tau_public)


# ── (b) valid gated reports PASS ────────────────────────────────────────────


def test_a_gate_pass_crown_passes_the_audit(runs):
    run = runs("pass")
    r, x = run["s1"], run["x"]["digest"]
    assert r["crowned"] == x and _result(r, x)["outcome"] == "pass"
    assert _verdict(r, x)["outcome"] == "crowned"
    _passes(r, run["cfg"], "a first-verdict crown through a gate pass")


def test_a_gate_pass_crown_with_the_sentinel_recorded_passes_the_audit(runs):
    run = runs("sentinel")
    r, x = run["s1"], run["x"]["digest"]
    assert r["gate"]["sentinel"] == SENTINEL and r["gate"]["suite"] != SENTINEL
    assert SENTINEL not in r["gate"]["eligible"]
    assert r["crowned"] == x
    _passes(r, run["cfg"], "a crown with the era's sentinel recorded (a sentinel loss)")


def test_a_cleared_first_verdict_lost_to_gate_regression_passes_the_audit(runs):
    run = runs("regression")
    r, x = run["s1"], run["x"]["digest"]
    assert r["crowned"] is None
    assert _result(r, x)["outcome"] == "gate_regression" and _result(r, x)["blocked_by"]
    v = _verdict(r, x)
    assert v["cleared"] is True and v["outcome"] == "lost"
    assert (_entry(r, x)["status"], _entry(r, x)["reason"]) == ("lost", "gate_regression")
    _passes(r, run["cfg"], "a cleared first verdict lost via gate_regression")


def test_a_cleared_first_verdict_closed_on_checkpoint_fault_passes_the_audit(runs):
    run = runs("fault")
    r, x = run["s1"], run["x"]["digest"]
    assert r["crowned"] is None
    assert _result(r, x)["outcome"] == "checkpoint_fault" and _result(r, x)["lcb"] is None
    assert _verdict(r, x)["cleared"] is True and _verdict(r, x)["outcome"] != "crowned"
    assert (_entry(r, x)["status"], _entry(r, x)["reason"]) == ("closed", "checkpoint_fault")
    _passes(r, run["cfg"], "a cleared first verdict closed via checkpoint_fault")


def test_a_bench_hold_is_held_not_crowned_and_passes_the_audit(runs):
    run = runs("hold")
    r, x = run["s1"], run["x"]["digest"]
    assert r["crowned"] is None and r["king_after"]["digest"] == r["king_before"]["digest"]
    assert _result(r, x)["outcome"] == "bench_hold"
    v = _verdict(r, x)
    assert v["cleared"] is True and v["outcome"] not in ("crowned", "lost")
    assert (_entry(r, x)["status"], _entry(r, x)["reason"]) == ("hold", "bench_hold")
    _passes(r, run["cfg"], "a cleared first verdict held via bench_hold (not crowned)")


def test_a_held_candidate_regated_and_crowned_without_a_verdict_passes_the_audit(runs):
    run = runs("hold")
    r, x = run["s2"], run["x"]["digest"]
    assert r["crowned"] == x
    assert _verdict(r, x) is None and not [row for row in r["rows"] if row["digest"] == x]
    assert _result(r, x)["outcome"] == "pass"
    _passes(r, run["cfg"], "a held candidate re-gated and crowned at a later settlement")


def test_a_passing_runner_up_crowned_over_a_regressing_best_passes_the_audit(runs):
    run = runs("runner_up")
    r, x1, x2 = run["s1"], run["x1"]["digest"], run["x2"]["digest"]
    assert _result(r, x1)["outcome"] == "gate_regression"
    assert _result(r, x2)["outcome"] == "pass"
    assert r["crowned"] == x2 and _verdict(r, x1)["outcome"] == "lost"
    _passes(r, run["cfg"], "a passing runner-up crowned over the regressing max-T best")


def test_the_w7a_settlements_with_gate_inputs_pass_the_audit(tmp_path):
    """W7a's own two-settlement chain (`helpers.two_settlements`, a healthy
    bench through `helpers.settle`): a loser, then the gated first-verdict crown."""
    from . import helpers as w7a

    built = w7a.two_settlements(tmp_path)
    s2 = built["s2"].report
    assert s2["crowned"] == built["entry"]["digest"]
    assert isinstance(s2.get("gate"), dict), "fixture: W5e wrote no gate record"
    for name in ("s1", "s2"):
        _passes(built[name].report, built["cfg"], f"W7a {name}", helpers=w7a)


# ── (c) ungated reports with nothing to crown still PASS ────────────────────


def test_ungated_reports_without_a_crowning_candidate_pass_the_audit(runs):
    run = runs("losers")
    r = run["s1"]
    assert "gate" not in r and r["crowned"] is None
    assert _verdict(r, run["z"]["digest"])["outcome"] == "lost"
    assert _verdict(r, run["w"]["digest"])["outcome"] == "lost"
    _passes(r, run["cfg"], "an ungated report with two losers")


def test_an_ungated_armed_crowning_candidate_is_held_and_passes_the_audit(runs):
    """The legacy call (no bench, no block hash) fails closed: the cleared
    first verdict is `bench_hold`, nothing crowns and no gate record is
    written (W5e "Fail closed")."""
    run = runs("legacy")
    r, x = run["s1"], run["x"]["digest"]
    assert "gate" not in r and r["crowned"] is None
    assert _verdict(r, x)["outcome"] not in ("crowned", "lost")
    assert (_entry(r, x)["status"], _entry(r, x)["reason"]) == ("hold", "bench_hold")
    _passes(r, run["cfg"], "an ungated cleared first verdict held (fail closed)")


# ── (a) crowned reports the gate does not support FAIL ──────────────────────


def _doctor_crowned_pass(r: dict, how: str, cfg) -> dict:
    h = _h()
    out = copy.deepcopy(r)
    x = out["crowned"]
    gate = out.get("gate")
    if how == "gate_absent":
        out.pop("gate")
    elif how == "gate_null":
        out["gate"] = None
    elif how == "no_result_for_crowned":
        res = next(g for g in gate["results"] if g["digest"] == x)
        res["digest"] = h.digest("w7a audit: some other candidate")
    elif how == "crowned_result_held":
        res = next(g for g in gate["results"] if g["digest"] == x)
        res.update(outcome="bench_hold", lcb=None, blocked_by=[])
    elif how == "crowned_result_regressed":
        res = next(g for g in gate["results"] if g["digest"] == x)
        res["lcb"]["house"]["crps"] = -(_tau(cfg) + 0.05)
        res.update(outcome="gate_regression", blocked_by=["house"])
    elif how == "suite_not_the_draw":
        gate["suite"] = next(s for s in gate["eligible"] if s != gate["suite"])
    elif how == "block_hash_draws_another_suite":
        for i in range(256):
            other = h.block_hash(f"w7a audit other {i}")
            if h.drawn_suite(cfg, other, gate["sentinel"]) != gate["suite"]:
                gate["block_hash"] = other
                break
        else:
            raise AssertionError("fixture: no block hash draws another suite")
    elif how == "block_not_the_boundary":
        gate["block"] = int(out["boundary"]) + 1
    elif how == "pass_with_an_lcb_below_tau":
        res = next(g for g in gate["results"] if g["digest"] == x)
        res["lcb"]["king_d"]["mase"] = -(_tau(cfg) + 0.05)
    else:
        raise KeyError(how)
    return _resign(out)


CROWNED_DOCTORINGS = ("gate_absent", "gate_null", "no_result_for_crowned",
                      "crowned_result_held", "crowned_result_regressed",
                      "suite_not_the_draw", "block_hash_draws_another_suite",
                      "block_not_the_boundary", "pass_with_an_lcb_below_tau")


@pytest.mark.parametrize("how", CROWNED_DOCTORINGS)
def test_a_crowned_report_the_gate_does_not_support_fails_the_audit(runs, how):
    run = runs("pass")
    r = run["s1"]
    _passes(r, run["cfg"], "the honest crowned report")
    _fails(_doctor_crowned_pass(r, how, run["cfg"]), run["cfg"], f"crowned, {how}")


def test_a_crowned_report_with_a_sentinel_drawn_as_the_suite_fails_the_audit(runs):
    run = runs("sentinel")
    r = copy.deepcopy(run["s1"])
    _passes(r, run["cfg"], "the honest sentinel crown")
    r["gate"]["suite"] = SENTINEL
    _fails(_resign(r), run["cfg"], "the sentinel stated as the drawn suite")


def test_a_held_regate_crown_without_its_pass_fails_the_audit(runs):
    run = runs("hold")
    r = copy.deepcopy(run["s2"])
    _passes(r, run["cfg"], "the honest report the doctoring starts from")
    res = _result(r, r["crowned"])
    res.update(outcome="bench_hold", lcb=None, blocked_by=[])
    _fails(_resign(r), run["cfg"], "a re-gated crown whose result is bench_hold")
    r2 = copy.deepcopy(run["s2"])
    r2.pop("gate")
    _fails(_resign(r2), run["cfg"], "a re-gated crown with no gate record")


# ── (a) gate outcomes that disagree with their own LCBs or the verdicts FAIL ─


def test_a_regression_whose_lcbs_all_clear_tau_fails_the_audit(runs):
    run = runs("regression")
    r = copy.deepcopy(run["s1"])
    _passes(r, run["cfg"], "the honest report the doctoring starts from")
    res = _result(r, run["x"]["digest"])
    res["lcb"] = {ref: {m: 0.0 for m in ("crps", "mase")} for ref in ("king_d", "house")}
    _fails(_resign(r), run["cfg"], "gate_regression with every LCB >= -tau_public")


def test_a_regression_blocked_by_the_wrong_reference_fails_the_audit(runs):
    run = runs("regression")
    r = copy.deepcopy(run["s1"])
    _passes(r, run["cfg"], "the honest report the doctoring starts from")
    res = _result(r, run["x"]["digest"])
    tau = _tau(run["cfg"])
    below = {ref for ref in ("king_d", "house")
             if any(res["lcb"][ref][m] < -tau for m in ("crps", "mase"))}
    assert below and sorted(below) == res["blocked_by"], f"fixture: {res}"
    if below == {"king_d", "house"}:
        res["blocked_by"] = ["king_d"]
    else:
        res["blocked_by"] = sorted({"king_d", "house"} - below)
    _fails(_resign(r), run["cfg"], "blocked_by naming refs the LCBs do not block")


def test_a_regressed_verdict_stated_held_fails_the_audit(runs):
    run = runs("regression")
    r = copy.deepcopy(run["s1"])
    _passes(r, run["cfg"], "the honest report the doctoring starts from")
    held = next(v["outcome"] for v in runs("hold")["s1"]["verdicts"])
    _verdict(r, run["x"]["digest"])["outcome"] = held
    _fails(_resign(r), run["cfg"], "a gate_regression verdict left held")


def test_a_passing_result_that_is_not_crowned_fails_the_audit(runs):
    """A held report doctored to a gate `pass` for its only candidate, still
    uncrowned: the audit re-derives the crown from the gate."""
    run = runs("hold")
    r = copy.deepcopy(run["s1"])
    _passes(r, run["cfg"], "the honest report the doctoring starts from")
    res = _result(r, run["x"]["digest"])
    res.update(outcome="pass", blocked_by=[],
               lcb={ref: {m: 0.0 for m in ("crps", "mase")} for ref in ("king_d", "house")})
    _fails(_resign(r), run["cfg"], "a passing candidate left uncrowned")


def test_a_pass_doctored_onto_the_regressing_best_fails_the_audit(runs):
    """In the runner-up report the max-T best regressed. Doctored to a
    consistent `pass`, the re-derived crown is the best, not the runner-up."""
    run = runs("runner_up")
    r = copy.deepcopy(run["s1"])
    _passes(r, run["cfg"], "the honest report the doctoring starts from")
    res = _result(r, run["x1"]["digest"])
    res.update(outcome="pass", blocked_by=[],
               lcb={ref: {m: 0.0 for m in ("crps", "mase")} for ref in ("king_d", "house")})
    _fails(_resign(r), run["cfg"], "the crown went to the runner-up over a passing best")


def test_a_cleared_verdict_missing_from_the_gate_fails_the_audit(runs):
    run = runs("runner_up")
    r = copy.deepcopy(run["s1"])
    _passes(r, run["cfg"], "the honest report the doctoring starts from")
    x1 = run["x1"]["digest"]
    r["gate"]["results"] = [g for g in r["gate"]["results"] if g["digest"] != x1]
    _fails(_resign(r), run["cfg"], "a cleared verdict lost with no gate result")


def test_an_ungated_cleared_verdict_stated_crowned_fails_the_audit(runs):
    """The legacy (fail-closed) report doctored into a crown: no gate record,
    so nothing may crown."""
    run = runs("legacy")
    r = copy.deepcopy(run["s1"])
    _passes(r, run["cfg"], "the honest report the doctoring starts from")
    x = run["x"]["digest"]
    _verdict(r, x)["outcome"] = "crowned"
    r["crowned"] = x
    r["king_after"] = {**r["king_after"], "digest": x, "house": False,
                       "hotkey": run["x"]["hotkey"]}
    court = r["pay"]["court"]
    if court:
        court[0] = {**court[0], "digest": x, "hotkey": run["x"]["hotkey"]}
    _fails(_resign(r), run["cfg"], "an ungated crown")


# ── rev6.11 (b): a confirm stage is refused ─────────────────────────────────


@pytest.mark.parametrize("where", ["cohort", "verdict"])
def test_a_report_carrying_a_confirm_stage_fails_the_audit(runs, where):
    """No confirmation stage exists: a crowned report relabelled with
    `stage: "confirm"` (on its cohort or its verdict) is refused, even
    re-signed."""
    run = runs("pass")
    r = copy.deepcopy(run["s1"])
    _passes(r, run["cfg"], "the honest report the doctoring starts from")
    if where == "cohort":
        r["cohorts"][0]["stage"] = "confirm"
    else:
        _verdict(r, run["x"]["digest"])["stage"] = "confirm"
    _fails(_resign(r), run["cfg"], f"a confirm stage on the {where}")
