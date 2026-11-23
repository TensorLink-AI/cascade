"""W7a x RV7 B2: `check_lane_d` re-derives the crown's provenance, not only
its gate (plan rev6 §1.3, §1.3b A10, §1.5, §14.3 items 1-2; README
"Correction (RV7 B2)", "Correction (rev6.11)").

rev6.11 (b): the confirmation duel is dropped, so a crown is decided at the
first verdict and there is no first-clear chain to follow. What stays:

* (a) a gate `pass` for a listed entry with no verdict is a held candidate
  re-gated after a `bench_hold`. It must carry `held_from` = {boundary,
  report_sha256, set_sha256, lcb, king_d}: the settlement where its FIRST
  verdict cleared and the gate held it. A result with a verdict this
  settlement carries none. In chain mode (`prior_reports`, what W9 runs) the
  named report must show that hold;
* a crown needs a clear: a loser doctored into a crown FAILs;
* (c) `report["seed"]`, the KING_D a cohort is judged against (it must be
  `king_before`) and `gate.sentinel` (`era_sentinel`) are re-checked;
* (d) W3b5: a cleared verdict whose signed A10 block reads `fail` is `lost`,
  never gated and never crowned.

Reports come from real `settle_lane_d` calls (W5e's fixtures through
`test_w7a_audit_gate._build`, W3b5's for A10, W7a's `two_settlements`).
Every doctored report is re-signed by the validator's own key, and every
doctored case first requires the honest report it starts from to PASS.

On main (the confirm stage still in W5e) the scenario fixtures fail on an
`assert`: a first clear is `confirm_pending`, never gated or held.
"""

from __future__ import annotations

import copy
import hashlib
import inspect

import numpy as np
import pytest

from .test_w7a_audit_gate import SENTINEL, _build, _entry, _result, _verdict

_UNSET = object()
HELD_FROM_KEYS = {"boundary", "report_sha256", "set_sha256", "lcb", "king_d"}


def _h():
    from tests.accept.w5e import helpers as w5e

    return w5e


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    cache: dict[str, dict] = {}

    def get(name: str) -> dict:
        if name not in cache:
            tmp = tmp_path_factory.mktemp(f"w7a-prov-{name}")
            cache[name] = _challenger(tmp) if name == "challenger" else (
                _pending(tmp) if name == "pending" else _build(name, tmp))
        return cache[name]
    return get


def _pending(tmp) -> dict:
    """RV7's reproducer: X loses at boundary(1); Y is listed but has no
    forward window (`pending` / `pending_floor`), never judged."""
    h = _h()
    cfg = h.lane_d_cfg()
    h0, x = h.house(), h.entry("x", reveal_ts=h.T_X)
    y = h.entry("y", reveal_ts=h.T_X + 30 * h.DAY)
    pool1, recs1 = h.snapshot(tmp, 1)
    fc = {h0["digest"]: h.model_forecasts(recs1, "fair", "house-0"),
          x["digest"]: h.model_forecasts(recs1, "bad", "x"),
          y["digest"]: h.model_forecasts(recs1, "bad", "y")}
    s1 = h.settle(cfg, h.boundary(1), pool1, [x, y], h0, fc, h.uids(x, y),
                  tmp / "state" / "lane_d_state.json")
    e = _entry(s1.report, y["digest"])
    assert (e["status"], e["reason"]) == ("pending", "pending_floor"), f"fixture: {e}"
    assert y["digest"] not in s1.report["judged"]
    assert _verdict(s1.report, x["digest"])["outcome"] == "lost", "fixture: X must lose"
    return {"cfg": cfg, "h0": h0, "x": x, "y": y, "s1": s1.report}


def _challenger(tmp) -> dict:
    """The `pass` run (X crowned at its first verdict, boundary(1)), then Y's
    first verdict at boundary(2) against KING_D = X with the house H0 as the
    second bar. Y is `good`: it loses to X but beats the `bad` house."""
    h = _h()
    run = dict(_build("pass", tmp))
    x, h0 = run["x"], run["h0"]
    y = h.entry("y", reveal_ts=h.T_Y, after=1)
    pool2, recs2 = h.snapshot(tmp, 2, t=h.T_Y)
    fc = {h0["digest"]: h.model_forecasts(recs2, "bad", "house-0"),
          x["digest"]: h.model_forecasts(recs2, "sharp", "x"),
          y["digest"]: h.model_forecasts(recs2, "good", "y")}
    s2 = h.settle(run["cfg"], h.boundary(2), pool2, [x, y], h0, fc, h.uids(x, y), run["path"])
    run.update(y=y, s2=s2.report)
    v = _verdict(s2.report, y["digest"])
    assert v is not None and v.get("stage", "first") == "first" \
        and v["king_digest"] == x["digest"], \
        f"fixture: Y was not judged against KING_D = X: {v}"
    assert v["cleared"] is False and v["house"]["cleared"] is True, \
        f"fixture: Y must lose to X and beat the house: {v}"
    return run


def _held_outcome(runs) -> str:
    """The producer's verdict outcome for a held (bench_hold) first clear."""
    run = runs("hold")
    return _verdict(run["s1"], run["x"]["digest"])["outcome"]


# ── the audit ───────────────────────────────────────────────────────────────


def _audit(report, cfg, *, helpers=None, prior=None, era_sentinel=_UNSET):
    pytest.importorskip("bittensor.sp_core")
    from cascade.audit.checks import CheckResult
    from cascade.audit.lane_d import check_lane_d

    params = inspect.signature(check_lane_d).parameters
    kw = {}
    if prior is not None:
        assert "prior_reports" in params, (
            "check_lane_d takes no prior_reports: the chain-mode audit (RV7 B2) that "
            "verifies held_from and first clears against earlier signed reports is missing")
        kw["prior_reports"] = [copy.deepcopy(p) for p in prior]
    if era_sentinel is not _UNSET:
        assert "era_sentinel" in params, (
            "check_lane_d takes no era_sentinel: gate.sentinel is not checked against "
            "the era's W-S1 record (RV7 B2(c))")
        kw["era_sentinel"] = era_sentinel
    key = (helpers or _h()).fixture_key("signer")
    res = check_lane_d(report, cfg, signer_hotkey=key.ss58_address, **kw)
    assert isinstance(res, CheckResult), f"check_lane_d returned {type(res).__name__}"
    return res


def _passes(report, cfg, what: str, **kw) -> None:
    res = _audit(report, cfg, **kw)
    assert res.status == "PASS", f"{what}: check_lane_d {res.status}: {res.detail}"


def _fails(report, cfg, what: str, **kw) -> None:
    res = _audit(report, cfg, **kw)
    assert res.status == "FAIL", \
        f"{what}: a re-signed doctored report got check_lane_d {res.status}: {res.detail}"


def _resign(report: dict, helpers=None) -> dict:
    from cascade.validator.lane_d_report import rows_sha256, sign_report

    out = copy.deepcopy(report)
    out.pop("signature", None)
    out["rows_sha256"] = rows_sha256(out["rows"])
    return sign_report(out, (helpers or _h()).fixture_key("signer"))


def _digest_of(report: dict) -> str:
    from cascade.validator.lane_d_state import report_digest

    return report_digest(report)


def _hex(tag: str) -> str:
    return hashlib.sha256(f"w7a rv7 {tag}".encode()).hexdigest()


def _lcb_ok() -> dict:
    return {r: {m: 0.5 for m in ("crps", "mase")} for r in ("king_d", "house")}


def _crown(report: dict, entry: dict, uid: int) -> None:
    """Make ``entry`` the stated crown of ``report`` (in place)."""
    dg = entry["digest"]
    report["crowned"] = dg
    report["king_after"] = {"digest": dg, "hotkey": entry["hotkey"], "house": False,
                            "reveal_ts": entry["reveal_ts"]}
    report["pay"]["court"][0] = {"digest": dg, "hotkey": entry["hotkey"], "uid": uid,
                                 "burn_reason": None}
    for e in report["entries"]:
        if e["digest"] == dg:
            e["status"], e["reason"] = "crowned", "crowned"
    report["skipped"] = [s for s in report["skipped"] if s["digest"] != dg]
    for v in report["verdicts"]:
        if v["digest"] == dg:
            v["outcome"] = "crowned"


def _forged_gate(cfg, b: int, dg: str, tag: str, sentinel=None) -> dict:
    from cascade.validator.lane_d_gate import gate_record

    g = gate_record(cfg, int(b), "0x" + _hex(tag), sentinel=sentinel)
    g["results"] = [{"digest": dg, "outcome": "pass", "lcb": _lcb_ok(), "blocked_by": [],
                     "session": f"forged-{tag}"}]
    return g


def _cohort_of(report: dict, dg: str) -> dict:
    v = _verdict(report, dg)
    return report["cohorts"][int(v["cohort"])]


# ── (a) the producer signs held_from; the schema carries it ─────────────────


def test_a_held_regate_result_carries_held_from_naming_the_hold(runs):
    """Hold run: X's first verdict clears at boundary(1) and the gate holds it
    (`bench_hold`); at boundary(2) it is re-gated without a verdict and
    crowned. Its result binds the boundary(1) report that holds it."""
    run = runs("hold")
    s1, s2, x = run["s1"], run["s2"], run["x"]["digest"]
    res = _result(s2, x)
    assert _verdict(s2, x) is None and res["outcome"] == "pass"
    hf = res.get("held_from")
    assert isinstance(hf, dict) and set(hf) == HELD_FROM_KEYS, \
        f"a held candidate's gate result must carry held_from {sorted(HELD_FROM_KEYS)}: {res}"
    v1 = _verdict(s1, x)
    assert hf["boundary"] == s1["boundary"] and isinstance(hf["boundary"], int)
    assert hf["report_sha256"] == _digest_of(s1), \
        "held_from.report_sha256 is the hold report's body digest (lane_d_state.report_digest)"
    assert hf["set_sha256"] == _cohort_of(s1, x)["set_sha256"]
    assert hf["lcb"] == v1["lcb"]
    assert hf["king_d"] == v1["king_digest"] == s2["king_before"]["digest"]


def test_results_with_a_verdict_carry_no_held_from(runs):
    for name in ("pass", "regression", "fault", "hold", "runner_up"):
        r = runs(name)["s1"]
        for res in r["gate"]["results"]:
            assert "held_from" not in res, f"{name}: a verdict-backed result has held_from"


def test_a_report_carrying_held_from_validates_and_the_record_is_closed(runs):
    from cascade.validator.lane_d_report import ReportSchemaError, validate_report

    run = runs("hold")
    r = run["s2"]
    assert "held_from" in _result(r, run["x"]["digest"]), "fixture: no held_from"
    validate_report(r)
    for value, code in ((None, "bad_type"), ("x", "bad_type")):
        bad = copy.deepcopy(r)
        _result(bad, run["x"]["digest"])["held_from"] = value
        with pytest.raises(ReportSchemaError) as e:
            validate_report(_resign(bad))
        assert e.value.code == code, (value, e.value.code, e.value.field)
    bad = copy.deepcopy(r)
    _result(bad, run["x"]["digest"])["held_from"]["extra"] = 1
    with pytest.raises(ReportSchemaError) as e:
        validate_report(_resign(bad))
    assert e.value.code == "unknown_field", (e.value.code, e.value.field)
    for k in sorted(HELD_FROM_KEYS):
        bad = copy.deepcopy(r)
        _result(bad, run["x"]["digest"])["held_from"].pop(k)
        with pytest.raises(ReportSchemaError) as e:
            validate_report(_resign(bad))
        assert e.value.code == "missing_field", (k, e.value.code, e.value.field)


def _find_key(node, key):
    """Every value stored under ``key`` anywhere in the schema document."""
    if isinstance(node, dict):
        for k, v in node.items():
            if k == key:
                yield v
            yield from _find_key(v, key)
    elif isinstance(node, list):
        for v in node:
            yield from _find_key(v, key)


def test_the_schema_file_describes_held_from_as_an_optional_closed_record():
    from cascade.validator.lane_d_report import load_schema

    schema = load_schema()
    found = list(_find_key(schema, "held_from"))
    assert found, "the schema file does not describe gate.results[].held_from"
    defs = schema.get("$defs", {})
    for node in found:
        while isinstance(node, dict) and "$ref" in node:
            node = defs[node["$ref"].rsplit("/", 1)[-1]]
        assert isinstance(node, dict) and set(node.get("properties", {})) == HELD_FROM_KEYS, node
        assert set(node.get("required", [])) == HELD_FROM_KEYS, node
        assert node.get("additionalProperties") is False, node
    for req in _find_key(schema, "required"):
        assert "held_from" not in req, "held_from is optional (present only on a held result)"


# ── (a) held crowns need their hold ─────────────────────────────────────────


def test_rv7_a_a_never_judged_entry_crowned_as_held_fails_the_audit(runs):
    """RV7's reproducer: Y (`pending_floor`, never judged) gets a gate `pass`
    with no verdict and is crowned. Nothing binds it to a hold."""
    h = _h()
    run = runs("pending")
    s1, cfg, y = run["s1"], run["cfg"], run["y"]
    _passes(s1, cfg, "the honest settlement")
    d = copy.deepcopy(s1)
    d["gate"] = _forged_gate(cfg, h.boundary(1), y["digest"], "pending")
    _crown(d, y, 11)
    _fails(_resign(d), cfg, "a never-judged entry crowned as a held candidate")


def test_rv7_a_a_fabricated_held_from_fails_the_chain_audit(runs):
    """The same forgery with an invented `held_from`: structurally plausible,
    so the single-report audit cannot see it; the chain audit finds no hold
    report and FAILs."""
    h = _h()
    run = runs("pending")
    s1, cfg, y = run["s1"], run["cfg"], run["y"]
    _passes(s1, cfg, "the honest settlement", prior=[])
    d = copy.deepcopy(s1)
    d["gate"] = _forged_gate(cfg, h.boundary(1), y["digest"], "pending-hf")
    d["gate"]["results"][0]["held_from"] = {
        "boundary": h.ARM_BLOCK, "report_sha256": _hex("no such report"),
        "set_sha256": _hex("no such set"), "lcb": 0.5, "king_d": run["h0"]["digest"]}
    _crown(d, y, 11)
    _fails(_resign(d), cfg, "a fabricated held_from, chain audit", prior=[])


def test_the_honest_held_crown_passes_alone_and_in_chain_mode(runs):
    run = runs("hold")
    cfg = run["cfg"]
    _passes(run["s2"], cfg, "the held crown, single report")
    _passes(run["s2"], cfg, "the held crown, chain mode", prior=[run["s1"]])
    _passes(run["s1"], cfg, "the hold, chain mode", prior=[])


def _held_doctor(run, how: str) -> dict:
    r = copy.deepcopy(run["s2"])
    x = run["x"]["digest"]
    res = _result(r, x)
    hf = res.get("held_from")
    assert isinstance(hf, dict), f"fixture: the honest held result has no held_from: {res}"
    if how == "absent":
        res.pop("held_from")
    elif how == "boundary_not_before":
        hf["boundary"] = int(r["boundary"])
    elif how == "lcb_below_d_margin":
        hf["lcb"] = float(run["cfg"].d_margin) - 0.01
    elif how == "king_d_not_king_before":
        hf["king_d"] = _hex("another king")
    else:
        raise KeyError(how)
    return _resign(r)


@pytest.mark.parametrize("how", ["absent", "boundary_not_before", "lcb_below_d_margin",
                                 "king_d_not_king_before"])
def test_a_held_crown_with_a_bad_held_from_fails_the_audit(runs, how):
    """Single-report rules: a held result names its hold; the hold is at an
    earlier boundary; its LCB cleared `d_margin`; its KING_D is still
    `king_before` (a king change drops a hold: its LCB was against another
    king, and there is no second duel to redo it)."""
    run = runs("hold")
    _passes(run["s2"], run["cfg"], "the honest held crown")
    _fails(_held_doctor(run, how), run["cfg"], f"held crown, held_from {how}")


def test_held_from_on_a_verdict_backed_result_fails_the_audit(runs):
    run = runs("pass")
    r = copy.deepcopy(run["s1"])
    _passes(r, run["cfg"], "the honest crown")
    x = run["x"]["digest"]
    _result(r, x)["held_from"] = {"boundary": int(run["s1"]["boundary"]) - 900,
                                  "report_sha256": _hex("an earlier report"),
                                  "set_sha256": _cohort_of(run["s1"], x)["set_sha256"],
                                  "lcb": _verdict(run["s1"], x)["lcb"],
                                  "king_d": run["h0"]["digest"]}
    _fails(_resign(r), run["cfg"], "held_from on a candidate judged this settlement")


def _chain_doctor(run, how: str, runs) -> tuple[dict, list[dict]]:
    r = copy.deepcopy(run["s2"])
    s1 = copy.deepcopy(run["s1"])
    x = run["x"]["digest"]
    hf = _result(r, x).get("held_from")
    assert isinstance(hf, dict), "fixture: the honest held result has no held_from"
    if how == "names_a_report_that_crowned_it":
        # A real signed report (the `pass` run: same X, same house, same
        # boundary) where X's first verdict passed the gate: not a hold.
        p1 = copy.deepcopy(runs("pass")["s1"])
        hf.update(report_sha256=_digest_of(p1), set_sha256=_cohort_of(p1, x)["set_sha256"],
                  lcb=_verdict(p1, x)["lcb"])
        return _resign(r), [p1]
    if how == "report_sha256_differs":
        hf["report_sha256"] = _hex("another body")
    elif how == "set_sha256_differs":
        hf["set_sha256"] = _hex("another set")
    elif how == "lcb_differs":
        hf["lcb"] = float(hf["lcb"]) + 0.25
    elif how == "hold_report_missing":
        return _resign(r), []
    elif how == "hold_report_signature_invalid":
        # Same body (same digest), a signature that does not verify over it.
        s1["signature"] = runs("pass")["s1"]["signature"]
        return r, [s1]
    else:
        raise KeyError(how)
    return _resign(r), [s1]


CHAIN_HELD = ("names_a_report_that_crowned_it", "report_sha256_differs", "set_sha256_differs",
              "lcb_differs", "hold_report_missing", "hold_report_signature_invalid")


@pytest.mark.parametrize("how", CHAIN_HELD)
def test_a_held_crown_whose_hold_the_chain_does_not_show_fails_the_chain_audit(runs, how):
    run = runs("hold")
    cfg = run["cfg"]
    _passes(run["s2"], cfg, "the honest held crown, chain mode", prior=[run["s1"]])
    r, prior = _chain_doctor(run, how, runs)
    _fails(r, cfg, f"held crown, chain {how}", prior=prior)


# ── a crown needs a clear ───────────────────────────────────────────────────


def test_a_loser_doctored_into_a_first_verdict_crown_fails_the_audit(runs):
    """rev6.11: the crown is the first verdict, so the clear itself is the
    provenance. A lost verdict re-labelled cleared and crowned through a
    forged gate `pass` FAILs: the audit re-derives `cleared` from the LCB,
    `d_margin` and the gain."""
    h = _h()
    run = runs("pending")
    s1, cfg, x = run["s1"], run["cfg"], run["x"]
    _passes(s1, cfg, "the honest settlement")
    d = copy.deepcopy(s1)
    v = _verdict(d, x["digest"])
    assert v["lcb"] < float(cfg.d_margin), f"fixture: X's LCB must be below d_margin: {v}"
    v["cleared"] = True
    d["gate"] = _forged_gate(cfg, h.boundary(1), x["digest"], "loser")
    _crown(d, x, 10)
    _fails(_resign(d), cfg, "a loser crowned at its first verdict")


def test_honest_first_verdict_reports_pass_the_chain_audit(runs, tmp_path):
    for name in ("pass", "sentinel", "regression", "fault", "hold", "runner_up", "legacy"):
        run = runs(name)
        _passes(run["s1"], run["cfg"], f"{name} s1, chain mode", prior=[])
    from . import helpers as w7a

    built = w7a.two_settlements(tmp_path)
    _passes(built["s2"].report, built["cfg"], "W7a s2, chain mode", helpers=w7a,
            prior=[built["s1"].report])


# ── (c) the seed, king_before and the era sentinel ──────────────────────────


def _reseed(report: dict, cfg, seed: str) -> dict | None:
    """``report`` with ``seed`` and every verdict's LCBs re-run from its rows
    under that seed (the producer's own `judge_cohort`), so only the seed's
    binding can catch it; ``None`` if the new seed flips any clear."""
    from cascade.validator.lane_d import judge_cohort

    out = copy.deepcopy(report)
    out["seed"] = seed
    rows = {(int(r["cohort"]), r["window_id"], r["digest"]): r for r in out["rows"]}
    windows = {w["window_id"]: w for w in out["windows"]}
    for c in out["cohorts"]:
        idx, wids = int(c["index"]), list(c["window_ids"])

        def scores(dg, idx=idx, wids=wids):
            return (np.asarray([rows[(idx, w, dg)]["crps"] for w in wids]),
                    np.asarray([rows[(idx, w, dg)]["mase"] for w in wids]))
        clusters = [windows[w]["source"] for w in wids]
        members = list(c["members"])
        res = judge_cohort(scores(c["king_digest"]), [scores(m) for m in members],
                           clusters=clusters, cfg=cfg, seed=seed)
        hres = (judge_cohort(scores(c["house_digest"]), [scores(m) for m in members],
                             clusters=clusters, cfg=cfg, seed=seed)
                if "house_digest" in c else None)
        for i, m in enumerate(members):
            v = next(v for v in out["verdicts"] if v["digest"] == m)
            cleared = bool(res[i]["cleared"] and (hres is None or hres[i]["cleared"]))
            if cleared != bool(v["cleared"]):
                return None
            v["lcb"], v["gain"] = float(res[i]["lcb"]), float(res[i]["gain"])
            if hres is not None:
                v["house"].update(lcb=float(hres[i]["lcb"]), gain=float(hres[i]["gain"]))
    return _resign(out)


@pytest.mark.parametrize("which", [("pass", "s1"), ("challenger", "s2")])
def test_a_ground_bootstrap_seed_fails_the_audit(runs, which):
    run = runs(which[0])
    r, cfg = run[which[1]], run["cfg"]
    assert r["seed"] == f"lane-d/1:{r['boundary']}:{r['pool']['sha256']}", \
        "fixture: the producer's seed is lane-d/1:{boundary}:{pool.sha256}"
    _passes(r, cfg, f"the honest {which}")
    for i in range(32):
        forged = _reseed(r, cfg, f"lane-d/1:{r['boundary']}:ground-{i}")
        if forged is not None:
            break
    else:
        raise AssertionError("fixture: no other seed keeps every clear")
    _fails(forged, cfg, f"{which} re-judged under a ground seed")


def test_a_cohort_judged_against_another_king_fails_the_audit(runs):
    """Y's first verdict at boundary(2) relabelled as judged against the weak
    house instead of KING_D = X (rows, LCBs and outcome all consistent with
    that relabel, held as an ungated clear): a cohort's king must be
    `king_before`."""
    run = runs("challenger")
    r, cfg = copy.deepcopy(run["s2"]), run["cfg"]
    _passes(run["s2"], cfg, "the honest challenger duel")
    y, x = run["y"]["digest"], run["x"]["digest"]
    assert r["king_before"]["digest"] == x
    v = _verdict(r, y)
    c = r["cohorts"][int(v["cohort"])]
    hd = c.pop("house_digest")
    c["king_digest"], c["king_reveal_ts"] = hd, int(r["house"]["reveal_ts"])
    c["t_max"] = max([int(r["house"]["reveal_ts"])]
                     + [int(_entry(r, m)["reveal_ts"]) for m in c["members"]])
    r["rows"] = [row for row in r["rows"]
                 if not (int(row["cohort"]) == int(c["index"]) and row["digest"] == x)]
    house_j = v.pop("house")
    v.update(king_digest=hd, lcb=house_j["lcb"], gain=house_j["gain"],
             cleared=house_j["cleared"], outcome=_held_outcome(runs))
    e = _entry(r, y)
    e["status"], e["reason"] = "hold", "bench_hold"
    _fails(_resign(r), cfg, "a first verdict judged against the house, not king_before")


def _era(suite: str) -> dict:
    h = _h()
    return {"suite": suite, "block": h.ARM_BLOCK, "block_hash": h.block_hash("era")}


def test_the_era_sentinel_record_is_checked_against_gate_sentinel(runs):
    sen, plain = runs("sentinel"), runs("pass")
    _passes(sen["s1"], sen["cfg"], "the sentinel crown, its era's record",
            era_sentinel=_era(SENTINEL))
    _passes(plain["s1"], plain["cfg"], "an unarmed era's crown", era_sentinel=None)
    _fails(plain["s1"], plain["cfg"], "gate.sentinel null in an armed era",
           era_sentinel=_era(SENTINEL))
    _fails(sen["s1"], sen["cfg"], "a sentinel stated in an unarmed era", era_sentinel=None)
    other = next(s for s in _h().GATE_SUITES if s != SENTINEL)
    _fails(sen["s1"], sen["cfg"], "gate.sentinel is not the era's draw",
           era_sentinel=_era(other))


def test_a_restated_sentinel_with_a_consistent_draw_fails_against_the_era(runs):
    """The sentinel run's gate re-stated with another sentinel and the draw
    re-derived for it (eligible, suite): self-consistent, so only the era
    record catches it."""
    h = _h()
    run = runs("sentinel")
    cfg = run["cfg"]
    _passes(run["s1"], cfg, "the honest sentinel crown", era_sentinel=_era(SENTINEL))
    r = copy.deepcopy(run["s1"])
    g = r["gate"]
    other = next(s for s in h.GATE_SUITES if s != SENTINEL)
    want = h.expected_gate_record(cfg, int(r["boundary"]), g["block_hash"], sentinel=other)
    g.update({k: want[k] for k in ("suite", "sentinel", "eligible")})
    _passes(_resign(r), cfg, "fixture: the restated gate is self-consistent")
    _fails(_resign(r), cfg, "a restated sentinel", era_sentinel=_era(SENTINEL))


# ── (d) W3b5: a lowfreq fail is lost, never gated, never crowned ────────────


def _pin_embargo_zero(pool_dir) -> None:
    """rev6.10 (a): every listed feed pins its `embargo_hours` (W6b) and W5a
    fails a feed without it closed. W3b5's fixture predates the key, so its
    snapshot gets `embargo_hours: 0` (the rev3 rule) on every listed feed."""
    import json
    from pathlib import Path

    meta_path = Path(pool_dir) / "metadata.json"
    meta = json.loads(meta_path.read_text())
    for f in (meta.get("feeds") or {}).values():
        f.setdefault("embargo_hours", 0.0)
    meta_path.write_text(json.dumps(meta, sort_keys=True))


def _lowfreq_fail_run(tmp_path) -> dict:
    """W3b5's T4b case at the first verdict (rev6.11): X clears on the hourly
    set at boundary(1) but regresses on the snapshot's low-frequency slice
    (every level 1.3x further from the target)."""
    from tests.accept.w3b5 import helpers as w3

    cfg = w3.lane_d_cfg()
    h0, x = w3.house(), w3.entry("x", reveal_ts=w3.T_FIRST)
    hd, xd = h0["digest"], x["digest"]
    path = tmp_path / "state" / "lane_d_state.json"
    lf1 = w3.lowfreq_records("lf1", 8, "D")
    pool1, recs1 = w3.snapshot(tmp_path, 1, w3.T_FIRST, lf1)
    _pin_embargo_zero(pool1)
    fc1 = w3.merge({hd: w3.hourly_forecasts(recs1, "bad", "house-0"),
                    xd: w3.hourly_forecasts(recs1, "sharp", "x")},
                   {hd: w3.lowfreq_forecasts(lf1, w3.flat(1.0)),
                    xd: w3.lowfreq_forecasts(lf1, w3.flat(1.3))})
    s1 = w3.settle(cfg, w3.boundary(1), pool1, [x], h0, fc1, w3.uids(x), path)
    v = w3.verdict_of(s1.report, xd)
    assert v is not None and v.get("stage", "first") == "first" and v["cleared"], \
        f"fixture: {v}"
    if "lcb" not in v["lowfreq"]:
        pytest.skip("W3b5 (#651) not on this base: verdicts[].lowfreq.lcb is absent, the "
                    "A10 check does not decide yet")
    assert v["lowfreq"]["result"] == "fail" and v["outcome"] == "lost", f"fixture: {v}"
    return {"w3": w3, "cfg": cfg, "x": x, "s1": s1.report}


def test_w3b5_an_honest_lowfreq_fail_passes_the_audit(tmp_path):
    run = _lowfreq_fail_run(tmp_path)
    r, x = run["s1"], run["x"]["digest"]
    assert r["crowned"] is None
    assert "gate" not in r or all(g["digest"] != x for g in r["gate"]["results"]), \
        "fixture: a lowfreq fail is never swept by the gate"
    _passes(r, run["cfg"], "a cleared first verdict lost to lowfreq_regression",
            helpers=run["w3"])
    _passes(r, run["cfg"], "the same, chain mode", helpers=run["w3"], prior=[])


def _lowfreq_doctor(run, how: str, held_outcome: str) -> dict:
    w3 = run["w3"]
    r = copy.deepcopy(run["s1"])
    x = run["x"]
    v = _verdict(r, x["digest"])
    if how in ("crowned", "crowned_result_pass"):
        if how == "crowned_result_pass":
            v["lowfreq"]["result"] = "pass"
        r["gate"] = _forged_gate(run["cfg"], w3.boundary(1), x["digest"], "lowfreq")
        _crown(r, x, 10)
    elif how == "result_pass_still_lost":
        v["lowfreq"]["result"] = "pass"
    elif how == "held_not_lost":
        v["outcome"] = held_outcome
        e = _entry(r, x["digest"])
        e["status"], e["reason"] = "hold", "bench_hold"
    else:
        raise KeyError(how)
    return _resign(r, w3)


@pytest.mark.parametrize("how", ["crowned", "crowned_result_pass", "result_pass_still_lost",
                                 "held_not_lost"])
def test_w3b5_a_doctored_lowfreq_fail_fails_the_audit(tmp_path, runs, how):
    run = _lowfreq_fail_run(tmp_path)
    _passes(run["s1"], run["cfg"], "the honest lowfreq fail", helpers=run["w3"])
    res = _audit(_lowfreq_doctor(run, how, _held_outcome(runs)), run["cfg"],
                 helpers=run["w3"])
    assert res.status == "FAIL", f"lowfreq fail doctored {how}: check_lane_d {res.status}"
