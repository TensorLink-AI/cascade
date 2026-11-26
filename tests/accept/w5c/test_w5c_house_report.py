"""W5c (RV5 BLOCKING item on W5c): the signed report records the house
judgement, so a satellite or `cascade-audit` re-derives every verdict from
signed fields alone (plan rev6 §1.3, Proposed design change 3, §1.5, §14.3
rules 1-2: "every value a satellite needs ... is in the signed report").

When a miner holds KING_D and the house is a different model, `settle_lane_d`
scores the house as a second bar on the cohort's set (test_w5c_house.py). The
report must then carry:

* `cohorts[].house_digest`: the house digest the cohort was judged against
  (equal to `report["house"]["digest"]`);
* `verdicts[].house`: `{"digest", "lcb", "gain", "cleared"}`, the same max-T
  judgement as the KING_D one with the house as the bar, on the same rows,
  set, clusters and `report["seed"]`;
* `verdicts[].cleared` == (KING_D cleared: `lcb >= d_margin and gain > 0`)
  AND `house.cleared`; `lcb` / `gain` stay the KING_D judgement;
* the house's rows inside the cohort (rows per cohort =
  `len(window_ids) * (2 + len(members))`).

When the house holds KING_D there is one bar, and NEITHER key appears (the
report is byte-for-byte today's shape). See README.md "The house judgement in
the report" for why absent, not null and not a copy.

`check_lane_d` must PASS on every report these fixtures produce, and FAIL on
a re-signed report whose house judgement is omitted, nulled, swapped or
tampered. Schema validation of the new keys is W7a's (pending): nothing here
calls `validate_report` / `verify_report` on a report carrying them.

On main: the report has no house judgement and the audit FAILs every
miner-KING_D report ("72 rows; the cohorts account for 48"), so the tests
that need the new fields or an honest PASS fail on an assertion.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from .helpers import (
    ARM_BLOCK,
    BOUNDARY_1,
    BOUNDARY_2,
    BOUNDARY_3,
    DAY,
    SETTLEMENT,
    T0,
    T_A,
    T_B,
    T_C,
    band_start,
    entry,
    fixture_key,
    group,
    house,
    lane_d_cfg,
    model_forecasts,
    read_state,
    settle,
    uids,
    verdict_of,
    wids,
    write_pool,
)

T_K1 = T0
T_H1 = T0 + DAY
T_Y = T0 + 2 * DAY

HOUSE_KEYS = {"digest", "lcb", "gain", "cleared"}

# (house quality, Y quality or None = an exact copy of the house) ->
# (Y clears KING_D, Y clears the house). DEC-CA-0076: only the first applies
# (the house is no bar after a miner crown); the second is kept as the
# record of what the retired W5c change 3 would have decided.
SCENARIOS = {
    ("sharp", None): (True, False),      # copy of the house: ties, loses
    ("sharp", "good"): (True, False),    # beats KING_D, not the house
    ("bad", "good"): (True, True),       # beats both (weak house)
    ("fair", "sharp"): (True, True),     # beats both
}


def _chain(tmp_path, house_quality: str, y_quality: str | None) -> dict:
    """The test_w5c_house.py chain, keeping every report: BOUNDARY_1 (no
    entry listed), BOUNDARY_2 (K1 enters and is crowned over the genesis
    house H0 at its first verdict, rev6.11 (b)), BOUNDARY_3 (the house
    refreshes to H1; Y is judged against KING_D = K1 and the house, and is
    crowned when it clears both)."""
    cfg = lane_d_cfg()
    h0 = house("house-0")
    k1 = entry("k1", reveal_ts=T_K1)
    path = tmp_path / "state" / "lane_d_state.json"
    reports = []
    p1 = group("p1", 6, band_start(T_K1))
    p2 = group("p2", 6, band_start(T_K1) + DAY)
    for boundary, recs, name, listed in ((BOUNDARY_1, p1, "pool1", []),
                                         (BOUNDARY_2, p2, "pool2", [k1])):
        pool = write_pool(tmp_path / name, recs, f"w5c-{name}")
        fc = {h0["digest"]: model_forecasts(recs, "bad", "house-0"),
              k1["digest"]: model_forecasts(recs, "fair", "k1")}
        reports.append(settle(cfg, boundary, pool, listed, h0, fc, uids(k1), path).report)
    assert reports[-1]["crowned"] == k1["digest"], \
        "fixture: K1 was not crowned at its first verdict (BOUNDARY_2)"
    h1 = house("house-1", reveal_ts=T_H1)
    y = entry("y", reveal_ts=T_Y, ready_block=ARM_BLOCK + 2 * SETTLEMENT + 40,
              reveal_block=ARM_BLOCK + 2 * SETTLEMENT + 60)
    recs = group("p3", 6, band_start(T_Y))
    pool = write_pool(tmp_path / "pool3", recs, "w5c-pool3")
    h_fc = model_forecasts(recs, house_quality, "house-1")
    fc = {k1["digest"]: model_forecasts(recs, "fair", "k1"), h1["digest"]: h_fc,
          y["digest"]: dict(h_fc) if y_quality is None else model_forecasts(recs, y_quality, "y")}
    reports.append(settle(cfg, BOUNDARY_3, pool, [k1, y], h1, fc, uids(k1, y), path).report)
    state = read_state(path)
    assert state["house"]["digest"] == h1["digest"], "fixture: the house did not refresh"
    yv = verdict_of(reports[-1], y["digest"])
    want_king = y["digest"] if yv is not None and yv["cleared"] else k1["digest"]
    assert state["king"]["digest"] == want_king and not state["king"]["house"], \
        "fixture: a miner must hold KING_D (Y when it clears both bars, else K1)"
    return {"cfg": cfg, "reports": reports, "k1": k1, "h0": h0, "h1": h1, "y": y,
            "recs": recs}


@pytest.fixture(scope="module")
def chains(tmp_path_factory):
    """One chain per scenario, built once (the box is small)."""
    cache = {}

    def get(hq, yq):
        if (hq, yq) not in cache:
            cache[(hq, yq)] = _chain(tmp_path_factory.mktemp(f"w5c-{hq}-{yq}"), hq, yq)
        return cache[(hq, yq)]
    return get


def _y_cohort(report, y) -> dict:
    found = [c for c in report["cohorts"] if y["digest"] in c["members"]]
    assert len(found) == 1, "Y is not in exactly one cohort"
    return found[0]


def _scores(report, cohort, dg):
    by = {(r["window_id"], r["digest"]): (r["crps"], r["mase"])
          for r in report["rows"] if r["cohort"] == cohort["index"]}
    missing = [w for w in cohort["window_ids"] if (w, dg) not in by]
    assert not missing, f"no rows for {dg[:12]} on {len(missing)} of the cohort's windows"
    pairs = [by[(w, dg)] for w in cohort["window_ids"]]
    return (np.asarray([p[0] for p in pairs]), np.asarray([p[1] for p in pairs]))


def _judge(report, cohort, bar, cfg):
    from cascade.validator.lane_d import judge_cohort

    src = {w["window_id"]: w["source"] for w in report["windows"]}
    return judge_cohort(_scores(report, cohort, bar),
                        [_scores(report, cohort, m) for m in cohort["members"]],
                        clusters=[src[w] for w in cohort["window_ids"]], cfg=cfg,
                        seed=report["seed"])


def _audit(report, cfg):
    pytest.importorskip("bittensor.sp_core")
    from cascade.audit.lane_d import check_lane_d

    return check_lane_d(report, cfg, signer_hotkey=fixture_key("signer").ss58_address)


def _resign(report: dict) -> dict:
    """The validator itself re-signs a doctored body (a signature alone proves
    nothing about the verdict)."""
    from cascade.validator.lane_d_report import rows_sha256, sign_report

    out = copy.deepcopy(report)
    out["rows_sha256"] = rows_sha256(out["rows"])
    return sign_report(out, fixture_key("signer"))


# ── DEC-CA-0076: a miner KING_D is the only bar ──────────────────────────
#
# Owner 2026-11-24 (laned:tests-correction): "I just want [the house] to be
# for genesis of the comp until a king is crowned … remove the
# house-holds-king mechanism after that." W5c change 3 (the house as a
# second bar while a miner holds KING_D) is retired: these tests used to pin
# `cohorts[].house_digest`, `verdicts[].house` and the house's rows; they now
# pin that none of them appear and that Y crowns on beating KING_D alone.


@pytest.mark.parametrize("scenario", list(SCENARIOS), ids=lambda s: f"{s[0]}-{s[1]}")
def test_house_report_names_no_house_bar_after_a_miner_crown(chains, scenario):
    c = chains(*scenario)
    report, k1, h1, y = c["reports"][-1], c["k1"], c["h1"], c["y"]
    assert report["house"]["digest"] == h1["digest"]
    cohort = _y_cohort(report, y)
    assert cohort["king_digest"] == k1["digest"], "the cohort must be judged against KING_D"
    assert "house_digest" not in cohort, "the house is a second bar after a miner crown"
    v = verdict_of(report, y["digest"])
    assert v is not None and "house" not in v and v["king_digest"] == k1["digest"]


@pytest.mark.parametrize("scenario", list(SCENARIOS), ids=lambda s: f"{s[0]}-{s[1]}")
def test_house_report_verdict_is_the_king_d_judgement_alone(chains, scenario):
    c = chains(*scenario)
    report, cfg, k1, y = c["reports"][-1], c["cfg"], c["k1"], c["y"]
    cohort = _y_cohort(report, y)
    vs_king = _judge(report, cohort, k1["digest"], cfg)
    for m, rk in zip(cohort["members"], vs_king, strict=True):
        v = verdict_of(report, m)
        assert "house" not in v, "the verdict carries a house judgement after a miner crown"
        assert v["lcb"] == rk["lcb"] and v["gain"] == rk["gain"]
        king_cleared = bool(v["lcb"] >= float(cfg.d_margin) and v["gain"] > 0.0)
        assert v["cleared"] is king_cleared, "cleared is not the KING_D judgement alone"
        assert v["outcome"] == ("crowned" if v["cleared"] else "lost")
    assert verdict_of(report, y["digest"])["cleared"] is SCENARIOS[scenario][0], \
        "fixture: Y must beat KING_D in every scenario"


@pytest.mark.parametrize("scenario", list(SCENARIOS), ids=lambda s: f"{s[0]}-{s[1]}")
def test_house_report_rows_are_king_d_and_members_only(chains, scenario):
    c = chains(*scenario)
    report, h1 = c["reports"][-1], c["h1"]
    assert report["cohorts"], "fixture: no miner cohort was judged"
    for cohort in report["cohorts"]:
        assert h1["digest"] not in cohort["members"], "the house challenged a miner KING_D"
        got = [r for r in report["rows"] if r["cohort"] == cohort["index"]]
        assert len(got) == len(cohort["window_ids"]) * (1 + len(cohort["members"]))
    assert not [r for r in report["rows"] if r["digest"] == h1["digest"]], \
        "the house was scored after a miner crown"


# ── the house holds KING_D: one bar, no house judgement ────────────────────


def _house_king_reports(chains, tmp_path):
    """Reports where the house holds KING_D: BOUNDARY_1 (K1 vs H0) of the
    house chain, and a W5c three-member cohort vs the genesis house."""
    c = chains("bad", "good")
    a = group("bandA", 2, band_start(T_A))
    b = group("bandB", 2, band_start(T_B))
    cc = group("bandC", 6, band_start(T_C))
    recs = a + b + cc
    members = (entry("alpha", reveal_ts=T_A), entry("bravo", reveal_ts=T_B),
               entry("charlie", reveal_ts=T_C))
    h = house()
    fc = {h["digest"]: model_forecasts(recs, "bad", "house")}
    for e, q in zip(members, ("good", "fair", "sharp"), strict=True):
        fc[e["digest"]] = model_forecasts(recs, q, e["hotkey"])
    pool = write_pool(tmp_path / "pool", recs, "w5c-cohort")
    s = settle(lane_d_cfg(), BOUNDARY_1, pool, list(members), h, fc, uids(*members),
               tmp_path / "state" / "lane_d_state.json")
    assert s.report["cohorts"] and set(s.report["cohorts"][0]["window_ids"]) == set(wids(cc))
    # BOUNDARY_2: K1 is judged while the house holds KING_D (BOUNDARY_1
    # lists no entry, rev6.11 (b)).
    return [("house-chain-b2", c["reports"][1]), ("cohort", s.report)]


def test_house_holds_king_d_report_carries_no_house_judgement(chains, tmp_path):
    for name, report in _house_king_reports(chains, tmp_path):
        assert report["cohorts"], f"{name}: fixture judged nothing"
        for cohort in report["cohorts"]:
            assert cohort["king_digest"] == report["house"]["digest"]
            assert "house_digest" not in cohort, \
                f"{name}: the house holds KING_D; a second bar must not be named"
            rows = [r for r in report["rows"] if r["cohort"] == cohort["index"]]
            assert len(rows) == len(cohort["window_ids"]) * (1 + len(cohort["members"]))
        for v in report["verdicts"]:
            assert "house" not in v, f"{name}: a house judgement where the house is KING_D"


# ── the audit re-derives it ─────────────────────────────────────────────────


def test_audit_passes_every_report_the_w5c_fixtures_produce(chains, tmp_path):
    from cascade.audit.checks import PASS

    cfg = lane_d_cfg()
    for scenario in SCENARIOS:
        c = chains(*scenario)
        for boundary, report in zip((BOUNDARY_1, BOUNDARY_2, BOUNDARY_3), c["reports"],
                                    strict=True):
            res = _audit(report, c["cfg"])
            assert res.status == PASS, \
                f"{scenario} @ {boundary}: check_lane_d {res.status} {res.detail}"
    for name, report in _house_king_reports(chains, tmp_path):
        res = _audit(report, cfg)
        assert res.status == PASS, f"{name}: check_lane_d {res.status} {res.detail}"


def test_audit_fails_a_house_bar_added_after_a_miner_crown(chains):
    """DEC-CA-0076: a re-signed miner-KING_D report naming the house as a
    second bar is refused (a satellite never meets two readings)."""
    from cascade.audit.checks import FAIL, PASS

    c = chains("bad", "good")
    honest = c["reports"][-1]
    res = _audit(honest, c["cfg"])
    assert res.status == PASS, f"the honest report must pass first: {res.status} {res.detail}"
    doctored = copy.deepcopy(honest)
    _y_cohort(doctored, c["y"])["house_digest"] = c["h1"]["digest"]
    res = _audit(_resign(doctored), c["cfg"])
    assert res.status == FAIL, f"the audit accepted a house bar ({res.status})"
    assert "signature" not in res.detail and "rows_sha256" not in res.detail, res.detail


def test_audit_fails_a_house_judgement_where_the_house_holds_king_d(chains):
    """The house judgement is present iff the cohort's KING_D is not the
    report's house: a house-KING_D report carrying one (even a copy of the
    KING_D judgement) is refused, so a satellite never meets two readings."""
    from cascade.audit.checks import FAIL, PASS

    c = chains("bad", "good")
    honest = c["reports"][1]          # BOUNDARY_2: K1 against the house as KING_D
    res = _audit(honest, c["cfg"])
    assert res.status == PASS, f"the honest report must pass first: {res.status} {res.detail}"
    doctored = copy.deepcopy(honest)
    (cohort,) = doctored["cohorts"]
    hd = doctored["house"]["digest"]
    assert cohort["king_digest"] == hd, "fixture: the house must hold KING_D at BOUNDARY_2"
    cohort["house_digest"] = hd
    for v in doctored["verdicts"]:
        v["house"] = {"digest": hd, "lcb": v["lcb"], "gain": v["gain"], "cleared": v["cleared"]}
    res = _audit(_resign(doctored), c["cfg"])
    assert res.status == FAIL, f"a house judgement on a house-KING_D report passed ({res.status})"
    assert "signature" not in res.detail and "rows_sha256" not in res.detail, res.detail
