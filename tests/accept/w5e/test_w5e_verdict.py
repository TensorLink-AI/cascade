"""W5e through `settle_lane_d`: the verdict (paired cluster bootstrap,
`d_margin`, max-T), crown at the FIRST verdict, `rows_sha256` (plan rev6
§1.3 Verdict, §1.5; rev6.11 (b): the confirmation duel is dropped). README.md
has the contract.

Every state comes from real settlements: X clears against the genesis house
H0 on snapshot 1 at boundary(1) and, with a passing §1.3b gate, is crowned
there. There is no confirmation stage, no `confirm_pending` state and no D20
king-change rule; `confirm_on_king_change` is parsed and ignored.

On main a first clear goes `confirm_pending` and waits for a second set, so
the crown tests fail on an AssertionError. The non-crowning verdict tests
(W5b/W5c built them) pass on main.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from .helpers import (
    HOUR,
    HOUSE_T,
    T_X,
    boundary,
    entry,
    entry_of,
    house,
    lane_d_cfg,
    model_forecasts,
    read_state,
    rows_of,
    set_sha,
    settle,
    settle_crowning,
    snapshot,
    uids,
    verdict_of,
    wids,
)

REPO = Path(__file__).resolve().parents[3]


def _fc(recs, models: dict) -> dict:
    """`{digest: forecasts}` from `{digest: (quality, seed name)}`."""
    return {dg: model_forecasts(recs, q, name) for dg, (q, name) in models.items()}


def _first_verdict(tmp_path, cfg=None, *, x_quality="sharp", house_quality="bad", gated=True):
    """X judged at its first freeze (boundary 1). `gated` passes a healthy
    bench and a block hash (a clear may crown); otherwise the legacy call,
    only for a fixture in which X does not clear."""
    cfg = cfg or lane_d_cfg()
    h0 = house()
    x = entry("x", reveal_ts=T_X)
    path = tmp_path / "state" / "lane_d_state.json"
    pool1, recs1 = snapshot(tmp_path, 1)
    fc = _fc(recs1, {h0["digest"]: (house_quality, "house-0"), x["digest"]: (x_quality, "x")})
    if gated:
        s1 = settle_crowning(cfg, boundary(1), pool1, [x], h0, fc, uids(x), path,
                             candidates=[x["digest"]], references=[h0["digest"]], tag="b1")
    else:
        s1 = settle(cfg, boundary(1), pool1, [x], h0, fc, uids(x), path)
    return cfg, h0, x, path, (pool1, recs1), s1


def _stage_is_first(obj) -> bool:
    """rev6.11 / W7a: a verdict's or cohort's `stage` is fixed to `first`, or
    dropped."""
    return obj.get("stage", "first") == "first"


def test_a_first_clear_is_crowned_at_the_first_verdict(tmp_path):
    cfg, h0, x, path, (_, recs1), s1 = _first_verdict(tmp_path)
    v = verdict_of(s1.report, x["digest"])
    assert v is not None, "X was not judged at its first freeze"
    assert _stage_is_first(v) and v["cleared"] is True, v
    assert v["outcome"] == "crowned", \
        f"rev6.11 (b): a first clear that passes A10 and the gate is crowned; got {v['outcome']}"
    assert v.get("confirmation") is None, "the confirmation block is removed (rev6.11 (b))"
    assert s1.report["crowned"] == x["digest"]
    assert s1.report["king_after"]["digest"] == x["digest"]
    (cohort,) = s1.report["cohorts"]
    assert _stage_is_first(cohort)
    assert cohort["set_sha256"] == set_sha(wids(recs1))
    state = read_state(path)
    assert state["king"]["digest"] == x["digest"] and not state["king"]["house"]
    rec = state["entries"][x["digest"]]
    assert rec["status"] == "crowned"
    assert rec.get("confirm") is None, "no `confirm` record (rev6.11 (b))"
    assert sorted(rec["used_windows"]) == wids(recs1)
    assert entry_of(s1.report, x["digest"])["status"] == "crowned"
    assert s1.pay["court_uids"][0] == uids(x)[x["hotkey"]]


def test_no_confirm_stage_or_status_exists(tmp_path):
    """`confirm_pending` leaves every open-status tuple, and no report or
    state the settlement writes names a confirm stage."""
    from cascade.validator import lane_d, lane_d_queue, lane_d_status

    for mod in (lane_d, lane_d_queue, lane_d_status):
        assert "confirm_pending" not in tuple(mod.OPEN_STATUSES), \
            f"{mod.__name__}.OPEN_STATUSES still holds confirm_pending"
    cfg, h0, x, path, _, s1 = _first_verdict(tmp_path)
    text = json.dumps(s1.report) + Path(path).read_text()
    for word in ("confirm_pending", "lost_confirmation"):
        assert word not in text, f"the settlement still writes {word}"


def test_the_crowned_king_is_not_judged_again_on_a_second_set(tmp_path):
    cfg, h0, x, path, (pool1, recs1), s1 = _first_verdict(tmp_path)
    assert s1.report["crowned"] == x["digest"], "fixture: X was not crowned"
    pool2, recs2 = snapshot(tmp_path, 2)
    fc = _fc(recs1 + recs2, {h0["digest"]: ("bad", "house-0"), x["digest"]: ("sharp", "x")})
    s2 = settle(cfg, boundary(2), pool2, [x], h0, fc, uids(x), path, prior=[pool1])
    assert verdict_of(s2.report, x["digest"]) is None, "the crowned king was judged again"
    assert s2.report["crowned"] is None
    assert read_state(path)["king"]["digest"] == x["digest"]
    assert all(_stage_is_first(c) for c in s2.report["cohorts"])


def test_a_failed_first_verdict_is_lost_duel(tmp_path):
    cfg, h0, x, path, _, s1 = _first_verdict(tmp_path, x_quality="bad",
                                             house_quality="fair", gated=False)
    v = verdict_of(s1.report, x["digest"])
    assert v is not None and _stage_is_first(v)
    assert v["cleared"] is False and v["outcome"] == "lost"
    rec = read_state(path)["entries"][x["digest"]]
    assert (rec["status"], rec["reason"]) == ("lost", "lost_duel")
    assert s1.report["crowned"] is None
    assert read_state(path)["king"]["digest"] == h0["digest"]


def test_the_first_verdict_is_restart_safe(tmp_path):
    """A restart (a fresh call on a copy of the prior state file) gives the
    same verdict, crown and `rows_sha256`."""
    cfg = lane_d_cfg()
    h0 = house()
    z = entry("z", reveal_ts=T_X)
    path = tmp_path / "state" / "lane_d_state.json"
    pool1, recs1 = snapshot(tmp_path, 1)
    models = {h0["digest"]: ("fair", "house-0"), z["digest"]: ("bad", "z")}
    settle(cfg, boundary(1), pool1, [z], h0, _fc(recs1, models), uids(z), path)
    restarted = tmp_path / "restarted" / "lane_d_state.json"
    restarted.parent.mkdir()
    shutil.copyfile(path, restarted)
    pool2, recs2 = snapshot(tmp_path, 2)
    x = entry("x", reveal_ts=T_X, after=1)
    models = {h0["digest"]: ("bad", "house-0"), x["digest"]: ("sharp", "x")}
    gate = {"candidates": [x["digest"]], "references": [h0["digest"]], "tag": "b2"}
    a = settle_crowning(cfg, boundary(2), pool2, [z, x], h0, _fc(recs2, models), uids(z, x),
                        path, **gate)
    b = settle_crowning(cfg, boundary(2), pool2, [z, x], h0, _fc(recs2, models), uids(z, x),
                        restarted, **gate)
    for s in (a, b):
        v = verdict_of(s.report, x["digest"])
        assert v is not None and _stage_is_first(v) and v["cleared"], v
        assert s.report["crowned"] == x["digest"]
    assert a.report["rows_sha256"] == b.report["rows_sha256"]
    assert a.report["verdicts"] == b.report["verdicts"]


def _verdict_recompute(report, cohort, cfg):
    """`judge_cohort` recomputed from the report's own rows over
    `cohort["window_ids"]`, members in cohort order, against KING_D."""
    from cascade.validator.lane_d import judge_cohort

    by = {(r["window_id"], r["digest"]): r for r in report["rows"]}
    src = {w["window_id"]: w["source"] for w in report["windows"]}
    ids = cohort["window_ids"]

    def scores(dg):
        return (np.asarray([by[(w, dg)]["crps"] for w in ids]),
                np.asarray([by[(w, dg)]["mase"] for w in ids]))

    return judge_cohort(scores(cohort["king_digest"]), [scores(m) for m in cohort["members"]],
                        clusters=[src[w] for w in ids], cfg=cfg, seed=report["seed"])


def test_d_margin_is_an_inclusive_bar_on_the_lcb(tmp_path):
    _, _, x, _, _, s1 = _first_verdict(tmp_path / "probe")
    lcb = verdict_of(s1.report, x["digest"])["lcb"]
    assert 0.0 < lcb < 1.0, f"fixture: X's LCB {lcb} is not inside (0, 1)"
    # rev6.23: the bar decays from d_margin_start to d_margin; both at the
    # LCB pin the bar there at every tenure.
    at = lane_d_cfg(d_margin=lcb, d_margin_start=lcb)
    _, _, x, path, _, s = _first_verdict(tmp_path / "at", at)
    v = verdict_of(s.report, x["digest"])
    assert v["lcb"] == lcb, "the LCB must not depend on d_margin"
    assert v["cleared"] is True, "LCB == d_margin must clear (LCB_α ≥ d_margin)"
    above = lane_d_cfg(d_margin=math.nextafter(lcb, 1.0),
                       d_margin_start=math.nextafter(lcb, 1.0))
    _, _, x, path, _, s = _first_verdict(tmp_path / "above", above)
    v = verdict_of(s.report, x["digest"])
    assert v["cleared"] is False and v["outcome"] == "lost"
    rec = read_state(path)["entries"][x["digest"]]
    assert (rec["status"], rec["reason"]) == ("lost", "lost_duel")
    assert s.report["params"]["d_margin"] == above.d_margin


def _two_entry_first(tmp_path, order=(0, 1), *, gated=False):
    cfg = lane_d_cfg()
    h0 = house()
    x1 = entry("x1", reveal_ts=T_X)
    x2 = entry("x2", reveal_ts=T_X)
    path = tmp_path / "state" / "lane_d_state.json"
    pool1, recs1 = snapshot(tmp_path, 1)
    models = {h0["digest"]: ("bad", "house-0"), x1["digest"]: ("sharp", "x1"),
              x2["digest"]: ("good", "x2")}
    ents = [x1, x2]
    ents = [ents[i] for i in order]
    if gated:
        s1 = settle_crowning(cfg, boundary(1), pool1, ents, h0, _fc(recs1, models),
                             uids(x1, x2), path, candidates=[x1["digest"], x2["digest"]],
                             references=[h0["digest"]], tag="b1")
    else:
        s1 = settle(cfg, boundary(1), pool1, ents, h0, _fc(recs1, models), uids(x1, x2), path)
    return cfg, h0, (x1, x2), path, models, s1


def test_max_t_crowns_the_best_of_a_clearing_cohort_at_the_first_verdict(tmp_path):
    cfg, h0, (x1, x2), path, models, s1 = _two_entry_first(tmp_path, gated=True)
    (cohort,) = s1.report["cohorts"]
    assert _stage_is_first(cohort) and len(cohort["members"]) == 2
    vs = {x["digest"]: verdict_of(s1.report, x["digest"]) for x in (x1, x2)}
    assert all(v["cleared"] for v in vs.values()), f"fixture: {vs}"
    # The verdict is the cohort's max-T judgment against KING_D on its set.
    again = _verdict_recompute(s1.report, cohort, cfg)
    for m, r in zip(cohort["members"], again, strict=True):
        for k in ("lcb", "gain", "cleared"):
            assert vs[m][k] == r[k], (m, k)
    best = max(vs, key=lambda d: vs[d]["lcb"])
    assert s1.report["crowned"] == best
    assert read_state(path)["king"]["digest"] == best
    other = next(d for d in vs if d != best)
    assert vs[other]["outcome"] != "crowned"
    assert read_state(path)["entries"][other]["status"] != "crowned"


def test_every_verdict_clears_iff_lcb_reaches_d_margin_and_gain_is_positive(tmp_path):
    cfg, _, _, _, _, s1 = _two_entry_first(tmp_path)
    assert s1.report["verdicts"]
    # rev6.23: the bar in force at the settlement (tenure 0: d_margin_start).
    cb = s1.report.get("crown_bar")
    assert isinstance(cb, dict), "the report records no crown_bar (rev6.23)"
    bar = cb["margin"]
    assert bar == cfg.crown_bar(s1.report["crown_bar"]["tenure"]) >= cfg.d_margin
    for v in s1.report["verdicts"]:
        assert v["cleared"] == (v["lcb"] >= bar and v["gain"] > 0.0), v


def test_confirm_on_king_change_is_parsed_and_ignored(tmp_path):
    """rev6.11 (b): `confirm_on_king_change` is a legacy key, parsed and
    ignored. Both values give the same settlement, and it leaves the
    report's config echo."""
    reps = {}
    for rule in ("restart", "original"):
        cfg = lane_d_cfg(confirm_on_king_change=rule)
        *_, s1 = _first_verdict(tmp_path / rule, cfg)
        reps[rule] = s1.report
        assert "confirm_on_king_change" not in s1.report["params"], \
            "confirm_on_king_change must leave the config echo (rev6.11 (b))"
    a, b = reps["restart"], reps["original"]
    for k in ("cohorts", "verdicts", "rows_sha256", "entries", "crowned", "params"):
        assert a[k] == b[k], f"{k} depends on the ignored confirm_on_king_change"


def test_a_house_refresh_never_discards_or_restarts_a_crown(tmp_path):
    """No D20: a house refresh after X's first-verdict crown leaves X king
    (a new house enters as a challenger, W5f), and X is not re-judged."""
    cfg, h0, x, path, (pool1, recs1), s1 = _first_verdict(tmp_path)
    assert s1.report["crowned"] == x["digest"], "fixture: X was not crowned"
    h1 = house("house-1", reveal_ts=HOUSE_T + HOUR)
    pool2, recs2 = snapshot(tmp_path, 2)
    fc = _fc(recs2, {h0["digest"]: ("bad", "house-0"), h1["digest"]: ("bad", "house-1"),
                     x["digest"]: ("sharp", "x")})
    s2 = settle(cfg, boundary(2), pool2, [x], h1, fc, uids(x), path)
    assert read_state(path)["king"]["digest"] == x["digest"]
    assert s2.report["crowned"] is None
    v = verdict_of(s2.report, x["digest"])
    assert v is None, f"the crowned king was re-judged after a house refresh: {v}"


def test_rows_sha256_is_the_canonical_digest_of_the_ordered_rows(tmp_path):
    cfg, h0, (x1, x2), path, _, s1 = _two_entry_first(tmp_path, gated=True)
    rep = s1.report
    canon = json.dumps(rep["rows"], sort_keys=True, separators=(",", ":"),
                       allow_nan=False).encode("utf-8")
    assert rep["rows_sha256"] == hashlib.sha256(canon).hexdigest()
    (cohort,) = rep["cohorts"]
    scored = {cohort["king_digest"], *cohort["members"]}
    # Window-major in the cohort's set order, every scored model on every window.
    order = []
    for r in rep["rows"]:
        if not order or order[-1] != r["window_id"]:
            order.append(r["window_id"])
    assert order == cohort["window_ids"]
    for w in cohort["window_ids"]:
        assert {r["digest"] for r in rep["rows"] if r["window_id"] == w} == scored
    for r in rep["rows"]:
        assert r["cohort"] == cohort["index"]
        assert math.isfinite(r["crps"]) and math.isfinite(r["mase"])
    assert read_state(path)["last_report"]["rows_sha256"] == rep["rows_sha256"]
    tampered = [dict(r) for r in rep["rows"]]
    tampered[0]["crps"] = tampered[0]["crps"] * 1.000001
    from cascade.validator.lane_d_report import rows_sha256

    assert rows_sha256(tampered) != rep["rows_sha256"]
    assert rows_sha256(rep["rows"]) == rep["rows_sha256"]
    assert rows_of(rep, x1["digest"]) == set(cohort["window_ids"])


def _fingerprint(report) -> dict:
    return {k: report[k] for k in ("cohorts", "verdicts", "rows", "rows_sha256", "judged",
                                   "entries", "seed", "crowned")}


def test_verdicts_are_deterministic_in_entry_input_order(tmp_path):
    *_, a = _two_entry_first(tmp_path / "a", order=(0, 1), gated=True)
    *_, b = _two_entry_first(tmp_path / "b", order=(1, 0), gated=True)
    assert _fingerprint(a.report) == _fingerprint(b.report)


HASHSEED_SCRIPT = r"""
import json, sys
from pathlib import Path
from tests.accept.w5e import test_w5e_verdict as t
*_, s = t._two_entry_first(Path(sys.argv[1]), gated=True)
print(json.dumps(t._fingerprint(s.report), sort_keys=True))
"""


def test_verdicts_are_independent_of_the_hash_seed(tmp_path):
    outs = []
    for seed in ("1", "2"):
        env = {"PYTHONPATH": str(REPO), "PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin",
               "HOME": os.environ.get("HOME", "/tmp")}
        proc = subprocess.run([sys.executable, "-P", "-c", HASHSEED_SCRIPT,
                               str(tmp_path / f"seed{seed}")],
                              capture_output=True, text=True, timeout=300, check=False,
                              env=env, cwd=str(REPO))
        assert proc.returncode == 0, proc.stderr[-4000:]
        outs.append(json.loads(proc.stdout))
    assert outs[0] == outs[1]
