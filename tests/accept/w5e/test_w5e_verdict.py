"""W5e through `settle_lane_d`: the verdict (paired cluster bootstrap,
`d_margin`, max-T), the mandatory confirmation duel on the next disjoint set,
the D20 king-change rule and `rows_sha256` (plan rev6 §1.3 Verdict, §1.5,
D20, T9b). README.md has the contract.

Every state comes from real settlements: X first clears against the genesis
house H0 on snapshot 1 at boundary(1) and confirms on snapshot 2 at
boundary(2). Most of these rules are already on main (W5b/W5c built them);
W5e owns them as its assertions. The D20 `"original"` rule after a house
refresh fails on main: the first-clear king is no longer known to the state,
so the confirmation is skipped as `king_unscored`.

RV5: an armed crowning fails closed without the gate inputs, so every
settlement here that may crown goes through `helpers.settle_crowning` (a
passing bench and a block hash); on main those fail on the AssertionError in
`helpers.settle_gated`. Settlements without a confirming clear keep the
legacy call.
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


def _first_clear(tmp_path, cfg=None, *, x_quality="sharp", house_quality="bad"):
    cfg = cfg or lane_d_cfg()
    h0 = house()
    x = entry("x", reveal_ts=T_X)
    path = tmp_path / "state" / "lane_d_state.json"
    pool1, recs1 = snapshot(tmp_path, 1)
    fc = _fc(recs1, {h0["digest"]: (house_quality, "house-0"), x["digest"]: (x_quality, "x")})
    s1 = settle(cfg, boundary(1), pool1, [x], h0, fc, uids(x), path)
    return cfg, h0, x, path, (pool1, recs1), s1


def test_a_first_clear_is_not_crowned_and_goes_confirm_pending(tmp_path):
    cfg, h0, x, path, (_, recs1), s1 = _first_clear(tmp_path)
    v = verdict_of(s1.report, x["digest"])
    assert v is not None, "X was not judged at its first freeze"
    assert v["stage"] == "first" and v["cleared"] is True, v
    assert v["outcome"] == "confirm_pending", "a first clear must not crown"
    assert s1.report["crowned"] is None
    assert s1.report["king_after"]["digest"] == h0["digest"]
    assert s1.pay["court_uids"][0] is None, "the house still holds KING_D (its slot burns)"
    (cohort,) = s1.report["cohorts"]
    assert cohort["set_sha256"] == set_sha(wids(recs1))
    rec = read_state(path)["entries"][x["digest"]]
    assert (rec["status"], rec["reason"]) == ("confirm_pending", "confirm_pending")
    conf = rec["confirm"]
    assert conf["first_set_sha256"] == cohort["set_sha256"]
    assert conf["king_digest"] == h0["digest"]
    assert conf["rule"] == "restart"
    assert conf["boundary"] == boundary(1)
    assert sorted(rec["used_windows"]) == wids(recs1)
    e = entry_of(s1.report, x["digest"])
    assert (e["status"], e["reason"]) == ("confirm_pending", "confirm_pending")


def test_confirmation_runs_on_the_next_disjoint_set_and_crowns(tmp_path):
    cfg, h0, x, path, (pool1, recs1), s1 = _first_clear(tmp_path)
    pool2, recs2 = snapshot(tmp_path, 2)
    fc = _fc(recs1 + recs2, {h0["digest"]: ("bad", "house-0"), x["digest"]: ("sharp", "x")})
    # Snapshot 1 is still retained: its windows are eligible for X but already
    # used, so the confirmation set must be snapshot 2's windows alone.
    s2 = settle_crowning(cfg, boundary(2), pool2, [x], h0, fc, uids(x), path, prior=[pool1],
                         candidates=[x["digest"]], references=[h0["digest"]], tag="b2")
    v = verdict_of(s2.report, x["digest"])
    assert v is not None, "X was not judged on its confirmation set"
    assert v["stage"] == "confirm" and v["cleared"] is True, v
    (cohort,) = [c for c in s2.report["cohorts"] if x["digest"] in c["members"]]
    assert sorted(cohort["window_ids"]) == wids(recs2)
    assert not set(cohort["window_ids"]) & set(wids(recs1)), \
        "the confirmation set shares a window with the first-clear set"
    assert rows_of(s2.report, x["digest"]) == set(wids(recs2))
    assert v["confirmation"]["stage"] == "confirm"
    assert v["confirmation"]["first_set_sha256"] == s1.report["cohorts"][0]["set_sha256"]
    assert v["confirmation"]["first_king_digest"] == h0["digest"]
    assert v["outcome"] == "crowned"
    assert s2.report["crowned"] == x["digest"]
    state = read_state(path)
    assert state["king"]["digest"] == x["digest"] and not state["king"]["house"]
    assert state["entries"][x["digest"]]["status"] == "crowned"
    assert s2.pay["court_uids"][0] == uids(x)[x["hotkey"]]


def test_a_clear_on_the_confirmation_set_fails_and_is_lost_confirmation(tmp_path):
    cfg, h0, x, path, _, _ = _first_clear(tmp_path)
    pool2, recs2 = snapshot(tmp_path, 2)
    fc = _fc(recs2, {h0["digest"]: ("fair", "house-0"), x["digest"]: ("bad", "x")})
    s2 = settle(cfg, boundary(2), pool2, [x], h0, fc, uids(x), path)
    v = verdict_of(s2.report, x["digest"])
    assert v is not None and v["stage"] == "confirm"
    assert v["cleared"] is False and v["outcome"] == "lost"
    rec = read_state(path)["entries"][x["digest"]]
    assert (rec["status"], rec["reason"]) == ("lost", "lost_confirmation")
    assert s2.report["crowned"] is None
    assert read_state(path)["king"]["digest"] == h0["digest"]


def test_confirm_pending_persists_until_a_fresh_set_freezes_and_across_restarts(tmp_path):
    cfg, h0, x, path, (pool1, recs1), s1 = _first_clear(tmp_path)
    conf1 = read_state(path)["entries"][x["digest"]]["confirm"]
    # boundary(2): no new snapshot, so no fresh window: X is not judged and
    # keeps its first clear.
    fc1 = _fc(recs1, {h0["digest"]: ("bad", "house-0"), x["digest"]: ("sharp", "x")})
    s2 = settle(cfg, boundary(2), pool1, [x], h0, fc1, uids(x), path)
    assert verdict_of(s2.report, x["digest"]) is None, "X was re-judged on used windows"
    rec = read_state(path)["entries"][x["digest"]]
    assert rec["status"] == "confirm_pending"
    assert rec["confirm"] == conf1, "the first clear was lost while waiting for a fresh set"
    # A restart: a new process reads the persisted state file.
    restarted = tmp_path / "restarted" / "lane_d_state.json"
    restarted.parent.mkdir()
    shutil.copyfile(path, restarted)
    pool2, recs2 = snapshot(tmp_path, 2)
    fc = _fc(recs1 + recs2, {h0["digest"]: ("bad", "house-0"), x["digest"]: ("sharp", "x")})
    gate = {"candidates": [x["digest"]], "references": [h0["digest"]], "tag": "b3"}
    a = settle_crowning(cfg, boundary(3), pool2, [x], h0, fc, uids(x), path, prior=[pool1],
                        **gate)
    b = settle_crowning(cfg, boundary(3), pool2, [x], h0, fc, uids(x), restarted,
                        prior=[pool1], **gate)
    for s in (a, b):
        v = verdict_of(s.report, x["digest"])
        assert v is not None and v["stage"] == "confirm", v
        assert v["confirmation"]["first_set_sha256"] == conf1["first_set_sha256"]
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
    _, _, x, _, _, s1 = _first_clear(tmp_path / "probe")
    lcb = verdict_of(s1.report, x["digest"])["lcb"]
    assert 0.0 < lcb < 1.0, f"fixture: X's LCB {lcb} is not inside (0, 1)"
    at = lane_d_cfg(d_margin=lcb)
    _, _, x, path, _, s = _first_clear(tmp_path / "at", at)
    v = verdict_of(s.report, x["digest"])
    assert v["lcb"] == lcb, "the LCB must not depend on d_margin"
    assert v["cleared"] is True, "LCB == d_margin must clear (LCB_α ≥ d_margin)"
    above = lane_d_cfg(d_margin=math.nextafter(lcb, 1.0))
    _, _, x, path, _, s = _first_clear(tmp_path / "above", above)
    v = verdict_of(s.report, x["digest"])
    assert v["cleared"] is False and v["outcome"] == "lost"
    rec = read_state(path)["entries"][x["digest"]]
    assert (rec["status"], rec["reason"]) == ("lost", "lost_duel")
    assert s.report["params"]["d_margin"] == above.d_margin


def _two_entry_first(tmp_path, order=(0, 1)):
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
    s1 = settle(cfg, boundary(1), pool1, ents, h0, _fc(recs1, models), uids(x1, x2), path)
    return cfg, h0, (x1, x2), path, models, s1


def test_max_t_crowns_the_best_of_a_confirming_cohort(tmp_path):
    cfg, h0, (x1, x2), path, models, s1 = _two_entry_first(tmp_path)
    for x in (x1, x2):
        v = verdict_of(s1.report, x["digest"])
        assert v is not None and v["cleared"] is True, f"fixture: {v}"
    pool2, recs2 = snapshot(tmp_path, 2)
    s2 = settle_crowning(cfg, boundary(2), pool2, [x1, x2], h0, _fc(recs2, models),
                         uids(x1, x2), path, candidates=[x1["digest"], x2["digest"]],
                         references=[h0["digest"]], tag="b2")
    (cohort,) = s2.report["cohorts"]
    assert cohort["stage"] == "confirm" and len(cohort["members"]) == 2
    vs = {x["digest"]: verdict_of(s2.report, x["digest"]) for x in (x1, x2)}
    assert all(v["cleared"] for v in vs.values()), vs
    # The verdict is the cohort's max-T judgment against KING_D on its set.
    again = _verdict_recompute(s2.report, cohort, cfg)
    for m, r in zip(cohort["members"], again, strict=True):
        for k in ("lcb", "gain", "cleared"):
            assert vs[m][k] == r[k], (m, k)
    best = max(vs, key=lambda d: vs[d]["lcb"])
    assert s2.report["crowned"] == best
    assert read_state(path)["king"]["digest"] == best
    other = next(d for d in vs if d != best)
    assert vs[other]["outcome"] != "crowned"
    assert read_state(path)["entries"][other]["status"] != "crowned"


def test_every_verdict_clears_iff_lcb_reaches_d_margin_and_gain_is_positive(tmp_path):
    cfg, _, _, _, _, s1 = _two_entry_first(tmp_path)
    assert s1.report["verdicts"]
    for v in s1.report["verdicts"]:
        assert v["cleared"] == (v["lcb"] >= cfg.d_margin and v["gain"] > 0.0), v


def test_t9b_d20_restart_house_refresh_discards_the_first_clear(tmp_path):
    cfg, h0, x, path, _, s1 = _first_clear(tmp_path)
    assert cfg.confirm_on_king_change == "restart"
    h1 = house("house-1", reveal_ts=HOUSE_T + HOUR)
    pool2, recs2 = snapshot(tmp_path, 2)
    fc = _fc(recs2, {h0["digest"]: ("bad", "house-0"), h1["digest"]: ("bad", "house-1"),
                     x["digest"]: ("sharp", "x")})
    s2 = settle(cfg, boundary(2), pool2, [x], h1, fc, uids(x), path)
    v = verdict_of(s2.report, x["digest"])
    assert v is not None, "X was not re-judged against the new king"
    assert v["stage"] == "first", "the first clear against H0 must be discarded (restart)"
    assert v["king_digest"] == h1["digest"]
    assert v["confirmation"]["stage"] == "first"
    assert v["confirmation"]["first_set_sha256"] is None
    assert v["cleared"] is True and v["outcome"] == "confirm_pending"
    assert s2.report["crowned"] is None, "a restarted duel crowned without a confirmation"
    rec = read_state(path)["entries"][x["digest"]]
    assert rec["status"] == "confirm_pending"
    assert rec["confirm"]["king_digest"] == h1["digest"]
    assert rec["confirm"]["boundary"] == boundary(2)
    assert rec["confirm"]["first_set_sha256"] == set_sha(wids(recs2))
    assert rows_of(s2.report, x["digest"]) == set(wids(recs2))


def test_t9b_d20_original_confirms_against_the_king_of_the_first_clear(tmp_path):
    cfg = lane_d_cfg(confirm_on_king_change="original")
    cfg, h0, x, path, _, s1 = _first_clear(tmp_path, cfg)
    h1 = house("house-1", reveal_ts=HOUSE_T + HOUR)
    pool2, recs2 = snapshot(tmp_path, 2)
    fc = _fc(recs2, {h0["digest"]: ("bad", "house-0"), h1["digest"]: ("bad", "house-1"),
                     x["digest"]: ("sharp", "x")})
    s2 = settle_crowning(cfg, boundary(2), pool2, [x], h1, fc, uids(x), path,
                         candidates=[x["digest"]], references=[h0["digest"], h1["digest"]],
                         tag="b2")
    v = verdict_of(s2.report, x["digest"])
    skips = [k["reason"] for k in s2.report["skipped"] if k["digest"] == x["digest"]]
    # The D20 "original" bug on main: H0 is no longer the king, the house or a
    # prior king, so its T is unknown and the confirmation is skipped.
    assert "king_unscored" not in skips, \
        "D20 original: the confirmation against the first-clear king was skipped as king_unscored"
    assert v is not None, f"X's confirmation against H0 did not run (skipped: {skips})"
    assert v["stage"] == "confirm" and v["king_digest"] == h0["digest"]
    assert v["confirmation"]["rule"] == "original"
    assert v["confirmation"]["first_king_digest"] == h0["digest"]
    assert v["cleared"] is True
    assert s2.report["crowned"] == x["digest"]
    assert read_state(path)["king"]["digest"] == x["digest"]


def test_rows_sha256_is_the_canonical_digest_of_the_ordered_rows(tmp_path):
    cfg, h0, (x1, x2), path, _, s1 = _two_entry_first(tmp_path)
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


def _fingerprint(report) -> dict:
    return {k: report[k] for k in ("cohorts", "verdicts", "rows", "rows_sha256", "judged",
                                   "entries", "seed", "crowned")}


def test_verdicts_are_deterministic_in_entry_input_order(tmp_path):
    *_, a = _two_entry_first(tmp_path / "a", order=(0, 1))
    *_, b = _two_entry_first(tmp_path / "b", order=(1, 0))
    assert _fingerprint(a.report) == _fingerprint(b.report)


HASHSEED_SCRIPT = r"""
import json, sys
from pathlib import Path
from tests.accept.w5e import test_w5e_verdict as t
*_, s = t._two_entry_first(Path(sys.argv[1]))
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
