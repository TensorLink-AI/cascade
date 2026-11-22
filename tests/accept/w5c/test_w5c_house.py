"""W5c: accepted Proposed design change 3 ("Copy of a fresh house can win"):
a miner challenger must also clear `d_margin` against the current house on
the same set, and the house ranks first on ties (plan rev6 §1.3, T16).

Setup (real settlements, no hand-written state): K1 clears against the
genesis house H0 at BOUNDARY_1 and confirms on a disjoint snapshot at
BOUNDARY_2, so a MINER holds KING_D. At BOUNDARY_3 lane A's king has moved:
the house refreshes to H1 (K1 keeps the throne), and challenger Y is judged
against KING_D = K1 on a fresh snapshot.

On main the verdict compares Y with K1 only, so the tie and the
worse-than-house cases fail on an assertion (Y goes `confirm_pending`), and
the house is not scored on the cohort's set.
"""

from __future__ import annotations

from .helpers import (
    ARM_BLOCK,
    BOUNDARY_1,
    BOUNDARY_2,
    BOUNDARY_3,
    DAY,
    SETTLEMENT,
    T0,
    band_start,
    entry,
    group,
    house,
    lane_d_cfg,
    model_forecasts,
    read_state,
    rows_of,
    settle,
    uids,
    verdict_of,
    wids,
    write_pool,
)

T_K1 = T0
T_H1 = T0 + DAY
T_Y = T0 + 2 * DAY


def _crowned_k1(tmp_path):
    """Run BOUNDARY_1 and BOUNDARY_2: K1 is crowned. Returns the state path,
    cfg, K1 and the uid map."""
    cfg = lane_d_cfg()
    h0 = house("house-0")
    k1 = entry("k1", reveal_ts=T_K1)
    p1 = group("p1", 6, band_start(T_K1))
    p2 = group("p2", 6, band_start(T_K1) + DAY)
    path = tmp_path / "state" / "lane_d_state.json"
    for boundary, recs, name in ((BOUNDARY_1, p1, "pool1"), (BOUNDARY_2, p2, "pool2")):
        pool = write_pool(tmp_path / name, recs, f"w5c-{name}")
        fc = {h0["digest"]: model_forecasts(recs, "bad", "house-0"),
              k1["digest"]: model_forecasts(recs, "fair", "k1")}
        settle(cfg, boundary, pool, [k1], h0, fc, uids(k1), path)
    state = read_state(path)
    assert state["king"]["digest"] == k1["digest"] and not state["king"]["house"], \
        "fixture: K1 was not crowned"
    return path, cfg, k1


def _duel(tmp_path, house_quality: str, y_quality: str | None):
    """BOUNDARY_3: house refreshed to H1 of `house_quality`; Y of `y_quality`,
    or an exact copy of H1's forecasts when `y_quality` is None."""
    path, cfg, k1 = _crowned_k1(tmp_path)
    h1 = house("house-1", reveal_ts=T_H1)
    y = entry("y", reveal_ts=T_Y, ready_block=ARM_BLOCK + 2 * SETTLEMENT + 40,
              reveal_block=ARM_BLOCK + 2 * SETTLEMENT + 60)
    recs = group("p3", 6, band_start(T_Y))
    pool = write_pool(tmp_path / "pool3", recs, "w5c-pool3")
    h_fc = model_forecasts(recs, house_quality, "house-1")
    fc = {k1["digest"]: model_forecasts(recs, "fair", "k1"), h1["digest"]: h_fc,
          y["digest"]: dict(h_fc) if y_quality is None else model_forecasts(recs, y_quality, "y")}
    s = settle(cfg, BOUNDARY_3, pool, [k1, y], h1, fc, uids(k1, y), path)
    state = read_state(path)
    assert state["house"]["digest"] == h1["digest"], "fixture: the house did not refresh"
    return s, state, k1, h1, y, recs


def _cohort_of(report, dg) -> dict:
    (c,) = [c for c in report["cohorts"] if dg in c["members"]]
    return c


def test_house_change3_copy_of_the_house_ties_and_loses(tmp_path):
    s, state, k1, h1, y, recs = _duel(tmp_path, "sharp", None)
    v = verdict_of(s.report, y["digest"])
    assert v is not None, "Y was not judged"
    assert v["king_digest"] == k1["digest"], "Y must be judged against KING_D"
    rec = state["entries"][y["digest"]]
    assert (rec["status"], rec["reason"]) == ("lost", "lost_duel"), \
        "a copy of the current house cleared: the house must rank first on ties"
    assert v["cleared"] is False and v["outcome"] == "lost"
    assert state["king"]["digest"] == k1["digest"]


def test_house_change3_challenger_short_of_d_margin_vs_house_loses(tmp_path):
    s, state, k1, h1, y, _ = _duel(tmp_path, "sharp", "good")
    v = verdict_of(s.report, y["digest"])
    assert v is not None and v["king_digest"] == k1["digest"]
    rec = state["entries"][y["digest"]]
    assert (rec["status"], rec["reason"]) == ("lost", "lost_duel"), \
        "Y beat KING_D but not the current house by d_margin, and still cleared"
    assert v["cleared"] is False


def test_house_change3_control_beating_king_and_a_weak_house_clears(tmp_path):
    s, state, k1, h1, y, _ = _duel(tmp_path, "bad", "good")
    v = verdict_of(s.report, y["digest"])
    assert v is not None and v["cleared"] is True, v
    assert state["entries"][y["digest"]]["status"] == "confirm_pending"


def test_house_change3_challenger_clearing_king_and_house_clears(tmp_path):
    s, state, k1, h1, y, _ = _duel(tmp_path, "fair", "sharp")
    v = verdict_of(s.report, y["digest"])
    assert v is not None and v["cleared"] is True, v
    assert state["entries"][y["digest"]]["status"] == "confirm_pending"


def test_house_change3_house_is_scored_on_the_cohorts_set(tmp_path):
    s, _, k1, h1, y, recs = _duel(tmp_path, "sharp", None)
    cohort = _cohort_of(s.report, y["digest"])
    shared = set(cohort["window_ids"])
    assert shared == set(wids(recs))
    assert rows_of(s.report, k1["digest"]) == shared
    assert rows_of(s.report, h1["digest"]) == shared, \
        "the house was not scored on the same set as the cohort"
