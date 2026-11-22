"""W5c through `settle_lane_d`: cohort formation, the intersection eligible for
the cohort's max-T, latest-T deferral, floors after the intersection, the
max-T comparison against KING_D, determinism (plan rev6 §1.3 Cohorts, T9).

The pool holds three §1.3 bands, a day apart:

* band A (8 windows): eligible for T_A only;
* band B (8 windows): eligible for T_A and T_B;
* band C (24 windows): eligible for T_A, T_B and T_C.

So A freezes on 40 windows, B on 32, C on 24, and the cohort {A, B, C} is
judged on band C alone. The genesis house holds KING_D (T earlier than all).
"""

from __future__ import annotations

import json

import numpy as np

from .helpers import (
    BOUNDARY_1,
    BOUNDARY_2,
    T_A,
    T_B,
    T_C,
    T_LATE,
    band_start,
    entry,
    group,
    house,
    lane_d_cfg,
    model_forecasts,
    read_state,
    rows_of,
    set_sha,
    settle,
    skipped_reasons,
    uids,
    verdict_of,
    wids,
    write_pool,
)


def _bands(a=2, b=2, c=6):
    return (group("bandA", a, band_start(T_A)), group("bandB", b, band_start(T_B)),
            group("bandC", c, band_start(T_C)))


def _entries():
    # Names chosen so digest order is NOT T order (checked in the sanity test).
    return (entry("alpha", reveal_ts=T_A), entry("bravo", reveal_ts=T_B),
            entry("charlie", reveal_ts=T_C))


def _forecasts(records, h, members, skip=None, house_quality="bad"):
    skip = skip or {}
    fc = {h["digest"]: model_forecasts(records, house_quality, "house")}
    for e in members:
        fc[e["digest"]] = model_forecasts(records, "good", e["digest"],
                                          skip=skip.get(e["digest"], ()))
    return fc


def _run(tmp_path, records, members, *, skip=None, boundary=BOUNDARY_1, order=None,
         state_dir="state", house_quality="bad"):
    cfg = lane_d_cfg()
    h = house()
    pool = write_pool(tmp_path / "pool", records, "w5c-cohort")
    fc = _forecasts(records, h, members, skip, house_quality)
    path = tmp_path / state_dir / "lane_d_state.json"
    s = settle(cfg, boundary, pool, list(order or members), h, fc, uids(*members), path)
    return s, path, h, pool, fc, cfg


def _short_c(band_c):
    """Two band-C windows per source that C has no forecast for: C's frozen set
    (24) minus them is 16, under the 24-window floor."""
    return wids(band_c[:8])


# ── fixture sanity ──────────────────────────────────────────────────────────


def test_cohort_fixture_bands_and_frozen_sets(tmp_path):
    a_band, b_band, c_band = _bands()
    A, B, C = _entries()
    assert [e["digest"] for e in sorted((A, B, C), key=lambda e: e["digest"])] != \
        [A["digest"], B["digest"], C["digest"]], "digest order must differ from T order"
    # A sharp house: every member loses, so state keeps each frozen set (a
    # first clear resets it for the disjoint confirmation set).
    s, path, *_ = _run(tmp_path, a_band + b_band + c_band, (A, B, C), house_quality="sharp")
    state = read_state(path)
    assert {state["entries"][e["digest"]]["status"] for e in (A, B, C)} == {"lost"}
    assert state["entries"][A["digest"]]["frozen_set"]["window_ids"] == \
        wids(a_band + b_band + c_band)
    assert state["entries"][B["digest"]]["frozen_set"]["window_ids"] == wids(b_band + c_band)
    assert state["entries"][C["digest"]]["frozen_set"]["window_ids"] == wids(c_band)
    for e in (A, B, C):
        assert state["entries"][e["digest"]]["frozen_set"]["frozen"] is True


# ── cohort formation + intersection for max-T ───────────────────────────────


def test_t9_cohort_is_every_entry_frozen_this_settlement_ordered_by_t(tmp_path):
    a_band, b_band, c_band = _bands()
    A, B, C = _entries()
    s, *_ = _run(tmp_path, a_band + b_band + c_band, (A, B, C), order=(C, A, B))
    (cohort,) = s.report["cohorts"]
    assert cohort["members"] == [A["digest"], B["digest"], C["digest"]], \
        "the cohort is not the frozen entries ordered by T"
    assert cohort["t_max"] == T_C, "the cohort's T is not max(T over cohort and king)"
    assert sorted(s.report["judged"]) == sorted(cohort["members"])


def test_t9_cohort_is_judged_on_the_intersection_eligible_for_max_t(tmp_path):
    a_band, b_band, c_band = _bands()
    A, B, C = _entries()
    s, _, h, *_ = _run(tmp_path, a_band + b_band + c_band, (A, B, C))
    (cohort,) = s.report["cohorts"]
    want = set(wids(c_band))
    assert sorted(cohort["window_ids"]) == sorted(want), \
        "windows eligible for a member's own T but not the cohort's max-T were judged"
    assert cohort["set_sha256"] == set_sha(want)
    for dg in (A["digest"], B["digest"], C["digest"], h["digest"]):
        assert rows_of(s.report, dg) == want, f"{dg[:8]} not scored on the shared set"
    assert not rows_of(s.report, A["digest"]) & set(wids(a_band + b_band))


def test_an_unfrozen_entry_is_not_a_cohort_member(tmp_path):
    a_band, b_band, c_band = _bands()
    A, B, C = _entries()
    late = entry("late", reveal_ts=T_LATE)       # no window is eligible for its T
    s, path, *_ = _run(tmp_path, a_band + b_band + c_band, (A, B, C, late))
    (cohort,) = s.report["cohorts"]
    assert late["digest"] not in cohort["members"]
    assert cohort["t_max"] == T_C, "an unfrozen entry's T entered the cohort's max-T"
    assert sorted(cohort["window_ids"]) == wids(c_band)
    rec = read_state(path)["entries"][late["digest"]]
    assert (rec["status"], rec["reason"]) == ("pending", "pending_floor")
    assert rec["frozen_set"]["frozen"] is False
    assert rows_of(s.report, late["digest"]) == set()


# ── floors after the intersection: latest-T deferral ────────────────────────


def test_t9_latest_t_member_is_deferred_when_the_intersection_misses_the_floors(tmp_path):
    from cascade.validator.lane_d_state import ENTRY_REASONS

    a_band, b_band, c_band = _bands()
    A, B, C = _entries()
    s, path, h, *_ = _run(tmp_path, a_band + b_band + c_band, (A, B, C),
                          skip={C["digest"]: _short_c(c_band)})
    (cohort,) = s.report["cohorts"]
    assert cohort["members"] == [A["digest"], B["digest"]], \
        "the latest-T member was not the one deferred"
    assert cohort["t_max"] == T_B, "the cohort's max-T was not recomputed after the deferral"
    want = set(wids(b_band + c_band))
    assert sorted(cohort["window_ids"]) == sorted(want)
    assert rows_of(s.report, A["digest"]) == want
    assert rows_of(s.report, B["digest"]) == want
    assert rows_of(s.report, h["digest"]) == want

    assert rows_of(s.report, C["digest"]) == set(), "a deferred entry was judged"
    assert verdict_of(s.report, C["digest"]) is None
    assert C["digest"] not in s.report["judged"]
    assert skipped_reasons(s.report, C["digest"]) == ["pending_floor"]
    rec = read_state(path)["entries"][C["digest"]]
    assert (rec["status"], rec["reason"]) == ("pending", "pending_floor"), rec
    assert rec["reason"] in ENTRY_REASONS
    assert rec["used_windows"] == [], "a deferred entry's windows were marked used"
    assert rec["frozen_set"]["frozen"] is True
    assert rec["frozen_set"]["window_ids"] == wids(c_band), "deferral changed the frozen set"


def test_deferred_entry_is_judged_next_settlement_on_its_frozen_set(tmp_path):
    a_band, b_band, c_band = _bands()
    A, B, C = _entries()
    records = a_band + b_band + c_band
    _, path, h, pool, _, cfg = _run(tmp_path, records, (A, B, C),
                                    skip={C["digest"]: _short_c(c_band)})
    frozen = read_state(path)["entries"][C["digest"]]["frozen_set"]

    fc = _forecasts(records, h, (A, B, C))         # the eval host now has C's windows
    s2 = settle(cfg, BOUNDARY_2, pool, [A, B, C], h, fc, uids(A, B, C), path)
    mine = [c for c in s2.report["cohorts"] if C["digest"] in c["members"]]
    assert len(mine) == 1, "the deferred entry was not judged at the next settlement"
    (cohort,) = mine
    assert cohort["members"] == [C["digest"]]
    assert sorted(cohort["window_ids"]) == frozen["window_ids"]
    assert cohort["set_sha256"] == frozen["set_sha256"]
    assert rows_of(s2.report, C["digest"]) == set(frozen["window_ids"])
    after = read_state(path)["entries"][C["digest"]]
    assert after["status"] == "confirm_pending", after
    assert after["confirm"]["first_set_sha256"] == frozen["set_sha256"]


def test_deferral_repeats_latest_t_first_until_the_floors_hold(tmp_path):
    a_band, b_band, c_band = _bands(a=6, b=2, c=6)
    A, B, C = _entries()
    skip = {C["digest"]: _short_c(c_band),
            # B lacks band B and 8 of band C: B's part of the intersection is 16
            B["digest"]: wids(b_band) + wids(c_band[8:16])}
    s, path, *_ = _run(tmp_path, a_band + b_band + c_band, (A, B, C), skip=skip)
    (cohort,) = s.report["cohorts"]
    assert cohort["members"] == [A["digest"]]
    assert cohort["t_max"] == T_A
    assert sorted(cohort["window_ids"]) == wids(a_band + b_band + c_band)
    state = read_state(path)
    for e in (B, C):
        rec = state["entries"][e["digest"]]
        assert (rec["status"], rec["reason"]) == ("pending", "pending_floor"), rec
        assert rows_of(s.report, e["digest"]) == set()
        assert skipped_reasons(s.report, e["digest"]) == ["pending_floor"]


def test_a_lone_member_short_after_the_intersection_is_deferred_not_judged(tmp_path):
    _, _, c_band = _bands()
    C = entry("charlie", reveal_ts=T_C)
    s, path, h, *_ = _run(tmp_path, c_band, (C,), skip={C["digest"]: _short_c(c_band)})
    assert s.report["cohorts"] == []
    assert s.report["rows"] == []
    assert s.report["verdicts"] == []
    rec = read_state(path)["entries"][C["digest"]]
    assert (rec["status"], rec["reason"]) == ("pending", "pending_floor")
    assert rec["frozen_set"]["frozen"] is True
    assert read_state(path)["king"]["digest"] == h["digest"]


# ── max-T against KING_D on the shared set ──────────────────────────────────


def test_verdicts_are_cohort_max_t_against_king_d_on_the_shared_set(tmp_path):
    from cascade.validator.lane_d import judge_cohort

    a_band, b_band, c_band = _bands()
    A, B, C = _entries()
    s, _, h, _, _, cfg = _run(tmp_path, a_band + b_band + c_band, (A, B, C))
    (cohort,) = s.report["cohorts"]
    assert cohort["king_digest"] == h["digest"]
    order = list(cohort["window_ids"])
    src = {w["window_id"]: w["source"] for w in s.report["windows"]}
    by = {(r["digest"], r["window_id"]): (r["crps"], r["mase"]) for r in s.report["rows"]}

    def scores(dg):
        pairs = [by[(dg, w)] for w in order]
        return (np.asarray([p[0] for p in pairs]), np.asarray([p[1] for p in pairs]))

    want = judge_cohort(scores(h["digest"]), [scores(m) for m in cohort["members"]],
                        clusters=[src[w] for w in order], cfg=cfg, seed=s.report["seed"])
    for m, w in zip(cohort["members"], want, strict=True):
        v = verdict_of(s.report, m)
        assert v["king_digest"] == h["digest"] and v["cohort"] == cohort["index"]
        assert v["lcb"] == w["lcb"] and v["gain"] == w["gain"], \
            "a verdict is not the cohort's joint max-T result against KING_D"
        assert v["cleared"] is w["cleared"]


# ── determinism ─────────────────────────────────────────────────────────────


def _strip(state: dict) -> dict:
    s = json.loads(json.dumps(state))
    s.pop("last_report", None)
    return s


def test_cohort_and_deferral_are_deterministic_in_input_order(tmp_path):
    a_band, b_band, c_band = _bands()
    A, B, C = _entries()
    skip = {C["digest"]: _short_c(c_band)}
    out = []
    for name, order in (("one", (A, B, C)), ("two", (C, B, A))):
        s, path, *_ = _run(tmp_path / name, a_band + b_band + c_band, (A, B, C), skip=skip, order=order)
        out.append((s.report, _strip(read_state(path))))
    (r1, st1), (r2, st2) = out
    for key in ("cohorts", "verdicts", "rows", "rows_sha256", "judged", "entries"):
        assert r1[key] == r2[key], f"report {key} depends on input order"
    assert sorted(map(json.dumps, r1["skipped"])) == sorted(map(json.dumps, r2["skipped"]))
    assert st1 == st2, "state depends on input order"
