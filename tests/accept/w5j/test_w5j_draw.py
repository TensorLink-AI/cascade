"""W5j: the seeded per-bin draw as a pure function (README "Algorithm"; rev6.12 fresh-first draw and minimum fresh share).

Deterministic and re-derivable (consensus: external validators re-derive it):
the draw equals the reference algorithm exactly, is independent of input
order, and depends on the boundary block hash."""

from __future__ import annotations

import copy

import pytest

from .helpers import (
    HASH_A,
    HASH_B,
    HORIZON,
    cells_of,
    draw,
    forward_cell,
    lane_d_cfg,
    ref_for,
    seed_string,
    series,
    shuffled,
    view,
)

# Three fast timeframes, 8 sources, one window per series.
MIXED = {"10s": {f"src{i}": 6 for i in range(8)},
         "1min": {f"src{i}": 4 for i in range(8)},
         "5min": {f"src{i}": 2 for i in range(8)}}


def _ids(result) -> list[str]:
    return [w["window_id"] for w in result["windows"]]


def test_w5j_seed_is_the_block_hash_tagged_lane_d_windows():
    cfg = lane_d_cfg()
    out = draw(cells_of(MIXED), cfg, HASH_A)
    assert out["seed"] == seed_string(HASH_A) == f"{HASH_A}:lane_d_windows"
    assert out["boundary_block_hash"] == HASH_A
    assert out["max_windows_per_settlement"] == 60


@pytest.mark.parametrize("block_hash", [HASH_A, HASH_B])
def test_w5j_draw_equals_the_reference_algorithm(block_hash):
    cfg = lane_d_cfg()
    cells = cells_of(MIXED)
    out, ref = draw(cells, cfg, block_hash), ref_for(cells, cfg, block_hash)
    assert out["capacity"] == ref["capacity"]
    assert out["quota"] == ref["quota"]
    assert view(out) == ref["windows"], "the draw differs from the pinned algorithm"
    assert out["meets_floors"] is ref["meets_floors"]
    assert out["deferred"] is None


def test_w5j_draw_is_independent_of_input_order_and_never_mutates():
    cfg = lane_d_cfg()
    cells = cells_of(MIXED)
    before = copy.deepcopy(cells)
    a = draw(cells, cfg, HASH_A)
    assert cells == before, "draw_windows mutated its input"
    for s in (1, 2, 3):
        assert view(draw(shuffled(cells, s), cfg, HASH_A)) == view(a)


def test_w5j_draw_depends_on_the_block_hash():
    cfg = lane_d_cfg()
    cells = cells_of(MIXED)
    a, b = draw(cells, cfg, HASH_A), draw(cells, cfg, HASH_B)
    assert _ids(a) != _ids(b), "two boundary hashes drew the same windows"
    assert view(draw(cells, cfg, HASH_A)) == view(a), "the same hash drew twice differently"


def test_w5j_equal_quota_per_cell_when_every_cell_is_large():
    """60 over three large cells: equal shares, differing by at most one."""
    cfg = lane_d_cfg()
    cells = cells_of({f: {f"src{i}": 10 for i in range(8)} for f in ("10s", "1min", "5min")})
    out = draw(cells, cfg, HASH_A)
    q = out["quota"]
    assert sum(q.values()) == 60 and max(q.values()) - min(q.values()) <= 1, q
    assert q == ref_for(cells, cfg, HASH_A)["quota"]
    per_cell: dict[str, int] = {}
    for w in out["windows"]:
        k = f"{w['timeframe']}|{w['bin']}"
        per_cell[k] = per_cell.get(k, 0) + 1
    assert per_cell == q, "a cell was drawn above or below its quota"


def test_w5j_small_cell_taken_whole_and_surplus_re_split():
    """Capacities 5 / 80 / 80 under 60: the scarce cell gives all 5, the
    other two split the 55 left (27 then 28, smallest key first)."""
    cfg = lane_d_cfg()
    cells = cells_of({"10s": {f"src{i}": 10 for i in range(8)},
                      "1min": {f"src{i}": 10 for i in range(8)},
                      "5min": {"src0": 2, "src1": 1, "src2": 1, "src3": 1}})
    out = draw(cells, cfg, HASH_A)
    assert out["capacity"] == {"10s|forward": 80, "1min|forward": 80, "5min|forward": 5}
    assert out["quota"] == {"10s|forward": 27, "1min|forward": 28, "5min|forward": 5}
    assert sum(1 for w in out["windows"] if w["timeframe"] == "5min") == 5


def test_w5j_capacity_counts_non_overlapping_windows_at_h_lo():
    """A 200-point series with a gap: runs 130 and 69 hold 2 + 1 windows."""
    flags = [True] * 130 + [False] + [True] * 69
    cell = forward_cell("10s", [series("gap", "src0", "10s", 200, eligible=flags)])
    out = draw([cell], lane_d_cfg(min_windows=1, min_clusters=1,
                                  max_windows_per_settlement=60, max_source_share=1.0),
               HASH_A)
    assert out["capacity"] == {"10s|forward": 3}


def test_w5j_two_windows_of_one_series_never_overlap():
    cfg = lane_d_cfg()
    cells = cells_of({"10s": {f"src{i}": 2 for i in range(8)}}, windows_per_series=6)
    out = draw(cells, cfg, HASH_A)
    assert view(out) == ref_for(cells, cfg, HASH_A)["windows"]
    by_series: dict[str, list[int]] = {}
    for w in out["windows"]:
        by_series.setdefault(w["series_id"], []).append(int(w["origin"]))
    for sid, origins in by_series.items():
        origins.sort()
        gaps = [b - a for a, b in zip(origins, origins[1:], strict=False)]
        assert all(g >= HORIZON * 10 for g in gaps), f"{sid}: overlapping windows {origins}"


def test_w5j_window_fields_and_identity():
    cfg = lane_d_cfg()
    out = draw(cells_of(MIXED), cfg, HASH_A)
    for w in view(out):
        assert w["window_id"] == f"{w['series_id']}:{w['origin']}:h{w['h']}"
        assert w["bin"] == "forward" and w["kind"] == "forward" and w["h"] == HORIZON
        assert w["timeframe"] in MIXED and w["embargo_window"] is False
    assert _ids(out) == sorted(_ids(out))


def test_w5j_source_cap_after_the_draw_without_refill():
    """A dominant source is capped after the picks; dropped windows are not
    replaced, so the set ends below the quota sum."""
    cfg = lane_d_cfg()
    spec = {"10s": {"big": 40, **{f"src{i}": 3 for i in range(6)}}}
    cells = cells_of(spec)
    out = draw(cells, cfg, HASH_A)
    ref = ref_for(cells, cfg, HASH_A)
    assert view(out) == ref["windows"]
    n = len(out["windows"])
    big = sum(1 for w in out["windows"] if w["source"] == "big")
    assert big <= 0.25 * n, f"source 'big' holds {big}/{n} > max_source_share"
    assert n < sum(out["quota"].values()), "dropped windows were refilled"


def _embargo_cells(n_series: int, embargoed: int) -> list[dict]:
    """Hourly forward cell: ``embargoed`` of the series pass only at e > 0."""
    members = []
    for i in range(n_series):
        e0 = [False] * HORIZON if i < embargoed else None
        members.append(series(f"emb-{i:02d}", f"src{i % 8}", "H", HORIZON, e0=e0))
    return [forward_cell("H", members)]


@pytest.mark.parametrize(("s", "kept_embargo"), [(0.0, 30), (0.5, 18), (0.75, 6), (1.0, 0)])
def test_w5j_min_fresh_window_share(s, kept_embargo):
    """rev6.12 (b) step 3: after the source cap, embargo windows are trimmed
    (last-drawn first, no refill) until fresh windows make up >= s of the set;
    no fresh window is ever dropped for it."""
    cfg = lane_d_cfg(min_fresh_window_share=s, max_source_share=1.0)
    cells = _embargo_cells(48, 30)          # 18 fresh + 30 embargo, all drawn
    out = draw(cells, cfg, HASH_A)
    ref = ref_for(cells, cfg, HASH_A, fresh_share=s, max_source_share=1.0)
    assert view(out) == ref["windows"]
    emb = sum(1 for w in out["windows"] if w["embargo_window"])
    fresh = len(out["windows"]) - emb
    assert fresh == 18, "a fresh window was dropped by the fresh-share trim"
    assert emb == kept_embargo
    assert fresh >= s * len(out["windows"])
    assert (out["fresh_windows"], out["embargo_windows"]) == (fresh, emb)


def test_w5j_fresh_windows_are_drawn_first_in_each_cell():
    """Quota 24 over 20 fresh + 20 embargo candidates: every fresh one, then
    4 embargo ones; over 30 fresh + 20 embargo: fresh only."""
    cfg = lane_d_cfg(max_source_share=1.0, max_windows_per_settlement=24)
    out = draw(_embargo_cells(40, 20), cfg, HASH_A)
    assert view(out) == ref_for(_embargo_cells(40, 20), cfg, HASH_A, max_source_share=1.0,
                                max_windows=24)["windows"]
    assert (out["fresh_windows"], out["embargo_windows"]) == (20, 4)
    out2 = draw(_embargo_cells(50, 20), cfg, HASH_A)
    assert (out2["fresh_windows"], out2["embargo_windows"]) == (24, 0)


def test_w5j_the_trim_drops_the_last_drawn_embargo_windows():
    """The kept embargo windows are the earliest drawn (the reference keeps
    the draw-order prefix): checked across two hashes."""
    cfg = lane_d_cfg(min_fresh_window_share=0.75, max_source_share=1.0)
    for h in (HASH_A, HASH_B):
        cells = _embargo_cells(48, 30)
        assert view(draw(cells, cfg, h)) == ref_for(cells, cfg, h, fresh_share=0.75,
                                                    max_source_share=1.0)["windows"]


def test_w5j_fresh_share_short_when_the_trim_breaks_the_floors():
    """30 drawn windows (10 fresh) meet the floors; after the 0.5 trim 20 are
    left, below min_windows 24: not frozen, reason fresh_share_short."""
    cfg = lane_d_cfg(max_source_share=1.0, min_windows=24)
    out = draw(_embargo_cells(30, 20), cfg, HASH_A)
    assert len(out["windows"]) == 20 and out["meets_floors"] is False
    assert out["short_reason"] == "fresh_share_short"
    assert out["deferred"] is None


def test_w5j_plain_floor_miss_is_not_fresh_share_short():
    cfg = lane_d_cfg(max_source_share=1.0, min_windows=24)
    out = draw(_embargo_cells(20, 0), cfg, HASH_A)      # 20 fresh: too few anyway
    assert out["meets_floors"] is False and out["short_reason"] != "fresh_share_short"
    ok = draw(_embargo_cells(48, 0), cfg, HASH_A)
    assert ok["meets_floors"] is True and ok["short_reason"] is None


def test_w5j_total_never_exceeds_max_windows_per_settlement():
    cfg = lane_d_cfg(max_windows_per_settlement=40)
    cells = cells_of({f: {f"src{i}": 20 for i in range(8)} for f in ("10s", "1min")})
    out = draw(cells, cfg, HASH_A)
    assert len(out["windows"]) <= 40 and sum(out["quota"].values()) == 40
    assert view(out) == ref_for(cells, cfg, HASH_A, max_windows=40)["windows"]


def test_w5j_budget_below_min_windows_defers_window_budget():
    cfg = lane_d_cfg()
    out = draw(cells_of(MIXED), cfg, HASH_A, budget=23)
    assert out["deferred"] == "window_budget"
    assert out["windows"] == [] and set(out["quota"].values()) == {0}
    ok = draw(cells_of(MIXED), cfg, HASH_A, budget=30)
    assert ok["deferred"] is None and len(ok["windows"]) <= 30
    assert view(ok) == ref_for(cells_of(MIXED), cfg, HASH_A, budget=30)["windows"]


def test_w5j_result_is_plain_json():
    import json

    out = draw(cells_of(MIXED), lane_d_cfg(), HASH_A)
    assert json.loads(json.dumps(out, allow_nan=False)) == out
