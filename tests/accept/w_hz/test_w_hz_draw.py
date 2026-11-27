"""W-HZ: the band draw (owner 2026-11-27): one seeded draw per cohort from
the boundary block hash, the band and cut of every window the hash's, no
target overlap inside a series across bands, and equal counts per
(frequency class, band) cell. Contract: README.md."""

from __future__ import annotations

import random

from .helpers import HASH_A, HASH_B, band_cfg, need

DRAW = "cascade.validator.lane_d_draw"


def _cands(n_series=12, sources=4, *, embargo_every=0):
    """Per series one candidate per band, all ending at the series' newest
    point (``band_slots = 1``): they overlap, so a series gives one window."""
    out = []
    for i in range(n_series):
        end = 1_800_000_000 + i * 7
        for band, h in (("short", 64), ("medium", 480), ("long", 736)):
            out.append({"window_id": f"s{i}:{end - (h - 1) * 60}:h{h}", "series_id": f"s{i}",
                        "source": f"src{i % sources}", "freq": "min", "band": band, "h": h,
                        "t0": end - (h - 1) * 60, "t1": end,
                        "embargo_window": bool(embargo_every and i % embargo_every == 0)})
    return out


def test_the_draw_is_the_block_hashes_and_order_free():
    d = need(DRAW, "draw_band_windows")
    cfg = band_cfg(min_windows=3, min_clusters=2)
    c = _cands()
    a = d.draw_band_windows(c, boundary_block_hash=HASH_A, cfg=cfg)
    shuffled = list(c)
    random.Random(3).shuffle(shuffled)
    assert a == d.draw_band_windows(shuffled, boundary_block_hash=HASH_A, cfg=cfg)
    b = d.draw_band_windows(c, boundary_block_hash=HASH_B, cfg=cfg)
    assert [w["window_id"] for w in a["windows"]] != [w["window_id"] for w in b["windows"]]
    assert a["seed"] == f"{HASH_A}:lane_d_windows"


def test_no_two_windows_of_a_series_overlap_across_bands():
    d = need(DRAW, "draw_band_windows")
    cfg = band_cfg(min_windows=3, min_clusters=2)
    got = d.draw_band_windows(_cands(), boundary_block_hash=HASH_A, cfg=cfg)
    by: dict[str, list] = {}
    for w in got["windows"]:
        for a, b in by.get(w["series_id"], []):
            assert w["t1"] < a or b < w["t0"], w["window_id"]
        by.setdefault(w["series_id"], []).append((w["t0"], w["t1"]))


def test_cells_are_balanced_to_equal_counts_and_small_cells_left_out():
    d = need(DRAW, "draw_band_windows", "balance_cells")
    cfg = band_cfg(min_windows=3, min_clusters=2, min_cell_windows=3)
    got = d.draw_band_windows(_cands(n_series=15), boundary_block_hash=HASH_A, cfg=cfg)
    counts = {}
    for w in got["windows"]:
        counts[w["cell"]] = counts.get(w["cell"], 0) + 1
    assert counts == got["cells"] and len(set(counts.values())) == 1
    assert set(counts) == {"subdaily|short", "subdaily|medium", "subdaily|long"}
    # a cell below min_cell_windows is left out; the rest stay equal
    ws = [{"window_id": f"x{i}", "cell": "subdaily|short"} for i in range(5)]
    ws += [{"window_id": f"y{i}", "cell": "daily|short"} for i in range(4)]
    ws += [{"window_id": f"z{i}", "cell": "daily|long"} for i in range(2)]
    kept, cells, dropped = d.balance_cells(ws, 3)
    assert cells == {"daily|short": 4, "subdaily|short": 4} and dropped == {"daily|long": 2}
    assert [w["window_id"] for w in kept] == ["x0", "x1", "x2", "x3", "y0", "y1", "y2", "y3"]


def test_fresh_windows_are_drawn_first_and_the_fresh_share_holds():
    d = need(DRAW, "draw_band_windows")
    cfg = band_cfg(min_windows=3, min_clusters=2, min_fresh_window_share=0.75)
    got = d.draw_band_windows(_cands(n_series=24, embargo_every=2),
                              boundary_block_hash=HASH_A, cfg=cfg)
    n = len(got["windows"])
    assert n and got["fresh_windows"] >= 0.75 * n
    assert len({w["cell"] for w in got["windows"]}) == len(got["cells"])
    assert len(set(got["cells"].values())) == 1


def test_a_budget_below_the_floor_draws_nothing():
    d = need(DRAW, "draw_band_windows")
    got = d.draw_band_windows(_cands(), boundary_block_hash=HASH_A, cfg=band_cfg(), budget=2)
    assert got["windows"] == [] and got["deferred"] == "window_budget"
