"""§1.3 "Evidence accumulates, then freezes": `advance_frozen_set` (plan rev6
W5b row, T18e floors re-check).

An entry's eligible set is the union over every pool_d snapshot since T,
deduplicated by window id, of the windows W5a's `eligible_windows` admits for
max(T_E, T_K). It freezes at the first step whose union, after the
`max_source_share` cap, meets `min_windows` and `min_clusters`; a frozen set
never changes again.
"""

from __future__ import annotations

import json

from .helpers import (
    BOUND,
    DAY,
    HORIZON,
    HOUR,
    HOUSE_T,
    T0,
    capped_counts,
    lane_d_cfg,
    per_source,
    record,
    set_sha,
    snapshot_records,
    trap,
    wids,
    write_pool,
)

FOUR = {"src0": 3, "src1": 3, "src2": 3, "src3": 3}      # 12 windows, 4 sources


def _pool(tmp_path, name, records):
    from cascade.validator.lane_d import load_pool

    return load_pool(write_pool(tmp_path / name, records, snapshot=name), HORIZON)


def _advance(progress, pools, cfg, seed="lane-d/1:union", **kw):
    from cascade.validator.lane_d_frozen import advance_frozen_set

    kw.setdefault("t_entry", T0)
    kw.setdefault("t_king", HOUSE_T)
    return advance_frozen_set(progress, pools, cfg=cfg, seed=seed, **kw)


def _core(p: dict) -> tuple:
    """The pinned keys (README): other progress keys are the builder's."""
    return p["frozen"], p["window_ids"], p["set_sha256"]


def _source_of(*record_lists) -> dict[str, str]:
    return {w: r["source"] for rs in record_lists for r in rs
            for w in wids([r])}


def test_union_accumulates_across_snapshots_until_the_floors(tmp_path):
    cfg = lane_d_cfg()
    a, b = snapshot_records("a", FOUR, 0), snapshot_records("b", FOUR, 2)
    pa, pb = _pool(tmp_path, "a", a), _pool(tmp_path, "b", b)
    p1 = _advance(None, [pa], cfg)
    assert p1["frozen"] is False, "12 windows < min_windows 24 froze"
    assert p1["window_ids"] == sorted(wids(a))
    assert p1["set_sha256"] is None
    p2 = _advance(p1, [pb], cfg)
    assert p2["frozen"] is True, "the union of two snapshots meets the floors"
    assert p2["window_ids"] == sorted(wids(a) | wids(b))
    assert p2["set_sha256"] == set_sha(wids(a) | wids(b))
    # several snapshots in one step equal the same snapshots step by step
    assert _core(_advance(None, [pa, pb], cfg)) == _core(p2)
    assert _core(_advance(None, [pb, pa], cfg)) == _core(p2)


def test_union_dedups_by_window_id_across_snapshots(tmp_path):
    cfg = lane_d_cfg()
    a = snapshot_records("a", FOUR, 0)
    pa = _pool(tmp_path, "a", a)
    again = _pool(tmp_path, "a-again", a)          # the same windows, a new snapshot
    p1 = _advance(None, [pa], cfg)
    assert _core(_advance(p1, [pa], cfg)) == _core(p1), "re-reading a snapshot changed the union"
    assert _core(_advance(p1, [again], cfg)) == _core(p1), \
        "a window seen in two snapshots counted twice"
    half = _pool(tmp_path, "b", a[:6] + snapshot_records("b", FOUR, 2)[:6])
    p2 = _advance(p1, [half], cfg)
    assert len(p2["window_ids"]) == 18 and p2["frozen"] is False


def test_union_counts_only_forward_eligible_windows(tmp_path):
    cfg = lane_d_cfg()
    good = snapshot_records("a", FOUR, 0)
    traps = [trap("before_margin", "t-bm", "src0"), trap("publish_ahead", "t-pa", "src1")]
    p = _advance(None, [_pool(tmp_path, "a", good + traps)], cfg)
    assert p["window_ids"] == sorted(wids(good)), "a §1.3-refused window joined the union"
    # max(T_E, T_K): a king revealed after these targets makes them all ineligible
    late = _advance(None, [_pool(tmp_path, "a2", good)], cfg, t_king=T0 + 30 * DAY)
    assert late["window_ids"] == [] and late["frozen"] is False


def test_frozen_set_never_changes_after_freeze(tmp_path):
    cfg = lane_d_cfg()
    a, b = snapshot_records("a", FOUR, 0), snapshot_records("b", FOUR, 2)
    c = snapshot_records("c", {"src4": 9, "src5": 9}, 4)
    frozen = _advance(None, [_pool(tmp_path, "a", a), _pool(tmp_path, "b", b)], cfg)
    assert frozen["frozen"] is True
    before = json.dumps(frozen, sort_keys=True)
    later = _advance(frozen, [_pool(tmp_path, "c", c)], cfg, seed="lane-d/1:other")
    assert later == frozen, "a frozen set changed"
    assert json.dumps(frozen, sort_keys=True) == before, "advance mutated its input"


def test_union_skips_excluded_windows(tmp_path):
    """The confirmation set is disjoint (§1.3): windows a member was already
    judged on are passed as `exclude` and never re-enter its union."""
    cfg = lane_d_cfg()
    a = snapshot_records("a", FOUR, 0)
    used = sorted(wids(a))[:5]
    p = _advance(None, [_pool(tmp_path, "a", a)], cfg, exclude=used)
    assert p["window_ids"] == sorted(wids(a) - set(used))


def test_union_progress_round_trips_through_json(tmp_path):
    """Persistence in lane_d_state.json: the progress is plain JSON and a
    reloaded progress continues exactly like the in-memory one."""
    cfg = lane_d_cfg()
    a, b = snapshot_records("a", FOUR, 0), snapshot_records("b", FOUR, 2)
    pa, pb = _pool(tmp_path, "a", a), _pool(tmp_path, "b", b)
    p1 = _advance(None, [pa], cfg)
    text = json.dumps(p1, sort_keys=True, allow_nan=False)
    reloaded = json.loads(text)
    assert reloaded == p1
    assert _advance(reloaded, [pb], cfg) == _advance(p1, [pb], cfg)
    p2 = _advance(reloaded, [pb], cfg)
    assert json.loads(json.dumps(p2, allow_nan=False)) == p2


def test_t18e_floors_are_rechecked_after_the_cap(tmp_path):
    """24 windows over 4 sources meet the floors before the cap, but one source
    holds 15 of them: after the cap (Q = 3) only 12 remain, below
    min_windows, so the set does not freeze and keeps its uncapped union.
    Once other sources catch up the capped union freezes."""
    cfg = lane_d_cfg()
    assert cfg.max_source_share == 0.25
    a = snapshot_records("a", {"big": 15, "src1": 3, "src2": 3, "src3": 3}, 0)
    assert capped_counts({"big": 15, "src1": 3, "src2": 3, "src3": 3}, 0.25) == \
        {"big": 3, "src1": 3, "src2": 3, "src3": 3}
    p1 = _advance(None, [_pool(tmp_path, "a", a)], cfg)
    assert p1["frozen"] is False, "a set below the floors after the cap froze"
    assert p1["window_ids"] == sorted(wids(a)), "the uncapped union was not kept"
    b = snapshot_records("b", {"src1": 6, "src2": 6, "src3": 6, "src4": 6}, 2)
    p2 = _advance(p1, [_pool(tmp_path, "b", b)], cfg, seed="lane-d/1:freeze")
    assert p2["frozen"] is True
    src = _source_of(a, b)
    counts = {"big": 15, "src1": 9, "src2": 9, "src3": 9, "src4": 6}
    assert per_source(p2["window_ids"], src) == capped_counts(counts, 0.25)
    assert set(p2["window_ids"]) <= wids(a) | wids(b)
    assert p2["set_sha256"] == set_sha(p2["window_ids"])
    # the drop is the seeded one: same inputs, same frozen set
    again = _advance(json.loads(json.dumps(p1)), [_pool(tmp_path, "b2", b)], cfg,
                     seed="lane-d/1:freeze")
    assert _core(again) == _core(p2)


def test_union_reads_the_floors_from_config(tmp_path):
    a = snapshot_records("a", FOUR, 0)
    pa = _pool(tmp_path, "a", a)
    assert _advance(None, [pa], lane_d_cfg(min_windows=12, min_clusters=4))["frozen"] is True
    assert _advance(None, [pa], lane_d_cfg(min_windows=12, min_clusters=5))["frozen"] is False
    assert _advance(None, [pa], lane_d_cfg(min_windows=13, min_clusters=4))["frozen"] is False
    # max_source_share 0.5 admits a 3-source set that 0.25 empties
    three = snapshot_records("t", {"x": 4, "y": 4, "z": 4}, 0)
    pt = _pool(tmp_path, "t", three)
    loose = _advance(None, [pt], lane_d_cfg(min_windows=12, min_clusters=3,
                                            max_source_share=0.5))
    tight = _advance(None, [pt], lane_d_cfg(min_windows=12, min_clusters=3))
    assert loose["frozen"] is True and tight["frozen"] is False


def test_union_window_record_sanity():
    """The fixture's records are forward-eligible at BOUND (guards the
    fixture, not the code under test)."""
    r = record("x", "src0", BOUND + 3 * HOUR)
    assert min(r["target_ts"]) > BOUND and min(r["first_seen_ts"]) > BOUND
    assert r["end_ts"] == r["target_ts"][-1] and len(r["target_ts"]) == HORIZON
