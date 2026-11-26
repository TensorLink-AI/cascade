"""W6a: `pool_d` selection = `pool`'s selected series ids, joined on
(source, pre-suffix source series key, channel) (plan §3.0 W6a row, §1.7).

On main each test fails at its first import of `cascade.pool.pool_d_select`
(in PATHS).
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pytest

from cascade.pool import builder
from cascade.pool.sources.tsbench_forge import TsbenchForgeSource

from .helpers import (
    CTX,
    ListSource,
    hs,
    lane_a_cfg,
    no_fetch,
    select_mod,
    standard_sources,
    tree_sha,
    wave,
)


def _run(sources, cfg):
    return select_mod().collect_with_selection(sources, CTX, cfg, no_fetch)


@pytest.mark.parametrize("selection", ["first", "balanced"])
def test_pool_bytes_identical_to_build_pool(tmp_path, selection):
    """`pool` is untouched: same inputs, same bytes (§1.7 "Lane A must not change")."""
    sel = select_mod()
    cfg = lane_a_cfg(max_series_per_domain=2, selection=selection,
                     selection_seed="as_of:2026-11-20" if selection == "balanced" else "")
    plain = builder.build_pool(standard_sources(), tmp_path / "plain", CTX, cfg, fetch=no_fetch)
    summary, selection_out = sel.build_pool_with_selection(
        standard_sources(), tmp_path / "joined", CTX, cfg, fetch=no_fetch)
    assert tree_sha(tmp_path / "plain") == tree_sha(tmp_path / "joined")
    assert summary.n_series == plain.n_series
    assert len(selection_out.series) > 0


def test_each_source_harvested_once(tmp_path):
    """Lane A and the join read the SAME records of ONE harvest."""
    sel = select_mod()
    sources = standard_sources()
    sel.build_pool_with_selection(sources, tmp_path / "p", CTX, lane_a_cfg(), fetch=no_fetch)
    assert [s.calls for s in sources] == [1, 1]


def test_selection_is_lane_a_selected_records(tmp_path):
    """Joined, not re-selected: every selected series is a lane A record and
    every lane A record is either selected or refused with a typed reason."""
    sel = select_mod()
    cfg = lane_a_cfg(max_series_per_domain=2)
    records, _drops, selection = _run(standard_sources(), cfg)
    lane_a_ids = {r.series_id for r in records}
    selected_ids = {s.pool_series_id for s in selection.series}
    refused_ids = {sid for sid, _ in selection.refused}
    assert selected_ids <= lane_a_ids
    assert selected_ids | refused_ids == lane_a_ids
    assert not (selected_ids & refused_ids)
    assert all(reason in sel.REFUSAL_REASONS for _, reason in selection.refused)
    by_id = {r.series_id: r for r in records}
    for s in selection.series:
        rec = by_id[s.pool_series_id]
        assert s.cluster == rec.metadata["source"]
        assert s.freq == rec.metadata["freq"]
        assert s.domain == rec.domain


def test_lane_a_dropped_series_never_selected():
    cfg = lane_a_cfg(max_series_per_domain=2)
    records, drops, selection = _run(standard_sources(), cfg)
    assert drops["degenerate"] == 1 and drops["too_short"] == 1 and drops["domain_cap"] == 2
    bases = {s.key[1] for s in selection.series}
    assert not ({"flat", "short", "osaka"} & bases)
    assert ("grid", "st_a", 0) not in selection.join_keys()      # energy cap: 2 of 3
    assert {"grid_load_r1", "grid_load_r2", "tokyo"} <= bases


def test_join_key_survives_suffix_reassignment():
    """Emitted ids move with the collision suffixes (`st_a` <-> `st_a-2` when
    the source order flips); the join keys and pool_d ids do not."""
    sel = select_mod()
    cfg = lane_a_cfg()
    fwd_records, _, fwd = _run(standard_sources(), cfg)
    rev_records, _, rev = _run(list(reversed(standard_sources())), cfg)

    def lane_a_id(records, source):
        return {r.series_id for r in records
                if r.series_id.startswith("st_a") and r.metadata["source"] == source}

    assert lane_a_id(fwd_records, "grid") == {"st_a"}
    assert lane_a_id(rev_records, "grid") == {"st_a-2"}          # the suffix moved
    for result in (fwd, rev):
        assert ("grid", "st_a", 0) in result.join_keys()
        assert ("weather", "st_a", 0) in result.join_keys()
    assert fwd.join_keys() == rev.join_keys()
    ids = {s.key: s.pool_d_series_id for s in fwd.series}
    assert ids == {s.key: s.pool_d_series_id for s in rev.series}
    assert ids[("grid", "st_a", 0)] != ids[("weather", "st_a", 0)]
    assert ids[("grid", "st_a", 0)] == sel.pool_d_series_id(("grid", "st_a", 0)) == "st_a~grid.c0"
    for pid in ids.values():
        assert ":" not in pid and "/" not in pid


def test_join_key_survives_tail_move():
    """The content hash changes whenever the tail moves; the key does not."""
    cfg = lane_a_cfg()
    a = ListSource("grid", [hs("meter 1", wave(600, 21), source="grid")])
    b = ListSource("grid", [hs("meter 1", wave(660, 21), source="grid")])   # 60 newer points
    rec_a, _, sel_a = _run([a], cfg)
    rec_b, _, sel_b = _run([b], cfg)
    assert rec_a[0].content_hash != rec_b[0].content_hash
    assert sel_a.join_keys() == sel_b.join_keys() == {("grid", "meter_1", 0)}
    assert [s.pool_d_series_id for s in sel_a.series] == [s.pool_d_series_id for s in sel_b.series]


def test_key_collision_in_one_source_refused_and_logged(tmp_path, caplog):
    """Two series of ONE source whose pre-suffix keys collide: both refused
    from pool_d (logged); lane A keeps both, bytes unchanged."""
    sel = select_mod()
    cfg = lane_a_cfg()

    def sources():
        return [ListSource("grid", [hs("a b", wave(600, 31), source="grid"),
                                    hs("a_b", wave(600, 32), source="grid"),
                                    hs("c", wave(600, 33), source="grid")])]

    builder.build_pool(sources(), tmp_path / "plain", CTX, cfg, fetch=no_fetch)
    with caplog.at_level(logging.WARNING, logger="cascade.pool"):
        _summary, selection = sel.build_pool_with_selection(
            sources(), tmp_path / "joined", CTX, cfg, fetch=no_fetch)
    assert tree_sha(tmp_path / "plain") == tree_sha(tmp_path / "joined")
    meta = json.loads((tmp_path / "joined" / "metadata.json").read_text())
    assert {"a_b", "a_b-2", "c"} == set(meta)                  # lane A keeps both
    assert dict(selection.refused) == {"a_b": "key_collision", "a_b-2": "key_collision"}
    assert selection.join_keys() == {("grid", "c", 0)}
    msgs = [r.getMessage() for r in caplog.records if r.name.startswith("cascade.pool")]
    for sid in ("a_b", "a_b-2"):
        assert any("pool_d_join_refused" in m and sid in m for m in msgs), msgs


def test_multichannel_record_one_key_per_channel():
    cfg = lane_a_cfg(max_channels=3)
    src = ListSource("grid", [hs("substation", wave(600, 41, channels=3), source="grid")])
    _records, _, selection = _run([src], cfg)
    assert selection.join_keys() == {("grid", "substation", c) for c in range(3)}
    assert sorted(s.pool_d_series_id for s in selection.series) == [
        "substation~grid.c0", "substation~grid.c1", "substation~grid.c2"]
    assert {s.n_channels for s in selection.series} == {3}
    assert {s.pool_series_id for s in selection.series} == {"substation"}


def test_no_source_refused():
    cfg = lane_a_cfg()
    src = ListSource("anon", [hs("x1", wave(600, 51), source=None),
                              hs("x2", wave(600, 52), source="feed2")])
    records, _, selection = _run([src], cfg)
    assert {r.series_id for r in records} == {"x1", "x2"}
    assert dict(selection.refused) == {"x1": "no_source"}
    assert selection.join_keys() == {("feed2", "x2", 0)}


def test_ambiguous_record_refused():
    """Lane A dedups two harvested series with identical values but different
    keys: the surviving record's key is undetermined, so it fails closed."""
    cfg = lane_a_cfg()
    same = wave(600, 61)
    src = ListSource("grid", [hs("left", same.copy(), source="grid"),
                              hs("right", same.copy(), source="grid"),
                              hs("other", wave(600, 62), source="grid")])
    records, drops, selection = _run([src], cfg)
    assert drops["duplicate"] == 1
    assert dict(selection.refused) == {"left": "ambiguous_record"}
    assert selection.join_keys() == {("grid", "other", 0)}


def test_selection_deterministic_and_hashable():
    cfg = lane_a_cfg(max_series_per_domain=2)
    _, _, one = _run(standard_sources(), cfg)
    _, _, two = _run(standard_sources(), cfg)
    doc = one.to_json()
    assert doc == two.to_json()
    assert doc["schema"] == "pool-d-selection/1"
    assert json.loads(json.dumps(doc)) == doc                     # plain JSON types
    assert one.sha256() == two.sha256()
    assert len(one.sha256()) == 64
    assert [s["pool_d_series_id"] for s in doc["series"]] == sorted(
        s["pool_d_series_id"] for s in doc["series"])
    other = lane_a_cfg(max_series_per_domain=1)
    _, _, three = _run(standard_sources(), other)
    assert three.sha256() != one.sha256()


def test_join_key_function():
    sel = select_mod()
    s = hs("grid load/r1", np.arange(10.0), source="grid")
    assert sel.join_key(s) == ("grid", "grid_load_r1", 0)
    assert sel.join_key(s, 2) == ("grid", "grid_load_r1", 2)


# ── forge mirror ────────────────────────────────────────────────────────────

def _forge_source(root):
    return TsbenchForgeSource(root, max_series_per_source=5)


def test_forge_mirror_pool_bytes_identical(tmp_path):
    from .helpers import forge_mirror

    sel = select_mod()
    root = forge_mirror(tmp_path / "forge")
    cfg = lane_a_cfg(cluster_by="source_family")
    builder.build_pool([_forge_source(root)], tmp_path / "plain", CTX, cfg, fetch=no_fetch)
    _summary, selection = sel.build_pool_with_selection(
        [_forge_source(root)], tmp_path / "joined", CTX, cfg, fetch=no_fetch)
    assert tree_sha(tmp_path / "plain") == tree_sha(tmp_path / "joined")
    assert selection.series


def test_forge_mirror_keys_are_catalog_feed_and_panel(tmp_path):
    from .helpers import forge_mirror

    root = forge_mirror(tmp_path / "forge")
    cfg = lane_a_cfg(cluster_by="source_family")
    records, _, selection = select_mod().collect_with_selection(
        [_forge_source(root)], CTX, cfg, no_fetch)
    assert selection.join_keys() == {
        ("grid_load", "tsforge__grid_load__region_r1", 0),
        ("grid_load", "tsforge__grid_load__region_r2", 0),
        ("air_temp", "tsforge__air_temp", 0),
    }
    by_key = {s.key: s for s in selection.series}
    r1 = by_key[("grid_load", "tsforge__grid_load__region_r1", 0)]
    assert r1.feed == "grid_load"
    lane_a = {r.series_id: r for r in records}[r1.pool_series_id]
    assert r1.cluster == lane_a.metadata["source"] == lane_a.metadata["source_family"]
    assert "fixture:grid" in r1.cluster              # lane A cluster key (source_family)
    assert r1.attrs.get("region") == "r1"
    assert dict(selection.refused) == {
        "tsforge__dup_panel__station_a_b": "key_collision",
        "tsforge__dup_panel__station_a_b-2": "key_collision",
    }
