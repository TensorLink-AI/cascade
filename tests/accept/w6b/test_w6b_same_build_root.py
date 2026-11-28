"""W6b: the snapshot uses the first-seen root THIS build committed (2026-11-28).

The builder's sequence at ``now`` (just after the 3-h boundary ``as_of``) is
``first_seen record`` -> ``first_seen commit`` -> ``pool_d_build --as-of-ts
as_of``. The commit at ``now`` writes the root of the previous build's period
(its points all first seen before ``as_of``), but the rev6.11 (d) rule only
took roots committed by ``as_of``, so that data reached lane D one build
(3 h) late. The CLI now builds under the ``ended`` rule: the same snapshot
bytes as if the root had been committed exactly at ``as_of``, re-derivable
later from the log.
"""

from __future__ import annotations

from .helpers import (
    AS1,
    CTX_A,
    LANE_A_CFG,
    PERIOD,
    T0,
    by_id,
    forge_mirror,
    forge_source,
    meta,
    mods,
    no_fetch,
    tree_sha,
)


def _lane_a(root, tmp_path):
    from cascade.pool import builder

    out = tmp_path / "lane_a"
    builder.build_pool([forge_source(root)], out, CTX_A, LANE_A_CFG, fetch=no_fetch)
    return out


def _build(b, root, lane_a, out, *extra):
    return b.main(["--forge-dir", str(root), "--first-seen-dir", str(root / "first_seen"),
                   "--lane-a-pool", str(lane_a), "--out", str(out), "--as-of-ts", str(AS1),
                   "--max-panel-series-per-feed", "5", "--lane-a-selection", "replay",
                   *extra])


def test_snapshot_uses_the_root_this_build_committed(tmp_path):
    b, _sel, fs = mods()
    # The live order: build k recorded at T0 + 10 min; build k+1 runs at
    # AS1 + 10 min: record, commit (period T0, committed after AS1), build at AS1.
    late = forge_mirror(tmp_path / "late")
    log = fs.FirstSeenLog(late / "first_seen")
    log.record(late, now_ts=T0 + 600)
    log.record(late, now_ts=AS1 + 600)
    log.commit(now_ts=AS1 + 600)
    assert log.roots()[T0]["committed_ts"] == AS1 + 600 > AS1
    lane_a = _lane_a(late, tmp_path)
    rc = _build(b, late, lane_a, tmp_path / "late_out")
    assert rc == 0, "the build did not use the root its own commit wrote"
    m = meta(tmp_path / "late_out")
    assert m["series"] and m["first_seen_roots"] == [log.roots()[T0]["root"]]
    assert max(max(s["first_seen_ts"]) for s in m["series"]) == T0 + 600
    assert not b.validate_pool_d(tmp_path / "late_out")

    # Byte-identical to the snapshot cut when that root was committed at AS1
    # exactly (the old rule's best case): nothing a satellite checks changes.
    ontime = forge_mirror(tmp_path / "ontime")
    log2 = fs.FirstSeenLog(ontime / "first_seen")
    log2.record(ontime, now_ts=T0 + 600)
    log2.commit(now_ts=AS1)
    assert _build(b, ontime, lane_a, tmp_path / "ontime_out", "--first-seen-roots",
                  "committed") == 0
    assert tree_sha(tmp_path / "late_out") == tree_sha(tmp_path / "ontime_out")
    assert by_id(meta(tmp_path / "late_out")) == by_id(meta(tmp_path / "ontime_out"))

    # Re-derived later (more records and roots since): the same bytes.
    log.commit(now_ts=AS1 + PERIOD + 600)
    log.record(late, now_ts=AS1 + PERIOD + 700)
    assert _build(b, late, lane_a, tmp_path / "again", "--no-first-seen-cache") == 0
    assert tree_sha(tmp_path / "again") == tree_sha(tmp_path / "late_out")

    # The old rule stays available: it sees no committed-by-AS1 root here.
    assert _build(b, late, lane_a, tmp_path / "old", "--first-seen-roots", "committed") != 0
