"""W-LAUNCH pool_d builder OOM (2026-11-26): the first-seen index used by
``pool_d_build`` is bounded by one feed.

``python -P -m cascade.pool.pool_d_build`` over a 20 GB / 7,002-feed
``records.jsonl`` reached 39.3 GB RSS within 5 min on the 39 GB builder and
filled swap: ``FirstSeenLog.index()`` held every record of the log AND a
``(feed, panel_json, ts) -> (seen, root, baseline)`` dict of every committed
point before the build looked at its first feed.

Contract: ``FirstSeenLog.feed_index()`` (what ``pool_d_build`` and
``verify_snapshot`` use) has a peak memory that does not grow with the NUMBER
of feeds in the log: it streams the log once (32-byte leaf hashes for the root
checks, per-feed byte spans) and holds one feed's points at a time while the
caller walks the feeds. Its answers are identical to the previous whole-log
``index()`` (``tests/unit/test_first_seen_index_memory.py`` pins that, and the
``pool_d_build`` output bytes, against the previous implementation).

No pandas, no torch: the log is written line by line.
"""

from __future__ import annotations

import json
import tracemalloc
from pathlib import Path

T0 = 1_795_132_800            # 2026-11-20 00:00 UTC, a 3 h period boundary
PERIOD = 10_800
POINTS = 3000                 # per record, two records per feed
PANEL = '{"site":"s1"}'


def _log(fs, root: Path, n_feeds: int) -> Path:
    log = fs.FirstSeenLog(root)
    root.mkdir(parents=True, exist_ok=True)
    with open(log.records_path, "w", encoding="utf-8") as f:
        for k in range(2):
            for i in range(n_feeds):
                start = T0 - 86400 * 30 + k * POINTS * 600
                doc = {"schema": fs.LOG_SCHEMA, "feed": f"feed{i:03d}",
                       "partition": f"d{k}.parquet", "partition_sha256": f"{i:060d}{k:04d}",
                       "seen_ts": T0 + 10 + i, "period_start": T0, "baseline": k == 0,
                       "points": [[PANEL, start + j * 600] for j in range(POINTS)]}
                f.write(fs.canonical_json(doc).decode() + "\n")
    log.commit(now_ts=T0 + PERIOD)
    return root


def _peak(fn) -> int:
    tracemalloc.start()
    try:
        fn()
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def _walk(fs, log_dir: Path) -> None:
    """What pool_d_build does: feed by feed, the lag then point lookups."""
    idx = fs.FirstSeenLog(log_dir).feed_index(until_ts=T0 + PERIOD)
    for feed in idx.feeds():
        assert idx.feed_lag([feed])[feed]["max_lead_hours"] is not None
        for j in range(0, 2 * POINTS, 97):
            assert idx.first_seen(feed, PANEL, T0 - 86400 * 30 + j * 600) is not None
            assert idx.root_for(feed, PANEL, T0 - 86400 * 30 + j * 600)
    idx.release()


def test_pool_d_index_walk_peak_does_not_grow_with_feed_count(tmp_path):
    from cascade.pool import first_seen as fs

    small, large = _log(fs, tmp_path / "s", 2), _log(fs, tmp_path / "l", 16)
    p_small = _peak(lambda: _walk(fs, small))
    p_large = _peak(lambda: _walk(fs, large))
    assert p_large < 2.0 * p_small, (
        f"the pool_d first-seen index peak grows with the feed count: {p_small} B over 2 "
        f"feeds, {p_large} B over 16 (every feed's points held at once)")


def test_verify_snapshot_peak_does_not_grow_with_feed_count(tmp_path):
    from cascade.pool import first_seen as fs

    def run(log_dir: Path) -> int:
        idx = fs.FirstSeenLog(log_dir).index(until_ts=T0 + PERIOD)     # setup only
        roots = sorted({idx.root_for("feed001", PANEL, T0 - 86400 * 30)})
        ts = [T0 - 86400 * 30 + j * 600 for j in range(8)]
        meta = {"as_of_ts": T0 + PERIOD, "first_seen_roots": roots, "series": [
            {"series_id": "a", "feed": "feed001", "panel": json.loads(PANEL),
             "target_ts": ts, "first_seen_ts": [T0 + 11] * len(ts)}]}
        del idx
        return _peak(lambda: (fs.verify_snapshot(meta, log_dir) == []) or 1 / 0)

    p_small = run(_log(fs, tmp_path / "s", 2))
    p_large = run(_log(fs, tmp_path / "l", 16))
    assert p_large < 2.0 * p_small, (
        f"verify_snapshot loads every feed of the log: {p_small} B with 2 feeds logged, "
        f"{p_large} B with 16")
