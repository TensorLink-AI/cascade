"""W-LAUNCH pool_d builder OOM (2026-11-25): ``FirstSeenLog.record()`` is
bounded by one feed.

The first baseline ``python -P -m cascade.pool.first_seen record`` over 7,002
feeds / 14 GB of parquet reached 39.8 GB RSS on the 39 GB builder and was
OOM-killed: it kept every point of every feed (``known``) and every new record
until the end of the run, and loaded the whole log history the same way.

Contract: the peak memory of ``record()`` does not grow with the NUMBER of
feeds, on a baseline run (nothing logged yet) and on a later run over a long
history (the history is loaded one feed at a time). The logs it writes are
byte-identical to before (``tests/unit/test_first_seen_record_memory.py``
pins that against the previous implementation).

No pandas, no torch: ``partition_points`` is stubbed with a JSON reader.
"""

from __future__ import annotations

import json
import tracemalloc
from pathlib import Path

import pytest

T0 = 1_795_132_800            # 2026-11-20 00:00 UTC, a 3 h period boundary
PERIOD = 10_800
POINTS = 4000                 # per feed
PANEL = '{"site":"s1"}'


def _json_points(path: Path):
    doc = json.loads(path.read_text(encoding="utf-8"))
    return sorted({(str(pj), int(t)) for pj, t in doc["points"]})


@pytest.fixture
def fs(monkeypatch):
    from cascade.pool import first_seen

    monkeypatch.setattr(first_seen, "partition_points", _json_points)
    return first_seen


def _forge(root: Path, n_feeds: int, day: str = "2026-11-19", n_points: int = POINTS) -> Path:
    for i in range(n_feeds):
        d = root / "data" / f"feed{i:03d}"
        d.mkdir(parents=True, exist_ok=True)
        pts = [[PANEL, T0 - 86400 * 30 + k * 600] for k in range(n_points)]
        (d / f"{day}.parquet").write_text(json.dumps({"points": pts}), encoding="utf-8")
    return root


def _peak(fn) -> int:
    tracemalloc.start()
    try:
        fn()
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def test_baseline_peak_does_not_grow_with_feed_count(fs, tmp_path):
    small = _forge(tmp_path / "small", 2)
    large = _forge(tmp_path / "large", 16)
    p_small = _peak(lambda: fs.FirstSeenLog(tmp_path / "log-s").record(small, now_ts=T0))
    p_large = _peak(lambda: fs.FirstSeenLog(tmp_path / "log-l").record(large, now_ts=T0))
    assert fs.FirstSeenLog(tmp_path / "log-l").records()[-1]["baseline"] is True
    assert p_large < 2.0 * p_small, (
        f"baseline record() peak grows with the feed count: {p_small} B over 2 feeds, "
        f"{p_large} B over 16 (every feed's points held at once)")


def test_history_is_loaded_one_feed_at_a_time(fs, tmp_path):
    def history_run(name: str, n_feeds: int) -> int:
        forge = _forge(tmp_path / name, n_feeds)
        log = fs.FirstSeenLog(tmp_path / f"log-{name}")
        log.record(forge, now_ts=T0)
        log.commit(now_ts=T0 + PERIOD)
        # one small new partition in one feed: the rest of the history is
        # needed only for that feed's known points
        _forge(forge, 1, day="2026-11-20", n_points=10)
        return _peak(lambda: log.record(forge, now_ts=T0 + PERIOD + 1))

    p_small = history_run("small", 2)
    p_large = history_run("large", 16)
    assert p_large < 2.0 * p_small, (
        f"record() over a long history loads every feed at once: {p_small} B with 2 "
        f"feeds logged, {p_large} B with 16")
