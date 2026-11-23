"""W6fs: first-seen recording, 3-hourly Merkle roots, per-feed lag, snapshot
verification (plan §3.0 W6fs row; rev6.11 (d)).

On main each test fails at its first import of `cascade.pool.first_seen`
(in PATHS).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import statistics

import pytest

from .helpers import (
    HOUR,
    PERIOD,
    T0,
    canonical,
    catalog,
    day_of,
    fs_mod,
    hourly,
    ref_merkle,
    scrape,
    tree_sha,
)

DAY = day_of(T0)


def _mirror(tmp_path, feeds=("meter",)):
    root = tmp_path / "forge"
    catalog(root, [{"id": f, "domain": "energy", "frequency": "PT1H",
                    "source_family": f"fixture:{f}"} for f in feeds])
    return root


def _first_two_scrapes(tmp_path):
    """Scrape 1 (48 points, baseline) recorded in period T0; scrape 2 adds 3
    newer points, recorded in period T0 + 3 h."""
    fs = fs_mod()
    root = _mirror(tmp_path)
    log = fs.FirstSeenLog(root / "first_seen")
    old = hourly(T0 - 48 * HOUR, 48)
    scrape(root, "meter", DAY, old)
    assert log.record(root, now_ts=T0 + 600) == {"records": 1, "points": 48}
    new = hourly(T0, 3)
    scrape(root, "meter", DAY, old + new)
    assert log.record(root, now_ts=T0 + PERIOD + 600) == {"records": 1, "points": 3}
    return fs, root, log, old, new


def test_constants_and_helpers():
    fs = fs_mod()
    assert fs.PERIOD_SECONDS == 10800
    assert fs.LOG_SCHEMA == "pool-d-first-seen/1"
    assert fs.period_start(T0 + 10799) == T0 and fs.period_start(T0 + 10800) == T0 + 10800
    assert fs.canonical_json({"b": 1, "a": [1, 2]}) == b'{"a":[1,2],"b":1}'
    assert fs.panel_json({}) == "{}"
    assert fs.panel_json({"z": "1", "a": "x"}) == '{"a":"x","z":"1"}'
    leaves = [b"r1", b"r2", b"r3"]
    assert fs.merkle_root(leaves) == ref_merkle(leaves)
    assert fs.merkle_root([b"only"]) == ref_merkle([b"only"])
    with pytest.raises(ValueError):
        fs.merkle_root([])
    assert issubclass(fs.FirstSeenError, ValueError)


def test_record_first_seen_per_point(tmp_path):
    fs, root, log, old, new = _first_two_scrapes(tmp_path)
    log.commit(now_ts=T0 + 2 * PERIOD)
    idx = log.index(until_ts=T0 + 2 * PERIOD)
    for t in old:
        assert idx.first_seen("meter", {}, t) == T0 + 600
    for t in new:
        assert idx.first_seen("meter", {}, t) == T0 + PERIOD + 600
    assert idx.first_seen("meter", {}, T0 + 10 * HOUR) is None
    recs = log.records()
    assert [r["baseline"] for r in recs] == [True, False]
    for r in recs:
        assert r["schema"] == fs.LOG_SCHEMA and r["feed"] == "meter"
        assert r["partition"] == f"{DAY}.parquet"
        assert len(r["partition_sha256"]) == 64
        assert r["period_start"] == fs.period_start(r["seen_ts"])
        assert r["points"] == sorted(r["points"])
    assert recs[1]["points"] == [["{}", t] for t in new]


def test_unchanged_partition_skipped(tmp_path):
    fs, root, log, _old, _new = _first_two_scrapes(tmp_path)
    before = (root / "first_seen" / "records.jsonl").read_bytes()
    assert log.record(root, now_ts=T0 + PERIOD + 1200) == {"records": 0, "points": 0}
    assert (root / "first_seen" / "records.jsonl").read_bytes() == before


def test_log_is_append_only(tmp_path):
    fs = fs_mod()
    root = _mirror(tmp_path)
    log = fs.FirstSeenLog(root / "first_seen")
    pts = hourly(T0 - 10 * HOUR, 10)
    scrape(root, "meter", DAY, pts)
    log.record(root, now_ts=T0 + 60)
    first = (root / "first_seen" / "records.jsonl").read_bytes()
    scrape(root, "meter", DAY, pts + hourly(T0, 2))
    log.record(root, now_ts=T0 + PERIOD + 60)
    second = (root / "first_seen" / "records.jsonl").read_bytes()
    assert second.startswith(first) and len(second) > len(first)


def test_rewrite_keeps_first_time(tmp_path):
    """A partition rewritten with new values for the same timestamps adds no
    point and moves no first-seen time."""
    fs, root, log, old, new = _first_two_scrapes(tmp_path)
    scrape(root, "meter", DAY, old + new, values=[99.0 + i for i in range(len(old + new))])
    assert log.record(root, now_ts=T0 + 2 * PERIOD + 600)["points"] == 0
    log.commit(now_ts=T0 + 3 * PERIOD)
    idx = log.index(until_ts=T0 + 3 * PERIOD)
    assert idx.first_seen("meter", {}, old[0]) == T0 + 600
    assert idx.first_seen("meter", {}, new[-1]) == T0 + PERIOD + 600


def test_backfill_new_points_get_their_own_time(tmp_path):
    """Backfilled older points are new to the mirror: they get the time the
    mirror first saw them; existing points keep theirs."""
    fs, root, log, old, new = _first_two_scrapes(tmp_path)
    back = hourly(T0 - 60 * HOUR, 12)                       # older than anything seen
    scrape(root, "meter", day_of(T0 - 2 * 86_400), back)    # a second partition
    assert log.record(root, now_ts=T0 + 2 * PERIOD + 600) == {"records": 1, "points": 12}
    log.commit(now_ts=T0 + 3 * PERIOD)
    idx = log.index(until_ts=T0 + 3 * PERIOD)
    for t in back:
        assert idx.first_seen("meter", {}, t) == T0 + 2 * PERIOD + 600
    assert idx.first_seen("meter", {}, old[-1]) == T0 + 600


def test_panel_points_keyed_by_panel(tmp_path):
    fs = fs_mod()
    root = _mirror(tmp_path, feeds=("grid",))
    log = fs.FirstSeenLog(root / "first_seen")
    ts = hourly(T0 - 5 * HOUR, 5)
    scrape(root, "grid", DAY, ts + ts, panel={"region": ["r1"] * 5 + ["r2"] * 5})
    log.record(root, now_ts=T0 + 30)
    scrape(root, "grid", DAY, ts + ts + [T0], panel={"region": ["r1"] * 5 + ["r2"] * 6})
    log.record(root, now_ts=T0 + PERIOD + 30)
    log.commit(now_ts=T0 + 2 * PERIOD)
    idx = log.index(until_ts=T0 + 2 * PERIOD)
    assert idx.first_seen("grid", {"region": "r1"}, ts[0]) == T0 + 30
    assert idx.first_seen("grid", {"region": "r2"}, T0) == T0 + PERIOD + 30
    assert idx.first_seen("grid", {"region": "r1"}, T0) is None


def test_commit_roots_per_3h_period(tmp_path):
    fs, root, log, _old, _new = _first_two_scrapes(tmp_path)
    added = log.commit(now_ts=T0 + 2 * PERIOD)
    assert [e["period_start"] for e in added] == [T0, T0 + PERIOD]
    recs = log.records()
    roots = log.roots()
    assert set(roots) == {T0, T0 + PERIOD}
    for start, entry in roots.items():
        assert start % fs.PERIOD_SECONDS == 0
        mine = [r for r in recs if r["period_start"] == start]
        assert entry["n_records"] == len(mine)
        assert entry["root"] == ref_merkle([canonical(r) for r in mine])
        assert entry["committed_ts"] == T0 + 2 * PERIOD
    lines = (root / "first_seen" / "roots.jsonl").read_text().splitlines()
    assert len(lines) == 2
    assert log.commit(now_ts=T0 + 2 * PERIOD + 5) == []            # idempotent


def test_commit_skips_open_period(tmp_path):
    fs, root, log, _old, _new = _first_two_scrapes(tmp_path)
    added = log.commit(now_ts=T0 + PERIOD + 900)       # period T0+3h still open
    assert [e["period_start"] for e in added] == [T0]
    assert set(log.roots()) == {T0}


def test_record_refuses_committed_period(tmp_path):
    fs, root, log, old, new = _first_two_scrapes(tmp_path)
    log.commit(now_ts=T0 + 2 * PERIOD)
    scrape(root, "meter", DAY, old + new + [T0 + 3 * HOUR])
    with pytest.raises(fs.FirstSeenError) as err:
        log.record(root, now_ts=T0 + PERIOD + 4000)     # inside a committed period
    assert err.value.code == "period_committed"


def test_index_until_never_exposes_later_records(tmp_path):
    """First-seen is never later than a snapshot containing the point: an
    index cut at `until` holds only records of periods whose root was
    committed by then, so every first-seen it returns is < until."""
    fs, root, log, old, new = _first_two_scrapes(tmp_path)
    log.commit(now_ts=T0 + PERIOD)                  # period T0, committed at its end
    log.commit(now_ts=T0 + 2 * PERIOD)              # period T0 + 3 h
    idx = log.index(until_ts=T0 + PERIOD)
    assert idx.first_seen("meter", {}, old[0]) == T0 + 600 < T0 + PERIOD
    assert all(idx.first_seen("meter", {}, t) is None for t in new)
    # a root committed after `until` is not usable at `until`, even for an
    # ended period (re-deriving an old snapshot sees what it saw)
    late = fs.FirstSeenLog(tmp_path / "late")
    root2 = tmp_path / "forge"
    late.record(root2, now_ts=T0 + 600)
    late.commit(now_ts=T0 + 2 * PERIOD)
    assert late.index(until_ts=T0 + PERIOD).first_seen("meter", {}, old[0]) is None
    assert late.index(until_ts=T0 + 2 * PERIOD).first_seen("meter", {}, old[0]) == T0 + 600


def test_index_excludes_uncommitted_period(tmp_path):
    fs, root, log, old, new = _first_two_scrapes(tmp_path)
    log.commit(now_ts=T0 + PERIOD + 900)                  # only period T0 committed
    idx = log.index(until_ts=T0 + 5 * PERIOD)
    assert idx.first_seen("meter", {}, old[0]) == T0 + 600
    assert all(idx.first_seen("meter", {}, t) is None for t in new)
    assert idx.root_for("meter", {}, old[0]) == log.roots()[T0]["root"]
    assert idx.root_for("meter", {}, new[0]) is None


def test_index_refuses_rewritten_log(tmp_path):
    fs, root, log, _old, _new = _first_two_scrapes(tmp_path)
    log.commit(now_ts=T0 + 2 * PERIOD)
    path = root / "first_seen" / "records.jsonl"
    lines = path.read_text().splitlines()
    rec = json.loads(lines[1])
    rec["seen_ts"] -= 3600                                 # backdate a first-seen time
    lines[1] = json.dumps(rec)
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(fs.FirstSeenError) as err:
        fs.FirstSeenLog(root / "first_seen").index(until_ts=T0 + 2 * PERIOD)
    assert err.value.code == "root_mismatch"


def _meta(log, idx, ts_list, *, as_of, roots=None, first_seen=None):
    seen = first_seen or [idx.first_seen("meter", {}, t) for t in ts_list]
    used = roots if roots is not None else sorted(
        {idx.root_for("meter", {}, t) for t in ts_list} - {None})
    return {"schema": "pool_d/1", "as_of_ts": as_of, "first_seen_roots": used,
            "series": [{"series_id": "m", "feed": "meter", "panel": {},
                        "target_ts": list(ts_list), "first_seen_ts": seen}]}


def test_verify_snapshot_accepts_consistent_snapshot(tmp_path):
    fs, root, log, old, new = _first_two_scrapes(tmp_path)
    log.commit(now_ts=T0 + 2 * PERIOD)
    idx = log.index(until_ts=T0 + 2 * PERIOD)
    meta = _meta(log, idx, old[-3:] + new, as_of=T0 + 2 * PERIOD)
    assert fs.verify_snapshot(meta, root / "first_seen") == []


def test_verify_snapshot_refuses_uncommitted_and_late_roots(tmp_path):
    """A snapshot whose first-seen records postdate their committed root (or
    whose root is not committed by its as_of) is refused."""
    fs, root, log, old, new = _first_two_scrapes(tmp_path)
    log.commit(now_ts=T0 + PERIOD + 900)                  # period T0 only
    idx0 = log.index(until_ts=T0 + PERIOD)
    seen = [T0 + 600] * 3 + [T0 + PERIOD + 600] * 3
    # uses the new points, whose period has no committed root
    meta = _meta(log, idx0, old[-3:] + new, as_of=T0 + 2 * PERIOD,
                 roots=[log.roots()[T0]["root"]], first_seen=seen)
    assert fs.verify_snapshot(meta, root / "first_seen")
    # the root exists but was committed after the snapshot's as_of
    log.commit(now_ts=T0 + 3 * PERIOD)
    idx = log.index(until_ts=T0 + 3 * PERIOD)
    late = _meta(log, idx, old[-3:] + new, as_of=T0 + 2 * PERIOD)
    assert fs.verify_snapshot(late, root / "first_seen")
    ok = _meta(log, idx, old[-3:] + new, as_of=T0 + 3 * PERIOD)
    assert fs.verify_snapshot(ok, root / "first_seen") == []
    # a first-seen time that is not the log's, an unlisted root, an unknown point
    wrong = _meta(log, idx, old[-3:] + new, as_of=T0 + 3 * PERIOD,
                  first_seen=[t + 1 for t in seen])
    assert fs.verify_snapshot(wrong, root / "first_seen")
    unlisted = _meta(log, idx, old[-3:] + new, as_of=T0 + 3 * PERIOD, roots=[])
    assert fs.verify_snapshot(unlisted, root / "first_seen")
    unknown = _meta(log, idx, [T0 + 40 * HOUR], as_of=T0 + 3 * PERIOD, roots=[],
                    first_seen=[T0])
    assert fs.verify_snapshot(unknown, root / "first_seen")


def _lag_mirror(tmp_path):
    """Three feeds recorded every 3 h: `slow` publishes 2 h after data time,
    `fast` 30 min after, `ahead` carries values stamped 24 h in the future."""
    fs = fs_mod()
    root = _mirror(tmp_path, feeds=("slow", "fast", "ahead"))
    log = fs.FirstSeenLog(root / "first_seen")
    grid = hourly(T0 - 24 * HOUR, 24 + 72)
    lags = {"slow": 2 * HOUR, "fast": HOUR // 2, "ahead": -24 * HOUR}
    seen: dict[str, dict[int, int]] = {f: {} for f in lags}
    for k in range(5):
        now = T0 + k * PERIOD + 1800
        for feed, lag in lags.items():
            pts = [t for t in grid if t <= now - lag]
            scrape(tmp_path / "forge", feed, DAY, pts)
            for t in pts:
                seen[feed].setdefault(t, (now, k == 0))
        log.record(root, now_ts=now)
    log.commit(now_ts=T0 + 6 * PERIOD)
    return fs, log, seen


def test_mirror_lag_per_feed(tmp_path):
    fs, log, seen = _lag_mirror(tmp_path)
    lag = log.index(until_ts=T0 + 6 * PERIOD).feed_lag()
    assert set(lag) == {"slow", "fast", "ahead"}
    for feed in ("slow", "fast"):
        diffs = [(s - t) / 3600 for t, (s, base) in seen[feed].items() if not base and t <= s]
        assert lag[feed]["mirror_lag_hours"] == pytest.approx(statistics.median(diffs))
        assert lag[feed]["publish_ahead"] is False
    assert lag["slow"]["mirror_lag_hours"] > lag["fast"]["mirror_lag_hours"]


def test_publish_ahead_feed_measured(tmp_path):
    fs, log, seen = _lag_mirror(tmp_path)
    lag = log.index(until_ts=T0 + 6 * PERIOD).feed_lag()
    lead = max((t - s) / 3600 for t, (s, _base) in seen["ahead"].items())
    assert lag["ahead"]["max_lead_hours"] == pytest.approx(lead)
    assert lead > 0 and lag["ahead"]["publish_ahead"] is True
    assert lag["slow"]["max_lead_hours"] < 0


def test_release_recomputable_from_record_and_e(tmp_path):
    """rev6.10: release(p) = first_seen(p) + e is never stored; it is
    recomputed from the signed first-seen record and the pinned e alone."""
    fs, root, log, old, new = _first_two_scrapes(tmp_path)
    log.commit(now_ts=T0 + 2 * PERIOD)
    idx = log.index(until_ts=T0 + 2 * PERIOD)
    for e in (0, 6, 48, 1.5):
        assert fs.release_ts(idx.first_seen("meter", {}, new[0]), e) == \
            T0 + PERIOD + 600 + round(e * 3600)
    text = (root / "first_seen" / "records.jsonl").read_text()
    assert "release" not in text and "embargo" not in text


def test_pool_bytes_identical_with_and_without_recorder(tmp_path):
    """The recorder writes only inside its own log dir: a lane A `pool` built
    from the same mirror is byte-identical before and after recording."""
    from cascade.pool import builder
    from cascade.pool.source import HarvestContext
    from cascade.pool.sources.tsbench_forge import TsbenchForgeSource

    fs = fs_mod()
    root = _mirror(tmp_path, feeds=("meter", "grid"))
    pts = hourly(T0 - 700 * HOUR, 700)
    scrape(root, "meter", DAY, pts)
    scrape(root, "grid", DAY, pts + pts, panel={"region": ["r1"] * 700 + ["r2"] * 700},
           values=[float(i % 24) + (i // 700) for i in range(1400)])
    ctx = HarvestContext(as_of=dt.date.fromisoformat(DAY), context_length=256, horizon=32)
    cfg = builder.PoolBuildConfig(context_length=256, horizon=32, min_context=64)

    def build(out):
        builder.build_pool([TsbenchForgeSource(root)], out, ctx, cfg,
                           fetch=lambda u, p=None: None)
        return tree_sha(out)

    before = build(tmp_path / "pool_before")
    fs.FirstSeenLog(root / "first_seen").record(root, now_ts=T0 + 60)
    fs.FirstSeenLog(root / "first_seen").commit(now_ts=T0 + PERIOD)
    assert (root / "first_seen" / "records.jsonl").is_file()
    assert build(tmp_path / "pool_after") == before


def test_cli_record_commit_evidence(tmp_path):
    fs = fs_mod()
    root = _mirror(tmp_path)
    scrape(root, "meter", DAY, hourly(T0 - 10 * HOUR, 10))
    log_dir = root / "first_seen"
    assert fs.main(["record", "--forge-dir", str(root), "--log-dir", str(log_dir),
                    "--now-ts", str(T0 + 60)]) == 0
    assert fs.main(["commit", "--log-dir", str(log_dir), "--now-ts", str(T0 + PERIOD)]) == 0
    out = tmp_path / "w6fs.json"
    assert fs.main(["evidence", "--forge-dir", str(root), "--log-dir", str(log_dir),
                    "--out", str(out), "--now-ts", str(T0 + PERIOD + 5)]) == 0
    doc = json.loads(out.read_text())
    _check_evidence(fs, doc, allow_fixture=True)
    assert doc["n_points"] == 10 and doc["n_records"] == 1


# ── evidence (the owner's builder run; skipped without $LANED_EVIDENCE) ──────

def _check_evidence(fs, doc, *, allow_fixture=False):
    assert doc["schema"] == fs.EVIDENCE_SCHEMA == "laned-evidence/w6fs/1"
    assert doc["wp"] == "W6fs"
    assert doc["generated_by"] == "cascade.pool.first_seen evidence"
    if not allow_fixture:
        assert doc["source"] == "builder"
    body = {k: v for k, v in doc.items() if k != "digest"}
    assert doc["digest"] == hashlib.sha256(canonical(body)).hexdigest(), \
        "evidence edited after it was generated"
    assert isinstance(doc["n_records"], int) and doc["n_records"] > 0
    assert isinstance(doc["n_points"], int) and doc["n_points"] > 0
    assert doc["roots"], "no committed 3-hourly root"
    for r in doc["roots"]:
        assert r["period_start"] % 10800 == 0
        assert len(r["root"]) == 64 and set(r["root"]) <= set("0123456789abcdef")
        assert r["n_records"] > 0
        assert r["committed_ts"] >= r["period_start"] + 10800
    assert doc["feeds"]
    for feed, rec in doc["feeds"].items():
        assert set(rec) >= {"mirror_lag_hours", "max_lead_hours", "publish_ahead"}, feed
        assert isinstance(rec["publish_ahead"], bool)


def _evidence_doc():
    import os

    path = os.environ.get("LANED_EVIDENCE")
    if not path:
        pytest.skip("LANED_EVIDENCE unset: the owner's builder run writes evidence/w6fs.json")
    from pathlib import Path

    return json.loads(Path(path).read_text(encoding="utf-8"))


def test_evidence_from_builder_run():
    fs = fs_mod()
    _check_evidence(fs, _evidence_doc())


def test_evidence_roots_cover_3h_periods():
    fs = fs_mod()
    doc = _evidence_doc()
    starts = [r["period_start"] for r in doc["roots"]]
    assert starts == sorted(set(starts))
    assert doc["source"] != "fixture" and fs.PERIOD_SECONDS == 10800
