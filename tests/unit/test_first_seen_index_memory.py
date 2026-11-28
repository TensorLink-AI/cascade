"""FirstSeenLog.index() / pool_d_build bounded by one feed (2026-11-26 pool_d OOM).

``python -P -m cascade.pool.pool_d_build`` on the 39 GB builder reached 39.3 GB
RSS within 5 min over a 20 GB / 7,002-feed ``records.jsonl``: ``index()`` held
every record AND a ``(feed, panel_json, ts) -> (seen, root, baseline)`` dict of
every committed point. ``FirstSeenLog.feed_index()`` now streams the log once
(root checks on 32-byte leaf hashes, per-feed byte spans) and loads one feed at
a time; ``index()`` (the whole-log dict) is built from it feed by feed.

These tests pin that every answer is IDENTICAL to the previous ``index()``
(kept below verbatim as the reference): first_seen / root_for / baseline per
point, feed_points, feed_lag, len, the errors and which one wins, verify_snapshot,
and the ``pool_d_build`` output files (that one needs the ``pool-forge`` extra).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import statistics
from pathlib import Path

import numpy as np
import pytest

from cascade.pool import first_seen as fs

T0 = 1_795_132_800            # 2026-11-20 00:00 UTC, a 3 h period boundary
P = fs.PERIOD_SECONDS
A = '{"site":"a1"}'
B = '{"site":"b1"}'
U = "{}"


# ── the reference: FirstSeenIndex + index() before the fix, verbatim ─────────


class OldIndex:
    def __init__(self, entries: dict, until_ts: int):
        self._entries = entries
        self.until_ts = int(until_ts)

    def __len__(self) -> int:
        return len(self._entries)

    def _get(self, feed, panel, data_ts):
        return self._entries.get((str(feed), fs.panel_json(panel), int(data_ts)))

    def first_seen(self, feed, panel, data_ts):
        hit = self._get(feed, panel, data_ts)
        return None if hit is None else hit[0]

    def root_for(self, feed, panel, data_ts):
        hit = self._get(feed, panel, data_ts)
        return None if hit is None else hit[1]

    def feed_points(self, feed):
        return {(p, t): v[0] for (f, p, t), v in self._entries.items() if f == feed}

    def feed_lag(self):
        lags: dict[str, list[int]] = {}
        leads: dict[str, int] = {}
        for (feed, _panel, ts), (seen, _root, baseline) in self._entries.items():
            lead = ts - seen
            leads[feed] = lead if feed not in leads else max(leads[feed], lead)
            lags.setdefault(feed, [])
            if not baseline and ts <= seen:
                lags[feed].append(seen - ts)
        out = {}
        for feed in sorted(leads):
            lag = lags.get(feed) or []
            lead_h = leads[feed] / 3600.0
            out[feed] = {
                "mirror_lag_hours": (statistics.median(lag) / 3600.0) if lag else None,
                "max_lead_hours": lead_h,
                "publish_ahead": bool(lead_h > 0),
            }
        return out


def _old_lines(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    out = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise fs.FirstSeenError("log_corrupt", f"{path}:{n}: {e}") from e
    return out


class OldLog(fs.FirstSeenLog):
    """The pre-fix whole-log index(), verbatim."""

    _lines = staticmethod(_old_lines)

    def index(self, *, until_ts):
        until_ts = int(until_ts)
        records = self.records()
        roots = self.roots()
        by_period: dict[int, list[dict]] = {}
        for r in records:
            if int(r["period_start"]) != fs.period_start(int(r["seen_ts"])):
                raise fs.FirstSeenError("root_mismatch",
                                        f"record of {r.get('feed')} has a wrong period_start")
            by_period.setdefault(int(r["period_start"]), []).append(r)
        for p, entry in roots.items():
            recs = by_period.get(p, [])
            got = fs.merkle_root([fs.canonical_json(r) for r in recs]) if recs else None
            if got != entry["root"] or len(recs) != int(entry["n_records"]):
                raise fs.FirstSeenError("root_mismatch",
                                        f"period {p}: log no longer matches its committed root")
        usable = {p: e["root"] for p, e in roots.items() if int(e["committed_ts"]) <= until_ts}
        quarantined = self.quarantined(until_ts=until_ts)
        entries: dict = {}
        for r in records:
            p = int(r["period_start"])
            if p not in usable:
                continue
            seen = int(r["seen_ts"])
            q_ts = quarantined.get((str(r["feed"]), str(r["partition"])))
            if q_ts is not None and seen >= q_ts:
                continue
            for pj, t in r["points"]:
                key = (str(r["feed"]), str(pj), int(t))
                if key not in entries or seen < entries[key][0]:
                    entries[key] = (seen, usable[p], bool(r["baseline"]))
        return OldIndex(entries, until_ts)


def old_verify_snapshot(meta, log_dir) -> list[str]:
    problems: list[str] = []
    try:
        as_of = int(meta["as_of_ts"])
    except (KeyError, TypeError, ValueError):
        return ["metadata has no integer as_of_ts"]
    try:
        idx = OldLog(log_dir).index(until_ts=as_of)
    except fs.FirstSeenError as e:
        return [f"first-seen log: {e}"]
    listed = set(meta.get("first_seen_roots") or [])
    for s in meta.get("series") or []:
        sid = s.get("series_id")
        ts, seen = s.get("target_ts") or [], s.get("first_seen_ts") or []
        if len(ts) != len(seen):
            problems.append(f"{sid}: target_ts and first_seen_ts differ in length")
            continue
        feed, panel = s.get("feed") or s.get("source"), s.get("panel") or {}
        for t, claimed in zip(ts, seen, strict=True):
            got = idx.first_seen(feed, panel, t)
            if got is None:
                problems.append(f"{sid}: point {t} has no first-seen record committed by "
                                f"as_of {as_of}")
                continue
            if int(claimed) != got:
                problems.append(f"{sid}: point {t} first_seen_ts {claimed} != log {got}")
            root = idx.root_for(feed, panel, t)
            if root not in listed:
                problems.append(f"{sid}: point {t} root {root} not in first_seen_roots")
    return problems


# ── a synthetic multi-feed log, written line by line ─────────────────────────


def rec(feed, part, seen, points, *, baseline=False, period=None):
    return {"schema": fs.LOG_SCHEMA, "feed": feed, "partition": part,
            "partition_sha256": hashlib.sha256(f"{feed}/{part}/{seen}".encode()).hexdigest(),
            "seen_ts": seen, "period_start": fs.period_start(seen) if period is None else period,
            "baseline": baseline, "points": [[pj, int(t)] for pj, t in points]}


def pts(panel, start, n, step=3600):
    return [(panel, start + i * step) for i in range(n)]


def qentry(kind, feed, part, qts, points=()):
    e = {"schema": fs.QUARANTINE_SCHEMA, "kind": kind, "feed": feed, "partition": part,
         "partition_sha256": "0" * 64, "size": 10, "quarantined_ts": qts}
    if kind == "unreadable":
        e["reason"] = "OSError: no footer"
    else:
        e["points"] = [[pj, int(t)] for pj, t in points]
    return e


def write_lines(path: Path, docs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for d in docs:
            f.write(fs.canonical_json(d).decode() + "\n")


H = T0 - 86400                # data time base


def synthetic_log(log_dir: Path) -> Path:
    """Four feeds interleaved in file order: baselines, overlaps (earliest seen
    wins, the first record on a tie), a later-in-file record with an earlier
    seen, an uncommitted period, a late-committed period, empty points,
    quarantined names (before/after the record, in effect early/late),
    blocked points, a blank line."""
    log = fs.FirstSeenLog(log_dir)
    write_lines(log.records_path, [
        rec("f0", "d1.parquet", T0 + 10, pts(A, H, 6) + pts(B, H, 3), baseline=True),
        rec("f1", "d1.parquet", T0 + 20, pts(U, H, 5), baseline=True),
        rec("f2", "q.parquet", T0 + 5, pts(U, H, 2), baseline=True),       # before quarantine
        rec("f3", "late.parquet", T0 + P + 50, pts(A, H, 4)),               # P1, earlier in file
        rec("f3", "e.parquet", T0 + 40, pts(A, H + 3600, 2), baseline=True),  # P0 seen wins
        rec("f3", "tie1.parquet", T0 + 60, pts(B, H, 2), baseline=True),
        rec("f3", "tie2.parquet", T0 + 60, pts(B, H, 3)),                   # tie: first wins
        rec("f2", "q.parquet", T0 + 30, pts(U, H + 7200, 3)),               # after quarantine
        rec("f2", "r.parquet", T0 + 31, pts(U, H + 7200, 1)),
        rec("f0", "empty.parquet", T0 + 70, []),
    ])
    with open(log.records_path, "a", encoding="utf-8") as f:
        f.write("\n")
    write_lines(log.records_path, [
        rec("f0", "d2.parquet", T0 + P + 5, pts(A, H + 4 * 3600, 6)),       # overlap later seen
        rec("f1", "d2.parquet", T0 + P + 7, pts(U, H + 5 * 3600, 30, step=600)),
        rec("f2", "s.parquet", T0 + P + 9, pts(U, T0 + 6 * 3600, 2)),       # publish-ahead
        rec("f4", "lateq.parquet", T0 + P + 11, pts(A, H, 3), baseline=True),
    ])
    log.commit(now_ts=T0 + 2 * P)                                           # P0, P1 at T0+2P
    write_lines(log.records_path, [
        rec("f0", "d3.parquet", T0 + 2 * P + 1, pts(A, H + 10 * 3600, 3)),
        rec("f1", "d3.parquet", T0 + 2 * P + 2, pts(U, H, 2) + pts(B, H, 2)),
    ])
    log.commit(now_ts=T0 + 3 * P + 1)                                       # P2, after T0+3P
    write_lines(log.records_path, [
        rec("f0", "d4.parquet", T0 + 3 * P + 3, pts(A, H + 20 * 3600, 2)),  # uncommitted
    ])
    write_lines(log.quarantine_path, [
        qentry("unreadable", "f2", "q.parquet", T0 + 15),
        qentry("blocked", "f0", "d1.parquet", T0 + P + 1, pts(A, H + 99 * 3600, 2)),
        qentry("unreadable", "f4", "lateq.parquet", T0 + 2 * P + 30),       # after its record
        qentry("unreadable", "f2", "q.parquet", T0 + 12),                   # earliest wins
    ])
    return log_dir


UNTILS = (T0, T0 + 2 * P - 1, T0 + 2 * P, T0 + 3 * P, T0 + 3 * P + 1, T0 + 2 * P + 30,
          T0 + 9 * P)


def all_keys(log_dir: Path):
    keys = set()
    for r in _old_lines(log_dir / fs.RECORDS_FILE):
        keys.update((r["feed"], pj, int(t)) for pj, t in r["points"])
    keys |= {("f0", A, H - 1), ("nofeed", A, H), ("f1", A, H)}
    return sorted(keys)


def outcome(fn):
    try:
        return ("ok", fn())
    except fs.FirstSeenError as e:
        return ("err", e.code, str(e))


def test_lookups_identical_to_whole_log_index(tmp_path):
    log_dir = synthetic_log(tmp_path / "log")
    old, new = OldLog(log_dir), fs.FirstSeenLog(log_dir)
    keys = all_keys(log_dir)
    nonempty = 0
    for until in UNTILS:
        o = old.index(until_ts=until)
        full = new.index(until_ts=until)
        lazy = new.feed_index(until_ts=until)
        assert full._entries == o._entries, until
        assert len(full) == len(lazy) == len(o), until
        nonempty += bool(len(o))
        for feed, pj, t in keys:                    # alternating feeds: still correct
            want = o._get(feed, pj, t)
            for idx in (full, lazy):
                assert idx._get(feed, pj, t) == want
                assert idx.first_seen(feed, json.loads(pj), t) == o.first_seen(feed, pj, t)
                assert idx.root_for(feed, pj, t) == o.root_for(feed, pj, t)
        assert full.feed_lag() == lazy.feed_lag() == o.feed_lag(), until
        for feed in ("f0", "f1", "f2", "f3", "f4", "nofeed"):
            assert full.feed_points(feed) == lazy.feed_points(feed) == o.feed_points(feed)
            want = {k: v for k, v in o.feed_lag().items() if k == feed}
            assert full.feed_lag([feed]) == lazy.feed_lag([feed]) == want
        assert lazy.feed_lag(["f3", "f0", "nofeed"]) == {
            k: v for k, v in o.feed_lag().items() if k in ("f0", "f3")}
    assert nonempty >= 4
    # the fixture exercises what it claims
    o = old.index(until_ts=T0 + 9 * P)
    assert o.first_seen("f3", A, H + 3600) == T0 + 40              # later-in-file, earlier seen
    assert o._get("f3", B, H) == (T0 + 60, o.root_for("f3", B, H), True)   # tie: first record
    assert o.first_seen("f2", U, H + 7200) == T0 + 31              # q.parquet's T0+30 dropped
    assert o.first_seen("f2", U, H + 10800) is None                # quarantined name, after
    assert o.first_seen("f2", U, H) == T0 + 5                      # before its quarantine
    assert o.first_seen("f4", A, H) == T0 + P + 11                 # recorded before its quarantine
    assert o.first_seen("f0", A, H + 20 * 3600) is None            # uncommitted period
    assert old.index(until_ts=T0 + 3 * P).first_seen("f0", A, H + 10 * 3600) is None
    assert o.feed_lag()["f2"]["publish_ahead"] is True


def test_lazy_index_holds_one_feed(tmp_path):
    log_dir = synthetic_log(tmp_path / "log")
    lazy = fs.FirstSeenLog(log_dir).feed_index(until_ts=T0 + 9 * P)
    assert lazy.feeds() == ["f0", "f1", "f2", "f3", "f4"]
    assert lazy.loads == 0                                         # nothing loaded up front
    for feed in lazy.feeds():
        lazy.first_seen(feed, A, H)
        lazy.root_for(feed, A, H)
        assert set(lazy._cur) == set(lazy.feed_points(feed))
        assert lazy._cur_feed == feed
    assert lazy.loads == 5                                         # one load per feed walk
    lazy.release()
    assert lazy._cur == {} and lazy._cur_feed is None
    # the spans hold offsets only, never parsed records
    assert all(isinstance(o, int) and isinstance(n, int)
               for spans in lazy._spans.values() for o, n in spans)


def _tamper(log_dir: Path, fn) -> None:
    path = log_dir / fs.RECORDS_FILE
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    path.write_text("".join(fn(lines)), encoding="utf-8")


def _edit_record(i, **changes):
    def fn(lines):
        nonblank = [k for k, ln in enumerate(lines) if ln.strip()]
        k = nonblank[i]
        doc = json.loads(lines[k])
        doc.update(changes)
        lines[k] = fs.canonical_json(doc).decode() + "\n"
        return lines
    return fn


@pytest.mark.parametrize("case", [
    "points_changed", "wrong_period", "dropped_record", "corrupt_record", "corrupt_roots",
    "root_conflict", "corrupt_quarantine", "wrong_period_and_corrupt_roots",
    "mismatch_and_corrupt_quarantine", "corrupt_record_and_roots",
])
def test_errors_identical_to_whole_log_index(tmp_path, case):
    log_dir = synthetic_log(tmp_path / "log")
    roots_path, q_path = log_dir / fs.ROOTS_FILE, log_dir / fs.QUARANTINE_FILE
    if case in ("points_changed", "mismatch_and_corrupt_quarantine"):
        _tamper(log_dir, _edit_record(3, points=[[A, H]]))
    if case in ("wrong_period", "wrong_period_and_corrupt_roots"):
        _tamper(log_dir, _edit_record(5, period_start=T0 + P))
    if case == "dropped_record":
        _tamper(log_dir, lambda lines: lines[:1] + lines[2:])
    if case in ("corrupt_record", "corrupt_record_and_roots"):
        _tamper(log_dir, lambda lines: [*lines[:4], "{oops\n", *lines[4:]])
    if case in ("corrupt_roots", "wrong_period_and_corrupt_roots", "corrupt_record_and_roots"):
        with open(roots_path, "a", encoding="utf-8") as f:
            f.write("{nope\n")
    if case == "root_conflict":
        first = json.loads(roots_path.read_text(encoding="utf-8").splitlines()[0])
        write_lines(roots_path, [dict(first, root="0" * 64)])
    if case in ("corrupt_quarantine", "mismatch_and_corrupt_quarantine"):
        with open(q_path, "a", encoding="utf-8") as f:
            f.write("[broken\n")
    old, new = OldLog(log_dir), fs.FirstSeenLog(log_dir)
    for until in (T0, T0 + 9 * P):
        want = outcome(lambda u=until: old.index(until_ts=u)._entries)
        assert want[0] == "err", (case, want)
        assert outcome(lambda u=until: new.index(until_ts=u)._entries) == want
        assert outcome(lambda u=until: new.feed_index(until_ts=u)) == want


def test_verify_snapshot_identical(tmp_path):
    log_dir = synthetic_log(tmp_path / "log")
    o = OldLog(log_dir).index(until_ts=T0 + 9 * P)
    roots = sorted({v[1] for v in o._entries.values()})

    def series(sid, feed, panel, ts, seen=None):
        seen = [o.first_seen(feed, panel, t) or 0 for t in ts] if seen is None else seen
        return {"series_id": sid, "feed": feed, "panel": json.loads(panel),
                "target_ts": ts, "first_seen_ts": seen}

    good = [series("s0", "f0", A, [H, H + 3600, H + 4 * 3600]),
            series("s1", "f1", U, [H, H + 3600]),
            series("s2", "f0", B, [H, H + 7200]),
            series("s3", "f3", A, [H + 3600]),
            series("s4", "f1", U, [H + 5 * 3600 + 600])]
    metas = {
        "ok": {"as_of_ts": T0 + 9 * P, "first_seen_roots": roots, "series": good},
        "bad": {"as_of_ts": T0 + 9 * P, "first_seen_roots": roots[:1], "series": [
            *good[:2], series("x1", "f0", A, [H, H - 5]),
            series("x2", "f2", U, [H + 7200], [T0 + 30]),
            series("x3", "f1", U, [H], [T0 + 1]),
            {"series_id": "x4", "source": "f0", "panel": json.loads(A),
             "target_ts": [H], "first_seen_ts": [T0 + 10]},
            {"series_id": "x5", "feed": "f3", "target_ts": [H, H], "first_seen_ts": [1]},
            series("x6", "nofeed", U, [H], [1]), *good[2:]]},
        "early": {"as_of_ts": T0 + 2 * P - 1, "first_seen_roots": roots, "series": good},
        "no_as_of": {"series": good},
    }
    for name, meta in metas.items():
        want = old_verify_snapshot(meta, log_dir)
        assert fs.verify_snapshot(meta, log_dir) == want, name
        assert (want == []) == (name == "ok"), (name, want)
    _tamper(log_dir, _edit_record(0, baseline=False))
    assert fs.verify_snapshot(metas["ok"], log_dir) == old_verify_snapshot(metas["ok"], log_dir)
    assert fs.verify_snapshot(metas["ok"], log_dir)[0].startswith("first-seen log: root_mismatch")


def test_evidence_identical(tmp_path, monkeypatch):
    log_dir = synthetic_log(tmp_path / "log")
    new = fs._evidence(fs.FirstSeenLog(log_dir), T0 + 9 * P)
    monkeypatch.setattr(fs.FirstSeenLog, "index", OldLog.index)
    monkeypatch.setattr(fs.FirstSeenLog, "feed_index", OldLog.index)
    monkeypatch.setattr(fs.FirstSeenLog, "_lines", staticmethod(_old_lines))
    records = fs.FirstSeenLog(log_dir).records()
    old = fs._evidence(fs.FirstSeenLog(log_dir), T0 + 9 * P)
    assert new == old
    assert new["n_records"] == len(records) and new["feeds"]


# ── pool_d_build: the same files over the old and the new index ──────────────


class _OldIndexAdapter:
    """The old whole-log index behind the calls build_pool_d makes now."""

    def __init__(self, old: OldIndex):
        self._old = old
        self.first_seen, self.root_for = old.first_seen, old.root_for

    def feed_lag(self, feeds=None):
        lags = self._old.feed_lag()
        return lags if feeds is None else {k: v for k, v in lags.items() if k in set(feeds)}

    def release(self):
        pass


def _forge(root: Path) -> Path:
    import pandas as pd
    import yaml

    catalog = [
        {"id": "grid", "domain": "energy", "frequency": "PT1H", "source_family": "fx:grid"},
        {"id": "air", "domain": "weather", "frequency": "PT1H", "source_family": "fx:air",
         "embargo_hours": 6},
        {"id": "ahead", "domain": "nature", "frequency": "PT1H", "source_family": "fx:ahead"},
        {"id": "gone", "domain": "web", "frequency": "PT1H", "source_family": "fx:gone"},
    ]
    root.mkdir(parents=True, exist_ok=True)
    (root / "sources.yaml").write_text(yaml.safe_dump(catalog), encoding="utf-8")

    def iso(t):
        return dt.datetime.fromtimestamp(int(t), tz=dt.UTC).isoformat()

    def write(feed, day, frame):
        d = root / "data" / feed
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(frame).to_parquet(d / f"{day}.parquet")

    n = 120
    rng = np.random.default_rng(7)
    ts = [T0 - 3600 * (n - i) for i in range(n)]
    sig = 20 + 5 * np.sin(np.arange(2 * n) / 4.0) + rng.normal(0, 0.3, 2 * n)
    write("grid", "2026-11-19", {"timestamp": [iso(t) for t in ts] * 2,
                                 "_panel_region": ["r1"] * n + ["r2"] * n, "mw": sig})
    write("air", "2026-11-19", {"timestamp": [iso(t) for t in ts], "t": sig[:n] + 3})
    ahead = [T0 + 6 * 3600 - 3600 * (n - i) for i in range(n)]
    write("ahead", "2026-11-19", {"timestamp": [iso(t) for t in ahead], "a": sig[n:] - 2})
    write("gone", "2026-11-19", {"timestamp": [iso(t) for t in ts], "g": sig[:n]})
    return root


def _second_scrape(root: Path) -> None:
    import pandas as pd

    ts = [T0 + 3600 * i for i in range(5)]
    pd.DataFrame({"timestamp": [dt.datetime.fromtimestamp(t, tz=dt.UTC).isoformat()
                                for t in ts],
                  "t": np.linspace(20.0, 24.0, 5)}).to_parquet(
        root / "data" / "air" / "2026-11-20.parquet")


def _selection(b):
    from cascade.pool.pool_d_select import PoolDSelection, SelectedSeries, pool_d_series_id

    rows = [("grid", {"region": "r1"}), ("grid", {"region": "r2"}), ("air", {}),
            ("ahead", {}), ("gone", {}), ("nosuch", {})]
    series = []
    for feed, panel in rows:
        key = (f"tsforge:{feed}", b._series_base(feed, panel), 0)
        series.append(SelectedSeries(key=key, pool_series_id=f"p-{key[1]}",
                                     pool_d_series_id=pool_d_series_id(key), feed=feed,
                                     cluster=f"fx:{feed}", channel=0, n_channels=1,
                                     freq="H", domain="x"))
    return PoolDSelection(series=tuple(series), refused=())


def _tree(d: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(Path(d).iterdir()) if p.is_file()}


def test_pool_d_build_output_byte_identical(tmp_path, monkeypatch):
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    pytest.importorskip("yaml")
    from cascade.pool import pool_d_build as b

    root = _forge(tmp_path / "forge")
    log = fs.FirstSeenLog(root / "first_seen")
    log.record(root, now_ts=T0 + 600)
    log.commit(now_ts=T0 + P)
    # a quarantined name and a later (uncommitted at AS1) scrape
    write_lines(log.quarantine_path, [qentry("unreadable", "gone", "2026-11-19.parquet", T0)])
    _second_scrape(root)
    log.record(root, now_ts=T0 + P + 600)
    as_ofs = (T0 + P, T0 + 2 * P)
    log.commit(now_ts=T0 + 2 * P)
    selection = _selection(b)
    cfg = b.PoolDBuildConfig(horizon=8, min_context=16, history_length=64)

    built = {}
    loads = []
    real_feed_index = fs.FirstSeenLog.feed_index

    def counting_feed_index(self, *, until_ts, roots="committed"):
        idx = real_feed_index(self, until_ts=until_ts, roots=roots)
        loads.append(idx)
        return idx

    for name in ("new", "old"):
        if name == "old":
            monkeypatch.setattr(
                fs.FirstSeenLog, "feed_index",
                lambda self, *, until_ts, roots="committed": _OldIndexAdapter(
                    OldLog(self.log_dir).index(until_ts=until_ts)))
        else:
            monkeypatch.setattr(fs.FirstSeenLog, "feed_index", counting_feed_index)
        for as_of in as_ofs:
            out = tmp_path / f"{name}-{as_of}"
            s = b.build_pool_d(forge_dir=root, selection=selection,
                               first_seen_dir=root / "first_seen", as_of_ts=as_of,
                               out_dir=out, cfg=cfg)
            built[(name, as_of)] = (_tree(out), s.sha256, s.refused)
    monkeypatch.undo()
    for as_of in as_ofs:
        assert built[("new", as_of)] == built[("old", as_of)], as_of
    meta = json.loads(built[("new", as_ofs[0])][0]["metadata.json"])
    assert len(meta["series"]) >= 3 and meta["first_seen_roots"]
    assert meta["feeds"]["ahead"]["publish_ahead"] is True
    # one load per feed walked (lag + lookups of a feed share it), freed at the end
    assert loads and all(idx.loads <= 3 and idx._cur == {} for idx in loads)
