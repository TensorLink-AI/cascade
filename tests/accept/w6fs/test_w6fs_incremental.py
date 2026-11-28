"""W6fs incremental scans + the ``ended`` root rule (2026-11-28 pool_d speed-up).

Every pool_d build re-hashed all ~630k forge partitions and re-parsed the
whole 20 GB ``records.jsonl`` in record, commit and the build's index; and it
used the root committed one build earlier. These tests pin that the cached
(incremental) scans give exactly the full recompute's logs, roots and index,
that they read only new or changed input, that a stale cache is caught, and
the ``ended`` rule (README.md).

Partitions are JSON stand-ins named ``*.parquet`` (``partition_points`` is
stubbed), so nothing here needs pandas / pyarrow. Each test asserts the new
API first, so on a base without it every test fails by assertion.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from .helpers import HOUR, PERIOD, T0, fs_mod

A = '{"site":"a1"}'
B = '{"site":"b1"}'


def _points(path: Path):
    doc = json.loads(path.read_text(encoding="utf-8"))   # corrupt bytes raise
    return sorted({(str(pj), int(t)) for pj, t in doc["points"]})


@pytest.fixture
def fs(monkeypatch):
    mod = fs_mod()
    monkeypatch.setattr(mod, "partition_points", _points)
    return mod


def needs_api(mod) -> None:
    """Each test's first line: on a base without the feature it fails here."""
    assert "cache_dir" in inspect.signature(mod.FirstSeenLog).parameters, \
        "FirstSeenLog has no incremental scan cache"
    assert "roots" in inspect.signature(mod.FirstSeenLog.feed_index).parameters, \
        "feed_index has no root rule"


def put(root: Path, feed: str, name: str, points=None, *, corrupt: bytes | None = None):
    d = root / "data" / feed
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{name}.parquet"
    if corrupt is not None:
        path.write_bytes(corrupt)
    else:
        path.write_text(json.dumps({"points": [list(x) for x in points]}), encoding="utf-8")
    return path


def pts(panel: str, start: int, n: int, step: int = HOUR):
    return [(panel, start + i * step) for i in range(n)]


def logs(log_dir: Path) -> dict[str, bytes]:
    return {n: (log_dir / n).read_bytes() for n in ("records.jsonl", "roots.jsonl",
                                                    "quarantine.jsonl")
            if (log_dir / n).is_file()}


def entries(log, until, roots="committed"):
    return log.index(until_ts=until, roots=roots)._entries


class Clock:
    def __init__(self, t: float = 1_000_000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


def _steps():
    """A builder history: baseline, a rewrite + backfill + a new day, an
    unreadable partition, its repair (blocked), a duplicate across feeds."""
    def s1(f):
        put(f, "a", "2026-11-18", pts(A, T0 - 86400, 24))
        put(f, "a", "2026-11-19", pts(A, T0 - 6 * HOUR, 6) + pts("{}", T0 - 2 * HOUR, 2))
        put(f, "b", "2026-11-19", pts(B, T0 - 86400, 5))
        put(f, "c", "2026-11-19", corrupt=b"PAR1 no footer")
        return T0 + 600

    def s2(f):
        put(f, "a", "2026-11-19", pts(A, T0 - 6 * HOUR, 9))          # rewrite: 3 new
        put(f, "a", "2026-11-17", pts(A, T0 - 2 * 86400, 4))         # backfill
        put(f, "a", "2026-11-20", pts(A, T0, 3))                     # a new day
        put(f, "b", "2026-11-20", pts(B, T0 - 86400, 5) + pts(B, T0, 2))  # dup + 2 new
        return T0 + PERIOD + 600

    def s3(f):
        put(f, "c", "2026-11-19", pts("{}", T0 - 86400, 3))          # repaired: blocked
        put(f, "a", "2026-11-20", pts(A, T0, 6))                     # 3 more
        return T0 + 2 * PERIOD + 600

    return [s1, s2, s3]


def test_cached_runs_equal_the_full_recompute(tmp_path, fs):
    """record / commit / index with the cache = without it, byte for byte,
    after every builder step (and under ``full=True``)."""
    needs_api(fs)
    forge = tmp_path / "forge"
    plain = fs.FirstSeenLog(tmp_path / "plain")
    cached = fs.FirstSeenLog(tmp_path / "cached", cache_dir=tmp_path / "cached.cache")
    forced = fs.FirstSeenLog(tmp_path / "forced", cache_dir=tmp_path / "forced.cache",
                             full=True)
    for step in _steps():
        now = step(forge)
        outs = [log.record(forge, now_ts=now) for log in (plain, cached, forced)]
        assert outs[0] == outs[1] == outs[2]
        commits = [log.commit(now_ts=now) for log in (plain, cached, forced)]
        assert commits[0] == commits[1] == commits[2]
        assert logs(tmp_path / "plain") == logs(tmp_path / "cached") == logs(tmp_path / "forced")
        for rule in ("committed", "ended"):
            want = entries(plain, now, rule)
            assert entries(cached, now, rule) == want
            assert entries(forced, now, rule) == want
        lag = plain.feed_index(until_ts=now).feed_lag()
        assert cached.feed_index(until_ts=now).feed_lag() == lag
    assert (tmp_path / "cached.cache" / fs.CACHE_META_FILE).is_file()
    assert (tmp_path / "cached.cache" / fs.CACHE_PARTS_FILE).is_file()
    # the cache never lands among the log files state-push uploads
    assert not (tmp_path / "cached" / fs.CACHE_META_FILE).exists()


def test_only_new_or_changed_partitions_are_hashed(tmp_path, fs, monkeypatch):
    """The CLI (cached by default) hashes a partition again only when its
    size / mtime / inode changed."""
    needs_api(fs)
    forge = tmp_path / "forge"
    for i in range(6):
        put(forge, f"f{i}", "2026-11-19", pts(A, T0 - 86400 + i, 10))
    log_dir = forge / "first_seen"
    assert fs.main(["record", "--forge-dir", str(forge), "--log-dir", str(log_dir),
                    "--now-ts", str(T0 + 60)]) == 0
    hashed = []
    real = fs._file_sha
    monkeypatch.setattr(fs, "_file_sha", lambda p: hashed.append(p.name + "@" + p.parent.name)
                        or real(p))
    put(forge, "f0", "2026-11-20", pts(A, T0, 3))                       # new
    put(forge, "f1", "2026-11-19", pts(A, T0 - 86400 + 1, 12))          # rewritten
    assert fs.main(["record", "--forge-dir", str(forge), "--log-dir", str(log_dir),
                    "--now-ts", str(T0 + PERIOD + 60)]) == 0
    assert sorted(hashed) == ["2026-11-19.parquet@f1", "2026-11-20.parquet@f0"]
    # --full re-hashes everything (the escape hatch)
    hashed.clear()
    assert fs.main(["record", "--forge-dir", str(forge), "--log-dir", str(log_dir),
                    "--now-ts", str(T0 + 2 * PERIOD + 60), "--full"]) == 0
    assert len(hashed) == 7


def test_only_appended_log_lines_are_parsed(tmp_path, fs, monkeypatch):
    """After the first cached scan, commit parses only the records appended
    since; record loads only the records whose data time meets a new
    partition's (not the feed's whole history)."""
    needs_api(fs)
    forge = tmp_path / "forge"
    log_dir = forge / "first_seen"
    for day in range(10):                                   # 10 daily partitions
        put(forge, "a", f"2026-11-{10 + day:02d}", pts(A, T0 - (10 - day) * 86400, 24))
    assert fs.main(["record", "--forge-dir", str(forge), "--log-dir", str(log_dir),
                    "--now-ts", str(T0 + 60)]) == 0
    assert fs.main(["commit", "--log-dir", str(log_dir), "--now-ts", str(T0 + PERIOD)]) == 0
    n_lines = len((log_dir / "records.jsonl").read_text().splitlines())
    assert n_lines == 10

    parsed = []
    real_loads = json.loads

    class CountingJson:
        def __getattr__(self, name):
            return getattr(json, name)

        @staticmethod
        def loads(s, *a, **k):
            doc = real_loads(s, *a, **k)
            if isinstance(doc, dict) and doc.get("schema") == fs.LOG_SCHEMA:
                parsed.append(doc["partition"])
            return doc

    monkeypatch.setattr(fs, "json", CountingJson())
    put(forge, "a", "2026-11-20", pts(A, T0, 3))            # one new day
    assert fs.main(["record", "--forge-dir", str(forge), "--log-dir", str(log_dir),
                    "--now-ts", str(T0 + PERIOD + 60)]) == 0
    assert parsed == [], parsed                             # no history overlaps it
    put(forge, "a", "2026-11-19", pts(A, T0 - 86400, 24) + pts(A, T0 - 86400 + 1800, 2))
    assert fs.main(["record", "--forge-dir", str(forge), "--log-dir", str(log_dir),
                    "--now-ts", str(T0 + 2 * PERIOD + 60)]) == 0
    # the one line appended since the last scan, then the one overlapping record
    assert parsed == ["2026-11-20.parquet", "2026-11-19.parquet"], parsed
    parsed.clear()
    assert fs.main(["commit", "--log-dir", str(log_dir),
                    "--now-ts", str(T0 + 3 * PERIOD)]) == 0
    assert len(parsed) == 1, parsed                         # only the newest record line
    roots = fs.FirstSeenLog(log_dir).roots()
    assert sorted(roots) == [T0, T0 + PERIOD, T0 + 2 * PERIOD]


def test_rewritten_log_is_caught_through_the_cache(tmp_path, fs):
    """A cache that no longer fingerprints its log is dropped: the full scan
    refuses a backdated record (root_mismatch) as before."""
    needs_api(fs)
    forge = tmp_path / "forge"
    log = fs.FirstSeenLog(tmp_path / "log", cache_dir=tmp_path / "c")
    put(forge, "a", "d1", pts(A, T0 - 48 * HOUR, 48))
    log.record(forge, now_ts=T0 + 600)
    put(forge, "a", "d1", pts(A, T0 - 48 * HOUR, 51))
    log.record(forge, now_ts=T0 + PERIOD + 600)
    log.commit(now_ts=T0 + 2 * PERIOD)
    assert log.index(until_ts=T0 + 2 * PERIOD)._entries           # cache written
    path = tmp_path / "log" / "records.jsonl"
    lines = path.read_text().splitlines()
    rec = json.loads(lines[1])
    rec["seen_ts"] -= 3600
    lines[1] = json.dumps(rec)
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(fs.FirstSeenError) as err:
        fs.FirstSeenLog(tmp_path / "log", cache_dir=tmp_path / "c").index(
            until_ts=T0 + 2 * PERIOD)
    assert err.value.code == "root_mismatch"


def test_periodic_full_verify_rebuilds_a_diverged_cache(tmp_path, fs, capsys):
    """Once a day the cached scan is replaced by a full one; a cache that
    diverged (here: hand-edited) is reported loudly and rebuilt."""
    needs_api(fs)
    forge = tmp_path / "forge"
    clock = Clock()
    cdir = tmp_path / "c"

    def log():
        return fs.FirstSeenLog(tmp_path / "log", cache_dir=cdir, clock=clock)

    put(forge, "a", "d1", pts(A, T0 - 24 * HOUR, 24))
    put(forge, "b", "d1", pts(B, T0 - 24 * HOUR, 24))
    log().record(forge, now_ts=T0 + 600)
    log().commit(now_ts=T0 + PERIOD)
    want = fs.FirstSeenLog(tmp_path / "log").index(until_ts=T0 + PERIOD)._entries
    meta = json.loads((cdir / fs.CACHE_META_FILE).read_text())
    meta["lines"][0][2] = "zzz"                               # the feed of line 0
    (cdir / fs.CACHE_META_FILE).write_text(json.dumps(meta))
    capsys.readouterr()
    clock.t += fs.FULL_VERIFY_SECONDS - 10                     # not due: the cache is used
    assert log().index(until_ts=T0 + PERIOD)._entries != want
    clock.t += 20                                              # due: full scan
    assert log().index(until_ts=T0 + PERIOD)._entries == want
    assert "DIVERGED" in capsys.readouterr().err
    assert log().index(until_ts=T0 + PERIOD)._entries == want   # the rebuilt cache


# ── the "ended" root rule ───────────────────────────────────────────────────


def test_ended_rule_uses_the_root_this_build_committed(tmp_path, fs):
    """The builder at T0 + 3 h + 10 min commits period T0 (recorded at
    T0 + 10 min) and cuts the snapshot at as_of T0 + 3 h: under ``ended``
    those points are usable now, not one build later."""
    needs_api(fs)
    forge = tmp_path / "forge"
    log = fs.FirstSeenLog(tmp_path / "log")
    put(forge, "a", "d1", pts(A, T0 - 24 * HOUR, 24))
    log.record(forge, now_ts=T0 + 600)                          # build k: period T0
    put(forge, "a", "d1", pts(A, T0 - 24 * HOUR, 27))
    as_of, now = T0 + PERIOD, T0 + PERIOD + 600
    log.record(forge, now_ts=now)                               # build k+1
    log.commit(now_ts=now)                                      # commits period T0 at `now`
    old = log.index(until_ts=as_of)                             # rev6.11 (d) rule
    new = log.index(until_ts=as_of, roots="ended")
    t = T0 - HOUR
    assert old.first_seen("a", A, t) is None
    assert new.first_seen("a", A, t) == T0 + 600 < as_of
    assert new.root_for("a", A, t) == log.roots()[T0]["root"]
    # the build-k+1 points (seen after as_of) stay out under both rules
    assert new.first_seen("a", A, T0 + 2 * HOUR) is None
    meta = {"schema": "pool_d/1", "as_of_ts": as_of,
            "first_seen_roots": [log.roots()[T0]["root"]],
            "series": [{"series_id": "s", "feed": "a", "panel": json.loads(A),
                        "target_ts": [t], "first_seen_ts": [T0 + 600]}]}
    assert fs.verify_snapshot(meta, tmp_path / "log", roots="ended") == []
    assert fs.verify_snapshot(meta, tmp_path / "log")           # the old rule refuses it


def test_ended_rule_refuses_an_uncommitted_ended_period(tmp_path, fs):
    needs_api(fs)
    forge = tmp_path / "forge"
    log = fs.FirstSeenLog(tmp_path / "log")
    put(forge, "a", "d1", pts(A, T0 - 24 * HOUR, 24))
    log.record(forge, now_ts=T0 + 600)
    with pytest.raises(fs.FirstSeenError) as err:
        log.index(until_ts=T0 + PERIOD, roots="ended")
    assert err.value.code == "uncommitted_period"
    assert log.index(until_ts=T0 + PERIOD)._entries == {}        # old rule: just nothing
    log.commit(now_ts=T0 + PERIOD + 5)
    assert log.index(until_ts=T0 + PERIOD, roots="ended")._entries


def test_record_refuses_a_period_frozen_by_a_commit(tmp_path, fs):
    """After a commit at C no record may enter a period that ended by C (an
    ``ended`` snapshot relied on it), so a re-derived snapshot never changes."""
    needs_api(fs)
    forge = tmp_path / "forge"
    log = fs.FirstSeenLog(tmp_path / "log")
    put(forge, "a", "d1", pts(A, T0 - 24 * HOUR, 24))
    log.record(forge, now_ts=T0 + 600)
    log.commit(now_ts=T0 + 2 * PERIOD + 600)
    put(forge, "a", "d1", pts(A, T0 - 24 * HOUR, 26))
    with pytest.raises(fs.FirstSeenError) as err:
        log.record(forge, now_ts=T0 + PERIOD + 600)              # ended at T0 + 6 h
    assert err.value.code == "period_frozen"
    assert log.record(forge, now_ts=T0 + 2 * PERIOD + 900) == {"records": 1, "points": 2}
