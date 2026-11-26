"""FirstSeenLog.record() / commit() bounded by one feed (2026-11-25 pool_d OOM).

The mainnet baseline run over 7,002 feeds / 14 GB of parquet held every point
of every feed (``known``) plus every new record in memory until the end and
OOM-killed a 39 GB builder. record() now processes one feed at a time and
stages its lines in unlinked temp files; commit() streams the log. These tests
pin that the logs it writes are BYTE-IDENTICAL to the whole-forge version
(kept below verbatim as the reference), and that only one feed's state is
alive at a time.

The record() tests stub ``partition_points`` with a JSON reader (files still
named ``*.parquet``), so they need no pandas/pyarrow; the parquet reader's own
equivalence test skips without the ``pool-forge`` extra.
"""

from __future__ import annotations

import gc
import json
import weakref
from pathlib import Path

import numpy as np
import pytest

from cascade.pool import first_seen as fs

T0 = 1_795_132_800            # 2026-11-20 00:00 UTC, a 3 h period boundary
P = fs.PERIOD_SECONDS


# ── the reference: record()/commit()/partition_points before the fix ─────────


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
    """The pre-fix whole-forge record() and commit(), verbatim."""

    _lines = staticmethod(_old_lines)

    def record(self, forge_dir, *, now_ts, max_unreadable_files=fs.DEFAULT_MAX_UNREADABLE_FILES,
               max_unreadable_frac=fs.DEFAULT_MAX_UNREADABLE_FRAC):
        now_ts = int(now_ts)
        p = fs.period_start(now_ts)
        if p in self.roots():
            raise fs.FirstSeenError("period_committed",
                                    f"period {p} already has a committed root; now_ts={now_ts}")
        existing = self.records()
        quarantine = self.quarantine()
        seen_parts = {(r["feed"], r["partition"], r["partition_sha256"]) for r in existing}
        known: dict[str, set] = {}
        for r in existing:
            known.setdefault(r["feed"], set()).update((pj, int(t)) for pj, t in r["points"])
        q_names = {(str(e["feed"]), str(e["partition"])) for e in quarantine
                   if e.get("kind") == "unreadable"}
        q_versions = {(str(e["feed"]), str(e["partition"]), str(e["partition_sha256"])):
                      e.get("kind") for e in quarantine}
        for e in quarantine:
            if e.get("kind") == "blocked":
                known.setdefault(str(e["feed"]), set()).update(
                    (pj, int(t)) for pj, t in e.get("points") or [])
        logged_feeds = {r["feed"] for r in existing}
        data = Path(forge_dir) / "data"
        new: list[dict] = []
        new_q: list[dict] = []
        unreadable: list[str] = []
        n_points = n_blocked = n_parts = 0
        feed_dirs = sorted(d for d in data.iterdir() if d.is_dir()) if data.is_dir() else []
        for feed_dir in feed_dirs:
            feed = feed_dir.name
            for part in sorted(feed_dir.glob("*.parquet")):
                n_parts += 1
                sha = fs._file_sha(part)
                if (feed, part.name, sha) in seen_parts:
                    continue
                kind = q_versions.get((feed, part.name, sha))
                if kind == "unreadable":
                    unreadable.append(f"{feed}/{part.name}")
                    continue
                if kind == "blocked":
                    continue
                size = part.stat().st_size
                try:
                    points = fs.partition_points(part)
                except Exception as e:  # noqa: BLE001
                    reason = fs.unreadable_reason(e)
                    unreadable.append(f"{feed}/{part.name}")
                    new_q.append({"schema": fs.QUARANTINE_SCHEMA, "kind": "unreadable",
                                  "feed": feed, "partition": part.name,
                                  "partition_sha256": sha, "size": int(size),
                                  "reason": reason, "quarantined_ts": now_ts})
                    q_versions[(feed, part.name, sha)] = "unreadable"
                    q_names.add((feed, part.name))
                    continue
                have = known.setdefault(feed, set())
                fresh = sorted(set(points) - have)
                have.update(fresh)
                if (feed, part.name) in q_names:
                    n_blocked += len(fresh)
                    new_q.append({"schema": fs.QUARANTINE_SCHEMA, "kind": "blocked",
                                  "feed": feed, "partition": part.name,
                                  "partition_sha256": sha, "size": int(size),
                                  "quarantined_ts": now_ts,
                                  "points": [[pj, int(t)] for pj, t in fresh]})
                    q_versions[(feed, part.name, sha)] = "blocked"
                    continue
                n_points += len(fresh)
                new.append({
                    "schema": fs.LOG_SCHEMA, "feed": feed, "partition": part.name,
                    "partition_sha256": sha, "seen_ts": now_ts, "period_start": p,
                    "baseline": feed not in logged_feeds,
                    "points": [[pj, int(t)] for pj, t in fresh],
                })
        limit = fs.unreadable_limit(n_parts, max_files=max_unreadable_files,
                                    max_frac=max_unreadable_frac)
        if len(unreadable) > limit:
            raise fs.FirstSeenError(
                "too_many_unreadable",
                f"{len(unreadable)} of {n_parts} partitions are unreadable (limit {limit}); "
                f"a systemic read failure? nothing recorded. First: "
                f"{', '.join(unreadable[:10])}")
        self._append(self.quarantine_path, new_q)
        self._append(self.records_path, new)
        out = {"records": len(new), "points": n_points}
        n_q = sum(1 for e in new_q if e["kind"] == "unreadable")
        if n_q:
            out["quarantined"] = n_q
        if n_blocked or any(e["kind"] == "blocked" for e in new_q):
            out["blocked"] = n_blocked
        if unreadable:
            out["unreadable"] = len(unreadable)
        return out

    def commit(self, *, now_ts):
        now_ts = int(now_ts)
        roots = self.roots()
        by_period: dict[int, list[dict]] = {}
        for r in self.records():
            by_period.setdefault(int(r["period_start"]), []).append(r)
        added = []
        for p in sorted(by_period):
            if p in roots or p + fs.PERIOD_SECONDS > now_ts:
                continue
            recs = by_period[p]
            added.append({"period_start": p,
                          "root": fs.merkle_root([fs.canonical_json(r) for r in recs]),
                          "n_records": len(recs), "committed_ts": now_ts})
        self._append(self.roots_path, added)
        return added


def _old_partition_points(path):
    from cascade.pool.sources.tsbench_forge import TsbenchForgeSource

    pd, pq, _yaml = TsbenchForgeSource._require_deps()
    df = pq.read_table(path).to_pandas()
    if "timestamp" not in df.columns or df.empty:
        return []
    ok, secs = fs._epoch_seconds(pd, df["timestamp"])
    panel_cols = [c for c in df.columns if c.startswith("_panel_")]
    cols = [[str(v) for v in df[c].tolist()] for c in panel_cols]
    names = [c.removeprefix("_panel_") for c in panel_cols]
    out = set()
    for i in np.flatnonzero(ok):
        panel = {n: col[i] for n, col in zip(names, cols, strict=True)}
        out.add((fs.panel_json(panel), int(secs[i])))
    return sorted(out)


# ── a JSON-backed synthetic forge ────────────────────────────────────────────


def _json_points(path: Path):
    doc = json.loads(path.read_text(encoding="utf-8"))   # corrupt bytes raise
    return sorted({(str(pj), int(t)) for pj, t in doc["points"]})


@pytest.fixture
def json_forge(monkeypatch):
    monkeypatch.setattr(fs, "partition_points", _json_points)


def put(root: Path, feed: str, name: str, points=None, *, corrupt: bytes | None = None):
    d = root / "data" / feed
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{name}.parquet"
    if corrupt is not None:
        path.write_bytes(corrupt)
    else:
        path.write_text(json.dumps({"points": [list(x) for x in points]}), encoding="utf-8")
    return path


def pts(panel: str, start: int, n: int, step: int = 3600):
    return [(panel, start + i * step) for i in range(n)]


def snapshot(log_dir: Path) -> dict[str, bytes] | None:
    if not log_dir.exists():
        return None
    return {p.name: p.read_bytes() for p in sorted(log_dir.iterdir())}


def run_both(tmp_path, forge, step, fn):
    """Apply ``fn(log)`` to the old and new log; same result (or same error)."""
    res = {}
    for name, cls in (("old", OldLog), ("new", fs.FirstSeenLog)):
        log = cls(tmp_path / name)
        try:
            res[name] = ("ok", fn(log))
        except fs.FirstSeenError as e:
            res[name] = ("err", e.code, str(e).replace(str(tmp_path / name), "<log>"))
    assert res["old"] == res["new"], step
    assert snapshot(tmp_path / "old") == snapshot(tmp_path / "new"), step
    return res["new"]


A = '{"site":"a1"}'
B = '{"site":"b1"}'


def test_record_and_commit_byte_identical_to_whole_forge_version(tmp_path, json_forge):
    forge = tmp_path / "forge"
    # run 1 (baseline): three feeds, a duplicated point across b's partitions,
    # an unreadable partition in c, an empty partition, an empty feed dir.
    put(forge, "a", "2026-11-18", pts(A, T0 - 86400, 24))
    put(forge, "a", "2026-11-19", pts(A, T0 - 3600 * 6, 6) + pts("{}", T0 - 7200, 2))
    put(forge, "b", "2026-11-18", pts(B, T0 - 86400, 5))
    put(forge, "b", "2026-11-19", pts(B, T0 - 86400 + 3600 * 4, 5))
    put(forge, "c", "2026-11-18", pts("{}", T0 - 86400, 3))
    put(forge, "c", "2026-11-19", corrupt=b"PAR1\x00no footer")
    put(forge, "c", "2026-11-20", [])
    (forge / "data" / "empty").mkdir(parents=True)
    out = run_both(tmp_path, forge, "baseline", lambda log: log.record(forge, now_ts=T0 + 10))
    assert out[1]["quarantined"] == 1 and out[1]["unreadable"] == 1

    run_both(tmp_path, forge, "commit 1", lambda log: log.commit(now_ts=T0 + P))

    # run 2: a rewritten (new sha) + a new partition with overlap; c's corrupt
    # partition readable again (blocked); a new feed d (baseline for d only).
    put(forge, "a", "2026-11-19", pts(A, T0 - 3600 * 6, 9) + pts("{}", T0 - 7200, 2))
    put(forge, "a", "2026-11-20", pts(A, T0, 4))
    put(forge, "c", "2026-11-19", pts("{}", T0 - 86400, 6))
    put(forge, "d", "2026-11-20", pts(A, T0, 3))
    out = run_both(tmp_path, forge, "run 2", lambda log: log.record(forge, now_ts=T0 + P + 5))
    assert out[1]["blocked"] == 3 and out[1]["records"] == 3

    # run 3: c's name re-corrupted (new bytes -> quarantined again) and a new
    # corrupt partition in b; unchanged partitions skipped unread.
    put(forge, "c", "2026-11-19", corrupt=b"\x00\x01garbage")
    put(forge, "b", "2026-11-21", corrupt=b"{not json")
    put(forge, "b", "2026-11-22", pts(B, T0 + 3600, 3))
    run_both(tmp_path, forge, "commit 2", lambda log: log.commit(now_ts=T0 + 2 * P))
    out = run_both(tmp_path, forge, "run 3", lambda log: log.record(forge, now_ts=T0 + 2 * P + 7,
                                                                max_unreadable_frac=1.0))
    assert out[1]["unreadable"] == 2

    # run 4: the same corrupt bytes again (counted unreadable, not re-quarantined)
    run_both(tmp_path, forge, "run 4", lambda log: log.record(forge, now_ts=T0 + 2 * P + 9,
                                                                max_unreadable_frac=1.0))
    run_both(tmp_path, forge, "commit 3", lambda log: log.commit(now_ts=T0 + 4 * P))
    # a committed period refuses records
    out = run_both(tmp_path, forge, "committed", lambda log: log.record(forge, now_ts=T0 + 2 * P))
    assert out[:2] == ("err", "period_committed")

    # the derived views agree too
    old, new = OldLog(tmp_path / "old"), fs.FirstSeenLog(tmp_path / "new")
    assert old.index(until_ts=T0 + 5 * P)._entries == new.index(until_ts=T0 + 5 * P)._entries


def test_too_many_unreadable_writes_nothing_like_before(tmp_path, json_forge):
    forge = tmp_path / "forge"
    for i in range(3):
        put(forge, f"f{i}", "2026-11-19", pts(A, T0 - 86400, 4))
    put(forge, "f1", "2026-11-20", corrupt=b"bad")
    put(forge, "f2", "2026-11-20", corrupt=b"worse")
    # fresh log dir: neither version creates it (the new staging dir is removed)
    out = run_both(tmp_path, forge, "fresh",
                   lambda log: log.record(forge, now_ts=T0 + 1, max_unreadable_files=1))
    assert out[:2] == ("err", "too_many_unreadable")
    assert not (tmp_path / "new").exists()
    # nested missing parents are not left behind either
    nested = tmp_path / "x" / "y" / "first_seen"
    with pytest.raises(fs.FirstSeenError):
        fs.FirstSeenLog(nested).record(forge, now_ts=T0 + 1, max_unreadable_files=1)
    assert not (tmp_path / "x").exists()
    # an existing log: unchanged byte for byte, no staging file left behind
    (forge / "data" / "f1" / "2026-11-20.parquet").unlink()
    (forge / "data" / "f2" / "2026-11-20.parquet").unlink()
    run_both(tmp_path, forge, "base", lambda log: log.record(forge, now_ts=T0 + 2))
    put(forge, "f0", "2026-11-20", pts(A, T0, 2))
    put(forge, "f1", "2026-11-20", corrupt=b"bad")
    put(forge, "f2", "2026-11-20", corrupt=b"worse")
    out = run_both(tmp_path, forge, "existing",
                   lambda log: log.record(forge, now_ts=T0 + 3, max_unreadable_frac=0.0,
                                          max_unreadable_files=1))
    assert out[:2] == ("err", "too_many_unreadable")
    assert sorted(snapshot(tmp_path / "new")) == ["records.jsonl"]
    # a nothing-new run over a fresh dir creates nothing (like before)
    empty = tmp_path / "empty-forge"
    run_both(tmp_path / "e", empty, "no data", lambda log: log.record(empty, now_ts=T0))
    assert not (tmp_path / "e" / "new").exists()


def test_only_one_feed_state_alive_at_a_time(tmp_path, json_forge, monkeypatch):
    forge = tmp_path / "forge"
    feeds = [f"feed{i:02d}" for i in range(6)]
    for i, feed in enumerate(feeds):
        put(forge, feed, "2026-11-18", pts(A, T0 - 86400 + i, 10))
        put(forge, feed, "2026-11-19", pts(A, T0 - 3600 * 5 + i, 10))

    class Tracked(set):
        pass

    alive: dict[str, weakref.ref] = {}
    orig_known = fs.FirstSeenLog._feed_known

    def tracked_known(self, feed, hist):
        got = Tracked(orig_known(self, feed, hist))
        alive[feed] = weakref.ref(got)
        return got

    def live_feeds():
        gc.collect()
        return sorted(f for f, ref in alive.items() if ref() is not None)

    peaks = []

    def reading_points(path):
        peaks.append(live_feeds())
        return _json_points(path)

    freed = []

    def on_freed(self, feed):
        assert feed not in live_feeds(), f"{feed}'s known set outlived its feed"
        freed.append(feed)

    monkeypatch.setattr(fs.FirstSeenLog, "_feed_known", tracked_known)
    monkeypatch.setattr(fs.FirstSeenLog, "_feed_state_freed", on_freed)
    monkeypatch.setattr(fs, "partition_points", reading_points)

    log = fs.FirstSeenLog(tmp_path / "log")
    log.record(forge, now_ts=T0 + 1)                        # baseline: no history
    for feed in feeds:                                      # run 2: history per feed
        put(forge, feed, "2026-11-20", pts(A, T0 - 3600 * 5, 20))
    peaks.clear()
    freed.clear()
    log.record(forge, now_ts=T0 + P + 1)
    assert freed == feeds
    assert peaks and all(len(p) <= 1 for p in peaks), peaks
    # the history loaded for a feed is that feed's points only
    hist = log._scan_history()
    for feed in feeds:
        known = orig_known(log, feed, hist)
        want = {(pj, int(t)) for r in log.records() if r["feed"] == feed for pj, t in r["points"]}
        assert known == want
    assert not any(isinstance(v, list) and v and isinstance(v[0], dict) for v in hist.values())


def test_scan_reports_corrupt_line_before_writing(tmp_path, json_forge):
    forge = tmp_path / "forge"
    put(forge, "a", "2026-11-19", pts(A, T0, 2))
    for name in ("old", "new"):
        (tmp_path / name).mkdir()
        good = {"schema": fs.LOG_SCHEMA, "feed": "a", "partition": "x.parquet",
                "partition_sha256": "0" * 64, "seen_ts": T0 - P, "period_start": T0 - P,
                "baseline": True, "points": [[A, T0 - P]]}
        (tmp_path / name / "records.jsonl").write_text(
            fs.canonical_json(good).decode() + "\n\n{oops\n")
    out = run_both(tmp_path, forge, "corrupt", lambda log: log.record(forge, now_ts=T0))
    assert out[:2] == ("err", "log_corrupt") and ":3:" in out[2]


def test_partition_points_matches_previous_reader(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    n = 50
    frame = pd.DataFrame({
        "timestamp": [f"2026-11-{19 + h // 24}T{h % 24:02d}:00:00+00:00" for h in range(n)],
        "v": np.arange(n, dtype=float),
        "_panel_site": [f"s{i % 3}" for i in range(n)],
        "_panel_kind": [i % 2 for i in range(n)],
    })
    frame.loc[7, "timestamp"] = "not a time"
    paneled = tmp_path / "p.parquet"
    frame.to_parquet(paneled)
    plain = tmp_path / "u.parquet"
    frame[["timestamp", "v"]].to_parquet(plain)
    for path in (paneled, plain):
        assert fs.partition_points(path) == _old_partition_points(path)
    assert len(fs.partition_points(paneled)) == n - 1
