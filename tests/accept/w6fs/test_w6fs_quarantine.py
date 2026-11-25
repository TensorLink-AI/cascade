"""W6fs: unreadable partitions are quarantined, never fresh (2026-11-25).

Two forge objects with no parquet footer aborted every ``first_seen record``.
A corrupt partition is now skipped, logged and appended to the append-only
``quarantine.jsonl``; its name stays quarantined forever, so when it becomes
readable again its unseen (old) points are blocked, never recorded as first
seen. Too many unreadable partitions (a systemic read failure) still abort.
"""

from __future__ import annotations

import json

import pytest

from .helpers import HOUR, PERIOD, T0, day_of, fs_mod, hourly, scrape

CORRUPT = b"PAR1" + b"\x00" * 64          # parquet magic, no footer


def _corrupt(root, feed, day):
    d = root / "data" / feed
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{day}.parquet"
    path.write_bytes(CORRUPT)
    return path


def _mirror(root):
    """Feed ``good`` readable; feed ``bad`` with one readable and one corrupt
    partition (the corrupt one, once repaired, holds old July-like points)."""
    scrape(root, "good", day_of(T0), hourly(T0 - 24 * HOUR, 24))
    scrape(root, "bad", day_of(T0), hourly(T0 - 12 * HOUR, 12))
    _corrupt(root, "bad", "2026-07-22")
    return root


def test_unreadable_partition_is_skipped_and_quarantined(tmp_path, capsys):
    fs = fs_mod()
    root = _mirror(tmp_path / "forge")
    log = fs.FirstSeenLog(root / "first_seen")
    out = log.record(root, now_ts=T0 + 600)
    assert out == {"records": 2, "points": 36, "quarantined": 1, "unreadable": 1}
    err = capsys.readouterr().err
    assert "QUARANTINE" in err and "data/bad/2026-07-22.parquet" in err
    q = log.quarantine()
    assert len(q) == 1
    e = q[0]
    assert e["schema"] == fs.QUARANTINE_SCHEMA and e["kind"] == "unreadable"
    assert (e["feed"], e["partition"]) == ("bad", "2026-07-22.parquet")
    assert e["size"] == len(CORRUPT) and len(e["partition_sha256"]) == 64
    assert e["quarantined_ts"] == T0 + 600 and e["reason"]
    assert {r["partition"] for r in log.records()} == {f"{day_of(T0)}.parquet"}


def test_quarantine_persists_and_is_append_only(tmp_path):
    fs = fs_mod()
    root = _mirror(tmp_path / "forge")
    log = fs.FirstSeenLog(root / "first_seen")
    log.record(root, now_ts=T0 + 600)
    before = log.quarantine_path.read_bytes()
    # a second run (fresh object, same files): no duplicate entry, still counted
    again = fs.FirstSeenLog(root / "first_seen").record(root, now_ts=T0 + PERIOD + 600)
    assert again == {"records": 0, "points": 0, "unreadable": 1}
    assert log.quarantine_path.read_bytes() == before
    assert log.quarantined() == {("bad", "2026-07-22.parquet"): T0 + 600}


def test_repaired_partition_never_becomes_fresh(tmp_path):
    fs = fs_mod()
    root = _mirror(tmp_path / "forge")
    log = fs.FirstSeenLog(root / "first_seen")
    log.record(root, now_ts=T0 + 600)
    log.commit(now_ts=T0 + PERIOD)
    # the object is repaired: it now holds old points (and one shared point)
    july = hourly(T0 - 120 * 24 * HOUR, 48)
    shared = T0 - 1 * HOUR                     # already recorded from the good partition
    scrape(root, "bad", "2026-07-22", july + [shared])
    out = log.record(root, now_ts=T0 + 2 * PERIOD + 600)
    assert out["records"] == 0 and out["points"] == 0 and out["blocked"] == 48
    blocked = [e for e in log.quarantine() if e["kind"] == "blocked"]
    assert len(blocked) == 1 and len(blocked[0]["points"]) == 48
    log.commit(now_ts=T0 + 4 * PERIOD)
    idx = log.index(until_ts=T0 + 4 * PERIOD)
    assert all(idx.first_seen("bad", {}, t) is None for t in july)
    assert idx.first_seen("bad", {}, shared) == T0 + 600     # genuine time kept
    # the same old points re-delivered in a NEW partition stay blocked
    scrape(root, "bad", "2026-11-21", july[:10] + [T0 + 5 * HOUR])
    out2 = log.record(root, now_ts=T0 + 4 * PERIOD + 600)
    assert out2["points"] == 1
    log.commit(now_ts=T0 + 6 * PERIOD)
    idx = log.index(until_ts=T0 + 6 * PERIOD)
    assert all(idx.first_seen("bad", {}, t) is None for t in july)
    assert idx.first_seen("bad", {}, T0 + 5 * HOUR) == T0 + 4 * PERIOD + 600
    # a further rewrite of the quarantined name is still never recorded
    scrape(root, "bad", "2026-07-22", july + [T0 + 9 * HOUR])
    log.record(root, now_ts=T0 + 6 * PERIOD + 600)
    assert not any(r["partition"] == "2026-07-22.parquet" for r in log.records())


def test_index_ignores_records_of_a_quarantined_name_after_quarantine(tmp_path):
    """A record a quarantine-unaware release wrote for a repaired object
    (seen after the quarantine) never yields a first-seen time."""
    fs = fs_mod()
    root = _mirror(tmp_path / "forge")
    log = fs.FirstSeenLog(root / "first_seen")
    log.record(root, now_ts=T0 + 600)
    stale = {"schema": fs.LOG_SCHEMA, "feed": "bad", "partition": "2026-07-22.parquet",
             "partition_sha256": "0" * 64, "seen_ts": T0 + PERIOD + 600,
             "period_start": T0 + PERIOD, "baseline": False,
             "points": [["{}", T0 - 100 * 24 * HOUR]]}
    with open(log.records_path, "a", encoding="utf-8") as f:
        f.write(fs.canonical_json(stale).decode() + "\n")
    log.commit(now_ts=T0 + 2 * PERIOD)
    idx = log.index(until_ts=T0 + 2 * PERIOD)
    assert idx.first_seen("bad", {}, T0 - 100 * 24 * HOUR) is None
    assert idx.first_seen("bad", {}, T0 - 1 * HOUR) == T0 + 600


def test_too_many_unreadable_aborts_and_writes_nothing(tmp_path):
    fs = fs_mod()
    root = _mirror(tmp_path / "forge")
    _corrupt(root, "good", "2026-07-14")
    log = fs.FirstSeenLog(root / "first_seen")
    with pytest.raises(fs.FirstSeenError) as ei:
        log.record(root, now_ts=T0 + 600, max_unreadable_files=1)
    assert ei.value.code == "too_many_unreadable"
    assert not log.records_path.exists() and not log.quarantine_path.exists()
    # the fraction bound: 2 of 4 partitions is over 25%
    with pytest.raises(fs.FirstSeenError):
        log.record(root, now_ts=T0 + 600, max_unreadable_frac=0.25)
    assert log.record(root, now_ts=T0 + 600, max_unreadable_frac=0.5)["quarantined"] == 2


def test_unreadable_limit_defaults():
    fs = fs_mod()
    assert fs.DEFAULT_MAX_UNREADABLE_FILES == 50
    assert fs.DEFAULT_MAX_UNREADABLE_FRAC == 0.001
    assert fs.unreadable_limit(477_428) == 50          # the live forge mirror: 2 pass
    assert fs.unreadable_limit(20_000) == 20
    assert fs.unreadable_limit(4) == 1                 # never below one


def test_cli_record_reports_quarantine(tmp_path, capsys):
    fs = fs_mod()
    root = _mirror(tmp_path / "forge")
    assert fs.main(["record", "--forge-dir", str(root), "--now-ts", str(T0 + 600)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["quarantined"] == 1
    _corrupt(root, "good", "2026-07-14")
    rc = fs.main(["record", "--forge-dir", str(root), "--now-ts", str(T0 + PERIOD + 600),
                  "--max-unreadable-files", "1"])
    assert rc == 2 and "too_many_unreadable" in capsys.readouterr().err
