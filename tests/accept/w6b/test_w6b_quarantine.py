"""W6b: the pool_d build skips unreadable / quarantined partitions (2026-11-25).

A partition the W6fs quarantine names is never read at ``as_of_ts``, even
once repaired (its values must not enter lane D); any other unreadable
partition is skipped and logged. Both lists land in the signed metadata so
audits see them; a systemic read failure still aborts the build.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from .helpers import (
    AS1,
    AS2,
    CTX_A,
    LANE_A_AS_OF,
    LANE_A_CFG,
    T0,
    _write,
    by_id,
    forge_mirror,
    forge_source,
    hourly_ts,
    iso,
    meta,
    mods,
    no_fetch,
)

CORRUPT = b"PAR1" + b"\x00" * 64          # parquet magic, no footer
# Sorts after the lane A day partition and is <= the as_of day, so its rows
# would win the per-timestamp dedup if it were read.
LATE_DAY = "2026-11-20"


def _world(base):
    b, sel, fs = mods()
    root = forge_mirror(base / "forge")
    (root / "data" / "air_temp" / f"{LATE_DAY}.parquet").write_bytes(CORRUPT)
    log = fs.FirstSeenLog(root / "first_seen")
    try:
        log.record(root, now_ts=T0 + 600)
    except Exception as e:  # noqa: BLE001 — the pre-fix abort is the defect under test
        raise AssertionError(f"first_seen record aborted on an unreadable partition: "
                             f"{e!r}") from e
    log.commit(now_ts=AS1)
    _summary, selection = sel.build_pool_with_selection(
        [forge_source(root)], base / "pool", CTX_A, LANE_A_CFG, fetch=no_fetch)
    return b, fs, root, log, selection


def _air(m):
    return [s for s in m["series"] if s["feed"] == "air_temp"]


def test_build_continues_and_records_the_quarantine(tmp_path, capsys):
    b, _fs, root, _log, selection = _world(tmp_path)
    s1 = b.build_pool_d(forge_dir=root, selection=selection, first_seen_dir=root / "first_seen",
                        as_of_ts=AS1, out_dir=tmp_path / "d1")
    m = meta(tmp_path / "d1")
    assert _air(m), "the feed with a quarantined partition still yields its series"
    assert [(q["kind"], q["feed"], q["partition"]) for q in m["quarantine"]] == [
        ("unreadable", "air_temp", f"{LATE_DAY}.parquet")]
    assert m["quarantine"][0]["size"] == len(CORRUPT) and m["quarantine"][0]["reason"]
    assert "unreadable_partitions" not in m        # quarantined ones are not even read
    assert s1.quarantine == m["quarantine"]
    assert b.validate_pool_d(tmp_path / "d1") == []


def test_repaired_quarantined_partition_stays_out(tmp_path):
    b, _fs, root, log, selection = _world(tmp_path)
    b.build_pool_d(forge_dir=root, selection=selection, first_seen_dir=root / "first_seen",
                   as_of_ts=AS1, out_dir=tmp_path / "d1")
    before = {s["series_id"]: s for s in _air(meta(tmp_path / "d1"))}
    with np.load(tmp_path / "d1" / "tgt.npz") as t:
        tgt_before = {sid: t[sid].copy() for sid in before}
    # repaired: same timestamps, different values (it would win the dedup)
    ts = hourly_ts(3_000)
    _write(root, "air_temp", LATE_DAY, {"timestamp": [iso(t) for t in ts],
                                       "t": np.full(len(ts), 999.0)})
    out = log.record(root, now_ts=AS1 + 600)
    assert out.get("blocked") == 0 and out["records"] == 0
    log.commit(now_ts=AS2)
    b.build_pool_d(forge_dir=root, selection=selection, first_seen_dir=root / "first_seen",
                   as_of_ts=AS2, out_dir=tmp_path / "d2")
    m2 = meta(tmp_path / "d2")
    assert {s["series_id"] for s in _air(m2)} == set(before)
    with np.load(tmp_path / "d2" / "tgt.npz") as t:
        for sid in before:
            assert not np.any(t[sid] == 999.0)
            np.testing.assert_array_equal(t[sid], tgt_before[sid])
    kinds = sorted(q["kind"] for q in m2["quarantine"])
    assert kinds == ["blocked", "unreadable"]
    blocked = next(q for q in m2["quarantine"] if q["kind"] == "blocked")
    assert blocked["n_blocked_points"] == 0 and "points" not in blocked


def test_unquarantined_unreadable_partition_is_logged_and_recorded(tmp_path, capsys):
    b, _fs, root, _log, selection = _world(tmp_path)
    # corrupted after the record step: unknown to the quarantine
    (root / "data" / "fx_fwd" / f"{LANE_A_AS_OF.isoformat()}.parquet").write_bytes(CORRUPT)
    s = b.build_pool_d(forge_dir=root, selection=selection, first_seen_dir=root / "first_seen",
                       as_of_ts=AS1, out_dir=tmp_path / "d1")
    assert "skipping unreadable partition data/fx_fwd/" in capsys.readouterr().err
    m = meta(tmp_path / "d1")
    assert [u[:2] for u in m["unreadable_partitions"]] == [
        ["fx_fwd", f"{LANE_A_AS_OF.isoformat()}.parquet"]]
    assert s.unreadable == m["unreadable_partitions"]
    assert not [x for x in m["series"] if x["feed"] == "fx_fwd"]


def test_systemic_read_failure_aborts_the_build(tmp_path):
    b, _fs, root, _log, selection = _world(tmp_path)
    (root / "data" / "fx_fwd" / f"{LANE_A_AS_OF.isoformat()}.parquet").write_bytes(CORRUPT)
    cfg = dataclasses.replace(b.PoolDBuildConfig(), max_unreadable_files=0)
    with pytest.raises(ValueError, match="too_many_unreadable"):
        b.build_pool_d(forge_dir=root, selection=selection, first_seen_dir=root / "first_seen",
                       as_of_ts=AS1, out_dir=tmp_path / "d1", cfg=cfg)
    assert not (tmp_path / "d1" / "metadata.json").exists()


def test_no_quarantine_keys_when_nothing_is_skipped(tmp_path):
    b, sel, fs = mods()
    root = forge_mirror(tmp_path / "forge")
    log = fs.FirstSeenLog(root / "first_seen")
    log.record(root, now_ts=T0 + 600)
    log.commit(now_ts=AS1)
    _summary, selection = sel.build_pool_with_selection(
        [forge_source(root)], tmp_path / "pool", CTX_A, LANE_A_CFG, fetch=no_fetch)
    s = b.build_pool_d(forge_dir=root, selection=selection, first_seen_dir=root / "first_seen",
                       as_of_ts=AS1, out_dir=tmp_path / "d1")
    m = meta(tmp_path / "d1")
    assert "quarantine" not in m and "unreadable_partitions" not in m
    assert by_id(m)
    # the summary always reports both lists (empty here)
    assert getattr(s, "quarantine", None) == [] and getattr(s, "unreadable", None) == []
