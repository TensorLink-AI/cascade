"""W6b (#841): event / irregular feeds with repeated timestamps never void
the whole `pool_d/1` snapshot.

Testnet RC 02335f09 refused the whole snapshot because two event feeds
(`emsc_seismic_events`, `usgs_earthquakes_realtime`) carried repeated data
timestamps. The fixture reproduces both: one feed whose repeated rows agree
(de-duplicated deterministically at selection, then built), one whose
repeated rows disagree (refused per series, typed). The rest of the snapshot
is built, and every series without repeats is byte-identical to a build
without the event feeds.
"""

from __future__ import annotations

import numpy as np

from .helpers import (
    AS1,
    CATALOG,
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
    signal,
)

EMSC = "emsc_seismic_events"
USGS = "usgs_earthquakes_realtime"
EMSC_SID = f"tsforge__{EMSC}~{EMSC}.c0"
USGS_SID = f"tsforge__{USGS}~{USGS}.c0"
AIR = "tsforge__air_temp~air_temp.c0"
GRID_R1 = "tsforge__grid_load__region_r1~grid_load.c0"
N_EVENT = 2_000
# repeated rows land inside the target window and inside the history
REPEAT_AT = (N_EVENT - 3, N_EVENT - 30, N_EVENT - 500)

EVENT_CATALOG = [
    {"id": EMSC, "domain": "nature", "frequency": "PT1H",
     "source_family": "fixture:emsc"},
    {"id": USGS, "domain": "nature", "frequency": "PT1H",
     "source_family": "fixture:usgs"},
]


def _subsecond(t: int, frac: str) -> str:
    """``iso(t)`` with a fractional second: distinct raw strings (so the
    forge reader's raw-``timestamp`` dedup keeps both rows) that floor to
    the SAME epoch second, as the testnet event feeds do."""
    return iso(t).replace("+00:00", f".{frac}+00:00")


def _event_frame(seed: int, *, conflict: bool) -> dict:
    ts = hourly_ts(N_EVENT)
    vals = list(signal(N_EVENT, seed, 24.0))
    rows_ts, rows_v = [], []
    for i, (t, v) in enumerate(zip(ts, vals, strict=True)):
        if i in REPEAT_AT:                 # two events reported inside one second
            rows_ts += [_subsecond(t, "250000"), _subsecond(t, "750000")]
            rows_v += [v, v + 1.0 if conflict else v]
        else:
            rows_ts.append(iso(t))
            rows_v.append(v)
    return {"timestamp": rows_ts, "m": np.asarray(rows_v)}


def _world(base, *, with_events: bool):
    b, sel, fs = mods()
    catalog = CATALOG + (EVENT_CATALOG if with_events else [])
    root = forge_mirror(base / "forge", catalog)
    if with_events:
        day = LANE_A_AS_OF.isoformat()
        _write(root, EMSC, day, _event_frame(11, conflict=False))
        _write(root, USGS, day, _event_frame(12, conflict=True))
    log = fs.FirstSeenLog(root / "first_seen")
    log.record(root, now_ts=T0 + 600)
    log.commit(now_ts=AS1)
    _summary, selection = sel.build_pool_with_selection(
        [forge_source(root)], base / "pool", CTX_A, LANE_A_CFG, fetch=no_fetch)
    return b, root, selection


def _build(b, root, selection, out):
    return b.build_pool_d(forge_dir=root, selection=selection,
                          first_seen_dir=root / "first_seen", as_of_ts=AS1, out_dir=out)


def test_repeated_timestamps_refuse_per_series_not_the_snapshot(tmp_path):
    b, root, selection = _world(tmp_path / "ev", with_events=True)
    picked = {s.pool_d_series_id for s in selection.series}
    assert {EMSC_SID, USGS_SID} <= picked, picked   # lane A selected both feeds
    summary = _build(b, root, selection, tmp_path / "d")
    assert b.validate_pool_d(tmp_path / "d") == []
    m = meta(tmp_path / "d")
    series, refused = by_id(m), dict(map(tuple, m["refused"]))
    # disagreeing repeats: refused per series, typed, out of the arrays
    assert refused[USGS_SID] == "duplicate_ts_conflict"
    assert summary.refused[USGS_SID] == "duplicate_ts_conflict"
    assert USGS_SID not in series
    assert USGS not in m["feeds"]
    # agreeing repeats: de-duplicated, the series is built
    assert EMSC_SID in series and EMSC_SID not in refused
    ts = series[EMSC_SID]["target_ts"]
    assert all(b2 > a for a, b2 in zip(ts, ts[1:], strict=False))
    assert ts == [int(t) for t in hourly_ts(N_EVENT)[-m["horizon"]:]]
    assert EMSC in m["feeds"]
    # the rest of the snapshot is built
    assert AIR in series and GRID_R1 in series
    with np.load(tmp_path / "d" / "tgt.npz") as tgt:
        assert set(tgt.files) == set(series)


def test_event_feed_build_is_deterministic(tmp_path):
    b, root, selection = _world(tmp_path / "ev", with_events=True)
    s1 = _build(b, root, selection, tmp_path / "d1")
    s2 = _build(b, root, selection, tmp_path / "d2")
    assert s1.sha256 == s2.sha256
    for n in b.SNAPSHOT_FILES:
        assert (tmp_path / "d1" / n).read_bytes() == (tmp_path / "d2" / n).read_bytes()
    assert s1.sha256 == b.snapshot_sha256(tmp_path / "d1")


def test_series_without_repeats_unchanged(tmp_path):
    b, root_e, sel_e = _world(tmp_path / "ev", with_events=True)
    _b, root_p, sel_p = _world(tmp_path / "plain", with_events=False)
    _build(b, root_e, sel_e, tmp_path / "de")
    _build(b, root_p, sel_p, tmp_path / "dp")
    me, mp = by_id(meta(tmp_path / "de")), by_id(meta(tmp_path / "dp"))
    for sid in (AIR, GRID_R1):
        assert me[sid] == mp[sid], sid
    with np.load(tmp_path / "de" / "hist.npz") as he, np.load(tmp_path / "dp" / "hist.npz") as hp:
        for sid in (AIR, GRID_R1):
            assert he[sid].tobytes() == hp[sid].tobytes(), sid


def test_dedupe_helper_rules():
    b, _sel, _fs = mods()
    ts = np.array([1, 2, 2, 3, 3, 3, 4], dtype=np.int64)
    seen = np.array([9, 9, 9, 9, 9, 9, 9], dtype=np.int64)
    vals = np.array([1.0, 2.0, 2.0, np.nan, np.nan, np.nan, 4.0])
    got = b._dedupe_repeated_ts(ts, seen, vals)
    assert got is not None
    assert got[0].tolist() == [1, 2, 3, 4]
    assert got[2][[0, 1, 3]].tolist() == [1.0, 2.0, 4.0] and np.isnan(got[2][2])
    clash = vals.copy()
    clash[2] = 2.5
    assert b._dedupe_repeated_ts(ts, seen, clash) is None
    plain = np.array([1, 2, 3], dtype=np.int64)
    out = b._dedupe_repeated_ts(plain, plain, plain.astype(float))
    assert out[0] is plain                  # no repeats: untouched


def test_per_series_invariant_refusals():
    b, _sel, _fs = mods()
    assert b._series_invariant_refusal([1, 2, 3], [0, 0, 0], 10) is None
    assert b._series_invariant_refusal([1, 2, 2], [0, 0, 0], 10) == "target_ts_not_increasing"
    assert b._series_invariant_refusal([1, 2, 3], [0, 10, 0], 10) == \
        "first_seen_not_before_as_of"
