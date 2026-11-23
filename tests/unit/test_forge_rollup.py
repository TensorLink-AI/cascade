"""Forge reader rollups (catalog tag ``rollup: sum|mean|last``) and freshness
by newest observation relative to cadence.

The bucketing and freshness rules are pure functions (numpy only); the
harvest-level tests build a miniature forge mirror and need the ``pool-forge``
extra (pandas / pyarrow / pyyaml), like the other forge tests.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from cascade.pool.source import HarvestContext
from cascade.pool.sources.tsbench_forge import (
    ROLLUP_TARGETS,
    freshness_window,
    is_fresh,
    rollup_agg,
    rollup_buckets,
)

H = 3_600
DAY = 86_400
WEEK = 604_800
MON = int(dt.datetime(2026, 3, 2, tzinfo=dt.UTC).timestamp())  # a Monday


def _hourly(n_days, *, start=MON, drop=()):
    sec = start + np.arange(n_days * 24, dtype=np.int64) * H
    vals = np.arange(n_days * 24, dtype=np.float64)
    keep = np.ones(sec.size, bool)
    keep[list(drop)] = False
    return sec[keep], vals[keep]


# ── rollup_buckets: aggregation + complete-bucket rule ──────────────────────

def test_rollup_mean_sum_last_on_complete_days():
    sec, v = _hourly(3)
    starts, mean = rollup_buckets(sec, v, agg="mean", raw_sec=H, bucket_sec=DAY)
    assert list(starts) == [MON, MON + DAY, MON + 2 * DAY]
    assert list(mean) == [11.5, 35.5, 59.5]
    _, s = rollup_buckets(sec, v, agg="sum", raw_sec=H, bucket_sec=DAY)
    assert list(s) == [sum(range(24)), sum(range(24, 48)), sum(range(48, 72))]
    _, last = rollup_buckets(sec, v, agg="last", raw_sec=H, bucket_sec=DAY)
    assert list(last) == [23.0, 47.0, 71.0]


def test_rollup_partial_bucket_is_nan_never_partial_aggregate():
    # Day 1 loses 5 of 24 points (19 < 0.8 * 24 = 19.2) → NaN for every agg.
    # Day 2 loses 4 (20 >= 19.2) → complete; sum is pro-rated to 24 points.
    sec, v = _hourly(3, drop=[24, 25, 26, 27, 28, 48, 49, 50, 71])
    for agg in ("sum", "mean", "last"):
        _, r = rollup_buckets(sec, v, agg=agg, raw_sec=H, bucket_sec=DAY)
        assert np.isnan(r[1]), agg
        assert np.isfinite(r[0]) and np.isfinite(r[2]), agg
    day2 = [float(x) for x in range(51, 71)]
    _, s = rollup_buckets(sec, v, agg="sum", raw_sec=H, bucket_sec=DAY)
    assert s[2] == pytest.approx(np.mean(day2) * 24)
    _, m = rollup_buckets(sec, v, agg="mean", raw_sec=H, bucket_sec=DAY)
    assert m[2] == pytest.approx(np.mean(day2))
    _, last = rollup_buckets(sec, v, agg="last", raw_sec=H, bucket_sec=DAY)
    assert last[2] == 70.0  # 71 is missing: the latest PRESENT observation


def test_rollup_nonfinite_values_count_as_missing():
    sec, v = _hourly(1)
    v = v.copy()
    v[:5] = np.nan
    _, r = rollup_buckets(sec, v, agg="mean", raw_sec=H, bucket_sec=DAY)
    assert np.isnan(r[0])


def test_rollup_leading_partial_day_is_nan():
    sec, v = _hourly(2, start=MON + 12 * H)  # starts at noon
    starts, r = rollup_buckets(sec, v, agg="mean", raw_sec=H, bucket_sec=DAY)
    assert list(starts) == [MON, MON + DAY, MON + 2 * DAY]
    assert np.isnan(r[0]) and np.isfinite(r[1]) and np.isnan(r[2])


def test_rollup_weeks_are_monday_start_utc():
    sec, v = _hourly(14, start=MON - 3 * DAY)  # Friday → ... → Thursday
    starts, r = rollup_buckets(sec, v, agg="mean", raw_sec=H, bucket_sec=WEEK)
    assert all(dt.datetime.fromtimestamp(int(s), tz=dt.UTC).weekday() == 0
               for s in starts)
    assert list(starts) == [MON - WEEK, MON, MON + WEEK]
    assert np.isnan(r[0]) and np.isfinite(r[1]) and np.isnan(r[2])
    assert r[1] == pytest.approx(np.mean(np.arange(72, 72 + 168)))


def test_rollup_weekly_from_daily_raw():
    sec = MON + np.arange(21, dtype=np.int64) * DAY
    v = np.ones(21)
    _, s = rollup_buckets(sec, v, agg="sum", raw_sec=DAY, bucket_sec=WEEK)
    assert list(s) == [7.0, 7.0, 7.0]


def test_rollup_agg_reads_only_known_catalog_tags():
    assert rollup_agg({"id": "x"}) is None
    assert rollup_agg({"id": "x", "rollup": "sum"}) == "sum"
    assert rollup_agg({"id": "x", "rollup": "last"}) == "last"
    assert rollup_agg({"id": "x", "rollup": "median"}) is None   # not in the contract
    assert rollup_agg({"id": "x", "rollup": "Mean"}) is None     # exact values only


def test_rollup_targets_metadata():
    assert ROLLUP_TARGETS == (("D", DAY, "D", 7), ("W", WEEK, "W", 52))


# ── freshness by newest observation relative to cadence ─────────────────────

AS_OF = dt.date(2026, 6, 1)


def _utc(*a):
    return dt.datetime(*a, tzinfo=dt.UTC)


def test_freshness_window_scales_with_cadence():
    assert freshness_window(10, 4) == dt.timedelta(days=4)        # 10 s feed
    assert freshness_window(H, 4) == dt.timedelta(days=4)
    assert freshness_window(DAY, 4) == dt.timedelta(days=4)       # historical rule
    assert freshness_window(WEEK, 4) == dt.timedelta(weeks=4)
    assert freshness_window(2_629_800, 4) == dt.timedelta(seconds=4 * 2_629_800)
    assert freshness_window(None, 4) == dt.timedelta(days=4)


@pytest.mark.parametrize("cadence", [10, DAY])
def test_daily_and_fast_file_rule_is_unchanged(cadence):
    """With an up-to-date observation the verdict equals the old file-date rule
    (fresh iff newest file >= as_of - max_stale_days) at every offset."""
    for lag in range(0, 10):
        newest_file = AS_OF - dt.timedelta(days=lag)
        old = not (newest_file < AS_OF - dt.timedelta(days=4))
        obs = _utc(newest_file.year, newest_file.month, newest_file.day, 23, 59)
        assert is_fresh(as_of=AS_OF, cadence_sec=cadence, max_stale_days=4,
                        newest_file=newest_file, newest_obs=obs) is old, lag
        assert is_fresh(as_of=AS_OF, cadence_sec=cadence, max_stale_days=4,
                        newest_file=newest_file, newest_obs=None) is old, lag


@pytest.mark.parametrize("cadence", [10, DAY])
def test_fresh_file_with_stale_observation_is_stale(cadence):
    # The case the file-date rule got strictly wrong: today's scrape, old data.
    kw = dict(as_of=AS_OF, cadence_sec=cadence, max_stale_days=4, newest_file=AS_OF)
    assert is_fresh(**kw, newest_obs=_utc(2026, 5, 28)) is True     # exactly 4 days
    assert is_fresh(**kw, newest_obs=_utc(2026, 5, 27, 23)) is False


def test_weekly_freshness_by_observation():
    kw = dict(as_of=AS_OF, cadence_sec=WEEK, max_stale_days=4)
    # Weekly data scraped two weeks ago is fresh (the file-date rule dropped it).
    assert is_fresh(**kw, newest_file=AS_OF - dt.timedelta(days=14),
                    newest_obs=_utc(2026, 5, 18)) is True
    assert is_fresh(**kw, newest_file=AS_OF, newest_obs=_utc(2026, 5, 4)) is True   # 28 d
    assert is_fresh(**kw, newest_file=AS_OF, newest_obs=_utc(2026, 5, 3)) is False


def test_monthly_freshness_by_observation():
    kw = dict(as_of=AS_OF, cadence_sec=2_629_800, max_stale_days=4, newest_file=AS_OF)
    assert is_fresh(**kw, newest_obs=_utc(2026, 3, 1)) is True       # ~3 months
    assert is_fresh(**kw, newest_obs=_utc(2026, 1, 15)) is False     # > 4 nominal months


# ── harvest level (forge mirror; needs the pool-forge extra) ────────────────

def _mirror(tmp_path, catalog, frames, as_of):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    yaml = pytest.importorskip("yaml")
    root = tmp_path / "forge"
    root.mkdir()
    (root / "sources.yaml").write_text(yaml.safe_dump(catalog), encoding="utf-8")
    for sid, (day, data) in frames.items():
        d = root / "data" / sid
        d.mkdir(parents=True)
        pd.DataFrame(data).to_parquet(d / f"{day}.parquet")
    return root


ROLL_AS_OF = dt.date(2026, 4, 27)   # Monday after 8 full weeks from MON
CTX = HarvestContext(as_of=ROLL_AS_OF, context_length=512, horizon=16, max_series=100)


def _rollup_mirror(tmp_path, tag="sum"):
    pd = pytest.importorskip("pandas")
    n = 56 * 24
    hours = pd.date_range("2026-03-02", periods=n, freq="h", tz="UTC")
    keep = np.ones(n, bool)
    keep[[72, 73, 74, 75, 76]] = False          # day 3 incomplete (19/24)
    keep[[120, 121, 122, 123]] = False          # day 5 complete within tolerance (20/24)
    vals = np.arange(n, dtype=float)
    days = pd.date_range("2026-03-02", periods=56, freq="D", tz="UTC")
    tagged = {"id": "meter", "domain": "energy", "frequency": "PT1H",
              "source_family": "fixture:meter"}
    if tag is not None:
        tagged["rollup"] = tag
    catalog = [
        tagged,
        {"id": "plain", "domain": "energy", "frequency": "PT1H", "source_family": "fixture:p"},
        {"id": "daily", "domain": "sales", "frequency": "P1D", "rollup": "mean",
         "source_family": "fixture:d"},
    ]
    frames = {
        "meter": (ROLL_AS_OF, {"timestamp": hours[keep].astype(str), "kwh": vals[keep]}),
        "plain": (ROLL_AS_OF, {"timestamp": hours.astype(str), "kwh": vals % 24}),
        "daily": (ROLL_AS_OF, {"timestamp": days.astype(str),
                               "units": np.arange(56, dtype=float)}),
    }
    return _mirror(tmp_path, catalog, frames, ROLL_AS_OF)


def _harvest(root, **kw):
    from cascade.pool.sources.tsbench_forge import TsbenchForgeSource
    return list(TsbenchForgeSource(root, **kw).harvest(lambda u, p: None, CTX))


def test_harvest_emits_tagged_rollups_with_ids_and_cadence(tmp_path):
    out = _harvest(_rollup_mirror(tmp_path, "sum"))
    by_id = {s.series_id: s for s in out}
    assert [s.series_id for s in out] == [
        "tsforge__meter", "tsforge__meter.rollup_D", "tsforge__meter.rollup_W",
        "tsforge__plain",                                 # untagged: no rollup
        "tsforge__daily", "tsforge__daily.rollup_W",      # daily raw: weekly only
    ]
    d, w = by_id["tsforge__meter.rollup_D"], by_id["tsforge__meter.rollup_W"]
    assert (d.freq, d.seasonal_period, w.freq, w.seasonal_period) == ("D", 7, "W", 52)
    for s in (d, w):
        assert s.source == "meter" and s.domain == "energy"
        assert s.attrs["rollup"] == "sum"
        assert s.attrs["source_family"] == by_id["tsforge__meter"].attrs["source_family"]
    assert d.values.shape == (56,) and w.values.shape == (8,)
    assert np.isnan(d.values[3])                                  # incomplete day
    assert d.values[0] == sum(range(24))
    present5 = [x for x in range(120, 144) if x not in (120, 121, 122, 123)]
    assert d.values[5] == pytest.approx(np.mean(present5) * 24)  # pro-rated
    assert w.values[1] == sum(range(168, 336))
    assert np.isfinite(w.values).all()                            # 163/168 >= 0.8
    assert by_id["tsforge__daily.rollup_W"].values.tolist() == [
        float(np.mean(np.arange(7 * k, 7 * k + 7))) for k in range(8)]


@pytest.mark.parametrize("agg", ["mean", "last"])
def test_harvest_rollup_mean_and_last(tmp_path, agg):
    out = {s.series_id: s for s in _harvest(_rollup_mirror(tmp_path, agg))}
    d = out["tsforge__meter.rollup_D"].values
    expect0 = 11.5 if agg == "mean" else 23.0
    assert d[0] == expect0 and np.isnan(d[3])


def test_untagged_and_invalid_tags_emit_no_rollup(tmp_path):
    for tag in (None, "median"):
        sub = tmp_path / str(tag)
        sub.mkdir()
        ids = [s.series_id for s in _harvest(_rollup_mirror(sub, tag))]
        assert not any(i.startswith("tsforge__meter.") for i in ids), tag
        assert "tsforge__meter" in ids


def test_rollup_harvest_and_build_are_deterministic(tmp_path):
    from cascade.pool.builder import PoolBuildConfig, build_pool
    from cascade.pool.sources.tsbench_forge import TsbenchForgeSource

    root = _rollup_mirror(tmp_path, "sum")
    a, b = _harvest(root), _harvest(root)
    assert [s.series_id for s in a] == [s.series_id for s in b]
    for x, y in zip(a, b, strict=True):
        assert x.values.tobytes() == y.values.tobytes()

    cfg = PoolBuildConfig(context_length=64, horizon=4, min_context=4)
    dirs = []
    for name in ("p1", "p2"):
        build_pool([TsbenchForgeSource(root)], tmp_path / name, CTX, cfg)
        dirs.append(tmp_path / name)
    files = sorted(p.name for p in dirs[0].iterdir())
    assert "tsforge__meter.rollup_D.npy" in files and "tsforge__meter.rollup_W.npy" in files
    assert files == sorted(p.name for p in dirs[1].iterdir())
    for f in files:
        a_bytes, b_bytes = (dirs[0] / f).read_bytes(), (dirs[1] / f).read_bytes()
        if f == "provenance.json":  # records the (differing) output path
            a_bytes = a_bytes.replace(str(dirs[0]).encode(), b"OUT")
            b_bytes = b_bytes.replace(str(dirs[1]).encode(), b"OUT")
        assert a_bytes == b_bytes, f


def test_fresh_file_with_stale_observations_is_skipped(tmp_path):
    """A 10 s feed scraped today whose newest observation is 10 days old is
    stale (the file-date rule kept it); a 10 s feed with current data is not."""
    pd = pytest.importorskip("pandas")
    old = pd.date_range("2026-04-10", periods=500, freq="10s", tz="UTC")
    new = pd.date_range("2026-04-26", periods=500, freq="10s", tz="UTC")
    root = _mirror(tmp_path, [
        {"id": "dead", "domain": "nature", "frequency": "PT10S"},
        {"id": "live", "domain": "nature", "frequency": "PT10S"},
    ], {
        "dead": (ROLL_AS_OF, {"timestamp": old.astype(str), "v": np.sin(np.arange(500.))}),
        "live": (ROLL_AS_OF, {"timestamp": new.astype(str), "v": np.sin(np.arange(500.))}),
    }, ROLL_AS_OF)
    assert [s.series_id for s in _harvest(root)] == ["tsforge__live"]
