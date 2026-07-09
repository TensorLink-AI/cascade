"""Data sources — parsing against canned API payloads (no live network) and the
synthetic source's determinism."""

from __future__ import annotations

import datetime as dt

import numpy as np

from cascade.pool.source import HarvestContext
from cascade.pool.sources.openmeteo import OpenMeteoSource
from cascade.pool.sources.synthetic import SyntheticSource
from cascade.pool.sources.wikimedia import WikimediaSource

CTX = HarvestContext(as_of=dt.date(2026, 6, 1), context_length=512, horizon=16, max_series=1000)


def test_openmeteo_parses_hourly_and_maps_nulls_to_nan():
    captured = {}

    def fake_fetch(url, params):
        captured["url"] = url
        captured["params"] = params
        return {
            "hourly": {
                "time": ["2026-01-01T00:00", "2026-01-01T01:00", "2026-01-01T02:00"],
                "temperature_2m": [1.0, None, 3.0],
                "relative_humidity_2m": [50.0, 55.0, 60.0],
                "surface_pressure": [1010.0, 1011.0, 1012.0],
                "wind_speed_10m": [2.0, 2.5, 3.0],
            }
        }

    src = OpenMeteoSource(locations=(("tokyo", 35.69, 139.69),))
    out = list(src.harvest(fake_fetch, CTX))
    assert len(out) == 4  # one series per variable
    temp = next(s for s in out if s.series_id.endswith("temperature_2m"))
    assert temp.freq == "H" and temp.domain == "weather" and temp.seasonal_period == 24
    assert np.isnan(temp.values[1]) and temp.values[0] == 1.0
    assert captured["params"]["latitude"] == 35.69
    assert "archive-api.open-meteo.com" in captured["url"]


def test_openmeteo_handles_empty_response():
    src = OpenMeteoSource(locations=(("nowhere", 0.0, 0.0),))
    assert list(src.harvest(lambda u, p: {}, CTX)) == []


def test_openmeteo_respects_max_series():
    def fake_fetch(url, params):
        return {
            "hourly": {
                "time": ["t0", "t1"],
                "temperature_2m": [1.0, 2.0],
                "relative_humidity_2m": [3.0, 4.0],
                "surface_pressure": [5.0, 6.0],
                "wind_speed_10m": [7.0, 8.0],
            }
        }

    ctx = HarvestContext(as_of=dt.date(2026, 6, 1), max_series=3)
    src = OpenMeteoSource(locations=(("a", 0, 0), ("b", 0, 0)))
    out = list(src.harvest(fake_fetch, ctx))
    assert len(out) == 3  # capped mid-stream


def test_wikimedia_parses_items_and_builds_url():
    captured = {}

    def fake_fetch(url, params):
        captured["url"] = url
        return {"items": [{"timestamp": "2024010100", "views": 100}, {"timestamp": "2024010200", "views": 120}, {"timestamp": "2024010300", "views": 90}]}

    src = WikimediaSource(articles=("Apple_Inc.",))
    out = list(src.harvest(fake_fetch, CTX))
    assert len(out) == 1
    s = out[0]
    assert s.freq == "D" and s.domain == "web_traffic" and s.seasonal_period == 7
    assert list(s.values) == [100.0, 120.0, 90.0]
    # special chars in the article are percent-encoded in the path
    assert "Apple_Inc.%2C" not in captured["url"]  # '.' is safe; comma would be encoded
    assert "per-article/en.wikipedia/all-access/all-agents/Apple_Inc." in captured["url"]


def test_wikimedia_skips_sparse_articles():
    src = WikimediaSource(articles=("X",))
    assert list(src.harvest(lambda u, p: {"items": [{"views": 1}]}, CTX)) == []


def test_wikimedia_end_date_lags_as_of():
    # Pageviews lag ~2 days; requesting up to as_of 404s the whole range. The
    # requested end must be as_of - PAGEVIEW_LAG_DAYS.
    from cascade.pool.sources.wikimedia import PAGEVIEW_LAG_DAYS

    captured = {}

    def fake_fetch(url, params):
        captured["url"] = url
        return {"items": [{"views": 1}, {"views": 2}, {"views": 3}]}

    ctx = HarvestContext(as_of=dt.date(2026, 7, 9), context_length=512, horizon=16, max_series=10)
    list(WikimediaSource(articles=("Physics",)).harvest(fake_fetch, ctx))
    end = (dt.date(2026, 7, 9) - dt.timedelta(days=PAGEVIEW_LAG_DAYS)).strftime("%Y%m%d")
    assert captured["url"].rstrip("/").endswith(end)
    assert "20260709" not in captured["url"]  # never requests today


# ── HttpFetcher rate-limit (429) handling ──────────────────────────────────────


def _fetcher_with_responses(responses, sleeps):
    """An HttpFetcher whose urlopen yields ``responses`` in order (an int status
    raises that HTTPError; a dict is returned as JSON), recording sleeps."""
    import io
    import json as _json
    import urllib.error

    from cascade.pool.source import HttpFetcher

    seq = iter(responses)

    class _Resp(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): self.close()

    def fake_urlopen(req, timeout=None):
        item = next(seq)
        if isinstance(item, tuple):  # (code, headers)
            code, headers = item
            raise urllib.error.HTTPError(req.full_url, code, "err", headers, None)
        return _Resp(_json.dumps(item).encode())

    import cascade.pool.source as src_mod
    orig = src_mod.urllib.request.urlopen
    src_mod.urllib.request.urlopen = fake_urlopen
    f = HttpFetcher(_sleep=lambda s: sleeps.append(s))
    return f, (lambda: setattr(src_mod.urllib.request, "urlopen", orig))


def test_fetcher_retries_429_with_own_budget():
    from email.message import Message

    sleeps = []
    hdr = Message()  # no Retry-After → geometric backoff
    f, restore = _fetcher_with_responses(
        [(429, hdr), (429, hdr), {"ok": 1}], sleeps)
    try:
        assert f("https://x/y") == {"ok": 1}
    finally:
        restore()
    # two 429s → two backoff sleeps (rate_limit_backoff ** 0, ** 1) = 1.0, 2.0
    assert sleeps == [1.0, 2.0]


def test_fetcher_honors_retry_after_header():
    from email.message import Message

    sleeps = []
    hdr = Message()
    hdr["Retry-After"] = "5"
    f, restore = _fetcher_with_responses([(429, hdr), {"ok": 2}], sleeps)
    try:
        assert f("https://x/y") == {"ok": 2}
    finally:
        restore()
    assert sleeps == [5.0]   # slept exactly the header value


def test_fetcher_429_exhausts_and_raises():
    from email.message import Message

    from cascade.pool.source import HarvestError

    sleeps = []
    hdr = Message()
    f, restore = _fetcher_with_responses([(429, hdr)] * 10, sleeps)
    f.rate_limit_retries = 3
    try:
        import pytest
        with pytest.raises(HarvestError, match="fetch_failed"):
            f("https://x/y")
    finally:
        restore()
    assert len(sleeps) == 3   # rate_limit_retries attempts, then give up


def test_fetcher_4xx_fails_fast_no_retry():
    from email.message import Message

    from cascade.pool.source import HarvestError

    sleeps = []
    hdr = Message()
    f, restore = _fetcher_with_responses([(404, hdr), {"never": 1}], sleeps)
    try:
        import pytest
        with pytest.raises(HarvestError, match="http_404"):
            f("https://x/y")
    finally:
        restore()
    assert sleeps == []   # 404 is terminal, no backoff


def test_default_sources_can_supply_2000_windows():
    from cascade.pool.sources.openmeteo import OpenMeteoSource, global_grid
    from cascade.pool.sources.wikimedia import WikimediaSource

    grid = global_grid()
    assert len(grid) > 200
    src = OpenMeteoSource()
    weather = len(src.locations) * len(src.variables)
    web = len(WikimediaSource().articles)
    # Default pool must clear [eval] n_windows = 2000 with margin for validation drops.
    assert weather + web >= 2500


def test_global_grid_denser_with_smaller_steps():
    from cascade.pool.sources.openmeteo import global_grid

    assert len(global_grid(lat_step=6, lon_step=8)) > len(global_grid())


def test_synthetic_is_deterministic_and_sized():
    src = SyntheticSource(n_series=5)
    a = list(src.harvest(None, CTX))
    b = list(src.harvest(None, CTX))
    assert len(a) == 5
    assert all(s.values.shape[0] == CTX.context_length + CTX.horizon for s in a)
    for x, y in zip(a, b, strict=True):
        assert np.array_equal(x.values, y.values)


# ── tsbench-forge (parquet mirror; needs the pool-forge extra) ────────────────

pytest = __import__("pytest")


def _write_forge_mirror(tmp_path, as_of):
    """A miniature forge mirror: catalog + dated parquet under data/<id>/."""
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    yaml = pytest.importorskip("yaml")

    catalog = [
        {  # hourly, panel-expanded into two series
            "id": "grid_load", "domain": "energy", "dgp_class": "load_curve",
            "frequency": "PT1H",
        },
        {  # daily, single series
            "id": "wiki_views", "domain": "web_traffic", "dgp_class": "attention",
            "frequency": "P1D",
        },
        {"id": "switched_off", "domain": "energy", "frequency": "PT1H", "disabled": True},
        {"id": "too_slow", "domain": "econ_fin", "frequency": "P1W"},
        {"id": "gone_stale", "domain": "nature", "frequency": "PT1H"},
    ]
    root = tmp_path / "forge"
    root.mkdir()
    (root / "sources.yaml").write_text(yaml.safe_dump(catalog), encoding="utf-8")

    def write(sid, day, df):
        d = root / "data" / sid
        d.mkdir(parents=True, exist_ok=True)
        df.to_parquet(d / f"{day}.parquet")

    hours = pd.date_range("2026-05-20", periods=400, freq="h", tz="UTC")
    write("grid_load", as_of.isoformat(), pd.DataFrame({
        "timestamp": list(hours.astype(str)) * 2,
        "load_mw": [float(i % 24) for i in range(400)] + [float((i * 3) % 24) for i in range(400)],
        "_panel_REGION": ["north"] * 400 + ["south"] * 400,
    }))
    days = pd.date_range("2025-06-01", periods=400, freq="D", tz="UTC")
    write("wiki_views", as_of.isoformat(), pd.DataFrame({
        "timestamp": days.astype(str),
        "views": [float(100 + (i % 7) * 10) for i in range(400)],
    }))
    # A future-dated snapshot must be ignored for a build pinned at as_of.
    write("wiki_views", (as_of + dt.timedelta(days=2)).isoformat(), pd.DataFrame({
        "timestamp": ["2026-09-01T00:00:00+00:00"], "views": [1e9],
    }))
    # Newest snapshot far older than max_stale_days before as_of → skipped.
    write("gone_stale", (as_of - dt.timedelta(days=30)).isoformat(), pd.DataFrame({
        "timestamp": hours.astype(str), "v": [1.0 * i for i in range(400)],
    }))
    write("switched_off", as_of.isoformat(), pd.DataFrame({
        "timestamp": hours.astype(str), "v": [2.0 * i for i in range(400)],
    }))
    return root


def test_tsbench_forge_reads_mirror_and_labels_clusters(tmp_path):
    from cascade.pool.sources.tsbench_forge import TsbenchForgeSource

    as_of = dt.date(2026, 6, 1)
    root = _write_forge_mirror(tmp_path, as_of)
    ctx = HarvestContext(as_of=as_of, context_length=512, horizon=16, max_series=100)
    out = list(TsbenchForgeSource(root).harvest(lambda u, p: None, ctx))

    ids = sorted(s.series_id for s in out)
    assert ids == [
        "tsforge__grid_load__REGION_north",
        "tsforge__grid_load__REGION_south",
        "tsforge__wiki_views",
    ]  # disabled, weekly, and stale sources are all skipped
    hourly = next(s for s in out if s.series_id.endswith("north"))
    assert hourly.freq == "H" and hourly.seasonal_period == 24
    assert hourly.domain == "energy" and hourly.source == "grid_load"
    daily = next(s for s in out if "wiki" in s.series_id)
    assert daily.freq == "D" and daily.seasonal_period == 7
    # The future-dated snapshot (and its absurd row) never entered the series.
    assert float(np.nanmax(daily.values)) < 1e6


def test_tsbench_forge_raises_when_everything_is_stale(tmp_path):
    from cascade.pool.source import HarvestError
    from cascade.pool.sources.tsbench_forge import TsbenchForgeSource

    as_of = dt.date(2026, 6, 1)
    root = _write_forge_mirror(tmp_path, as_of)
    late = dt.date(2026, 8, 1)  # far past every snapshot
    ctx = HarvestContext(as_of=late, context_length=512, horizon=16, max_series=100)
    with pytest.raises(HarvestError, match="no fresh tsbench-forge data"):
        list(TsbenchForgeSource(root).harvest(lambda u, p: None, ctx))


def test_tsbench_forge_respects_max_series(tmp_path):
    from cascade.pool.sources.tsbench_forge import TsbenchForgeSource

    as_of = dt.date(2026, 6, 1)
    root = _write_forge_mirror(tmp_path, as_of)
    ctx = HarvestContext(as_of=as_of, context_length=512, horizon=16, max_series=1)
    out = list(TsbenchForgeSource(root).harvest(lambda u, p: None, ctx))
    assert len(out) == 1


def test_resolve_frequency_derivation_matches_the_pinned_table():
    """The generic ISO-duration path must reproduce every canonical entry, so a
    catalog cadence dropping out of FREQ_MAP can never change behaviour."""
    from cascade.pool.sources.tsbench_forge import FREQ_MAP, _iso_duration_seconds

    for iso, (_, period) in FREQ_MAP.items():
        sec = int(_iso_duration_seconds(iso))
        if sec == 86_400:
            assert period == 7
        elif sec >= 300:
            assert period == max(2, round(86_400 / sec)), iso
        else:
            assert period == max(2, round(3_600 / sec)), iso


def test_resolve_frequency_handles_future_cadences():
    from cascade.pool.sources.tsbench_forge import resolve_frequency

    assert resolve_frequency("PT20M") == ("1200S", 72)     # new sub-daily feed
    assert resolve_frequency("PT2M") == ("120S", 30)       # faster than 5min → hourly cycle
    assert resolve_frequency("PT4H") == ("14400S", 6)
    assert resolve_frequency("PT1H30M") == ("5400S", 16)   # compound duration
    assert resolve_frequency("P1W") is None                # weekly+ still skipped
    assert resolve_frequency("P1M") is None
    assert resolve_frequency("fortnightly") is None        # unparseable
    assert resolve_frequency("PT1H") == ("H", 24)          # table stays canonical


def test_tsbench_forge_picks_up_a_new_cadence_source(tmp_path):
    """A catalog entry at a cadence FREQ_MAP has never seen must still be
    harvested — the future-proofing contract for a growing catalog."""
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    yaml = pytest.importorskip("yaml")
    from cascade.pool.sources.tsbench_forge import TsbenchForgeSource

    as_of = dt.date(2026, 6, 1)
    root = tmp_path / "forge"
    (root / "data" / "new_feed").mkdir(parents=True)
    (root / "sources.yaml").write_text(yaml.safe_dump([
        {"id": "new_feed", "domain": "transport", "frequency": "PT20M"},
    ]), encoding="utf-8")
    ts = pd.date_range("2026-05-25", periods=500, freq="20min", tz="UTC")
    pd.DataFrame({"timestamp": ts.astype(str),
                  "count": [float(i % 72) for i in range(500)]}).to_parquet(
        root / "data" / "new_feed" / f"{as_of}.parquet")

    ctx = HarvestContext(as_of=as_of, context_length=512, horizon=16, max_series=10)
    out = list(TsbenchForgeSource(root).harvest(lambda u, p: None, ctx))
    assert len(out) == 1
    assert out[0].freq == "1200S" and out[0].seasonal_period == 72
    assert out[0].source == "new_feed"
