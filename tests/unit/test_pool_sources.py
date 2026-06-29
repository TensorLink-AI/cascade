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
