"""Shared fixtures for the W6a tests (README.md has the contract).

Nothing here imports ``cascade.pool.pool_d_select``: each test imports it
first, so on main every test fails at that import (a module in PATHS).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np

from cascade.pool.builder import PoolBuildConfig
from cascade.pool.source import HarvestContext, HarvestedSeries

AS_OF = dt.date(2026, 11, 20)
CTX = HarvestContext(as_of=AS_OF, context_length=256, horizon=32, max_series=10_000)


def select_mod():
    return importlib.import_module("cascade.pool.pool_d_select")


def lane_a_cfg(**over) -> PoolBuildConfig:
    base = dict(context_length=256, horizon=32, min_context=64, cluster_by="source")
    base.update(over)
    return PoolBuildConfig(**base)


def wave(n: int, seed: int, period: int = 24, channels: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64)
    rows = [10.0 + seed + 3.0 * np.sin(2 * np.pi * (t + c) / period) + rng.normal(0, 0.3, n)
            for c in range(channels)]
    return rows[0] if channels == 1 else np.stack(rows, axis=0)


def hs(series_id: str, values, *, source: str | None, freq: str = "H",
       domain: str = "energy", attrs: dict | None = None) -> HarvestedSeries:
    return HarvestedSeries(series_id=series_id, values=np.asarray(values), freq=freq,
                           domain=domain, source=source, attrs=dict(attrs or {}))


class ListSource:
    """A DataSource over fixed series; counts how often it is harvested."""

    def __init__(self, name: str, items: list[HarvestedSeries]):
        self.name = name
        self.items = list(items)
        self.calls = 0

    def harvest(self, fetch, ctx):
        self.calls += 1
        yield from self.items


def no_fetch(url, params=None):
    raise AssertionError(f"no network in W6a tests: {url}")


def tree_sha(directory: Path) -> dict[str, str]:
    """``{relative path: sha256}`` of a pool directory. ``provenance.json`` is
    compared without ``summary.out_dir`` (the directory name it was written
    to); every other byte must match."""
    out = {}
    for p in sorted(Path(directory).rglob("*")):
        if not p.is_file():
            continue
        data = p.read_bytes()
        if p.name == "provenance.json":
            doc = json.loads(data)
            doc.get("summary", {}).pop("out_dir", None)
            data = json.dumps(doc, sort_keys=True).encode()
        out[p.relative_to(directory).as_posix()] = hashlib.sha256(data).hexdigest()
    return out


def standard_sources() -> list[ListSource]:
    """Two sources: valid series, one degenerate, one too short, one that the
    domain cap drops, and a cross-source sanitised-id collision."""
    grid = ListSource("grid", [
        hs("grid load/r1", wave(600, 1), source="grid"),
        hs("grid load/r2", wave(600, 2), source="grid"),
        hs("st a", wave(600, 3), source="grid"),
        hs("flat", np.full(600, 5.0), source="grid"),            # degenerate
        hs("short", wave(40, 4), source="grid"),                  # too short
    ])
    weather = ListSource("weather", [
        hs("st_a", wave(600, 5), source="weather", domain="weather"),
        hs("tokyo", wave(600, 6), source="weather", domain="weather"),
        hs("osaka", wave(600, 7), source="weather", domain="weather"),
    ])
    return [grid, weather]


# ── forge mirror (the real TsbenchForgeSource contract) ─────────────────────

def forge_mirror(root: Path) -> Path:
    """``sources.yaml`` + ``data/<feed>/<YYYY-MM-DD>.parquet``: a paneled
    5-minute feed (two regions), an unpaneled hourly feed and a paneled feed
    whose two panel values sanitise to the same key (``"a b"``, ``"a_b"``)."""
    import pandas as pd
    import yaml

    root.mkdir(parents=True, exist_ok=True)
    catalog = [
        {"id": "grid_load", "domain": "energy", "frequency": "PT5M",
         "source_family": "fixture:grid", "dgp_class": "load"},
        {"id": "air_temp", "domain": "weather", "frequency": "PT1H",
         "source_family": "fixture:air"},
        {"id": "dup_panel", "domain": "transport", "frequency": "PT1H",
         "source_family": "fixture:dup"},
    ]
    (root / "sources.yaml").write_text(yaml.safe_dump(catalog), encoding="utf-8")
    end = pd.Timestamp(AS_OF, tz="UTC") + pd.Timedelta(hours=23)

    def write(feed: str, frame: dict) -> None:
        d = root / "data" / feed
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(frame).to_parquet(d / f"{AS_OF.isoformat()}.parquet")

    n5 = 900
    ts5 = pd.date_range(end=end, periods=n5, freq="5min")
    write("grid_load", {
        "timestamp": [str(t) for t in ts5] * 2,
        "_panel_region": ["r1"] * n5 + ["r2"] * n5,
        "mw": np.concatenate([wave(n5, 11), wave(n5, 12)]),
    })
    n1 = 700
    ts1 = pd.date_range(end=end, periods=n1, freq="h")
    write("air_temp", {"timestamp": [str(t) for t in ts1], "t": wave(n1, 13)})
    write("dup_panel", {
        "timestamp": [str(t) for t in ts1] * 2,
        "_panel_station": ["a b"] * n1 + ["a_b"] * n1,
        "flow": np.concatenate([wave(n1, 14), wave(n1, 15)]),
    })
    return root
