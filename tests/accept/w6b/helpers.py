"""Shared fixtures for the W6b tests (README.md has the contract).

Nothing here imports ``cascade.pool.pool_d_build`` (W6b), ``pool_d_select``
(W6a) or ``first_seen`` (W6fs) at module level: :func:`mods` imports W6b's
module FIRST, so on main every test fails at that import (a module in PATHS).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np

from cascade.pool.builder import PoolBuildConfig
from cascade.pool.source import HarvestContext

T0 = 1_795_132_800                 # 2026-11-20 00:00 UTC (3 h boundary)
HOUR = 3600
PERIOD = 3 * HOUR
FIVE_MIN = 300
AS1 = T0 + PERIOD                  # first snapshot
AS2 = T0 + 2 * PERIOD              # second snapshot (after a second scrape)
LANE_A_AS_OF = dt.date(2026, 11, 19)
LANE_A_CFG = PoolBuildConfig(context_length=512, horizon=16, min_context=32,
                             cluster_by="source_family")
CTX_A = HarvestContext(as_of=LANE_A_AS_OF, context_length=512, horizon=16, max_series=10_000)
N_GRID = 17_000                    # > 16384 + 64
N_AIR = 3_000                      # < 16384 + 64
N_HOURLY = 600


def mods():
    """``(pool_d_build, pool_d_select, first_seen)``: W6b's module first."""
    b = importlib.import_module("cascade.pool.pool_d_build")
    sel = importlib.import_module("cascade.pool.pool_d_select")
    fs = importlib.import_module("cascade.pool.first_seen")
    return b, sel, fs


def iso(ts: int) -> str:
    return dt.datetime.fromtimestamp(int(ts), tz=dt.UTC).isoformat()


def no_fetch(url, params=None):
    raise AssertionError(f"no network in W6b tests: {url}")


def signal(n: int, seed: int, period: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64)
    return 50.0 + seed + 8.0 * np.sin(2 * np.pi * t / period) + rng.normal(0, 0.5, n)


def grid_ts(n: int = N_GRID) -> list[int]:
    return [T0 - FIVE_MIN * (n - i) for i in range(n)]


def hourly_ts(n: int, end: int = T0) -> list[int]:
    return [end - HOUR * (n - i) for i in range(n)]


GRID_EXTRA = 24                    # 5-min points T0 .. T0 + 115 min (second scrape)

CATALOG = [
    {"id": "grid_load", "domain": "energy", "frequency": "PT5M",
     "source_family": "fixture:grid", "dgp_class": "load"},
    {"id": "air_temp", "domain": "weather", "frequency": "PT1H",
     "source_family": "fixture:air", "embargo_hours": 12},
    {"id": "fx_fwd", "domain": "econ", "frequency": "PT1H",
     "source_family": "fixture:fx", "forecast_product": True},
    {"id": "capped", "domain": "econ", "frequency": "PT1H",
     "source_family": "fixture:capped", "embargo_hours": 100},
    {"id": "bad_emb", "domain": "sales", "frequency": "PT1H",
     "source_family": "fixture:bad", "embargo_hours": -1},
    {"id": "ahead", "domain": "nature", "frequency": "PT1H",
     "source_family": "fixture:ahead"},
    {"id": "gappy", "domain": "transport", "frequency": "PT1H",
     "source_family": "fixture:gappy"},
    {"id": "roll", "domain": "web", "frequency": "PT1H",
     "source_family": "fixture:roll", "rollup": "mean"},
]


def grid_values(region: str, n: int = N_GRID + GRID_EXTRA) -> np.ndarray:
    return signal(n, 1 if region == "r1" else 2, 288.0)


def hourly_values(feed: str, n: int) -> np.ndarray:
    seed = sum(ord(c) for c in feed)
    v = signal(n, seed % 50, 24.0)
    if feed == "gappy":
        v[-3:] = np.nan                    # a gap inside the last horizon points
    return v


def _write(root: Path, feed: str, day: str, frame: dict) -> None:
    import pandas as pd

    d = root / "data" / feed
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(frame).to_parquet(d / f"{day}.parquet")


def forge_mirror(root: Path, catalog: list[dict] | None = None) -> Path:
    import yaml

    root.mkdir(parents=True, exist_ok=True)
    (root / "sources.yaml").write_text(yaml.safe_dump(catalog or CATALOG), encoding="utf-8")
    day = LANE_A_AS_OF.isoformat()
    ts = grid_ts()
    _write(root, "grid_load", day, {
        "timestamp": [iso(t) for t in ts] * 2,
        "_panel_region": ["r1"] * N_GRID + ["r2"] * N_GRID,
        "mw": np.concatenate([grid_values("r1")[:N_GRID], grid_values("r2")[:N_GRID]]),
    })
    _write(root, "air_temp", day, {"timestamp": [iso(t) for t in hourly_ts(N_AIR)],
                                   "t": hourly_values("air_temp", N_AIR)})
    _write(root, "roll", day, {"timestamp": [iso(t) for t in hourly_ts(N_AIR)],
                               "r": hourly_values("roll", N_AIR)})
    for feed in ("fx_fwd", "capped", "bad_emb", "gappy"):
        _write(root, feed, day, {"timestamp": [iso(t) for t in hourly_ts(N_HOURLY)],
                                 "x": hourly_values(feed, N_HOURLY)})
    # future-stamped values (publish-ahead): data time up to T0 + 23 h
    _write(root, "ahead", day, {"timestamp": [iso(t) for t in hourly_ts(N_HOURLY, T0 + 24 * HOUR)],
                                "a": hourly_values("ahead", N_HOURLY)})
    return root


def second_scrape(root: Path) -> None:
    """24 newer 5-minute grid points per region, in the next day's partition."""
    ts = [T0 + FIVE_MIN * i for i in range(GRID_EXTRA)]
    _write(root, "grid_load", "2026-11-20", {
        "timestamp": [iso(t) for t in ts] * 2,
        "_panel_region": ["r1"] * GRID_EXTRA + ["r2"] * GRID_EXTRA,
        "mw": np.concatenate([grid_values("r1")[N_GRID:], grid_values("r2")[N_GRID:]]),
    })


def forge_source(root: Path):
    from cascade.pool.sources.tsbench_forge import TsbenchForgeSource

    return TsbenchForgeSource(root, max_series_per_source=5)


def build_world(base: Path) -> dict:
    """The full builder run twice: record + commit (W6fs), lane A + join
    (W6a), pool_d at AS1; a second scrape; pool_d at AS2 with AS1 prior."""
    b, sel, fs = mods()
    root = forge_mirror(base / "forge")
    log = fs.FirstSeenLog(root / "first_seen")
    log.record(root, now_ts=T0 + 600)
    log.commit(now_ts=AS1)
    _summary, selection = sel.build_pool_with_selection(
        [forge_source(root)], base / "pool", CTX_A, LANE_A_CFG, fetch=no_fetch)
    s1 = b.build_pool_d(forge_dir=root, selection=selection, first_seen_dir=root / "first_seen",
                        as_of_ts=AS1, out_dir=base / "d1")
    second_scrape(root)
    log.record(root, now_ts=AS1 + 600)
    # the AS1 rebuild after the second scrape must not see its points
    s1b = b.build_pool_d(forge_dir=root, selection=selection, first_seen_dir=root / "first_seen",
                         as_of_ts=AS1, out_dir=base / "d1b")
    log.commit(now_ts=AS2)
    s2 = b.build_pool_d(forge_dir=root, selection=selection, first_seen_dir=root / "first_seen",
                        as_of_ts=AS2, out_dir=base / "d2", prior_dirs=[base / "d1"])
    return {"root": root, "selection": selection, "s1": s1, "s1b": s1b, "s2": s2,
            "d1": base / "d1", "d1b": base / "d1b", "d2": base / "d2", "pool": base / "pool"}


def meta(d: Path) -> dict:
    return json.loads((Path(d) / "metadata.json").read_text(encoding="utf-8"))


def by_id(m: dict) -> dict:
    return {s["series_id"]: s for s in m["series"]}


def tree_sha(directory: Path) -> dict[str, str]:
    """``{relative path: sha256}``; ``provenance.json`` without its
    ``summary.out_dir`` (the directory name it was written to)."""
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


def canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
