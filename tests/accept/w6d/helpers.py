"""Shared fixtures for the W6d tests (README.md has the contract).

The ``pool_d/1`` fixture is written by hand in exactly the shape the W6b
builder writes (``cascade/pool/pool_d_build.py``, tests/accept/w6b/README.md)
and passes its ``validate_pool_d``. Nothing here imports
``cascade.pool.pool_d_publish``: each test imports it first, so on main every
test fails at that import (a module in PATHS).
"""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import numpy as np

T0 = 1_795_132_800                 # 2026-11-20 00:00 UTC (3 h boundary)
HOUR = 3600
AS_OF = T0 + 3 * HOUR
HORIZON = 64


def pub_mod():
    return importlib.import_module("cascade.pool.pool_d_publish")


def series_values(sid: str, n: int) -> np.ndarray:
    rng = np.random.default_rng(sum(ord(c) * (i + 1) for i, c in enumerate(sid)))
    t = np.arange(n, dtype=np.float64)
    return (30.0 + 4.0 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 0.4, n)).astype(np.float32)


def series_record(sid: str, feed: str, hist_len: int, *, end: int = T0 - HOUR) -> dict:
    ts = [end - HOUR * (HORIZON - 1 - i) for i in range(HORIZON)]
    return {
        "series_id": sid, "pool_series_id": sid.split("~")[0], "source": f"curated:{feed}",
        "feed": feed, "panel": {}, "channel": 0, "column": "v", "freq": "H",
        "seasonal_period": 24, "domain": "energy", "history_len": hist_len,
        "target_ts": ts, "first_seen_ts": [T0 + 600] * HORIZON,
        "first_target_ts": ts[0], "end_ts": ts[-1], "frontier": [],
    }


SERIES = (("tsforge__meter~meter.c0", "meter", 4000),
          ("tsforge__grid__region_r1~grid.c0", "grid", 2500),
          ("tsforge__grid__region_r2~grid.c0", "grid", 300))


def write_pool_d(d: Path, *, snapshot: str | None = None, as_of: int = AS_OF,
                 series=SERIES, shift: float = 0.0) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    recs, hist, tgt = [], {}, {}
    for sid, feed, n in series:
        recs.append(series_record(sid, feed, n))
        full = series_values(sid, n + HORIZON) + np.float32(shift)
        hist[sid], tgt[sid] = full[:n], full[n:]
    feeds = {f: {"mirror_lag_hours": 0.5, "max_lead_hours": -0.5, "publish_ahead": False,
                 "forecast_product": False, "embargo_hours": 48.0}
             for f in sorted({r["feed"] for r in recs})}
    meta = {
        "schema": "pool_d/1", "snapshot": snapshot or f"pool_d-{as_of}", "as_of_ts": as_of,
        "horizon": HORIZON, "history_length": 16384, "builder_version": "cascade-pool-d/1",
        "catalog_sha256": hashlib.sha256(b"sources.yaml fixture").hexdigest(),
        "embargo_hours_global": 48.0, "selection_sha256": hashlib.sha256(b"sel").hexdigest(),
        "first_seen_roots": [hashlib.sha256(b"root").hexdigest()],
        "feeds": feeds, "series": sorted(recs, key=lambda r: r["series_id"]), "refused": [],
    }
    (d / "metadata.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    np.savez(d / "hist.npz", **hist)
    np.savez(d / "tgt.npz", **tgt)
    return d


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot_sha(d: Path) -> str:
    """The digest ``cascade.validator.lane_d.load_pool`` computes."""
    return hashlib.sha256("".join(sha(Path(d) / n) for n in
                                  ("metadata.json", "hist.npz", "tgt.npz")).encode()).hexdigest()


class Recording:
    """Wraps a pool store and records every key read or written."""

    def __init__(self, inner):
        self.inner = inner
        self.reads: list[str] = []
        self.writes: list[str] = []

    def put_file(self, key, path):
        self.writes.append(key)
        return self.inner.put_file(key, path)

    def get_file(self, key, path):
        self.reads.append(key)
        return self.inner.get_file(key, path)

    def put_bytes(self, key, data):
        self.writes.append(key)
        return self.inner.put_bytes(key, data)

    def get_bytes(self, key):
        self.reads.append(key)
        return self.inner.get_bytes(key)
