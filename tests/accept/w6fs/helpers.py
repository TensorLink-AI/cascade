"""Shared fixtures for the W6fs tests (README.md has the contract).

Nothing here imports ``cascade.pool.first_seen``: each test imports it first,
so on main every test fails at that import (a module in PATHS).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np

T0 = 1_795_132_800            # 2026-11-20 00:00 UTC, a 3 h period boundary
HOUR = 3600
PERIOD = 3 * HOUR


def fs_mod():
    return importlib.import_module("cascade.pool.first_seen")


def day_of(ts: int) -> str:
    return dt.datetime.fromtimestamp(int(ts), tz=dt.UTC).date().isoformat()


def iso(ts: int) -> str:
    return dt.datetime.fromtimestamp(int(ts), tz=dt.UTC).isoformat()


def catalog(root: Path, feeds: list[dict]) -> None:
    import yaml

    root.mkdir(parents=True, exist_ok=True)
    (root / "sources.yaml").write_text(yaml.safe_dump(feeds), encoding="utf-8")


def scrape(root: Path, feed: str, day: str, ts: list[int], values=None, *,
           panel: dict[str, list] | None = None, column: str = "v") -> Path:
    """Write ``data/<feed>/<day>.parquet`` (overwrites: a rewrite)."""
    import pandas as pd

    vals = np.asarray(values if values is not None else
                      [10.0 + np.sin(i / 3.0) for i in range(len(ts))], dtype=float)
    frame = {"timestamp": [iso(t) for t in ts], column: vals}
    for k, v in (panel or {}).items():
        frame[f"_panel_{k}"] = list(v)
    d = root / "data" / feed
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{day}.parquet"
    pd.DataFrame(frame).to_parquet(path)
    return path


def hourly(start: int, n: int) -> list[int]:
    return [int(start + i * HOUR) for i in range(n)]


def canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def ref_merkle(leaves: list[bytes]) -> str:
    """Independent implementation of the README Merkle rule."""
    level = [hashlib.sha256(b"\x00" + leaf).digest() for leaf in leaves]
    while len(level) > 1:
        nxt = []
        for i in range(0, len(level), 2):
            if i + 1 < len(level):
                nxt.append(hashlib.sha256(b"\x01" + level[i] + level[i + 1]).digest())
            else:
                nxt.append(level[i])
        level = nxt
    return level[0].hex()


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