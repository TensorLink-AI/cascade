"""F3 (plan rev6.18 A1/A2/A5/A6): tiny synthetic public-source mirrors,
shared by the W6p, W5k, W7a and W9 acceptance tests.

Two layouts, as the real sources store them (pyarrow writes them; nothing is
downloaded):

* ``write_gift`` — a ``Salesforce/GiftEvalPretrain``-style folder: one HF
  ``datasets`` arrow IPC stream ``data-00000-of-00001.arrow`` with columns
  ``item_id``, ``start``, ``freq``, ``target`` (``list<float32>``, or
  ``list<list<float32>>`` for a ``(C, L)`` multivariate dataset);
* ``write_rmisc`` — an ``nju-zhangsq/RMISC``-style folder: long-form
  ``part<k>.parquet`` files (one row per time step, one column per
  channel, ``_original_filename`` naming the series; columns absent from a
  series are all-null, as RMISC's file merge leaves them) plus
  ``meta.json``.

Values are deterministic functions of (dataset, series, channel, t), so a
test can recompute any point. Nothing here imports a module in PATHS.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

GIFT = "gift_eval_pretrain"
RMISC = "rmisc"
BLOCK_HASH = "0x" + "ab" * 32
OTHER_BLOCK_HASH = "0x" + "cd" * 32


def value(dataset: str, series: int, channel: int, t) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64)
    k = sum(map(ord, dataset)) % 17
    return (10.0 + series + 3.0 * channel + k
            + np.sin(2 * np.pi * t / (12 + channel)) * (1 + 0.1 * series) + 0.01 * t)


def series_values(dataset: str, series: int, length: int, channels: int = 1) -> np.ndarray:
    t = np.arange(length)
    return np.stack([value(dataset, series, c, t) for c in range(channels)])


def write_gift(root: Path, dataset: str, lengths: list[int], *, freq: str = "H",
               channels: int = 1, nan_at: dict | None = None) -> Path:
    """One GiftEvalPretrain-style folder; ``nan_at`` = {series: [(channel, t)]}."""
    import pyarrow as pa

    d = Path(root) / dataset
    d.mkdir(parents=True, exist_ok=True)
    targets = []
    for i, L in enumerate(lengths):
        a = series_values(dataset, i, L, channels).astype(np.float32)
        for c, t in (nan_at or {}).get(i, ()):
            a[c, t] = np.nan
        targets.append(a[0].tolist() if channels == 1 else a.tolist())
    ty = pa.list_(pa.float32()) if channels == 1 else pa.list_(pa.list_(pa.float32()))
    t = pa.table({"item_id": pa.array([f"{dataset}_{i}" for i in range(len(lengths))]),
                  "start": pa.array([0] * len(lengths), type=pa.timestamp("s")),
                  "freq": pa.array([freq] * len(lengths)),
                  "target": pa.array(targets, type=ty)})
    path = d / "data-00000-of-00001.arrow"
    with pa.OSFile(str(path), "wb") as sink, pa.ipc.new_stream(sink, t.schema) as w:
        w.write_table(t, max_chunksize=max(1, len(lengths) // 2 or 1))
    (d / "dataset_info.json").write_text(json.dumps({"dataset_name": dataset}), encoding="utf-8")
    return d


def write_rmisc(root: Path, dataset: str, series: dict[str, tuple[int, int]], *,
                freq: str | None = "h", licence: str = "CC BY 4.0", parts: int = 1,
                row_group_size: int = 50, empty: dict | None = None,
                domain: str = "Energy", desp: str = "Synthetic sensor readings.") -> Path:
    """One RMISC-style folder. ``series`` = {name: (length, channels)}; each
    series uses columns ``c0 … c<channels-1>`` of the merged schema (the
    widest series sets it; narrower ones leave the rest null). ``empty`` =
    {name: [channel]} makes those columns all-null in that series."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    d = Path(root) / dataset
    d.mkdir(parents=True, exist_ok=True)
    width = max(c for _, c in series.values())
    names = sorted(series)
    chunks = [names[i::parts] for i in range(parts)]
    for k, chunk in enumerate(chunks):
        cols = {f"c{j}": [] for j in range(width)}
        ts, fname = [], []
        for name in chunk:
            L, C = series[name]
            i = names.index(name)
            a = series_values(dataset, i, L, C)
            for j in range(width):
                if j < C and j not in (empty or {}).get(name, ()):
                    cols[f"c{j}"] += a[j].tolist()
                else:
                    cols[f"c{j}"] += [None] * L
            ts += list(range(L))
            fname += [name] * L
        t = pa.table({"timestamp": pa.array(ts, type=pa.int64()),
                      **{c: pa.array(v, type=pa.float64()) for c, v in cols.items()},
                      "_original_filename": pa.array(fname)})
        pq.write_table(t, str(d / f"part{k}.parquet"), row_group_size=row_group_size)
    meta = {"name": dataset, "source": "https://example.org/" + dataset, "desp": desp,
            "domain": domain, "license": licence, "targets": [], "covariates": [],
            "timestamp": "timestamp", "freq": None if freq is None else [freq],
            "num_series": len(series)}
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return d


def gift_mirror(tmp: Path) -> tuple[Path, list[str]]:
    """A small GiftEvalPretrain mirror: two climate year folders (one family),
    a scarce hourly folder, a daily folder and a multivariate folder."""
    root = Path(tmp) / "gift"
    write_gift(root, "cmip6_1850", [400] * 6, freq="6H")
    write_gift(root, "cmip6_1855", [400] * 6, freq="6H")
    write_gift(root, "era5_1999", [500] * 5, freq="H")
    write_gift(root, "scarce_hourly", [300, 300], freq="H")
    write_gift(root, "daily_sales", [260 + 10 * i for i in range(12)], freq="D")
    write_gift(root, "multi_traffic", [300] * 4, freq="H", channels=3)
    write_gift(root, "too_short", [100, 150], freq="H")
    return root, sorted(p.name for p in root.iterdir())


def rmisc_mirror(tmp: Path) -> tuple[Path, list[str]]:
    """A small RMISC mirror: a folder of few long series (several windows
    per series), a wide multichannel folder with an empty column, and a
    two-part folder."""
    root = Path(tmp) / "rmisc"
    write_rmisc(root, "FewLong", {"s0": (2000, 2), "s1": (1800, 2)}, freq="h")
    write_rmisc(root, "WideSensors", {f"w{i}": (400, 40) for i in range(3)}, freq="10min",
                empty={"w1": [5, 6]})
    write_rmisc(root, "TwoPart", {f"p{i}": (300 + 20 * i, 3) for i in range(6)}, freq="D",
                parts=2, row_group_size=40)
    return root, ["FewLong", "TwoPart", "WideSensors"]


def build(tmp: Path, source: str) -> dict:
    """Mirror + index of ``source`` under ``tmp`` (the index built by the
    code under test, imported lazily)."""
    from cascade.eval.lane_d_public_index import build_index

    if source == GIFT:
        root, ds = gift_mirror(tmp)
    else:
        root, ds = rmisc_mirror(tmp)
    index = Path(tmp) / f"index_{source}"
    manifest = build_index(root, source=source, datasets=ds, out_dir=index)
    return {"root": root, "datasets": ds, "index": index, "manifest": manifest}
