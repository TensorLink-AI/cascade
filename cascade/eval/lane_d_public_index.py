"""Lane D public sources: the pinned sources, the per-dataset series index
and the series reader (LANE_D_PLAN rev6.18 A1, A5, A6; F3, W6p).

Two public sources make up lane D's public half, each mirrored from ONE
pinned Hugging Face revision (:data:`PUBLIC_SOURCES`):

- ``gift_eval_pretrain`` = ``Salesforce/GiftEvalPretrain`` (Apache-2.0): one
  folder per dataset, HF ``datasets`` arrow files, one row per series
  (``item_id``, ``start``, ``freq``, ``target``; ``target`` is ``(L,)`` or,
  for the 17 multivariate datasets, ``(C, L)``);
- ``rmisc`` = ``nju-zhangsq/RMISC``, the allowed folders only
  (``data/lane_d/rmisc_allowed.toml``, :func:`load_allowed_list`): one
  folder per dataset, ``part<k>.parquet`` files in long form (one row per
  time step, one column per channel) whose rows are grouped into series by
  ``_original_filename``, plus ``meta.json``.

**The index** (one per source; :func:`write_index` / :func:`iter_index`) has
one row per series::

    {"ordinal", "series_id", "dataset", "family", "freq", "length",
     "n_channels", "file", "rows"}

``ordinal`` is the row's line number (0-based), ``file`` the data file's
path relative to the source root, ``rows`` the ``[start, stop)`` row
ranges of the series in that file (GiftEvalPretrain: ``[[i, i + 1]]``),
``n_channels`` the channels holding at least one value. The index file is
``index.jsonl.gz``: canonical JSON lines (sorted keys, no spaces) in
``(dataset, file, first row)`` order; ``index_sha256`` is the sha256 of the
uncompressed bytes. ``manifest.json`` carries the source, repo, revision,
``index_sha256``, the mirrored file list (path, size, sha256) and
``files_sha256`` = sha256 of that list's canonical JSON. The signed lane D
report carries all three per source (W7a ``public``).

**Families** (A2): year-split folders merge into one family
(:func:`family_of`: ``cmip6_*``, ``era5_*``, ``largest_*``, any case, ``_``
or ``-`` separated); every other folder is its own family.

The index, hashes and allowed-list format are pure stdlib; building an index
and reading series need ``pyarrow`` (imported inside those functions: the
validator box draws from the index alone and never reads data).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import tomllib
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

INDEX_SCHEMA = "lane-d-public-index/1"
ALLOWED_SCHEMA = "lane-d-rmisc-allowed/1"
INDEX_FILE = "index.jsonl.gz"
MANIFEST_FILE = "manifest.json"
INDEX_FIELDS = ("ordinal", "series_id", "dataset", "family", "freq", "length", "n_channels",
                "file", "rows")
RMISC_SERIES_COLUMN = "_original_filename"
# A5 / A4: RMISC's duplicate folders, never allowed.
RMISC_DUPLICATE_MARKERS = ("KnownOpen", "KnownOil")
FAMILY_PREFIXES = ("cmip6", "era5", "largest")
UNKNOWN_FREQ = "unknown"
ALLOWED_LIST_PATH = Path(__file__).resolve().parents[2] / "data" / "lane_d" / "rmisc_allowed.toml"


@dataclass(frozen=True)
class PublicSource:
    """One public source: the HF repo, its ONE pinned revision, the data
    layout and whether a series may yield several windows (A2: RMISC's small
    folders with few long series)."""

    name: str
    repo: str
    revision: str
    layout: str
    multi_window: bool
    licence: str


PUBLIC_SOURCES: dict[str, PublicSource] = {
    "gift_eval_pretrain": PublicSource(
        name="gift_eval_pretrain", repo="Salesforce/GiftEvalPretrain",
        revision="6830b624de7ed2b3d3e5b85bb6959d81dcc5d874", layout="gift_arrow",
        multi_window=False, licence="apache-2.0"),
    "rmisc": PublicSource(
        name="rmisc", repo="nju-zhangsq/RMISC",
        revision="d0cdd935300ca9b3b2d6f23e6085db24cb7560c2", layout="rmisc_parquet",
        multi_window=True, licence="mit (per-folder licences in meta.json)"),
}


def family_of(dataset: str) -> str:
    """A2: ``cmip6_1850`` / ``CMIP6-2000-PartI`` -> ``cmip6``, ``era5_*`` ->
    ``era5``, ``largest_*`` -> ``largest``; any other folder is its own
    family (its name)."""
    low = str(dataset).lower()
    for p in FAMILY_PREFIXES:
        if low == p or low.startswith(p + "_") or low.startswith(p + "-"):
            return p
    return str(dataset)


def canonical_json(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _check_row(row: dict, ordinal: int) -> None:
    if set(row) != set(INDEX_FIELDS):
        raise ValueError(f"index row {ordinal}: fields {sorted(row)} != {sorted(INDEX_FIELDS)}")
    if row["ordinal"] != ordinal:
        raise ValueError(f"index row {ordinal}: ordinal {row['ordinal']!r}")
    for k in ("series_id", "dataset", "family", "freq", "file"):
        if not isinstance(row[k], str) or not row[k]:
            raise ValueError(f"index row {ordinal}: {k} must be a non-empty string")
    for k in ("length", "n_channels"):
        if not isinstance(row[k], int) or isinstance(row[k], bool) or row[k] < 0:
            raise ValueError(f"index row {ordinal}: {k} must be a non-negative integer")
    rows = row["rows"]
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"index row {ordinal}: rows must be a non-empty list")
    last = 0
    for r in rows:
        if (not isinstance(r, list) or len(r) != 2
                or not all(isinstance(x, int) and not isinstance(x, bool) for x in r)
                or not last <= r[0] < r[1]):
            raise ValueError(f"index row {ordinal}: bad row range {r!r}")
        last = r[1]


def index_lines(rows: Iterable[dict]) -> Iterator[bytes]:
    """Canonical index lines (``\\n``-terminated), ordinals checked."""
    for i, row in enumerate(rows):
        _check_row(row, i)
        yield canonical_json(row) + b"\n"


def index_sha256(rows: Iterable[dict]) -> str:
    h = hashlib.sha256()
    for line in index_lines(rows):
        h.update(line)
    return h.hexdigest()


def files_sha256(files: Iterable[dict]) -> str:
    """sha256 of the canonical JSON list of ``{path, size, sha256}``, by path."""
    items = sorted(({"path": str(f["path"]), "size": int(f["size"]), "sha256": str(f["sha256"])}
                    for f in files), key=lambda f: f["path"])
    return hashlib.sha256(canonical_json(items)).hexdigest()


def file_sha256(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def assign_ordinals(rows: Iterable[dict]) -> list[dict]:
    """Rows sorted in index order ``(dataset, file, first row)`` with their
    ``ordinal`` set."""
    ordered = sorted(rows, key=lambda r: (r["dataset"], r["file"], r["rows"][0][0]))
    return [{**r, "ordinal": i} for i, r in enumerate(ordered)]


def write_index(out_dir, *, source: str, rows: list[dict], files: list[dict]) -> dict:
    """Write ``index.jsonl.gz`` + ``manifest.json`` under ``out_dir``; return
    the manifest. ``rows`` must already carry their ordinals
    (:func:`assign_ordinals`)."""
    spec = PUBLIC_SOURCES[source]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    n = 0
    with gzip.GzipFile(out / INDEX_FILE, "wb", mtime=0) as gz:
        for line in index_lines(rows):
            gz.write(line)
            h.update(line)
            n += 1
    files = sorted(({"path": str(f["path"]), "size": int(f["size"]),
                     "sha256": str(f["sha256"])} for f in files), key=lambda f: f["path"])
    manifest = {"schema": INDEX_SCHEMA, "source": source, "repo": spec.repo,
                "revision": spec.revision, "n_series": n, "index_sha256": h.hexdigest(),
                "files": files, "files_sha256": files_sha256(files),
                "datasets": sorted({r["dataset"] for r in rows})}
    (out / MANIFEST_FILE).write_bytes(canonical_json(manifest) + b"\n")
    return manifest


def load_manifest(index_dir) -> dict:
    m = json.loads((Path(index_dir) / MANIFEST_FILE).read_text(encoding="utf-8"))
    if m.get("schema") != INDEX_SCHEMA:
        raise ValueError(f"{index_dir}: manifest schema {m.get('schema')!r}")
    spec = PUBLIC_SOURCES.get(m.get("source"))
    if spec is None or m.get("repo") != spec.repo or m.get("revision") != spec.revision:
        raise ValueError(f"{index_dir}: manifest source/repo/revision is not a pinned source")
    if files_sha256(m["files"]) != m.get("files_sha256"):
        raise ValueError(f"{index_dir}: files_sha256 does not match the file list")
    return m


def iter_index(index_dir) -> Iterator[dict]:
    """Stream the index rows (ordinals checked); at the end the bytes must
    hash to the manifest's ``index_sha256`` (``ValueError`` otherwise)."""
    m = load_manifest(index_dir)
    h = hashlib.sha256()
    n = 0
    with gzip.open(Path(index_dir) / INDEX_FILE, "rb") as gz:
        for line in gz:
            h.update(line)
            row = json.loads(line)
            _check_row(row, n)
            if canonical_json(row) + b"\n" != line:
                raise ValueError(f"{index_dir}: index line {n} is not canonical")
            n += 1
            yield row
    if h.hexdigest() != m["index_sha256"] or n != int(m["n_series"]):
        raise ValueError(f"{index_dir}: index bytes do not hash to the manifest's index_sha256")


# ---------------------------------------------------------------------------
# The RMISC allowed list (A5)
# ---------------------------------------------------------------------------

ALLOWED_STATUSES = ("placeholder", "final")
ALLOWED_FOLDER_FIELDS = ("name", "reason", "licence", "domain", "freq", "data_check")


def _toml_str(s: str) -> str:
    return json.dumps(str(s), ensure_ascii=False)


def render_allowed_list(*, status: str, folders: list[dict], generated_by: str,
                        revision: str | None = None) -> str:
    """The canonical TOML text of an allowed list (what the finalize step
    writes; :func:`load_allowed_list` reads it back)."""
    rev = PUBLIC_SOURCES["rmisc"].revision if revision is None else revision
    out = [
        "# Lane D RMISC allowed list (LANE_D_PLAN rev6.18 A5). Written by",
        "# scripts/laned_rmisc_allowlist.py finalize; do not hand-edit.",
        f"schema = {_toml_str(ALLOWED_SCHEMA)}",
        f"repo = {_toml_str(PUBLIC_SOURCES['rmisc'].repo)}",
        f"revision = {_toml_str(rev)}",
        f"status = {_toml_str(status)}",
        f"generated_by = {_toml_str(generated_by)}",
    ]
    for f in sorted(folders, key=lambda f: f["name"]):
        out += ["", "[[folder]]"]
        out += [f"{k} = {_toml_str(f[k])}" for k in ALLOWED_FOLDER_FIELDS]
    return "\n".join(out) + "\n"


def parse_allowed_list(text: str) -> dict:
    """Parse and validate an allowed list (``ValueError`` on any defect):
    the schema, the pinned RMISC repo and revision, ``status`` placeholder
    (no folder) or final (at least one); each folder carries every field as
    a non-empty string, names unique, sorted, never a KnownOpen / KnownOil
    duplicate folder, ``data_check`` = ``pass``."""
    doc = tomllib.loads(text)
    spec = PUBLIC_SOURCES["rmisc"]
    want = {"schema", "repo", "revision", "status", "generated_by"}
    extra = set(doc) - want - {"folder"}
    if extra:
        raise ValueError(f"allowed list: unknown keys {sorted(extra)}")
    if doc.get("schema") != ALLOWED_SCHEMA:
        raise ValueError(f"allowed list: schema {doc.get('schema')!r}")
    if doc.get("repo") != spec.repo or doc.get("revision") != spec.revision:
        raise ValueError("allowed list: repo/revision is not the pinned RMISC revision")
    status = doc.get("status")
    if status not in ALLOWED_STATUSES:
        raise ValueError(f"allowed list: status {status!r}")
    if not isinstance(doc.get("generated_by"), str):
        raise ValueError("allowed list: generated_by must be a string")
    folders = doc.get("folder", [])
    if not isinstance(folders, list):
        raise ValueError("allowed list: folder must be an array of tables")
    if status == "placeholder" and folders:
        raise ValueError("allowed list: a placeholder lists no folder")
    if status == "final" and not folders:
        raise ValueError("allowed list: a final list names at least one folder")
    names = []
    for i, f in enumerate(folders):
        if not isinstance(f, dict) or set(f) != set(ALLOWED_FOLDER_FIELDS):
            raise ValueError(f"allowed list: folder {i} fields must be {list(ALLOWED_FOLDER_FIELDS)}")
        for k in ALLOWED_FOLDER_FIELDS:
            if not isinstance(f[k], str) or not f[k].strip():
                raise ValueError(f"allowed list: folder {i} {k} must be a non-empty string")
        if any(m in f["name"] for m in RMISC_DUPLICATE_MARKERS):
            raise ValueError(f"allowed list: {f['name']} is a KnownOpen/KnownOil duplicate")
        if f["data_check"] != "pass":
            raise ValueError(f"allowed list: {f['name']} data_check is {f['data_check']!r}")
        names.append(f["name"])
    if len(set(names)) != len(names) or names != sorted(names):
        raise ValueError("allowed list: folder names must be unique and sorted")
    return {"status": status, "revision": doc["revision"], "folders": folders,
            "names": names, "generated_by": doc["generated_by"]}


def load_allowed_list(path=None) -> dict:
    return parse_allowed_list(Path(path or ALLOWED_LIST_PATH).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Building the index (pyarrow; run by the owner on a builder, never the box)
# ---------------------------------------------------------------------------

def _freq_str(freq) -> str:
    if isinstance(freq, list):
        freq = freq[0] if len(freq) == 1 else ",".join(str(x) for x in freq) if freq else None
    s = str(freq).strip() if freq is not None else ""
    return s or UNKNOWN_FREQ


def _open_arrow(path: Path):
    import pyarrow as pa

    src = pa.memory_map(str(path), "r")
    try:
        return pa.ipc.open_stream(src).read_all()
    except pa.ArrowInvalid:
        return pa.ipc.open_file(src).read_all()


def _list_shape(col) -> tuple[np.ndarray, np.ndarray]:
    """Per row ``(n_channels, length)`` of a ``target`` column: list<float>
    (one channel) or list<list<float>> (``C`` channels of equal length)."""
    import pyarrow as pa
    import pyarrow.compute as pc

    col = col.combine_chunks() if isinstance(col, pa.ChunkedArray) else col
    outer = pc.list_value_length(col).to_numpy(zero_copy_only=False).astype(np.int64)
    if not pa.types.is_list(col.type.value_type) and not pa.types.is_large_list(
            col.type.value_type):
        return np.ones_like(outer), outer
    inner = pc.list_value_length(pc.list_flatten(col)).to_numpy(
        zero_copy_only=False).astype(np.int64)
    lengths = np.zeros_like(outer)
    starts = np.concatenate([[0], np.cumsum(outer)[:-1]]).astype(np.int64)
    for i, (s, c) in enumerate(zip(starts, outer, strict=True)):
        if c == 0:
            continue
        seg = inner[s:s + c]
        if seg.min() != seg.max():
            raise ValueError(f"row {i}: channels of unequal length {sorted(set(seg.tolist()))}")
        lengths[i] = seg[0]
    return outer, lengths


def index_gift_dataset(root, dataset: str) -> tuple[list[dict], list[Path]]:
    """Index rows of one GiftEvalPretrain folder (ordinals unset) and the
    data files read."""
    base = Path(root) / dataset
    files = sorted(base.glob("*.arrow"))
    if not files:
        raise ValueError(f"{base}: no arrow files")
    rows = []
    for f in files:
        t = _open_arrow(f)
        ch, ln = _list_shape(t.column("target"))
        freqs = t.column("freq").to_pylist()
        rel = f"{dataset}/{f.name}"
        for i in range(t.num_rows):
            rows.append({"ordinal": -1, "series_id": f"{dataset}/{f.stem}#{i}",
                         "dataset": dataset, "family": family_of(dataset),
                         "freq": _freq_str(freqs[i]), "length": int(ln[i]),
                         "n_channels": int(ch[i]), "file": rel, "rows": [[i, i + 1]]})
    return rows, files


def _value_columns(schema, timestamp) -> list[str]:
    import pyarrow as pa

    out = []
    for field in schema:
        if field.name in (RMISC_SERIES_COLUMN, timestamp):
            continue
        t = field.type
        if pa.types.is_integer(t) or pa.types.is_floating(t) or pa.types.is_decimal(t):
            out.append(field.name)
    return out


def index_rmisc_dataset(root, dataset: str) -> tuple[list[dict], list[Path]]:
    """Index rows of one RMISC folder: series = runs of equal
    ``_original_filename`` per part file (a name split into several runs
    keeps every run in ``rows``); ``n_channels`` = value columns with at
    least one value in the series; ``freq`` from ``meta.json``."""
    import pyarrow.parquet as pq

    base = Path(root) / dataset
    meta = json.loads((base / "meta.json").read_text(encoding="utf-8"))
    freq = _freq_str(meta.get("freq"))
    ts_col = meta.get("timestamp")
    parts = sorted(base.glob("part*.parquet"), key=lambda p: int(re.sub(r"\D", "", p.stem) or 0))
    if not parts:
        raise ValueError(f"{base}: no part*.parquet files")
    rows = []
    for f in parts:
        pf = pq.ParquetFile(str(f))
        cols = _value_columns(pf.schema_arrow, ts_col)
        t = pf.read(columns=[RMISC_SERIES_COLUMN, *cols])
        names = t.column(RMISC_SERIES_COLUMN).to_pylist()
        valid = np.stack([t.column(c).is_valid().to_numpy(zero_copy_only=False)
                          for c in cols]) if cols else np.zeros((0, len(names)), dtype=bool)
        runs: dict[str, list[list[int]]] = {}
        start = 0
        for i in range(1, len(names) + 1):
            if i == len(names) or names[i] != names[start]:
                runs.setdefault(str(names[start]), []).append([start, i])
                start = i
        rel = f"{dataset}/{f.name}"
        for name, rr in runs.items():
            idx = np.concatenate([np.arange(a, b) for a, b in rr])
            n_ch = int(valid[:, idx].any(axis=1).sum()) if cols else 0
            rows.append({"ordinal": -1, "series_id": f"{dataset}/{name}", "dataset": dataset,
                         "family": family_of(dataset), "freq": freq, "length": int(idx.size),
                         "n_channels": n_ch, "file": rel, "rows": rr})
    return rows, parts


def build_index(data_root, *, source: str, datasets: list[str], out_dir) -> dict:
    """Index every dataset folder of a local mirror of ``source`` and write
    the index + manifest (file sha256s computed here) under ``out_dir``."""
    spec = PUBLIC_SOURCES[source]
    fn = index_gift_dataset if spec.layout == "gift_arrow" else index_rmisc_dataset
    rows, files = [], []
    root = Path(data_root)
    for ds in sorted(datasets):
        r, fs = fn(root, ds)
        rows += r
        files += fs
        extra = sorted(p for p in (root / ds).iterdir() if p.is_file() and p not in fs)
        files += extra
    recs = [{"path": p.relative_to(root).as_posix(), "size": p.stat().st_size,
             "sha256": file_sha256(p)} for p in sorted(set(files))]
    return write_index(out_dir, source=source, rows=assign_ordinals(rows), files=recs)


# ---------------------------------------------------------------------------
# Reading one series (only the drawn ones)
# ---------------------------------------------------------------------------

def read_series(data_root, row: dict, *, source: str) -> np.ndarray:
    """The series of one index row as a float64 ``(C, L)`` array (``C = 1``
    for a univariate series; nulls become NaN). Reads only that series'
    rows: one arrow row (memory-mapped) or the parquet row groups covering
    its ranges. RMISC channels are the folder's numeric value columns in
    schema order (the ``meta.json`` timestamp and ``_original_filename``
    excluded; columns empty in this series stay, all-NaN, for F2's
    empty-channel drop)."""
    path = Path(data_root) / row["file"]
    layout = PUBLIC_SOURCES[source].layout
    if layout == "gift_arrow":
        t = _open_arrow(path)
        (i, _j), = row["rows"]
        v = t.column("target").slice(i, 1).to_pylist()[0]
        a = np.asarray([np.nan if x is None else x for x in v] if v and not isinstance(v[0], list)
                       else [[np.nan if x is None else x for x in ch] for ch in v],
                       dtype=np.float64)
        return a.reshape(1, -1) if a.ndim == 1 else a
    import pyarrow as pa
    import pyarrow.parquet as pq

    meta = json.loads((Path(data_root) / row["dataset"] / "meta.json").read_text(encoding="utf-8"))
    pf = pq.ParquetFile(str(path))
    cols = _value_columns(pf.schema_arrow, meta.get("timestamp"))
    md = pf.metadata
    bounds = np.cumsum([0] + [md.row_group(g).num_rows for g in range(md.num_row_groups)])
    pieces = []
    for a, b in row["rows"]:
        groups = [g for g in range(md.num_row_groups) if bounds[g] < b and bounds[g + 1] > a]
        t = pf.read_row_groups(groups, columns=cols)
        off = int(bounds[groups[0]])
        pieces.append(t.slice(a - off, b - a))
    t = pa.concat_tables(pieces)
    out = np.empty((len(cols), t.num_rows), dtype=np.float64)
    for k, c in enumerate(cols):
        col = t.column(c).cast(pa.float64())
        out[k] = col.to_numpy(zero_copy_only=False).astype(np.float64)
    return out


__all__ = ["ALLOWED_LIST_PATH", "ALLOWED_SCHEMA", "INDEX_FIELDS", "INDEX_FILE", "INDEX_SCHEMA",
           "MANIFEST_FILE", "PUBLIC_SOURCES", "PublicSource", "assign_ordinals", "build_index",
           "canonical_json", "family_of", "file_sha256", "files_sha256", "index_gift_dataset",
           "index_lines", "index_rmisc_dataset", "index_sha256", "iter_index", "load_allowed_list",
           "load_manifest", "parse_allowed_list", "read_series", "render_allowed_list",
           "write_index"]
