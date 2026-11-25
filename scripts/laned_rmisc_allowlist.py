#!/usr/bin/env python3
"""Lane D RMISC allowed-list tooling (LANE_D_PLAN rev6.18 A5; F3, W6p).

The allowed list ``data/lane_d/rmisc_allowed.toml`` names the RMISC folders
lane D draws public windows from (and publishes as the official training
source). It is built in four steps; only the first two run anywhere, the
data compare runs on a builder / pod that holds the data (it is never run on
the validator box, and nothing here downloads data except ``download``):

1. ``fetch-listings --out listings.json``: Hugging Face METADATA only (repo
   tree listings at the pinned revisions and each RMISC folder's
   ``meta.json``; never a data file). ``HF_TOKEN`` is read from the
   environment when set (never printed).
2. ``candidates --listings listings.json --out candidates.json``: offline.
   RMISC folders whose normalised name matches no folder of the exclusion
   sources (:data:`EXCLUSION_SOURCES`: GIFT-Eval test, GiftEvalPretrain,
   fev_datasets, chronos_datasets, TIME @ 83e3d0b, BOOM @ 69325b5), minus
   the KnownOpen / KnownOil duplicates, non-forecasting sets (EEG/EMG/BCI,
   activity recognition, classification, anomaly; by name, ``meta.json``
   description, domain and source URL) and licences that forbid
   redistribution (``meta.json`` ``license``).
3. ``datacompare`` (OWNER, on a builder/pod; see ``download``): compares
   DATA, not names: winnowed K-gram hashes of quantised values from a
   deterministic sample of every candidate folder's series, looked up in
   the same hashes over every exclusion source's data. A folder whose hit
   fraction against any source reaches ``--dup-threshold`` is
   ``duplicate``.
4. ``finalize --candidates … --datacompare … --out data/lane_d/rmisc_allowed.toml``:
   the folders that passed every step (unknown licences only with
   ``--allow-unknown-licence``), written in the canonical TOML format
   (:func:`cascade.eval.lane_d_public_index.render_allowed_list`).
   ``validate [FILE]`` checks a list file.

Output formats (JSON, ``schema`` field first):

- ``lane-d-rmisc-listings/1``: ``{"sources": {key: {"repo", "revision",
  "names": [top-level folders]}}, "rmisc": {"repo", "revision", "folders":
  [names], "meta": {folder: meta.json object | null}}}``
- ``lane-d-rmisc-candidates/1``: ``{"rmisc": {"repo", "revision"},
  "exclusion_sources": [{"key", "repo", "revision", "n_names"}],
  "n_folders", "candidates": [{"name", "domain", "freq", "licence",
  "licence_class", "source_url", "near": [names]}], "excluded": [{"name",
  "reason", "detail"}]}`` (``reason`` one of :data:`EXCLUDE_REASONS`;
  ``licence_class`` ``ok`` | ``unknown``)
- ``lane-d-rmisc-datacompare/1``: ``{"params": {...}, "folders": [{"name",
  "n_series_sampled", "n_hashes", "hits": {source key: n}, "max_hit_fraction",
  "verdict": "pass" | "duplicate" | "no_data"}]}``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LISTINGS_SCHEMA = "lane-d-rmisc-listings/1"
CANDIDATES_SCHEMA = "lane-d-rmisc-candidates/1"
DATACOMPARE_SCHEMA = "lane-d-rmisc-datacompare/1"
RMISC_REPO = "nju-zhangsq/RMISC"
RMISC_REVISION = "d0cdd935300ca9b3b2d6f23e6085db24cb7560c2"
# key -> (repo, pinned revision). GiftEval / BOOM / TIME are the revisions
# the bench sidecar pins (benchmarks/cascade_benchmark/datasets.py); the
# others were the repos' heads when F3 pinned them.
EXCLUSION_SOURCES: dict[str, tuple[str, str]] = {
    "gift_eval": ("Salesforce/GiftEval", "30841734ac5cfddbd0c3bad6d09d2b6b32becbb0"),
    "gift_eval_pretrain": ("Salesforce/GiftEvalPretrain",
                           "6830b624de7ed2b3d3e5b85bb6959d81dcc5d874"),
    "fev_datasets": ("autogluon/fev_datasets", "d3e7e720f4158eab03ebcb021f1cce779cd92acc"),
    "chronos_datasets": ("autogluon/chronos_datasets",
                         "eeecad0b82a8c237e212ce6f8d1abecb513e2cec"),
    "time": ("Real-TSF/TIME", "83e3d0b3be28d11c7182bffcc1892d19b36c4da1"),
    "boom": ("Datadog/BOOM", "69325b544c45ff0d6c43c7a99c49a6601a01725b"),
}
EXCLUDE_REASONS = ("duplicate_folder", "name_match", "non_forecasting", "licence_forbids",
                   "no_meta")
DUPLICATE_MARKERS = ("KnownOpen", "KnownOil")
NON_FORECASTING = ("eeg", "emg", "electroencephalog", "electromyog", "brain-computer",
                   "brain computer", "bci", "motor imagery", "activity recognition",
                   "human activity", "accelerometer", "gyroscope", "gesture", "gait",
                   "classification", "anomal", "intrusion", "cybersecurity",
                   "localisation", "localization", "timeseriesclassification.com")
LICENCE_FORBIDS = ("no redistribution", "not be redistributed", "not redistribute",
                   "non-redistributable", "redistribution prohibited", "redistribution is not",
                   "proprietary", "all rights reserved", "not for redistribution")
LICENCE_UNKNOWN = ("", "unknown", "n/a", "na", "none", "unspecified", "null", "-")
HF = "https://huggingface.co"


# ── step 1: metadata listings ───────────────────────────────────────────────

def _get(url: str) -> tuple[bytes, dict]:
    req = urllib.request.Request(url, headers={"User-Agent": "cascade-laned-f3"})
    tok = os.environ.get("HF_TOKEN")
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
    with urllib.request.urlopen(req, timeout=60) as r:          # noqa: S310 (https only)
        return r.read(), dict(r.headers)


def hf_tree(repo: str, revision: str, path: str = "", recursive: bool = False) -> list[dict]:
    """Every entry of a dataset repo's tree at ``revision`` (metadata only,
    following the API's ``Link: next`` pagination)."""
    q = "?recursive=1" if recursive else ""
    url = f"{HF}/api/datasets/{repo}/tree/{revision}" + (f"/{urllib.parse.quote(path)}"
                                                        if path else "") + q
    out = []
    while url:
        body, headers = _get(url)
        out += json.loads(body)
        link = headers.get("Link") or headers.get("link") or ""
        m = re.search(r'<([^>]+)>;\s*rel="next"', link)
        url = m.group(1) if m else ""
    return out


def top_level_names(entries: list[dict]) -> list[str]:
    return sorted({e["path"].split("/", 1)[0] for e in entries
                   if e.get("type") == "directory" or "/" in e.get("path", "")})


def cmd_fetch_listings(args) -> int:
    sources = {}
    for key, (repo, rev) in EXCLUSION_SOURCES.items():
        names = top_level_names(hf_tree(repo, rev))
        sources[key] = {"repo": repo, "revision": rev, "names": names}
        print(f"listed {repo}@{rev[:7]}: {len(names)} folders", flush=True)
    folders = top_level_names(hf_tree(RMISC_REPO, RMISC_REVISION))
    meta = {}
    for f in folders:
        url = f"{HF}/datasets/{RMISC_REPO}/resolve/{RMISC_REVISION}/{urllib.parse.quote(f)}/meta.json"
        try:
            meta[f] = json.loads(_get(url)[0])
        except Exception as e:                                   # noqa: BLE001
            print(f"no meta.json for {f}: {type(e).__name__}", flush=True)
            meta[f] = None
    doc = {"schema": LISTINGS_SCHEMA, "sources": sources,
           "rmisc": {"repo": RMISC_REPO, "revision": RMISC_REVISION, "folders": folders,
                     "meta": meta}}
    Path(args.out).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.out}: {len(folders)} RMISC folders")
    return 0


# ── step 2: candidates (offline) ────────────────────────────────────────────

def norm(name: str) -> str:
    n = re.sub(r"[^a-z0-9]", "", str(name).lower())
    return n.replace("withmissing", "")


def tokens(name: str) -> list[str]:
    """Lower-case word tokens: split on non-alphanumerics and camelCase."""
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(name))
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)
    return [t for t in re.split(r"[^A-Za-z0-9]+", s.lower()) if t and t != "with"
            and t != "missing"]


def name_matches(name: str, others: list[str]) -> tuple[list[str], list[str]]:
    """``(matches, near)``: a match is an equal normalised name, a prefix
    either way of the normalised names (the shorter at least 6 characters)
    or a token prefix either way (``M4`` ~ ``m4_daily``, ``Covid`` ~
    ``covid_deaths``); ``near`` (recorded for the data compare, not
    excluded): a normalised substring (at least 6) or the same first token
    (at least 3 characters)."""
    a, ta = norm(name), tokens(name)
    match, near = [], []
    for o in others:
        b, tb = norm(o), tokens(o)
        if not a or not b:
            continue
        short = min(len(a), len(b))
        k = min(len(ta), len(tb))
        if (a == b or (short >= 6 and (a.startswith(b) or b.startswith(a)))
                or (k >= 1 and ta[:k] == tb[:k])):
            match.append(o)
        elif (short >= 6 and (a in b or b in a)) or (ta and tb and ta[0] == tb[0]
                                                    and len(ta[0]) >= 3):
            near.append(o)
    return match, near


def non_forecasting(name: str, meta: dict) -> str | None:
    text = " ".join(str(x) for x in (name, meta.get("desp", ""), meta.get("domain", ""),
                                     meta.get("source", ""), meta.get("name", ""))).lower()
    words = re.sub(r"[_\-/.]", " ", text)
    for kw in NON_FORECASTING:
        if len(kw) <= 3:
            if re.search(rf"\b{re.escape(kw)}\b", words):
                return kw
        elif kw in text:
            return kw
    return None


def licence_class(lic) -> str:
    s = str(lic or "").strip().lower()
    if s in LICENCE_UNKNOWN:
        return "unknown"
    if any(p in s for p in LICENCE_FORBIDS):
        return "forbids"
    return "ok"


def candidates(listings: dict) -> dict:
    if listings.get("schema") != LISTINGS_SCHEMA:
        raise SystemExit(f"listings schema {listings.get('schema')!r}")
    rm = listings["rmisc"]
    if (rm["repo"], rm["revision"]) != (RMISC_REPO, RMISC_REVISION):
        raise SystemExit("listings are not of the pinned RMISC revision")
    srcs = listings["sources"]
    for key, (repo, rev) in EXCLUSION_SOURCES.items():
        s = srcs.get(key)
        if not s or (s["repo"], s["revision"]) != (repo, rev):
            raise SystemExit(f"listings lack the pinned {repo}@{rev}")
    from cascade.eval.lane_d_public_index import FAMILY_PREFIXES, family_of

    all_names = [(k, n) for k in sorted(srcs) for n in srcs[k]["names"]]
    keep, out = [], []
    for f in sorted(rm["folders"]):
        meta = rm["meta"].get(f)
        if any(m in f for m in DUPLICATE_MARKERS):
            out.append({"name": f, "reason": "duplicate_folder", "detail": "KnownOpen/KnownOil"})
            continue
        fam = family_of(f)
        same = [f"{k}/{n}" for k, n in all_names if fam in FAMILY_PREFIXES and family_of(n) == fam]
        if same:
            # A year-split family (cmip6 / era5 / largest) already in a source.
            out.append({"name": f, "reason": "name_match",
                        "detail": f"family {fam}: " + ", ".join(same[:3])})
            continue
        match, near = [], []
        for k, n in all_names:
            m, nr = name_matches(f, [n])
            match += [f"{k}/{x}" for x in m]
            near += [f"{k}/{x}" for x in nr]
        if match:
            out.append({"name": f, "reason": "name_match", "detail": ", ".join(match[:5])})
            continue
        if not isinstance(meta, dict):
            out.append({"name": f, "reason": "no_meta", "detail": "meta.json missing"})
            continue
        kw = non_forecasting(f, meta)
        if kw:
            out.append({"name": f, "reason": "non_forecasting", "detail": kw})
            continue
        lc = licence_class(meta.get("license"))
        if lc == "forbids":
            out.append({"name": f, "reason": "licence_forbids", "detail": str(meta.get("license"))})
            continue
        freq = meta.get("freq")
        keep.append({"name": f, "domain": str(meta.get("domain") or ""),
                     "freq": ",".join(map(str, freq)) if isinstance(freq, list) else str(freq or ""),
                     "licence": str(meta.get("license") or ""), "licence_class": lc,
                     "source_url": str(meta.get("source") or ""), "near": near})
    return {"schema": CANDIDATES_SCHEMA, "rmisc": {"repo": RMISC_REPO, "revision": RMISC_REVISION},
            "exclusion_sources": [{"key": k, "repo": r, "revision": v,
                                   "n_names": len(srcs[k]["names"])}
                                  for k, (r, v) in EXCLUSION_SOURCES.items()],
            "n_folders": len(rm["folders"]), "candidates": keep, "excluded": out}


def cmd_candidates(args) -> int:
    doc = candidates(json.loads(Path(args.listings).read_text(encoding="utf-8")))
    Path(args.out).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    by = {}
    for e in doc["excluded"]:
        by[e["reason"]] = by.get(e["reason"], 0) + 1
    print(f"CANDIDATES {len(doc['candidates'])} of {doc['n_folders']} RMISC folders; excluded "
          + ", ".join(f"{k}={v}" for k, v in sorted(by.items())))
    return 0


# ── step 3: data compare (owner, on a builder/pod) ──────────────────────────

def download_command(key: str, local_dir: str) -> list[str]:
    """The download the owner runs on the pod before ``datacompare``."""
    repo, rev = (RMISC_REPO, RMISC_REVISION) if key == "rmisc" else EXCLUSION_SOURCES[key]
    return ["hf", "download", repo, "--repo-type", "dataset", "--revision", rev,
            "--local-dir", local_dir]


def cmd_download(args) -> int:
    from huggingface_hub import snapshot_download

    repo, rev = (RMISC_REPO, RMISC_REVISION) if args.source == "rmisc" \
        else EXCLUSION_SOURCES[args.source]
    pats = None
    if args.source == "rmisc" and args.candidates:
        c = json.loads(Path(args.candidates).read_text(encoding="utf-8"))
        pats = [f"{x['name']}/*" for x in c["candidates"]]
    snapshot_download(repo_id=repo, repo_type="dataset", revision=rev, local_dir=args.local_dir,
                      allow_patterns=pats)
    print(f"downloaded {repo}@{rev} -> {args.local_dir}")
    return 0


def _quantise(v):
    import numpy as np

    a = np.asarray(v, dtype=np.float32)
    bits = a.view(np.int32) & np.int32(-256)            # ~4.5 significant digits
    return bits.astype(np.int64), np.isfinite(a)


def kgram_hashes(values, k: int, mod: int):
    """Winnowed hashes of every K-gram of finite quantised values: the
    polynomial hash mod 2^64, kept iff ``hash % mod == 0``."""
    import numpy as np

    if np.asarray(values).size < k:
        return np.empty(0, dtype=np.uint64)
    h, okw = _window_hashes(values, k)
    h = h[okw]
    return np.unique(h[h % np.uint64(mod) == 0])


def _window_hashes(values, k: int):
    """``(hash, finite)`` of every K-gram window of ``values`` (``size - k +
    1`` windows; ``size >= k``): the one hash :func:`kgram_hashes` and the
    vectorised exclusion scan (:func:`file_hits`) share."""
    import numpy as np

    q, ok = _quantise(values)
    mult = (np.arange(1, k + 1, dtype=np.uint64) * np.uint64(0x9E3779B97F4A7C15)) | np.uint64(1)
    win = np.lib.stride_tricks.sliding_window_view(q.astype(np.uint64), k)
    okw = np.lib.stride_tricks.sliding_window_view(ok, k).all(axis=1)
    with np.errstate(over="ignore"):
        h = (win * mult).sum(axis=1, dtype=np.uint64)
        h ^= h >> np.uint64(29)
        h *= np.uint64(0xBF58476D1CE4E5B9)
        h ^= h >> np.uint64(32)
    return h, okw


def iter_channels(path: Path, max_points: int):
    """Every numeric sequence of one data file (arrow / parquet / csv): list
    columns row by row (one per channel), scalar numeric columns whole."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc

    suf = path.suffix.lower()
    if suf == ".arrow":
        src = pa.memory_map(str(path), "r")
        try:
            t = pa.ipc.open_stream(src).read_all()
        except pa.ArrowInvalid:
            t = pa.ipc.open_file(src).read_all()
    elif suf == ".parquet":
        import pyarrow.parquet as pq
        t = pq.read_table(str(path))
    elif suf == ".csv":
        import pyarrow.csv as pcsv
        t = pcsv.read_csv(str(path))
    else:
        return
    for name in t.column_names:
        col = t.column(name)
        ty = col.type
        if pa.types.is_list(ty) or pa.types.is_large_list(ty):
            for v in col.to_pylist():
                if not v:
                    continue
                chans = v if isinstance(v[0], list) else [v]
                for ch in chans:
                    yield np.asarray([np.nan if x is None else x for x in ch[:max_points]],
                                     dtype=np.float64)
        elif pa.types.is_integer(ty) or pa.types.is_floating(ty):
            arr = pc.cast(col, pa.float64()).to_numpy(zero_copy_only=False)
            for i in range(0, min(arr.size, max_points * 64), max_points):
                yield arr[i:i + max_points]


def _batches(path: Path, columns=None):
    """The record batches of one data file (arrow stream / arrow file /
    parquet / csv), so a multi-GB file is never materialised whole."""
    import pyarrow as pa

    suf = path.suffix.lower()
    if suf == ".arrow":
        src = pa.memory_map(str(path), "r")
        try:
            rd = pa.ipc.open_stream(src)
        except pa.ArrowInvalid:
            f = pa.ipc.open_file(src)
            for i in range(f.num_record_batches):
                b = f.get_batch(i)
                yield b.select(columns) if columns is not None else b
            return
        for b in rd:
            yield b.select(columns) if columns is not None else b
    elif suf == ".parquet":
        import pyarrow.parquet as pq
        yield from pq.ParquetFile(str(path)).iter_batches(batch_size=1024, columns=columns)
    elif suf == ".csv":
        import pyarrow.csv as pcsv
        t = pcsv.read_csv(str(path))
        if columns is not None:
            t = t.select(columns)
        yield from t.to_batches()


def _schema(path: Path):
    import pyarrow as pa

    suf = path.suffix.lower()
    if suf == ".arrow":
        src = pa.memory_map(str(path), "r")
        try:
            return pa.ipc.open_stream(src).schema
        except pa.ArrowInvalid:
            return pa.ipc.open_file(src).schema
    if suf == ".parquet":
        import pyarrow.parquet as pq
        return pq.read_schema(str(path))
    if suf == ".csv":
        import pyarrow.csv as pcsv
        return pcsv.read_csv(str(path)).schema
    return None


def _is_num(ty) -> bool:
    import pyarrow as pa

    return pa.types.is_integer(ty) or pa.types.is_floating(ty)


def _hits_of(values, lengths, k: int, mod: int, all_c, max_points: int):
    """The candidate hashes (``all_c``, sorted) hit by any channel: ``values``
    is the channels laid end to end, ``lengths`` their lengths. Each channel
    is truncated to ``max_points`` and no window crosses a channel boundary,
    so the set equals the union of :func:`kgram_hashes` over the channels
    (the same hash, :func:`_window_hashes`) — vectorised over a whole batch."""
    import numpy as np

    lengths = np.asarray(lengths, dtype=np.int64)
    if not lengths.size or not all_c.size:
        return np.empty(0, dtype=np.uint64)
    vals = np.asarray(values, dtype=np.float64)
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]]).tolist()
    use = np.minimum(lengths, max_points).tolist()
    out = []

    def flush(segs, lens):
        cat = np.concatenate(segs)
        ln = np.asarray(lens, dtype=np.int64)
        ends = np.cumsum(ln)
        # window j is valid iff it ends inside its own channel
        pos = np.arange(cat.size, dtype=np.int64) - np.repeat(ends - ln, ln)
        lim = np.repeat(ln, ln)
        h, okw = _window_hashes(cat, k)
        valid = okw & (pos[:h.size] + k <= lim[:h.size])
        h = h[valid]
        h = h[h % np.uint64(mod) == 0]
        if h.size:
            p = np.searchsorted(all_c, h)
            p[p >= all_c.size] = 0
            out.append(np.unique(h[all_c[p] == h]))

    segs, lens, n = [], [], 0
    for s0, u in zip(starts, use, strict=True):
        if u < k:
            continue                                   # no window (kgram_hashes: empty)
        segs.append(vals[s0:s0 + u])
        lens.append(u)
        n += u
        if n >= (1 << 22):                             # ~4M points per block
            flush(segs, lens)
            segs, lens, n = [], [], 0
    if segs:
        flush(segs, lens)
    return np.unique(np.concatenate(out)) if out else np.empty(0, dtype=np.uint64)


def file_hits(path: Path, all_c, k: int, mod: int, max_points: int):
    """The candidate hashes one exclusion file hits: the same channels as
    :func:`iter_channels` (list columns row by row, one per channel, nested
    lists one per inner list, nulls as NaN; scalar numeric columns whole, in
    ``max_points`` chunks over the first ``64 * max_points`` values), read
    batch by batch with Arrow's buffers instead of Python lists."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc

    schema = _schema(path)
    if schema is None:
        return np.empty(0, dtype=np.uint64)
    list_cols, scalar_cols = [], []
    for fld in schema:
        ty = fld.type
        if pa.types.is_list(ty) or pa.types.is_large_list(ty):
            inner = ty.value_type
            if _is_num(inner) or ((pa.types.is_list(inner) or pa.types.is_large_list(inner))
                                  and _is_num(inner.value_type)):
                list_cols.append(fld.name)
        elif _is_num(ty):
            scalar_cols.append(fld.name)
    hits = []
    cap = max_points * 64
    scal = {c: [] for c in scalar_cols}
    got = dict.fromkeys(scalar_cols, 0)
    cols = list_cols + scalar_cols
    if not cols:
        return np.empty(0, dtype=np.uint64)
    for b in _batches(path, cols):
        for name in list_cols:
            arr = b.column(name)
            nested = not _is_num(arr.type.value_type)
            if nested:
                arr = arr.flatten()                    # one element per channel
            ln = pc.fill_null(pc.list_value_length(arr), 0).to_numpy(zero_copy_only=False)
            vals = pc.fill_null(pc.cast(arr.flatten(), pa.float64()), float("nan"))
            hits.append(_hits_of(vals.to_numpy(zero_copy_only=False), ln, k, mod, all_c,
                                 max_points))
        for name in scalar_cols:
            if got[name] >= cap:
                continue
            a = pc.cast(b.column(name), pa.float64()).to_numpy(zero_copy_only=False)
            a = a[:cap - got[name]]
            scal[name].append(a)
            got[name] += a.size
    for name in scalar_cols:
        if not scal[name]:
            continue
        arr = np.concatenate(scal[name])
        ln = [min(max_points, arr.size - i) for i in range(0, arr.size, max_points)]
        hits.append(_hits_of(arr, ln, k, mod, all_c, max_points))
    hits = [h for h in hits if h.size]
    return np.unique(np.concatenate(hits)) if hits else np.empty(0, dtype=np.uint64)


_POOL_STATE: dict = {}


def _pool_init(all_c, k, mod, max_points):
    _POOL_STATE.update(all_c=all_c, k=k, mod=mod, max_points=max_points)


def _pool_file(path: str):
    s = _POOL_STATE
    return path, file_hits(Path(path), s["all_c"], s["k"], s["mod"], s["max_points"])


def _hf_data_files(repo: str, rev: str) -> list[tuple[str, int]]:
    return sorted((e["path"], int(e.get("size") or 0)) for e in hf_tree(repo, rev, recursive=True)
                  if e.get("type") == "file"
                  and e["path"].lower().endswith((".arrow", ".parquet", ".csv")))


def _size_batches(files: list[tuple[str, int]], budget: int) -> list[list[str]]:
    out, cur, n = [], [], 0
    for p, sz in files:
        if cur and n + sz > budget:
            out.append(cur)
            cur, n = [], 0
        cur.append(p)
        n += sz
    if cur:
        out.append(cur)
    return out


def _stream_source(key: str, stream_dir: Path, budget: int):
    """Yields ``(dir, [file paths])`` per download batch of ``key``'s data at
    its pinned revision (the next batch downloads while the caller hashes this
    one; the caller deletes each dir when done, so disk stays ~2 batches)."""
    import queue
    import shutil
    import threading

    from huggingface_hub import snapshot_download

    repo, rev = EXCLUSION_SOURCES[key]
    batches = _size_batches(_hf_data_files(repo, rev), budget)
    q: queue.Queue = queue.Queue(maxsize=1)

    def produce():
        try:
            for i, paths in enumerate(batches):
                d = stream_dir / key / f"b{i:05d}"
                if d.exists():
                    shutil.rmtree(d)
                snapshot_download(repo_id=repo, repo_type="dataset", revision=rev,
                                  local_dir=str(d), allow_patterns=paths, max_workers=16)
                q.put((d, [d / p for p in paths]))
            q.put(None)
        except BaseException as e:  # noqa: BLE001 — re-raised in the consumer
            q.put(e)

    threading.Thread(target=produce, daemon=True).start()
    print(f"stream {key}: {repo}@{rev[:7]} {len(batches)} batches", flush=True)
    while True:
        item = q.get()
        if item is None:
            return
        if isinstance(item, BaseException):
            raise SystemExit(f"stream {key}: download failed ({item})")
        yield item


def data_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file()
                  and p.suffix.lower() in (".arrow", ".parquet", ".csv"))


def cand_file_hashes(path, sample_series: int, max_points: int, k: int, mod: int,
                     *, reference: bool = False):
    """``(n_series, hashes)`` of one RMISC candidate parquet file: the
    ``sample_series`` series with the smallest sha256(name), each numeric
    column of each truncated to ``max_points``, K-gram hashed. ``reference``
    is the original per-row Python path; the default does the same in Arrow
    (``index_in`` instead of an object-array compare per series)."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    from cascade.eval.lane_d_public_index import RMISC_SERIES_COLUMN

    key = lambda s: hashlib.sha256(s.encode()).hexdigest()  # noqa: E731
    if reference:
        t = pq.read_table(str(path))
        cols = [x for x in t.column_names if x != RMISC_SERIES_COLUMN
                and (pa.types.is_integer(t.column(x).type)
                     or pa.types.is_floating(t.column(x).type))]
        vals = {x: t.column(x).cast(pa.float64()).to_numpy(zero_copy_only=False) for x in cols}
        names = list(map(str, t.column(RMISC_SERIES_COLUMN).to_pylist())) \
            if RMISC_SERIES_COLUMN in t.column_names else ["_all"] * t.num_rows
        uniq = sorted(set(names), key=key)[:sample_series]
        arr_names = np.asarray(names, dtype=object)
        hs = [kgram_hashes(vals[c][np.flatnonzero(arr_names == s)][:max_points], k, mod)
              for s in uniq for c in cols]
        h = np.unique(np.concatenate(hs)) if hs else np.empty(0, dtype=np.uint64)
        return len(uniq), h
    # Memory-bounded: the name column first, then one numeric column at a time,
    # taking only the sampled rows (a wide table is never materialised whole).
    schema = pq.read_schema(str(path))
    cols = [f.name for f in schema if f.name != RMISC_SERIES_COLUMN
            and (pa.types.is_integer(f.type) or pa.types.is_floating(f.type))]
    has = RMISC_SERIES_COLUMN in schema.names
    if has:
        col = pq.read_table(str(path), columns=[RMISC_SERIES_COLUMN]).column(RMISC_SERIES_COLUMN)
        if pa.types.is_dictionary(col.type):
            col = col.cast(col.type.value_type)               # RMISC: dictionary<string>
        if pa.types.is_string(col.type) or pa.types.is_large_string(col.type):
            col = pc.fill_null(col, "None")                   # str(None), as the reference
            uniq = sorted(pc.unique(col).to_pylist(), key=key)[:sample_series]
            pos = pc.fill_null(pc.index_in(col, value_set=pa.array(uniq, type=col.type)), -1)
            pos = np.asarray(pos.to_numpy(zero_copy_only=False))
            groups = [np.flatnonzero(pos == j) for j in range(len(uniq))]
        else:
            arr_names = np.asarray(list(map(str, col.to_pylist())), dtype=object)
            uniq = sorted(set(arr_names.tolist()), key=key)[:sample_series]
            groups = [np.flatnonzero(arr_names == s) for s in uniq]
    else:
        n = pq.ParquetFile(str(path)).metadata.num_rows
        uniq = ["_all"]
        groups = [np.arange(n)]
    groups = [g[:max_points] for g in groups]                 # vals[idx][:m] == vals[idx[:m]]
    sel = np.concatenate(groups) if groups else np.empty(0, dtype=np.int64)
    offs = np.concatenate([[0], np.cumsum([g.size for g in groups])]).tolist()
    hs = []
    for c in cols:
        v = pq.read_table(str(path), columns=[c]).column(c).take(pa.array(sel)) \
            .cast(pa.float64()).to_numpy(zero_copy_only=False)
        for j in range(len(groups)):
            hs.append(kgram_hashes(v[offs[j]:offs[j + 1]], k, mod))
    h = np.unique(np.concatenate(hs)) if hs else np.empty(0, dtype=np.uint64)
    return len(uniq), h


def _cand_job(job):
    name, f, sample_series, max_points, k, mod = job
    n, h = cand_file_hashes(f, sample_series, max_points, k, mod)
    return name, f, n, h


def cmd_datacompare(args) -> int:
    import numpy as np

    cands = json.loads(Path(args.candidates).read_text(encoding="utf-8"))
    if cands.get("schema") != CANDIDATES_SCHEMA:
        raise SystemExit("not a candidates file")
    k, mod = int(args.k), int(args.mod)
    roots = dict(x.split("=", 1) for x in args.exclude_root)
    for key in getattr(args, "stream", None) or []:
        if key not in EXCLUSION_SOURCES:
            raise SystemExit(f"--stream {key}: not an exclusion source")
        roots.setdefault(key, "hf")
    missing = sorted(set(EXCLUSION_SOURCES) - set(roots))
    if missing:
        raise SystemExit(f"--exclude-root missing for {missing} (every source is compared)")
    folders = []
    owner: dict[int, set[str]] = {}
    per_folder: dict[str, np.ndarray] = {}
    jobs = [(c["name"], str(f)) for c in cands["candidates"]
            for f in sorted((Path(args.rmisc_root) / c["name"]).glob("*.parquet"))]
    cargs = (int(args.sample_series), int(args.max_points), k, mod)
    cw = max(1, min(int(getattr(args, "workers", 1) or 1), int(args.candidate_workers)))
    if cw > 1:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(cw, mp_context=mp.get_context("fork")) as cpool:
            done = list(cpool.map(_cand_job, [(n, f, *cargs) for n, f in jobs], chunksize=1))
    else:
        done = [_cand_job((n, f, *cargs)) for n, f in jobs]
    by: dict[str, list] = {}
    for name, f, n, h in done:
        by.setdefault(name, []).append((n, h))
        print(f"hashed {name}: {f} series={n} hashes={h.size}", flush=True)
    for c in cands["candidates"]:
        got = by.get(c["name"], [])
        hs = [h for _, h in got]
        n_series = sum(n for n, _ in got)
        arr = np.unique(np.concatenate(hs)) if hs else np.empty(0, dtype=np.uint64)
        per_folder[c["name"]] = arr
        for x in arr.tolist():
            owner.setdefault(x, set()).add(c["name"])
        folders.append({"name": c["name"], "n_series_sampled": n_series, "n_hashes": int(arr.size)})
    all_c = np.asarray(sorted(owner), dtype=np.uint64)
    hits = {f["name"]: {s: set() for s in EXCLUSION_SOURCES} for f in folders}
    emax = int(args.exclude_max_points)
    workers = max(1, int(getattr(args, "workers", 1) or 1))
    pool = None
    if workers > 1:
        import gc
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        gc.freeze()          # forked workers never touch (copy) the parent's hash maps
        pool = ProcessPoolExecutor(workers, mp_context=mp.get_context("fork"),
                                   initializer=_pool_init, initargs=(all_c, k, mod, emax))

    def scan(key, files):
        if pool is None:
            res = ((str(f), file_hits(f, all_c, k, mod, emax)) for f in files)
        else:
            res = pool.map(_pool_file, [str(f) for f in files], chunksize=1)
        for f, h in res:
            for x in h.tolist():
                for name in owner[x]:
                    hits[name][key].add(x)
            print(f"compared {key}: {f}", flush=True)

    import shutil

    for key in sorted(EXCLUSION_SOURCES):
        if roots[key] == "hf":
            if not getattr(args, "stream_dir", None):
                raise SystemExit(f"{key}=hf needs --stream-dir")
            budget = int(float(args.stream_batch_gib) * 2**30)
            for d, files in _stream_source(key, Path(args.stream_dir), budget):
                scan(key, [f for f in files if f.is_file()])
                shutil.rmtree(d, ignore_errors=True)
        else:
            scan(key, data_files(Path(roots[key])))
    if pool is not None:
        pool.shutdown()
    for f in folders:
        n = max(1, f["n_hashes"])
        f["hits"] = {s: len(v) for s, v in sorted(hits[f["name"]].items())}
        f["max_hit_fraction"] = max(f["hits"].values()) / n if f["n_hashes"] else 0.0
        f["verdict"] = ("no_data" if not f["n_hashes"] else
                        "duplicate" if f["max_hit_fraction"] >= float(args.dup_threshold)
                        else "pass")
    doc = {"schema": DATACOMPARE_SCHEMA,
           "params": {"k": k, "mod": mod, "sample_series": int(args.sample_series),
                      "max_points": int(args.max_points),
                      "exclude_max_points": int(args.exclude_max_points),
                      "dup_threshold": float(args.dup_threshold),
                      "candidates_sha256": hashlib.sha256(
                          Path(args.candidates).read_bytes()).hexdigest()},
           "folders": folders}
    Path(args.out).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print("DATACOMPARE " + ", ".join(f"{v}={sum(f['verdict'] == v for f in folders)}"
                                     for v in ("pass", "duplicate", "no_data")))
    return 0


# ── step 4: finalize / validate ─────────────────────────────────────────────

def finalize(cands: dict, dc: dict, *, allow_unknown_licence: bool, generated_by: str) -> str:
    from cascade.eval.lane_d_public_index import parse_allowed_list, render_allowed_list

    if cands.get("schema") != CANDIDATES_SCHEMA or dc.get("schema") != DATACOMPARE_SCHEMA:
        raise SystemExit("finalize needs a candidates file and a datacompare file")
    verdict = {f["name"]: f for f in dc["folders"]}
    folders = []
    for c in cands["candidates"]:
        v = verdict.get(c["name"])
        if v is None or v["verdict"] != "pass":
            continue
        if c["licence_class"] == "unknown" and not allow_unknown_licence:
            continue
        folders.append({"name": c["name"],
                        "reason": (f"no name match in the exclusion sources; data compare pass "
                                   f"(max hit fraction {v['max_hit_fraction']:.4f})"),
                        "licence": c["licence"] or "unknown", "domain": c["domain"] or "unknown",
                        "freq": c["freq"] or "unknown", "data_check": "pass"})
    text = render_allowed_list(status="final", folders=folders, generated_by=generated_by)
    parse_allowed_list(text)
    return text


def cmd_finalize(args) -> int:
    cands = json.loads(Path(args.candidates).read_text(encoding="utf-8"))
    dc = json.loads(Path(args.datacompare).read_text(encoding="utf-8"))
    gen = (f"laned_rmisc_allowlist.py finalize; candidates sha256 "
           f"{hashlib.sha256(Path(args.candidates).read_bytes()).hexdigest()[:16]}; datacompare "
           f"sha256 {hashlib.sha256(Path(args.datacompare).read_bytes()).hexdigest()[:16]}")
    text = finalize(cands, dc, allow_unknown_licence=args.allow_unknown_licence, generated_by=gen)
    Path(args.out).write_text(text, encoding="utf-8")
    n = text.count("[[folder]]")
    print(f"ALLOWED {n} folders -> {args.out}")
    return 0


def cmd_validate(args) -> int:
    from cascade.eval.lane_d_public_index import ALLOWED_LIST_PATH, load_allowed_list

    path = args.file or str(ALLOWED_LIST_PATH)
    doc = load_allowed_list(path)
    print(f"OK {path}: status {doc['status']}, {len(doc['names'])} folders")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fetch-listings")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_fetch_listings)
    p = sub.add_parser("candidates")
    p.add_argument("--listings", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_candidates)
    p = sub.add_parser("download")
    p.add_argument("--source", required=True, choices=["rmisc", *EXCLUSION_SOURCES])
    p.add_argument("--local-dir", required=True)
    p.add_argument("--candidates", help="rmisc: download only the candidate folders")
    p.set_defaults(fn=cmd_download)
    p = sub.add_parser("datacompare")
    p.add_argument("--candidates", required=True)
    p.add_argument("--rmisc-root", required=True)
    p.add_argument("--exclude-root", action="append", default=[], metavar="KEY=DIR")
    p.add_argument("--out", required=True)
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--mod", type=int, default=32)
    p.add_argument("--sample-series", type=int, default=64)
    p.add_argument("--max-points", type=int, default=16384)
    p.add_argument("--exclude-max-points", type=int, default=1_000_000)
    p.add_argument("--dup-threshold", type=float, default=0.02)
    p.add_argument("--workers", type=int, default=1,
                   help="processes scanning exclusion files (the result is identical)")
    p.add_argument("--stream", action="append", default=[], metavar="KEY",
                   help="stream this exclusion source from HF at its pinned revision in "
                        "download batches (hash, then delete) instead of --exclude-root")
    p.add_argument("--stream-dir", help="scratch dir for --stream batches")
    p.add_argument("--stream-batch-gib", type=float, default=64.0)
    p.add_argument("--candidate-workers", type=int, default=12,
                   help="cap on processes hashing candidate files (each reads a file whole)")
    p.set_defaults(fn=cmd_datacompare)
    p = sub.add_parser("finalize")
    p.add_argument("--candidates", required=True)
    p.add_argument("--datacompare", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--allow-unknown-licence", action="store_true")
    p.set_defaults(fn=cmd_finalize)
    p = sub.add_parser("validate")
    p.add_argument("file", nargs="?")
    p.set_defaults(fn=cmd_validate)
    args = ap.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
