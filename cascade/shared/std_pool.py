"""The STANDARD POOL (DEC-CA-0075, W-CMP): one content-addressed, signed
schema every lane-C competition's eval / training data is published in.

Owner 2026-11-24: "new test or formula and embargo days etc. should be up to
our choice, and we should bring new lane C competitions without validators
updating"; a new data source must NOT need a validator release. So every
source-specific step (market APIs, relays, energy feeds, scrapers) lives only
in a pool BUILDER (owner infra); validators, the trainer and the audit read
one schema through one reader (:func:`load_std_pool`). A new domain is a new
builder job publishing a standard pool plus an overlay spec naming it
(:mod:`cascade.shared.competition_spec`) — no validator code.

Layout of a pool directory::

    pool.json                    signed body (below) + ``signature``
    <any relative path>.npy      float64 channel arrays / int64 timestamps,
                                 every one listed in ``files`` with its sha256

``pool.json`` body (schema :data:`SCHEMA`):

* ``schema`` — ``"cascade-std-pool/1"``; ``pool_id`` — the builder job name;
* ``series`` — ``{series_id: record}``, a record being
  ``grid`` (``{t0, step_seconds, steps}``: a regular UTC grid, the
  timestamp array is ``t0 + step_seconds * arange(steps)``) OR
  ``timestamps`` (relative path of an int64 unix-second array); ``channels``
  (``{name: relative path}``, each float64 of the series length, NaN =
  missing — together the (C, L) values); ``freq`` (``"1m"``, ``"1h"``, ...);
  ``tags`` (``{domain, source, ...}`` string tags the spec selects series
  by); ``first_seen`` / ``embargo_until`` (unix seconds, 0 = none);
* ``clock`` — optional ``{anchor_block, anchor_time, block_seconds}``: the
  block → time map the forward-only rules use;
* ``files`` — ``{relative path: sha256}`` of every data file (nothing
  unlisted under a listed directory is accepted);
* ``source_meta`` — builder metadata carried verbatim (never scored);
* ``signer`` — the ss58 hotkey of the pool signer.

The digest is the sha256 of the canonical body (sorted-key compact JSON
without ``signature``); the signature (sr25519) covers the same bytes.

Older formats are SPECIAL CASES of this schema: a MarketCast snapshot
(``snapshot.json``, ``cascade.pool.marketcast`` v1) is the standard pool
whose series are ``<bank>/<class>/<asset>`` with the OHLCV(+funding) fields
as channels (:func:`std_body_from_mkt_meta`), and the MarketCast builder
emits that form (:func:`write_std_pool_from_mkt`). :func:`mkt_meta_view`
projects a standard pool back onto the snapshot API, so the MarketCast
judge reads the standard pool with byte-identical arrays (pinned by
``tests/accept/w_cmp``).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SCHEMA = "cascade-std-pool/1"
SCHEMAS = (SCHEMA,)
POOL_FILE = "pool.json"
BODY_FIELDS = frozenset({"schema", "pool_id", "series", "files", "signer"})
OPTIONAL_FIELDS = frozenset({"clock", "source_meta"})
SERIES_FIELDS = frozenset({"channels", "freq", "tags"})
SERIES_OPTIONAL = frozenset({"grid", "timestamps", "first_seen", "embargo_until"})
_ID_RE = re.compile(r"^[A-Za-z0-9_.:\-]+(?:/[A-Za-z0-9_.:\-]+)*$")
MAX_SERIES = 200_000


class StdPoolError(ValueError):
    """A standard pool is malformed, tampered or unsigned (typed: the
    caller fails closed — the competition holds, never a crash)."""


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_body(body: dict) -> bytes:
    d = {k: v for k, v in body.items() if k != "signature"}
    return json.dumps(d, sort_keys=True, separators=(",", ":")).encode("utf-8")


def pool_digest(body: dict) -> str:
    return hashlib.sha256(canonical_body(body)).hexdigest()


def _safe_rel(rel: object) -> bool:
    return (isinstance(rel, str) and bool(rel) and not rel.startswith("/")
            and ".." not in Path(rel).parts and rel.endswith(".npy"))


def check_body(body: object) -> dict:
    """The shape rules of a pool body (no file I/O); returns it unchanged."""
    if not isinstance(body, dict):
        raise StdPoolError("pool body must be an object")
    keys = set(body) - {"signature"}
    if not keys >= BODY_FIELDS or keys - BODY_FIELDS - OPTIONAL_FIELDS:
        raise StdPoolError(f"pool fields must be {sorted(BODY_FIELDS)} (+ optional "
                           f"{sorted(OPTIONAL_FIELDS)}); got {sorted(keys)}")
    if body["schema"] not in SCHEMAS:
        raise StdPoolError(f"unsupported pool schema {body['schema']!r} (this release "
                           f"reads {list(SCHEMAS)})")
    files = body["files"]
    if not isinstance(files, dict) or not all(_safe_rel(r) and isinstance(s, str)
                                              and len(s) == 64 for r, s in files.items()):
        raise StdPoolError("pool files must map safe relative .npy paths to sha256 hex")
    series = body["series"]
    if not isinstance(series, dict) or len(series) > MAX_SERIES:
        raise StdPoolError(f"pool series must be a table of at most {MAX_SERIES} records")
    for sid, rec in series.items():
        where = f"series {sid!r}"
        if not isinstance(sid, str) or not _ID_RE.match(sid):
            raise StdPoolError(f"{where}: bad series id")
        if not isinstance(rec, dict):
            raise StdPoolError(f"{where} must be an object")
        rk = set(rec)
        if not rk >= SERIES_FIELDS or rk - SERIES_FIELDS - SERIES_OPTIONAL:
            raise StdPoolError(f"{where}: fields must be {sorted(SERIES_FIELDS)} (+ "
                               f"{sorted(SERIES_OPTIONAL)}); got {sorted(rk)}")
        if ("grid" in rec) == ("timestamps" in rec):
            raise StdPoolError(f"{where}: exactly one of grid / timestamps")
        if "grid" in rec:
            g = rec["grid"]
            if (not isinstance(g, dict) or set(g) != {"t0", "step_seconds", "steps"}
                    or any(type(g[k]) is not int for k in g)
                    or g["step_seconds"] <= 0 or g["steps"] < 0):
                raise StdPoolError(f"{where}: grid must be integer {{t0, step_seconds > 0, "
                                   "steps >= 0}")
        elif rec["timestamps"] not in files:
            raise StdPoolError(f"{where}: timestamps file is not listed")
        ch = rec["channels"]
        if not isinstance(ch, dict) or not ch or not all(
                isinstance(n, str) and n and r in files for n, r in ch.items()):
            raise StdPoolError(f"{where}: channels must map names to listed files")
        if not isinstance(rec["freq"], str) or not rec["freq"]:
            raise StdPoolError(f"{where}: freq must be a non-empty string")
        tags = rec["tags"]
        if not isinstance(tags, dict) or not all(isinstance(k, str) and isinstance(v, str)
                                                 for k, v in tags.items()):
            raise StdPoolError(f"{where}: tags must be string -> string")
        for k in ("first_seen", "embargo_until"):
            if k in rec and (type(rec[k]) is not int or rec[k] < 0):
                raise StdPoolError(f"{where}: {k} must be an integer >= 0")
    clock = body.get("clock")
    if clock is not None and (not isinstance(clock, dict)
                              or set(clock) != {"anchor_block", "anchor_time",
                                                "block_seconds"}):
        raise StdPoolError("pool clock must be {anchor_block, anchor_time, block_seconds}")
    if not isinstance(body["signer"], str):
        raise StdPoolError("pool signer must be a string")
    return body


def sign_body(body: dict, wallet) -> dict:
    hotkey = getattr(wallet, "hotkey", wallet)
    sig = hotkey.sign(canonical_body(check_body(body)))
    return {**body, "signature": sig.hex() if isinstance(sig, (bytes, bytearray)) else str(sig)}


def verify_signature(body: dict, signer_hotkey: str) -> bool:
    sig = str(body.get("signature", "") or "")
    if not sig or not signer_hotkey:
        return False
    from .sr25519 import verify_hotkey

    return verify_hotkey(signer_hotkey, canonical_body(body), sig)


@dataclass
class StdPool:
    """A verified standard pool directory."""
    root: Path
    body: dict
    digest: str
    _cache: dict = field(default_factory=dict, repr=False)

    @property
    def series_ids(self) -> list[str]:
        return sorted(self.body["series"])

    def record(self, sid: str) -> dict:
        return self.body["series"][sid]

    def select(self, **tags: str) -> list[str]:
        """The series ids whose tags carry every ``tags`` item, sorted."""
        return [s for s in self.series_ids
                if all(self.record(s)["tags"].get(k) == v for k, v in tags.items())]

    def timestamps(self, sid: str) -> np.ndarray:
        rec = self.record(sid)
        if "grid" in rec:
            g = rec["grid"]
            return int(g["t0"]) + int(g["step_seconds"]) * np.arange(int(g["steps"]),
                                                                    dtype=np.int64)
        return np.load(self.root / rec["timestamps"], allow_pickle=False)

    def channel(self, sid: str, name: str) -> np.ndarray:
        key = (sid, name)
        hit = self._cache.get(key)
        if hit is None:
            hit = np.load(self.root / self.record(sid)["channels"][name], allow_pickle=False)
            self._cache[key] = hit
        return hit

    def values(self, sid: str, channels: tuple[str, ...] | None = None) -> np.ndarray:
        """``(C, L)`` float64 values of ``channels`` (default: every channel,
        sorted by name)."""
        names = tuple(sorted(self.record(sid)["channels"])) if channels is None else channels
        return np.stack([np.asarray(self.channel(sid, n), dtype=np.float64) for n in names])


def load_std_pool(root: Path | str, *, signer_hotkey: str = "",
                  verify: bool = True, schema: str = SCHEMA) -> StdPool:
    """THE reader: ``pool.json`` under ``root``, its shape, every listed
    file's sha256, no unlisted file beside listed ones, every series' arrays
    of its declared length, and (``verify``) the signature by
    ``signer_hotkey``. ``schema`` = the version the competition spec names
    (a pool of another version is refused). Raises :class:`StdPoolError`."""
    root = Path(root)
    try:
        body = json.loads((root / POOL_FILE).read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001 — any read failure is a typed refusal
        raise StdPoolError(f"{root}/{POOL_FILE} unreadable: {e}") from e
    check_body(body)
    if body["schema"] != schema:
        raise StdPoolError(f"pool schema {body['schema']!r} != the spec's {schema!r}")
    files = body["files"]
    for rel, sha in files.items():
        p = root / rel
        if not p.is_file() or p.is_symlink() or sha256_file(p) != sha:
            raise StdPoolError(f"pool file {rel} missing or sha256 mismatch")
    dirs = {str(Path(r).parent) for r in files}
    extra = sorted(str(p.relative_to(root)) for d in dirs if (root / d).is_dir()
                   for p in (root / d).iterdir()
                   if p.is_file() and str(p.relative_to(root)) not in files)
    if extra:
        raise StdPoolError(f"unlisted pool files {extra[:3]}")
    if verify and not verify_signature(body, signer_hotkey):
        raise StdPoolError(f"pool signature does not verify against {signer_hotkey or 'no signer'}")
    pool = StdPool(root, body, pool_digest(body))
    for sid in pool.series_ids:
        n = len(pool.timestamps(sid))
        for name in pool.record(sid)["channels"]:
            if pool.channel(sid, name).shape != (n,):
                raise StdPoolError(f"series {sid!r} channel {name!r}: length != {n}")
    return pool


def write_std_pool(out_dir: Path | str, *, pool_id: str, series: dict[str, dict],
                   arrays: dict[str, np.ndarray], signer: str = "", wallet=None,
                   clock: dict | None = None, source_meta: dict | None = None) -> dict:
    """BUILDER side: write ``arrays`` (relative path → array; float64
    channels, int64 timestamps) and ``pool.json`` with ``series`` records,
    signed when ``wallet`` is given. Deterministic bytes."""
    out = Path(out_dir)
    files: dict[str, str] = {}
    for rel, arr in sorted(arrays.items()):
        if not _safe_rel(rel):
            raise StdPoolError(f"unsafe pool path {rel!r}")
        p = out / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        a = np.asarray(arr)
        a = a.astype(np.int64 if a.dtype.kind in "iu" else np.float64, copy=False)
        np.save(p, a, allow_pickle=False)
        files[rel] = sha256_file(p)
    body: dict = {"schema": SCHEMA, "pool_id": str(pool_id),
                  "series": {k: series[k] for k in sorted(series)},
                  "files": dict(sorted(files.items())), "signer": str(signer or "")}
    if clock is not None:
        body["clock"] = dict(clock)
    if source_meta:
        body["source_meta"] = dict(source_meta)
    check_body(body)
    if wallet is not None:
        body = sign_body(body, wallet)
    out.mkdir(parents=True, exist_ok=True)
    (out / POOL_FILE).write_text(json.dumps(body, sort_keys=True, indent=1), encoding="utf-8")
    return body


# ── MarketCast: the snapshot as a special case of the standard pool ──────────

MKT_POOL_ID = "marketcast"
MKT_DOMAIN = "market"


def _mkt_banks(meta: dict) -> dict[str, tuple[int, int, int]]:
    """bank dir → (t0, step_seconds, steps) of a MarketCast snapshot meta."""
    from ..pool.marketcast import native_dir

    out = {"bars": (int(meta["t0"]), 60, int(meta["minutes"]))}
    c = meta.get("coarse")
    if c:
        out["coarse"] = (int(c["t0"]), 60 * int(c.get("bar_minutes", 60)), int(c["steps"]))
    for iv, rec in sorted((meta.get("native") or {}).items()):
        out[native_dir(iv)] = (int(rec["t0"]), 60 * int(rec["bar_minutes"]), int(rec["steps"]))
    return out


def std_body_from_mkt_meta(meta: dict) -> dict:
    """The standard-pool body (unsigned, ``signer`` kept) of a MarketCast
    snapshot meta: series ``<bank>/<class>/<asset>``, channels = its fields
    at the SAME relative paths, the clock, and the snapshot's own keys in
    ``source_meta`` so :func:`mkt_meta_view` restores them exactly."""
    banks = _mkt_banks(meta)
    series: dict[str, dict] = {}
    for rel in sorted(meta["files"]):
        parts = rel[:-4].split("/")
        if len(parts) != 4 or parts[0] not in banks:
            raise StdPoolError(f"MarketCast file {rel!r} is not <bank>/<class>/<asset>/<f>.npy")
        bank, cls, asset, fld = parts
        t0, step, n = banks[bank]
        sid = f"{bank}/{cls}/{asset}"
        rec = series.setdefault(sid, {
            "grid": {"t0": t0, "step_seconds": step, "steps": n}, "channels": {},
            "freq": f"{step // 60}m",
            "tags": {"domain": MKT_DOMAIN, "class": cls, "asset": asset, "bank": bank,
                     "source": str((meta.get("sources") or {}).get(cls, ""))}})
        rec["channels"][fld] = rel
    keep = {k: v for k, v in meta.items() if k not in ("files", "signer", "signature")}
    body = {"schema": SCHEMA, "pool_id": MKT_POOL_ID, "series": series,
            "files": dict(sorted(meta["files"].items())), "signer": str(meta.get("signer", "")),
            "clock": {"anchor_block": int(meta["anchor_block"]),
                      "anchor_time": float(meta["anchor_time"]),
                      "block_seconds": float(meta["block_seconds"])},
            "source_meta": {"marketcast_snapshot": keep}}
    return check_body(body)


def mkt_meta_view(body: dict, select: dict | None = None) -> dict:
    """The MarketCast snapshot meta a standard pool built from one projects
    to (the generic reader's view for the MarketCast judge): the snapshot's
    own keys from ``source_meta`` and the pool's ``files`` / ``signer``.
    ``select`` (tag → value, the spec's ``select``) keeps only the series
    carrying every tag; ``{}`` / None = every series (unchanged view)."""
    sm = dict((body.get("source_meta") or {}).get("marketcast_snapshot") or {})
    if not sm:
        raise StdPoolError("pool carries no MarketCast series (source_meta.marketcast_snapshot)")
    view = {**sm, "files": dict(body["files"]), "signer": str(body.get("signer", ""))}
    if select:
        return select_mkt_view(view, body, select)
    return view


def select_mkt_view(view: dict, body: dict, select: dict) -> dict:
    """``view`` restricted to the series of ``body`` whose tags carry every
    ``select`` item (their files and assets only)."""
    keep = {sid for sid, rec in body["series"].items()
            if all(rec["tags"].get(k) == v for k, v in select.items())}
    files = {rel: sha for rel, sha in view["files"].items()
             if "/".join(rel.split("/")[:3]) in keep}
    bars = {tuple(s.split("/")[1:3]) for s in keep if s.startswith("bars/")}
    assets = {c: [a for a in syms if (c, a) in bars]
              for c, syms in (view.get("assets") or {}).items()}
    return {**view, "files": files, "assets": {c: v for c, v in assets.items() if v}}


# W-CMP2: a standard pool from ANY builder, projected onto the MarketCast
# snapshot API: each series' grid step names its bank (1 m = the minute bars,
# 1 h = the coarse bank, 5 m / 15 m = the native banks), its ``class`` tag
# (else ``domain``) the asset class, its ``asset`` tag (else the last id
# component) the symbol, and each channel a field (renamed by the spec's
# ``channels`` map: metric channel -> pool channel). Every series of one bank
# must share one grid (a builder aligns them).
_STEP_BANKS = {60: ("bars", ""), 3600: ("coarse", ""), 300: ("n5m", "5m"), 900: ("n15m", "15m")}


def generic_mkt_view(body: dict, select: dict | None = None,
                     channels: dict | None = None) -> dict:
    """The MarketCast snapshot meta (with ``aliases``: virtual path -> the
    pool's file) of a standard pool that carries no MarketCast source meta."""
    clock = body.get("clock")
    if not clock:
        raise StdPoolError("a pool read by the MarketCast runner needs its block clock")
    inv = {pool_ch: metric_ch for metric_ch, pool_ch in dict(channels or {}).items()}
    grids: dict[str, tuple[int, int, int]] = {}
    files: dict[str, str] = {}
    aliases: dict[str, str] = {}
    assets: dict[str, list[str]] = {}
    sources: dict[str, str] = {}
    for sid in sorted(body["series"]):
        rec = body["series"][sid]
        tags = rec["tags"]
        if select and not all(tags.get(k) == v for k, v in select.items()):
            continue
        g = rec.get("grid")
        if not g or int(g["step_seconds"]) not in _STEP_BANKS:
            continue                    # explicit timestamps / unsupported step: not judged
        bank, _iv = _STEP_BANKS[int(g["step_seconds"])]
        grid = (int(g["t0"]), int(g["step_seconds"]), int(g["steps"]))
        if grids.setdefault(bank, grid) != grid:
            raise StdPoolError(f"series {sid!r}: bank {bank} grid {grid} differs from "
                               f"{grids[bank]} (a builder aligns one grid per step)")
        cls = str(tags.get("class") or tags.get("domain") or "")
        asset = str(tags.get("asset") or sid.rsplit("/", 1)[-1])
        if not cls or "/" in cls or "/" in asset:
            raise StdPoolError(f"series {sid!r}: class / asset tags are not path-safe")
        for ch, real in sorted(rec["channels"].items()):
            fld = inv.get(ch, ch)
            v = f"{bank}/{cls}/{asset}/{fld}.npy"
            files[v] = body["files"][real]
            aliases[v] = real
        if bank == "bars" or asset not in assets.get(cls, ()):
            assets.setdefault(cls, [])
            if asset not in assets[cls]:
                assets[cls].append(asset)
        sources.setdefault(cls, str(tags.get("source", "")))
    if not grids:
        raise StdPoolError("no series of the pool lies on a 1m / 5m / 15m / 1h grid")
    t0, step, n = grids.get("bars") or grids.get("coarse") or next(iter(grids.values()))
    minutes = n * step // 60
    meta: dict = {"version": 1, "t0": int(t0), "minutes": int(minutes),
                  "end": int(t0) + 60 * int(minutes),
                  "anchor_block": int(clock["anchor_block"]),
                  "anchor_time": float(clock["anchor_time"]),
                  "block_seconds": float(clock["block_seconds"]),
                  "sources": dict(sorted(sources.items())),
                  "assets": {c: sorted(v) for c, v in sorted(assets.items())},
                  "files": dict(sorted(files.items())), "cross_check": {},
                  "signer": str(body.get("signer", "")), "aliases": aliases}
    if "coarse" in grids:
        c0, _s, cn = grids["coarse"]
        meta["coarse"] = {"t0": int(c0), "steps": int(cn), "bar_minutes": 60}
        if "bars" not in grids:
            meta["end"] = max(int(meta["end"]), int(c0) + 3600 * int(cn))
    nat = {iv: grids[b] for b, iv in (("n5m", "5m"), ("n15m", "15m")) if b in grids}
    if nat:
        meta["native"] = {iv: {"t0": int(g0), "steps": int(gn), "bar_minutes": gs // 60}
                          for iv, (g0, gs, gn) in sorted(nat.items())}
    return meta


def pool_mkt_view(body: dict, select: dict | None = None,
                  channels: dict | None = None) -> dict:
    """THE projection the MarketCast runner reads a standard pool through: a
    MarketCast builder's pool (its own source meta) or any builder's pool
    (:func:`generic_mkt_view`)."""
    if (body.get("source_meta") or {}).get("marketcast_snapshot"):
        return mkt_meta_view(body, select)
    return generic_mkt_view(body, select, channels)


def write_std_pool_from_mkt(src_dir: Path | str, out_dir: Path | str, *, wallet=None) -> dict:
    """BUILDER: re-emit a MarketCast snapshot directory as a standard pool
    (same array bytes, same relative paths; signed when ``wallet``)."""
    import shutil

    from ..pool.marketcast import SNAPSHOT_FILE

    src, out = Path(src_dir), Path(out_dir)
    meta = json.loads((src / SNAPSHOT_FILE).read_text(encoding="utf-8"))
    body = std_body_from_mkt_meta(meta)
    for rel in body["files"]:
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src / rel, out / rel)
    if wallet is not None:
        body = sign_body(body, wallet)
    out.mkdir(parents=True, exist_ok=True)
    (out / POOL_FILE).write_text(json.dumps(body, sort_keys=True, indent=1), encoding="utf-8")
    return body


__all__ = ["generic_mkt_view", "pool_mkt_view", "MKT_DOMAIN", "MKT_POOL_ID", "POOL_FILE", "SCHEMA", "SCHEMAS", "StdPool",
           "StdPoolError", "canonical_body", "check_body", "load_std_pool", "mkt_meta_view",
           "pool_digest", "sign_body", "std_body_from_mkt_meta", "verify_signature",
           "write_std_pool", "write_std_pool_from_mkt"]
