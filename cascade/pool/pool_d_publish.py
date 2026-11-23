"""Lane D ``pool_d`` pinning, retention and cadence (plan W6d; rev6.11 (d)).

One ``pool_d/1`` snapshot is built every ``CADENCE_SECONDS`` (3 h) on the
dedicated builder (W6b), split into ``pool_d_hist`` / ``pool_d_tgt`` (W6c) and
published here under ``<prefix>/<snapshot>/`` in the private
``[lane_d_ops] pool_d_bucket``, content-addressed by sha256:

* builder: :func:`publish_snapshot` uploads both tars, a manifest and an
  index entry; a snapshot name never changes bytes (``snapshot_conflict``);
* box: :func:`sync_pins` downloads, re-checks every digest, re-joins the
  tars and appends a ``PinFilePool`` pin (``lane-d-pool-pins/1``), the input
  :class:`~cascade.validator.lane_d_inputs.PinFilePool` reads;
* eval host: :func:`fetch_hist` reads only the manifest and ``hist.tar``
  (targets never leave the box, invariant 8);
* :func:`retention_plan` / :func:`apply_retention` keep every snapshot an open
  entry's eligible union or frozen set can still reference.

The lane D report already carries each judged snapshot's sha256; that digest
covers ``metadata.json`` and with it the effective per-feed ``embargo_hours``
(W6b) and ``catalog_sha256``, the provenance hash of the forge catalog the
builder read. Nothing here re-reads the catalog. :func:`check_build_wall`
alerts (``pool_d_build_wall_exceeded``) when a build takes 3 h or more or ends
after the next boundary. Uploads above one part go multipart in parts of
``MULTIPART_CHUNK_BYTES`` (W3a: single PUTs are only safe up to 512 MiB).
Credentials come from the ``HIPPIUS_S3_*`` environment only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import socket
import sys
import tempfile
import time
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from ..shared.hippius import ObjectNotFound, S3Config, StorageError

log = logging.getLogger("cascade.pool")

CADENCE_SECONDS = 10_800
MAX_BUILD_WALL_SECONDS = 10_800
PINS_SCHEMA = "lane-d-pool-pins/1"
MANIFEST_SCHEMA = "lane-d-pool-manifest/1"
INDEX_SCHEMA = "lane-d-pool-index/1"
EVIDENCE_SCHEMA = "laned-evidence/w6d/1"
MULTIPART_CHUNK_BYTES = 256 * 1024 * 1024
TERMINAL_STATUSES = ("crowned", "lost", "closed")
DEFAULT_PREFIX = "pool_d"


class PublishError(ValueError):
    """A refused publish / sync / retention step; ``code`` is a stable tag."""

    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code


def _canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# ── cadence ─────────────────────────────────────────────────────────────────


def due_as_of(now_ts: int) -> int:
    """The ``as_of_ts`` of the snapshot due at ``now_ts`` (3-hourly)."""
    now_ts = int(now_ts)
    return now_ts - now_ts % CADENCE_SECONDS


def check_build_wall(*, as_of_ts: int, started_ts: int, finished_ts: int) -> dict:
    """The build-wall check: under 3 h and done before the next boundary."""
    wall = int(finished_ts) - int(started_ts)
    deadline = int(as_of_ts) + CADENCE_SECONDS
    ok = wall < MAX_BUILD_WALL_SECONDS and int(finished_ts) < deadline
    if not ok:
        log.warning("pool_d_build_wall_exceeded as_of=%s wall=%ss limit=%ss deadline=%s "
                    "finished=%s", as_of_ts, wall, MAX_BUILD_WALL_SECONDS, deadline, finished_ts)
    return {"ok": bool(ok), "wall_seconds": wall, "limit_seconds": MAX_BUILD_WALL_SECONDS,
            "deadline_ts": deadline}


# ── stores ──────────────────────────────────────────────────────────────────


@dataclass
class FsPoolStore:
    """A pool store on a local filesystem (testnet / local runs)."""

    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)

    @property
    def bucket(self) -> str:
        return str(self.root)

    def _path(self, key: str) -> Path:
        parts = [p for p in key.split("/") if p]
        if not parts or any(p in (".", "..") for p in parts):
            raise StorageError(f"bad key {key!r}")
        return self.root.joinpath(*parts)

    def put_file(self, key: str, path: Path | str) -> None:
        dest = self._path(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)

    def get_file(self, key: str, path: Path | str) -> None:
        src = self._path(key)
        if not src.is_file():
            raise ObjectNotFound(f"fs_get_missing: {key}")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, path)

    def put_bytes(self, key: str, data: bytes) -> None:
        _write_atomic(self._path(key), bytes(data))

    def get_bytes(self, key: str) -> bytes:
        src = self._path(key)
        if not src.is_file():
            raise ObjectNotFound(f"fs_get_missing: {key}")
        return src.read_bytes()


@dataclass
class HippiusPoolStore:
    """The pool store over the private Hippius S3 bucket (lazy client)."""

    cfg: S3Config
    prefix: str = DEFAULT_PREFIX
    _client: object = None

    @property
    def bucket(self) -> str:
        return self.cfg.bucket

    @property
    def endpoint(self) -> str:
        return self.cfg.endpoint

    def client(self):
        if self._client is None:
            from ..shared.hippius import _s3_client

            self._client = _s3_client(self.cfg)
        return self._client

    def _transfer(self):
        from boto3.s3.transfer import TransferConfig  # type: ignore

        return TransferConfig(multipart_threshold=MULTIPART_CHUNK_BYTES,
                              multipart_chunksize=MULTIPART_CHUNK_BYTES)

    def _missing(self, e: Exception) -> bool:
        from ..shared.hippius import _is_missing_object

        return _is_missing_object(e)

    def put_file(self, key: str, path: Path | str) -> None:
        c = self.client()
        try:
            c.upload_file(str(path), self.cfg.bucket, key, Config=self._transfer())
        except Exception as e:  # noqa: BLE001
            raise StorageError(f"s3_put_failed: {key}: {e}") from e

    def get_file(self, key: str, path: Path | str) -> None:
        c = self.client()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        try:
            c.download_file(self.cfg.bucket, key, str(path), Config=self._transfer())
        except Exception as e:  # noqa: BLE001
            if self._missing(e) or "404" in str(e):
                raise ObjectNotFound(f"s3_get_missing: {key}") from e
            raise StorageError(f"s3_get_failed: {key}: {e}") from e

    def put_bytes(self, key: str, data: bytes) -> None:
        c = self.client()
        try:
            c.put_object(Bucket=self.cfg.bucket, Key=key, Body=bytes(data),
                         ContentType="application/octet-stream")
        except Exception as e:  # noqa: BLE001
            raise StorageError(f"s3_put_failed: {key}: {e}") from e

    def get_bytes(self, key: str) -> bytes:
        c = self.client()
        try:
            return c.get_object(Bucket=self.cfg.bucket, Key=key)["Body"].read()
        except Exception as e:  # noqa: BLE001
            if self._missing(e):
                raise ObjectNotFound(f"s3_get_missing: {key}") from e
            raise StorageError(f"s3_get_failed: {key}: {e}") from e


def open_pool_store(chain_toml: Path | str) -> HippiusPoolStore:
    """The Hippius pool store named by a chain.toml (``[lane_d_ops]``)."""
    from ..shared.config import load_chain_config
    from ..shared.config.lane_d_ops import parse_lane_d_ops

    path = Path(chain_toml)
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    ops = parse_lane_d_ops(raw.get("lane_d_ops"), raw.get("storage"))
    storage = load_chain_config(path).storage
    return HippiusPoolStore(S3Config.from_storage(storage, bucket=ops.pool_d_bucket),
                            prefix=ops.pool_d_prefix)


# ── publish (builder) ───────────────────────────────────────────────────────


def _keys(prefix: str, snapshot: str) -> tuple[str, str, str]:
    base = f"{prefix.strip('/')}/{snapshot}"
    return f"{base}/manifest.json", f"{base}/hist.tar", f"{base}/tgt.tar"


def _index_key(prefix: str) -> str:
    return f"{prefix.strip('/')}/index.json"


def _get_json(store, key: str):
    try:
        return json.loads(store.get_bytes(key))
    except ObjectNotFound:
        return None


def _add_to_index(store, prefix: str, manifest: dict, manifest_key: str) -> None:
    index = _get_json(store, _index_key(prefix)) or {"schema": INDEX_SCHEMA, "snapshots": []}
    if any(e["snapshot"] == manifest["snapshot"] for e in index["snapshots"]):
        return
    index["snapshots"].append({"snapshot": manifest["snapshot"],
                               "as_of_ts": int(manifest["as_of_ts"]),
                               "manifest_key": manifest_key})
    index["snapshots"].sort(key=lambda e: (int(e["as_of_ts"]), e["snapshot"]))
    store.put_bytes(_index_key(prefix), _canonical(index))


def publish_snapshot(pool_dir: Path | str, store, *, prefix: str = DEFAULT_PREFIX,
                     build_wall_seconds: int | None = None) -> dict:
    """Upload one snapshot's two tars, its manifest and its index entry."""
    from .pool_d_split import split_pool_d

    meta = json.loads((Path(pool_dir) / "metadata.json").read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="pool-d-publish-") as tmp:
        res = split_pool_d(pool_dir, tmp)
        man_key, hist_key, tgt_key = _keys(prefix, res.snapshot)
        manifest = {
            "schema": MANIFEST_SCHEMA, "snapshot": res.snapshot,
            "as_of_ts": int(meta["as_of_ts"]), "sha256": res.snapshot_sha256,
            "hist_sha256": res.hist_sha256, "tgt_sha256": res.tgt_sha256,
            "hist_key": hist_key, "tgt_key": tgt_key,
            "catalog_sha256": meta.get("catalog_sha256"),
            "first_seen_roots": list(meta.get("first_seen_roots") or []),
            "n_series": len(meta.get("series") or []),
            "build_wall_seconds": None if build_wall_seconds is None else int(build_wall_seconds),
        }
        existing = _get_json(store, man_key)
        if existing is not None:
            same = all(existing.get(k) == manifest[k]
                       for k in ("sha256", "hist_sha256", "tgt_sha256"))
            if not same:
                raise PublishError("snapshot_conflict",
                                   f"{res.snapshot} is already published with sha256 "
                                   f"{existing.get('sha256')}, not {res.snapshot_sha256}")
            _add_to_index(store, prefix, existing, man_key)
            return existing
        store.put_file(hist_key, res.hist_tar)
        store.put_file(tgt_key, res.tgt_tar)
    store.put_bytes(man_key, _canonical(manifest))
    _add_to_index(store, prefix, manifest, man_key)
    return manifest


# ── pins (box) and history fetch (eval host) ────────────────────────────────


def _load_pins(pins_path: Path) -> dict:
    if not pins_path.is_file():
        return {"schema": PINS_SCHEMA, "pins": []}
    doc = json.loads(pins_path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or doc.get("schema") != PINS_SCHEMA \
            or not isinstance(doc.get("pins"), list):
        raise PublishError("bad_pins", f"{pins_path} is not a {PINS_SCHEMA} file")
    return doc


def _save_pins(pins_path: Path, doc: dict) -> None:
    doc["pins"].sort(key=lambda p: (int(p["as_of_ts"]), p["snapshot"]))
    _write_atomic(pins_path, (json.dumps(doc, indent=1, sort_keys=True) + "\n").encode())


def _fetch_checked(store, key: str, dest: Path, want: str) -> Path:
    store.get_file(key, dest)
    got = _file_sha(dest)
    if got != want:
        dest.unlink(missing_ok=True)
        raise PublishError("digest_mismatch", f"{key} hashes to {got}, manifest pins {want}")
    return dest


def sync_pins(store, *, prefix: str, local_root: Path | str, pins_path: Path | str,
              from_block: int) -> list[dict]:
    """Pin every indexed snapshot not yet pinned (box side)."""
    from .pool_d_build import snapshot_sha256
    from .pool_d_split import SplitError, join_split

    pins_path, root = Path(pins_path), Path(local_root)
    doc = _load_pins(pins_path)
    pinned = {p.get("snapshot") for p in doc["pins"]}
    index = _get_json(store, _index_key(prefix)) or {"snapshots": []}
    added = []
    for ent in sorted(index["snapshots"], key=lambda e: (int(e["as_of_ts"]), e["snapshot"])):
        if ent["snapshot"] in pinned:
            continue
        man = json.loads(store.get_bytes(ent["manifest_key"]))
        dest = root / man["snapshot"]
        with tempfile.TemporaryDirectory(prefix="pool-d-sync-") as tmp:
            hist = _fetch_checked(store, man["hist_key"], Path(tmp) / "hist.tar",
                                  man["hist_sha256"])
            tgt = _fetch_checked(store, man["tgt_key"], Path(tmp) / "tgt.tar", man["tgt_sha256"])
            try:
                join_split(hist, tgt, dest)
            except SplitError as e:
                raise PublishError("digest_mismatch", f"{man['snapshot']}: {e}") from e
        if snapshot_sha256(dest) != man["sha256"]:
            shutil.rmtree(dest, ignore_errors=True)
            raise PublishError("digest_mismatch", f"{dest} does not hash to {man['sha256']}")
        pin = {"dir": os.path.relpath(dest, pins_path.parent), "sha256": man["sha256"],
               "from_block": int(from_block), "snapshot": man["snapshot"],
               "as_of_ts": int(man["as_of_ts"]), "hist_sha256": man["hist_sha256"],
               "tgt_sha256": man["tgt_sha256"], "catalog_sha256": man.get("catalog_sha256")}
        doc["pins"].append(pin)
        _save_pins(pins_path, doc)
        added.append(pin)
    return added


def fetch_hist(store, *, prefix: str, snapshot: str, out_dir: Path | str) -> Path:
    """Eval host: the manifest and ``hist.tar`` only, digest-checked."""
    man_key, _hist_key, _tgt_key = _keys(prefix, snapshot)
    man = json.loads(store.get_bytes(man_key))
    dest = Path(out_dir) / f"{snapshot}.hist.tar"
    return _fetch_checked(store, man["hist_key"], dest, man["hist_sha256"])


# ── retention ───────────────────────────────────────────────────────────────


def _open_entries(state: Mapping) -> list[Mapping]:
    ents = list((state.get("entries") or {}).values())
    if state.get("house_entry"):
        ents.append(state["house_entry"])
    return [e for e in ents if isinstance(e, Mapping)
            and e.get("status") not in TERMINAL_STATUSES]


def _window_ids(pool_dir: Path) -> set[str]:
    meta = json.loads((pool_dir / "metadata.json").read_text(encoding="utf-8"))
    h = int(meta["horizon"])
    return {f"{s['series_id']}:{int(s['target_ts'][0])}:h{h}"
            for s in meta.get("series") or [] if s.get("target_ts")}


def retention_plan(pins_path: Path | str, state: Mapping, *,
                   embargo_hours: float = 48.0) -> dict:
    """``{"keep", "expire"}``: the snapshots open entries can still reference."""
    pins_path = Path(pins_path)
    pins = _load_pins(pins_path)["pins"]
    names = [p["snapshot"] for p in pins]
    if not pins:
        return {"keep": [], "expire": []}
    open_ = _open_entries(state)
    reveals = [e.get("reveal_ts") for e in open_]
    if any(isinstance(r, bool) or not isinstance(r, int) for r in reveals):
        return {"keep": names, "expire": []}
    keep = {max(pins, key=lambda p: (int(p["as_of_ts"]), p["snapshot"]))["snapshot"]}
    if reveals:
        cut = min(reveals) - round(float(embargo_hours) * 3600)
        keep |= {p["snapshot"] for p in pins if int(p["as_of_ts"]) > cut}
    frozen: set[str] = set()
    for e in open_:
        fs = e.get("frozen_set")
        if isinstance(fs, Mapping):
            frozen |= {str(w) for w in fs.get("window_ids") or []}
    for p in pins:
        if p["snapshot"] not in keep and frozen:
            d = pins_path.parent / p["dir"]
            if not (d / "metadata.json").is_file() or _window_ids(d) & frozen:
                keep.add(p["snapshot"])
    return {"keep": [n for n in names if n in keep], "expire": [n for n in names if n not in keep]}


def apply_retention(pins_path: Path | str, local_root: Path | str, plan: Mapping) -> list[str]:
    """Drop the expired pins and their directories (only under ``local_root``)."""
    pins_path = Path(pins_path)
    root = Path(local_root).resolve()
    doc = _load_pins(pins_path)
    expire = set(plan.get("expire") or [])
    gone = [p for p in doc["pins"] if p["snapshot"] in expire]
    for p in gone:
        d = (pins_path.parent / p["dir"]).resolve()
        if root not in d.parents:
            raise PublishError("outside_local_root", f"{d} is not under {root}")
    doc["pins"] = [p for p in doc["pins"] if p["snapshot"] not in expire]
    _save_pins(pins_path, doc)
    for p in gone:
        shutil.rmtree((pins_path.parent / p["dir"]).resolve(), ignore_errors=True)
    return [p["snapshot"] for p in gone]


# ── CLI ─────────────────────────────────────────────────────────────────────


def _evidence(store, prefix: str, last: int) -> dict:
    from .pool_d_build import snapshot_sha256
    from .pool_d_split import join_split

    index = _get_json(store, _index_key(prefix)) or {"snapshots": []}
    rows = []
    for ent in index["snapshots"][-int(last):]:
        man = json.loads(store.get_bytes(ent["manifest_key"]))
        with tempfile.TemporaryDirectory(prefix="pool-d-evidence-") as tmp:
            hist = _fetch_checked(store, man["hist_key"], Path(tmp) / "h.tar", man["hist_sha256"])
            tgt = _fetch_checked(store, man["tgt_key"], Path(tmp) / "t.tar", man["tgt_sha256"])
            joined = join_split(hist, tgt, Path(tmp) / "joined")
            verified = snapshot_sha256(joined) == man["sha256"]
        rows.append({"snapshot": man["snapshot"], "as_of_ts": int(man["as_of_ts"]),
                     "sha256": man["sha256"], "hist_sha256": man["hist_sha256"],
                     "tgt_sha256": man["tgt_sha256"],
                     "build_wall_seconds": man.get("build_wall_seconds"),
                     "verified": bool(verified)})
    gaps = [b["as_of_ts"] - a["as_of_ts"] for a, b in zip(rows, rows[1:], strict=False)]
    doc = {
        "schema": EVIDENCE_SCHEMA, "wp": "W6d", "source": "builder",
        "generated_by": "cascade.pool.pool_d_publish evidence",
        "host": socket.gethostname(), "generated_ts": int(time.time()),
        "bucket": store.bucket, "prefix": prefix, "snapshots": rows,
        "cadence_ok": len(rows) >= 2 and all(g == CADENCE_SECONDS for g in gaps),
        "walls_ok": bool(rows) and all(r["build_wall_seconds"] is not None
                                       and r["build_wall_seconds"] < MAX_BUILD_WALL_SECONDS
                                       for r in rows),
    }
    doc["digest"] = hashlib.sha256(_canonical(doc)).hexdigest()
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -P -m cascade.pool.pool_d_publish",
                                 description="pool_d pinning, retention and cadence (W6d)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def store_args(p):
        g = p.add_mutually_exclusive_group(required=True)
        g.add_argument("--chain-toml", type=Path)
        g.add_argument("--fs-store", type=Path)
        p.add_argument("--prefix", default=None)

    pub = sub.add_parser("publish")
    store_args(pub)
    pub.add_argument("--pool-d", type=Path, required=True)
    pub.add_argument("--started-ts", type=int, default=None)
    pub.add_argument("--finished-ts", type=int, default=None)
    syn = sub.add_parser("sync")
    store_args(syn)
    syn.add_argument("--pins", type=Path, required=True)
    syn.add_argument("--local-root", type=Path, required=True)
    syn.add_argument("--from-block", type=int, required=True)
    ret = sub.add_parser("retention")
    ret.add_argument("--pins", type=Path, required=True)
    ret.add_argument("--local-root", type=Path, required=True)
    ret.add_argument("--state", type=Path, required=True)
    ret.add_argument("--embargo-hours", type=float, default=48.0)
    ret.add_argument("--apply", action="store_true")
    ev = sub.add_parser("evidence")
    store_args(ev)
    ev.add_argument("--out", type=Path, required=True)
    ev.add_argument("--last", type=int, default=8)
    args = ap.parse_args(argv)

    try:
        if args.cmd == "retention":
            state = json.loads(args.state.read_text(encoding="utf-8"))
            plan = retention_plan(args.pins, state, embargo_hours=args.embargo_hours)
            print(json.dumps(plan, sort_keys=True))
            if args.apply:
                print(json.dumps({"removed": apply_retention(args.pins, args.local_root, plan)}))
            return 0
        if args.fs_store is not None:
            store, prefix = FsPoolStore(args.fs_store), args.prefix or DEFAULT_PREFIX
        else:
            store = open_pool_store(args.chain_toml)
            prefix = args.prefix or store.prefix
        if args.cmd == "publish":
            wall = None
            if args.started_ts is not None and args.finished_ts is not None:
                meta = json.loads((args.pool_d / "metadata.json").read_text(encoding="utf-8"))
                check = check_build_wall(as_of_ts=int(meta["as_of_ts"]),
                                         started_ts=args.started_ts, finished_ts=args.finished_ts)
                wall = check["wall_seconds"]
                print(json.dumps({"build_wall": check}, sort_keys=True))
            man = publish_snapshot(args.pool_d, store, prefix=prefix, build_wall_seconds=wall)
            print(json.dumps(man, sort_keys=True))
        elif args.cmd == "sync":
            added = sync_pins(store, prefix=prefix, local_root=args.local_root,
                              pins_path=args.pins, from_block=args.from_block)
            print(json.dumps({"pinned": [p["snapshot"] for p in added]}))
        else:
            doc = _evidence(store, prefix, args.last)
            _write_atomic(args.out, (json.dumps(doc, indent=2, sort_keys=True) + "\n").encode())
            print(f"wrote {args.out} ({len(doc['snapshots'])} snapshots, "
                  f"cadence_ok={doc['cadence_ok']}, walls_ok={doc['walls_ok']})")
    except (PublishError, StorageError, ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "CADENCE_SECONDS",
    "EVIDENCE_SCHEMA",
    "INDEX_SCHEMA",
    "MANIFEST_SCHEMA",
    "MAX_BUILD_WALL_SECONDS",
    "MULTIPART_CHUNK_BYTES",
    "PINS_SCHEMA",
    "TERMINAL_STATUSES",
    "FsPoolStore",
    "HippiusPoolStore",
    "PublishError",
    "apply_retention",
    "check_build_wall",
    "due_as_of",
    "fetch_hist",
    "main",
    "open_pool_store",
    "publish_snapshot",
    "retention_plan",
    "sync_pins",
]
