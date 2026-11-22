"""W18: the lane D vault TTL sweeper (LANE_D_PLAN §1.2 "Visibility and
retention", §6 D8 and D16, §3.0 W18 row).

Losers leave the private vault ``loser_ttl_eras`` after judgement, sit in an
owner-only archive for the dispute window, then are deleted::

    vault -> owner-only archive (sha256 read back) -> delete every vault version
    archive object older than archive_retention_days (and not disputed) -> delete

Hippius accepts lifecycle rules but does not apply them, so this is our own
job. Judgement never comes from storage: the sweeper reads a retention ledger
(``laned_vault_ledger/1``, a snapshot of lane D state) and keeps anything it
can't prove is due (typed keep reasons, fail closed). An entry is swept whole:
if one of its files is locked or its archive copy fails, none of its files
leaves the vault in that run. Re-runs resume: a verified archive copy is never
written twice.

A live run needs the owner's standing approval D16, read from the plan's
§17.4 decision register; ``--dry-run`` needs none and makes no storage write.

Usage::

    python -P -m cascade.laned.vault_sweeper --vault-dir D --archive-dir D --ledger F \\
        --config chain.toml --log-dir D (--dry-run | --live) [--plan P] [--now ISO] \\
        [--evidence-out F]

    python -P -m cascade.laned.vault_sweeper --s3 --archive-bucket B [--archive-prefix P] \\
        --ledger F --config chain.toml --log-dir D (--dry-run | --live) ...

The S3 mode takes the vault from ``[lane_d_ops] vault_bucket`` /
``vault_prefix`` and the endpoint from ``[storage] s3_endpoint`` /
``s3_region``; credentials come from ``HIPPIUS_S3_ACCESS_KEY`` /
``HIPPIUS_S3_SECRET_KEY`` (env only, never argv, never logged). The fs mode
imports no boto3 and opens no connection.

Exit codes: 0 clean, 1 an action failed, 2 refused or usage error.
Contract: ``tests/accept/w18/README.md``.
"""

from __future__ import annotations

import argparse
import contextlib
import errno
import hashlib
import json
import os
import re
import secrets
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

MODULE = "cascade.laned.vault_sweeper"
LOG_SCHEMA = "laned_vault_sweep/1"
LEDGER_SCHEMA = "laned_vault_ledger/1"
EVIDENCE_SCHEMA = "laned_w18_evidence/1"
WP = "W18"

ARCHIVE_RETENTION_DAYS = 14          # D8: owner-only archive for the 14-day dispute window
REPO = Path(__file__).resolve().parents[2]
DEFAULT_PLAN = REPO / "docs" / "plans" / "LANE_D_PLAN.md"
DEFAULT_ARCHIVE_PREFIX = "lane-d-archive"

OPEN_STATUSES = frozenset({"submitted", "ready", "revealed", "pending"})
SETTLEMENT_STATUSES = frozenset({"judging"})
KING_STATUSES = frozenset({"king"})
LOSER_STATUSES = frozenset({"lost", "closed_unjudged", "ready_expired", "voided"})

REGISTER_HEADER = ("D", "Status", "Option chosen", "Class", "Decided")
DECISION = "D16"

ACCESS_ENV = "HIPPIUS_S3_ACCESS_KEY"
SECRET_ENV = "HIPPIUS_S3_SECRET_KEY"

ARCHIVED_AT_META = "laned-archived-at"   # S3 user metadata: the sweeper's write time
META_DIR = ".laned-meta"                 # FsStore's own metadata, never listed as objects

CHUNK = 8 * 1024 * 1024
PART_BYTES = 64 * 1024 * 1024            # Hippius: parts <= 512 MiB, <= 10,000 parts

_HEX64 = re.compile(r"[0-9a-f]{64}")
_NO_LOCK_CODES = frozenset({"NoSuchObjectLockConfiguration", "ObjectLockConfigurationNotFoundError",
                            "InvalidRequest"})
_MISSING_CODES = frozenset({"NoSuchKey", "NotFound", "404", "NoSuchVersion"})


# ── errors ───────────────────────────────────────────────────────────────────


class SweepRefused(Exception):
    """The run is refused before any storage change; ``code`` is typed."""

    def __init__(self, code: str, message: str = ""):
        super().__init__(f"{code}: {message}" if message else code)
        self.code = code


class ObjectLocked(Exception):
    """An adapter's ``delete`` on an object under unexpired retention."""

    def __init__(self, key: str, retain_until: datetime):
        super().__init__(f"{key} is under retention until {_iso(retain_until)}")
        self.key = key
        self.retain_until = retain_until


class StoreError(Exception):
    """A storage call failed; ``code`` is a short, secret-free error code."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


# ── time and JSON ────────────────────────────────────────────────────────────


def _iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat()


def _aware(dt: Any) -> datetime | None:
    if isinstance(dt, datetime):
        return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)
    if isinstance(dt, str) and dt:
        try:
            parsed = datetime.fromisoformat(dt.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
    return None


def _no_constant(name: str):
    raise ValueError(f"non-strict JSON constant {name}")


def _strict_loads(text: str):
    return json.loads(text, parse_constant=_no_constant)


def _dumps(obj: Any) -> str:
    return json.dumps(obj, allow_nan=False, sort_keys=False, indent=1)


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _err_code(exc: BaseException) -> str:
    """A short, secret-free code for a failed storage call (a botocore-shaped
    ``.response`` code, an errno name, or the exception class)."""
    if isinstance(exc, StoreError):
        return exc.code
    resp = getattr(exc, "response", None)
    if isinstance(resp, dict):
        code = (resp.get("Error") or {}).get("Code")
        if code:
            return str(code)
        status = (resp.get("ResponseMetadata") or {}).get("HTTPStatusCode")
        if status:
            return str(status)
    if isinstance(exc, OSError) and exc.errno is not None:
        return errno.errorcode.get(exc.errno, type(exc).__name__)
    return type(exc).__name__


def _client_code(exc: BaseException) -> str | None:
    resp = getattr(exc, "response", None)
    if not isinstance(resp, dict):
        return None
    code = (resp.get("Error") or {}).get("Code")
    status = (resp.get("ResponseMetadata") or {}).get("HTTPStatusCode")
    if code:
        return str(code)
    return str(status) if status else None


def _is_missing(exc: BaseException) -> bool:
    resp = getattr(exc, "response", None)
    if not isinstance(resp, dict):
        return False
    status = (resp.get("ResponseMetadata") or {}).get("HTTPStatusCode")
    return _client_code(exc) in _MISSING_CODES or status == 404


# ── the policy and the D16 gate ──────────────────────────────────────────────


@dataclass(frozen=True)
class SweepPolicy:
    """``loser_ttl_eras``: a loser is due when ``era - closed_era >=`` this
    (``[lane_d] loser_ttl_eras``, D8). ``archive_retention_days``: the
    owner-only archive's dispute window (D8, 14)."""

    loser_ttl_eras: int
    archive_retention_days: int = ARCHIVE_RETENTION_DAYS

    def __post_init__(self):
        for name in ("loser_ttl_eras", "archive_retention_days"):
            v = getattr(self, name)
            if type(v) is not int or v < 0:
                raise ValueError(f"SweepPolicy.{name} must be an int >= 0; got {v!r}")

    def as_dict(self) -> dict:
        return {"loser_ttl_eras": self.loser_ttl_eras,
                "archive_retention_days": self.archive_retention_days}


def policy_from_config(cfg: Any) -> SweepPolicy:
    """The policy from a loaded ``ChainConfig``: the TTL is ``[lane_d]
    loser_ttl_eras`` (overlay-settable), the archive window is D8's."""
    return SweepPolicy(loser_ttl_eras=cfg.lane_d.loser_ttl_eras)


def _cells(line: str) -> list[str] | None:
    s = line.strip()
    if not s.startswith("|"):
        return None
    s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


def d16_status(plan_path: Path | str | None) -> str | None:
    """The ``Status`` cell of the D16 row in the plan's §17.4 decision register
    (the table headed ``| D | Status | Option chosen | Class | Decided |``),
    stripped; None if the file, the table or the row is missing. No other
    table (the §6 recommendations) is ever read as a decision."""
    if plan_path is None:
        return None
    try:
        text = Path(plan_path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    lines = text.splitlines()
    for i, line in enumerate(lines):
        cells = _cells(line)
        if cells is None or tuple(cells) != REGISTER_HEADER:
            continue
        for row in lines[i + 1:]:
            rc = _cells(row)
            if rc is None:
                break
            if rc and rc[0] == DECISION and len(rc) >= 2:
                return rc[1]
    return None


# ── the ledger ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LedgerEntry:
    hotkey: str
    digest: str
    status: str
    closed_era: int | None


@dataclass(frozen=True)
class Ledger:
    era: int
    entries: dict[tuple[str, str], LedgerEntry]
    disputes: frozenset[str]


def _bad_ledger(msg: str) -> SweepRefused:
    return SweepRefused("ledger_invalid", msg)


def _is_int(v: Any) -> bool:
    return type(v) is int


def parse_ledger(raw: Any) -> Ledger:
    """Validate a ``laned_vault_ledger/1`` object; ``SweepRefused("ledger_invalid")``
    on any defect."""
    if not isinstance(raw, dict):
        raise _bad_ledger("the ledger is not an object")
    if raw.get("schema") != LEDGER_SCHEMA:
        raise _bad_ledger(f"schema is not {LEDGER_SCHEMA}")
    era = raw.get("era")
    if not _is_int(era) or era < 0:
        raise _bad_ledger("era must be an int >= 0")
    rows = raw.get("entries")
    if not isinstance(rows, list):
        raise _bad_ledger("entries must be a list")
    disputes = raw.get("disputes")
    if not isinstance(disputes, list):
        raise _bad_ledger("disputes must be a list")
    entries: dict[tuple[str, str], LedgerEntry] = {}
    for n, row in enumerate(rows):
        if not isinstance(row, dict):
            raise _bad_ledger(f"entry {n} is not an object")
        hk, dg, st, ce = (row.get("hotkey"), row.get("digest"), row.get("status"),
                          row.get("closed_era"))
        if not isinstance(hk, str) or not hk or "/" in hk or hk != hk.strip():
            raise _bad_ledger(f"entry {n}: bad hotkey")
        if not isinstance(dg, str) or not _HEX64.fullmatch(dg):
            raise _bad_ledger(f"entry {n}: digest must be 64 lowercase hex")
        if not isinstance(st, str) or not st:
            raise _bad_ledger(f"entry {n}: bad status")
        if ce is not None and (not _is_int(ce) or ce < 0):
            raise _bad_ledger(f"entry {n}: closed_era must be an int >= 0 or null")
        if st in LOSER_STATUSES and not _is_int(ce):
            raise _bad_ledger(f"entry {n}: loser status {st!r} needs an int closed_era")
        if (hk, dg) in entries:
            raise _bad_ledger(f"entry {n}: duplicate entry")
        entries[(hk, dg)] = LedgerEntry(hk, dg, st, ce)
    for n, d in enumerate(disputes):
        if not isinstance(d, str) or not _HEX64.fullmatch(d):
            raise _bad_ledger(f"dispute {n} must be a 64 lowercase hex digest")
    return Ledger(era=era, entries=entries, disputes=frozenset(disputes))


def split_key(key: str) -> tuple[str, str, str] | None:
    """``<hotkey>/<digest>/<file>`` -> (hotkey, digest, file), else None."""
    parts = key.split("/", 2)
    if len(parts) != 3:
        return None
    hk, dg, name = parts
    if not hk or not _HEX64.fullmatch(dg) or not name:
        return None
    if any(seg in ("", ".", "..") for seg in name.split("/")):
        return None
    return hk, dg, name


# ── storage adapters ─────────────────────────────────────────────────────────


def _chunks_hashed(chunks: Iterator[bytes], h) -> Iterator[bytes]:
    for c in chunks:
        h.update(c)
        yield c


class FsStore:
    """A filesystem store: the object ``key`` is the regular file
    ``root/<key>``. Lock state and archive times live only under
    ``root/.laned-meta/``, never listed. Everything this store writes is
    owner-only (dirs 0700, files 0600) whatever the umask."""

    kind = "fs"

    def __init__(self, root: Path | str):
        self.root = Path(root)

    def describe(self) -> dict:
        return {"kind": "fs", "root": str(self.root)}

    # paths
    def _path(self, key: str) -> Path:
        if not key or key.startswith("/") or any(s in ("", ".", "..") for s in key.split("/")):
            raise StoreError("bad_key")
        if key.split("/", 1)[0] == META_DIR:
            raise StoreError("bad_key")
        return self.root / key

    def _meta_path(self, key: str) -> Path:
        return self.root / META_DIR / "objects" / f"{key}.json"

    def _read_meta(self, key: str) -> dict:
        try:
            m = _strict_loads(self._meta_path(key).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return m if isinstance(m, dict) else {}

    def _ensure_dirs(self, d: Path) -> None:
        """Create ``d`` (under root) with every new directory 0700."""
        rel = d.relative_to(self.root)
        cur = self.root
        for part in (None, *rel.parts):
            if part is not None:
                cur = cur / part
            if not cur.exists():
                cur.mkdir(mode=0o700)
            elif not cur.is_dir() or cur.is_symlink():
                raise StoreError("ENOTDIR")
            os.chmod(cur, 0o700)

    def _write_file(self, target: Path, chunks: Iterator[bytes]) -> None:
        self._ensure_dirs(target.parent)
        tmpdir = self.root / META_DIR / "tmp"
        self._ensure_dirs(tmpdir)
        tmp = tmpdir / f"{secrets.token_hex(8)}.part"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "wb") as fh:
                for c in chunks:
                    fh.write(c)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, target)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        os.chmod(target, 0o600)

    def _write_meta(self, key: str, meta: dict) -> None:
        mp = self._meta_path(key)
        if not meta:
            mp.unlink(missing_ok=True)
            return
        data = json.dumps(meta, allow_nan=False, sort_keys=True).encode()
        self._write_file(mp, iter((data,)))

    # the store surface
    def list_keys(self) -> list[str]:
        if not self.root.is_dir():
            return []
        out = []
        for dirpath, dirnames, filenames in os.walk(self.root):
            d = Path(dirpath)
            if d == self.root:
                dirnames[:] = [n for n in dirnames if n != META_DIR]
            dirnames[:] = [n for n in dirnames if not (d / n).is_symlink()]
            for name in filenames:
                p = d / name
                if p.is_symlink() or not p.is_file():
                    continue
                out.append(p.relative_to(self.root).as_posix())
        return sorted(out)

    def exists(self, key: str) -> bool:
        p = self._path(key)
        return p.is_file() and not p.is_symlink()

    def iter_bytes(self, key: str) -> Iterator[bytes]:
        path = self._path(key)
        try:
            with open(path, "rb") as fh:
                while True:
                    c = fh.read(CHUNK)
                    if not c:
                        return
                    yield c
        except OSError as e:
            raise StoreError(_err_code(e)) from None

    def sha256(self, key: str) -> str | None:
        if not self.exists(key):
            return None
        h = hashlib.sha256()
        for c in self.iter_bytes(key):
            h.update(c)
        return h.hexdigest()

    def locked_until(self, key: str, now: datetime) -> datetime | None:
        m = self._read_meta(key)
        ru = _aware(m.get("retain_until"))
        if m.get("lock_mode") and ru is not None and ru > now:
            return ru
        return None

    def archived_at(self, key: str) -> datetime | None:
        return _aware(self._read_meta(key).get("archived_at"))

    def put(self, key: str, data: bytes, *, lock_mode: str | None = None,
            retain_until: datetime | None = None) -> None:
        """Write an object, optionally under retention (``lock_mode`` +
        ``retain_until``)."""
        self._write_file(self._path(key), iter((bytes(data),)))
        meta = {}
        if lock_mode and retain_until is not None:
            meta = {"lock_mode": str(lock_mode), "retain_until": _iso(retain_until)}
        self._write_meta(key, meta)

    def write_archive(self, key: str, chunks: Iterator[bytes], archived_at: datetime) -> None:
        self._write_file(self._path(key), chunks)
        meta = self._read_meta(key)
        meta["archived_at"] = _iso(archived_at)
        self._write_meta(key, meta)

    def _prune(self, start: Path, stop: Path) -> None:
        d = start
        while d != stop and stop in d.parents:
            try:
                d.rmdir()
            except OSError:
                return
            d = d.parent

    def delete(self, key: str, *, now: datetime) -> None:
        """Remove the object; ``ObjectLocked`` while ``retain_until > now``."""
        ru = self.locked_until(key, now)
        if ru is not None:
            raise ObjectLocked(key, ru)
        p = self._path(key)
        try:
            p.unlink(missing_ok=True)
        except OSError as e:
            raise StoreError(_err_code(e)) from None
        mp = self._meta_path(key)
        mp.unlink(missing_ok=True)
        self._prune(p.parent, self.root)
        self._prune(mp.parent, self.root / META_DIR / "objects")

    def overlaps(self, other: Any) -> bool:
        if not isinstance(other, FsStore):
            return False
        a, b = self.root.resolve(), other.root.resolve()
        return a == b or a in b.parents or b in a.parents


class S3Store:
    """A boto3-shaped store over ``bucket`` / ``prefix``. Keys are relative to
    the prefix. Archive writes carry ``ACL="private"`` and no grant; deletes
    remove every version by ``VersionId`` and never bypass governance."""

    kind = "s3"

    def __init__(self, client: Any, bucket: str, prefix: str):
        self.client = client
        self.bucket = bucket
        self.prefix = prefix

    def describe(self) -> dict:
        return {"kind": "s3", "bucket": self.bucket, "prefix": self.prefix}

    @property
    def _base(self) -> str:
        return self.prefix.strip("/")

    def _full(self, key: str) -> str:
        return f"{self._base}/{key}" if self._base else key

    def _rel(self, full: str) -> str | None:
        if not self._base:
            return full
        pre = self._base + "/"
        return full[len(pre):] if full.startswith(pre) else None

    def _call(self, op: str, **kw):
        try:
            return getattr(self.client, op)(**kw)
        except Exception as e:  # botocore ClientError and transport errors alike
            raise StoreError(_err_code(e)) from None

    def list_keys(self) -> list[str]:
        kw = {"Bucket": self.bucket}
        if self._base:
            kw["Prefix"] = self._base + "/"
        out = []
        try:
            for page in self.client.get_paginator("list_objects_v2").paginate(**kw):
                for o in page.get("Contents") or []:
                    rel = self._rel(o.get("Key", ""))
                    if rel:
                        out.append(rel)
        except Exception as e:
            raise StoreError(_err_code(e)) from None
        return sorted(set(out))

    def _versions(self, key: str) -> list[tuple[str, bool]]:
        """(VersionId, is_delete_marker) of every version of exactly ``key``."""
        full = self._full(key)
        out = []
        try:
            for page in self.client.get_paginator("list_object_versions").paginate(
                    Bucket=self.bucket, Prefix=full):
                for v in page.get("Versions") or []:
                    if v.get("Key") == full:
                        out.append((v.get("VersionId"), False))
                for v in page.get("DeleteMarkers") or []:
                    if v.get("Key") == full:
                        out.append((v.get("VersionId"), True))
        except Exception as e:
            raise StoreError(_err_code(e)) from None
        return out

    def _head(self, key: str, version: str | None = None) -> dict | None:
        kw = {"Bucket": self.bucket, "Key": self._full(key)}
        if version is not None:
            kw["VersionId"] = version
        try:
            return self.client.head_object(**kw)
        except Exception as e:
            if _is_missing(e):
                return None
            raise StoreError(_err_code(e)) from None

    def exists(self, key: str) -> bool:
        return self._head(key) is not None

    def _version_retain_until(self, key: str, version: str) -> datetime | None:
        full = self._full(key)
        try:
            r = self.client.get_object_retention(Bucket=self.bucket, Key=full, VersionId=version)
        except Exception as e:
            if _client_code(e) in _NO_LOCK_CODES:
                return None
            # Retention unreadable: fall back to the version's head (fail closed
            # if that fails too: StoreError propagates).
            h = self._head(key, version)
            if h is None or not h.get("ObjectLockMode"):
                return None
            return _aware(h.get("ObjectLockRetainUntilDate"))
        ret = (r or {}).get("Retention") or {}
        if not ret.get("Mode"):
            return None
        return _aware(ret.get("RetainUntilDate"))

    def locked_until(self, key: str, now: datetime) -> datetime | None:
        latest = None
        for vid, marker in self._versions(key):
            if marker or vid is None:
                continue
            ru = self._version_retain_until(key, vid)
            if ru is not None and ru > now and (latest is None or ru > latest):
                latest = ru
        return latest

    def iter_bytes(self, key: str) -> Iterator[bytes]:
        resp = self._call("get_object", Bucket=self.bucket, Key=self._full(key))
        body = resp["Body"]
        try:
            while True:
                try:
                    c = body.read(CHUNK)
                except Exception as e:
                    raise StoreError(_err_code(e)) from None
                if not c:
                    return
                yield c
        finally:
            close = getattr(body, "close", None)
            if close is not None:
                close()

    def sha256(self, key: str) -> str | None:
        if not self.exists(key):
            return None
        h = hashlib.sha256()
        for c in self.iter_bytes(key):
            h.update(c)
        return h.hexdigest()

    def archived_at(self, key: str) -> datetime | None:
        h = self._head(key)
        if h is None:
            return None
        meta = {str(k).lower(): v for k, v in (h.get("Metadata") or {}).items()}
        return _aware(meta.get(ARCHIVED_AT_META))

    def write_archive(self, key: str, chunks: Iterator[bytes], archived_at: datetime) -> None:
        full = self._full(key)
        meta = {ARCHIVED_AT_META: _iso(archived_at)}
        buf = bytearray()
        upload_id = None
        parts: list[dict] = []
        try:
            for c in chunks:
                buf += c
                while len(buf) >= PART_BYTES:
                    if upload_id is None:
                        upload_id = self._call("create_multipart_upload", Bucket=self.bucket,
                                               Key=full, ACL="private",
                                               Metadata=meta)["UploadId"]
                    part, buf = bytes(buf[:PART_BYTES]), bytearray(buf[PART_BYTES:])
                    n = len(parts) + 1
                    r = self._call("upload_part", Bucket=self.bucket, Key=full,
                                   UploadId=upload_id, PartNumber=n, Body=part)
                    parts.append({"ETag": r["ETag"], "PartNumber": n})
            if upload_id is None:
                self._call("put_object", Bucket=self.bucket, Key=full, Body=bytes(buf),
                           ACL="private", Metadata=meta)
                return
            if buf:
                n = len(parts) + 1
                r = self._call("upload_part", Bucket=self.bucket, Key=full,
                               UploadId=upload_id, PartNumber=n, Body=bytes(buf))
                parts.append({"ETag": r["ETag"], "PartNumber": n})
            self._call("complete_multipart_upload", Bucket=self.bucket, Key=full,
                       UploadId=upload_id, MultipartUpload={"Parts": parts})
            upload_id = None
        finally:
            if upload_id is not None:
                # The upload is abandoned either way; an abort error never masks the cause.
                with contextlib.suppress(Exception):
                    self.client.abort_multipart_upload(Bucket=self.bucket, Key=full,
                                                       UploadId=upload_id)

    def delete(self, key: str, *, now: datetime) -> None:
        """Delete every version (and delete marker) by VersionId; ObjectLocked
        (no delete call made) if any version is under unexpired retention."""
        ru = self.locked_until(key, now)
        if ru is not None:
            raise ObjectLocked(key, ru)
        full = self._full(key)
        for vid, _marker in self._versions(key):
            if vid is None:
                raise StoreError("unversioned_object")
            self._call("delete_object", Bucket=self.bucket, Key=full, VersionId=vid)
        if self._versions(key):
            raise StoreError("versions_remain")

    def overlaps(self, other: Any) -> bool:
        if not isinstance(other, S3Store) or other.bucket != self.bucket:
            return False
        a, b = self._base, other._base
        if not a or not b:
            return True
        return (a + "/").startswith(b + "/") or (b + "/").startswith(a + "/")


# ── the sweep ────────────────────────────────────────────────────────────────


def _rec(store: str, key: str, op: str, outcome: str, reason: str | None, **extra) -> dict:
    r = {"store": store, "key": key, "op": op, "outcome": outcome, "reason": reason}
    r.update(extra)
    return r


def _counts(actions: list[dict]) -> dict:
    c = {"archived": 0, "deleted": 0, "expired": 0, "kept": 0, "failed": 0, "planned": 0}
    for a in actions:
        if a["op"] == "archive" and a["outcome"] == "done":
            c["archived"] += 1
        if a["op"] == "delete" and a["outcome"] == "done":
            c["deleted" if a["store"] == "vault" else "expired"] += 1
        if a["op"] == "keep":
            c["kept"] += 1
        if a["outcome"] == "failed":
            c["failed"] += 1
        if a["outcome"] == "planned":
            c["planned"] += 1
    return c


def _entry_reason(entry: LedgerEntry | None, led: Ledger, policy: SweepPolicy) -> str | None:
    """The keep reason for an entry's files from the ledger alone, or None if
    the entry is due (subject to Object Lock and the archive copy)."""
    if entry is None:
        return "unreferenced"
    if entry.status in OPEN_STATUSES:
        return "open_entry"
    if entry.status in SETTLEMENT_STATUSES:
        return "settlement_open"
    if entry.status in KING_STATUSES:
        return "king"
    if entry.status not in LOSER_STATUSES:
        return "unknown_status"
    if entry.digest in led.disputes:
        return "dispute_open"
    if led.era - int(entry.closed_era) < policy.loser_ttl_eras:
        return "ttl_not_expired"
    return None


class _Run:
    def __init__(self, vault, archive, led: Ledger, policy: SweepPolicy, now: datetime,
                 live: bool):
        self.vault, self.archive, self.led = vault, archive, led
        self.policy, self.now, self.live = policy, now, live
        self.actions: list[dict] = []

    def add(self, *a, **kw) -> None:
        self.actions.append(_rec(*a, **kw))

    # vault phase
    def _archive_one(self, key: str) -> tuple[bool, str | None]:
        """Archive ``key`` (live) or plan it (dry-run). Returns (ok, sha256)."""
        try:
            existing = self.archive.exists(key)
            src = None
            if existing:
                src = self.vault.sha256(key)
                if src is not None and self.archive.sha256(key) == src:
                    self.add("vault", key, "archive", "already", None, sha256=src)
                    return True, src
            if not self.live:
                src = src or self.vault.sha256(key)
                if src is None:
                    raise StoreError("vault_object_missing")
                self.add("vault", key, "archive", "planned", None, sha256=src)
                return True, src
            h = hashlib.sha256()
            self.archive.write_archive(key, _chunks_hashed(self.vault.iter_bytes(key), h),
                                       self.now)
            src = h.hexdigest()
        except (StoreError, OSError) as e:
            self.add("vault", key, "archive", "failed", "archive_copy_failed",
                     error=_err_code(e))
            return False, None
        try:
            back = self.archive.sha256(key)
        except (StoreError, OSError) as e:
            self.add("vault", key, "archive", "failed", "archive_verify_failed",
                     error=_err_code(e))
            return False, None
        if back != src:
            self.add("vault", key, "archive", "failed", "archive_verify_failed")
            return False, None
        self.add("vault", key, "archive", "done", None, sha256=src)
        return True, src

    def sweep_entry(self, keys: list[str]) -> list[str]:
        """Sweep one due entry whole; returns the vault keys deleted."""
        locked = {}
        for k in keys:
            ru = self.vault.locked_until(k, self.now)
            if ru is not None:
                locked[k] = ru
        if locked:
            for k in keys:
                if k in locked:
                    self.add("vault", k, "keep", "kept", "object_lock",
                             retain_until=_iso(locked[k]))
                else:
                    self.add("vault", k, "keep", "kept", "entry_incomplete")
            return []
        done = [k for k in keys if self._archive_one(k)[0]]
        if len(done) != len(keys):
            for k in done:
                self.add("vault", k, "keep", "kept", "entry_incomplete")
            return []
        deleted = []
        for k in keys:
            if not self.live:
                self.add("vault", k, "delete", "planned", None)
                continue
            try:
                self.vault.delete(k, now=self.now)
            except ObjectLocked as e:
                self.add("vault", k, "delete", "failed", "object_lock",
                         retain_until=_iso(e.retain_until))
                continue
            except (StoreError, OSError) as e:
                self.add("vault", k, "delete", "failed", "delete_failed", error=_err_code(e))
                continue
            self.add("vault", k, "delete", "done", None)
            deleted.append(k)
        return deleted

    def vault_phase(self, vault_keys: list[str]) -> set[str]:
        groups: dict[tuple[str, str], list[str]] = {}
        for k in vault_keys:
            parts = split_key(k)
            entry = self.led.entries.get((parts[0], parts[1])) if parts else None
            reason = _entry_reason(entry, self.led, self.policy)
            if reason is not None:
                self.add("vault", k, "keep", "kept", reason)
                continue
            groups.setdefault((parts[0], parts[1]), []).append(k)
        deleted: set[str] = set()
        for ident in sorted(groups):
            deleted.update(self.sweep_entry(sorted(groups[ident])))
        return deleted

    # archive phase
    def archive_phase(self, archive_keys: list[str], vault_left: set[str]) -> None:
        window = timedelta(days=self.policy.archive_retention_days)
        for k in archive_keys:
            parts = split_key(k)
            if parts is None:
                self.add("archive", k, "keep", "kept", "unreferenced")
                continue
            if k in vault_left:
                # Its vault copy is still there: the archive copy is the only
                # verified one the next vault run builds on.
                self.add("archive", k, "keep", "kept", "entry_incomplete")
                continue
            if parts[1] in self.led.disputes:
                self.add("archive", k, "keep", "kept", "dispute_open")
                continue
            at = self.archive.archived_at(k)
            if at is None:
                self.add("archive", k, "keep", "kept", "archive_age_unknown")
                continue
            if self.now - at <= window:
                self.add("archive", k, "keep", "kept", "archive_retention_not_expired")
                continue
            ru = self.archive.locked_until(k, self.now)
            if ru is not None:
                self.add("archive", k, "keep", "kept", "object_lock", retain_until=_iso(ru))
                continue
            if not self.live:
                self.add("archive", k, "delete", "planned", None)
                continue
            try:
                self.archive.delete(k, now=self.now)
            except ObjectLocked as e:
                self.add("archive", k, "delete", "failed", "object_lock",
                         retain_until=_iso(e.retain_until))
                continue
            except (StoreError, OSError) as e:
                self.add("archive", k, "delete", "failed", "delete_failed", error=_err_code(e))
                continue
            self.add("archive", k, "delete", "done", None)


def _check_store(store: Any, side: str) -> None:
    if not isinstance(store, (FsStore, S3Store)):
        raise TypeError(f"{side} must be an FsStore or S3Store, got {type(store).__name__}")


def sweep(*, vault: Any, archive: Any, ledger: Any, policy: SweepPolicy, now: datetime,
          live: bool, plan_path: Path | str | None, log_dir: Path | str) -> dict:
    """One sweeper run. Returns the ``laned_vault_sweep/1`` action log and
    writes it to ``log_dir/vault-sweep-<run_id>.json``.

    Refusals (``SweepRefused``, no storage call): ``ledger_invalid``,
    ``d16_not_decided`` (live only), ``archive_overlaps_vault``."""
    started = datetime.now(UTC)
    _check_store(vault, "vault")
    _check_store(archive, "archive")
    if not isinstance(policy, SweepPolicy):
        raise TypeError("policy must be a SweepPolicy")
    if not isinstance(now, datetime) or now.tzinfo is None:
        raise ValueError("now must be a timezone-aware datetime")
    led = parse_ledger(ledger)
    d16 = d16_status(plan_path) if plan_path is not None else None
    if live and d16 != "decided":
        raise SweepRefused("d16_not_decided",
                           "a live run needs D16 decided in the plan's §17.4 register")
    if vault.overlaps(archive) or archive.overlaps(vault):
        raise SweepRefused("archive_overlaps_vault")

    run = _Run(vault, archive, led, policy, now, bool(live))
    vault_keys = vault.list_keys()
    archive_keys = archive.list_keys()
    deleted = run.vault_phase(vault_keys)
    run.archive_phase(archive_keys, set(vault_keys) - deleted)

    log = {
        "schema": LOG_SCHEMA,
        "run_id": secrets.token_hex(8),
        "mode": "live" if live else "dry-run",
        "now": _iso(now),
        "started_at": _iso(started),
        "finished_at": _iso(datetime.now(UTC)),
        "d16": d16,
        "policy": policy.as_dict(),
        "ledger_era": led.era,
        "vault": vault.describe(),
        "archive": archive.describe(),
        "actions": run.actions,
        "counts": _counts(run.actions),
    }
    text = _dumps(log)
    _write_text_atomic(Path(log_dir) / f"vault-sweep-{log['run_id']}.json", text + "\n")
    return log


def evidence(log: dict) -> dict:
    """The ``laned_w18_evidence/1`` record of a live run's log."""
    failed = any(a["outcome"] == "failed" for a in log["actions"])
    return {"schema": EVIDENCE_SCHEMA, "wp": WP, "passed": not failed, "log": log}


# ── CLI ──────────────────────────────────────────────────────────────────────


class _Usage(Exception):
    pass


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog=f"python -P -m {MODULE}",
                                 description=__doc__.split("\n", 1)[0])
    ap.add_argument("--vault-dir", help="fs mode: the vault root")
    ap.add_argument("--archive-dir", help="fs mode: the owner-only archive root")
    ap.add_argument("--s3", action="store_true",
                    help="S3 mode: the vault from [lane_d_ops], the endpoint from [storage]")
    ap.add_argument("--archive-bucket", help="S3 mode: the owner-only archive bucket")
    ap.add_argument("--archive-prefix", default=DEFAULT_ARCHIVE_PREFIX,
                    help="S3 mode: the archive prefix (default %(default)s)")
    ap.add_argument("--ledger", required=True, help="the laned_vault_ledger/1 JSON file")
    ap.add_argument("--config", required=True, help="chain.toml (load_chain_config)")
    ap.add_argument("--log-dir", required=True)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--live", action="store_true")
    ap.add_argument("--plan", default=None,
                    help=f"the plan holding the §17.4 register (default {DEFAULT_PLAN})")
    ap.add_argument("--now", default=None, help="ISO 8601 with timezone (default: now, UTC)")
    ap.add_argument("--evidence-out", default=None, help="live only: write laned_w18_evidence/1")
    return ap


def _parse_now(s: str | None) -> datetime:
    if s is None:
        return datetime.now(UTC)
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        raise _Usage(f"--now {s!r} is not ISO 8601") from None
    if dt.tzinfo is None:
        raise _Usage("--now needs a timezone")
    return dt


def _s3_client(storage: Any):
    import boto3  # lazy: the fs mode never imports boto3
    from botocore.config import Config

    access, secret = os.environ.get(ACCESS_ENV), os.environ.get(SECRET_ENV)
    if not access or not secret:
        raise SweepRefused("credentials_missing", f"set {ACCESS_ENV} / {SECRET_ENV}")
    cfg = Config(signature_version="s3v4", s3={"addressing_style": "path"},
                 connect_timeout=30, read_timeout=600,
                 retries={"max_attempts": 3, "mode": "standard"},
                 request_checksum_calculation="when_required",
                 response_checksum_validation="when_required")
    return boto3.client("s3", endpoint_url=storage.s3_endpoint, region_name=storage.s3_region,
                        aws_access_key_id=access, aws_secret_access_key=secret, config=cfg)


def _stores(args: argparse.Namespace, cfg: Any, client_factory: Callable[[Any], Any]):
    fs = args.vault_dir is not None or args.archive_dir is not None
    if fs and args.s3:
        raise _Usage("give either --vault-dir/--archive-dir or --s3, not both")
    if fs:
        if args.vault_dir is None or args.archive_dir is None:
            raise _Usage("the fs mode needs both --vault-dir and --archive-dir")
        return FsStore(Path(args.vault_dir)), FsStore(Path(args.archive_dir))
    if not args.s3:
        raise _Usage("give --vault-dir/--archive-dir (fs) or --s3")
    if not args.archive_bucket:
        raise _Usage("--s3 needs --archive-bucket (the owner-only archive)")
    from cascade.shared.config.lane_d_ops import lane_d_ops_config
    ops = lane_d_ops_config(cfg)
    client = client_factory(cfg.storage)
    return (S3Store(client, ops.vault_bucket, ops.vault_prefix),
            S3Store(client, args.archive_bucket, args.archive_prefix))


def main(argv: list[str] | None = None, *,
         client_factory: Callable[[Any], Any] = _s3_client) -> int:
    args = _parser().parse_args(argv)

    def refused(code: str) -> int:
        print(json.dumps({"refused": code}))
        return 2

    try:
        if args.evidence_out and not args.live:
            raise _Usage("--evidence-out is for --live runs only")
        now = _parse_now(args.now)
        try:
            ledger_raw = _strict_loads(Path(args.ledger).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, ValueError):
            return refused("ledger_invalid")
        from cascade.shared.config import load_chain_config
        cfg = load_chain_config(Path(args.config))
        policy = policy_from_config(cfg)
        plan = Path(args.plan) if args.plan else DEFAULT_PLAN
        parse_ledger(ledger_raw)                       # refuse before any client exists
        if args.live and d16_status(plan) != "decided":
            return refused("d16_not_decided")
        vault, archive = _stores(args, cfg, client_factory)
        log = sweep(vault=vault, archive=archive, ledger=ledger_raw, policy=policy, now=now,
                    live=bool(args.live), plan_path=plan, log_dir=Path(args.log_dir))
    except SweepRefused as e:
        return refused(e.code)
    except _Usage as e:
        print(f"{MODULE}: {e}", file=sys.stderr)
        return 2
    except StoreError as e:
        print(json.dumps({"error": "storage_failed", "code": e.code}))
        return 1
    print(_dumps(log))
    if args.evidence_out:
        _write_text_atomic(Path(args.evidence_out), _dumps(evidence(log)) + "\n")
    return 1 if log["counts"]["failed"] else 0


__all__ = ["ARCHIVE_RETENTION_DAYS", "FsStore", "ObjectLocked", "S3Store", "StoreError",
           "SweepPolicy", "SweepRefused", "d16_status", "evidence", "main", "parse_ledger",
           "policy_from_config", "split_key", "sweep"]


if __name__ == "__main__":
    sys.exit(main())
