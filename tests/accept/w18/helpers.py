"""Shared helpers for the W18 acceptance tests (lane D vault TTL sweeper).

* :class:`FakeS3` is a boto3-shaped, in-memory, VERSIONED multi-bucket S3
  service (the "fixture S3 model" of tests/accept/w3a, made writable). It
  models Object Lock (a locked version refuses a versioned delete), delete
  markers (an unversioned delete on a versioned bucket hides, never removes),
  pagination (2 items per page), legal holds, and failure injection (write
  failure, corrupted write, delete failure; :meth:`FakeS3.outage`: a latched
  503 on any operation from a trigger point on; :attr:`FakeS3.retention_override`:
  an error or a malformed answer from ``GetObjectRetention``). Every call is
  recorded; any method outside the README's allowed surface is recorded in
  ``forbidden`` and raises.
* :func:`scenario` builds the vault and its retention ledger.
* :func:`run_cli` drives ``python -P -m cascade.laned.vault_sweeper`` with
  ``boto3``/``botocore``/``bittensor`` unimportable and every socket connect
  refused (the fs adapter needs neither).
* :func:`evidence_problems` is the owner-evidence verifier. Contract: README.md.

The module under test is imported only inside functions (:func:`vs`), never at
helpers import time.
"""

from __future__ import annotations

import hashlib
import importlib
import io
import json
import os
import re
import socket
import stat
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
MODULE = "cascade.laned.vault_sweeper"
PLAN = REPO / "docs" / "plans" / "LANE_D_PLAN.md"
CHAIN_TOML = REPO / "chain.toml"

LOG_SCHEMA = "laned_vault_sweep/1"
LEDGER_SCHEMA = "laned_vault_ledger/1"
EVIDENCE_SCHEMA = "laned_w18_evidence/1"
WP = "W18"

VAULT_BUCKET = "cascade-lane-d-vault"
VAULT_PREFIX = "lane-d"
ARCHIVE_BUCKET = "cascade-lane-d-archive"
ARCHIVE_PREFIX = "lane-d-archive"

TOP_KEYS = ("schema", "run_id", "mode", "now", "started_at", "finished_at", "d16", "policy",
            "ledger_era", "vault", "archive", "actions", "counts")
ACTION_KEYS = ("store", "key", "op", "outcome", "reason")
OPS = ("archive", "delete", "keep", "list")
OUTCOMES = {"archive": ("done", "planned", "already", "failed"),
            "delete": ("done", "planned", "failed"),
            "keep": ("kept",),
            "list": ("failed",)}
KEEP_REASONS = ("object_lock", "open_entry", "settlement_open", "king", "ttl_not_expired",
                "dispute_open", "unreferenced", "unknown_status", "entry_incomplete",
                "archive_retention_not_expired", "archive_age_unknown",
                "lock_unknown", "legal_hold", "run_aborted")
FAIL_REASONS = ("archive_copy_failed", "archive_verify_failed", "delete_failed", "object_lock",
                "storage_error")
# keep reasons that mean "may be under Object Lock": never deleted
LOCK_KEEP_REASONS = ("object_lock", "lock_unknown", "legal_hold")
COUNT_KEYS = ("archived", "deleted", "expired", "kept", "failed", "planned")

HEX64_RE = re.compile(r"[0-9a-f]{64}")
RUN_ID_RE = re.compile(r"[0-9a-f]{16}")
LEAK_RES = [re.compile(p, re.I) for p in (
    r"https?://", r"x-amz-signature", r"x-amz-credential", r"x-amz-security-token",
    r"awsaccesskeyid", r"signature=", r"authorization:", r"secret")]

NOW = datetime(2026, 12, 1, 12, 0, 0, tzinfo=UTC)
ERA = 20
TTL = 4
FILES = ("model.safetensors", "manifest.json")


def vs():
    """The module under test (a ModuleNotFoundError here is the right reason
    on main: cascade/laned/vault_sweeper.py is in PATHS)."""
    return importlib.import_module(MODULE)


def iso(dt: datetime) -> str:
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_ts(s) -> datetime:
    assert isinstance(s, str), f"timestamp must be an ISO 8601 string, got {s!r}"
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    assert dt.tzinfo is not None, f"timestamp {s!r} carries no timezone"
    return dt


# ── strict JSON ──────────────────────────────────────────────────────────────


def _no_constant(name):
    raise ValueError(f"non-strict JSON constant {name}")


def strict_loads(text: str):
    return json.loads(text, parse_constant=_no_constant)


# ── the scenario: a vault and its retention ledger ───────────────────────────


def _hk(tag: str) -> str:
    base = "5" + hashlib.sha256(tag.encode()).hexdigest()
    alpha = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
    return "".join(alpha[int(c, 16) % len(alpha)] for c in base)[:48]


def _digest(tag: str) -> str:
    return hashlib.sha256(("w18-entry-" + tag).encode()).hexdigest()


@dataclass
class Entry:
    tag: str
    status: str
    closed_era: int | None
    hotkey: str = ""
    digest: str = ""
    files: dict = field(default_factory=dict)

    def __post_init__(self):
        self.hotkey = self.hotkey or _hk(self.tag)
        self.digest = self.digest or _digest(self.tag)
        if not self.files:
            self.files = {f: hashlib.sha256(f"{self.tag}/{f}".encode()).digest() * 3
                          for f in FILES}

    def key(self, name: str) -> str:
        return f"{self.hotkey}/{self.digest}/{name}"

    def rels(self) -> list[str]:
        return [self.key(f) for f in self.files]

    def row(self) -> dict:
        return {"hotkey": self.hotkey, "digest": self.digest, "status": self.status,
                "closed_era": self.closed_era}


def scenario_entries() -> dict[str, Entry]:
    """era = 20, loser_ttl_eras = 4."""
    return {e.tag: e for e in (
        Entry("lost_due", "lost", 16),            # 20 - 16 = 4 >= 4: swept
        Entry("lost_young", "lost", 17),          # 3 < 4: ttl_not_expired
        Entry("unjudged_due", "closed_unjudged", 10),
        Entry("expired_due", "ready_expired", 12),
        Entry("voided_due", "voided", 15),
        Entry("open", "ready", None),             # open_entry
        Entry("judging", "judging", None),        # settlement_open
        Entry("king", "king", 5),                 # king (kept permanently)
        Entry("disputed", "lost", 3),             # dispute_open
        Entry("weird", "mystery", 1),             # unknown_status
    )}


DUE = ("lost_due", "unjudged_due", "expired_due", "voided_due")
EXPECTED_KEEP = {"lost_young": "ttl_not_expired", "open": "open_entry",
                 "judging": "settlement_open", "king": "king", "disputed": "dispute_open",
                 "weird": "unknown_status"}
ORPHAN_KEY = f"{_hk('orphan')}/{_digest('orphan')}/model.safetensors"
STRAY_KEY = "stray.txt"


def ledger(entries: dict[str, Entry], *, era: int = ERA, disputes=("disputed",)) -> dict:
    return {"schema": LEDGER_SCHEMA, "era": era,
            "entries": [e.row() for e in entries.values()],
            "disputes": [entries[t].digest for t in disputes if t in entries]}


def all_vault_objects(entries: dict[str, Entry]) -> dict[str, bytes]:
    out = {k: e.files[k.rsplit("/", 1)[1]] for e in entries.values() for k in e.rels()}
    out[ORPHAN_KEY] = b"orphan-bytes"
    out[STRAY_KEY] = b"stray-bytes"
    return out


def policy(ttl: int = TTL, days: int | None = None):
    m = vs()
    return m.SweepPolicy(loser_ttl_eras=ttl) if days is None else m.SweepPolicy(
        loser_ttl_eras=ttl, archive_retention_days=days)


def write_plan(path: Path, d16: str | None, *, register: bool = True) -> Path:
    """A minimal plan file: the §6 table always lists D16; the §17.4 register
    lists it with status ``d16`` (omitted when ``d16`` is None)."""
    lines = ["# Lane D plan (fixture)", "", "## 6. Owner decisions", "",
             "| # | Decision | Recommended |", "|---|---|---|",
             "| D16 | Standing approval | Automated vault TTL deletion and archive expiry |",
             "", "### 17.4 Decision register and owner inputs", ""]
    if register:
        lines += ["| D | Status | Option chosen | Class | Decided |",
                  "|---|---|---|---|---|",
                  "| D8 | decided | Losers private; vault TTL 4 eras; archive 14 days | O | owner |"]
        if d16 is not None:
            lines.append(f"| D16 | {d16} | Standing approval for automated TTL deletion | "
                         f"O (ops) | owner |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ── a boto3-shaped, versioned, multi-bucket fake S3 ──────────────────────────


class FakeClientError(Exception):
    """Shaped like ``botocore.exceptions.ClientError``: ``.response`` carries
    ``Error.Code`` and ``ResponseMetadata.HTTPStatusCode``."""

    def __init__(self, op: str, code: str, status: int):
        super().__init__(f"An error occurred ({code}) when calling the {op} operation")
        self.operation_name = op
        self.response = {"Error": {"Code": code, "Message": code},
                         "ResponseMetadata": {"HTTPStatusCode": status}}


class _Body:
    def __init__(self, data: bytes):
        self._b = io.BytesIO(data)

    def read(self, amt=None):
        return self._b.read() if amt is None else self._b.read(amt)

    def iter_chunks(self, chunk_size=1024 * 1024):
        while True:
            c = self._b.read(chunk_size)
            if not c:
                return
            yield c

    def close(self):
        self._b.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


class _Paginator:
    def __init__(self, fn, token_in: tuple[str, ...], token_out: tuple[str, ...]):
        self._fn, self._in, self._out = fn, token_in, token_out

    def paginate(self, **kw):
        kw = dict(kw)
        while True:
            page = self._fn(**kw)
            yield page
            if not page.get("IsTruncated"):
                return
            for i, o in zip(self._in, self._out, strict=True):
                kw[i] = page[o]


WRITE_OPS = ("put_object", "copy_object", "create_multipart_upload", "upload_part",
             "upload_part_copy", "complete_multipart_upload")
ALLOWED_OPS = ("list_objects_v2", "list_object_versions", "head_object", "get_object",
               "get_object_retention", "get_object_legal_hold", "delete_object", "delete_objects",
               "abort_multipart_upload", "get_paginator") + WRITE_OPS
PAGE = 2


class FakeS3:
    """In-memory S3 service holding several versioned buckets. ``clock`` is
    the service's "now" for Object Lock enforcement (the tests set it to the
    sweep's ``now``). Versions: ``{"vid", "data", "mode", "retain_until",
    "meta", "marker", "hold"}`` per key, oldest first (``hold`` is the legal
    hold, ``"ON"``/``"OFF"``/None; an ``"ON"`` hold refuses a versioned delete
    like unexpired retention)."""

    def __init__(self, buckets=(VAULT_BUCKET, ARCHIVE_BUCKET), clock: datetime = NOW):
        self.buckets: dict[str, dict[str, list[dict]]] = {b: {} for b in buckets}
        self.clock = clock
        self.calls: list[tuple[str, dict]] = []
        self.forbidden: list[str] = []
        self.violations: list[str] = []
        self.fail_put: set[tuple[str, str]] = set()
        self.corrupt: set[tuple[str, str]] = set()
        self.fail_delete: set[tuple[str, str]] = set()
        # (bucket, full key) -> ("error", code, status) | ("response", dict): what
        # GetObjectRetention answers for every version of that key instead of the
        # truth (an unreadable or malformed retention answer).
        self.retention_override: dict[tuple[str, str], tuple] = {}
        self.rules: list[dict] = []          # outage rules, see outage()
        self.injected: list[int] = []        # indices into ``calls`` of injected failures
        self.deleted_ok: list[tuple[str, str, str | None]] = []   # successful deletes
        self._uploads: dict[str, dict] = {}
        self._n = 0

    # failure injection
    def outage(self, op: str, bucket: str | None = None, *, when=None, code: str = "SlowDown",
               status: int = 503) -> dict:
        """From the first ``op`` call (on ``bucket``, if given) at which
        ``when(self)`` is true (or from the start when ``when`` is None), every
        such call fails with ``code``/``status`` (latched: a storage outage)."""
        rule = {"op": op, "bucket": bucket, "when": when, "code": code, "status": status,
                "armed": when is None}
        self.rules.append(rule)
        return rule

    def clear_faults(self) -> None:
        self.rules.clear()
        self.retention_override.clear()
        self.fail_put.clear()
        self.corrupt.clear()
        self.fail_delete.clear()

    def _inject(self, op: str, kw: dict) -> None:
        for rule in self.rules:
            if rule["op"] != op or (rule["bucket"] is not None and kw.get("Bucket") != rule["bucket"]):
                continue
            if not rule["armed"] and rule["when"](self):
                rule["armed"] = True
            if rule["armed"]:
                self.injected.append(len(self.calls) - 1)
                raise FakeClientError(op, rule["code"], rule["status"])

    def _record(self, op: str, kw: dict) -> None:
        self.calls.append((op, kw))
        self._inject(op, kw)

    # fixture state
    def _vid(self) -> str:
        self._n += 1
        return f"v{self._n:06d}"

    def seed(self, bucket: str, key: str, data: bytes, *, mode: str | None = None,
             retain_until: datetime | None = None, meta: dict | None = None,
             hold: str | None = None) -> str:
        vid = self._vid()
        self.buckets[bucket].setdefault(key, []).append(
            {"vid": vid, "data": data, "mode": mode, "retain_until": retain_until,
             "meta": dict(meta or {}), "marker": False, "hold": hold})
        return vid

    def live_data(self, bucket: str, key: str) -> bytes | None:
        vs_ = self.buckets[bucket].get(key) or []
        if not vs_ or vs_[-1]["marker"]:
            return None
        return vs_[-1]["data"]

    def versions(self, bucket: str, key: str) -> list[dict]:
        return list(self.buckets[bucket].get(key) or [])

    def writes(self, bucket: str | None = None) -> list[tuple[str, dict]]:
        return [(op, kw) for op, kw in self.calls
                if op in WRITE_OPS and (bucket is None or kw.get("Bucket") == bucket)]

    def deletes(self, bucket: str | None = None) -> list[tuple[str, dict]]:
        return [(op, kw) for op, kw in self.calls if op in ("delete_object", "delete_objects")
                and (bucket is None or kw.get("Bucket") == bucket)]

    def deleted_keys(self, bucket: str) -> list[str]:
        out = []
        for op, kw in self.deletes(bucket):
            if op == "delete_object":
                out.append(kw.get("Key"))
            else:
                out += [o.get("Key") for o in (kw.get("Delete") or {}).get("Objects", [])]
        return out

    # plumbing
    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        self.forbidden.append(name)

        def _refuse(*a, **k):
            raise AssertionError(f"W18: the sweeper called S3 {name!r}, outside the allowed "
                                 f"surface {ALLOWED_OPS} (README)")
        return _refuse

    def _bucket(self, op: str, kw: dict) -> dict:
        b = kw.get("Bucket")
        if b not in self.buckets:
            raise FakeClientError(op, "NoSuchBucket", 404)
        return self.buckets[b]

    def _locked(self, v: dict) -> bool:
        if v.get("hold") == "ON":
            return True
        return bool(v["mode"]) and v["retain_until"] is not None and v["retain_until"] > self.clock

    def _check_acl(self, op: str, kw: dict) -> None:
        if kw.get("Bucket") == ARCHIVE_BUCKET:
            if kw.get("ACL") != "private":
                self.violations.append(f"{op} to the archive without ACL='private': "
                                       f"ACL={kw.get('ACL')!r}")
            grants = [k for k in kw if k.startswith("Grant")]
            if grants:
                self.violations.append(f"{op} to the archive with grants {grants}")

    def _store(self, op: str, kw: dict, data: bytes) -> dict:
        bucket = self._bucket(op, kw)
        key = kw.get("Key")
        if (kw.get("Bucket"), key) in self.fail_put:
            raise FakeClientError(op, "InternalError", 500)
        if (kw.get("Bucket"), key) in self.corrupt and data:
            data = bytes([data[0] ^ 0xFF]) + data[1:]
        vid = self._vid()
        mode = kw.get("ObjectLockMode")
        ru = kw.get("ObjectLockRetainUntilDate")
        bucket.setdefault(key, []).append(
            {"vid": vid, "data": data, "mode": mode, "retain_until": ru,
             "meta": dict(kw.get("Metadata") or {}), "marker": False})
        return {"VersionId": vid, "ETag": '"' + hashlib.md5(data).hexdigest() + '"',
                "ResponseMetadata": {"HTTPStatusCode": 200}}

    def _find(self, op: str, kw: dict) -> dict:
        vs_ = self._bucket(op, kw).get(kw.get("Key")) or []
        vid = kw.get("VersionId")
        if vid is None:
            if vs_ and not vs_[-1]["marker"]:
                return vs_[-1]
        else:
            for v in vs_:
                if v["vid"] == vid and not v["marker"]:
                    return v
        raise FakeClientError(op, "NoSuchKey" if op == "GetObject" else "404", 404)

    @staticmethod
    def _read_body(body) -> bytes:
        if isinstance(body, (bytes, bytearray)):
            return bytes(body)
        if hasattr(body, "read"):
            return body.read()
        raise AssertionError(f"unsupported Body type {type(body).__name__}")

    def _source(self, op: str, src) -> bytes:
        if isinstance(src, str):
            b, _, rest = src.lstrip("/").partition("/")
            k, _, q = rest.partition("?versionId=")
            src = {"Bucket": b, "Key": k, **({"VersionId": q} if q else {})}
        return self._find(op, dict(src))["data"]

    # the boto3-shaped surface
    def list_objects_v2(self, **kw):
        self._record("list_objects_v2", dict(kw))
        bucket = self._bucket("ListObjectsV2", kw)
        pre = kw.get("Prefix", "")
        keys = [k for k in sorted(bucket) if k.startswith(pre) and bucket[k]
                and not bucket[k][-1]["marker"]]
        start = int(kw.get("ContinuationToken") or 0)
        page = keys[start:start + PAGE]
        out = {"Contents": [{"Key": k, "Size": len(bucket[k][-1]["data"])} for k in page],
               "KeyCount": len(page), "IsTruncated": start + PAGE < len(keys),
               "ResponseMetadata": {"HTTPStatusCode": 200}}
        if out["IsTruncated"]:
            out["NextContinuationToken"] = str(start + PAGE)
        return out

    def list_object_versions(self, **kw):
        self._record("list_object_versions", dict(kw))
        bucket = self._bucket("ListObjectVersions", kw)
        pre = kw.get("Prefix", "")
        items = [(k, i, v) for k in sorted(bucket) if k.startswith(pre)
                 for i, v in enumerate(bucket[k])]
        start = 0
        if kw.get("KeyMarker") is not None:
            for n, (k, _, v) in enumerate(items):
                if k == kw["KeyMarker"] and v["vid"] == kw.get("VersionIdMarker"):
                    start = n + 1
        page = items[start:start + PAGE]
        vers, marks = [], []
        for k, i, v in page:
            latest = i == len(bucket[k]) - 1
            (marks if v["marker"] else vers).append(
                {"Key": k, "VersionId": v["vid"], "IsLatest": latest,
                 **({} if v["marker"] else {"Size": len(v["data"])})})
        out = {"Versions": vers, "DeleteMarkers": marks,
               "IsTruncated": start + PAGE < len(items),
               "ResponseMetadata": {"HTTPStatusCode": 200}}
        if out["IsTruncated"]:
            out["NextKeyMarker"], out["NextVersionIdMarker"] = page[-1][0], page[-1][2]["vid"]
        return out

    def get_paginator(self, name: str):
        self._record("get_paginator", {"name": name})
        if name == "list_objects_v2":
            return _Paginator(self.list_objects_v2, ("ContinuationToken",),
                              ("NextContinuationToken",))
        if name == "list_object_versions":
            return _Paginator(self.list_object_versions, ("KeyMarker", "VersionIdMarker"),
                              ("NextKeyMarker", "NextVersionIdMarker"))
        raise AssertionError(f"W18: unexpected paginator {name!r}")

    def head_object(self, **kw):
        self._record("head_object", dict(kw))
        v = self._find("HeadObject", kw)
        out = {"ContentLength": len(v["data"]), "VersionId": v["vid"],
               "ETag": '"' + hashlib.md5(v["data"]).hexdigest() + '"',
               "Metadata": dict(v["meta"]), "ResponseMetadata": {"HTTPStatusCode": 200}}
        if v["mode"]:
            out["ObjectLockMode"] = v["mode"]
            out["ObjectLockRetainUntilDate"] = v["retain_until"]
        if v.get("hold"):
            out["ObjectLockLegalHoldStatus"] = v["hold"]
        return out

    def get_object_retention(self, **kw):
        self._record("get_object_retention", dict(kw))
        v = self._find("GetObjectRetention", kw)
        ov = self.retention_override.get((kw.get("Bucket"), kw.get("Key")))
        if ov is not None:
            if ov[0] == "error":
                raise FakeClientError("GetObjectRetention", ov[1], ov[2])
            return dict(ov[1])
        if not v["mode"]:
            raise FakeClientError("GetObjectRetention", "NoSuchObjectLockConfiguration", 404)
        return {"Retention": {"Mode": v["mode"], "RetainUntilDate": v["retain_until"]},
                "ResponseMetadata": {"HTTPStatusCode": 200}}

    def get_object_legal_hold(self, **kw):
        self._record("get_object_legal_hold", dict(kw))
        v = self._find("GetObjectLegalHold", kw)
        if not v.get("hold"):
            raise FakeClientError("GetObjectLegalHold", "NoSuchObjectLockConfiguration", 404)
        return {"LegalHold": {"Status": v["hold"]}, "ResponseMetadata": {"HTTPStatusCode": 200}}

    def get_object(self, **kw):
        self._record("get_object", dict(kw))
        v = self._find("GetObject", kw)
        return {"Body": _Body(v["data"]), "ContentLength": len(v["data"]),
                "VersionId": v["vid"], "Metadata": dict(v["meta"]),
                "ResponseMetadata": {"HTTPStatusCode": 200}}

    def put_object(self, **kw):
        rec = {k: v for k, v in kw.items() if k != "Body"}
        self._record("put_object", rec)
        self._check_acl("put_object", kw)
        return self._store("PutObject", kw, self._read_body(kw.get("Body", b"")))

    def copy_object(self, **kw):
        self._record("copy_object", dict(kw))
        self._check_acl("copy_object", kw)
        return self._store("CopyObject", kw, self._source("CopyObject", kw.get("CopySource")))

    def create_multipart_upload(self, **kw):
        self._record("create_multipart_upload", dict(kw))
        self._check_acl("create_multipart_upload", kw)
        self._bucket("CreateMultipartUpload", kw)
        self._n += 1
        uid = f"upload-{self._n}"
        self._uploads[uid] = {"kw": dict(kw), "parts": {}}
        return {"UploadId": uid, "Bucket": kw.get("Bucket"), "Key": kw.get("Key")}

    def upload_part(self, **kw):
        rec = {k: v for k, v in kw.items() if k != "Body"}
        self._record("upload_part", rec)
        up = self._uploads[kw["UploadId"]]
        data = self._read_body(kw.get("Body", b""))
        up["parts"][int(kw["PartNumber"])] = data
        return {"ETag": '"' + hashlib.md5(data).hexdigest() + '"'}

    def upload_part_copy(self, **kw):
        self._record("upload_part_copy", dict(kw))
        up = self._uploads[kw["UploadId"]]
        data = self._source("UploadPartCopy", kw.get("CopySource"))
        rng = kw.get("CopySourceRange")
        if rng:
            a, b = rng.removeprefix("bytes=").split("-")
            data = data[int(a):int(b) + 1]
        up["parts"][int(kw["PartNumber"])] = data
        return {"CopyPartResult": {"ETag": '"' + hashlib.md5(data).hexdigest() + '"'}}

    def complete_multipart_upload(self, **kw):
        self._record("complete_multipart_upload", dict(kw))
        up = self._uploads.pop(kw["UploadId"])
        nums = [int(p["PartNumber"]) for p in (kw.get("MultipartUpload") or {}).get("Parts", [])]
        data = b"".join(up["parts"][n] for n in sorted(nums))
        return self._store("CompleteMultipartUpload", up["kw"], data)

    def abort_multipart_upload(self, **kw):
        self._record("abort_multipart_upload", dict(kw))
        self._uploads.pop(kw.get("UploadId"), None)
        return {}

    def _delete_one(self, op: str, bucket_name: str, key: str, vid) -> None:
        bucket = self.buckets[bucket_name]
        if (bucket_name, key) in self.fail_delete:
            raise FakeClientError(op, "InternalError", 500)
        vs_ = bucket.get(key) or []
        if vid is None:
            # A versioned bucket: an unversioned delete only adds a marker.
            vs_.append({"vid": self._vid(), "data": b"", "mode": None, "retain_until": None,
                        "meta": {}, "marker": True})
            bucket[key] = vs_
            self.deleted_ok.append((bucket_name, key, None))
            return
        for i, v in enumerate(vs_):
            if v["vid"] == vid:
                if self._locked(v):
                    self.violations.append(f"delete of a locked version {bucket_name}/{key}@{vid}")
                    raise FakeClientError(op, "AccessDenied", 403)
                del vs_[i]
                if not vs_:
                    bucket.pop(key, None)
                self.deleted_ok.append((bucket_name, key, vid))
                return

    def delete_object(self, **kw):
        self._record("delete_object", dict(kw))
        self._bucket("DeleteObject", kw)
        if "BypassGovernanceRetention" in kw:
            self.violations.append("delete_object with BypassGovernanceRetention")
        self._delete_one("DeleteObject", kw["Bucket"], kw.get("Key"), kw.get("VersionId"))
        return {"ResponseMetadata": {"HTTPStatusCode": 204}}

    def delete_objects(self, **kw):
        self._record("delete_objects", dict(kw))
        self._bucket("DeleteObjects", kw)
        if "BypassGovernanceRetention" in kw:
            self.violations.append("delete_objects with BypassGovernanceRetention")
        deleted, errors = [], []
        for o in (kw.get("Delete") or {}).get("Objects", []):
            try:
                self._delete_one("DeleteObjects", kw["Bucket"], o.get("Key"), o.get("VersionId"))
                deleted.append(dict(o))
            except FakeClientError as e:
                errors.append({**o, "Code": e.response["Error"]["Code"]})
        return {"Deleted": deleted, "Errors": errors, "ResponseMetadata": {"HTTPStatusCode": 200}}


def seeded_fake(entries: dict[str, Entry], clock: datetime = NOW) -> FakeS3:
    fake = FakeS3(clock=clock)
    for k, data in all_vault_objects(entries).items():
        fake.seed(VAULT_BUCKET, f"{VAULT_PREFIX}/{k}", data)
    return fake


def s3_stores(fake: FakeS3):
    m = vs()
    return (m.S3Store(fake, VAULT_BUCKET, VAULT_PREFIX),
            m.S3Store(fake, ARCHIVE_BUCKET, ARCHIVE_PREFIX))


def vkey(rel: str) -> str:
    return f"{VAULT_PREFIX}/{rel}"


def akey(rel: str) -> str:
    return f"{ARCHIVE_PREFIX}/{rel}"


def run_sweep(vault, archive, led: dict, tmp_path: Path, *, now: datetime = NOW,
              live: bool = True, plan: Path | None | str = "decided", pol=None,
              log_dir: Path | None = None) -> dict:
    m = vs()
    tmp_path.mkdir(parents=True, exist_ok=True)
    if plan == "decided":
        plan = write_plan(tmp_path / "plan_decided.md", "decided")
    log_dir = log_dir or (tmp_path / "logs")
    return m.sweep(vault=vault, archive=archive, ledger=led, policy=pol or policy(), now=now,
                   live=live, plan_path=plan, log_dir=log_dir)


def sweep_logged(vault, archive, led: dict, tmp_path: Path, **kw) -> tuple[dict, Exception | None]:
    """One run that may end on a storage error. ``sweep`` may return the log
    or raise (anything but ``SweepRefused``), but only AFTER writing exactly
    one complete strict-JSON action log to its ``log_dir`` (README "Storage
    errors"). Returns (the log read back from disk, the exception or None)."""
    m = vs()
    log_dir = tmp_path / "logs"
    ret, raised = None, None
    try:
        ret = run_sweep(vault, archive, led, tmp_path, log_dir=log_dir, **kw)
    except AssertionError:
        raise
    except m.SweepRefused as e:
        raise AssertionError(f"W18: a run that touched storage ended in SweepRefused({e.code!r}); "
                             "a refusal is only ever raised before any storage call") from None
    except Exception as e:  # noqa: BLE001 - any escape is allowed once the log is on disk
        raised = e
    files = sorted(log_dir.glob("vault-sweep-*.json")) if log_dir.is_dir() else []
    assert len(files) == 1, (
        f"W18: a storage error must never escape without the action log: {len(files)} "
        f"log file(s) in {log_dir} (sweep raised {raised!r}). The log of what was already "
        "archived/deleted is written BEFORE sweep returns or raises (README \"Storage errors\")")
    try:
        log = strict_loads(files[0].read_text(encoding="utf-8"))
    except ValueError as e:
        raise AssertionError(f"W18: the action log on disk is not strict JSON: {e}") from None
    assert isinstance(log, dict) and files[0].name == f"vault-sweep-{log.get('run_id')}.json", (
        files[0].name)
    if ret is not None:
        assert ret == log, "sweep returned a log that differs from the one it wrote"
    return log, raised


def by_key(log: dict, store: str = "vault") -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for a in log["actions"]:
        if a["store"] == store:
            out.setdefault(a["key"], []).append(a)
    return out


def ops_of(log: dict, key: str, store: str = "vault") -> list[tuple[str, str]]:
    return [(a["op"], a["outcome"]) for a in by_key(log, store).get(key, [])]


def keep_reason(log: dict, key: str, store: str = "vault") -> str | None:
    rs = [a["reason"] for a in by_key(log, store).get(key, []) if a["op"] == "keep"]
    return rs[0] if rs else None


# ── action-log shape ─────────────────────────────────────────────────────────


def derive_counts(actions: list[dict]) -> dict:
    c = dict.fromkeys(COUNT_KEYS, 0)
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


def log_problems(log) -> list[str]:
    p: list[str] = []
    if not isinstance(log, dict):
        return [f"log is not an object: {type(log).__name__}"]
    if tuple(sorted(log)) != tuple(sorted(TOP_KEYS)):
        p.append(f"top-level keys {sorted(log)} != {sorted(TOP_KEYS)}")
        return p
    if log["schema"] != LOG_SCHEMA:
        p.append(f"schema {log['schema']!r}")
    if not isinstance(log["run_id"], str) or not RUN_ID_RE.fullmatch(log["run_id"]):
        p.append(f"run_id {log['run_id']!r} is not 16 lowercase hex")
    if log["mode"] not in ("dry-run", "live"):
        p.append(f"mode {log['mode']!r}")
    try:
        if parse_ts(log["started_at"]) > parse_ts(log["finished_at"]):
            p.append("started_at after finished_at")
        parse_ts(log["now"])
    except (AssertionError, ValueError) as e:
        p.append(f"timestamps: {e}")
    if not isinstance(log["policy"], dict) or set(log["policy"]) != {
            "loser_ttl_eras", "archive_retention_days"}:
        p.append(f"policy {log['policy']!r}")
    for side in ("vault", "archive"):
        d = log[side]
        if not isinstance(d, dict) or d.get("kind") not in ("s3", "fs"):
            p.append(f"{side} descriptor {d!r}")
        elif d["kind"] == "s3" and not ({"bucket", "prefix"} <= set(d)):
            p.append(f"{side} s3 descriptor lacks bucket/prefix: {d!r}")
        elif d["kind"] == "fs" and "root" not in d:
            p.append(f"{side} fs descriptor lacks root: {d!r}")
    if not isinstance(log["actions"], list):
        p.append("actions is not a list")
        return p
    for i, a in enumerate(log["actions"]):
        if not isinstance(a, dict) or not set(ACTION_KEYS) <= set(a):
            p.append(f"action {i} lacks {ACTION_KEYS}: {a!r}")
            continue
        if a["store"] not in ("vault", "archive"):
            p.append(f"action {i} store {a['store']!r}")
        if a["op"] not in OPS or a["outcome"] not in OUTCOMES.get(a["op"], ()):
            p.append(f"action {i} op/outcome {a['op']!r}/{a['outcome']!r}")
        if a["op"] == "keep" and a["reason"] not in KEEP_REASONS:
            p.append(f"action {i} keep reason {a['reason']!r} not in {KEEP_REASONS}")
        if a["outcome"] == "failed" and a["reason"] not in FAIL_REASONS:
            p.append(f"action {i} failure reason {a['reason']!r} not in {FAIL_REASONS}")
        if a["op"] == "archive" and a["outcome"] in ("done", "already", "planned") and not (
                isinstance(a.get("sha256"), str) and HEX64_RE.fullmatch(a["sha256"])):
            p.append(f"action {i} archive without a 64-hex sha256: {a!r}")
        if a["op"] == "keep" and a["reason"] == "object_lock":
            try:
                parse_ts(a.get("retain_until"))
            except (AssertionError, ValueError) as e:
                p.append(f"action {i} object_lock keep without retain_until: {e}")
    if log["counts"] != derive_counts(log["actions"]):
        p.append(f"counts {log['counts']} != re-derived {derive_counts(log['actions'])}")
    text = json.dumps(log)
    for rx in LEAK_RES:
        if rx.search(text):
            p.append(f"log text matches leak pattern {rx.pattern!r}")
    return p


# ── fs helpers ───────────────────────────────────────────────────────────────


def fs_seed(root: Path, entries: dict[str, Entry]) -> None:
    for k, data in all_vault_objects(entries).items():
        f = root / k
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(data)


def tree(root: Path) -> dict[str, tuple]:
    out = {}
    if not root.exists():
        return out
    for p in sorted(root.rglob("*")):
        st = p.lstat()
        rel = p.relative_to(root).as_posix()
        if stat.S_ISREG(st.st_mode):
            out[rel] = ("f", p.read_bytes(), st.st_mode, st.st_mtime_ns)
        else:
            out[rel] = ("d", st.st_mode, st.st_mtime_ns)
    return out


# ── the CLI (subprocess, network and boto3 blocked) ──────────────────────────

_SITECUSTOMIZE = '''\
import os, socket
def _blocked(*a, **k):
    raise OSError("W18 test: network is blocked; the fs adapter needs none")
socket.socket.connect = _blocked
socket.socket.connect_ex = _blocked
socket.create_connection = _blocked
socket.getaddrinfo = _blocked
_m = os.environ.get("W18_NET_BLOCK_MARKER")
if _m:
    open(_m, "w").write("blocked")
'''


def _sandbox(tmp_path: Path) -> Path:
    d = tmp_path / "w18_sandbox"
    for pkg in ("boto3", "botocore", "bittensor"):
        (d / pkg).mkdir(parents=True, exist_ok=True)
        (d / pkg / "__init__.py").write_text(
            f"raise ImportError('W18 test: {pkg} is blocked for the fs adapter')\n")
    (d / "sitecustomize.py").write_text(_SITECUSTOMIZE)
    return d


def run_cli(tmp_path: Path, *args: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items()
           if not (k.startswith("HIPPIUS_") or k.startswith("AWS_"))}
    env["PYTHONPATH"] = os.pathsep.join([str(_sandbox(tmp_path)), str(REPO)])
    marker = tmp_path / "net_block_marker"
    env["W18_NET_BLOCK_MARKER"] = str(marker)
    proc = subprocess.run([sys.executable, "-P", "-m", MODULE, *args], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=120, check=False)
    assert marker.is_file(), "the network block (sitecustomize) did not load in the CLI run"
    return proc


def no_network_in_process():
    """A context manager refusing socket connects in the test process."""
    class _Guard:
        def __enter__(self):
            self._saved = (socket.socket.connect, socket.create_connection)

            def _blocked(*a, **k):
                raise AssertionError("W18 test: the sweeper opened a network connection")
            socket.socket.connect = _blocked
            socket.create_connection = _blocked
            return self

        def __exit__(self, *a):
            socket.socket.connect, socket.create_connection = self._saved
    return _Guard()


# ── the owner-evidence verifier ──────────────────────────────────────────────


def _overlap(a: dict, b: dict) -> bool:
    if a.get("bucket") != b.get("bucket"):
        return False
    pa, pb = str(a.get("prefix", "")).rstrip("/") + "/", str(b.get("prefix", "")).rstrip("/") + "/"
    return pa.startswith(pb) or pb.startswith(pa)


def evidence_problems(ev) -> list[str]:
    """Re-derives every claim of ``evidence/w18.json`` from its embedded log."""
    if not isinstance(ev, dict):
        return [f"evidence is not an object: {type(ev).__name__}"]
    p: list[str] = []
    if set(ev) != {"schema", "wp", "passed", "log"}:
        p.append(f"evidence keys {sorted(ev)} != ['log', 'passed', 'schema', 'wp']")
        return p
    if ev["schema"] != EVIDENCE_SCHEMA:
        p.append(f"schema {ev['schema']!r} != {EVIDENCE_SCHEMA!r}")
    if ev["wp"] != WP:
        p.append(f"wp {ev['wp']!r} != {WP!r}")
    log = ev["log"]
    lp = log_problems(log)
    if lp:
        return p + [f"log: {x}" for x in lp]
    if log["mode"] != "live":
        p.append(f"log mode {log['mode']!r}: L4 needs a live run")
    if log["d16"] != "decided":
        p.append(f"log d16 {log['d16']!r}: a live run needs D16 decided")
    v, a = log["vault"], log["archive"]
    if v.get("kind") != "s3" or v.get("bucket") != VAULT_BUCKET:
        p.append(f"vault {v!r} is not the live s3 vault bucket {VAULT_BUCKET!r}")
    if a.get("kind") != "s3":
        p.append(f"archive {a!r} is not an s3 store")
    elif _overlap(v, a):
        p.append(f"archive {a!r} overlaps the vault {v!r}")
    archived: set[str] = set()
    locked: set[str] = set()
    failed = False
    for x in log["actions"]:
        if x["outcome"] == "failed":
            failed = True
        if x["store"] != "vault":
            continue
        if x["op"] == "archive" and x["outcome"] in ("done", "already"):
            archived.add(x["key"])
        if x["op"] == "keep" and x["reason"] in LOCK_KEEP_REASONS:
            locked.add(x["key"])
        if x["op"] == "delete" and x["outcome"] == "done":
            if x["key"] not in archived:
                p.append(f"vault delete of {x['key']!r} with no earlier verified archive")
            if x["key"] in locked:
                p.append(f"vault delete of locked {x['key']!r}")
    if ev["passed"] is not (not failed):
        p.append(f"passed {ev['passed']!r} but failed outcomes present = {failed}")
    if ev["passed"] is not True:
        p.append("passed is not true")
    return p


def make_evidence(log: dict) -> dict:
    failed = any(x["outcome"] == "failed" for x in log["actions"])
    return {"schema": EVIDENCE_SCHEMA, "wp": WP, "passed": not failed, "log": log}


def days(n: float) -> timedelta:
    return timedelta(days=n)
