"""W3a: the owner's live Hippius S3 storage probe (LANE_D_PLAN §1.2 "Storage").

Seven probes answer what the lane D vault can rely on:

* ``presigned_upload_part`` — can a miner PUT multipart parts through
  presigned URLs (no credential on the miner)?
* ``single_put_limit`` — does one presigned PUT carry a ~2.5 GB BF16 entry?
* ``object_lock_multipart`` — do COMPLIANCE lock headers on
  CreateMultipartUpload lock the completed object?
* ``prefix_bucket_policy`` — can a bucket policy scope writes by prefix?
* ``subtoken_scope`` — is a console sub-token confined to its bucket?
* ``upload_part_copy_2_5gb`` — can a ~2.5 GB king be server-side copied by
  UploadPartCopy (locking a king by copy at crowning)?
* ``compliance_retention_unshortenable`` — is COMPLIANCE retention really
  impossible to shorten or bypass?

Usage (owner, from a clean main checkout, with the ``HIPPIUS_*`` env set)::

    python -P -m cascade.laned.storage_probe --bucket <throwaway lock-enabled bucket> \\
        --prefix <throwaway prefix>/ --other-bucket <bucket outside the sub-token> \\
        --out evidence/w3a_probe.json --confirm-live

Credentials are read from the environment by name only
(``HIPPIUS_S3_ACCESS_KEY`` / ``HIPPIUS_S3_SECRET_KEY`` and the read-write
sub-token ``HIPPIUS_PROBE_SUBTOKEN_ACCESS_KEY`` /
``HIPPIUS_PROBE_SUBTOKEN_SECRET_KEY``); their values, presigned URLs and
signatures are never printed or written. The probe never administers buckets
(no create/delete, versioning, lock configuration, ACL or lifecycle calls).
Every object it creates lives under ``<prefix><run_id>/`` and is deleted at
the end, except the two tiny (<= 4096 bytes) COMPLIANCE-locked objects, which
are reported with their retain-until date (at most one day by default).

``--dry-run`` opens no connection: without ``--fixture`` it prints the plan;
with ``--fixture MODEL`` it runs every probe against an in-process backend
that behaves as the model says and writes ``source: "fixture"`` evidence,
which the verifier never accepts. Contract: tests/accept/w3a/README.md.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import io
import json
import math
import os
import re
import secrets
import struct
import subprocess
import sys
import tomllib
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import urlsplit

SCHEMA = "laned_w3a_probe/1"
FIXTURE_SCHEMA = "laned_w3a_fixture/1"
WP = "W3a"
MODULE = "cascade.laned.storage_probe"
REPO = Path(__file__).resolve().parents[2]

PROBES = ("presigned_upload_part", "single_put_limit", "object_lock_multipart",
          "prefix_bucket_policy", "subtoken_scope", "upload_part_copy_2_5gb",
          "compliance_retention_unshortenable")

MAIN_ACCESS_ENV = "HIPPIUS_S3_ACCESS_KEY"
MAIN_SECRET_ENV = "HIPPIUS_S3_SECRET_KEY"
SUB_ACCESS_ENV = "HIPPIUS_PROBE_SUBTOKEN_ACCESS_KEY"
SUB_SECRET_ENV = "HIPPIUS_PROBE_SUBTOKEN_SECRET_KEY"
CREDENTIAL_ENVS = (MAIN_ACCESS_ENV, MAIN_SECRET_ENV, SUB_ACCESS_ENV, SUB_SECRET_ENV)

DEFAULT_ENDPOINT = "https://s3.hippius.com"
DEFAULT_REGION = "decentralized"
MIB = 1024 * 1024
LIVE_DEFAULTS = {"big_object_bytes": 2_684_354_560, "part_bytes": 256 * MIB,
                 "single_put_bytes": [512 * MIB, 2_684_354_560]}
FIXTURE_DEFAULTS = {"big_object_bytes": 1_048_576, "part_bytes": 262_144,
                    "single_put_bytes": [524_288, 1_048_576]}
FIXTURE_MAX_BYTES = 16 * MIB
MIN_PART_BYTES = 5 * MIB
MAX_PART_BYTES = 512 * MIB
MAX_PARTS = 10_000
MIN_LOCK_S, MAX_LOCK_S = 60, 86_400
MAX_PRESIGN_S = 604_800
PRESIGNED_PART_SIZES = (5 * MIB, 1024)
LOCKED_OBJECT_BYTES = 1024
CHUNK = MIB

PREFIX_RE = re.compile(r"[a-z0-9][a-z0-9._-]*(/[a-z0-9][a-z0-9._-]*)*/")
URL_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9+.-]*://\S*")
REQUEST_ID_HEADERS = ("x-amz-request-id", "x-amz-id-2", "x-request-id")
LOCK_MODE = "COMPLIANCE"


class UsageError(ValueError):
    """A refusal before any storage access (exit 2)."""


# ── helpers ──────────────────────────────────────────────────────────────────


def _now() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def _iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).replace(microsecond=0).isoformat()


def scrub(text: str) -> str:
    """Strip URLs (a presigned URL carries a signature) and any credential
    value from a message before it is recorded or printed."""
    out = URL_RE.sub("<url>", str(text))
    for name in CREDENTIAL_ENVS:
        val = os.environ.get(name)
        if val:
            out = out.replace(val, "<redacted>")
    return out[:500]


def production_buckets(repo: Path = REPO) -> set[str]:
    """Every non-empty string value of a key ending in ``bucket`` in the
    repo's chain.toml / chain.testnet.toml, any table."""
    found: set[str] = set()

    def walk(node: object) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                if isinstance(v, str) and k.endswith("bucket") and v:
                    found.add(v)
                else:
                    walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    for name in ("chain.toml", "chain.testnet.toml"):
        p = repo / name
        if p.is_file():
            walk(tomllib.loads(p.read_text(encoding="utf-8")))
    return found


def tool_info() -> dict:
    def git(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True,
                              check=False)

    head = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")
    sha = head.stdout.strip() if head.returncode == 0 else ""
    dirty = status.returncode != 0 or bool(status.stdout.strip())
    return {"module": MODULE, "git_sha": sha, "git_dirty": dirty}


class Payload:
    """A deterministic byte pattern of ``size`` bytes, produced in 1 MiB
    chunks on demand (never one big buffer). Chunk ``i`` is a per-run base
    block with its first 8 bytes replaced by ``i``."""

    def __init__(self, size: int, seed: str):
        self.size = size
        base = hashlib.sha256(seed.encode()).digest()
        self._base = (base * (CHUNK // len(base) + 1))[:CHUNK]
        self._sha: str | None = None

    def _chunk(self, i: int) -> bytes:
        return struct.pack(">Q", i) + self._base[8:]

    def read_at(self, offset: int, n: int) -> bytes:
        end = min(self.size, offset + n)
        parts = []
        pos = offset
        while pos < end:
            i, off = divmod(pos, CHUNK)
            take = min(CHUNK - off, end - pos)
            parts.append(self._chunk(i)[off:off + take])
            pos += take
        return b"".join(parts)

    def chunks(self, start: int = 0, end: int | None = None):
        end = self.size if end is None else end
        pos = start
        while pos < end:
            take = min(CHUNK - pos % CHUNK, end - pos)
            yield self.read_at(pos, take)
            pos += take

    def sha256(self, start: int = 0, end: int | None = None) -> str:
        if start == 0 and end in (None, self.size) and self._sha is not None:
            return self._sha
        h = hashlib.sha256()
        for c in self.chunks(start, end):
            h.update(c)
        digest = h.hexdigest()
        if start == 0 and end in (None, self.size):
            self._sha = digest
        return digest


class PayloadReader(io.RawIOBase):
    """A seekable read-only file over ``payload[start:end]`` (for boto3
    bodies, which may seek to retry or checksum)."""

    def __init__(self, payload: Payload, start: int, end: int):
        self._p, self._start, self._end, self._pos = payload, start, end, 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = 0) -> int:
        length = self._end - self._start
        base = {0: 0, 1: self._pos, 2: length}[whence]
        self._pos = max(0, min(length, base + offset))
        return self._pos

    def readinto(self, b) -> int:
        n = min(len(b), self._end - self._start - self._pos)
        if n <= 0:
            return 0
        data = self._p.read_at(self._start + self._pos, n)
        b[:n] = data
        self._pos += n
        return n


@dataclass
class Resp:
    """One storage response: HTTP status (None = no response), request id,
    S3 error code and op-specific data."""

    status: int | None
    request_id: str | None = None
    error_code: str | None = None
    data: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status is not None and 200 <= self.status < 300


# ── the live backend (boto3 + plain HTTP for presigned URLs) ─────────────────


def _rid_from_headers(headers: dict) -> str | None:
    low = {str(k).lower(): v for k, v in (headers or {}).items()}
    for h in REQUEST_ID_HEADERS:
        if low.get(h):
            return str(low[h])
    return None


class LiveBackend:
    """Hippius S3 through boto3 (credentialed ops) and http.client (presigned
    URLs, sent with no Authorization header)."""

    def __init__(self, endpoint: str, region: str, bucket: str, timeout: float = 600.0):
        import boto3  # lazy: never imported in --dry-run
        from botocore.config import Config

        self.bucket = bucket
        self.timeout = timeout
        cfg = Config(signature_version="s3v4", s3={"addressing_style": "path"},
                     connect_timeout=30, read_timeout=600,
                     retries={"max_attempts": 2, "mode": "standard"},
                     request_checksum_calculation="when_required",
                     response_checksum_validation="when_required")

        def client(access_env: str, secret_env: str):
            return boto3.client("s3", endpoint_url=endpoint, region_name=region,
                                aws_access_key_id=os.environ[access_env],
                                aws_secret_access_key=os.environ[secret_env], config=cfg)

        self._main = client(MAIN_ACCESS_ENV, MAIN_SECRET_ENV)
        self._sub = client(SUB_ACCESS_ENV, SUB_SECRET_ENV) if self.has_subtoken() else None

    @staticmethod
    def has_subtoken() -> bool:
        return bool(os.environ.get(SUB_ACCESS_ENV) and os.environ.get(SUB_SECRET_ENV))

    def _c(self, cred: str):
        if cred == "subtoken":
            if self._sub is None:
                raise RuntimeError("no sub-token configured")
            return self._sub
        return self._main

    def _call(self, cred: str, op: str, **kw) -> Resp:
        from botocore.exceptions import ClientError

        try:
            out = getattr(self._c(cred), op)(**kw)
        except ClientError as e:
            meta = e.response.get("ResponseMetadata", {})
            return Resp(meta.get("HTTPStatusCode"),
                        meta.get("RequestId") or _rid_from_headers(meta.get("HTTPHeaders", {})),
                        e.response.get("Error", {}).get("Code"))
        meta = out.get("ResponseMetadata", {})
        return Resp(meta.get("HTTPStatusCode"),
                    meta.get("RequestId") or _rid_from_headers(meta.get("HTTPHeaders", {})),
                    None, {"raw": out})

    # presigned (no credential on the wire)
    def _http_put(self, url: str, payload: Payload, start: int, end: int) -> Resp:
        parts = urlsplit(url)
        conn_cls = http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
        conn = conn_cls(parts.hostname, parts.port, timeout=self.timeout)
        try:
            target = parts.path + ("?" + parts.query if parts.query else "")
            conn.putrequest("PUT", target, skip_accept_encoding=True)
            conn.putheader("Content-Length", str(end - start))
            conn.endheaders()
            for c in payload.chunks(start, end):
                conn.send(c)
            r = conn.getresponse()
            body = r.read(65536)
            headers = dict(r.getheaders())
        finally:
            conn.close()
        code = None
        m = re.search(rb"<Code>([^<]+)</Code>", body or b"")
        if m:
            code = m.group(1).decode("utf-8", "replace")
        etag = {k.lower(): v for k, v in headers.items()}.get("etag")
        return Resp(r.status, _rid_from_headers(headers), code, {"etag": etag})

    def presigned_upload_part(self, key, upload_id, part_number, payload, start, end, expires):
        url = self._main.generate_presigned_url(
            "upload_part", Params={"Bucket": self.bucket, "Key": key, "UploadId": upload_id,
                                   "PartNumber": part_number},
            ExpiresIn=expires, HttpMethod="PUT")
        return self._http_put(url, payload, start, end)

    def presigned_put(self, key, payload, expires):
        url = self._main.generate_presigned_url(
            "put_object", Params={"Bucket": self.bucket, "Key": key}, ExpiresIn=expires,
            HttpMethod="PUT")
        return self._http_put(url, payload, 0, payload.size)

    # credentialed object ops
    def create_multipart(self, key, lock=None):
        kw = {"Bucket": self.bucket, "Key": key}
        if lock:
            kw.update(ObjectLockMode=lock[0], ObjectLockRetainUntilDate=lock[1])
        r = self._call("main", "create_multipart_upload", **kw)
        if r.ok:
            r.data = {"upload_id": r.data["raw"].get("UploadId")}
        return r

    def upload_part(self, key, upload_id, part_number, payload, start, end):
        r = self._call("main", "upload_part", Bucket=self.bucket, Key=key, UploadId=upload_id,
                       PartNumber=part_number, Body=PayloadReader(payload, start, end),
                       ContentLength=end - start)
        if r.ok:
            r.data = {"etag": r.data["raw"].get("ETag")}
        return r

    def upload_part_copy(self, key, upload_id, part_number, src_key, start, end):
        r = self._call("main", "upload_part_copy", Bucket=self.bucket, Key=key,
                       UploadId=upload_id, PartNumber=part_number,
                       CopySource={"Bucket": self.bucket, "Key": src_key},
                       CopySourceRange=f"bytes={start}-{end}")
        if r.ok:
            r.data = {"etag": r.data["raw"].get("CopyPartResult", {}).get("ETag")}
        return r

    def complete_multipart(self, key, upload_id, parts):
        return self._call("main", "complete_multipart_upload", Bucket=self.bucket, Key=key,
                          UploadId=upload_id,
                          MultipartUpload={"Parts": [{"PartNumber": n, "ETag": e}
                                                     for n, e in parts]})

    def abort_multipart(self, key, upload_id):
        return self._call("main", "abort_multipart_upload", Bucket=self.bucket, Key=key,
                          UploadId=upload_id)

    def list_open_uploads(self, prefix):
        r = self._call("main", "list_multipart_uploads", Bucket=self.bucket, Prefix=prefix)
        if not r.ok:
            return []
        return [(u["Key"], u["UploadId"]) for u in r.data["raw"].get("Uploads", []) or []]

    def put_object(self, key, payload, lock=None, cred="main"):
        kw = {"Bucket": self.bucket, "Key": key, "Body": PayloadReader(payload, 0, payload.size),
              "ContentLength": payload.size}
        if lock:
            kw.update(ObjectLockMode=lock[0], ObjectLockRetainUntilDate=lock[1])
        r = self._call(cred, "put_object", **kw)
        if r.ok:
            r.data = {"version_id": r.data["raw"].get("VersionId")}
        return r

    def get_object(self, key, cred="main"):
        from botocore.exceptions import ClientError

        try:
            out = self._c(cred).get_object(Bucket=self.bucket, Key=key)
        except ClientError as e:
            meta = e.response.get("ResponseMetadata", {})
            return Resp(meta.get("HTTPStatusCode"),
                        meta.get("RequestId") or _rid_from_headers(meta.get("HTTPHeaders", {})),
                        e.response.get("Error", {}).get("Code"))
        meta = out.get("ResponseMetadata", {})
        h, size = hashlib.sha256(), 0
        for c in out["Body"].iter_chunks(CHUNK):
            h.update(c)
            size += len(c)
        return Resp(meta.get("HTTPStatusCode"),
                    meta.get("RequestId") or _rid_from_headers(meta.get("HTTPHeaders", {})),
                    None, {"sha256": h.hexdigest(), "size": size})

    def head_object(self, key):
        r = self._call("main", "head_object", Bucket=self.bucket, Key=key)
        if r.ok:
            raw = r.data["raw"]
            r.data = {"size": raw.get("ContentLength"), "mode": raw.get("ObjectLockMode"),
                      "retain_until": _iso(raw.get("ObjectLockRetainUntilDate")),
                      "version_id": raw.get("VersionId")}
        return r

    def copy_object(self, dst_key, src_key):
        return self._call("main", "copy_object", Bucket=self.bucket, Key=dst_key,
                          CopySource={"Bucket": self.bucket, "Key": src_key})

    def put_object_retention(self, key, version_id, until, bypass):
        kw = {"Bucket": self.bucket, "Key": key,
              "Retention": {"Mode": LOCK_MODE, "RetainUntilDate": until}}
        if version_id:
            kw["VersionId"] = version_id
        if bypass:
            kw["BypassGovernanceRetention"] = True
        return self._call("main", "put_object_retention", **kw)

    def delete_object(self, key, version_id=None):
        kw = {"Bucket": self.bucket, "Key": key}
        if version_id:
            kw["VersionId"] = version_id
        return self._call("main", "delete_object", **kw)

    # bucket reads, the policy probe's own policy, other-bucket reads
    def bucket_lock_enabled(self):
        r = self._call("main", "get_object_lock_configuration", Bucket=self.bucket)
        enabled = r.ok and (r.data["raw"].get("ObjectLockConfiguration", {})
                            .get("ObjectLockEnabled") == "Enabled")
        r.data = {"enabled": enabled}
        return r

    def get_bucket_policy(self):
        r = self._call("main", "get_bucket_policy", Bucket=self.bucket)
        r.data = {"exists": r.ok}
        return r

    def put_bucket_policy(self, doc):
        return self._call("main", "put_bucket_policy", Bucket=self.bucket, Policy=json.dumps(doc))

    def delete_bucket_policy(self):
        return self._call("main", "delete_bucket_policy", Bucket=self.bucket)

    def head_bucket(self, bucket, cred="main"):
        return self._call(cred, "head_bucket", Bucket=bucket)

    def list_bucket(self, bucket, cred="main"):
        return self._call(cred, "list_objects_v2", Bucket=bucket, MaxKeys=1)

    def count_prefix(self, prefix):
        r = self._call("main", "list_objects_v2", Bucket=self.bucket, Prefix=prefix, MaxKeys=1)
        if not r.ok:
            raise RuntimeError(f"cannot list the prefix (status {r.status}, {r.error_code})")
        return int(r.data["raw"].get("KeyCount", 0))

    def list_versions(self, prefix):
        """[(key, version_id)] of every version and delete marker under
        ``prefix``; falls back to plain keys if versions can't be listed."""
        out = []
        try:
            pages = self._main.get_paginator("list_object_versions").paginate(
                Bucket=self.bucket, Prefix=prefix)
            for page in pages:
                for v in (page.get("Versions") or []) + (page.get("DeleteMarkers") or []):
                    out.append((v["Key"], v.get("VersionId")))
            return out
        except Exception:  # noqa: BLE001 — no version listing: fall back to keys
            out = []
        for page in self._main.get_paginator("list_objects_v2").paginate(Bucket=self.bucket,
                                                                        Prefix=prefix):
            for o in page.get("Contents") or []:
                out.append((o["Key"], None))
        return out


# ── the fixture backend (dry-run only; no network) ──────────────────────────


def load_fixture(path: Path) -> dict:
    try:
        model = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise UsageError(f"--fixture: cannot read the model: {e}") from None
    if not isinstance(model, dict) or model.get("schema") != FIXTURE_SCHEMA:
        raise UsageError(f"--fixture: want schema {FIXTURE_SCHEMA!r}")
    want = {"endpoint_host": str, "region": str, "prefix_object_count": int,
            "bucket_lock_enabled": bool, "bucket_has_policy": bool,
            "presigned_upload_part": ("supported", "rejected"),
            "single_put_max_bytes": int, "copy_object_max_bytes": int,
            "upload_part_copy": ("supported", "rejected"),
            "lock_on_multipart": ("applied", "ignored", "rejected"),
            "bucket_policy": ("enforced", "accepted_not_enforced", "rejected"),
            "subtoken": ("bucket_scoped", "unscoped", None),
            "compliance": ("unshortenable", "shortenable")}
    for k, t in want.items():
        if k not in model:
            raise UsageError(f"--fixture: missing key {k!r}")
        v = model[k]
        ok = v in t if isinstance(t, tuple) else (isinstance(v, t) and not
                                                  (t is int and isinstance(v, bool)))
        if not ok:
            raise UsageError(f"--fixture: bad value for {k!r}")
    return model


class FixtureBackend:
    """An in-process object store that behaves as the fixture model says.
    Request ids are ``fixture-<n>``; no URL or credential exists here."""

    def __init__(self, model: dict, bucket: str):
        self.m = model
        self.bucket = bucket
        self._n = 0
        self._objects: dict[str, list[dict]] = {}
        self._uploads: dict[str, dict] = {}
        self._policy: dict | None = None

    def _r(self, status: int, code: str | None = None, **data) -> Resp:
        self._n += 1
        return Resp(status, f"fixture-{self._n}", code, data)

    def has_subtoken(self) -> bool:
        return self.m["subtoken"] is not None

    def _lock_for(self, lock):
        if lock and self.m["bucket_lock_enabled"]:
            return {"mode": lock[0], "retain_until": lock[1]}
        return None

    def _new_version(self, key: str, data: bytes, lock) -> str:
        self._n += 1
        vid = f"v{self._n}"
        self._objects.setdefault(key, []).append({"vid": vid, "data": data, "lock": lock})
        return vid

    def _latest(self, key: str):
        vs = self._objects.get(key) or []
        return vs[-1] if vs else None

    def _denied(self, key: str) -> bool:
        if not self._policy or self.m["bucket_policy"] != "enforced":
            return False
        for s in self._policy.get("Statement", []):
            if s.get("Effect") != "Deny":
                continue
            for res in s.get("Resource", []):
                pre = res.removeprefix(f"arn:aws:s3:::{self.bucket}/").rstrip("*")
                if key.startswith(pre):
                    return True
        return False

    def presigned_upload_part(self, key, upload_id, part_number, payload, start, end, expires):
        if self.m["presigned_upload_part"] == "rejected":
            return self._r(403, "SignatureDoesNotMatch")
        return self.upload_part(key, upload_id, part_number, payload, start, end)

    def presigned_put(self, key, payload, expires):
        if payload.size > self.m["single_put_max_bytes"]:
            return self._r(400, "EntityTooLarge")
        self._new_version(key, b"".join(payload.chunks()), None)
        return self._r(200)

    def create_multipart(self, key, lock=None):
        if lock and self.m["lock_on_multipart"] == "rejected":
            return self._r(400, "InvalidRequest")
        applied = lock if self.m["lock_on_multipart"] == "applied" else None
        uid = secrets.token_hex(8)
        self._uploads[uid] = {"key": key, "parts": {}, "lock": self._lock_for(applied)}
        return self._r(200, upload_id=uid)

    def upload_part(self, key, upload_id, part_number, payload, start, end):
        up = self._uploads.get(upload_id)
        if not up or up["key"] != key:
            return self._r(404, "NoSuchUpload")
        up["parts"][part_number] = b"".join(payload.chunks(start, end))
        return self._r(200, etag=f'"{hashlib.md5(up["parts"][part_number]).hexdigest()}"')

    def upload_part_copy(self, key, upload_id, part_number, src_key, start, end):
        if self.m["upload_part_copy"] == "rejected":
            return self._r(501, "NotImplemented")
        up, src = self._uploads.get(upload_id), self._latest(src_key)
        if not up or src is None:
            return self._r(404, "NoSuchKey")
        up["parts"][part_number] = src["data"][start:end + 1]
        return self._r(200, etag=f'"{hashlib.md5(up["parts"][part_number]).hexdigest()}"')

    def complete_multipart(self, key, upload_id, parts):
        up = self._uploads.get(upload_id)
        if not up or any(n not in up["parts"] for n, _ in parts):
            return self._r(400, "InvalidPart")
        data = b"".join(up["parts"][n] for n, _ in parts)
        self._new_version(key, data, up["lock"])
        del self._uploads[upload_id]
        return self._r(200)

    def abort_multipart(self, key, upload_id):
        self._uploads.pop(upload_id, None)
        return self._r(204)

    def list_open_uploads(self, prefix):
        return [(u["key"], uid) for uid, u in self._uploads.items() if u["key"].startswith(prefix)]

    def put_object(self, key, payload, lock=None, cred="main"):
        if cred == "subtoken" and not self.has_subtoken():
            return self._r(403, "InvalidAccessKeyId")
        if self._denied(key):
            return self._r(403, "AccessDenied")
        vid = self._new_version(key, b"".join(payload.chunks()), self._lock_for(lock))
        return self._r(200, version_id=vid)

    def get_object(self, key, cred="main"):
        v = self._latest(key)
        if v is None:
            return self._r(404, "NoSuchKey")
        return self._r(200, sha256=hashlib.sha256(v["data"]).hexdigest(), size=len(v["data"]))

    def head_object(self, key):
        v = self._latest(key)
        if v is None:
            return self._r(404, "NotFound")
        lock = v["lock"] or {}
        return self._r(200, size=len(v["data"]), mode=lock.get("mode"),
                       retain_until=_iso(lock.get("retain_until")), version_id=v["vid"])

    def copy_object(self, dst_key, src_key):
        src = self._latest(src_key)
        if src is None:
            return self._r(404, "NoSuchKey")
        if len(src["data"]) > self.m["copy_object_max_bytes"]:
            return self._r(400, "InvalidRequest")
        self._new_version(dst_key, src["data"], None)
        return self._r(200)

    def _version(self, key, version_id):
        for v in self._objects.get(key) or []:
            if v["vid"] == version_id or version_id is None:
                return v
        return None

    def put_object_retention(self, key, version_id, until, bypass):
        v = self._version(key, version_id)
        if v is None:
            return self._r(404, "NoSuchKey")
        lock = v["lock"]
        if lock and until < lock["retain_until"] and self.m["compliance"] == "unshortenable":
            return self._r(403, "AccessDenied")
        v["lock"] = {"mode": LOCK_MODE, "retain_until": until}
        return self._r(200)

    def delete_object(self, key, version_id=None):
        v = self._version(key, version_id)
        if v is None:
            return self._r(404, "NoSuchKey")
        lock = v["lock"]
        if lock and lock["retain_until"] > _now():
            return self._r(403, "AccessDenied")
        self._objects[key].remove(v)
        if not self._objects[key]:
            del self._objects[key]
        return self._r(204)

    def bucket_lock_enabled(self):
        return self._r(200, enabled=self.m["bucket_lock_enabled"])

    def get_bucket_policy(self):
        exists = self.m["bucket_has_policy"] or self._policy is not None
        return self._r(200 if exists else 404, None if exists else "NoSuchBucketPolicy",
                       exists=exists)

    def put_bucket_policy(self, doc):
        if self.m["bucket_policy"] == "rejected":
            return self._r(501, "NotImplemented")
        self._policy = doc
        return self._r(204)

    def delete_bucket_policy(self):
        self._policy = None
        return self._r(204)

    def head_bucket(self, bucket, cred="main"):
        return self._r(200)

    def list_bucket(self, bucket, cred="main"):
        if cred == "subtoken" and bucket != self.bucket and self.m["subtoken"] == "bucket_scoped":
            return self._r(403, "AccessDenied")
        return self._r(200)

    def count_prefix(self, prefix):
        return self.m["prefix_object_count"] + sum(1 for k in self._objects if k.startswith(prefix))

    def list_versions(self, prefix):
        return [(k, v["vid"]) for k, vs in self._objects.items() if k.startswith(prefix)
                for v in vs]


# ── the probes ───────────────────────────────────────────────────────────────


@dataclass
class Run:
    backend: object
    bucket: str
    prefix: str
    run_id: str
    other_bucket: str
    params: dict
    locked: list = field(default_factory=list)
    uploads: list = field(default_factory=list)
    policy_left: bool = False

    @property
    def root(self) -> str:
        return f"{self.prefix}{self.run_id}/"


class Rec:
    """One probe's record: its requests, objects and observations."""

    def __init__(self, name: str):
        self.name = name
        self.requests: list[dict] = []
        self.objects: list[dict] = []
        self.obs: dict = {}

    def call(self, op: str, r: Resp) -> Resp:
        self.requests.append({"op": op, "http_status": r.status, "request_id": r.request_id})
        return r

    def obj(self, key: str, size: int, sha256: str | None) -> None:
        self.objects.append({"key": key, "size": size, "sha256": sha256})


def _ranges(size: int, part: int) -> list[tuple[int, int]]:
    """Half-open ranges of at most ``part`` bytes covering ``[0, size)``."""
    return [(s, min(size, s + part)) for s in range(0, size, part)]


def _upload_whole(run: Run, rec: Rec, key: str, payload: Payload, label: str) -> int | None:
    """Credentialed multipart upload of ``payload``; the completing status."""
    b = run.backend
    r = rec.call(f"CreateMultipartUpload({label})", b.create_multipart(key))
    if not r.ok:
        return r.status
    uid = r.data["upload_id"]
    run.uploads.append((key, uid))
    etags = []
    for n, (s, e) in enumerate(_ranges(payload.size, run.params["part_bytes"]), start=1):
        p = rec.call(f"UploadPart({label})", b.upload_part(key, uid, n, payload, s, e))
        if not p.ok:
            return p.status
        etags.append((n, p.data.get("etag")))
    c = rec.call(f"CompleteMultipartUpload({label})", b.complete_multipart(key, uid, etags))
    return c.status


def probe_presigned_upload_part(run: Run, rec: Rec) -> str:
    b, key = run.backend, f"{run.root}presigned-mpu.bin"
    sizes = PRESIGNED_PART_SIZES
    payload = Payload(sum(sizes), f"{run.run_id}:presigned")
    expires = run.params["presign_expires_seconds"]
    o = rec.obs
    o.update(create_status=None, url_expires_seconds=expires, authorization_header_sent=False,
             parts=[], complete_status=None, uploaded_sha256=payload.sha256(),
             object_size=None, object_sha256=None)
    r = rec.call("CreateMultipartUpload", b.create_multipart(key))
    o["create_status"] = r.status
    rec.obj(key, payload.size, None)
    if not r.ok:
        return "error"
    uid = r.data["upload_id"]
    run.uploads.append((key, uid))
    etags, start = [], 0
    for n, size in enumerate(sizes, start=1):
        p = rec.call("UploadPart(presigned)",
                     b.presigned_upload_part(key, uid, n, payload, start, start + size, expires))
        etag = p.data.get("etag") if p.ok else None
        o["parts"].append({"part_number": n, "size": size, "http_status": p.status,
                           "etag_returned": bool(etag)})
        etags.append((n, etag))
        start += size
    if all(p["http_status"] == 200 and p["etag_returned"] for p in o["parts"]):
        c = rec.call("CompleteMultipartUpload", b.complete_multipart(key, uid, etags))
        o["complete_status"] = c.status
        if c.ok:
            g = rec.call("GetObject", b.get_object(key))
            if g.ok:
                o["object_size"], o["object_sha256"] = g.data["size"], g.data["sha256"]
                rec.objects[-1]["sha256"] = g.data["sha256"]
    ok = (o["complete_status"] == 200 and o["object_size"] == payload.size
          and o["object_sha256"] == o["uploaded_sha256"])
    return "pass" if ok else "fail"


def probe_single_put_limit(run: Run, rec: Rec) -> str:
    b = run.backend
    required = run.params["big_object_bytes"]
    o = rec.obs
    o.update(required_bytes=required, authorization_header_sent=False, attempts=[],
             max_accepted_bytes=None, min_rejected_bytes=None)
    for size in run.params["single_put_bytes"]:
        key = f"{run.root}single-put-{size}.bin"
        payload = Payload(size, f"{run.run_id}:single:{size}")
        r = rec.call("PutObject(presigned)",
                     b.presigned_put(key, payload, run.params["presign_expires_seconds"]))
        obj_size = None
        if r.ok:
            h = rec.call("HeadObject", b.head_object(key))
            obj_size = h.data.get("size") if h.ok else None
        accepted = r.status == 200 and obj_size == size
        sha = payload.sha256() if accepted else None
        o["attempts"].append({"size": size, "http_status": r.status,
                              "error_code": r.error_code, "object_size": obj_size,
                              "sha256": sha})
        rec.obj(key, size, sha)
        if accepted:
            o["max_accepted_bytes"] = size
        else:
            o["min_rejected_bytes"] = size
            break
    ok = o["max_accepted_bytes"] is not None and o["max_accepted_bytes"] >= required
    return "pass" if ok else "fail"


def _retain_until(run: Run) -> datetime:
    return _now() + timedelta(seconds=run.params["lock_retention_seconds"])


def probe_object_lock_multipart(run: Run, rec: Rec) -> str:
    b, key = run.backend, f"{run.root}lock-mpu.bin"
    payload = Payload(LOCKED_OBJECT_BYTES, f"{run.run_id}:lock-mpu")
    lk = rec.call("GetObjectLockConfiguration", b.bucket_lock_enabled())
    until = _retain_until(run)
    o = rec.obs
    o.update(bucket_lock_enabled=bool(lk.data.get("enabled")), requested_mode=LOCK_MODE,
             requested_retain_until=_iso(until), create_status=None, complete_status=None,
             head_mode=None, head_retain_until=None, object_size=None)
    rec.obj(key, payload.size, None)
    if not o["bucket_lock_enabled"]:
        return "error"
    r = rec.call("CreateMultipartUpload(lock headers)",
                 b.create_multipart(key, lock=(LOCK_MODE, until)))
    o["create_status"] = r.status
    if r.ok:
        uid = r.data["upload_id"]
        run.uploads.append((key, uid))
        p = rec.call("UploadPart", b.upload_part(key, uid, 1, payload, 0, payload.size))
        if p.ok:
            c = rec.call("CompleteMultipartUpload",
                         b.complete_multipart(key, uid, [(1, p.data.get("etag"))]))
            o["complete_status"] = c.status
            if c.ok:
                h = rec.call("HeadObject", b.head_object(key))
                if h.ok:
                    o.update(head_mode=h.data.get("mode"),
                             head_retain_until=h.data.get("retain_until"),
                             object_size=h.data.get("size"))
                    rec.objects[-1]["sha256"] = payload.sha256()
                    if o["head_mode"] == LOCK_MODE:
                        run.locked.append({"key": key, "version_id": h.data.get("version_id"),
                                           "mode": LOCK_MODE,
                                           "retain_until": o["head_retain_until"],
                                           "size": payload.size})
    ok = (o["create_status"] == 200 and o["complete_status"] == 200
          and o["head_mode"] == LOCK_MODE and o["head_retain_until"] is not None
          and abs((datetime.fromisoformat(o["head_retain_until"]) - until).total_seconds()) <= 1)
    return "pass" if ok else "fail"


def probe_prefix_bucket_policy(run: Run, rec: Rec) -> str:
    b = run.backend
    denied_key = f"{run.root}policy/denied/object.bin"
    allowed_key = f"{run.root}policy/allowed/object.bin"
    o = rec.obs
    o.update(preexisting_policy=None, policy=None, credential=None, put_policy_status=None,
             denied_put_status=None, allowed_put_status=None, policy_removed=None)
    g = rec.call("GetBucketPolicy", b.get_bucket_policy())
    if g.data.get("exists"):
        o["preexisting_policy"] = True
        rec.obj(allowed_key, 0, None)
        return "error"
    if g.status != 404:
        o["preexisting_policy"] = None
        rec.obj(allowed_key, 0, None)
        raise RuntimeError(f"GetBucketPolicy answered {g.status} ({g.error_code}): cannot "
                           "tell whether a policy exists, so it is not touched")
    o["preexisting_policy"] = False
    doc = {"Version": "2012-10-17",
           "Statement": [{"Sid": "W3aProbeDenyPrefix", "Effect": "Deny", "Principal": "*",
                          "Action": ["s3:PutObject"],
                          "Resource": [f"arn:aws:s3:::{run.bucket}/{run.root}policy/denied/*"]}]}
    o["policy"] = doc
    cred = "subtoken" if b.has_subtoken() else "main"
    o["credential"] = cred
    payload = Payload(1024, f"{run.run_id}:policy")
    try:
        p = rec.call("PutBucketPolicy", b.put_bucket_policy(doc))
        o["put_policy_status"] = p.status
        if p.ok:
            run.policy_left = True
            d = rec.call(f"PutObject(denied,{cred})", b.put_object(denied_key, payload, cred=cred))
            a = rec.call(f"PutObject(allowed,{cred})", b.put_object(allowed_key, payload, cred=cred))
            o["denied_put_status"], o["allowed_put_status"] = d.status, a.status
            rec.obj(denied_key, payload.size, payload.sha256() if d.ok else None)
            rec.obj(allowed_key, payload.size, payload.sha256() if a.ok else None)
        else:
            rec.obj(allowed_key, payload.size, None)
    finally:
        if run.policy_left:
            rec.call("DeleteBucketPolicy", b.delete_bucket_policy())
        after = rec.call("GetBucketPolicy", b.get_bucket_policy())
        o["policy_removed"] = not after.data.get("exists")
        run.policy_left = not o["policy_removed"]
    ok = (o["put_policy_status"] in (200, 204) and o["denied_put_status"] == 403
          and o["allowed_put_status"] == 200)
    return "pass" if ok else "fail"


def probe_subtoken_scope(run: Run, rec: Rec) -> str:
    b, key = run.backend, f"{run.root}subtoken/in-bucket.bin"
    payload = Payload(1024, f"{run.run_id}:subtoken")
    o = rec.obs
    o.update(subtoken_present=b.has_subtoken(), other_bucket=run.other_bucket,
             other_bucket_exists=None, in_bucket_put_status=None, in_bucket_get_status=None,
             other_bucket_list_status=None)
    rec.obj(key, payload.size, None)
    if not o["subtoken_present"]:
        return "error"
    h = rec.call("HeadBucket(other,main)", b.head_bucket(run.other_bucket, cred="main"))
    o["other_bucket_exists"] = h.status == 200
    if not o["other_bucket_exists"]:
        return "error"
    p = rec.call("PutObject(subtoken)", b.put_object(key, payload, cred="subtoken"))
    o["in_bucket_put_status"] = p.status
    if p.ok:
        rec.objects[-1]["sha256"] = payload.sha256()
        g = rec.call("GetObject(subtoken)", b.get_object(key, cred="subtoken"))
        o["in_bucket_get_status"] = g.status
    lst = rec.call("ListObjectsV2(other,subtoken)", b.list_bucket(run.other_bucket, cred="subtoken"))
    o["other_bucket_list_status"] = lst.status
    ok = (o["in_bucket_put_status"] == 200 and o["in_bucket_get_status"] == 200
          and o["other_bucket_list_status"] == 403)
    return "pass" if ok else "fail"


def probe_upload_part_copy(run: Run, rec: Rec) -> str:
    b = run.backend
    size = run.params["big_object_bytes"]
    src, single, dst = (f"{run.root}copy-src.bin", f"{run.root}copy-single.bin",
                        f"{run.root}copy-dst.bin")
    payload = Payload(size, f"{run.run_id}:copy")
    o = rec.obs
    o.update(source_status=None, source_size=size, source_sha256=None, copy_object_status=None,
             copy_object_error_code=None, parts=[], complete_status=None, dest_size=None,
             dest_sha256=None)
    o["source_status"] = _upload_whole(run, rec, src, payload, "source")
    rec.obj(src, size, None)
    if o["source_status"] != 200:
        return "error"
    o["source_sha256"] = payload.sha256()
    rec.objects[-1]["sha256"] = o["source_sha256"]
    c1 = rec.call("CopyObject(single)", b.copy_object(single, src))
    o["copy_object_status"], o["copy_object_error_code"] = c1.status, c1.error_code
    rec.obj(single, size, o["source_sha256"] if c1.ok else None)
    r = rec.call("CreateMultipartUpload(dest)", b.create_multipart(dst))
    rec.obj(dst, size, None)
    if r.ok:
        uid = r.data["upload_id"]
        run.uploads.append((dst, uid))
        etags = []
        for n, (s, e) in enumerate(_ranges(size, run.params["part_bytes"]), start=1):
            p = rec.call("UploadPartCopy", b.upload_part_copy(dst, uid, n, src, s, e - 1))
            o["parts"].append({"part_number": n, "range_start": s, "range_end": e - 1,
                               "http_status": p.status})
            etags.append((n, p.data.get("etag") if p.ok else None))
            if not p.ok:
                break
        if o["parts"] and all(p["http_status"] == 200 for p in o["parts"]):
            c = rec.call("CompleteMultipartUpload(dest)", b.complete_multipart(dst, uid, etags))
            o["complete_status"] = c.status
            if c.ok:
                g = rec.call("GetObject(dest)", b.get_object(dst))
                if g.ok:
                    o["dest_size"], o["dest_sha256"] = g.data["size"], g.data["sha256"]
                    rec.objects[-1]["sha256"] = g.data["sha256"]
    ok = (o["complete_status"] == 200 and o["dest_size"] == size
          and o["dest_sha256"] == o["source_sha256"])
    return "pass" if ok else "fail"


def probe_compliance_retention(run: Run, rec: Rec) -> str:
    b, key = run.backend, f"{run.root}compliance.bin"
    payload = Payload(LOCKED_OBJECT_BYTES, f"{run.run_id}:compliance")
    lk = rec.call("GetObjectLockConfiguration", b.bucket_lock_enabled())
    o = rec.obs
    o.update(bucket_lock_enabled=bool(lk.data.get("enabled")), put_status=None, mode=None,
             retain_until=None, shorten_to=None, shorten_status=None,
             shorten_bypass_status=None, delete_version_status=None, retain_until_after=None,
             object_size=payload.size)
    rec.obj(key, payload.size, None)
    if not o["bucket_lock_enabled"]:
        return "error"
    until = _retain_until(run)
    p = rec.call("PutObject(COMPLIANCE)", b.put_object(key, payload, lock=(LOCK_MODE, until)))
    o["put_status"] = p.status
    if not p.ok:
        return "error"
    rec.objects[-1]["sha256"] = payload.sha256()
    h = rec.call("HeadObject", b.head_object(key))
    vid = h.data.get("version_id") or p.data.get("version_id")
    o["mode"], o["retain_until"] = h.data.get("mode"), h.data.get("retain_until")
    if o["mode"] != LOCK_MODE or o["retain_until"] is None:
        return "error"
    retain = datetime.fromisoformat(o["retain_until"])
    shorten_to = retain - timedelta(seconds=max(30, run.params["lock_retention_seconds"] // 2))
    o["shorten_to"] = _iso(shorten_to)
    s = rec.call("PutObjectRetention(shorten)", b.put_object_retention(key, vid, shorten_to, False))
    o["shorten_status"] = s.status
    s2 = rec.call("PutObjectRetention(shorten,bypass)",
                  b.put_object_retention(key, vid, shorten_to, True))
    o["shorten_bypass_status"] = s2.status
    d = rec.call("DeleteObject(version)", b.delete_object(key, vid))
    o["delete_version_status"] = d.status
    h2 = rec.call("HeadObject(after)", b.head_object(key))
    o["retain_until_after"] = h2.data.get("retain_until") if h2.ok else None
    run.locked.append({"key": key, "version_id": vid, "mode": LOCK_MODE,
                       "retain_until": o["retain_until_after"] or o["retain_until"],
                       "size": payload.size})
    refused = all((x or 0) >= 400 for x in (s.status, s2.status, d.status))
    ok = refused and o["retain_until_after"] is not None and abs(
        (datetime.fromisoformat(o["retain_until_after"]) - retain).total_seconds()) <= 1
    return "pass" if ok else "fail"


PROBE_FUNCS = {"presigned_upload_part": probe_presigned_upload_part,
               "single_put_limit": probe_single_put_limit,
               "object_lock_multipart": probe_object_lock_multipart,
               "prefix_bucket_policy": probe_prefix_bucket_policy,
               "subtoken_scope": probe_subtoken_scope,
               "upload_part_copy_2_5gb": probe_upload_part_copy,
               "compliance_retention_unshortenable": probe_compliance_retention}

# The order the probes run in: the small ones first, the 2.5 GB ones last.
RUN_ORDER = ("compliance_retention_unshortenable", "object_lock_multipart",
             "prefix_bucket_policy", "subtoken_scope", "presigned_upload_part",
             "single_put_limit", "upload_part_copy_2_5gb")

REASONS = {"pass": "the capability behaves as lane D needs",
           "fail": "recorded live result: the capability is absent or not enforced",
           "error": "no conclusive result: see the observations"}


def run_probe(run: Run, name: str) -> dict:
    rec = Rec(name)
    try:
        result = PROBE_FUNCS[name](run, rec)
        reason = REASONS[result]
    except Exception as e:  # noqa: BLE001 — one probe's failure never stops the run
        result, reason = "error", scrub(f"{type(e).__name__}: {e}")
    return {"name": name, "result": result, "passed": result == "pass", "reason": reason,
            "requests": rec.requests, "objects": rec.objects, "observations": rec.obs}


def cleanup(run: Run) -> dict:
    """Abort unfinished uploads and delete every version under the run root;
    only COMPLIANCE-locked objects may survive."""
    b = run.backend
    aborted = 0
    seen = set()
    for key, uid in list(run.uploads) + list(b.list_open_uploads(run.root)):
        if uid in seen:
            continue
        seen.add(uid)
        if key.startswith(run.root) and b.abort_multipart(key, uid).ok:
            aborted += 1
    deleted = 0
    for key, vid in b.list_versions(run.root):
        if key.startswith(run.root) and b.delete_object(key, vid).ok:
            deleted += 1
    locked_keys = {lo["key"] for lo in run.locked}
    left = sorted({k for k, _ in b.list_versions(run.root)})
    return {"deleted_versions": deleted, "aborted_uploads": aborted,
            "remaining_unlocked": [k for k in left if k not in locked_keys],
            "remaining_locked": [k for k in left if k in locked_keys]}


def recommended_path(results: dict) -> str:
    if results.get("presigned_upload_part") == "pass":
        return "presigned_multipart"
    if results.get("single_put_limit") == "pass":
        return "single_presigned_put"
    return "upload_relay"


def would_do(args, params: dict) -> dict:
    root = f"{args.prefix}<run_id>/"
    big, part = params["big_object_bytes"], params["part_bytes"]
    ret = params["lock_retention_seconds"]
    return {
        "presigned_upload_part": [
            f"CreateMultipartUpload {root}presigned-mpu.bin (main key)",
            f"PUT {len(PRESIGNED_PART_SIZES)} parts ({', '.join(map(str, PRESIGNED_PART_SIZES))} "
            "bytes) through presigned UploadPart URLs, no Authorization header",
            "CompleteMultipartUpload, GET and compare sha256"],
        "single_put_limit": [
            f"presigned PUT of {n} bytes to {root}single-put-{n}.bin, HEAD the size "
            "(stop at the first rejection)" for n in params["single_put_bytes"]],
        "object_lock_multipart": [
            "GetObjectLockConfiguration on the bucket",
            f"CreateMultipartUpload {root}lock-mpu.bin with COMPLIANCE lock headers, "
            f"retain {ret} s ({LOCKED_OBJECT_BYTES} bytes, stays locked)",
            "UploadPart, CompleteMultipartUpload, HEAD the lock mode and date"],
        "prefix_bucket_policy": [
            "GetBucketPolicy (an existing policy is never touched)",
            f"PutBucketPolicy: Deny s3:PutObject on {root}policy/denied/*",
            "PUT under denied/ and allowed/ with the sub-token, then DeleteBucketPolicy"],
        "subtoken_scope": [
            f"HeadBucket {args.other_bucket} (main key)",
            f"PUT and GET {root}subtoken/in-bucket.bin with the sub-token",
            f"ListObjectsV2 {args.other_bucket} with the sub-token (want 403)"],
        "upload_part_copy_2_5gb": [
            f"multipart upload {big} bytes to {root}copy-src.bin in {part}-byte parts",
            f"CopyObject to {root}copy-single.bin (single-copy cap, informational)",
            f"UploadPartCopy {math.ceil(big / part)} ranges to {root}copy-dst.bin, "
            "complete, GET and compare sha256"],
        "compliance_retention_unshortenable": [
            f"PUT {root}compliance.bin ({LOCKED_OBJECT_BYTES} bytes) with COMPLIANCE, "
            f"retain {ret} s (stays locked)",
            "PutObjectRetention earlier, with and without BypassGovernanceRetention; "
            "DeleteObject by version; HEAD the retention again"],
    }


# ── CLI ──────────────────────────────────────────────────────────────────────


def _int_list(text: str) -> list[int]:
    try:
        vals = [int(x) for x in text.split(",") if x.strip()]
    except ValueError:
        raise UsageError("--single-put-bytes: want comma-separated ints") from None
    return vals


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog=f"python -P -m {MODULE}", description=__doc__.split("\n")[0])
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--other-bucket", required=True)
    ap.add_argument("--out")
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    ap.add_argument("--region", default=DEFAULT_REGION)
    ap.add_argument("--lock-retention-seconds", type=int, default=MAX_LOCK_S)
    ap.add_argument("--presign-expires-seconds", type=int, default=3600)
    ap.add_argument("--big-object-bytes", type=int)
    ap.add_argument("--part-bytes", type=int)
    ap.add_argument("--single-put-bytes")
    ap.add_argument("--confirm-live", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--fixture")
    return ap.parse_args(argv)


def check_endpoint(endpoint: str) -> tuple[str, str]:
    """(endpoint URL, host); refuses userinfo, path, query and fragment
    without echoing the value (it may carry a credential)."""
    msg = "--endpoint must be http(s)://host[:port] with no userinfo, path, query or fragment"
    try:
        u = urlsplit(endpoint)
        port = u.port
    except ValueError:
        raise UsageError(msg) from None
    if u.scheme not in ("http", "https") or not u.hostname or "@" in u.netloc \
            or u.query or u.fragment or u.path not in ("", "/"):
        raise UsageError(msg)
    host = u.hostname.lower()
    return f"{u.scheme}://{host}" + (f":{port}" if port else ""), host


def build_params(args, fixture: bool) -> dict:
    d = FIXTURE_DEFAULTS if fixture else LIVE_DEFAULTS
    params = {"big_object_bytes": args.big_object_bytes or d["big_object_bytes"],
              "part_bytes": args.part_bytes or d["part_bytes"],
              "single_put_bytes": (_int_list(args.single_put_bytes) if args.single_put_bytes
                                   else list(d["single_put_bytes"])),
              "lock_retention_seconds": args.lock_retention_seconds,
              "presign_expires_seconds": args.presign_expires_seconds}
    lr = params["lock_retention_seconds"]
    if not MIN_LOCK_S <= lr <= MAX_LOCK_S:
        raise UsageError(f"--lock-retention-seconds: want {MIN_LOCK_S}..{MAX_LOCK_S} (COMPLIANCE "
                         "locks can't be removed early; the probe never locks longer than a day)")
    if not 1 <= params["presign_expires_seconds"] <= MAX_PRESIGN_S:
        raise UsageError(f"--presign-expires-seconds: want 1..{MAX_PRESIGN_S}")
    ladder = params["single_put_bytes"]
    if not ladder or any(x <= 0 for x in ladder) or ladder != sorted(set(ladder)):
        raise UsageError("--single-put-bytes: want a strictly ascending list of sizes > 0")
    big, part = params["big_object_bytes"], params["part_bytes"]
    if big <= 0 or part <= 0:
        raise UsageError("--big-object-bytes / --part-bytes must be > 0")
    if fixture:
        if max([big, part, *ladder]) > FIXTURE_MAX_BYTES:
            raise UsageError(f"--fixture runs cap every payload at {FIXTURE_MAX_BYTES} bytes")
    else:
        if not MIN_PART_BYTES <= part <= MAX_PART_BYTES:
            raise UsageError(f"--part-bytes: want {MIN_PART_BYTES}..{MAX_PART_BYTES}")
        if math.ceil(big / part) > MAX_PARTS:
            raise UsageError(f"--big-object-bytes / --part-bytes needs more than {MAX_PARTS} parts")
    return params


def validate(args) -> tuple[str, str, dict]:
    if not PREFIX_RE.fullmatch(args.prefix) or args.prefix.startswith("lane-d/"):
        raise UsageError("--prefix: want a throwaway prefix like 'w3a-probe/' (lowercase "
                         "segments, trailing '/', never under 'lane-d/')")
    if args.bucket in production_buckets():
        raise UsageError(f"--bucket {args.bucket!r} is a chain.toml bucket: use a throwaway, "
                         "Object-Lock-enabled bucket created for this probe")
    if args.other_bucket == args.bucket:
        raise UsageError("--other-bucket must differ from --bucket")
    if args.fixture and not args.dry_run:
        raise UsageError("--fixture needs --dry-run")
    endpoint, host = check_endpoint(args.endpoint)
    params = build_params(args, fixture=bool(args.fixture))
    if not args.dry_run:
        if not args.out:
            raise UsageError("live mode needs --out")
        if not args.confirm_live:
            raise UsageError("live mode needs --confirm-live: it creates real objects, two of "
                             "them COMPLIANCE-locked until their retain-until date")
        missing = [n for n in (MAIN_ACCESS_ENV, MAIN_SECRET_ENV) if not os.environ.get(n)]
        if missing:
            raise UsageError(f"missing credentials: set {' and '.join(missing)} in the "
                             "environment")
    return endpoint, host, params


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    try:
        args = parse_args(argv)
    except SystemExit as e:
        return 2 if e.code else 0
    try:
        endpoint, host, params = validate(args)
        model = load_fixture(Path(args.fixture)) if args.fixture else None
    except UsageError as e:
        print(f"storage_probe: {e}", file=sys.stderr)
        return 2
    summary = {"dry_run": bool(args.dry_run), "fixture": args.fixture, "out": args.out,
               "endpoint_host": model["endpoint_host"] if model else host,
               "bucket": args.bucket, "prefix": args.prefix, "run_id": None,
               "would_do": would_do(args, params), "results": None, "complete": None,
               "recommended_path": None, "locked_objects": None, "cleanup": None,
               "passed": False}
    if args.dry_run and model is None:
        print(json.dumps(summary, indent=2))
        return 0
    try:
        if model is not None:
            backend = FixtureBackend(model, args.bucket)
            region = model["region"]
        else:
            backend = LiveBackend(endpoint, args.region, args.bucket)
            region = args.region
        if backend.count_prefix(args.prefix) != 0:
            raise UsageError(f"--prefix {args.prefix!r} already holds objects: the probe needs "
                             "an empty throwaway prefix")
    except UsageError as e:
        print(f"storage_probe: {e}", file=sys.stderr)
        return 2
    except Exception as e:  # noqa: BLE001 — setup failure is an input error
        print(f"storage_probe: cannot start: {scrub(f'{type(e).__name__}: {e}')}", file=sys.stderr)
        return 2

    run = Run(backend, args.bucket, args.prefix, secrets.token_hex(8), args.other_bucket, params)
    started = _now()
    rows = {}
    for name in RUN_ORDER:
        print(f"storage_probe: running {name}", file=sys.stderr, flush=True)
        rows[name] = run_probe(run, name)
    try:
        clean = cleanup(run)
    except Exception as e:  # noqa: BLE001 — report, never hide, a failed cleanup
        print(f"storage_probe: cleanup failed: {scrub(f'{type(e).__name__}: {e}')}",
              file=sys.stderr)
        clean = {"deleted_versions": 0, "aborted_uploads": 0,
                 "remaining_unlocked": [f"{run.root}(cleanup failed: list the prefix)"],
                 "remaining_locked": []}
    finished = _now()
    probes = [rows[n] for n in PROBES]
    results = {p["name"]: p["result"] for p in probes}
    complete = (all(r in ("pass", "fail") for r in results.values())
                and clean["remaining_unlocked"] == [] and not run.policy_left)
    source = "fixture" if model is not None else "live"
    evidence = {"schema": SCHEMA, "wp": WP, "source": source,
                "endpoint_host": summary["endpoint_host"], "region": region,
                "bucket": args.bucket, "prefix": args.prefix, "run_id": run.run_id,
                "started_at": _iso(started), "finished_at": _iso(finished),
                "tool": tool_info(), "params": params, "probes": probes,
                "locked_objects": run.locked, "cleanup": clean,
                "recommended_path": recommended_path(results),
                "passed": complete and source == "live"}
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    summary.update(run_id=run.run_id, results=results, complete=complete,
                   recommended_path=evidence["recommended_path"],
                   locked_objects=run.locked, cleanup=clean, passed=evidence["passed"])
    print(json.dumps(summary, indent=2))
    return 0 if complete else 1


if __name__ == "__main__":
    sys.exit(main())
