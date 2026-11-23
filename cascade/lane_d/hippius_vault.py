"""The lane D vault on Hippius S3 (LANE_D_PLAN §1.2 "Storage"; W3a decision
``presigned_multipart``).

* :class:`S3VaultStorage` is the :class:`~cascade.lane_d.storage.LaneDStorage`
  over the private ``[lane_d_ops] vault_bucket``: what the intake reads at
  ``ready`` (the object list, sizes and ``manifest.json`` only) and where it
  writes the canonical manifest at ``submit``.
* :class:`HippiusMultipartMinter` is the one production
  :class:`~cascade.lane_d.service.GrantMinter`. At ``submit`` it creates one
  multipart upload per file and mints one short-lived presigned UploadPart URL
  per part (parts of at most :data:`MAX_PART_BYTES`, at most
  :data:`MAX_PARTS` per file); at ``ready`` the intake completes each upload
  itself after checking every part's size against the plan.

Rev6.4: Hippius has no Object Lock, so nothing here asks for one. A pending
entry's integrity rests on content addressing (the manifest digest is in the
object path) and the ingest re-hash.

Credentials are env-only (``HIPPIUS_S3_ACCESS_KEY`` / ``HIPPIUS_S3_SECRET_KEY``)
and never logged; presigned URLs leave this module only inside the grants the
submit answer carries. ``boto3`` is imported lazily.
"""

from __future__ import annotations

import math
import os
import threading
from dataclasses import dataclass

from .intake import IntakeError
from .service import UPLOAD_PATH, GrantMinter
from .storage import LaneDStorage, check_key

MIB = 1 << 20
# Hippius multipart limits (plan §1.2): parts of at most 512 MiB, at most
# 10,000 parts per object.
MAX_PART_BYTES = 512 * MIB
MAX_PARTS = 10_000
# The part size the minter starts from (a 2.5 GB BF16 checkpoint is 38 parts).
DEFAULT_PART_BYTES = 64 * MIB
# Presigned UploadPart URLs expire after this many seconds (Hippius allows up
# to 7 days; a slow 2.5 GB upload still fits comfortably).
GRANT_EXPIRES_SECONDS = 6 * 3600
# Per-socket read inactivity: CompleteMultipartUpload of a 2.5 GB object can
# take the gateway a while to answer.
S3_READ_TIMEOUT_S = 120
S3_CONNECT_TIMEOUT_S = 10
ACCESS_KEY_ENV = "HIPPIUS_S3_ACCESS_KEY"
SECRET_KEY_ENV = "HIPPIUS_S3_SECRET_KEY"
_MISSING_CODES = frozenset(("NoSuchKey", "NotFound", "404", "NoSuchUpload"))


@dataclass(frozen=True)
class VaultEndpoint:
    """The vault bucket and its S3 endpoint (credentials stay in the env)."""

    endpoint: str
    region: str
    bucket: str

    @classmethod
    def from_storage(cls, storage: object, *, bucket: str) -> VaultEndpoint:
        """From a loaded ``[storage]`` table and ``[lane_d_ops] vault_bucket``."""
        return cls(endpoint=str(getattr(storage, "s3_endpoint", "") or "https://s3.hippius.com"),
                   region=str(getattr(storage, "s3_region", "") or "decentralized"),
                   bucket=str(bucket))


class VaultClient:
    """One lazily created boto3 S3 client for the vault, shared by the
    storage adapter and the minter."""

    def __init__(self, endpoint: VaultEndpoint, *, environ=None):
        self.endpoint = endpoint
        self._environ = os.environ if environ is None else environ
        self._client = None
        self._lock = threading.Lock()

    @property
    def bucket(self) -> str:
        return self.endpoint.bucket

    def __call__(self):
        with self._lock:
            if self._client is None:
                self._client = self._build()
            return self._client

    def _build(self):
        import boto3  # noqa: PLC0415 — the [hippius] extra, loaded on first use
        from botocore.config import Config  # noqa: PLC0415

        access = self._environ.get(ACCESS_KEY_ENV)
        secret = self._environ.get(SECRET_KEY_ENV)
        if not access or not secret:
            raise RuntimeError(f"missing vault credentials: set {ACCESS_KEY_ENV} / "
                               f"{SECRET_KEY_ENV} in the service environment")
        return boto3.client(
            "s3", endpoint_url=self.endpoint.endpoint, region_name=self.endpoint.region,
            aws_access_key_id=access, aws_secret_access_key=secret,
            config=Config(signature_version="s3v4", s3={"addressing_style": "path"},
                          connect_timeout=S3_CONNECT_TIMEOUT_S, read_timeout=S3_READ_TIMEOUT_S,
                          retries={"max_attempts": 2, "mode": "standard"}))


def _is_missing(exc: BaseException) -> bool:
    resp = getattr(exc, "response", None)
    if not isinstance(resp, dict):
        return False
    code = str((resp.get("Error") or {}).get("Code", ""))
    status = (resp.get("ResponseMetadata") or {}).get("HTTPStatusCode")
    return code in _MISSING_CODES or status == 404


class S3VaultStorage(LaneDStorage):
    """The vault bucket behind :class:`LaneDStorage`: a missing object is
    ``KeyError``; every other failure propagates (the intake answers 503)."""

    def __init__(self, client: VaultClient):
        self._c = client

    def put(self, key: str, data: bytes) -> None:
        self._c().put_object(Bucket=self._c.bucket, Key=check_key(key), Body=bytes(data),
                             ContentType="application/octet-stream")

    def get(self, key: str) -> bytes:
        try:
            resp = self._c().get_object(Bucket=self._c.bucket, Key=check_key(key))
        except Exception as e:
            if _is_missing(e):
                raise KeyError(key) from None
            raise
        body = resp["Body"]
        try:
            return body.read()
        finally:
            body.close()

    def list(self, prefix: str) -> list[str]:
        keys: list[str] = []
        kwargs = {"Bucket": self._c.bucket, "Prefix": prefix}
        while True:
            resp = self._c().list_objects_v2(**kwargs)
            keys.extend(str(o["Key"]) for o in resp.get("Contents", ()))
            if not resp.get("IsTruncated"):
                break
            kwargs["ContinuationToken"] = resp["NextContinuationToken"]
        return sorted(keys)

    def size(self, key: str) -> int:
        try:
            resp = self._c().head_object(Bucket=self._c.bucket, Key=check_key(key))
        except Exception as e:
            if _is_missing(e):
                raise KeyError(key) from None
            raise
        return int(resp["ContentLength"])


def part_plan(size: int) -> list[tuple[int, int]]:
    """``[(offset, length)]`` covering ``[0, size)`` in order: parts of
    :data:`DEFAULT_PART_BYTES`, grown (in whole MiB) so a file never needs
    more than :data:`MAX_PARTS`; ``IntakeError("file_too_large")`` past
    ``MAX_PARTS * MAX_PART_BYTES``."""
    size = int(size)
    if size <= 0:
        raise IntakeError("bad_manifest", f"size {size} is not positive")
    part = max(DEFAULT_PART_BYTES, math.ceil(math.ceil(size / MAX_PARTS) / MIB) * MIB)
    if part > MAX_PART_BYTES:
        raise IntakeError("file_too_large",
                          f"{size} bytes exceeds {MAX_PARTS} parts of {MAX_PART_BYTES}")
    return [(off, min(part, size - off)) for off in range(0, size, part)]


class HippiusMultipartMinter(GrantMinter):
    """Presigned multipart grants on the vault bucket; the intake completes.

    The grants are W10's format: ``{"files": {name: {"requests": [{"method":
    "PUT", "url", "offset", "length", "headers"}]}}}``, one request per part in
    order, each covering ``file[offset:offset+length]``."""

    upload_path = UPLOAD_PATH

    def __init__(self, client: VaultClient, *, expires_seconds: int = GRANT_EXPIRES_SECONDS):
        if not 60 <= int(expires_seconds) <= 7 * 86400:
            raise ValueError(f"expires_seconds must be in [60, 604800]; got {expires_seconds}")
        self._c = client
        self.expires_seconds = int(expires_seconds)

    def _pending(self, key: str) -> list[dict]:
        """In-progress multipart uploads of exactly ``key``, oldest first."""
        out: list[dict] = []
        kwargs = {"Bucket": self._c.bucket, "Prefix": key}
        while True:
            resp = self._c().list_multipart_uploads(**kwargs)
            out.extend(u for u in resp.get("Uploads", ()) if u.get("Key") == key)
            if not resp.get("IsTruncated"):
                break
            kwargs["KeyMarker"] = resp.get("NextKeyMarker", "")
            kwargs["UploadIdMarker"] = resp.get("NextUploadIdMarker", "")
        return sorted(out, key=lambda u: str(u.get("Initiated", "")))

    def _abort(self, key: str, upload_id: str) -> None:
        try:
            self._c().abort_multipart_upload(Bucket=self._c.bucket, Key=key, UploadId=upload_id)
        except Exception as e:
            if not _is_missing(e):
                raise

    def mint(self, prefix: str, files: dict[str, int]) -> dict:
        plans = {name: part_plan(size) for name, size in sorted(files.items())}
        out: dict[str, dict] = {}
        for name, plan in plans.items():
            key = check_key(prefix + name)
            # A resubmit (retry) starts over: at most one upload per object.
            for u in self._pending(key):
                self._abort(key, str(u["UploadId"]))
            upload_id = str(self._c().create_multipart_upload(
                Bucket=self._c.bucket, Key=key,
                ContentType="application/octet-stream")["UploadId"])
            requests = []
            for i, (offset, length) in enumerate(plan, start=1):
                url = self._c().generate_presigned_url(
                    "upload_part",
                    Params={"Bucket": self._c.bucket, "Key": key, "UploadId": upload_id,
                            "PartNumber": i},
                    ExpiresIn=self.expires_seconds, HttpMethod="PUT")
                requests.append({"method": "PUT", "url": str(url), "offset": offset,
                                 "length": length, "headers": {}})
            out[name] = {"requests": requests}
        return {"files": out}

    def _parts(self, key: str, upload_id: str) -> list[dict]:
        parts: list[dict] = []
        kwargs = {"Bucket": self._c.bucket, "Key": key, "UploadId": upload_id, "MaxParts": 1000}
        while True:
            resp = self._c().list_parts(**kwargs)
            parts.extend(resp.get("Parts", ()))
            if not resp.get("IsTruncated"):
                break
            kwargs["PartNumberMarker"] = resp["NextPartNumberMarker"]
        return sorted(parts, key=lambda p: int(p["PartNumber"]))

    def finalize(self, prefix: str, files: dict[str, int]) -> None:
        for name, size in sorted(files.items()):
            key = check_key(prefix + name)
            plan = part_plan(size)
            pending = self._pending(key)
            if not pending:
                # Completed by an earlier ready that was refused later (a cap):
                # the object must already be there at its size.
                try:
                    head = self._c().head_object(Bucket=self._c.bucket, Key=key)
                except Exception as e:
                    if _is_missing(e):
                        raise IntakeError("missing_object", name) from None
                    raise
                if int(head["ContentLength"]) != int(size):
                    raise IntakeError("size_mismatch", name)
                continue
            upload = pending[-1]
            for older in pending[:-1]:
                self._abort(key, str(older["UploadId"]))
            upload_id = str(upload["UploadId"])
            parts = self._parts(key, upload_id)
            got = [(int(p["PartNumber"]), int(p["Size"])) for p in parts]
            want = [(i, length) for i, (_, length) in enumerate(plan, start=1)]
            if got != want:
                raise IntakeError("upload_incomplete",
                                  f"{name}: {len(got)} of {len(want)} part(s) as planned")
            self._c().complete_multipart_upload(
                Bucket=self._c.bucket, Key=key, UploadId=upload_id,
                MultipartUpload={"Parts": [{"PartNumber": int(p["PartNumber"]),
                                            "ETag": str(p["ETag"])} for p in parts]})


__all__ = [
    "DEFAULT_PART_BYTES", "GRANT_EXPIRES_SECONDS", "HippiusMultipartMinter", "MAX_PARTS",
    "MAX_PART_BYTES", "S3VaultStorage", "VaultClient", "VaultEndpoint", "part_plan",
]
