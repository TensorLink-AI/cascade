"""Hippius storage backends — the registry (models) and S3 (logs/manifests).

metronome stores three kinds of artefact on Hippius:

* **Models / checkpoints / generators → the Hippius *registry*** (the
  ``hippius_sdk`` IPFS layer). Content-addressed by **CID**: a directory is
  packed into a deterministic tar, uploaded, and referenced everywhere by the
  returned CID. A CID *is* the content hash, so it doubles as the integrity
  digest — there is no separate ``@sha`` the way a HuggingFace ``repo@revision``
  pointer needed one. Miners commit ``metro-v1:gen:hippius:<cid>``; the trainer
  publishes ``metro-v1:trained:hippius:<cid>`` checkpoints.
* **Training manifests → Hippius S3** (a standard boto3 endpoint). Small JSON
  the validator polls; the trainer writes ``round-<id>.json`` and updates a
  ``latest.json`` pointer.
* **Training logs / metrics → Hippius S3.** Per-round, per-role JSONL emitted by
  the reference trainer (train loss, lr, throughput, eval-on-train metrics) for
  observability.

Both backends are behind **lazy imports** so the core package stays installable
without ``hippius_sdk`` / ``boto3`` (unit tests, the miner's static path). The
``hippius_sdk`` client is async; the sync wrappers here run it on a private event
loop so the rest of metronome stays synchronous.

Credentials are read from the environment, never from ``chain.toml`` (which is a
public, committed file):

* registry  — ``IPFS_NODE_URL`` (or the configured ``ipfs_api_url``),
  ``HIPPIUS_ENCRYPTION_KEY`` (optional; models are stored unencrypted by default
  so validators can read them).
* S3        — ``HIPPIUS_S3_ACCESS_KEY`` / ``HIPPIUS_S3_SECRET_KEY``.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import os

# CID grammar: CIDv0 (base58btc, ``Qm…46 chars``) or CIDv1 (multibase-prefixed,
# lowercase base32 is the common ``bafy…`` form). Kept permissive but bounded so
# a malformed on-chain payload is rejected, not fetched.
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path

CIDV0_RE = re.compile(r"^Qm[1-9A-HJ-NP-Za-km-z]{44}$")
CIDV1_RE = re.compile(r"^b[a-z2-7]{20,}$")


def is_cid(value: str) -> bool:
    """True if ``value`` looks like an IPFS CID (v0 or v1)."""
    v = value.strip()
    return bool(CIDV0_RE.match(v) or CIDV1_RE.match(v))


class StorageError(RuntimeError):
    """Any Hippius registry or S3 operation failed."""


# ───────────────────────────── deterministic tar ────────────────────────────


def pack_dir_to_tar(local_dir: Path | str) -> bytes:
    """Pack a directory into a reproducible (sorted, zeroed-metadata) tar blob.

    Two callers packing the same file tree get byte-identical tar bytes — so the
    registry CID is stable across machines, which is what makes a stored
    checkpoint / generator auditable (re-pack ⇒ same CID).
    """
    d = Path(local_dir)
    if not d.is_dir():
        raise StorageError(f"not_a_directory: {d}")
    files = sorted(p for p in d.rglob("*") if p.is_file())
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for p in files:
            arcname = p.relative_to(d).as_posix()
            info = tarfile.TarInfo(name=arcname)
            data = p.read_bytes()
            info.size = len(data)
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def unpack_tar_to_dir(tar_bytes: bytes, dest_dir: Path | str) -> Path:
    """Inverse of :func:`pack_dir_to_tar`; extracts safely under ``dest_dir``.

    Generator tars are miner-controlled, so every member is vetted: only regular
    files and directories are allowed (no symlinks/hardlinks/devices), and every
    resolved path must stay strictly inside ``dest`` (a plain string-prefix check
    is unsafe — ``/dest`` prefixes the sibling ``/dest-evil``).
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    dest_resolved = dest.resolve()
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:*") as tar:
        for member in tar.getmembers():
            if member.issym() or member.islnk() or member.isdev():
                raise StorageError(f"unsafe_tar_member (link/dev): {member.name}")
            target = (dest / member.name).resolve()
            if target != dest_resolved and dest_resolved not in target.parents:
                raise StorageError(f"unsafe_tar_member (escapes dest): {member.name}")
        tar.extractall(dest)  # noqa: S202 — members vetted above
    return dest


def tar_cid_digest(tar_bytes: bytes) -> str:
    """sha256 of the packed tar — a backend-independent integrity digest that
    travels next to the CID (a sanity check the fetched bytes match)."""
    return hashlib.sha256(tar_bytes).hexdigest()


# ───────────────────────────── registry (IPFS) ──────────────────────────────


@dataclass(frozen=True)
class RegistryConfig:
    """How to reach the Hippius registry (IPFS) backend."""

    ipfs_api_url: str
    ipfs_gateway: str = "https://get.hippius.network"
    encrypt: bool = False  # models/generators are public; keep them readable

    @classmethod
    def from_storage(cls, storage: object) -> RegistryConfig:
        """Build from a :class:`metronome.shared.config.StorageConfig`, letting
        ``IPFS_NODE_URL`` override the committed default."""
        api = os.environ.get("IPFS_NODE_URL") or getattr(storage, "ipfs_api_url", "")
        if not api:
            raise StorageError(
                "no IPFS node configured: set IPFS_NODE_URL or [storage] ipfs_api_url"
            )
        return cls(
            ipfs_api_url=api,
            ipfs_gateway=getattr(storage, "ipfs_gateway", "https://get.hippius.network"),
            encrypt=bool(getattr(storage, "registry_encrypt", False)),
        )


def _run(coro):
    """Run an async coroutine to completion on a fresh event loop (the rest of
    metronome is synchronous)."""
    try:
        return asyncio.run(coro)
    except RuntimeError as e:  # pragma: no cover - only if called inside a loop
        raise StorageError(f"event_loop_error: {e}") from e


def _registry_client(reg: RegistryConfig):
    try:
        from hippius_sdk import HippiusClient  # type: ignore
    except ImportError as e:
        raise StorageError(
            "hippius_sdk not installed; install the [hippius] extra to use the registry"
        ) from e
    key = os.environ.get("HIPPIUS_ENCRYPTION_KEY")
    kwargs: dict = {"ipfs_api_url": reg.ipfs_api_url, "ipfs_gateway": reg.ipfs_gateway}
    if key:
        import base64

        kwargs["encryption_key"] = base64.b64decode(key)
    return HippiusClient(**kwargs)


@dataclass(frozen=True)
class RegistryUpload:
    cid: str
    tar_digest: str
    size_bytes: int


def upload_dir_to_registry(local_dir: Path | str, reg: RegistryConfig) -> RegistryUpload:
    """Pack ``local_dir`` to a deterministic tar and upload it to the registry.

    Returns the IPFS CID (the pointer used everywhere) plus the tar's sha256 and
    size. The CID is content-addressed, so re-uploading identical content yields
    the same CID — the audit hook for re-derived runs.
    """
    tar_bytes = pack_dir_to_tar(local_dir)
    digest = tar_cid_digest(tar_bytes)
    client = _registry_client(reg)

    async def _go() -> str:
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as fh:
            fh.write(tar_bytes)
            tmp = fh.name
        try:
            result = await client.upload_file(tmp, encrypt=reg.encrypt)
        finally:
            os.unlink(tmp)
        cid = result.get("cid") if isinstance(result, dict) else getattr(result, "cid", None)
        if not cid:
            raise StorageError(f"registry upload returned no cid: {result!r}")
        return str(cid)

    cid = _run(_go())
    return RegistryUpload(cid=cid, tar_digest=digest, size_bytes=len(tar_bytes))


def fetch_from_registry(
    cid: str,
    dest_dir: Path | str,
    reg: RegistryConfig,
    *,
    expected_tar_digest: str | None = None,
) -> Path:
    """Download ``cid`` from the registry and extract it into ``dest_dir``.

    If ``expected_tar_digest`` is given, the fetched tar's sha256 must match
    (defence against a backend serving the wrong bytes for a CID).
    """
    if not is_cid(cid):
        raise StorageError(f"not_a_cid: {cid!r}")
    client = _registry_client(reg)

    async def _go() -> bytes:
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as fh:
            tmp = fh.name
        try:
            await client.download_file(cid, tmp, decrypt=reg.encrypt)
            return Path(tmp).read_bytes()
        finally:
            os.unlink(tmp)

    tar_bytes = _run(_go())
    if expected_tar_digest is not None:
        got = tar_cid_digest(tar_bytes)
        if got != expected_tar_digest:
            raise StorageError(f"tar_digest_mismatch: {got} != {expected_tar_digest}")
    return unpack_tar_to_dir(tar_bytes, dest_dir)


# ─────────────────────────────────── S3 ─────────────────────────────────────


@dataclass(frozen=True)
class S3Config:
    """Hippius S3 endpoint + bucket. Credentials come from the environment."""

    endpoint: str
    region: str
    bucket: str

    @classmethod
    def from_storage(cls, storage: object, *, bucket: str) -> S3Config:
        return cls(
            endpoint=getattr(storage, "s3_endpoint", "https://s3.hippius.com"),
            region=getattr(storage, "s3_region", "decentralized"),
            bucket=bucket,
        )


def _s3_client(s3cfg: S3Config):
    try:
        import boto3  # type: ignore
        from botocore.config import Config  # type: ignore
    except ImportError as e:
        raise StorageError(
            "boto3 not installed; install the [hippius] extra to use Hippius S3"
        ) from e
    access = os.environ.get("HIPPIUS_S3_ACCESS_KEY")
    secret = os.environ.get("HIPPIUS_S3_SECRET_KEY")
    if not access or not secret:
        raise StorageError(
            "missing Hippius S3 credentials: set HIPPIUS_S3_ACCESS_KEY / HIPPIUS_S3_SECRET_KEY"
        )
    return boto3.client(
        "s3",
        endpoint_url=s3cfg.endpoint,
        region_name=s3cfg.region,
        aws_access_key_id=access,
        aws_secret_access_key=secret,
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )


@dataclass
class S3Store:
    """Thin boto3 facade over one Hippius S3 bucket (lazy client)."""

    cfg: S3Config
    _client: object = None

    def client(self):
        if self._client is None:
            self._client = _s3_client(self.cfg)
        return self._client

    def _ensure_bucket(self) -> None:
        c = self.client()
        try:
            c.head_bucket(Bucket=self.cfg.bucket)
        except Exception:  # noqa: BLE001 — create if missing/inaccessible
            try:
                c.create_bucket(Bucket=self.cfg.bucket)
            except Exception as e:  # noqa: BLE001
                raise StorageError(f"bucket_unavailable: {self.cfg.bucket}: {e}") from e

    def put_bytes(self, key: str, data: bytes, *, content_type: str = "application/octet-stream") -> None:
        self._ensure_bucket()
        try:
            self.client().put_object(
                Bucket=self.cfg.bucket, Key=key, Body=data, ContentType=content_type
            )
        except Exception as e:  # noqa: BLE001
            raise StorageError(f"s3_put_failed: {key}: {e}") from e

    def put_text(self, key: str, text: str, *, content_type: str = "text/plain") -> None:
        self.put_bytes(key, text.encode("utf-8"), content_type=content_type)

    def get_bytes(self, key: str) -> bytes:
        try:
            resp = self.client().get_object(Bucket=self.cfg.bucket, Key=key)
            return resp["Body"].read()
        except Exception as e:  # noqa: BLE001
            raise StorageError(f"s3_get_failed: {key}: {e}") from e

    def get_text(self, key: str) -> str:
        return self.get_bytes(key).decode("utf-8")


# ───────────────────────── manifests + logs over S3 ─────────────────────────

MANIFEST_LATEST_KEY = "manifests/latest.json"


def manifest_round_key(round_id: str) -> str:
    return f"manifests/round-{round_id}.json"


def publish_manifest(store: S3Store, manifest_text: str, round_id: str) -> str:
    """Write the round manifest and update the ``latest.json`` pointer.

    Returns the per-round key. Validators read :data:`MANIFEST_LATEST_KEY`.
    """
    key = manifest_round_key(round_id)
    store.put_text(key, manifest_text, content_type="application/json")
    store.put_text(MANIFEST_LATEST_KEY, manifest_text, content_type="application/json")
    return key


def read_latest_manifest(store: S3Store) -> str:
    """Read the current manifest JSON from ``latest.json``."""
    return store.get_text(MANIFEST_LATEST_KEY)


def log_key(round_id: str, role: str) -> str:
    return f"logs/round-{round_id}/{role}.jsonl"


@dataclass
class LogSink:
    """Buffer training log records and flush them as one JSONL object to S3.

    S3 has no append, so the reference trainer accumulates per-step records and
    :meth:`flush` writes the whole JSONL blob (idempotent — the latest flush wins
    for a (round, role) key). Use :meth:`emit` per step.
    """

    store: S3Store
    round_id: str
    role: str
    _records: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self._records is None:
            self._records = []

    def emit(self, record: dict) -> None:
        import json

        self._records.append(json.dumps(record, sort_keys=True, separators=(",", ":")))

    def flush(self) -> str | None:
        if not self._records:
            return None
        key = log_key(self.round_id, self.role)
        self.store.put_text(key, "\n".join(self._records) + "\n", content_type="application/x-ndjson")
        return key
