"""Hippius storage backends — the registry (models) and S3 (logs/manifests).

cascade stores three kinds of artefact on Hippius:

* **Models / checkpoints / generators → the Hippius *registry* (Hippius Hub)** —
  the OCI model registry documented at https://docs.hippius.com/registry (the
  same backend teutonic uses, via ``hippius_hub``). An artefact lives in a Hub
  **repo** and is pinned by an immutable OCI manifest **digest** (``sha256:…``).
  It is referenced everywhere by ``repo@digest``: the digest *is* the content
  hash, so it both locates the artefact and doubles as the integrity digest —
  ``snapshot_download`` verifies the layer blobs against it on fetch. Miners
  commit ``metro-v1:gen:hippius:<repo>@<digest>``; the trainer publishes
  ``metro-v1:trained:hippius:<repo>@<digest>`` checkpoints.
* **Training manifests → Hippius S3** (a standard boto3 endpoint). Small JSON
  the validator polls; the trainer writes ``round-<id>.json`` and updates a
  ``latest.json`` pointer.
* **Training logs / metrics → Hippius S3.** Per-round, per-role JSONL emitted by
  the reference trainer (train loss, lr, throughput, eval-on-train metrics) for
  observability.

Both backends are behind **lazy imports** so the core package stays installable
without ``hippius-hub`` / ``boto3`` (unit tests, the miner's static path). The
``hippius_hub`` push/pull helpers are synchronous, so no event loop is needed.

Credentials are read from the environment, never from ``chain.toml`` (which is a
public, committed file):

* registry  — a Hub token (``HIPPIUS_HUB_TOKEN`` / ``HIPPIUS_TOKEN``) or a
  username/password pair (``HIPPIUS_HUB_USERNAME`` + ``HIPPIUS_HUB_PASSWORD``);
  ``HF_TOKEN`` for any ``hf:``-pinned artefact.
* S3        — ``HIPPIUS_S3_ACCESS_KEY`` / ``HIPPIUS_S3_SECRET_KEY``.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import tarfile
from dataclasses import asdict, dataclass
from pathlib import Path


class StorageError(RuntimeError):
    """Any Hippius registry or S3 operation failed."""


# A Hippius Hub model reference is ``<repo>@<digest>``: a repo id plus an
# immutable OCI manifest digest. Two digest shapes are accepted — ``sha256:``
# (the canonical Hub OCI digest a push returns) and ``hf:`` (a vanilla
# HuggingFace commit SHA, for a genesis/eval artefact mirrored on HF without a
# Hub copy), mirroring teutonic's ``ModelRef``.
REPO_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*/[a-zA-Z0-9][a-zA-Z0-9._/-]*$")
DIGEST_RE = re.compile(r"^(sha256:[0-9a-f]{64}|hf:[0-9a-f]{40})$")


@dataclass(frozen=True)
class HubRef:
    """An immutable Hippius Hub reference: ``repo@digest``."""

    repo: str
    digest: str

    def __post_init__(self) -> None:
        repo = (self.repo or "").strip()
        digest = (self.digest or "").strip()
        if not REPO_RE.match(repo):
            raise StorageError(f"invalid Hippius Hub repo id: {self.repo!r}")
        if not DIGEST_RE.match(digest):
            raise StorageError(f"invalid Hippius Hub OCI digest: {self.digest!r}")
        object.__setattr__(self, "repo", repo)
        object.__setattr__(self, "digest", digest)

    @property
    def immutable_ref(self) -> str:
        return f"{self.repo}@{self.digest}"

    @classmethod
    def parse(cls, ref: str) -> HubRef:
        """Parse a ``repo@digest`` string; raise StorageError if malformed."""
        repo, sep, digest = (ref or "").strip().partition("@")
        if not sep:
            raise StorageError(f"not a Hippius Hub ref (expected repo@digest): {ref!r}")
        return cls(repo, digest)


def is_hub_ref(value: str) -> bool:
    """True if ``value`` parses as a ``repo@digest`` Hippius Hub reference."""
    try:
        HubRef.parse(value)
        return True
    except StorageError:
        return False


# ──────────────── deterministic tar (S3 eval-pool snapshots) ─────────────────
#
# Models/checkpoints/generators go to the Hub registry (above); these helpers
# pack the daily eval-pool snapshot into a reproducible tar stored on S3
# (``publish_pool_snapshot`` / ``fetch_pool_snapshot``), where the sha256 of the
# tar is the integrity check (S3 has no content-addressing of its own).


def pack_dir_to_tar(local_dir: Path | str) -> bytes:
    """Pack a directory into a reproducible (sorted, zeroed-metadata) tar blob.

    Two callers packing the same file tree get byte-identical tar bytes — so the
    snapshot's sha256 is stable across machines, which is what lets every
    validator verify it fetched the same eval-pool bytes (re-pack ⇒ same sha256).
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

    Every member is vetted: only regular files and directories are allowed (no
    symlinks/hardlinks/devices), and every resolved path must stay strictly
    inside ``dest`` (a plain string-prefix check is unsafe — ``/dest`` prefixes
    the sibling ``/dest-evil``).
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
    """sha256 of the packed tar — the integrity digest stored in the eval-pool
    snapshot index so a validator can verify the bytes it fetched from S3."""
    return hashlib.sha256(tar_bytes).hexdigest()


# ───────────────────────────── registry (Hippius Hub) ───────────────────────
#
# Models, checkpoints, and generators live on the Hippius Hub OCI registry
# (https://docs.hippius.com/registry — the same backend teutonic uses). A push
# returns an immutable ``sha256:`` manifest digest; ``repo@digest`` both locates
# and pins the artefact, so the digest doubles as the integrity hash (the fetch
# verifies layer blobs against it). Auth is read from the environment.

HUB_TOKEN_ENV_NAMES = ("HIPPIUS_HUB_TOKEN", "HIPPIUS_TOKEN", "CASCADE_HIPPIUS_TOKEN")
HUB_USERNAME_ENV_NAMES = ("HIPPIUS_HUB_USERNAME", "HIPPIUS_REGISTRY_USERNAME")
HUB_PASSWORD_ENV_NAMES = ("HIPPIUS_HUB_PASSWORD", "HIPPIUS_REGISTRY_PASSWORD")
HUB_TOKEN_PATH = Path("~/.cache/hippius/hub/token").expanduser()

# Files materialised from a fetched snapshot. Generators ship code + config +
# optional safetensors weights; checkpoints ship safetensors + config; eval
# pools ship .npy/.npz + metadata.json. Pickle weights are rejected later by
# cascade.interface.validation (loading them runs arbitrary code).
ALLOW_PATTERNS = [
    "*.py", "*.json", "*.txt", "*.md",
    "*.safetensors", "*.model", "tokenizer*", "special_tokens*",
    "*.npy", "*.npz",
]


class HubAuthError(StorageError):
    """Hippius Hub auth is unavailable or clearly misconfigured."""


def _get_first_env(names: tuple[str, ...]) -> str | None:
    for name in names:
        value = (os.environ.get(name) or "").strip()
        if value:
            return value
    return None


def _read_cached_hub_token() -> str | None:
    if HUB_TOKEN_PATH.exists():
        cached = HUB_TOKEN_PATH.read_text().strip()
        if cached:
            return cached
    return None


def _resolve_hub_token(action: str) -> str:
    """A Hub bearer token from an env token, the cached token, or a login.

    Raises :class:`HubAuthError` (a StorageError) if no usable credential is set,
    so a missing credential surfaces as a clear, non-retryable error.
    """
    token = _get_first_env(HUB_TOKEN_ENV_NAMES) or _read_cached_hub_token()
    if token:
        return token
    username = _get_first_env(HUB_USERNAME_ENV_NAMES)
    password = _get_first_env(HUB_PASSWORD_ENV_NAMES)
    if username and password:
        from hippius_hub import login as hub_login

        hub_login(username=username, password=password)
        cached = _read_cached_hub_token()
        if cached:
            return cached
    raise HubAuthError(
        f"{action} requires Hippius Hub auth: set a token "
        f"({', '.join(HUB_TOKEN_ENV_NAMES)}) or username+password "
        f"({HUB_USERNAME_ENV_NAMES[0]} + {HUB_PASSWORD_ENV_NAMES[0]})."
    )


@dataclass(frozen=True)
class HubConfig:
    """How to reach the Hippius Hub OCI registry (credentials come from env).

    ``registry_url`` is informational — ``hippius_hub`` targets its own default
    endpoint and is not redirected by this value. ``namespace`` *is* used: it is
    the repo prefix the trainer/owner push under.
    """

    registry_url: str = "https://registry.hippius.com"
    namespace: str = "cascade"

    @classmethod
    def from_storage(cls, storage: object) -> HubConfig:
        """Build from a :class:`cascade.shared.config.StorageConfig`."""
        return cls(
            registry_url=getattr(storage, "hub_registry_url", "https://registry.hippius.com"),
            namespace=getattr(storage, "hub_namespace", "cascade"),
        )


@dataclass(frozen=True)
class HubUpload:
    ref: HubRef
    size_bytes: int


def _dir_size_bytes(local_dir: Path | str) -> int:
    return sum(p.stat().st_size for p in Path(local_dir).rglob("*") if p.is_file())


def upload_dir_to_hub(local_dir: Path | str, repo: str, hub: HubConfig | None = None) -> HubUpload:
    """Upload a folder to a Hippius Hub ``repo`` and return its immutable ref.

    The push returns an OCI ``sha256:`` manifest digest; re-uploading identical
    content to the same repo yields the same digest — the audit hook for
    re-derived runs. ``hub`` is accepted for symmetry but the Hub endpoint is a
    package/server default; auth comes from the environment.
    """
    d = Path(local_dir)
    if not d.is_dir():
        raise StorageError(f"not_a_directory: {d}")
    try:
        from hippius_hub import upload_folder
    except ImportError as e:
        raise StorageError(
            "hippius-hub not installed; install the [hippius] extra to use the registry"
        ) from e
    token = _resolve_hub_token(f"Uploading {d} to {repo}")
    result = upload_folder(
        repo_id=str(repo), folder_path=str(d), allow_patterns=ALLOW_PATTERNS, token=token,
    )
    digest = str(getattr(result, "oid", "") or "")
    if not DIGEST_RE.match(digest):
        raise StorageError(f"hub upload returned no usable sha256 digest: {result!r}")
    return HubUpload(ref=HubRef(str(repo), digest), size_bytes=_dir_size_bytes(d))


def fetch_from_hub(ref: HubRef | str, dest_dir: Path | str, hub: HubConfig | None = None) -> Path:
    """Download an immutable Hub (or ``hf:``) snapshot into ``dest_dir``.

    ``ref`` may be a :class:`HubRef` or a ``repo@digest`` string. The OCI digest
    pins the content, so no separate integrity check is needed on fetch — a Hub
    that served the wrong bytes for a digest would fail the layer verification.
    """
    ref = ref if isinstance(ref, HubRef) else HubRef.parse(ref)
    dest = Path(dest_dir)
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if ref.digest.startswith("hf:"):
        from huggingface_hub import snapshot_download as hf_snapshot_download

        path = hf_snapshot_download(
            repo_id=ref.repo, revision=ref.digest[3:], local_dir=str(dest),
            allow_patterns=ALLOW_PATTERNS,
            token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY"),
        )
    else:
        try:
            from hippius_hub import snapshot_download
        except ImportError as e:
            raise StorageError(
                "hippius-hub not installed; install the [hippius] extra to use the registry"
            ) from e
        path = snapshot_download(
            repo_id=ref.repo, revision=ref.digest, local_dir=str(dest),
            allow_patterns=ALLOW_PATTERNS,
            token=_resolve_hub_token(f"Downloading {ref.immutable_ref}"),
        )
    return Path(path)


# ─────────────────────────────────── S3 ─────────────────────────────────────


@dataclass(frozen=True)
class S3Config:
    """An S3-compatible endpoint + bucket. Credentials come from the environment
    (named by ``access_key_env`` / ``secret_key_env``, never stored here).

    Defaults target Hippius S3; the env-name fields let a second store (e.g. the
    eval-pool bucket on Cloudflare R2) read different credentials without
    touching the manifest store."""

    endpoint: str
    region: str
    bucket: str
    access_key_env: str = "HIPPIUS_S3_ACCESS_KEY"
    secret_key_env: str = "HIPPIUS_S3_SECRET_KEY"

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
    access = os.environ.get(s3cfg.access_key_env)
    secret = os.environ.get(s3cfg.secret_key_env)
    if not access or not secret:
        raise StorageError(
            f"missing S3 credentials: set {s3cfg.access_key_env} / {s3cfg.secret_key_env}"
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

    def put_bytes(
        self, key: str, data: bytes, *,
        content_type: str = "application/octet-stream", acl: str | None = None,
    ) -> None:
        """Write one object. ``acl`` sets a canned object ACL (e.g.
        ``"public-read"`` for the audit-facing receipts); None keeps the
        bucket default (private)."""
        self._ensure_bucket()
        kwargs = {"Bucket": self.cfg.bucket, "Key": key, "Body": data,
                  "ContentType": content_type}
        if acl:
            kwargs["ACL"] = acl
        try:
            self.client().put_object(**kwargs)
        except Exception as e:  # noqa: BLE001
            raise StorageError(f"s3_put_failed: {key}: {e}") from e

    def put_text(
        self, key: str, text: str, *,
        content_type: str = "text/plain", acl: str | None = None,
    ) -> None:
        self.put_bytes(key, text.encode("utf-8"), content_type=content_type, acl=acl)

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


RECEIPT_LATEST_KEY = "receipts/latest.json"


def receipt_round_key(round_id: str) -> str:
    return f"receipts/round-{round_id}.json"


def publish_receipt(store: S3Store, receipt_text: str, round_id: str) -> str:
    """Write the round receipt and update the ``receipts/latest.json`` pointer.

    Mirrors :func:`publish_manifest` — the validator publishes its signed
    :class:`cascade.shared.receipt.RoundReceipt` here after weights are set, and
    auditors read :data:`RECEIPT_LATEST_KEY` (or a specific round's key).
    Returns the per-round key.

    Receipts are the audit-facing artefact, so each object is written with a
    ``public-read`` ACL: third parties can then GET it (and run
    ``cascade-audit``) with zero credentials while the bucket — manifests,
    logs — stays private. On a backend that rejects object ACLs the write is
    retried private (the audit's anonymous fetch then falls back to
    credentials, as documented in docs/AUDIT.md).
    """
    key = receipt_round_key(round_id)
    try:
        store.put_text(key, receipt_text, content_type="application/json", acl="public-read")
        store.put_text(RECEIPT_LATEST_KEY, receipt_text, content_type="application/json",
                       acl="public-read")
    except StorageError:
        # ACL unsupported on this backend: publish private rather than not at all.
        store.put_text(key, receipt_text, content_type="application/json")
        store.put_text(RECEIPT_LATEST_KEY, receipt_text, content_type="application/json")
    return key


def read_receipt(store: S3Store, round_id: str) -> str:
    """Read one round's receipt JSON by round id."""
    return store.get_text(receipt_round_key(round_id))


def read_latest_receipt(store: S3Store) -> str:
    """Read the current receipt JSON from ``receipts/latest.json``."""
    return store.get_text(RECEIPT_LATEST_KEY)


# ── receipts index (dashboard-facing) ───────────────────────────────────────
#
# The per-round receipts are the audit source of truth, but a static dashboard
# can't *list* a bucket to discover them. So the validator also maintains one
# small public-read ``receipts/index.json`` — a rolling window of compact
# per-round summaries (see :func:`cascade.shared.receipt.summarize_receipt`) with
# a ``receipt_key`` pointer back to each signed receipt. Presentational only:
# nothing here is signed or part of the audit contract, and a stale/absent index
# never affects weights (the update is best-effort, like receipt publication).

RECEIPT_INDEX_KEY = "receipts/index.json"
RECEIPT_INDEX_SCHEMA = 1
RECEIPT_INDEX_MAX_KEEP = 400


def read_receipt_index(store: S3Store) -> dict:
    """Read ``receipts/index.json``; return an empty index if absent/malformed."""
    empty = {"schema": RECEIPT_INDEX_SCHEMA, "rounds": []}
    try:
        text = store.get_text(RECEIPT_INDEX_KEY)
    except StorageError:
        return empty
    try:
        doc = json.loads(text)
    except (ValueError, TypeError):
        return empty
    if not isinstance(doc, dict) or not isinstance(doc.get("rounds"), list):
        return empty
    return doc


def update_receipt_index(
    store: S3Store,
    summary: dict,
    *,
    updated_at: str = "",
    subnet: dict | None = None,
    max_keep: int = RECEIPT_INDEX_MAX_KEEP,
) -> dict:
    """Append/replace one round in ``receipts/index.json`` and write it public-read.

    Idempotent per ``round_id`` (a re-published round replaces its entry), sorted
    by ``epoch_start_block`` then ``round_id`` (chronological — round ids are
    block-hash seeds, not monotonic), and capped at ``max_keep`` most-recent
    rounds. ``updated_at`` (an ISO stamp) and ``subnet`` (``{"netuid", "name"}``)
    are optional header fields the dashboard shows. Returns the stored entry.
    """
    entry = dict(summary)
    entry["receipt_key"] = receipt_round_key(str(entry.get("round_id", "")))
    if updated_at:
        entry["published_at"] = updated_at

    doc = read_receipt_index(store)
    rid = str(entry.get("round_id"))
    rounds = [r for r in doc.get("rounds", []) if str(r.get("round_id")) != rid]
    rounds.append(entry)
    rounds.sort(key=lambda r: (int(r.get("epoch_start_block", 0)), str(r.get("round_id", ""))))
    rounds = rounds[-max_keep:]

    out: dict = {"schema": RECEIPT_INDEX_SCHEMA, "rounds": rounds}
    if updated_at:
        out["updated_at"] = updated_at
    if subnet:
        out["subnet"] = subnet

    text = json.dumps(out, indent=2, sort_keys=True)
    try:
        store.put_text(RECEIPT_INDEX_KEY, text, content_type="application/json", acl="public-read")
    except StorageError:
        # ACL unsupported on this backend: publish private rather than not at all.
        store.put_text(RECEIPT_INDEX_KEY, text, content_type="application/json")
    return entry


# ── static dashboard site ────────────────────────────────────────────────────
#
# The "notebook" — a single self-contained ``index.html`` — is served straight
# from the manifest bucket (public-read), reading the public receipts + index
# above. Mirrors teutonic, whose validator re-uploads its dashboard on restart.

WEBSITE_INDEX_KEY = "index.html"


def publish_website(store: S3Store, html: str, *, key: str = WEBSITE_INDEX_KEY) -> str:
    """Upload the dashboard HTML to the bucket root, public-read. Returns the key."""
    try:
        store.put_text(key, html, content_type="text/html; charset=utf-8", acl="public-read")
    except StorageError:
        store.put_text(key, html, content_type="text/html; charset=utf-8")
    return key


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


# ───────────────────────── eval-pool snapshots over S3 ──────────────────────
#
# The eval pool refreshes daily without a chain.toml edit: the owner orchestrator
# publishes a new snapshot (a deterministic tar of the pool directory) to the
# pool bucket and appends it to ``pool/index.json``. Every validator reads the
# same owner-controlled index and selects, for a round, the snapshot whose
# ``effective_round`` is the greatest ``<= round_id`` — so all validators score
# the identical pool for a given round REGARDLESS of when they polled (no
# latest-wins divergence at the daily rollover). Integrity is the tar sha256.
#
# Invariant the publisher MUST hold: a new snapshot's ``effective_round`` is in
# the FUTURE (greater than the current round). Never publish a snapshot that
# becomes active for an already-processed round, or validators that already
# scored it would disagree with those that re-select it.

POOL_INDEX_KEY = "pool/index.json"
POOL_INDEX_SCHEMA = 1


def pool_snapshot_key(effective_round: int) -> str:
    return f"pool/snapshots/{int(effective_round)}.tar"


@dataclass(frozen=True)
class PoolSnapshotMeta:
    """One published eval-pool snapshot, listed in ``pool/index.json``."""

    effective_round: int
    key: str
    sha256: str
    size_bytes: int
    as_of: str
    n_series: int
    context_length: int
    horizon: int

    @classmethod
    def from_dict(cls, d: dict) -> PoolSnapshotMeta:
        return cls(
            effective_round=int(d["effective_round"]),
            key=str(d["key"]),
            sha256=str(d["sha256"]),
            size_bytes=int(d.get("size_bytes", 0)),
            as_of=str(d.get("as_of", "")),
            n_series=int(d.get("n_series", 0)),
            context_length=int(d.get("context_length", 0)),
            horizon=int(d.get("horizon", 0)),
        )


def read_pool_index(store: S3Store) -> list[PoolSnapshotMeta]:
    """Read the snapshot index, sorted by ``effective_round``. Empty if absent."""
    try:
        text = store.get_text(POOL_INDEX_KEY)
    except StorageError:
        return []
    doc = json.loads(text)
    snaps = [PoolSnapshotMeta.from_dict(s) for s in doc.get("snapshots", [])]
    return sorted(snaps, key=lambda s: s.effective_round)


def select_snapshot(index: list[PoolSnapshotMeta], round_id: int) -> PoolSnapshotMeta | None:
    """The snapshot active for ``round_id``: greatest ``effective_round <= round_id``.

    Falls back to the earliest snapshot when ``round_id`` precedes them all (so a
    validator always has a pool); returns ``None`` only for an empty index. The
    rule is deterministic, so every validator selects the same snapshot.
    """
    if not index:
        return None
    eligible = [s for s in index if s.effective_round <= round_id]
    if eligible:
        return max(eligible, key=lambda s: s.effective_round)
    return min(index, key=lambda s: s.effective_round)


def publish_pool_snapshot(
    store: S3Store,
    tar_bytes: bytes,
    *,
    effective_round: int,
    as_of: str,
    n_series: int,
    context_length: int,
    horizon: int,
    max_keep: int = 14,
) -> PoolSnapshotMeta:
    """Upload a pool snapshot tar and register it in ``pool/index.json``.

    Idempotent per ``effective_round`` (re-publishing replaces that entry). Keeps
    the most recent ``max_keep`` entries in the index (old tars are left in the
    bucket for any validator still resolving an older round; prune out-of-band).
    """
    sha = tar_cid_digest(tar_bytes)
    key = pool_snapshot_key(effective_round)
    store.put_bytes(key, tar_bytes, content_type="application/x-tar")

    meta = PoolSnapshotMeta(
        effective_round=int(effective_round),
        key=key,
        sha256=sha,
        size_bytes=len(tar_bytes),
        as_of=as_of,
        n_series=n_series,
        context_length=context_length,
        horizon=horizon,
    )
    index = [s for s in read_pool_index(store) if s.effective_round != meta.effective_round]
    index.append(meta)
    index.sort(key=lambda s: s.effective_round)
    index = index[-max_keep:]
    doc = {"schema": POOL_INDEX_SCHEMA, "snapshots": [asdict(s) for s in index]}
    store.put_text(POOL_INDEX_KEY, json.dumps(doc, indent=2, sort_keys=True),
                   content_type="application/json")
    return meta


def fetch_pool_snapshot(store: S3Store, meta: PoolSnapshotMeta, dest_dir: Path | str) -> Path:
    """Download a snapshot tar, verify its sha256 against the index, and unpack."""
    data = store.get_bytes(meta.key)
    got = tar_cid_digest(data)
    if got != meta.sha256:
        raise StorageError(f"pool_snapshot_digest_mismatch: {got} != {meta.sha256}")
    return unpack_tar_to_dir(data, dest_dir)


def pool_s3_store(storage: object, *, bucket: str | None = None) -> S3Store:
    """Build an :class:`S3Store` for the eval-pool bucket.

    Backend-agnostic: defaults to the Hippius S3 endpoint/credentials, but a
    ``[storage] pool_s3_endpoint`` / ``pool_s3_region`` (e.g. Cloudflare R2) and
    ``POOL_S3_ACCESS_KEY`` / ``POOL_S3_SECRET_KEY`` env override it. When the
    POOL_* env is unset it falls back to the HIPPIUS_S3_* credentials, so a
    Hippius-only operator needs no extra config.
    """
    bkt = bucket or getattr(storage, "pool_bucket", "") or "cascade-eval-pool"
    use_pool_env = bool(os.environ.get("POOL_S3_ACCESS_KEY"))
    endpoint = getattr(storage, "pool_s3_endpoint", "") or getattr(
        storage, "s3_endpoint", "https://s3.hippius.com"
    )
    region = getattr(storage, "pool_s3_region", "") or getattr(storage, "s3_region", "decentralized")
    cfg = S3Config(
        endpoint=endpoint,
        region=region,
        bucket=bkt,
        access_key_env="POOL_S3_ACCESS_KEY" if use_pool_env else "HIPPIUS_S3_ACCESS_KEY",
        secret_key_env="POOL_S3_SECRET_KEY" if use_pool_env else "HIPPIUS_S3_SECRET_KEY",
    )
    return S3Store(cfg)
