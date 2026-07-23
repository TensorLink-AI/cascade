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

import contextlib
import hashlib
import io
import json
import os
import re
import shutil
import tarfile
import time
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


def _resolve_hub_token_for_pull(action: str) -> str | bool:
    """Like :func:`_resolve_hub_token`, but pulls fall back to ANONYMOUS.

    The Hub token service grants anonymous pull tokens for public projects,
    and the cascade checkpoint/generator repos are public by design — they
    are the artefacts validators and auditors verify. A validator therefore
    needs no Hub account to eval a round (external validators hit exactly
    this on 2026-07-18: checkpoint fetch raised HubAuthError although the
    registry would have served the pull anonymously). Credentials, when
    present, are still used. Uploads keep the strict resolver — pushing
    always needs an identity. ``False`` is the library's explicit
    "anonymous; do not auto-discover" sentinel.
    """
    try:
        return _resolve_hub_token(action)
    except HubAuthError:
        import logging

        logging.getLogger("cascade.storage").info(
            "%s: no Hub credentials in env; using anonymous pull "
            "(public repos only)", action)
        return False


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


# A Hub push/pull is one HTTP round-trip to an IPFS-backed OCI registry, so a
# single chunk can silently stall ("The read operation timed out") even when the
# operation would sail through on a fresh attempt — a miner's whole `cascade
# deploy` should not die on one flaky chunk after it already passed verify.
# Retry transient network failures with exponential backoff; a HubAuthError, an
# invalid ref, or a genuine "not found" is deterministic and surfaces at once.
HUB_MAX_ATTEMPTS = 4
HUB_BACKOFF_BASE_S = 2.0

# Substrings (lower-cased) that mark a Hub error as a transient network blip
# worth retrying. reqwest/requests timeouts surface as "read operation timed
# out" / "read timed out"; 5xx are the registry itself being briefly unhappy.
_RETRYABLE_HUB_ERROR_SUBSTRINGS = (
    "timed out", "timeout", "read operation",
    "connection reset", "connection aborted", "connection error",
    "broken pipe", "temporarily unavailable", "try again",
    " 500 ", " 502 ", " 503 ", " 504 ",
    "500 server error", "502 server error", "503 server error", "504 server error",
)


def _is_retryable_hub_error(exc: BaseException) -> bool:
    """True if ``exc`` looks like a transient network failure, not a permanent one.

    Auth failures (:class:`HubAuthError`) are deterministic and never retried.
    Timeouts and connection resets are — whether they arrive as a stdlib
    ``TimeoutError``/``ConnectionError`` or, more commonly, wrapped in a library
    HTTP error whose *message* carries the timeout text.
    """
    if isinstance(exc, HubAuthError):
        return False
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(sub in text for sub in _RETRYABLE_HUB_ERROR_SUBSTRINGS)


def _retry_hub_op(op, what: str, *, attempts: int = HUB_MAX_ATTEMPTS,
                  base_delay: float = HUB_BACKOFF_BASE_S, sleep=time.sleep):
    """Run ``op`` (a Hub upload/download), retrying transient network failures.

    Backs off exponentially (``base_delay`` × 2ⁿ) between attempts. A
    non-retryable error (auth, bad ref) is re-raised immediately; exhausting the
    retries raises a :class:`StorageError` that names ``what`` and the last cause.
    """
    last_exc: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return op()
        except StorageError:
            raise  # our own validation errors (auth, bad digest) are deterministic
        except Exception as exc:  # noqa: BLE001 — classify below, re-raise if permanent
            last_exc = exc
            if attempt >= attempts or not _is_retryable_hub_error(exc):
                break
            sleep(base_delay * (2 ** (attempt - 1)))
    raise StorageError(f"{what} failed after {attempt} attempt(s): {last_exc}") from last_exc


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
    result = _retry_hub_op(
        lambda: upload_folder(
            repo_id=str(repo), folder_path=str(d), allow_patterns=ALLOW_PATTERNS, token=token,
        ),
        f"upload of {d} to {repo}",
    )
    digest = str(getattr(result, "oid", "") or "")
    if not DIGEST_RE.match(digest):
        raise StorageError(f"hub upload returned no usable sha256 digest: {result!r}")
    return HubUpload(ref=HubRef(str(repo), digest), size_bytes=_dir_size_bytes(d))


def upload_dir_to_hf(local_dir: Path | str, repo: str, *, token: str | None = None) -> HubUpload:
    """Mirror a generator folder to a HuggingFace **model** repo and return its
    immutable ``repo@hf:<commit_sha>`` ref — a miner's escape hatch for submitting
    when the Hippius Hub OCI registry is down.

    The whole downstream path already understands ``hf:`` refs: ``fetch_from_hub``
    snapshot-downloads that exact revision (also a **model** repo, no repo_type),
    and ``parse_commit`` accepts the ``hf:[0-9a-f]{40}`` digest grammar — so an
    HF-mirrored submission is trained and audited exactly like a Hub one. Only
    files matching :data:`ALLOW_PATTERNS` are pushed (the same allow-list the Hub
    path uses), so the fetched content matches byte-for-byte.

    Caveat vs the Hub: the Hub's ``sha256:`` OCI digest is a *content* hash
    (identical content ⇒ same digest, the audit re-derivation anchor), whereas an
    HF ``oid`` is a git commit SHA — it pins an immutable revision but is not
    content-addressed, so re-uploading identical content yields a *new* ref. It
    still locates + pins the submission, which is all the chain commit needs.
    Auth: ``HF_TOKEN`` / ``HUGGINGFACE_API_KEY``.
    """
    d = Path(local_dir)
    if not d.is_dir():
        raise StorageError(f"not_a_directory: {d}")
    try:
        from huggingface_hub import HfApi
    except ImportError as e:
        raise StorageError(
            "huggingface_hub not installed; install the [hippius] extra to mirror to HF"
        ) from e
    tok = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY")
    if not tok:
        raise HubAuthError(
            f"mirroring {d} to HF repo {repo} requires an HF token: set HF_TOKEN "
            "(or HUGGINGFACE_API_KEY)."
        )
    api = HfApi(token=tok)
    with contextlib.suppress(Exception):  # may already exist / no create perm
        api.create_repo(str(repo), repo_type="model", private=False, exist_ok=True)
    info = api.upload_folder(
        repo_id=str(repo), folder_path=str(d), repo_type="model",
        allow_patterns=ALLOW_PATTERNS,
        commit_message=f"cascade generator submission: {d.name}",
    )
    oid = str(getattr(info, "oid", "") or "")
    if not re.fullmatch(r"[0-9a-f]{40}", oid):
        # A short oid / tag won't satisfy DIGEST_RE — resolve main's full sha so
        # the ref pins an immutable revision the trainer can fetch.
        with contextlib.suppress(Exception):
            refs = api.list_repo_refs(str(repo), repo_type="model")
            oid = next((b.target_commit for b in refs.branches if b.name == "main"), oid)
    if not re.fullmatch(r"[0-9a-f]{40}", oid):
        raise StorageError(f"HF upload returned no usable commit sha: {info!r}")
    return HubUpload(ref=HubRef(str(repo), f"hf:{oid}"), size_bytes=_dir_size_bytes(d))


def upload_dir_to_hub_or_hf(
    local_dir: Path | str,
    repo: str,
    hub: HubConfig | None = None,
    *,
    hf_repo: str | None = None,
    hf_token: str | None = None,
) -> HubUpload:
    """Push a folder to the Hippius Hub, mirroring to a HuggingFace **model** repo
    when the Hub is unreachable — the training path's counterpart to the miner's
    ``cascade deploy --hf-repo`` escape hatch.

    The Hub is priority-one: its ``sha256:`` OCI digest is content-addressed and
    is the audit re-derivation anchor, so it is always tried first (with the usual
    retry/backoff in :func:`upload_dir_to_hub`). Only if that ultimately raises a
    :class:`StorageError` — a Hub outage, the ``_ensure_config_blob_uploaded``
    class of failure that otherwise aborts a round — *and* ``hf_repo`` is set do we
    fall back to :func:`upload_dir_to_hf` and return its ``repo@hf:<sha>`` ref,
    which the whole downstream (``fetch_from_hub``, ``parse_commit``) already
    handles. The trade: an ``hf:`` ref pins an immutable revision but is a git
    commit sha, not a content hash, so a fallback round stays auditable by
    corpus/contract digest but not by a re-derived checkpoint digest. With no
    ``hf_repo`` the Hub error propagates unchanged (the prior, Hub-only behaviour),
    so callers that must not silently mirror off-Hub are unaffected.
    """
    try:
        return upload_dir_to_hub(local_dir, repo, hub)
    except StorageError as hub_exc:
        if not hf_repo:
            raise
        import logging

        logging.getLogger("cascade.storage").warning(
            "Hub upload of %s to %s failed (%s); mirroring to HuggingFace %s",
            local_dir, repo, hub_exc, hf_repo,
        )
        return upload_dir_to_hf(local_dir, hf_repo, token=hf_token)


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

        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY")
        path = _retry_hub_op(
            lambda: hf_snapshot_download(
                repo_id=ref.repo, revision=ref.digest[3:], local_dir=str(dest),
                allow_patterns=ALLOW_PATTERNS, token=hf_token,
            ),
            f"fetch of {ref.immutable_ref}",
        )
    else:
        try:
            from hippius_hub import snapshot_download
        except ImportError as e:
            raise StorageError(
                "hippius-hub not installed; install the [hippius] extra to use the registry"
            ) from e
        hub_token = _resolve_hub_token_for_pull(f"Downloading {ref.immutable_ref}")
        try:
            path = _retry_hub_op(
                lambda: snapshot_download(
                    repo_id=ref.repo, revision=ref.digest, local_dir=str(dest),
                    allow_patterns=ALLOW_PATTERNS, token=hub_token,
                ),
                f"fetch of {ref.immutable_ref}",
            )
        except Exception as e:
            denied = any(s in str(e).lower()
                         for s in ("401", "403", "unauthorized", "forbidden"))
            if hub_token is False and denied:
                raise HubAuthError(
                    f"Anonymous pull of {ref.immutable_ref} was refused (private "
                    f"repo?): set a token ({', '.join(HUB_TOKEN_ENV_NAMES)}) or "
                    f"username+password ({HUB_USERNAME_ENV_NAMES[0]} + "
                    f"{HUB_PASSWORD_ENV_NAMES[0]})."
                ) from e
            raise
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
        # Tight deadlines: Hippius's gateway has slow nights, and every caller
        # here either has a fallback store (manifests) or retries on its own
        # poll cadence (pool index, provisioner watches). boto's defaults
        # (60s connect/read × legacy retries) let ONE bad GET burn minutes and
        # starve poll loops — observed live 2026-07-14 (provisioner cycles
        # taking minutes each, heartbeat/trigger starved). read_timeout is
        # per-socket-read inactivity, not total transfer, so multi-MB objects
        # still stream fine as long as bytes flow.
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"},
                      connect_timeout=10, read_timeout=20,
                      retries={"max_attempts": 2, "mode": "standard"}),
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


@dataclass
class HFFallbackStore:
    """Hippius S3 primary + a HuggingFace-Hub dataset fallback that engages ONLY
    when S3 is unavailable.

    Hippius S3 has had degraded windows that 500 the manifest/receipt objects and
    stall the whole round loop. This wrapper keeps the loop running through such
    an outage without changing the happy path:

    * **reads** try S3 first, then the HF mirror;
    * **writes** go to S3 first and, only if S3 fails, land on HF so the object is
      not lost.

    So when Hippius is healthy there is **zero HF traffic** (no commit spam, no
    latency), and during a Hippius outage the trainer's manifest write and the
    validator's read (and receipt publish) both transparently ride the HF mirror
    — the round completes end-to-end. Duck-types :class:`S3Store` (same
    ``put_text``/``get_text``/``put_bytes``/``get_bytes``), so it drops into every
    manifest/receipt call site. Uses a HF **dataset** repo (``hf_backup_repo``);
    make it public so receipts stay auditable during an outage. Auth: ``HF_TOKEN``.
    """

    primary: S3Store
    hf_repo: str
    _api: object = None
    _ensured: bool = False

    def _token(self) -> str | None:
        return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY")

    def _hf(self):
        if self._api is None:
            from huggingface_hub import HfApi

            self._api = HfApi(token=self._token())
            if not self._ensured:
                # may already exist / caller lacks create perm — the upload will
                # surface the real error if the repo is genuinely unusable.
                with contextlib.suppress(Exception):
                    self._api.create_repo(self.hf_repo, repo_type="dataset",
                                          private=False, exist_ok=True)
                self._ensured = True
        return self._api

    def _hf_put(self, key: str, data: bytes) -> None:
        import io

        self._hf().upload_file(
            path_or_fileobj=io.BytesIO(data), path_in_repo=key,
            repo_id=self.hf_repo, repo_type="dataset",
            commit_message=f"cascade S3-fallback: {key}",
        )

    def _hf_get(self, key: str) -> bytes:
        from huggingface_hub import hf_hub_download

        # force_download: latest.json changes every round, so never serve a
        # cached (stale) revision from a prior fetch.
        path = hf_hub_download(
            repo_id=self.hf_repo, filename=key, repo_type="dataset",
            force_download=True, token=self._token(),
        )
        return Path(path).read_bytes()

    def put_bytes(self, key: str, data: bytes, *,
                  content_type: str = "application/octet-stream", acl: str | None = None) -> None:
        try:
            self.primary.put_bytes(key, data, content_type=content_type, acl=acl)
            return
        except StorageError as e:
            import logging
            logging.getLogger("cascade.storage").warning(
                "S3 put failed for %s (%s); writing HF fallback %s", key, e, self.hf_repo)
        try:
            self._hf_put(key, data)
        except Exception as e:  # noqa: BLE001
            raise StorageError(f"both S3 and HF put failed for {key}: {e}") from e

    def put_text(self, key: str, text: str, *,
                 content_type: str = "text/plain", acl: str | None = None) -> None:
        self.put_bytes(key, text.encode("utf-8"), content_type=content_type, acl=acl)

    def get_bytes(self, key: str) -> bytes:
        try:
            return self.primary.get_bytes(key)
        except StorageError as e:
            import logging
            logging.getLogger("cascade.storage").warning(
                "S3 get failed for %s (%s); reading HF fallback %s", key, e, self.hf_repo)
        try:
            return self._hf_get(key)
        except Exception as e:  # noqa: BLE001
            raise StorageError(f"both S3 and HF get failed for {key}: {e}") from e

    def get_text(self, key: str) -> str:
        return self.get_bytes(key).decode("utf-8")


@dataclass
class S3MirrorStore:
    """Hippius S3 primary + an S3-compatible (Cloudflare R2) mirror that receives
    a copy of every write — a genuine off-Hippius **backup**, not just a failover.

    Where :class:`HFFallbackStore` lands objects on HF *only while S3 is down* (so
    the mirror holds just the outage-era writes), this **dual-writes** every put to
    both stores, so the mirror always carries a complete, current copy of the
    manifest/receipt namespace. It therefore survives not only a Hippius S3
    *outage* but a Hippius S3 *data-loss* event — R2 keeps its own copy of every
    object that was ever written, healthy or not.

    * **writes** go to the primary first, then the mirror. A mirror-only failure
      is logged and swallowed — the object is safely on the primary, and a backup
      outage must never break the round loop. A primary failure still writes the
      mirror so the object is not lost (exactly like :class:`HFFallbackStore`);
      only if *both* fail does the put raise.
    * **reads** try the primary first, then fall back to the mirror.

    Duck-types :class:`S3Store` (``put_text``/``get_text``/``put_bytes``/
    ``get_bytes``), so it drops into every manifest/receipt call site through
    :func:`open_manifest_store`. The ``primary`` may itself be an
    :class:`HFFallbackStore`, so R2 and the HF failover can be stacked. R2 does
    not honour per-object canned ACLs, so a mirror write rejected for its ``acl``
    is retried without one — the backed-up bytes matter more than the ACL on the
    mirror copy (the primary keeps the public-read receipts).
    """

    primary: S3Store | HFFallbackStore
    mirror: S3Store

    @property
    def _mirror_label(self) -> str:
        return getattr(getattr(self.mirror, "cfg", None), "bucket", "r2-backup")

    def _mirror_put(self, key: str, data: bytes, *, content_type: str, acl: str | None) -> None:
        try:
            self.mirror.put_bytes(key, data, content_type=content_type, acl=acl)
        except StorageError:
            if not acl:
                raise
            # R2 rejects canned object ACLs — retry without so the backup lands.
            self.mirror.put_bytes(key, data, content_type=content_type)

    def put_bytes(self, key: str, data: bytes, *,
                  content_type: str = "application/octet-stream", acl: str | None = None) -> None:
        import logging
        log = logging.getLogger("cascade.storage")
        primary_ok = True
        try:
            self.primary.put_bytes(key, data, content_type=content_type, acl=acl)
        except StorageError as e:
            primary_ok = False
            log.warning("primary put failed for %s (%s); relying on R2 backup %s",
                        key, e, self._mirror_label)
        try:
            self._mirror_put(key, data, content_type=content_type, acl=acl)
        except Exception as e:  # noqa: BLE001 — a backup failure is not fatal on its own
            if not primary_ok:
                raise StorageError(f"both primary and R2 put failed for {key}: {e}") from e
            log.warning("R2 backup put failed for %s (%s); primary copy is intact",
                        key, e)

    def put_text(self, key: str, text: str, *,
                 content_type: str = "text/plain", acl: str | None = None) -> None:
        self.put_bytes(key, text.encode("utf-8"), content_type=content_type, acl=acl)

    def get_bytes(self, key: str) -> bytes:
        try:
            return self.primary.get_bytes(key)
        except StorageError as e:
            import logging
            logging.getLogger("cascade.storage").warning(
                "primary get failed for %s (%s); reading R2 backup %s",
                key, e, self._mirror_label)
        try:
            return self.mirror.get_bytes(key)
        except Exception as e:  # noqa: BLE001
            raise StorageError(f"both primary and R2 get failed for {key}: {e}") from e

    def get_text(self, key: str) -> str:
        return self.get_bytes(key).decode("utf-8")


def backup_s3_store(storage: object, *, bucket: str) -> S3Store | None:
    """Build an :class:`S3Store` for the Cloudflare R2 (or any S3-compatible)
    backup of the manifest/receipt bucket, or ``None`` when no backup is configured.

    Enabled by ``[storage] backup_s3_endpoint`` (R2's
    ``https://<account>.r2.cloudflarestorage.com``); the mirror bucket defaults to
    ``[storage] backup_bucket`` and falls back to the same ``bucket`` name as the
    primary. ``backup_s3_region`` defaults to ``"auto"`` (R2's region). Credentials
    come from ``BACKUP_S3_ACCESS_KEY`` / ``BACKUP_S3_SECRET_KEY`` — a distinct pair
    from the Hippius keys, since the backup lives on a different provider.
    """
    endpoint = getattr(storage, "backup_s3_endpoint", "") or ""
    if not endpoint:
        return None
    region = getattr(storage, "backup_s3_region", "") or "auto"
    bkt = getattr(storage, "backup_bucket", "") or bucket
    cfg = S3Config(
        endpoint=endpoint,
        region=region,
        bucket=bkt,
        access_key_env="BACKUP_S3_ACCESS_KEY",
        secret_key_env="BACKUP_S3_SECRET_KEY",
    )
    return S3Store(cfg)


def open_manifest_store(storage: object) -> S3Store | HFFallbackStore | S3MirrorStore:
    """The manifest/receipt bucket store, with optional backups layered on.

    Base is a plain :class:`S3Store`; if ``[storage] hf_backup_repo`` is set it is
    wrapped in an :class:`HFFallbackStore` (HF failover, unchanged), and if
    ``[storage] backup_s3_endpoint`` is set the result is wrapped in an
    :class:`S3MirrorStore` that dual-writes every object to a Cloudflare R2 backup.
    With neither configured this is exactly a :class:`S3Store` — no behaviour
    change. Every manifest/receipt call site builds its store through here."""
    bucket = getattr(storage, "manifest_bucket", "cascade-manifests")
    s3 = S3Store(S3Config.from_storage(storage, bucket=bucket))
    repo = getattr(storage, "hf_backup_repo", "") or ""
    base: S3Store | HFFallbackStore = HFFallbackStore(s3, repo) if repo else s3
    mirror = backup_s3_store(storage, bucket=bucket)
    return S3MirrorStore(base, mirror) if mirror else base


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


def receipt_latest_key(validator_hotkey: str = "") -> str:
    """The validator's own latest-receipt pointer; the shared legacy key when
    no hotkey is given."""
    if validator_hotkey:
        return f"receipts/{validator_hotkey}/latest.json"
    return RECEIPT_LATEST_KEY


def receipt_round_key(round_id: str, validator_hotkey: str = "") -> str:
    """Per-round receipt key, namespaced per validator.

    With a hotkey the key is ``receipts/<hotkey>/round-<id>.json`` — every
    validator owns its own prefix, so any number of validators can publish
    receipts for the same round without overwriting each other. The bare
    form is the legacy single-writer layout, kept so old rounds stay
    readable and local/test paths without a wallet still work.
    """
    if validator_hotkey:
        return f"receipts/{validator_hotkey}/round-{round_id}.json"
    return f"receipts/round-{round_id}.json"


def publish_receipt(
    store: S3Store, receipt_text: str, round_id: str, *, validator_hotkey: str = ""
) -> str:
    """Write the round receipt under the validator's own prefix.

    Mirrors :func:`publish_manifest` — the validator publishes its signed
    :class:`cascade.shared.receipt.RoundReceipt` here after weights are set.
    Three objects are written: the validator's per-round receipt and
    ``latest.json`` (both under ``receipts/<hotkey>/`` — single-writer keys,
    so concurrent validators never clobber each other's audit trail), plus
    the shared :data:`RECEIPT_LATEST_KEY` convenience pointer. That shared
    pointer is last-writer-wins by design: honest validators agree on the
    verdict, and each validator's authoritative copy lives under its own
    prefix regardless. Returns the per-round key.

    Exception: a REJECTED receipt for a round whose round key already holds a
    SCORED one publishes nothing (see the guard below) — such a same-round
    downgrade is a restarted validator re-gating an old manifest, and letting
    it take the latest pointers blanked the public dashboard until the next
    scored round.

    Receipts are the audit-facing artefact, so each object is written with a
    ``public-read`` ACL: third parties can then GET it (and run
    ``cascade-audit``) with zero credentials while the bucket — manifests,
    logs — stays private. On a backend that rejects object ACLs the write is
    retried private (the audit's anonymous fetch then falls back to
    credentials, as documented in docs/AUDIT.md).
    """
    round_key = receipt_round_key(round_id, validator_hotkey)
    keys = [round_key, receipt_latest_key(validator_hotkey)]
    if validator_hotkey:
        keys.append(RECEIPT_LATEST_KEY)
    # A SCORED verdict is only ever superseded by another scored verdict: on a
    # same-round-id re-judgement that ends in a gate rejection, the rejected
    # receipt must not clobber the signed scored record at the round key
    # (2026-07-15: two contract-switch rejections erased the receipt that
    # crowned the first king). The rejection stays visible via the latest
    # pointers; the round key keeps the authoritative scored copy.
    if _receipt_status(receipt_text) == "rejected" and _scored_receipt_at(store, round_key):
        import logging

        logging.getLogger("cascade.storage").warning(
            "suppressing rejected receipt for round %s: a scored receipt already "
            "sits at %s. A same-round scored→rejected re-judgement is a restart "
            "re-gating an old manifest (king_resyncing), not a new verdict — "
            "publishing it blanked the public dashboard's king/verdict for days "
            "(2026-07-21). Round key AND latest pointers keep the scored copy; "
            "the rejection stays diagnosable here and in the journal.",
            round_id, round_key)
        return round_key
    # Shared-pointer variant of the same protection: a validator that never
    # scored this round (nothing at its own round key — e.g. it rejected at a
    # gate) must still not steal the last-writer-wins shared pointer from
    # another validator's scored verdict of the SAME round. Its own prefix
    # keeps the rejection (that trail is the operator's diagnostic surface).
    if (_receipt_status(receipt_text) == "rejected" and RECEIPT_LATEST_KEY in keys
            and _scored_same_round_at(store, RECEIPT_LATEST_KEY, round_id)):
        import logging

        logging.getLogger("cascade.storage").warning(
            "keeping shared %s on the scored round-%s receipt; this rejected "
            "receipt publishes only under the validator's own prefix",
            RECEIPT_LATEST_KEY, round_id)
        keys.remove(RECEIPT_LATEST_KEY)
    try:
        for key in keys:
            store.put_text(key, receipt_text, content_type="application/json",
                           acl="public-read")
    except StorageError:
        # ACL unsupported on this backend: publish private rather than not at all.
        for key in keys:
            store.put_text(key, receipt_text, content_type="application/json")
    return round_key


def _receipt_status(receipt_text: str) -> str:
    try:
        doc = json.loads(receipt_text)
        return str(doc.get("status", "")) if isinstance(doc, dict) else ""
    except (ValueError, TypeError):
        return ""


def _scored_receipt_at(store: S3Store, key: str) -> bool:
    """Whether a receipt with ``status == "scored"`` already sits at ``key``.
    Unreadable/absent/malformed all read as False — precedence is best-effort,
    never a reason to fail a publish."""
    try:
        return _receipt_status(store.get_text(key)) == "scored"
    except Exception:  # noqa: BLE001 — absent key or store hiccup
        return False


def _scored_same_round_at(store: S3Store, key: str, round_id: str) -> bool:
    """Whether ``key`` holds a SCORED receipt for exactly ``round_id``.
    A scored receipt for a DIFFERENT round reads False: a newer round's
    rejection is real information and may move the pointer."""
    try:
        doc = json.loads(store.get_text(key))
        return (isinstance(doc, dict) and str(doc.get("status")) == "scored"
                and str(doc.get("round_id")) == str(round_id))
    except Exception:  # noqa: BLE001 — absent key or store hiccup
        return False


def read_receipt(store: S3Store, round_id: str, validator_hotkey: str = "") -> str:
    """Read one round's receipt JSON by round id (a validator's, or legacy)."""
    return store.get_text(receipt_round_key(round_id, validator_hotkey))


def read_latest_receipt(store: S3Store, validator_hotkey: str = "") -> str:
    """Read the newest receipt — a specific validator's, or the shared pointer."""
    return store.get_text(receipt_latest_key(validator_hotkey))


# ── receipts index (dashboard-facing) ───────────────────────────────────────
#
# The per-round receipts are the audit source of truth, but a static dashboard
# can't *list* a bucket to discover them. So every validator also maintains one
# shared public-read ``receipts/index.json`` — a rolling window of compact
# per-round summaries (see :func:`cascade.shared.receipt.summarize_receipt`) with
# a ``receipt_key`` pointer back to each signed receipt. Entries are keyed by
# ``(round_id, validator_hotkey)`` so a validator's update preserves its peers'
# entries; the read-modify-write itself is uncoordinated, so two validators
# publishing in the same instant can still drop one entry until that
# validator's next round. Acceptable because the index is presentational only:
# nothing here is signed or part of the audit contract (the per-validator
# receipts above are single-writer), and a stale/absent index never affects
# weights (the update is best-effort, like receipt publication).

RECEIPT_INDEX_KEY = "receipts/index.json"
RECEIPT_INDEX_SCHEMA = 2  # 2: entries keyed by (round_id, validator_hotkey)
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
    chain: dict | None = None,
    max_keep: int = RECEIPT_INDEX_MAX_KEEP,
) -> dict:
    """Append/replace one entry in ``receipts/index.json`` and write it public-read.

    Idempotent per ``(round_id, validator_hotkey)`` — a re-published round
    replaces the same validator's entry and leaves other validators' entries
    for that round intact. Sorted by ``epoch_start_block`` then ``round_id``
    then ``validator_hotkey`` (chronological — round ids are block-hash seeds,
    not monotonic), and capped at ``max_keep`` most-recent entries.
    ``updated_at`` (an ISO stamp), ``subnet`` (``{"netuid", "name"}``),
    and ``chain`` (schedule anchor for the next-round countdown:
    ``{as_of, current_block, epoch_start_block, epoch_blocks, block_time_s}``)
    are optional header fields the dashboard shows. Returns the stored entry.
    """
    entry = dict(summary)
    hotkey = str(entry.get("validator_hotkey") or "")
    entry["receipt_key"] = receipt_round_key(str(entry.get("round_id", "")), hotkey)
    if updated_at:
        entry["published_at"] = updated_at

    doc = read_receipt_index(store)
    rid = str(entry.get("round_id"))
    prior = [r for r in doc.get("rounds", [])
             if (str(r.get("round_id")), str(r.get("validator_hotkey") or ""))
             == (rid, hotkey)]
    rounds = [r for r in doc.get("rounds", [])
              if (str(r.get("round_id")), str(r.get("validator_hotkey") or ""))
              != (rid, hotkey)]
    # Scored precedence (mirrors publish_receipt): a rejected re-judgement of
    # the same round must not erase this validator's scored entry — that is
    # what blanked the dashboard's king on 2026-07-15. The rejected receipt
    # remains reachable via receipts/<hotkey>/latest.json.
    if (str(entry.get("status")) == "rejected"
            and any(str(r.get("status")) == "scored" for r in prior)):
        entry = next(r for r in prior if str(r.get("status")) == "scored")
    rounds.append(entry)
    rounds.sort(key=lambda r: (int(r.get("epoch_start_block", 0)),
                               str(r.get("round_id", "")),
                               str(r.get("validator_hotkey") or "")))
    rounds = rounds[-max_keep:]

    out: dict = {"schema": RECEIPT_INDEX_SCHEMA, "rounds": rounds}
    if updated_at:
        out["updated_at"] = updated_at
    if subnet:
        out["subnet"] = subnet
    if chain:
        out["chain"] = chain

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
# ``effective_block`` is the greatest ``<= the round's epoch-boundary block`` —
# so all validators score the identical pool for a given round REGARDLESS of
# when they polled (no latest-wins divergence at the daily rollover). Integrity
# is the tar sha256.
#
# Why the EPOCH BLOCK, not the round id: a round id is the epoch-boundary block
# HASH folded to a 64-bit seed (``ChainClient.block_seed``) — deliberately
# unpredictable and therefore NON-monotonic. Selecting "greatest effective_round
# <= round_id" over random seeds is meaningless. The epoch-boundary block NUMBER
# (``created_block // epoch_blocks * epoch_blocks``) is monotonic and derivable
# by every validator from the manifest's ``created_block``, so it is the correct
# ordering key for "which daily snapshot is active for this round".
#
# Invariant the publisher MUST hold: a new snapshot's ``effective_block`` is in
# the FUTURE (a later epoch than the current round). Never publish a snapshot
# that becomes active for an already-processed round, or validators that already
# scored it would disagree with those that re-select it.

POOL_INDEX_KEY = "pool/index.json"
POOL_INDEX_SCHEMA = 2   # v1 keyed snapshots by round_id (non-monotonic; broken)


def pool_snapshot_key(effective_block: int) -> str:
    return f"pool/snapshots/block-{int(effective_block)}.tar"


@dataclass(frozen=True)
class PoolSnapshotMeta:
    """One published eval-pool snapshot, listed in ``pool/index.json``.

    ``effective_block`` is the epoch-boundary block from which this snapshot is
    active; validators select by it (greatest ``<=`` the round's epoch block).
    """

    effective_block: int
    key: str
    sha256: str
    size_bytes: int
    as_of: str
    n_series: int
    context_length: int
    horizon: int

    @classmethod
    def from_dict(cls, d: dict) -> PoolSnapshotMeta:
        # ``effective_round`` is the retired v1 key; read it as a fallback so an
        # old index still parses (a redeploy republishes with ``effective_block``).
        block = d.get("effective_block", d.get("effective_round"))
        return cls(
            effective_block=int(block),
            key=str(d["key"]),
            sha256=str(d["sha256"]),
            size_bytes=int(d.get("size_bytes", 0)),
            as_of=str(d.get("as_of", "")),
            n_series=int(d.get("n_series", 0)),
            context_length=int(d.get("context_length", 0)),
            horizon=int(d.get("horizon", 0)),
        )


def read_pool_index(store: S3Store) -> list[PoolSnapshotMeta]:
    """Read the snapshot index, sorted by ``effective_block``. Empty if absent."""
    try:
        text = store.get_text(POOL_INDEX_KEY)
    except StorageError:
        return []
    doc = json.loads(text)
    snaps = [PoolSnapshotMeta.from_dict(s) for s in doc.get("snapshots", [])]
    return sorted(snaps, key=lambda s: s.effective_block)


def select_snapshot(index: list[PoolSnapshotMeta], epoch_block: int) -> PoolSnapshotMeta | None:
    """The snapshot active for a round at ``epoch_block`` (its epoch-boundary block
    number): greatest ``effective_block <= epoch_block``.

    Falls back to the earliest snapshot when ``epoch_block`` precedes them all (so
    a validator always has a pool); returns ``None`` only for an empty index. The
    rule is deterministic over a monotonic key, so every validator selects the
    same snapshot for the same round regardless of when it polled.
    """
    if not index:
        return None
    eligible = [s for s in index if s.effective_block <= epoch_block]
    if eligible:
        return max(eligible, key=lambda s: s.effective_block)
    return min(index, key=lambda s: s.effective_block)


def publish_pool_snapshot(
    store: S3Store,
    tar_bytes: bytes,
    *,
    effective_block: int,
    as_of: str,
    n_series: int,
    context_length: int,
    horizon: int,
    max_keep: int = 14,
) -> PoolSnapshotMeta:
    """Upload a pool snapshot tar and register it in ``pool/index.json``.

    Idempotent per ``effective_block`` (re-publishing replaces that entry). Keeps
    the most recent ``max_keep`` entries in the index (old tars are left in the
    bucket for any validator still resolving an older round; prune out-of-band).
    """
    sha = tar_cid_digest(tar_bytes)
    key = pool_snapshot_key(effective_block)
    store.put_bytes(key, tar_bytes, content_type="application/x-tar")

    meta = PoolSnapshotMeta(
        effective_block=int(effective_block),
        key=key,
        sha256=sha,
        size_bytes=len(tar_bytes),
        as_of=as_of,
        n_series=n_series,
        context_length=context_length,
        horizon=horizon,
    )
    index = [s for s in read_pool_index(store) if s.effective_block != meta.effective_block]
    index.append(meta)
    index.sort(key=lambda s: s.effective_block)
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
