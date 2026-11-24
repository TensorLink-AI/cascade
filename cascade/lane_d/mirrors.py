"""Production mirror clients for lane D king publication (LANE_D_PLAN §11.1, W13).

:mod:`cascade.lane_d.publish` programs against a mirror client: ``put(key,
data)``, ``get(key)`` (``KeyError`` when absent), ``list(prefix)`` (sorted full
keys), ``size(key)`` and ``delete(key)``, with keys
``lane-d-kings/<digest>/<file>``. This module adapts the existing storage
backends to that interface; it adds no bucket, no repo namespace and no
credential name:

* **hub**: the public Hippius Hub repo ``{[storage] hub_namespace}/laned-king-
  c{context}-{hotkey8}`` (:func:`cascade.lane_d.publish.hub_repo`; one repo per
  king), through ``hippius_hub.HippiusApi``; the token is the existing
  ``HIPPIUS_HUB_TOKEN`` / ``HIPPIUS_TOKEN`` (or the username/password pair),
  resolved by :mod:`cascade.shared.hippius`. A file is ``<file>`` in the repo.
* **hf**: the HF dataset :data:`cascade.lane_d.publish.HF_MIRROR_REPO` through
  ``huggingface_hub.HfApi`` with ``HF_TOKEN`` / ``HUGGINGFACE_API_KEY``; the
  file path in the dataset is the full key.
* **r2**: the existing Cloudflare R2 backup endpoint (``[storage]
  backup_s3_endpoint``, bucket ``backup_bucket`` else ``manifest_bucket``,
  credentials ``BACKUP_S3_ACCESS_KEY`` / ``BACKUP_S3_SECRET_KEY``) under the
  ``lane-d-kings/`` prefix. A missing bucket is an error, never created.

:func:`open_live_vault` is the private vault (``[lane_d_ops] vault_bucket``,
read only here) and :func:`open_index_store` the public manifest store the
signed ``champions/lane_d/index.json`` goes to (``open_manifest_store``: Hippius
S3 with the R2 dual-write), written ``public-read``.

Every client is lazy: nothing is imported, read or opened until a call. The
mirror objects are what the ``king-publish`` / ``takedown`` jobs pass to
:func:`~cascade.lane_d.publish.publish_king` / :func:`~cascade.lane_d.publish.
takedown`; tests pass fakes of the ``api`` / ``client`` objects.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path

from .publish import HF_MIRROR_REPO, KINGS_PREFIX, hub_repo, mirror_prefix
from .storage import check_key

HUB_REPO_TYPE = "model"
HF_REPO_TYPE = "dataset"
R2_ACCESS_KEY_ENV = "BACKUP_S3_ACCESS_KEY"
R2_SECRET_KEY_ENV = "BACKUP_S3_SECRET_KEY"
HF_TOKEN_ENVS = ("HF_TOKEN", "HUGGINGFACE_API_KEY")
_MISSING_NAMES = frozenset(("EntryNotFoundError", "RemoteEntryNotFoundError",
                            "RepositoryNotFoundError", "RevisionNotFoundError",
                            "LocalEntryNotFoundError"))
_MISSING_CODES = frozenset(("NoSuchKey", "NotFound", "404"))


class MirrorError(RuntimeError):
    """A mirror cannot be opened or answered something unusable."""


def _missing(exc: BaseException) -> bool:
    """A not-found answer from an HF-shaped API or boto3."""
    for cls in type(exc).__mro__:
        if cls.__name__ in _MISSING_NAMES:
            return True
    resp = getattr(exc, "response", None)
    if isinstance(resp, dict):
        code = str((resp.get("Error") or {}).get("Code", ""))
        status = (resp.get("ResponseMetadata") or {}).get("HTTPStatusCode")
        return code in _MISSING_CODES or status == 404
    status = getattr(resp, "status_code", None)
    return status == 404


class RepoMirror:
    """A mirror over one HF-shaped repo (``hippius_hub.HippiusApi`` or
    ``huggingface_hub.HfApi``). ``path_of(key)`` maps a mirror key to the
    file path in the repo (``None``: the key is outside this mirror)."""

    def __init__(self, api, *, repo_id: str, repo_type: str,
                 path_of: Callable[[str], str | None], key_of: Callable[[str], str | None]):
        self.api = api
        self.repo_id = str(repo_id)
        self.repo_type = str(repo_type)
        self._path_of = path_of
        self._key_of = key_of
        self._created = False

    def _path(self, key: str) -> str:
        path = self._path_of(check_key(key))
        if path is None:
            raise KeyError(key)
        return path

    def put(self, key: str, data: bytes) -> None:
        path = self._path(key)
        if not self._created:
            self.api.create_repo(self.repo_id, repo_type=self.repo_type, private=False,
                                 exist_ok=True)
            self._created = True
        self.api.upload_file(path_or_fileobj=bytes(data), path_in_repo=path,
                             repo_id=self.repo_id, repo_type=self.repo_type,
                             commit_message=f"lane D king: {key}")

    def get(self, key: str) -> bytes:
        path = self._path(key)
        tmp = tempfile.mkdtemp(prefix="laned-mirror-")
        try:
            try:
                local = self.api.hf_hub_download(repo_id=self.repo_id, filename=path,
                                                 repo_type=self.repo_type, local_dir=tmp)
            except Exception as e:
                if _missing(e):
                    raise KeyError(key) from None
                raise
            return Path(local).read_bytes()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def _files(self) -> list[str]:
        try:
            return list(self.api.list_repo_files(self.repo_id, repo_type=self.repo_type))
        except Exception as e:
            if _missing(e):
                return []
            raise

    def list(self, prefix: str) -> list[str]:
        keys = (self._key_of(p) for p in self._files())
        return sorted(k for k in keys if k is not None and k.startswith(prefix))

    def size(self, key: str) -> int:
        path = self._path(key)
        try:
            infos = list(self.api.get_paths_info(self.repo_id, [path], repo_type=self.repo_type))
        except Exception as e:
            if _missing(e):
                raise KeyError(key) from None
            raise
        for info in infos:
            if getattr(info, "path", None) == path and getattr(info, "size", None) is not None:
                return int(info.size)
        raise KeyError(key)

    def delete(self, key: str) -> None:
        path = self._path(key)
        try:
            self.api.delete_file(path, self.repo_id, repo_type=self.repo_type,
                                 commit_message=f"lane D takedown: {key}")
        except Exception as e:
            if _missing(e):
                return
            raise


def hub_mirror(api, *, namespace: str, context: int, hotkey: str, digest: str) -> RepoMirror:
    """The Hub mirror of ONE king: its repo holds ``<file>`` for the key
    ``lane-d-kings/<digest>/<file>``; any other key is outside it."""
    prefix = mirror_prefix(digest)

    def path_of(key: str) -> str | None:
        return key[len(prefix):] if key.startswith(prefix) and len(key) > len(prefix) else None

    return RepoMirror(api, repo_id=hub_repo(namespace, context, hotkey),
                      repo_type=HUB_REPO_TYPE, path_of=path_of, key_of=lambda p: prefix + p)


def hf_mirror(api, *, repo_id: str = HF_MIRROR_REPO) -> RepoMirror:
    """The HF dataset mirror: the file path in the dataset is the key."""

    def under(key: str) -> str | None:
        return key if key.startswith(f"{KINGS_PREFIX}/") else None

    return RepoMirror(api, repo_id=repo_id, repo_type=HF_REPO_TYPE, path_of=under, key_of=under)


class S3Mirror:
    """A mirror over one existing S3-compatible bucket (boto3-shaped client
    factory ``client()``); keys are object keys. Never creates the bucket."""

    def __init__(self, client: Callable[[], object], *, bucket: str):
        self._client = client
        self.bucket = str(bucket)

    def put(self, key: str, data: bytes) -> None:
        self._client().put_object(Bucket=self.bucket, Key=check_key(key), Body=bytes(data),
                                  ContentType="application/octet-stream")

    def get(self, key: str) -> bytes:
        try:
            resp = self._client().get_object(Bucket=self.bucket, Key=check_key(key))
        except Exception as e:
            if _missing(e):
                raise KeyError(key) from None
            raise
        body = resp["Body"]
        try:
            return body.read()
        finally:
            close = getattr(body, "close", None)
            if callable(close):
                close()

    def list(self, prefix: str) -> list[str]:
        keys: list[str] = []
        kwargs = {"Bucket": self.bucket, "Prefix": prefix}
        while True:
            resp = self._client().list_objects_v2(**kwargs)
            keys.extend(str(o["Key"]) for o in resp.get("Contents", ()))
            if not resp.get("IsTruncated"):
                break
            kwargs["ContinuationToken"] = resp["NextContinuationToken"]
        return sorted(keys)

    def size(self, key: str) -> int:
        try:
            resp = self._client().head_object(Bucket=self.bucket, Key=check_key(key))
        except Exception as e:
            if _missing(e):
                raise KeyError(key) from None
            raise
        return int(resp["ContentLength"])

    def delete(self, key: str) -> None:
        self._client().delete_object(Bucket=self.bucket, Key=check_key(key))


class PublicStore:
    """``put(key, data)`` onto a manifest store (``put_bytes``), public-read:
    where :func:`~cascade.lane_d.publish.publish_index` puts the index."""

    def __init__(self, store):
        self.store = store

    def put(self, key: str, data: bytes) -> None:
        self.store.put_bytes(check_key(key), bytes(data), content_type="application/json",
                             acl="public-read")


# ── the live clients (env credentials, names only here) ─────────────────────


def _lazy(build: Callable[[], object]) -> Callable[[], object]:
    box: list = []

    def get():
        if not box:
            box.append(build())
        return box[0]

    return get


def _hub_api(environ: Mapping[str, str]):
    from hippius_hub import HippiusApi

    from ..shared.hippius import _resolve_hub_token

    del environ                     # the shared resolver reads os.environ itself
    return HippiusApi(token=_resolve_hub_token("Publishing a lane D king"))


def _hf_api(environ: Mapping[str, str]):
    from huggingface_hub import HfApi

    token = next((environ[n] for n in HF_TOKEN_ENVS if environ.get(n)), None)
    if not token:
        raise MirrorError(f"the HF mirror needs {HF_TOKEN_ENVS[0]} in the job environment")
    return HfApi(token=token)


def r2_bucket(storage) -> str:
    """The existing R2 bucket the king mirror uses."""
    bucket = str(getattr(storage, "backup_bucket", "") or getattr(storage, "manifest_bucket", "")
                 or "")
    if not bucket:
        raise MirrorError("no [storage] backup_bucket / manifest_bucket for the R2 mirror")
    return bucket


def _r2_client(storage, environ: Mapping[str, str]):
    endpoint = str(getattr(storage, "backup_s3_endpoint", "") or "")
    if not endpoint:
        raise MirrorError("no [storage] backup_s3_endpoint: the R2 mirror is not configured")
    access, secret = environ.get(R2_ACCESS_KEY_ENV), environ.get(R2_SECRET_KEY_ENV)
    if not access or not secret:
        raise MirrorError(f"the R2 mirror needs {R2_ACCESS_KEY_ENV} / {R2_SECRET_KEY_ENV}")
    import boto3
    from botocore.config import Config

    return boto3.client("s3", endpoint_url=endpoint,
                        region_name=str(getattr(storage, "backup_s3_region", "") or "auto"),
                        aws_access_key_id=access, aws_secret_access_key=secret,
                        config=Config(signature_version="s3v4", s3={"addressing_style": "path"},
                                      connect_timeout=10, read_timeout=120,
                                      retries={"max_attempts": 2, "mode": "standard"}))


def open_live_mirrors(cfg, *, context: int, hotkey: str, digest: str,
                      environ: Mapping[str, str] | None = None,
                      hub_api=None, hf_api=None, r2_client=None) -> dict:
    """``{"hub", "r2", "hf"}`` mirror clients for one king from the loaded
    ``ChainConfig`` (``[storage]``). The ``*_api`` / ``r2_client`` arguments
    replace the live backends (tests)."""
    env = os.environ if environ is None else environ
    storage = cfg.storage
    hub = hub_api if hub_api is not None else _LazyApi(lambda: _hub_api(env))
    hf = hf_api if hf_api is not None else _LazyApi(lambda: _hf_api(env))
    r2 = (lambda: r2_client) if r2_client is not None else _lazy(lambda: _r2_client(storage, env))
    return {"hub": hub_mirror(hub, namespace=str(getattr(storage, "hub_namespace", "cascade")),
                              context=context, hotkey=hotkey, digest=digest),
            "r2": S3Mirror(r2, bucket=r2_bucket(storage)),
            "hf": hf_mirror(hf)}


class _LazyApi:
    """Builds the real API object at its first method call."""

    def __init__(self, build: Callable[[], object]):
        self._get = _lazy(build)

    def __getattr__(self, name: str):
        return getattr(self._get(), name)


def open_live_vault(cfg, environ: Mapping[str, str] | None = None):
    """The private vault (read by publication only)."""
    from .hippius_vault import S3VaultStorage, VaultClient, VaultEndpoint

    ep = VaultEndpoint.from_storage(cfg.storage, bucket=cfg.lane_d_ops.vault_bucket)
    return S3VaultStorage(VaultClient(ep, environ=environ))


def open_index_store(cfg) -> PublicStore:
    """The public store of ``champions/lane_d/index.json``."""
    from ..shared.hippius import open_manifest_store

    return PublicStore(open_manifest_store(cfg.storage))


__all__ = ["HF_REPO_TYPE", "HUB_REPO_TYPE", "MirrorError", "PublicStore", "RepoMirror",
           "S3Mirror", "hf_mirror", "hub_mirror", "open_index_store", "open_live_mirrors",
           "open_live_vault", "r2_bucket"]
