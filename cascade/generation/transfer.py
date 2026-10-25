"""Moving corpus shards with pre-signed links (DEC-CA-0053).

Neither the CPU VM nor the GPU pod ever holds a storage credential. The
orchestrator signs per-object, time-limited links with its own key:

* **upload links** (``PUT``) for every shard name the tier could write plus
  ``corpus.json`` — the VM uploads what it wrote and nothing else;
* **download links** (``GET``) for the shards one leg's index lists — the GPU
  pod streams exactly its own corpus (:func:`cascade.generation.shards.
  iter_corpus` reads ``urls.json``).

A link authorises one object, one verb, until it expires; it is a capability
for that object, not an account key.
"""

from __future__ import annotations

import logging
import math
import shutil
import time
import urllib.request
from pathlib import Path

from ..shared.config import GenerationConfig
from .shards import INDEX_NAME, URLS_NAME  # noqa: F401 — URLS_NAME re-exported

log = logging.getLogger("cascade.generation.transfer")

UPLOAD_EXPIRES_S = 12 * 3600       # window (5 h) + bootstrap + upload, with slack
DOWNLOAD_EXPIRES_S = 24 * 3600     # queue + rent + a 5 h wall, with slack
_ATTEMPTS = 5


def presign(store, method: str, key: str, expires_s: int) -> str:
    """A pre-signed ``get_object`` / ``put_object`` link for ``key`` on the
    store's bucket (an :class:`cascade.shared.hippius.S3Store`)."""
    if method not in ("get_object", "put_object"):
        raise ValueError(f"unsupported presign method {method!r}")
    return store.client().generate_presigned_url(
        method, Params={"Bucket": store.cfg.bucket, "Key": key}, ExpiresIn=int(expires_s))


def shard_names(gcfg: GenerationConfig) -> list[str]:
    """Every shard name the tier can write for ``gcfg``: per child, enough
    for its share of ``target_points`` at ``shard_points`` per shard plus one
    (a shard closes on the series that crosses the size)."""
    n = gcfg.n_children
    per_child = math.ceil(gcfg.target_points / n)
    shards = math.ceil(per_child / gcfg.shard_points) + 1
    return [f"c{i}-{k:05d}.parquet" for i in range(n) for k in range(shards)]


def upload_plan(store, base: str, gcfg: GenerationConfig,
                expires_s: int = UPLOAD_EXPIRES_S) -> dict[str, str]:
    """``{name: PUT link}`` under ``base`` for every possible shard and the index."""
    names = [*shard_names(gcfg), INDEX_NAME]
    return {name: presign(store, "put_object", base + name, expires_s) for name in names}


def download_plan(store, base: str, index: dict,
                  expires_s: int = DOWNLOAD_EXPIRES_S) -> dict[str, str]:
    """``{name: GET link}`` for the shards ``index`` lists."""
    names = [sh["name"] for c in index["children"] for sh in c["shards"]]
    return {name: presign(store, "get_object", base + name, expires_s) for name in names}


def _retry(what: str, fn):
    for attempt in range(1, _ATTEMPTS + 1):
        try:
            return fn()
        except OSError as e:
            if attempt == _ATTEMPTS:
                raise
            delay = min(60.0, 2.0 ** attempt)
            log.warning("%s failed (%s); retry %d/%d in %.0fs", what, e, attempt,
                        _ATTEMPTS - 1, delay)
            time.sleep(delay)
    return None


def put_file(url: str, path: Path | str, *, timeout: float = 600.0) -> None:
    """Upload ``path`` to a pre-signed PUT link, with retries."""
    p = Path(path)

    def _once() -> None:
        with open(p, "rb") as fh:
            req = urllib.request.Request(url, data=fh, method="PUT", headers={
                "Content-Length": str(p.stat().st_size)})
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 — operator-signed link
                if resp.status >= 300:
                    raise OSError(f"PUT {p.name}: HTTP {resp.status}")

    _retry(f"upload {p.name}", _once)


def get_file(url: str, path: Path | str, *, timeout: float = 600.0) -> None:
    """Download a pre-signed GET link into ``path`` (written via a temp file)."""
    p = Path(path)
    tmp = p.with_suffix(p.suffix + ".part")

    def _once() -> None:
        with urllib.request.urlopen(url, timeout=timeout) as resp, open(tmp, "wb") as fh:  # noqa: S310
            shutil.copyfileobj(resp, fh, length=1 << 20)
        tmp.replace(p)

    _retry(f"download {p.name}", _once)
