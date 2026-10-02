"""The worker image a round actually trained on, resolved to a rentable ref.

A round's signed contract pins ``train_image_digest`` (``sha256:…``). Lium
cannot launch a digest-only ref, so the digest is matched to its tag through
the registry's public API (anonymous pull token) and returned as
``<repo>:<tag>@sha256:…`` — the executor launches the tag and the template
carries the full pin. Resolutions are cached on disk: a tag→digest pair never
changes once published (a moved tag simply stops matching and is re-resolved).
"""

from __future__ import annotations

import json
import logging
import urllib.request
from pathlib import Path

log = logging.getLogger("cascade.miner.harness.images")

WORKER_REPO = "ghcr.io/tensorlink-ai/cascade-worker"
_ACCEPT = ", ".join((
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.docker.distribution.manifest.v2+json",
    "application/vnd.oci.image.manifest.v1+json",
))


def _get(url: str, headers: dict, method: str = "GET", timeout: float = 20.0):
    req = urllib.request.Request(url, headers=headers, method=method)
    return urllib.request.urlopen(req, timeout=timeout)


def _ghcr_tags_with_digests(repo: str) -> dict[str, str]:
    """``{tag: digest}`` for a public GHCR repo (newest tags last)."""
    host, _, name = repo.partition("/")
    with _get(f"https://{host}/token?scope=repository:{name}:pull", {}) as r:
        token = json.loads(r.read())["token"]
    auth = {"Authorization": f"Bearer {token}"}
    with _get(f"https://{host}/v2/{name}/tags/list", auth) as r:
        tags = json.loads(r.read()).get("tags", [])
    out = {}
    for tag in tags:
        if not tag.startswith("worker-v"):
            continue
        try:
            with _get(f"https://{host}/v2/{name}/manifests/{tag}",
                      {**auth, "Accept": _ACCEPT}, method="HEAD") as r:
                out[tag] = r.headers.get("Docker-Content-Digest", "")
        except Exception:  # noqa: BLE001
            continue
    return out


def resolve_worker_image(digest: str, *, cache: Path | None = None,
                         repo: str = WORKER_REPO, lister=None) -> str | None:
    """``repo:tag@digest`` for ``digest``, or None when no published tag has it."""
    d = digest if digest.startswith("sha256:") else f"sha256:{digest}"
    known = {}
    if cache is not None and cache.is_file():
        try:
            known = json.loads(cache.read_text(encoding="utf-8"))
        except ValueError:
            known = {}
    if d in known:
        return known[d]
    try:
        tags = (lister or _ghcr_tags_with_digests)(repo)
    except Exception as e:  # noqa: BLE001
        log.warning("cannot list %s tags: %s", repo, e)
        return None
    tag = next((t for t, td in tags.items() if td == d), None)
    if tag is None:
        return None
    ref = f"{repo}:{tag}@{d}"
    if cache is not None:
        known[d] = ref
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(known, indent=1), encoding="utf-8")
    return ref
