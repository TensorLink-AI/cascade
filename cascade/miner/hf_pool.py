"""``--pool-hf`` — score on a RETIRED eval-pool snapshot from the public HF reveal.

The validator's private pool is revealed on HF (``Tensor-Link/cascade-eval-pool``)
once a newer snapshot has superseded it (:mod:`cascade.pool.reveal`). Each
``snapshots/<as_of>-block-<N>/`` folder holds a ``block-<N>.tar`` plus its
``POOL_SHA256``. Legacy ``snapshots/<YYYY-MM-DD>/`` folders were rebuilds, not
byte mirrors, so they are never selected.

This module pins the dataset's current commit revision, downloads the tar from
that revision, checks its sha256 against the dataset's own ``POOL_SHA256``
(integrity of the download, NOT authenticity: both files come from the same
repo; nothing here cross-checks a signed round receipt), and scores from the
extracted tar (never the convenience ``.npy`` copies next to it). The tar is
deleted after a verified extract. Snapshots are immutable, so the extraction
is cached under ``<cache_root>/hf-pool/<sha256>/pool/`` and a named snapshot
is reused with no network at all; ``latest`` falls back to the newest cached
snapshot when HF cannot be reached.

Pure stdlib (``urllib``): ``huggingface_hub`` is only in optional extras, and
the miner path must not need it. The data is evaluation data, never training
data, and a retired pool is directional only.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tarfile
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

HF_POOL_REPO = "Tensor-Link/cascade-eval-pool"
LATEST = "latest"
# ASCII digits only: \d would also match other Unicode digit characters.
SNAPSHOT_RE = re.compile(r"^(?P<as_of>[0-9]{4}-[0-9]{2}-[0-9]{2})-block-(?P<block>[0-9]+)$", re.ASCII)
LEGACY_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$", re.ASCII)
SHA_RE = re.compile(r"[0-9a-f]{64}", re.ASCII)
REVISION_RE = re.compile(r"[0-9a-f]{40}", re.ASCII)
SHA_MARKER = "POOL_SHA256"
COMPLETE_MARKER = "snapshot.json"
TIMEOUT_S = 60

RETIRED_POOL_NOTE = (
    "note: --pool-hf scores on a RETIRED eval-pool snapshot (public reveal of a pool "
    "validators no longer use). The number is directional only; live duels use a newer "
    "private pool. This is EVALUATION data: never train on it or fold it into your "
    "generator; doing so only overfits a retired benchmark. The tar is checked against the "
    "dataset's own POOL_SHA256 (download integrity, not authenticity)."
)


class HfPoolError(RuntimeError):
    """The HF pool could not be resolved, fetched, verified or extracted."""


class HfPoolIntegrityError(HfPoolError):
    """The downloaded tar does not match its published ``POOL_SHA256``."""


@dataclass(frozen=True)
class HfPool:
    snapshot: str      # e.g. "2026-10-31-block-9388800"
    sha256: str        # sha256 of block-<N>.tar, as published in the folder's POOL_SHA256
    block: int
    path: Path         # the extracted tar: a pool dir for window_source_from_dir
    revision: str = ""  # the HF dataset commit the tar was downloaded from

    @property
    def label(self) -> str:
        return f"hf:{HF_POOL_REPO}/snapshots/{self.snapshot} sha256={self.sha256[:12]} (retired)"

    def as_dict(self) -> dict:
        return {"repo": HF_POOL_REPO, "snapshot": self.snapshot, "sha256": self.sha256,
                "block": self.block, "revision": self.revision}


def _endpoint() -> str:
    return os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")


def _open(url: str):
    """The one network seam (tests replace it). Anonymous: the dataset is public
    and no token is ever sent."""
    req = urllib.request.Request(url, headers={"User-Agent": "cascade-miner"})
    return urllib.request.urlopen(req, timeout=TIMEOUT_S)  # noqa: S310 — fixed https endpoint


def _resolve_url(path: str, revision: str) -> str:
    return f"{_endpoint()}/datasets/{HF_POOL_REPO}/resolve/{revision}/{path}"


def validate_spec(spec: str) -> str:
    """``latest`` or an exact ``<as_of>-block-<N>`` folder name; anything else
    (a legacy date folder, a path swallowed by the optional argument) is refused."""
    s = str(spec or LATEST).strip().strip("/")
    s = s.removeprefix("snapshots/")
    if s.lower() == LATEST:
        return LATEST
    if SNAPSHOT_RE.match(s):
        return s
    if LEGACY_RE.match(s):
        raise HfPoolError(
            f"--pool-hf {s!r} is a legacy rebuild folder, not a byte mirror of a scored pool; "
            "use 'latest' or a '<YYYY-MM-DD>-block-<N>' folder")
    raise HfPoolError(
        f"--pool-hf expects 'latest' or a snapshot folder like '2026-10-31-block-9388800'; "
        f"got {s!r} (put --pool-hf after the repo path, or pass its value explicitly)")


def pick_latest(names) -> str:
    """The newest ``<as_of>-block-<N>`` folder by block. Legacy date-only
    folders never qualify, however recent their date."""
    best = None
    for raw in names:
        name = str(raw).strip("/").rsplit("/", 1)[-1]
        m = SNAPSHOT_RE.match(name)
        if m and (best is None or int(m["block"]) > best[0]):
            best = (int(m["block"]), name)
    if best is None:
        raise HfPoolError(f"no '<as_of>-block-<N>' snapshot folders found in {HF_POOL_REPO}")
    return best[1]


def current_revision() -> str:
    """The dataset's current commit sha (``/api/datasets/<repo>/revision/main``),
    so the listing, POOL_SHA256 and tar all come from ONE commit."""
    with _open(f"{_endpoint()}/api/datasets/{HF_POOL_REPO}/revision/main") as resp:
        doc = json.loads(resp.read().decode("utf-8"))
    sha = str((doc or {}).get("sha") or "")
    if not REVISION_RE.fullmatch(sha):
        raise HfPoolError(f"HF returned no commit revision for {HF_POOL_REPO}: {sha[:60]!r}")
    return sha


def list_snapshots(revision: str = "main") -> list[str]:
    """Folder names under ``snapshots/`` from the HF tree API (follows pagination)."""
    url = f"{_endpoint()}/api/datasets/{HF_POOL_REPO}/tree/{revision}/snapshots"
    names: list[str] = []
    for _ in range(100):
        with _open(url) as resp:
            rows = json.loads(resp.read().decode("utf-8"))
            link = (getattr(resp, "headers", None) or {}).get("Link", "") or ""
        names.extend(r["path"].rsplit("/", 1)[-1] for r in rows
                     if isinstance(r, dict) and r.get("type") == "directory"
                     and isinstance(r.get("path"), str))
        m = re.search(r'<([^>]+)>;\s*rel="next"', link)
        if not m:
            return names
        url = m.group(1)
    return names


def _block_of(snapshot: str) -> int:
    return int(SNAPSHOT_RE.match(snapshot)["block"])


def _cached(cache: Path) -> dict[str, HfPool]:
    """Every complete cached snapshot, by folder name."""
    out: dict[str, HfPool] = {}
    if not cache.is_dir():
        return out
    for marker in cache.glob(f"*/{COMPLETE_MARKER}"):
        try:
            doc = json.loads(marker.read_text())
            snap, sha = str(doc["snapshot"]), str(doc["sha256"])
            revision = str(doc.get("revision") or "")
        except (OSError, ValueError, KeyError, TypeError):
            continue
        pool = marker.parent / "pool"
        if marker.parent.name == sha and SNAPSHOT_RE.match(snap) and pool.is_dir():
            out[snap] = HfPool(snapshot=snap, sha256=sha, block=_block_of(snap), path=pool,
                               revision=revision)
    return out


def _download(url: str, dest: Path) -> str:
    """Stream ``url`` to ``dest``; returns the sha256 of the bytes written."""
    h = hashlib.sha256()
    with _open(url) as resp, dest.open("wb") as f:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
            f.write(chunk)
    return h.hexdigest()


def safe_extract(tar_path: Path, dest: Path) -> Path:
    """Extract an UNTRUSTED tar under ``dest``: regular files and directories
    only (no links, devices or fifos), every path strictly inside ``dest``, and
    the stdlib ``data`` filter on top where this Python has it. Kept local to
    the miner path; the validator/trainer unpacker is unchanged."""
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    with tarfile.open(tar_path, mode="r:*") as tar:
        for member in tar.getmembers():
            if not (member.isreg() or member.isdir()):
                raise HfPoolError(f"unsafe tar member (not a regular file/dir): {member.name!r}")
            target = (dest / member.name).resolve()
            if target != root and root not in target.parents:
                raise HfPoolError(f"unsafe tar member (escapes the pool dir): {member.name!r}")
        if hasattr(tarfile, "data_filter"):
            tar.extractall(dest, filter="data")
        else:  # pragma: no cover — Pythons without PEP 706; members vetted above
            tar.extractall(dest)  # noqa: S202
    return dest


def _fetch(snapshot: str, cache: Path, revision: str) -> HfPool:
    block = _block_of(snapshot)
    with _open(_resolve_url(f"snapshots/{snapshot}/{SHA_MARKER}", revision)) as resp:
        text = resp.read().decode("utf-8", "replace").strip()
    want = text.split()[0].lower() if text else ""
    if not SHA_RE.fullmatch(want):
        raise HfPoolError(f"{snapshot}/{SHA_MARKER} is not a sha256: {text[:80]!r}")
    cache.mkdir(parents=True, exist_ok=True)
    stage = cache / f".incoming-{want[:12]}"
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    try:
        tar = stage / f"block-{block}.tar"
        got = _download(_resolve_url(f"snapshots/{snapshot}/block-{block}.tar", revision), tar)
        if got != want:
            raise HfPoolIntegrityError(
                f"refusing {snapshot}: block-{block}.tar sha256 {got} != {SHA_MARKER} {want}")
        # Score from the verified tar, never the unpacked convenience copies.
        safe_extract(tar, stage / "pool")
        tar.unlink()  # the extraction is what is scored; keep disk use to one copy
        if not any((stage / "pool").rglob("*.npy")) and not any((stage / "pool").rglob("*.npz")):
            raise HfPoolError(f"{snapshot}: verified tar holds no .npy/.npz series")
        (stage / COMPLETE_MARKER).write_text(json.dumps(
            {"repo": HF_POOL_REPO, "snapshot": snapshot, "sha256": want, "block": block,
             "revision": revision}, sort_keys=True))
        final = cache / want
        shutil.rmtree(final, ignore_errors=True)
        stage.rename(final)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return HfPool(snapshot=snapshot, sha256=want, block=block, path=final / "pool",
                  revision=revision)


def _resolve(snap: str, cache: Path, warn) -> HfPool:
    cached = _cached(cache)
    if snap != LATEST and snap in cached:
        return cached[snap]
    try:
        revision = current_revision()
        if snap == LATEST:
            snap = pick_latest(list_snapshots(revision))
            if snap in cached:
                return cached[snap]
    except (OSError, ValueError, HfPoolError) as e:  # URLError/HTTPError are OSErrors
        if snap != LATEST or not cached:
            raise HfPoolError(f"cannot reach {HF_POOL_REPO} ({e}) and the snapshot is not "
                              "cached; retry online or pass --pool-dir") from e
        best = max(cached.values(), key=lambda p: p.block)
        if warn:
            warn(f"warning: could not reach HF ({e}); reusing newest cached snapshot "
                 f"{best.snapshot} — 'latest' may be stale")
        return best
    return _fetch(snap, cache, revision)


def resolve_hf_pool(spec: str, cache_root: Path | str, *,
                    warn: Callable[[str], None] | None = None) -> HfPool:
    """Resolve ``--pool-hf`` to a verified, extracted local pool.

    A named snapshot already in the cache needs no network. ``latest`` asks HF
    for the newest ``-block-<N>`` folder at the current commit and, if HF is
    unreachable, reuses the newest cached snapshot (with a warning). A sha256
    mismatch raises :class:`HfPoolIntegrityError` and nothing is cached; every
    other failure (network, malformed tar, unsafe member, storage) is an
    :class:`HfPoolError`."""
    snap = validate_spec(spec)
    cache = Path(cache_root) / "hf-pool"
    try:
        return _resolve(snap, cache, warn)
    except HfPoolError:
        raise
    except (OSError, ValueError, UnicodeError, tarfile.TarError, EOFError) as e:
        raise HfPoolError(f"--pool-hf {snap}: {type(e).__name__}: {e}") from e
