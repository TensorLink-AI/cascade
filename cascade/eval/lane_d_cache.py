"""Lane D prefetch cache (LANE_D_PLAN §1.2 step 5, §10 "Data paths" 2; W15).

On reveal the launch eval host prefetches each lane D entry from the vault,
through presigned GETs, into a local cache keyed by the entry digest:

* **Content addressed.** An entry lives at ``root/<digest>``, where the
  digest is the W1.5/W4 manifest digest
  (:func:`cascade.lane_d.intake.manifest_digest` of the canonical
  ``{"files": {name: {"sha256", "size"}}}`` over every regular file). A fill
  is admitted only if its bytes hash to the digest, and every read re-hashes
  every file (sizes and mtimes are never trusted): corrupt bytes are dropped,
  logged and never served.
* **Byte-budget LRU** (§1.6, #467). Admission evicts the least recently used
  entries until the new one fits; ``total_bytes() <= budget_bytes`` holds.
* **KING_D and the house are pinned.** They are never evicted. An admission
  that would fit only by evicting a pin is refused and evicts nothing.
* **Resumable.** The LRU order is a logical counter recorded in an index file
  under ``root`` (never file mtimes); a new cache on the same root recovers
  the entries and their order. Staging directories left by a crash are
  removed at open and are never entries.

The cache owns ``root``: one :class:`PrefetchCache` per root at a time.

:func:`prefetch_grant` fills the cache through W4's
:func:`cascade.eval.lane_d_ingest.ingest_presigned`, so every ingest guard
runs before an entry is admitted. :func:`open_prefetch_cache` returns ``None``
and creates nothing while lane D is not active (§3.2 invariant 1).

This module is torch-free; the only torch use on the prefetch path is W4's
finite check inside ``ingest_presigned``.

Log tags (logger ``cascade.eval.lane_d_cache``): INFO ``lane_d_cache_evict``,
WARNING ``lane_d_cache_corrupt``, WARNING ``lane_d_cache_refused <code>``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import stat
from pathlib import Path

from ..lane_d.intake import manifest_digest

log = logging.getLogger("cascade.eval.lane_d_cache")

CACHE_CODES = ("bad_digest", "digest_mismatch", "over_budget")
INDEX_FILE = ".lane_d_cache_index.json"
INDEX_VERSION = 1
STAGING_MARK = ".staging-"
INGEST_SUBDIR = ".ingest"
_HEX64 = re.compile(r"[0-9a-f]{64}")
_CHUNK = 1 << 20


class CacheError(RuntimeError):
    """A typed cache refusal; ``code`` is one of :data:`CACHE_CODES`."""

    def __init__(self, code: str, detail: str = ""):
        if code not in CACHE_CODES:
            raise ValueError(f"unknown cache code {code!r}")
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


def _check_digest(digest) -> str:
    if not isinstance(digest, str) or not _HEX64.fullmatch(digest):
        raise CacheError("bad_digest", repr(digest)[:80])
    return digest


def _pin(digest) -> str | None:
    if digest is None or digest == "":
        return None
    return _check_digest(digest)


def _hash_file(path: Path) -> tuple[str, int]:
    """sha256 and byte count of a regular file, read without following a
    symlink; the size is the count of bytes read, not ``st_size``."""
    h = hashlib.sha256()
    n = 0
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(fd, "rb") as f:
        while True:
            block = f.read(_CHUNK)
            if not block:
                break
            h.update(block)
            n += len(block)
    return h.hexdigest(), n


def tree_digest(directory: str | os.PathLike) -> tuple[str, int] | None:
    """``(digest, total_bytes)`` of every regular file below ``directory``
    under the W4 manifest rule, or ``None`` when the tree is not a plain
    checkpoint (empty, a symlink, or anything but directories and regular
    files)."""
    root = Path(directory)
    try:
        if not stat.S_ISDIR(os.lstat(root).st_mode):
            return None
    except FileNotFoundError:
        return None
    files: dict[str, dict] = {}
    total = 0
    for here, dirnames, filenames in os.walk(root, followlinks=False):
        base = Path(here)
        for name in dirnames:
            if not stat.S_ISDIR(os.lstat(base / name).st_mode):
                return None
        for name in filenames:
            p = base / name
            if not stat.S_ISREG(os.lstat(p).st_mode):
                return None
            sha, size = _hash_file(p)
            files[p.relative_to(root).as_posix()] = {"sha256": sha, "size": size}
            total += size
    if not files:
        return None
    return manifest_digest({"files": files}), total


def _rmtree(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


class PrefetchCache:
    """Sha-keyed, resumable, byte-budget LRU with KING_D and the house pinned
    (contract: ``tests/accept/w15/README.md``)."""

    def __init__(self, root: str | os.PathLike, *, budget_bytes: int,
                 king_d: str | None = None, house: str | None = None):
        if isinstance(budget_bytes, bool) or not isinstance(budget_bytes, int):
            raise TypeError(f"budget_bytes must be an int; got {budget_bytes!r}")
        if budget_bytes <= 0:
            raise ValueError(f"budget_bytes must be positive; got {budget_bytes!r}")
        pins = frozenset(p for p in (_pin(king_d), _pin(house)) if p)
        self.root = Path(root)
        self.budget_bytes = budget_bytes
        self._pinned = pins
        self._clock = 0
        self._entries: dict[str, tuple[int, int]] = {}   # digest -> (tick, size)
        self._staged = 0
        self.root.mkdir(parents=True, exist_ok=True)
        self._recover()
        self._fit_budget_at_open()
        self._save()

    # ── public API ──────────────────────────────────────────────────────────

    @property
    def pinned(self) -> frozenset:
        return self._pinned

    def set_pinned(self, *, king_d: str | None, house: str | None) -> None:
        """Replace the pinned set (KING_D and the house). A pin may name an
        entry not yet cached; the previous pins become ordinary entries."""
        self._pinned = frozenset(p for p in (_pin(king_d), _pin(house)) if p)

    def entries(self) -> list[str]:
        """Cached digests, least recently used first."""
        return [d for d, _ in sorted(self._entries.items(), key=lambda kv: kv[1][0])]

    def total_bytes(self) -> int:
        return sum(size for _, size in self._entries.values())

    def get(self, digest: str) -> Path | None:
        """The verified entry at ``root/<digest>`` (now most recently used), or
        ``None``. Every call re-hashes every file; a corrupt entry is deleted
        and never served."""
        _check_digest(digest)
        if digest not in self._entries:
            return None
        path = self.root / digest
        got = tree_digest(path)
        if got is None or got[0] != digest:
            log.warning("lane_d_cache_corrupt digest=%s hashed=%s pinned=%s", digest,
                        got[0] if got else "unreadable", digest in self._pinned)
            _rmtree(path)
            del self._entries[digest]
            self._save()
            return None
        self._touch(digest)
        self._save()
        return path

    def put(self, digest: str, fill) -> Path:
        """Admit ``digest`` by running ``fill(staging_dir)``; a verified cached
        digest is a hit and ``fill`` is not called."""
        try:
            _check_digest(digest)
        except CacheError as e:
            self._refused(e, digest)
            raise
        if digest in self._entries:
            hit = self.get(digest)
            if hit is not None:
                return hit
        staging = self._staging_dir(digest)
        try:
            fill(staging)
            got = tree_digest(staging)
            if got is None or got[0] != digest:
                raise CacheError("digest_mismatch",
                                 f"fill hashed to {got[0] if got else 'no files'}")
            size = got[1]
            for victim in self._victims(digest, size):
                self._evict(victim)
            final = self.root / digest
            _rmtree(final)
            os.replace(staging, final)
            self._clock += 1
            self._entries[digest] = (self._clock, size)
            self._save()
        except CacheError as e:
            self._refused(e, digest)
            raise
        finally:
            _rmtree(staging)
        return final

    def ensure(self, digest: str, fill) -> Path:
        """``get`` the entry, else ``put`` it (a corrupt entry is refetched)."""
        hit = self.get(digest)
        if hit is not None:
            return hit
        return self.put(digest, fill)

    # ── internals ───────────────────────────────────────────────────────────

    def _touch(self, digest: str) -> None:
        self._clock += 1
        self._entries[digest] = (self._clock, self._entries[digest][1])

    def _staging_dir(self, digest: str) -> Path:
        self._staged += 1
        staging = self.root / f".{digest}{STAGING_MARK}{os.getpid()}-{self._staged}"
        _rmtree(staging)
        staging.mkdir()
        return staging

    def _victims(self, digest: str, size: int) -> list[str]:
        """The LRU unpinned entries to evict so ``size`` more bytes fit; raises
        ``over_budget`` (evicting nothing) when only pins stand in the way."""
        if size > self.budget_bytes:
            raise CacheError("over_budget", f"{size} bytes > budget {self.budget_bytes}")
        need = self.total_bytes() + size - self.budget_bytes
        victims: list[str] = []
        for d in self.entries():
            if need <= 0:
                break
            if d in self._pinned or d == digest:
                continue
            victims.append(d)
            need -= self._entries[d][1]
        if need > 0:
            raise CacheError("over_budget",
                             f"{size} bytes fit only by evicting a pinned entry")
        return victims

    def _evict(self, digest: str) -> None:
        size = self._entries[digest][1]
        _rmtree(self.root / digest)
        del self._entries[digest]
        self._save()
        log.info("lane_d_cache_evict digest=%s size=%d total=%d budget=%d", digest, size,
                 self.total_bytes(), self.budget_bytes)

    def _refused(self, e: CacheError, digest) -> None:
        log.warning("lane_d_cache_refused %s digest=%s detail=%s", e.code,
                    str(digest)[:80], e.detail)

    def _load_index(self) -> dict[str, tuple[int, int]]:
        path = self.root / INDEX_FILE
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError, OSError):
            return {}
        if not isinstance(raw, dict) or raw.get("version") != INDEX_VERSION:
            return {}
        out: dict[str, tuple[int, int]] = {}
        entries = raw.get("entries")
        if not isinstance(entries, dict):
            return {}
        for d, meta in entries.items():
            if not (isinstance(d, str) and _HEX64.fullmatch(d) and isinstance(meta, dict)):
                continue
            tick, size = meta.get("tick"), meta.get("size")
            if all(isinstance(v, int) and not isinstance(v, bool) and v >= 0
                   for v in (tick, size)):
                out[d] = (tick, size)
        clock = raw.get("clock")
        if isinstance(clock, int) and not isinstance(clock, bool):
            self._clock = max(self._clock, clock)
        return out

    def _recover(self) -> None:
        indexed = self._load_index()
        on_disk: list[str] = []
        for p in sorted(self.root.iterdir()):
            name = p.name
            if name.startswith(".") and STAGING_MARK in name:
                _rmtree(p)          # a crash's staging leftover: never an entry
            elif _HEX64.fullmatch(name) and p.is_dir() and not p.is_symlink():
                on_disk.append(name)
        for d in on_disk:
            if d in indexed:
                self._entries[d] = indexed[d]
        if self._entries:
            self._clock = max(self._clock, max(t for t, _ in self._entries.values()))
        # A directory the index does not know (a crash between publish and the
        # index write) is adopted only if it verifies, as most recently used.
        for d in on_disk:
            if d in self._entries:
                continue
            got = tree_digest(self.root / d)
            if got is None or got[0] != d:
                log.warning("lane_d_cache_corrupt digest=%s hashed=%s at=open", d,
                            got[0] if got else "unreadable")
                _rmtree(self.root / d)
                continue
            self._clock += 1
            self._entries[d] = (self._clock, got[1])

    def _fit_budget_at_open(self) -> None:
        over = self.total_bytes() - self.budget_bytes
        for d in self.entries():
            if over <= 0:
                break
            if d in self._pinned:
                continue
            over -= self._entries[d][1]
            self._evict(d)
        if over > 0:
            log.warning("lane_d_cache_refused over_budget at=open pinned_bytes=%d budget=%d",
                        self.total_bytes(), self.budget_bytes)

    def _save(self) -> None:
        body = {"version": INDEX_VERSION, "clock": self._clock,
                "entries": {d: {"tick": t, "size": s}
                            for d, (t, s) in sorted(self._entries.items())}}
        tmp = self.root / f"{INDEX_FILE}.tmp-{os.getpid()}"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(body, f, sort_keys=True, separators=(",", ":"))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.root / INDEX_FILE)


def prefetch_grant(cache: PrefetchCache, grant: dict, *, contract=None,
                   expect=None) -> Path:
    """Prefetch a revealed entry from W4's presigned ``grant`` (``{"hotkey",
    "digest", "manifest_url", "files"}``) into ``cache``. A verified hit makes
    no vault request; a miss ingests through ``ingest_presigned`` (every W4
    guard runs before admission; an ``IngestError`` propagates and admits
    nothing)."""
    digest = grant.get("digest") if isinstance(grant, dict) else None
    _check_digest(digest)

    def fill(staging: Path) -> None:
        from .lane_d_ingest import ingest_presigned

        landing = staging / INGEST_SUBDIR
        out = ingest_presigned(grant, landing, contract=contract, expect=expect)
        for p in sorted(Path(out).iterdir()):
            os.replace(p, staging / p.name)
        _rmtree(landing)

    return cache.ensure(digest, fill)


def open_prefetch_cache(lane_d_cfg, boundary, root: str | os.PathLike, *,
                        budget_bytes: int, king_d: str | None = None,
                        house: str | None = None) -> PrefetchCache | None:
    """The eval host's prefetch cache, or ``None`` (creating nothing on disk)
    unless lane D is active at ``boundary`` (§3.2 invariant 1)."""
    if lane_d_cfg is None or not lane_d_cfg.active(boundary):
        return None
    return PrefetchCache(root, budget_bytes=budget_bytes, king_d=king_d, house=house)


__all__ = [
    "CACHE_CODES", "CacheError", "PrefetchCache", "open_prefetch_cache", "prefetch_grant",
    "tree_digest",
]
