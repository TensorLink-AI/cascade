"""The lane D vault behind one interface (LANE_D_PLAN §1.2 "Storage").

:class:`LaneDStorage` is what intake and ingest program against. Keys are
``/``-separated object names (``lane-d/<hotkey>/<digest>/<file>``); a missing
key raises ``KeyError``. :class:`FilesystemStorage` is the adapter CI and the
walking skeleton use; the private ``cascade-lane-d-vault`` bucket adapter
(W3a) implements the same four methods.

:func:`fetch_presigned` is the eval pod's only way to read the vault
(§1.2: presigned GETs only, never a credential on a pod): one size-capped,
sha256-checked GET of a presigned URL, with no ``Authorization`` header and
no credential read from the environment. :func:`fetch_presigned_bytes` is the
same capped GET into memory for small objects of unknown hash (the manifest).
"""

from __future__ import annotations

import abc
import hashlib
import os
import urllib.error
import urllib.request
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

# FetchError codes (they equal the matching lane D ingest codes).
FETCH_CODES = ("oversize", "size_mismatch", "digest_mismatch", "fetch_failed")
# Seconds any single connect/read may block before the GET fails.
FETCH_TIMEOUT_S = 120.0
_CHUNK = 1 << 20
_SCHEMES = ("http", "https")


class LaneDStorage(abc.ABC):
    """A flat object store addressed by ``/``-separated keys."""

    @abc.abstractmethod
    def put(self, key: str, data: bytes) -> None:
        """Store ``data`` under ``key`` (replacing any previous object)."""

    @abc.abstractmethod
    def get(self, key: str) -> bytes:
        """The object's bytes; ``KeyError`` when ``key`` holds no object."""

    @abc.abstractmethod
    def list(self, prefix: str) -> list[str]:
        """Every key starting with ``prefix``, sorted, as full keys."""

    @abc.abstractmethod
    def size(self, key: str) -> int:
        """The object's size in bytes; ``KeyError`` when absent."""


def check_key(key: str) -> str:
    """A key is a relative ``/``-path with no empty, ``.`` or ``..`` part."""
    if not isinstance(key, str) or not key:
        raise ValueError(f"storage key must be a non-empty string; got {key!r}")
    parts = key.split("/")
    if key.startswith("/") or any(p in ("", ".", "..") for p in parts) or "\\" in key:
        raise ValueError(f"storage key {key!r} is not a clean relative path")
    return key


class FilesystemStorage(LaneDStorage):
    """Objects as files under ``root`` (one file per key)."""

    def __init__(self, root: str | os.PathLike):
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        return self.root.joinpath(*PurePosixPath(check_key(key)).parts)

    def put(self, key: str, data: bytes) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(f".{p.name}.part-{os.getpid()}")
        tmp.write_bytes(bytes(data))
        os.replace(tmp, p)

    def get(self, key: str) -> bytes:
        p = self._path(key)
        if not p.is_file():
            raise KeyError(key)
        return p.read_bytes()

    def list(self, prefix: str) -> list[str]:
        if not self.root.is_dir():
            return []
        out = []
        for p in self.root.rglob("*"):
            if not p.is_file() or p.name.startswith("."):
                continue
            key = p.relative_to(self.root).as_posix()
            if key.startswith(prefix):
                out.append(key)
        return sorted(out)

    def size(self, key: str) -> int:
        p = self._path(key)
        if not p.is_file():
            raise KeyError(key)
        return p.stat().st_size


class FetchError(RuntimeError):
    """A refused presigned GET; ``code`` is one of :data:`FETCH_CODES`."""

    def __init__(self, code: str, detail: str = ""):
        if code not in FETCH_CODES:
            raise ValueError(f"unknown fetch code {code!r}")
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


class _HttpOnlyRedirects(urllib.request.HTTPRedirectHandler):
    """Follows a redirect only to another http(s) URL (urllib would also
    follow ftp)."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urlsplit(newurl).scheme.lower() not in _SCHEMES:
            raise FetchError("fetch_failed", f"redirect to a non-http(s) URL ({code})")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _redact(url: str) -> str:
    """The URL without its query (the presigned signature) for messages."""
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}{parts.path}"


def _open(url: str, timeout_s: float):
    if not isinstance(url, str) or urlsplit(url).scheme.lower() not in _SCHEMES or not urlsplit(url).netloc:
        raise FetchError("fetch_failed", "only http(s) presigned URLs are fetched")
    # A bare request: no Authorization or any other credential header. The
    # presigned query string is the whole authorisation.
    req = urllib.request.Request(url, method="GET", headers={"Accept-Encoding": "identity"})
    opener = urllib.request.build_opener(_HttpOnlyRedirects())
    try:
        return opener.open(req, timeout=timeout_s)
    except urllib.error.HTTPError as e:
        e.close()
        raise FetchError("fetch_failed", f"HTTP {e.code} for {_redact(url)}") from None
    except (OSError, ValueError) as e:  # URLError and timeouts are OSErrors
        raise FetchError("fetch_failed", f"{type(e).__name__} for {_redact(url)}") from None


def _stream(url: str, sink, *, max_size: int, timeout_s: float) -> tuple[str, int]:
    """GET ``url`` into ``sink`` (a callable taking bytes), never reading more
    than ``max_size + 1`` bytes; returns ``(sha256 hex, size)``."""
    resp = _open(url, timeout_s)
    with resp:
        declared = resp.headers.get("Content-Length")
        if declared is not None:
            try:
                n = int(declared)
            except ValueError:
                raise FetchError("fetch_failed", "unparseable Content-Length") from None
            if n > max_size:
                raise FetchError("oversize", f"Content-Length {n} > {max_size}")
        h, got = hashlib.sha256(), 0
        while True:
            try:
                chunk = resp.read(min(_CHUNK, max_size + 1 - got))
            except (OSError, ValueError) as e:
                raise FetchError("fetch_failed", f"{type(e).__name__} reading {_redact(url)}") from None
            if not chunk:
                break
            got += len(chunk)
            if got > max_size:
                raise FetchError("oversize", f"body exceeds {max_size} bytes")
            h.update(chunk)
            sink(chunk)
    return h.hexdigest(), got


def _check_hex(sha256: str) -> str:
    s = str(sha256)
    if len(s) != 64 or any(c not in "0123456789abcdef" for c in s):
        raise ValueError(f"sha256 must be 64 lowercase hex characters; got {sha256!r}")
    return s


def fetch_presigned(url: str, dest: str | os.PathLike, *, sha256: str, size: int,
                    timeout_s: float = FETCH_TIMEOUT_S) -> Path:
    """Stream one presigned GET to ``dest`` (parents created); the body must
    be exactly ``size`` bytes hashing to ``sha256``. Raises :class:`FetchError`
    and then leaves neither ``dest`` nor a partial file behind."""
    want = _check_hex(sha256)
    size = int(size)
    if size < 0:
        raise ValueError(f"size must be >= 0; got {size}")
    target = Path(dest)
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(f".{target.name}.part-{os.getpid()}")
    try:
        with open(part, "wb") as f:
            got_sha, got = _stream(url, f.write, max_size=size, timeout_s=timeout_s)
        if got != size:
            raise FetchError("size_mismatch", f"{got} bytes != {size}")
        if got_sha != want:
            raise FetchError("digest_mismatch", f"sha256 of {_redact(url)} is not the expected one")
        os.replace(part, target)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    return target


def fetch_presigned_bytes(url: str, *, max_size: int, timeout_s: float = FETCH_TIMEOUT_S) -> bytes:
    """One presigned GET of a small object into memory, refused as
    ``oversize`` past ``max_size`` bytes (no hash is known in advance)."""
    buf = bytearray()
    _stream(url, buf.extend, max_size=int(max_size), timeout_s=timeout_s)
    return bytes(buf)


__all__ = ["FETCH_CODES", "FETCH_TIMEOUT_S", "FetchError", "FilesystemStorage", "LaneDStorage",
           "check_key", "fetch_presigned", "fetch_presigned_bytes"]
