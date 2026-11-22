"""The lane D vault behind one interface (LANE_D_PLAN §1.2 "Storage").

:class:`LaneDStorage` is what intake and ingest program against. Keys are
``/``-separated object names (``lane-d/<hotkey>/<digest>/<file>``); a missing
key raises ``KeyError``. :class:`FilesystemStorage` is the adapter CI and the
walking skeleton use; the private ``cascade-lane-d-vault`` bucket adapter
(W3a) implements the same four methods.
"""

from __future__ import annotations

import abc
import os
from pathlib import Path, PurePosixPath


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


__all__ = ["FilesystemStorage", "LaneDStorage", "check_key"]
