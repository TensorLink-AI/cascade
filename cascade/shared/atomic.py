"""Crash-safe state files shared by the validator and the trainer.

``write_text_atomic`` is tmp + fsync + rename (+ directory fsync): a crash
leaves the old file or the new one, never a torn one. With ``backup=True`` the
previous good file is kept as ``<name>.bak`` first, and ``read_text_with_backup``
falls back to it when the main file is missing or does not parse.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import threading
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

T = TypeVar("T")


def acquire_state_lock(path: str | os.PathLike):
    """Hold the returned file open for the lifetime of a single state writer."""
    import fcntl

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lock = open(p, "a+")  # noqa: SIM115 — caller holds it for the service lifetime
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as e:
        lock.close()
        raise RuntimeError(f"another validator already holds state lock {p}") from e
    lock.seek(0)
    lock.truncate()
    lock.write(f"{os.getpid()}\n")
    lock.flush()
    return lock


_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def path_lock(path: str | os.PathLike) -> threading.Lock:
    """One process-wide lock per resolved path, for read-modify-write callers."""
    key = str(Path(path).resolve())
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(key, threading.Lock())


def _fsync_dir(d: Path) -> None:
    fd = os.open(str(d), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_text_atomic(path: str | os.PathLike, text: str, *, backup: bool = False) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f"{p.name}.tmp.{os.getpid()}.{threading.get_ident()}")
    with open(tmp, "wb") as f:
        f.write(text.encode("utf-8"))
        f.flush()
        os.fsync(f.fileno())
    if backup and p.is_file():
        # copy (never move) the last good file, so ``path`` is never absent
        with contextlib.suppress(OSError):
            shutil.copy2(p, p.with_name(p.name + ".bak"))
    os.replace(tmp, p)
    _fsync_dir(p.parent)


def read_text_with_backup(path: str | os.PathLike, parse: Callable[[str], T]) -> tuple[T | None, str]:
    """``(parse(main), "main")``, else ``(parse(bak), "backup")``, else
    ``(None, "absent")`` when neither file exists, else ``(None, "unreadable")``."""
    p = Path(path)
    bak = p.with_name(p.name + ".bak")
    seen = False
    for src, tag in ((p, "main"), (bak, "backup")):
        if not src.is_file():
            continue
        seen = True
        try:
            return parse(src.read_text(encoding="utf-8")), tag
        except Exception:  # noqa: BLE001 — torn / foreign file: try the next
            continue
    return None, ("unreadable" if seen else "absent")
