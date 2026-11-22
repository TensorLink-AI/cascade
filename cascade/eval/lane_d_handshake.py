"""Lane D worker handshake (LANE_D_PLAN W4, §10.4/§10.5, T19/T28).

Before an eval worker's forecasts are used, the validator compares the
worker's code identity with its own: the git ``HEAD`` of the release
directory and a hash of the ``cascade`` package's source bytes. Any
difference — or anything unclear — fails closed with a CRITICAL
``lane_d_handshake_mismatch`` line and nothing is scored, so a stale or
hot-patched eval host can never feed a settlement (T19). The documented
upgrade order (§10.5 S5: eval host first, then the validator) therefore
passes through mismatched steps that score nothing and ends matched (T28).

The git head alone is not enough (an edited checkout keeps its head) and the
package hash alone does not name a release, so both must match and neither
may be empty.
"""

from __future__ import annotations

import hashlib
import logging
import os
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

log = logging.getLogger("cascade.eval.lane_d_handshake")

HANDSHAKE_SCHEMA = "lane-d-handshake/1"
PACKAGE_DIR = "cascade"
HANDSHAKE_FIELDS = ("git_head", "package_hash", "schema")
_GIT_TIMEOUT_S = 30
# Environment that would point git somewhere other than ``root``.
_GIT_ENV_DROP = ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY",
                 "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_CEILING_DIRECTORIES",
                 "GIT_COMMON_DIR", "GIT_NAMESPACE")

T = TypeVar("T")


@dataclass(frozen=True)
class CodeIdentity:
    """What a host runs: the release checkout's ``HEAD`` (``""`` when the
    root is not the top of a git checkout) and :func:`package_hash`."""

    git_head: str
    package_hash: str


class HandshakeMismatch(RuntimeError):
    """The worker's code identity differs from the validator's (or cannot be
    established); ``fields`` is the sorted tuple of mismatching fields."""

    def __init__(self, fields, detail: str = ""):
        self.fields = tuple(sorted(set(fields)))
        msg = f"lane D handshake mismatch on {list(self.fields)}"
        super().__init__(f"{msg}: {detail}" if detail else msg)


def _source_files(pkg: Path) -> list[tuple[str, Path]]:
    out = []
    for dirpath, dirnames, filenames in os.walk(pkg):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        for name in filenames:
            if name.endswith((".pyc", ".pyo")):
                continue
            p = Path(dirpath) / name
            if p.is_file():
                out.append((p.relative_to(pkg).as_posix(), p))
    return sorted(out)


def package_hash(root: str | os.PathLike) -> str:
    """sha256 hex over the source files under ``<root>/cascade/``: their
    relative paths and bytes only (no mtimes, no location, no bytecode)."""
    pkg = Path(root) / PACKAGE_DIR
    if not pkg.is_dir():
        raise FileNotFoundError(f"no {PACKAGE_DIR}/ package under {root}")
    h = hashlib.sha256()
    for rel, p in _source_files(pkg):
        rel_b = rel.encode("utf-8")
        h.update(len(rel_b).to_bytes(8, "big"))
        h.update(rel_b)
        h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def _git(root: Path, *args: str) -> str:
    env = {k: v for k, v in os.environ.items() if k not in _GIT_ENV_DROP}
    proc = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True,
                          env=env, timeout=_GIT_TIMEOUT_S, check=False)
    return proc.stdout.strip() if proc.returncode == 0 else ""


def _git_head(root: Path) -> str:
    """The full ``HEAD`` when ``root`` is itself the top of a git checkout."""
    try:
        top = _git(root, "rev-parse", "--show-toplevel")
        if not top or os.path.realpath(top) != os.path.realpath(root):
            return ""
        head = _git(root, "rev-parse", "--verify", "HEAD^{commit}")
    except (OSError, subprocess.SubprocessError):
        return ""
    if len(head) != 40 or any(c not in "0123456789abcdef" for c in head):
        return ""
    return head


def _default_root() -> Path:
    import cascade

    return Path(os.path.realpath(cascade.__file__)).parent.parent


def local_identity(root: str | os.PathLike | None = None) -> CodeIdentity:
    """This host's :class:`CodeIdentity`; ``root`` defaults to the directory
    containing the imported ``cascade`` package."""
    r = _default_root() if root is None else Path(os.path.realpath(root))
    return CodeIdentity(git_head=_git_head(r), package_hash=package_hash(r))


def worker_hello(root: str | os.PathLike | None = None) -> dict:
    """The eval worker's hello: ``{"schema", "git_head", "package_hash"}``."""
    ident = local_identity(root)
    return {"schema": HANDSHAKE_SCHEMA, "git_head": ident.git_head,
            "package_hash": ident.package_hash}


def _mismatches(hello, expected: CodeIdentity) -> list[str]:
    if not isinstance(hello, dict):
        return list(HANDSHAKE_FIELDS)
    bad = []
    if hello.get("schema") != HANDSHAKE_SCHEMA:
        bad.append("schema")
    for field in ("git_head", "package_hash"):
        got, want = hello.get(field), getattr(expected, field)
        if not isinstance(got, str) or not got or not want or got != want:
            bad.append(field)
    return bad


def _short(value) -> str:
    return value[:12] if isinstance(value, str) and value else repr(value)[:24]


def check_handshake(hello, *, expected: CodeIdentity | None = None) -> CodeIdentity:
    """Compare a worker's hello with ``expected`` (default: this host);
    return the identity on a match, else log CRITICAL and raise
    :class:`HandshakeMismatch`. Fails closed on anything unclear."""
    want = local_identity() if expected is None else expected
    bad = _mismatches(hello, want)
    if bad:
        fields = sorted(bad)
        got = hello if isinstance(hello, dict) else {}
        log.critical("lane_d_handshake_mismatch fields=%s worker_git_head=%s "
                     "worker_package_hash=%s expected_git_head=%s expected_package_hash=%s",
                     ",".join(fields), _short(got.get("git_head")),
                     _short(got.get("package_hash")), _short(want.git_head),
                     _short(want.package_hash))
        raise HandshakeMismatch(fields)
    log.info("lane_d_handshake_ok git_head=%s package_hash=%s", want.git_head,
             want.package_hash[:12])
    return want


def with_handshake(hello, work: Callable[[], T], *,
                   expected: CodeIdentity | None = None) -> T:
    """Run ``work()`` only after :func:`check_handshake` passes; on a
    mismatch nothing runs (T19)."""
    check_handshake(hello, expected=expected)
    return work()


__all__ = [
    "CodeIdentity", "HANDSHAKE_FIELDS", "HANDSHAKE_SCHEMA", "HandshakeMismatch",
    "check_handshake", "local_identity", "package_hash", "with_handshake", "worker_hello",
]
