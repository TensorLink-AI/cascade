"""Which ``cascade`` is actually running — checked at every service start.

Incident 2026-11-04: the funding intake was launched as ``python -m
cascade.funding.main`` with ``PYTHONPATH=/root/cascade-rc20`` from a cwd that
held a stale copy of the package. ``-m`` puts the cwd FIRST on ``sys.path``, so
the stale copy shadowed the release and the service ran Oct-30 code for a day:
a withdraw-guard fix merged on Nov 3 never ran, and a miner's key was dropped
mid-harvest ($90 lost). Nothing in the logs said which code was loaded.

Every entry point therefore calls :func:`verify_release_code_path` first. It
logs the imported package's location (plus the git head when the release dir
is a checkout) and, when the operator declared where the release lives — the
first ``PYTHONPATH`` entry, or ``CASCADE_RELEASE_PATH`` — refuses to start if
the import resolved anywhere else. ``CASCADE_ALLOW_CODE_PATH_MISMATCH=1``
downgrades the refusal to a CRITICAL log line (dev loops only).
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

log = logging.getLogger("cascade.codepath")

ALLOW_MISMATCH_ENV = "CASCADE_ALLOW_CODE_PATH_MISMATCH"
RELEASE_PATH_ENV = "CASCADE_RELEASE_PATH"
EXIT_CODE_PATH_MISMATCH = 78   # EX_CONFIG


class CodePathMismatch(RuntimeError):
    """The imported ``cascade`` package is not the declared release."""


def imported_package_root(package=None) -> Path:
    """The directory that CONTAINS the imported ``cascade`` package."""
    if package is None:
        import cascade as package  # noqa: PLC0415 — the thing being checked
    return Path(package.__file__).resolve().parent.parent


def declared_release_root(environ=None) -> Path | None:
    """Where the operator says the release lives: ``CASCADE_RELEASE_PATH``,
    else the first ``PYTHONPATH`` entry. None when neither is set (a plain
    venv install: nothing to compare against)."""
    env = os.environ if environ is None else environ
    explicit = (env.get(RELEASE_PATH_ENV) or "").strip()
    if explicit:
        return Path(explicit).resolve()
    pp = (env.get("PYTHONPATH") or "").split(os.pathsep)
    for entry in pp:
        if entry.strip():
            return Path(entry.strip()).resolve()
    return None


def git_head(root: Path) -> str:
    """``<short sha>[+dirty]`` of ``root`` when it is a git checkout/worktree,
    else ``""`` (best-effort, never raises)."""
    try:
        if not (root / ".git").exists():
            return ""
        sha = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5).stdout.strip()
        if not sha:
            return ""
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain",
                                "--untracked-files=no"],
                               capture_output=True, text=True, timeout=5).stdout.strip()
        return sha + ("+dirty" if dirty else "")
    except Exception:  # noqa: BLE001 — a label, never a failure
        return ""


def check_release_code_path(*, package=None, environ=None) -> tuple[Path, Path | None]:
    """Return ``(imported_root, declared_root)``; raise
    :class:`CodePathMismatch` when a release was declared and the import
    resolved elsewhere."""
    imported = imported_package_root(package)
    declared = declared_release_root(environ)
    if declared is not None and imported != declared:
        raise CodePathMismatch(
            f"cascade imported from {imported} but the declared release is {declared} "
            f"(sys.path[0]={sys.path[0]!r}, cwd={os.getcwd()}). A stale copy is "
            f"shadowing the release — launch with `python -P`/`-I`, or from a cwd "
            f"without a `cascade/` dir, or set {RELEASE_PATH_ENV}.")
    return imported, declared


def verify_release_code_path(service: str, *, package=None, environ=None,
                             exit_fn=sys.exit) -> Path:
    """Log the code that runs ``service`` and refuse a shadowed release.

    Call this FIRST in every service ``main()``. Returns the imported root.
    On a mismatch: a CRITICAL line, then ``exit_fn(EXIT_CODE_PATH_MISMATCH)``
    unless ``CASCADE_ALLOW_CODE_PATH_MISMATCH=1``."""
    env = os.environ if environ is None else environ
    try:
        imported, declared = check_release_code_path(package=package, environ=env)
    except CodePathMismatch as e:
        if (env.get(ALLOW_MISMATCH_ENV) or "").strip() in ("1", "true", "yes"):
            log.critical("%s: %s — continuing because %s is set", service, e,
                         ALLOW_MISMATCH_ENV)
            return imported_package_root(package)
        log.critical("%s: REFUSING TO START — %s", service, e)
        exit_fn(EXIT_CODE_PATH_MISMATCH)
        raise  # exit_fn returned (tests): surface it
    head = git_head(imported)
    log.info("%s: code %s%s%s", service, imported, f" @ {head}" if head else "",
             "" if declared is not None else " (no release declared: PYTHONPATH unset)")
    return imported
