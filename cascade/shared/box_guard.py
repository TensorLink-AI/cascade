"""Runtime box guard: the 7 GB box never imports torch & co. (retro P2, C2).

Owner rule (2026-10-31, restated 2026-11-26 and 2026-11-28): nothing loads a
model, a checkpoint or torch on the trainer/validator box. The import-time
test (``tests/unit/test_box_side_torch_free.py``) only sees module-level
imports of a hand-kept list; a FUNCTION-LOCAL ``import torch`` /
``import timestral`` passes it and then, at call time, either loads GBs into
the validator (OOM 11-28 04:39, 05:41, 10:54) or fails inside a generic
``except`` and becomes a silent ``infra_deferred`` (#1003: the validator box
imported timestral for days).

:func:`install` (called first thing in every box service's ``main()``) makes
that a loud, named error at run time:

* a ``sys.meta_path`` finder refuses the FIRST import of a heavy package
  (``importlib.import_module`` included), and
* a ``builtins.__import__`` wrapper refuses every ``import`` STATEMENT of one
  from box code even after the package is already in ``sys.modules`` (a
  finder is never consulted for a cached module).

The refusal raises :class:`BoxImportRefused` (a ``ModuleNotFoundError``, so a
legitimate ``try: import torch / except ImportError:`` fallback still falls
back) whose message names the service, the package and the calling
``module (file:line)``, and it is ALSO logged at ERROR (once per caller), so a
refusal swallowed by a broad ``except`` is still visible in the service log
and to cascade-ops (``box_guard_refused``).

A package's own internal imports (``torch`` importing ``torch.nn``) are never
refused: only the entry into it. :func:`allow` opens a named, thread-local
window for a box path that truly needs one (today only the checkpoint
guard's pinned-model shape spec, built on the meta device: no weights).

Modes (``CASCADE_BOX_GUARD``): ``enforce`` (refuse + ERROR), ``warn`` (ERROR,
import proceeds: the first-deploy shadow mode), ``off`` (not installed: pods
and dev machines). ``CASCADE_BOX_GUARD_ALLOW=pkg[,pkg]`` exempts packages
explicitly (the owner's escape hatch; logged at start).

Pure stdlib; importing this module imports nothing heavy.
"""

from __future__ import annotations

import builtins
import contextlib
import logging
import os
import sys
import threading
from collections.abc import Iterator
from importlib.machinery import ModuleSpec

log = logging.getLogger(__name__)

# The packages that mean "model work": torch and everything that drags it in.
HEAVY_PACKAGES = frozenset({"torch", "safetensors", "transformers", "triton", "timestral"})
MODES = ("enforce", "warn", "off")
ENV_MODE = "CASCADE_BOX_GUARD"
ENV_ALLOW = "CASCADE_BOX_GUARD_ALLOW"
LOG_TOKEN = "box_guard_refused"


class BoxImportRefused(ModuleNotFoundError):
    """A box service tried to import a heavy package (torch & co.)."""


class _State:
    def __init__(self) -> None:
        self.service = ""
        self.mode = "off"
        self.allow_env: frozenset[str] = frozenset()
        self.local = threading.local()
        self.seen: set[tuple[str, str]] = set()
        self.refusals: list[dict] = []
        self.lock = threading.Lock()
        self.orig_import = None
        self.finder: _Finder | None = None


_S = _State()


def _top(name: str) -> str:
    return name.partition(".")[0]


def _scopes() -> list[tuple[str, str]]:
    st = getattr(_S.local, "scopes", None)
    if st is None:
        st = _S.local.scopes = []
    return st


def _allowed(top: str) -> bool:
    if top in _S.allow_env:
        return True
    return any(top in pkgs.split(",") for pkgs, _why in _scopes())


def _module_of(frame) -> str:
    return str(frame.f_globals.get("__name__", "?"))


def _caller(skip_internal: bool = True) -> tuple[str, str]:
    """``(module, "file:line")`` of the first frame outside the import
    machinery and this module."""
    f = sys._getframe(1)
    while f is not None:
        fn = f.f_code.co_filename
        mod = _module_of(f)
        if skip_internal and (fn.startswith("<frozen importlib") or mod == __name__
                              or mod.startswith("importlib")):
            f = f.f_back
            continue
        return mod, f"{fn.rsplit('/cascade/', 1)[-1]}:{f.f_lineno}"
    return "?", "?"


def _decide(name: str, caller_mod: str, where: str) -> None:
    """Refuse (enforce) or record (warn) an import of ``name`` by
    ``caller_mod``; returns quietly when it is allowed."""
    top = _top(name)
    if top not in HEAVY_PACKAGES or _S.mode == "off" or _allowed(top):
        return
    if _top(caller_mod) == top:          # the package's own internal import
        return
    msg = (f"{LOG_TOKEN}: box service {_S.service or '?'} imported {name!r} from "
           f"{caller_mod} ({where}); the box never runs model code (owner rule "
           f"2026-10-31): move this onto a pod. Exempt explicitly with "
           f"{ENV_ALLOW}={top} (mode {_S.mode})")
    key = (top, where)
    with _S.lock:
        first = key not in _S.seen
        _S.seen.add(key)
        if first:
            _S.refusals.append({"service": _S.service, "module": name, "caller": caller_mod,
                                "where": where, "mode": _S.mode})
    if first:
        log.error("%s", msg)
    if _S.mode == "enforce":
        raise BoxImportRefused(msg, name=name)


class _RefusingLoader:
    def __init__(self, msg: str):
        self.msg = msg

    def create_module(self, spec):
        raise BoxImportRefused(self.msg, name=spec.name)

    def exec_module(self, module):  # pragma: no cover - create_module raises first
        raise BoxImportRefused(self.msg, name=module.__name__)


class _Finder:
    """Refuses the FIRST load of a heavy top-level package. A ``find_spec``
    probe (``importlib.util.find_spec("torch")``) gets a spec back (the
    package is installed: true), only an actual load raises."""

    def find_spec(self, name, path=None, target=None):
        top = _top(name)
        if top not in HEAVY_PACKAGES or name != top or top in sys.modules:
            return None
        mod, where = _caller()
        try:
            _decide(name, mod, where)
        except BoxImportRefused as e:
            return ModuleSpec(name, _RefusingLoader(str(e)))
        return None


def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and _top(name) in HEAVY_PACKAGES and _S.mode != "off":
        caller_mod = str((globals or {}).get("__name__", "?"))
        if _top(caller_mod) != _top(name):
            f = sys._getframe(1)
            where = f"{f.f_code.co_filename.rsplit('/cascade/', 1)[-1]}:{f.f_lineno}"
            _decide(name, caller_mod, where)
    return _S.orig_import(name, globals, locals, fromlist, level)


def install(service: str, *, default_mode: str = "enforce") -> str:
    """Install the guard for box service ``service``; returns the active
    mode. Idempotent (a second call only renames the service / mode).
    ``CASCADE_BOX_GUARD`` overrides ``default_mode``."""
    mode = (os.environ.get(ENV_MODE, "") or default_mode).strip().lower()
    if mode not in MODES:
        log.error("%s=%r is not one of %s; using enforce", ENV_MODE, mode, MODES)
        mode = "enforce"
    allow = frozenset(p.strip() for p in os.environ.get(ENV_ALLOW, "").split(",") if p.strip())
    _S.service, _S.mode, _S.allow_env = str(service), mode, allow
    if mode == "off":
        log.info("box guard off for %s (%s=off)", service, ENV_MODE)
        return mode
    if _S.orig_import is None:
        _S.orig_import = builtins.__import__
        builtins.__import__ = _guarded_import
    if _S.finder is None:
        _S.finder = _Finder()
        sys.meta_path.insert(0, _S.finder)
    loaded = sorted(p for p in HEAVY_PACKAGES if p in sys.modules)
    log.info("box guard %s for %s: refusing %s%s%s", mode, service,
             ",".join(sorted(HEAVY_PACKAGES - allow)),
             f"; explicitly allowed {sorted(allow)}" if allow else "",
             f"; ALREADY LOADED before install: {loaded}" if loaded else "")
    return mode


def uninstall() -> None:
    """Remove the guard (tests)."""
    if _S.orig_import is not None:
        builtins.__import__ = _S.orig_import
        _S.orig_import = None
    if _S.finder is not None:
        with contextlib.suppress(ValueError):
            sys.meta_path.remove(_S.finder)
        _S.finder = None
    _S.mode, _S.service, _S.allow_env = "off", "", frozenset()
    with _S.lock:
        _S.seen.clear()
        _S.refusals.clear()


@contextlib.contextmanager
def allow(*packages: str, why: str) -> Iterator[None]:
    """A named, thread-local window in which ``packages`` may be imported by
    box code. Every use must say ``why``; keep the window to the one call
    that needs it."""
    if not why:
        raise ValueError("box_guard.allow needs a reason")
    st = _scopes()
    st.append((",".join(packages), why))
    try:
        yield
    finally:
        st.pop()


def mode() -> str:
    return _S.mode


def refusals() -> list[dict]:
    """Every distinct refused (or, in warn mode, recorded) import so far."""
    with _S.lock:
        return [dict(r) for r in _S.refusals]


__all__ = ["ENV_ALLOW", "ENV_MODE", "HEAVY_PACKAGES", "LOG_TOKEN", "BoxImportRefused",
           "allow", "install", "mode", "refusals", "uninstall"]
