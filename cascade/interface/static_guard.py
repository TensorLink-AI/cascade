"""AST-level import scan for a submitted generator tree.

Runs before the trainer spawns the sandbox subprocess. Catches lazy attacks (a
generator that opens a socket, shells out, or reaches into the trainer/chain
modules) without paying the cost of starting the sandbox.

Scope: EVERY ``.py`` file under the repo (:func:`scan_tree`), not just
``generator.py`` — a sibling module imported from ``generator.py`` is the same
code — and, inside each file, every long string/bytes constant that decodes to
Python source (plain, base64, zlib, base64+zlib, hex; nested up to
:data:`MAX_PACK_DEPTH`). Packing a module into a string constant and
``exec``-ing it is a common, legitimate submission style, so the packed source
is scanned exactly like a file rather than rejected.

One layer of defense-in-depth: static_guard is cheap and catches obvious cases
at submit time; the trainer's corpus sandbox (network namespace, disk
restrictions, rlimit — see :mod:`cascade.trainer.corpus`) is the backstop for
what the static scan misses (e.g. ``importlib.import_module`` with a computed
argument, or source assembled at runtime).
"""

from __future__ import annotations

import ast
import base64
import binascii
import zlib
from dataclasses import dataclass
from pathlib import Path

# A string constant shorter than this is never a packed module worth decoding.
MIN_PACKED_CHARS = 200
# Packed-inside-packed recursion bound (the observed field packs one or two deep).
MAX_PACK_DEPTH = 4
# Per-file decode budget: every candidate blob (raw + decoded forms) is counted;
# past the budget the remaining constants are left to the sandbox backstop.
MAX_UNPACK_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class GuardResult:
    ok: bool
    blocked_module: str | None = None
    reason: str | None = None
    file: str | None = None


def _module_is_blocked(name: str, blocked: tuple[str, ...]) -> str | None:
    """Return the matching blocked prefix, or None.

    Matches both exact and dotted-prefix forms: ``http.client`` blocks
    ``http.client`` and anything under it; ``socket`` blocks ``socket`` and
    ``socket.foo``.
    """
    for b in blocked:
        if name == b or name.startswith(b + "."):
            return b
    return None


def _decodings(blob: bytes) -> list[bytes]:
    """Every plausible decoding of a packed constant (raw form first)."""
    out = [blob]
    stripped = b"".join(blob.split())
    for fn in (
        lambda b: base64.b64decode(b, validate=True),
        lambda b: zlib.decompress(base64.b64decode(b, validate=True)),
        zlib.decompress,
        lambda b: zlib.decompress(b, -15),
        binascii.unhexlify,
    ):
        try:
            dec = fn(stripped)
        except Exception:  # noqa: BLE001 — not that encoding
            continue
        if dec and dec not in out:
            out.append(dec)
    return out


def _looks_like_python(blob: bytes) -> bool:
    """Cheap pre-filter before the (comparatively) expensive ``ast.parse``."""
    if len(blob) < MIN_PACKED_CHARS:
        return False
    head = blob[: 256 * 1024]
    return b"import " in head or b"def " in head or b"class " in head


def scan_source(
    source: str | bytes,
    blocked: tuple[str, ...],
    *,
    unpack: bool = True,
    _depth: int = 0,
) -> GuardResult:
    """Parse ``source`` as Python and reject if it imports any blocked module.

    Catches ``import X``, ``from X import Y``, ``__import__("X")``, and
    ``importlib.import_module("X")`` with a literal argument, in the source
    itself and in every string constant that decodes to Python (packed
    modules). Dynamic imports with a computed argument are by design left to
    the sandbox backstop.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError) as e:
        return GuardResult(ok=False, reason=f"syntax_error: {e}")

    packed: list[bytes] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                hit = _module_is_blocked(alias.name, blocked)
                if hit is not None:
                    return GuardResult(ok=False, blocked_module=hit, reason="import")
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                # ``from . import x`` — relative import scoped to the cloned
                # repo; every sibling module is scanned by scan_tree.
                continue
            hit = _module_is_blocked(node.module, blocked)
            if hit is not None:
                return GuardResult(ok=False, blocked_module=hit, reason="from_import")
        elif isinstance(node, ast.Call):
            if (
                isinstance(node.func, ast.Name)
                and node.func.id == "__import__"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                hit = _module_is_blocked(node.args[0].value, blocked)
                if hit is not None:
                    return GuardResult(ok=False, blocked_module=hit, reason="__import__")
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "import_module"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                hit = _module_is_blocked(node.args[0].value, blocked)
                if hit is not None:
                    return GuardResult(
                        ok=False, blocked_module=hit, reason="importlib.import_module"
                    )
        elif (
            unpack
            and isinstance(node, ast.Constant)
            and isinstance(node.value, (str, bytes))
            and len(node.value) >= MIN_PACKED_CHARS
        ):
            packed.append(
                node.value.encode("utf-8", "surrogatepass")
                if isinstance(node.value, str) else node.value
            )

    if not unpack or _depth >= MAX_PACK_DEPTH:
        return GuardResult(ok=True)

    budget = MAX_UNPACK_BYTES
    for blob in packed:
        for dec in _decodings(blob):
            budget -= len(dec)
            if budget < 0:
                return GuardResult(ok=True)
            if not _looks_like_python(dec):
                continue
            try:
                inner = ast.parse(dec)
            except (SyntaxError, ValueError):
                continue
            if not inner.body:
                continue
            res = scan_source(dec, blocked, unpack=True, _depth=_depth + 1)
            if not res.ok and res.blocked_module is not None:
                reason = res.reason or ""
                if not reason.startswith("packed["):
                    reason = f"packed[{_depth + 1}]:{reason}"
                return GuardResult(ok=False, blocked_module=res.blocked_module, reason=reason)
    return GuardResult(ok=True)


def scan_file(path: Path | str, blocked: tuple[str, ...]) -> GuardResult:
    p = Path(path)
    if not p.exists():
        return GuardResult(ok=False, reason="missing_file", file=p.name)
    res = scan_source(p.read_bytes(), blocked)
    return GuardResult(ok=res.ok, blocked_module=res.blocked_module,
                       reason=res.reason, file=p.name) if not res.ok else res


def scan_tree(repo_dir: Path | str, blocked: tuple[str, ...]) -> GuardResult:
    """Scan ``generator.py`` and every other ``.py`` under ``repo_dir``.

    ``generator.py`` is scanned first (a missing one is ``missing_file``); the
    first failing file wins and is named in ``GuardResult.file`` (repo-relative).
    Files that do not parse are reported as ``syntax_error`` too: a module the
    interpreter cannot import is still a module the guard cannot vouch for.
    """
    d = Path(repo_dir)
    entry = d / "generator.py"
    if not entry.is_file():
        return GuardResult(ok=False, reason="missing_file", file="generator.py")
    others = sorted(p for p in d.rglob("*.py") if p.is_file() and p != entry)
    for p in (entry, *others):
        res = scan_source(p.read_bytes(), blocked)
        if not res.ok:
            return GuardResult(ok=False, blocked_module=res.blocked_module,
                               reason=res.reason, file=p.relative_to(d).as_posix())
    return GuardResult(ok=True)
