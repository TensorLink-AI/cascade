"""AST-level import scan for a submitted ``generator.py``.

Runs before the trainer spawns the sandbox subprocess. Catches lazy attacks (a
generator that opens a socket, shells out, or reaches into the trainer/chain
modules) without paying the cost of starting the sandbox.

One layer of defense-in-depth: static_guard is cheap and catches obvious cases
at submit time; the trainer's corpus sandbox (network namespace, disk
restrictions, rlimit — see :mod:`cascade.trainer.corpus`) is the backstop for
what the static scan misses (e.g. ``importlib.import_module`` with a computed
argument).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GuardResult:
    ok: bool
    blocked_module: str | None = None
    reason: str | None = None


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


def scan_source(source: str, blocked: tuple[str, ...]) -> GuardResult:
    """Parse ``source`` as Python and reject if it imports any blocked module.

    Catches ``import X``, ``from X import Y``, ``__import__("X")``, and
    ``importlib.import_module("X")`` with a literal argument. Dynamic imports
    with a computed argument are by design left to the sandbox backstop.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return GuardResult(ok=False, reason=f"syntax_error: {e}")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                hit = _module_is_blocked(alias.name, blocked)
                if hit is not None:
                    return GuardResult(ok=False, blocked_module=hit, reason="import")
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                # ``from . import x`` — relative import scoped to the cloned
                # repo; the sandbox bounds what it can reach.
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

    return GuardResult(ok=True)


def scan_file(path: Path | str, blocked: tuple[str, ...]) -> GuardResult:
    p = Path(path)
    if not p.exists():
        return GuardResult(ok=False, reason="missing_file")
    return scan_source(p.read_text(encoding="utf-8"), blocked)
