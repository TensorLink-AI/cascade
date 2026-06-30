"""Mechanical source-similarity for the copy-of-king screen — no LLM.

The LLM is *not* the detector here. Two generators that are byte-identical are
already dropped in :func:`cascade.trainer.loop.plan_round` (same ``repo@digest``).
This catches the next tier: a challenger that copied the published king and
re-laid-out the source (renamed nothing structural, just reflowed comments and
whitespace) to dodge the digest check.

We normalise both generators — strip comments and docstrings, canonicalise
whitespace and formatting by round-tripping through the AST — then compute a
similarity ratio on the canonical text. The trainer rejects mechanically above a
high threshold (this is the cascade analogue of tau's copy thresholds) and only
escalates the *middle band* — similar-but-restructured, the cases a ratio can't
call confidently — to the LLM, feeding it both generators.

Only relevant when the king's source is published. Skip the whole screen in a
private-only deployment (``[judge] publish_king = false``).
"""

from __future__ import annotations

import ast
import difflib


def canonical_source(source: str) -> str:
    """Comment/whitespace-insensitive canonical form of a Python source string.

    Round-tripping through :func:`ast.parse` → :func:`ast.unparse` drops every
    comment and all original spacing/line-breaks, then re-emits with one fixed
    style — so two files that differ only in formatting collapse to the same
    text. Module/class/function docstrings (bare string-literal statements) are
    stripped too, since re-wording a docstring is the cheapest possible "edit".

    A file that does not parse falls back to a whitespace-collapsed form so the
    similarity check still produces a usable number rather than raising.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return " ".join(source.split())
    _strip_docstrings(tree)
    try:
        return ast.unparse(tree)
    except (AttributeError, ValueError):  # pragma: no cover - unparse is total on parsed trees
        return " ".join(source.split())


def _strip_docstrings(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            node.body = body[1:] or [ast.Pass()]


def source_similarity(a: str, b: str) -> float:
    """Similarity in ``[0.0, 1.0]`` between two Python sources after canonicalising.

    ``1.0`` = identical once comments/whitespace are normalised away; ``0.0`` =
    nothing in common. Uses :class:`difflib.SequenceMatcher` (``autojunk=False``
    so long files aren't silently down-weighted) on the canonical text.
    """
    ca, cb = canonical_source(a), canonical_source(b)
    if not ca and not cb:
        return 1.0
    if not ca or not cb:
        return 0.0
    return difflib.SequenceMatcher(None, ca, cb, autojunk=False).ratio()
