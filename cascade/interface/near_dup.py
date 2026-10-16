"""Near-duplicate signatures for the admission guard (DEC-CA-0048 amendment).

The exact-identity tiers (DEC-CA-0008, :mod:`cascade.interface.dedup`) miss a
variant that changes a few dozen lines of a twenty-thousand-line generator:
every digest moves. This module gives a cheap, text-only similarity instead —
the set of hashed k-line shingles of the normalised ``.py`` source:

* docstrings and full-line / trailing comments are dropped,
* numeric literals are masked (a constant sweep is the same code),
* whitespace is collapsed and blank lines skipped.

Generator code is UNTRUSTED: it is read as text and never imported, compiled
or executed here. The signature is a sorted list of 64-bit ints, a few hundred
KB for the largest observed tree, cached per ref on the trainer box.

Similarity is information, not proof (DEC-CA-0008): the caller decides what a
score means (shadow log vs enforce), and subtracts PUBLIC material first so
building on a published king — the design — never reads as a copy.
"""
from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from pathlib import Path

SHINGLE_LINES = 3
DEFAULT_MAX_TEXT_MB = 4.0

_DOCSTRING = re.compile(r'("""|\'\'\')(?:.|\n)*?\1')
_NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d*)?(?:[eE][-+]?\d+)?|(?<![\w.])\.\d+(?:[eE][-+]?\d+)?")
_SPACE = re.compile(r"\s+")


def normalized_lines(src: str) -> list[str]:
    """The source's lines with docstrings, comments, numeric literals and
    whitespace normalised away (blank lines dropped). Comment stripping is
    deliberately conservative: a ``#`` on a line that also carries a quote is
    kept (it may be inside a string) — stable beats clever for a fingerprint."""
    src = _DOCSTRING.sub("", src)
    out: list[str] = []
    for line in src.splitlines():
        if "#" in line and "'" not in line and '"' not in line:
            line = line.split("#", 1)[0]
        line = _SPACE.sub(" ", _NUMBER.sub("0", line)).strip()
        if line:
            out.append(line)
    return out


def shingle_signature(lines: list[str], k: int = SHINGLE_LINES) -> frozenset[int]:
    """Hashed k-line shingles of ``lines`` (64-bit blake2b)."""
    if not lines:
        return frozenset()
    n = max(1, len(lines) - k + 1)
    return frozenset(
        int.from_bytes(hashlib.blake2b("\n".join(lines[i:i + k]).encode("utf-8"),
                                       digest_size=8).digest(), "big")
        for i in range(n))


def signature_of_sources(sources: Iterable[str], k: int = SHINGLE_LINES) -> frozenset[int]:
    """Signature of the concatenated sources (in the order given)."""
    return shingle_signature(normalized_lines("\n".join(sources)), k)


def signature_of_dir(root: Path, *, max_text_mb: float = DEFAULT_MAX_TEXT_MB,
                     k: int = SHINGLE_LINES) -> frozenset[int]:
    """Signature of every ``.py`` file under ``root`` (sorted by relative
    path), reading at most ``max_text_mb`` of text. Text only."""
    root = Path(root)
    budget = int(max_text_mb * 1024 * 1024)
    sources: list[str] = []
    for p in sorted((p for p in root.rglob("*.py") if p.is_file() and not p.is_symlink()),
                    key=lambda p: p.relative_to(root).as_posix()):
        if budget <= 0:
            break
        raw = p.read_bytes()[:budget]
        budget -= len(raw)
        sources.append(raw.decode("utf-8", "ignore"))
    return signature_of_sources(sources, k)


def private_jaccard(a: frozenset[int], b: frozenset[int], public: frozenset[int], *,
                    min_private: int) -> float | None:
    """Jaccard of the two signatures' PRIVATE parts (``public`` removed);
    ``None`` when either private part is smaller than ``min_private`` (a tree
    that is mostly public lineage has nothing of its own to compare)."""
    pa, pb = a - public, b - public
    if len(pa) < min_private or len(pb) < min_private:
        return None
    union = len(pa | pb)
    return (len(pa & pb) / union) if union else None
