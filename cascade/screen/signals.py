"""Mechanical evidence for the distillation judge — no LLM, no miner code run.

The LLM judge catches *blatant* distillation (big literal arrays, base64/
compressed blobs decoded at runtime, libraries that only load pretrained models)
but misses well-laundered weights. To narrow that gap we hand it two cheap,
objective signals computed straight from the repo source:

* **total bytes of numeric literals** — honest priors are mostly *logic* (a
  handful of small constants: periods, scales, lengths); distilled weights are
  mostly *numbers* (long arrays of high-mantissa floats). A repo whose source is
  dominated by numeric literals is suspicious on its face.
* **byte entropy of those literals** — fitted weights look like noise: their
  digit stream is near-uniform (high entropy). Hand-written constants (``7``,
  ``0.5``, ``2.0``) are short and low-entropy.

Both are evidence, not a verdict — they're fed to :mod:`cascade.screen.judge`
alongside the source so the model can weigh them. A string-blob signal
(base64/compressed payloads are *string* literals, not numeric) rides along for
the same reason.
"""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass, field
from pathlib import Path

# Files we scan for literals. The whole submission is code-only and tiny, so we
# walk every Python source file, not just generator.py — a distilled blob can
# hide in a helper module just as easily.
_SOURCE_GLOB = "*.py"


@dataclass(frozen=True)
class NumericEvidence:
    """Objective literal-density signals for one generator repo.

    ``constant_entropy`` is Shannon entropy in **bits per byte** (0..8) over the
    concatenated source text of every numeric literal; ``0.0`` when there are no
    numeric literals at all.
    """

    total_literal_bytes: int
    constant_entropy: float
    numeric_literal_count: int
    largest_numeric_array_len: int
    # Supplementary (string blobs — base64/compressed payloads hide here, not in
    # numbers). Not one of the two required signals, but cheap and useful to the
    # judge, which is told to flag base64/compressed blobs decoded at runtime.
    total_string_literal_bytes: int = 0
    max_string_literal_bytes: int = 0
    files_scanned: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict:
        return {
            "total_numeric_literal_bytes": self.total_literal_bytes,
            "numeric_literal_byte_entropy_bits": round(self.constant_entropy, 4),
            "numeric_literal_count": self.numeric_literal_count,
            "largest_numeric_array_len": self.largest_numeric_array_len,
            "total_string_literal_bytes": self.total_string_literal_bytes,
            "max_string_literal_bytes": self.max_string_literal_bytes,
            "files_scanned": list(self.files_scanned),
        }


def _byte_entropy(data: bytes) -> float:
    """Shannon entropy of ``data`` in bits per byte (0..8). Empty ⇒ 0.0."""
    if not data:
        return 0.0
    counts = [0] * 256
    for b in data:
        counts[b] += 1
    n = len(data)
    h = 0.0
    for c in counts:
        if c:
            p = c / n
            h -= p * math.log2(p)
    return h


def _literal_text(source: str, node: ast.AST) -> str:
    """Source text of a literal node, falling back to ``repr`` of its value.

    ``ast.get_source_segment`` returns the bytes as the miner wrote them
    (``1.4142135623730951``), which is exactly what we want to measure; the
    ``repr`` fallback covers the rare node with no position info.
    """
    seg = ast.get_source_segment(source, node)
    if seg is not None:
        return seg
    value = getattr(node, "value", None)
    return repr(value)


def _scan_source(source: str) -> tuple[bytes, int, int, int, int]:
    """Return ``(numeric_bytes, numeric_count, largest_array_len, str_bytes,
    max_str_bytes)`` for one Python source string. A syntax error contributes
    nothing (the static guard / determinism check reports the real failure)."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return b"", 0, 0, 0, 0

    numeric_chunks: list[bytes] = []
    numeric_count = 0
    str_bytes = 0
    max_str_bytes = 0
    largest_array = 0

    for node in ast.walk(tree):
        # A literal collection of numbers (the classic distillation tell: a big
        # array of weights pasted into the source).
        if isinstance(node, ast.List | ast.Tuple):
            n_numeric = sum(
                1
                for el in node.elts
                if isinstance(el, ast.Constant) and isinstance(el.value, int | float | complex)
            )
            largest_array = max(largest_array, n_numeric)
        if isinstance(node, ast.Constant):
            v = node.value
            if isinstance(v, bool):
                continue  # bool is an int subclass; not a weight
            if isinstance(v, int | float | complex):
                txt = _literal_text(source, node)
                numeric_chunks.append(txt.encode("utf-8", "replace"))
                numeric_count += 1
            elif isinstance(v, str | bytes):
                blob = v.encode("utf-8", "replace") if isinstance(v, str) else v
                str_bytes += len(blob)
                max_str_bytes = max(max_str_bytes, len(blob))

    return b"".join(numeric_chunks), numeric_count, largest_array, str_bytes, max_str_bytes


def numeric_evidence(repo_dir: Path | str) -> NumericEvidence:
    """Compute the literal-density evidence for every ``*.py`` file in the repo.

    Pure and offline: parses source, never imports or runs it.
    """
    d = Path(repo_dir)
    all_numeric = bytearray()
    numeric_count = 0
    largest_array = 0
    total_str = 0
    max_str = 0
    scanned: list[str] = []

    for path in sorted(d.rglob(_SOURCE_GLOB)):
        if not path.is_file():
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        chunk, count, arr, sbytes, smax = _scan_source(source)
        all_numeric += chunk
        numeric_count += count
        largest_array = max(largest_array, arr)
        total_str += sbytes
        max_str = max(max_str, smax)
        scanned.append(str(path.relative_to(d)))

    return NumericEvidence(
        total_literal_bytes=len(all_numeric),
        constant_entropy=_byte_entropy(bytes(all_numeric)),
        numeric_literal_count=numeric_count,
        largest_numeric_array_len=largest_array,
        total_string_literal_bytes=total_str,
        max_string_literal_bytes=max_str,
        files_scanned=tuple(scanned),
    )
