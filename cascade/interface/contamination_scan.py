"""Static contamination scan of a submitted lane B generator (LANE_D_PLAN
§12b.1 LB1, §3.0 WB13).

:func:`scan_repo` matches every numeric sequence a submitted tree carries
against the WB11 hash set (:mod:`cascade.eval.contamination`) and returns
the receipt field ``{hashset_version, result, suite, window_id}``. It runs at
intake, before any compute, and it NEVER executes the tree: ``.py`` files are
parsed with :mod:`ast`, ``.json`` files with :mod:`json`, and everything else
is read as bytes. Symlinks are not followed, so nothing outside the tree is
read.

The sequences, per file, in this order:

- ``.py`` (when it parses): every number constant of the file in source
  order, a unary minus or plus folded into its number. Each list, tuple or
  ``np.asarray([...])`` literal is a contiguous run of that stream, so a
  literal, a window split across nested lists and a window spread over
  scalar assignments are all one sequence.
- ``.json`` (when it parses): every number of the document in document
  order, nested arrays and objects included.
- any other file (and a ``.py`` / ``.json`` that does not parse): the bytes
  decoded at offset 0 as little-endian float64, float32, int64 and int32
  arrays, then the numbers of a one-column text table (one number per line;
  a non-numeric line, such as a header, starts a new sequence).

Files are visited in sorted relative-path order and the first sequence that
meets the hit rule names the receipt's ``suite`` / ``window_id``, so the
receipt depends only on the tree's contents (not on write order or
``PYTHONHASHSEED``).

The hit rule is WB10's: ``exact_hits >= min_hits["exact"]`` or
``normalised_hits >= min_hits["normalised"]`` per sequence, with the WB11
lookup (:func:`~cascade.eval.contamination.match_series`) doing the
matching. Long sequences are matched in overlapping chunks (``window − 1``
values of overlap, so every window is seen exactly once) to bound memory.

:func:`load_pinned_hashset` maps the ``[lane_b] contamination_hashset`` pin
to the ``benchmarks/contamination/hashset-v<n>.json`` artefact it names, and
fails closed when no artefact matches.
"""

from __future__ import annotations

import ast
import json
import math
import os
import re
import struct
from collections.abc import Iterator, Mapping
from pathlib import Path

RESULT_CLEAN = "clean"
RESULT_EMBEDDED = "embedded"
HASHSET_DIR = Path("benchmarks") / "contamination"
_HASHSET_NAME = re.compile(r"^hashset-v([1-9][0-9]*)\.json(\.gz)?$")

# Values matched per match_series call (memory bound for large data files).
_CHUNK = 1 << 16
# Little-endian decodings of non-code bytes (LB1 "decoded as
# float32/float64/int arrays"), in the order they are tried.
_DECODINGS = (("<d", 8), ("<f", 4), ("<q", 8), ("<i", 4))


def _wb11():
    from cascade.eval import contamination

    return contamination


# ── hit rule ─────────────────────────────────────────────────────────────────


def _thresholds(artefact: Mapping, min_hits: object) -> dict[str, int]:
    """The WB10 hit thresholds ``min_hits`` resolves to. They are validated
    through WB11's :func:`is_contaminated` (on an empty sequence, which
    matches nothing), so a bad ``min_hits`` raises exactly as it would there."""
    from cascade.shared.config import LaneBConfig

    _wb11().is_contaminated(artefact, (), min_hits)
    out = dict(LaneBConfig().contamination_min_hits)
    if min_hits is not None:
        out.update(dict(min_hits))
    return {"exact": int(out["exact"]), "normalised": int(out["normalised"])}


def _chunks(values: list, window: int) -> Iterator[list]:
    """``values`` in pieces of at most ``_CHUNK`` that overlap by
    ``window − 1``: every full window lies in exactly one piece."""
    n = len(values)
    if n < window:
        return
    step = max(_CHUNK - (window - 1), 1)
    start = 0
    while True:
        yield values[start:start + _CHUNK]
        if start + _CHUNK >= n:
            return
        start += step


def _byte_chunks(data: bytes, code: str, width: int, window: int) -> Iterator[list]:
    """``data`` decoded as ``code`` values at offset 0 (a trailing partial
    value is ignored), in the same overlapping pieces as :func:`_chunks`."""
    n = len(data) // width
    if n < window:
        return
    step = max(_CHUNK - (window - 1), 1)
    start = 0
    while True:
        count = min(_CHUNK, n - start)
        yield list(struct.unpack_from(f"{code[0]}{count}{code[1]}", data, start * width))
        if start + _CHUNK >= n:
            return
        start += step


def _match(artefact: Mapping, pieces: Iterator[list], thresholds: dict[str, int]) -> dict | None:
    """The WB11 ``first`` hit of one sequence (given as overlapping pieces)
    when the sequence meets the hit rule, else ``None``."""
    match_series = _wb11().match_series
    exact = normalised = 0
    first = None
    for piece in pieces:
        r = match_series(artefact, piece)
        exact += r["exact_hits"]
        normalised += r["normalised_hits"]
        if first is None and r["first"] is not None:
            first = r["first"]
    if first is not None and (exact >= thresholds["exact"]
                              or normalised >= thresholds["normalised"]):
        return first
    return None


# ── extraction (parse only, never execute) ───────────────────────────────────


def _number(v: object) -> float | int | None:
    """A JSON / AST number as a value ``match_series`` accepts (a number too
    large for a float becomes NaN, which breaks a window like a gap)."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    if isinstance(v, int):
        try:
            float(v)
        except OverflowError:
            return math.nan
    return v


def _py_numbers(source: bytes) -> list | None:
    """Every number constant of a Python source in source order (unary
    minus / plus folded in), or ``None`` when it does not parse."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return None
    found: list[tuple[int, int, float | int]] = []
    folded: set[int] = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd))
                and isinstance(node.operand, ast.Constant)):
            v = _number(node.operand.value)
            if v is None:
                continue
            folded.add(id(node.operand))
            found.append((node.lineno, node.col_offset, -v if isinstance(node.op, ast.USub) else v))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and id(node) not in folded:
            v = _number(node.value)
            if v is not None:
                found.append((node.lineno, node.col_offset, v))
    found.sort(key=lambda t: (t[0], t[1]))
    return [v for _, _, v in found]


def _json_numbers(data: bytes) -> list | None:
    """Every number of a JSON document in document order, or ``None`` when
    it does not parse."""
    try:
        doc = json.loads(data.decode("utf-8"), parse_constant=lambda _c: math.nan)
    except (UnicodeDecodeError, ValueError, RecursionError):
        return None
    out: list = []
    stack = [doc]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            stack.extend(reversed(list(node.values())))
        elif isinstance(node, list):
            stack.extend(reversed(node))
        else:
            v = _number(node)
            if v is not None:
                out.append(v)
    return out


def _text_columns(data: bytes) -> Iterator[list[float]]:
    """The runs of one-number lines of a text file; a line that is not a
    single number ends a run. Nothing for bytes that are not UTF-8."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return
    run: list[float] = []
    for line in text.splitlines():
        token = line.strip().rstrip(",;").strip()
        try:
            run.append(float(token))
        except ValueError:
            if run:
                yield run
            run = []
    if run:
        yield run


def _files(root: Path) -> list[tuple[str, Path]]:
    """``(relative posix path, path)`` of every regular file under ``root``,
    sorted; symlinks (to files or directories) are never followed."""
    out = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = [d for d in dirnames if not os.path.islink(os.path.join(dirpath, d))]
        for name in filenames:
            p = Path(dirpath) / name
            if p.is_symlink() or not p.is_file():
                continue
            out.append((p.relative_to(root).as_posix(), p))
    out.sort(key=lambda t: t[0])
    return out


def _sequences(path: Path, data: bytes, window: int) -> Iterator[Iterator[list]]:
    """The sequences of one file, each as overlapping pieces."""
    suffix = path.suffix.lower()
    if suffix == ".py":
        nums = _py_numbers(data)
        if nums is not None:
            yield _chunks(nums, window)
            return
    elif suffix == ".json":
        nums = _json_numbers(data)
        if nums is not None:
            yield _chunks(nums, window)
            return
    for code, width in _DECODINGS:
        yield _byte_chunks(data, code, width, window)
    for run in _text_columns(data):
        yield _chunks(run, window)


# ── public API ───────────────────────────────────────────────────────────────


def scan_repo(repo_dir: str | os.PathLike, artefact: Mapping, *,
              min_hits: object = None) -> dict:
    """The receipt field ``{hashset_version, result, suite, window_id}`` of
    the tree at ``repo_dir`` against the WB11 hash-set ``artefact``.

    ``result`` is ``"embedded"`` when any extracted sequence meets the hit
    rule (``min_hits``: anything ``dict()`` accepts, e.g.
    ``LaneBConfig.contamination_min_hits``; ``None`` is the WB10 default),
    with ``suite`` / ``window_id`` from that sequence's first hit; else
    ``"clean"`` with both ``None``. The tree is read, never executed, and
    ``artefact`` is not mutated. Raises ``NotADirectoryError`` when
    ``repo_dir`` is not a directory."""
    root = Path(repo_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"not a directory: {root}")
    thresholds = _thresholds(artefact, min_hits)
    window = int(artefact["window"])
    version = int(artefact["version"])
    for _rel, path in _files(root):
        data = path.read_bytes()
        for pieces in _sequences(path, data, window):
            first = _match(artefact, pieces, thresholds)
            if first is not None:
                return {"hashset_version": version, "result": RESULT_EMBEDDED,
                        "suite": first["suite"], "window_id": first["window_id"]}
    return {"hashset_version": version, "result": RESULT_CLEAN, "suite": None,
            "window_id": None}


def load_pinned_hashset(pin: str, root: str | os.PathLike) -> dict | None:
    """The hash-set artefact ``[lane_b] contamination_hashset = pin`` names.

    ``""`` (unpinned) is ``None``. Otherwise the
    ``benchmarks/contamination/hashset-v<n>.json[.gz]`` under ``root`` whose
    WB11 ``artefact_sha256`` equals ``pin`` (and whose ``version`` is its
    ``n``). Raises ``ValueError`` when no artefact matches: a file edited
    after pinning or a ``root`` without artefacts never falls back to
    another version."""
    pin = str(pin or "")
    if not pin:
        return None
    wb11 = _wb11()
    d = Path(root) / HASHSET_DIR
    candidates = []
    if d.is_dir():
        for p in d.iterdir():
            m = _HASHSET_NAME.match(p.name)
            if m and p.is_file() and not p.is_symlink():
                candidates.append((int(m.group(1)), p.name, p))
    for n, _name, p in sorted(candidates):
        try:
            art = wb11.load_artefact(p)
        except (OSError, ValueError):
            continue
        if art.get("version") == n and wb11.artefact_sha256(art) == pin:
            return art
    raise ValueError(f"[lane_b] contamination_hashset {pin} matches no artefact under "
                     f"{d} (looked at {len(candidates)} hashset-v<n>.json file(s))")


__all__ = ["HASHSET_DIR", "RESULT_CLEAN", "RESULT_EMBEDDED", "load_pinned_hashset",
           "scan_repo"]
