"""Static contamination scan of a submitted lane B generator (LANE_D_PLAN
§12b.1 LB1, §3.0 WB13; RV6 B6 / B7).

:func:`scan_repo` matches every numeric sequence a submitted tree carries
against the WB11 hash set (:mod:`cascade.eval.contamination`) and returns
the receipt field ``{hashset_version, result, suite, window_id}``. It runs at
intake, before any compute, and it NEVER executes the tree: ``.py`` files are
parsed with :mod:`ast`, ``.json`` files with :mod:`json`, ``.npy`` headers
with :func:`ast.literal_eval` (never unpickled), zip archives with
:mod:`zipfile`, and everything else is read as bytes. Symlinks are not
followed, so nothing outside the tree is read.

Every file is first classified, so a text file is never decoded as every
binary layout:

- ``.py`` (when it parses): every number constant of the file in source
  order, a unary minus or plus folded into its number (so a list literal, a
  window split across nested lists and a window spread over scalar
  assignments are all one sequence); then, in source order, every ``str``
  constant as text and every ``bytes`` constant as file content (below).
- ``.json`` (when it parses): every number of the document in document
  order, nested arrays and objects included; then every string value as text.
- a zip archive (``.npz``, ``np.savez`` stored or ``np.savez_compressed``
  deflated): each member, inflated under a size cap, as file content (no
  nested archives).
- a ``.npy`` array (also a zip member): the header is parsed (``descr``,
  ``shape``) and the data decoded as that numeric dtype. An object or
  structured dtype is skipped, never unpickled.
- text (UTF-8 without NUL or other control bytes): the numeric tokens, split
  on whitespace, ``,`` and ``;``. A non-numeric token ends a run, so a
  one-column table, a one-line CSV row after a label and a whitespace row
  are each one sequence.
- anything else (binary): float64 at every byte offset 0–7 in both byte
  orders, float32 at offset 0 in both byte orders, then little-endian int64
  and int32 at offset 0.

Files are visited in sorted relative-path order and the first sequence that
meets the hit rule names the receipt's ``suite`` / ``window_id``, so the
receipt depends only on the tree's contents (not on write order or
``PYTHONHASHSEED``).

The hit rule is WB10's: ``exact_hits >= min_hits["exact"]`` or
``normalised_hits >= min_hits["normalised"]`` per sequence, with the WB11
lookup (:func:`~cascade.eval.contamination.match_series`) doing the
matching one sub-window at a time (memory stays O(window)).

The scan is bounded (RV6 B6). It matches at most ``max_windows``
(:data:`MAX_SCAN_WINDOWS`) sub-windows over every sequence and decoding,
runs at most ``deadline_s`` (:data:`SCAN_DEADLINE_S`) seconds, checked
before every sub-window, and refuses inputs that would need unbounded memory
(an inflated archive member over :data:`MAX_INFLATED_MEMBER_BYTES`, an
archive over :data:`MAX_INFLATED_ARCHIVE_BYTES` or
:data:`MAX_ARCHIVE_MEMBERS`, an archive member it cannot read (encrypted,
corrupt or an unsupported method), a file over :data:`MAX_FILE_BYTES`, a
``.py`` source over :data:`MAX_PARSE_BYTES`). Any of these gives the result
``budget_exceeded``: the intake refuses it (413), it never passes as clean.
The returned :class:`ScanReceipt` records why the scan stopped
(``stopped_by``, internal, not in the receipt): only a deadline overrun
depends on load, so the intake never caches it (RV7 note 9).

:func:`load_pinned_hashset` maps the ``[lane_b] contamination_hashset`` pin
to the ``benchmarks/contamination/hashset-v<n>.json`` file it names, found by
a streaming sha256 of the raw bytes, loads only that file with WB11's bounded
``load_lookup`` (RV7 B3) and returns it as a :class:`PinnedHashset` carrying
the pin; it fails closed when no artefact matches.
"""

from __future__ import annotations

import ast
import collections
import gzip
import hashlib
import io
import json
import math
import os
import re
import struct
import time
import zipfile
import zlib
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path

RESULT_CLEAN = "clean"
RESULT_EMBEDDED = "embedded"
RESULT_BUDGET_EXCEEDED = "budget_exceeded"
HASHSET_DIR = Path("benchmarks") / "contamination"
_HASHSET_NAME = re.compile(r"^hashset-v([1-9][0-9]*)\.json(\.gz)?$")

# RV6 B6: the scan budget. Both are read when a scan runs (``None`` keyword).
# A random sub-window costs ~0.2 ms and a crafted near-edge one ~21 ms, so the
# deadline is the binding bound on adversarial input and the window budget
# the bound on bulk data (~3 windows per byte of a binary file).
MAX_SCAN_WINDOWS = 150_000
SCAN_DEADLINE_S = 20.0
# Memory bounds: an archive member is inflated up to this many bytes (more
# is ``budget_exceeded``, never skipped), an archive up to this many in all.
MAX_INFLATED_MEMBER_BYTES = 8 * 1024 * 1024
MAX_INFLATED_ARCHIVE_BYTES = 32 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 1024
# A single file is read whole; a ``.py`` source is parsed into an AST (about
# 100x its size in memory), so it has a tighter cap.
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_PARSE_BYTES = 2 * 1024 * 1024
_MAX_NPY_HEADER = 64 * 1024

# Binary decodings, in the order they are tried: float64 at every byte
# offset in both byte orders, float32 in both byte orders, then the integer
# widths (LB1 "decoded as float32/float64/int arrays").
_BINARY_DECODINGS = (tuple(("<", "d", 8, off) for off in range(8))
                     + tuple((">", "d", 8, off) for off in range(8))
                     + (("<", "f", 4, 0), (">", "f", 4, 0), ("<", "q", 8, 0),
                        ("<", "i", 4, 0)))
# ``.npy`` dtypes decoded (kind, itemsize) -> struct code.
_NPY_CODES = {("f", 4): "f", ("f", 8): "d", ("i", 1): "b", ("i", 2): "h", ("i", 4): "i",
              ("i", 8): "q", ("u", 1): "B", ("u", 2): "H", ("u", 4): "I", ("u", 8): "Q"}
_NPY_DESCR = re.compile(r"^([<>=|])([fiu])([1248])$")
_NPY_MAGIC = b"\x93NUMPY"
_ZIP_MAGIC = b"PK\x03\x04"
# Text tokens: runs of anything but whitespace, ``,`` and ``;``; brackets and
# quotes around a token are stripped before it is read as a number.
_TOKEN = re.compile(r"[^\s,;]+")
_TOKEN_STRIP = "[](){}\"'"
# Bytes that make a buffer binary even when it decodes as UTF-8.
_CONTROL = re.compile(rb"[\x00-\x08\x0e-\x1f\x7f]")


def _wb11():
    from cascade.eval import contamination

    return contamination


class _OverBudget(Exception):
    """The scan ran past its window budget, its deadline or a memory cap.
    ``stop`` is the :class:`ScanReceipt` ``stopped_by`` value it gives."""

    def __init__(self, message: str, stop: str = "") -> None:
        super().__init__(message)
        self.stop = stop or STOP_CAP


# Why a scan stopped (RV7 note 9), on :attr:`ScanReceipt.stopped_by`. Only a
# deadline overrun depends on load rather than on the tree's content.
STOP_CAP = "cap"
STOP_DEADLINE = "deadline"
STOP_WINDOWS = "windows"


class ScanReceipt(dict):
    """The receipt field :func:`scan_repo` returns: a plain ``dict`` of
    exactly ``{hashset_version, result, suite, window_id}`` (equal to, and
    serialised as, that dict), plus the internal :attr:`stopped_by`, which is
    never part of the receipt: for ``budget_exceeded`` it is
    :data:`STOP_DEADLINE` (the scan deadline, a property of load),
    :data:`STOP_WINDOWS` (the window budget) or :data:`STOP_CAP` (a memory
    cap), both properties of the tree; ``None`` for ``clean`` / ``embedded``.
    The intake caches a receipt only when its result is content-determined
    (:attr:`load_dependent` is false)."""

    stopped_by: str | None = None

    @property
    def load_dependent(self) -> bool:
        return self.stopped_by == STOP_DEADLINE


class _Budget:
    """Windows matched so far against ``max_windows``, and the deadline."""

    def __init__(self, max_windows: int, deadline_s: float) -> None:
        self.max_windows = int(max_windows)
        self.end = time.monotonic() + float(deadline_s)
        self.used = 0

    def charge(self, n: int) -> None:
        self.used += n
        if self.used > self.max_windows:
            raise _OverBudget(f"more than {self.max_windows} sub-windows", STOP_WINDOWS)

    def tick(self) -> None:
        if time.monotonic() > self.end:
            raise _OverBudget("past the scan deadline", STOP_DEADLINE)


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


class _Matcher:
    """Matches sequences against the artefact one sub-window at a time,
    charging the budget per window and checking the deadline before each."""

    def __init__(self, artefact: Mapping, thresholds: dict[str, int], budget: _Budget) -> None:
        self.artefact = artefact
        self.window = int(artefact["window"])
        self.thresholds = thresholds
        self.budget = budget
        self.match_series = _wb11().match_series

    def match(self, values: Iterable, *, precharged: bool = False) -> dict | None:
        """The WB11 ``first`` hit of one sequence once it meets the hit rule,
        else ``None``. ``precharged``: the caller already charged this
        sequence's windows (a binary decoding of known length)."""
        window, budget = self.window, self.budget
        buf: collections.deque = collections.deque(maxlen=window)
        exact = normalised = 0
        first = None
        for v in values:
            buf.append(v)
            if len(buf) < window:
                continue
            if not precharged:
                budget.charge(1)
            budget.tick()
            r = self.match_series(self.artefact, list(buf))
            if r["first"] is None:
                continue
            exact += r["exact_hits"]
            normalised += r["normalised_hits"]
            if first is None:
                first = r["first"]
            if (exact >= self.thresholds["exact"]
                    or normalised >= self.thresholds["normalised"]):
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


def _py_constants(source: bytes) -> tuple[list, list] | None:
    """``(numbers, strings)`` of a Python source: every number constant in
    source order (unary minus / plus folded in), and every ``str`` / ``bytes``
    constant in source order. ``None`` when it does not parse."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return None
    found: list[tuple[int, int, float | int]] = []
    strings: list[tuple[int, int, str | bytes]] = []
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
            if isinstance(node.value, (str, bytes)):
                strings.append((node.lineno, node.col_offset, node.value))
                continue
            v = _number(node.value)
            if v is not None:
                found.append((node.lineno, node.col_offset, v))
    found.sort(key=lambda t: (t[0], t[1]))
    strings.sort(key=lambda t: (t[0], t[1]))
    return [v for _, _, v in found], [s for _, _, s in strings]


def _json_constants(data: bytes) -> tuple[list, list] | None:
    """``(numbers, strings)`` of a JSON document in document order, or
    ``None`` when it does not parse."""
    try:
        doc = json.loads(data.decode("utf-8"), parse_constant=lambda _c: math.nan)
    except (UnicodeDecodeError, ValueError, RecursionError):
        return None
    out: list = []
    strings: list[str] = []
    stack = [doc]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            stack.extend(reversed(list(node.values())))
        elif isinstance(node, list):
            stack.extend(reversed(node))
        elif isinstance(node, str):
            strings.append(node)
        else:
            v = _number(node)
            if v is not None:
                out.append(v)
    return out, strings


def _text_runs(text: str) -> Iterator[Iterator[float]]:
    """The runs of numeric tokens of a text, each lazily: tokens are split on
    whitespace, ``,`` and ``;`` and a non-numeric token ends a run."""
    tokens = _TOKEN.finditer(text)
    pending: list[float] = []

    def run() -> Iterator[float]:
        yield from pending
        pending.clear()
        for m in tokens:
            try:
                yield float(m.group().strip(_TOKEN_STRIP))
            except ValueError:
                return

    for m in tokens:
        try:
            pending.append(float(m.group().strip(_TOKEN_STRIP)))
        except ValueError:
            continue
        yield run()


def _as_text(data: bytes) -> str | None:
    """``data`` as text when it is UTF-8 without NUL or control bytes, else
    ``None`` (binary)."""
    if _CONTROL.search(data):
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _decoded(data: bytes, order: str, code: str, width: int, offset: int) -> Iterator:
    """``data[offset:]`` as ``order``/``code`` values (a trailing partial
    value is ignored), lazily."""
    n = (len(data) - offset) // width
    if n <= 0:
        return iter(())
    view = memoryview(data)[offset:offset + n * width]
    return (t[0] for t in struct.iter_unpack(order + code, view))


def _npy(data: bytes) -> tuple[str, str, int, int] | None:
    """``(byte order, struct code, itemsize, data offset)`` of a ``.npy``
    buffer with a plain numeric dtype; ``None`` for anything else (an object
    or structured dtype is never unpickled, just skipped). The header is read
    with :func:`ast.literal_eval`, which evaluates literals only."""
    if not data.startswith(_NPY_MAGIC) or len(data) < 10:
        return None
    major = data[6]
    if major == 1:
        hlen, start = struct.unpack_from("<H", data, 8)[0], 10
    elif major in (2, 3) and len(data) >= 12:
        hlen, start = struct.unpack_from("<I", data, 8)[0], 12
    else:
        return None
    if hlen > _MAX_NPY_HEADER or start + hlen > len(data):
        return None
    raw = data[start:start + hlen]
    try:
        header = ast.literal_eval(raw.decode("utf-8" if major == 3 else "latin1"))
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError,
            UnicodeDecodeError):
        return None
    if not isinstance(header, dict):
        return None
    descr = header.get("descr")
    m = _NPY_DESCR.match(descr) if isinstance(descr, str) else None
    if m is None:
        return None
    order, kind, size = m.group(1), m.group(2), int(m.group(3))
    code = _NPY_CODES.get((kind, size))
    if code is None:
        return None
    return (">" if order == ">" else "<"), code, size, start + hlen


def _archive_members(data: bytes) -> Iterator[bytes] | None:
    """The inflated members of a zip archive (sorted by name), each capped;
    ``None`` when ``data`` is not a readable zip."""
    if not data.startswith(_ZIP_MAGIC):
        return None
    try:
        z = zipfile.ZipFile(io.BytesIO(data))
        infos = sorted((i for i in z.infolist() if not i.is_dir()),
                       key=lambda i: i.filename)
    except (zipfile.BadZipFile, zipfile.LargeZipFile, OSError, ValueError, EOFError):
        return None
    if len(infos) > MAX_ARCHIVE_MEMBERS:
        raise _OverBudget(f"archive has more than {MAX_ARCHIVE_MEMBERS} members")

    def members() -> Iterator[bytes]:
        total = 0
        for info in infos:
            try:
                with z.open(info) as f:
                    blob = f.read(MAX_INFLATED_MEMBER_BYTES + 1)
            # RuntimeError: an encrypted member or an unsupported compression
            # method (zipfile raises a RuntimeError subclass for it). A member
            # the scan cannot read could hide anything: it fails closed.
            except (zipfile.BadZipFile, RuntimeError, OSError, EOFError, ValueError,
                    zlib.error) as e:
                raise _OverBudget(f"unreadable archive member {info.filename!r}: {e}") from None
            if len(blob) > MAX_INFLATED_MEMBER_BYTES:
                raise _OverBudget(f"archive member inflates past "
                                  f"{MAX_INFLATED_MEMBER_BYTES} bytes")
            total += len(blob)
            if total > MAX_INFLATED_ARCHIVE_BYTES:
                raise _OverBudget(f"archive inflates past {MAX_INFLATED_ARCHIVE_BYTES} bytes")
            yield blob

    return members()


def _content_sequences(data: bytes, matcher: _Matcher, *,
                       archives: bool = True) -> Iterator[tuple[Iterator, bool]]:
    """``(values, precharged)`` sequences of an opaque buffer (a non-code
    file, a ``bytes`` constant or an archive member), classified first."""
    window = matcher.window
    if archives:
        members = _archive_members(data)
        if members is not None:
            for blob in members:
                yield from _content_sequences(blob, matcher, archives=False)
            return
    npy = _npy(data)
    if npy is not None:
        order, code, width, offset = npy
        n = (len(data) - offset) // width
        if n >= window:
            matcher.budget.charge(n - window + 1)
            yield _decoded(data, order, code, width, offset), True
        return
    text = _as_text(data)
    if text is not None:
        for run in _text_runs(text):
            yield run, False
        return
    for order, code, width, offset in _BINARY_DECODINGS:
        n = (len(data) - offset) // width
        if n < window:
            continue
        matcher.budget.charge(n - window + 1)
        yield _decoded(data, order, code, width, offset), True


def _string_sequences(strings: list, matcher: _Matcher) -> Iterator[tuple[Iterator, bool]]:
    for s in strings:
        if isinstance(s, bytes):
            yield from _content_sequences(s, matcher, archives=False)
        else:
            for run in _text_runs(s):
                yield run, False


def _sequences(path: Path, data: bytes, matcher: _Matcher) -> Iterator[tuple[Iterator, bool]]:
    """The sequences of one file, in a fixed order."""
    suffix = path.suffix.lower()
    if suffix == ".py":
        if len(data) > MAX_PARSE_BYTES:
            raise _OverBudget(f"{path.name}: a .py source over {MAX_PARSE_BYTES} bytes")
        parsed = _py_constants(data)
        if parsed is not None:
            yield iter(parsed[0]), False
            yield from _string_sequences(parsed[1], matcher)
            return
    elif suffix == ".json":
        parsed = _json_constants(data)
        if parsed is not None:
            yield iter(parsed[0]), False
            yield from _string_sequences(parsed[1], matcher)
            return
    yield from _content_sequences(data, matcher)


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


# ── public API ───────────────────────────────────────────────────────────────


def _receipt(version: int, result: str, first: Mapping | None = None, *,
             stopped_by: str | None = None) -> ScanReceipt:
    r = ScanReceipt(hashset_version=version, result=result,
                    suite=first["suite"] if first is not None else None,
                    window_id=first["window_id"] if first is not None else None)
    r.stopped_by = stopped_by
    return r


def scan_repo(repo_dir: str | os.PathLike, artefact: Mapping, *,
              min_hits: object = None, max_windows: int | None = None,
              deadline_s: float | None = None) -> ScanReceipt:
    """The receipt field ``{hashset_version, result, suite, window_id}`` of
    the tree at ``repo_dir`` against the WB11 hash-set ``artefact``.

    ``result`` is ``"embedded"`` when any extracted sequence meets the hit
    rule (``min_hits``: anything ``dict()`` accepts, e.g.
    ``LaneBConfig.contamination_min_hits``; ``None`` is the WB10 default),
    with ``suite`` / ``window_id`` from that sequence's first hit;
    ``"budget_exceeded"`` (both ``None``) when the scan matches more than
    ``max_windows`` sub-windows, runs past ``deadline_s`` seconds or meets a
    memory cap before it finds one (``None``: :data:`MAX_SCAN_WINDOWS` /
    :data:`SCAN_DEADLINE_S`, read now); else ``"clean"`` with both ``None``.
    The tree is read, never executed, and ``artefact`` (the dict, or the
    bounded lookup :func:`load_pinned_hashset` returns) is not mutated. The
    return is a :class:`ScanReceipt`: the receipt dict, whose internal
    ``stopped_by`` says why a ``budget_exceeded`` scan stopped.
    Raises ``NotADirectoryError`` when ``repo_dir`` is not a directory."""
    root = Path(repo_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"not a directory: {root}")
    thresholds = _thresholds(artefact, min_hits)
    version = int(artefact["version"])
    budget = _Budget(MAX_SCAN_WINDOWS if max_windows is None else max_windows,
                     SCAN_DEADLINE_S if deadline_s is None else deadline_s)
    matcher = _Matcher(artefact, thresholds, budget)
    try:
        for _rel, path in _files(root):
            budget.tick()
            if path.stat().st_size > MAX_FILE_BYTES:
                raise _OverBudget(f"{path.name}: a file over {MAX_FILE_BYTES} bytes")
            data = path.read_bytes()
            for values, precharged in _sequences(path, data, matcher):
                first = matcher.match(values, precharged=precharged)
                if first is not None:
                    return _receipt(version, RESULT_EMBEDDED, first)
    except _OverBudget as e:
        return _receipt(version, RESULT_BUDGET_EXCEEDED, stopped_by=e.stop)
    return _receipt(version, RESULT_CLEAN)


class PinnedHashset(Mapping):
    """The bounded WB11 lookup (:func:`~cascade.eval.contamination.load_lookup`)
    of a pinned hash-set file, carrying the ``pin`` it was verified against.

    A read-only ``Mapping`` that answers exactly like the lookup it wraps
    (header keys, the ``exact`` / ``normalised`` hash tables), so
    ``match_series`` and :func:`scan_repo` take it like the dict artefact.
    The intake reads :attr:`pin` instead of serialising the hash set."""

    __slots__ = ("_lookup", "_pin")

    def __init__(self, lookup: Mapping, pin: str) -> None:
        self._lookup = lookup
        self._pin = str(pin)

    @property
    def pin(self) -> str:
        """``sha256:<hex>`` of the file's canonical JSON (WB11
        ``artefact_sha256``), checked by ``load_lookup`` as it was read."""
        return self._pin

    def __getitem__(self, key: str) -> object:
        return self._lookup[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._lookup)

    def __len__(self) -> int:
        return len(self._lookup)


_HASH_CHUNK = 1 << 16


def _raw_sha256(path: Path) -> str | None:
    """``sha256:<hex>`` of the file's raw bytes (decompressed for ``.gz``),
    streamed in fixed chunks: O(1) memory, nothing parsed. ``None`` when it
    cannot be read (or is not gzip although named ``.gz``)."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as raw:
            f = gzip.GzipFile(fileobj=raw, mode="rb") if path.name.endswith(".gz") else raw
            with f:
                while True:
                    chunk = f.read(_HASH_CHUNK)
                    if not chunk:
                        break
                    h.update(chunk)
    except (OSError, EOFError, zlib.error):
        return None
    return "sha256:" + h.hexdigest()


def load_pinned_hashset(pin: str, root: str | os.PathLike) -> PinnedHashset | None:
    """The bounded hash set ``[lane_b] contamination_hashset = pin`` names.

    ``""`` (unpinned) is ``None``. Otherwise the
    ``benchmarks/contamination/hashset-v<n>.json[.gz]`` under ``root`` whose
    canonical JSON hashes to ``pin`` (WB11 ``artefact_sha256``, which is the
    sha256 of the bytes ``write_artefact`` writes) and whose ``version`` is
    its ``n``, loaded with WB11 ``load_lookup(path, expect_sha256=pin)``
    (RV7 B3: about 64 B of heap per window, never the whole dict) and
    returned as a :class:`PinnedHashset` carrying the pin.

    The pinned file is found by a streaming sha256 of each candidate's raw
    (decompressed) bytes, so a non-matching file is never parsed and only
    the matching one is loaded. Raises ``ValueError`` when no artefact
    matches: a file edited after pinning or a ``root`` without artefacts
    never falls back to another version."""
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
        if _raw_sha256(p) != pin:
            continue
        lookup = wb11.load_lookup(p, expect_sha256=pin)
        if lookup["version"] != n:
            raise ValueError(f"{p}: hash set version {lookup['version']} does not match its "
                             f"file name (v{n})")
        return PinnedHashset(lookup, pin)
    raise ValueError(f"[lane_b] contamination_hashset {pin} matches no artefact under "
                     f"{d} (looked at {len(candidates)} hashset-v<n>.json file(s))")


__all__ = ["HASHSET_DIR", "MAX_SCAN_WINDOWS", "RESULT_BUDGET_EXCEEDED", "RESULT_CLEAN",
           "RESULT_EMBEDDED", "SCAN_DEADLINE_S", "STOP_CAP", "STOP_DEADLINE", "STOP_WINDOWS",
           "PinnedHashset", "ScanReceipt", "load_pinned_hashset", "scan_repo"]
