"""Contamination hash set over the public benchmark suites (LANE_D_PLAN §12b.1
LB1, §3.0 WB11).

The hash set holds two hashes of every informative sliding sub-window of
every suite test window:

- **exact**: sha256 of the window's values packed as little-endian IEEE 754
  float64 (``struct.pack("<32d", ...)``);
- **normalised** (``cascade-zgrid/2``): the window z-scored on its own values
  (population std), each z-score put on a fixed-point grid
  (``floor(z / GRID)``, absolute bins of width :data:`GRID`), and the bin
  ints hashed with sha256 under a domain tag.

A **low-information** window (:func:`is_low_information`: fewer than
:data:`MIN_DISTINCT` distinct values, one value filling 3/4 of the window, a
std within 2**-46 of the level, or an exact straight line) is in neither set,
and a low-information submission window is never counted as a hit: zero
runs, one-hot lists, counters and ramps turn up in benign code too, so they
are not evidence of contamination (RV6 B5).

The lookup is tolerant: a submission value whose z-score lies within
:data:`TOLERANCE` of a bin edge also probes the neighbouring bin (the
:data:`MAX_EDGE_VALUES` nearest-to-edge values, so at most
``2 ** MAX_EDGE_VALUES`` probes), for the window and its negation. So a
scaled (any sign), shifted, re-rounded or float32-cast copy of a fragment
still matches whenever every z-score moved by at most :data:`TOLERANCE`.

Each hash maps to ``[suite, "<item id>@<offset>"]``, the smallest
``(suite, item, offset)`` that produces it. The artefact records the suites'
W3b4 identities (:func:`cascade.eval.suite_identity.suite_identity`), is
serialised as canonical JSON (:func:`artefact_bytes`) and pinned by
:func:`artefact_sha256` in ``chain.toml [lane_b] contamination_hashset``.

LB1's intake check (WB13) loads the pinned file with :func:`load_lookup`, a
read-only structure of sorted 128-bit hash prefixes built by stream-parsing
the canonical JSON (at most :data:`LOOKUP_PEAK_BYTES_PER_WINDOW` bytes of heap
per window plus :data:`LOOKUP_FIXED_BYTES`), and calls :func:`match_series` /
:func:`is_contaminated` on it. :func:`load_artefact` returns the full dict
(CLI, evidence check).

The owner builds the real artefact on the bench pod::

    python -P -m cascade.eval.contamination build --input SUITES.json --version N --out PATH
    python -P -m cascade.eval.contamination verify --artefact PATH --sha256 sha256:<hex> [--input SUITES.json]

``SUITES.json`` is ``{"suites": {suite: {"version": str, "series": {item: [float|null]}}}}``.
``verify`` also reports ``n_exact``, ``n_normalised``, ``file_bytes`` and
``lookup_peak_bytes`` (the traced heap peak of :func:`load_lookup`), which the
pin evidence records.
The byte encoding of the normalised hash is fixed by :data:`SCHEMA` version 2;
changing it needs a new schema. Schema 1 (z-scores at 4 significant figures)
missed rounded copies at small series levels and is refused on load.
"""

from __future__ import annotations

import argparse
import copy
import functools
import gzip
import hashlib
import json
import math
import os
import re
import struct
import sys
import tempfile
import tracemalloc
import zlib
from array import array
from bisect import bisect_left
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path

from cascade.eval.suite_identity import KNOWN_SUITES, suite_identity

SCHEMA = "cascade_contamination_hashset/2"
_SCHEMA_V1 = "cascade_contamination_hashset/1"
WINDOW = 32
STRIDE = 1
HASH = "sha256"
# Width of the absolute z-score bins.
GRID = 0.5
# A lookup value within this distance (in z) of a bin edge also probes the
# neighbouring bin. Rounding a copy to 3 dp inside the promised bound moves a
# z-score by at most ~0.02; float32 by ~1e-5.
TOLERANCE = 0.03
# At most this many near-edge values get an alternative bin (the nearest to
# their edge), so a lookup tries at most 2**MAX_EDGE_VALUES probes per
# orientation.
MAX_EDGE_VALUES = 12
NORMALISATION = {"method": "zscore", "scope": "subwindow", "std": "population",
                 "grid": GRID, "tolerance": TOLERANCE, "orientation": "both",
                 "encoding": "cascade-zgrid/2"}

# Low-information windows (RV6 B5). A window is low-information when ANY of:
# (a) it has fewer than MIN_DISTINCT distinct values;
MIN_DISTINCT = 4
# (b) one value fills at least MODE_SHARE[0] / MODE_SHARE[1] of it;
MODE_SHARE = (3, 4)
# (c) its population std is <= STD_REL_FLOOR * |mean| (under 64 float64 ulps
#     of its level; std 0 included);
STD_REL_FLOOR = 2.0 ** -46
# (d) a least-squares line in the index leaves <= AFFINE_RESIDUAL of its
#     variance (range / arange / linspace / exact counters).
AFFINE_RESIDUAL = 1e-6

# Memory budget of load_lookup (RV6 B8): traced heap peak per suite window
# (the larger set's size) plus a fixed allowance, and an absolute cap for the
# 7 GB production box whatever the artefact's size. The committed .json.gz
# stays within FILE_BYTES_PER_WINDOW per window plus FILE_FIXED_BYTES.
LOOKUP_PEAK_BYTES_PER_WINDOW = 64
LOOKUP_FIXED_BYTES = 1 << 20
LOOKUP_CAP_BYTES = 3 << 29
FILE_BYTES_PER_WINDOW = 96
FILE_FIXED_BYTES = 64 << 10

# Domain tag of the normalised preimage: a normalised hash can never equal an
# exact hash of the same window (different preimages).
_NORM_TAG = b"cascade-contamination-zgrid/2\n"
# |z| below this is taken as 0: a value equal to the window mean in exact
# arithmetic comes out as ~1e-16 in floating point, whose sign would otherwise
# depend on the window's scale and shift.
_ZERO_Z = 1e-9

_PIN = re.compile(r"sha256:[0-9a-f]{64}")
# Stream-parser tokens of the canonical artefact JSON (ensure_ascii, no
# whitespace, keys sorted).
_STR = r'"(?:[^"\\\x00-\x1f]|\\.)*"'
_KEY = re.compile("(" + _STR + "):")
_ENTRY = re.compile(r'"([0-9a-f]{64})":\[(' + _STR + "),(" + _STR + r")\]([,}])")
_DECODER = json.JSONDecoder()
_CHUNK = 1 << 16
# A single token (one entry or one header value) longer than this is refused,
# so a malformed file cannot grow the parse buffer without bound.
_MAX_PENDING = 4 << 20
# Unsigned array type of at least 32 bits for item indices and offsets.
_U32 = "I" if array("I").itemsize >= 4 else "L"
_U32_MAX = (1 << (8 * array(_U32).itemsize)) - 1


# ── values and windows ───────────────────────────────────────────────────────


def _as_values(values: object, *, where: str) -> list[float]:
    """Floats of one series; ``None`` and non-finite values become NaN."""
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise TypeError(f"{where}: values must be a sequence of numbers; got {type(values).__name__}")
    out: list[float] = []
    for i, v in enumerate(values):
        if v is None:
            out.append(math.nan)
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise TypeError(f"{where}[{i}]: {v!r} is not a number or null")
        f = float(v)
        out.append(f if math.isfinite(f) else math.nan)
    return out


def _windows(vals: list[float], window: int, stride: int):
    """``(offset, window)`` for every full sub-window with no missing value."""
    for off in range(0, len(vals) - window + 1, stride):
        w = vals[off:off + window]
        if all(math.isfinite(x) for x in w):
            yield off, w


@functools.lru_cache(maxsize=8)
def _index_terms(n: int) -> tuple[tuple[float, ...], float]:
    """Centred index ``t_i = i - (n-1)/2`` of an ``n``-point window and ``sum t_i**2``."""
    t = tuple(i - (n - 1) / 2 for i in range(n))
    return t, math.fsum(ti * ti for ti in t)


def is_low_information(window: Sequence[float]) -> bool:
    """True when the window carries too little shape to be contamination
    evidence: (a) fewer than :data:`MIN_DISTINCT` distinct values, (b) one value
    fills at least 3/4 of it, (c) population std <= :data:`STD_REL_FLOOR` *
    |mean|, or (d) a least-squares line in the index leaves at most
    :data:`AFFINE_RESIDUAL` of its variance. Every rule is invariant under
    ``a*x + b`` with ``a != 0`` ((c) while the shift stays below 2**46 window
    stds)."""
    n = len(window)
    counts = Counter(window)
    if len(counts) < MIN_DISTINCT:
        return True
    if max(counts.values()) * MODE_SHARE[1] >= MODE_SHARE[0] * n:
        return True
    mean = math.fsum(window) / n
    dev = [x - mean for x in window]
    ss_tot = math.fsum(d * d for d in dev)
    std = math.sqrt(ss_tot / n)
    if not std > STD_REL_FLOOR * abs(mean):
        return True
    t, tt = _index_terms(n)
    slope = math.fsum(ti * d for ti, d in zip(t, dev, strict=True)) / tt
    ss_res = math.fsum((d - slope * ti) ** 2 for ti, d in zip(t, dev, strict=True))
    return ss_res <= AFFINE_RESIDUAL * ss_tot


def _informative_windows(vals: list[float], window: int, stride: int):
    """:func:`_windows` without the low-information ones."""
    for off, w in _windows(vals, window, stride):
        if not is_low_information(w):
            yield off, w


def exact_hash(window: list[float]) -> str:
    """sha256 (lower-case hex) of the window as little-endian float64."""
    return hashlib.sha256(struct.pack(f"<{len(window)}d", *window)).hexdigest()


def _zscores(window: Sequence[float]) -> list[float] | None:
    """The window's z-scores (population std); ``None`` for a constant window."""
    if max(window) == min(window):
        return None
    n = len(window)
    mean = math.fsum(window) / n
    dev = [x - mean for x in window]
    std = math.sqrt(math.fsum(d * d for d in dev) / n)
    if not std > 0.0 or not math.isfinite(std):
        return None
    out = []
    for d in dev:
        z = d / std
        out.append(0.0 if abs(z) < _ZERO_Z else z)
    return out


def _bins_hash(parts: Sequence[str]) -> str:
    return hashlib.sha256(_NORM_TAG + ",".join(parts).encode("ascii")).hexdigest()


def normalised_hash(window: Sequence[float]) -> str | None:
    """sha256 of the window's z-score bins ``floor(z / GRID)``; ``None`` for a
    constant window. This is the hash stored at build time and the first
    probe of a lookup."""
    z = _zscores(window)
    if z is None:
        return None
    return _bins_hash([str(math.floor(v / GRID)) for v in z])


def _probe_hashes(z: Sequence[float]) -> Iterator[str]:
    """Lookup probes of one orientation of a z-scored window.

    The first probe is the plain floor-bin hash (:func:`normalised_hash`).
    Every value within :data:`TOLERANCE` of a bin edge has the neighbouring
    bin as its alternative; the :data:`MAX_EDGE_VALUES` nearest to an edge
    are kept, and every combination of them is tried, nearest-to-edge
    alternatives first."""
    edge = TOLERANCE / GRID
    parts: list[str] = []
    alts: list[tuple[float, int, str]] = []
    for i, v in enumerate(z):
        q = v / GRID
        b = math.floor(q)
        parts.append(str(b))
        below = q - b          # distance (in bins) to the lower edge
        above = 1.0 - below    # ... and to the upper edge
        if below <= edge:
            alts.append((below, i, str(b - 1)))
        elif above <= edge:
            alts.append((above, i, str(b + 1)))
    alts.sort()
    chosen = [(i, s) for _, i, s in alts[:MAX_EDGE_VALUES]]
    yield _bins_hash(parts)
    for mask in range(1, 1 << len(chosen)):
        probe = list(parts)
        for j, (i, s) in enumerate(chosen):
            if mask >> j & 1:
                probe[i] = s
        yield _bins_hash(probe)


def _normalised_lookup(normalised: Mapping, window: Sequence[float]) -> object:
    """The set entry a window hits under the tolerant probe, or ``None``."""
    z = _zscores(window)
    if z is None:
        return None
    for orientation in (z, [-v for v in z]):
        for h in _probe_hashes(orientation):
            hit = normalised.get(h)
            if hit is not None:
                return hit
    return None


# ── input validation ─────────────────────────────────────────────────────────


def _check_int(name: str, v: object, minimum: int) -> int:
    if isinstance(v, bool) or not isinstance(v, int):
        raise TypeError(f"{name} must be an int; got {v!r}")
    if v < minimum:
        raise ValueError(f"{name} must be >= {minimum}; got {v}")
    return v


def _normalise_suites(suites: object) -> dict[str, tuple[str, dict[str, list[float]]]]:
    """``{suite: (version, {item: floats})}``, validated, never the caller's objects."""
    if not isinstance(suites, Mapping):
        raise TypeError(f"suites must be a mapping; got {type(suites).__name__}")
    if not suites:
        raise ValueError("suites is empty: the hash set must cover the pinned suites")
    out: dict[str, tuple[str, dict[str, list[float]]]] = {}
    for suite, spec in suites.items():
        if suite not in KNOWN_SUITES:
            raise ValueError(f"suite {suite!r} is not one of {sorted(KNOWN_SUITES)}")
        if not isinstance(spec, Mapping):
            raise TypeError(f"suite {suite!r}: spec must be a mapping")
        version = spec.get("version")
        if not isinstance(version, str):
            raise TypeError(f"suite {suite!r}: version must be the pinned data revision (str); "
                            f"got {version!r}")
        if not version:
            raise ValueError(f"suite {suite!r}: version is empty (unpinned data)")
        series = spec.get("series")
        if not isinstance(series, Mapping):
            raise TypeError(f"suite {suite!r}: series must be a mapping of item id -> values")
        items: dict[str, list[float]] = {}
        for item, values in series.items():
            if not isinstance(item, str) or not item:
                raise ValueError(f"suite {suite!r}: item id {item!r} is not a non-empty str")
            items[item] = _as_values(values, where=f"{suite}/{item}")
        out[suite] = (version, items)
    return out


# ── build ────────────────────────────────────────────────────────────────────


def build_hashset(suites: Mapping, *, version: int, window: int = WINDOW,
                  stride: int = STRIDE) -> dict:
    """The contamination hash-set artefact of ``suites``
    (``{suite: {"version": str, "series": {item: [float | None]}}}``).

    ``version`` is the artefact version ``n`` (``hashset-v<n>``). Raises
    ``ValueError`` / ``TypeError`` for unknown suite keys, unpinned versions,
    malformed series or bad parameters. Windows with a missing value and
    low-information windows (:func:`is_low_information`) are in neither set.
    The result depends only on the data, the suite versions, ``window``,
    ``stride`` and ``version``."""
    _check_int("version", version, 1)
    _check_int("window", window, 2)
    _check_int("stride", stride, 1)
    norm = _normalise_suites(suites)

    identities = {}
    exact: dict[str, tuple[str, str, int]] = {}
    normalised: dict[str, tuple[str, str, int]] = {}
    for suite in sorted(norm):
        sver, items = norm[suite]
        identities[suite] = suite_identity(suite, {
            "status": "ok", "data_revision": sver,
            "rows": [{"item": i} for i in sorted(items)]})
        for item in sorted(items):
            for off, w in _informative_windows(items[item], window, stride):
                key = (suite, item, off)
                h = exact_hash(w)
                if h not in exact or key < exact[h]:
                    exact[h] = key
                nh = normalised_hash(w)
                if nh is not None and (nh not in normalised or key < normalised[nh]):
                    normalised[nh] = key

    def _ids(m: dict[str, tuple[str, str, int]]) -> dict[str, list[str]]:
        return {h: [s, f"{i}@{o}"] for h, (s, i, o) in sorted(m.items())}

    return {
        "schema": SCHEMA,
        "version": version,
        "suites": identities,
        "window": window,
        "stride": stride,
        "hash": HASH,
        "normalisation": dict(NORMALISATION),
        "exact": _ids(exact),
        "normalised": _ids(normalised),
    }


# ── bytes, digest, file ──────────────────────────────────────────────────────


def artefact_bytes(artefact: Mapping) -> bytes:
    """The canonical JSON encoding the pin is computed over."""
    return json.dumps(artefact, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("ascii")


def artefact_sha256(artefact: Mapping) -> str:
    """``"sha256:<hex>"`` of :func:`artefact_bytes`: the
    ``[lane_b] contamination_hashset`` pin."""
    return "sha256:" + hashlib.sha256(artefact_bytes(artefact)).hexdigest()


def write_artefact(artefact: Mapping, path: str | os.PathLike) -> str:
    """Write :func:`artefact_bytes` to ``path`` (gzip-compressed, with a zero
    mtime, when it ends in ``.gz``) atomically; return :func:`artefact_sha256`."""
    p = Path(path)
    data = artefact_bytes(artefact)
    gz = p.name.endswith(".gz")
    data_out = gzip.compress(data, compresslevel=9, mtime=0) if gz else data
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{p.name}.", dir=str(p.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data_out)
        os.replace(tmp, p)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _check_shape(a: Mapping, where: str) -> None:
    """Refuse a wrong schema (schema 1 included) or a malformed header."""
    if a.get("schema") == _SCHEMA_V1:
        raise ValueError(f"{where}: schema {_SCHEMA_V1!r} uses the retired 4-significant-figure "
                         f"normalised encoding; rebuild it as {SCHEMA!r}")
    if a.get("schema") != SCHEMA:
        raise ValueError(f"{where}: schema {a.get('schema')!r} is not {SCHEMA!r}")
    for key in ("exact", "normalised"):
        if not isinstance(a.get(key), (dict, _HashTable)):
            raise ValueError(f"{where}: {key!r} is not an object")
    if not isinstance(a.get("suites"), dict):
        raise ValueError(f"{where}: 'suites' is not an object")
    for key in ("window", "stride", "version"):
        v = a.get(key)
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            raise ValueError(f"{where}: {key}={v!r} is not a positive int")
    if a["window"] < 2:
        raise ValueError(f"{where}: window={a['window']} < 2")


def load_artefact(path: str | os.PathLike) -> dict:
    """The artefact at ``path`` (``.json`` or ``.json.gz``). Raises
    ``ValueError`` for a wrong schema or a malformed artefact."""
    p = Path(path)
    raw = p.read_bytes()
    if p.name.endswith(".gz"):
        try:
            raw = gzip.decompress(raw)
        except (OSError, EOFError) as e:
            raise ValueError(f"{p}: not a gzip file: {e}") from None
    elif p.name.endswith(".zst"):
        raise ValueError(f"{p}: .zst artefacts are not supported (zstandard is not a dependency)")
    try:
        a = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ValueError(f"{p}: not canonical ASCII JSON: {e}") from None
    if not isinstance(a, dict):
        raise ValueError(f"{p}: artefact is not a JSON object")
    _check_shape(a, str(p))
    return a


# ── the intake's bounded lookup (RV6 B8) ─────────────────────────────────────


class _HashTable:
    """A read-only ``{sha256 hex: [suite, window_id]}`` over sorted 128-bit
    hash prefixes (two ``array("Q")``) with an item index and an offset per
    hash: 24 bytes per hash instead of a dict entry with three strings."""

    __slots__ = ("_hi", "_lo", "_item", "_off", "_items")

    def __init__(self, items: list[tuple[str, str]]) -> None:
        self._hi = array("Q")
        self._lo = array("Q")
        self._item = array(_U32)
        self._off = array(_U32)
        self._items = items

    def _append(self, h: str, item: int, off: int) -> None:
        hi, lo = int(h[:16], 16), int(h[16:32], 16)
        n = len(self._hi)
        if n and self._hi[n - 1] == hi and self._lo[n - 1] == lo:
            raise ValueError(f"two hashes share the 128-bit prefix {h[:32]}: the lookup cannot "
                             "tell them apart")
        self._hi.append(hi)
        self._lo.append(lo)
        self._item.append(item)
        self._off.append(off)

    def __len__(self) -> int:
        return len(self._hi)

    def get(self, h: str, default: object = None) -> object:
        """``[suite, window_id]`` of the hash ``h`` (lower-case hex), or ``default``."""
        if len(h) != 64:
            return default
        try:
            hi, lo = int(h[:16], 16), int(h[16:32], 16)
        except ValueError:
            return default
        i = bisect_left(self._hi, hi)
        n = len(self._hi)
        while i < n and self._hi[i] == hi:
            if self._lo[i] == lo:
                suite, item = self._items[self._item[i]]
                return [suite, f"{item}@{self._off[i]}"]
            i += 1
        return default

    def __contains__(self, h: object) -> bool:
        return isinstance(h, str) and self.get(h) is not None


class ContaminationLookup(Mapping):
    """The read-only lookup :func:`load_lookup` returns. ``match_series`` and
    ``is_contaminated`` accept it like the dict artefact. It exposes the
    artefact's header keys (``schema``, ``version``, ``window``, ``stride``,
    ``suites``, ``hash``, ``normalisation``; container values are returned as
    copies) and ``exact`` / ``normalised`` as read-only hash tables."""

    __slots__ = ("_header", "_tables")

    def __init__(self, header: dict, tables: dict[str, _HashTable]) -> None:
        self._header = header
        self._tables = tables

    def __getitem__(self, key: str) -> object:
        if key in self._tables:
            return self._tables[key]
        v = self._header[key]
        return copy.deepcopy(v) if isinstance(v, (dict, list)) else v

    def __iter__(self) -> Iterator[str]:
        return iter(sorted([*self._header, *self._tables]))

    def __len__(self) -> int:
        return len(self._header) + len(self._tables)


class _CanonicalReader:
    """Incremental reader of the canonical artefact JSON: hashes the bytes as
    they are read and keeps only a bounded window of the text."""

    def __init__(self, f, where: str) -> None:
        self._f = f
        self._where = where
        self._sha = hashlib.sha256()
        self.buf = ""
        self.pos = 0
        self.eof = False

    def _fail(self, what: str) -> ValueError:
        return ValueError(f"{self._where}: not a canonical artefact: {what}")

    def fill(self) -> bool:
        if self.eof:
            return False
        try:
            data = self._f.read(_CHUNK)
        except (OSError, EOFError, zlib.error) as e:
            raise ValueError(f"{self._where}: unreadable: {type(e).__name__}: {e}") from None
        if not data:
            self.eof = True
            return False
        self._sha.update(data)
        try:
            text = data.decode("ascii")
        except UnicodeDecodeError:
            raise self._fail("non-ASCII bytes") from None
        rest = self.buf[self.pos:]
        if len(rest) > _MAX_PENDING:
            raise self._fail(f"a token longer than {_MAX_PENDING} bytes")
        self.buf = rest + text
        self.pos = 0
        return True

    def match(self, rx: re.Pattern, what: str) -> re.Match:
        while True:
            m = rx.match(self.buf, self.pos)
            if m is not None:
                self.pos = m.end()
                return m
            if not self.fill():
                raise self._fail(f"expected {what}")

    def char(self) -> str:
        while self.pos >= len(self.buf):
            if not self.fill():
                raise self._fail("unexpected end of file")
        c = self.buf[self.pos]
        self.pos += 1
        return c

    def peek(self) -> str:
        c = self.char()
        self.pos -= 1
        return c

    def value(self) -> object:
        while True:
            try:
                v, end = _DECODER.raw_decode(self.buf, self.pos)
            except json.JSONDecodeError:
                if self.fill():
                    continue
                raise self._fail("a malformed header value") from None
            if end >= len(self.buf) and self.fill():
                continue
            self.pos = end
            return v

    def finish(self) -> str:
        """Refuse trailing bytes; the ``sha256:<hex>`` of everything read."""
        while True:
            if self.pos < len(self.buf):
                raise self._fail("trailing data after the artefact object")
            if not self.fill():
                return "sha256:" + self._sha.hexdigest()


def _decode_str(tok: str, r: _CanonicalReader) -> str:
    if "\\" not in tok:
        return tok[1:-1]
    try:
        return json.loads(tok)
    except json.JSONDecodeError:
        raise r._fail(f"a malformed string {tok[:40]!r}") from None


def _read_table(r: _CanonicalReader, index: dict[tuple[str, str], int],
                items: list[tuple[str, str]]) -> _HashTable:
    """One of the ``exact`` / ``normalised`` objects, entry by entry."""
    if r.char() != "{":
        raise r._fail("a hash set is not an object")
    table = _HashTable(items)
    if r.peek() == "}":
        r.pos += 1
        return table
    prev = ""
    suites: dict[str, str] = {}
    while True:
        m = r.match(_ENTRY, '"<sha256 hex>":["<suite>","<item>@<offset>"]')
        h = m.group(1)
        if h <= prev:
            raise r._fail(f"hash {h} is not in sorted order")
        suite = _decode_str(m.group(2), r)
        suite = suites.setdefault(suite, suite)
        item, at, off_s = _decode_str(m.group(3), r).rpartition("@")
        if not at or not item or not off_s.isascii() or not off_s.isdigit() \
                or str(int(off_s)) != off_s:
            raise r._fail(f"window id of {h} is not '<item>@<offset>'")
        off = int(off_s)
        if off > _U32_MAX:
            raise r._fail(f"offset {off} of {h} is too large")
        key = (suite, item)
        idx = index.get(key)
        if idx is None:
            idx = index[key] = len(items)
            items.append(key)
        table._append(h, idx, off)
        prev = h
        if m.group(4) == "}":
            return table


def _parse_lookup(f, where: str) -> tuple[dict, dict[str, _HashTable], str]:
    r = _CanonicalReader(f, where)
    if r.char() != "{":
        raise r._fail("not a JSON object")
    header: dict = {}
    tables: dict[str, _HashTable] = {}
    index: dict[tuple[str, str], int] = {}
    items: list[tuple[str, str]] = []
    while True:
        key = _decode_str(r.match(_KEY, "an object key").group(1), r)
        if key in header or key in tables:
            raise r._fail(f"duplicate key {key!r}")
        if key in ("exact", "normalised"):
            tables[key] = _read_table(r, index, items)
        else:
            header[key] = r.value()
        sep = r.char()
        if sep == "}":
            break
        if sep != ",":
            raise r._fail(f"unexpected {sep!r} after key {key!r}")
    return header, tables, r.finish()


def load_lookup(path: str | os.PathLike, *, expect_sha256: str | None = None) -> ContaminationLookup:
    """The intake's memory-bounded lookup of the artefact at ``path``
    (``.json`` or ``.json.gz``).

    The canonical JSON is stream-parsed and hashed as it is read, and each
    set is kept as sorted 128-bit hash prefixes with an item index and an
    offset per hash, so the load stays within
    :data:`LOOKUP_PEAK_BYTES_PER_WINDOW` bytes of heap per window plus
    :data:`LOOKUP_FIXED_BYTES`. Raises ``ValueError`` when the sha256 of the
    file's canonical JSON is not ``expect_sha256`` (given), for a tampered or
    malformed file and for any schema other than :data:`SCHEMA` (schema 1
    included)."""
    if expect_sha256 is not None and (not isinstance(expect_sha256, str)
                                      or not _PIN.fullmatch(expect_sha256)):
        raise ValueError(f"expect_sha256={expect_sha256!r} is not sha256:<64 hex>")
    p = Path(path)
    if p.name.endswith(".zst"):
        raise ValueError(f"{p}: .zst artefacts are not supported (zstandard is not a dependency)")
    with open(p, "rb") as raw:
        if p.name.endswith(".gz"):
            with gzip.GzipFile(fileobj=raw, mode="rb") as f:
                header, tables, sha = _parse_lookup(f, str(p))
        else:
            header, tables, sha = _parse_lookup(raw, str(p))
    if expect_sha256 is not None and sha != expect_sha256:
        raise ValueError(f"{p}: sha256 {sha} != expected {expect_sha256} (wrong pin or tampered)")
    _check_shape({**header, **tables}, str(p))
    return ContaminationLookup(header, tables)


def lookup_peak_bytes(path: str | os.PathLike) -> int:
    """Traced heap peak (bytes) of :func:`load_lookup` on ``path``: what the
    pin evidence records as ``lookup_peak_bytes``."""
    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
        tracemalloc.start()
    try:
        base = tracemalloc.get_traced_memory()[0]
        tracemalloc.reset_peak()
        lk = load_lookup(path)
        peak = tracemalloc.get_traced_memory()[1] - base
        del lk
    finally:
        if not was_tracing:
            tracemalloc.stop()
    return max(1, peak)


def lookup_budget(n_windows: int) -> int:
    """The heap budget of :func:`load_lookup` for ``n_windows`` suite windows."""
    return LOOKUP_PEAK_BYTES_PER_WINDOW * n_windows + LOOKUP_FIXED_BYTES


# ── lookup ───────────────────────────────────────────────────────────────────


def match_series(artefact: Mapping, values: Iterable) -> dict:
    """Hits of a submitted series against the hash set (the dict artefact or
    a :func:`load_lookup` structure).

    Every offset (stride 1) is scanned with the artefact's window; windows
    with a missing value and low-information windows
    (:func:`is_low_information`) never count. Returns
    ``{"exact_hits", "normalised_hits", "first"}``; ``first`` is the hit at
    the lowest submission offset ``at`` (exact before normalised) as
    ``{"kind", "suite", "window_id", "at"}``, or ``None``."""
    window = artefact["window"]
    exact = artefact["exact"]
    normalised = artefact["normalised"]
    vals = _as_values(values, where="submission")
    exact_hits = normalised_hits = 0
    first = None
    for off, w in _informative_windows(vals, window, 1):
        hit = exact.get(exact_hash(w))
        if hit is not None:
            exact_hits += 1
            if first is None:
                first = {"kind": "exact", "suite": hit[0], "window_id": hit[1], "at": off}
        nhit = _normalised_lookup(normalised, w)
        if nhit is not None:
            normalised_hits += 1
            if first is None:
                first = {"kind": "normalised", "suite": nhit[0], "window_id": nhit[1], "at": off}
    return {"exact_hits": exact_hits, "normalised_hits": normalised_hits, "first": first}


def _default_min_hits() -> dict[str, int]:
    from cascade.shared.config.v2 import LaneBConfig

    return dict(LaneBConfig().contamination_min_hits)


def is_contaminated(artefact: Mapping, values: Iterable, min_hits: object = None) -> bool:
    """``exact_hits >= min_hits["exact"]`` or ``normalised_hits >=
    min_hits["normalised"]``. ``artefact`` is the dict artefact or a
    :func:`load_lookup` structure. ``min_hits`` is anything ``dict()``
    accepts (e.g. ``LaneBConfig.contamination_min_hits``); ``None`` is the
    WB10 default."""
    thresholds = _default_min_hits()
    if min_hits is not None:
        thresholds.update(dict(min_hits))
    for k in ("exact", "normalised"):
        _check_int(f"min_hits[{k!r}]", thresholds[k], 1)
    r = match_series(artefact, values)
    return r["exact_hits"] >= thresholds["exact"] or r["normalised_hits"] >= thresholds["normalised"]


# ── CLI ──────────────────────────────────────────────────────────────────────


def _read_suites(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    if not isinstance(doc, dict) or not isinstance(doc.get("suites"), dict):
        raise ValueError(f"{path}: expected {{\"suites\": {{...}}}}")
    return doc["suites"]


def _emit(obj: Mapping) -> None:
    sys.stdout.write(json.dumps(obj, sort_keys=True) + "\n")


def _cmd_build(args: argparse.Namespace) -> int:
    try:
        suites = _read_suites(args.input)
        a = build_hashset(suites, version=args.version, window=args.window, stride=args.stride)
    except (OSError, ValueError, TypeError) as e:
        sys.stderr.write(f"contamination build: refused: {type(e).__name__}: {e}\n")
        return 2
    sha = write_artefact(a, args.out)
    _emit({"sha256": sha, "version": a["version"], "suites": a["suites"],
           "window": a["window"], "stride": a["stride"],
           "n_exact": len(a["exact"]), "n_normalised": len(a["normalised"]),
           "out": str(args.out)})
    return 0


def _budget_problems(report: dict) -> list[str]:
    """The pinned file's size and lookup peak against the box's budget."""
    problems = []
    n = max(report["n_exact"], report["n_normalised"])
    size, peak = report["file_bytes"], report["lookup_peak_bytes"]
    if size > FILE_BYTES_PER_WINDOW * n + FILE_FIXED_BYTES:
        problems.append(f"file {size} B > {FILE_BYTES_PER_WINDOW} B/window x {n} + "
                        f"{FILE_FIXED_BYTES} B")
    if peak > lookup_budget(n):
        problems.append(f"load_lookup peak {peak} B > {LOOKUP_PEAK_BYTES_PER_WINDOW} B/window x "
                        f"{n} + {LOOKUP_FIXED_BYTES} B")
    if peak > LOOKUP_CAP_BYTES:
        problems.append(f"load_lookup peak {peak} B > the box cap {LOOKUP_CAP_BYTES} B")
    return problems


def _cmd_verify(args: argparse.Namespace) -> int:
    report: dict = {"ok": False, "artefact": str(args.artefact), "expected": args.sha256}
    try:
        a = load_artefact(args.artefact)
        sha = artefact_sha256(a)
    except (OSError, ValueError, TypeError) as e:
        report["problems"] = [f"load: {type(e).__name__}: {e}"]
        _emit(report)
        return 1
    problems = []
    report["sha256"] = sha
    if sha != args.sha256:
        problems.append(f"sha256 {sha} != expected {args.sha256}")
    report["n_exact"] = len(a["exact"])
    report["n_normalised"] = len(a["normalised"])
    report["file_bytes"] = os.path.getsize(args.artefact)
    try:
        report["lookup_peak_bytes"] = lookup_peak_bytes(args.artefact)
    except (OSError, ValueError) as e:
        problems.append(f"load_lookup: {type(e).__name__}: {e}")
    else:
        problems += _budget_problems(report)
    if args.input:
        try:
            rebuilt = build_hashset(_read_suites(args.input), version=a["version"],
                                    window=a["window"], stride=a["stride"])
        except (OSError, ValueError, TypeError) as e:
            problems.append(f"rebuild: {type(e).__name__}: {e}")
        else:
            report["rebuild_sha256"] = artefact_sha256(rebuilt)
            if artefact_bytes(rebuilt) != artefact_bytes(a):
                problems.append(f"rebuild from {args.input} is not byte-identical "
                                f"({report['rebuild_sha256']} != {sha})")
    report["suites"] = a["suites"]
    report["version"] = a["version"]
    report["problems"] = problems
    report["ok"] = not problems
    _emit(report)
    return 0 if report["ok"] else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -P -m cascade.eval.contamination",
                                 description="Build or verify the LB1 contamination hash set.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build the hash-set artefact from suite data")
    b.add_argument("--input", required=True, help='SUITES.json: {"suites": {...}}')
    b.add_argument("--version", required=True, type=int, help="artefact version n (hashset-v<n>)")
    b.add_argument("--out", required=True, help="artefact path (.json or .json.gz)")
    b.add_argument("--window", type=int, default=WINDOW)
    b.add_argument("--stride", type=int, default=STRIDE)
    v = sub.add_parser("verify", help="check an artefact against its pin (and a rebuild)")
    v.add_argument("--artefact", required=True)
    v.add_argument("--sha256", required=True, help="the expected pin, sha256:<hex>")
    v.add_argument("--input", help="SUITES.json to rebuild from and compare byte for byte")
    args = ap.parse_args(argv)
    return _cmd_build(args) if args.cmd == "build" else _cmd_verify(args)


__all__ = ["AFFINE_RESIDUAL", "FILE_BYTES_PER_WINDOW", "FILE_FIXED_BYTES", "GRID", "HASH",
           "LOOKUP_CAP_BYTES", "LOOKUP_FIXED_BYTES", "LOOKUP_PEAK_BYTES_PER_WINDOW",
           "MAX_EDGE_VALUES", "MIN_DISTINCT", "MODE_SHARE", "NORMALISATION", "SCHEMA",
           "STD_REL_FLOOR", "STRIDE", "TOLERANCE", "WINDOW",
           "ContaminationLookup", "artefact_bytes", "artefact_sha256", "build_hashset",
           "exact_hash", "is_contaminated", "is_low_information", "load_artefact",
           "load_lookup", "lookup_budget", "lookup_peak_bytes", "match_series",
           "normalised_hash", "write_artefact"]


if __name__ == "__main__":
    sys.exit(main())
