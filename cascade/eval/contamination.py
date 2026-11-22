"""Contamination hash set over the public benchmark suites (LANE_D_PLAN §12b.1
LB1, §3.0 WB11).

The hash set holds two hashes of every sliding sub-window of every suite test
window:

- **exact**: sha256 of the window's values packed as little-endian IEEE 754
  float64 (``struct.pack("<32d", ...)``);
- **normalised** (``cascade-zgrid/2``): the window z-scored on its own values
  (population std), each z-score put on a fixed-point grid
  (``floor(z / GRID)``, absolute bins of width :data:`GRID`), and the bin
  ints hashed with sha256 under a domain tag. A constant window has no
  z-score and no normalised hash.

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
LB1's intake check (WB13) calls :func:`match_series` / :func:`is_contaminated`.

The owner builds the real artefact on the bench pod::

    python -P -m cascade.eval.contamination build --input SUITES.json --version N --out PATH
    python -P -m cascade.eval.contamination verify --artefact PATH --sha256 sha256:<hex> [--input SUITES.json]

``SUITES.json`` is ``{"suites": {suite: {"version": str, "series": {item: [float|null]}}}}``.
The byte encoding of the normalised hash is fixed by :data:`SCHEMA` version 2;
changing it needs a new schema. Schema 1 (z-scores at 4 significant figures)
missed rounded copies at small series levels and is refused on load.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import struct
import sys
import tempfile
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

# Domain tag of the normalised preimage: a normalised hash can never equal an
# exact hash of the same window (different preimages).
_NORM_TAG = b"cascade-contamination-zgrid/2\n"
# |z| below this is taken as 0: a value equal to the window mean in exact
# arithmetic comes out as ~1e-16 in floating point, whose sign would otherwise
# depend on the window's scale and shift.
_ZERO_Z = 1e-9


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
    malformed series or bad parameters. The result depends only on the data,
    the suite versions, ``window``, ``stride`` and ``version``."""
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
            for off, w in _windows(items[item], window, stride):
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
    if a.get("schema") == _SCHEMA_V1:
        raise ValueError(f"{p}: schema {_SCHEMA_V1!r} uses the retired 4-significant-figure "
                         f"normalised encoding; rebuild it as {SCHEMA!r}")
    if a.get("schema") != SCHEMA:
        raise ValueError(f"{p}: schema {a.get('schema')!r} is not {SCHEMA!r}")
    for key in ("exact", "normalised", "suites"):
        if not isinstance(a.get(key), dict):
            raise ValueError(f"{p}: {key!r} is not an object")
    for key in ("window", "stride", "version"):
        v = a.get(key)
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            raise ValueError(f"{p}: {key}={v!r} is not a positive int")
    if a["window"] < 2:
        raise ValueError(f"{p}: window={a['window']} < 2")
    return a


# ── lookup ───────────────────────────────────────────────────────────────────


def match_series(artefact: Mapping, values: Iterable) -> dict:
    """Hits of a submitted series against the hash set.

    Every offset (stride 1) is scanned with the artefact's window. Returns
    ``{"exact_hits", "normalised_hits", "first"}``; ``first`` is the hit at
    the lowest submission offset ``at`` (exact before normalised) as
    ``{"kind", "suite", "window_id", "at"}``, or ``None``."""
    window = artefact["window"]
    exact = artefact["exact"]
    normalised = artefact["normalised"]
    vals = _as_values(values, where="submission")
    exact_hits = normalised_hits = 0
    first = None
    for off, w in _windows(vals, window, 1):
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
    min_hits["normalised"]``. ``min_hits`` is anything ``dict()`` accepts
    (e.g. ``LaneBConfig.contamination_min_hits``); ``None`` is the WB10
    default."""
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


__all__ = ["GRID", "HASH", "MAX_EDGE_VALUES", "NORMALISATION", "SCHEMA", "STRIDE",
           "TOLERANCE", "WINDOW",
           "artefact_bytes", "artefact_sha256", "build_hashset", "exact_hash",
           "is_contaminated", "load_artefact", "match_series", "normalised_hash",
           "write_artefact"]


if __name__ == "__main__":
    sys.exit(main())
