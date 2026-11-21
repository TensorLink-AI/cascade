"""S0a de-drift classifier (plan §3.0 R6, §5 S0a, §14).

    python -m cascade.laned.dedrift --deployed F --main G --out DIR --effective-block N

Reads a deployed ``chain.toml`` and the release ``chain.toml``, classifies
every ``LIVE OVERRIDE`` marker of the deployed file as ``main``, ``overlay``
or ``operator-only`` and writes ``classification.json``, ``main.diff`` (the
release PR that types the main-class drift) and ``overlay.json`` (the
DEC-CA-0057 body for the overlay-class drift). Both inputs are read only.

Every rule reads the installed ``cascade.shared`` code at run time
(``netconfig.overlay_policy``, ``REFUSED_TABLES``, ``check_overrides``,
``effective_raw``; ``decided.TYPED_KEY_BY_PREFIX`` / ``PIN_KEYS``;
``reliability.DEFAULTS``), so a release that changes them changes the
classification without editing this file. Nothing here touches the chain,
storage or the network.
"""
from __future__ import annotations

import argparse
import copy
import difflib
import hashlib
import json
import math
import re
import sys
import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from cascade.shared import decided, netconfig
from cascade.shared.config import parse_chain_config
from cascade.shared.config.reliability import DEFAULTS as RELIABILITY_DEFAULTS

SCHEMA = "cascade-dedrift/1"
MARKER = "LIVE OVERRIDE"
CLASSES = ("main", "overlay", "operator-only")
OUTPUTS = ("classification.json", "main.diff", "overlay.json")

# Rules 3 and 4 name single fields; no code table lists them.
# config/scoring.py: ``gift_gate_data_dir`` points at benchmark data on the
# validator host; ``calib_horizons`` is LOG-ONLY telemetry that never feeds
# the verdict.
HOST_PATH_KEYS = frozenset({("eval", "gift_gate_data_dir")})
LOG_ONLY_KEYS = frozenset({("eval", "calib_horizons")})
# The resolved ``launch_at_utc`` block, typed per the runbook (§5 S0a).
LAUNCH_BLOCK_KEY = "v2.from_block"

_KEY_LINE = re.compile(r"""^\s*(?:[A-Za-z0-9_\-]+|"(?:[^"\\]|\\.)*"|'[^']*')"""
                       r"""(?:\s*\.\s*(?:[A-Za-z0-9_\-]+|"(?:[^"\\]|\\.)*"|'[^']*'))*\s*=""")
_HEADER_LINE = re.compile(r"^\s*\[")
_BARE_KEY = re.compile(r"^[A-Za-z0-9_\-]+$")


class DedriftError(ValueError):
    """Input the classifier refuses (dangling marker, bad TOML, ...)."""


# ── the rules ────────────────────────────────────────────────────────────────


class _AnyBlock:
    """Answers every attribute with itself and ``int()`` with 1, so running
    ``netconfig.effective_raw`` on it reports every key that function writes
    from a resolved activation (all ``resolved_*`` read as set)."""

    def __getattr__(self, name: str) -> _AnyBlock:
        if name.startswith("__"):
            raise AttributeError(name)
        return self

    def __int__(self) -> int:
        return 1


@dataclass(frozen=True)
class _ResolvedProbe:
    raw: dict
    activation: _AnyBlock
    round: _AnyBlock
    scoring: _AnyBlock
    v2: _AnyBlock
    eval: _AnyBlock


@lru_cache(maxsize=1)
def decided_block_keys() -> frozenset[str]:
    """Dotted keys of decided blocks: what ``effective_raw`` writes from a
    resolved activation, every ``TYPED_KEY_BY_PREFIX`` key, and the launch
    block."""
    any_block = _AnyBlock()
    probe = _ResolvedProbe(raw={}, activation=any_block, round=any_block,
                           scoring=any_block, v2=any_block, eval=any_block)
    written = netconfig.effective_raw(probe)
    keys = {f"{table}.{key}" for table, kv in written.items() for key in kv}
    keys |= {key for _prefix, key in decided.TYPED_KEY_BY_PREFIX}
    keys.add(LAUNCH_BLOCK_KEY)
    return frozenset(keys)


def _nest(table: str, key: str, value: object) -> dict:
    parts = table.split(".")
    inner: dict = {key: value}
    for part in reversed(parts[1:]):
        inner = {part: inner}
    return {parts[0]: inner}


def classify(table: str, key: str, value: object = 0) -> tuple[str, str]:
    """``(class, reason)`` for one ``[table] key``, first matching rule wins
    (tests/accept/r6/README.md). ``value`` only matters to
    ``check_overrides`` (a table value in ``[v2]`` is a nested table)."""
    top = table.split(".", 1)[0]
    dotted = f"{table}.{key}"
    if top in netconfig.overlay_policy()["operator_only"]:
        return "operator-only", "operator_only_table"
    if table == "round" and key in RELIABILITY_DEFAULTS:
        return "operator-only", "trainer_only_policy"
    if (table, key) in HOST_PATH_KEYS:
        return "operator-only", "host_path"
    if (table, key) in LOG_ONLY_KEYS:
        return "operator-only", "log_only"
    if dotted in decided_block_keys():
        return "main", "decided_block"
    if dotted in decided.PIN_KEYS:
        return "main", "decided_pin"
    if top in netconfig.REFUSED_TABLES:
        return "main", "release_only_table"
    try:
        netconfig.check_overrides(_nest(table, key, value))
    except netconfig.NetConfigError:
        return "main", "release_only_key"
    return "overlay", "overlay_key"


# ── reading the TOML text ────────────────────────────────────────────────────


def comment_start(line: str) -> int:
    """Index of the ``#`` that opens the line's comment, or -1."""
    quote = ""
    i = 0
    while i < len(line):
        ch = line[i]
        if quote:
            if quote == '"' and ch == "\\":
                i += 2
                continue
            if ch == quote:
                quote = ""
        elif ch in "\"'":
            quote = ch
        elif ch == "#":
            return i
        i += 1
    return -1


def _has_marker(line: str) -> bool:
    cs = comment_start(line)
    return cs >= 0 and MARKER in line[cs:]


@dataclass(frozen=True)
class KeySpan:
    start: int          # 0-based first line
    end: int            # 0-based last line (inclusive)
    table: str
    key: str


@dataclass(frozen=True)
class Scan:
    keys: list[KeySpan]
    headers: list[tuple[int, str, bool]]   # (line, dotted table, array-of-tables)
    markers: list[tuple[int, KeySpan]]     # (marker line, the key it attaches to)


def _path_of(parsed: dict) -> list[str]:
    path: list[str] = []
    node: object = parsed
    while isinstance(node, dict) and len(node) == 1:
        name = next(iter(node))
        path.append(name)
        node = node[name]
        if isinstance(node, list):
            node = node[-1] if node else None
    return path


def scan(text: str, *, name: str) -> Scan:
    """Key spans, table headers and attached markers of one TOML text."""
    lines = text.splitlines()
    keys: list[KeySpan] = []
    headers: list[tuple[int, str, bool]] = []
    markers: list[tuple[int, KeySpan]] = []
    pending: list[int] = []
    table = ""
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            if _has_marker(line):
                pending.append(i)
            i += 1
            continue
        if _HEADER_LINE.match(line) and not _KEY_LINE.match(line):
            if pending or _has_marker(line):
                bad = pending[0] if pending else i
                raise DedriftError(f"{name}:{bad + 1}: {MARKER} marker is not followed by a "
                                   f"key before the table header at line {i + 1}")
            try:
                parsed = tomllib.loads(line)
            except tomllib.TOMLDecodeError as e:
                raise DedriftError(f"{name}:{i + 1}: bad table header ({e})") from e
            table = ".".join(_path_of(parsed))
            headers.append((i, table, stripped.startswith("[[")))
            i += 1
            continue
        if not _KEY_LINE.match(line):
            raise DedriftError(f"{name}:{i + 1}: not a key, header or comment line")
        end = i
        while True:
            try:
                parsed = tomllib.loads("\n".join(lines[i:end + 1]))
                break
            except tomllib.TOMLDecodeError as e:
                if end + 1 >= len(lines):
                    raise DedriftError(f"{name}:{i + 1}: value does not parse ({e})") from e
                end += 1
        path = _path_of(parsed)
        if len(path) == 1:
            span = KeySpan(i, end, table, path[0])
        else:
            span = KeySpan(i, end, ".".join(([table] if table else []) + path[:-1]), path[-1])
        keys.append(span)
        for at in pending:
            markers.append((at, span))
        pending = []
        for j in range(i, end + 1):
            if _has_marker(lines[j]):
                markers.append((j, span))
        i = end + 1
    if pending:
        raise DedriftError(f"{name}:{pending[0] + 1}: {MARKER} marker is not followed by a "
                           f"key before the end of the file")
    for at, span in markers:
        if not span.table:
            raise DedriftError(f"{name}:{at + 1}: {MARKER} marker on a top-level key "
                               f"({span.key}); no table to classify it under")
        if any(t == span.table and arr for _l, t, arr in headers):
            raise DedriftError(f"{name}:{at + 1}: {MARKER} marker inside an array of "
                               f"tables [[{span.table}]] is not supported")
    return Scan(keys, headers, sorted(markers, key=lambda m: m[0]))


_MISSING = object()


def lookup(raw: dict, table: str, key: str) -> object:
    node: object = raw
    for part in table.split("."):
        if not isinstance(node, dict) or part not in node:
            return _MISSING
        node = node[part]
    if not isinstance(node, dict) or key not in node:
        return _MISSING
    return node[key]


# ── writing TOML values ──────────────────────────────────────────────────────


def toml_key(key: str) -> str:
    return key if _BARE_KEY.match(key) else json.dumps(key, ensure_ascii=False)


def toml_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return "[" + ", ".join(toml_value(v) for v in value) + "]"
    if isinstance(value, dict):
        if not value:
            return "{}"
        return "{ " + ", ".join(f"{toml_key(k)} = {toml_value(v)}"
                                for k, v in value.items()) + " }"
    raise DedriftError(f"cannot write a {type(value).__name__} value into chain.toml")


def _json_value(value: object, where: str) -> object:
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as e:
        raise DedriftError(f"{where}: value is not representable as JSON ({e})") from e


# ── the main-PR patch ────────────────────────────────────────────────────────


def _replacement(lines: list[str], span: KeySpan, value: object) -> list[str]:
    first = lines[span.start]
    key_part = _KEY_LINE.match(first)
    if key_part is None:
        raise DedriftError(f"line {span.start + 1}: not a key line")
    eq = key_part.end() - 1
    rest = first[eq + 1:]
    gap = rest[:len(rest) - len(rest.lstrip())] or " "
    prefix = first[:eq + 1] + gap
    last = lines[span.end]
    cs = comment_start(last)
    comment = ""
    if cs >= 0 and MARKER not in last[cs:]:
        before = last[:cs]
        comment = before[len(before.rstrip()):] + last[cs:]
    if isinstance(value, list) and value and span.end > span.start:
        indent = " " * (len(first) - len(first.lstrip()) + 4)
        body = [prefix + "["] + [f"{indent}{toml_value(v)}," for v in value]
        return body + ["]" + comment]
    return [prefix + toml_value(value) + comment]


def patch_main(main_text: str, main_scan: Scan, edits: list[tuple[str, str, object]]) -> str:
    """``main_text`` with each ``(table, key, value)`` typed in."""
    lines = main_text.splitlines()
    by_key = {(s.table, s.key): s for s in main_scan.keys}
    replace: dict[int, tuple[int, list[str]]] = {}
    insert_after: dict[int, list[str]] = {}
    appended: dict[str, list[str]] = {}
    for table, key, value in edits:
        span = by_key.get((table, key))
        if span is not None:
            replace[span.start] = (span.end, _replacement(lines, span, value))
            continue
        new_line = f"{toml_key(key)} = {toml_value(value)}"
        in_table = [s.end for s in main_scan.keys if s.table == table]
        header = [ln for ln, t, _arr in main_scan.headers if t == table]
        if in_table:
            insert_after.setdefault(max(in_table), []).append(new_line)
        elif header:
            insert_after.setdefault(header[0], []).append(new_line)
        else:
            appended.setdefault(table, []).append(new_line)
    out: list[str] = []
    i = 0
    while i < len(lines):
        if i in replace:
            end, new = replace[i]
            out.extend(new)
            last = end
        else:
            out.append(lines[i])
            last = i
        for j in range(i, last + 1):
            out.extend(insert_after.get(j, []))
        i = last + 1
    for table, new_lines in appended.items():
        out.extend(["", "[" + ".".join(toml_key(p) for p in table.split(".")) + "]", *new_lines])
    return "\n".join(out) + "\n" if out else ""


def unified_diff(old: str, new: str) -> str:
    if old == new:
        return ""
    a, b = old.splitlines(keepends=True), new.splitlines(keepends=True)
    out: list[str] = []
    for line in difflib.unified_diff(a, b, "a/chain.toml", "b/chain.toml", n=3):
        out.append(line if line.endswith("\n") else line + "\n\\ No newline at end of file\n")
    return "".join(out)


# ── the run ──────────────────────────────────────────────────────────────────


def _load(path: Path, name: str) -> tuple[bytes, str, dict]:
    data = path.read_bytes()
    text = data.decode("utf-8")
    try:
        raw = tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise DedriftError(f"{name}: not valid TOML ({e})") from e
    return data, text, raw


def dedrift(deployed: Path, main: Path, effective_block: int) -> dict[str, str]:
    """The three output files' text, after every check has passed."""
    dep_bytes, dep_text, dep_raw = _load(deployed, "--deployed")
    main_bytes, main_text, main_raw = _load(main, "--main")
    dep_scan = scan(dep_text, name="--deployed")
    main_scan = scan(main_text, name="--main")
    if MARKER in main_text:
        raise DedriftError(f"--main carries {MARKER} text; the release file must not")

    markers: list[dict] = []
    for at, span in dep_scan.markers:
        where = f"--deployed:{at + 1} [{span.table}] {span.key}"
        dep_val = lookup(dep_raw, span.table, span.key)
        if dep_val is _MISSING:
            raise DedriftError(f"{where}: {MARKER} key not found in the parsed TOML")
        main_val = lookup(main_raw, span.table, span.key)
        cls, reason = classify(span.table, span.key, dep_val)
        dep_json = _json_value(dep_val, where)
        main_json = None if main_val is _MISSING else _json_value(main_val, where)
        markers.append({
            "line": at + 1, "table": span.table, "key": span.key,
            "deployed": dep_json, "main": main_json,
            "drift": main_val is _MISSING or dep_val != main_val,
            "class": cls, "reason": reason,
        })

    main_edits: list[tuple[str, str, object]] = []
    overrides: dict[str, dict] = {}
    seen: set[tuple[str, str]] = set()
    for m in markers:
        if not m["drift"] or (m["table"], m["key"]) in seen:
            continue
        seen.add((m["table"], m["key"]))
        dep_val = lookup(dep_raw, m["table"], m["key"])
        if m["class"] == "main":
            main_edits.append((m["table"], m["key"], dep_val))
        elif m["class"] == "overlay":
            if "." in m["table"]:
                raise DedriftError(
                    f"--deployed:{m['line']} [{m['table']}] {m['key']}: an overlay sets a "
                    f"top-level table's key; a nested table cannot be overridden key by key")
            overrides.setdefault(m["table"], {})[m["key"]] = copy.deepcopy(dep_val)

    patched_text = patch_main(main_text, main_scan, main_edits) if main_edits else main_text
    want = copy.deepcopy(main_raw)
    for table, key, value in main_edits:
        node = want
        for part in table.split("."):
            node = node.setdefault(part, {})
        node[key] = copy.deepcopy(value)
    patched_raw = tomllib.loads(patched_text)
    if patched_raw != want:
        raise DedriftError("the typed main file does not parse to main + the main-class drift")
    if MARKER in patched_text:
        raise DedriftError(f"the typed main file carries {MARKER} text")
    patched_cfg = parse_chain_config(copy.deepcopy(patched_raw))
    main_cfg = parse_chain_config(copy.deepcopy(main_raw))

    body = netconfig.next_overlay(overrides, effective_block, None,
                                  network=netconfig.network_identity(main_cfg))
    netconfig.validate_overlay(patched_cfg, body)

    counts = {c: sum(1 for m in markers if m["class"] == c) for c in CLASSES}
    doc = {
        "schema": SCHEMA,
        "deployed_sha256": hashlib.sha256(dep_bytes).hexdigest(),
        "main_sha256": hashlib.sha256(main_bytes).hexdigest(),
        "markers": markers,
        "counts": counts,
    }
    return {
        "classification.json": json.dumps(doc, indent=2, ensure_ascii=False) + "\n",
        "main.diff": unified_diff(main_text, patched_text),
        "overlay.json": json.dumps(body, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m cascade.laned.dedrift",
                                 description=__doc__.split("\n\n")[0])
    ap.add_argument("--deployed", required=True, type=Path, help="deployed chain.toml (read only)")
    ap.add_argument("--main", required=True, type=Path, help="release chain.toml (read only)")
    ap.add_argument("--out", required=True, type=Path, help="output directory")
    ap.add_argument("--effective-block", required=True, type=int,
                    help="the overlay body's effective_block (an era start)")
    args = ap.parse_args(argv)
    try:
        files = dedrift(args.deployed, args.main, args.effective_block)
    except (DedriftError, netconfig.NetConfigError, ValueError, OSError) as e:
        print(f"dedrift: {e}", file=sys.stderr)
        return 1
    args.out.mkdir(parents=True, exist_ok=True)
    for name in OUTPUTS:
        (args.out / name).write_text(files[name], encoding="utf-8")
    counts = json.loads(files["classification.json"])["counts"]
    print("dedrift: " + " ".join(f"{c}={counts[c]}" for c in CLASSES))
    return 0


if __name__ == "__main__":
    sys.exit(main())
