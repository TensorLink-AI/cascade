#!/usr/bin/env python3
"""Stub scan (plan §3.1 build rule 3).

    stub_scan.py --root R [PATH_OR_GLOB ...] [--paths-file FILE]

Refuses placeholder markers, ``pass``/``...``-only function bodies, uses of the
not-implemented exception, hardcoded sample values and mock imports in every
scanned file outside ``tests/``. Prints ``<relpath>:<line>: <rule-id> ...``
per finding and exits 1 when there is any.

The marker words and sample values below are assembled from fragments so this
file passes its own scan (W0c's Accept runs it over ``scripts/laned/``).
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

import lanedlib

MARKERS = (("TO" + "DO", "todo"), ("FIX" + "ME", "fixme"))
MARKER_RES = [(re.compile(r"\b" + word + r"\b"), rule) for word, rule in MARKERS]
NOT_IMPL = "NotImpl" + "ementedError"
SAMPLE_WORDS = frozenset((
    "sam" + "ple", "exam" + "ple", "dum" + "my", "place" + "holder", "f" + "oo", "b" + "ar",
    "b" + "az", "change" + "me", "t" + "bd", "x" + "xx",
))
SAMPLE_SUBSTRINGS = ("lorem " + "ipsum", "dead" + "beef")
REPEATED_HEX = re.compile(r"^(?:[a-z0-9_\-]+:)?([0-9a-f])\1{7,}$")
MOCK_MODULES = ("mo" + "ck", "pytest_" + "mo" + "ck")
UNITTEST_MOCK = "unittest." + "mo" + "ck"


def is_python(path: Path, text: str) -> bool:
    if path.suffix == ".py":
        return True
    first = text.split("\n", 1)[0]
    return first.startswith("#!") and "python" in first


def sample_value_reason(value: str) -> str | None:
    norm = value.strip().lower()
    if norm in SAMPLE_WORDS:
        return f"string constant {value!r}"
    for sub in SAMPLE_SUBSTRINGS:
        if sub in norm:
            return f"string constant contains {sub!r}"
    if REPEATED_HEX.match(norm):
        return f"repeated-character digest {value!r}"
    return None


def _mod_is_mock(name: str) -> bool:
    return any(name == mod or name.startswith(mod + ".") for mod in MOCK_MODULES + (UNITTEST_MOCK,))


def _body_after_docstring(body: list[ast.stmt]) -> list[ast.stmt]:
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        return body[1:]
    return body


def scan_python(tree: ast.AST) -> list[tuple[int, str, str]]:
    findings: list[tuple[int, str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            rest = _body_after_docstring(node.body)
            if len(rest) == 1:
                stmt = rest[0]
                if isinstance(stmt, ast.Pass):
                    findings.append((node.lineno, "pass-body", f"function {node.name} has only pass"))
                elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) \
                        and stmt.value.value is Ellipsis:
                    findings.append((node.lineno, "ellipsis-body", f"function {node.name} has only ..."))
        elif isinstance(node, ast.Name) and node.id == NOT_IMPL:
            findings.append((node.lineno, "not-implemented", node.id))
        elif isinstance(node, ast.Attribute) and node.attr == NOT_IMPL:
            findings.append((node.lineno, "not-implemented", node.attr))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            reason = sample_value_reason(node.value)
            if reason:
                findings.append((node.lineno, "sample-value", reason))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if _mod_is_mock(alias.name):
                    findings.append((node.lineno, "mock-import", f"import {alias.name}"))
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if _mod_is_mock(mod):
                findings.append((node.lineno, "mock-import", f"from {mod} import ..."))
            elif mod == "unittest" and any(a.name == UNITTEST_MOCK.split(".")[1] for a in node.names):
                findings.append((node.lineno, "mock-import", f"from {mod} import {UNITTEST_MOCK.split('.')[1]}"))
    return findings


def scan_file(path: Path, rel: str) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []
    findings: list[tuple[int, str, str]] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        for regex, rule in MARKER_RES:
            if regex.search(line):
                findings.append((lineno, rule, line.strip()[:80]))
    if is_python(path, text):
        try:
            tree = ast.parse(text, filename=rel)
        except SyntaxError as exc:
            findings.append((exc.lineno or 1, "syntax-error", str(exc.msg)))
        else:
            findings.extend(scan_python(tree))
    return [f"{rel}:{ln}: {rule} {detail}" for ln, rule, detail in sorted(findings)]


def scan(root: Path, entries: list[str]) -> tuple[list[str], int]:
    lines: list[str] = []
    files = lanedlib.expand_entries(root, entries)
    scanned = 0
    for path in files:
        try:
            rel = path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            rel = path.as_posix()
        if rel.split("/", 1)[0] == "tests":
            continue
        scanned += 1
        lines.extend(scan_file(path, rel))
    return lines, scanned


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=".")
    ap.add_argument("--paths-file", action="append", default=[])
    ap.add_argument("paths", nargs="*")
    args = ap.parse_args(argv)
    root = Path(args.root)
    entries = list(args.paths)
    for pf in args.paths_file:
        entries.extend(lanedlib.parse_paths(lanedlib.resolve(root, pf).read_text()))
    if not entries:
        print("stub_scan: no paths given", file=sys.stderr)
        return 2
    lines, scanned = scan(root, entries)
    for line in lines:
        print(line)
    print(f"stub_scan: {len(lines)} finding(s) in {scanned} file(s)", file=sys.stderr)
    return 1 if lines else 0


if __name__ == "__main__":
    sys.exit(main())
