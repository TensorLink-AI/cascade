#!/usr/bin/env python3
"""Follower-pattern check (plan §3.1 build rule 5).

    follower_patterns.py --root R --source DIR [...] --patterns FILE [...]

A lane D log call is ``<recv>.info|warning|error|critical|exception(msg, ...)``
in a ``.py`` file whose repo-relative path contains ``lane_d``, where the
receiver's last name segment contains ``log``. Its message must be literal and
start with an event tag ``lane_d_[a-z0-9_]+`` (else ``untagged <path>:<line>``),
and the tag must occur as a whole word in a string literal of a patterns file,
e.g. cascade-ops ``cascade_ops/events.py`` (else
``uncovered <path>:<line> <tag>``). Exits 1 on any finding.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

import lanedlib

LEVELS = frozenset(("info", "warning", "error", "critical", "exception"))
TAG = re.compile(r"^lane_d_[a-z0-9_]+")


def receiver_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Call):
        return receiver_name(node.func)
    return ""


def literal_start(msg: ast.AST) -> str | None:
    if isinstance(msg, ast.Constant) and isinstance(msg.value, str):
        return msg.value
    if isinstance(msg, ast.JoinedStr):
        first = msg.values[0] if msg.values else None
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            return first.value
        return None
    if isinstance(msg, ast.BinOp) and isinstance(msg.op, ast.Mod):
        return literal_start(msg.left) if isinstance(msg.left, ast.Constant) else None
    return None


def log_calls(tree: ast.AST) -> list[tuple[int, ast.AST | None]]:
    calls = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr not in LEVELS:
            continue
        if "log" not in receiver_name(node.func.value).lower():
            continue
        msg = node.args[0] if node.args else next(
            (kw.value for kw in node.keywords if kw.arg == "msg"), None)
        calls.append((node.lineno, msg))
    return sorted(calls, key=lambda c: c[0])


def pattern_literals(root: Path, files: list[str]) -> list[str]:
    literals: list[str] = []
    for f in files:
        path = lanedlib.resolve(root, f)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        literals.extend(n.value for n in ast.walk(tree)
                        if isinstance(n, ast.Constant) and isinstance(n.value, str))
    return literals


def covered(tag: str, literals: list[str]) -> bool:
    word = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(tag) + r"(?![A-Za-z0-9_])")
    return any(word.search(s) for s in literals)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=".")
    ap.add_argument("--source", action="append", required=True)
    ap.add_argument("--patterns", action="append", required=True)
    args = ap.parse_args(argv)
    root = Path(args.root)
    literals = pattern_literals(root, args.patterns)
    findings: list[str] = []
    n_calls = 0
    for src in args.source:
        for path in lanedlib.walk_files(root, lanedlib.resolve(root, src)):
            if path.suffix != ".py":
                continue
            try:
                rel = path.resolve().relative_to(root.resolve()).as_posix()
            except ValueError:
                rel = path.as_posix()
            if "lane_d" not in rel:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
            for lineno, msg in log_calls(tree):
                n_calls += 1
                start = literal_start(msg) if msg is not None else None
                m = TAG.match(start) if start is not None else None
                if m is None:
                    findings.append(f"untagged {rel}:{lineno}")
                elif not covered(m.group(0), literals):
                    findings.append(f"uncovered {rel}:{lineno} {m.group(0)}")
    for line in findings:
        print(line)
    print(f"follower_patterns: {len(findings)} finding(s) over {n_calls} lane D log call(s)",
          file=sys.stderr)
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
