#!/usr/bin/env python3
"""Unread-key check (plan §3.1 build rule 4).

    unread_keys.py --root R --config FILE [--config ...] --source DIR [--source ...]

Every key of the ``[lane_d]`` and ``[lane_d_ops]`` tables in each config must be
read by some ``.py`` file under a ``--source`` directory. A read is the key as a
string constant in a Load subscript (``raw["k"]``), as the first argument of
``.get``/``.pop``/``.setdefault``, or as the left operand of ``in``. Mentions in
an ``ALLOWED`` tuple, docstrings, comments and dict-literal keys don't count.
Prints ``unread <table>.<key>`` per unread key and exits 1 when there is any.
"""

from __future__ import annotations

import argparse
import ast
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # `python -P` drops the script dir
import lanedlib  # noqa: E402

TABLES = ("lane_d", "lane_d_ops")
READ_METHODS = frozenset(("get", "pop", "setdefault"))


def _str_const(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def reads_in(tree: ast.AST) -> set[str]:
    keys: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load):
            key = _str_const(node.slice)
            if key is not None:
                keys.add(key)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr in READ_METHODS and node.args:
            key = _str_const(node.args[0])
            if key is not None:
                keys.add(key)
        elif isinstance(node, ast.Compare) and node.ops and isinstance(node.ops[0], ast.In):
            key = _str_const(node.left)
            if key is not None:
                keys.add(key)
    return keys


def collect_reads(root: Path, sources: list[str]) -> set[str]:
    keys: set[str] = set()
    for src in sources:
        for path in lanedlib.walk_files(root, lanedlib.resolve(root, src)):
            if path.suffix != ".py":
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except (SyntaxError, UnicodeDecodeError) as exc:
                print(f"unparseable {path}: {exc}", file=sys.stderr)
                continue
            keys |= reads_in(tree)
    return keys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=".")
    ap.add_argument("--config", action="append", required=True)
    ap.add_argument("--source", action="append", required=True)
    args = ap.parse_args(argv)
    root = Path(args.root)
    reads = collect_reads(root, args.source)
    unread: list[str] = []
    checked = 0
    for cfg in args.config:
        with open(lanedlib.resolve(root, cfg), "rb") as fh:
            data = tomllib.load(fh)
        for table in TABLES:
            section = data.get(table)
            if not isinstance(section, dict):
                continue
            for key in section:
                checked += 1
                name = f"{table}.{key}"
                if key not in reads and name not in unread:
                    unread.append(name)
    for name in unread:
        print(f"unread {name}")
    print(f"unread_keys: {len(unread)} unread of {checked} key(s)", file=sys.stderr)
    return 1 if unread else 0


if __name__ == "__main__":
    sys.exit(main())
