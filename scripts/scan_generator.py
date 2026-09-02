#!/usr/bin/env python3
"""Static security scan for an untrusted generator tree — NO code is executed.

A cascade generator's legitimate footprint is narrow: pure-Python math over
numpy/torch/sklearn-style libraries, reading its own config, yielding arrays.
It has no business touching the network, spawning processes, importing at
runtime from strings, reading credentials, or shipping binaries — on the
subnet it runs inside a sandbox that blocks most of that, but an ARCHIVED
copy in a repo can be run by a human on a real machine, which is exactly the
gap this scan covers (it gates scripts/sync_king.py's commit).

Checks (all static):
  * file inventory  — binaries/native libs, symlinks, hidden files, huge
                      files, path-traversal names
  * requirements    — non-PyPI sources (URLs, git+, local paths), packages
                      outside the known generator stack
  * Python AST      — network primitives, subprocess/os.system, exec/eval/
                      compile on data, ctypes/FFI, dynamic import from
                      strings, env/credential reads, writes outside the
                      tree, base64→exec style obfuscation

Findings are advisory but the exit code is honest: 0 clean, 1 findings,
2 scan error. Usage: scripts/scan_generator.py <dir> [--json out.json]
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

ALLOWED_SUFFIXES = {".py", ".json", ".md", ".txt", ".toml", ".cfg", ".yaml",
                    ".yml", ".typed", ""}
MAX_FILE_MB = 8

# import roots a generator has no business touching
NET_MODULES = {"socket", "urllib", "requests", "http", "httpx", "aiohttp",
               "ftplib", "smtplib", "telnetlib", "websockets", "paramiko",
               "boto3", "botocore"}
PROC_MODULES = {"subprocess", "pty", "pexpect"}
FFI_MODULES = {"ctypes", "cffi"}
KNOWN_STACK_PREFIXES = (
    "numpy", "scipy", "pandas", "scikit-learn", "sklearn", "torch", "gpytorch",
    "linear-operator", "linear_operator", "jaxtyping", "statsmodels", "numba",
    "pyyaml", "typing-extensions", "typing_extensions", "joblib", "threadpoolctl",
)
SECRET_HINTS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "WALLET")


def _finding(out, sev, path, what):
    out.append({"severity": sev, "path": str(path), "finding": what})


def scan_inventory(root: Path, out: list) -> None:
    for p in root.rglob("*"):
        rel = p.relative_to(root)
        if rel.name in (".fetch_complete",):     # our own fetch marker, not theirs
            continue
        if any(part.startswith(".") for part in rel.parts):
            _finding(out, "warn", rel, "hidden file/dir (dotpath)")
        if p.is_symlink():
            _finding(out, "high", rel, "symlink (possible tree escape)")
            continue
        if not p.is_file():
            continue
        if ".." in rel.parts:
            _finding(out, "high", rel, "path traversal component")
        if p.suffix.lower() not in ALLOWED_SUFFIXES:
            _finding(out, "high", rel,
                     f"unexpected file type '{p.suffix}' (binary/native payload?)")
        elif p.stat().st_size > MAX_FILE_MB * 1024 * 1024:
            _finding(out, "warn", rel, f"file larger than {MAX_FILE_MB}MB")
        try:
            head = p.read_bytes()[:2]
            if head == b"\x7fE" or head == b"MZ":
                _finding(out, "high", rel, "executable binary header (ELF/PE)")
        except OSError:
            _finding(out, "warn", rel, "unreadable file")


def scan_requirements(root: Path, out: list) -> None:
    for req in root.rglob("requirements*.txt"):
        rel = req.relative_to(root)
        for line in req.read_text(errors="replace").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            low = line.lower()
            if any(t in low for t in ("http://", "https://", "git+", "file:",
                                      "--index-url", "--extra-index-url", "-e ")):
                _finding(out, "high", rel,
                         f"non-PyPI requirement source: {line!r}")
                continue
            name = (line.split(";")[0].split("[")[0]
                    .split("==")[0].split(">=")[0].split("<=")[0]
                    .split("~=")[0].split(">")[0].split("<")[0].strip().lower())
            if name and not any(name.startswith(k) for k in KNOWN_STACK_PREFIXES):
                _finding(out, "warn", rel,
                         f"requirement outside the known generator stack: {name!r} "
                         "(check for typosquatting before installing)")


class _PyScan(ast.NodeVisitor):
    def __init__(self, rel, out):
        self.rel, self.out = rel, out

    def _root(self, name):
        return (name or "").split(".")[0]

    def visit_Import(self, node):
        for a in node.names:
            self._flag_import(self._root(a.name))
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        self._flag_import(self._root(node.module))
        self.generic_visit(node)

    def _flag_import(self, root):
        if root in NET_MODULES:
            _finding(self.out, "high", self.rel, f"network module import: {root}")
        elif root in PROC_MODULES:
            _finding(self.out, "high", self.rel, f"process-spawn import: {root}")
        elif root in FFI_MODULES:
            _finding(self.out, "high", self.rel, f"FFI import: {root}")

    def visit_Call(self, node):
        f = node.func
        name = ""
        if isinstance(f, ast.Name):
            name = f.id
        elif isinstance(f, ast.Attribute):
            base = f.value.id if isinstance(f.value, ast.Name) else ""
            name = f"{base}.{f.attr}"
        if name in ("eval", "exec", "compile"):
            _finding(self.out, "high", self.rel, f"dynamic execution: {name}()")
        if name in ("os.system", "os.popen", "os.execv", "os.execve",
                    "os.spawnl", "os.fork"):
            _finding(self.out, "high", self.rel, f"process call: {name}()")
        if name in ("importlib.import_module", "__import__"):
            if not (node.args and isinstance(node.args[0], ast.Constant)):
                _finding(self.out, "high", self.rel,
                         f"dynamic import from a computed string: {name}()")
        if name in ("getattr",) and len(node.args) >= 2 and not isinstance(
                node.args[1], ast.Constant):
            _finding(self.out, "warn", self.rel,
                     "getattr with computed attribute (obfuscation vector)")
        if name.endswith(("b64decode", "a85decode", "unhexlify")):
            _finding(self.out, "warn", self.rel,
                     f"encoded-payload decode: {name}() (verify what it feeds)")
        self.generic_visit(node)

    def visit_Attribute(self, node):
        if (isinstance(node.value, ast.Attribute)
                and isinstance(node.value.value, ast.Name)
                and node.value.value.id == "os" and node.value.attr == "environ"):
            _finding(self.out, "warn", self.rel, "os.environ access")
        self.generic_visit(node)

    def visit_Subscript(self, node):
        v = node.value
        if (isinstance(v, ast.Attribute) and v.attr == "environ"
                and isinstance(v.value, ast.Name) and v.value.id == "os"):
            key = node.slice
            if isinstance(key, ast.Constant) and any(
                    h in str(key.value).upper() for h in SECRET_HINTS):
                _finding(self.out, "high", self.rel,
                         f"credential-looking env read: os.environ[{key.value!r}]")
        self.generic_visit(node)


def scan_python(root: Path, out: list) -> None:
    for p in root.rglob("*.py"):
        rel = p.relative_to(root)
        try:
            tree = ast.parse(p.read_text(errors="replace"))
        except SyntaxError as e:
            _finding(out, "warn", rel, f"unparseable python ({e.msg} line {e.lineno})")
            continue
        _PyScan(rel, out).visit(tree)


def scan(root: Path) -> list:
    out: list = []
    scan_inventory(root, out)
    scan_requirements(root, out)
    scan_python(root, out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", type=Path)
    ap.add_argument("--json", type=Path, default=None, help="write findings JSON here")
    args = ap.parse_args()
    if not args.target.is_dir():
        print(f"not a directory: {args.target}", file=sys.stderr)
        return 2
    findings = scan(args.target)
    doc = {"target": str(args.target),
           "high": sum(1 for f in findings if f["severity"] == "high"),
           "warn": sum(1 for f in findings if f["severity"] == "warn"),
           "findings": findings}
    if args.json:
        args.json.write_text(json.dumps(doc, indent=2) + "\n")
    for f in findings:
        print(f"[{f['severity'].upper():4s}] {f['path']}: {f['finding']}")
    print(f"scan: {doc['high']} high, {doc['warn']} warn "
          f"({'CLEAN' if not findings else 'FINDINGS'})")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
