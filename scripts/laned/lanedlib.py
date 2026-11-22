"""Shared helpers for the lane D loop tooling (plan §3 conventions, §17).

Imported by ``accept``, ``status`` and the scanners, which all live next to
this file and are run as ``python scripts/laned/<tool>``; Python puts the
script's directory first on ``sys.path``, so ``import lanedlib`` resolves here.
Stdlib only.
"""

from __future__ import annotations

import fnmatch
import json
import re
import subprocess
from pathlib import Path

GLOB_CHARS = ("*", "?", "[")
PRUNE_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".mypy_cache",
              ".pytest_cache", ".ruff_cache"}


def wp_dir(wp: str) -> str:
    """WP id -> its tests/accept directory name (``W1.5`` -> ``w1_5``)."""
    return wp.lower().replace(".", "_").replace("-", "_")


def resolve(root: Path, path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p


# ---------------------------------------------------------------------------
# PATHS files
# ---------------------------------------------------------------------------

def parse_paths(text: str) -> list[str]:
    entries = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        entries.append(line)
    return entries


def path_matches(rel: str, entries: list[str]) -> bool:
    """True when the repo-relative path ``rel`` is matched by a PATHS entry."""
    while rel.startswith("./"):
        rel = rel[2:]
    for entry in entries:
        if any(c in entry for c in GLOB_CHARS):
            if fnmatch.fnmatchcase(rel, entry):
                return True
        elif entry.endswith("/"):
            if rel.startswith(entry) or rel + "/" == entry:
                return True
        elif rel == entry:
            return True
    return False


def module_in_paths(module: str, entries: list[str]) -> bool:
    """True when module ``a.b`` maps to ``a/b.py`` or ``a/b/`` under PATHS."""
    base = module.replace(".", "/")
    candidates = (base + ".py", base + "/", base + "/__init__.py")
    return any(path_matches(c, entries) for c in candidates)


def walk_files(root: Path, start: Path) -> list[Path]:
    """Every regular file under ``start`` (pruning VCS/venv/cache dirs)."""
    if start.is_file():
        return [start]
    out: list[Path] = []
    if not start.is_dir():
        return out
    for path in sorted(start.iterdir()):
        if path.is_dir():
            if path.name in PRUNE_DIRS:
                continue
            out.extend(walk_files(root, path))
        elif path.is_file():
            out.append(path)
    return out


def expand_entries(root: Path, entries: list[str]) -> list[Path]:
    """Files under ``root`` selected by PATHS-style entries or CLI paths."""
    selected: dict[str, Path] = {}
    all_files: list[Path] | None = None
    for entry in entries:
        if any(c in entry for c in GLOB_CHARS):
            if all_files is None:
                all_files = walk_files(root, root)
            for f in all_files:
                rel = f.relative_to(root).as_posix()
                if fnmatch.fnmatchcase(rel, entry):
                    selected[rel] = f
            continue
        target = resolve(root, entry)
        for f in walk_files(root, target):
            try:
                rel = f.resolve().relative_to(root.resolve()).as_posix()
            except ValueError:
                rel = f.as_posix()
            selected[rel] = f
    return [selected[k] for k in sorted(selected)]


# ---------------------------------------------------------------------------
# Evidence verification (shared by accept --evidence, accept --gate, status)
# ---------------------------------------------------------------------------

OPEN_BLOCKING = re.compile(r"^\s*[-*]\s*\[ \]\s*BLOCKING\b(.*)$", re.M)


def check_json_evidence(text: str | None, wp: str) -> tuple[bool, str]:
    if text is None:
        return False, "evidence file missing"
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        return False, f"evidence is not JSON: {exc}"
    if not isinstance(data, dict):
        return False, "evidence is not a JSON object"
    if data.get("wp") != wp:
        return False, f"evidence wp={data.get('wp')!r}, expected {wp!r}"
    if data.get("passed") is not True:
        return False, "evidence does not record passed: true"
    return True, "ok"


def open_blocking_items(text: str) -> list[str]:
    return [m.group(0).strip() for m in OPEN_BLOCKING.finditer(text)]


def check_review_evidence(text: str | None) -> tuple[bool, str]:
    if text is None:
        return False, "review file missing"
    items = open_blocking_items(text)
    if items:
        return False, "open BLOCKING items: " + "; ".join(items)
    return True, "ok"


def check_evidence(text: str | None, name: str, wp: str) -> tuple[bool, str]:
    if name.endswith(".md"):
        return check_review_evidence(text)
    return check_json_evidence(text, wp)


def check_gate(text: str | None, stage: str) -> tuple[bool, str]:
    if text is None:
        return False, "gate evidence missing"
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        return False, f"gate evidence is not JSON: {exc}"
    if not isinstance(data, dict):
        return False, "gate evidence is not a JSON object"
    if data.get("stage") != stage:
        return False, f"gate evidence stage={data.get('stage')!r}, expected {stage!r}"
    checks = data.get("checks")
    if not isinstance(checks, list) or not checks:
        return False, "gate evidence records no checks"
    failed = [c.get("name", "?") if isinstance(c, dict) else repr(c)
              for c in checks if not (isinstance(c, dict) and c.get("passed") is True)]
    if failed:
        return False, "gate checks not passed: " + ", ".join(map(str, failed))
    return True, "ok"


def gate_path(stage: str) -> str:
    return f"evidence/gates/{stage.lower()}.json"


# ---------------------------------------------------------------------------
# Plan Accept cells (§3.0): the evidence files a WP declares
# ---------------------------------------------------------------------------

PLAN_PATH = "docs/plans/LANE_D_PLAN.md"
EVIDENCE_ARG = re.compile(r"--evidence\s+([^\s`)|]+)")


def _plan_cell(text: str) -> str:
    return text.replace("**", "").replace("`", "").strip()


def _plan_row(line: str) -> list[str]:
    body = line.strip()
    if body.startswith("|"):
        body = body[1:]
    if body.endswith("|"):
        body = body[:-1]
    return [c.strip() for c in body.split("|")]


def accept_evidence_files(plan_text: str, wp: str) -> list[str]:
    """Every ``--evidence <file>`` in the Accept cell of ``wp``'s row in a §3.0
    WP table (first header cell ``WP``, with ``Runs in`` and ``Accept``
    columns). The first row naming the WP wins, as on the status board."""
    lines = plan_text.splitlines()
    i = 0
    while i < len(lines):
        if not lines[i].lstrip().startswith("|"):
            i += 1
            continue
        block = []
        while i < len(lines) and lines[i].lstrip().startswith("|"):
            block.append(lines[i])
            i += 1
        header = [_plan_cell(c) for c in _plan_row(block[0])]
        if not header or header[0] != "WP" or "Runs in" not in header or "Accept" not in header:
            continue
        k = header.index("Accept")
        for raw in block[2:]:
            row = _plan_row(raw)
            if row and _plan_cell(row[0]) == wp:
                # Accept is the last column in §3.0; a `a|b` inside it must not
                # cut it short, so everything from its index on is the cell.
                accept = " | ".join(row[k:]) if k == len(header) - 1 else (row[k] if k < len(row) else "")
                return list(dict.fromkeys(m.group(1) for m in EVIDENCE_ARG.finditer(accept)))
    return []


# ---------------------------------------------------------------------------
# Token matching of WP / decision ids in plan cells
# ---------------------------------------------------------------------------

def token_re(token: str) -> re.Pattern[str]:
    """Whole-token matcher: ``W0`` never matches ``W0c`` or ``W1`` ``W1.5``."""
    return re.compile(r"(?<![A-Za-z0-9_.\-])" + re.escape(token) + r"(?![A-Za-z0-9_\-]|\.[A-Za-z0-9])")


def has_token(text: str, token: str) -> bool:
    return bool(token_re(token).search(text))


# ---------------------------------------------------------------------------
# git
# ---------------------------------------------------------------------------

def git(root: Path, *args: str, check: bool = True,
        env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    proc = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, env=env)
    if check and proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc


def first_parent_history(root: Path, ref: str) -> list[tuple[str, list[str]]]:
    """``[(sha, changed files)]`` along ``ref``'s first-parent line, oldest first."""
    log = git(root, "log", "--first-parent", "--reverse", "--diff-merges=first-parent",
              "--name-only", "--format=%x01%H", ref).stdout
    history: list[tuple[str, list[str]]] = []
    for chunk in log.split("\x01")[1:]:
        lines = [ln for ln in chunk.split("\n") if ln.strip()]
        history.append((lines[0].strip(), lines[1:]))
    return history


def impl_commit_index(history: list[tuple[str, list[str]]], wp: str, paths: list[str]) -> int | None:
    """The L2 commit (plan §17.1): the first commit after the one that added
    the WP's PATHS file that changes something outside tests/accept/ and only
    paths inside PATHS, i.e. the WP's implementation merge. None when absent."""
    rel = f"tests/accept/{wp_dir(wp)}/PATHS"
    start = next((i for i, (_, files) in enumerate(history) if rel in files), None)
    if start is None:
        return None
    for i in range(start + 1, len(history)):
        files = history[i][1]
        if not files or all(f.startswith("tests/accept/") for f in files):
            continue
        if all(path_matches(f, paths) for f in files):
            return i
    return None
