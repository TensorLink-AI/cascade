#!/usr/bin/env python3
"""Incident lint: every incident a pattern check could have caught becomes a rule.

Run offline from the checkout (stdlib only, no cascade imports, well under a
second):

    python3 scripts/check_incident_rules.py            # whole repo, exit 1 on a hit
    python3 scripts/check_incident_rules.py --list     # the rules and their incidents
    python3 scripts/check_incident_rules.py --rule IR001 docs/VALIDATOR.md

Each rule is written down in docs/INCIDENT_RULES.md: the incident (date,
issue/PR, what broke), the rule, how to fix a hit and the process for adding a
rule. A justified exception carries an allowlist comment on the hit's line or
the line above it, with a reason:

    # incident-ok: IR004 <why this one is safe>
    <!-- incident-ok: IR001 <why> -->        (Markdown)

An allowlist comment without a reason, or naming an unknown rule, is itself a
hit (IR000).
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import re
import shlex
import subprocess
import sys
import tomllib
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Never scanned: tests and fixtures quote bad patterns on purpose; the plan,
# evidence and archive record history verbatim; the rule book and this checker
# spell out every pattern they forbid.
GLOBAL_EXCLUDE = (
    "tests/*", "evidence/*", "docs/plans/*", "docs/archive/*",
    "docs/INCIDENT_RULES.md", "scripts/check_incident_rules.py",
)
SKIP_DIRS = {".git", ".venv", "node_modules", "__pycache__", ".mypy_cache", ".ruff_cache"}

ALLOW_RE = re.compile(r"incident-ok:\s*(?P<rule>[A-Za-z]+\d*)?(?P<reason>[^\n]*)")
SHELL_LANGS = {"", "sh", "bash", "shell", "console", "zsh", "shell-session"}


@dataclass(frozen=True)
class Hit:
    rule: str
    path: str
    line: int
    message: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.rule} {self.message}"


@dataclass(frozen=True)
class Rule:
    id: str
    title: str
    incident: str
    fix: str
    globs: tuple[str, ...]
    check_file: Callable[[str, str], list[tuple[int, str]]] | None = None
    check_tree: Callable[[dict[str, str]], list[tuple[str, int, str]]] | None = None
    exclude: tuple[str, ...] = field(default=())


# ── text contexts ──────────────────────────────────────────────────────────────

def kind_of(rel: str) -> str:
    name = rel.rsplit("/", 1)[-1]
    if name.endswith((".sh", ".bash")):
        return "shell"
    if name.endswith((".service", ".timer")):
        return "unit"
    if name.endswith(".py"):
        return "python"
    if name.endswith(".md"):
        return "markdown"
    if name.endswith(".toml"):
        return "toml"
    if name == "Makefile" or name.startswith("Dockerfile"):
        return "shell"
    return "other"


def _strip_shell_comment(line: str) -> str:
    """Drop a trailing `` # comment`` that sits outside quotes."""
    single = double = False
    for i, ch in enumerate(line):
        if ch == "'" and not double:
            single = not single
        elif ch == '"' and not single:
            double = not double
        elif ch == "#" and not single and not double and (i == 0 or line[i - 1].isspace()):
            return line[:i]
    return line


def _logical(lines: list[tuple[int, str]]) -> Iterator[tuple[int, str]]:
    """Join backslash continuations; yield (first line number, joined text)."""
    buf: list[str] = []
    start = 0
    for no, raw in lines:
        if not buf:
            start = no
        if raw.rstrip().endswith("\\"):
            buf.append(raw.rstrip()[:-1])
            continue
        buf.append(raw)
        yield start, " ".join(buf)
        buf = []
    if buf:
        yield start, " ".join(buf)


def markdown_fences(text: str) -> Iterator[tuple[int, str]]:
    """Lines inside shell-ish fenced code blocks, continuations joined."""
    inside = False
    marker = ""
    shellish = False
    block: list[tuple[int, str]] = []
    for no, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if not inside:
            m = re.match(r"^(`{3,}|~{3,})\s*([\w+-]*)", stripped)
            if m:
                inside, marker = True, m.group(1)
                shellish = m.group(2).lower() in SHELL_LANGS
                block = []
            continue
        if stripped.startswith(marker) and stripped.strip("`~") == "":
            inside = False
            if shellish:
                body = [(n, s) for n, s in block if not s.lstrip().startswith("#")]
                yield from _logical([(n, _strip_shell_comment(s)) for n, s in body])
            continue
        block.append((no, line))


def markdown_paragraphs(text: str) -> Iterator[tuple[int, str]]:
    """Prose paragraphs (outside fences): (first line number, text with newlines)."""
    inside = False
    marker = ""
    para: list[str] = []
    start = 0
    for no, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if inside:
            if stripped.startswith(marker) and stripped.strip("`~") == "":
                inside = False
            continue
        m = re.match(r"^(`{3,}|~{3,})", stripped)
        if m or not stripped:
            if para:
                yield start, "\n".join(para)
                para = []
            if m:
                inside, marker = True, m.group(1)
            continue
        if not para:
            start = no
        para.append(line)
    if para:
        yield start, "\n".join(para)


def command_lines(rel: str, text: str) -> Iterator[tuple[int, str]]:
    """The executable text of a file: shell/unit logical lines without comments,
    Python non-comment lines, Markdown shell fences. Other kinds yield nothing."""
    kind = kind_of(rel)
    lines = list(enumerate(text.splitlines(), 1))
    if kind in ("shell", "unit"):
        body = [(n, _strip_shell_comment(s) if kind == "shell" else s)
                for n, s in lines if not s.lstrip().startswith(("#", ";"))]
        yield from _logical(body)
    elif kind == "python":
        for n, s in lines:
            if not s.lstrip().startswith("#"):
                yield n, s
    elif kind == "markdown":
        yield from markdown_fences(text)


# ── IR001: bare `uv sync --frozen` ─────────────────────────────────────────────

# `uv sync` where a command starts: line start, after a shell separator or an
# opening quote/backtick/`$ ` prompt, after env assignments, under any path.
UV_SYNC_CMD = re.compile(
    r"(?:^|&&|\|\||[;|(`\"']|\$\(|^\s*\$)\s*(?:sudo\s+)?(?:env\s+)?"
    r"(?:[A-Za-z_][A-Za-z0-9_]*=\S*\s+)*(?:[\w.~/${}:-]*/)?uv\s+sync\b")
# Python: only a real command string (after a separator or the opening quote).
UV_SYNC_CMD_PY = re.compile(r"(?:&&|\|\||[;|\"'])\s*(?:[\w.~/${}:-]*/)?uv\s+sync\b")
UV_SEGMENT_END = re.compile(r"&&|\|\||[;|)`\"']")
UV_KEEPS_EXTRAS = re.compile(r"--extra\b|--all-extras\b|--inexact\b|--group\b|\{|\$")
UV_INLINE = re.compile(r"^\s*(?:[A-Za-z_][A-Za-z0-9_]*=\S*\s+)*(?:[\w.~/-]*/)?uv\s+sync\b")


def _bare_frozen_syncs(text: str, python: bool = False) -> list[str]:
    out = []
    for m in (UV_SYNC_CMD_PY if python else UV_SYNC_CMD).finditer(text):
        rest = text[m.end():]
        end = UV_SEGMENT_END.search(rest)
        seg = "uv sync" + (rest[:end.start()] if end else rest)
        if "--frozen" in seg and not UV_KEEPS_EXTRAS.search(seg):
            out.append(" ".join(seg.split()))
    return out


def check_ir001(rel: str, text: str) -> list[tuple[int, str]]:
    hits = [(no, f"bare `{seg}` drops every installed extra (bittensor, torch, ...)")
            for no, line in command_lines(rel, text)
            for seg in _bare_frozen_syncs(line, kind_of(rel) == "python")]
    if kind_of(rel) == "markdown":
        for start, para in markdown_paragraphs(text):
            if re.search(r"extra|--inexact", para, re.I):
                continue  # the paragraph says which extras to keep
            for m in re.finditer(r"(?<!`)`([^`]+)`(?!`)", para):
                span = " ".join(m.group(1).split())
                if UV_INLINE.match(span) and "--frozen" in span and not UV_KEEPS_EXTRAS.search(span):
                    hits.append((start + para.count("\n", 0, m.start()),
                                 f"instruction `{span}` names no extras: a bare frozen sync "
                                 "uninstalls them"))
    return hits


# ── IR002: `python -m` without `-P` ────────────────────────────────────────────

PY_DASH_M = re.compile(
    r"(?<![\w.-])(?:[\w.~/${}-]*/)?python(?:3(?:\.\d+)?)?((?:\s+-[A-Za-z]+)*)\s+-m\s+([\w.]+)")


def check_ir002(rel: str, text: str) -> list[tuple[int, str]]:
    hits = []
    for no, line in command_lines(rel, text):
        for m in PY_DASH_M.finditer(line):
            flags = m.group(1).split()
            if not any(f[1:].isalpha() and ("P" in f or "I" in f) for f in flags):
                hits.append((no, f"`python -m {m.group(2)}` without -P: the working directory "
                                 "shadows the release on sys.path"))
    return hits


# ── IR003: `pgrep -f` / `pkill -f` with a literal pattern ──────────────────────

PROC_TOOL = re.compile(r"(?<![\w-])(pgrep|pkill)(?![\w-])")
PROC_OPT_WITH_ARG = {"-u", "-U", "-g", "-G", "-P", "-s", "-t", "-F", "--signal",
                     "--pidfile", "--euid", "--uid", "--group", "--parent", "--session",
                     "--terminal", "--ns", "--nslist"}
PROC_SEGMENT_END = re.compile(r"&&|\|\||[;|)`<>]|\d>")


def _proc_pattern(rest: str) -> tuple[bool, str | None]:
    """(uses -f, the pattern argument or None) for the text after pgrep/pkill."""
    end = PROC_SEGMENT_END.search(rest)
    seg = rest[:end.start()] if end else rest
    try:
        toks = shlex.split(seg)
    except ValueError:
        toks = [t.strip("\"'") for t in seg.split()]
    full = False
    i = 0
    while i < len(toks):
        tok = toks[i]
        if tok == "--":
            return full, toks[i + 1] if i + 1 < len(toks) else None
        if tok in ("-f", "--full"):
            full = True
        elif tok in PROC_OPT_WITH_ARG:
            i += 1
        elif re.fullmatch(r"-[a-z]+", tok):
            full = full or "f" in tok
        elif not tok.startswith("-"):
            return full, tok
        i += 1
    return full, None


def check_ir003(rel: str, text: str) -> list[tuple[int, str]]:
    hits = []
    for no, line in command_lines(rel, text):
        for m in PROC_TOOL.finditer(line):
            full, pat = _proc_pattern(line[m.end():])
            if not full or not pat or pat.startswith(("$", "{")) or "[" in pat:
                continue
            hits.append((no, f"`{m.group(1)} -f {pat}` also matches the invoking shell "
                             "(its command line contains the pattern): bracket one character "
                             f"(`{_bracketed(pat)}`) or use a pid file"))
    return hits


def _bracketed(pat: str) -> str:
    for i, ch in enumerate(pat):
        if ch.isalnum() or ch in "-_":
            return f"{pat[:i]}[{ch}]{pat[i + 1:]}"
    return pat


# ── IR004: unpinned ssh host keys ──────────────────────────────────────────────

HK_NO = re.compile(r"StrictHostKeyChecking\s*=\s*no\b")
HK_DEVNULL = re.compile(r"UserKnownHostsFile\s*=\s*/dev/null")
HK_ACCEPT_NEW = re.compile(r"StrictHostKeyChecking\s*=\s*accept-new\b")


def check_ir004(rel: str, text: str) -> list[tuple[int, str]]:
    hits = []
    for no, line in command_lines(rel, text):
        if HK_NO.search(line) or HK_DEVNULL.search(line):
            hits.append((no, "host key not verified at all (StrictHostKeyChecking=no / "
                             "UserKnownHostsFile=/dev/null): pin the pod's scanned key"))
        elif HK_ACCEPT_NEW.search(line) and "UserKnownHostsFile" not in line:
            hits.append((no, "accept-new against the shared ~/.ssh/known_hosts: a reused "
                             "ip:port with a fresh host key fails verification; pin a "
                             "per-pod UserKnownHostsFile"))
    return hits


# ── IR005: bulk git staging ────────────────────────────────────────────────────

GIT_BULK = re.compile(
    r"\bgit\s+(?:-[Cc]\s+\S+\s+)*(?:"
    r"add\s+(?:[^;&|\n]*?\s)?(?:-u|-A|--all|--update)(?=\s|$|[;&|])"
    r"|add\s+\.(?=\s|$|[;&|])"
    r"|commit\s+(?:[^;&|\n]*?\s)?(?:-[A-Za-z]*a[A-Za-z]*|--all)(?=\s|$|[;&|]))")


def check_ir005(rel: str, text: str) -> list[tuple[int, str]]:
    return [(no, f"`{m.group(0)}` stages every modified tracked file, live overrides "
                 "included (chain.toml): stage explicit paths")
            for no, line in command_lines(rel, text) for m in GIT_BULK.finditer(line)]


# ── IR006: Shadeform GPU rentals ───────────────────────────────────────────────

GPU_KEYS = ("gpus", "gpus_per_pod", "num_gpus", "gpu_count")


def _shadeform_gpu_tables(node, path=()) -> Iterator[tuple[tuple[str, ...], str]]:
    if isinstance(node, list):
        for item in node:
            yield from _shadeform_gpu_tables(item, path)
        return
    if not isinstance(node, dict):
        return
    providers = node.get("providers", ())
    provider = node.get("provider", "")
    names = [str(p).lower() for p in (providers if isinstance(providers, list) else ())]
    names.append(str(provider).lower())
    gpus = [node[k] for k in GPU_KEYS if isinstance(node.get(k), int | float)]
    if "shadeform" in names and any(g > 0 for g in gpus):
        yield path, "providers" if "providers" in node else "provider"
    for k, v in node.items():
        yield from _shadeform_gpu_tables(v, (*path, k))


def check_ir006(rel: str, text: str) -> list[tuple[int, str]]:
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        return []
    lines = text.splitlines()
    hits = []
    for path, key in _shadeform_gpu_tables(data):
        table = ".".join(path)
        no = _toml_key_line(lines, table, key)
        hits.append((no, f"[{table}] rents a GPU (gpus > 0) with Shadeform as a provider: "
                         "GPUs are Lium only, Shadeform is CPU only"))
    return hits


def _toml_key_line(lines: list[str], table: str, key: str) -> int:
    header = re.compile(r"^\s*\[{1,2}\s*" + re.escape(table) + r"\s*\]{1,2}")
    start = next((i for i, s in enumerate(lines) if header.match(s)), -1) if table else -1
    for i in range(start + 1, len(lines)):
        if i > start + 1 and lines[i].lstrip().startswith("["):
            break
        if re.match(r"^\s*" + re.escape(key) + r"\s*=", lines[i]):
            return i + 1
    return start + 1 if start >= 0 else 1


# ── IR007: config dataclass fields the loader never parses ─────────────────────

# Fields set at runtime, never read from chain.toml (each says so in a comment
# or docstring at its definition). A new runtime-only field takes an inline
# `# incident-ok: IR007 runtime-only: <who sets it>` instead of a line here.
IR007_RUNTIME_ONLY = {
    "GeneratorConfig.real_corpus_dir": "RUNTIME-resolved local corpus path",
    "GeneratorConfig.curriculum_stage": "WB27 runtime-only CLI stage",
    "ActivationConfig.resolved_block": "written by apply_activation",
    "ActivationConfig.resolved_forfeit_block": "written by apply_forfeit_activation",
    "ActivationConfig.resolved_margin_v2_block": "written by apply_margin_v2_activation",
    "ActivationConfig.resolved_prelaunch_burn_block": "written by apply_prelaunch_burn_activation",
    "ActivationConfig.resolved_image_decoupled_block": "written by apply_image_decoupled_activation",
    "ActivationConfig.resolved_shared_ladder_block": "written by apply_shared_ladder_activation",
    "NetConfigConfig.seq": "set by netconfig.build_config",
    "NetConfigConfig.digest": "set by netconfig.build_config",
    "NetConfigConfig.applied_block": "set by netconfig.build_config",
    "TrainingContractConfig.lane_b_lr_scale": "set by lane_b_levers.apply_levers",
    "TrainingContractConfig.curriculum": "set by lane_b_levers.apply_levers",
    "TrainingContractConfig.lane_b_stop_record": "set by lane_b_leg for a limits leg",
    "TrainingContractConfig.router_slow": "set by the worker's --router-slow flag",
}


def _is_dataclass(cls: ast.ClassDef) -> bool:
    for d in cls.decorator_list:
        target = d.func if isinstance(d, ast.Call) else d
        name = target.id if isinstance(target, ast.Name) else getattr(target, "attr", "")
        if name == "dataclass":
            return True
    return False


def check_ir007(files: dict[str, str]) -> list[tuple[str, int, str]]:
    trees = {}
    for rel, text in files.items():
        try:
            trees[rel] = ast.parse(text)
        except SyntaxError:
            continue
    mentioned: set[str] = set()
    generic: set[str] = set()
    for tree in trees.values():
        for n in ast.walk(tree):
            if isinstance(n, ast.Constant) and isinstance(n.value, str):
                mentioned.add(n.value)
            elif isinstance(n, ast.keyword) and n.arg:
                mentioned.add(n.arg)
            elif (isinstance(n, ast.Call) and n.args and isinstance(n.args[0], ast.Name)
                  and getattr(n.func, "id", getattr(n.func, "attr", "")) == "fields"):
                generic.add(n.args[0].id)
            elif (isinstance(n, ast.Attribute) and n.attr == "__dataclass_fields__"
                  and isinstance(n.value, ast.Name)):
                generic.add(n.value.id)
    hits = []
    for rel, tree in trees.items():
        for cls in tree.body:
            if not isinstance(cls, ast.ClassDef) or not _is_dataclass(cls) or cls.name in generic:
                continue
            for st in cls.body:
                if not (isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name)):
                    continue
                name = st.target.id
                if name.startswith("_") or name in mentioned:
                    continue
                if f"{cls.name}.{name}" in IR007_RUNTIME_ONLY:
                    continue
                if "ClassVar" in ast.unparse(st.annotation):
                    continue
                hits.append((rel, st.lineno,
                             f"{cls.name}.{name} is never parsed by the config loader (no "
                             f"\"{name}\" key read, no {name}= kwarg): setting it in chain.toml "
                             "silently does nothing"))
    return hits


# ── IR008: remote python over ssh with no on-host bound ───────────────────────

# A `-m` argument (the remote `python -m <module>` launch), a `timeout`
# command token, a detached launch.
IR008_M = re.compile(r"(?:^|\s)-m(?:\s|$)")
IR008_TIMEOUT = re.compile(r"(?:^|[\s\"'(;&|])timeout(?:\s|$)")
IR008_DETACHED = re.compile(r"\b(?:nohup|setsid)\b")
# Reviewed builders, keyed "<path>::<function>". Each reason says why the
# remote process cannot outlive its caller, or names the open follow-up.
IR008_REVIEWED = {
    "cascade/trainer/remote.py::worker_argv":
        "legs launch detached (run_detached: nohup + kill on its own deadline); the "
        "synchronous path only runs with [round] detached_dispatch = false",
    "cascade/trainer/remote.py::probe_worker_runtime":
        "`worker --help` exits in seconds and starts no work",
    "cascade/validator/validator_bench.py::_vbench_current_data_revisions":
        "bench_identity reads file revisions and exits; no GPU, no long work",
    "cascade/validator/mkt_offload.py::start":
        "a --serve session over stdin: the worker returns at stdin EOF (after its current "
        "frame) once the ssh dies",
    "cascade/validator/lane_d_bench_host.py::__call__":
        "KNOWN VIOLATION pending owner: the lane D bench sweep has the pre-#969 "
        "SSHEvalHost shape (named as a follow-up in #969)",
    "cascade/trainer/r2_transfer.py::relay_command":
        "KNOWN VIOLATION pending owner: the operator R2 relay push (4 h local timeout) "
        "is not bounded on the relay",
}


def _ir008_strings(fn: ast.AST) -> list[str]:
    return [n.value for n in ast.walk(fn) if isinstance(n, ast.Constant) and isinstance(n.value, str)]


def check_ir008(rel: str, text: str) -> list[tuple[int, str]]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    hits = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if f"{rel}::{fn.name}" in IR008_REVIEWED:
            continue
        if not any(isinstance(n, ast.Attribute) and n.attr == "remote_python" for n in ast.walk(fn)):
            continue
        strings = _ir008_strings(fn)
        if not any(IR008_M.search(s) for s in strings):
            continue
        if any(IR008_TIMEOUT.search(s) or IR008_DETACHED.search(s) for s in strings):
            continue
        hits.append((fn.lineno,
                     f"{fn.name}() builds a remote `<remote_python> -m …` command with no on-host "
                     "`timeout`: subprocess.run(timeout=) kills only the local ssh, the remote "
                     "process keeps running and every retry stacks another one"))
    return hits


# ── IR009: an ssh reply's raw stdout parsed as JSON ────────────────────────────

# A pod command whose stdout is parsed as JSON shares that stdout with every
# library the remote process imports: on 2026-11-27 hippius_hub printed
# "Downloading <file> (parallel)..." ahead of the `cat`ed result.json and the
# lane B settlement rebase failed "pod reply is not JSON" (#983). Safe shapes:
# a reply from a separate `cat <file>`-only ssh command, a last-line parse
# (`.splitlines()[-1]`), or a tolerant parser (lane_b_offload.pod_reply_json).
IR009_SSH_CALL = re.compile(r"ssh", re.IGNORECASE)
IR009_CAT_ONLY = re.compile(r"^\s*cat\s")
IR009_CHAIN = re.compile(r"&&|\|\||;|\|")


def _call_name(node: ast.AST) -> str:
    f = node.func if isinstance(node, ast.Call) else None
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return ""


def _first_text(node: ast.AST) -> str | None:
    """The leading literal text of a str constant or an f-string."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        lit = []
        for part in node.values:
            if isinstance(part, ast.Constant) and isinstance(part.value, str):
                lit.append(part.value)
            else:
                lit.append("\0")
        return "".join(lit)
    return None


def _cat_only(value: ast.AST) -> bool:
    """``value`` (an ssh call) carries a literal ``cat <file>`` command and nothing else."""
    for n in ast.walk(value):
        text = _first_text(n)
        if text is not None and IR009_CAT_ONLY.match(text):
            return not IR009_CHAIN.search(text.replace("\0", ""))
    return False


def _stdout_owner(arg: ast.AST) -> str | None:
    """``X`` when ``arg`` is ``X.stdout`` or ``X.stdout.strip()`` (no last-line pick)."""
    if any(isinstance(n, ast.Attribute) and n.attr == "splitlines" for n in ast.walk(arg)):
        return None
    node = arg
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
            and node.func.attr in ("strip", "rstrip", "lstrip"):
        node = node.func.value
    if isinstance(node, ast.Attribute) and node.attr == "stdout":
        return node.value.id if isinstance(node.value, ast.Name) else ""
    return None


def check_ir009(rel: str, text: str) -> list[tuple[int, str]]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    hits: dict[int, str] = {}
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)]
        if not any(IR009_SSH_CALL.search(_call_name(c)) for c in calls):
            continue
        assigned: dict[str, list[ast.AST]] = {}
        for n in ast.walk(fn):
            if isinstance(n, ast.Assign):
                for t in n.targets:
                    if isinstance(t, ast.Name):
                        assigned.setdefault(t.id, []).append(n.value)
        for c in calls:
            f = c.func
            if not (isinstance(f, ast.Attribute) and f.attr == "loads"
                    and isinstance(f.value, ast.Name) and f.value.id == "json" and c.args):
                continue
            owner = _stdout_owner(c.args[0])
            if owner is None:
                continue
            sources = assigned.get(owner or "", [])
            if sources and all(_cat_only(v) for v in sources):
                continue
            hits[c.lineno] = (
                "json.loads() over an ssh reply's raw stdout: library chatter on the pod "
                "(hippius_hub's 'Downloading …' line, #983) breaks the parse. Read the reply "
                "with a separate `cat <file>` ssh call, parse its last line, or use "
                "lane_b_offload.pod_reply_json")
    return sorted(hits.items())


# ── the rule table ─────────────────────────────────────────────────────────────

RULES: tuple[Rule, ...] = (
    Rule("IR001", "bare `uv sync --frozen` uninstalls the extras",
         "2026-11-24 #892/#895/#897: the documented upgrade step removed bittensor",
         "repeat the install extras (`--extra train --extra hippius --extra chain`), or "
         "`--inexact` on a host whose profile installed them",
         ("docs/*.md", "README.md", "deploy/*", "scripts/*"),
         check_file=check_ir001),
    Rule("IR002", "`python -m` without `-P` in deploy scripts and units",
         "2026-11-04: intake ran Oct-30 code for a day from cwd (miner lost $90)",
         "`python -P -m …` (or the venv entry point)",
         ("deploy/*", "scripts/*.sh"),
         check_file=check_ir002),
    Rule("IR003", "`pgrep -f`/`pkill -f` with a plain literal pattern",
         "2026-09-14..2026-11-24: self-matching pgrep/pkill killed shells and stalled "
         "the EXP-LB-KING experiment driver",
         "bracket one character (`cascade[-]trainer`) or kill by pid file",
         ("deploy/*", "scripts/*"),
         check_file=check_ir003),
    Rule("IR004", "ssh host key not pinned",
         "2026-09-15: king leg died on 'Host key verification failed' (Lium reused ip:port)",
         "pin the scanned key: `-o UserKnownHostsFile=<per-pod file> "
         "-o StrictHostKeyChecking=yes`",
         ("deploy/*", "scripts/*"),
         check_file=check_ir004),
    Rule("IR005", "bulk git staging (`git add -u/-A/.`, `git commit -a`)",
         "2026-08-31 PR #239: `git add -u` pushed the live chain.toml override to main",
         "stage explicit paths and check the commit's file list",
         ("deploy/*", "scripts/*", "docs/*.md", "README.md"),
         check_file=check_ir005),
    Rule("IR006", "Shadeform as a GPU provider",
         "owner rule 2026-11-24: GPUs are Lium only, Shadeform is CPU only",
         "drop shadeform from that GPU table's providers",
         ("deploy/*.toml", "scripts/*.toml"),
         check_file=check_ir006),
    Rule("IR007", "config dataclass field the loader never parses",
         "2026-08-27 #223/#224: knobs landed without load_chain_config parsing; "
         "arming them in TOML silently no-oped",
         "add the loader parse + a loader round-trip test, or mark the field "
         "`# incident-ok: IR007 runtime-only: <who sets it>`",
         ("cascade/shared/config/*.py",),
         check_tree=check_ir007),
    Rule("IR008", "remote python over ssh with no on-host `timeout`",
         "2026-11-27 #969: lane D eval-host forecasts orphaned on the 4090 after the local "
         "ssh timeout; retries stacked three copies on the GPU",
         "wrap the remote command in `timeout -k <grace> <T - grace>` (see "
         "SSHEvalHost.remote_command), or launch it detached with its own deadline",
         ("cascade/*.py",),
         check_file=check_ir008),
    Rule("IR009", "ssh reply stdout parsed as JSON whole",
         "2026-11-27 #983: hippius_hub printed 'Downloading <file> (parallel)...' to the pod's "
         "stdout ahead of the cat'd result.json; the lane B settlement rebase failed "
         "'pod reply is not JSON' and the 19:27 settlement published without lane B",
         "read the reply with its own `cat <file>`-only ssh call, parse the last line "
         "(`.splitlines()[-1]`), or use `lane_b_offload.pod_reply_json`",
         ("cascade/*.py", "scripts/*.py"),
         check_file=check_ir009),
)
RULE_IDS = {r.id for r in RULES}


# ── runner ─────────────────────────────────────────────────────────────────────

def repo_files(root: Path) -> list[str]:
    if (root / ".git").exists():
        try:
            out = subprocess.run(["git", "-C", str(root), "ls-files", "-z"],
                                 capture_output=True, check=True).stdout
            return sorted(p for p in out.decode().split("\0") if p and (root / p).is_file())
        except (OSError, subprocess.CalledProcessError):
            pass
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*")
                  if p.is_file() and not SKIP_DIRS.intersection(p.relative_to(root).parts))


def _matches(rel: str, globs: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatchcase(rel, g) for g in globs)


def allowlist(rel: str, text: str) -> tuple[dict[int, set[str]], list[Hit]]:
    """Line -> rule ids allowed there (the comment's line and the next line)."""
    allowed: dict[int, set[str]] = {}
    bad = []
    for no, line in enumerate(text.splitlines(), 1):
        for m in ALLOW_RE.finditer(line):
            rule = (m.group("rule") or "").upper()
            reason = re.sub(r"-->\s*$", "", m.group("reason")).strip(" :-\"'")
            if rule not in RULE_IDS:
                bad.append(Hit("IR000", rel, no, f"allowlist names unknown rule {rule or '?'}"))
            elif len(reason) < 3:
                bad.append(Hit("IR000", rel, no, f"allowlist for {rule} gives no reason"))
            else:
                allowed.setdefault(no, set()).add(rule)
                allowed.setdefault(no + 1, set()).add(rule)
    return allowed, bad


def check(root: Path = ROOT, paths: list[str] | None = None,
          rules: tuple[Rule, ...] = RULES) -> list[Hit]:
    files = paths if paths is not None else repo_files(root)
    files = [f for f in files if not _matches(f, GLOBAL_EXCLUDE)]
    texts: dict[str, str] = {}

    def text_of(rel: str) -> str | None:
        if rel not in texts:
            try:
                texts[rel] = (root / rel).read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                texts[rel] = None
        return texts[rel]

    raw: list[Hit] = []
    for rule in rules:
        mine = [f for f in files if _matches(f, rule.globs) and not _matches(f, rule.exclude)]
        if rule.check_file:
            for rel in mine:
                text = text_of(rel)
                if text is not None:
                    raw += [Hit(rule.id, rel, no, msg) for no, msg in rule.check_file(rel, text)]
        if rule.check_tree:
            tree = {rel: t for rel in mine if (t := text_of(rel)) is not None}
            raw += [Hit(rule.id, rel, no, msg) for rel, no, msg in rule.check_tree(tree)]

    hits: list[Hit] = []
    allow_cache: dict[str, dict[int, set[str]]] = {}
    for rel in sorted({h.path for h in raw} | set(texts)):
        text = texts.get(rel)
        if text is None or "incident-ok" not in text:
            allow_cache[rel] = {}
            continue
        allow_cache[rel], bad = allowlist(rel, text)
        hits += bad
    hits += [h for h in raw if h.rule not in allow_cache.get(h.path, {}).get(h.line, set())]
    return sorted(set(hits), key=lambda h: (h.path, h.line, h.rule))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("paths", nargs="*", help="repo-relative files (default: every tracked file)")
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--rule", action="append", help="run only this rule id (repeatable)")
    ap.add_argument("--list", action="store_true", help="print the rules and exit")
    args = ap.parse_args(argv)
    if args.list:
        for r in RULES:
            print(f"{r.id}  {r.title}\n       incident: {r.incident}\n       fix: {r.fix}")
        return 0
    rules = RULES
    if args.rule:
        wanted = {x.upper() for x in args.rule}
        unknown = wanted - RULE_IDS
        if unknown:
            ap.error(f"unknown rule(s): {', '.join(sorted(unknown))}")
        rules = tuple(r for r in RULES if r.id in wanted)
    hits = check(args.root.resolve(), args.paths or None, rules)
    fixes = {r.id: r.fix for r in RULES}
    for h in hits:
        print(h)
    if hits:
        print(f"\n{len(hits)} incident-rule hit(s). Fix each (see docs/INCIDENT_RULES.md):")
        for rid in sorted({h.rule for h in hits}):
            print(f"  {rid}: {fixes.get(rid, 'give the allowlist a known rule id and a reason')}")
        print("or, when the exception is justified, add `incident-ok: <rule-id> <reason>` "
              "on the line or the line above.")
        return 1
    print(f"incident rules: 0 hits ({len(rules)} rules)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
