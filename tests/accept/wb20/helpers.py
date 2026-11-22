"""Code truth and doc parsing for the WB20 acceptance tests (lane B docs).

Every fact the docs must state is derived from the code on every run, never
from a hardcoded list, so a code change that adds a key or a refusal code
makes a stale doc fail. Sources (all on main, all torch-free):

- ``cascade/shared/config/lane_b.py``: ``LANE_B_TRACK_DEFAULTS`` (the WB10
  track keys and defaults), ``LANE_B_OPEN_RULES`` (WB17);
- ``cascade/shared/config/v2.py``: ``parse_lane_b`` (every other ``[lane_b]``
  key a node reads), ``SUITE_TARGET_SUITES`` / ``SUITE_TARGET_PREFIX`` /
  ``SUITE_CODE_PREFIX`` (WB18 suite targets and their ``s-<suite>`` codes);
- ``cascade/funding/intake.py``: the ``code`` values of
  ``FundingIntake._lane_b_contamination`` (WB13 refusals), the
  ``LANE_B_DECLARED_ROWS_HEADER`` name and the ``rows=`` signed tag (WB14);
- ``cascade/trainer/declared_rows.py``: ``DECLARED_ROWS_CODES``,
  ``model_experts_per_layer``, ``layer_clamp``, ``parse_declared_rows``;
- ``cascade/trainer/row_claims.py``: ``ROWS_HELD`` (WB15), ``SENTINEL_TARGET``
  (WB18) deferrals;
- ``cascade/trainer/expert_lane.py``: ``GATE_CLOSED_REASONS`` (WB17), by AST
  (the module imports numpy and is the trainer's, not the docs');
- ``cascade/funding/queue.py``: the ``last_error_class`` of
  ``FundedQueue.expire_stale`` (``funding_expired``; LaneBQueue inherits it);
- ``cascade/interface/contamination_scan.py``: the receipt keys
  ``scan_repo`` returns (WB13).

Only stdlib and these torch-free modules are imported.
"""

from __future__ import annotations

import ast
import json
import re
import tomllib
from functools import cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
CASCADE = REPO / "cascade"
DOCS = REPO / "docs"
MINER = DOCS / "MINER.md"
INTERFACE = DOCS / "INTERFACE.md"
AGENT_MINING = DOCS / "AGENT_MINING.md"
VALIDATOR = DOCS / "VALIDATOR.md"
WB20_DOCS = (MINER, INTERFACE, AGENT_MINING, VALIDATOR)

LANE_B_PY = CASCADE / "shared" / "config" / "lane_b.py"
V2_PY = CASCADE / "shared" / "config" / "v2.py"
INTAKE_PY = CASCADE / "funding" / "intake.py"
QUEUE_PY = CASCADE / "funding" / "queue.py"
EXPERT_LANE_PY = CASCADE / "trainer" / "expert_lane.py"
SCAN_PY = CASCADE / "interface" / "contamination_scan.py"
CHAIN_TOMLS = (REPO / "chain.toml", REPO / "chain.testnet.toml")

# A heading whose text names lane B opens a "lane B section" (to the next
# heading of the same or a higher level).
LANE_B_HEADING = re.compile(r"\blane[ -]?b\b", re.I)
HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
FENCE = re.compile(r"^\s*(```|~~~)")
SPAN = re.compile(r"(?<!`)`([^`\n]+)`(?!`)")
SNAKE = re.compile(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)+")
RANGE_SEP = r"\s*(?:\.\.|–|—|-|to)\s*"


# ── code truth ──────────────────────────────────────────────────────────────

def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), str(path))


def _function(tree: ast.AST, name: str, cls: str | None = None) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if cls is not None and isinstance(node, ast.ClassDef) and node.name == cls:
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and sub.name == name:
                    return sub
        if cls is None and isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"code truth: {cls + '.' if cls else ''}{name} not found")


def _module_strings(tree: ast.Module) -> dict[str, object]:
    """Module-level ``NAME = <constant or tuple of names/constants>``."""
    out: dict[str, object] = {}
    for node in tree.body:
        targets, value = [], None
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        for t in targets:
            if not isinstance(t, ast.Name):
                continue
            if isinstance(value, ast.Constant):
                out[t.id] = value.value
            elif isinstance(value, ast.Tuple):
                items = []
                for e in value.elts:
                    if isinstance(e, ast.Constant):
                        items.append(e.value)
                    elif isinstance(e, ast.Name) and e.id in out:
                        items.append(out[e.id])
                out[t.id] = tuple(items)
    return out


def _reads(fn: ast.AST, receiver: str) -> set[str]:
    """String keys read from ``receiver`` (``r["k"]``, ``r.get("k")``, ``"k" in r``)."""
    keys: set[str] = set()
    for node in ast.walk(fn):
        if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
                and node.value.id == receiver and isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, str)):
            keys.add(node.slice.value)
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
              and node.func.attr in ("get", "pop", "setdefault")
              and isinstance(node.func.value, ast.Name) and node.func.value.id == receiver
              and node.args and isinstance(node.args[0], ast.Constant)
              and isinstance(node.args[0].value, str)):
            keys.add(node.args[0].value)
        elif (isinstance(node, ast.Compare) and len(node.ops) == 1
              and isinstance(node.ops[0], ast.In) and isinstance(node.left, ast.Constant)
              and isinstance(node.left.value, str) and isinstance(node.comparators[0], ast.Name)
              and node.comparators[0].id == receiver):
            keys.add(node.left.value)
    return keys


@cache
def track_defaults() -> dict[str, object]:
    """WB10: the lane B track keys and their defaults."""
    from cascade.shared.config.lane_b import LANE_B_TRACK_DEFAULTS

    return dict(LANE_B_TRACK_DEFAULTS)


@cache
def lane_b_keys() -> frozenset[str]:
    """Every ``[lane_b]`` key some parser reads (v2.parse_lane_b + the WB10 track)."""
    keys = set(track_defaults())
    keys |= _reads(_function(_tree(V2_PY), "parse_lane_b"), "lb")
    keys |= _reads(_function(_tree(LANE_B_PY), "parse_lane_b_track"), "lb")
    return frozenset(keys)


def funded_pod_skus_defaults_to_round() -> bool:
    """D42: ``funded_pod_skus`` has no literal default; the parser takes the
    same file's ``[round] funded_pod_skus`` (its ``round_funded_pod_skus``)."""
    fn = _function(_tree(LANE_B_PY), "parse_lane_b_track")
    names = {a.arg for a in fn.args.kwonlyargs}
    return "round_funded_pod_skus" in names


@cache
def open_rules() -> tuple[str, ...]:
    from cascade.shared.config.lane_b import LANE_B_OPEN_RULES

    return tuple(LANE_B_OPEN_RULES)


@cache
def suites() -> tuple[str, ...]:
    from cascade.shared.config.v2 import SUITE_TARGET_SUITES

    return tuple(SUITE_TARGET_SUITES)


@cache
def suite_prefixes() -> tuple[str, str]:
    """(target prefix ``suite:``, commit code prefix ``s-``)."""
    from cascade.shared.config.v2 import SUITE_CODE_PREFIX, SUITE_TARGET_PREFIX

    return SUITE_TARGET_PREFIX, SUITE_CODE_PREFIX


@cache
def contamination_refusal_codes() -> tuple[str, ...]:
    """WB13: every ``"code"`` the lane B intake contamination check returns."""
    fn = _function(_tree(INTAKE_PY), "_lane_b_contamination", cls="FundingIntake")
    codes: list[str] = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.Dict):
            continue
        for k, v in zip(node.keys, node.values, strict=True):
            if isinstance(k, ast.Constant) and k.value == "code":
                for c in ast.walk(v):
                    if isinstance(c, ast.Constant) and isinstance(c.value, str) \
                            and c.value not in codes:
                        codes.append(c.value)
    assert codes, "code truth: no refusal codes found in _lane_b_contamination"
    return tuple(codes)


@cache
def declared_rows_codes() -> tuple[str, ...]:
    from cascade.trainer.declared_rows import DECLARED_ROWS_CODES

    return tuple(DECLARED_ROWS_CODES)


@cache
def deferral_codes() -> tuple[str, ...]:
    from cascade.trainer.row_claims import ROWS_HELD, SENTINEL_TARGET

    return (ROWS_HELD, SENTINEL_TARGET)


@cache
def gate_closed_reasons() -> tuple[str, ...]:
    consts = _module_strings(_tree(EXPERT_LANE_PY))
    reasons = consts.get("GATE_CLOSED_REASONS")
    assert isinstance(reasons, tuple) and reasons, "code truth: GATE_CLOSED_REASONS"
    return tuple(str(r) for r in reasons)


@cache
def expiry_codes() -> tuple[str, ...]:
    """The terminal class an idle funded (lane B) entry expires with."""
    fn = _function(_tree(QUEUE_PY), "expire_stale", cls="FundedQueue")
    out = []
    for node in ast.walk(fn):
        if isinstance(node, ast.keyword) and node.arg == "last_error_class" \
                and isinstance(node.value, ast.Constant):
            out.append(node.value.value)
    assert out, "code truth: FundedQueue.expire_stale error class"
    return tuple(out)


@cache
def receipt_keys() -> tuple[str, ...]:
    """WB13: the ``contamination`` receipt field's keys.

    ``scan_repo`` either returns the receipt dict literal directly or returns
    a call to a same-module helper (``_receipt(...)`` since #643) that builds
    it; follow such calls one level and read the helper's dict literal."""
    tree = _tree(SCAN_PY)
    module_fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}

    def dict_keys(fn: ast.FunctionDef) -> tuple[str, ...]:
        for node in ast.walk(fn):
            if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict):
                keys = tuple(k.value for k in node.value.keys
                             if isinstance(k, ast.Constant) and isinstance(k.value, str))
                if keys:
                    return keys
        return ()

    fn = _function(tree, "scan_repo")
    keys = dict_keys(fn)
    if keys:
        return keys
    for node in ast.walk(fn):
        if (isinstance(node, ast.Return) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id in module_fns):
            keys = dict_keys(module_fns[node.value.func.id])
            if keys:
                return keys
    raise AssertionError(
        "code truth: scan_repo receipt keys (no returned dict literal in scan_repo "
        "nor in a same-module helper it returns a call to)")


@cache
def intake_constants() -> dict[str, object]:
    return _module_strings(_tree(INTAKE_PY))


def declared_rows_header() -> str:
    return str(intake_constants()["LANE_B_DECLARED_ROWS_HEADER"])


def declared_rows_signed_tag() -> str:
    return str(intake_constants()["_DECLARED_ROWS_SIGNED_TAG"])


@cache
def lane_b_headers() -> frozenset[str]:
    """Every ``X-Lane-B-*`` header string the code knows."""
    return frozenset(s for s in code_strings() if s.startswith("X-Lane-B-"))


@cache
def experts_per_layer() -> tuple[int, ...]:
    from cascade.trainer.declared_rows import model_experts_per_layer

    return tuple(model_experts_per_layer())


def clamp(n_experts: int) -> tuple[int, int]:
    from cascade.trainer.declared_rows import layer_clamp

    d = track_defaults()
    return layer_clamp(n_experts, float(d["min_experts_frac"]), float(d["max_experts_frac"]))


def parse_rows(raw: str):
    """WB14's parser under the default clamp; returns ``(rows, None)`` or
    ``(None, code)``."""
    from cascade.trainer.declared_rows import (
        DeclaredRowsError,
        model_experts_per_layer,
        parse_declared_rows,
    )

    d = track_defaults()
    try:
        return parse_declared_rows(raw, experts_per_layer=model_experts_per_layer(),
                                   min_frac=float(d["min_experts_frac"]),
                                   max_frac=float(d["max_experts_frac"])), None
    except DeclaredRowsError as e:
        return None, e.code


@cache
def code_strings() -> frozenset[str]:
    """Every string constant and identifier under ``cascade/`` (the corpus a
    doc token must exist in) plus every key and string value of the
    checked-in chain TOMLs."""
    out: set[str] = set()
    for path in sorted(CASCADE.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                out.add(node.value)
            elif isinstance(node, ast.Name):
                out.add(node.id)
            elif isinstance(node, ast.Attribute):
                out.add(node.attr)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                out.add(node.name)
            elif isinstance(node, ast.arg) or (isinstance(node, ast.keyword) and node.arg):
                out.add(node.arg)
            elif isinstance(node, ast.alias):
                out.add(node.asname or node.name.split(".")[-1])

    def walk(table: object) -> None:
        if isinstance(table, dict):
            for k, v in table.items():
                out.add(str(k))
                walk(v)
        elif isinstance(table, list):
            for v in table:
                walk(v)
        elif isinstance(table, str):
            out.add(table)

    for toml in CHAIN_TOMLS:
        if toml.exists():
            walk(tomllib.loads(toml.read_text(encoding="utf-8")))
    return frozenset(out)


def code_has_token(token: str) -> bool:
    """A doc token exists in code: as a whole string/identifier, or as a
    whole ``key=`` / ``key:`` word of a string constant (log and wire forms)."""
    corpus = code_strings()
    if token in corpus:
        return True
    pat = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(token)}(?![A-Za-z0-9_])")
    return any(pat.search(s) for s in corpus if len(s) < 400 and token[:3] in s)


# ── doc parsing ─────────────────────────────────────────────────────────────

def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def lane_b_sections(path: Path) -> list[str]:
    """The text of every lane B section of ``path`` (heading line included).
    Headings inside fenced code are ignored."""
    lines = read(path).splitlines()
    heads: list[tuple[int, int, str]] = []
    fenced = False
    for i, line in enumerate(lines):
        if FENCE.match(line):
            fenced = not fenced
            continue
        if fenced:
            continue
        m = HEADING.match(line)
        if m:
            heads.append((i, len(m.group(1)), m.group(2)))
    out: list[str] = []
    for n, (i, level, text) in enumerate(heads):
        if not LANE_B_HEADING.search(text):
            continue
        end = len(lines)
        for j, lv, _ in heads[n + 1:]:
            if lv <= level:
                end = j
                break
        out.append("\n".join(lines[i:end]))
    # nested lane B headings inside an outer lane B section repeat text; harmless
    return out


def lane_b_text(path: Path) -> str:
    return "\n\n".join(lane_b_sections(path))


def spans(text: str) -> list[str]:
    """Inline code spans (outside fenced blocks) plus every fenced line."""
    out: list[str] = []
    fenced = False
    for line in text.splitlines():
        if FENCE.match(line):
            fenced = not fenced
            continue
        if fenced:
            out.append(line.strip())
        else:
            out.extend(s.strip() for s in SPAN.findall(line))
    return out


def inline_spans(line: str) -> list[str]:
    return [s.strip() for s in SPAN.findall(line)]


def paragraphs(text: str) -> list[str]:
    """Blank-line separated blocks (a table or a list counts as one block)."""
    return [p for p in re.split(r"\n\s*\n", text) if p.strip()]


def has_token(text: str, token: str) -> bool:
    """``token`` appears as an inline code span (alone, or as a whole word
    inside a span) or as a whole word in a fenced line."""
    pat = re.compile(rf"(?<![A-Za-z0-9_\-]){re.escape(token)}(?![A-Za-z0-9_\-])")
    return any(pat.search(s) for s in spans(text))


def toml_value(literal: str):
    """``literal`` parsed as a TOML value, or a sentinel ``_NOPE``."""
    try:
        return tomllib.loads(f"v = {literal}")["v"]
    except (tomllib.TOMLDecodeError, ValueError):
        return _NOPE


_NOPE = object()


def default_equal(parsed: object, default: object) -> bool:
    if parsed is _NOPE:
        return False
    if isinstance(default, bool) or isinstance(parsed, bool):
        return type(parsed) is type(default) and parsed == default
    if isinstance(default, str):
        return isinstance(parsed, str) and parsed == default
    if isinstance(default, (int, float)):
        return isinstance(parsed, (int, float)) and float(parsed) == float(default)
    if isinstance(default, tuple) and all(isinstance(p, tuple) and len(p) == 2 for p in default):
        return isinstance(parsed, dict) and parsed == dict(default)
    if isinstance(default, tuple):
        return isinstance(parsed, list) and tuple(parsed) == default
    return parsed == default


def names_key(span: str, key: str) -> bool:
    """A code span names the ``[lane_b]`` key: ``key``, ``[lane_b] key`` or
    ``[lane_b] key = …`` / ``key = …`` (never ``[other] key``)."""
    s = span.strip()
    return bool(re.fullmatch(rf"(?:\[lane_b\]\s+)?{re.escape(key)}(?:\s*=.*)?", s))


def documents_default(text: str, key: str, default: object) -> bool:
    """Some line of ``text`` names ``key`` in a code span and gives its
    default: either a span ``key = <toml>`` / ``[lane_b] key = <toml>`` or a
    separate span on the same line whose TOML value equals the default."""
    for line in text.splitlines():
        ss = inline_spans(line)
        if not any(names_key(s, key) for s in ss):
            continue
        for s in ss:
            m = re.fullmatch(rf"(?:\[lane_b\]\s+)?{re.escape(key)}\s*=\s*(.+)", s)
            if m and default_equal(toml_value(m.group(1)), default):
                return True
            if not names_key(s, key) and default_equal(toml_value(s), default):
                return True
    return False


def declared_rows_examples(text: str) -> list[tuple[str, str]]:
    """``(line, json)`` for every ``<header>: {json}`` in ``text``."""
    header = declared_rows_header()
    out = []
    for line in text.splitlines():
        for m in re.finditer(rf"{re.escape(header)}\s*:\s*", line):
            rest = line[m.end():]
            i = rest.find("{")
            if i != 0:
                continue
            try:
                obj, end = json.JSONDecoder().raw_decode(rest)
            except ValueError:
                out.append((line, rest))
                continue
            out.append((line, rest[:end]))
    return out


def range_on_line(text: str, number: int, lo: int, hi: int) -> bool:
    pat = re.compile(rf"(?<!\d)(?<!\d\.){lo}{RANGE_SEP}{hi}(?!\d|\.\d)")
    num = re.compile(rf"(?<!\d)(?<!\d\.){number}(?!\d|\.\d)")
    return any(pat.search(ln) and num.search(pat.sub(" ", ln)) for ln in text.splitlines())


def ordinal_range_documented(text: str, n_layers: int) -> bool:
    pat = re.compile(rf"(?<!\d)(?<!\d\.)0{RANGE_SEP}{n_layers - 1}(?!\d|\.\d)")
    return any("ordinal" in ln.lower() and pat.search(ln) for ln in text.splitlines())


# ── links, secrets ──────────────────────────────────────────────────────────

def anchors(text: str) -> set[str]:
    """GitHub-style heading slugs (same algorithm as scripts/check_docs.py)."""
    result, used = set(), {}
    for heading in re.findall(r"^#{1,6}\s+(.+?)\s*#*$", text, re.M):
        heading = re.sub(r"<[^>]+>", "", heading).strip().lower()
        slug = "".join(c for c in heading if c.isalnum() or c in " _-").replace(" ", "-")
        number = used.get(slug, 0)
        used[slug] = number + 1
        result.add(slug + (f"-{number}" if number else ""))
    return result


def broken_links(path: Path) -> list[str]:
    from urllib.parse import unquote

    text = re.sub(r"^```[^\n]*\n.*?^```", "", read(path), flags=re.M | re.S)
    issues = []
    for target in re.findall(r"\]\(([^)\s]+)(?:\s+[^)]*)?\)", text):
        if re.match(r"\w+://|mailto:|data:", target):
            continue
        destination, _, anchor = unquote(target.strip("<>")).partition("#")
        resolved = path.parent / destination if destination else path
        if not resolved.exists():
            issues.append(f"{path.name}: missing local link {target}")
        elif anchor and resolved.suffix in (".md", ".txt") \
                and anchor not in anchors(read(resolved)):
            issues.append(f"{path.name}: missing heading {target}")
    return issues


IPV4 = re.compile(r"(?<![\w.])(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})(?![\w.])")
# RFC 5737 documentation ranges, loopback and the unspecified address are fine
_OK_IP_PREFIXES = ("192.0.2.", "198.51.100.", "203.0.113.", "127.", "0.0.0.0")
SECRET_PATTERNS = (
    ("aws access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("hugging face token", re.compile(r"\bhf_[A-Za-z0-9]{30,}\b")),
    ("github token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b")),
    ("openai/anthropic style key", re.compile(r"\bsk-(?:ant-)?[A-Za-z0-9_-]{20,}\b")),
    ("private key block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("slack webhook", re.compile(r"hooks\.slack\.com/services/[A-Za-z0-9/]+")),
    ("discord webhook", re.compile(r"discord(?:app)?\.com/api/webhooks/\d+")),
    ("telegram bot token", re.compile(r"\b\d{8,10}:[A-Za-z0-9_-]{35}\b")),
    ("credential assignment", re.compile(
        r"\b(?:HIPPIUS_[A-Z_]+|HF_TOKEN|WANDB_API_KEY|LIUM_API_KEY|BT_WALLET_PASSWORD\w*"
        r"|CASCADE_OPS_WEBHOOK_URL|AWS_SECRET_ACCESS_KEY)\s*=\s*['\"]?"
        r"(?![\s'\"$<{]|\.\.\.|YOUR|your|xxx|XXX|\*)[A-Za-z0-9/+_\-]{12,}")),
    ("ssh target with an address", re.compile(r"\b[a-z_][a-z0-9_-]*@\d{1,3}(?:\.\d{1,3}){3}\b")),
    # a login on a concrete host (user@host.domain): pods, the box, executors
    ("user@host login", re.compile(
        r"(?<![\w.+-])(?:root|ubuntu|admin|user|cascade)@[a-z0-9-]+(?:\.[a-z0-9-]+)+\b")),
    ("private hostname", re.compile(
        r"\b[a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:internal|local|lan|localdomain|svc|corp|home\.arpa)"
        r"(?![\w.-])")),
    ("ssh with a concrete port and host", re.compile(
        r"\bssh\b[^\n`]*-p\s*\d{2,5}\s+[a-z_][a-z0-9_-]*@[a-z0-9]")),
)


def leaks(path: Path) -> list[str]:
    out = []
    for n, line in enumerate(read(path).splitlines(), 1):
        for m in IPV4.finditer(line):
            ip = m.group(0)
            if all(0 <= int(p) <= 255 for p in m.groups()) \
                    and not ip.startswith(_OK_IP_PREFIXES):
                out.append(f"{path.name}:{n}: IPv4 address {ip}")
        for name, pat in SECRET_PATTERNS:
            if pat.search(line):
                out.append(f"{path.name}:{n}: {name}")
    return out
