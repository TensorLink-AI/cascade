"""Shared spec and doc parsers for the W12a acceptance tests.

Copied from docs/plans/LANE_D_PLAN.md (rev6.4):

- W12a row: "Docs and agent surfaces (§11.2 text surfaces, website; rev3:
  LANE_D.md FAQ on coldkey caps, confirmation latency, both clocks, BF16;
  VALIDATOR.md on what satellites verify for the new report fields;
  OPERATOR.md on launch eval host, S8, first-seen recorder) | W10 | ci |
  `scripts/laned/accept W12a` (every §11.2 surface mentions lane D; links
  resolve; `llms.txt` lane list matches the status fixture)".
- §11.2 (the table) and its "§11.2 → WPs" row for W12a: "LANE_D.md,
  INTERFACE, MINER, AGENT_MINING, AGENTS/agent.md, llms.txt (both),
  VALIDATOR, OPERATOR, ARCHITECTURE, website, README/CHANGES".

**The code is the truth** (as in WB20): every key, default, file name, reason
code and report field the tests require is read from main on each run, never
copied here. The parsers (sections, spans, links, leaks) are WB20's
(`tests/accept/wb20/helpers.py`). Nothing here imports a module in PATHS
(W12a's PATHS are docs only).
"""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path

from tests.accept.wb20 import helpers as wb

REPO = Path(__file__).resolve().parents[3]
DOCS = REPO / "docs"
LANE_D_MD = DOCS / "LANE_D.md"
INTERFACE = DOCS / "INTERFACE.md"
MINER = DOCS / "MINER.md"
AGENT_MINING = DOCS / "AGENT_MINING.md"
VALIDATOR = DOCS / "VALIDATOR.md"
OPERATOR = DOCS / "OPERATOR.md"
ARCHITECTURE = DOCS / "ARCHITECTURE.md"
AGENTS = REPO / "AGENTS.md"
AGENT = REPO / "agent.md"
LLMS = REPO / "llms.txt"
WEB_LLMS = REPO / "cascade" / "website" / "llms.txt"
WEB_INDEX = REPO / "cascade" / "website" / "index.html"
README = REPO / "README.md"
CHANGES = REPO / "CHANGES.md"
SCHEMA = REPO / "cascade" / "validator" / "schemas" / "lane_d_report_1.json"

# §11.2 text surfaces that must carry a lane D SECTION (a heading).
SECTION_DOCS = (LANE_D_MD, INTERFACE, MINER, AGENT_MINING, VALIDATOR, OPERATOR, ARCHITECTURE,
                AGENTS, AGENT)
# §11.2 surfaces that must at least mention lane D.
MENTION_SURFACES = (*SECTION_DOCS, LLMS, WEB_LLMS, WEB_INDEX, README, CHANGES)
# Surfaces whose relative links must resolve. CHANGES.md is a historical log
# that cites absolute /tmp paths of old runs; its links are not checked.
LINK_SURFACES = (*SECTION_DOCS, LLMS, README)

LANE_D_HEADING = re.compile(r"\blane[ -]?d\b", re.I)
LANE_D_WORD = re.compile(r"\blane[ _-]?d\b", re.I)
NEGATION = re.compile(r"\b(?:no|not|never|cannot|can't|without|nor|neither)\b", re.I)

# The five commands of the plan's W10 row (the parser is the truth once W10
# is on main: `cli_subcommands`).
PLAN_SUBCOMMANDS = ("export", "verify", "submit", "ready", "status")

read = wb.read
spans = wb.spans
paragraphs = wb.paragraphs
has_token = wb.has_token
broken_links = wb.broken_links
leaks = wb.leaks
HEADING = wb.HEADING
FENCE = wb.FENCE


def rel(path: Path) -> str:
    return path.relative_to(REPO).as_posix()


def sections(path: Path, pattern: re.Pattern = LANE_D_HEADING) -> list[str]:
    """The text of every section of ``path`` whose heading matches
    ``pattern`` (heading line included), to the next heading of the same or a
    higher level. Headings inside fenced code are ignored."""
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
        if not pattern.search(text):
            continue
        end = len(lines)
        for j, lv, _ in heads[n + 1:]:
            if lv <= level:
                end = j
                break
        out.append("\n".join(lines[i:end]))
    return out


def lane_d_text(path: Path) -> str:
    return "\n\n".join(sections(path))


def require(path: Path) -> str:
    assert path.is_file(), f"{rel(path)} does not exist"
    return read(path)


def require_sections(path: Path, pattern: re.Pattern = LANE_D_HEADING, what="lane D") -> str:
    require(path)
    secs = sections(path, pattern)
    assert secs, f"{rel(path)} has no {what} section (a heading matching {pattern.pattern!r})"
    return "\n\n".join(secs)


def paragraph_with(text: str, *needles, negation: bool = False) -> str | None:
    """The first paragraph of ``text`` that contains every needle (a str is
    matched case-insensitively, a compiled pattern by search), plus a
    negation word when asked."""
    for p in paragraphs(text):
        low = p.lower()
        ok = all((n.search(p) if isinstance(n, re.Pattern) else n.lower() in low)
                 for n in needles)
        if ok and (not negation or NEGATION.search(p)):
            return p
    return None


def documents_default(text: str, key: str, default) -> bool:
    """One line carries a code span naming ``key`` (`` `k` ``,
    `` `[lane_d] k` `` or `` `k = v` ``) and the default: either inside a
    `` `k = v` `` span or as another span on that line (TOML value equality)."""
    for line in text.splitlines():
        ss = wb.inline_spans(line)
        named = [s for s in ss if re.fullmatch(rf"(?:\[lane_d\]\s*)?{re.escape(key)}(?:\s*=.*)?", s)]
        if not named:
            continue
        values = [s.split("=", 1)[1].strip() for s in named if "=" in s]
        values += [s for s in ss if s not in named]
        for v in values:
            parsed = wb.toml_value(v)
            if parsed is not wb._NOPE and wb.default_equal(parsed, default):
                return True
    return False


# ── the code is the truth ───────────────────────────────────────────────────


def lane_d_defaults() -> dict:
    from cascade.shared.config.lane_d import LaneDConfig

    return {f.name: f.default for f in dataclasses.fields(LaneDConfig)
            if f.default is not dataclasses.MISSING}


def lane_d_keys() -> frozenset[str]:
    from cascade.shared.config.lane_d import LaneDConfig

    return frozenset(f.name for f in dataclasses.fields(LaneDConfig))


def allowed_files() -> tuple[str, ...]:
    from cascade.lane_d.intake import ALLOWED_FILES

    return tuple(ALLOWED_FILES)


def skip_reasons() -> tuple[str, ...]:
    from cascade.validator.lane_d_report import SKIP_REASONS

    return tuple(SKIP_REASONS)


def report_fields() -> dict[str, set[str]]:
    """Top-level and per-record property names of `lane-d-report/1`."""
    s = json.loads(SCHEMA.read_text(encoding="utf-8"))
    defs = s.get("$defs") or s.get("definitions") or {}
    out = {"": set(s["properties"])}
    for k, p in s["properties"].items():
        it = p.get("items", p)
        if "$ref" in it:
            it = defs[it["$ref"].split("/")[-1]]
        out[k] = set(it.get("properties", {}))
    return out


def cli_subcommands() -> tuple[str, ...]:
    """The `cascade lane-d` children of the real parser (W10), else the plan's
    five (W10 not on main)."""
    import argparse

    from cascade.miner.cli import build_parser

    root = build_parser()
    (sub,) = [a for a in root._actions if isinstance(a, argparse._SubParsersAction)]
    group = sub.choices.get("lane-d")
    if group is None:
        return PLAN_SUBCOMMANDS
    kids = [a for a in group._actions if isinstance(a, argparse._SubParsersAction)]
    return tuple(sorted(kids[0].choices)) if kids else PLAN_SUBCOMMANDS


def status_lanes() -> tuple[str, ...]:
    """The lane list of the status fixture: the `lanes` keys of
    `cascade status --json`'s snapshot (`live_status.status_snapshot`) over
    an empty fetch, upper-cased."""
    from datetime import UTC, datetime

    from cascade.miner.live_status import status_snapshot

    snap = status_snapshot({}, now=datetime(2026, 11, 23, tzinfo=UTC), max_age=3600.0)
    return tuple(sorted(str(k).upper() for k in snap["lanes"]))


def lane_table_letters(text: str) -> tuple[str, ...]:
    """The first-column values of every Markdown table whose header's first
    cell is `Lane` (single upper-case letters only)."""
    letters, in_table = [], False
    for line in text.splitlines():
        s = line.strip()
        if not s.startswith("|"):
            in_table = False
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if cells and cells[0] == "Lane":
            in_table = True
            continue
        if in_table and re.fullmatch(r"[A-Z]", cells[0] or ""):
            letters.append(cells[0])
    return tuple(sorted(set(letters)))
