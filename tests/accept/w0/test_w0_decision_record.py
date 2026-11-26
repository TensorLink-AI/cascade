"""W0 acceptance: DEC-CA-0070 records every lane D decision as settled.

Plan §3.0 W0 row: DEC-CA-0070 (supersedes DEC-CA-0062's recipe design)
records D1-D43, D45-D48 and D42b (D44 withdrawn). Accept: front matter valid;
``supersedes: DEC-CA-0062``; every decision answered or marked resolved.

The expected choices are parsed from the §17.4 decision register in
``docs/plans/LANE_D_PLAN.md`` on every run, so these tests follow the plan.
The front-matter rules are the ones every accepted/active record on main
already obeys (there is no decision validator in the repo); the
``test_rules_hold_for_existing_*`` tests pin that the rules are the repo's.
See README.md in this directory for the full contract.
"""

from __future__ import annotations

import datetime as _dt
import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
DECISIONS = REPO / "decisions"
PLAN = REPO / "docs" / "plans" / "LANE_D_PLAN.md"
DEC70 = DECISIONS / "DEC-CA-0070-lane-d-open-weights.md"
DEC62 = DECISIONS / "DEC-CA-0062-lane-d-distillation-real-data.md"

# The W0 row, verbatim: D1-D43, D45-D48 and D42b, with D44 withdrawn.
W0_IDS = (
    {f"D{n}" for n in range(1, 44)} | {f"D{n}" for n in range(45, 49)} | {"D42b", "D44"}
)

# Every record on main carries UNIVERSAL_KEYS; all but a handful also carry
# revisit_when and relations. A new record (DEC-CA-0070) must carry all of
# REQUIRED_KEYS and relate only to DEC-CA records that exist.
UNIVERSAL_KEYS = ("id", "type", "title", "status", "date", "tags")
REQUIRED_KEYS = UNIVERSAL_KEYS + ("revisit_when", "relations")
OPTIONAL_KEYS = ("note", "amended")
STATUSES = {"proposed", "accepted", "active", "superseded"}
DECIDED_STATUSES = {"accepted", "active"}
REGISTER_STATUSES = {"decided", "withdrawn", "deferred-to-stage"}
DEC_ID = re.compile(r"^DEC-CA-\d{4}$")


# --------------------------------------------------------------------------
# Front matter (the subset of YAML the decision records use)
# --------------------------------------------------------------------------

def _split_top(text: str, sep: str = ",") -> list[str]:
    """Split on ``sep`` at bracket depth 0, outside double quotes."""
    parts, depth, quoted, cur, i = [], 0, False, [], 0
    while i < len(text):
        ch = text[i]
        if quoted:
            cur.append(ch)
            if ch == "\\" and i + 1 < len(text):
                cur.append(text[i + 1])
                i += 1
            elif ch == '"':
                quoted = False
        elif ch == '"':
            quoted = True
            cur.append(ch)
        elif ch in "[{":
            depth += 1
            cur.append(ch)
        elif ch in "]}":
            depth -= 1
            cur.append(ch)
        elif ch == sep and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
        i += 1
    tail = "".join(cur).strip()
    if tail:
        parts.append(tail)
    return parts


def _scalar(raw: str):
    raw = raw.strip()
    if raw.startswith('"'):
        assert raw.endswith('"'), f"unterminated string: {raw!r}"
        return json.loads(raw)
    if raw.startswith("["):
        assert raw.endswith("]"), f"unterminated list: {raw!r}"
        return [_scalar(p) for p in _split_top(raw[1:-1])]
    if raw.startswith("{"):
        assert raw.endswith("}"), f"unterminated mapping: {raw!r}"
        out = {}
        for part in _split_top(raw[1:-1]):
            key, sep, value = part.partition(":")
            assert sep, f"mapping entry without ':': {part!r}"
            out[key.strip()] = _scalar(value)
        return out
    return raw


def parse_record(path: Path) -> tuple[dict, str]:
    """Return (front matter, body). Fails with an assertion if malformed."""
    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")
    assert lines[0] == "---", f"{path.name}: first line must be '---'"
    try:
        end = lines.index("---", 1)
    except ValueError:
        raise AssertionError(f"{path.name}: front matter is not closed by '---'") from None
    fm: dict = {}
    block_key, block_style, block_lines = None, "", []

    def _close_block() -> None:
        if block_key is not None:
            assert block_lines, f"{path.name}: empty block scalar for {block_key!r}"
            joiner = "\n" if block_style.startswith("|") else " "
            fm[block_key] = joiner.join(block_lines)

    for line in lines[1:end]:
        if block_key is not None and line.startswith((" ", "\t")):
            block_lines.append(line.strip())
            continue
        _close_block()
        block_key, block_style, block_lines = None, "", []
        if not line.strip():
            continue
        assert not line.startswith((" ", "\t")), f"{path.name}: nested front matter: {line!r}"
        key, sep, value = line.partition(":")
        assert sep, f"{path.name}: front matter line without ':': {line!r}"
        key = key.strip()
        assert key not in fm, f"{path.name}: duplicate front matter key {key!r}"
        if value.strip() in (">", ">-", "|", "|-"):
            block_key, block_style = key, value.strip()
            continue
        fm[key] = _scalar(value)
    _close_block()
    return fm, "\n".join(lines[end + 1:])


def _relation_targets(value) -> list:
    return value if isinstance(value, list) else [value]


def front_matter_errors(path: Path, fm: dict, strict: bool = True) -> list[str]:
    """The decision-record rules. ``strict=False`` is the floor every record on
    main obeys; ``strict=True`` adds revisit_when + relations as required and
    restricts relation targets to existing DEC-CA records."""
    errors = []
    for key in REQUIRED_KEYS if strict else UNIVERSAL_KEYS:
        if key not in fm:
            errors.append(f"missing key {key!r}")
    for key in fm:
        if key not in REQUIRED_KEYS + OPTIONAL_KEYS:
            errors.append(f"unknown key {key!r}")
    m = re.match(r"^(DEC-CA-\d{4})-[a-z0-9-]+\.md$", path.name)
    if not m:
        errors.append(f"file name {path.name!r} is not DEC-CA-NNNN-<slug>.md")
    elif fm.get("id") != m.group(1):
        errors.append(f"id {fm.get('id')!r} does not match the file name {m.group(1)!r}")
    if fm.get("type") != "decision":
        errors.append(f"type is {fm.get('type')!r}, want 'decision'")
    if not isinstance(fm.get("title"), str) or not fm.get("title", "").strip():
        errors.append("title must be a non-empty string")
    if fm.get("status") not in STATUSES:
        errors.append(f"status {fm.get('status')!r} not in {sorted(STATUSES)}")
    try:
        _dt.date.fromisoformat(str(fm.get("date")))
    except ValueError:
        errors.append(f"date {fm.get('date')!r} is not an ISO date (YYYY-MM-DD)")
    tags = fm.get("tags")
    if not isinstance(tags, list) or not tags or not all(isinstance(t, str) and t for t in tags):
        errors.append(f"tags must be a non-empty flow list of strings, got {tags!r}")
    if (strict or "revisit_when" in fm) and (
        not isinstance(fm.get("revisit_when"), str) or not fm.get("revisit_when", "").strip()
    ):
        errors.append("revisit_when must be a non-empty string")
    rel = fm.get("relations", {} if not strict else None)
    if not isinstance(rel, dict):
        errors.append(f"relations must be a flow mapping, got {rel!r}")
    else:
        for kind, value in rel.items():
            if not re.match(r"^[a-z_]+$", kind):
                errors.append(f"relation kind {kind!r} is not snake_case")
            for target in _relation_targets(value):
                if not isinstance(target, str) or not target.strip():
                    errors.append(f"relation {kind}: empty target")
                elif not strict:
                    continue
                elif not DEC_ID.match(target):
                    errors.append(f"relation {kind}: target {target!r} is not a DEC-CA id")
                elif not list(DECISIONS.glob(f"{target}-*.md")):
                    errors.append(f"relation {kind}: {target} has no file in decisions/")
    if "note" in fm and (not isinstance(fm["note"], str) or not fm["note"].strip()):
        errors.append("note must be a non-empty string")
    return errors


# --------------------------------------------------------------------------
# Markdown tables
# --------------------------------------------------------------------------

def _cells(line: str) -> list[str]:
    body = line.strip()
    assert body.startswith("|"), line
    body = body[1:-1] if body.endswith("|") else body[1:]
    return [c.strip() for c in body.split("|")]


def _is_separator(line: str) -> bool:
    return bool(re.match(r"^\|(\s*:?-{3,}:?\s*\|)+\s*$", line.strip()))


def tables(text: str) -> list[tuple[list[str], list[list[str]]]]:
    """Every pipe table in ``text`` as (header cells, row cells)."""
    out, lines, i = [], text.split("\n"), 0
    while i < len(lines) - 1:
        if lines[i].strip().startswith("|") and _is_separator(lines[i + 1]):
            header, rows, j = _cells(lines[i]), [], i + 2
            while j < len(lines) and lines[j].strip().startswith("|"):
                rows.append(_cells(lines[j]))
                j += 1
            out.append((header, rows))
            i = j
        else:
            i += 1
    return out


def norm(cell: str) -> str:
    return re.sub(r"\s+", " ", cell.replace("**", "")).strip()


def _register_rows(header: list[str], rows: list[list[str]], where: str) -> dict[str, dict]:
    keys = [norm(h).lower() for h in header]
    out: dict[str, dict] = {}
    for row in rows:
        did = norm(row[0])
        assert re.match(r"^D\d+[a-z]?$", did), f"{where}: register row id {row[0]!r} is not Dnn"
        assert did not in out, f"{where}: {did} appears in more than one register row"
        assert len(row) == len(header), f"{where}: {did} row has {len(row)} cells, header has {len(header)}"
        out[did] = {k: norm(c) for k, c in zip(keys, row, strict=True)}
    return out


def plan_register() -> dict[str, dict]:
    text = PLAN.read_text(encoding="utf-8")
    m = re.search(r"^### 17\.4 Decision register.*?$(.*?)^### ", text, re.M | re.S)
    assert m, "plan §17.4 'Decision register' section not found"
    for header, rows in tables(m.group(1)):
        if [norm(h).lower() for h in header[:3]] == ["d", "status", "option chosen"]:
            return _register_rows(header, rows, "plan §17.4")
    raise AssertionError("plan §17.4 has no '| D | Status | Option chosen |' table")


def record_register(body: str) -> dict[str, dict]:
    found: dict[str, dict] = {}
    for header, rows in tables(body):
        if [norm(h).lower() for h in header[:3]] == ["d", "status", "option chosen"]:
            for did, row in _register_rows(header, rows, DEC70.name).items():
                assert did not in found, f"{DEC70.name}: {did} recorded in two register tables"
                found[did] = row
    return found


# --------------------------------------------------------------------------
# Sanity: the rules and the register parser are the repo's / plan's
# --------------------------------------------------------------------------

def _decided_records() -> list[Path]:
    out = []
    for p in sorted(DECISIONS.glob("DEC-CA-*.md")):
        if p == DEC70:
            continue
        fm, _ = parse_record(p)
        if fm.get("status") in DECIDED_STATUSES:
            out.append(p)
    return out


def test_rules_hold_for_existing_decided_records():
    records = _decided_records()
    assert len(records) >= 10, "expected the repo's accepted/active records to exist"
    bad = {
        p.name: errs for p in records if (errs := front_matter_errors(p, parse_record(p)[0], strict=False))
    }
    assert not bad, f"the W0 front-matter rules reject existing decided records: {bad}"


def test_rules_hold_for_existing_superseded_records():
    for p in sorted(DECISIONS.glob("DEC-CA-*.md")):
        fm, _ = parse_record(p)
        if fm.get("status") == "superseded" and p != DEC62:
            assert not front_matter_errors(p, fm, strict=False), p.name
            assert str(fm.get("note", "")).startswith("Superseded by DEC-CA-"), (
                f"{p.name}: superseded records carry a 'Superseded by DEC-CA-NNNN' note"
            )


def test_plan_register_covers_w0_and_nothing_is_open():
    reg = plan_register()
    missing = sorted(W0_IDS - set(reg))
    assert not missing, f"plan §17.4 register lacks W0 decisions {missing}"
    bad = {d: r["status"] for d, r in reg.items() if r["status"] not in REGISTER_STATUSES}
    assert not bad, f"plan §17.4 has decisions not answered or resolved: {bad}"
    assert reg["D44"]["status"] == "withdrawn"


# --------------------------------------------------------------------------
# (1) Front matter of DEC-CA-0070
# --------------------------------------------------------------------------

def test_dec70_front_matter_valid():
    fm, _ = parse_record(DEC70)
    errors = front_matter_errors(DEC70, fm)
    assert not errors, f"{DEC70.name} front matter: {errors}"


def test_dec70_id_unique():
    owners = [p.name for p in DECISIONS.glob("DEC-CA-*.md") if parse_record(p)[0].get("id") == "DEC-CA-0070"]
    assert owners == [DEC70.name], f"DEC-CA-0070 claimed by {owners}"


def test_dec70_status_is_decided():
    fm, _ = parse_record(DEC70)
    assert fm.get("status") in DECIDED_STATUSES, (
        f"{DEC70.name} status is {fm.get('status')!r}; W0 needs one of {sorted(DECIDED_STATUSES)}"
    )


def test_dec70_supersedes_dec62():
    fm, _ = parse_record(DEC70)
    rel = fm.get("relations")
    assert isinstance(rel, dict), f"relations is {rel!r}"
    assert "DEC-CA-0062" in _relation_targets(rel.get("supersedes", [])), (
        f"relations must carry supersedes: [DEC-CA-0062]; got {rel}"
    )
    stale = [k for k in rel if k != "supersedes" and "DEC-CA-0062" in _relation_targets(rel[k])
             and k.startswith("supersedes")]
    assert not stale, f"conditional supersession kinds remain: {stale}"


# --------------------------------------------------------------------------
# (2) + (3) Every decision recorded, matching the §17.4 register
# --------------------------------------------------------------------------

def test_dec70_records_every_decision():
    _, body = parse_record(DEC70)
    rec = record_register(body)
    want = W0_IDS | set(plan_register())
    missing = sorted(want - set(rec), key=lambda d: (int(re.sub(r"\D", "", d)), d))
    assert not missing, (
        f"{DEC70.name} has no '| D | Status | Option chosen |' register row for {missing}"
    )
    extra = sorted(set(rec) - want)
    assert not extra, f"{DEC70.name} records decisions the plan register lacks: {extra}"


def test_dec70_d44_withdrawn():
    _, body = parse_record(DEC70)
    rec = record_register(body)
    assert "D44" in rec, f"{DEC70.name} does not record D44"
    assert rec["D44"]["status"] == "withdrawn", f"D44 status is {rec['D44']['status']!r}"


def test_dec70_choices_match_plan_register():
    _, body = parse_record(DEC70)
    rec = record_register(body)
    plan = plan_register()
    assert rec, f"{DEC70.name} has no '| D | Status | Option chosen |' register table"
    mismatches = []
    for did, want in plan.items():
        got = rec.get(did)
        if got is None:
            mismatches.append(f"{did}: not recorded")
            continue
        if got["status"] not in REGISTER_STATUSES:
            mismatches.append(f"{did}: status {got['status']!r} is not answered/resolved")
        # Status and option chosen always; any other register column the
        # record carries (Class, Decided) must agree with the plan too.
        for col in [c for c in got if c in want and c != "d"]:
            if got[col] != want[col]:
                mismatches.append(f"{did} {col}: record {got[col]!r} != plan {want[col]!r}")
        if "option chosen" not in got:
            mismatches.append(f"{did}: no option chosen")
    assert not mismatches, "DEC-CA-0070 disagrees with plan §17.4:\n" + "\n".join(mismatches)


# --------------------------------------------------------------------------
# (4) No undecided wording
# --------------------------------------------------------------------------

_NEGATION = re.compile(r"\b(no|not|never|without|nor|instead of|replaces?|replaced)\b", re.I)


def undecided_wording(text: str) -> list[str]:
    hits = []
    for n, line in enumerate(text.split("\n"), 1):
        if "[DECIDE]" in line:
            hits.append(f"{n}: [DECIDE]")
        for m in re.finditer(r"\bS6[abc]\b", line):
            hits.append(f"{n}: {m.group(0)} (stage does not exist in rev6)")
        if re.search(r"\bpending\b", line, re.I):
            hits.append(f"{n}: 'pending'")
        for m in re.finditer(r"\bproposed\b", line, re.I):
            if not re.match(r"\s+(design\s+)?changes?\b", line[m.end():], re.I):
                hits.append(f"{n}: 'proposed'")
        for m in re.finditer(r"\bshadow\w*", line, re.I):
            before = line[max(0, m.start() - 60):m.start()]
            before = re.split(r"[.;!?]", before)[-1]
            if not _NEGATION.search(before):
                hits.append(f"{n}: '{m.group(0)}' (not negated)")
        if re.match(r"^\s*#+.*\bopen decisions\b", line, re.I):
            hits.append(f"{n}: 'Open decisions' heading")
        if re.search(r"\bnone (is|are) adopted\b", line, re.I):
            hits.append(f"{n}: 'none is adopted'")
    return hits


def test_undecided_wording_scanner_accepts_register_text():
    """The plan's own register cells must pass the scan, so a faithful copy can."""
    cells = [v for row in plan_register().values() for v in row.values()]
    assert not undecided_wording("\n".join(cells))


def test_dec70_has_no_undecided_wording():
    hits = undecided_wording(DEC70.read_text(encoding="utf-8"))
    assert not hits, f"{DEC70.name} still has undecided wording:\n" + "\n".join(hits)


# --------------------------------------------------------------------------
# (5) DEC-CA-0062 back-link
# --------------------------------------------------------------------------

def test_dec62_marked_superseded_by_dec70():
    fm, _ = parse_record(DEC62)
    assert not front_matter_errors(DEC62, fm), front_matter_errors(DEC62, fm)
    assert fm.get("status") == "superseded", f"{DEC62.name} status is {fm.get('status')!r}"
    note = fm.get("note", "")
    assert isinstance(note, str) and note.startswith("Superseded by DEC-CA-0070"), (
        f"{DEC62.name} needs note: \"Superseded by DEC-CA-0070 ...\"; got {note!r}"
    )


@pytest.mark.parametrize("path", [DEC70, DEC62], ids=lambda p: p.name)
def test_records_exist(path):
    assert path.is_file(), f"{path} missing"
