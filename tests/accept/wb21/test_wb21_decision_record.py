"""WB21 acceptance: the DEC-CA-0071 amendment (lane B rev4) is recorded as decided.

Plan §3.0 WB21 row: "DEC-CA-0071 amendment (this PR drafts it; owner
accepts)", runs in ``ci + owner``. Plan §12b says the lane B rev4 rules are
recorded in the DEC-CA-0071 amendment (§"Amendment rev4"); Appendix C is the
text of that amendment; §17.4 is the decision register.

Everything the record must say is parsed from ``docs/plans/LANE_D_PLAN.md``
on every run (Appendix C's items, its new-key list, the §17.4 rows for
D35-D42 and D42b, the §12b.6 D42 rules), so an owner-merged plan change
flows through. The front-matter rules and the undecided-wording scan are the
same as W0's (tests/accept/w0/ on PR #489), so the two decision WPs agree.
See README.md in this directory for the full contract.
"""

from __future__ import annotations

import datetime as _dt
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DECISIONS = REPO / "decisions"
PLAN = REPO / "docs" / "plans" / "LANE_D_PLAN.md"
DEC71 = DECISIONS / "DEC-CA-0071-lane-b-public-suite-targets.md"

# The lane B decisions this record carries (§12b, register rows D35-D42, D42b).
WB21_IDS = {f"D{n}" for n in range(35, 43)} | {"D42b"}

# W0's front-matter rules (identical constants).
UNIVERSAL_KEYS = ("id", "type", "title", "status", "date", "tags")
REQUIRED_KEYS = UNIVERSAL_KEYS + ("revisit_when", "relations")
OPTIONAL_KEYS = ("note", "amended")
STATUSES = {"proposed", "accepted", "active", "superseded"}
DECIDED_STATUSES = {"accepted", "active"}
REGISTER_STATUSES = {"decided", "withdrawn", "deferred-to-stage"}
DEC_ID = re.compile(r"^DEC-CA-\d{4}$")


# --------------------------------------------------------------------------
# Front matter (the subset of YAML the decision records use; same as W0)
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
    assert path.is_file(), f"{path} missing"
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
    """W0's decision-record rules. ``strict=False`` is the floor every record
    on main obeys; ``strict=True`` adds revisit_when + relations as required
    and restricts relation targets to existing DEC-CA records."""
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
# Markdown helpers
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


def norm(text: str) -> str:
    """Drop ``**`` and collapse whitespace (also used for register cells)."""
    return re.sub(r"\s+", " ", text.replace("**", "")).strip()


def section(text: str, heading: re.Pattern, where: str) -> str:
    """The body under the first heading matching ``heading``, up to the next
    heading of the same or a higher level."""
    lines = text.split("\n")
    for i, line in enumerate(lines):
        m = re.match(r"^(#+)\s+(.*)$", line)
        if m and heading.match(m.group(2)):
            level = len(m.group(1))
            out = []
            for nxt in lines[i + 1:]:
                h = re.match(r"^(#+)\s", nxt)
                if h and len(h.group(1)) <= level:
                    break
                out.append(nxt)
            return "\n".join(out)
    raise AssertionError(f"{where}: no heading matching {heading.pattern!r}")


AMENDMENT_HEADING = re.compile(r"^Amendment rev4\b", re.I)


def numbered_items(text: str) -> dict[int, str]:
    """Top-level ``N. ...`` items: the item line plus following blank or
    indented lines, up to the next unindented non-empty line."""
    items: dict[int, str] = {}
    cur: int | None = None
    buf: list[str] = []
    for line in text.split("\n"):
        m = re.match(r"^(\d+)\.\s+", line)
        if m:
            if cur is not None:
                items[cur] = "\n".join(buf)
            cur, buf = int(m.group(1)), [line]
            assert cur not in items, f"item {cur} appears twice"
        elif cur is not None and (not line.strip() or line.startswith((" ", "\t"))):
            buf.append(line)
        elif cur is not None:
            items[cur] = "\n".join(buf)
            cur, buf = None, []
    if cur is not None:
        items[cur] = "\n".join(buf)
    return items


def code_tokens(text: str) -> list[str]:
    """Every inline-code span, whitespace-collapsed, in order, deduplicated."""
    out: list[str] = []
    for m in re.finditer(r"`([^`]+)`", text):
        tok = norm(m.group(1))
        if tok not in out:
            out.append(tok)
    return out


def item_lead(item: str) -> str:
    """The bold lead of an item with the undecided parts removed: a
    parenthetical holding ``[DECIDE]``, any bare ``[DECIDE]``, and trailing
    punctuation. ``**Sentinel ([DECIDE] D35).**`` -> ``Sentinel``."""
    m = re.match(r"^\d+\.\s+\*\*(.+?)\*\*", item.strip(), re.S)
    assert m, f"item has no bold lead: {item[:80]!r}"
    lead = re.sub(r"\s*\([^()]*\[DECIDE\][^()]*\)", "", m.group(1))
    lead = lead.replace("[DECIDE]", "")
    return norm(lead).rstrip(".:; ")


def d_numbers(text: str) -> set[str]:
    return set(re.findall(r"\bD\d+[a-z]?\b", text))


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


def _is_register(header: list[str]) -> bool:
    return [norm(h).lower() for h in header[:3]] == ["d", "status", "option chosen"]


# --------------------------------------------------------------------------
# What the plan says the record must hold
# --------------------------------------------------------------------------

def plan_text() -> str:
    return PLAN.read_text(encoding="utf-8")


def plan_register() -> dict[str, dict]:
    m = re.search(r"^### 17\.4 Decision register.*?$(.*?)^### ", plan_text(), re.M | re.S)
    assert m, "plan §17.4 'Decision register' section not found"
    for header, rows in tables(m.group(1)):
        if _is_register(header):
            return _register_rows(header, rows, "plan §17.4")
    raise AssertionError("plan §17.4 has no '| D | Status | Option chosen |' table")


def plan_amendment() -> str:
    appendix = section(plan_text(), re.compile(r"^Appendix C\b"), "plan")
    return section(appendix, AMENDMENT_HEADING, "plan Appendix C")


def plan_lb6() -> str:
    return section(plan_text(), re.compile(r"^12b\.6\b"), "plan §12b")


def new_keys(text: str) -> list[str]:
    """The bullets under the ``New `[lane_b]` keys:`` line."""
    m = re.search(r"^New `\[lane_b\]` keys:\s*\n((?:\s*\n|\s*- .*\n?)+)", text, re.M)
    assert m, "no 'New `[lane_b]` keys:' list"
    return [key for line in m.group(1).split("\n") for key in re.findall(r"^\s*- `([^`]+)`", line)]


def each_key_paragraph(text: str) -> str:
    m = re.search(r"^Each key:.*?(?=\n\s*\n|\Z)", text, re.M | re.S)
    assert m, "no 'Each key:' paragraph"
    return m.group(0)


# D42 / D42b rules (§12b.6 "D42 decided" + Appendix C item 6). Every pattern
# is checked against the plan itself (test_plan_states_every_d42_rule), so
# the list is the plan's wording, not ours. Patterns run on norm()'d text.
D42_RULES = (
    ("lane B GPU list `[lane_b] funded_pod_skus`", r"\[lane_b\] funded_pod_skus"),
    ("the lane B list is separate from `[round]`", r"separate from `?\[round\]"),
    ("a GPU type is image-qualified before it is offered", r"qualified"),
    ("a GPU type is measured before it is offered", r"\bmeasured\b"),
    ("miner-chosen `train_hours`", r"\btrain_hours\b"),
    ("`train_hours` lower bound 0.5", r"\b0\.5\b"),
    ("`max_leg_hours` is 6", r"max_leg_hours`?\s*(?:=|\(|is)?\s*6(?:\.0)?\b"),
    ("in 0.5 h steps", r"\b0\.5\s*h\s+steps\b"),
    ("guard factor `guard_factor`", r"\bguard_factor\b"),
    ("guard capped by `[lane_b] max_leg_seconds`", r"\[lane_b\] max_leg_seconds"),
    ("`[training] max_train_seconds` is not changed",
     r"\[training\] max_train_seconds`?[^.]{0,80}\b(?:not changed|never edited|unchanged|untouched)\b"),
    ("D42b: wall hours on the chosen GPU", r"\b(?:wall|real) hours on the chosen GPU\b"),
    ("D42b: steps from `measured_steps_per_s`", r"\bmeasured_steps_per_s\b"),
    ("steps fixed at admission", r"\bfixed at admission\b"),
    ("the step count is signed", r"\bsigned\b"),
)

# MoE-ordinal declared_rows keys (§12b.2 rev5, Appendix C item 1 rev5).
MOE_RULES = (
    ("declared_rows keys are MoE ordinals 0-6", r"\bMoE ordinals?\s+0\s*(?:–|-|\.\.)\s*6\b"),
    ("declared_rows keys are not block indices", r"\bnot (?:the )?block indices\b"),
    ("the clamp minimum applies only to layers with a declared row",
     r"\bminimum applies only to layers with (?:at least one|≥\s*1|one or more) declared rows?\b"),
)


def _missing_rules(text: str, rules) -> list[str]:
    flat = norm(text)
    return [label for label, pat in rules if not re.search(pat, flat, re.I)]


# --------------------------------------------------------------------------
# What the record holds
# --------------------------------------------------------------------------

def record_amendment() -> str:
    _, body = parse_record(DEC71)
    return section(body, AMENDMENT_HEADING, DEC71.name)


def record_register() -> dict[str, dict]:
    _, body = parse_record(DEC71)
    found: dict[str, dict] = {}
    for header, rows in tables(body):
        if _is_register(header):
            for did, row in _register_rows(header, rows, DEC71.name).items():
                assert did not in found, f"{DEC71.name}: {did} recorded in two register tables"
                found[did] = row
    return found


# --------------------------------------------------------------------------
# Undecided wording (W0's scan, unchanged)
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


# --------------------------------------------------------------------------
# Sanity: the plan parses and says what the tests rely on
# --------------------------------------------------------------------------

def test_rules_hold_for_existing_decided_records():
    records = []
    for p in sorted(DECISIONS.glob("DEC-CA-*.md")):
        if p != DEC71 and parse_record(p)[0].get("status") in DECIDED_STATUSES:
            records.append(p)
    assert len(records) >= 10, "expected the repo's accepted/active records to exist"
    bad = {
        p.name: errs for p in records if (errs := front_matter_errors(p, parse_record(p)[0], strict=False))
    }
    assert not bad, f"the W0/WB21 front-matter rules reject existing decided records: {bad}"


def test_plan_register_has_wb21_decisions_decided():
    reg = plan_register()
    missing = sorted(WB21_IDS - set(reg))
    assert not missing, f"plan §17.4 register lacks lane B decisions {missing}"
    bad = {d: reg[d]["status"] for d in WB21_IDS if reg[d]["status"] not in REGISTER_STATUSES}
    assert not bad, f"plan §17.4 lane B decisions not answered or resolved: {bad}"


def test_plan_appendix_c_parses():
    amend = plan_amendment()
    items = numbered_items(amend)
    assert sorted(items) == list(range(1, len(items) + 1)) and len(items) >= 6, (
        f"plan Appendix C amendment items are {sorted(items)}"
    )
    for n, item in items.items():
        assert item_lead(item), f"plan Appendix C item {n} has an empty lead"
    assert len(new_keys(amend)) >= 11, new_keys(amend)
    assert "contamination_hashset" in each_key_paragraph(amend)


def test_plan_states_every_d42_rule():
    plan_side = plan_lb6() + "\n" + plan_amendment()
    missing = _missing_rules(plan_side, D42_RULES)
    assert not missing, f"plan §12b.6 / Appendix C no longer state: {missing}"


def test_plan_states_every_moe_ordinal_rule():
    plan_side = section(plan_text(), re.compile(r"^12b\.2\b"), "plan §12b") + "\n" + plan_amendment()
    missing = _missing_rules(plan_side, MOE_RULES)
    assert not missing, f"plan §12b.2 / Appendix C no longer state: {missing}"


def test_undecided_wording_scanner_accepts_register_text():
    """The plan's own register cells must pass the scan, so a faithful copy can."""
    reg = plan_register()
    cells = [v for d in WB21_IDS for v in reg[d].values()]
    assert not undecided_wording("\n".join(cells))


# --------------------------------------------------------------------------
# (1) Front matter: valid, unique, decided
# --------------------------------------------------------------------------

def test_dec71_front_matter_valid():
    fm, _ = parse_record(DEC71)
    errors = front_matter_errors(DEC71, fm)
    assert not errors, f"{DEC71.name} front matter: {errors}"


def test_dec71_id_unique():
    owners = [p.name for p in DECISIONS.glob("DEC-CA-*.md") if parse_record(p)[0].get("id") == "DEC-CA-0071"]
    assert owners == [DEC71.name], f"DEC-CA-0071 claimed by {owners}"


def test_dec71_status_is_decided():
    fm, _ = parse_record(DEC71)
    assert fm.get("status") in DECIDED_STATUSES, (
        f"{DEC71.name} status is {fm.get('status')!r}; WB21 needs one of {sorted(DECIDED_STATUSES)}"
    )


# --------------------------------------------------------------------------
# (2) The Appendix C amendment is in the record
# --------------------------------------------------------------------------

def test_dec71_amendment_items_match_appendix_c():
    plan_items = numbered_items(plan_amendment())
    rec_items = numbered_items(record_amendment())
    problems = []
    for n, want in sorted(plan_items.items()):
        got = rec_items.get(n)
        if got is None:
            problems.append(f"item {n}: missing from the record's 'Amendment rev4' section")
            continue
        flat = norm(got).casefold()
        lead = item_lead(want)
        if lead.casefold() not in flat:
            problems.append(f"item {n}: lead {lead!r} not stated")
        tokens = [t for t in code_tokens(want) if t.casefold() not in flat]
        if tokens:
            problems.append(f"item {n}: missing {tokens}")
        ds = sorted(d_numbers(want) - d_numbers(got))
        if ds:
            problems.append(f"item {n}: does not name {ds}")
    extra = sorted(set(rec_items) - set(plan_items))
    if extra:
        problems.append(f"record has amendment items the plan lacks: {extra}")
    assert not problems, (
        f"{DEC71.name} 'Amendment rev4' differs from plan Appendix C:\n" + "\n".join(problems)
    )


def test_dec71_amendment_lists_new_lane_b_keys():
    amend = record_amendment()
    want = new_keys(plan_amendment())
    try:
        got = new_keys(amend)
    except AssertionError:
        got = []
    missing = [k for k in want if k not in got]
    assert not missing, f"{DEC71.name} 'New `[lane_b]` keys:' list lacks {missing}"


def test_dec71_amendment_key_rules_match_appendix_c():
    want_para = each_key_paragraph(plan_amendment())
    got_para = norm(each_key_paragraph(record_amendment()))
    missing = [t for t in code_tokens(want_para) if t not in got_para]
    assert not missing, f"{DEC71.name} 'Each key:' paragraph lacks {missing}"
    assert re.search(r"`?contamination_hashset`?[^.]{0,120}\brelease-only\b"
                     r"|\brelease-only\b[^.]{0,120}`?contamination_hashset\b", got_para), (
        f"{DEC71.name} 'Each key:' paragraph must state contamination_hashset is release-only "
        f"(plan Appendix C rev5); got {got_para!r}"
    )


def test_dec71_owner_direction_recorded():
    m = re.search(r"\*\*(Owner direction \([^)]*\))\.?\*\*", plan_amendment())
    assert m, "plan Appendix C has no 'Owner direction (...)' lead"
    assert norm(m.group(1)).casefold() in norm(record_amendment()).casefold(), (
        f"{DEC71.name} amendment lacks {m.group(1)!r}"
    )


# --------------------------------------------------------------------------
# (3) D35-D42 and D42b recorded with the register's chosen option
# --------------------------------------------------------------------------

def test_dec71_records_lane_b_decisions():
    rec = record_register()
    missing = sorted(WB21_IDS - set(rec), key=lambda d: (int(re.sub(r"\D", "", d)), d))
    assert not missing, (
        f"{DEC71.name} has no '| D | Status | Option chosen |' register row for {missing}"
    )
    unknown = sorted(set(rec) - set(plan_register()))
    assert not unknown, f"{DEC71.name} records decisions the plan register lacks: {unknown}"


def test_dec71_choices_match_plan_register():
    rec = record_register()
    plan = plan_register()
    assert rec, f"{DEC71.name} has no '| D | Status | Option chosen |' register table"
    mismatches = []
    for did in sorted(WB21_IDS | set(rec)):
        want, got = plan.get(did), rec.get(did)
        if got is None:
            mismatches.append(f"{did}: not recorded")
            continue
        if want is None:
            mismatches.append(f"{did}: not in plan §17.4")
            continue
        if got["status"] not in REGISTER_STATUSES:
            mismatches.append(f"{did}: status {got['status']!r} is not answered/resolved")
        if "option chosen" not in got or not got["option chosen"]:
            mismatches.append(f"{did}: no option chosen")
        # Status and option chosen always; any other register column the
        # record carries (Class, Decided) must agree with the plan too.
        for col in [c for c in got if c in want and c != "d"]:
            if got[col] != want[col]:
                mismatches.append(f"{did} {col}: record {got[col]!r} != plan {want[col]!r}")
    assert not mismatches, "DEC-CA-0071 disagrees with plan §17.4:\n" + "\n".join(mismatches)


# --------------------------------------------------------------------------
# (4) MoE-ordinal declared_rows keys and the D42 rules
# --------------------------------------------------------------------------

def test_dec71_declared_rows_keys_are_moe_ordinals():
    missing = _missing_rules(record_amendment(), MOE_RULES)
    assert not missing, f"{DEC71.name} amendment does not state: {missing}"


def test_dec71_d42_gpu_list_and_leg_hours_rules():
    missing = _missing_rules(record_amendment(), D42_RULES)
    assert not missing, f"{DEC71.name} amendment does not state the D42/D42b rules: {missing}"


# --------------------------------------------------------------------------
# (5) No undecided wording; consistent numbering
# --------------------------------------------------------------------------

def test_dec71_has_no_undecided_wording():
    hits = undecided_wording(DEC71.read_text(encoding="utf-8"))
    assert not hits, f"{DEC71.name} still has undecided wording:\n" + "\n".join(hits)


def test_dec71_numbering_consistent():
    text = DEC71.read_text(encoding="utf-8")
    assert "DEC-CA-0069" not in text, (
        f"{DEC71.name} still refers to DEC-CA-0069 (the record was renumbered 0071; plan rev5 A7)"
    )
    fm, _ = parse_record(DEC71)
    assert fm.get("id") == "DEC-CA-0071", f"id is {fm.get('id')!r}"
