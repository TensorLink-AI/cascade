"""Shared helpers for the lane D loop tooling (plan §3 conventions, §17).

Imported by ``accept``, ``status`` and the scanners, which all live next to
this file and are run as ``python scripts/laned/<tool>``; Python puts the
script's directory first on ``sys.path``, so ``import lanedlib`` resolves here.
Stdlib only.
"""

from __future__ import annotations

import fnmatch
import hashlib
import itertools
import json
import os
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
    if wp == "W3a" and w3a_nolock_path_verified(data):
        return True, "ok"
    if wp == "WB19" and data.get("passed") is False:
        why = wb19_partial_problem(data)
        return (True, "ok (partial WB19 evidence, rev6.16)") if why is None else (False, why)
    if data.get("passed") is not True:
        return False, "evidence does not record passed: true"
    return True, "ok"


_WB19_MISSING = re.compile(r"missing expert runs: \[(.*)\]")


def wb19_partial_problem(data: dict) -> str | None:
    """Partial WB19 evidence (owner 2026-11-23, plan rev6.16) records
    ``passed: false`` by design. It is accepted only when ``coverage`` is
    its sole failing check, that check's detail names the missing SKUs (a
    non-empty list, none of which has a table row), and ``body_sha256``
    matches the body. WB19's own ``test_evidence_*`` tests re-derive the
    candidate list from the shipped chain.toml and OA8; this is the
    tooling's shape check. None = accepted, else the refusal reason."""
    checks = data.get("checks")
    if not isinstance(checks, list) or not checks or not all(
            isinstance(c, dict) and isinstance(c.get("name"), str)
            and isinstance(c.get("passed"), bool) for c in checks):
        return "partial WB19 evidence: checks malformed"
    failing = [c for c in checks if not c["passed"]]
    if [c["name"] for c in failing] != ["coverage"]:
        return ("partial WB19 evidence: coverage must be the sole failing check (failing: "
                + ", ".join(sorted(c["name"] for c in failing)) + ")")
    m = _WB19_MISSING.search(str(failing[0].get("detail", "")))
    missing = re.findall(r"'([^']+)'", m.group(1)) if m else []
    if not missing:
        return "partial WB19 evidence: the coverage detail names no missing SKU"
    table = data.get("table")
    rows = table.get("rows") if isinstance(table, dict) else None
    if not isinstance(rows, dict) or not rows:
        return "partial WB19 evidence: no measured rows"
    measured = {str(k).lower() for k in rows}
    named_measured = sorted(s for s in missing if s.lower() in measured)
    if named_measured:
        return (f"partial WB19 evidence: coverage names measured SKU(s) as missing: "
                f"{named_measured}")
    sha = data.get("body_sha256")
    body = {k: v for k, v in data.items() if k != "body_sha256"}
    try:
        want = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"),
                                         allow_nan=False).encode()).hexdigest()
    except ValueError:
        return "partial WB19 evidence: body is not canonical JSON"
    if sha != want:
        return "partial WB19 evidence: body_sha256 does not match the body"
    return None


def w3a_nolock_path_verified(data: dict) -> bool:
    """A no-lock W3a run (bucket_lock asserted by the owner flag) records
    top-level passed: false by design (its non-path probes error) and is
    sha-bound; its gate verdict is verified_live.verified AND
    verified_live.path_verdict.passed. A lock-enabled run still needs
    top-level passed: true."""
    lock = data.get("bucket_lock")
    if not isinstance(lock, dict) or lock.get("asserted_by") != "owner_flag":
        return False
    live = data.get("verified_live")
    if not isinstance(live, dict) or live.get("verified") is not True:
        return False
    verdict = live.get("path_verdict")
    return isinstance(verdict, dict) and verdict.get("passed") is True


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


def plan_wp_ids(plan_text: str) -> list[str]:
    """Every WP id in the §3.0 WP tables (first header cell ``WP``, with a
    ``Runs in`` column), in plan order, first occurrence only."""
    ids: list[str] = []
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
        if not header or header[0] != "WP" or "Runs in" not in header:
            continue
        for raw in block[2:]:
            row = _plan_row(raw)
            wp = _plan_cell(row[0]) if row else ""
            if wp and wp not in ids:
                ids.append(wp)
    return ids


def plan_wp_depends(plan_text: str) -> dict[str, str | None]:
    """``{WP: Depends cell}`` for every WP in the §3.0 WP tables (first header
    cell ``WP``, with a ``Runs in`` column), first occurrence only. The value
    is None when the WP's table has no ``Depends`` column (its dependencies
    cannot be told from the plan)."""
    out: dict[str, str | None] = {}
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
        if not header or header[0] != "WP" or "Runs in" not in header:
            continue
        k = header.index("Depends") if "Depends" in header else None
        for raw in block[2:]:
            row = _plan_row(raw)
            wp = _plan_cell(row[0]) if row else ""
            if wp and wp not in out:
                out[wp] = _plan_cell(row[k]) if k is not None and k < len(row) else None
    return out


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


BATCH_IMPL_MAX = 3  # plan rev6.3 (b): an implementation PR carries at most 3 WPs


def is_batch_impl_commit(files: list[str], paths: list[str],
                         others: list[list[str]] | None) -> bool:
    """A batch implementation squash merge (plan rev6.3 (b)) for the WP with
    ``paths``: no tests/accept/ file, some file in the WP's own PATHS, and at
    most BATCH_IMPL_MAX - 1 sets of ``others`` (the other WPs' PATHS) that,
    with the WP's own, match every file exactly once and each match a file
    (the batch's check-pr guaranteed disjointness only among its own WPs, so
    a broad PATHS of a WP outside the batch never spoils the attribution)."""
    if not files or not others or any(f.startswith("tests/accept/") for f in files):
        return False
    rest = [f for f in files if not path_matches(f, paths)]
    if not rest or len(rest) == len(files):
        return False
    cands = [s for s in others if any(path_matches(f, s) for f in rest)]
    for n in range(1, BATCH_IMPL_MAX):
        for combo in itertools.combinations(cands, n):
            chosen = [paths, *combo]
            if all(sum(path_matches(f, s) for s in chosen) == 1 for f in files) \
                    and all(any(path_matches(f, s) for f in files) for s in combo):
                return True
    return False


def is_coupled_impl_commit(files: list[str], group: list[list[str]]) -> bool:
    """A coupled implementation squash merge (scripts/laned/coupled_batches)
    for a WP of ``group`` (the PATHS of every member of its declared coupled
    set, its own included): no tests/accept/ file, every file in the union
    of the members' PATHS, and every member's PATHS matching a file (the
    whole set landed together)."""
    if not files or len(group) < 2 or any(f.startswith("tests/accept/") for f in files):
        return False
    return all(any(path_matches(f, s) for s in group) for f in files) \
        and all(any(path_matches(f, s) for f in files) for s in group)


def impl_commit_index(history: list[tuple[str, list[str]]], wp: str, paths: list[str],
                      others: list[list[str]] | None = None,
                      coupled: list[list[list[str]]] | None = None,
                      subjects: dict[str, str] | None = None,
                      pr_labels: dict[int, set[str]] | None = None) -> int | None:
    """The L2 commit (plan §17.1).

    LABELLED rule (rev6.17): a commit whose subject ends ``(#N)`` with N's
    labels known (``pr_labels``: from the GitHub PR list and the ledger's PR
    rows) is the WP's L2 commit iff the PR carried ``laned:<WP>`` AND the
    commit changes at least one non-test path (outside tests/) its PATHS
    match. A known PR without the label is never the WP's L2, whatever paths
    it touches (#817: broad PATHS made W5i read L2 from another WP's merge).
    It may be the commit that adds the PATHS file (a combined PR).

    Legacy git-only rule (a commit with no PR number or unknown labels): the
    first commit after the one that added the WP's PATHS file that changes
    something outside tests/accept/ and only paths inside PATHS. With
    ``others`` (every other WP's PATHS) a batch implementation merge counts
    too (is_batch_impl_commit); with ``coupled`` (per declared coupled set
    holding the WP, every member's PATHS) a coupled implementation merge
    counts too (is_coupled_impl_commit). None when absent."""
    rel = f"tests/accept/{wp_dir(wp)}/PATHS"
    start = next((i for i, (_, files) in enumerate(history) if rel in files), None)
    if start is None:
        return None
    for i in range(start, len(history)):
        sha, files = history[i]
        labels = labels_of_commit(subjects, pr_labels, sha)
        if labels is not None:
            if wp in labels and any(not is_test_path(f) and path_matches(f, paths) for f in files):
                return i
            continue
        if i == start or not files or all(f.startswith("tests/accept/") for f in files):
            continue
        if all(path_matches(f, paths) for f in files) or is_batch_impl_commit(files, paths, others) \
                or any(is_coupled_impl_commit(files, g) for g in coupled or ()):
            return i
    return None


# ---------------------------------------------------------------------------
# Feature PRs (rev6.17): the squash merge of a green laned:* PR, by PR number
# ---------------------------------------------------------------------------

# The GitHub squash / merge-queue subject ends with "(#<PR number>)".
PR_SUBJECT = re.compile(r"\(#(\d+)\)\s*$")
# A PR label that is a flag, never a WP: lets a combined (tests + code) PR
# delete or weaken existing assertions of its WPs' tests (accept --check-pr).
CORRECTION_LABEL = "tests-correction"
# PR-run ledger modes whose green result attributes the PR's squash merge.
PR_MERGE_MODES = ("accept", "combined")
LEDGER_REFS = ("origin/laned/build-state", "laned/build-state")
LEDGER_FILE = "status/ci_runs.json"


def pr_number(subject: str) -> int | None:
    m = PR_SUBJECT.search(subject or "")
    return int(m.group(1)) if m else None


def labels_of_commit(subjects: dict[str, str] | None, pr_labels: dict[int, set[str]] | None,
                     sha: str) -> set[str] | None:
    """The laned WP labels of the PR whose squash merge is ``sha`` (subject
    ``... (#N)``), or None when unknown (no PR number, or N not in ``pr_labels``)."""
    if not subjects or pr_labels is None:
        return None
    n = pr_number(subjects.get(sha, ""))
    return pr_labels.get(n) if n is not None else None


def parse_pr_labels(text: str | None) -> dict[int, set[str]]:
    """``{PR number: {WP, ...}}`` from ``gh pr list --json number,labels``
    output (a list of ``{"number", "labels": [{"name"}]}``) or a plain
    ``{"<N>": ["laned:W1", ...]}`` map. Only ``laned:<WP>`` labels count
    (the laned:tests-correction flag is not a WP); a PR with none maps to an
    empty set (known: carries no WP label)."""
    try:
        data = json.loads(text) if text else None
    except json.JSONDecodeError:
        return {}
    items: list[tuple[object, object]] = []
    if isinstance(data, list):
        items = [(d.get("number"), [x.get("name") if isinstance(x, dict) else x
                                    for x in d.get("labels") or []])
                 for d in data if isinstance(d, dict)]
    elif isinstance(data, dict):
        items = list(data.items())
    out: dict[int, set[str]] = {}
    for n, labels in items:
        try:
            num = int(n)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        out[num] = {str(lab).split(":", 1)[1] for lab in labels or []  # type: ignore[union-attr]
                    if isinstance(lab, str) and lab.startswith("laned:") and lab != f"laned:{CORRECTION_LABEL}"}
    return out


def pr_labels_from_runs(runs: list[dict]) -> dict[int, set[str]]:
    """``{PR number: {WP, ...}}`` from ledger rows with a ``pr`` field (the PR
    job records a row per laned:<WP> label)."""
    out: dict[int, set[str]] = {}
    for r in runs:
        pr = r.get("pr")
        if isinstance(pr, int) and not isinstance(pr, bool) and isinstance(r.get("wp"), str) \
                and r.get("event") == "pull_request":
            out.setdefault(pr, set()).add(r["wp"])
    return out


def merge_pr_labels(*maps: dict[int, set[str]]) -> dict[int, set[str]]:
    out: dict[int, set[str]] = {}
    for m in maps:
        for n, labels in m.items():
            out.setdefault(n, set()).update(labels)
    return out


def pr_labels_file() -> dict[int, set[str]]:
    """``$LANED_PR_LABELS`` (``gh pr list --state merged --json number,labels``
    written by the workflow), or none."""
    env = os.environ.get("LANED_PR_LABELS")
    if not env or not Path(env).is_file():
        return {}
    return parse_pr_labels(Path(env).read_text(encoding="utf-8"))


def first_parent_subjects(root: Path, ref: str) -> dict[str, str]:
    """``{sha: subject}`` along ``ref``'s first-parent line."""
    out = git(root, "log", "--first-parent", "--format=%H%x09%s", ref, check=False).stdout
    subjects: dict[str, str] = {}
    for line in out.splitlines():
        sha, _, subject = line.partition("\t")
        if sha:
            subjects[sha] = subject
    return subjects


def green_pr_numbers(runs: list[dict], wp: str, modes: tuple[str, ...] = PR_MERGE_MODES) -> set[int]:
    """PR numbers whose LAST pull_request ledger row for ``wp`` (any of
    ``modes``, or a check-pr row) is a green row in one of ``modes``: a PR
    that went red (or failed check-pr) after a green run never counts."""
    last: dict[int, dict] = {}
    for r in runs:
        pr = r.get("pr")
        if (r.get("wp") != wp or r.get("event") != "pull_request"
                or r.get("mode") not in (*PR_MERGE_MODES, "check-pr")
                or not isinstance(pr, int) or isinstance(pr, bool)):
            continue
        if pr not in last or str(r.get("created_at", "")) >= str(last[pr].get("created_at", "")):
            last[pr] = r
    return {pr for pr, r in last.items() if r.get("mode") in modes and r.get("conclusion") == "success"}


def pr_merge_index(history: list[tuple[str, list[str]]], subjects: dict[str, str], wp: str,
                   paths: list[str], prs: set[int], need_tests: bool = False) -> int | None:
    """The first first-parent commit that is the squash merge (subject
    ``... (#N)``) of a PR in ``prs`` and changes a path the WP owns: one its
    PATHS match or one under tests/accept/<wp>/ (with ``need_tests``: some
    file under tests/accept/<wp>/). At or after the commit adding the PATHS
    file (a combined PR may add it in the same merge). None when absent."""
    if not prs:
        return None
    own = f"tests/accept/{wp_dir(wp)}/"
    rel = own + "PATHS"
    start = next((i for i, (_, files) in enumerate(history) if rel in files), None)
    if start is None:
        return None
    for i in range(start, len(history)):
        sha, files = history[i]
        m = PR_SUBJECT.search(subjects.get(sha, ""))
        if not m or int(m.group(1)) not in prs:
            continue
        tests = any(f.startswith(own) for f in files)
        if (tests if need_tests else tests or any(path_matches(f, paths) for f in files)):
            return i
    return None


def ledger_runs(root: Path) -> list[dict]:
    """The CI results ledger (``status/ci_runs.json`` on laned/build-state):
    ``$LANED_LEDGER`` (a file) when set, else the first of LEDGER_REFS that
    has it, else none. Used by ``accept`` to see feature-PR L2 merges."""
    env = os.environ.get("LANED_LEDGER")
    text: str | None = None
    if env:
        p = Path(env)
        text = p.read_text(encoding="utf-8") if p.is_file() else None
    else:
        for ref in LEDGER_REFS:
            proc = git(root, "show", f"{ref}:{LEDGER_FILE}", check=False)
            if proc.returncode == 0:
                text = proc.stdout
                break
    try:
        data = json.loads(text) if text else {}
    except json.JSONDecodeError:
        return []
    runs = data.get("runs") if isinstance(data, dict) else None
    return [r for r in runs if isinstance(r, dict)] if isinstance(runs, list) else []


# ---------------------------------------------------------------------------
# Coupled batches (legacy since rev6.17: still read for L2 attribution of
# merges made under the COUPLED rule; check-pr no longer needs them)
# ---------------------------------------------------------------------------

COUPLED_BATCHES_PATH = "scripts/laned/coupled_batches"


def parse_coupled_batches(text: str | None) -> list[tuple[str, ...]]:
    """Declared coupled sets in a ``coupled_batches`` file: one set per line,
    WP ids separated by blanks, ``#`` comments (whole-line or trailing) and
    blank lines ignored. A line naming fewer than two distinct WPs is not a
    set. Duplicates within a line collapse; order is kept."""
    out: list[tuple[str, ...]] = []
    for line in (text or "").splitlines():
        ids = tuple(dict.fromkeys(line.split("#", 1)[0].split()))
        if len(ids) >= 2:
            out.append(ids)
    return out


def coupled_batches_at(root: Path, ref: str) -> list[tuple[str, ...]]:
    """The declared coupled sets at ``ref``. Read at the base / board ref,
    never from the PR, and only a W0c PR may edit scripts/laned/, so a PR can
    never declare its own WPs coupled. No file at ``ref`` means none."""
    proc = git(root, "show", f"{ref}:{COUPLED_BATCHES_PATH}", check=False)
    return parse_coupled_batches(proc.stdout) if proc.returncode == 0 else []


def coupled_set_of(wps, batches: list[tuple[str, ...]]) -> tuple[str, ...] | None:
    """The declared coupled set equal to ``wps`` (as a set), or None. Only the
    exact set is coupled: a subset or superset keeps the ordinary rules."""
    want = set(wps)
    return next((b for b in batches if set(b) == want), None)


def coupled_paths(batches: list[tuple[str, ...]], sets: dict[str, list[str]],
                  wp: str) -> list[list[list[str]]]:
    """For each declared coupled set holding ``wp`` whose every member has a
    PATHS file in ``sets`` (dir -> entries), the members' PATHS (for
    impl_commit_index)."""
    out: list[list[list[str]]] = []
    for b in batches:
        if wp in b and all(wp_dir(m) in sets for m in b):
            out.append([sets[wp_dir(m)] for m in b])
    return out


def paths_sets_at(root: Path, ref: str) -> dict[str, list[str]]:
    """``{tests/accept dir name: PATHS entries}`` for every WP dir with a PATHS file at ``ref``."""
    proc = git(root, "ls-tree", "-r", "--name-only", ref, "--", "tests/accept", check=False)
    out: dict[str, list[str]] = {}
    for rel in proc.stdout.split("\n"):
        parts = rel.split("/")
        if len(parts) == 4 and parts[3] == "PATHS":
            shown = git(root, "show", f"{ref}:{rel}", check=False)
            if shown.returncode == 0:
                out[parts[2]] = parse_paths(shown.stdout)
    return out


def other_paths(sets: dict[str, list[str]], wp: str) -> list[list[str]]:
    """Every PATHS set of ``sets`` but the one of ``wp``."""
    own = wp_dir(wp)
    return [s for d, s in sorted(sets.items()) if d != own]


# ---------------------------------------------------------------------------
# Verification WPs (a tests-only PR whose tests pass on code another WP built)
# ---------------------------------------------------------------------------

VERIFICATION_WPS_PATH = "scripts/laned/verification_wps"


def parse_verification_wps(text: str | None) -> set[str]:
    """WP ids in a ``verification_wps`` file: one per line, ``#`` comments
    (whole-line or trailing) and blank lines ignored."""
    out: set[str] = set()
    for line in (text or "").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.add(line)
    return out


def verification_wps_at(root: Path, ref: str) -> set[str]:
    """The declared verification WPs at ``ref`` (RV5): only these may take MODE
    verification or an L2 via verification. The list lives under
    scripts/laned/, which only a W0c PR may edit, and is read at the base /
    board ref, so a PR can never declare its own WP a verification WP.
    No file at ``ref`` means no verification WP."""
    proc = git(root, "show", f"{ref}:{VERIFICATION_WPS_PATH}", check=False)
    return parse_verification_wps(proc.stdout) if proc.returncode == 0 else set()


# ---------------------------------------------------------------------------
# Retired WPs (an owner decision dropped the WP: it is never built)
# ---------------------------------------------------------------------------

RETIRED_WPS_PATH = "scripts/laned/retired_wps"


def parse_retired_wps(text: str | None) -> dict[str, str]:
    """``{WP id: owner decision reference}`` from a ``retired_wps`` file: one
    WP per line, ``<WP> <reference>``; ``#`` comments (whole-line or
    trailing) and blank lines ignored. A line naming a WP without a
    reference raises ValueError: a WP is retired only by a cited owner
    decision, so a bare id fails closed instead of retiring silently."""
    out: dict[str, str] = {}
    for n, line in enumerate((text or "").splitlines(), 1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        wp, _, ref = line.partition(" ")
        if not ref.strip():
            raise ValueError(f"{RETIRED_WPS_PATH}:{n}: {wp} has no owner decision reference")
        out[wp] = ref.strip()
    return out


def retired_wps_at(root: Path, ref: str) -> dict[str, str]:
    """The retired WPs at ``ref``. Like verification_wps, the list lives under
    scripts/laned/ (only a W0c PR may edit it) and is read at the base / board
    ref, never from the PR, so a PR cannot retire a WP to dodge its Accept.
    No file at ``ref`` means no retired WP."""
    proc = git(root, "show", f"{ref}:{RETIRED_WPS_PATH}", check=False)
    return parse_retired_wps(proc.stdout) if proc.returncode == 0 else {}


def retired_dirs(retired: dict[str, str]) -> set[str]:
    """The tests/accept dir names of the retired WPs."""
    return {wp_dir(w) for w in retired}


def is_test_path(rel: str) -> bool:
    return rel.split("/", 1)[0] == "tests"


def missing_paths_at(root: Path, ref: str, paths: list[str]) -> list[str]:
    """The non-test PATHS entries with no file at ``ref`` (an exact entry: that
    file; a ``dir/`` entry: some file under it; a glob: some matching file).
    A PATHS file with no non-test entry at all has nothing to verify, so it is
    reported as missing too. Empty list: every non-test entry exists at ``ref``."""
    entries = [e for e in paths if not is_test_path(e)]
    if not entries:
        return ["(no non-test PATHS entry)"]
    proc = git(root, "ls-tree", "-r", "--name-only", f"{ref}^{{commit}}", check=False)
    if proc.returncode != 0:
        return list(entries)
    files = [f for f in proc.stdout.split("\n") if f]
    return [e for e in entries if not any(path_matches(f, [e]) for f in files)]


def wp_tree_at(root: Path, ref: str, wp: str) -> str | None:
    """Tree sha of ``tests/accept/<wp>/`` at ``ref``: what a verification run
    records, so the board can find the merge commit after a squash merge."""
    proc = git(root, "rev-parse", "-q", "--verify", f"{ref}:tests/accept/{wp_dir(wp)}", check=False)
    return proc.stdout.strip() if proc.returncode == 0 else None


def tests_commit_index(root: Path, history: list[tuple[str, list[str]]], wp: str,
                       trees: set[str], shas: set[str]) -> int | None:
    """The merge of a green tests-only PR run for ``wp``: the first commit on
    the first-parent line, at or after the one that added the WP's PATHS, that
    changes only files under tests/accept/ (some of them under
    tests/accept/<wp>/) and whose test dir is the one the run checked (its
    ``tests_tree``), or whose sha is that run's ``head_sha``. A batch
    tests-only squash commit (several WPs' test dirs, plan rev6.2 (h)) is
    found for each of its WPs through that WP's own tree. Such a commit never
    is an L2 implementation commit (impl_commit_index skips tests-only
    commits). None when absent."""
    own = f"tests/accept/{wp_dir(wp)}/"
    rel = own + "PATHS"
    start = next((i for i, (_, files) in enumerate(history) if rel in files), None)
    if start is None or not (trees or shas):
        return None
    for i in range(start, len(history)):
        sha, files = history[i]
        if not files or not all(f.startswith("tests/accept/") for f in files) \
                or not any(f.startswith(own) for f in files):
            continue
        if sha in shas or (trees and wp_tree_at(root, sha, wp) in trees):
            return i
    return None


def verification_commit_index(root: Path, history: list[tuple[str, list[str]]], wp: str,
                              trees: set[str], shas: set[str]) -> int | None:
    """The L2 commit of a verification WP: the merge of its green
    ``verification`` run (tests_commit_index), batch squash commits included."""
    return tests_commit_index(root, history, wp, trees, shas)
