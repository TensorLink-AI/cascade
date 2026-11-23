"""W13: ``.github/workflows/laned-king-sync.yml`` and the publication code's
static properties (no Object Lock API, the workflow runs the tested code
path, lane A's king-sync untouched). README.md has the contract.

On main the workflow, the sync script and ``cascade/lane_d/publish.py`` do
not exist: those tests fail on an AssertionError naming the missing file.
"""

from __future__ import annotations

import ast
import re

import pytest
import yaml

from . import helpers as h
from .helpers import no_network  # noqa: F401  (autouse: no socket connects)


def _workflow() -> dict:
    assert h.WORKFLOW.is_file(), f"missing {h.WORKFLOW.relative_to(h.REPO)}"
    doc = yaml.safe_load(h.WORKFLOW.read_text(encoding="utf-8"))
    assert isinstance(doc, dict), "workflow is not a mapping"
    return doc


def _triggers(doc: dict) -> dict:
    on = doc.get("on", doc.get(True))     # YAML 1.1 reads a bare `on:` as True
    if isinstance(on, str):
        return {on: None}
    if isinstance(on, list):
        return {k: None for k in on}
    assert isinstance(on, dict), on
    return on


def _runs(doc: dict) -> list[str]:
    out = []
    for job in (doc.get("jobs") or {}).values():
        for step in job.get("steps") or []:
            if isinstance(step, dict) and isinstance(step.get("run"), str):
                out.append(step["run"])
    return out


def test_workflow_runs_on_a_schedule_and_by_hand():
    on = _triggers(_workflow())
    assert "schedule" in on and on["schedule"], on
    assert "workflow_dispatch" in on, on


def test_workflow_may_commit_the_cards():
    perms = _workflow().get("permissions") or {}
    assert perms.get("contents") == "write", perms


def test_workflow_runs_the_sync_script():
    """The workflow calls scripts/laned_king_sync.py, which imports the
    tested module (no copy of the publication logic in the workflow)."""
    runs = "\n".join(_runs(_workflow()))
    assert "scripts/laned_king_sync.py" in runs, runs
    assert h.SYNC_SCRIPT.is_file(), f"missing {h.SYNC_SCRIPT.relative_to(h.REPO)}"
    tree = ast.parse(h.SYNC_SCRIPT.read_text(encoding="utf-8"))
    mods = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module:
            mods.add(n.module)
            mods |= {f"{n.module}.{a.name}" for a in n.names}
    assert h.MODULE in mods, sorted(mods)


def test_workflow_needs_no_secret_but_its_own_token():
    """The sync reads the public signed index and writes cards; it holds no
    storage or HF credential (those stay with the validator's publish)."""
    text = h.WORKFLOW.read_text(encoding="utf-8") if h.WORKFLOW.is_file() else ""
    assert text, f"missing {h.WORKFLOW.relative_to(h.REPO)}"
    used = set(re.findall(r"secrets\.([A-Za-z0-9_]+)", text))
    assert used <= {"GITHUB_TOKEN"}, sorted(used)


def test_workflow_only_stages_lane_d_cards():
    """Every `git add` in the workflow names a path under champions/lane_d
    (no weights, nothing of lane A)."""
    adds = []
    for run in _runs(_workflow()):
        for line in run.splitlines():
            m = re.search(r"\bgit\s+add\b(.*)", line)
            if m:
                adds.append(m.group(1).split())
    for args in adds:
        paths = [a for a in args if not a.startswith("-")]
        assert paths, args
        assert all(p.strip("'\"").startswith("champions/lane_d") for p in paths), args
        assert not any(p in ("-A", "--all", "-u", "--update") for p in args), args


def test_publication_code_has_no_object_lock_api():
    """rev6.4: Hippius has no Object Lock. The publication path never asks
    for one (no lock or retention parameter, no lock call)."""
    for path in (h.PUBLISH_SRC, h.SYNC_SCRIPT):
        assert path.is_file(), f"missing {path.relative_to(h.REPO)}"
        src = path.read_text(encoding="utf-8")
        found = [api for api in h.LOCK_API if api.lower() in src.lower()]
        assert not found, (path.name, found)


@pytest.mark.parametrize("path", [h.LANE_A_WORKFLOW])
def test_lane_a_king_sync_is_untouched(path):
    """Invariant 1 for lane A: its king-sync never learns about lane D."""
    text = path.read_text(encoding="utf-8")
    assert "lane_d" not in text and "laned" not in text
