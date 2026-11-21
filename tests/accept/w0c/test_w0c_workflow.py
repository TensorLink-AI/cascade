"""W0c (rev6, ii): .github/workflows/laned-accept.yml exists, triggers on
labelled PRs and on push to main, and drives scripts/laned/accept in the
check-pr, expect-fail and accept modes."""

from __future__ import annotations

import yaml

from .helpers import REPO

WORKFLOW = REPO / ".github" / "workflows" / "laned-accept.yml"


def _load():
    assert WORKFLOW.is_file(), f"missing {WORKFLOW.relative_to(REPO)}"
    doc = yaml.safe_load(WORKFLOW.read_text())
    assert isinstance(doc, dict), "workflow is not a YAML mapping"
    # PyYAML (YAML 1.1) reads the bare key `on` as boolean True.
    triggers = doc.get("on", doc.get(True))
    assert isinstance(triggers, dict), f"workflow triggers must be a mapping, got {triggers!r}"
    return doc, triggers


def test_workflow_triggers_on_labelled_pull_requests():
    _, triggers = _load()
    pr = triggers.get("pull_request")
    assert isinstance(pr, dict), "pull_request trigger missing"
    types = pr.get("types") or []
    assert "labeled" in types, types
    assert "synchronize" in types, types


def test_workflow_triggers_on_push_to_main():
    _, triggers = _load()
    push = triggers.get("push")
    assert isinstance(push, dict), "push trigger missing"
    assert "main" in (push.get("branches") or []), push


def test_workflow_runs_accept_in_every_mode():
    doc, _ = _load()
    assert doc.get("jobs"), "workflow has no jobs"
    text = WORKFLOW.read_text()
    assert "scripts/laned/accept" in text
    for flag in ("--check-pr", "--expect-fail"):
        assert flag in text, f"workflow never runs accept {flag}"
    assert "laned:" in text, "workflow does not select WPs from laned:<WP> labels"
    assert "scripts/laned/status" in text, "push to main must recompute the status board"
