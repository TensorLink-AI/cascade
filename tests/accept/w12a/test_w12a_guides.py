"""W12a: the lane D sections of the existing guides and agent surfaces (plan
§11.2 rows INTERFACE, MINER, AGENT_MINING, AGENTS/agent.md, VALIDATOR,
OPERATOR, ARCHITECTURE). Contract: README.md "The guides"."""

from __future__ import annotations

import re

import pytest

from . import helpers as h

# §11.2 VALIDATOR row (rev3): "the new report fields (`rows_sha256`,
# `coldkey`, judged set, `confirmation`, low-frequency result, both clocks)"
# as `lane-d-report/1` names them (checked against the schema below).
REPORT_FIELDS = {"rows_sha256": "", "judged": "", "skipped": "", "coldkey": "entries",
                 "confirmation": "verdicts", "lowfreq": "verdicts",
                 "min_target_ts": "windows", "min_first_seen_ts": "windows",
                 "mirror_lag_hours": "windows"}


def _need(path, text, *needles, negation=False, what=""):
    p = h.paragraph_with(text, *needles, negation=negation)
    assert p is not None, f"{h.rel(path)} lane D section: no paragraph {what or needles!r}"
    return p


def test_interface_has_the_checkpoint_interface():
    sec = h.require_sections(h.INTERFACE)
    for name in h.allowed_files():
        assert h.has_token(sec, name), f"INTERFACE lane D section does not name `{name}`"
    assert "BF16" in sec


def test_miner_lane_table_has_lane_d():
    text = h.require(h.MINER)
    assert "D" in h.lane_table_letters(text), "docs/MINER.md's lane table has no D row"


def test_miner_links_the_lane_d_guide():
    assert "LANE_D.md" in h.require_sections(h.MINER), \
        "docs/MINER.md's lane D section does not link LANE_D.md"


def test_miner_says_lane_d_needs_no_compute_key():
    sec = h.require_sections(h.MINER)
    _need(h.MINER, sec, re.compile(r"\bcompute[- ]keys?\b", re.I), negation=True,
          what="(no compute keys needed for lane D)")


def test_agent_mining_warns_local_scores_overstate_forward_gains():
    sec = h.require_sections(h.AGENT_MINING)
    _need(h.AGENT_MINING, sec, "overstate", "forward",
          re.compile(r"\b(?:revealed|local)\b", re.I),
          what="(local scores on the revealed pool overstate forward gains)")


@pytest.mark.parametrize("path", (h.AGENTS, h.AGENT), ids=h.rel)
def test_agent_surfaces_keep_human_authorization_for_lane_d(path):
    sec = h.require_sections(path)
    _need(path, sec, "human", re.compile(r"\bauthori[sz]", re.I),
          what="(the lane D role; spend and publish need the human's authorization)")


def test_validator_names_the_new_report_fields():
    fields = h.report_fields()
    for name, where in REPORT_FIELDS.items():
        assert name in fields[where], f"self-check: `{name}` is not a report field ({where!r})"
    sec = h.require_sections(h.VALIDATOR)
    missing = [n for n in REPORT_FIELDS if not h.has_token(sec, n)]
    assert not missing, f"VALIDATOR lane D section does not name the report fields {missing}"


def test_validator_says_what_satellites_cannot_verify():
    sec = h.require_sections(h.VALIDATOR)
    _need(h.VALIDATOR, sec, re.compile(r"\bsatellites?\b", re.I), "target", negation=True,
          what="(satellites cannot re-derive the targets)")


@pytest.mark.parametrize("needle", [
    "valeval-laned", "S8", "first-seen", "kill switch", "Blackwell", "vault", "retention",
    "`cascade-laned-intake`", "`lane_d_coldkey_unresolved`"])
def test_operator_lane_d_sections_cover_the_runbook(needle):
    sec = h.require_sections(h.OPERATOR)
    if needle.startswith("`"):
        assert h.has_token(sec, needle.strip("`")), f"OPERATOR lane D sections lack {needle}"
    else:
        assert needle.lower() in sec.lower(), f"OPERATOR lane D sections lack {needle!r}"


def test_operator_names_the_launch_eval_host():
    sec = h.require_sections(h.OPERATOR)
    _need(h.OPERATOR, sec, "eval host", re.compile(r"\bpinned\b", re.I), "torch",
          what="(the launch eval host runs the pinned torch)")


@pytest.mark.parametrize("token", ["cascade-laned-intake", "lane_d_state.json"])
def test_architecture_lane_d_flow(token):
    sec = h.require_sections(h.ARCHITECTURE)
    assert h.has_token(sec, token), f"ARCHITECTURE lane D section does not name `{token}`"
    assert re.search(r"\bsigned\b.*\breport\b|\breport\b.*\bsigned\b", sec, re.I | re.S)
