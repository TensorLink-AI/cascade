"""W12a: every §11.2 surface mentions lane D, links resolve, nothing leaks,
`scripts/check_docs.py` passes, and `llms.txt`'s lane list matches the status
fixture (plan §3.0 W12a Accept). Contract: README.md "Every surface"."""

from __future__ import annotations

import os
import re
import subprocess
import sys

import pytest

from . import helpers as h


def _id(p):
    return h.rel(p)


@pytest.mark.parametrize("path", h.MENTION_SURFACES, ids=_id)
def test_every_surface_mentions_lane_d(path):
    text = h.require(path)
    assert h.LANE_D_WORD.search(text), f"{h.rel(path)} never mentions lane D"


@pytest.mark.parametrize("path", h.SECTION_DOCS, ids=_id)
def test_every_text_surface_has_a_lane_d_section(path):
    h.require_sections(path)


@pytest.mark.parametrize("path", h.LINK_SURFACES, ids=_id)
def test_every_relative_link_resolves(path):
    h.require(path)
    assert not h.broken_links(path)


@pytest.mark.parametrize("path", h.SECTION_DOCS, ids=_id)
def test_no_secret_host_or_ip_in_the_lane_d_text(path):
    h.require(path)
    if path == h.LANE_D_MD:
        bad = h.leaks(path)
    else:
        bad = []
        for sec in h.sections(path):
            for line in sec.splitlines():
                for name, pat in h.wb.SECRET_PATTERNS:
                    if pat.search(line):
                        bad.append(f"{h.rel(path)}: {name}")
                for m in h.wb.IPV4.finditer(line):
                    ip = m.group(0)
                    if all(0 <= int(x) <= 255 for x in m.groups()) \
                            and not ip.startswith(h.wb._OK_IP_PREFIXES):
                        bad.append(f"{h.rel(path)}: IPv4 {ip}")
    assert not bad, bad


def test_check_docs_passes():
    env = {**os.environ, "PYTHONPATH": str(h.REPO)}
    proc = subprocess.run([sys.executable, "-P", str(h.REPO / "scripts" / "check_docs.py")],
                          cwd=h.REPO, env=env, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, f"scripts/check_docs.py failed:\n{proc.stdout}\n{proc.stderr}"


def test_llms_lane_list_matches_the_status_fixture():
    """The repo llms.txt lane table lists exactly the lanes `cascade status
    --json` reports (a/b/c on main; d once W12b lands)."""
    got = h.lane_table_letters(h.require(h.LLMS))
    assert got == h.status_lanes(), f"llms.txt lanes {got} != status lanes {h.status_lanes()}"


@pytest.mark.parametrize("path", (h.LLMS, h.WEB_LLMS), ids=_id)
def test_llms_links_lane_d_and_drops_the_stale_planned_order(path):
    text = h.require(path)
    assert "LANE_D.md" in text, f"{h.rel(path)} does not link docs/LANE_D.md"
    assert not re.search(r"A\s*(?:→|->)\s*C\s*(?:→|->)\s*B", text), \
        f"{h.rel(path)} still states the stale planned order A → C → B"


def test_readme_links_lane_d_and_changes_records_it():
    readme = h.require(h.README)
    assert "docs/LANE_D.md" in readme, "README.md does not link docs/LANE_D.md"
    changes = h.require(h.CHANGES)
    assert h.paragraph_with(changes, h.LANE_D_WORD, "LANE_D.md") or \
        h.paragraph_with(changes, h.LANE_D_WORD, "lane-d"), \
        "CHANGES.md has no release note naming lane D and its guide or CLI"


def test_no_surface_names_a_lane_d_key_the_code_does_not_have():
    """Stale guard: every `[lane_d] <key>` on any surface is a real
    `LaneDConfig` field."""
    keys = h.lane_d_keys()
    bad = []
    for path in h.MENTION_SURFACES:
        if not path.is_file():
            continue
        for m in re.finditer(r"\[lane_d\]\s+`?([a-z][a-z0-9_]*)", h.read(path)):
            if m.group(1) not in keys:
                bad.append(f"{h.rel(path)}: [lane_d] {m.group(1)}")
    assert not bad, bad


def test_no_surface_names_a_lane_d_command_the_cli_does_not_have():
    cmds = set(h.cli_subcommands())
    bad = []
    for path in h.MENTION_SURFACES:
        if not path.is_file():
            continue
        for m in re.finditer(r"cascade lane-d ([a-z][a-z-]*)", h.read(path)):
            if m.group(1) not in cmds:
                bad.append(f"{h.rel(path)}: cascade lane-d {m.group(1)}")
    assert not bad, bad
