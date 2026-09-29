"""Both dashboards link the sitting king's generator to its GitHub archive
(``champions/king``, re-synced from the on-chain commitment by
``.github/workflows/king-sync.yml``), keeping the exact registry digest as a
secondary link. Text checks only."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

WEBSITE = Path(__file__).resolve().parents[2] / "cascade" / "website"
KING_GITHUB = "https://github.com/TensorLink-AI/cascade/tree/main/champions/king"


@pytest.mark.parametrize("page", ["index.html", "stakeholders.html"])
def test_king_generator_links_to_the_github_archive(page: str):
    html = (WEBSITE / page).read_text(encoding="utf-8")
    assert re.search(rf'var KING_GITHUB\s*=\s*"{re.escape(KING_GITHUB)}"', html), f"{page}: KING_GITHUB missing"
    assert "KING_GITHUB" in html.split("var KING_GITHUB")[1], f"{page}: KING_GITHUB never used"
    assert "genUrl(" in html, f"{page}: the registry digest link is gone"


def test_champions_king_is_the_synced_archive():
    wf = (WEBSITE.parents[1] / ".github" / "workflows" / "king-sync.yml").read_text(encoding="utf-8")
    assert "champions/king" in wf
