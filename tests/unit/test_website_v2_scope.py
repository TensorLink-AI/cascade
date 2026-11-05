"""Both public dashboards scope the receipts index to the v2 era (2026-11-04,
owner: "it still reports 16 legs / 149 competitions — it should reset to this
fresh round"): rounds settled before the v2 activation block are the previous
contract's history and must not feed the tiles, tables or miner counts."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PAGES = ("cascade/website/index.html", "cascade/website/stakeholders.html")
V2_FROM_BLOCK = 9399600          # status/v2.json golive.from_block on mainnet


@pytest.mark.parametrize("page", PAGES)
def test_dashboard_scopes_the_index_to_the_v2_era(page):
    html = (ROOT / page).read_text()
    assert f"var V2_FROM_BLOCK = {V2_FROM_BLOCK};" in html
    assert "function v2Scope(idx)" in html
    # the index is scoped the moment it loads, and re-scoped once status/v2.json
    # (the authoritative from_block) arrives
    assert "_rawIndex=collapseByRound(d); _index=v2Scope(_rawIndex);" in html
    assert re.search(r'fetchJSON\("status/v2\.json"\)\.then\(function\(d\)\{ _v2=d; if\(_rawIndex', html)
    # the filter keys on the settlement's era / epoch start block, never on a count
    assert "return b>=from;" in html
    assert "v2FromBlock()" in html


def test_v2_scope_filter_logic_matches_the_announced_block():
    """Pure re-implementation of the page filter over a representative index."""
    rows = [{"epoch_start_block": 9142200}, {"epoch_start_block": 9399600},
            {"era_start_block": 9430200, "epoch_start_block": 0}, {"epoch_start_block": 9399599}]

    def keep(r):
        b = r.get("era_start_block") or 0
        b = b if b > 0 else (r.get("epoch_start_block") or 0)
        return b >= V2_FROM_BLOCK

    assert [keep(r) for r in rows] == [False, True, True, False]
