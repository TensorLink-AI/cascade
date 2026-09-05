"""The public website (cascade/website/index.html) renders the duel-only round
fields this branch adds to the trainer's heat status and the validator's
receipt/index rows — so the Vercel page, not just the ``cascade round`` CLI,
shows the whole seated field and the per-horizon verdict breakdown.

The page is static JS fed by S3 JSON, so these are contract checks: every
heat standing the trainer can publish has a pill, and every display field the
receipt index row carries is read somewhere. A silent field rename on either
side would otherwise leave the page rendering the pre-ladder view forever."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from cascade.shared.manifest import HEAT_STATUSES

INDEX = Path(__file__).resolve().parents[2] / "cascade" / "website" / "index.html"


@pytest.fixture(scope="module")
def page() -> str:
    return INDEX.read_text(encoding="utf-8")


def _heat_status_meta(page: str) -> str:
    m = re.search(r"function heatStatusMeta\(s\)\{(.*?)\n\}", page, re.S)
    assert m, "heatStatusMeta() missing from index.html"
    return m.group(1)


@pytest.mark.parametrize("status", HEAT_STATUSES)
def test_every_heat_status_has_a_pill(page: str, status: str):
    assert f's==="{status}"' in _heat_status_meta(page), (
        f"heat status {status!r} has no pill in heatStatusMeta() — it would render "
        "as a bare muted token")


@pytest.mark.parametrize("field", ["per_horizon", "cohort_geomeans", "cohort_per_horizon",
                                   "cohort_lcbs", "duel_only"])
def test_duel_fields_are_read(page: str, field: str):
    assert re.search(rf'["\.]{field}\b', page), f"index.html never reads {field!r}"


def test_cohort_panel_is_wired(page: str):
    assert 'id="cohort-panel"' in page
    assert "function renderCohort()" in page
    m = re.search(r"function renderAll\(\)\{(.*?)\}", page)
    assert m and "renderCohort()" in m.group(1), "renderCohort() is not called from renderAll()"


def test_duel_only_heat_panel_drops_the_score_columns(page: str):
    m = re.search(r"if\(duel\)\{(.*?)\n    return;\n  \}", page, re.S)
    assert m, "renderHeat() has no duel-only branch"
    body = m.group(1)
    assert "<th class=\"num\">Seat</th>" in body
    assert "CRPS" not in body and "MASE" not in body, "no screen ran — there are no heat scores to show"
