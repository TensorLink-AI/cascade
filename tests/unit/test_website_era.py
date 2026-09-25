"""The public website (cascade/website/index.html) under rolling eras: one
round is one era of ``era_blocks`` with a settlement receipt at every
``epoch_blocks`` boundary inside it, and the live field is the trainer's
funded roster (``funded/latest.json``), not the pre-rollover heat doc.

Contract checks on the static page (pure text, no network): the era grid is
derived from the status feed, settlement rows group by era, every roster
standing has a pill, the roster is fetched, and the stale pre-rollover docs are
guarded against — a silent regression on any of these leaves the page counting
3 h rounds or showing a field from before the current era."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

INDEX = Path(__file__).resolve().parents[2] / "cascade" / "website" / "index.html"
ROSTER_STATUSES = ("training", "renting", "waiting", "seated", "failed")


@pytest.fixture(scope="module")
def page() -> str:
    return INDEX.read_text(encoding="utf-8")


def _fn(page: str, name: str) -> str:
    m = re.search(rf"function {name}\([^)]*\)\{{(.*?)\n\}}", page, re.S)
    assert m, f"{name}() missing from index.html"
    return m.group(1)


def test_chain_clock_carries_the_era_grid(page: str):
    body = _fn(page, "chainNow")
    for key in ("eraBlocks", "eraStart", "eraEnd", "settlements", "settleNo", "rolling"):
        assert f"{key}:" in body, f"chainNow() does not expose {key}"
    assert "eraBlocksOf(c)" in body


def test_era_length_comes_from_the_status_feed_with_the_grid_fallback(page: str):
    body = _fn(page, "eraBlocksOf")
    assert "era_blocks" in body, "the validator's era_blocks field is never read"
    assert "ERA_BLOCKS_FALLBACK" in body
    assert re.search(r"var ERA_BLOCKS_FALLBACK=3600;", page)


@pytest.mark.parametrize("field", ["era_start_block", "activation_block", "era_blocks"])
def test_era_fields_are_read(page: str, field: str):
    assert re.search(rf'["\.]{field}\b', page), f"index.html never reads {field!r}"


def test_settlement_rows_group_by_the_receipts_era(page: str):
    body = _fn(page, "eraStartOfRow")
    assert "r.era_start_block" in body and "_latest.era_start_block" in body
    assert "Math.floor(b/era)*era" in body


def test_roster_is_fetched_and_rendered_in_rolling_mode(page: str):
    assert re.search(r'fetchJSON\(\s*"funded/latest\.json"', page)
    assert "function renderRoster(" in page
    heat = _fn(page, "renderHeat")
    assert "renderRoster(" in heat, "renderHeat() never hands off to the roster"
    rows = _fn(page, "rosterRows")
    for key in ("in_flight", "rents", "waiting", "seated", "terminal"):
        assert f"d.{key}" in rows, f"roster list {key!r} is never read"


@pytest.mark.parametrize("status", ROSTER_STATUSES)
def test_every_roster_standing_has_a_pill(page: str, status: str):
    m = re.search(r"function heatStatusMeta\(s\)\{(.*?)\n\}", page, re.S)
    assert m and f's==="{status}"' in m.group(1)


def test_stale_pre_rollover_heat_doc_is_never_shown_as_live(page: str):
    body = _fn(page, "heatSource")
    assert "live.esb<cn.eraStart" in body


def test_live_strip_renders_one_pill_per_settlement(page: str):
    assert "function renderSettlementStrip(" in page
    stage = _fn(page, "renderStage")
    assert "renderSettlementStrip(" in stage
    strip = _fn(page, "renderSettlementStrip")
    assert "cn.settlements" in strip and "blockClock(" in strip


def test_verdict_panel_lists_the_eras_settlements(page: str):
    assert 'id="era-settlements"' in page
    verdict = _fn(page, "renderVerdict")
    assert "renderEraSettlements(" in verdict
    body = _fn(page, "renderEraSettlements")
    assert "eraRows(" in body and "cohort_lcbs" in body


def test_countdown_counts_the_era_and_the_next_settlement(page: str):
    body = _fn(page, "renderCountdown")
    assert "era ends" in body and "nextSettlementSecs()" in body


def test_warm_start_reads_the_settlement_manifest_in_rolling_mode(page: str):
    body = _fn(page, "renderWarmEra")
    assert "warm_start_ckpt" in body and "m.era" in body and "effective_era" in body
    warm = _fn(page, "renderWarm")
    assert "renderWarmEra(" in warm


def test_round_labels_carry_the_settlement_number(page: str):
    body = _fn(page, "roundLabel")
    assert "settlementNoOf(" in body and "S\"" in body
    rounds = _fn(page, "renderRounds")
    assert "roundLabel(r)" in rounds
