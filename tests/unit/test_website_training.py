"""The Training tab of the public dashboard (``cascade/website/index.html``).

The tab charts, round by round, the public-benchmark numbers (GIFT-Eval /
BOOM / TIME, CRPS + MASE) of the king's checkpoint — the lineage's training
trajectory under warm-start — and, on request, every checkpoint the trainer
benched (each challenger's, plus the from-scratch shadow control). It reads
the trainer-signed per-round bench reports and the scratch-shadow roll-up.

The page is static JS fed by S3 JSON, so these are contract checks: the six
score keys it reads must be exactly :class:`BenchScores`' fields, the object
keys it fetches must be the ones the trainer publishes, every role a report
can carry must render, and the tab must be wired into the page switcher.
Pure text assertions; no browser, no network.
"""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path

import pytest

from cascade.shared.bench_report import bench_report_key
from cascade.shared.config import load_chain_config
from cascade.shared.manifest import VALID_ROLES, BenchScores
from cascade.shared.scratch_report import SCRATCH_INDEX_KEY

REPO = Path(__file__).resolve().parents[2]
INDEX = REPO / "cascade" / "website" / "index.html"


@pytest.fixture(scope="module")
def page() -> str:
    return INDEX.read_text(encoding="utf-8")


def _js_string_list(page: str, name: str) -> list[str]:
    m = re.search(rf"var {re.escape(name)}\s*=\s*\[(.*?)\]\s*;", page)
    assert m, f"{name} not found in the page"
    return re.findall(r'"([^"]+)"', m.group(1))


def test_tab_is_wired_into_the_page_switcher(page: str):
    assert 'data-ptab="train"' in page, "no Training tab button"
    assert 'id="page-train"' in page
    m = re.search(r"function setPage\(p\)\{(.*?)\n\}", page, re.S)
    assert m, "setPage() missing"
    body = m.group(1)
    assert '"page-train"' in body and 'p!=="train"' in body, "setPage() never shows/hides the tab"
    assert "loadBench()" in body, "opening the tab never loads the bench reports"
    # the hidden-page rule must cover the new page or it stays visible under
    # the others (display:flex beats the hidden attribute)
    assert re.search(r"#page-train\[hidden\]", page), "#page-train[hidden] has no display:none rule"


def test_six_score_keys_are_exactly_bench_scores(page: str):
    fields = [f.name for f in dataclasses.fields(BenchScores)]
    assert _js_string_list(page, "SIX_KEYS") == fields, (
        "SIX_KEYS drifted from BenchScores — the charts and the table would read "
        "undefined and render --")


def test_fetches_the_objects_the_trainer_publishes(page: str):
    # per-round signed report: benchmarks/round-<id>.json
    prefix, suffix = bench_report_key("XYZ").split("XYZ")
    assert re.search(rf'fetchJSON\("{re.escape(prefix)}"\s*\+\s*id\s*\+\s*"{re.escape(suffix)}"', page), (
        "the tab does not fetch bench_report_key() objects")
    # scratch-shadow roll-up (DEC-CA-0014), read for the control series
    assert f'fetchJSON("{SCRATCH_INDEX_KEY}")' in page
    # immutable per-round reports are fetched cache-friendly, never busted
    assert re.search(r'fetchJSON\("benchmarks/round-".*?\{bust:false\}', page)


@pytest.mark.parametrize("field", ["scratch_scores", "warm_start_ckpt", "generation", "round_id"])
def test_scratch_index_rows_are_read(page: str, field: str):
    body = re.search(r"function trainRows\(\)\{(.*?)\n\}", page, re.S)
    assert body, "trainRows() missing"
    assert re.search(rf"\.{field}\b", body.group(1)), f"trainRows() never reads scratch row {field!r}"


@pytest.mark.parametrize("field", ["role", "miner_hotkey", "miner_uid", "size", "trained_pointer"])
def test_bench_entry_fields_are_read(page: str, field: str):
    body = re.search(r"function trainRows\(\)\{(.*?)\n\}", page, re.S)
    assert body and re.search(rf"\.{field}\b", body.group(1)), f"trainRows() never reads BenchEntry.{field}"


def test_every_report_role_renders_a_pill(page: str):
    tbl = re.search(r"function renderTrainTable\(rows\)\{(.*?)\n\}", page, re.S)
    assert tbl, "renderTrainTable() missing"
    assert VALID_ROLES == ("king", "challenger"), "a new role needs its own pill in renderTrainTable()"
    assert 'role-pill king' in tbl.group(1) and 'role-pill chal' in tbl.group(1)
    assert 'role-pill scratch' in tbl.group(1), "the scratch control row has no pill"


def test_king_lineage_is_the_default_and_everything_trained_is_offered(page: str):
    assert re.search(r'data-tscope="king"[^>]*>', page)
    assert re.search(r'data-tscope="all"[^>]*>', page)
    assert 'var _trainMetric="crps", _trainScope="king"' in page
    for metric in ("crps", "mase", "geomean"):
        assert f'data-tmetric="{metric}"' in page, f"no {metric} metric button"


def test_charts_are_per_suite_small_multiples_with_a_hover_layer(page: str):
    suites = _js_string_list(page, "SUITES")
    assert suites[::2] == ["gifteval", "boom", "time"], suites
    fn = re.search(r"function trainChartSVG\(rows, S, W, H\)\{(.*?)\n\}", page, re.S)
    assert fn, "trainChartSVG() missing"
    body = fn.group(1)
    assert 'class="col"' in body and 'class="xh"' in body, "no per-round hit targets / crosshair"
    assert "r.dethroned" in body, "dethrones are not marked on the round axis"
    assert "function wireTrainChart" in page and "train-tip" in page


def test_tab_polls_while_open(page: str):
    m = re.search(r"function poll\(\)\{(.*?)\n\}", page, re.S)
    assert m and 'if(_page==="train") loadBench()' in m.group(1), (
        "a report landing while the tab is open would never appear until reload")


def _train_budget(page: str) -> dict:
    m = re.search(r"var TRAIN_BUDGET\s*=\s*(\{[^}]*\})\s*;", page)
    assert m, "TRAIN_BUDGET not found in the page"
    return json.loads(re.sub(r"(\w+):", r'"\1":', m.group(1)))


def test_x_axis_budget_matches_chain_toml(page: str):
    """The x axis is cumulative training (round × steps per leg). The leg
    budget is transcribed from ``[training]`` in chain.toml — if the contract
    moves, the axis must move with it or every tick is silently wrong."""
    b = _train_budget(page)
    cfg = load_chain_config(REPO / "chain.toml")
    assert b["ref_tps"] == cfg.training.ref_throughput_tokens_per_s
    assert b["train_h"] == pytest.approx(cfg.training.target_train_hours)
    assert b["batch"] == cfg.training.batch_size
    assert b["ctx"] == cfg.training.context_length
    # sanity on the derived denomination the page states: ~152k steps / ~40B
    # series-points per leg at the live contract
    steps = b["train_h"] * 3600 * b["ref_tps"] / (b["batch"] * b["ctx"])
    assert 100_000 < steps < 200_000


def test_lineage_totals_sum_the_king_legs_from_the_training_summary(page: str):
    """Steps and tokens accumulate over the KING legs only (one per checkpoint),
    each priced from the trainer's ``training/round-<id>.json`` row when it
    is measured and at the contracted budget otherwise; the tokens are the
    leg's channel tokens (steps × batch × C × context)."""
    body = re.search(r"function trainRows\(\)\{(.*?)\n\}", page, re.S)
    assert body
    b = body.group(1)
    assert 'l.role==="king"' in b, "legs other than the king's would be counted"
    assert "seen[l.pointer]" in b, "a king leg benched at several settlements would be counted twice"
    assert "cum.steps+=l.steps" in b and "cum.tokens+=l.tokens" in b
    for field in ("steps", "tokens_seen", "channel_tokens", "max_channels_seen", "deadline_hit", "measured"):
        assert re.search(rf"\.{field}\b", b), f"trainRows() never reads the training summary's {field!r}"
    assert "stepsPerLeg()" in b and "tokensPerLeg()" in b, "no contracted fallback for an unmeasured leg"
    chart = re.search(r"function trainChartSVG\(rows, S, W, H\)\{(.*?)\n\}", page, re.S)
    assert chart and "rows[i].steps" in chart.group(1), "the chart does not place rounds by cumulative steps"
    tip = re.search(r"function trainTipHTML\(row, S, i, unit\)\{(.*?)\n\}", page, re.S)
    assert tip and "fmtTokens(row.tokens)" in tip.group(1), "the tooltip does not show the token count"


def test_training_summary_is_fetched_immutably(page: str):
    from cascade.shared.training_summary import training_summary_key

    prefix, suffix = training_summary_key("XYZ").split("XYZ")
    assert re.search(rf'fetchJSON\("{re.escape(prefix)}"\s*\+\s*id\s*\+\s*"{re.escape(suffix)}",\{{bust:false\}}', page), (
        "the tab does not fetch training_summary_key() objects cache-friendly")


@pytest.mark.parametrize("tile", ["Steps trained", "Tokens trained", "Width C", "Batch × context", "King legs"])
def test_top_box_has_a_tile_per_trained_quantity(page: str, tile: str):
    body = re.search(r"function renderTrainStats\(rows, info\)\{(.*?)\n\}", page, re.S)
    assert body and f'tile("{tile}"' in body.group(1), f"no {tile!r} tile in the top box"


def test_top_box_is_a_grid_of_bordered_tiles(page: str):
    assert 'class="train-tiles" id="train-stats"' in page
    assert re.search(r"\.train-tiles \.tile \{[^}]*border:", page), "tiles carry no border"
    m = re.search(r"function tile\(k,v,sub,o\)\{(.*?)\n\}", page, re.S)
    assert m and 'class="tile' in m.group(1) and 'class="k"' in m.group(1) and 'class="v"' in m.group(1)
