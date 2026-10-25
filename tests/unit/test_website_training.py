"""The Training tab of the public dashboard (``cascade/website/index.html``).

The tab charts, round by round, the public-benchmark numbers (GIFT-Eval /
BOOM / TIME, CRPS + MASE) of the king's checkpoint — the lineage's training
trajectory under warm-start — and, on request, every checkpoint the trainer
benched (each challenger's). It reads the trainer-signed per-round bench
reports.

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
from cascade.shared.manifest import FEV_SCORE_KEYS, VALID_ROLES, BenchScores

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
    # FEV is charted separately; the promotion geomean still uses six numbers.
    fields = [f.name for f in dataclasses.fields(BenchScores) if f.name not in FEV_SCORE_KEYS]
    assert _js_string_list(page, "SIX_KEYS") == fields, (
        "SIX_KEYS drifted from BenchScores — the charts and the table would read "
        "undefined and render --")
    assert _js_string_list(page, "FEV_KEYS") == list(FEV_SCORE_KEYS)


def test_fetches_the_objects_the_trainer_publishes(page: str):
    # per-round signed report: benchmarks/round-<id>.json
    prefix, suffix = bench_report_key("XYZ").split("XYZ")
    assert re.search(rf'fetchJSON\("{re.escape(prefix)}"\s*\+\s*id\s*\+\s*"{re.escape(suffix)}"', page), (
        "the tab does not fetch bench_report_key() objects")
    # immutable per-round reports are fetched cache-friendly, never busted
    assert re.search(r'fetchJSON\("benchmarks/round-".*?\{bust:false\}', page)


@pytest.mark.parametrize("field", ["role", "miner_hotkey", "miner_uid", "size", "trained_pointer"])
def test_bench_entry_fields_are_read(page: str, field: str):
    body = re.search(r"function trainRows\(\)\{(.*?)\n\}", page, re.S)
    assert body and re.search(rf"\.{field}\b", body.group(1)), f"trainRows() never reads BenchEntry.{field}"


def test_every_report_role_renders_a_pill(page: str):
    tbl = re.search(r"function renderTrainTable\(rows\)\{(.*?)\n\}", page, re.S)
    assert tbl, "renderTrainTable() missing"
    assert VALID_ROLES == ("king", "challenger"), "a new role needs its own pill in renderTrainTable()"
    assert 'role-pill king' in tbl.group(1) and 'role-pill chal' in tbl.group(1)


def test_benchmark_and_axes_selectors(page: str):
    assert 'var _trainScope="king"' in page
    for suite in ("gifteval", "boom", "time", "fev"):
        assert f'<option value="{suite}"' in page
    for axes in ("size-mase", "size-crps", "mase-crps"):
        assert f'<option value="{axes}"' in page
    assert 'id="pareto-history"' in page
    assert 'data-tscope="all"' in page


def test_one_pareto_chart_with_accessible_point_details(page: str):
    assert page.count('<svg id="train-svg"') == 1
    assert 'function paretoFrontier(points)' in page
    assert 'function paretoChartSVG(points,W,H)' in page
    assert 'class="king-trajectory"' in page
    assert 'class="pareto-frontier"' in page
    assert 'tabindex="0" role="button"' in page
    assert 'node.addEventListener("keydown"' in page
    assert 'id="pareto-detail"' in page
    assert 'id="pareto-rows"' in page


def test_tab_polls_while_open(page: str):
    m = re.search(r"function poll\(\)\{(.*?)\n\}", page, re.S)
    assert m and 'if(_page==="train") loadBench()' in m.group(1), (
        "a report landing while the tab is open would never appear until reload")


def _train_budget(page: str) -> dict:
    m = re.search(r"var TRAIN_BUDGET\s*=\s*(\{[^}]*\})\s*;", page)
    assert m, "TRAIN_BUDGET not found in the page"
    return json.loads(re.sub(r"(\w+):", r'"\1":', m.group(1)))


def test_lineage_budget_matches_chain_toml(page: str):
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


def test_lineage_depth_is_one_leg_per_generation(page: str):
    """Steps and tokens are the LINEAGE of the king checkpoint: one leg per
    warm-start generation behind it plus its own (every round of a
    generation trains from that generation's promoted set, so rounds do not
    accumulate — promotions do). The walk follows each training summary's
    warm_start_ckpt to the promoted member's own leg; an unresolved leg is
    priced at the contracted budget."""
    body = re.search(r"function trainRows\(\)\{(.*?)\n\}", page, re.S)
    assert body
    b = body.group(1)
    assert "genOfBlock(" in b and "depth=gen+1" in b, "depth is not one leg per generation"
    assert "warm_start_ckpt" in b, "the lineage walk never follows the round's init"
    assert 'l.role==="king"' in b, "a challenger's leg would price the king's lineage"
    assert "stepsPerLeg()" in b and "tokensPerLeg()" in b, "no contracted fallback for an unresolved leg"
    for field in ("steps", "tokens_seen", "channel_tokens", "max_channels_seen", "deadline_hit", "measured"):
        assert re.search(rf"\.{field}\b", page[page.index("function legPrice"):page.index("function legIndex")]), (
            f"legPrice() never reads the training summary's {field!r}")
    gens = re.search(r"function loadGenerations\(\)\{(.*?)\n\}", page, re.S)
    assert gens and 'fetchJSON("promotions/index.json")' in gens.group(1)
    assert re.search(r"\.fired_block\b", gens.group(1)) and re.search(r"\.effective_era\b", gens.group(1)), (
        "a generation's activation block is not read off its promotion record")
    assert "function kingTrajectory(rows)" in page
    assert "if(entry.crowned) add(entry,row)" in page


def test_training_summary_is_fetched_immutably(page: str):
    from cascade.shared.training_summary import training_summary_key

    prefix, suffix = training_summary_key("XYZ").split("XYZ")
    assert re.search(rf'fetchJSON\("{re.escape(prefix)}"\s*\+\s*id\s*\+\s*"{re.escape(suffix)}",\{{bust:false\}}', page), (
        "the tab does not fetch training_summary_key() objects cache-friendly")


@pytest.mark.parametrize("tile", ["Generations", "Steps trained", "Tokens trained", "Series-points", "Lineage legs"])
def test_top_box_has_a_tile_per_trained_quantity(page: str, tile: str):
    body = re.search(r"function renderTrainStats\(rows, info\)\{(.*?)\n\}", page, re.S)
    assert body and f'tile("{tile}"' in body.group(1), f"no {tile!r} tile in the top box"


def test_top_box_is_a_grid_of_bordered_tiles(page: str):
    assert 'class="train-tiles" id="train-stats"' in page
    assert re.search(r"\.train-tiles \.tile \{[^}]*border:", page), "tiles carry no border"
    m = re.search(r"function tile\(k,v,sub,o\)\{(.*?)\n\}", page, re.S)
    assert m and 'class="tile' in m.group(1) and 'class="k"' in m.group(1) and 'class="v"' in m.group(1)


def test_reference_catalog_and_same_pipeline_override(page: str):
    assert 'fetchJSON("benchmarks/reference-timestral-209m.json")' in page
    assert 'fetchJSON("benchmarks/model-frontier.json")' in page
    assert 'fetch("data/benchmark-models.json"' in page
    assert 'parameters:metadata?metadata.parameters:null' in page
    assert 'source:"same-pipeline reference"' in page


def test_old_toto2_comparison_is_removed(page: str):
    assert "TOTO2_PRETRAIN" not in page and 'tile("Tokens vs Toto2"' not in page
    assert "reference-toto2-" not in page


@pytest.mark.parametrize("gone", ["Width C", "Benched", "Batch × context", "King checkpoint", "Geomean", "Best ever"])
def test_removed_tiles_stay_removed(page: str, gone: str):
    body = re.search(r"function renderTrainStats\(rows, info\)\{(.*?)\n\}", page, re.S)
    assert body and f'tile("{gone}"' not in body.group(1), f"{gone!r} tile is back"


def test_benchmark_cells_are_tagged_when_the_king_beats_timestral(page: str):
    body = re.search(r"function renderTrainStats\(rows, info\)\{(.*?)\n\}", page, re.S)
    assert body and 'return (rv!=null&&s[key]<rv)?"beats "+ref.label:""' in body.group(1)
    assert 'tag=beat(key)' in body.group(1)
    assert '["mase","crps"]' in body.group(1)
