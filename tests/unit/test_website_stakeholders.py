"""The stakeholder scoreboard (``cascade/website/stakeholders.html``).

Owner 2026-10-10: a short, visual page — the king's GIFT-Eval gap to a few
headline leaderboard models, five big-number tiles, one trend chart. The page
is a static asset with no build step, so these tests pin what it copies or
reads against the code that produces it:

* the trust anchor, index cap and per-leg budget (``chain.toml`` / hippius);
* the GIFT-Eval reference file (``scripts/refresh_leaderboard_reference.py``)
  — schema, provenance, and that the page reads the reading computed with the
  king's own aggregation;
* the receipt-summary, bench-report and ``status/v2.json`` fields it reads.

Pure text/JSON assertions; no browser, no network.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from cascade.shared.config import LaneBConfig, load_chain_config
from cascade.shared.hippius import RECEIPT_INDEX_MAX_KEEP, WEBSITE_STAKEHOLDERS_KEY
from cascade.shared.receipt import summarize_receipt
from cascade.shared.v2_status import build_v2_status

from .receipt_fixture import make_scored_receipt

REPO = Path(__file__).resolve().parents[2]
PAGE = REPO / "cascade" / "website" / WEBSITE_STAKEHOLDERS_KEY
REFERENCE = REPO / "cascade" / "website" / "data" / "gift-eval-reference.json"


@pytest.fixture(scope="module")
def html() -> str:
    return PAGE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def ref() -> dict:
    return json.loads(REFERENCE.read_text(encoding="utf-8"))


def _fn(html: str, name: str) -> str:
    m = re.search(rf"function {name}\([^)]*\)\{{(.*?)\n\}}", html, re.S)
    assert m, f"{name}() missing"
    return m.group(1)


def test_page_is_self_contained(html: str):
    for tag in re.findall(r"<(?:script|link|img|iframe)\b[^>]*>", html, re.I):
        src = re.search(r'(?:src|href)\s*=\s*"([^"]+)"', tag)
        if src:
            assert not src.group(1).startswith(("http://", "https://", "//")), f"external asset: {tag}"
    assert "@import" not in html


def test_trust_anchor_and_index_cap(html: str):
    cfg = load_chain_config(REPO / "chain.toml")
    assert re.search(r'var RECEIPT_ANCHOR = "([^"]+)"', html).group(1) == cfg.manifest.validator_hotkey
    assert float(re.search(r"var INDEX_MAX_KEEP = (\d+);", html).group(1)) == RECEIPT_INDEX_MAX_KEEP


def test_leg_budget_matches_the_training_contract(html: str):
    """Pre-v2 lineage tokens = legs × this budget; it must be chain.toml's."""
    m = re.search(r"var LEG_BUDGET = (\{[^}]*\});", html)
    b = json.loads(re.sub(r"(\w+):", r'"\1":', m.group(1)))
    cfg = load_chain_config(REPO / "chain.toml")
    assert b["train_h"] == pytest.approx(cfg.training.target_train_hours)
    assert b["ref_tps"] == cfg.training.ref_throughput_tokens_per_s


# ── the GIFT-Eval reference ──────────────────────────────────────────────────


def test_reference_file_is_real_and_versioned(ref: dict):
    assert ref["schema"] == 1 and ref["benchmark"] == "GIFT-Eval"
    assert ref["source"] == "https://huggingface.co/spaces/Salesforce/GIFT-Eval"
    assert re.fullmatch(r"[0-9a-f]{40}", ref["space_revision"]), "pin the space revision"
    assert ref["retrieved_at"] and ref["n_models"] >= len(ref["models"]) >= 3
    names = [m["model"] for m in ref["models"]]
    assert "Timestral-209M" in names, "the owner's headline model"
    for m in ref["models"]:
        for reading in ("leaderboard", "cascade"):
            for k in ("mase", "crps"):
                v = m[reading][k]
                assert 0 < v < 2, f"{m['model']} {reading} {k} = {v}"
        # GIFT-Eval has no zero-MASE config, so the sidecar's aggregation and
        # the space's agree — the comparison on the page is apples to apples
        assert m["cascade"]["n_scored"] == m["cascade"]["n_nonzero"] == 97
        assert m["cascade"]["mase"] == pytest.approx(m["leaderboard"]["mase"], rel=1e-4)
        assert m["testdata_leakage"] == "No"
    for k in ("mase", "crps"):
        f = ref["field"][k]
        assert f == sorted(f) and len(f) == ref["n_no_leak"]


def test_page_reads_the_kings_own_aggregation(html: str, ref: dict):
    assert 'fetchJSON("benchmarks/leaderboard-reference.json")' in html
    assert re.search(r'var REF_HEADLINE = "([^"]+)"', html).group(1) in [m["model"] for m in ref["models"]]
    assert "m&&m.cascade" in _fn(html, "refVal"), "the hero must read the cascade-aggregation reading"
    assert "_lb.field" in _fn(html, "placeInField")


def test_king_numbers_come_from_signed_bench_reports(html: str):
    from cascade.shared.bench_report import bench_report_key

    prefix, suffix = bench_report_key("XYZ").split("XYZ")
    assert re.search(rf'fetchJSON\("{re.escape(prefix)}"\+id\+"{re.escape(suffix)}",\{{bust:false\}}\)', html)
    body = _fn(html, "kingSeries")
    assert 'String(e.role||"")!=="king"' in body and "e.trained_pointer" in body
    g = _fn(html, "giftOf")
    assert "gifteval_mase" in g and "gifteval_crps" in g


def test_refresh_script_aggregates_with_the_sidecar():
    src = (REPO / "scripts" / "refresh_leaderboard_reference.py").read_text(encoding="utf-8")
    assert "from cascade_benchmark.aggregate import official_aggregate" in src
    assert "gifteval_seasonal_naive.json" in src
    assert '"Timestral-209M"' in src and "testdata_leakage" in src


# ── tiles ────────────────────────────────────────────────────────────────────


def test_page_reads_only_fields_the_summary_publishes(html: str):
    receipt, _k, _c = make_scored_receipt()
    summary = summarize_receipt(receipt)
    consumed = {"status", "round_id", "epoch_start_block", "cohort_k", "cohort_lcbs",
                "chal_hotkey", "king_hotkey", "n_participants", "king_gen_ref",
                "post_round_king_hotkey", "validator_hotkey"}
    assert not consumed - set(summary)
    for f in consumed:
        assert f"r.{f}" in html, f"the page no longer reads {f}"


@pytest.mark.parametrize("tile", ["Lineage tokens", "Competitions", "Miners", "Experts merged", "v2 · MoE"])
def test_the_five_tiles(html: str, tile: str):
    assert f'k:"{tile}"' in _fn(html, "renderTiles")


def test_v2_status_fields_the_page_reads_exist():
    cfg = SimpleNamespace(lane_b=LaneBConfig(from_block=10, domains=("energy",)),
                          subnet=SimpleNamespace(netuid=91),
                          v2=SimpleNamespace(from_block=10))
    doc = build_v2_status(cfg, block=20, state={"lineage_tokens": 5, "merged_inits": {}})
    assert doc["golive"]["state"] == "live" and doc["golive"]["from_block"] == 10
    assert doc["lane_b"]["gate"]["lineage_tokens"] == 5 and doc["lane_b"]["merged_total"] == 0
    assert {"total_params", "active_params"} <= set(doc["model"])


def test_v2_tiles_read_the_status_doc(html: str):
    assert 'fetchJSON("status/v2.json")' in html
    tiles = _fn(html, "renderTiles")
    for k in ("golive", "merged_total", "total_params", "active_params", '"scheduled"', '"live"'):
        assert k in tiles, f"renderTiles() never reads {k}"
    assert "lineage_tokens" in _fn(html, "lineageTokens")
    assert "go-live not announced" in tiles, "the 0 placeholder must render as not announced"


@pytest.mark.parametrize("gone", ["Is the competition working?", "How big are the models?",
                                  "Is the progress worth the compute?", "Does progress carry forward?",
                                  "TOKEN_ERAS", "TOTO2_PRETRAIN", "Checking these numbers"])
def test_cut_sections_stay_cut(html: str, gone: str):
    assert gone not in html


def test_links_to_and_from_the_mining_dashboard(html: str):
    assert 'href="index.html"' in html
    index = (REPO / "cascade" / "website" / "index.html").read_text(encoding="utf-8")
    assert f'href="{WEBSITE_STAKEHOLDERS_KEY}"' in index
