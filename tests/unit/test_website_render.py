"""Fixture-driven render of both dashboards (``index.html`` and
``stakeholders.html``) in node with a minimal DOM shim
(``website_render_harness.js``) — no browser, no network.

Each scenario feeds the pages bucket JSON built by the REAL producers
(``summarize_receipt`` + ``v2_index_fields`` for the index rows,
``build_v2_status`` for ``status/v2.json``, the checked-in GIFT-Eval reference)
and then calls the render entry points directly, so a render bug throws here
instead of being swallowed by a fetch ``.catch()``:

* mainnet today — v2 go-live block 0 (not announced): the toto2 competition
  renders as now, v2 reads "not announced";
* mainnet scheduled — a typed go-live block: a countdown;
* testnet live — lane B with a winner, a failed guard, routing history and
  the king's MoE health;
* nothing published — every panel renders its pending state.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cascade.shared.config import LaneBConfig
from cascade.shared.receipt import summarize_receipt
from cascade.shared.v2_status import build_v2_status, king_routing_health, v2_index_fields

from .receipt_fixture import make_scored_receipt

REPO = Path(__file__).resolve().parents[2]
WEB = REPO / "cascade" / "website"
HARNESS = Path(__file__).with_name("website_render_harness.js")
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node not installed")

PTR = "metro-v1:trained:hippius:cascade/ckpt-r9-king@sha256:" + "a" * 64
DOMAINS = ("energy", "retail")
NOW = datetime.now(UTC).replace(microsecond=0).isoformat()
SIX = dict(gifteval_crps=0.47, gifteval_mase=0.69, boom_crps=0.4, boom_mase=0.6,
           time_crps=0.5, time_mase=0.7)


def _render(page: str, fixtures: dict, host: str, calls: list[str], tmp: Path) -> dict:
    fx, cl = tmp / "fixtures.json", tmp / "calls.json"
    fx.write_text(json.dumps(fixtures))
    cl.write_text(json.dumps(calls))
    out = subprocess.run([NODE, str(HARNESS), str(WEB / page), str(fx), host, str(cl)],
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    res = json.loads(out.stdout)
    assert res["errors"] == [], "\n".join(res["errors"])
    return res["els"]


def _html(els: dict, el: str) -> str:
    return els.get(el, {}).get("html", "") + els.get(el, {}).get("text", "")


def _cfg(from_block: int):
    lb = LaneBConfig(from_block=from_block + 1 if from_block else 0,
                    domains=DOMAINS if from_block else (), win_ttl_eras=28)
    return SimpleNamespace(lane_b=lb, subnet=SimpleNamespace(netuid=91),
                           training=SimpleNamespace(v2_from_block=from_block))


def _bucket(*, v2_from: int, block: int, lane_b: bool) -> dict:
    receipt, _k, _c = make_scored_receipt()
    body = json.loads(receipt.canonical_body())
    row = {**summarize_receipt(receipt), "published_at": NOW,
           "receipt_key": f"receipts/round-{receipt.round_id}.json"}
    if lane_b:
        n = len(body["weights"])
        weights = [0.0] * max(n, 20)
        weights[0], weights[1] = 0.25, 0.25          # lane A court (uids 0, 1 below)
        weights[12], weights[19] = 0.25, 0.1         # energy paid in full, retail at 0.4
        weights[2] = 0.15                            # the unpaid lane-B share burns (uid 2)
        body.update(weights=weights, reward_uids=[0, 1])
        body["lane_b"] = [
            {"hotkey": "hk-energy", "domain": "energy", "stage": "verdict", "passed": True,
             "reason": "merge", "target_lcb": 0.031, "guard_lcbs": {"retail": -0.002},
             "public_lcbs": {"gift_crps": 0.001}},
            {"hotkey": "hk-loser", "domain": "retail", "stage": "verdict", "passed": False,
             "reason": "regression on domain energy: LCB -0.0200 < -0.01", "target_lcb": 0.02,
             "guard_lcbs": {"energy": -0.02}, "public_lcbs": {}},
        ]
        body["lane_b_pay"] = {"block": block, "epoch_blocks": 900, "wins": [
            {"domain": "energy", "hotkey": "hk-energy", "uid": 12, "merge_block": block - 900,
             "age": 0, "multiplier": 1.0},
            {"domain": "retail", "hotkey": "hk-old", "uid": 19, "merge_block": block - 30000,
             "age": 7, "multiplier": 0.4}]}
        row.update(v2_index_fields(SimpleNamespace(
            lane_b=tuple(body["lane_b"]), lane_b_pay=body["lane_b_pay"],
            weights=tuple(weights), reward_uids=(0, 1))))
    index = {"subnet": {"netuid": 91}, "updated_at": NOW, "rounds": [row]}
    state = None
    health = {}
    if lane_b:
        state = {
            "queue": json.dumps({"pending": [], "open": [{"sub": {
                "hotkey": "hk-open", "uid": 21, "gen_ref": "ns/g@sha256:" + "c" * 64,
                "domain": "retail", "commit_block": block - 50},
                "selection": {"1": [3, 4], "2": [9]}, "base_king_ref": PTR, "base_era": 4,
                "exclude": {}}]}),
            "lineage_tokens": 260_000_000_000, "gate_opened_block": block - 9000,
            "routing": {"era": 5, "block": block, "domains": {
                d: {"overlap": [0.9] * 7, "jaccard": [0.8] * 7, "tokens": 512} for d in DOMAINS},
                "open": True, "reason": "open"},
            "routing_history": [{"era": e, "block": e * 3600, "lineage_tokens": e * 40_000_000_000,
                                 "domains": {d: {"overlap": [0.5 + e / 20] * 7,
                                                 "jaccard": [0.4 + e / 20] * 7} for d in DOMAINS}}
                                for e in range(1, 7)],
            "merged_inits": {"5": {"pointer": PTR, "merges": [{"rebased_pointer":
                "metro-v1:trained:hippius:cascade/ckpt-r70-expert-energy-u12@sha256:" + "f" * 64,
                "round_id": "70", "selection": {"1": [1]}}]}},
            "outcomes": [{"round_id": "71", "hotkey": "hk-x", "uid": 3, "domain": "energy",
                          "outcome": "routing_stale", "detail": {}}],
        }
        load = np.full((7, 256), 10.0)
        load[3, :20] = 0
        health = king_routing_health(load, layers=[1, 2, 3, 5, 6, 7, 8])
    cfg = _cfg(v2_from)
    if not lane_b:
        cfg.lane_b = replace(cfg.lane_b, from_block=0)
    v2 = build_v2_status(cfg, block=block, state=state, health=health, era_index=5,
                         king_pointer=PTR, health_domains=2 if lane_b else 0, as_of=NOW)
    bench = {"round_id": receipt.round_id, "created_block": 1, "entries": [
        {"role": "king", "size": "toto2-22m", "miner_hotkey": "hk-king", "miner_uid": 0,
         "trained_pointer": PTR, "scores": dict(SIX)}]}
    return {
        "receipts/index.json": index,
        "receipts/5FcCsoB2R5M55RdrNrvPQv7xmLJRFzs7R7C73UUSxgUCyfsw/latest.json": body,
        "status/chain.json": {"current_block": block, "epoch_blocks": 900, "era_blocks": 3600,
                              "block_time_s": 12, "as_of": NOW, "submissions": []},
        "status/v2.json": v2,
        f"benchmarks/round-{receipt.round_id}.json": bench,
        "benchmarks/model-frontier.json":
            json.loads((WEB / "data" / "benchmark-models.json").read_text()),
        "benchmarks/leaderboard-reference.json":
            json.loads((WEB / "data" / "gift-eval-reference.json").read_text()),
        "promotions/index.json": {"latest_generation": 2},
    }


INDEX_CALLS = ["renderAll()", "setPage('laneb')", "renderLaneB()", "setPage('train')",
               "renderTrain()", "setPage('main')", "renderV2Strip()"]
STAKE_CALLS = ["renderAll()", "_metric='crps'; renderHero(); renderTrend();", "renderTiles()"]


@pytest.mark.parametrize("era_start,current", [(46800, True), (43200, False), (50400, False)])
def test_header_links_current_warm_init_not_trained_king_or_future_promotion(tmp_path, era_start, current):
    fx = _bucket(v2_from=1000, block=50000, lane_b=True)
    init = "metro-v1:trained:hippius:cascade/accepted-init@sha256:" + "b" * 64
    fx["status/v2.json"]["warm_start"] = {
        "pointer": init, "mode": "checkpoint", "era_start_block": era_start,
        "generation": 3, "member_index": 0}
    calls = ["renderAll()", "renderWarmStartLink()"]
    if current:
        calls.append("if(document.getElementById('warm-start-link').href!=='https://hub.hippius.com/models/cascade/accepted-init/main') throw Error('wrong checkpoint link')")
    els = _render("index.html", fx, "localhost", calls, tmp_path)
    assert els["warm-start-link"]["hidden"] != current
    if current:
        assert "accepted-init" in _html(els, "warm-start-link")
    else:
        assert "awaiting this era" in _html(els, "warm-start-state")


@pytest.mark.parametrize("fev", [None, {"fev_crps": .31, "fev_mase": .62},
                                 {"fev_crps": "bad", "fev_mase": None}])
def test_fev_report_display_and_timestral_reference(tmp_path, fev):
    fx = _bucket(v2_from=1000, block=50000, lane_b=True)
    fx["receipts/index.json"]["rounds"][0].update(
        post_round_king_hotkey="hk-king", dethroned=False)
    bench = next(v for k, v in fx.items() if k.startswith("benchmarks/round-"))
    if fev:
        bench["entries"][0]["scores"].update(fev)
    fx["benchmarks/reference-timestral-209m.json"] = {
        "preset": "timestral-209m", "source": "mistralai/Timestral-209M@" + "a" * 40,
        "scores": {**{k: .9 for k in SIX}, "fev_crps": .8, "fev_mase": .9}}
    valid = fev is not None and isinstance(fev["fev_crps"], float)
    calls = [f"_bench[{json.dumps(bench['round_id'])}]={json.dumps(bench)}",
             "setPage('train')", "renderTrain()",
             "_paretoBenchmark='fev'; _paretoAxes='mase-crps'; renderTrain()",
             f"if(modelPoints(trainRows().rows).kings.filter(p=>paretoCoordinates(p)).length!=={int(valid)}) throw Error('FEV plotted without a numeric score')",
             "var scores=readSix(_bench[Object.keys(_bench)[0]].entries[0]);"
             f"if(Math.abs(sixGeomean(scores)-sixGeomean({json.dumps(SIX)}))>1e-12) throw Error('FEV changed the promotion score')"]
    els = _render("index.html", fx, "localhost", calls, tmp_path)
    stats = _html(els, "train-stats")
    fev_tile = stats.split('data-metric="fev_crps">', 1)[1].split('</td>', 1)[0]
    assert "0.3100" in fev_tile if valid else "--" in fev_tile
    assert ("beats Timestral-209M" in fev_tile) == valid
    assert 'data-metric="fev_mase"' in stats
    assert stats.index('scope="row">MASE') < stats.index('scope="row">CRPS')
    assert "King lineage" in stats and "Tokens trained" in stats
    assert '<details class="train-history"' not in stats
    assert "Same-pipeline model reference" in _html(els, "train-ref-status")
    assert "Timestral-209M" in _html(els, "pareto-rows")
    assert "Tokens vs Toto2" not in stats


@pytest.mark.parametrize("reference", [None, {"preset": "toto2-4m", "source": "old", "scores": SIX}])
def test_timestral_reference_pending_never_uses_toto2(tmp_path, reference):
    fx = _bucket(v2_from=1000, block=50000, lane_b=True)
    if reference:
        fx["benchmarks/reference-timestral-209m.json"] = reference
    bench = next(v for k, v in fx.items() if k.startswith("benchmarks/round-"))
    calls = [f"_bench[{json.dumps(bench['round_id'])}]={json.dumps(bench)}",
             "setPage('train')", "renderTrain()"]
    els = _render("index.html", fx, "localhost", calls, tmp_path)
    assert "Published reference" in _html(els, "train-ref-status")
    assert "Timestral-209M" in _html(els, "pareto-rows")
    assert "beats Timestral" not in _html(els, "train-stats")


def test_mainnet_today_v2_not_announced(tmp_path):
    fx = _bucket(v2_from=0, block=9_500_000, lane_b=False)
    els = _render("index.html", fx, "dashboard.casdesub.net", INDEX_CALLS, tmp_path)
    assert _html(els, "v2-state") == "v2 · not announced"
    assert "go-live block not yet announced" in _html(els, "v2-body")
    assert _html(els, "base-arch") == "toto2"
    assert "not been announced" in _html(els, "lb-gate")
    assert "verdict" in els and _html(els, "verdict")          # lane A renders as before
    assert "lane A" in _html(els, "v2-split")
    st = _render("stakeholders.html", fx, "dashboard.casdesub.net", STAKE_CALLS, tmp_path)
    tiles = _html(st, "tiles")
    assert "go-live not announced" in tiles and "Lineage tokens" in tiles
    hero = _html(st, "hero")
    assert "Timestral-209M" in hero and "Cascade king" in hero and "% " not in hero[:0]
    assert "would place" in hero


def test_mainnet_scheduled_counts_down(tmp_path):
    fx = _bucket(v2_from=9_600_000, block=9_500_000, lane_b=False)
    els = _render("index.html", fx, "dashboard.casdesub.net", INDEX_CALLS, tmp_path)
    assert _html(els, "v2-state") == "v2 · block 9600000"
    assert "lane A goes live at block <b>9600000</b>" in _html(els, "v2-body")
    st = _render("stakeholders.html", fx, "dashboard.casdesub.net", STAKE_CALLS, tmp_path)
    assert "#9600000" in _html(st, "tiles") and "in ~" in _html(st, "tiles")


@pytest.mark.parametrize("b_start", [0, 60000])
def test_v2_live_with_lane_b_off_or_scheduled(tmp_path, b_start):
    fx = _bucket(v2_from=1000, block=50000, lane_b=False)
    fx["status/v2.json"]["lane_b"]["from_block"] = b_start
    els = _render("index.html", fx, "dashboard.casdesub.net", INDEX_CALLS, tmp_path)
    assert _html(els, "v2-state") == "v2 live"
    if b_start:
        assert "lane B scheduled at block <b>60000</b>" in _html(els, "v2-body")
        assert "full allocation" in _html(els, "lb-gate")
    else:
        assert "lane A has the full allocation" in _html(els, "v2-body")
        assert "Lane B has not launched" in _html(els, "lb-gate")
    assert "Lane B not launched" in _html(els, "lb-domains")


def test_testnet_live_lane_b(tmp_path):
    fx = _bucket(v2_from=1000, block=50_000, lane_b=True)
    els = _render("index.html", fx, "testnet.casdesub.net", INDEX_CALLS, tmp_path)
    assert _html(els, "v2-state") == "v2 live"
    assert _html(els, "base-arch") == "Cascade-MoE-28M"
    split = _html(els, "v2-split")
    assert "lane A 50%" in split and "lane B 35%" in split and "burn 15%" in split
    gate = _html(els, "lb-gate")
    assert "● open" in gate and "260B" in gate
    doms = _html(els, "lb-domains")
    for want in ("energy", "retail", "uid 12", "40% of the slot", "guard: energy", "✓ pass",
                 "routing_stale", "expires in 21 eras"):
        assert want in doms, want
    assert "<polyline" in _html(els, "lb-sat-svg")
    assert "<path" in _html(els, "lb-pay-svg") and "expires" in _html(els, "lb-pay-svg")
    moe = _html(els, "lb-moe")
    assert "block 1" in moe and 'class="dead"' in moe
    assert "metro-v1:expert:" in _html(els, "lb-enter")
    st = _render("stakeholders.html", fx, "testnet.casdesub.net", STAKE_CALLS, tmp_path)
    tiles = _html(st, "tiles")
    assert "Live" in tiles and "260B" in tiles and ">1<" in tiles.replace("Experts merged", "")
    assert "not comparable" in _html(st, "hero")       # testnet's truncated battery


def test_nothing_published_renders_pending_states(tmp_path):
    els = _render("index.html", {}, "dashboard.casdesub.net", INDEX_CALLS, tmp_path)
    assert _html(els, "v2-state") == "v2 · status pending"
    assert "not published" in _html(els, "lb-gate")
    st = _render("stakeholders.html", {}, "dashboard.casdesub.net", STAKE_CALLS, tmp_path)
    assert "awaiting" in _html(st, "hero") and "awaiting" in _html(st, "tiles")
