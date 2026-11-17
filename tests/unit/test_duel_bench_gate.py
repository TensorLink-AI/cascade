"""`cascade duel` and the website show the public-benchmark gate per suite
(owner 2026-11-17: two challengers cleared the private pool and were blocked by
the gate, and nobody outside the validator could see which suite did it). The
detail comes from the round's signed receipt, ``verdict.bench_gate_detail``."""

from __future__ import annotations

import re
from pathlib import Path

from cascade.miner import dashboard as dash

# round 7979878267927883534, uid 69 vs the king (abridged, real numbers)
DETAIL = {
    "avg_ratio": 0.99444, "computed": True, "min_win_rate": 0.5, "passed": False,
    "reason": "win rate 0.4845 < 0.5000", "rule": "avg_noninf_winrate",
    "suites": {
        "boom": {"n": 7371, "ratio": 0.98253, "ratio_lcb": 0.97846, "win_rate": 0.50265},
        "fev": {"n": 65, "ratio": 0.99618, "ratio_lcb": 0.98493, "win_rate": 0.49231},
        "gift": {"n": 97, "ratio": 0.99911, "ratio_lcb": 0.99098, "win_rate": 0.43299},
        "time": {"n": 98, "ratio": 1.00004, "ratio_lcb": 0.99651, "win_rate": 0.51020},
    },
    "tolerance": 0.01, "win_rate": 0.48454, "worst_lcb": 0.99651,
}


def test_lines_weakest_suite_first_and_flag_the_bar():
    lines = dash.bench_gate_lines(DETAIL)
    assert "BLOCKED the dethrone" in lines[0] and "48.5%" in lines[0] and "needs 50%" in lines[0]
    body = lines[2:]
    assert [ln.split()[0] for ln in body] == ["GIFT-Eval", "FEV", "BOOM", "TIME"]
    assert "below the bar" in body[0] and "43.3%" in body[0]
    assert "below the bar" not in body[-1]


def test_lines_uncomputed_and_absent():
    assert dash.bench_gate_lines(None) == []
    out = dash.bench_gate_lines({"computed": False, "reason": "validator bench infra"})
    assert out == ["  public bench   not computed — validator bench infra"]


def test_fetch_reads_the_scored_receipt(monkeypatch):
    seen = []

    def fake(storage, key, *, timeout=10.0):
        seen.append(key)
        return {"verdict": {"bench_gate_detail": DETAIL}}

    monkeypatch.setattr(dash, "fetch_public_json", fake)
    rows = [{"status": "rejected", "receipt_key": "receipts/a/x.json"},
            {"status": "scored", "receipt_key": "receipts/b/x.json"}]
    assert dash.fetch_bench_gate(object(), rows) == DETAIL
    assert seen == ["receipts/b/x.json"]
    assert dash.fetch_bench_gate(object(), [{"status": "rejected"}]) is None
    monkeypatch.setattr(dash, "fetch_public_json", lambda *a, **k: None)
    assert dash.fetch_bench_gate(object(), rows) is None


def test_render_duel_appends_the_gate():
    row = {"round_id": "7979878267927883534", "status": "scored", "epoch_start_block": 9522900,
           "lcb": 0.2782, "margin": 0.01}
    text = dash.render_duel([row], bench_gate=DETAIL)
    assert "public bench" in text and "GIFT-Eval" in text
    assert "public bench" not in dash.render_duel([row])


def test_website_renders_the_gate():
    page = (Path(__file__).resolve().parents[2] / "cascade" / "website" / "index.html").read_text()
    assert "bench_gate_detail" in page and "function benchGateHtml(r)" in page
    m = re.search(r"function renderCohort\(\)\{(.*?)\nfunction renderChart", page, re.S)
    assert m and "benchGateHtml(r)" in m.group(1)
