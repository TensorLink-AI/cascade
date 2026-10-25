"""DEC-CA-0061 rule v2 ("avg_noninf_winrate", owner 2026-10-20): the
four-suite lane-A bench gate (GIFT-Eval, BOOM, TIME, FEV) over per-task
ratios — suite-equal average < 1, no suite's ratio LCB above 1 + tolerance at
alpha/4, suite-equal win rate >= min — versioned by ``[v2]
bench_gate_v2_from_block``."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from cascade.audit import checks as C
from cascade.eval import bench_gate as BG
from cascade.eval.bench_gate import (
    RULE_V1,
    RULE_V2,
    SUITES_V2,
    evaluate_bench_gate,
    uncomputable_bench_gate,
)
from cascade.eval.benchmarks import extract_bench_scores
from cascade.eval.koth import RoundResult, apply_bench_gate
from cascade.shared.bench_report import (
    BenchEntry,
    BenchReport,
    dump_bench_report,
    dump_bench_rows,
    load_bench_report,
)
from cascade.shared.config.v2 import parse_bench_gate_v2_from_block
from cascade.shared.era import bench_gate_rule_at, bench_gate_v2_configured
from cascade.shared.manifest import BenchScores
from cascade.shared.receipt import VerdictRecord, _verdict_body, dump_receipt, load_receipt

from .receipt_fixture import EPOCH_START, make_scored_receipt

SIX = ("gifteval_crps", "gifteval_mase", "boom_crps", "boom_mase", "time_crps", "time_mase")
EIGHT = (*SIX, "fev_crps", "fev_mase")


def _rows(scale: dict[str, float] | None = None, n: int = 40, seed: int = 0,
          suites=SUITES_V2) -> dict:
    rng = np.random.default_rng(seed)
    out = {}
    for suite in suites:
        k = (scale or {}).get(suite, 1.0)
        base = rng.uniform(0.5, 1.5, size=(n, 2))
        out[suite] = {"status": "ok", "rows": [
            {"item": f"{suite}/{i}", "crps": float(c * k), "mase": float(m * k)}
            for i, (c, m) in enumerate(base)]}
    return out


def _eight(v: float = 1.0, **over) -> dict:
    return {**dict.fromkeys(EIGHT, v), **over}


def _gate(king_rows, chal_rows, king_s, chal_s, rule=RULE_V2, **kw):
    return evaluate_bench_gate(king_rows, chal_rows, king_s, chal_s,
                               alpha=0.05, B=400, seed="blockhash", rule=rule, **kw)


def _win() -> RoundResult:
    return RoundResult(lcb=0.05, margin=0.01, challenger_wins_round=True, n_windows=100,
                       king_geomean=1.0, chal_geomean=0.9, inconclusive=False)


# ── the rule ─────────────────────────────────────────────────────────────────


def _scaled(per_suite: dict[str, list[float]] | dict[str, float], n: int = 40,
            seed: int = 0) -> tuple[dict, dict]:
    """(king rows, chal rows): the challenger's task ``i`` of a suite is the
    king's times ``per_suite[suite][i]`` (a float = every task), CRPS and MASE
    alike."""
    king = _rows(n=n, seed=seed)
    chal = json.loads(json.dumps(king))
    for suite, f in per_suite.items():
        fs = [f] * n if isinstance(f, (int, float)) else list(f)
        for r, x in zip(chal[suite]["rows"], fs, strict=True):
            r["crps"] *= x
            r["mase"] *= x
    return king, chal


def test_better_everywhere_passes_and_records_the_three_numbers():
    g = _gate(_rows(), _rows(dict.fromkeys(SUITES_V2, 0.95)), _eight(1.0), _eight(0.95))
    assert g.computed and g.passed, g.reason
    assert set(g.suite_ucb) == set(SUITES_V2) == {"gift", "boom", "time", "fev"}
    d = g.detail()
    assert d["rule"] == RULE_V2 == "avg_noninf_winrate" and d["passed"] is True
    assert d["avg_ratio"] == pytest.approx(0.95) and d["win_rate"] == 1.0
    assert d["worst_lcb"] == max(d["suites"][s]["ratio_lcb"] for s in SUITES_V2) < 1.0
    assert d["tolerance"] == 0.01 and d["min_win_rate"] == 0.5
    assert all(d["suites"][s]["n"] == 40 and d["suites"][s]["masked"] == 0 for s in SUITES_V2)
    assert BG.detail_verdict(d) is True


def test_hair_worse_on_one_suites_mase_but_better_on_average_now_passes():
    # the owner's case: the suite-level FEV MASE is a hair worse (the strict
    # rule blocked this), the tasks are better on average and FEV is flat
    king, chal = _scaled({"gift": 0.95, "boom": 0.95, "time": 0.95, "fev": 1.0005})
    g = _gate(king, chal, _eight(1.0), _eight(0.95, fev_mase=1.0005))
    assert g.computed and g.passed, g.reason
    assert BG.strict_eight(_eight(1.0), _eight(0.95, fev_mase=1.0005))["fev_mase"] is False


def test_average_alone_blocks():
    # every suite flat (no regression, win rate 1/2 via exact ties): average = 1
    g = _gate(_rows(), _rows(), _eight(1.0), _eight(1.0))
    assert g.computed and g.passed is False
    assert g.avg_ratio == 1.0 and g.win_rate == 0.5 and g.worst_lcb <= 1.0 + 0.01
    assert "average ratio" in g.reason and "regression" not in g.reason
    assert "win rate" not in g.reason


def test_significant_regression_on_one_suite_blocks():
    # big wins elsewhere carry the average, FEV is 15 % worse on every task
    king, chal = _scaled({"gift": 0.8, "boom": 0.8, "time": 0.8, "fev": 1.15})
    g = _gate(king, chal, _eight(1.0), _eight(0.9))
    assert g.computed and g.passed is False
    assert g.avg_ratio < 1.0 and g.win_rate == 0.75
    assert "significant regression on fev" in g.reason
    assert g.suites["fev"]["ratio_lcb"] > 1.01 and g.worst_lcb == g.suites["fev"]["ratio_lcb"]
    assert "average" not in g.reason and "win rate" not in g.reason


def test_tolerance_is_the_regression_slack():
    king, chal = _scaled({"gift": 0.9, "boom": 0.9, "time": 0.9, "fev": 1.02})
    assert "regression on fev" in _gate(king, chal, _eight(1.0), _eight(0.9)).reason
    g = _gate(king, chal, _eight(1.0), _eight(0.9), tolerance=0.05)
    assert g.passed, g.reason


def test_win_rate_below_half_with_big_wins_blocks():
    # each suite: 15 tasks 40 % better, 25 tasks 1 % worse — the average is
    # far below 1 and no suite regresses significantly, but 15/40 < 1/2
    f = [0.6] * 15 + [1.01] * 25
    king, chal = _scaled(dict.fromkeys(SUITES_V2, f))
    g = _gate(king, chal, _eight(1.0), _eight(0.9))
    assert g.computed and g.passed is False
    assert g.avg_ratio < 0.9 and g.worst_lcb <= 1.01
    assert g.win_rate == pytest.approx(15 / 40)
    assert g.reason == f"win rate {15 / 40:.4f} < 0.5000"
    assert _gate(king, chal, _eight(1.0), _eight(0.9), min_win_rate=0.3).passed


def test_exact_ties_count_half():
    assert BG.suite_win_rate([-0.1, 0.0, 0.1, 0.0]) == 0.5
    assert BG.suite_win_rate([0.0, 0.0, -1e-300]) == pytest.approx(2 / 3)


def test_suites_weigh_equally_not_tasks():
    # GIFT has 200 tasks 10 % better; BOOM/TIME/FEV 10 tasks each 8 % worse.
    # Pooled over tasks the average would be ~0.94 (pass); suite-equal it is
    # (0.9 * 1.08^3)^(1/4) > 1 (block). Win rate likewise: 1/4, not 200/230.
    king = {**_rows(n=10, seed=1)}
    king["gift"] = _rows(n=200, seed=2)["gift"]
    chal = json.loads(json.dumps(king))
    for suite, f in {"gift": 0.9, "boom": 1.08, "time": 1.08, "fev": 1.08}.items():
        for r in chal[suite]["rows"]:
            r["crps"] *= f
            r["mase"] *= f
    g = _gate(king, chal, _eight(1.0), _eight(1.0), tolerance=0.2)
    assert g.suite_n == {"gift": 200, "boom": 10, "time": 10, "fev": 10}
    assert g.avg_ratio == pytest.approx((0.9 * 1.08 ** 3) ** 0.25)
    assert g.avg_ratio > 1.0 and g.win_rate == pytest.approx(0.25)
    assert not g.passed and "average ratio" in g.reason and "win rate" in g.reason


def test_task_score_combines_crps_and_mase():
    # CRPS 20 % better, MASE 20 % worse on every task: r = sqrt(0.8 * 1.2) < 1
    king = _rows()
    chal = json.loads(json.dumps(king))
    for suite in SUITES_V2:
        for r in chal[suite]["rows"]:
            r["crps"] *= 0.8
            r["mase"] *= 1.2
    g = _gate(king, chal, _eight(1.0), _eight(1.0))
    assert g.avg_ratio == pytest.approx((0.8 * 1.2) ** 0.5)
    assert g.win_rate == 1.0 and g.passed, g.reason


def test_zero_and_non_finite_denominators_are_masked_and_counted():
    king, chal = _scaled(dict.fromkeys(SUITES_V2, 0.9))
    king["boom"]["rows"][0]["mase"] = 0.0           # zero denominator
    king["boom"]["rows"][1]["crps"] = None          # non-finite (parsed to None)
    king["boom"]["rows"][2]["crps"] = float("inf")
    g = _gate(king, chal, _eight(1.0), _eight(0.9))
    assert g.passed, g.reason
    assert g.suites["boom"]["n"] == 37 and g.suites["boom"]["masked"] == 3
    assert g.detail()["suites"]["boom"]["masked"] == 3
    # below the minimum task count ⇒ uncomputable, never a silent pass
    for r in king["time"]["rows"][:31]:
        r["crps"] = 0.0
    u = _gate(king, chal, _eight(1.0), _eight(0.9))
    assert not u.computed and u.passed is None
    assert "time" in u.reason
    assert u.detail()["computed"] is False and BG.detail_verdict(u.detail()) is None


def test_challenger_cannot_win_by_omitting_its_bad_tasks():
    king, chal = _scaled(dict.fromkeys(SUITES_V2, 0.9))
    chal["boom"]["rows"][0]["crps"] = float("nan")
    result = _gate(king, chal, _eight(1.0), _eight(0.9))
    assert result.computed and result.passed is False
    assert "challenger failed required task" in result.reason
    chal["boom"]["rows"].pop(0)
    assert _gate(king, chal, _eight(1.0), _eight(0.9)).passed is False


def test_alpha_split_over_four_suites(monkeypatch):
    seen: list[tuple[str, float]] = []
    real = BG.paired_geomean_lcb

    def spy(k, c, *, alpha, B, seed):
        seen.append((str(seed).rsplit("|", 1)[-1], alpha))
        return real(k, c, alpha=alpha, B=B, seed=seed)

    monkeypatch.setattr(BG, "paired_geomean_lcb", spy)
    _gate(_rows(), _rows(dict.fromkeys(SUITES_V2, 0.9)), _eight(1.0), _eight(0.9))
    assert [s for s, _ in seen] == list(SUITES_V2)
    assert all(a == pytest.approx(1.0 - 0.05 / 4) for _, a in seen)
    seen.clear()
    _gate(_rows(), _rows(dict.fromkeys(SUITES_V2, 0.9)), _eight(1.0), _eight(0.9),
          rule=RULE_V1)
    assert [s for s, _ in seen] == ["gift", "boom", "time"]
    assert all(a == pytest.approx(1.0 - 0.05 / 3) for _, a in seen)


def test_replay_is_deterministic_and_the_detail_implies_the_outcome():
    king, chal = _scaled({"gift": 0.97, "boom": 1.01, "time": 0.95, "fev": 0.99}, seed=3)
    a = _gate(king, chal, _eight(1.0), _eight(0.9))
    b = _gate(json.loads(json.dumps(king)), json.loads(json.dumps(chal)),
              _eight(1.0), _eight(0.9))
    assert a == b and a.detail() == b.detail()
    d = json.loads(json.dumps(a.detail()))
    assert BG.detail_verdict(d) is a.passed
    d["win_rate"] = 0.1                                       # recorded number edited
    assert BG.detail_verdict(d) is False
    # a different seed moves the bootstrap LCBs, never the point numbers
    c = evaluate_bench_gate(king, chal, _eight(1.0), _eight(0.9), alpha=0.05, B=400,
                            seed="other", rule=RULE_V2)
    assert c.avg_ratio == a.avg_ratio and c.win_rate == a.win_rate


def test_old_rule_is_the_default_and_unchanged():
    k, c = _rows(suites=("gift", "boom", "time")), _rows({"boom": 1.02},
                                                        suites=("gift", "boom", "time"))
    a = evaluate_bench_gate(k, c, _eight(1.0), _eight(0.99), alpha=0.05, B=400, seed=7)
    b = evaluate_bench_gate(k, c, _eight(1.0), _eight(0.99), alpha=0.05, B=400, seed=7,
                            rule=RULE_V1)
    assert a == b and a.rule == RULE_V1 and a.detail() is None and a.suites == {}
    # the DEC-CA-0054 rule ignores FEV entirely (no fev rows, a worse fev number)
    ok = evaluate_bench_gate(k, _rows(dict.fromkeys(SUITES_V2, 0.9)),
                             _eight(1.0, fev_crps=0.1), _eight(0.9, fev_crps=5.0),
                             alpha=0.05, B=400, seed=7)
    assert ok.passed is True
    # and receipts it judges record no detail
    assert apply_bench_gate(_win(), a, mode="enforce").bench_gate_detail is None
    with pytest.raises(ValueError):
        evaluate_bench_gate(k, c, _eight(), _eight(), alpha=0.05, B=10, seed=1, rule="x")


def test_blocking_folds_into_the_round():
    king, chal = _scaled({"gift": 0.8, "boom": 0.8, "time": 0.8, "fev": 1.15})
    g = _gate(king, chal, _eight(1.0), _eight(0.9))
    r = apply_bench_gate(_win(), g, mode="enforce")
    assert not r.challenger_wins_round and not r.inconclusive
    assert r.bench_gate_detail["passed"] is False
    s = apply_bench_gate(_win(), g, mode="shadow")
    assert s.challenger_wins_round and s.bench_gate_detail["passed"] is False


# ── missing FEV: uncomputable, like a missing suite under DEC-CA-0054 ─────────


def test_missing_fev_rows_is_uncomputable_and_holds_under_enforce():
    chal = _rows(dict.fromkeys(SUITES_V2, 0.9))
    del chal["fev"]
    g = _gate(_rows(), chal, _eight(1.0), _eight(0.9))
    assert not g.computed and g.passed is None and "fev" in g.reason
    r = apply_bench_gate(_win(), g, mode="enforce")
    assert r.inconclusive and not r.challenger_wins_round and r.bench_gate_passed is None
    assert r.bench_gate_detail["computed"] is False
    bad = _rows()
    bad["fev"]["status"] = "skipped"
    assert not _gate(bad, _rows(), _eight(1.0), _eight(0.9)).computed


def test_rule_v2_reads_rows_not_suite_numbers():
    # the suite-level numbers are not read (FEV's absent, or NaN): rows decide
    six = {k: 0.9 for k in SIX}
    g = _gate(_rows(), _rows(dict.fromkeys(SUITES_V2, 0.9)), _eight(1.0), six)
    assert g.computed and g.passed, g.reason
    u = uncomputable_bench_gate("signed bench rows missing for king", rule=RULE_V2)
    assert u.detail()["rule"] == RULE_V2 and u.detail()["computed"] is False


def test_extract_bench_scores_fev_optional_or_required():
    def suite(name, crps=0.5, mase=0.8, status="ok"):
        return {"suite": name, "status": status, "metrics": {"crps": crps, "mase": mase}}

    base = [suite("gift-eval"), suite("boom"), suite("time")]
    assert set(extract_bench_scores({"suites": base})) == set(SIX)
    assert extract_bench_scores({"suites": base}, require_fev=True) is None
    with_fev = {"suites": [*base, suite("fev", 0.4, 0.7)]}
    got = extract_bench_scores(with_fev, require_fev=True)
    assert got["fev_crps"] == 0.4 and got["fev_mase"] == 0.7
    failed = {"suites": [*base, suite("fev", status="error")]}
    assert extract_bench_scores(failed, require_fev=True) is None
    assert set(extract_bench_scores(failed)) == set(SIX)


# ── the trainer's signed report carries FEV ──────────────────────────────────


def _entry(scores: BenchScores, **kw) -> BenchEntry:
    return BenchEntry(role="challenger", size="moe", miner_hotkey="hk", miner_uid=3,
                      trained_pointer="metro-v1:trained:hippius:r@sha256:" + "0" * 64,
                      scores=scores, **kw)


def test_bench_report_carries_fev_and_old_reports_keep_their_bytes():
    six = BenchScores(**{k: 0.9 for k in SIX})
    body = BenchReport(round_id="1", created_block=9, entries=(_entry(six),)).canonical_body()
    assert b"fev_" not in body                          # archived signatures still verify
    eight = BenchScores(**_eight(0.9, fev_crps=0.7, fev_mase=0.8))
    rep = BenchReport(round_id="1", created_block=9, entries=(_entry(eight),))
    assert b'"fev_crps":0.7' in rep.canonical_body()
    back = load_bench_report(dump_bench_report(rep))
    assert back.entries[0].scores == eight
    assert back.canonical_body() == rep.canonical_body()
    assert load_bench_report(dump_bench_report(
        BenchReport(round_id="1", created_block=9, entries=(_entry(six),)))).entries[0] \
        .scores.fev_crps is None


def test_manifest_bench_scores_keep_their_bytes():
    from cascade.shared.manifest import _bench_from_json, bench_scores_body

    six = BenchScores(**{k: 0.9 for k in SIX})
    assert set(bench_scores_body(six)) == set(SIX)
    assert _bench_from_json(bench_scores_body(six)) == six


def test_trainer_sweep_adds_fev_only_when_the_rule_is_configured(cfg):
    from cascade.trainer.bench_hook import BenchPlan, bench_suites_for, build_bench_remote_command

    assert bench_suites_for(cfg, "gift-eval,boom,time") == "gift-eval,boom,time"
    armed = _armed(cfg, 600)
    assert bench_gate_v2_configured(armed)
    assert bench_suites_for(armed, "gift-eval,boom,time") == "gift-eval,boom,time,fev"
    assert bench_suites_for(armed, "gift-eval,fev") == "gift-eval,fev"
    host = SimpleNamespace(workdir="/w", cuda_device=None)
    cmd, _ = build_bench_remote_command(host, "1", "moe", BenchPlan(
        suites="gift-eval,boom,time,fev", data_dir="/d"))
    assert "--extra time --extra fev" in cmd and "fev" in cmd
    assert "test -f /d/fev/manifest.json &&" in cmd    # no mutable FEV download fallback
    cmd, _ = build_bench_remote_command(host, "1", "moe", BenchPlan(
        suites="gift-eval,boom,time", data_dir="/d"))
    assert "--extra fev" not in cmd


# ── config + versioning ──────────────────────────────────────────────────────


def _armed(cfg, block: int, mode: str = "enforce"):
    return replace(cfg, v2=replace(cfg.v2, bench_gate_v2_from_block=block,
                                   bench_gate_mode=mode, from_block=cfg.v2.from_block),
                   scoring=replace(cfg.scoring, bench_gate_mode=mode))


def test_parse_key():
    assert parse_bench_gate_v2_from_block({}) == 0
    assert parse_bench_gate_v2_from_block({"bench_gate_v2_from_block": 1200}) == 1200
    assert parse_bench_gate_v2_from_block({"bench_gate_v2_from_block": "v2"}) == 0
    assert parse_bench_gate_v2_from_block(
        {"bench_gate_v2_from_block": "v2", "from_block": 8190000}) == 8190000
    for bad in (-1, "soon", True, 1.5):
        with pytest.raises(ValueError):
            parse_bench_gate_v2_from_block({"bench_gate_v2_from_block": bad})


def test_shipped_configs():
    from cascade.shared.config import load_chain_config

    main = load_chain_config("chain.toml")
    assert main.v2.from_block == 0 and main.v2.bench_gate_v2_from_block == 0
    assert main.scoring.bench_gate_tolerance == 0.01
    assert main.scoring.bench_gate_min_win_rate == 0.5
    test = load_chain_config("chain.testnet.toml")
    # testnet aligned with mainnet ("v2"); inert while its gate mode is off
    assert test.v2.bench_gate_v2_from_block == test.v2.from_block > 0
    assert bench_gate_rule_at(test, 10**9) == RULE_V2
    assert not bench_gate_v2_configured(test)
    from cascade.trainer.runner_rolling import _bench_fev_required
    assert not _bench_fev_required(test, SimpleNamespace(start_block=10**9))
    assert (test.scoring.bench_gate_tolerance, test.scoring.bench_gate_min_win_rate) == (
        0.01, 0.5)


def test_threshold_keys_parse_and_are_dropped_at_default(cfg):
    from cascade.shared.config.scoring import _bench_gate_tolerance, _bench_gate_win_rate

    assert _bench_gate_tolerance(0) == 0.0 and _bench_gate_win_rate(1) == 1.0
    for bad in (-0.01, 0.5, True, "x", float("nan")):
        with pytest.raises(ValueError):
            _bench_gate_tolerance(bad)
    for bad in (-0.1, 1.01, False, None):
        with pytest.raises(ValueError):
            _bench_gate_win_rate(bad)
    tr = SimpleNamespace(dethroned=False, note="", state=SimpleNamespace(
        king_hotkey="k", king_uid=0))
    base = VerdictRecord.from_round(_win(), tr, params=cfg.koth_params(block=0),
                                    bootstrap_seed="1")
    assert "bench_gate_tolerance" not in base.params
    assert "bench_gate_min_win_rate" not in base.params
    tuned = replace(cfg, scoring=replace(cfg.scoring, bench_gate_tolerance=0.02,
                                         bench_gate_min_win_rate=0.6))
    rec = VerdictRecord.from_round(_win(), tr, params=tuned.koth_params(block=0),
                                   bootstrap_seed="1")
    assert rec.params["bench_gate_tolerance"] == 0.02
    assert rec.params["bench_gate_min_win_rate"] == 0.6


def test_rule_switches_at_an_era_start_only(cfg):
    era = int(cfg.round.epoch_blocks) * max(1, int(cfg.round.era_settlements))
    n = 1000 * era
    armed = _armed(cfg, n)
    assert bench_gate_rule_at(armed, n) == RULE_V1        # the settlement AT n: last old era
    assert bench_gate_rule_at(armed, n + int(cfg.round.epoch_blocks)) == RULE_V2
    assert bench_gate_rule_at(armed, n - era) == RULE_V1
    assert bench_gate_rule_at(cfg, n + 10 * era) == RULE_V1
    # stamped on KothParams; the default is dropped from receipts
    assert armed.koth_params(block=n).bench_gate_rule == RULE_V1
    assert armed.koth_params(block=n + era).bench_gate_rule == RULE_V2
    tr = SimpleNamespace(dethroned=False, note="", state=SimpleNamespace(
        king_hotkey="k", king_uid=0))
    old = VerdictRecord.from_round(_win(), tr, params=armed.koth_params(block=n),
                                   bootstrap_seed="1")
    assert "bench_gate_rule" not in old.params
    assert "bench_gate_detail" not in _verdict_body(old)
    new = VerdictRecord.from_round(_win(), tr, params=armed.koth_params(block=n + era),
                                   bootstrap_seed="1")
    assert new.params["bench_gate_rule"] == RULE_V2


# ── audit replay ─────────────────────────────────────────────────────────────


def _audit_case(cfg, chal_scale: float = 0.9):
    """A scored receipt judged under DEC-CA-0061, plus the signed-report
    replay input its validator read."""
    from cascade.shared.era import settlement_contract_block

    armed = _armed(cfg, settlement_contract_block(cfg.round, EPOCH_START))
    receipt, _, _ = make_scored_receipt(armed)
    assert bench_gate_rule_at(armed, receipt.epoch_start_block) == RULE_V2
    m = receipt.load_embedded_manifest()
    king = m.entry_for_role("king")
    chal = m.entries_for_role("challenger")[-1]
    king_rows, chal_rows = _rows(), _rows(dict.fromkeys(SUITES_V2, chal_scale))
    king_s, chal_s = _eight(1.0), _eight(chal_scale)
    p = armed.koth_params(block=receipt.epoch_start_block)
    gate = evaluate_bench_gate(king_rows, chal_rows, king_s, chal_s, alpha=p.bootstrap_alpha,
                               B=p.bootstrap_B, seed=C._bootstrap_seed(
                                   receipt.verdict.bootstrap_seed),
                               min_items=p.bench_gate_min_items, rule=RULE_V2)
    v = receipt.verdict
    params = {**dict(v.params), "bench_gate_rule": RULE_V2, "bench_gate_mode": "enforce"}
    receipt = replace(receipt, verdict=replace(
        v, params=params, bench_gate_passed=gate.passed, bench_gate_detail=gate.detail()))
    bench = {king.trained_pointer: (king_rows, king_s),
             chal.trained_pointer: (chal_rows, chal_s)}
    return armed, receipt, bench, chal


def test_audit_replays_rule_v2_deterministically(cfg):
    armed, receipt, bench, _ = _audit_case(cfg)
    r = C.check_bench_gate(receipt, armed, bench)
    assert r.status == C.PASS, r
    # a JSON round trip keeps the detail (and the signed bytes)
    back = load_receipt(dump_receipt(receipt))
    assert back.verdict.bench_gate_detail == json.loads(
        json.dumps(receipt.verdict.bench_gate_detail))
    assert C.check_bench_gate(back, armed, bench).status == C.PASS
    # without the report: consistency only, never a silent pass
    assert C.check_bench_gate(receipt, armed, None).status == C.WARN


def test_audit_tamper_fails(cfg):
    armed, receipt, bench, chal = _audit_case(cfg)
    v = receipt.verdict
    # 1. the signed rows say the challenger regressed on FEV's tasks
    rows, scores = bench[chal.trained_pointer]
    worse = json.loads(json.dumps(rows))
    for r in worse["fev"]["rows"]:
        r["crps"] *= 1.5
        r["mase"] *= 1.5
    forged = {**bench, chal.trained_pointer: (worse, scores)}
    assert C.check_bench_gate(receipt, armed, forged).status == C.FAIL
    # 2. a recorded per-suite ratio edited (the three numbers re-derive
    # differently ⇒ the detail no longer implies its outcome)
    d = json.loads(json.dumps(v.bench_gate_detail))
    d["suites"]["time"]["ratio"] = 0.5
    r = C.check_bench_gate(replace(receipt, verdict=replace(v, bench_gate_detail=d)),
                           armed, bench)
    assert r.status == C.FAIL
    # 2b. the recorded numbers edited consistently: the replay still differs
    d = json.loads(json.dumps(v.bench_gate_detail))
    d["suites"]["time"]["win_rate"] = 0.9
    d["win_rate"] = BG.combine_suites(d["suites"], tolerance=0.01, min_win_rate=0.5)[2]
    assert BG.detail_verdict(d) is True
    r = C.check_bench_gate(replace(receipt, verdict=replace(v, bench_gate_detail=d)),
                           armed, bench)
    assert r.status == C.FAIL and "differs from the replay" in r.detail
    # 2c. a recorded tolerance other than the params'
    d = json.loads(json.dumps(v.bench_gate_detail))
    d["tolerance"] = 0.2
    r = C.check_bench_gate(replace(receipt, verdict=replace(v, bench_gate_detail=d)),
                           armed, bench)
    assert r.status == C.FAIL
    # 3. a flipped outcome contradicts its own recorded numbers (no rows needed)
    flipped = replace(receipt, verdict=replace(v, bench_gate_passed=False))
    assert C.check_bench_gate(flipped, armed, None).status == C.FAIL
    # 4. the old rule stamped where the new one is in force
    old = replace(receipt, verdict=replace(
        v, params={**dict(v.params), "bench_gate_rule": RULE_V1}, bench_gate_detail=None))
    assert C.check_bench_gate(old, armed, bench).status == C.FAIL
    # 5. detail stripped from a DEC-CA-0061 receipt
    bare = replace(receipt, verdict=replace(v, bench_gate_detail=None))
    assert C.check_bench_gate(bare, armed, bench).status == C.FAIL
    # 6. rows absent: the replay is uncomputable, the recorded pass is not
    assert C.check_bench_gate(receipt, armed, {}).status == C.FAIL


def test_audit_replays_the_old_rule(cfg):
    receipt, _, _ = make_scored_receipt(cfg)
    m = receipt.load_embedded_manifest()
    king, chal = m.entry_for_role("king"), m.entries_for_role("challenger")[-1]
    k3, c3 = _rows(suites=("gift", "boom", "time")), _rows({"boom": 1.2},
                                                          suites=("gift", "boom", "time"))
    bench = {king.trained_pointer: (k3, _eight(1.0)), chal.trained_pointer: (c3, _eight(0.9))}
    v = receipt.verdict
    blocked = replace(receipt, verdict=replace(v, bench_gate_passed=False))
    assert C.check_bench_gate(blocked, cfg, bench).status == C.PASS
    passed = replace(receipt, verdict=replace(v, bench_gate_passed=True))
    assert C.check_bench_gate(passed, cfg, bench).status == C.FAIL
    assert C.check_bench_gate(receipt, cfg, bench).status == C.SKIP     # gate never ran


def test_fetch_bench_inputs_pins_rows_by_sha(cfg, monkeypatch):
    from cascade.audit.main import fetch_bench_inputs
    from cascade.shared import bench_report as BR
    from cascade.shared.hippius import ObjectNotFound

    rows = _rows()
    text = dump_bench_rows(rows)
    sha = hashlib.sha256(text.encode()).hexdigest()
    eight = BenchScores(**_eight(0.9))
    rep = BenchReport(round_id="7", created_block=1, entries=(
        _entry(eight, rows_key="benchmarks/rows/x.json", rows_sha256=sha),))
    objects = {BR.bench_report_key("7"): dump_bench_report(rep),
               "benchmarks/rows/x.json": text}

    class Store:
        def get_text(self, key):
            if key not in objects:
                raise ObjectNotFound(key)
            return objects[key]

    receipt = SimpleNamespace(round_id="7")
    monkeypatch.setattr(BR, "verify_bench_report_signature", lambda r, hk: True)
    got = fetch_bench_inputs(cfg, receipt, Store())
    ptr = rep.entries[0].trained_pointer
    assert got[ptr][0] == rows and got[ptr][1] == dataclasses.asdict(eight)
    objects["benchmarks/rows/x.json"] = text + " "                  # sha mismatch
    assert fetch_bench_inputs(cfg, receipt, Store())[ptr] == (None, None)
    monkeypatch.setattr(BR, "verify_bench_report_signature", lambda r, hk: False)
    assert fetch_bench_inputs(cfg, receipt, Store()) == {}         # unsigned: absent
    assert fetch_bench_inputs(cfg, SimpleNamespace(round_id="8"), Store()) == {}


def test_trainer_requires_fev_only_in_rule_eras(cfg):
    from cascade.trainer.runner_rolling import _bench_fev_required

    armed = _armed(cfg, 14400)
    assert _bench_fev_required(armed, SimpleNamespace(start_block=14400))
    assert not _bench_fev_required(armed, SimpleNamespace(start_block=0))
    assert not _bench_fev_required(cfg, SimpleNamespace())          # rule unset: era unread
