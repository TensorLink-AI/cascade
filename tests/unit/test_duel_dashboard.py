"""Duel-only dashboard: per-horizon verdict breakdown + the whole seated field.

Validator side: ``per_horizon`` / ``cohort_geomeans`` / ``cohort_per_horizon``
ride the receipt (dropped from the canonical body when absent, so archived
signatures survive) and the public index. Trainer side: a duel-only round's
standings list every entrant as ``seated`` or ``waiting``. CLI: ``cascade
heat`` and ``cascade duel`` render both.
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from cascade.eval.koth import per_horizon_breakdown
from cascade.eval.scoring import WindowScore


def _scores(scale, seed, ids):
    rng = np.random.default_rng(seed)
    return [WindowScore(series_id=i, mase=float(rng.uniform(0.5, 1.5) * scale),
                        qloss_per_q=rng.uniform(0.1, 1.0, size=9) * scale,
                        abs_target=float(rng.uniform(5.0, 10.0)))
            for i in ids]


# 3 × 80 = 240 windows: above [scoring] min_windows (200), so the duel is decided
LADDER_IDS = ([f"h64-s{i}" for i in range(80)] + [f"h256-s{i}" for i in range(80)]
              + [f"h720-s{i}" for i in range(80)])


# ── eval: the breakdown ──────────────────────────────────────────────────────


def test_per_horizon_breakdown_only_on_ladder_ids():
    king = _scores(1.0, 0, LADDER_IDS)
    chal = [WindowScore(s.series_id, s.mase * 0.8, s.qloss_per_q * 0.8, s.abs_target)
            for s in king]
    ph = per_horizon_breakdown(king, chal)
    assert list(ph) == ["64", "256", "720"]
    for h in ph:
        assert ph[h]["n"] == 80
        assert ph[h]["chal"] < ph[h]["king"]
        assert ph[h]["win_rate"] == 1.0
    # legacy single-horizon ids ⇒ nothing (every archived round)
    assert per_horizon_breakdown(_scores(1.0, 0, [str(i) for i in range(10)]),
                                 _scores(1.0, 1, [str(i) for i in range(10)])) is None
    assert per_horizon_breakdown([], []) is None


def test_evaluate_round_carries_per_horizon(cfg):
    from cascade.eval.koth import evaluate_round

    king = _scores(1.0, 0, LADDER_IDS)
    chal = [WindowScore(s.series_id, s.mase * 0.7, s.qloss_per_q * 0.7, s.abs_target)
            for s in king]
    res = evaluate_round(king, chal, cfg.koth_params(), seed=3, king_tenure_rounds=0)
    assert res.per_horizon is not None and set(res.per_horizon) == {"64", "256", "720"}
    legacy = evaluate_round(_scores(1.0, 0, [str(i) for i in range(240)]),
                            _scores(0.7, 0, [str(i) for i in range(240)]),
                            cfg.koth_params(), seed=3, king_tenure_rounds=0)
    assert legacy.per_horizon is None


# ── eval: the per-DOMAIN breakdown (which domains did I beat the king in) ────


DOMS = ["nature", "energy", "sales", "web"]


def _domained(scores, doms=DOMS):
    return [replace(s, domain=doms[i % len(doms)]) for i, s in enumerate(scores)]


def _rescaled(king, factor_by_domain):
    """A paired challenger whose scores are ``factor`` × the king's, per domain
    — so the per-domain gaps are known exactly."""
    return [WindowScore(s.series_id, s.mase * factor_by_domain[s.domain],
                        s.qloss_per_q * factor_by_domain[s.domain], s.abs_target,
                        domain=s.domain, source=s.source)
            for s in king]


def test_per_domain_breakdown_reports_scores_not_just_win_rates():
    from cascade.eval.koth import per_domain_breakdown
    from cascade.eval.scoring import global_geomean

    king = _domained(_scores(1.0, 0, [str(i) for i in range(240)]))
    chal = _rescaled(king, {"nature": 0.8, "energy": 0.9, "sales": 1.1, "web": 1.0})
    pd = per_domain_breakdown(king, chal)
    assert list(pd) == sorted(DOMS)                          # sorted by name
    for d in DOMS:
        assert pd[d]["n"] == 60
        # the domain geomeans ARE the verdict statistic on the domain's windows
        ks = [s for s in king if s.domain == d]
        assert pd[d]["king"] == pytest.approx(global_geomean(ks))
    # the gap carries the magnitude the win rate cannot: 20% better on nature,
    # 10% worse on sales, and a tie (factor 1.0) on web is win_rate 0 (not <)
    assert pd["nature"]["chal"] / pd["nature"]["king"] == pytest.approx(0.8)
    assert pd["sales"]["chal"] / pd["sales"]["king"] == pytest.approx(1.1)
    assert pd["nature"]["win_rate"] == 1.0 and pd["sales"]["win_rate"] == 0.0
    assert pd["web"]["chal"] == pytest.approx(pd["web"]["king"])
    # a domain-less pool (every archived round) ⇒ None, nothing changes for it
    plain = _scores(1.0, 0, [str(i) for i in range(10)])
    assert per_domain_breakdown(plain, plain) is None
    assert per_domain_breakdown([], []) is None
    assert per_domain_breakdown(king, king[:10]) is None                # unpaired


def test_evaluate_round_carries_per_domain(cfg):
    from cascade.eval.koth import evaluate_round

    king = _domained(_scores(1.0, 0, [str(i) for i in range(240)]))
    chal = _rescaled(king, {"nature": 0.7, "energy": 0.7, "sales": 0.7, "web": 0.7})
    res = evaluate_round(king, chal, cfg.koth_params(), seed=3, king_tenure_rounds=0)
    assert res.per_domain is not None and set(res.per_domain) == set(DOMS)
    # and the how-often half is still there beside the by-how-much half
    assert set(res.per_domain_win_rate) == set(DOMS)
    plain = evaluate_round(_scores(1.0, 0, [str(i) for i in range(240)]),
                           _scores(0.7, 0, [str(i) for i in range(240)]),
                           cfg.koth_params(), seed=3, king_tenure_rounds=0)
    assert plain.per_domain is None


def test_receipt_carries_per_domain_and_drops_it_when_absent(cfg):
    from cascade.shared.receipt import (
        VerdictRecord,
        _verdict_body,
        dump_receipt,
        load_receipt,
        summarize_receipt,
    )
    from tests.unit.receipt_fixture import make_scored_receipt

    receipt, _, _ = make_scored_receipt(cfg)
    v = receipt.verdict
    body = _verdict_body(v)
    assert "per_domain" not in body and "cohort_per_domain" not in body   # legacy: dropped

    pd = {"nature": {"king": 0.5, "chal": 0.45, "win_rate": 0.8, "n": 60},
          "web": {"king": 0.9, "chal": 0.91, "win_rate": 0.4, "n": 60}}
    rich = replace(v, per_domain=pd, cohort_per_domain={"a": pd})
    body = _verdict_body(rich)
    assert body["per_domain"] == pd and body["cohort_per_domain"] == {"a": pd}
    again = load_receipt(dump_receipt(replace(receipt, verdict=rich)))
    assert again.verdict.per_domain == pd
    assert again.verdict.cohort_per_domain == {"a": pd}
    assert _verdict_body(again.verdict) == _verdict_body(rich)

    # from_round: NaN-scrubbed, sorted by domain, cohort kwarg lands
    class _T:
        dethroned = False
        note = ""
        state = type("S", (), {"king_hotkey": "k", "king_uid": 0})()

    class _R:
        lcb = 0.01
        margin = 0.02
        challenger_wins_round = False
        inconclusive = False
        n_windows = 120
        king_geomean = 0.5
        chal_geomean = 0.49
        gift_lcb = None
        gift_gate_passed = None
        per_domain = {"web": {"king": 0.5, "chal": float("nan"), "win_rate": 0.5, "n": 40},
                      "nature": {"king": 0.4, "chal": 0.39, "win_rate": 0.6, "n": 80}}

    rec = VerdictRecord.from_round(_R(), _T(), params=cfg.koth_params(), bootstrap_seed=1,
                                   cohort_per_domain={"a": _R.per_domain, "b": None})
    assert list(rec.per_domain) == ["nature", "web"]
    assert rec.per_domain["web"] == {"king": 0.5, "chal": None, "win_rate": 0.5, "n": 40}
    assert set(rec.cohort_per_domain) == {"a"}            # the empty one is dropped

    entry = summarize_receipt(replace(receipt, verdict=rich))
    assert entry["per_domain"] == pd and entry["cohort_per_domain"] == {"a": pd}
    assert summarize_receipt(receipt)["per_domain"] is None


def test_validator_records_cohort_per_domain(cfg):
    from cascade.shared.manifest import (
        TrainedEntry,
        TrainingManifest,
        contract_digest,
        format_trained_pointer,
    )
    from cascade.validator.loop import ValidatorRunner
    from cascade.validator.state import genesis

    cid = "alice/gen@sha256:" + "a" * 64
    ptr = format_trained_pointer("cascade/ckpt@sha256:" + "b" * 64)
    king = _domained(_scores(1.0, 0, [str(i) for i in range(240)]))
    by_hk = {"chal_a": _rescaled(king, {"nature": 0.7, "energy": 0.7, "sales": 0.7, "web": 0.7}),
             "chal_b": _rescaled(king, {"nature": 0.9, "energy": 1.2, "sales": 1.1, "web": 1.1})}

    def fake_eval(entry, windows):
        return king if entry.role == "king" else by_hk[entry.miner_hotkey]

    m = TrainingManifest(
        round_id="1", created_block=10, contract_digest=contract_digest(cfg.training),
        base_arch_digest=cfg.training.base_arch_digest, eval_dataset=cfg.eval.eval_dataset,
        entries=[TrainedEntry("king_hk", 0, "king", cid, ptr, "d", 10),
                 TrainedEntry("chal_a", 1, "challenger", cid, ptr, "d", 10),
                 TrainedEntry("chal_b", 2, "challenger", cid, ptr, "d", 10)])
    armed = replace(cfg, round=replace(cfg.round, max_finalists=3))
    runner = ValidatorRunner(cfg=armed, state=genesis("king_hk", 0), evaluate_fn=fake_eval,
                             verify_signatures=False)
    outcome = runner.process_round(m, windows=[], base_seed=7)
    assert outcome is not None
    # EVERY judged challenger gets a breakdown — the loser too, which is the
    # one who needs to know where they lost.
    assert set(outcome.cohort_per_domain) == {"chal_a", "chal_b"}
    b = outcome.cohort_per_domain["chal_b"]
    assert b["nature"]["chal"] < b["nature"]["king"]          # beat the king on nature …
    assert b["energy"]["chal"] > b["energy"]["king"]          # … lost energy
    receipt = runner.build_round_receipt(
        m, base_seed=7, epoch_start_block=10, epoch_block_hash="0x" + "ab" * 32,
        outcome=outcome, windows=[])
    assert set(receipt.verdict.cohort_per_domain) == {"chal_a", "chal_b"}
    assert receipt.verdict.per_domain is not None


# ── receipt: round-trip, canonical drop, index ───────────────────────────────


def test_receipt_carries_ladder_fields_and_drops_them_when_absent(cfg):
    from cascade.shared.receipt import VerdictRecord, _verdict_body
    from tests.unit.receipt_fixture import make_scored_receipt

    receipt, _, _ = make_scored_receipt(cfg)
    v = receipt.verdict
    body = _verdict_body(v)
    assert "per_horizon" not in body and "cohort_geomeans" not in body   # legacy receipt: dropped

    ph = {"64": {"king": 0.5, "chal": 0.45, "win_rate": 0.8, "n": 10},
          "720": {"king": 0.9, "chal": 0.91, "win_rate": 0.4, "n": 10}}
    rich = replace(v, per_horizon=ph, cohort_geomeans={"a": 0.45, "b": 0.5},
                   cohort_per_horizon={"a": ph})
    body = _verdict_body(rich)
    assert body["per_horizon"] == ph and body["cohort_geomeans"] == {"a": 0.45, "b": 0.5}

    from cascade.shared.receipt import dump_receipt, load_receipt

    again = load_receipt(dump_receipt(replace(receipt, verdict=rich)))
    assert again.verdict.per_horizon == ph
    assert again.verdict.cohort_geomeans == {"a": 0.45, "b": 0.5}
    assert again.verdict.cohort_per_horizon == {"a": ph}
    assert _verdict_body(again.verdict) == _verdict_body(rich)

    # from_round: the result's breakdown and the cohort kwargs land, NaN-scrubbed
    class _T:
        dethroned = False
        note = ""
        state = type("S", (), {"king_hotkey": "k", "king_uid": 0})()

    class _R:
        lcb = 0.01
        margin = 0.02
        challenger_wins_round = False
        inconclusive = False
        n_windows = 120
        king_geomean = 0.5
        chal_geomean = 0.49
        gift_lcb = None
        gift_gate_passed = None
        per_horizon = {"64": {"king": 0.5, "chal": float("nan"), "win_rate": 0.5, "n": 40}}

    rec = VerdictRecord.from_round(_R(), _T(), params=cfg.koth_params(), bootstrap_seed=1,
                                   cohort_geomeans={"a": 0.49}, cohort_per_horizon={"a": _R.per_horizon})
    assert rec.per_horizon == {"64": {"king": 0.5, "chal": None, "win_rate": 0.5, "n": 40}}
    assert rec.cohort_geomeans == {"a": 0.49}

    from cascade.shared.receipt import summarize_receipt

    entry = summarize_receipt(replace(receipt, verdict=rich))
    assert entry["per_horizon"] == ph and entry["cohort_geomeans"] == {"a": 0.45, "b": 0.5}
    assert summarize_receipt(receipt)["per_horizon"] is None


def test_validator_records_cohort_and_per_horizon(cfg):
    from cascade.shared.manifest import (
        TrainedEntry,
        TrainingManifest,
        contract_digest,
        format_trained_pointer,
    )
    from cascade.validator.loop import ValidatorRunner
    from cascade.validator.state import genesis

    cid = "alice/gen@sha256:" + "a" * 64
    ptr = format_trained_pointer("cascade/ckpt@sha256:" + "b" * 64)
    king = _scores(1.0, 0, LADDER_IDS)
    by_hk = {"chal_a": [WindowScore(s.series_id, s.mase * 0.7, s.qloss_per_q * 0.7, s.abs_target)
                        for s in king],
             "chal_b": [WindowScore(s.series_id, s.mase * 1.1, s.qloss_per_q * 1.1, s.abs_target)
                        for s in king]}

    def fake_eval(entry, windows):
        return king if entry.role == "king" else by_hk[entry.miner_hotkey]

    m = TrainingManifest(
        round_id="1", created_block=10, contract_digest=contract_digest(cfg.training),
        base_arch_digest=cfg.training.base_arch_digest, eval_dataset=cfg.eval.eval_dataset,
        entries=[TrainedEntry("king_hk", 0, "king", cid, ptr, "d", 10),
                 TrainedEntry("chal_a", 1, "challenger", cid, ptr, "d", 10),
                 TrainedEntry("chal_b", 2, "challenger", cid, ptr, "d", 10)])
    armed = replace(cfg, round=replace(cfg.round, max_finalists=3))
    runner = ValidatorRunner(cfg=armed, state=genesis("king_hk", 0), evaluate_fn=fake_eval,
                             verify_signatures=False)
    outcome = runner.process_round(m, windows=[], base_seed=7)
    assert outcome is not None
    assert set(outcome.cohort_geomeans) == {"chal_a", "chal_b"}
    assert outcome.cohort_geomeans["chal_a"] < outcome.cohort_geomeans["chal_b"]
    assert set(outcome.cohort_per_horizon["chal_a"]) == {"64", "256", "720"}
    assert outcome.result.per_horizon is not None


# ── trainer standings + dashboard ────────────────────────────────────────────


def test_heat_status_marks_duel_only_and_counts_seated(cfg):
    from cascade.shared.heat_status import build_heat_status, heat_summary
    from cascade.shared.manifest import HeatEntrant, HeatResult

    heat = HeatResult(screen_size="", finalists=2, entrants=(
        HeatEntrant(uid=5, hotkey="hk5", gen_ref="a/b@sha256:" + "a" * 64, status="seated",
                    rank=1, rel_score=None, p_best=None, crps=None, mase=None),
        HeatEntrant(uid=9, hotkey="hk9", gen_ref="c/d@sha256:" + "c" * 64, status="waiting",
                    rank=None, rel_score=None, p_best=None, crps=None, mase=None)))
    doc = build_heat_status(heat, round_id="1", epoch_start_block=3600, as_of="now",
                            screened=1, no_screen_reason="duel-only round", duel_only=True)
    assert doc["duel_only"] is True and doc["no_screen_reason"] == "duel-only round"
    assert [e["status"] for e in doc["entrants"]] == ["seated", "waiting"]
    s = heat_summary(doc)
    assert s["duel_only"] is True and s["n_advanced"] == 1 and s["n_entrants"] == 2
    # a legacy doc is unchanged
    legacy = build_heat_status(None, round_id="2", epoch_start_block=7200, as_of="now",
                               no_screen_reason="fit the slots")
    assert "duel_only" not in legacy and legacy["no_screen"] is True


def test_render_heat_lists_seated_and_waiting():
    from cascade.miner.dashboard import heat_block, heat_headline, render_heat

    doc = {"schema": 1, "round_id": "9", "epoch_start_block": 3600, "as_of": "now",
           "screened": 2, "screen_size": "", "finalists": 2, "duel_only": True,
           "no_screen_reason": "duel-only round: 2 seated", "entrants": [
               {"uid": 7, "hotkey": "hk-g", "gen_ref": "g/g@sha256:" + "0" * 64,
                "status": "seated", "rank": 1},
               {"uid": 3, "hotkey": "hk-c", "gen_ref": "c/c@sha256:" + "1" * 64,
                "status": "seated", "rank": 2},
               {"uid": 8, "hotkey": "hk-h", "gen_ref": "h/h@sha256:" + "2" * 64,
                "status": "waiting"}],
           "leader_lcb": None, "n_windows": None, "n_clusters": None}
    assert heat_headline(doc).startswith("duel-only round — 2 seated")
    assert "1 waiting" in heat_headline(doc)
    lines = heat_block(doc, me="hk-h")
    assert any("seated — duels the king" in ln and "uid    7" in ln for ln in lines)
    assert any("waiting — next round" in ln and "← you" in ln for ln in lines)
    text = render_heat(doc)
    assert "duel-only" in text and "advancing" not in text


def test_render_duel_shows_horizons_and_the_whole_cohort():
    from cascade.miner.dashboard import render_duel
    from tests.unit.test_miner_dashboard import _duel_row

    ph = {"64": {"king": 0.5, "chal": 0.48, "win_rate": 0.62, "n": 666},
          "256": {"king": 0.36, "chal": 0.355, "win_rate": 0.55, "n": 666},
          "720": {"king": 0.49, "chal": 0.495, "win_rate": 0.47, "n": 666}}
    row = _duel_row("12", per_horizon=ph, cohort_k=3,
                    cohort_lcbs={"5Chal" + "c" * 43: 0.022, "5Other" + "o" * 42: -0.01,
                                 "5Third" + "t" * 42: -0.03},
                    cohort_geomeans={"5Chal" + "c" * 43: 0.23881, "5Other" + "o" * 42: 0.25,
                                     "5Third" + "t" * 42: 0.26},
                    cohort_per_horizon={"5Other" + "o" * 42: ph})
    text = render_duel([row])
    assert "by horizon" in text
    assert "h=64" in text and "(-4.00%)" in text and "win 62%" in text
    assert "h=720" in text and "(+1.02%)" in text
    assert "cohort         3 challenger(s) judged at α/3" in text
    lines = text.splitlines()
    first = next(ln for ln in lines if ln.strip().startswith("#1"))
    assert "5Chal" in first and "✓ cleared" in first and "← took the throne" in first
    second = next(ln for ln in lines if ln.strip().startswith("#2"))
    assert "✗" in second and "[h64 -4.0% · h256 -1.4% · h720 +1.0%]" in second
    # a legacy row renders exactly as before: no horizon / cohort blocks
    legacy = render_duel([_duel_row("10")])
    assert "by horizon" not in legacy and "cohort  " not in legacy


_PD_CHAL = {"nature": {"king": 0.50, "chal": 0.47, "win_rate": 0.66, "n": 300},
            "energy": {"king": 0.40, "chal": 0.41, "win_rate": 0.45, "n": 300},
            "sales": {"king": 0.30, "chal": 0.33, "win_rate": 0.30, "n": 12}}
_PD_OTHER = {"nature": {"king": 0.50, "chal": 0.52, "win_rate": 0.40, "n": 300},
             "energy": {"king": 0.40, "chal": 0.38, "win_rate": 0.60, "n": 300},
             "sales": {"king": 0.30, "chal": 0.30, "win_rate": 0.50, "n": 12}}
_OTHER = "5Other" + "o" * 42


def _domain_row(round_id="12", **extra):
    from tests.unit.test_miner_dashboard import _duel_row

    chal = "5Chal" + "c" * 43
    fields = dict(per_domain=_PD_CHAL, cohort_k=2,
                  cohort_lcbs={chal: 0.022, _OTHER: -0.01},
                  cohort_geomeans={chal: 0.23881, _OTHER: 0.25},
                  cohort_per_domain={chal: _PD_CHAL, _OTHER: _PD_OTHER})
    fields.update(extra)
    return _duel_row(round_id, **fields)


def test_render_duel_shows_domain_scores_for_the_decided_pair_and_the_cohort():
    from cascade.miner.dashboard import render_duel

    text = render_duel([_domain_row()])
    assert "by domain" in text
    lines = text.splitlines()
    dom = [ln for ln in lines if ln.strip().startswith(("nature", "energy", "sales"))]
    # strongest domain for the challenger first, magnitude beside the win rate
    assert dom[0].strip().startswith("nature") and "(-6.00%)" in dom[0] and "win 66%" in dom[0]
    assert "(+2.50%)" in dom[1]
    assert "(+10.00%)" in dom[2] and "few windows" in dom[2]      # n=12 flagged as noise
    # the win-rate-only block is superseded once scores are present
    assert "per-domain win rate" not in text
    # every cohort member gets a by-domain continuation line
    second = next(i for i, ln in enumerate(lines) if ln.strip().startswith("#2"))
    assert "by domain: energy -5.0% · sales +0.0% · nature +4.0%" in lines[second + 1]
    # a legacy row (win rates only) renders exactly as before
    from tests.unit.test_miner_dashboard import _duel_row
    legacy = render_duel([_duel_row("10")])
    assert "per-domain win rate" in legacy and "by domain" not in legacy


def test_render_duel_hotkey_adds_your_domains_block():
    from cascade.miner.dashboard import my_domain_lines, render_duel

    row = _domain_row()
    # the loser of the cohort: their own table, not the decided pair's
    text = render_duel([row], me=_OTHER)
    assert "your domains   5Other…oooo vs the king — 1 better, 2 worse" in text
    mine = text[text.index("your domains"):].splitlines()
    assert mine[1].strip().startswith("energy") and "you  0.38000" in mine[1] and "(-5.00%)" in mine[1]
    # the decided challenger on a k=1 receipt (no cohort_per_domain) falls back
    # to the headline per_domain — the usual rolling settlement
    solo = _domain_row(cohort_per_domain=None, cohort_geomeans=None, cohort_lcbs=None, cohort_k=None)
    text = render_duel([solo], me="5Chal" + "c" * 43)
    assert "1 better, 2 worse" in text and "nature" in text[text.index("your domains"):]
    # the king, a bystander, and a judged miner on a pre-field receipt
    assert "was the king this round" in my_domain_lines(row, "5King" + "k" * 43)[0]
    assert "was not judged this round" in my_domain_lines(row, "5Nobody" + "n" * 41)[0]
    old = _domain_row(per_domain=None, cohort_per_domain=None)
    assert "carries no per-domain scores" in my_domain_lines(old, _OTHER)[0]


def test_render_duel_domain_history_is_the_per_miner_trend():
    from cascade.miner.dashboard import render_duel_domain_history

    later = {"nature": {"king": 0.50, "chal": 0.46, "win_rate": 0.7, "n": 300},
             "energy": {"king": 0.40, "chal": 0.39, "win_rate": 0.55, "n": 300}}
    doc = {"rounds": [
        _domain_row("8", epoch_start_block=8_802_000, dethroned=False),
        _domain_row("10", epoch_start_block=8_809_200, cohort_per_domain={_OTHER: later},
                    cohort_geomeans={_OTHER: 0.24}),
        # a round the miner sat out is skipped, not shown as dashes
        _domain_row("11", epoch_start_block=8_812_800, cohort_per_domain={}, cohort_geomeans={}),
    ]}
    text = render_duel_domain_history(doc, _OTHER)
    assert "5Other…oooo by domain over 2 judged round(s)" in text
    lines = text.splitlines()
    assert lines[1].split() == ["round", "overall", "energy", "nature", "sales"]
    r8 = next(ln for ln in lines if ln.strip().startswith("8 "))
    r10 = next(ln for ln in lines if ln.strip().startswith("10 "))
    assert "+0.66%" in r8 and "-5.00%" in r8 and "+4.00%" in r8      # overall, energy, nature
    assert "-3.37%" in r10 and "-2.50%" in r10 and "-8.00%" in r10 and r10.rstrip().endswith("--")
    avg = next(ln for ln in lines if ln.strip().startswith("average"))
    assert "-3.75%" in avg and "-2.00%" in avg                      # energy, nature means
    assert "better than the king on average: energy, nature  ·  worse: sales" in text
    # nobody / no data
    assert "no per-domain scores for 5Nob" in render_duel_domain_history(doc, "5Nob" + "x" * 44)
    assert "pass --hotkey" in render_duel_domain_history(doc, "")


def test_trainer_publishes_seated_and_waiting_entrants(cfg, tmp_path, monkeypatch):
    from tests.unit.test_duel_only_rounds import _field, _runner, _screen_must_not_run, _Store

    duel_cfg = replace(cfg, round=replace(cfg.round, max_finalists=1, finalists=1,
                                          duel_from_block=1000, duel_field_cap=2,
                                          epoch_blocks_prev=0, epoch_activation_block=0))
    store = _Store()
    runner = _runner(duel_cfg, tmp_path, monkeypatch, publish_stage_status=True,
                     screen_fn=_screen_must_not_run)
    runner._manifest_store = store
    runner.run_round(_field(), king_hotkey="a", base_seed=1, block=5000)
    doc = json.loads(store.objects["status/heat.json"])
    assert doc["duel_only"] is True
    assert [(e["uid"], e["status"], e.get("rank")) for e in doc["entrants"]] == [
        (2, "seated", 1), (1, "seated", 2), (3, "waiting", None)]
    assert doc["finalists"] == 2
    idx = json.loads(store.objects["heats/index.json"])
    assert idx["heats"][0]["duel_only"] is True and idx["heats"][0]["n_advanced"] == 2
    with pytest.raises(KeyError):
        _ = doc["no_screen"]        # a duel-only doc is not a "no screen" doc
