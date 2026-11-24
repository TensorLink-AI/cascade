"""DEC-CA-0073 (owner 2026-11-24): a MarketCast verdict within 13 hours —
every horizon judged on windows DRAWN from the embargo by the round block's
hash, the generator scanned for embargoed prices before a leg rents, and the
draw recorded, re-derived and audited."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest

from cascade.pool import marketcast as P
from cascade.shared.config import load_chain_config
from cascade.shared.config.marketcast import EMBARGO_RULE, VERDICT_WAIT_MAX_BLOCKS
from cascade.shared.marketcast_report import MktEntryRows, MktWindowRow, kind_scores
from cascade.validator.marketcast import (
    _draw_mismatch,
    embargo_scan_problem,
    entry_draw,
    mkt_prechecks,
)

DAY = 86400
H = "0x" + "5a" * 32
MC = load_chain_config("chain.toml").marketcast


def _entry(tt, scan=None):
    return SimpleNamespace(marketcast={"train_time": float(tt),
                                       **({"embargo_scan": scan} if scan is not None else {})})


# ── the draw ────────────────────────────────────────────────────────────────


def test_shipped_rule_is_a_13h_verdict_on_the_embargo():
    for toml in ("chain.toml", "chain.testnet.toml"):
        mc = load_chain_config(toml).marketcast
        assert mc.wait_blocks == VERDICT_WAIT_MAX_BLOCKS == 3900          # 13 h of 12 s blocks
        assert (mc.embargo_eval_days, mc.backtest_windows_per_cell) == (28, 48)
        assert mc.embargo_eval_days < mc.embargo_days
        assert mc.forward_horizons == ("1mx60", "5mx36")
    from cascade.shared import netconfig as N

    # the eval-draw knobs are overlay-settable (code-bounded by the loader)
    assert {"embargo_eval_days", "backtest_windows_per_cell",
            "verdict_wait_blocks"} <= N.ALLOWED["marketcast"]
    assert "embargo_days" not in N.ALLOWED["marketcast"]          # a raise scores trained days


def test_draw_spans_lie_in_the_embargo_no_leg_reads():
    tt = datetime(2026, 12, 1, 7, 30, tzinfo=UTC).timestamp()
    d = P.embargo_draw(MC, train_time=tt, phase="judge", block=9_600_000, block_hash=H,
                       after=tt + MC.start_margin_seconds)
    assert d["rule"] == EMBARGO_RULE and d["half"] == 0 and d["block"] == 9_600_000
    assert d["hi"] == tt and d["lo"] == tt - 28 * DAY
    assert d["forward_hi"] == tt + 3900 * 12 and d["forward_horizons"] == ["1mx60", "5mx36"]
    # the latest data.end a recipe committed at the dispatch may use is the commit
    # day − 30 d, read whole (+1 day): the draw starts after it
    from cascade.shared.marketcast_recipe import parse_marketcast_recipe
    from cascade.trainer.marketcast_leg import training_span

    from .test_marketcast import _recipe

    sub = datetime.fromtimestamp(tt, tz=UTC).date()
    end = (sub - timedelta(days=MC.embargo_days)).isoformat()
    obj = _recipe(target="mkt:market", horizon_mix={"1hx48": 1.0},
                  objective={"suite_weights": {"returns": 1.0}, "coverage_penalty": 0.5},
                  data={"assets": {"crypto": ["BTC"]}, "class_weights": {"crypto": 1.0},
                        "frequencies": ["1m", "1h"], "start": "2026-01-01", "end": end})
    rc = parse_marketcast_recipe(obj, MC, submission_time=tt)
    _t0, t1 = training_span(rc, not_after=tt)
    assert t1 <= d["lo"] < d["hi"]
    with pytest.raises(ValueError, match="embargo"):
        late = {**obj, "data": {**obj["data"],
                                "end": (sub - timedelta(days=MC.embargo_days - 1)).isoformat()}}
        parse_marketcast_recipe(late, MC, submission_time=tt)
    # the longest allowed eval span still never reaches back past the embargo
    lo_max = P.embargo_draw(replace(MC, embargo_eval_days=29), train_time=tt, phase="judge",
                            block=1, block_hash=H, after=tt)["lo"]
    assert lo_max >= P.embargo_span(MC, tt)[0] >= t1 - 1
    early = P.embargo_draw(replace(MC, embargo_eval_days=29), train_time=float(
        int(tt) // DAY * DAY), phase="judge", block=1, block_hash=H, after=tt)
    assert early["lo"] >= P.embargo_span(MC, early["hi"])[0]          # at a midnight too


def test_draw_is_deterministic_hash_ranked_and_halves_are_disjoint():
    ids = [f"crypto:BTC|returns|1mx60|{1_790_000_000 + 3600 * i}" for i in range(2000)]
    halves = [P.window_half(w) for w in ids]
    assert 0.4 < sum(halves) / len(halves) < 0.6
    assert halves == [P.window_half(w) for w in ids]                 # fixed, public
    r1 = sorted(ids, key=lambda w: P.draw_rank(H, w))
    r2 = sorted(ids, key=lambda w: P.draw_rank("0x" + "5b" * 32, w))
    assert r1 == sorted(ids, key=lambda w: P.draw_rank(H, w)) and r1[:48] != r2[:48]
    with pytest.raises(ValueError, match="0x"):
        P.embargo_draw(MC, train_time=1.0, phase="judge", block=1, block_hash="5a" * 32, after=1)
    with pytest.raises(ValueError, match="phase"):
        P.embargo_draw(MC, train_time=1.0, phase="x", block=1, block_hash=H, after=1)


def test_signed_draw_must_be_the_rules():
    tt = 1_795_000_000.0
    e = _entry(tt)
    d = entry_draw(e, MC, phase="judge", block=9_600_000, block_hash=H)
    rows = MktEntryRows("hk", "p", "mkt:market", "judge", tt + 3600, draw=d)
    assert _draw_mismatch(e, rows, MC, phase="judge", block=9_600_000, fresh_after=0.0) == ""
    # another round block, phase, span or count is not the rule's draw
    assert "block" in _draw_mismatch(e, rows, MC, phase="judge", block=9_600_900,
                                     fresh_after=0.0)
    why = _draw_mismatch(e, rows, MC, phase="confirm", block=9_600_000, fresh_after=0.0)
    assert "half" in why or "phase" in why
    for k, v in (("lo", d["lo"] - DAY), ("n", 500), ("forward_hi", d["forward_hi"] + 3600)):
        assert k in _draw_mismatch(e, replace(rows, draw={**d, k: v}), MC, phase="judge",
                                   block=9_600_000, fresh_after=0.0)
    assert "draw record" in _draw_mismatch(e, replace(rows, draw={}), MC, phase="judge",
                                           block=9_600_000, fresh_after=0.0)
    assert "0x" in _draw_mismatch(e, replace(rows, draw={**d, "block_hash": "nope"}), MC,
                                  phase="judge", block=9_600_000, fresh_after=0.0)
    # a confirm's forward windows start after the judge snapshot's end
    dc = entry_draw(e, MC, phase="confirm", block=9_600_900, block_hash=H,
                    fresh_after=tt + 7 * 3600)
    assert dc["half"] == 1 and dc["forward_after"] == tt + 7 * 3600


def test_kind_subscores_split_forward_from_backtest():
    tt = 1_795_000_000
    d = {"hi": float(tt)}

    def w(start, ratio):
        return MktWindowRow(f"crypto:BTC|returns|1mx60|{start}", 1.0, ratio, (30, 48, 59), 60,
                            float("nan"), (3, 3))

    rows = MktEntryRows("hk", "p", "mkt:returns", "judge", tt + 3600, draw=d,
                        windows=tuple([w(tt - 3600 * (i + 1), 0.9) for i in range(30)]
                                      + [w(tt + 3600 * (i + 2), 1.1) for i in range(10)]))
    mc = replace(MC, composite=False, min_windows=5, min_clusters=1, min_time_blocks=1)
    ks = kind_scores(rows, mc, suite="returns", k=1, seed="s")
    assert ks["backtest"]["n"] == 30 and ks["forward"]["n"] == 10
    assert ks["backtest"]["gain"] > 0 > ks["forward"]["gain"]
    assert kind_scores(replace(rows, draw={}), mc, suite="returns", k=1, seed="s") == {}


# ── guard (b): the generator scan ──────────────────────────────────────────


def _snapshot(tmp_path, end):
    """A snapshot whose hourly bank covers 40 days to ``end``."""
    ct0, hours = P.coarse_grid(end, 40)
    t0, minutes = P.snapshot_grid(end, 1)
    rng = np.random.default_rng(3)
    c = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.004, hours)))
    one = np.ones(minutes)
    P.write_snapshot(tmp_path, t0=t0, minutes=minutes,
                     bars={("crypto", "BTC"): {k: one for k in ("open", "high", "low", "close",
                                                                "volume")}},
                     sources={"crypto": "hyperliquid"}, anchor_block=0, anchor_time=float(t0),
                     coarse={("crypto", "BTC"): {"open": c, "high": c, "low": c, "close": c,
                                                 "volume": np.ones(hours)}},
                     coarse_t0=ct0, coarse_steps=hours)
    return P.load_snapshot(tmp_path, verify_signature=False), ct0, c


def _tree(tmp_path, name, body):
    d = tmp_path / name
    d.mkdir()
    (d / "marketcast.json").write_text("{}")
    (d / "generator.py").write_text(body)
    return d


def test_generator_embedding_embargoed_bars_is_refused(tmp_path):
    from cascade.shared.marketcast_embargo import embargo_series, scan_receipt

    end = 1_795_000_000 // 3600 * 3600
    snap, ct0, c = _snapshot(tmp_path / "s", end)
    tt = float(end)
    ser = embargo_series(snap, tt - MC.embargo_eval_days * DAY, tt)
    assert {"crypto/BTC/coarse/close", "crypto/BTC/coarse/ret_bp"} <= set(ser)
    i = (int(tt) - 10 * DAY - ct0) // 3600             # 10 days before dispatch: embargoed
    j = (int(tt) - 35 * DAY - ct0) // 3600             # 35 days before: released history
    clean = _tree(tmp_path, "clean", "import numpy as np\n\ndef generate(n, rng):\n"
                                     "    return rng.normal(size=(n, 64))\n")
    exact = _tree(tmp_path, "exact", f"PRICES = {[float(x) for x in c[i:i + 40]]}\n")
    scaled = _tree(tmp_path, "scaled", "PRICES = "
                   + json.dumps([round(float(x) * 3.7 + 11, 3) for x in c[i:i + 64]]) + "\n")
    rets = _tree(tmp_path, "rets", "R = "
                 + json.dumps([float(x) for x in np.diff(np.log(c[i:i + 50])) * 1e4]) + "\n")
    old = _tree(tmp_path, "old", f"PRICES = {[float(x) for x in c[j:j + 40]]}\n")
    got = {t.name: scan_receipt(t, snap, MC, train_time=tt)
           for t in (clean, exact, scaled, rets, old)}
    assert got["clean"]["result"] == "clean" and got["old"]["result"] == "clean"
    assert got["exact"]["result"] == got["scaled"]["result"] == "embedded"
    assert got["rets"]["result"] == "embedded"
    assert scan_receipt(None, snap, MC, train_time=tt)["result"] == "no_generator"
    r = got["clean"]
    assert (r["rule"], r["snapshot"], r["hi"]) == (EMBARGO_RULE, snap.digest, tt)
    # the trainer refuses an embedded tree before anything is rented
    from cascade.trainer.finetune import RecipeError
    from cascade.trainer.marketcast_runner import embargo_scan_or_refuse

    def scan(tree, *, train_time):
        return scan_receipt(tree, snap, MC, train_time=train_time)

    with pytest.raises(RecipeError, match="embargo_series_embedded"):
        embargo_scan_or_refuse(exact, True, scan, train_time=tt)
    assert dict(embargo_scan_or_refuse(clean, True, scan, train_time=tt))["result"] == "clean"
    # no generator ⇒ nothing to scan, the receipt says so
    assert dict(embargo_scan_or_refuse(exact, False, scan, train_time=tt))["result"] == \
        "no_generator"
    with pytest.raises(RuntimeError, match="train_time"):
        embargo_scan_or_refuse(clean, True, scan, train_time=None)


def test_trainer_never_dispatches_an_unscanned_leg(tmp_path):
    from cascade.trainer.lane_c_competitions import MarketCastTrainer

    ops = SimpleNamespace(r=SimpleNamespace(cfg=SimpleNamespace(marketcast=MC),
                                            work_root=tmp_path))
    with pytest.raises(RuntimeError, match="no embargo scanner"):
        MarketCastTrainer().read_recipe(ops, tmp_path, target="mkt:market",
                                        submission_time=None, train_time=1.0)


def test_validators_judge_only_entries_with_a_passing_scan():
    tt = 1_795_000_000.0
    assert embargo_scan_problem(_entry(tt, {"result": "clean"})) is None
    assert embargo_scan_problem(_entry(tt, {"result": "no_generator"})) is None
    st, why = embargo_scan_problem(_entry(tt, {"result": "embedded", "window_id": "x"}))
    assert st == "fault" and "embargo_series_embedded" in why
    for rec in (None, {"result": "budget_exceeded"}, {}):
        assert embargo_scan_problem(_entry(tt, rec))[0] == "stale"
    # the scan is the last pre-check, after target / base / binding
    lc = SimpleNamespace(targets=("mkt:market",))
    bad_target = SimpleNamespace(target="mkt:nope", marketcast={})
    assert mkt_prechecks(bad_target, lc=lc, mc=MC, commitments=[])[0] == "target"
