"""MarketCast (lane C) gain-scaled pay duration (owner 2026-11-24, DEC-CA-0073).

A confirmed win pays its slot in FULL for
``5 + 15 * clamp((lcb - margin) / (0.05 - margin), 0, 1)`` days, rounded UP
to eras on the grid in force at the confirming settlement, then the slot ENDS
(burn) — no decay, no TTL. The gain-tiered keys are retired. The audit
re-derives the duration and rejects a forged one or a payment past it."""

from __future__ import annotations

import json
import math
import tomllib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from cascade.shared.config import RETIRED_KEYS, load_chain_config, parse_lane_c
from cascade.shared.config.marketcast import parse_marketcast
from cascade.shared.era import era_length_blocks
from cascade.shared.lane_c import (
    LaneCState,
    TargetWin,
    duration_full_pay,
    full_pay_record_mismatch,
    lane_c_pay_record,
    lane_c_payouts,
    target_full_pay_record,
    win_expired,
)

ROOT = Path(__file__).resolve().parents[2]
ERA = 7200
T = "mkt:returns"
MC = parse_marketcast({"from_block": 2 * ERA})
RETIRED_LC = ("full_pay_tiers", "half_life_rounds", "floor", "win_ttl_eras")


def _lc(**kw):
    raw = {"targets": [T], "from_block": 2 * ERA, **kw}
    return parse_lane_c(raw, v2_from_block=ERA, era_blocks=ERA, era_settlements=4,
                        lane_b_share=0.25, kind_from_blocks={"mkt": MC.from_block})


def _dur(lcb, era_blocks=3600, margin=0.01):
    return duration_full_pay(lcb, margin=margin, min_days=5, max_days=20, max_gain=0.05,
                             era_blocks=era_blocks)


# ── the rule ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("lcb,eras", [
    (0.01, 10),                           # the margin: 5 days = 10 twelve-hour eras
    (0.02, 18),                           # 8.75 days = 17.5 eras -> rounded up
    (0.03, 25),                           # 12.5 days
    (0.05, 40),                           # 20 days
    (0.30, 40),                           # capped at max_gain
    (math.nextafter(0.05, 0), 40),
])
def test_duration_scales_linearly_with_the_gain(lcb, eras):
    r = _dur(lcb)
    assert r == {"gain_lcb": lcb, "tier": min(max(lcb, 0.01), 0.05), "full_eras": eras,
                 "below_floor": False}


def test_duration_follows_the_era_length_in_force():
    # 3 h settlements x 4 = 12 h eras (3600 blocks); 24 h eras halve the count
    assert _dur(0.01, era_blocks=7200)["full_eras"] == 5
    assert _dur(0.05, era_blocks=7200)["full_eras"] == 20
    # a grid that does not divide the duration rounds UP (never under 5 days)
    n = _dur(0.01, era_blocks=2700)["full_eras"]
    assert n * 2700 * 12 >= 5 * 86400 > (n - 1) * 2700 * 12


def test_below_margin_or_non_finite_pays_the_minimum_flagged():
    for lcb in (0.005, None, float("nan")):
        r = _dur(lcb)
        assert r["full_eras"] == 10 and r["below_floor"] is True and r["tier"] == 0.01


def test_duration_is_monotone_in_the_gain():
    ns = [_dur(0.01 + i * 0.0005)["full_eras"] for i in range(90)]
    assert ns == sorted(ns) and ns[0] == 10 and ns[-1] == 40


def test_mkt_ramp_starts_at_the_marketcast_margin():
    lc = _lc()
    mc = replace(MC, gain_margin=0.02)
    assert target_full_pay_record(T, 0.02, lc, mc, era_blocks=3600)["full_eras"] == 10
    assert target_full_pay_record(T, 0.02, lc, MC, era_blocks=3600)["full_eras"] == 18
    with pytest.raises(ValueError, match="marketcast"):
        target_full_pay_record(T, 0.02, lc, None, era_blocks=3600)


# ── config ───────────────────────────────────────────────────────────────────


def test_shipped_configs_carry_the_duration_terms_and_no_retired_keys():
    for f in ("chain.toml", "chain.testnet.toml"):
        raw = tomllib.loads((ROOT / f).read_text())
        assert not set(raw["lane_c"]) & set(RETIRED_LC), f
        assert "full_pay_tiers" not in raw["marketcast"], f
        lc = load_chain_config(ROOT / f).lane_c
        assert (lc.full_pay_min_days, lc.full_pay_max_days, lc.full_pay_max_gain) == (
            5.0, 20.0, 0.05)
    # the lane C activation the owner typed is untouched
    assert load_chain_config(ROOT / "chain.toml").lane_c.from_block == 9579600


def test_old_keys_are_retired_ignored_at_load_and_refused_in_overlays():
    from cascade.shared.netconfig import NetConfigError, check_overrides

    assert set(RETIRED_LC) <= RETIRED_KEYS["lane_c"]
    assert "full_pay_tiers" in RETIRED_KEYS["marketcast"]
    raw = tomllib.loads((ROOT / "chain.toml").read_text())
    raw["lane_c"].update(full_pay_tiers=[[0.05, 99]], win_ttl_eras=1, floor=0.5,
                         half_life_rounds=9.0)
    raw["marketcast"]["full_pay_tiers"] = [[0.05, 99]]
    c = load_chain_config(raw=raw)                      # loads, ignored
    assert not hasattr(c.lane_c, "win_ttl_eras") and not hasattr(c.marketcast,
                                                                  "full_pay_tiers")
    for table, key in [("lane_c", k) for k in RETIRED_LC] + [("marketcast", "full_pay_tiers")]:
        with pytest.raises(NetConfigError, match="retired"):
            check_overrides({table: {key: 1}})
    check_overrides({"lane_c": {"full_pay_max_days": 30.0}})   # the new keys are overlayable


@pytest.mark.parametrize("kw,msg", [
    ({"full_pay_min_days": 21.0}, "full_pay_min_days"),
    ({"full_pay_min_days": 0}, "full_pay_min_days"),
    ({"full_pay_max_gain": 1.0}, "full_pay_max_gain"),
    ({"full_pay_max_gain": 0.01}, "gain_margin"),             # == [lane_c] gain_margin
    ({"full_pay_max_days": "20"}, "full_pay_max_days"),
    ({"full_pay_max_days": float("inf")}, "finite"),
])
def test_loader_checks(kw, msg):
    with pytest.raises(ValueError, match=msg):
        _lc(**kw)


def test_marketcast_margin_must_sit_below_max_gain():
    raw = tomllib.loads((ROOT / "chain.toml").read_text())
    raw["marketcast"]["gain_margin"] = 0.05
    with pytest.raises(ValueError, match=r"\[marketcast\] gain_margin"):
        load_chain_config(raw=raw)


# ── judging and the win record ───────────────────────────────────────────────


def test_marketcast_judgement_records_the_duration(monkeypatch):
    from cascade.eval.marketcast import SuiteVerdict
    from cascade.validator import marketcast as vmc

    monkeypatch.setattr(vmc, "verdict_from_rows", lambda *a, **k: SuiteVerdict(
        "returns", True, "win", 300, 9, gain=0.04, gain_lcb=0.03))
    monkeypatch.setattr(vmc, "base_cov_gain", lambda rows: None)
    rows = SimpleNamespace(status="ok", after=1.0, fault="", windows=[], near_dup=None)
    entry = SimpleNamespace(target=T, miner_hotkey="h", miner_uid=3, trained_pointer="t",
                            recipe_ref="r")
    j = vmc.judge_marketcast_entry(entry, rows, mc=MC, k=1, phase="confirm", seed="s",
                                   round_block=10, origin_block=10, lc=_lc(), era_blocks=3600)
    assert j.full_pay == _dur(0.03)


def test_confirmed_win_carries_the_duration_and_round_trips():
    from cascade.validator.lane_c import LaneCJudgement, _confirmed_win

    entry = SimpleNamespace(target=T, miner_hotkey="h", miner_uid=3, trained_pointer="t",
                            recipe_ref="r")
    fp = target_full_pay_record(T, 0.05, _lc(), MC, era_blocks=ERA)
    j = LaneCJudgement(entry, True, "verdict", "win", gain_lcb=0.05, phase="confirm",
                       full_pay=fp)
    w = _confirmed_win(j, round_id="r", settle_block=4 * ERA, era_index=4)
    assert (w.full_eras, w.tier, w.below_floor) == (20, 0.05, False)    # 20 days / 1-day eras
    st = LaneCState(slots={T: w})
    assert LaneCState.from_json(st.to_json()).slots[T] == w


# ── payouts: full, then the slot ends ────────────────────────────────────────


def _win(block, full_eras, lcb=0.03):
    return TargetWin(target=T, hotkey="alice", uid=4, round_id=f"r{block}", win_block=block,
                     base_pointer=MC.base_pointer, gain_lcb=lcb, tier=lcb, full_eras=full_eras)


def test_win_pays_full_then_ends_with_no_decay_or_ttl():
    lc = _lc()
    w = _win(4 * ERA, 40)                  # 40 eras: past the retired 28-era TTL
    st = LaneCState(slots={T: w})

    def mult(age):
        rows = lane_c_payouts(st, block=4 * ERA + age * ERA, era_blocks=ERA, cfg=lc,
                              n_uids=8, targets=(T,))
        return rows[0]["multiplier"] if rows else None

    assert [mult(a) for a in (0, 1, 27, 28, 39)] == [1.0] * 5
    assert mult(40) is None and mult(41) is None          # ended: the slot burns
    assert win_expired(40, w) and not win_expired(39, w)
    with pytest.raises(ValueError, match="full_eras"):
        win_expired(0, replace(w, full_eras=None))


def test_an_ended_holder_is_evicted_by_the_marketcast_fold():
    """The fold's expiry eviction reads the holder's own duration."""
    import inspect

    from cascade.validator import marketcast as vmc

    src = inspect.getsource(vmc)
    assert "era_length_blocks(self.cfg.round, block)),\n                                  holder)" in src


# ── audit ────────────────────────────────────────────────────────────────────


def test_mismatch_rejects_a_forged_duration():
    lc = _lc()
    good = _dur(0.03, era_blocks=ERA, margin=MC.gain_margin)
    assert full_pay_record_mismatch(good, T, lc, MC, era_blocks=ERA) is None
    assert "full_eras" in full_pay_record_mismatch(dict(good, full_eras=20), T, lc, MC,
                                                   era_blocks=ERA)
    assert full_pay_record_mismatch(dict(good, gain_lcb=0.05), T, lc, MC, era_blocks=ERA)
    assert "below_floor" in full_pay_record_mismatch(dict(good, below_floor=True), T, lc, MC,
                                                     era_blocks=ERA)
    assert "carries no" in full_pay_record_mismatch({"gain_lcb": 0.03}, T, lc, MC,
                                                    era_blocks=ERA)
    # the gain-tiered record (tier 0.02 -> 5 eras) is no longer valid
    assert full_pay_record_mismatch({"gain_lcb": 0.03, "tier": 0.02, "full_eras": 5,
                                     "below_floor": False}, T, lc, MC, era_blocks=ERA)


def test_audit_replays_payouts_and_the_end_of_the_slot(cfg):
    from cascade.audit.checks import _lane_c_paid_wins

    lc = _lc()
    c = replace(cfg, lane_c=lc, marketcast=MC)
    win_block = 4 * ERA
    eb = int(era_length_blocks(c.round, win_block))
    fp = target_full_pay_record(T, 0.03, lc, MC, era_blocks=eb)
    w = replace(_win(win_block, fp["full_eras"]), tier=fp["tier"])
    st = LaneCState(slots={T: w})

    def why(block, edit=None):
        rec = json.loads(json.dumps(lane_c_pay_record(st, block=block, era_blocks=eb,
                                                      cfg=lc, n_uids=8, mc=MC)))
        if edit:
            rec = edit(rec)
        return _lane_c_paid_wins(SimpleNamespace(epoch_start_block=block, lane_c_pay=rec),
                                 c, 8)[1]

    def win0(**kw):
        return lambda r: {**r, "wins": [dict(r["wins"][0], **kw)]}

    assert why(win_block) is None                              # re-derived at the win
    last = win_block + (fp["full_eras"] - 1) * eb
    assert why(last) is None
    # forged duration at the win
    assert "full_eras" in why(win_block, win0(full_eras=40))
    # a decayed multiplier
    assert "multiplier" in why(win_block + eb, win0(multiplier=0.5))
    # a record still paying the slot after its duration ended
    end = win_block + fp["full_eras"] * eb
    rec_last = json.loads(json.dumps(lane_c_pay_record(st, block=last, era_blocks=eb,
                                                       cfg=lc, n_uids=8, mc=MC)))
    rec_end = json.loads(json.dumps(lane_c_pay_record(st, block=end, era_blocks=eb,
                                                      cfg=lc, n_uids=8, mc=MC)))
    assert rec_end["wins"] == []                               # the slot ended
    forged = {**rec_end, "wins": [dict(rec_last["wins"][0], age=fp["full_eras"])]}
    assert "slot has ended" in _lane_c_paid_wins(
        SimpleNamespace(epoch_start_block=end, lane_c_pay=forged), c, 8)[1]
