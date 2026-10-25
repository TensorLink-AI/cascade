"""Gain-tiered full-pay period for lane B / lane C (owner 2026-10-20).

A lane-B / lane-C win pays its slot in full for the ``full_eras`` of the
highest ``full_pay_tiers`` threshold its gain LCB meets (fixed at the win,
recorded on it), then the unchanged decay / TTL. MarketCast ``mkt:`` wins use
``[marketcast] full_pay_tiers``. The audit re-derives the tier and rejects a
forged one."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from cascade.eval.expert_lane import (
    ExpertVerdict,
    TargetsVerdict,
    full_pay,
    full_pay_mismatch,
    reward_multiplier,
)
from cascade.shared.config import load_chain_config, parse_lane_b, parse_lane_c
from cascade.shared.config.v2 import FULL_PAY_TIERS, parse_full_pay_tiers
from cascade.shared.lane_b import DomainWin, LaneBState, lane_b_payouts
from cascade.shared.lane_c import (
    LaneCState,
    full_pay_tiers_for,
    lane_c_payouts,
    target_full_pay,
)
from cascade.shared.manifest import format_trained_pointer
from cascade.shared.settlement_state import settlement_state

ROOT = Path(__file__).resolve().parents[2]
TIERS = ((0.05, 10), (0.02, 5), (0.01, 2))


def _ptr(tag: str) -> str:
    import hashlib

    return format_trained_pointer("cascade/ckpt@sha256:" + hashlib.sha256(
        tag.encode()).hexdigest())


# ── the rule ─────────────────────────────────────────────────────────────────


def test_defaults_are_the_placeholders():
    assert FULL_PAY_TIERS == TIERS
    for f in ("chain.toml", "chain.testnet.toml"):
        c = load_chain_config(ROOT / f)
        assert c.lane_b.full_pay_tiers == TIERS
        assert c.lane_c.full_pay_tiers == TIERS
        assert c.marketcast.full_pay_tiers == TIERS


@pytest.mark.parametrize("threshold,n,lower", [(0.05, 10, 5), (0.02, 5, 2)])
def test_each_tier_boundary_at_and_just_below(threshold, n, lower):
    at = full_pay(threshold, TIERS)
    assert (at.tier, at.full_eras, at.below_floor) == (threshold, n, False)
    below = full_pay(math.nextafter(threshold, 0.0), TIERS)
    assert below.full_eras == lower and not below.below_floor


def test_lowest_tier_boundary_and_below_floor():
    assert full_pay(0.01, TIERS).full_eras == 2
    under = full_pay(math.nextafter(0.01, 0.0), TIERS)
    assert (under.tier, under.full_eras, under.below_floor) == (0.01, 2, True)
    for bad in (None, float("nan"), float("-inf")):
        fp = full_pay(bad, TIERS)
        assert fp.gain_lcb is None and fp.full_eras == 2 and fp.below_floor
    assert full_pay(0.9, TIERS).full_eras == 10


def test_payout_curve_is_tiered_full_period_then_the_unchanged_decay():
    for n in (10, 5, 2, 0):
        curve = [reward_multiplier(a, full_eras=n, half_life_rounds=1.0, floor=0.1)
                 for a in range(n + 6)]
        assert curve[:n] == [1.0] * n
        assert curve[n:] == [0.5, 0.25, 0.125, 0.1, 0.1, 0.1]   # decay starts after it
    with pytest.raises(ValueError):
        reward_multiplier(0, full_eras=-1)


def test_lane_b_payout_reads_the_recorded_tier_and_stops_at_the_ttl():
    lb = parse_lane_b({"from_block": 1, "domains": ["energy", "web"], "win_ttl_eras": 12})
    st = LaneBState()
    st.record(DomainWin(domain="energy", hotkey="a", uid=3, round_id="r", merge_block=0,
                        target_lcb=0.07).with_full_pay(full_pay(0.07, lb.full_pay_tiers)))
    st.record(DomainWin(domain="web", hotkey="b", uid=4, round_id="r", merge_block=0,
                        target_lcb=0.015).with_full_pay(full_pay(0.015, lb.full_pay_tiers)))

    def pays(age):
        return {p["domain"]: p for p in lane_b_payouts(st, block=age * 10, era_blocks=10,
                                                      cfg=lb, n_uids=8)}

    assert pays(9)["energy"]["multiplier"] == 1.0          # 10 full eras
    assert pays(10)["energy"]["multiplier"] == 0.5
    assert pays(1)["web"]["multiplier"] == 1.0             # 2 full eras
    assert pays(2)["web"]["multiplier"] == 0.5
    p = pays(9)["energy"]
    assert (p["gain_lcb"], p["tier"], p["full_eras"], p["below_floor"]) == (0.07, 0.05, 10,
                                                                             False)
    assert pays(12) == {}                                  # TTL unchanged: both expired
    # an untiered win is refused, never paid on a default
    st.wins["web"] = replace(st.wins["web"], full_eras=None)
    with pytest.raises(ValueError, match="full-pay tier"):
        pays(1)


def test_the_tier_is_fixed_at_the_win():
    lb = parse_lane_b({"from_block": 1, "domains": ["energy"]})
    st = LaneBState()
    st.record(DomainWin(domain="energy", hotkey="a", uid=3, round_id="r", merge_block=0,
                        target_lcb=0.03).with_full_pay(full_pay(0.03, lb.full_pay_tiers)))
    later = replace(lb, full_pay_tiers=((0.02, 1),))      # e.g. a later overlay
    p = lane_b_payouts(st, block=40, era_blocks=10, cfg=later, n_uids=8)[0]
    assert p["full_eras"] == 5 and p["multiplier"] == 1.0  # still the win's 5 eras


# ── lane B: tiers at the win, per target ────────────────────────────────────


def _verdict(target, lcb):
    return ExpertVerdict(passed=True, reason="merge", target_domain=target, target_lcb=lcb,
                         target_n=200, target_gain=lcb + 0.01, target_clusters=8)


def test_multi_target_leg_tiers_each_target_by_its_own_lcb():
    from cascade.shared.expert_manifest import ExpertEntry
    from cascade.validator import expert_lane as vel

    e = ExpertEntry(miner_hotkey="alice", miner_uid=7, gen_ref="g", domain="energy",
                    selection={0: [1]}, exclude={}, base_king_pointer=_ptr("k"),
                    rebased_pointer=_ptr("r"), corpus_digest="c", train_block=5,
                    commit_block=4, targets=("energy", "web", "sales"))
    tv = TargetsVerdict(targets=e.declared, alpha=0.05 / 3,
                        verdicts={"energy": _verdict("energy", 0.06),
                                  "web": _verdict("web", 0.02),
                                  "sales": _verdict("sales", 0.012)})
    j = vel.ExpertJudgement(e, True, "verdict", "won", tv.verdicts["energy"],
                            king_pointer=_ptr("k"), targets=tv)
    st = LaneBState()
    wins = {w.domain: w for w in vel.apply_judgements(st, [j], round_id="r1",
                                                       settle_block=100, tiers=TIERS)}
    assert {d: (w.tier, w.full_eras) for d, w in wins.items()} == {
        "energy": (0.05, 10), "web": (0.02, 5), "sales": (0.01, 2)}
    # persisted and reloaded verbatim
    again = LaneBState.from_json(st.to_json())
    assert again.wins == st.wins
    # the settlement state carries the tier (bound across receipts)
    post = settlement_state(SimpleNamespace(), lane_b=st)
    assert post["lane_b"]["web"][4:] == [0.02, 0.02, 5]


# ── lane C / MarketCast ─────────────────────────────────────────────────────


def _mc(tiers):
    return SimpleNamespace(full_pay_tiers=tiers)


def test_lane_c_and_marketcast_tiers():
    # MarketCast-only lane C: an mkt: win is tiered by [marketcast]'s table
    lc = parse_lane_c({})
    mc = _mc(((0.10, 7), (0.01, 1)))
    assert full_pay_tiers_for("mkt:returns", lc, mc) == ((0.10, 7), (0.01, 1))
    assert target_full_pay("mkt:returns", 0.05, lc, mc).full_eras == 1
    with pytest.raises(ValueError):
        full_pay_tiers_for("mkt:returns", lc, None)


@pytest.mark.parametrize("lcb,n", [(0.05, 10), (math.nextafter(0.05, 0), 5), (0.02, 5),
                                   (math.nextafter(0.02, 0), 2), (0.01, 2)])
def test_lane_c_confirmed_win_records_its_judge_time_tier(lcb, n):
    from cascade.validator.lane_c import LaneCJudgement, _confirmed_win

    lc = parse_lane_c({})
    entry = SimpleNamespace(target="mkt:returns", miner_hotkey="h", miner_uid=3,
                            trained_pointer="t", recipe_ref="r")
    j = LaneCJudgement(entry, True, "verdict", "win", gain_lcb=lcb, phase="confirm",
                       full_pay=target_full_pay("mkt:returns", lcb, lc, _mc(TIERS)).record())
    w = _confirmed_win(j, round_id="r", settle_block=100, era_index=0)
    assert (w.gain_lcb, w.full_eras, w.below_floor) == (lcb, n, False)
    st = LaneCState()
    st.record(w)
    p = lane_c_payouts(st, block=100 + n * 10, era_blocks=10, cfg=lc, n_uids=8)[0]
    assert p["full_eras"] == n and p["multiplier"] == 0.5     # first decayed era
    if n:
        p0 = lane_c_payouts(st, block=100 + (n - 1) * 10, era_blocks=10, cfg=lc, n_uids=8)[0]
        assert p0["multiplier"] == 1.0
    # a confirm without a tier is refused
    with pytest.raises(ValueError, match="full-pay tier"):
        _confirmed_win(replace(j, full_pay=None), round_id="r", settle_block=100,
                       era_index=0)


@pytest.mark.parametrize("lcb,passed,want", [(0.12, True, 7), (0.05, True, 1),
                                              (0.005, False, None)])
def test_marketcast_judgement_tiers_by_marketcast_tiers(monkeypatch, lcb, passed, want):
    from cascade.eval.marketcast import SuiteVerdict
    from cascade.shared.config.marketcast import MarketCastConfig
    from cascade.validator import marketcast as vmc

    mc = replace(MarketCastConfig(), full_pay_tiers=((0.10, 7), (0.01, 1)))
    monkeypatch.setattr(vmc, "verdict_from_rows", lambda *a, **k: SuiteVerdict(
        "returns", passed, "win" if passed else "gain LCB < margin", 300, 9,
        gain=lcb + 0.01, gain_lcb=lcb))
    monkeypatch.setattr(vmc, "base_cov_gain", lambda rows: None)
    rows = SimpleNamespace(status="ok", after=1.0, fault="", windows=[], near_dup=None)
    entry = SimpleNamespace(target="mkt:returns", miner_hotkey="h", miner_uid=3,
                            trained_pointer="t", recipe_ref="r")
    j = vmc.judge_marketcast_entry(entry, rows, mc=mc, k=1, phase="confirm", seed="s",
                                   round_block=10, origin_block=10)
    assert j.passed is passed
    assert (j.full_pay or {}).get("full_eras") == want
    if passed:      # [marketcast]'s tiers, not [lane_c]'s (0.05 would be 10 there)
        assert j.full_pay == target_full_pay("mkt:returns", lcb, parse_lane_c({}), mc).record()


# ── audit: forged tier / full_eras / gain_lcb ───────────────────────────────


def _pay_win(**kw):
    w = {"domain": "energy", "hotkey": "alice", "uid": 9, "merge_block": 7200, "age": 0,
         "multiplier": 1.0, "gain_lcb": 0.03, "tier": 0.02, "full_eras": 5,
         "below_floor": False}
    w.update(kw)
    return w


def test_full_pay_mismatch_rejects_forgeries():
    ok = _pay_win()
    assert full_pay_mismatch(ok, TIERS) is None
    assert "full_eras" in full_pay_mismatch(_pay_win(full_eras=10), TIERS)
    assert "tier" in full_pay_mismatch(_pay_win(tier=0.05, full_eras=10), TIERS)
    assert full_pay_mismatch(_pay_win(gain_lcb=0.06), TIERS)        # lcb inflated → tier off
    assert "below_floor" in full_pay_mismatch(_pay_win(below_floor=True), TIERS)
    assert "carries no" in full_pay_mismatch({"gain_lcb": 0.03}, TIERS)
    assert full_pay_mismatch(_pay_win(full_eras=True, tier=0.01, gain_lcb=0.015), TIERS)


def _lane_b_receipt(block, win, *, state_post=None):
    return SimpleNamespace(epoch_start_block=block, state_post=state_post,
                           lane_b_pay={"block": block, "era_blocks": None, "wins": [win]})


def _audit_cfg(cfg):
    lb = parse_lane_b({"from_block": 1, "domains": ["energy", "web"]})
    return replace(cfg, lane_b=lb)


def test_audit_payout_replay_rejects_forged_tiers(cfg):
    from cascade.audit.checks import _lane_b_paid_wins
    from cascade.shared.era import era_length_blocks

    acfg = _audit_cfg(cfg)
    eb = int(era_length_blocks(acfg.round, 7200))

    def run(win, block=7200, post=None):
        rc = _lane_b_receipt(block, win, state_post=post)
        rc.lane_b_pay["era_blocks"] = eb
        return _lane_b_paid_wins(rc, acfg, 16)

    assert run(_pay_win())[1] is None
    for bad in (_pay_win(full_eras=10), _pay_win(tier=0.05), _pay_win(gain_lcb=0.07),
                _pay_win(full_eras=29), {k: v for k, v in _pay_win().items()
                                         if k != "tier"}):
        assert run(bad)[1] is not None, bad
    # an OLDER win: fixed at its own settlement (not re-derived under today's
    # tiers) but bound to the settlement state and self-consistent
    old = _pay_win(merge_block=7200 - eb, age=1)
    post_ok = {"lane_b": {"energy": ["alice", 9, 7200 - eb, "r", 0.03, 0.02, 5]}}
    assert run(old, post=post_ok)[1] is None
    post_bad = {"lane_b": {"energy": ["alice", 9, 7200 - eb, "r", 0.03, 0.05, 10]}}
    assert "settlement state" in run(old, post=post_bad)[1]
    assert "below_floor" in run(_pay_win(merge_block=7200 - eb, age=1, gain_lcb=0.01))[1]
    # the multiplier is replayed from the RECORDED full_eras
    late = _pay_win(merge_block=7200 - 6 * eb, age=6, multiplier=1.0)
    assert "multiplier" in run(late)[1]
    assert run(dict(late, multiplier=0.25))[1] is None     # full 5, then 0.5, 0.25


def test_check_lane_b_ties_the_pay_record_to_the_winning_row(cfg):
    from cascade.audit.checks import check_lane_b

    acfg = _audit_cfg(cfg)
    row = {"hotkey": "alice", "domain": "energy", "stage": "verdict", "passed": True,
           "reason": "merge", "target_lcb": 0.03, "guard_lcbs": {}, "public_lcbs": {}}

    def rc(win):
        return SimpleNamespace(epoch_start_block=7200, lane_b=[row], manifest={},
                               lane_b_pay={"block": 7200, "wins": [win]})

    assert check_lane_b(rc(_pay_win()), acfg).status == "PASS"
    r = check_lane_b(rc(_pay_win(gain_lcb=0.06, tier=0.05, full_eras=10)), acfg)
    assert r.status == "FAIL" and "gain_lcb" in r.detail
    r = check_lane_b(rc(_pay_win(full_eras=10)), acfg)
    assert r.status == "FAIL" and "forged" in r.detail


def test_check_lane_c_rejects_a_forged_confirm_tier(cfg):
    from cascade.audit.checks import check_lane_c

    lc = parse_lane_c({"from_block": 0})
    lc = replace(lc, from_block=7200 - 1, targets=("domain:energy",))
    acfg = replace(cfg, lane_c=lc)
    fp = target_full_pay("domain:energy", 0.03, lc).record()
    row = {"hotkey": "alice", "uid": 9, "target": "domain:energy", "stage": "verdict",
           "passed": True, "reason": "win", "gain_lcb": 0.03, "gain": 0.04, "n": 200,
           "public_lcbs": {}, "trained_pointer": "t", "train_block": 1, "phase": "confirm",
           "k": 1, "alpha": lc.alpha, "clusters": 8, "outcome": "confirmed",
           "base_pointer": "k", "origin_round": "r0", "holder": None,
           "full_pay": {k: fp[k] for k in ("tier", "full_eras", "below_floor")}}
    win = {"target": "domain:energy", "hotkey": "alice", "uid": 9, "win_block": 7200,
           "age": 0, "multiplier": 1.0, **fp}

    def rc(r, w):
        return SimpleNamespace(epoch_start_block=7200, lane_c=[r], status="scored",
                               round_id="r1", lane_c_pay={"wins": [w]})

    assert check_lane_c(rc(row, win), acfg).status == "PASS"
    forged = dict(row, full_pay={"tier": 0.05, "full_eras": 10, "below_floor": False})
    r = check_lane_c(rc(forged, dict(win, tier=0.05, full_eras=10)), acfg)
    assert r.status == "FAIL" and "forged" in r.detail
    r = check_lane_c(rc(row, dict(win, full_eras=10)), acfg)
    assert r.status == "FAIL" and "full_eras" in r.detail
    r = check_lane_c(rc(row, dict(win, gain_lcb=0.2)), acfg)
    assert r.status == "FAIL" and "gain_lcb" in r.detail


# ── config validation + overlay ─────────────────────────────────────────────


@pytest.mark.parametrize("raw,msg", [
    ([], "non-empty"),
    ("x", "non-empty"),
    ([[0.05, 10, 1]], "each tier"),
    ([[0.0, 1]], "threshold"),
    ([[1.0, 1]], "threshold"),
    ([[True, 1]], "threshold"),
    ([[0.05, -1]], "full_eras"),
    ([[0.05, 2.5]], "full_eras"),
    ([[0.05, 30]], "win_ttl_eras"),
    ([[0.02, 5], [0.05, 10]], "strictly descending"),
    ([[0.05, 5], [0.05, 2]], "strictly descending"),
    ([[0.05, 5], [0.02, 10]], "must not increase"),
])
def test_full_pay_tiers_validation(raw, msg):
    with pytest.raises(ValueError, match=msg):
        parse_full_pay_tiers(raw, section="lane_b", win_ttl_eras=28)
    with pytest.raises(ValueError, match=msg):
        parse_lane_b({"full_pay_tiers": raw})
    with pytest.raises(ValueError, match=msg):
        parse_lane_c({"full_pay_tiers": raw})


def test_full_pay_tiers_accepts_equal_full_eras_and_ttl_bound():
    assert parse_full_pay_tiers([[0.3, 28], [0.1, 28], [0.01, 0]], section="lane_c",
                                win_ttl_eras=28) == ((0.3, 28), (0.1, 28), (0.01, 0))
    assert parse_full_pay_tiers([[0.3, 99]], section="lane_c", win_ttl_eras=0) == ((0.3, 99),)
    with pytest.raises(ValueError, match="win_ttl_eras"):
        parse_lane_b({"full_pay_tiers": [[0.05, 10]], "win_ttl_eras": 9})


def test_full_rounds_is_replaced_not_read():
    with pytest.raises(ValueError, match="full_rounds was replaced"):
        parse_lane_b({"full_rounds": 5})
    with pytest.raises(ValueError, match="full_rounds was replaced"):
        parse_lane_c({"full_rounds": 5})


def test_marketcast_tiers_are_bounded_by_the_lane_c_ttl(tmp_path):
    text = (ROOT / "chain.toml").read_text()
    assert text.count("full_pay_tiers     = [[0.05, 10], [0.02, 5], [0.01, 2]]") == 1
    bad = text.replace("full_pay_tiers     = [[0.05, 10], [0.02, 5], [0.01, 2]]",
                      "full_pay_tiers     = [[0.05, 40], [0.02, 5], [0.01, 2]]")
    p = tmp_path / "chain.toml"
    p.write_text(bad)
    with pytest.raises(ValueError, match="win_ttl_eras"):
        load_chain_config(p)


def test_overlay_consistency():
    """``full_pay_tiers`` is overlay-settable on [lane_b] / [lane_c] exactly as
    ``full_rounds`` was (every key of both tables), and on [marketcast]
    through its explicit runtime-policy allow-list."""
    from cascade.shared.netconfig import ALLOWED, NetConfigError, check_overrides

    assert ALLOWED["lane_b"] is None and ALLOWED["lane_c"] is None
    check_overrides({"lane_b": {"full_pay_tiers": [[0.1, 8], [0.01, 1]]},
                     "lane_c": {"full_pay_tiers": [[0.1, 8], [0.01, 1]]}})
    check_overrides({"marketcast": {"full_pay_tiers": [[0.1, 8]]}})
    with pytest.raises(NetConfigError):
        check_overrides({"marketcast": {"base_ref": "other@pin"}})


def test_overlay_applies_new_tiers_and_refuses_bad_ones():
    from cascade.shared.netconfig import NetConfigError, build_config

    base = load_chain_config(ROOT / "chain.toml")
    body = {"protocol": "cascade-netconfig/2", "network": {"netuid": 91, "chain": "bittensor:finney"}, "seq": 1, "prev": "", "effective_block": 0,
            "overrides": {"lane_b": {"full_pay_tiers": [[0.3, 7], [0.01, 1]]}}}
    out = build_config(base, body)
    assert out.lane_b.full_pay_tiers == ((0.3, 7), (0.01, 1))
    body["overrides"]["lane_b"]["full_pay_tiers"] = [[0.01, 1], [0.3, 7]]
    with pytest.raises((NetConfigError, ValueError)):
        build_config(base, body)
