"""W-CMP (DEC-CA-0075): the rule menu — test families {plain_lcb, maxt,
bonferroni} and pay curves {tiered_decay, fixed_term, indefinite_until_beaten}
— selectable per lane / per competition; defaults are today's rules
byte-for-byte; a curve other than a lane's legacy one is recorded on the win
and the audit re-derives it."""

from __future__ import annotations

import pytest

from .helpers import parse, shipped_raw


def test_w_cmp_menu_defaults_are_todays_rules():
    from cascade.shared import rules_menu as R

    cfg = parse(shipped_raw())
    assert (cfg.lane_b.test_family, cfg.lane_b.pay_curve) == R.LANE_DEFAULTS["lane_b"]
    assert (cfg.lane_c.test_family, cfg.lane_c.pay_curve) == R.LANE_DEFAULTS["lane_c"]
    assert (cfg.lane_d.test_family, cfg.lane_d.pay_curve) == R.LANE_DEFAULTS["lane_d"]
    assert cfg.lane_b.verdict_params().bonferroni is True
    assert cfg.lane_c.verdict_params(4).alpha == pytest.approx(cfg.lane_c.alpha / 4)


def test_w_cmp_menu_choices_parse_and_unsupported_ones_are_refused():
    import copy

    import cascade.shared.competition_spec  # noqa: F401  (W-CMP)

    raw = shipped_raw()
    r = copy.deepcopy(raw)
    r["lane_b"]["test_family"], r["lane_b"]["pay_curve"] = "plain_lcb", "indefinite_until_beaten"
    r.setdefault("lane_c", {})["test_family"] = "plain_lcb"
    cfg = parse(r)
    assert cfg.lane_b.verdict_params().bonferroni is False
    assert cfg.lane_c.verdict_params(4).alpha == pytest.approx(cfg.lane_c.alpha)
    for table, key, val in (("lane_b", "test_family", "holm"), ("lane_d", "test_family",
                                                                "plain_lcb"),
                            ("lane_d", "pay_curve", "fixed_term"),
                            ("lane_c", "pay_curve", "lottery")):
        bad = copy.deepcopy(raw)
        bad.setdefault(table, {})[key] = val
        with pytest.raises(ValueError, match=key):
            parse(bad)


def test_w_cmp_family_math():
    from cascade.shared import rules_menu as R

    assert R.family_alpha("plain_lcb", 0.05, 5) == 0.05
    assert R.family_alpha("bonferroni", 0.05, 5) == pytest.approx(0.01)
    assert R.family_size("plain_lcb", 7) == 1 and R.family_size("bonferroni", 7) == 7
    with pytest.raises(R.MenuError):
        R.family_alpha("maxt", 0.05, 3)


def test_w_cmp_lane_b_plain_lcb_drops_the_per_leg_correction():
    import numpy as np

    import cascade.shared.competition_spec  # noqa: F401  (W-CMP)
    from cascade.eval.expert_lane import ExpertLaneParams, judge_expert_targets

    rng = np.random.default_rng(0)
    n = 240
    dom = ["a"] * (n // 2) + ["b"] * (n // 2)
    kq = rng.uniform(1.0, 2.0, (n, 3))
    ka, km = kq.sum(axis=1), kq.mean(axis=1)
    args = dict(domains=dom, king_qloss=kq, king_abs_target=ka, king_mase=km,
                chal_qloss=kq * 0.9, chal_abs_target=ka, chal_mase=km * 0.9,
                clusters=[f"c{i % 12}" for i in range(n)], targets=("a", "b"))
    p = ExpertLaneParams(min_target_windows=10, min_target_clusters=2, min_guard_windows=10)
    assert judge_expert_targets(**args, params=p).alpha == pytest.approx(p.alpha / 2)
    from dataclasses import replace

    assert judge_expert_targets(**args, params=replace(p, bonferroni=False)).alpha == p.alpha


def test_w_cmp_pay_curves():
    from cascade.shared import rules_menu as R

    assert not R.curve_expired("indefinite_until_beaten", 10**6, full_eras=1)
    assert R.curve_expired("fixed_term", 10, full_eras=10)
    assert not R.curve_expired("fixed_term", 9, full_eras=10)
    assert R.curve_expired("tiered_decay", 28, full_eras=5, ttl_eras=28)
    assert R.curve_multiplier("tiered_decay", 5, full_eras=5) == 0.5
    assert R.curve_multiplier("tiered_decay", 2, full_eras=5) == 1.0
    assert R.curve_multiplier("indefinite_until_beaten", 99, full_eras=5) == 1.0
    assert R.curve_record("fixed_term", "lane_c") == "" and R.curve_record(
        "tiered_decay", "lane_c") == "tiered_decay"


def test_w_cmp_lane_c_win_records_its_curve_and_pays_on_it():
    from dataclasses import replace

    import cascade.shared.competition_spec  # noqa: F401  (W-CMP)
    from cascade.shared.lane_c import (
        LaneCState,
        TargetWin,
        full_pay_record_mismatch,
        lane_c_payouts,
        target_full_pay_record,
    )

    cfg = parse(shipped_raw())
    lc, mc = cfg.lane_c, cfg.marketcast
    t = lc.targets[0]
    legacy = target_full_pay_record(t, 0.03, lc, mc, era_blocks=3600)
    assert "curve" not in legacy                         # byte-identical default
    mc2 = replace(mc, pay_curve="indefinite_until_beaten")
    rec = target_full_pay_record(t, 0.03, lc, mc2, era_blocks=3600)
    assert rec["curve"] == "indefinite_until_beaten"
    assert full_pay_record_mismatch(rec, t, lc, mc2, era_blocks=3600) is None
    assert "curve" in (full_pay_record_mismatch(rec, t, lc, mc, era_blocks=3600) or "")
    w = TargetWin(target=t, hotkey="hk", uid=3, round_id="r", win_block=3600, tier=0.03,
                  full_eras=rec["full_eras"], gain_lcb=0.03)
    late = 3600 * (rec["full_eras"] + 5)
    st = LaneCState(slots={t: w})
    assert lane_c_payouts(st, block=late, era_blocks=3600, cfg=lc, n_uids=10) == []
    st = LaneCState(slots={t: replace(w, curve="indefinite_until_beaten")})
    rows = lane_c_payouts(st, block=late, era_blocks=3600, cfg=lc, n_uids=10)
    assert rows and rows[0]["multiplier"] == 1.0 and rows[0]["curve"] == "indefinite_until_beaten"
    st = LaneCState(slots={t: replace(w, curve="tiered_decay")})
    rows = lane_c_payouts(st, block=late, era_blocks=3600, cfg=lc, n_uids=10)
    assert rows and 0.0 < rows[0]["multiplier"] < 1.0
    js = LaneCState(slots={t: w}).to_json()
    assert '"curve"' not in js                           # legacy state bytes unchanged


def test_w_cmp_lane_b_win_records_its_curve_and_pays_on_it():
    from dataclasses import replace

    import cascade.shared.competition_spec  # noqa: F401  (W-CMP)
    from cascade.shared.lane_b import DomainWin, LaneBState, lane_b_payouts

    cfg = parse(shipped_raw())
    lb = cfg.lane_b
    d = (getattr(lb, "targets", None) or lb.domains)[0]
    w = DomainWin(domain=d, hotkey="hk", uid=2, round_id="r", merge_block=3600, tier=0.01,
                  full_eras=2, target_lcb=0.02)
    late = 3600 * (int(lb.win_ttl_eras or 40) + 3)
    assert lane_b_payouts(LaneBState(wins={d: w}), block=late, era_blocks=3600, cfg=lb,
                          n_uids=10) == ([] if lb.win_ttl_eras else
                                         lane_b_payouts(LaneBState(wins={d: w}), block=late,
                                                        era_blocks=3600, cfg=lb, n_uids=10))
    ind = replace(w, curve="indefinite_until_beaten")
    rows = lane_b_payouts(LaneBState(wins={d: ind}), block=late, era_blocks=3600, cfg=lb,
                          n_uids=10)
    assert rows and rows[0]["multiplier"] == 1.0 and rows[0]["curve"] == "indefinite_until_beaten"
    assert '"curve"' not in LaneBState(wins={d: w}).to_json()
