"""WB26: the validator wiring — the real ``judge_manifest`` /
``judgements_record`` and ``LaneBMixin._lane_b_judge_experts`` — for suite
slots: the anchor king (DEC-CA-0071 §3 guard 5) comes from the signed
training manifest, is scored at most once per manifest and only when a
suite slot would be claimed, and a claimed suite slot's row records
``suite_guards``."""

from __future__ import annotations

from tests.accept.w_s2 import helpers as ws2

from . import helpers as h

GIFT = "suite:gift"


def _em(*entries):
    return ws2.expert_manifest(*entries)


def _row(rows, slot, hk=ws2.HK):
    (r,) = [r for r in rows if r["domain"] == slot and r["hotkey"] == hk]
    return r


def test_suite_slot_claimed_with_anchor_and_row_records_suite_guards(tmp_path):
    cfg = h.lane_b_cfg()
    assert GIFT in cfg.targets
    em = _em(h.entry(GIFT, h.bench_rows(h.GIFT_GAIN), anchor=h.bench_rows({})))
    calls: list = []
    js = h.judge_manifest(em, cfg, tmp_path, calls=calls)
    assert js[0].passed, js[0].reason
    row = _row(h.record(js), GIFT)
    assert row["passed"], row["reason"]
    sg = row.get("suite_guards")
    assert isinstance(sg, dict) and sg.get("anchor_pointer") == h.ANCHOR_PTR, sorted(row)
    assert {"pool_gain", "anchor_guard_lcbs", "anchor_public_lcbs"} <= set(sg)
    assert calls.count(h.ANCHOR_PTR) == 1, calls
    for r in h.record(js):
        if not r["passed"]:
            assert "suite_guards" not in r, r["domain"]


def test_anchor_scored_once_per_manifest(tmp_path):
    cfg = h.lane_b_cfg()
    em = _em(h.entry(GIFT, h.bench_rows(h.GIFT_GAIN), anchor=h.bench_rows({})),
             h.entry("suite:boom", h.bench_rows({("boom", "crps"): 0.90}),
                     anchor=h.bench_rows({}), uid=12, hk="5OtherHotkey"))
    calls: list = []
    js = h.judge_manifest(em, cfg, tmp_path, calls=calls)
    assert [j.passed for j in js] == [True, True], [j.reason for j in js]
    assert calls.count(h.ANCHOR_PTR) == 1, calls


def test_missing_anchor_rows_fail_closed(tmp_path):
    cfg = h.lane_b_cfg()
    em = _em(h.entry(GIFT, h.bench_rows(h.GIFT_GAIN)))          # no "anchor" rows
    js = h.judge_manifest(em, cfg, tmp_path)
    row = _row(h.record(js), GIFT)
    assert not row["passed"] and "anchor_missing" in row["reason"], row["reason"]


def test_no_anchor_pointer_fails_closed(tmp_path):
    cfg = h.lane_b_cfg()
    em = _em(h.entry(GIFT, h.bench_rows(h.GIFT_GAIN), anchor=h.bench_rows({})))
    js = h.judge_manifest(em, cfg, tmp_path, anchor_pointer="")
    row = _row(h.record(js), GIFT)
    assert not row["passed"] and "anchor_missing" in row["reason"], row["reason"]


def test_domain_leg_never_scores_the_anchor(tmp_path):
    """A domain leg that clears no suite slot is judged exactly as #854
    judges it: no anchor scoring, no suite_guards in any row."""
    cfg = h.lane_b_cfg()
    em = _em(h.entry("energy", h.bench_rows({}), anchor=h.bench_rows({})))
    calls: list = []
    js = h.judge_manifest(em, cfg, tmp_path, calls=calls, cand_effect=ws2.GAIN_EFFECT)
    assert js[0].passed, js[0].reason
    assert h.ANCHOR_PTR not in calls, calls
    assert all("suite_guards" not in r for r in h.record(js))


def test_anchor_pointer_is_the_era_promotion_member():
    """DEC-CA-0071 §3 guard 5: the anchor is the era's promotion member — the
    merged init's base when lane-B merges are applied, else the warm-start
    checkpoint; a random-init era has none."""
    from types import SimpleNamespace

    from cascade.validator.lane_b import lane_b_anchor_pointer

    merged = SimpleNamespace(warm_start_ckpt="merged", warm_start_base="member",
                             warm_start_merges=({"rebased_pointer": "x"},))
    plain = SimpleNamespace(warm_start_ckpt="member", warm_start_base="", warm_start_merges=())
    none = SimpleNamespace(warm_start_ckpt="", warm_start_base="", warm_start_merges=())
    assert lane_b_anchor_pointer(merged) == "member"
    assert lane_b_anchor_pointer(plain) == "member"
    assert lane_b_anchor_pointer(none) is None


def test_validator_scores_the_manifest_anchor(tmp_path):
    cfg = h.lane_b_cfg()
    em = _em(h.entry(GIFT, h.bench_rows(h.GIFT_GAIN), anchor=h.bench_rows({})))
    rnd, seen = h.validator_round(tmp_path, cfg, em, warm_start_ckpt=h.ANCHOR_PTR)
    assert h.ANCHOR_PTR in seen, f"the validator scores the era's anchor king; scored {seen}"
    row = _row(rnd.rows, GIFT)
    assert row.get("suite_guards", {}).get("anchor_pointer") == h.ANCHOR_PTR, row
    assert row["passed"], row["reason"]


def test_inv1_validator_does_not_score_an_anchor_with_suite_targets_off(tmp_path):
    """``suite_targets = false``: no suite slot, so although the judge is
    handed the anchor (WB26 is not block-gated, owner 2026-11-24) nothing
    but the king and the candidate is scored, and no row carries
    suite_guards."""
    from cascade.validator.lane_b import lane_b_anchor_pointer  # noqa: F401  (WB26 helper)

    cfg = h.lane_b_cfg(suite_targets=False)
    em = _em(h.entry("energy", h.bench_rows({}), anchor=h.bench_rows({})))
    rnd, seen = h.validator_round(tmp_path, cfg, em, warm_start_ckpt=h.ANCHOR_PTR)
    assert h.ANCHOR_PTR not in seen and set(seen) <= {h.KING_PTR, h.CAND_PTR}, seen
    assert all("suite_guards" not in r for r in rnd.rows)


def test_rule_is_not_behind_an_activation_block():
    """Owner 2026-11-24: lane B is not rolled out, so the WB26 guards are the
    default rule — no ``[lane_b]`` activation-block key gates them."""
    from cascade.shared.config.lane_b import LANE_B_TRACK_DEFAULTS

    new = {"tau_anchor_public", "tau_anchor_domain"}
    assert new <= set(LANE_B_TRACK_DEFAULTS), sorted(LANE_B_TRACK_DEFAULTS)
    assert not [k for k in LANE_B_TRACK_DEFAULTS if "anchor" in k and k.endswith("_from_block")]
