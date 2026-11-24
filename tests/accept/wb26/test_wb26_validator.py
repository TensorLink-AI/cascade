"""WB26: the validator wiring — the real ``judge_manifest`` /
``judgements_record`` and ``LaneBMixin._lane_b_judge_experts`` — for suite
targets: the anchor king (DEC-CA-0071 §3 guard 5) comes from the signed
training manifest, is scored once, and every suite-target row records the
rev6.19 (G) fields."""

from __future__ import annotations

from tests.accept.w_s2 import helpers as ws2

from . import helpers as h

GIFT = "suite:gift"
CAND_GIFT_CRPS = {("gift", "crps"): 0.95}
CAND_GIFT_MASE = {("gift", "mase"): 0.95}


def _em(*entries):
    return ws2.expert_manifest(*entries)


def test_suite_target_entry_wins_with_anchor_and_records_both_lcbs(tmp_path):
    cfg = h.lane_b_cfg()
    assert "suite:gift" in cfg.targets
    em = _em(h.entry(GIFT, h.bench_rows(CAND_GIFT_MASE), anchor=h.bench_rows({})))
    calls: list = []
    js = h.judge_manifest(em, cfg, tmp_path, calls=calls)
    assert js[0].passed, js[0].reason
    (row,) = h.record(js)
    g = row.get("suite_gain")
    assert isinstance(g, dict), f"a suite-target row records `suite_gain`; got {sorted(row)}"
    assert g["cleared_metric"] == "mase"
    assert {"crps_lcb", "mase_lcb", "crps_gain", "mase_gain", "noninf_metric", "noninf_lcb",
            "gain_alpha", "n_items"} <= set(g)
    sg = row.get("suite_guards")
    assert isinstance(sg, dict) and sg["anchor_pointer"] == h.ANCHOR_PTR, sg
    assert {"pool_gain", "anchor_guard_lcbs", "anchor_public_lcbs"} <= set(sg)
    assert row["target_lcb"] == g["mase_lcb"]
    assert h.ANCHOR_PTR in calls


def test_anchor_scored_once_per_manifest(tmp_path):
    cfg = h.lane_b_cfg()
    em = _em(h.entry(GIFT, h.bench_rows(CAND_GIFT_CRPS), anchor=h.bench_rows({})),
             h.entry("suite:boom", h.bench_rows({("boom", "crps"): 0.95}),
                     anchor=h.bench_rows({}), uid=12, hk="5OtherHotkey"))
    calls: list = []
    js = h.judge_manifest(em, cfg, tmp_path, calls=calls)
    assert [j.passed for j in js] == [True, True], [j.reason for j in js]
    assert calls.count(h.ANCHOR_PTR) == 1, calls


def test_missing_anchor_rows_fail_closed(tmp_path):
    cfg = h.lane_b_cfg()
    em = _em(h.entry(GIFT, h.bench_rows(CAND_GIFT_CRPS)))          # no "anchor" rows
    js = h.judge_manifest(em, cfg, tmp_path)
    assert not js[0].passed and "anchor_missing" in js[0].reason, js[0].reason


def test_no_anchor_pointer_fails_closed(tmp_path):
    cfg = h.lane_b_cfg()
    em = _em(h.entry(GIFT, h.bench_rows(CAND_GIFT_CRPS), anchor=h.bench_rows({})))
    js = h.judge_manifest(em, cfg, tmp_path, anchor_pointer="")
    assert not js[0].passed and "anchor_missing" in js[0].reason, js[0].reason


def test_domain_entry_never_scores_the_anchor(tmp_path):
    """A domain leg under armed suite targets is judged exactly as before:
    no anchor scoring, no suite fields in its row."""
    cfg = h.lane_b_cfg()
    em = _em(h.entry("energy", h.bench_rows({}), anchor=h.bench_rows({})))
    calls: list = []
    js = h.judge_manifest(em, cfg, tmp_path, calls=calls, cand_effect=ws2.GAIN_EFFECT)
    assert js[0].passed, js[0].reason
    assert h.ANCHOR_PTR not in calls, calls
    (row,) = h.record(js)
    assert "suite_gain" not in row and "suite_guards" not in row, sorted(row)


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
    em = _em(h.entry(GIFT, h.bench_rows(CAND_GIFT_CRPS), anchor=h.bench_rows({})))
    rnd, seen = h.validator_round(tmp_path, cfg, em, warm_start_ckpt=h.ANCHOR_PTR)
    assert h.ANCHOR_PTR in seen, f"the validator scores the era's anchor king; scored {seen}"
    (row,) = rnd.rows
    assert row.get("suite_guards", {}).get("anchor_pointer") == h.ANCHOR_PTR, row
    assert row["passed"], row["reason"]


def test_inv1_validator_does_not_score_an_anchor_with_suite_targets_off(tmp_path):
    """``suite_targets = false``: no suite slot, the judge call carries no
    anchor and nothing but the king and the candidate is scored."""
    from cascade.validator.lane_b import lane_b_anchor_pointer  # noqa: F401  (WB26 helper)

    cfg = h.lane_b_cfg(suite_targets=False)
    em = _em(h.entry("energy", h.bench_rows({}), anchor=h.bench_rows({})))
    rnd, seen = h.validator_round(tmp_path, cfg, em, warm_start_ckpt=h.ANCHOR_PTR)
    assert h.ANCHOR_PTR not in seen and set(seen) <= {h.KING_PTR, h.CAND_PTR}, seen
    (row,) = rnd.rows
    assert "suite_gain" not in row and "suite_guards" not in row
