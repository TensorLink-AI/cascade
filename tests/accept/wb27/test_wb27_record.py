"""WB27 signed record and judging (plan rev6.21): the expert entry carries
``levers`` and the limits-leg record (token limit, own-risk GPU, what the leg
trained and which limit stopped it); every field is dropped from the
canonical body at its default; judges re-derive from the record."""

from __future__ import annotations

import json

from . import helpers as h

KING = "metro-v1:trained:hippius:org/king@sha256:" + "1" * 64
LEG = "metro-v1:trained:hippius:org/leg@sha256:" + "2" * 64


def _entry(**kw):
    from cascade.shared.expert_manifest import ExpertEntry

    base = dict(miner_hotkey=h.HK, miner_uid=3, gen_ref=h.REF, domain="energy",
                selection={0: [1, 2]}, exclude={}, base_king_pointer=KING,
                rebased_pointer=LEG, corpus_digest="c" * 64, train_block=100,
                commit_block=50)
    base.update(kw)
    return ExpertEntry(**base)


def _manifest(*entries):
    from cascade.shared.expert_manifest import ExpertManifest

    return ExpertManifest(round_id="r1", created_block=200, king_pointer=KING,
                          lineage_tokens=1, entries=tuple(entries))


def test_default_entry_keeps_todays_bytes():
    h.require_built()
    body = json.loads(_manifest(_entry()).canonical_body())
    e = body["entries"][0]
    for k in ("levers", "train_tokens", "gpu_measured", "trained_steps", "trained_tokens",
              "stop"):
        assert k not in e, k


def test_record_fields_round_trip():
    h.require_built()
    from cascade.shared.expert_manifest import load_expert_manifest

    e = _entry(levers={"lr_scale": 1.5}, train_tokens=20 * 10**9, gpu_measured=False,
               trained_steps=9000, trained_tokens=9000 * 2_097_152, stop="hours",
               train_hours=12.0, leg_steps=9536, leg_guard_seconds=43_200)
    m = _manifest(e)
    body = json.loads(m.canonical_body())["entries"][0]
    assert body["levers"] == {"lr_scale": 1.5} and body["gpu_measured"] is False
    assert (body["trained_steps"], body["stop"]) == (9000, "hours")
    back = load_expert_manifest(m.to_json())
    assert back.entries[0] == e and back.digest() == m.digest()


def test_leg_record_check_for_limits_legs():
    h.require_built()
    m = h.leg()
    cfg, b = h.armed_cfg()
    c = h.base_contract(cfg)
    tps = h.want_tokens_per_step(c)
    steps = 20 * 10**9 // tps
    ok = dict(train_hours=12.0, steps=steps, guard_seconds=43_200, lb_cfg=cfg.lane_b,
              contract=c, block=b, train_tokens=20 * 10**9, trained_steps=steps,
              trained_tokens=steps * tps, stop="steps")
    assert m.check_leg_record(**ok) is None
    assert m.check_leg_record(**{**ok, "steps": steps - 1}) == "steps_not_tokens"
    assert m.check_leg_record(**{**ok, "guard_seconds": 43_201}) == "guard_out_of_range"
    # the hour limit stopped it: a normal stop, judged on what it trained
    assert m.check_leg_record(**{**ok, "trained_steps": 10, "trained_tokens": 10 * tps,
                                 "stop": "hours"}) is None
    for bad in ({"stop": "steps", "trained_steps": 10}, {"stop": "crash"},
                {"trained_steps": 0}, {"trained_steps": steps + 1},
                {"trained_tokens": steps * tps + 1}):
        assert m.check_leg_record(**{**ok, **bad}) == "actuals_out_of_range", bad
    # an own-risk GPU with an hour limit only: steps = the steps cap
    cap = h.want_reference_steps(c, 4.0)
    own = dict(ok, train_tokens=0, gpu_measured=False, train_hours=4.0, steps=cap,
               guard_seconds=4 * 3600, trained_steps=cap, trained_tokens=0)
    assert m.check_leg_record(**own) is None
    assert m.check_leg_record(**{**own, "steps": cap - 1}) == "steps_out_of_range"


def test_judge_refuses_bad_levers_at_stage_levers():
    h.require_built()
    from cascade.validator.expert_lane import judge_entry

    cfg, b = h.armed_cfg()
    for lv, tb in (({"lr_scale": 3.0}, b), ({"lr_scale": 1.0}, b)):
        e = _entry(levers=lv, train_block=tb)
        j = judge_entry(e, manifest=_manifest(e), king_pointer=KING, king_scores=None,
                        evaluate=lambda p: [], fetch_dir=lambda p: None, cfg=cfg.lane_b,
                        seed="1")
        assert (j.passed, j.stage) == (False, "levers"), (j.stage, j.reason)


def test_judge_rederives_the_ema_choice():
    h.require_built()
    from cascade.validator.expert_lane import judge_entry

    cfg, b = h.armed_cfg()
    e = _entry(levers={"ema": 0.05}, train_block=b, scored_weights="ema010")
    j = judge_entry(e, manifest=_manifest(e), king_pointer=KING, king_scores=None,
                    evaluate=lambda p: [], fetch_dir=lambda p: None, cfg=cfg.lane_b, seed="1")
    assert (j.passed, j.stage) == (False, "scored_weights"), (j.stage, j.reason)


def test_audit_knows_the_levers_stage():
    h.require_built()
    from cascade.audit import checks

    assert "levers" in checks.LANE_B_STAGES and "scored_weights" in checks.LANE_B_STAGES


def test_settlement_reads_the_leg_record(tmp_path):
    h.require_built()
    from cascade.trainer.lane_b_driver import _scored_weights_for, leg_actuals

    d = tmp_path / "leg"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({
        "power_ema": {"sigma_rels": [0.05, 0.1], "scored_sigma_rel": 0.05,
                      "schedule": "warmup_constant_ema"},
        "lane_b_leg": {"steps": 7, "tokens": 70, "stop": "hours"}}))
    assert leg_actuals(d) == {"steps": 7, "tokens": 70, "stop": "hours"}
    lb = h.chain_cfg().lane_b
    for s, want in ((0.03, "ema003"), (0.15, "ema015"), (0.2, "ema020")):
        assert _scored_weights_for(lb, {"ema": s}) == want, s
    assert _scored_weights_for(lb, {}) == "ema010"
    assert _scored_weights_for(lb, {"ema": 0.5}) == "?"
    (d / "config.json").write_text("{}")
    assert leg_actuals(d) == {}
