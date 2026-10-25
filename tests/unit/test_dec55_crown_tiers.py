"""DEC-CA-0055 margin-tiered start share (owner 2026-10-20): a newly crowned
lane-A king starts at the ``[v2] ladder_start_tiers`` share its crowning
increment-unit LCB meets (a level-unit crowning at the fixed ``[v2]
ladder_start_level_bp``); everything after the start is unchanged. The audit
re-derives units, tier and start from the verdict and catches forgeries."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from types import SimpleNamespace

import pytest

from cascade.audit.checks import check_weights, crown_start_mismatch
from cascade.eval.koth import evaluate_round
from cascade.eval.scoring import WindowScore
from cascade.shared.config import load_chain_config
from cascade.shared.config.v2 import LADDER_START_TIERS, parse_v2_economics
from cascade.shared.emission import (
    crown_record,
    crown_start,
    crown_units,
    ladder_step,
)
from cascade.shared.era import era_length_blocks
from cascade.shared.manifest import TrainedEntry, TrainingManifest
from cascade.shared.netconfig import ALLOWED, build_config
from cascade.shared.receipt import EntryScores, VerdictRecord, WindowScoreRecord
from cascade.validator import state as state_mod
from cascade.validator.loop import RoundOutcome
from cascade.validator.state import ChampionState

from .receipt_fixture import make_scores
from .test_dec55_validator import CID, KING, KP, ROOT, _Client, _receipt, _runner

NEW = "5New"


@pytest.fixture
def env(tmp_path):
    cfg = load_chain_config(ROOT / "chain.toml")
    era = era_length_blocks(cfg.round, 0)
    gate = 10_000 * era
    cfg = replace(cfg, v2=replace(cfg.v2, from_block=gate),
                  training=replace(cfg.training, v2_from_block=gate),
                  lane_b=replace(cfg.lane_b, from_block=gate + 1, share=0.5),
                  validator=replace(cfg.validator, state_db_path=str(tmp_path / "s.json")))
    return cfg, gate, era


# ── the rule ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("threshold,start_bp", LADDER_START_TIERS)
def test_each_tier_boundary_at_and_just_below(threshold, start_bp):
    at = crown_start(threshold, "increment")
    assert (at.tier, at.start_bp, at.below_floor) == (threshold, start_bp, False)
    below = crown_start(math.nextafter(threshold, 0.0), "increment")
    lower = [t for t in LADDER_START_TIERS if t[0] < threshold]
    if lower:
        assert (below.tier, below.start_bp, below.below_floor) == (*lower[0], False)
    else:   # under the lowest threshold: paid at the lowest tier, flagged
        assert (below.tier, below.start_bp, below.below_floor) == (threshold, start_bp, True)


def test_defaults_and_non_finite_lcb():
    assert crown_start(1.7, "increment").start_bp == 8000      # the cap-2 statistic
    assert crown_start(0.3, "increment").start_bp == 6500
    for bad in (None, float("nan")):
        c = crown_start(bad, "increment")
        assert c.below_floor and c.start_bp == 2500 and c.lcb is None
    with pytest.raises(ValueError):
        crown_start(0.3, "pct")


def test_level_units_use_the_fixed_start(env):
    cfg, _, _ = env
    for lcb in (0.9, 0.3, 0.02, None):
        c = crown_start(lcb, "level", cfg)
        assert (c.tier, c.start_bp, c.below_floor) == (None, 5000, False)
    cfg2 = replace(cfg, v2=replace(cfg.v2, ladder_start_level_bp=4200))
    assert crown_start(0.9, "level", cfg2).start_bp == 4200


def test_crown_units_rule():
    kw = dict(margin_mode="increment", has_baseline=True, maxt=False, maxt_increment=False)
    assert crown_units(**kw) == "increment"
    assert crown_units(**{**kw, "has_baseline": False}) == "level"     # level fallback
    assert crown_units(**{**kw, "margin_mode": "level"}) == "level"
    assert crown_units(**{**kw, "maxt": True}) == "level"               # level max-T bound
    assert crown_units(**{**kw, "maxt": True, "maxt_increment": True}) == "increment"


def test_post_start_ladder_is_unchanged(env):
    cfg, _, _ = env
    start = crown_start(0.02, "increment", cfg).start_bp
    assert start == 2500
    assert ladder_step(start, crowned=False, check="pass", cfg=cfg) == 3500
    assert ladder_step(start, crowned=False, check="deferred", cfg=cfg) == 2500
    assert ladder_step(start, crowned=False, check="fail", cfg=cfg) == 1000
    assert ladder_step(9000, crowned=False, check="pass", cfg=cfg) == 9500
    assert ladder_step(None, crowned=True, check=None, cfg=cfg, start_bp=start) == 2500
    # a DEC-CA-0048 successor (no duel LCB) reads as the level start
    assert ladder_step(None, crowned=False, check="pass", cfg=cfg) == 6000


# ── config ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("tiers", [
    [[0.10, 5000], [0.25, 6500]],                    # not descending
    [[0.25, 6500], [0.25, 5000]],                    # not strictly descending
    [[1.0, 8000], [0.10, 5000]],                     # threshold out of (0, 1)
    [[0.0, 5000]],
    [[0.50, 9600], [0.10, 5000]],                    # start above cap
    [[0.50, 0]],                                     # start not > 0
    [[0.50, 500]],                                   # start below ladder_fail_bp
    [[0.50, 5000], [0.10, 6500]],                    # start increases as threshold falls
    [[0.50, 5000.0]],                                # start not an int
    [[0.50, True]],
    [[0.50]],
    [],
    "0.5:8000",
])
def test_tier_refusals(tiers):
    with pytest.raises(ValueError):
        parse_v2_economics({"ladder_start_tiers": tiers})


def test_tiers_parse_and_ship(env):
    cfg, _, _ = env
    assert cfg.v2.ladder_start_tiers == LADDER_START_TIERS
    assert cfg.v2.ladder_start_level_bp == 5000
    got = parse_v2_economics({"ladder_start_tiers": [[0.4, 7000], [0.02, 3000]]})
    assert got["ladder_start_tiers"] == ((0.4, 7000), (0.02, 3000))
    testnet = load_chain_config(ROOT / "chain.testnet.toml")
    assert testnet.v2.ladder_start_tiers == LADDER_START_TIERS


def test_overlay_allow_list_is_consistent(env):
    cfg, _, _ = env
    v2 = ALLOWED["v2"]
    ladder_keys = {k for k in vars(cfg.v2) if k.startswith("ladder_")}
    assert ladder_keys == {"ladder_start_tiers", "ladder_start_level_bp", "ladder_step_bp",
                           "ladder_cap_bp", "ladder_fail_bp"}
    assert ladder_keys <= v2 and "ladder_start_bp" not in v2
    body = {"protocol": "cascade-netconfig/2", "network": {"netuid": 91, "chain": "bittensor:finney"}, "seq": 1, "prev": "", "effective_block": 0,
            "overrides": {"v2": {"ladder_start_tiers": [[0.3, 7000], [0.01, 3000]],
                                 "ladder_start_level_bp": 4000}}}
    out = build_config(cfg, body)
    assert out.v2.ladder_start_tiers == ((0.3, 7000), (0.01, 3000))
    assert crown_start(0.31, "increment", out).start_bp == 7000
    assert crown_start(0.31, "level", out).start_bp == 4000
    bad = json.loads(json.dumps(body))
    bad["overrides"]["v2"]["ladder_start_tiers"] = [[0.01, 3000], [0.3, 7000]]
    with pytest.raises(ValueError):
        build_config(cfg, bad)


# ── the validator + the audit ────────────────────────────────────────────────


def _scaled(scores, f):
    return [WindowScore(s.series_id, s.mase * f, s.qloss_per_q * f, s.abs_target)
            for s in scores]


def _recs(scores):
    return tuple(WindowScoreRecord.from_score(s) for s in scores)


def _crowning(env, *, increment: bool, chal_f: float = 0.97):
    """A real dethrone at a v2 settlement: scores → verdict → the validator's
    ladder step + vector → a receipt the audit re-derives."""
    cfg, gate, era = env
    b = gate + 20 * era
    m = TrainingManifest(
        round_id=str(b + 5), created_block=b + 5, contract_digest="c", base_arch_digest="d",
        eval_dataset="e", prev_round_id="77",
        entries=[TrainedEntry(KING, 3, "king", CID, KP, "d", b - 10, funded_by="king"),
                 TrainedEntry(NEW, 7, "challenger", CID, KP, "d", b - 9)])
    king = make_scores(1.0, 0, n=max(256, cfg.scoring.min_windows))
    chal, init = _scaled(king, chal_f), _scaled(king, 1.05)
    params = cfg.koth_params(block=b)
    assert params.margin_mode == "increment"
    judged = params if increment else cfg.judged_level_params(params, b)
    result = evaluate_round(king, chal, judged, seed=1, king_tenure_rounds=0,
                            baseline_scores=init if increment else None)
    assert result.challenger_wins_round
    pre = ChampionState(king_hotkey=KING, king_uid=3, king_since_block=gate + era,
                        king_share_bp=9000, king_key_lapses=1, king_slash_until_block=10**12)
    tr = state_mod.apply_round(pre, challenger_hotkey=NEW, challenger_uid=7, result=result,
                               dethrone_cp=1, keep_former_kings=1, crowned_block=b,
                               crowned_pointer=KP)
    assert tr.dethroned
    units = crown_units(margin_mode=judged.margin_mode, has_baseline=increment, maxt=False,
                        maxt_increment=False)
    r = _runner(cfg, tr.state)
    r._v2_settlement_economics(m, [], 1, RoundOutcome(result=result, transition=tr,
                                                      lcb_units=units), KING)
    c = _Client(b + 50, {KING: 3, NEW: 7})
    uids = [7, 3]
    r._lane_b_block = b
    vec = r._apply_weights(c, m.round_id, uids, boundary=b)
    rows = [EntryScores("king", "", KING, 3, _recs(king)),
            EntryScores("challenger", "", NEW, 7, _recs(chal))]
    if increment:
        rows.append(EntryScores("baseline", "", "", -1, _recs(init)))
    rc = replace(_receipt(r, m, b, uids, vec), entry_scores=tuple(rows),
                 verdict=VerdictRecord.from_round(result, tr, params=params,
                                                  bootstrap_seed=1, king_tenure_rounds=0))
    return cfg, r, rc, result, vec


def test_increment_crowning_starts_at_its_tier_and_the_audit_rederives_it(env):
    cfg, r, rc, result, vec = _crowning(env, increment=True)
    want = crown_start(result.lcb, "increment", cfg)
    assert want.tier is not None and not want.below_floor
    assert r.state.king_share_bp == want.start_bp
    lad = rc.emission["ladder"]
    assert lad["event"] == "crowned" and lad["bp"] == want.start_bp
    assert {k: lad[k] for k in crown_record(want)} == crown_record(want)
    assert lad["crown_lcb"] == rc.verdict.lcb
    assert vec[7] == pytest.approx(want.start_bp / 10000 * 0.5)
    assert r._emission_policy(rc.epoch_start_block).king_share == want.start_bp / 10000
    assert check_weights(rc, cfg).status == "PASS"


def test_level_crowning_starts_at_the_fixed_value(env):
    cfg, r, rc, result, _ = _crowning(env, increment=False, chal_f=0.9)
    lad = rc.emission["ladder"]
    assert (lad["crown_units"], lad["crown_tier"], lad["bp"]) == ("level", None, 5000)
    assert r.state.king_share_bp == 5000
    assert check_weights(rc, cfg).status == "PASS"
    # a level-unit crowning claimed as increment: the units re-derive as level
    em = json.loads(json.dumps(rc.emission))
    em["ladder"]["crown_units"] = "increment"
    assert check_weights(replace(rc, emission=em), cfg).status == "FAIL"


@pytest.mark.parametrize("field,value", [
    ("crown_lcb", 0.9),
    ("crown_tier", 0.5),
    ("crown_start_bp", 8000),
    ("bp", 8000),
    ("crown_below_floor", True),
])
def test_audit_catches_a_forged_crown(env, field, value):
    cfg, r, rc, result, _ = _crowning(env, increment=True)
    em = json.loads(json.dumps(rc.emission))
    if em["ladder"][field] == value:
        value = 1234 if isinstance(value, int) and not isinstance(value, bool) else 0.011
    em["ladder"][field] = value
    if field == "bp":   # the policy follows the forged bp, so only the crown check differs
        em["policy"]["king_share"] = value / 10000
    assert check_weights(replace(rc, emission=em), cfg).status == "FAIL"


def test_audit_requires_the_crown_fields_and_refuses_them_elsewhere(env):
    cfg, r, rc, result, _ = _crowning(env, increment=True)
    lad = {k: v for k, v in rc.emission["ladder"].items() if not k.startswith("crown_")}
    assert "no crown_lcb" in crown_start_mismatch(lad, cfg, verdict_lcb=rc.verdict.lcb,
                                                  units="increment")
    assert crown_start_mismatch(rc.emission["ladder"], cfg, verdict_lcb=rc.verdict.lcb,
                                units="increment") == ""


def test_key_slash_still_applies_on_top_of_a_tiered_start(env):
    cfg, r, rc, result, _ = _crowning(env, increment=True)
    start = r.state.king_share_bp
    b = rc.epoch_start_block + cfg.round.epoch_blocks
    m = TrainingManifest(
        round_id=str(b + 5), created_block=b + 5, contract_digest="c", base_arch_digest="d",
        eval_dataset="e", prev_round_id=rc.round_id,
        entries=[TrainedEntry(NEW, 7, "king", CID, KP, "d", b - 10, funded_by="operator")])
    r._v2_settlement_economics(m, [], 1, None, NEW)
    assert r.state.king_slashed and r.state.king_share_bp == start     # ladder kept
    assert r._emission_policy(b).king_share == pytest.approx(0.01)
    assert "crown_lcb" not in r._settlement_ladder[1]


def test_crowning_ladder_record_drops_nothing_for_a_dethrone_without_lcb(env):
    # a NaN LCB on a dethrone (impossible under the bar; defensive) is paid at
    # the lowest tier and recorded as such
    cfg, gate, era = env
    b = gate + 20 * era
    r = _runner(cfg, ChampionState(king_hotkey=NEW, king_uid=7, king_since_block=b))
    out = SimpleNamespace(transition=SimpleNamespace(dethroned=True, new_king_hotkey=NEW),
                          result=SimpleNamespace(lcb=float("nan")), lcb_units="increment")
    from .test_dec55_validator import _manifest

    r._v2_settlement_economics(_manifest(b), [], 1, out, KING)
    lad = r._settlement_ladder[1]
    assert (lad["crown_lcb"], lad["crown_below_floor"], lad["bp"]) == (None, True, 2500)
