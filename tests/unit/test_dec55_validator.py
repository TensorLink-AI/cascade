"""DEC-CA-0055 on the validator: the ladder vector it sets (scored path and
re-assert), the king check (pass / fail / deferred), the court and the key
slash, k = 1 promotions backed by a pass, seed epochs — and the audit
re-deriving the stamped vector (pass + tamper fail)."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from cascade.audit.checks import check_weights
from cascade.shared.bench_report import BenchEntry, BenchReport
from cascade.shared.config import load_chain_config
from cascade.shared.emission import COURT_BLOCKS
from cascade.shared.era import (
    EraSpec,
    era_for_block,
    era_length_blocks,
    min_effective_era,
    v2_check_due,
    v2_first_era,
    v2_first_warm_era,
)
from cascade.shared.manifest import (
    BenchScores,
    TrainedEntry,
    TrainingManifest,
    dump_manifest,
    format_trained_pointer,
)
from cascade.shared.receipt import RoundReceipt
from cascade.validator.cascade import CascadeController, CascadeState
from cascade.validator.loop import ValidatorRunner
from cascade.validator.state import ChampionState

ROOT = Path(__file__).resolve().parents[2]
KING, OLD = "5King", "5Old"
CID = "alice/gen@sha256:" + "a" * 64
KP = format_trained_pointer("cascade/king@sha256:" + "b" * 64)
IP = format_trained_pointer("cascade/init@sha256:" + "c" * 64)


class _Client:
    def __init__(self, block, uids):
        self.block, self.uids = block, uids
        self.vectors: list[list[float]] = []

    def current_block(self):
        return self.block

    def n_uids(self):
        return 16

    def uid_for_hotkey(self, hk):
        return self.uids.get(hk)

    def set_equal_share_weights(self, *a, **k):
        raise AssertionError("a DEC-CA-0055 vector is never the legacy equal share")

    def set_weight_vector(self, vec):
        self.vectors.append(list(vec))


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


def _manifest(boundary, *, funded_by="", prev="77", warm=""):
    return TrainingManifest(
        round_id=str(boundary + 5), created_block=boundary + 5, contract_digest="c",
        base_arch_digest="d", eval_dataset="e", prev_round_id=prev, warm_start_ckpt=warm,
        entries=[TrainedEntry(KING, 3, "king", CID, KP, "d", boundary - 10,
                              funded_by=funded_by)])


def _runner(cfg, state, *, reign_start=None):
    r = ValidatorRunner(cfg=cfg, state=state)
    if reign_start is not None:
        r.cascade = CascadeController(reign_days=1.0, state=CascadeState(
            king_hotkey=state.king_hotkey, reign_start_block=reign_start,
            clock_observed=True))
    return r


def _receipt(r, manifest, boundary, uids, vec):
    return RoundReceipt(
        round_id=manifest.round_id, status="scored", epoch_start_block=boundary,
        epoch_block_hash="0x", base_seed=int(manifest.round_id), generation_seed=0,
        training_seed=0, manifest=json.loads(dump_manifest(manifest)),
        reward_uids=tuple(uids), weights=tuple(vec), emission=r._last_emission,
        lane_b_pay=r._last_lane_b_pay, king_check=r._king_check_for(manifest.round_id),
        v2_block=r.cfg.v2.from_block)


# ── the ladder vector + the audit ────────────────────────────────────────────


def test_ladder_vector_is_set_stamped_and_rebuilt_by_the_audit(env):
    cfg, gate, era = env
    b = gate + 20 * era
    st = ChampionState(king_hotkey=KING, king_uid=3, king_since_block=gate + 2 * era,
                       king_share_bp=7000, court_hotkey=OLD, court_until_block=b + 100,
                       former_kings=(OLD, "5Older"))
    r = _runner(cfg, st)
    m = _manifest(b)
    r._v2_settlement_economics(m, [], 1, None, KING)
    c = _Client(b + 50, {KING: 3, OLD: 4, "5Older": 5})
    uids = r._reward_uids(m, None, c)
    assert uids == [3, 4]                               # the court list is not paid
    r._lane_b_block = b
    vec = r._apply_weights(c, m.round_id, uids, boundary=b)
    assert vec[3] == pytest.approx(0.7 * 0.5) and vec[4] == pytest.approx(0.05 * 0.5)
    assert vec[5] == 0.0 and sum(vec) == pytest.approx(1.0)
    assert r._last_emission["ladder"]["event"] == "none"
    rc = _receipt(r, m, b, uids, vec)
    assert check_weights(rc, cfg).status == "PASS"
    # tamper: weight, ladder state, policy — each fails the rebuild
    bad = list(vec)
    bad[3] += 0.01
    bad[cfg.scoring.burn_uid] -= 0.01
    assert check_weights(replace(rc, weights=tuple(bad)), cfg).status == "FAIL"
    em = json.loads(json.dumps(rc.emission))
    em["ladder"]["bp"] = 9500
    assert check_weights(replace(rc, emission=em), cfg).status == "FAIL"
    em = json.loads(json.dumps(rc.emission))
    em["policy"]["king_share"] = 0.95
    assert check_weights(replace(rc, emission=em), cfg).status == "FAIL"
    # a v2 settlement under the cap/ladder with no record at all fails
    assert check_weights(replace(rc, emission=None), cfg).status == "FAIL"


def test_court_stops_after_36000_blocks_and_reassert_follows(env):
    cfg, gate, era = env
    b = gate + 20 * era
    st = ChampionState(king_hotkey=KING, king_uid=3, king_since_block=b,
                       court_hotkey=OLD, court_until_block=b + COURT_BLOCKS)
    r = _runner(cfg, st)
    r.cfg = replace(r.cfg, validator=replace(r.cfg.validator, weight_set_interval_blocks=1))
    c = _Client(b + COURT_BLOCKS - 1, {KING: 3, OLD: 4})
    r._maybe_reassert_weights(c)
    assert c.vectors[-1][4] > 0
    c.block = b + COURT_BLOCKS + cfg.round.epoch_blocks
    r._last_weight_block = None
    r._maybe_reassert_weights(c)
    assert c.vectors[-1][4] == 0.0 and c.vectors[-1][3] == pytest.approx(0.25)


def test_key_slash_pays_one_percent_for_that_settlement(env):
    cfg, gate, era = env
    b = gate + 20 * era
    r = _runner(cfg, ChampionState(king_hotkey=KING, king_uid=3,
                                   king_since_block=gate + era, king_share_bp=9000))
    m = _manifest(b, funded_by="operator")
    r._v2_settlement_economics(m, [], 1, None, KING)
    assert r.state.king_slashed and r.state.king_share_bp == 9000   # ladder unchanged
    r._lane_b_block = b
    c = _Client(b, {KING: 3})
    vec = r._apply_weights(c, m.round_id, [3], boundary=b)
    assert vec[3] == pytest.approx(0.01 * 0.5)
    assert check_weights(_receipt(r, m, b, [3], vec), cfg).status == "PASS"
    # the next settlement on the king's own key: still slashed — the first
    # lapse holds for [v2] key_slash_min_eras (escalating slash, 2026-10-20)
    m2 = _manifest(b + cfg.round.epoch_blocks, funded_by="king")
    r._v2_settlement_economics(m2, [], 1, None, KING)
    assert r.state.king_slashed and r._settlement_ladder[1]["slash_reason"] == "min_window"
    # past the window, on the king's own key: back to 90 %
    b3 = b + cfg.v2.key_slash_min_eras * era
    r._v2_settlement_economics(_manifest(b3, funded_by="king"), [], 1, None, KING)
    assert not r.state.king_slashed


def test_dethrone_crowns_at_its_tier_and_seats_the_court(env):
    cfg, gate, era = env
    b = gate + 20 * era
    r = _runner(cfg, ChampionState(king_hotkey="5New", king_uid=7, king_since_block=b))
    # an increment LCB of 0.12 meets the 0.10 tier: 50 % (owner 2026-10-20)
    out = SimpleNamespace(transition=SimpleNamespace(dethroned=True, new_king_hotkey="5New"),
                          result=SimpleNamespace(lcb=0.12), lcb_units="increment")
    r._v2_settlement_economics(_manifest(b), [], 1, out, KING)
    assert r.state.court_hotkey == KING and r.state.court_until_block == b + COURT_BLOCKS
    assert r._settlement_ladder[1]["event"] == "crowned"
    pol = r._emission_policy(b)
    assert pol.king_share == 0.5 and pol.former_share == pytest.approx(0.05)


def test_carried_king_stays_capped_beside_lane_b(env):
    cfg, gate, era = env
    b = gate + 3 * era
    r = _runner(cfg, ChampionState(king_hotkey=KING, king_uid=3, king_since_block=gate - era,
                                   former_kings=(OLD,)))
    m = _manifest(b)
    r._v2_settlement_economics(m, [], 1, None, KING)
    c = _Client(b, {KING: 3, OLD: 4})
    uids = r._reward_uids(m, None, c)
    assert uids == [3, 4]                               # the court vector, not the ladder
    r._lane_b_block = b
    vec = r._apply_weights(c, m.round_id, uids, boundary=b)
    assert vec[3] + vec[4] == pytest.approx(0.05)
    assert check_weights(_receipt(r, m, b, uids, vec), cfg).status == "PASS"


def test_prelaunch_burn_waits_for_the_vote(env):
    cfg, gate, era = env
    b = gate - 50 * era
    r = _runner(cfg, ChampionState(king_hotkey=KING, king_uid=3))
    c = _Client(b, {KING: 3})
    r._apply_weights(c, "x", [3], boundary=b)
    assert r._last_emission is None                     # not voted: the legacy vector
    r.cfg = replace(r.cfg, v2=replace(r.cfg.v2, prelaunch_burn_from_block=b - era))
    vec = r._apply_weights(c, "x", [3], boundary=b)
    assert vec[3] == pytest.approx(0.05)
    assert r._last_emission["policy"]["prelaunch_burn"] == 0.95


# ── the king check ───────────────────────────────────────────────────────────


def _due_boundary(cfg, gate, era):
    floor = gate + cfg.v2.random_init_eras * era
    return next(x for x in range(floor, floor + 10 * era, cfg.round.epoch_blocks)
                if v2_check_due(cfg, boundary=x, reign_start_block=0))


def _report(king_better=True, with_init=True):
    good, bad = BenchScores(*[1.0] * 6), BenchScores(*[2.0] * 6)
    entries = [BenchEntry(role="king", size="m", miner_hotkey=KING, miner_uid=3,
                          trained_pointer=KP, scores=good if king_better else bad)]
    if with_init:
        entries.append(BenchEntry(role="init", size="m", miner_hotkey="", miner_uid=-1,
                                  trained_pointer=IP, scores=BenchScores(*[1.5] * 6)))
    return BenchReport(round_id="77", created_block=1, entries=tuple(entries))


@pytest.fixture
def checked(env, monkeypatch):
    cfg, gate, era = env
    b = _due_boundary(cfg, gate, era)
    import cascade.eval.koth as koth

    pool = {"wins": True}
    monkeypatch.setattr(koth, "evaluate_round", lambda *a, **k: SimpleNamespace(
        inconclusive=False, lcb=0.5 if pool["wins"] else 0.0, margin=0.2,
        challenger_wins_round=pool["wins"]))

    def make(report):
        r = _runner(cfg, ChampionState(king_hotkey=KING, king_uid=3,
                                       king_since_block=gate + era, king_pointer=KP),
                    reign_start=gate)
        r._fetch_bench_report = lambda rid, **k: report
        r._evaluate = lambda entry, windows, training: []
        return r
    return cfg, gate, era, b, make, pool


def test_king_check_pass_climbs_and_backs_one_promotion(checked):
    import hashlib

    from cascade.shared.promotion import PromotedMember, PromotionRecord
    cfg, gate, era, b, make, _ = checked
    r = make(_report())
    r._v2_settlement_economics(_manifest(b), [], 1, None, KING)
    kc = r._king_check_for(str(b + 5))
    assert kc["result"] == "pass" and kc["king_pointer"] == KP
    assert r.state.king_share_bp == 6000
    assert r.state.king_check_pass["block"] == b
    assert r.cascade.state.reign_start_block == b      # the clock restarts here
    # the promotion it backs: k = 1, plain min_effective_era, the passing pointer
    eff = min_effective_era(cfg.round, b + 10)
    rec = PromotionRecord(generation=1, king_hotkey=KING, fired_round="notice", fired_block=b + 10,
                          effective_era=eff, members=(PromotedMember(KP, "m", "77", 0.5),))
    r.verify_signatures = False  # this test exercises the state/notice rules
    r._promotion_check_manifest = TrainingManifest("notice", b + 10, "c", "d", "e",
        promotion_notices={"1": hashlib.sha256(rec.canonical_body()).hexdigest()})
    assert eff >= v2_first_warm_era(cfg)
    assert r._v2_record(rec) and r._v2_promotion_reason(rec, attesting=True) is None
    two = SimpleNamespace(member_ids=lambda: (KP, IP), fired_block=b + 10, effective_era=eff)
    assert "k = 1" in r._v2_promotion_reason(two, attesting=True)
    other = SimpleNamespace(member_ids=lambda: (IP,), fired_block=b + 10, effective_era=eff)
    assert "passing check" in r._v2_promotion_reason(other, attesting=True)
    late = SimpleNamespace(member_ids=lambda: (KP,), fired_block=b + 10, effective_era=eff + 1)
    assert "min_effective_era" in r._v2_promotion_reason(late, attesting=True)
    early = SimpleNamespace(member_ids=lambda: (KP,), fired_block=b - 10,
                            effective_era=min_effective_era(cfg.round, b - 10))
    assert "before the passing check" in r._v2_promotion_reason(early, attesting=True)


def test_king_check_fail_drops_to_ten_percent(checked):
    cfg, gate, era, b, make, pool = checked
    r = make(_report(king_better=False))
    r.state = replace(r.state, king_share_bp=8000)
    r._v2_settlement_economics(_manifest(b), [], 1, None, KING)
    kc = r._king_check_for(str(b + 5))
    assert kc["result"] == "fail" and len(kc["reasons"]) == 6
    assert r.state.king_share_bp == 1000 and r.state.king_check_pass is None
    assert r.cascade.state.reign_start_block == b
    # the pool half alone can fail it too
    pool["wins"] = False
    r = make(_report())
    r._v2_settlement_economics(_manifest(b), [], 1, None, KING)
    assert r._king_check_for(str(b + 5))["result"] == "fail"
    rec = SimpleNamespace(member_ids=lambda: (KP,), fired_block=b + 10,
                          effective_era=min_effective_era(cfg.round, b + 10))
    assert "not backed" in r._v2_promotion_reason(rec, attesting=True)


def test_king_check_defers_without_the_init_entry(checked):
    cfg, gate, era, b, make, _ = checked
    r = make(_report(with_init=False))
    r._v2_settlement_economics(_manifest(b), [], 1, None, KING)
    kc = r._king_check_for(str(b + 5))
    assert kc["result"] == "deferred" and "init" in kc["reasons"][0]
    assert r.state.king_share_bp == 5000                 # unchanged (50 %)
    assert r.cascade.state.reign_start_block == gate     # clock not reset
    # no report at all ⇒ deferred too
    r = make(None)
    r._v2_settlement_economics(_manifest(b), [], 1, None, KING)
    assert r._king_check_for(str(b + 5))["result"] == "deferred"


def test_king_check_not_due_inside_the_window(checked):
    cfg, gate, era, b, make, _ = checked
    r = make(_report())
    early = gate + 3 * era
    r._v2_settlement_economics(_manifest(early), [], 1, None, KING)
    assert r._king_check_for(str(early + 5)) is None


def test_audit_follows_the_check_into_the_ladder(checked):
    cfg, gate, era, b, make, _ = checked
    r = make(_report())
    m = _manifest(b)
    r._v2_settlement_economics(m, [], 1, None, KING)
    r._lane_b_block = b
    c = _Client(b, {KING: 3})
    vec = r._apply_weights(c, m.round_id, [3], boundary=b)
    rc = _receipt(r, m, b, [3], vec)
    assert vec[3] == pytest.approx(0.6 * 0.5)
    assert check_weights(rc, cfg).status == "PASS"
    # a ladder step the stamped check does not justify
    assert check_weights(replace(rc, king_check=dict(rc.king_check, result="fail")),
                         cfg).status == "FAIL"


# ── seed epochs + the dethrone-safe staged promotion ─────────────────────────


def test_seed_epoch_seed_block_and_pointer_era(env):
    cfg, gate, era = env
    r = _runner(cfg, ChampionState(king_hotkey=KING), reign_start=gate)
    first = v2_first_era(cfg)
    mid = era_for_block(cfg.round, (first + 20) * era)
    assert r._era_seed_block(mid) == first * era - era   # the first v2 era's seed block
    assert r._pointer_era(mid) == first
    warm = first + 25
    r.cascade.state = replace(r.cascade.state, generation=1, members=(KP,),
                              generation_effective_era=warm)
    later = era_for_block(cfg.round, (warm + 3) * era)
    assert r._era_seed_block(later) == (warm - 1) * era and r._pointer_era(later) == warm
    # ladder off (testnet): every era its own seeds
    r.cfg = replace(r.cfg, v2=replace(r.cfg.v2, king_ladder=False))
    assert r._era_seed_block(later) == later.seed_block
    assert r._pointer_era(later) == later.index


def test_mid_epoch_init_change_is_refused(env):
    cfg, gate, era = env
    first = v2_first_era(cfg)
    r = _runner(cfg, ChampionState(king_hotkey=KING, epoch_init={"epoch": first, "init": ""}),
                reign_start=gate)
    r._fetch_latest_promotion_record = lambda: None
    era_spec = EraSpec.from_json(json.loads(json.dumps(
        era_for_block(cfg.round, (first + 5) * era).to_json())))
    m = replace(_manifest((first + 5) * era + cfg.round.epoch_blocks), warm_start_ckpt=IP)
    why = r._check_era_init(m, era_spec)
    assert why is not None and "era_init_mismatch" in why


def test_dethrone_keeps_a_staged_v2_promotion(env):
    cfg, gate, era = env
    r = _runner(cfg, ChampionState(king_hotkey=KING), reign_start=gate)
    warm = v2_first_warm_era(cfg) + 3
    r.cascade.stage_promotion(generation=1, members=(KP,), effective_era=warm)
    assert r._v2_pending_kept()
    r.cascade.note_dethrone("5New", block=gate + 5 * era, keep_pending=r._v2_pending_kept())
    assert r.cascade.state.pending_generation == 1 and r.cascade.state.pending_members == (KP,)
