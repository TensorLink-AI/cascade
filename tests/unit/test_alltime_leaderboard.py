"""All-time top-k warm-start leaderboard (DEC-CA-0044): the fixed population
ranked on the suite-weighted score, rank-insertion replacement, the 24h
announce -> notice -> fire cycle on the trainer, the block-gated validator
envelope (any-signed-report provenance, weighted floor, notice-period
spacing), and the miner-facing surfaces (status docs, `cascade round` /
`cascade heat` / `cascade leaderboard`, `--warm-start upcoming`).
"""

from __future__ import annotations

import io
import json
import math
from dataclasses import replace
from types import SimpleNamespace

import pytest

from cascade.shared.bench_report import BenchEntry, BenchReport
from cascade.shared.manifest import BenchScores
from cascade.shared.promotion import (
    LEADERBOARD_KEY,
    PromotedMember,
    annotate_upcoming,
    build_leaderboard_doc,
    publish_leaderboard,
    upcoming_from_doc,
)
from cascade.trainer.promotion import (
    LeaderEntry,
    PendingChange,
    TrainerPromotion,
    admit_leader,
    alltime_members,
)
from cascade.validator.cascade import (
    DEFAULT_SUITE_WEIGHTS,
    CascadeController,
    CascadeState,
    cascade_score,
    weighted_cascade_score,
)

DAY = 7_200          # blocks; one 7200-block round with round_cfg=None
NOTICE = 2 * DAY     # 24h at the 12h cadence = 2 rounds; tests use 2 "days"


# ── the weighted score ───────────────────────────────────────────────────────


def test_uniform_weights_reproduce_the_six_number_geomean():
    args = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
    assert math.isclose(weighted_cascade_score(*args, weights=(1, 1, 1)),
                        cascade_score(*args), rel_tol=1e-12)
    assert math.isclose(weighted_cascade_score(*args, weights=(1 / 3, 1 / 3, 1 / 3)),
                        cascade_score(*args), rel_tol=1e-12)


def test_default_weights_are_50_25_25_gift_boom_time():
    assert DEFAULT_SUITE_WEIGHTS == (0.5, 0.25, 0.25)
    # Suite scores: gift = geomean(crps, mase) etc.; weighted geomean over suites.
    gift, boom, tm = 0.8, 1.2, 1.0
    got = weighted_cascade_score(gift, gift, boom, boom, tm, tm)
    assert math.isclose(got, gift ** 0.5 * boom ** 0.25 * tm ** 0.25, rel_tol=1e-12)
    # GIFT-Eval moves the score twice as much as BOOM does.
    base = weighted_cascade_score(1, 1, 1, 1, 1, 1)
    gift_up = weighted_cascade_score(1.1, 1.1, 1, 1, 1, 1)
    boom_up = weighted_cascade_score(1, 1, 1.1, 1.1, 1, 1)
    assert math.isclose(math.log(gift_up / base), 2 * math.log(boom_up / base), rel_tol=1e-9)


def test_degenerate_weights_fall_back_to_uniform():
    args = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
    assert weighted_cascade_score(*args, weights=(0, 0, 0)) == cascade_score(*args)
    assert weighted_cascade_score(*args, weights=(-1, 0, 0)) == cascade_score(*args)


# ── the leaderboard (pure) ───────────────────────────────────────────────────


def _leader(cid, score, *, block=0, rnd="r1", hk="hk", role="king"):
    return LeaderEntry(checkpoint_id=cid, size="toto2-4m", hotkey=hk, role=role,
                       source_round=rnd, created_block=block,
                       gifteval_crps=score, gifteval_mase=score, boom_crps=score,
                       boom_mase=score, time_crps=score, time_mase=score, score=score)


def test_better_checkpoint_takes_the_rank_it_beats_and_the_last_drops_out():
    board = (_leader("a", 1.00), _leader("b", 1.02), _leader("c", 1.04))
    board, rank = admit_leader(board, _leader("new", 1.01), 3)
    assert rank == 2
    assert [e.checkpoint_id for e in board] == ["a", "new", "b"]   # 2nd -> 3rd, 3rd out
    board, rank = admit_leader(board, _leader("best", 0.9), 3)
    assert rank == 1 and [e.checkpoint_id for e in board] == ["best", "a", "new"]


def test_worse_than_a_full_board_and_equal_scores_do_not_enter():
    board = (_leader("a", 1.00), _leader("b", 1.02), _leader("c", 1.04))
    assert admit_leader(board, _leader("worse", 1.05), 3) == (board, None)
    # Equal is not better: a tie with the last member does not enter, and a
    # tie with 2nd ranks BELOW it (the incumbent keeps its rank).
    assert admit_leader(board, _leader("tie-last", 1.04), 3) == (board, None)
    tied, rank = admit_leader(board, _leader("tie", 1.02), 3)
    assert rank == 3 and [e.checkpoint_id for e in tied] == ["a", "b", "tie"]
    # An already-listed pointer is never re-admitted; NaN never enters.
    assert admit_leader(board, _leader("a", 0.5), 3) == (board, None)
    assert admit_leader(board, _leader("nan", float("nan")), 3) == (board, None)


def test_board_fills_up_to_k_in_rank_order():
    board = ()
    for cid, sc in (("x", 1.03), ("y", 1.01), ("z", 1.02)):
        board, rank = admit_leader(board, _leader(cid, sc), 3)
        assert rank is not None
    assert [e.checkpoint_id for e in board] == ["y", "z", "x"]
    board, rank = admit_leader(board, _leader("w", 1.10), 3)   # full, worse
    assert rank is None and len(board) == 3


def test_members_are_the_top_k_within_the_quality_epsilon():
    board = (_leader("a", 1.00), _leader("b", 1.03), _leader("c", 1.10))
    got = alltime_members(board, k_max=3, quality_epsilon=0.05)
    assert [e.checkpoint_id for e in got] == ["a", "b"]         # c is >5% off: not legal
    assert [e.checkpoint_id for e in alltime_members(board, k_max=1, quality_epsilon=0.05)] == ["a"]
    assert alltime_members((), k_max=3, quality_epsilon=0.05) == []


# ── the engine ───────────────────────────────────────────────────────────────


def _scores(v):
    return BenchScores(gifteval_crps=v, gifteval_mase=v, boom_crps=v,
                       boom_mase=v, time_crps=v, time_mase=v)


def _report(round_id, block, scored: dict[str, tuple[float, str, str]]):
    return BenchReport(
        round_id=round_id, created_block=block,
        entries=tuple(
            BenchEntry(role=role, size="toto2-4m", miner_hotkey=hk, miner_uid=0,
                       trained_pointer=ptr, scores=_scores(v))
            for ptr, (v, hk, role) in scored.items()
        ),
    )


def _manifest(round_id, warm_start_ckpt=""):
    return SimpleNamespace(round_id=round_id, warm_start_ckpt=warm_start_ckpt)


def _engine(tmp_path, *, k_max=3, alltime_from_block=1, notice=NOTICE):
    return TrainerPromotion(
        reign_threshold=5, k_max=k_max, quality_epsilon=0.05,
        state_path=tmp_path / "trainer_promotion.json",
        pointer_path=tmp_path / "warm_start_init.json",
        alltime_from_block=alltime_from_block, notice_blocks=notice,
    )


def test_engine_admits_every_signed_bench_into_the_leaderboard(tmp_path):
    eng = _engine(tmp_path)
    eng.note_round("hkKing", epoch_block=0)
    # A round trained off a NON-live init contributes no reign candidate but
    # still enters the all-time population: the board is all-time by definition.
    eng.generation, eng.members = 1, (PromotedMember("live", "toto2-4m", "r0", 1.0),)
    assert eng.record_bench(_manifest("r1", "stale"), _report("r1", 1 * DAY, {
        "ptr-a": (0.98, "hkA", "king"), "ptr-b": (1.01, "hkB", "challenger")})) == 0
    assert [e.checkpoint_id for e in eng.leaderboard] == ["ptr-a", "ptr-b"]
    # A king change resets the reign candidates, never the all-time board.
    eng.note_round("hkOther", epoch_block=2 * DAY)
    assert eng.candidates == () and len(eng.leaderboard) == 2


def test_announce_then_fire_after_the_notice_period(tmp_path):
    eng = _engine(tmp_path)
    eng.note_round("hkKing", epoch_block=0)
    eng.record_bench(_manifest("r1"), _report("r1", 1 * DAY, {
        "ptr-a": (1.00, "hkA", "king"), "ptr-b": (1.02, "hkB", "challenger")}))
    # Boundary 2: target {a, b} != live {} -> ANNOUNCED, nothing fires.
    assert eng.maybe_promote(epoch_block=2 * DAY, round_id="r2") is None
    up = eng.upcoming()
    assert up is not None and up["generation"] == 1
    assert [m["checkpoint_id"] for m in up["members"]] == ["ptr-a", "ptr-b"]
    assert up["announced_block"] == 2 * DAY and up["effective_block"] == 2 * DAY + NOTICE
    assert eng.generation == 0 and eng.init_for_epoch(0) is None   # field unchanged
    # Inside the notice window: held, still announced.
    assert eng.maybe_promote(epoch_block=3 * DAY, round_id="r3") is None
    assert eng.upcoming() is not None
    # Effective block reached: the FROZEN set fires as generation 1.
    rec = eng.maybe_promote(epoch_block=4 * DAY, round_id="r4")
    assert rec is not None and rec.generation == 1
    assert rec.member_ids() == ("ptr-a", "ptr-b") and rec.fired_block == 4 * DAY
    assert eng.upcoming() is None and eng.reign_start_block == 4 * DAY
    assert eng.init_for_epoch(0) == ("ptr-a", "toto2-4m")
    assert eng.init_for_epoch(1) == ("ptr-b", "toto2-4m")          # rotation unchanged
    obj = json.loads((tmp_path / "warm_start_init.json").read_text())
    assert obj["rule"] == "alltime_top_k" and obj["generation"] == 1
    # Same population, no change -> nothing announced.
    assert eng.maybe_promote(epoch_block=6 * DAY, round_id="r6") is None
    assert eng.upcoming() is None


def test_announced_set_is_frozen_and_the_better_arrival_waits_its_turn(tmp_path):
    eng = _engine(tmp_path, k_max=2)
    eng.note_round("hkKing", epoch_block=0)
    eng.record_bench(_manifest("r1"), _report("r1", 1 * DAY, {
        "ptr-a": (1.00, "hkA", "king"), "ptr-b": (1.02, "hkB", "challenger")}))
    eng.maybe_promote(epoch_block=2 * DAY, round_id="r2")            # announce {a, b}
    # A better checkpoint lands during the notice: it takes rank 2 on the
    # board (b drops out) but the ANNOUNCED set is frozen.
    eng.record_bench(_manifest("r3"), _report("r3", 3 * DAY, {"ptr-c": (1.01, "hkC", "king")}))
    assert [e.checkpoint_id for e in eng.leaderboard] == ["ptr-a", "ptr-c"]
    rec = eng.maybe_promote(epoch_block=4 * DAY, round_id="r4")
    assert rec is not None and rec.member_ids() == ("ptr-a", "ptr-b")
    # …then the new target {a, c} is announced at the next boundary and fires
    # a notice period later: generation 2.
    assert eng.maybe_promote(epoch_block=5 * DAY, round_id="r5") is None
    assert [m["checkpoint_id"] for m in eng.upcoming()["members"]] == ["ptr-a", "ptr-c"]
    assert eng.maybe_promote(epoch_block=6 * DAY, round_id="r6") is None   # inside notice
    rec2 = eng.maybe_promote(epoch_block=7 * DAY, round_id="r7")
    assert rec2 is not None and rec2.generation == 2 and rec2.member_ids() == ("ptr-a", "ptr-c")


def test_frozen_set_outside_the_envelope_is_reannounced(tmp_path):
    # A MUCH better checkpoint moves the floor so the announced member no
    # longer sits within epsilon of the best: firing it would be rejected by
    # every validator, so the engine re-announces the current target instead.
    eng = _engine(tmp_path, k_max=2)
    eng.note_round("hkKing", epoch_block=0)
    eng.record_bench(_manifest("r1"), _report("r1", 1 * DAY, {"ptr-a": (1.00, "hkA", "king")}))
    eng.maybe_promote(epoch_block=2 * DAY, round_id="r2")            # announce {a}
    eng.record_bench(_manifest("r3"), _report("r3", 3 * DAY, {"ptr-z": (0.80, "hkZ", "king")}))
    assert eng.maybe_promote(epoch_block=4 * DAY, round_id="r4") is None
    up = eng.upcoming()
    assert [m["checkpoint_id"] for m in up["members"]] == ["ptr-z"]
    assert up["announced_block"] == 4 * DAY and up["effective_block"] == 4 * DAY + NOTICE
    rec = eng.maybe_promote(epoch_block=6 * DAY, round_id="r6")
    assert rec is not None and rec.member_ids() == ("ptr-z",)


def test_dethrone_inside_the_notice_window_holds_until_spaced(tmp_path):
    eng = _engine(tmp_path)
    eng.note_round("hkKing", epoch_block=0)
    eng.record_bench(_manifest("r1"), _report("r1", 1 * DAY, {"ptr-a": (1.00, "hkA", "king")}))
    eng.maybe_promote(epoch_block=2 * DAY, round_id="r2")            # effective 4*DAY
    eng.note_round("hkNew", epoch_block=3 * DAY)                      # dethrone resets anchor
    assert eng.maybe_promote(epoch_block=4 * DAY, round_id="r4") is None   # 1 round since anchor
    assert eng.upcoming() is not None                                 # still announced
    rec = eng.maybe_promote(epoch_block=5 * DAY, round_id="r5")       # 2 rounds = NOTICE
    assert rec is not None and rec.fired_block == 5 * DAY


def test_legacy_rule_still_selects_before_the_activation_block(tmp_path):
    eng = _engine(tmp_path, k_max=2, alltime_from_block=100 * DAY)
    eng.note_round("hkKing", epoch_block=0)
    eng.record_bench(_manifest("r1"), _report("r1", 1 * DAY, {
        "ptr-a": (1.00, "hkA", "king"), "ptr-b": (1.02, "hkB", "challenger")}))
    assert eng.maybe_promote(epoch_block=2 * DAY, round_id="r2") is None
    assert eng.upcoming() is None                                     # nothing announced
    rec = eng.maybe_promote(epoch_block=5 * DAY, round_id="r5")       # reign ripe: legacy fire
    assert rec is not None and rec.member_ids() == ("ptr-a", "ptr-b")
    assert len(eng.leaderboard) == 2                                  # board kept in shadow
    assert json.loads((tmp_path / "warm_start_init.json").read_text())["rule"] == "reign_scoped"


def test_leaderboard_ranks_on_the_weighted_score(tmp_path):
    eng = _engine(tmp_path)
    eng.note_round("hkKing", epoch_block=0)
    # gift-strong beats boom/time-strong under 50/25/25 although the uniform
    # geomean would rank them the other way round.
    gift_strong = BenchScores(gifteval_crps=0.8, gifteval_mase=0.8, boom_crps=1.1,
                              boom_mase=1.1, time_crps=1.1, time_mase=1.1)
    other_strong = BenchScores(gifteval_crps=1.1, gifteval_mase=1.1, boom_crps=0.85,
                               boom_mase=0.85, time_crps=0.85, time_mase=0.85)
    assert cascade_score(*_six(gift_strong)) > cascade_score(*_six(other_strong))
    rep = BenchReport(round_id="r1", created_block=DAY, entries=(
        BenchEntry(role="king", size="toto2-4m", miner_hotkey="hkA", miner_uid=0,
                   trained_pointer="gift", scores=gift_strong),
        BenchEntry(role="challenger", size="toto2-4m", miner_hotkey="hkB", miner_uid=1,
                   trained_pointer="other", scores=other_strong)))
    eng.admit_report(rep)
    assert [e.checkpoint_id for e in eng.leaderboard] == ["gift", "other"]
    assert math.isclose(eng.leaderboard[0].score, weighted_cascade_score(*_six(gift_strong)))


def _six(s):
    return (s.gifteval_crps, s.gifteval_mase, s.boom_crps, s.boom_mase, s.time_crps, s.time_mase)


def test_engine_persists_the_board_and_the_announcement(tmp_path):
    eng = _engine(tmp_path)
    eng.note_round("hkKing", epoch_block=0)
    eng.record_bench(_manifest("r1"), _report("r1", 1 * DAY, {"ptr-a": (1.00, "hkA", "king")}))
    eng.maybe_promote(epoch_block=2 * DAY, round_id="r2")
    eng.mark_leaderboard_seeded()
    again = TrainerPromotion.load(
        reign_threshold=5, k_max=3, quality_epsilon=0.05,
        state_path=tmp_path / "trainer_promotion.json",
        pointer_path=tmp_path / "warm_start_init.json",
        alltime_from_block=1, notice_blocks=NOTICE)
    assert [e.checkpoint_id for e in again.leaderboard] == ["ptr-a"]
    assert again.pending_change == PendingChange(
        generation=1, members=(PromotedMember("ptr-a", "toto2-4m", "r1", 1.0),),
        announced_round="r2", announced_block=2 * DAY, effective_block=2 * DAY + NOTICE)
    assert again.leaderboard_seeded is True
    rec = again.maybe_promote(epoch_block=4 * DAY, round_id="r4")     # fires after reload
    assert rec is not None and rec.generation == 1


def test_reload_reranks_the_board_under_new_weights(tmp_path):
    eng = _engine(tmp_path)
    eng.note_round("hkKing", epoch_block=0)
    gift_strong = BenchScores(gifteval_crps=0.8, gifteval_mase=0.8, boom_crps=1.1,
                              boom_mase=1.1, time_crps=1.1, time_mase=1.1)
    other_strong = BenchScores(gifteval_crps=1.1, gifteval_mase=1.1, boom_crps=0.85,
                               boom_mase=0.85, time_crps=0.85, time_mase=0.85)
    eng.admit_report(BenchReport(round_id="r1", created_block=DAY, entries=(
        BenchEntry(role="king", size="toto2-4m", miner_hotkey="hkA", miner_uid=0,
                   trained_pointer="gift", scores=gift_strong),
        BenchEntry(role="challenger", size="toto2-4m", miner_hotkey="hkB", miner_uid=1,
                   trained_pointer="other", scores=other_strong))))
    again = TrainerPromotion.load(
        reign_threshold=5, k_max=3, quality_epsilon=0.05,
        state_path=tmp_path / "trainer_promotion.json",
        pointer_path=tmp_path / "warm_start_init.json",
        suite_weights=(1, 1, 1))
    assert [e.checkpoint_id for e in again.leaderboard] == ["other", "gift"]


def test_leaderboard_rows_and_public_doc_round_trip():
    rows = [{"rank": 1, **_leader("a", 1.0).to_json()}]
    up = {"generation": 2, "members": [{"checkpoint_id": "a", "size": "s",
                                        "source_round": "r1", "score": 1.0}],
          "announced_round": "r9", "announced_block": 100, "effective_block": 7300}
    doc = build_leaderboard_doc(as_of="t", rule_active=True, k=3, weights=(0.5, 0.25, 0.25),
                                notice_blocks=7200, entries=rows, live_generation=1,
                                live_members=["b"], upcoming=up, activation_block=1)
    assert doc["rule"] == "alltime_top_k" and doc["weights"] == {
        "gifteval": 0.5, "boom": 0.25, "time": 0.25}
    assert upcoming_from_doc(doc)["effective_block"] == 7300
    assert upcoming_from_doc({"warm_start": {"upcoming": up}})["generation"] == 2
    assert upcoming_from_doc({"warm_start": {"upcoming": {"members": []}}}) is None
    assert upcoming_from_doc({"upcoming": {"members": [{"checkpoint_id": "a"}]}}) is None
    assert upcoming_from_doc(None) is None
    ann = annotate_upcoming(up, now_block=7000, seconds_per_block=12.0, now_s=0.0)
    assert ann["blocks_remaining"] == 300 and ann["effective_at"] == "1970-01-01T01:00:00+00:00"

    class _Store:
        def __init__(self):
            self.texts, self.acls = {}, {}

        def put_text(self, key, text, content_type="", acl=None):
            self.texts[key], self.acls[key] = text, acl

    st = _Store()
    assert publish_leaderboard(st, doc) == LEADERBOARD_KEY
    assert st.acls[LEADERBOARD_KEY] == "public-read"
    assert json.loads(st.texts[LEADERBOARD_KEY])["upcoming"]["generation"] == 2


# ── the validator envelope under the all-time rule ───────────────────────────


def test_controller_spacing_predicate():
    ctl = CascadeController(reign_days=5, state=CascadeState(
        king_hotkey="hk", reign_start_block=1000, clock_observed=True))
    assert ctl.is_spaced(block=1000 + NOTICE, min_blocks=NOTICE)
    assert not ctl.is_spaced(block=1000 + NOTICE - 1, min_blocks=NOTICE)
    assert not CascadeController(reign_days=5).is_spaced(block=10 ** 9, min_blocks=1)


@pytest.fixture
def alltime_cfg(cfg):
    return replace(cfg, scoring=replace(
        cfg.scoring, cascade_alltime_from_block=1, cascade_notice_blocks=NOTICE,
        cascade_top_k=3, cascade_quality_epsilon=0.05))


def _ws_helpers():
    from tests.unit import test_warm_start as ws

    return ws


def test_alltime_rule_accepts_a_member_from_a_dead_reign(alltime_cfg, tmp_path):
    ws = _ws_helpers()
    # The bench report scoring the member has created_block=10, long before
    # this reign's anchor: rejected under the reign-scoped rule
    # (test_out_of_reign_member_rejected_when_clock_can_attest), accepted here.
    store = ws._promotion_store([(ws.PTR2, 1.0)], generation=2)
    ctl = CascadeController(reign_days=5, state=CascadeState(
        king_hotkey="hk0", reign_start_block=5 * DAY, generation=1, members=(ws.PTR,),
        clock_observed=True))
    r = ws._validator(alltime_cfg, tmp_path, cascade=ctl, store=store)
    assert r.check_manifest(ws._manifest(
        alltime_cfg, warm_start_ckpt=ws.PTR2, created_block=10 * DAY + 10)) is None
    assert ctl.state.generation == 2 and ctl.state.members == (ws.PTR2,)


def test_alltime_rule_enforces_the_notice_period_not_the_reign(alltime_cfg, tmp_path):
    ws = _ws_helpers()
    store = ws._promotion_store([(ws.PTR2, 1.0)], generation=2)
    ctl = CascadeController(reign_days=5, state=CascadeState(
        king_hotkey="hk0", reign_start_block=0, generation=1, members=(ws.PTR,),
        clock_observed=True))
    r = ws._validator(alltime_cfg, tmp_path, cascade=ctl, store=store)
    # One round after the anchor: inside the notice period -> early.
    reason = r.check_manifest(ws._manifest(
        alltime_cfg, warm_start_ckpt=ws.PTR2, created_block=1 * DAY + 10))
    assert reason is not None and "warm_start_promotion_early" in reason
    assert "notice period" in reason
    # Two rounds (= NOTICE blocks) after the anchor: accepted although the
    # 5-round reign clock is NOT ripe — the notice period is the predicate.
    assert r.check_manifest(ws._manifest(
        alltime_cfg, warm_start_ckpt=ws.PTR2, created_block=2 * DAY + 10)) is None


def test_alltime_rule_measures_the_quality_floor_on_the_weighted_score(alltime_cfg, tmp_path):
    ws = _ws_helpers()
    from cascade.shared.bench_report import bench_report_key, dump_bench_report
    from cascade.shared.promotion import (
        PromotionRecord,
        dump_promotion_record,
        promotion_index_key,
        promotion_record_key,
    )

    # Member B trails A by 8% on BOOM and TIME only: uniform geomean 1.0526
    # (>5%, would fail the legacy floor), weighted 1.039 (<5%, legal).
    a = BenchScores(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    b = BenchScores(1.0, 1.0, 1.08, 1.08, 1.08, 1.08)
    assert cascade_score(*_six(b)) > 1.05 > weighted_cascade_score(*_six(b))
    report = BenchReport(round_id="r1", created_block=10, entries=(
        BenchEntry(role="king", size="toto2-4m", miner_hotkey="hk", miner_uid=0,
                   trained_pointer=ws.PTR, scores=a),
        BenchEntry(role="challenger", size="toto2-4m", miner_hotkey="hk", miner_uid=1,
                   trained_pointer=ws.PTR2, scores=b)))
    record = PromotionRecord(generation=1, king_hotkey="hk0", fired_round="r1", fired_block=0,
                             members=(PromotedMember(ws.PTR, "toto2-4m", "r1", 1.0),
                                      PromotedMember(ws.PTR2, "toto2-4m", "r1", 1.039)))
    store = ws._Store({
        promotion_index_key(): json.dumps({"latest_generation": 1}),
        promotion_record_key(1): dump_promotion_record(record),
        bench_report_key("r1"): dump_bench_report(report)})
    ctl = CascadeController(reign_days=5, state=CascadeState(
        king_hotkey="hk0", reign_start_block=0, clock_observed=True))
    r = ws._validator(alltime_cfg, tmp_path, cascade=ctl, store=store)
    assert r.check_manifest(ws._manifest(
        alltime_cfg, warm_start_ckpt=ws.PTR2, created_block=2 * DAY + 10)) is None
    # The identical record under the reign-scoped rule fails the uniform floor.
    legacy = replace(alltime_cfg, scoring=replace(alltime_cfg.scoring, cascade_alltime_from_block=0))
    ctl2 = CascadeController(reign_days=5, state=CascadeState(
        king_hotkey="hk0", reign_start_block=0, clock_observed=True))
    r2 = ws._validator(legacy, tmp_path, cascade=ctl2, store=store)
    reason = r2.check_manifest(ws._manifest(
        legacy, warm_start_ckpt=ws.PTR2, created_block=5 * DAY + 10))
    assert reason is not None and "warm_start_member_below_floor" in reason


# ── config ───────────────────────────────────────────────────────────────────


def test_config_carries_the_alltime_knobs(cfg):
    from cascade.shared.config import (
        cascade_alltime_active,
        cascade_suite_weights,
        load_chain_config,
    )

    assert cfg.scoring.cascade_alltime_from_block == 0          # mainnet: unarmed
    assert cfg.scoring.cascade_notice_blocks == 7200            # 24h at 12 s/block
    assert cascade_suite_weights(cfg.scoring) == (0.5, 0.25, 0.25)
    assert not cascade_alltime_active(cfg.scoring, 10 ** 9)
    armed = replace(cfg.scoring, cascade_alltime_from_block=500)
    assert cascade_alltime_active(armed, 500) and not cascade_alltime_active(armed, 499)
    assert not cascade_alltime_active(armed, None)
    testnet = load_chain_config("chain.testnet.toml")
    assert testnet.scoring.cascade_alltime_from_block == 1      # testnet: armed


# ── miner surfaces ───────────────────────────────────────────────────────────

PTR_A = "metro-v1:trained:hippius:cascade/ckpt-r9-king-toto2-4m@sha256:" + "a" * 64
PTR_B = "metro-v1:trained:hippius:cascade/ckpt-r9-chal-toto2-4m@sha256:" + "b" * 64


def _upcoming():
    return {"generation": 5, "announced_round": "r40", "announced_block": 14_400,
            "effective_block": 21_600, "effective_at": "2026-09-18T08:30:00+00:00",
            "members": [{"checkpoint_id": PTR_A, "size": "toto2-4m", "score": 0.61},
                        {"checkpoint_id": PTR_B, "size": "toto2-4m", "score": 0.62}]}


def test_round_dashboard_shows_the_announced_change():
    from cascade.miner.dashboard import run_dashboard, upcoming_init_lines
    from cascade.shared.config import RoundConfig
    from tests.unit.test_miner_dashboard import _FakeClient, _live_doc

    rc = RoundConfig(epoch_blocks=7200, round_hours=24.0)
    out = io.StringIO()
    run_dashboard(_FakeClient(15_000), rc, "test", out=out,
                  status_fetch=lambda: _live_doc(
                      epoch_start_block=14_400,
                      warm_start={"init_checkpoint": PTR_A, "size": "toto2-4m",
                                  "generation": 4, "upcoming": _upcoming()}))
    text = out.getvalue()
    assert "warm start      this round trains from" in text
    assert "upcoming init   generation 5: cascade/ckpt-r9-king-toto2-4m@sha256:aaaaaaaaaaaa…" in text
    assert "(+1 more in rotation)" in text
    assert "takes effect at block 21,600" in text and "from now, estimated" in text
    assert "--warm-start upcoming" in text
    # Random-init round with an announced FIRST generation: the notice still shows.
    out = io.StringIO()
    run_dashboard(_FakeClient(15_000), rc, "test", out=out,
                  status_fetch=lambda: _live_doc(epoch_start_block=14_400,
                                                 warm_start={"upcoming": _upcoming()}))
    assert "upcoming init" in out.getvalue() and "warm start      this round" not in out.getvalue()
    assert upcoming_init_lines({"init_checkpoint": PTR_A}) == []
    assert upcoming_init_lines(None) == []


def test_heat_view_shows_the_announced_change():
    from cascade.miner.dashboard import render_heat
    from tests.unit.test_miner_dashboard import _heat_doc

    text = render_heat(_heat_doc(warm_start={"init_checkpoint": PTR_A, "size": "toto2-4m",
                                             "generation": 4, "upcoming": _upcoming()}))
    assert "upcoming init   generation 5" in text
    assert "takes effect at block 21,600  (~2026-09-18T08:30:00+00:00, estimated)" in text
    assert "upcoming init" not in render_heat(_heat_doc())


def test_leaderboard_view():
    from cascade.miner.dashboard import render_leaderboard

    assert "no all-time leaderboard published" in render_leaderboard(None)
    rows = [{"rank": 1, **_leader(PTR_A, 0.61, rnd="r39", role="king").to_json()},
            {"rank": 2, **_leader(PTR_B, 0.62, rnd="r40", role="challenger").to_json()}]
    doc = build_leaderboard_doc(as_of="t", rule_active=True, k=3, weights=(0.5, 0.25, 0.25),
                                notice_blocks=7200, entries=rows, live_generation=4,
                                live_members=[PTR_A], upcoming=_upcoming())
    text = render_leaderboard(doc, now_block=20_000, spb=12.0)
    assert "all-time top 3" in text
    assert "gift-eval 50% · boom 25% · time 25%" in text
    assert "ACTIVE" in text
    assert "← live" in text and "cascade/ckpt-r9-chal-toto2-4m@sha256:bbbbbbbbbbbb…" in text
    assert "upcoming init   generation 5" in text and "takes effect at block 21,600" in text
    shadow = build_leaderboard_doc(as_of="t", rule_active=False, k=3, weights=(0.5, 0.25, 0.25),
                                   notice_blocks=7200, entries=[], live_generation=0,
                                   live_members=[], upcoming=None, activation_block=9_000_000)
    text = render_leaderboard(shadow)
    assert "shadow until block 9,000,000" in text and "none announced" in text
    assert "(empty" in text


def test_score_warm_start_upcoming_resolves_the_announced_init(cfg, tmp_path, monkeypatch):
    from pathlib import Path

    from cascade.miner import score as score_mod

    ref = "cascade/ckpt-r1-challenger-toto2-4m-u5@sha256:" + "ef" * 32
    up = {**_upcoming(), "members": [{"checkpoint_id": f"metro-v1:trained:hippius:{ref}"}]}
    monkeypatch.setattr("cascade.miner.dashboard.fetch_public_round_status",
                        lambda storage, **k: {"warm_start": {"init_checkpoint": PTR_A,
                                                             "upcoming": up}})
    seen = []
    monkeypatch.setattr("cascade.shared.hippius.fetch_from_hub",
                        lambda r, dest, hub=None: (seen.append(r),
                                                   Path(dest).mkdir(parents=True, exist_ok=True), Path(dest))[-1])
    d, label = score_mod._resolve_warm_start(cfg, "upcoming", cache_dir=tmp_path)
    assert seen == [ref] and d.is_dir() and ref in label
    # No announcement anywhere -> a clear error, never a silent random init.
    monkeypatch.setattr("cascade.miner.dashboard.fetch_public_round_status",
                        lambda storage, **k: {"warm_start": {"init_checkpoint": PTR_A}})
    monkeypatch.setattr("cascade.miner.dashboard.fetch_public_leaderboard",
                        lambda storage, **k: None)
    with pytest.raises(ValueError, match="no warm-start change is announced"):
        score_mod._resolve_warm_start(cfg, "upcoming", cache_dir=tmp_path)
    # …and the leaderboard doc is the fallback source.
    monkeypatch.setattr("cascade.miner.dashboard.fetch_public_leaderboard",
                        lambda storage, **k: {"upcoming": up})
    d, _ = score_mod._resolve_warm_start(cfg, "upcoming", cache_dir=tmp_path)
    assert seen == [ref, ref]


def test_leaderboard_cli_command_prints_without_a_chain(monkeypatch, capsys):
    from cascade.miner.cli import main

    monkeypatch.setattr("cascade.miner.dashboard.fetch_public_leaderboard",
                        lambda storage, **k: build_leaderboard_doc(
                            as_of="t", rule_active=True, k=3, weights=(0.5, 0.25, 0.25),
                            notice_blocks=7200, entries=[], live_generation=0,
                            live_members=[], upcoming=_upcoming()))
    assert main(["leaderboard", "--no-chain"]) == 0
    out = capsys.readouterr().out
    assert "cascade leaderboard" in out and "upcoming init   generation 5" in out
    monkeypatch.setattr("cascade.miner.dashboard.fetch_public_leaderboard",
                        lambda storage, **k: None)
    assert main(["leaderboard", "--no-chain"]) == 1
