"""Trainer-side promotion (DEC-CA-0012): the signed PromotionRecord wire format
and the selection engine — quality-gated structural-diversity selection,
reign tracking, candidate admissibility, rotation, migration, persistence.
"""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

from cascade.shared.bench_report import BenchEntry, BenchReport
from cascade.shared.manifest import BenchScores
from cascade.shared.promotion import (
    PromotedMember,
    PromotionRecord,
    dump_promotion_record,
    load_promotion_index,
    load_promotion_record,
    promotion_index_key,
    promotion_record_key,
    publish_promotion_record,
)
from cascade.trainer.promotion import Candidate, TrainerPromotion, select_members

DAY = 7_200  # blocks; the engine runs with round_cfg=None in these tests


# ── the signed record ────────────────────────────────────────────────────────


def _record(generation=1, members=((("cascade/a@sha256:" + "a" * 64), 1.0),)):
    return PromotionRecord(
        generation=generation, king_hotkey="hk0", fired_round="r7",
        fired_block=7 * DAY,
        members=tuple(PromotedMember(checkpoint_id=p, size="toto2-4m",
                                     source_round="r3", score=s)
                      for p, s in members),
    )


def test_record_round_trips_through_json():
    rec = _record(generation=4, members=(("cascade/a@sha256:" + "a" * 64, 1.0),
                                         ("cascade/b@sha256:" + "b" * 64, 1.03)))
    again = load_promotion_record(dump_promotion_record(rec))
    assert again == rec
    assert again.member_ids() == rec.member_ids()


def test_canonical_body_excludes_signature_and_is_stable():
    rec = _record()
    from dataclasses import replace

    assert rec.canonical_body() == replace(rec, signature="ff").canonical_body()
    # sorted-keys + tight separators: byte-stable across dump order.
    assert rec.canonical_body() == load_promotion_record(
        dump_promotion_record(rec)).canonical_body()


class _Store:
    def __init__(self):
        self.texts = {}

    def put_text(self, key, text, content_type=""):
        self.texts[key] = text

    def get_text(self, key):
        return self.texts[key]


def test_publish_writes_record_and_locator_index():
    store = _Store()
    rec = _record(generation=3)
    key = publish_promotion_record(store, dump_promotion_record(rec), 3)
    assert key == promotion_record_key(3)
    assert load_promotion_record(store.get_text(key)).generation == 3
    assert load_promotion_index(store.get_text(promotion_index_key())) == 3


def test_index_is_best_effort_on_garbage():
    assert load_promotion_index("{not json") == 0
    assert load_promotion_index("{}") == 0
    # Valid JSON that is not a dict (partial write, error page) is equally
    # "nothing located" — never an exception.
    assert load_promotion_index("[]") == 0
    assert load_promotion_index("3") == 0
    assert load_promotion_index("null") == 0


# ── selection: quality gate, then structural diversity ───────────────────────


def _cand(cid, score, *, hotkey="hkA", epoch=0, role="king"):
    return Candidate(checkpoint_id=cid, size="toto2-4m", hotkey=hotkey,
                     role=role, round_id=f"r{epoch}", epoch_index=epoch, score=score)


def test_selection_anchors_on_the_best():
    got = select_members([_cand("b", 1.0, epoch=1), _cand("a", 0.9, epoch=2)],
                         k_max=1, quality_epsilon=0.05)
    assert [c.checkpoint_id for c in got] == ["a"]


def test_quality_gate_excludes_off_frontier_candidates():
    # 1.2 is 20% off the best: never promoted, no matter how "diverse".
    got = select_members(
        [_cand("best", 1.0, epoch=0, hotkey="hkA"),
         _cand("bad", 1.2, epoch=9, hotkey="hkZ")],
        k_max=3, quality_epsilon=0.05)
    assert [c.checkpoint_id for c in got] == ["best"]


def test_selection_prefers_a_different_generator():
    # DEC-CA-0012: a challenger checkpoint (different generator = different
    # data distribution) beats a same-generator sibling with wider spacing.
    got = select_members(
        [_cand("king1", 1.0, epoch=1, hotkey="hkKing"),
         _cand("king2", 1.01, epoch=9, hotkey="hkKing"),
         _cand("chal", 1.02, epoch=2, hotkey="hkChal", role="challenger")],
        k_max=2, quality_epsilon=0.05)
    assert [c.checkpoint_id for c in got] == ["king1", "chal"]


def test_selection_spaces_rounds_within_one_generator():
    # Same generator throughout: the second slot takes the farthest round, not
    # the adjacent near-duplicate.
    got = select_members(
        [_cand("r1", 1.0, epoch=1), _cand("r2", 1.001, epoch=2),
         _cand("r8", 1.01, epoch=8)],
        k_max=2, quality_epsilon=0.05)
    assert [c.checkpoint_id for c in got] == ["r1", "r8"]


def test_min_round_spacing_blocks_same_generator_siblings():
    # Only a SAME-generator same-round sibling available beyond the anchor →
    # the set stays smaller rather than padding with a near-duplicate.
    got = select_members(
        [_cand("r1", 1.0, epoch=1), _cand("r1b", 1.001, epoch=1)],
        k_max=2, quality_epsilon=0.05, min_round_spacing=1)
    assert [c.checkpoint_id for c in got] == ["r1"]


def test_spacing_does_not_bind_across_generators():
    # A challenger checkpoint from the SAME round as the king's is a different
    # generator — genuinely different data — and co-promotable.
    got = select_members(
        [_cand("king", 1.0, epoch=1, hotkey="hkA"),
         _cand("chal", 1.001, epoch=1, hotkey="hkB", role="challenger")],
        k_max=2, quality_epsilon=0.05, min_round_spacing=1)
    assert [c.checkpoint_id for c in got] == ["king", "chal"]


def test_selection_is_deterministic_on_ties():
    a = [_cand("zzz", 1.0, epoch=1), _cand("aaa", 1.0, epoch=1)]
    b = list(reversed(a))
    assert [c.checkpoint_id for c in select_members(a, k_max=1, quality_epsilon=0.05)] == \
           [c.checkpoint_id for c in select_members(b, k_max=1, quality_epsilon=0.05)] == ["aaa"]


def test_selection_caps_at_k_max():
    cands = [_cand(f"c{i}", 1.0 + i * 0.001, epoch=i * 3, hotkey=f"hk{i}")
             for i in range(6)]
    assert len(select_members(cands, k_max=3, quality_epsilon=0.05)) == 3
    assert select_members([], k_max=3, quality_epsilon=0.05) == []


# ── the engine ───────────────────────────────────────────────────────────────


def _scores(v):
    return BenchScores(gifteval_crps=v, gifteval_mase=v, boom_crps=v,
                       boom_mase=v, time_crps=v, time_mase=v)


def _report(round_id, block, scored: dict[str, tuple[float, str, str]]):
    """scored: pointer -> (score, hotkey, role)."""
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


def _engine(tmp_path, *, threshold=5, k_max=2):
    return TrainerPromotion(
        reign_threshold=threshold, k_max=k_max, quality_epsilon=0.05,
        state_path=tmp_path / "trainer_promotion.json",
        pointer_path=tmp_path / "warm_start_init.json",
    )


def test_engine_fires_after_a_ripe_reign_and_writes_pointer(tmp_path):
    eng = _engine(tmp_path)
    eng.note_round("hkKing", epoch_block=0)
    eng.record_bench(_manifest("r1"), _report("r1", 1 * DAY, {
        "ptr-king1": (1.0, "hkKing", "king"),
        "ptr-chal1": (1.02, "hkChal", "challenger"),
    }))
    assert eng.maybe_promote(epoch_block=4 * DAY, round_id="r4") is None  # not ripe
    rec = eng.maybe_promote(epoch_block=5 * DAY, round_id="r5")
    assert rec is not None and rec.generation == 1
    assert rec.member_ids() == ("ptr-king1", "ptr-chal1")
    assert eng.candidates == ()                     # log cleared for the new reign
    assert eng.reign_start_block == 5 * DAY         # clock reset, king persists
    assert eng.king_hotkey == "hkKing"
    # Pointer file carries the member set + legacy single-pointer mirror.
    obj = json.loads((tmp_path / "warm_start_init.json").read_text(encoding="utf-8"))
    assert [m["checkpoint_id"] for m in obj["members"]] == ["ptr-king1", "ptr-chal1"]
    assert obj["checkpoint_id"] == "ptr-king1"
    assert obj["generation"] == 1


def test_engine_rotates_members_by_epoch():
    eng = TrainerPromotion(reign_threshold=5, k_max=2, quality_epsilon=0.05)
    assert eng.init_for_epoch(0) is None            # random-init era
    eng.members = (PromotedMember("a", "toto2-4m", "r1", 1.0),
                   PromotedMember("b", "toto2-4m", "r2", 1.01))
    assert eng.init_for_epoch(10) == ("a", "toto2-4m")
    assert eng.init_for_epoch(11) == ("b", "toto2-4m")
    assert eng.init_for_epoch(12) == ("a", "toto2-4m")


def test_king_change_resets_reign_and_candidates(tmp_path):
    eng = _engine(tmp_path)
    eng.note_round("hkA", epoch_block=0)
    eng.record_bench(_manifest("r1"), _report("r1", 1 * DAY, {
        "ptr-a": (1.0, "hkA", "king")}))
    eng.note_round("hkB", epoch_block=3 * DAY)
    assert eng.candidates == () and eng.reign_start_block == 3 * DAY
    # A full fresh reign is needed under the new king.
    assert eng.maybe_promote(epoch_block=5 * DAY, round_id="r5") is None


def test_bench_from_a_previous_generation_is_not_a_candidate(tmp_path):
    # A late report for a round trained off the OLD generation must not seed
    # the new one: admissibility keys on the round's manifest pin.
    eng = _engine(tmp_path)
    eng.note_round("hkKing", epoch_block=0)
    eng.members = (PromotedMember("live-a", "toto2-4m", "r0", 1.0),)
    eng.generation = 1
    assert eng.record_bench(_manifest("r1", warm_start_ckpt=""),
                            _report("r1", 1 * DAY, {"ptr-x": (1.0, "hkKing", "king")})) == 0
    assert eng.record_bench(_manifest("r2", warm_start_ckpt="stale-ptr"),
                            _report("r2", 2 * DAY, {"ptr-y": (1.0, "hkKing", "king")})) == 0
    assert eng.record_bench(_manifest("r3", warm_start_ckpt="live-a"),
                            _report("r3", 3 * DAY, {"ptr-z": (1.0, "hkKing", "king")})) == 1


def test_record_bench_dedupes_pointers(tmp_path):
    eng = _engine(tmp_path)
    eng.note_round("hkKing", epoch_block=0)
    rep = _report("r1", 1 * DAY, {"ptr-a": (1.0, "hkKing", "king")})
    assert eng.record_bench(_manifest("r1"), rep) == 1
    assert eng.record_bench(_manifest("r1"), rep) == 0


def test_engine_persists_and_reloads(tmp_path):
    eng = _engine(tmp_path)
    eng.note_round("hkKing", epoch_block=0)
    eng.record_bench(_manifest("r1"), _report("r1", 1 * DAY, {
        "ptr-a": (1.0, "hkKing", "king")}))
    assert eng.maybe_promote(epoch_block=5 * DAY, round_id="r5") is not None

    again = TrainerPromotion.load(
        reign_threshold=5, k_max=2, quality_epsilon=0.05,
        state_path=tmp_path / "trainer_promotion.json",
        pointer_path=tmp_path / "warm_start_init.json",
    )
    assert again.generation == 1
    assert [m.checkpoint_id for m in again.members] == ["ptr-a"]
    assert again.king_hotkey == "hkKing"
    assert again.reign_start_block == 5 * DAY


def test_legacy_pointer_file_grandfathered_as_generation_one(tmp_path):
    # Migration: the old validator-installed single winner becomes generation 1
    # so an armed deployment upgrades without a mismatch round.
    (tmp_path / "warm_start_init.json").write_text(
        json.dumps({"checkpoint_id": "old-winner", "size": "toto2-4m", "score": 1.1}),
        encoding="utf-8")
    eng = TrainerPromotion.load(
        reign_threshold=5, k_max=2, quality_epsilon=0.05,
        state_path=tmp_path / "trainer_promotion.json",
        pointer_path=tmp_path / "warm_start_init.json",
    )
    assert eng.generation == 1
    assert [m.checkpoint_id for m in eng.members] == ["old-winner"]
    assert eng.init_for_epoch(7) == ("old-winner", "toto2-4m")
    # The pointer file was rewritten in the new schema (members + mirror).
    obj = json.loads((tmp_path / "warm_start_init.json").read_text(encoding="utf-8"))
    assert [m["checkpoint_id"] for m in obj["members"]] == ["old-winner"]


def test_fired_record_pends_until_publish_confirmed(tmp_path):
    # State advances at fire time, so the record must survive a publish
    # failure (and a restart) until the caller confirms it landed — otherwise
    # a store outage orphans a generation the pointer file already rotates on.
    eng = _engine(tmp_path)
    eng.note_round("hkKing", epoch_block=0)
    eng.record_bench(_manifest("r1"), _report("r1", 1 * DAY, {
        "ptr-a": (1.0, "hkKing", "king")}))
    rec = eng.maybe_promote(epoch_block=5 * DAY, round_id="r5")
    assert eng.unpublished_record() == rec

    # A restart re-offers the same pending record.
    again = TrainerPromotion.load(
        reign_threshold=5, k_max=2, quality_epsilon=0.05,
        state_path=tmp_path / "trainer_promotion.json",
        pointer_path=tmp_path / "warm_start_init.json",
    )
    assert again.unpublished_record() is not None
    assert again.unpublished_record().generation == 1
    assert again.unpublished_record().member_ids() == ("ptr-a",)

    again.mark_record_published()
    assert again.unpublished_record() is None
    # ...and the confirmation persists too.
    final = TrainerPromotion.load(
        reign_threshold=5, k_max=2, quality_epsilon=0.05,
        state_path=tmp_path / "trainer_promotion.json",
        pointer_path=tmp_path / "warm_start_init.json",
    )
    assert final.unpublished_record() is None


def test_lost_state_file_readopts_member_pointer(tmp_path):
    # Losing/corrupting trainer_promotion.json must degrade to a resumable
    # engine: the surviving member-set pointer file is re-adopted at its
    # recorded generation, so candidates keep accumulating and the generation
    # counter never rolls back below what validators accepted.
    eng = _engine(tmp_path)
    eng.note_round("hkKing", epoch_block=0)
    eng.record_bench(_manifest("r1"), _report("r1", 1 * DAY, {
        "ptr-a": (1.0, "hkKing", "king")}))
    assert eng.maybe_promote(epoch_block=5 * DAY, round_id="r5") is not None
    (tmp_path / "trainer_promotion.json").unlink()

    again = TrainerPromotion.load(
        reign_threshold=5, k_max=2, quality_epsilon=0.05,
        state_path=tmp_path / "trainer_promotion.json",
        pointer_path=tmp_path / "warm_start_init.json",
    )
    assert again.generation == 1
    assert [m.checkpoint_id for m in again.members] == ["ptr-a"]
    # Rounds pinning the live member are candidates again.
    again.note_round("hkKing", epoch_block=6 * DAY)
    assert again.record_bench(
        _manifest("r6", warm_start_ckpt="ptr-a"),
        _report("r6", 6 * DAY, {"ptr-b": (1.0, "hkKing", "king")})) == 1


def test_ripe_clock_with_no_candidates_holds(tmp_path):
    eng = _engine(tmp_path)
    eng.note_round("hkKing", epoch_block=0)
    assert eng.maybe_promote(epoch_block=30 * DAY, round_id="r30") is None
    assert eng.generation == 0


def test_promoted_score_is_geomean_of_the_six(tmp_path):
    eng = _engine(tmp_path, k_max=1)
    eng.note_round("hkKing", epoch_block=0)
    eng.record_bench(_manifest("r1"), _report("r1", 1 * DAY, {
        "ptr-a": (0.8, "hkKing", "king")}))
    rec = eng.maybe_promote(epoch_block=5 * DAY, round_id="r5")
    assert rec is not None
    assert math.isclose(rec.members[0].score, 0.8, rel_tol=1e-9)
