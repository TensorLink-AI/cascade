"""The trainer's reign clock anchors where VALIDATORS reset theirs.

In rolling (era) mode a validator installs a staged promotion at the start of
its ``effective_era`` and re-crowns the persisting king there with a fresh
clock (``_install_era_promotion`` → ``accept_promotion(block=era.start)``);
a dethrone resets it at the crowning. The trainer used to reset at the FIRE
block, a full notice era earlier, so its next promotion ripened early and
every attesting validator ignored the record forever (ripeness is judged at
the record's ``fired_block``) — the 2026-10-17 gen-15 split.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from cascade.shared.bench_report import BenchEntry, BenchReport
from cascade.shared.era import (
    min_effective_era,
    promotion_install_block,
    reign_anchor,
)
from cascade.shared.manifest import BenchScores
from cascade.shared.promotion import PromotedMember, PromotionRecord
from cascade.trainer.promotion import TrainerPromotion
from cascade.validator.cascade import (
    CascadeState,
    accept_promotion,
    crown,
    reign_rounds,
)

ROLLOVER = 9_000_000  # post-rollover grid: 900-block settlements, 4 per era
ERA = 3_600


def _cfgs():
    from cascade.shared.config import load_chain_config

    cfg = load_chain_config(Path(__file__).resolve().parents[2] / "chain.toml")
    round_cfg = replace(cfg.round, epoch_blocks=900, epoch_blocks_prev=3600,
                        epoch_activation_block=ROLLOVER, rolling_from_block=ROLLOVER,
                        era_settlements=4)
    scoring = replace(cfg.scoring, era_king_from_block=ROLLOVER,
                      tenure_blocks_from_block=ROLLOVER, cascade_reign_blocks=18_000,
                      cascade_reign_days=5)
    return round_cfg, scoring


def _engine(tmp_path, round_cfg, scoring):
    return TrainerPromotion(
        reign_threshold=5, k_max=2, quality_epsilon=0.05,
        state_path=tmp_path / "trainer_promotion.json",
        pointer_path=tmp_path / "warm_start_init.json",
        round_cfg=round_cfg, scoring_cfg=scoring,
    )


def _scores(v):
    return BenchScores(gifteval_crps=v, gifteval_mase=v, boom_crps=v,
                       boom_mase=v, time_crps=v, time_mase=v)


class _M:
    def __init__(self, rid, warm=""):
        self.round_id = rid
        self.warm_start_ckpt = warm


def _bench(eng, rid, block, ptrs, warm=""):
    eng.record_bench(_M(rid, warm), BenchReport(round_id=rid, created_block=block, entries=tuple(
        BenchEntry(role=role, size="toto2-4m", miner_hotkey=hk, miner_uid=0,
                   trained_pointer=p, scores=_scores(v))
        for p, (v, hk, role) in ptrs.items())))


def _fire(eng, round_cfg, block, rid, tag):
    _bench(eng, rid, block - 900, {f"ptr-{tag}-k": (1.0, "hkKing", "king"),
                                   f"ptr-{tag}-c": (1.01, "hkChal", "challenger")})
    return eng.maybe_promote(epoch_block=block, round_id=rid,
                             effective_era=min_effective_era(round_cfg, block))


def test_helpers():
    round_cfg, _ = _cfgs()
    assert promotion_install_block(round_cfg, 0, fired_block=ROLLOVER + 900) is None
    assert promotion_install_block(round_cfg, 2585, fired_block=9_299_700) == 2585 * ERA
    assert reign_anchor(crown_block=10, install_block=None) == 10
    assert reign_anchor(crown_block=10, install_block=30) == 30
    assert reign_anchor(crown_block=40, install_block=30) == 40
    assert reign_anchor(crown_block=None, install_block=None) is None


def test_fire_anchors_the_clock_at_the_effective_era_install(tmp_path):
    round_cfg, scoring = _cfgs()
    eng = _engine(tmp_path, round_cfg, scoring)
    crowned = ROLLOVER + 10 * 900
    eng.note_round("hkKing", epoch_block=crowned)
    fired = crowned + 20 * 900                      # ripe: 20 settlements
    rec = _fire(eng, round_cfg, fired, "r20", "a")
    assert rec is not None and rec.fired_block == fired
    install = rec.effective_era * ERA
    assert install > fired
    assert eng.reign_start_block == install         # NOT the fire block
    assert eng.live_install_block == install
    # persisted and reloaded
    again = TrainerPromotion.load(
        reign_threshold=5, k_max=2, quality_epsilon=0.05,
        state_path=tmp_path / "trainer_promotion.json",
        pointer_path=tmp_path / "warm_start_init.json",
        round_cfg=round_cfg, scoring_cfg=scoring)
    assert again.reign_start_block == install and again.live_install_block == install
    # the next generation never ripens before install + 18000 blocks
    _bench(eng, "r-late", install + 900, {"ptr-b-c": (0.99, "hkC2", "challenger")})
    assert eng.maybe_promote(epoch_block=install + 18_000 - 900, round_id="x",
                             effective_era=0) is None
    assert eng.generation == 1


def test_trainer_and_validator_clocks_agree_across_an_install(tmp_path):
    round_cfg, scoring = _cfgs()
    eng = _engine(tmp_path, round_cfg, scoring)
    crowned = ROLLOVER + 10 * 900
    eng.note_round("hkKing", epoch_block=crowned)
    rec = _fire(eng, round_cfg, crowned + 20 * 900, "r20", "a")
    # The validator: crowned at the same boundary, installs at the era start.
    v = crown(CascadeState(), king_hotkey="hkKing", block=crowned)
    v = accept_promotion(v, generation=1, members=rec.member_ids(),
                         block=rec.effective_era * ERA)
    t = CascadeState(king_hotkey="hkKing", reign_start_block=eng.reign_start_block)
    for b in range(v.reign_start_block, v.reign_start_block + 25 * 900, 900):
        assert reign_rounds(v, b, round_cfg) == reign_rounds(t, b, round_cfg)


def test_crowning_before_the_install_is_reset_again_at_the_install(tmp_path):
    """DEC-CA-0050 carry: a dethrone between fire and install — validators
    crown, then re-anchor at the install, so the trainer must too."""
    round_cfg, scoring = _cfgs()
    eng = _engine(tmp_path, round_cfg, scoring)
    eng.note_round("hkOld", epoch_block=ROLLOVER + 900)
    _bench(eng, "r1", ROLLOVER + 1800, {"ptr-k": (1.0, "hkOld", "king"),
                                       "ptr-c": (1.01, "hkChal", "challenger")})
    fired = ROLLOVER + 21 * 900
    rec = eng.maybe_promote(epoch_block=fired, round_id="r21",
                            effective_era=min_effective_era(round_cfg, fired))
    install = rec.effective_era * ERA
    new_crown = fired + 900                          # dethrone in the notice era
    assert new_crown < install
    eng.note_round("hkNew", epoch_block=new_crown)
    assert eng.reign_start_block == install
    # a crowning AFTER the install anchors at the crowning
    later = install + 2 * 900
    eng.note_round("hkThird", epoch_block=later)
    assert eng.reign_start_block == later


def test_gen15_regression_withdraw_to_the_validator_anchor(tmp_path):
    """2026-10-17: gen 14 fired 9271800 under the old king, u11 crowned at
    9276300 (trainer boundary), validators installed the carried gen 14 at
    era 2581 (9291600) and reset there. Withdrawn to that anchor, the engine
    cannot fire before 9291600 + 18000 = 9309600."""
    round_cfg, scoring = _cfgs()
    eng = _engine(tmp_path, round_cfg, scoring)
    gen14 = PromotionRecord(
        generation=14, king_hotkey="hkOld", fired_round="r14", fired_block=9_271_800,
        members=(PromotedMember("m14a", "toto2-4m", "s1", 0.597),),
        effective_era=2577)
    eng.generation = 15
    eng.members = (PromotedMember("m15a", "toto2-4m", "s2", 0.596),)
    eng.king_hotkey = "hkU11"
    eng.reign_start_block = 9_299_700
    eng.withdraw_to(gen14, reign_start_block=9_291_600)
    assert eng.generation == 14
    assert [m.checkpoint_id for m in eng.members] == ["m14a"]
    assert eng.pending_record is None and eng.candidates == ()
    assert eng.reign_start_block == 9_291_600
    _bench(eng, "r-c", 9_300_600, {"ptr-new": (0.590, "hkC", "challenger")}, warm="m14a")
    for b in range(9_299_700, 9_309_600, 900):
        assert eng.maybe_promote(epoch_block=b, round_id=f"r{b}",
                                 effective_era=min_effective_era(round_cfg, b)) is None
    rec = eng.maybe_promote(epoch_block=9_309_600, round_id="r-ripe",
                            effective_era=min_effective_era(round_cfg, 9_309_600))
    assert rec is not None and rec.generation == 15
    assert rec.effective_era == 2588
    assert eng.reign_start_block == 2588 * ERA


def test_withdraw_to_refuses_a_non_adjacent_generation(tmp_path):
    import pytest

    round_cfg, scoring = _cfgs()
    eng = _engine(tmp_path, round_cfg, scoring)
    eng.generation = 15
    rec = PromotionRecord(generation=13, king_hotkey="h", fired_round="r",
                          fired_block=1, members=(), effective_era=0)
    with pytest.raises(ValueError):
        eng.withdraw_to(rec, reign_start_block=5)
