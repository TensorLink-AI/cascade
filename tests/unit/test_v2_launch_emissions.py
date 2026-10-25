"""Owner rules around the v2 go-live: a random-init window before the first
v2 warm start, a pre-launch burn, and a capped carried king."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from cascade.shared.bench_report import BenchEntry, BenchReport
from cascade.shared.config import load_chain_config
from cascade.shared.emission import EmissionPolicy, apply_emission_policy, emission_policy
from cascade.shared.era import v2_first_era, v2_first_warm_era, v2_generation_usable
from cascade.shared.manifest import BenchScores

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def typed():
    cfg = load_chain_config(ROOT / "chain.toml")
    era = cfg.round.epoch_blocks * cfg.round.era_settlements
    gate = 10_000 * era
    # the shipped DEC-CA-0055 timing keys zeroed: each test arms what it models
    cfg = replace(cfg, v2=replace(cfg.v2, from_block=gate, random_init_eras=0,
                                  reign_warm_start_blocks=0),
                  training=replace(cfg.training, v2_from_block=gate))
    return cfg, gate, era


def test_random_init_window_holds_warm_starts(typed):
    cfg, gate, era = typed
    first = v2_first_era(cfg)
    assert v2_first_warm_era(replace(cfg, v2=replace(cfg.v2, random_init_eras=0))) == first + 2
    held = replace(cfg, v2=replace(cfg.v2, random_init_eras=20))
    assert v2_first_warm_era(held) == first + 20
    late = gate + 25 * era
    assert not v2_generation_usable(held, late, first + 19)
    assert v2_generation_usable(held, late, first + 20)


def test_prelaunch_burn_scales_every_recipient(typed):
    cfg, gate, era = typed
    cfg = replace(cfg, v2=replace(cfg.v2, prelaunch_burn=0.95))
    # nothing burns until the on-chain vote (or a typed block) sets the start
    assert emission_policy(cfg, boundary=gate - era, king_since_block=None).inert
    cfg = replace(cfg, v2=replace(cfg.v2, prelaunch_burn_from_block=gate - 2 * era))
    assert emission_policy(cfg, boundary=gate - 3 * era, king_since_block=None).inert
    pol = emission_policy(cfg, boundary=gate - era, king_since_block=None)
    assert pol == EmissionPolicy(prelaunch_burn=0.95)
    out = apply_emission_policy([0.0, 0.6, 0.4], pol, burn_uid=0)
    assert out == pytest.approx([0.95, 0.03, 0.02])
    # the settlement AT the v2 block still judges the last pre-v2 era
    assert emission_policy(cfg, boundary=gate, king_since_block=None).prelaunch_burn == 0.95
    assert emission_policy(cfg, boundary=gate + era, king_since_block=None).prelaunch_burn == 0.0


def test_carried_king_is_capped_until_a_v2_crowning(typed):
    cfg, gate, era = typed
    # the ladder off: this models the cap alone (DEC-CA-0055 item 2)
    cfg = replace(cfg, v2=replace(cfg.v2, carryover_king_share=0.05, king_ladder=False))
    mid = gate + era
    pol = emission_policy(cfg, boundary=mid, king_since_block=gate - 5 * era)
    assert pol.lane_a_cap == 0.05
    # lane A 0.5 (king 0.4, court 0.1), lane B pays uid 3 0.2, 0.3 already burned
    out = apply_emission_policy([0.3, 0.4, 0.1, 0.2], pol, burn_uid=0, lane_a_uids=[1, 2])
    assert out == pytest.approx([0.75, 0.04, 0.01, 0.2])
    assert sum(out) == pytest.approx(1.0)
    assert emission_policy(cfg, boundary=mid, king_since_block=None).lane_a_cap == 0.05
    assert emission_policy(cfg, boundary=mid + era, king_since_block=mid).inert


def test_inert_by_default():
    cfg = load_chain_config(ROOT / "chain.toml")
    # the shipped chain.toml: v2 untyped and the burn not yet voted ⇒ inert
    assert cfg.v2.prelaunch_burn_from_block == 0
    assert emission_policy(cfg, boundary=10**9, king_since_block=None).inert
    assert apply_emission_policy([1.0, 0.0], EmissionPolicy(), burn_uid=0) == [1.0, 0.0]


def test_reign_clock_counts_from_the_end_of_the_window(typed, tmp_path):
    from cascade.shared.era import v2_reign_floor_block
    from cascade.trainer.promotion import TrainerPromotion

    cfg, gate, era = typed
    assert v2_reign_floor_block(cfg) == 0                       # no window configured
    held = replace(cfg, v2=replace(cfg.v2, random_init_eras=14))
    floor = v2_reign_floor_block(held)
    assert floor == gate + 14 * era
    eng = TrainerPromotion(reign_threshold=5, k_max=2, quality_epsilon=0.05,
                           state_path=tmp_path / "p.json", pointer_path=tmp_path / "w.json")
    day = 7200
    eng.note_round("hkKing", epoch_block=0)                     # a long pre-v2 reign
    eng.record_bench(SimpleNamespace(round_id="r1", warm_start_ckpt=""), BenchReport(
        round_id="r1", created_block=day, entries=(BenchEntry(
            role="king", size="toto2-4m", miner_hotkey="hkKing", miner_uid=0,
            trained_pointer="ptr", scores=BenchScores(**{
                f: 1.0 for f in BenchScores.__dataclass_fields__})),)))
    assert eng.maybe_promote(epoch_block=floor + day, round_id="a",
                             clock_floor_block=floor) is None    # 1 day into the clock
    assert eng.maybe_promote(epoch_block=floor + 5 * day, round_id="b",
                             clock_floor_block=floor) is not None


def test_seed_epoch_runs_from_one_warm_start_to_the_next(typed):
    from cascade.shared.era import era_for_block, v2_era_seed_block, v2_seed_epoch_start

    cfg, gate, era = typed
    first = v2_first_era(cfg)
    assert v2_seed_epoch_start(cfg, first - 1, ()) is None             # pre-v2
    warm = (first + 21, first + 29)                                    # two promotions take effect
    assert v2_seed_epoch_start(cfg, first, warm) == first
    assert v2_seed_epoch_start(cfg, first + 20, warm) == first         # window + first reign
    assert v2_seed_epoch_start(cfg, first + 21, warm) == first + 21
    assert v2_seed_epoch_start(cfg, first + 35, warm) == first + 29
    e = era_for_block(cfg.round, (first + 25) * era)
    assert v2_era_seed_block(cfg, e, warm) == (first + 20) * era       # the epoch start's seed block
    e0 = era_for_block(cfg.round, (first + 21) * era)
    assert v2_era_seed_block(cfg, e0, warm) == e0.seed_block


def test_check_is_due_when_the_warm_start_would_land_three_days_in(typed):
    from cascade.shared.era import min_effective_era, v2_check_due

    cfg, gate, era = typed
    c = replace(cfg, v2=replace(cfg.v2, random_init_eras=14, reign_warm_start_blocks=21600))
    floor = gate + 14 * era                       # the window's end: the clock starts here
    assert not v2_check_due(cfg, boundary=floor + 10 * era, reign_start_block=0)   # key off
    due = [b for b in range(floor, floor + 8 * era + 1, cfg.round.epoch_blocks)
           if v2_check_due(c, boundary=b, reign_start_block=0)]
    first = due[0]
    landed = min_effective_era(c.round, first) * era
    assert landed - floor >= 21600 > landed - era - floor   # lands in the first era ≥ 3 days in
    assert first - floor < 21600                            # decided ~2 days into the reign
