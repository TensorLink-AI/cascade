"""Plan rev6.20 (owner 2026-11-24, lane B only: EMA, no cooldown): the lane B
judge (``cascade.validator.expert_lane.judge_entry``, shared by the
validator, the trainer's self-judge and cascade-audit's replay) re-derives
which weights a leg scored from its ``train_block`` and ``[lane_b]
ema_no_cooldown_from_block`` and refuses a disagreeing entry at stage
``scored_weights``, before any checkpoint is fetched. Torch-free."""

from __future__ import annotations

import dataclasses

import pytest

from . import helpers as h

FROM = 5 * h.ERA_LEN


class _Fetched(Exception):
    """The judge went on to fetch checkpoints (past the scored-weights stage)."""


def _fetch(pointer):
    raise _Fetched(pointer)


def _judge(*, scored: str, train_block: int, from_block: int):
    from cascade.shared.config.v2 import parse_lane_b
    from cascade.shared.expert_manifest import ExpertEntry, ExpertManifest
    from cascade.validator.expert_lane import judge_entry

    assert "scored_weights" in {f.name for f in dataclasses.fields(ExpertEntry)}, (
        "ExpertEntry has no scored_weights (plan rev6.20)")
    cfg = parse_lane_b({"from_block": 1, "domains": list(h.DOMAINS),
                        "ema_no_cooldown_from_block": from_block})
    king = h.pointer("king")
    entry = ExpertEntry(miner_hotkey=h.HK, miner_uid=3, gen_ref=h.GEN, domain="energy",
                        selection={0: [1]}, exclude={}, base_king_pointer=king,
                        rebased_pointer=h.pointer("rebased"), corpus_digest="c",
                        train_block=train_block, commit_block=1, scored_weights=scored)
    m = ExpertManifest(round_id="r", created_block=train_block + 1, king_pointer=king,
                       lineage_tokens=0, entries=(entry,))
    return judge_entry(entry, manifest=m, king_pointer=king, king_scores=[],
                       evaluate=lambda p: [], fetch_dir=_fetch, cfg=cfg, seed="r")


@pytest.mark.parametrize("scored,train_block,from_block", [
    ("", FROM, FROM),              # armed: a raw leg is refused
    ("", FROM + 1, FROM),
    ("ema010", FROM - 1, FROM),    # an EMA leg dispatched before the key
    ("ema010", FROM, 0),           # an EMA leg while the key is unset
])
def test_judge_refuses_a_leg_that_scored_other_weights(scored, train_block, from_block):
    j = _judge(scored=scored, train_block=train_block, from_block=from_block)
    assert (j.passed, j.stage) == (False, "scored_weights"), (j.stage, j.reason)


@pytest.mark.parametrize("scored,train_block,from_block", [
    ("ema010", FROM, FROM),        # armed EMA leg
    ("ema010", FROM + 7, FROM),
])
def test_judge_goes_past_the_stage_when_the_rule_matches(scored, train_block, from_block):
    with pytest.raises(_Fetched):
        _judge(scored=scored, train_block=train_block, from_block=from_block)

