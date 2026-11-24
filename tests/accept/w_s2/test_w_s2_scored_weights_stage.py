"""Plan rev6.20 (owner 2026-11-24, lane B only: EMA, no cooldown; no
activation block): the lane B judge (``cascade.validator.expert_lane.
judge_entry``, shared by the validator, the trainer's self-judge and
cascade-audit's replay) refuses an entry that does not record the lane B
scored weights (``scored_weights = "ema010"``) at stage ``scored_weights``,
before any checkpoint is fetched. Torch-free."""

from __future__ import annotations

import dataclasses

import pytest

from . import helpers as h


class _Fetched(Exception):
    """The judge went on to fetch checkpoints (past the scored-weights stage)."""


def _fetch(pointer):
    raise _Fetched(pointer)


def _judge(**kw):
    from cascade.shared.config.v2 import parse_lane_b
    from cascade.shared.expert_manifest import ExpertEntry, ExpertManifest
    from cascade.validator.expert_lane import judge_entry

    assert "scored_weights" in {f.name for f in dataclasses.fields(ExpertEntry)}, (
        "ExpertEntry has no scored_weights (plan rev6.20)")
    cfg = parse_lane_b({"from_block": 1, "domains": list(h.DOMAINS)})
    king = h.pointer("king")
    entry = ExpertEntry(miner_hotkey=h.HK, miner_uid=3, gen_ref=h.GEN, domain="energy",
                        selection={0: [1]}, exclude={}, base_king_pointer=king,
                        rebased_pointer=h.pointer("rebased"), corpus_digest="c",
                        train_block=h.SETTLE_ERA_START, commit_block=1, **kw)
    m = ExpertManifest(round_id="r", created_block=h.SETTLE_BLOCK, king_pointer=king,
                       lineage_tokens=0, entries=(entry,))
    return judge_entry(entry, manifest=m, king_pointer=king, king_scores=[],
                       evaluate=lambda p: [], fetch_dir=_fetch, cfg=cfg, seed="r")


def test_judge_refuses_a_leg_that_scored_the_raw_endpoint():
    j = _judge(scored_weights="")
    assert (j.passed, j.stage) == (False, "scored_weights"), (j.stage, j.reason)


def test_judge_refuses_a_manifest_without_the_record():
    import json

    from cascade.shared.expert_manifest import ExpertEntry, ExpertManifest, load_expert_manifest

    assert "scored_weights" in {f.name for f in dataclasses.fields(ExpertEntry)}

    king = h.pointer("king")
    m = ExpertManifest(round_id="r", created_block=h.SETTLE_BLOCK, king_pointer=king,
                       lineage_tokens=0, entries=(ExpertEntry(
                           miner_hotkey=h.HK, miner_uid=3, gen_ref=h.GEN, domain="energy",
                           selection={0: [1]}, exclude={}, base_king_pointer=king,
                           rebased_pointer=h.pointer("rebased"), corpus_digest="c",
                           train_block=h.SETTLE_ERA_START, commit_block=1),))
    o = json.loads(m.to_json())
    del o["entries"][0]["scored_weights"]
    (entry,) = load_expert_manifest(json.dumps(o)).entries
    j = _judge(scored_weights=entry.scored_weights)
    assert (j.passed, j.stage) == (False, "scored_weights"), (j.stage, j.reason)


def test_judge_goes_past_the_stage_for_an_ema_leg():
    with pytest.raises(_Fetched):
        _judge(scored_weights="ema010")
    with pytest.raises(_Fetched):
        _judge()                                    # the default record
