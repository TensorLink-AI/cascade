"""WB23 signed leg record and validator check (TB27): ``ExpertEntry`` carries
the leg length (dropped at 0, so a leg without it keeps main's bytes), and a
validator refuses an entry whose recorded steps or guard exceed the consensus
caps at the ``leg_length`` stage, before any checkpoint fetch or evaluation.
Every node computes it from the signed bytes alone (satellite replay agrees)."""

from __future__ import annotations

import dataclasses
import inspect
import json

import pytest

from .helpers import base_contract, lane_b_cfg, want_reference_steps

GEN = "alice/metro-gen@sha256:" + "a" * 64
PTR = "metro-v1:trained:hippius:org/ckpt@sha256:" + "b" * 64
FIELDS = (("train_hours", 0.0), ("leg_steps", 0), ("leg_guard_seconds", 0))


def _fields_present():
    from cascade.shared.expert_manifest import ExpertEntry

    names = {f.name: f.default for f in dataclasses.fields(ExpertEntry)}
    for k, d in FIELDS:
        assert k in names and names[k] == d, f"ExpertEntry.{k} missing or default != {d}"


def _entry(**kw):
    from cascade.shared.expert_manifest import ExpertEntry

    return ExpertEntry(miner_hotkey="hk", miner_uid=1, gen_ref=GEN, domain="energy",
                       selection={0: [1, 2]}, exclude={}, base_king_pointer=PTR,
                       rebased_pointer=PTR, corpus_digest="cd", train_block=5,
                       commit_block=4, gpu_name="NVIDIA H100 80GB HBM3", **kw)


def _manifest(entry):
    from cascade.shared.expert_manifest import ExpertManifest

    return ExpertManifest(round_id="r1", created_block=10, king_pointer=PTR,
                          lineage_tokens=1, entries=(entry,))


def test_expert_entry_leg_length_dropped_at_zero_and_round_trips():
    from cascade.shared.expert_manifest import load_expert_manifest

    _fields_present()
    legacy = _manifest(_entry())
    body = json.loads(legacy.canonical_body())["entries"][0]
    assert not {k for k, _ in FIELDS} & set(body), "zeros must keep main's bytes"
    m = _manifest(_entry(train_hours=6.0, leg_steps=9504, leg_guard_seconds=27_000))
    e = json.loads(m.canonical_body())["entries"][0]
    assert (e["train_hours"], e["leg_steps"], e["leg_guard_seconds"]) == (6.0, 9504, 27_000)
    back = load_expert_manifest(m.to_json())
    assert back.digest() == m.digest() and back.entries[0].leg_steps == 9504
    assert load_expert_manifest(legacy.to_json()).digest() == legacy.digest()


@pytest.mark.parametrize("bad", [{"leg_steps": -1}, {"leg_guard_seconds": -5},
                                 {"leg_steps": True}, {"train_hours": float("nan")},
                                 {"train_hours": -0.5}])
def test_expert_entry_refuses_bad_leg_length(bad):
    _fields_present()
    with pytest.raises(ValueError):
        _entry(**bad)


# ── the validator stage (judge_entry) ────────────────────────────────────────

def _judge(entry, fetch):
    from cascade.validator.expert_lane import judge_entry

    assert "contract" in inspect.signature(judge_entry).parameters, (
        "judge_entry takes no contract (WB23: the leg_length consensus check)")
    return judge_entry(entry, manifest=_manifest(entry), king_pointer=PTR, king_scores=[],
                       evaluate=lambda p: pytest.fail("evaluated"), fetch_dir=fetch,
                       cfg=lane_b_cfg(), seed="1", contract=base_contract())


class _Fetched(Exception):
    pass


def _fetch_fails(p):
    raise AssertionError("a leg_length refusal must come before any checkpoint fetch")


def _fetch_marks(p):
    raise _Fetched(p)


def test_tb27_steps_over_the_cap_refused_before_fetch():
    _fields_present()
    c = base_contract()
    e = _entry(train_hours=2.0, leg_steps=want_reference_steps(c, 2.0) + 1,
               leg_guard_seconds=9000)
    j = _judge(e, _fetch_fails)
    assert (j.passed, j.stage, j.reason) == (False, "leg_length", "steps_out_of_range")


def test_tb27_guard_over_the_cap_refused_before_fetch():
    _fields_present()
    e = _entry(train_hours=6.0, leg_steps=100,
               leg_guard_seconds=int(lane_b_cfg().max_leg_seconds) + 1)
    j = _judge(e, _fetch_fails)
    assert (j.passed, j.stage, j.reason) == (False, "leg_length", "guard_out_of_range")


def test_judge_leg_length_unchanged_when_unarmed():
    """An entry without a leg length goes on to the fetch, as today."""
    with pytest.raises(_Fetched):
        _judge(_entry(), _fetch_marks)


def test_judge_leg_length_armed_nonempty_lane_d_effect():
    """A recorded leg inside the caps also goes on to the fetch."""
    _fields_present()
    e = _entry(train_hours=6.0, leg_steps=100,
               leg_guard_seconds=int(lane_b_cfg().max_leg_seconds))
    with pytest.raises(_Fetched):
        _judge(e, _fetch_marks)


def test_judge_manifest_and_validator_pass_the_contract():
    from cascade.validator import expert_lane, lane_b

    assert "contract" in inspect.getsource(expert_lane.judge_manifest) or \
        "**kw" in inspect.getsource(expert_lane.judge_manifest)
    src = inspect.getsource(lane_b)
    call = src[src.index("judge_manifest("):]
    assert "contract=" in call[:900], (
        "the validator must pass the round's training contract into judge_manifest")
