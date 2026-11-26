"""WB22, plan rev6.20 (owner 2026-11-24, lane B only: EMA, no cooldown; no
activation block): every signed expert manifest entry records which weights
the leg scored (``scored_weights = "ema010"``, always in the signed body; a
manifest without the record loads as ``""``, which the judge refuses), and
the trainer's dispatch paths build the lane B EMA leg contract. Contract:
README.md "rev6.20"."""

from __future__ import annotations

import dataclasses
import inspect
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

GEN = "alice/metro-gen@sha256:" + "a" * 64
KING = "metro-v1:trained:hippius:org/king@sha256:" + "b" * 64
LEG = "metro-v1:trained:hippius:org/leg@sha256:" + "c" * 64
REB = "metro-v1:trained:hippius:org/rebased@sha256:" + "d" * 64


def _has_field():
    from cascade.shared.expert_manifest import ExpertEntry

    f = {f.name: f for f in dataclasses.fields(ExpertEntry)}
    assert "scored_weights" in f, "ExpertEntry has no scored_weights (plan rev6.20)"
    assert f["scored_weights"].default == "ema010"


def _entry(**kw):
    from cascade.shared.expert_manifest import ExpertEntry

    return ExpertEntry(miner_hotkey="hk", miner_uid=1, gen_ref=GEN, domain="energy",
                       selection={0: [1, 2]}, exclude={}, base_king_pointer=KING,
                       rebased_pointer=REB, corpus_digest="cd", train_block=5,
                       commit_block=4, **kw)


def _manifest(entry):
    from cascade.shared.expert_manifest import ExpertManifest

    return ExpertManifest(round_id="r1", created_block=10, king_pointer=KING,
                          lineage_tokens=1, entries=(entry,))


def test_scored_weights_is_signed_and_round_trips():
    _has_field()
    from cascade.shared.expert_manifest import load_expert_manifest

    m = _manifest(_entry())
    assert json.loads(m.canonical_body())["entries"][0]["scored_weights"] == "ema010"
    back = load_expert_manifest(m.to_json())
    assert back.entries[0].scored_weights == "ema010" and back.digest() == m.digest()
    raw = _manifest(_entry(scored_weights=""))
    assert raw.digest() != m.digest()           # the record is in the signed digest


def test_a_manifest_without_the_record_loads_as_raw():
    _has_field()
    from cascade.shared.expert_manifest import load_expert_manifest

    o = json.loads(_manifest(_entry()).to_json())
    del o["entries"][0]["scored_weights"]
    assert load_expert_manifest(json.dumps(o)).entries[0].scored_weights == ""


# rev6.21 (owner 2026-11-24; tests-correction): "ema005" etc. are the miner's
# EMA choices (WB27, widths 0.01-0.30) — valid labels now; "ema999" stays unknown
@pytest.mark.parametrize("bad", ["raw", "ema999", "EMA010", None])
def test_scored_weights_refuses_unknown_labels(bad):
    _has_field()
    with pytest.raises(ValueError):
        _entry(scored_weights=bad)


def test_dispatch_paths_build_the_lane_b_leg_contract():
    from cascade.trainer.lane_b_driver import RunnerLaneBOps

    src = {fn.__name__: inspect.getsource(fn)
           for fn in (RunnerLaneBOps._dispatch_funded, RunnerLaneBOps._dispatch_operator)}
    assert "contract = lane_b_leg_contract(contract)" in src["_dispatch_funded"]
    assert "lc = lane_b_leg_contract(lc)" in src["_dispatch_operator"]


class _Ops:
    def __init__(self, dirs):
        self.dirs = dirs
        self.published = []

    def fetch(self, p):
        return self.dirs[p]

    def upload(self, d, repo):
        return REB

    def bench_rows(self, d):
        return None

    def publish_expert_manifest(self, m):
        self.published.append(m)

    def judge(self, m, round_id, block):
        return []


def test_settle_records_the_ema_on_every_entry(tmp_path, monkeypatch):
    _has_field()
    from cascade.shared.config import parse_lane_b
    from cascade.trainer import lane_b_driver as d
    from cascade.trainer.expert_lane import LaneBSubmission, OpenLeg

    rebuilt = []

    def fake_settle_leg(king_now, base, leg, sel, out, **kw):
        rebuilt.append(Path(leg))
        Path(out).mkdir(parents=True, exist_ok=True)
        return Path(out)

    monkeypatch.setattr(d, "settle_leg", fake_settle_leg)
    cfg = SimpleNamespace(lane_b=parse_lane_b({"from_block": 1, "domains": ["energy"],
                                               "open_tokens": 0}))
    ops = _Ops({KING: tmp_path / "king", LEG: tmp_path / "leg"})
    drv = d.LaneBDriver(cfg, ops, tmp_path / "lane_b.json")
    leg = OpenLeg(sub=LaneBSubmission("alice", 7, GEN, "energy", 5), selection={0: [1]},
                  base_king_ref=KING, base_era=0)
    drv.st.finished = [asdict(d.FinishedExpertLeg(leg=asdict(leg), leg_pointer=LEG,
                                                  corpus_digest="c", train_block=30))]
    drv.settle(round_id="r1", block=40, era_index=0, king_pointer=KING,
               work_dir=tmp_path / "w")
    assert rebuilt == [tmp_path / "leg"]           # rebased from the leg's checkpoint
    (m,) = ops.published
    assert m.entries[0].scored_weights == "ema010"
    assert json.loads(m.canonical_body())["entries"][0]["scored_weights"] == "ema010"
